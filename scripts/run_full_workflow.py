#!/usr/bin/env python3
"""
Complete GSDB Workflow Example

This script demonstrates a complete workflow from data processing to model evaluation.
It serves as both documentation and a reproducible experiment script.

Usage:
    python run_full_workflow.py [--base_dir PATH] [--task {inflammation,tissue}] 
                               [--model MODEL] [--skip_preprocessing]
                               [--skip_training] [--epochs EPOCHS]
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import subprocess
import json
import yaml

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import project modules
from src.config.paths import get_project_paths, add_path_args

def setup_logging(log_dir):
    """Configure logging to both file and console."""
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"workflow_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def run_command(command, description=None, check=True):
    """Run a shell command with proper logging."""
    if description:
        logging.info(description)
    
    logging.info(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Log command output
        if result.stdout:
            for line in result.stdout.splitlines():
                logging.info(f"STDOUT: {line}")
        
        if result.stderr:
            for line in result.stderr.splitlines():
                logging.warning(f"STDERR: {line}")
        
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            for line in e.stdout.splitlines():
                logging.info(f"STDOUT: {line}")
        if e.stderr:
            for line in e.stderr.splitlines():
                logging.error(f"STDERR: {line}")
        if check:
            raise
        return e

def get_timestamp():
    """Get a formatted timestamp for directory names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def preprocess_data(paths):
    """Run the data preprocessing steps."""
    logging.info("Starting data preprocessing...")

    
    # Step 1: Extract tiles from WSIs
    logging.info("Step 3: Extracting tiles from whole slide images")
    cmd = f"python {project_root}/src/data/extract_tiles.py --base-dir {paths['BASE_DIR']} --tile-size 256 --downsample 10 --overlap 64"
    run_command(cmd)

    # Step 2: Process dataset metadata
    logging.info("Step 2: Processing dataset metadata")
    cmd = f"python {project_root}/src/data/process_dataset.py --base-dir {paths['BASE_DIR']}"
    run_command(cmd)
    
    # Step 3: Create dataset splits
    logging.info("Step 4: Creating dataset splits")
    cmd = f"python {project_root}/src/data/create_splits.py --base-dir {paths['BASE_DIR']} --seed 42"
    run_command(cmd)
    
    logging.info("Data preprocessing completed successfully")
    return True

def train_model(paths, args):
    """Train a model for the specified task."""
    logging.info(f"Starting model training for {args.task} task...")
    
    # Build the training command
    cmd = [
        f"python {project_root}/src/models/train.py",
        f"--task {args.task}",
        f"--model {args.model}",
        f"--base-dir {paths['BASE_DIR']}",
        f"--epochs {args.epochs}"
    ]
    
    # Add optional arguments if specified
    if args.batch_size:
        cmd.append(f"--batch-size {args.batch_size}")
    
    if args.learning_rate:
        cmd.append(f"--learning-rate {args.learning_rate}")
    
    # Run the training command
    run_command(" ".join(cmd))
    
    # Check if model was created successfully
    model_pattern = f"best_model_{args.task}_{args.model}.pt"
    model_files = list(paths["MODELS_DIR"].glob(model_pattern))
    
    if not model_files:
        logging.error(f"No model file found matching pattern: {model_pattern}")
        return None
    
    # Return the most recently created model file
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logging.info(f"Model training completed successfully. Model saved to: {latest_model}")
    return latest_model

def evaluate_model(paths, args, model_path):
    """Evaluate the trained model on test sets."""
    logging.info(f"Starting model evaluation for {args.task} task...")
    
    results = {}
    
    # Evaluate on both test and test_scanner2 splits
    for test_split in ["test", "test_scanner2"]:
        logging.info(f"Evaluating on {test_split} split...")
        
        cmd = [
            f"python {project_root}/src/models/evaluate.py",
            f"--task {args.task}",
            f"--test_split {test_split}",
            f"--model_path {model_path}",
            f"--architecture {args.model}",
            f"--base-dir {paths['BASE_DIR']}"
        ]
        
        run_command(" ".join(cmd))
        
        # Find the most recent evaluation results
        eval_dir = paths["RESULTS_DIR"] / "evaluations"
        eval_pattern = f"{args.task}_{test_split}_*"
        eval_dirs = list(eval_dir.glob(eval_pattern))
        
        if not eval_dirs:
            logging.error(f"No evaluation results found for {test_split}")
            continue
        
        latest_eval_dir = max(eval_dirs, key=lambda p: p.stat().st_mtime)
        
        # Find metrics file in the evaluation directory
        metrics_files = list(latest_eval_dir.glob("metrics_*.json"))
        if not metrics_files:
            logging.error(f"No metrics file found in {latest_eval_dir}")
            continue
        
        # Load and store the metrics
        latest_metrics_file = max(metrics_files, key=lambda p: p.stat().st_mtime)
        with open(latest_metrics_file, 'r') as f:
            metrics = json.load(f)
        
        results[test_split] = {
            'metrics_file': str(latest_metrics_file),
            'key_metrics': {
                'accuracy': metrics.get(f'slide_accuracy', metrics.get('particle_accuracy', metrics.get('tile_accuracy', 0))),
                'f1': metrics.get(f'slide_f1', metrics.get('particle_f1', metrics.get('tile_f1', 0))),
                'auc': metrics.get(f'slide_auroc', metrics.get('particle_auroc', metrics.get('tile_auroc', 0)))
            }
        }
    
    # Summarize results
    if results:
        logging.info("\nEvaluation Summary:")
        for split, result in results.items():
            logging.info(f"\n{split.upper()} Split:")
            logging.info(f"Metrics file: {result['metrics_file']}")
            logging.info(f"Accuracy: {result['key_metrics']['accuracy']:.2f}%")
            logging.info(f"F1 Score: {result['key_metrics']['f1']:.2f}%")
            logging.info(f"AUC: {result['key_metrics']['auc']:.2f}%")
    
    logging.info("Model evaluation completed")
    return results

def save_experiment_metadata(paths, args, model_path, evaluation_results):
    """Save experiment metadata for reproducibility."""
    timestamp = get_timestamp()
    experiment_dir = paths["RESULTS_DIR"] / "experiments" / f"{args.task}_{args.model}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect experiment metadata
    metadata = {
        "timestamp": timestamp,
        "task": args.task,
        "model": args.model,
        "model_path": str(model_path),
        "base_dir": str(paths["BASE_DIR"]),
        "args": vars(args),
        "evaluation_results": evaluation_results,
        "git_commit": None
    }
    
    # Try to get git commit hash if available
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd=project_root,
            text=True
        ).strip()
        metadata["git_commit"] = git_hash
    except:
        pass
    
    # Save metadata
    metadata_file = experiment_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy config files
    for config_file in paths["CONFIG_DIR"].glob("*.yaml"):
        with open(config_file, 'r') as src, open(experiment_dir / config_file.name, 'w') as dst:
            dst.write(src.read())
    
    logging.info(f"Experiment metadata saved to: {experiment_dir}")
    return experiment_dir

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run complete GSDB workflow")
    
    # Add path arguments
    parser = add_path_args(parser)
    
    # Add workflow configuration arguments
    parser.add_argument("--task", choices=["inflammation", "tissue"], 
                        default="inflammation", help="Classification task")
    parser.add_argument("--model", default="convnext_large",
                        choices=["resnet18", "convnext_large", "swin_v2_b", "gigapath"],
                        help="Model architecture to train")
    parser.add_argument("--skip_preprocessing", action="store_true",
                      help="Skip the preprocessing steps")
    parser.add_argument("--skip_training", action="store_true",
                      help="Skip model training")
    parser.add_argument("--model_path", type=Path,
                      help="Path to pre-trained model (if skipping training)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float,
                      help="Learning rate for training")
    
    return parser.parse_args()

def main():
    """Main workflow function."""
    # Parse arguments
    args = parse_args()
    
    # Get project paths
    paths = get_project_paths(base_dir=args.base_dir)
    
    # Ensure log directory exists
    paths["LOGS_DIR"].mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(paths["LOGS_DIR"])
    
    logging.info(f"Starting GSDB workflow at {datetime.now()}")
    logging.info(f"Log file: {log_file}")
    
    # Log paths
    logging.info("Project paths:")
    for path_name, path_value in paths.items():
        logging.info(f"  {path_name}: {path_value}")
    
    # Log arguments
    logging.info("Command line arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
    
    start_time = time.time()
    
    # Step 1: Preprocess data (unless skipped)
    if not args.skip_preprocessing:
        if not preprocess_data(paths):
            logging.error("Data preprocessing failed")
            return 1
    else:
        logging.info("Skipping data preprocessing (--skip_preprocessing)")
    
    # Step 2: Train model (unless skipped)
    model_path = args.model_path
    if not args.skip_training:
        model_path = train_model(paths, args)
        if not model_path:
            logging.error("Model training failed")
            return 1
    else:
        logging.info("Skipping model training (--skip_training)")
        if not model_path:
            logging.error("--model_path must be provided when using --skip_training")
            return 1
        if not model_path.exists():
            logging.error(f"Model file not found: {model_path}")
            return 1
    
    # Step 3: Evaluate model
    evaluation_results = evaluate_model(paths, args, model_path)
    
    # Step 4: Save experiment metadata
    experiment_dir = save_experiment_metadata(paths, args, model_path, evaluation_results)
    
    # Calculate total runtime
    elapsed_time = time.time() - start_time
    logging.info(f"Workflow completed in {elapsed_time:.1f} seconds")
    logging.info(f"Experiment metadata saved to: {experiment_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())