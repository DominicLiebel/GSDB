#!/usr/bin/env python3
"""
Train and evaluate all models for a specified task (inflammation or tissue).

This script:
1. Trains all models defined in the config file for the specified task
2. Evaluates each trained model on specified dataset split(s)
3. Skips models that have already been trained or evaluated

Usage:
    python train_evaluate_all.py --task inflammation
    python train_evaluate_all.py --task tissue --deterministic --seed 42
    python train_evaluate_all.py --task inflammation --eval-dataset test_scanner2
    python train_evaluate_all.py --task inflammation --eval-dataset all
"""

import argparse
import json
import os
import subprocess
import sys
import yaml
import time
from datetime import datetime
from pathlib import Path
import logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate all models for a specified task")
    parser.add_argument("--task", required=True, choices=["inflammation", "tissue"], 
                        help="Classification task")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", 
                        help="Enable deterministic mode for reproducibility")
    parser.add_argument("--config-path", type=Path, 
                        default=Path('/mnt/data/dliebel/2024_dliebel/configs/model_config.yaml'),
                        help="Path to model_config.yaml")
    parser.add_argument("--skip-training", action="store_true", 
                        help="Skip training and only run evaluation")
    parser.add_argument("--skip-evaluation", action="store_true", 
                        help="Skip evaluation and only run training")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="Project root directory")
    parser.add_argument("--eval-dataset", choices=["test", "test_scanner2", "all"], 
                        default="test",
                        help="Dataset to use for evaluation (default: test)")
    return parser.parse_args()

def setup_logging(task, log_dir):
    """Set up logging to file and console."""
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_all_{task}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Log file: {log_file}")
    return log_file

def load_config(config_path):
    """Load model configurations from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)

def get_models_for_task(config, task):
    """Get list of model names for the specified task."""
    if task in config:
        return list(config[task].keys())
    return []

def check_model_trained(model_path):
    """Check if a model has already been trained by looking for the model file."""
    return model_path.exists()

def check_evaluation_exists(eval_dir, task, model_name, dataset_split):
    """Check if evaluation results already exist for a model on a dataset split."""
    # Look for directories matching the pattern "task_dataset_split_*"
    pattern = f"{task}_{dataset_split}_*"
    eval_dirs = list(eval_dir.glob(pattern))
    
    for eval_path in eval_dirs:
        # Check for predictions.csv and statistics.json
        pred_file = eval_path / "predictions.csv"
        stats_file = eval_path / "statistics.json"
        
        if pred_file.exists() and stats_file.exists():
            try:
                # Check if statistics.json contains info about this model
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    if stats.get('model_name') == model_name:
                        logging.info(f"Found existing evaluation for {model_name} on {dataset_split} at {eval_path}")
                        return True
            except Exception as e:
                logging.warning(f"Error checking statistics file {stats_file}: {e}")
                continue
    
    return False

def run_command(cmd, description=None, check=True, timeout=None):
    """Run a shell command with proper logging."""
    if description:
        logging.info(description)
    
    cmd_str = " ".join(str(c) for c in cmd)
    logging.info(f"Running: {cmd_str}")
    
    try:
        result = subprocess.run(
            cmd,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        
        # Log command output
        if result.stdout:
            for line in result.stdout.splitlines():
                logging.info(f"STDOUT: {line}")
        
        if result.stderr:
            for line in result.stderr.splitlines():
                logging.warning(f"STDERR: {line}")
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logging.error(f"Command timed out after {timeout} seconds: {cmd_str}")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {cmd_str}")
        if e.stdout:
            for line in e.stdout.splitlines():
                logging.info(f"STDOUT: {line}")
        if e.stderr:
            for line in e.stderr.splitlines():
                logging.error(f"STDERR: {line}")
        return False
    except Exception as e:
        logging.error(f"Error executing command: {e}")
        return False

def train_model(project_root, task, model_name, seed, deterministic):
    """Train a model using train.py."""
    cmd = [
        "python", str(project_root / "src" / "models" / "train.py"),
        "--task", task,
        "--model", model_name,
        "--seed", str(seed)
    ]
    
    if deterministic:
        cmd.append("--deterministic")
    
    # Training can take a long time, so no timeout
    return run_command(cmd, f"Training {task} model: {model_name}")

def evaluate_model(project_root, task, model_path, dataset_split, architecture, deterministic):
    """Evaluate a model using evaluate.py."""
    cmd = [
        "python", str(project_root / "src" / "models" / "evaluate.py"),
        "--task", task,
        "--test_split", dataset_split,
        "--model_path", str(model_path),
        "--architecture", architecture
    ]
    
    if deterministic:
        cmd.append("--deterministic")
    
    # Set a reasonable timeout for evaluation (3 hours)
    return run_command(
        cmd, 
        f"Evaluating {task} model {architecture} on {dataset_split}",
        timeout=10800
    )

def main():
    """Main function to train and evaluate all models for a task."""
    start_time = time.time()
    args = parse_args()
    
    # Project paths
    project_root = Path("/mnt/data/dliebel/2024_dliebel")
    logs_dir = project_root / "logs"
    models_dir = project_root / "results" / "models"
    eval_dir = project_root / "results" / "evaluations"
    
    # Create directories
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.task, logs_dir)
    
    # Log script start information
    logging.info(f"Starting processing for {args.task} task")
    logging.info(f"Project root: {project_root}")
    logging.info(f"Models directory: {models_dir}")
    logging.info(f"Evaluations directory: {eval_dir}")
    
    # Log command line arguments
    for arg, value in vars(args).items():
        logging.info(f"Argument {arg}: {value}")
    
    # Verify config path exists
    config_path = args.config_path
    if not config_path.exists():
        # Try to find config in project directory
        alt_config_path = project_root / "configs" / "model_config.yaml"
        if alt_config_path.exists():
            config_path = alt_config_path
            logging.info(f"Using alternative config path: {config_path}")
        else:
            logging.error(f"Config file not found at {config_path} or {alt_config_path}")
            sys.exit(1)
    
    # Load configurations
    logging.info(f"Loading configurations from {config_path}")
    config = load_config(config_path)
    
    # Get models for the specified task
    models = get_models_for_task(config, args.task)
    
    if not models:
        logging.error(f"No models found for task: {args.task}")
        sys.exit(1)
    
    logging.info(f"Found {len(models)} models for {args.task} task: {models}")
    
    # Track successfully processed models
    trained_models = []
    
    # Determine which dataset splits to evaluate
    if args.eval_dataset == "all":
        evaluation_splits = ["test", "test_scanner2"]
    else:
        evaluation_splits = [args.eval_dataset]
    
    evaluated_models = {split: [] for split in evaluation_splits}
    
    # Process each model
    for model_name in models:
        model_path = models_dir / f"best_model_{args.task}_{model_name}.pt"
        
        # Training phase
        if not args.skip_training:
            # Check if model needs training
            if check_model_trained(model_path):
                logging.info(f"Model {model_name} already trained at {model_path}. Skipping training.")
                trained_models.append(model_name)
            else:
                logging.info(f"Model {model_name} not found at {model_path}. Starting training.")
                if train_model(project_root, args.task, model_name, args.seed, args.deterministic):
                    logging.info(f"Successfully trained model: {model_name}")
                    trained_models.append(model_name)
                else:
                    logging.error(f"Failed to train model: {model_name}")
                    continue  # Skip evaluation if training failed
        else:
            logging.info(f"Skipping training for model {model_name} as requested")
            if model_path.exists():
                trained_models.append(model_name)
        
        # Evaluation phase
        if not args.skip_evaluation:
            # Check if model file exists before attempting evaluation
            if not model_path.exists():
                logging.error(f"Model file {model_path} not found. Skipping evaluation.")
                continue
                
            # Evaluate model on each selected dataset split
            for dataset_split in evaluation_splits:
                if check_evaluation_exists(eval_dir, args.task, model_name, dataset_split):
                    logging.info(f"Evaluation for {model_name} on {dataset_split} already exists. Skipping.")
                    evaluated_models[dataset_split].append(model_name)
                else:
                    logging.info(f"Starting evaluation of {model_name} on {dataset_split}")
                    if evaluate_model(project_root, args.task, model_path, dataset_split, model_name, args.deterministic):
                        logging.info(f"Successfully evaluated {model_name} on {dataset_split}")
                        evaluated_models[dataset_split].append(model_name)
                    else:
                        logging.error(f"Failed to evaluate {model_name} on {dataset_split}")
        else:
            logging.info(f"Skipping evaluation for model {model_name} as requested")
    
    # Summary
    elapsed_time = time.time() - start_time
    logging.info(f"\nProcess completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    logging.info(f"\nSummary for {args.task} task:")
    logging.info(f"Models successfully trained ({len(trained_models)}): {', '.join(trained_models)}")
    
    for dataset_split, models_list in evaluated_models.items():
        logging.info(f"Models evaluated on {dataset_split} ({len(models_list)}): {', '.join(models_list)}")
    
    # List models that failed or were skipped
    all_models_set = set(models)
    trained_set = set(trained_models)
    
    if not args.skip_training:
        failed_training = all_models_set - trained_set
        if failed_training:
            logging.warning(f"Models that failed training or were skipped: {', '.join(failed_training)}")
    
    if not args.skip_evaluation:
        for dataset_split, models_list in evaluated_models.items():
            evaluated_set = set(models_list)
            failed_evaluation = trained_set - evaluated_set
            if failed_evaluation:
                logging.warning(f"Models that failed evaluation on {dataset_split} or were skipped: {', '.join(failed_evaluation)}")

if __name__ == "__main__":
    main()