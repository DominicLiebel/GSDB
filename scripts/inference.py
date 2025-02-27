#!/usr/bin/env python3
"""
Inference Script for Gastric Histology Classification

This script performs inference on new slides using trained models.
It supports both inflammation classification and tissue type classification.

Example usage:
    python inference.py --slides_dir /path/to/new/slides --model_path /path/to/model.pt \
                       --output_dir /path/to/results --task inflammation --batch_size 32
"""

import argparse
import logging
import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import project modules
from src.config.paths import get_project_paths
from src.models.architectures import HistologyClassifier, GigaPathClassifier
from src.utils.error_handling import handle_exception, DataLoadingError, ConfigurationError
from src.models.evaluate import get_transforms
from src.models.training_utils import set_all_seeds


def setup_logging(output_dir: Path) -> Path:
    """Configure logging to file and console.
    
    Args:
        output_dir: Directory to save log file
        
    Returns:
        Path to log file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"inference_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging configured. Log file: {log_file}")
    return log_file


def load_model(model_path: Path, device: torch.device, architecture: str = None) -> torch.nn.Module:
    """Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        architecture: Model architecture (optional, inferred from checkpoint if None)
        
    Returns:
        Loaded model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model couldn't be loaded
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
            # Get architecture from config if not provided
            architecture = architecture or config.get('architecture', {}).get('name')
            task = config.get('task')
            logging.info(f"Loaded configuration from checkpoint. Task: {task}, Architecture: {architecture}")
        else:
            # If no configuration present, require architecture to be specified
            if not architecture:
                raise ConfigurationError("Architecture not found in checkpoint and not provided")
            logging.warning("No configuration found in checkpoint, using provided architecture")
        
        # Create model based on architecture
        if architecture == 'gigapath':
            model = GigaPathClassifier(num_classes=1)
        else:
            model = HistologyClassifier(model_name=architecture, num_classes=1)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading the checkpoint directly as a state dict
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        # Log model information
        logging.info(f"Model loaded successfully: {architecture}")
        logging.info(f"Task: {task if 'task' in locals() else 'unknown'}")
        
        # Check if reproducibility info is present
        if 'reproducibility' in checkpoint:
            repro_info = checkpoint['reproducibility']
            seed = repro_info.get('random_seed')
            if seed is not None:
                logging.info(f"Model was trained with seed: {seed}")
                # Use the same seed for inference
                set_all_seeds(seed)

        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def process_slide(slide_path: Path, model: torch.nn.Module, device: torch.device, 
                  transform=None, batch_size: int = 32) -> Dict:
    """Process a whole slide by extracting and classifying tiles.
    
    Args:
        slide_path: Path to slide file
        model: Trained model
        device: Device to run inference on
        transform: Image transformations
        batch_size: Batch size for inference
        
    Returns:
        Dictionary of results including predictions and metadata
    """
    # Check if this is a single file or a directory of tiles
    if slide_path.is_file():
        # Process single image file
        return process_single_image(slide_path, model, device, transform)
    
    # Process directory of tiles
    tiles = list(slide_path.glob("*.png"))
    if not tiles:
        logging.warning(f"No image tiles found in {slide_path}")
        return {
            'slide_name': slide_path.name,
            'num_tiles': 0,
            'predictions': [],
            'probabilities': [],
            'tile_paths': []
        }
    
    # Process tiles in batches
    all_probs = []
    all_preds = []
    all_tile_paths = []
    
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i+batch_size]
        batch_images = []
        
        # Load and preprocess images
        for tile_path in batch_tiles:
            try:
                img = Image.open(tile_path).convert('RGB')
                if transform:
                    img = transform(img)
                batch_images.append(img)
                all_tile_paths.append(str(tile_path))
            except Exception as e:
                logging.warning(f"Error processing tile {tile_path}: {str(e)}")
                continue
        
        if not batch_images:
            continue
        
        # Stack images into a batch
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(batch_tensor).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (outputs > 0).float().cpu().numpy()
            
            all_probs.extend(probs.tolist() if isinstance(probs, np.ndarray) else [probs])
            all_preds.extend(preds.tolist() if isinstance(preds, np.ndarray) else [preds])
    
    # Aggregate results
    result = {
        'slide_name': slide_path.name,
        'num_tiles': len(all_probs),
        'predictions': all_preds,
        'probabilities': all_probs,
        'tile_paths': all_tile_paths,
        'mean_probability': np.mean(all_probs) if all_probs else 0.0,
        'median_probability': np.median(all_probs) if all_probs else 0.0,
        'max_probability': np.max(all_probs) if all_probs else 0.0,
        'prediction': 1 if np.mean(all_probs) > 0.5 else 0 if all_probs else 0
    }
    
    return result


def process_single_image(image_path: Path, model: torch.nn.Module, device: torch.device, 
                        transform=None) -> Dict:
    """Process a single image file.
    
    Args:
        image_path: Path to image file
        model: Trained model
        device: Device to run inference on
        transform: Image transformations
        
    Returns:
        Dictionary with inference results
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        if transform:
            img = transform(img)
        
        # Add batch dimension
        img_tensor = img.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(img_tensor).squeeze()
            prob = torch.sigmoid(output).cpu().item()
            pred = 1 if prob > 0.5 else 0
        
        # Return results
        return {
            'slide_name': image_path.name,
            'num_tiles': 1,
            'predictions': [pred],
            'probabilities': [prob],
            'tile_paths': [str(image_path)],
            'mean_probability': prob,
            'median_probability': prob,
            'max_probability': prob,
            'prediction': pred
        }
        
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return {
            'slide_name': image_path.name,
            'num_tiles': 0,
            'predictions': [],
            'probabilities': [],
            'tile_paths': [],
            'error': str(e)
        }


def save_results(results: List[Dict], output_dir: Path, task: str, threshold: float = 0.5):
    """Save inference results to files.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save results
        task: Task type ('inflammation' or 'tissue')
        threshold: Classification threshold
    """
    # Create CSV with slide-level results
    slide_results = []
    for result in results:
        slide_results.append({
            'slide_name': result['slide_name'],
            'num_tiles': result['num_tiles'],
            'mean_probability': result['mean_probability'],
            'median_probability': result['median_probability'],
            'max_probability': result['max_probability'],
            'prediction': 1 if result['mean_probability'] > threshold else 0,
            'prediction_label': 'Positive' if result['mean_probability'] > threshold else 'Negative'
        })
    
    # Save to CSV
    slides_df = pd.DataFrame(slide_results)
    csv_path = output_dir / f"{task}_slide_predictions.csv"
    slides_df.to_csv(csv_path, index=False)
    logging.info(f"Slide-level results saved to {csv_path}")
    
    # Save tile-level results
    all_tiles = []
    for result in results:
        for i, (pred, prob, tile_path) in enumerate(zip(
            result['predictions'], 
            result['probabilities'], 
            result['tile_paths']
        )):
            all_tiles.append({
                'slide_name': result['slide_name'],
                'tile_path': tile_path,
                'probability': prob,
                'prediction': pred,
                'prediction_label': 'Positive' if pred == 1 else 'Negative'
            })
    
    # Save to CSV
    tiles_df = pd.DataFrame(all_tiles)
    csv_path = output_dir / f"{task}_tile_predictions.csv"
    tiles_df.to_csv(csv_path, index=False)
    logging.info(f"Tile-level results saved to {csv_path}")
    
    # Save detailed JSON with all information
    json_path = output_dir / f"{task}_detailed_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Detailed results saved to {json_path}")
    
    # Create visualization
    create_visualization(slides_df, tiles_df, output_dir, task)


def create_visualization(slides_df: pd.DataFrame, tiles_df: pd.DataFrame, 
                         output_dir: Path, task: str):
    """Create visualization of inference results.
    
    Args:
        slides_df: DataFrame with slide-level results 
        tiles_df: DataFrame with tile-level results
        output_dir: Directory to save visualizations
        task: Task type ('inflammation' or 'tissue')
    """
    # Create figure with multiple subplots
    plt.figure(figsize=(18, 12))
    
    # 1. Probability distribution
    plt.subplot(2, 2, 1)
    sns.histplot(tiles_df['probability'], bins=50, kde=True)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    plt.title('Probability Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.legend()
    
    # 2. Slide-level summary
    plt.subplot(2, 2, 2)
    slide_counts = slides_df['prediction_label'].value_counts()
    plt.pie(slide_counts, labels=slide_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=['#ff9999','#66b3ff'])
    plt.title(f'Slide-level {task.capitalize()} Predictions')
    
    # 3. Probability by slide (top 10 slides by tile count)
    plt.subplot(2, 1, 2)
    top_slides = slides_df.nlargest(10, 'num_tiles')
    top_slide_names = top_slides['slide_name'].tolist()
    top_slide_tiles = tiles_df[tiles_df['slide_name'].isin(top_slide_names)]
    
    sns.boxplot(x='slide_name', y='probability', data=top_slide_tiles)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    plt.title('Probability Distribution by Slide (Top 10 by Tile Count)')
    plt.xlabel('Slide')
    plt.ylabel('Probability')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / f"{task}_visualization.png"
    plt.savefig(fig_path, dpi=300)
    logging.info(f"Visualization saved to {fig_path}")
    plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on histology slides')
    
    parser.add_argument('--slides_dir', type=Path, required=True,
                        help='Directory containing slides/tiles to process')
    parser.add_argument('--model_path', type=Path, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='Directory to save results')
    parser.add_argument('--task', choices=['inflammation', 'tissue'], default='inflammation',
                        help='Classification task')
    parser.add_argument('--architecture', type=str,
                        choices=['resnet18', 'densenet121', 'densenet169', 
                                'convnext_large', 'swin_v2_b', 'gigapath'],
                        help='Model architecture (only needed if not in checkpoint)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function for the inference script."""
    args = parse_args()
    
    try:
        # Setup logging
        log_file = setup_logging(args.output_dir)
        
        # Set random seed for reproducibility
        set_all_seeds(args.seed)
        
        # Get device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Load model
        model = load_model(args.model_path, device, args.architecture)
        
        # Get appropriate transforms
        transform = get_transforms(is_training=False)
        
        # Collect slides to process
        if args.slides_dir.is_file():
            slides = [args.slides_dir]
            logging.info(f"Processing single file: {args.slides_dir}")
        else:
            # Check if the directory contains tiles or subdirectories of tiles
            png_files = list(args.slides_dir.glob("*.png"))
            
            if png_files:
                # Directory contains tiles directly
                slides = [args.slides_dir]
                logging.info(f"Processing directory containing {len(png_files)} tiles")
            else:
                # Directory contains subdirectories for each slide
                slides = [d for d in args.slides_dir.iterdir() if d.is_dir()]
                logging.info(f"Found {len(slides)} slide directories to process")
        
        # Process each slide
        results = []
        for slide in tqdm(slides, desc="Processing slides"):
            logging.info(f"Processing slide: {slide}")
            result = process_slide(slide, model, device, transform, args.batch_size)
            results.append(result)
            logging.info(f"Completed slide {slide.name}. "
                         f"Tiles: {result['num_tiles']}, "
                         f"Mean probability: {result['mean_probability']:.4f}")
        
        # Save and visualize results
        save_results(results, args.output_dir, args.task, args.threshold)
        
        logging.info(f"Inference completed successfully")
        
    except Exception as e:
        handle_exception(e, exit_code=1)


if __name__ == "__main__":
    main()