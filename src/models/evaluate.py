"""
Histology Model Evaluation Module

This module provides functionality for evaluating trained histological image classifiers
at multiple hierarchical levels (tile, particle, and WSI).

Usage:
    python evaluate.py --task <inflammation|tissue> --test_split <test|test_scanner2> --model_path
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Local imports
from src.models.dataset import HistologyDataset
import src.models.metrics_utils as metrics_utils
from src.config.paths import get_project_paths, add_path_args
from src.models.model_utils import load_model, get_transforms

# Import shared model architectures from the dedicated module
from src.models.architectures import GigaPathClassifier, HistologyClassifier




def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    task: str,
    split: str,
    device: torch.device,
    output_dir: Optional[Path] = None,
    validation_thresholds: Optional[Dict] = None, 
    save_predictions: bool = True
) -> Dict:
    """Evaluate model at appropriate hierarchical levels.
    
    Args:
        model: Neural network model to evaluate
        data_loader: DataLoader for evaluation data
        task: 'inflammation' or 'tissue'
        split: Dataset split name ('test', 'test_scanner2')
        device: Device to run evaluation on
        output_dir: Directory to save evaluation results
        validation_thresholds: Pre-computed thresholds from validation data
        
    Returns:
        Dict: Dictionary of evaluation metrics
    """
    model.eval()
    predictions = []
    
    # Get model name - use attribute if available
    if hasattr(model, 'model_name'):
        model_name = model.model_name
    else:
        # Default to model class name
        model_name = model.__class__.__name__
    
    logging.info(f"Evaluating model: {model_name}")
    
    # Collect predictions
    with torch.no_grad():
        for inputs, labels, metadata_batch in tqdm(data_loader, desc=f"Evaluating {split}"):
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()  # Get raw logits
            
            # Process each sample in the batch
            for i in range(len(outputs)):
                metadata = {k: v[i] if isinstance(v, list) else v for k, v in metadata_batch.items()}
                
                pred_dict = {
                    'raw_pred': outputs[i].cpu().item(),  # Store raw logits
                    'label': labels[i].item(),
                    'particle_id': metadata['particle_id'],
                    'slide_name': metadata['slide_name'],
                }
                
                # Add task field to support task detection
                if task == 'inflammation':
                    pred_dict['inflammation_type'] = metadata.get('inflammation_type', 'unknown')
                
                predictions.append(pred_dict)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(predictions)
    
    # Log basic statistics
    logging.info(f"\nEvaluation Summary:")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Unique slides: {df['slide_name'].nunique()}")
    logging.info(f"Unique particles: {len(df.groupby(['slide_name', 'particle_id']))}")
    
    # Calculate class distribution
    pos_count = (df['label'] == 1).sum()
    neg_count = (df['label'] == 0).sum()
    logging.info(f"\nClass Distribution:")
    logging.info(f"Positive class: {pos_count} ({pos_count/len(df):.1%})")
    logging.info(f"Negative class: {neg_count} ({neg_count/len(df):.1%})")
    
    # Log prediction statistics
    logging.info("\nPrediction Statistics:")
    logging.info(f"Raw logits mean: {df['raw_pred'].mean():.3f}")
    logging.info(f"Raw logits std: {df['raw_pred'].std():.3f}")
    logging.info(f"Raw logits min: {df['raw_pred'].min():.3f}")
    logging.info(f"Raw logits max: {df['raw_pred'].max():.3f}")
    
    # FIXED: Always use validation thresholds or calculate them if not provided
    if validation_thresholds is None:
        logging.warning("No validation thresholds provided. Calculating thresholds from validation data...")
        # Calculate optimal thresholds on validation data - never on test data
        validation_thresholds = metrics_utils.calculate_validation_thresholds(
            model_path=None,  # We'll use the current model, not load from path
            model=model,      # Pass the current model directly
            task=task,
            output_dir=output_dir / "validation_thresholds" if output_dir else None
        )
        logging.info("Validation-optimized thresholds calculated.")

    # Log the validation thresholds
    logging.info("\nUsing validation-optimized thresholds:")
    optimal_thresholds = {k: v for k, v in validation_thresholds.items() 
                         if k != "optimal_aggregation"}
    for level, metrics in optimal_thresholds.items():
        logging.info(f"{level.capitalize()} level: {metrics['threshold']:.3f}")
        logging.info(f"  Sensitivity: {metrics['sensitivity']:.2f}")
        logging.info(f"  Specificity: {metrics['specificity']:.2f}")
        logging.info(f"  F1 Score: {metrics['f1']:.2f}")

    # Use validation-optimized aggregation strategy if available
    if "optimal_aggregation" in validation_thresholds:
        logging.info("\nUsing validation-optimized aggregation strategy:")
        agg_results = validation_thresholds["optimal_aggregation"]
        
        # Extract validation confusion matrix first
        val_tp = agg_results['best_metrics'].get('tp', 0)
        val_tn = agg_results['best_metrics'].get('tn', 0)
        val_fp = agg_results['best_metrics'].get('fp', 0)
        val_fn = agg_results['best_metrics'].get('fn', 0)
        
        # Add new field to store validation confusion matrix
        agg_results['validation_confusion_matrix'] = {
            'tp': val_tp,
            'tn': val_tn,
            'fp': val_fp,
            'fn': val_fn,
            'total': val_tp + val_tn + val_fp + val_fn
        }
        
        logging.info("\nValidation Confusion Matrix:")
        logging.info(f"  True Positives: {val_tp}")
        logging.info(f"  True Negatives: {val_tn}")
        logging.info(f"  False Positives: {val_fp}")
        logging.info(f"  False Negatives: {val_fn}")
        logging.info(f"  Total Samples: {val_tp + val_tn + val_fp + val_fn}")
        
        # Apply the optimal strategy to the current test set
        best_strategy = agg_results['best_strategy']
        best_threshold = agg_results['best_metrics']['threshold']
        
        # Define aggregation function based on strategy name
        if best_strategy == 'mean':
            agg_func = np.mean
        elif best_strategy == 'median':
            agg_func = np.median
        elif best_strategy == 'max':
            agg_func = np.max
        elif best_strategy == 'min':
            agg_func = np.min
        elif best_strategy == 'percentile_75':
            agg_func = lambda x: np.percentile(x, 75)
        elif best_strategy == 'percentile_90':
            agg_func = lambda x: np.percentile(x, 90)
        elif best_strategy == 'top_k_mean_10':
            agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.1)):]) if len(x) > 0 else 0
        elif best_strategy == 'top_k_mean_20':
            agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.2)):]) if len(x) > 0 else 0
        elif best_strategy == 'filter_90_mean':
            agg_func = lambda x: np.mean(np.sort(x)[int(len(x)*0.9):]) if len(x) > 0 else 0
        elif best_strategy == 'filter_80_mean':
            agg_func = lambda x: np.mean(np.sort(x)[int(len(x)*0.8):]) if len(x) > 0 else 0
        elif best_strategy == 'filter_70_mean':
            agg_func = lambda x: np.mean(np.sort(x)[int(len(x)*0.7):]) if len(x) > 0 else 0
        else:
            logging.warning(f"Unknown aggregation strategy: {best_strategy}, defaulting to mean")
            agg_func = np.mean
            
        # Log the description of the strategy
        if 'filter_90_mean' in best_strategy:
            logging.info("Strategy description: Filter out 90% of tiles with lowest activation, compute mean of top 10%")
        elif 'filter_80_mean' in best_strategy:
            logging.info("Strategy description: Filter out 80% of tiles with lowest activation, compute mean of top 20%")
        elif 'filter_70_mean' in best_strategy:
            logging.info("Strategy description: Filter out 70% of tiles with lowest activation, compute mean of top 30%")
        
        # Calculate probabilities from logits if needed
        if 'raw_pred' in df.columns and (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1):
            df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        elif 'raw_pred' in df.columns and 'prob' not in df.columns:
            df['prob'] = df['raw_pred']
        
        if task == 'inflammation':
            slide_df = df.groupby('slide_name').agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()
            
            # Apply threshold
            slide_preds = (slide_df['prob'] > best_threshold).astype(int)
            
            # Recalculate confusion matrix
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(slide_df['label'].values, slide_preds).ravel()
            
            # Calculate accuracy and F1 score on test data
            test_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            if tp + fp + fn == 0:
                test_f1 = 0
            else:
                test_f1 = 2 * tp / (2 * tp + fp + fn)
            
            # Create test confusion matrix
            agg_results['test_confusion_matrix'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'total': int(tp + tn + fp + fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2)
            }
            
            # Update the metrics with test confusion matrix for reporting
            agg_results['best_metrics']['tp'] = int(tp)
            agg_results['best_metrics']['tn'] = int(tn)
            agg_results['best_metrics']['fp'] = int(fp)
            agg_results['best_metrics']['fn'] = int(fn)
            agg_results['best_metrics']['accuracy'] = float(test_accuracy)
            agg_results['best_metrics']['sensitivity'] = float(test_sensitivity)
            agg_results['best_metrics']['specificity'] = float(test_specificity)
            agg_results['best_metrics']['f1'] = float(test_f1)
            agg_results['best_metrics']['balanced_acc'] = float((test_sensitivity + test_specificity) / 2)
            
            # Log test confusion matrix
            logging.info("\nTest Confusion Matrix:")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  F1 Score: {test_f1:.4f}")
        
        elif task == 'tissue':
            # Similar code for tissue task with particle grouping
            particle_df = df.groupby(['slide_name', 'particle_id']).agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()
            
            # Apply threshold
            particle_preds = (particle_df['prob'] > best_threshold).astype(int)
            
            # Recalculate confusion matrix
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(particle_df['label'].values, particle_preds).ravel()
            
            # Calculate accuracy and F1 score on test data
            test_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            test_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            if tp + fp + fn == 0:
                test_f1 = 0
            else:
                test_f1 = 2 * tp / (2 * tp + fp + fn)
                
            # Create test confusion matrix
            agg_results['test_confusion_matrix'] = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'total': int(tp + tn + fp + fn),
                'accuracy': float(test_accuracy),
                'sensitivity': float(test_sensitivity),
                'specificity': float(test_specificity),
                'f1': float(test_f1),
                'balanced_acc': float((test_sensitivity + test_specificity) / 2)
            }
            
            # Update the metrics with test confusion matrix for reporting
            agg_results['best_metrics']['tp'] = int(tp)
            agg_results['best_metrics']['tn'] = int(tn)
            agg_results['best_metrics']['fp'] = int(fp)
            agg_results['best_metrics']['fn'] = int(fn)
            agg_results['best_metrics']['accuracy'] = float(test_accuracy)
            agg_results['best_metrics']['sensitivity'] = float(test_sensitivity)
            agg_results['best_metrics']['specificity'] = float(test_specificity)
            agg_results['best_metrics']['f1'] = float(test_f1)
            agg_results['best_metrics']['balanced_acc'] = float((test_sensitivity + test_specificity) / 2)
            
            # Log test confusion matrix
            logging.info("\nTest Confusion Matrix:")
            logging.info(f"  True Positives: {tp}")
            logging.info(f"  True Negatives: {tn}")
            logging.info(f"  False Positives: {fp}")
            logging.info(f"  False Negatives: {fn}")
            logging.info(f"  Total Samples: {tp + tn + fp + fn}")
            logging.info(f"  Accuracy: {test_accuracy:.4f}")
            logging.info(f"  F1 Score: {test_f1:.4f}")
    else:
        # FIXED: Never optimize on test data - either use validation thresholds or defaults
        logging.info("\nNo pre-computed aggregation strategy found in validation thresholds.")
        agg_results = validation_thresholds.get("optimal_aggregation", {})
        if not agg_results:
            logging.warning("Using default mean aggregation with threshold of 0.5")
            agg_results = {
                "best_strategy": "mean",
                "best_metrics": {"threshold": 0.5}
            }
            logging.info("This default strategy was NOT optimized - results may be suboptimal.")
    
    # Log aggregation results
    best_strategy = agg_results.get('best_strategy', 'mean')
    best_metrics = agg_results.get('best_metrics', {})
    logging.info(f"  Strategy: {best_strategy}")
    logging.info(f"  Threshold: {best_metrics.get('threshold', 0.5):.3f}")
    logging.info(f"  Sensitivity: {best_metrics.get('sensitivity', 0):.2f}")
    logging.info(f"  Specificity: {best_metrics.get('specificity', 0):.2f}")
    logging.info(f"  F1 Score: {best_metrics.get('f1', 0):.2f}")
    
    # Create enhanced distribution plots
    if output_dir:
        # Convert logits to probabilities
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        
        # Create figure with two subplots
        plt.figure(figsize=(16, 7))
        
        # First subplot - Raw logits distribution by class
        plt.subplot(1, 2, 1)
        plt.hist(df['raw_pred'][df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(df['raw_pred'][df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.0, color='r', linestyle='--', label='Default threshold (logit=0)')
        
        # Add optimal threshold from hierarchical optimization (if available)
        if task == 'inflammation' and 'slide' in optimal_thresholds:
            # Convert probability to logit for visualization
            prob_threshold = optimal_thresholds['slide']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--', 
                      label=f'Optimal threshold (logit={logit_threshold:.2f})')
        elif task == 'tissue' and 'particle' in optimal_thresholds:
            # Convert probability to logit for visualization
            prob_threshold = optimal_thresholds['particle']['threshold']
            logit_threshold = np.log(prob_threshold/(1-prob_threshold)) if 0 < prob_threshold < 1 else (10 if prob_threshold >= 1 else -10)
            plt.axvline(x=logit_threshold, color='g', linestyle='--', 
                      label=f'Optimal threshold (logit={logit_threshold:.2f})')
        
        plt.title('Raw Logits Distribution by Class')
        plt.xlabel('Logit Value')
        plt.ylabel('Count')
        plt.legend()
        
        # Second subplot - Probability distribution by class
        plt.subplot(1, 2, 2)
        plt.hist(probs[df['label']==1], bins=50, alpha=0.6, label='Positive class', color='blue')
        plt.hist(probs[df['label']==0], bins=50, alpha=0.6, label='Negative class', color='orange')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Default threshold (prob=0.5)')
        
        # Add optimal threshold
        if task == 'inflammation' and 'slide' in optimal_thresholds:
            plt.axvline(x=optimal_thresholds['slide']['threshold'], color='g', linestyle='--', 
                      label=f'Optimal threshold (prob={optimal_thresholds["slide"]["threshold"]:.2f})')
        elif task == 'tissue' and 'particle' in optimal_thresholds:
            plt.axvline(x=optimal_thresholds['particle']['threshold'], color='g', linestyle='--', 
                      label=f'Optimal threshold (prob={optimal_thresholds["particle"]["threshold"]:.2f})')
        
        # Add optimal aggregation threshold if available
        if 'threshold' in best_metrics:
            plt.axvline(x=best_metrics['threshold'], color='purple', linestyle='-.', 
                      label=f'Best agg. threshold (prob={best_metrics["threshold"]:.2f})')
        
        plt.title('Probability Distribution by Class')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f'pred_distribution_{split}.png', dpi=300)
        plt.close()
    
    # Calculate metrics using validation-optimized thresholds
    if 'slide' in optimal_thresholds:
        slide_threshold = optimal_thresholds['slide']['threshold']
        threshold_for_metrics = np.log(slide_threshold/(1-slide_threshold)) if 0 < slide_threshold < 1 else 0.0
        
        logging.info(f"Calculating metrics using validation-optimized threshold: {threshold_for_metrics:.4f} (logit)")
    else:
        threshold_for_metrics = 0.0  # Default threshold
    
    # Calculate metrics using specified threshold
    metrics = metrics_utils.calculate_hierarchical_metrics(
        df=df,
        split=split,
        model_name=model_name,
        threshold=threshold_for_metrics  # Use validation-optimized threshold if available
    )
    
    # Combine results
    combined_metrics = {
        **metrics,
        'optimal_thresholds': optimal_thresholds,
        'optimal_aggregation': agg_results  # Assume agg_results is calculated as before
    }
    
    return combined_metrics

def setup_logging(output_dir: Path) -> None:
    """Configure logging for both file and console output."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    file_handler = logging.FileHandler(output_dir / 'evaluation.log')
    console_handler = logging.StreamHandler()
    
    # Set format
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging configured. Log file: {output_dir / 'evaluation.log'}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate histology classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--task',
        required=True,
        choices=['inflammation', 'tissue'],
        help='Classification task'
    )
    parser.add_argument(
        '--test_split',
        required=True,
        choices=['test', 'test_scanner2'],
        help='Test split to evaluate'
    )
    parser.add_argument(
        '--model_path',
        required=True,
        type=Path,
        help='Path to model weights'
    )

    parser.add_argument(
        '--optimization-mode',
        choices=['validation', 'test', 'roc-only'],
        default='validation',
        help='Threshold optimization mode'
    )
    
    parser.add_argument(
        '--threshold-path',
        type=Path,
        help='Path to pre-computed threshold file'
    )

    parser.add_argument(
        '--architecture',
        choices=['gigapath', 'resnet18', 'swin_v2_b', 'convnext_large', 'densenet121', 'densenet169'],
        default='gigapath',
        help='Model architecture'
    )
    
    # Reproducibility options
    parser.add_argument(
        '--seed', 
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Enable deterministic mode for reproducibility'
    )
    
    # Add path arguments
    parser = add_path_args(parser)
    
    return parser.parse_args()

def plot_roc_curves(df: pd.DataFrame, task: str, output_dir: Path) -> None:
    """Plot ROC curves for different hierarchical levels.
    
    Args:
        df: DataFrame with predictions
        task: Classification task
        output_dir: Directory to save plots
    """
    from sklearn.metrics import roc_curve, auc
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert logits to probabilities
    if 'raw_pred' in df.columns and (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1):
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
    else:
        probs = df['raw_pred'].values
    
    # Calculate ROC curve for tile level
    tile_fpr, tile_tpr, _ = roc_curve(df['label'], probs)
    tile_auc = auc(tile_fpr, tile_tpr)
    
    # Calculate ROC curve for slide/particle level
    if task == 'inflammation':
        slide_df = df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        if 'raw_pred' in slide_df.columns and (slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1):
            slide_probs = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
        else:
            slide_probs = slide_df['raw_pred'].values
        
        slide_fpr, slide_tpr, _ = roc_curve(slide_df['label'], slide_probs)
        slide_auc = auc(slide_fpr, slide_tpr)
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        plt.plot(tile_fpr, tile_tpr, label=f'Tile-level (AUC = {tile_auc:.3f})')
        plt.plot(slide_fpr, slide_tpr, label=f'Slide-level (AUC = {slide_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Hierarchical Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'roc_curves.png', dpi=300)
        plt.close()
    
    else:  # tissue task
        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        if 'raw_pred' in particle_df.columns and (particle_df['raw_pred'].min() < 0 or particle_df['raw_pred'].max() > 1):
            particle_probs = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
        else:
            particle_probs = particle_df['raw_pred'].values
        
        particle_fpr, particle_tpr, _ = roc_curve(particle_df['label'], particle_probs)
        particle_auc = auc(particle_fpr, particle_tpr)
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        plt.plot(tile_fpr, tile_tpr, label=f'Tile-level (AUC = {tile_auc:.3f})')
        plt.plot(particle_fpr, particle_tpr, label=f'Particle-level (AUC = {particle_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Hierarchical Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'roc_curves.png', dpi=300)
        plt.close()

def plot_precision_recall_curve(df: pd.DataFrame, task: str, output_dir: Path) -> None:
    """Plot precision-recall curves for different hierarchical levels.
    
    Args:
        df: DataFrame with predictions
        task: Classification task
        output_dir: Directory to save plots
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert logits to probabilities
    if 'raw_pred' in df.columns and (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1):
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
    else:
        probs = df['raw_pred'].values
    
    # Calculate precision-recall curve for tile level
    tile_precision, tile_recall, _ = precision_recall_curve(df['label'], probs)
    tile_ap = average_precision_score(df['label'], probs)
    
    # Calculate precision-recall curve for slide/particle level
    if task == 'inflammation':
        slide_df = df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        if 'raw_pred' in slide_df.columns and (slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1):
            slide_probs = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
        else:
            slide_probs = slide_df['raw_pred'].values
        
        slide_precision, slide_recall, _ = precision_recall_curve(slide_df['label'], slide_probs)
        slide_ap = average_precision_score(slide_df['label'], slide_probs)
        
        # Plot precision-recall curves
        plt.figure(figsize=(10, 8))
        plt.plot(tile_recall, tile_precision, label=f'Tile-level (AP = {tile_ap:.3f})')
        plt.plot(slide_recall, slide_precision, label=f'Slide-level (AP = {slide_ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Different Hierarchical Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'precision_recall_curves.png', dpi=300)
        plt.close()
    
    else:  # tissue task
        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        if 'raw_pred' in particle_df.columns and (particle_df['raw_pred'].min() < 0 or particle_df['raw_pred'].max() > 1):
            particle_probs = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
        else:
            particle_probs = particle_df['raw_pred'].values
        
        particle_precision, particle_recall, _ = precision_recall_curve(particle_df['label'], particle_probs)
        particle_ap = average_precision_score(particle_df['label'], particle_probs)
        
        # Plot precision-recall curves
        plt.figure(figsize=(10, 8))
        plt.plot(tile_recall, tile_precision, label=f'Tile-level (AP = {tile_ap:.3f})')
        plt.plot(particle_recall, particle_precision, label=f'Particle-level (AP = {particle_ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Different Hierarchical Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'precision_recall_curves.png', dpi=300)
        plt.close()

def main():
    args = parse_args()
    
    # Get project paths with any overrides from command line
    paths = get_project_paths(base_dir=args.base_dir)
    
    # Override specific directories if provided
    if args.data_dir:
        paths["DATA_DIR"] = args.data_dir
        paths["RAW_DIR"] = args.data_dir / "raw"
        paths["PROCESSED_DIR"] = args.data_dir / "processed" 
        paths["SPLITS_DIR"] = args.data_dir / "splits"
    
    if args.output_dir:
        paths["RESULTS_DIR"] = args.output_dir
    else:
        paths["RESULTS_DIR"] = paths["BASE_DIR"] / "results"
    
    # Create evaluation directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = paths["RESULTS_DIR"] / "evaluations" / f'{args.task}_{args.test_split}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logging.info(f"Starting evaluation for {args.task} task on {args.test_split} split")
    
    # Set random seeds for reproducibility
    # Import here to avoid circular imports
    from src.models.training_utils import set_all_seeds
    set_all_seeds(args.seed)
    logging.info(f"Random seed set to {args.seed} for reproducible evaluation")
    
    if args.deterministic:
        logging.info("Using deterministic mode for CUDA operations")
    
    # Log paths
    logging.info("Project paths:")
    for path_name, path_value in paths.items():
        logging.info(f"  {path_name}: {path_value}")
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Load model
        model = load_model(args.model_path, device, args.architecture)
        
        # ===== FIXED: Always use strictly separate validation data =====
        # Create validation data directory
        validation_dir = output_dir / "validation_thresholds"
        validation_dir.mkdir(exist_ok=True)
        
        # Read pre-computed thresholds if provided - using pre-computed thresholds is preferred
        # for scientific reproducibility and strict separation of validation/test data
        validation_thresholds = None
        
        if args.threshold_path and Path(args.threshold_path).exists():
            try:
                with open(args.threshold_path, 'r') as f:
                    validation_thresholds = json.load(f)
                logging.info(f"Loaded validation thresholds from {args.threshold_path}")
                logging.info("Using pre-computed thresholds ensures proper validation/test separation")
            except Exception as e:
                logging.error(f"Error loading threshold file: {str(e)}")
                
        # If no valid thresholds were loaded, calculate them on validation data
        if validation_thresholds is None:
            logging.info("No pre-computed thresholds found. Calculating using validation data only...")
            
            validation_thresholds = metrics_utils.calculate_validation_thresholds(
                model_path=args.model_path,
                model=model,  # Pass the model directly
                task=args.task,
                output_dir=validation_dir
            )
            
            # Also save these thresholds for future use
            threshold_save_path = validation_dir / "thresholds.json"
            with open(threshold_save_path, 'w') as f:
                json.dump(validation_thresholds, f, indent=2)
            logging.info(f"Saved calculated validation thresholds to {threshold_save_path}")
            
        # Now evaluate on test data using validation-optimized thresholds
        logging.info(f"Evaluating on {args.test_split} using validation-optimized thresholds...")
        
        # Create dataset and dataloader for test data
        dataset = HistologyDataset(
            split=args.test_split,
            transform=get_transforms(is_training=False),
            task=args.task,
            paths=paths
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Run evaluation with validation thresholds
        metrics = evaluate_model(
            model=model,
            data_loader=dataloader,
            task=args.task,
            split=args.test_split,
            device=device,
            output_dir=output_dir,
            validation_thresholds=validation_thresholds  # Always provide validation thresholds
        )
        
        # Save metrics using metrics_utils
        # This will save both statistics.json and predictions.csv
        metrics_utils.save_metrics(metrics, output_dir)
        
        # Create ROC curves and precision-recall curves
        if 'predictions_df' in metrics:
            df = metrics['predictions_df']
            plot_roc_curves(df, args.task, output_dir)
        else:
            logging.warning("Could not find predictions DataFrame in metrics. Skipping ROC curve generation.")
        
        # Create precision-recall curves
        if 'predictions_df' in metrics:
            try:
                plot_precision_recall_curve(metrics['predictions_df'], args.task, output_dir)
                logging.info("Precision-recall curves saved successfully")
            except Exception as e:
                logging.warning(f"Failed to create precision-recall curves: {str(e)}")
        else:
            logging.warning("Could not find predictions DataFrame in metrics. Skipping precision-recall curve generation.")
        
        logging.info("Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()