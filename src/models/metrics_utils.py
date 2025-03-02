"""
Simplified metrics utility functions for histology classification evaluation.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve
)
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import HistologyDataset
import sys

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Use absolute import path
from src.models.dataset import HistologyDataset
from src.models.model_utils import load_model, get_transforms

def calculate_metrics(y_true, y_pred, y_scores=None, prefix=''):
    """Calculate binary classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_scores: Raw prediction scores for AUROC calculation
        prefix: Prefix for metric names
        
    Returns:
        Dictionary of metrics (values as raw proportions, not percentages)
    """
    metrics = {}
    
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        metrics[f'{prefix}accuracy'] = 0
        metrics[f'{prefix}sensitivity'] = 0  # Recall
        metrics[f'{prefix}specificity'] = 0
        metrics[f'{prefix}precision'] = 0
        metrics[f'{prefix}f1'] = 0
        if y_scores is not None:
            metrics[f'{prefix}auroc'] = 0.5
        return metrics
    
    # Handle single class case
    unique_classes = np.unique(y_true)
    if len(unique_classes) <= 1:
        # All samples are from a single class
        if unique_classes[0] == 1:  # All positives
            tp = np.sum(y_pred == 1)
            fn = np.sum(y_pred == 0)
            fp = 0
            tn = 0
        else:  # All negatives
            tp = 0
            fn = 0
            fp = np.sum(y_pred == 1)
            tn = np.sum(y_pred == 0)
            
        # Calculate metrics
        metrics[f'{prefix}accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics[f'{prefix}sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/sensitivity
        metrics[f'{prefix}specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics[f'{prefix}precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics[f'{prefix}f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        if y_scores is not None:
            metrics[f'{prefix}auroc'] = 0.5  # Default for single class
            
        return metrics
    
    # Normal case with multiple classes
    try:
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics (returns values between 0 and 1)
        metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred)
        metrics[f'{prefix}sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall/sensitivity
        metrics[f'{prefix}specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics[f'{prefix}precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'{prefix}f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate AUROC if scores are provided
        if y_scores is not None:
            try:
                metrics[f'{prefix}auroc'] = roc_auc_score(y_true, y_scores)
            except ValueError as e:
                logging.warning(f"Could not calculate AUROC: {str(e)}")
                metrics[f'{prefix}auroc'] = 0.5  # Default value for random classifier
    except Exception as e:
        logging.warning(f"Error calculating metrics: {str(e)}")
        # Provide default values
        metrics[f'{prefix}accuracy'] = 0
        metrics[f'{prefix}sensitivity'] = 0
        metrics[f'{prefix}specificity'] = 0
        metrics[f'{prefix}precision'] = 0
        metrics[f'{prefix}f1'] = 0
        if y_scores is not None:
            metrics[f'{prefix}auroc'] = 0.5
    
    return metrics

def find_optimal_threshold(y_true: List, y_prob: List) -> Tuple[float, float, float]:
    """Find optimal classification threshold using geometric mean of sensitivity and specificity.
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities or raw scores
        
    Returns:
        Tuple containing (optimal_threshold, sensitivity, specificity)
    """
    # Get false positive rate, true positive rate and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Calculate the geometric mean of sensitivity and specificity
    gmeans = np.sqrt(tpr * (1-fpr))
    
    # Find the optimal threshold
    ix = np.argmax(gmeans)
    optimal_threshold = thresholds[ix]
    sensitivity = tpr[ix]
    specificity = 1-fpr[ix]
    
    # Ensure numerical consistency by recalculating metrics at the chosen threshold
    predictions = (np.array(y_prob) > optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    
    # Recalculate sensitivity and specificity
    recalc_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    recalc_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Check for inconsistency due to interpolation in ROC curve calculation
    if abs(recalc_sensitivity - sensitivity) > 1e-6 or abs(recalc_specificity - specificity) > 1e-6:
        logging.warning("Metrics inconsistency detected in threshold optimization.")
        logging.warning(f"ROC curve values: sens={sensitivity:.6f}, spec={specificity:.6f}")
        logging.warning(f"Recalculated: sens={recalc_sensitivity:.6f}, spec={recalc_specificity:.6f}")
        
        # Use the recalculated values for consistency
        sensitivity = recalc_sensitivity
        specificity = recalc_specificity
    
    return optimal_threshold, sensitivity, specificity

def calculate_basic_metrics(
    labels: List,
    preds: List,
    raw_preds: Optional[List] = None
) -> Dict:
    """Calculate basic classification metrics.
    
    Args:
        labels: Ground truth labels
        preds: Binary predictions
        raw_preds: Raw prediction scores/logits for AUC calculation
        
    Returns:
        Dict containing metrics (as percentages)
    """
    metrics = {}
    
    # Convert to numpy arrays if needed
    labels = np.array(labels)
    preds = np.array(preds)
    
    # Handle empty arrays or arrays with single class
    if len(labels) == 0 or len(preds) == 0:
        logging.warning("Empty labels or predictions array")
        metrics['accuracy'] = 0
        metrics['sensitivity'] = 0
        metrics['specificity'] = 0
        metrics['precision'] = 0
        metrics['f1'] = 0
        if raw_preds is not None:
            metrics['auc'] = 50  # Default for random classifier
        return metrics
    
    # Check if only one class is present
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        # If only one class (all positives or all negatives)
        if unique_labels[0] == 1:  # all positives
            # All true positives if predictions match, else false negatives
            tp = np.sum(preds == 1)
            fn = np.sum(preds == 0)
            fp = 0
            tn = 0
        else:  # all negatives
            # All true negatives if predictions match, else false positives
            tn = np.sum(preds == 0)
            fp = np.sum(preds == 1)
            tp = 0
            fn = 0
    else:
        # Calculate confusion matrix normally when both classes are present
        cm = confusion_matrix(labels, preds)
        if cm.size == 4:  # 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle unexpected matrix shape (should not happen)
            logging.warning(f"Unexpected confusion matrix shape: {cm.shape}")
            metrics['accuracy'] = np.mean(labels == preds) * 100
            metrics['sensitivity'] = 0
            metrics['specificity'] = 0
            metrics['precision'] = 0
            metrics['f1'] = 0
            if raw_preds is not None:
                metrics['auc'] = 50
            return metrics
    
    # Calculate metrics as percentages
    metrics['accuracy'] = 100 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    metrics['sensitivity'] = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['precision'] = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Calculate F1 score
    if tp + fp + fn == 0:
        metrics['f1'] = 0
    else:
        metrics['f1'] = 100 * (2 * tp) / (2 * tp + fp + fn)
    
    # Calculate AUC if raw predictions are provided
    if raw_preds is not None:
        try:
            metrics['auc'] = 100 * roc_auc_score(labels, raw_preds)
        except ValueError as e:
            logging.warning(f"Could not calculate AUC: {str(e)}")
            metrics['auc'] = 50  # Default for random classifier
    
    return metrics

def calculate_hierarchical_metrics(
    df: pd.DataFrame,
    split: str,
    model_name: str,
    threshold: float = 0.0  # Default threshold for logits
) -> Dict:
    """Calculate metrics at appropriate hierarchical levels based on metadata.
    
    Always calculates tile-level metrics for both tasks.
    For inflammation task: also calculates slide-level metrics.
    For tissue task: also calculates particle-level metrics.
    
    Args:
        df: DataFrame with predictions and metadata
        split: Dataset split name
        model_name: Model name
        threshold: Threshold for binary classification
        
    Returns:
        Dict with metrics at all required levels
    """
    metrics = {}
    
    # Determine task based on metadata columns
    is_inflammation_task = 'inflammation_type' in df.columns or any('inflammation' in col for col in df.columns)
    
    # ALWAYS calculate tile-level metrics for both tasks
    tile_preds = (df['raw_pred'] > threshold).astype(int)
    
    # For ROC-AUC, convert to probabilities if we have logits
    if threshold == 0.0:  # If we're using logits
        scores = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
    else:  # If we already have probabilities
        scores = df['raw_pred'].values
    
    tile_metrics = calculate_metrics(
        y_true=df['label'],
        y_pred=tile_preds,
        y_scores=scores,
        prefix='tile_'
    )
    metrics.update(tile_metrics)
    
    # For inflammation task: ALWAYS calculate slide-level metrics
    if is_inflammation_task:
        slide_df = df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'  # All tiles from a slide have the same label
        })
        
        slide_preds = (slide_df['raw_pred'] > threshold).astype(int)
        if threshold == 0.0:
            slide_scores = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
        else:
            slide_scores = slide_df['raw_pred'].values
            
        slide_metrics = calculate_metrics(
            y_true=slide_df['label'],
            y_pred=slide_preds,
            y_scores=slide_scores,
            prefix='slide_'
        )
        metrics.update(slide_metrics)
        
    # For tissue task: ALWAYS calculate particle-level metrics
    else:  # tissue task
        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'  # All tiles from a particle have the same label
        })
        
        particle_preds = (particle_df['raw_pred'] > threshold).astype(int)
        if threshold == 0.0:
            particle_scores = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
        else:
            particle_scores = particle_df['raw_pred'].values
            
        particle_metrics = calculate_metrics(
            y_true=particle_df['label'],
            y_pred=particle_preds,
            y_scores=particle_scores,
            prefix='particle_'
        )
        metrics.update(particle_metrics)
    
    # Add metadata and preserve the original dataframe
    metrics.update({
        'split': split,
        'model_name': model_name,
        'threshold': threshold,
        'total_samples': len(df),
        'task': 'inflammation' if is_inflammation_task else 'tissue',
        'predictions_df': df  # Store the original predictions DataFrame
    })
    
    return metrics

def optimize_hierarchical_thresholds(df: pd.DataFrame, task: str = 'inflammation', output_dir: Optional[Path] = None) -> Dict:
    """Find optimal thresholds for each hierarchical level.
    
    IMPORTANT: This function should ONLY be used with validation data, never with test data,
    to avoid threshold optimization bias in final evaluation.
    
    Args:
        df: DataFrame with model predictions from VALIDATION data only
        task: 'inflammation' or 'tissue' 
        output_dir: Directory to save visualization plots
        
    Returns:
        Dictionary with optimal thresholds and metrics for each level
    """
    results = {}
    
    # Always optimize tile-level
    if 'raw_pred' in df.columns:
        # Convert logits to probabilities if needed
        if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
            tile_probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        else:
            tile_probs = df['raw_pred'].values
            
        # Find optimal threshold using gmean
        opt_threshold, sensitivity, specificity = find_optimal_threshold(
            df['label'].values, tile_probs
        )
        
        # Calculate all metrics at this threshold
        tile_preds = (tile_probs > opt_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(df['label'].values, tile_preds).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = f1_score(df['label'].values, tile_preds, zero_division=0)
        
        results['tile'] = {
            'threshold': float(opt_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f1': float(f1)
        }
        
        # Visualize if output directory provided
        if output_dir:
            visualize_threshold_optimization(
                df['label'].values, 
                tile_probs, 
                opt_threshold, 
                'tile', 
                output_dir
            )
    
    # Task-specific higher level optimization
    if task == 'inflammation':
        # Slide-level optimization
        slide_df = df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        # Convert to probabilities if needed
        if slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1:
            slide_probs = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
        else:
            slide_probs = slide_df['raw_pred'].values
        
        # Find optimal threshold
        opt_threshold, sensitivity, specificity = find_optimal_threshold(
            slide_df['label'].values, slide_probs
        )
        
        # Calculate metrics at this threshold
        slide_preds = (slide_probs > opt_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(slide_df['label'].values, slide_preds).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = f1_score(slide_df['label'].values, slide_preds, zero_division=0)
        
        results['slide'] = {
            'threshold': float(opt_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f1': float(f1)
        }
        
        # Visualize if output directory provided
        if output_dir:
            visualize_threshold_optimization(
                slide_df['label'].values, 
                slide_probs, 
                opt_threshold, 
                'slide', 
                output_dir
            )
    
    else:  # tissue task
        # Particle-level optimization
        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        # Convert to probabilities if needed
        if particle_df['raw_pred'].min() < 0 or particle_df['raw_pred'].max() > 1:
            particle_probs = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
        else:
            particle_probs = particle_df['raw_pred'].values
        
        # Find optimal threshold
        opt_threshold, sensitivity, specificity = find_optimal_threshold(
            particle_df['label'].values, particle_probs
        )
        
        # Calculate metrics at this threshold
        particle_preds = (particle_probs > opt_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(particle_df['label'].values, particle_preds).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = f1_score(particle_df['label'].values, particle_preds, zero_division=0)
        
        results['particle'] = {
            'threshold': float(opt_threshold),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'accuracy': float(accuracy),
            'f1': float(f1)
        }
        
        # Visualize if output directory provided
        if output_dir:
            visualize_threshold_optimization(
                particle_df['label'].values, 
                particle_probs, 
                opt_threshold, 
                'particle', 
                output_dir
            )
    
    return results

def optimize_aggregation_strategy(df: pd.DataFrame, task: str = 'inflammation', 
                                output_dir: Optional[Path] = None) -> Dict:
    """Find optimal aggregation strategy for hierarchical predictions.
    
    IMPORTANT: This function should ONLY be used with validation data, never with test data,
    to avoid optimization bias in final evaluation.
    
    Args:
        df: DataFrame with tile-level predictions from VALIDATION data only
        task: 'inflammation' (slide-level) or 'tissue' (particle-level)
        output_dir: Directory to save visualization plots
        
    Returns:
        Dictionary with optimal aggregation strategy and metrics
    """
    import torch
    
    # Convert logits to probabilities if needed
    if 'raw_pred' in df.columns and (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1):
        df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
    else:
        df['prob'] = df['raw_pred'].values if 'raw_pred' in df.columns else df['prob']
    
    # Define aggregation strategies to test
    aggregation_strategies = {
        'mean': lambda x: np.mean(x),
        'median': lambda x: np.median(x),
        'max': lambda x: np.max(x),
        'min': lambda x: np.min(x),
        'percentile_75': lambda x: np.percentile(x, 75),
        'percentile_90': lambda x: np.percentile(x, 90),
        'top_k_mean_10': lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.1)):]) if len(x) > 0 else 0,
        'top_k_mean_20': lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.2)):]) if len(x) > 0 else 0,
        'filter_90_mean': lambda x: np.mean(np.sort(x)[int(len(x)*0.9):]) if len(x) > 0 else 0,  # Filter bottom 90%, mean of top 10%
        'filter_80_mean': lambda x: np.mean(np.sort(x)[int(len(x)*0.8):]) if len(x) > 0 else 0,  # Filter bottom 80%, mean of top 20%
        'filter_70_mean': lambda x: np.mean(np.sort(x)[int(len(x)*0.7):]) if len(x) > 0 else 0,  # Filter bottom 70%, mean of top 30%
    }
    
    # Results dictionary to store metrics for each strategy
    results = {}
    
    # Determine the level to optimize based on task
    if task == 'inflammation':
        # For inflammation, optimize at slide level
        for strategy_name, agg_func in aggregation_strategies.items():
            # Regular aggregation
            slide_agg_df = df.groupby('slide_name').agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()
            
            # Find optimal threshold
            opt_threshold, sensitivity, specificity = find_optimal_threshold(
                slide_agg_df['label'].values, slide_agg_df['prob'].values
            )
            
            # Calculate metrics at this threshold
            slide_preds = (slide_agg_df['prob'] > opt_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(slide_agg_df['label'].values, slide_preds).ravel()
            
            # FIX: Calculate accuracy directly from confusion matrix
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            # Use scikit-learn for F1 to handle edge cases
            f1 = f1_score(slide_agg_df['label'].values, slide_preds, zero_division=0)
            
            # Record all metrics with confusion matrix elements
            results[strategy_name] = {
                'threshold': float(opt_threshold),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'accuracy': float(accuracy),
                'f1': float(f1),
                'balanced_acc': float((sensitivity + specificity) / 2),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
    
    else:  # tissue task
        # For tissue, optimize at particle level
        for strategy_name, agg_func in aggregation_strategies.items():
            # Regular aggregation
            particle_agg_df = df.groupby(['slide_name', 'particle_id']).agg({
                'prob': agg_func,
                'label': 'first'
            }).reset_index()
            
            # Find optimal threshold
            opt_threshold, sensitivity, specificity = find_optimal_threshold(
                particle_agg_df['label'].values, particle_agg_df['prob'].values
            )
            
            # Calculate metrics at this threshold
            particle_preds = (particle_agg_df['prob'] > opt_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(particle_agg_df['label'].values, particle_preds).ravel()
            
            # FIX: Calculate accuracy directly from confusion matrix
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            # Use scikit-learn for F1 to handle edge cases
            f1 = f1_score(particle_agg_df['label'].values, particle_preds, zero_division=0)
            
            # Record all metrics with confusion matrix elements
            results[strategy_name] = {
                'threshold': float(opt_threshold),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'accuracy': float(accuracy),
                'f1': float(f1),
                'balanced_acc': float((sensitivity + specificity) / 2),
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            }
    
    # Find the best strategy based on F1 score
    best_strategy = max(results.items(), key=lambda x: x[1]['f1'])
    
    # Log detailed information about all strategies
    logging.info("\nAggregation Strategy Comparison:")
    logging.info("--------------------------------")
    # Sort strategies by F1 score in descending order
    sorted_strategies = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for strategy_name, metrics in sorted_strategies:
        logging.info(f"{strategy_name}:")
        logging.info(f"  F1: {metrics['f1']:.4f}, Sens: {metrics['sensitivity']:.4f}, " +
                   f"Spec: {metrics['specificity']:.4f}, Acc: {metrics['accuracy']:.4f}")
    
    # Create visualization if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        level = 'slide' if task == 'inflammation' else 'particle'
        
        # Plot metrics for different strategies
        plt.figure(figsize=(14, 8))
        strategies = list(results.keys())
        
        # Extract metrics
        metrics_to_plot = {
            'F1 Score': [results[s]['f1'] for s in strategies],
            'Sensitivity': [results[s]['sensitivity'] for s in strategies],
            'Specificity': [results[s]['specificity'] for s in strategies],
            'Balanced Acc': [results[s]['balanced_acc'] for s in strategies],
            'Accuracy': [results[s]['accuracy'] for s in strategies]  # Added accuracy to plot
        }
        
        # Plot as bar chart with grouped bars
        x = np.arange(len(strategies))
        width = 0.17  # Adjusted for 5 metrics
        multiplier = 0
        
        for metric_name, values in metrics_to_plot.items():
            offset = width * multiplier
            plt.bar(x + offset, values, width, label=metric_name)
            multiplier += 1
        
        # Highlight best strategy with a vertical line
        best_idx = strategies.index(best_strategy[0])
        plt.axvline(x=best_idx, color='r', alpha=0.3, linestyle='--', 
                   label=f'Best Strategy: {best_strategy[0]}')
        
        plt.ylabel('Score')
        plt.title(f'Metrics by Aggregation Strategy for {level.capitalize()} Level')
        plt.xticks(x + width*2, strategies, rotation=45, ha='right')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_dir / f'{level}_aggregation_comparison.png', bbox_inches='tight')
        plt.close()
    
    return {
        'best_strategy': best_strategy[0],
        'best_metrics': best_strategy[1],
        'all_results': results
    }

def visualize_threshold_optimization(y_true: List, y_prob: List, optimal_threshold: float, 
                                   level: str, output_dir: Path) -> None:
    """Create visualization of threshold optimization results.
    
    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities
        optimal_threshold: Optimal threshold identified
        level: Hierarchical level ('tile', 'slide', or 'particle')
        output_dir: Directory to save visualization
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Test different thresholds
    thresholds = np.linspace(0.1, 0.9, 100)
    metrics = {
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'f1': [],
        'gmean': []
    }
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        preds = (y_prob > threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate F1 score directly to avoid warnings
        if tp + fp + fn == 0:
            f1 = 0
        else:
            f1 = 2 * tp / (2 * tp + fp + fn)
        
        # Calculate geometric mean
        gmean = np.sqrt(sensitivity * specificity)
        
        # Store metrics
        metrics['accuracy'].append(accuracy)
        metrics['sensitivity'].append(sensitivity)
        metrics['specificity'].append(specificity)
        metrics['f1'].append(f1)
        metrics['gmean'].append(gmean)
    
    # Plot metrics
    for metric_name, values in metrics.items():
        plt.plot(thresholds, values, label=metric_name.capitalize())
    
    # Add vertical line for optimal threshold
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
               label=f'Optimal ({optimal_threshold:.3f})')
    
    # Add labels and title
    plt.title(f'Metrics vs Threshold for {level.capitalize()} Level')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.savefig(output_dir / f'{level}_threshold_optimization.png')
    plt.close()

def save_metrics(metrics: Dict, output_dir: Path) -> None:
    """Save evaluation metrics to JSON file and create human-readable summary.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save metrics
    """
    # Create a copy of metrics to avoid modifying the original
    metrics_copy = {}
    
    # Convert DataFrames to dict or list to make them JSON serializable
    for key, value in metrics.items():
        if hasattr(value, 'to_dict'):  # Check if it's a DataFrame or Series
            if hasattr(value, 'to_records'):  # DataFrame
                metrics_copy[key] = value.to_dict(orient='records')
            else:  # Series
                metrics_copy[key] = value.to_dict()
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            metrics_copy[key] = {}
            for sub_key, sub_value in value.items():
                if hasattr(sub_value, 'to_dict'):  # Check for DataFrame or Series
                    if hasattr(sub_value, 'to_records'):  # DataFrame
                        metrics_copy[key][sub_key] = sub_value.to_dict(orient='records')
                    else:  # Series
                        metrics_copy[key][sub_key] = sub_value.to_dict()
                else:
                    metrics_copy[key][sub_key] = sub_value
        else:
            metrics_copy[key] = value
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON format - use a standard name for the statistics.json file
    # This is used by the compare_roc_curves.py script
    stats_path = output_dir / "statistics.json"
    
    with open(stats_path, 'w') as f:
        json.dump(metrics_copy, f, indent=2)
    
    logging.info(f"Metrics saved to: {stats_path}")
    
    # Also save with timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_file = output_dir / f"metrics_{timestamp}.json"
    
    with open(timestamped_file, 'w') as f:
        json.dump(metrics_copy, f, indent=2)
    
    # Save DataFrame predictions if present
    if 'predictions_df' in metrics:
        predictions_path = output_dir / "predictions.csv"
        metrics['predictions_df'].to_csv(predictions_path, index=False)
        logging.info(f"Predictions saved to: {predictions_path}")
    
    # Create human-readable summary
    create_summary_file(metrics, output_dir)

def plot_roc_curves(df: pd.DataFrame, task: str, output_dir: Path) -> None:
    """Plot ROC curves for different hierarchical levels.
    
    Args:
        df: DataFrame with predictions
        task: Classification task
        output_dir: Directory to save plots
    """
    from sklearn.metrics import roc_curve, auc
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert logits to probabilities
    if 'raw_pred' in df.columns and (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1):
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
    else:
        probs = df['raw_pred'].values
    
    # Calculate ROC curve for tile level
    tile_fpr, tile_tpr, _ = roc_curve(df['label'], probs)
    tile_auc = auc(tile_fpr, tile_tpr)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot tile-level ROC
    plt.plot(tile_fpr, tile_tpr, 
             label=f'Tile-level (AUC = {tile_auc:.3f})',
             linewidth=2)
    
    # Calculate and plot higher-level ROC (slide or particle)
    if task == 'inflammation':
        # Slide-level analysis
        slide_df = df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        # Convert to probabilities if needed
        if 'raw_pred' in slide_df.columns and (slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1):
            slide_probs = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
        else:
            slide_probs = slide_df['raw_pred'].values
        
        # Calculate ROC
        slide_fpr, slide_tpr, _ = roc_curve(slide_df['label'], slide_probs)
        slide_auc = auc(slide_fpr, slide_tpr)
        
        # Plot
        plt.plot(slide_fpr, slide_tpr, 
                 label=f'Slide-level (AUC = {slide_auc:.3f})',
                 linewidth=2)
    
    else:  # tissue task
        # Particle-level analysis
        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        # Convert to probabilities if needed
        if 'raw_pred' in particle_df.columns and (particle_df['raw_pred'].min() < 0 or particle_df['raw_pred'].max() > 1):
            particle_probs = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
        else:
            particle_probs = particle_df['raw_pred'].values
        
        # Calculate ROC
        particle_fpr, particle_tpr, _ = roc_curve(particle_df['label'], particle_probs)
        particle_auc = auc(particle_fpr, particle_tpr)
        
        # Plot
        plt.plot(particle_fpr, particle_tpr, 
                 label=f'Particle-level (AUC = {particle_auc:.3f})',
                 linewidth=2)
    
    # Finishing touches
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves for {task.capitalize()} Classification', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save figure
    try:
        save_path = output_dir / f'{task}_roc_curves.png'
        plt.savefig(save_path, dpi=300)
        logging.info(f"Successfully saved ROC curve visualization to {save_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to save ROC visualization: {str(e)}")
        plt.close()

def validate_metrics_consistency(metrics: Dict) -> Dict:
    """Validate and ensure consistency among classification metrics.
    
    This function checks for mathematical consistency between sensitivity, specificity,
    and accuracy. It adds diagnostic information and corrects inconsistencies if detected.
    
    Args:
        metrics: Dictionary containing calculated metrics
        
    Returns:
        Dictionary with validated and consistent metrics
    """
    # Create a copy of the metrics to avoid modifying the original
    validated_metrics = metrics.copy()
    
    # Extract metrics for each hierarchical level
    for level in ['tile', 'slide', 'particle']:
        # Check if this level exists in the metrics
        sensitivity_key = f'{level}_sensitivity'
        specificity_key = f'{level}_specificity'
        accuracy_key = f'{level}_accuracy'
        
        if sensitivity_key in metrics and specificity_key in metrics and accuracy_key in metrics:
            sens = metrics[sensitivity_key]
            spec = metrics[specificity_key]
            acc = metrics[accuracy_key]
            
            # Check for perfect classification case (sens=1, spec=1)
            if abs(sens - 1.0) < 1e-6 and abs(spec - 1.0) < 1e-6:
                # In this case, accuracy must also be 1.0
                if abs(acc - 1.0) > 1e-6:
                    logging.warning(f"Inconsistency detected: {level} level has perfect "
                                   f"sensitivity ({sens:.6f}) and specificity ({spec:.6f}), "
                                   f"but accuracy is {acc:.6f}")
                    
                    # Correct the accuracy
                    validated_metrics[accuracy_key] = 1.0
                    
                    # Add diagnostic information
                    validated_metrics[f'{level}_metrics_corrected'] = True
                    validated_metrics[f'{level}_original_accuracy'] = acc
    
    # Add validation timestamp
    validated_metrics['validation_timestamp'] = datetime.now().isoformat()
    validated_metrics['metrics_validated'] = True
    
    return validated_metrics


def calculate_validation_thresholds(model_path: Optional[Path] = None, model: Optional[nn.Module] = None, 
                             task: str = None, output_dir: Optional[Path] = None) -> Dict:
    """Calculate optimal thresholds using validation data only.
    
    Args:
        model_path: Path to trained model (optional if model is provided directly)
        model: Trained model (optional if model_path is provided)
        task: Classification task ('inflammation' or 'tissue')
        output_dir: Directory to save thresholds
        
    Returns:
        Dictionary with optimal thresholds for each level
    """
    # Validate inputs - either model_path or model must be provided
    if model_path is None and model is None:
        raise ValueError("Either model_path or model must be provided")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model if not provided directly
    if model is None:
        model = load_model(model_path, device)
    else:
        # Ensure model is on the correct device
        model = model.to(device)
    
    # Create validation dataset and dataloader
    val_dataset = HistologyDataset(
        split='val',
        transform=get_transforms(is_training=False),
        task=task
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Collect predictions on validation set
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs, labels, metadata_batch in tqdm(val_loader, desc="Collecting validation predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            
            for i in range(len(outputs)):
                metadata = {k: v[i] if isinstance(v, list) else v for k, v in metadata_batch.items()}
                
                pred_dict = {
                    'raw_pred': outputs[i].cpu().item(),
                    'label': labels[i].item(),
                    'particle_id': metadata['particle_id'],
                    'slide_name': metadata['slide_name'],
                }
                
                if task == 'inflammation':
                    pred_dict['inflammation_type'] = metadata.get('inflammation_type', 'unknown')
                
                predictions.append(pred_dict)
    
    # Convert to DataFrame
    val_df = pd.DataFrame(predictions)
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate optimal thresholds on validation data
    optimal_thresholds = optimize_hierarchical_thresholds(
        df=val_df,
        task=task,
        output_dir=output_dir
    )

    optimal_aggregation = optimize_aggregation_strategy(
        df=val_df,
        task=task,
        output_dir=output_dir
    )
    
    validation_optimized = {
        **optimal_thresholds,
        "optimal_aggregation": optimal_aggregation  # Add aggregation strategy to results
    }

    # Save thresholds to file if output directory is provided
    if output_dir:
        thresholds_file = output_dir / "validation_thresholds.json"
        with open(thresholds_file, 'w') as f:
            json.dump(validation_optimized, f, indent=2)
            
        logging.info(f"Validation-optimized thresholds and aggregation saved to: {thresholds_file}")
    
    return validation_optimized

def create_summary_file(metrics: Dict, output_dir: Path):
    """Create a human-readable summary file of metrics."""
    summary_file = output_dir / "metrics_summary.txt"
    
    with open(summary_file, 'w') as f:
        # Write header
        f.write("Histology Classification Metrics Summary\n")
        f.write("=====================================\n\n")
        
        # Write basic information
        f.write(f"Task: {metrics.get('task', 'Unknown')}\n")
        f.write(f"Split: {metrics.get('split', 'Unknown')}\n")
        f.write(f"Model: {metrics.get('model_name', 'Unknown')}\n")
        f.write(f"Total Samples: {metrics.get('total_samples', 0)}\n")
        f.write(f"Threshold: {metrics.get('threshold', 0.5):.3f}\n\n")
        
        # Always include tile-level metrics
        f.write("TEST DATASET - TILE LEVEL METRICS\n")
        f.write("-" * 35 + "\n")
        metric_types = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auroc']
        for metric in metric_types:
            key = f'tile_{metric}'
            if key in metrics:
                value = metrics[key]
                if isinstance(value, (int, float)):
                    f.write(f"{metric.capitalize():10s}: {value:.2%}\n")
        f.write("\n")
        
        # Determine task and include appropriate higher-level metrics
        if metrics.get('task') == 'inflammation':
            # Include slide-level metrics for inflammation task
            f.write("TEST DATASET - SLIDE LEVEL METRICS\n")
            f.write("-" * 35 + "\n")
            for metric in metric_types:
                key = f'slide_{metric}'
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, (int, float)):
                        f.write(f"{metric.capitalize():10s}: {value:.2%}\n")
        else:  # tissue task
            # Include particle-level metrics for tissue task
            f.write("TEST DATASET - PARTICLE LEVEL METRICS\n")
            f.write("-" * 35 + "\n")
            for metric in metric_types:
                key = f'particle_{metric}'
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, (int, float)):
                        f.write(f"{metric.capitalize():10s}: {value:.2%}\n")
        
        # Include optimal threshold information if available
        if 'optimal_thresholds' in metrics:
            f.write("\nVALIDATION-OPTIMIZED THRESHOLDS\n")
            f.write("-" * 30 + "\n")
            for level, level_metrics in metrics['optimal_thresholds'].items():
                f.write(f"{level.capitalize()} level: {level_metrics.get('threshold', 0.5):.3f}\n")
                f.write(f"  Sensitivity: {level_metrics.get('sensitivity', 0):.2%}\n")
                f.write(f"  Specificity: {level_metrics.get('specificity', 0):.2%}\n")
                f.write(f"  F1 Score: {level_metrics.get('f1', 0):.2%}\n")
                f.write("\n")
        
        # Include optimal aggregation strategy if available
        if 'optimal_aggregation' in metrics:
            agg = metrics['optimal_aggregation']
            f.write("\nOPTIMAL AGGREGATION STRATEGY\n")
            f.write("-" * 28 + "\n")
            f.write(f"Strategy: {agg.get('best_strategy', 'unknown')}\n")
            
            best_metrics = agg.get('best_metrics', {})
            f.write(f"Threshold: {best_metrics.get('threshold', 0.5):.3f}\n")
            f.write(f"Sensitivity: {best_metrics.get('sensitivity', 0):.2%}\n")
            f.write(f"Specificity: {best_metrics.get('specificity', 0):.2%}\n")
            f.write(f"F1 Score: {best_metrics.get('f1', 0):.2%}\n")
            f.write(f"Balanced Accuracy: {best_metrics.get('balanced_acc', 0):.2%}\n")
            
            if 'validation_confusion_matrix' in agg:
                f.write("\nVALIDATION DATASET CONFUSION MATRIX (Optimized on this data):\n")
                f.write("-" * 55 + "\n")
                f.write(f"  True Positives: {agg['validation_confusion_matrix'].get('tp', 0)}\n")
                f.write(f"  True Negatives: {agg['validation_confusion_matrix'].get('tn', 0)}\n")
                f.write(f"  False Positives: {agg['validation_confusion_matrix'].get('fp', 0)}\n")
                f.write(f"  False Negatives: {agg['validation_confusion_matrix'].get('fn', 0)}\n")
                f.write(f"  Total Samples: {agg['validation_confusion_matrix'].get('total', 0)}\n")
            
            if 'test_confusion_matrix' in agg:
                f.write("\nTEST DATASET CONFUSION MATRIX (Using validation-optimized strategy):\n")
                f.write("-" * 60 + "\n")
                f.write(f"  True Positives: {agg['test_confusion_matrix'].get('tp', 0)}\n")
                f.write(f"  True Negatives: {agg['test_confusion_matrix'].get('tn', 0)}\n")
                f.write(f"  False Positives: {agg['test_confusion_matrix'].get('fp', 0)}\n")
                f.write(f"  False Negatives: {agg['test_confusion_matrix'].get('fn', 0)}\n")
                f.write(f"  Total Samples: {agg['test_confusion_matrix'].get('total', 0)}\n")
                f.write(f"  Accuracy: {agg['test_confusion_matrix'].get('accuracy', 0):.2%}\n")
                f.write(f"  F1 Score: {agg['test_confusion_matrix'].get('f1', 0):.2%}\n")

        # Include validation information if present
        if 'metrics_validated' in metrics and metrics['metrics_validated']:
            f.write("\nVALIDATION INFORMATION\n")
            f.write("-" * 23 + "\n")
            f.write(f"Validation timestamp: {metrics.get('validation_timestamp', 'unknown')}\n")
            
            # Report any corrections made
            corrections_found = False
            for level in ['tile', 'slide', 'particle']:
                correction_key = f'{level}_metrics_corrected'
                if correction_key in metrics and metrics[correction_key]:
                    if not corrections_found:
                        f.write("\nMetrics corrected for consistency:\n")
                        corrections_found = True
                    
                    f.write(f"  {level.capitalize()} level: ")
                    f.write(f"accuracy adjusted from {metrics.get(f'{level}_original_accuracy', 0):.4f} to 1.0000\n")
            
            if not corrections_found:
                f.write("No inconsistencies detected in metrics.\n")
                
    logging.info(f"Created summary file at: {summary_file}")
    return summary_file