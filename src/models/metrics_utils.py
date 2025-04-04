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
    # Basic data quality checks
    logging.info(f"Finding optimal threshold for {len(y_true)} samples")
    logging.info(f"Class distribution: {np.bincount(np.array(y_true, dtype=int))}")
    logging.info(f"Predictions range: min={min(y_prob):.4f}, max={max(y_prob):.4f}")
    
    # Check for empty inputs or only one class
    if len(y_true) == 0 or len(np.unique(y_true)) <= 1:
        logging.warning("find_optimal_threshold: Empty input or only one class present. Using default threshold of 0.5")
        return 0.5, 0.0, 0.0
        
    # Check for predictions all being the same value
    if max(y_prob) == min(y_prob):
        logging.warning(f"All predictions have the same value: {max(y_prob):.4f}. Using default threshold of 0.5")
        return 0.5, 0.0, 0.0
    
    # Convert to numpy arrays for better handling
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    
    # Basic check for NaN or Inf values
    if np.isnan(y_prob_np).any() or np.isinf(y_prob_np).any():
        logging.warning("NaN or Inf values detected in predictions. Cleaning data.")
        # Replace NaN/Inf with a safe value
        y_prob_np = np.nan_to_num(y_prob_np, nan=0.5, posinf=1.0, neginf=0.0)
        
    # Get false positive rate, true positive rate and thresholds
    try:
        fpr, tpr, thresholds = roc_curve(y_true_np, y_prob_np)
        
        # Calculate the geometric mean of sensitivity and specificity
        gmeans = np.sqrt(tpr * (1-fpr))
        
        # Log ROC curve statistics
        auc_score = roc_auc_score(y_true_np, y_prob_np)
        logging.info(f"ROC AUC: {auc_score:.4f}")
        
        # Find the optimal threshold with proper error handling
        if len(gmeans) > 0:
            ix = np.argmax(gmeans)
            optimal_threshold = thresholds[ix]
            
            # Directly compute metrics using the chosen threshold
            predictions = (y_prob_np > optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_np, predictions).ravel()
            
            # Calculate sensitivity and specificity directly
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Calculate additional metrics for logging
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            logging.info(f"Found optimal threshold: {optimal_threshold:.4f}")
            logging.info(f"Metrics at this threshold: Accuracy={accuracy:.4f}, Sens={sensitivity:.4f}, Spec={specificity:.4f}, F1={f1:.4f}")
            logging.info(f"Confusion matrix: [TP={tp}, TN={tn}, FP={fp}, FN={fn}]")
            
            # For logging purposes
            roc_sensitivity = tpr[ix] 
            roc_specificity = 1-fpr[ix]
            
            # Check for inconsistency due to interpolation in ROC curve calculation
            if abs(roc_sensitivity - sensitivity) > 1e-6 or abs(roc_specificity - specificity) > 1e-3:
                logging.warning("Metrics inconsistency detected in threshold optimization.")
                logging.warning(f"ROC curve values: sens={roc_sensitivity:.6f}, spec={roc_specificity:.6f}")
                logging.warning(f"Recalculated: sens={sensitivity:.6f}, spec={specificity:.6f}")
        else:
            # Handle the case where gmeans is empty (can happen with edge cases)
            logging.warning("Empty gmeans array in find_optimal_threshold. Using default threshold of 0.5")
            optimal_threshold = 0.5
            sensitivity = 0.0
            specificity = 0.0
    
    except Exception as e:
        logging.error(f"Error in ROC curve calculation: {str(e)}")
        logging.warning("Defaulting to threshold of 0.5")
        return 0.5, 0.0, 0.0
    
    # Enforce threshold bounds (0.1 to 0.9) to avoid extreme values that might overfit
    if optimal_threshold < 0.1:
        logging.warning(f"Threshold too low ({optimal_threshold:.4f}), adjusting to 0.1")
        optimal_threshold = 0.1
        # Recalculate metrics with the new threshold
        predictions = (y_prob_np > optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_np, predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    elif optimal_threshold > 0.9:
        logging.warning(f"Threshold too high ({optimal_threshold:.4f}), adjusting to 0.9")
        optimal_threshold = 0.9
        # Recalculate metrics with the new threshold
        predictions = (y_prob_np > optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_np, predictions).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
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
    
    # Calculate metrics as percentages with proper error handling
    total = tp + tn + fp + fn
    metrics['accuracy'] = 100 * (tp + tn) / total if total > 0 else 0
    
    # Handle division by zero for sensitivity (recall)
    if tp + fn > 0:
        metrics['sensitivity'] = 100 * tp / (tp + fn)
    else:
        metrics['sensitivity'] = 0 if fn > 0 else 100  # If no positives, max recall is 100%
    
    # Handle division by zero for specificity
    if tn + fp > 0:
        metrics['specificity'] = 100 * tn / (tn + fp)
    else:
        metrics['specificity'] = 0 if fp > 0 else 100  # If no negatives, max specificity is 100%
    
    # Handle division by zero for precision
    if tp + fp > 0:
        metrics['precision'] = 100 * tp / (tp + fp)
    else:
        metrics['precision'] = 0 if fp > 0 else 100  # No predictions as positive
    
    # Calculate F1 score with proper handling of edge cases
    if tp == 0 and (fp == 0 or fn == 0):
        # Edge case when no true positives and either no false positives or no false negatives
        metrics['f1'] = 0
    else:
        # Standard F1 calculation with denominator check
        denominator = 2 * tp + fp + fn
        metrics['f1'] = 100 * (2 * tp) / denominator if denominator > 0 else 0
    
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
    is_inflammation_task = 'inflammation_status' in df.columns or any('inflammation' in col for col in df.columns)
    
    # Check if we have logits or probabilities and log for debugging
    is_logits = df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1
    logging.info(f"Raw predictions range: min={df['raw_pred'].min():.4f}, max={df['raw_pred'].max():.4f}")
    logging.info(f"Predictions appear to be {'logits' if is_logits else 'probabilities'}")
    logging.info(f"Using threshold: {threshold:.4f} in {'logit' if is_logits else 'probability'} space")
    
    # ALWAYS calculate tile-level metrics for both tasks
    tile_preds = (df['raw_pred'] > threshold).astype(int)
    
    # Do a sanity check on predictions
    total_pos = tile_preds.sum()
    total_neg = len(tile_preds) - total_pos
    logging.info(f"Binary predictions: {total_pos} positive, {total_neg} negative out of {len(tile_preds)} total")
    
    # For ROC-AUC, convert to probabilities if we have logits (negative values)
    # Check minimum value as indicator of logits vs probabilities
    if is_logits:  # Using the same check as above
        scores = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        logging.info(f"Converted logits to probabilities for AUC calculation")
    else:  # Already have probabilities
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
        
        # Use the same logic for slide-level predictions
        is_slide_logits = slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1
        logging.info(f"Slide-level raw predictions range: min={slide_df['raw_pred'].min():.4f}, max={slide_df['raw_pred'].max():.4f}")
        
        slide_preds = (slide_df['raw_pred'] > threshold).astype(int)
        
        # Log slide-level prediction distribution
        slide_pos = slide_preds.sum()
        slide_neg = len(slide_preds) - slide_pos
        logging.info(f"Slide-level binary predictions: {slide_pos} positive, {slide_neg} negative out of {len(slide_preds)} total")
        
        # Check if we have logits (negative values)
        if is_slide_logits:
            slide_scores = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
            logging.info(f"Converted slide-level logits to probabilities for AUC calculation")
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
        # Check if we have logits (negative values)
        if particle_df['raw_pred'].min() < 0:
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
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
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
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0
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
    
    # Carefully convert logits to probabilities if needed and cache it to ensure
    # consistency in probability calculation across all aggregation strategies
    if 'prob' not in df.columns:
        if 'raw_pred' in df.columns:
            if df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1:
                logging.info("Converting raw logits to probabilities with sigmoid in optimize_aggregation_strategy")
                df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
            else:
                logging.info("Raw predictions are already in probability range [0,1]")
                df['prob'] = df['raw_pred'].copy()
        else:
            logging.warning("No raw_pred column found in dataframe for optimize_aggregation_strategy")
            df['prob'] = 0.5  # Default fallback
    
    # Define aggregation strategies to test
    aggregation_strategies = {
        'mean': lambda x: np.mean(x),
        'median': lambda x: np.median(x),
        'top_k_mean_10': lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.1)):]) if len(x) > 0 else 0,
        'top_k_mean_20': lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.2)):]) if len(x) > 0 else 0,
        'top_k_mean_30': lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.3)):]) if len(x) > 0 else 0,
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
            
            # Calculate accuracy directly from confusion matrix with proper error handling
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            
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
            
            # Calculate accuracy directly from confusion matrix with proper error handling
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            
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
    
    # Find the best strategy based on balanced accuracy instead of F1 score
    # This is more suitable for class imbalance and helps avoid overfitting to the majority class
    best_strategy = max(results.items(), key=lambda x: x[1]['balanced_acc'])
    
    # Log detailed information about all strategies
    logging.info("\nAggregation Strategy Comparison:")
    logging.info("--------------------------------")
    # Sort strategies by balanced accuracy in descending order
    sorted_strategies = sorted(results.items(), key=lambda x: x[1]['balanced_acc'], reverse=True)
    
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
        
        # Define metrics to plot
        metric_names = ['F1 Score', 'Sensitivity', 'Specificity', 'Balanced Acc', 'Accuracy']
        metric_keys = ['f1', 'sensitivity', 'specificity', 'balanced_acc', 'accuracy']
        
        # Restructure data to group by metrics instead of strategies
        strategies_to_plot = {}
        for strategy in strategies:
            strategies_to_plot[strategy] = [results[strategy][key] for key in metric_keys]
        
        # Plot as bar chart with grouped bars
        x = np.arange(len(metric_names))
        width = 0.15  # Adjust based on number of strategies
        if len(strategies) > 5:
            width = 0.12
        elif len(strategies) < 4:
            width = 0.2
            
        multiplier = 0
        
        # Create color map to highlight best strategy
        best_strategy_name = best_strategy[0]
        colors = {}
        for strategy in strategies:
            if strategy == best_strategy_name:
                colors[strategy] = 'darkred'  # Highlight the best strategy
            else:
                colors[strategy] = None  # Use default color cycle
        
        for strategy, values in strategies_to_plot.items():
            offset = width * multiplier
            plt.bar(x + offset, values, width, label=strategy, 
                    color=colors[strategy], alpha=0.75 if strategy == best_strategy_name else 0.65)
            multiplier += 1
        
        plt.ylabel('Score')
        plt.title(f'Aggregation Strategies by Metric for {level.capitalize()} Level')
        plt.xticks(x + width*(len(strategies)-1)/2, metric_names, rotation=30, ha='right')
        
        # Add a star in the legend for the best strategy
        if len(strategies) > 1:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                    ncol=min(5, len(strategies)),
                    title=f"★ Best Strategy: {best_strategy_name} ★")
        else:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
        
        # Add horizontal grid lines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'optimal_aggregation' in metrics and 'test_confusion_matrix' in metrics['optimal_aggregation']:
        optimal_agg_auc = metrics['optimal_aggregation']['test_confusion_matrix'].get('auc', 0)
        
        # For inflammation, update slide_auroc
        if metrics.get('task') == 'inflammation' and 'slide_auroc' in metrics_copy:
            metrics_copy['slide_auroc'] = optimal_agg_auc
            logging.info(f"Updated slide_auroc in statistics.json to match optimal aggregation: {optimal_agg_auc:.4f}")
        
        # For tissue, update particle_auroc
        elif metrics.get('task') == 'tissue' and 'particle_auroc' in metrics_copy:
            metrics_copy['particle_auroc'] = optimal_agg_auc
            logging.info(f"Updated particle_auroc in statistics.json to match optimal aggregation: {optimal_agg_auc:.4f}")
    
    # Save JSON format - use a standard name for the statistics.json file
    # This is used by the compare_roc_curves.py script
    stats_path = output_dir / "statistics.json"
    
    with open(stats_path, 'w') as f:
        json.dump(metrics_copy, f, indent=2)
    
    logging.info(f"Metrics saved to: {stats_path}")
    
    # Save DataFrame predictions if present
    if 'predictions_df' in metrics:
        predictions_path = output_dir / "predictions.csv"
        metrics['predictions_df'].to_csv(predictions_path, index=False)
        logging.info(f"Predictions saved to: {predictions_path}")
    
    # Create human-readable summary
    summary_path = output_dir / "metrics_summary.txt"
    create_summary_file(metrics, summary_path)

def plot_roc_curves(df: pd.DataFrame, task: str, output_dir: Path, 
                   optimal_aggregation: Optional[Dict] = None,
                   metrics: Optional[Dict] = None):
    """Plot ROC curves using consistent AUC values from the metrics dictionary."""
    from sklearn.metrics import roc_curve, auc
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Force use of pre-calculated metrics if available
    use_precalculated = False
    if metrics and 'tile_auroc' in metrics:
        use_precalculated = True
        logging.info(f"Will use pre-calculated metrics: tile_auroc={metrics['tile_auroc']:.4f}")
    else:
        logging.warning("No pre-calculated metrics found, will calculate AUC values")
    
    # Get aggregation function
    agg_strategy = None
    if optimal_aggregation and 'best_strategy' in optimal_aggregation:
        agg_strategy = optimal_aggregation['best_strategy']
        
    # Define aggregation function
    if agg_strategy == 'median':
        agg_func = np.median
    elif agg_strategy == 'top_k_mean_10':
        agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.1)):]) if len(x) > 0 else 0
    elif agg_strategy == 'top_k_mean_20':
        agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.2)):]) if len(x) > 0 else 0
    elif agg_strategy == 'top_k_mean_30':
        agg_func = lambda x: np.mean(np.sort(x)[-int(max(1, len(x)*0.3)):]) if len(x) > 0 else 0
    else:
        agg_func = np.mean
    
    # Ensure probabilities are available
    if 'prob' not in df.columns:
        if 'raw_pred' in df.columns and (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1):
            df['prob'] = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
        else:
            df['prob'] = df['raw_pred'].values
    
    # Calculate ROC curves (regardless of whether we'll use pre-calculated AUC)
    tile_fpr, tile_tpr, _ = roc_curve(df['label'], df['prob'])
    
    # CRITICAL: Always use pre-calculated AUC when available
    if use_precalculated:
        tile_auc = metrics['tile_auroc']
        logging.info(f"Using pre-calculated tile AUC: {tile_auc:.4f}")
    else:
        tile_auc = auc(tile_fpr, tile_tpr)
        logging.info(f"Calculated tile AUC: {tile_auc:.4f} for ROC curve")
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot tile-level ROC with consistent AUC
    plt.plot(tile_fpr, tile_tpr, 
             label=f'Tile-level (AUC = {tile_auc:.3f})',
             linewidth=2)
    
    # Higher level metrics (slide or particle)
    if task == 'inflammation':
        # Slide-level aggregation
        slide_df = df.groupby('slide_name').agg({
            'prob': agg_func,
            'label': 'first'
        }).reset_index()
        
        slide_fpr, slide_tpr, _ = roc_curve(slide_df['label'], slide_df['prob'])
        
        # Priority for slide AUC:
        # 1. Use optimal_aggregation test_confusion_matrix auc if available
        # 2. Fall back to slide_auroc
        # 3. Calculate from data as last resort
        if use_precalculated:
            if (optimal_aggregation and 
                'test_confusion_matrix' in optimal_aggregation and
                'auc' in optimal_aggregation['test_confusion_matrix']):
                slide_auc = optimal_aggregation['test_confusion_matrix']['auc']
                logging.info(f"Using optimal aggregation AUC: {slide_auc:.4f}")
            elif 'slide_auroc' in metrics:
                slide_auc = metrics['slide_auroc']
                logging.info(f"Using pre-calculated slide AUC: {slide_auc:.4f}")
            else:
                slide_auc = auc(slide_fpr, slide_tpr)
                logging.info(f"Calculated slide AUC: {slide_auc:.4f} for ROC curve")
        else:
            slide_auc = auc(slide_fpr, slide_tpr)
            logging.info(f"Calculated slide AUC: {slide_auc:.4f} for ROC curve")
        
        strategy_label = f" using {agg_strategy}" if agg_strategy else ""
        plt.plot(slide_fpr, slide_tpr, 
                 label=f'Slide-level{strategy_label} (AUC = {slide_auc:.3f})',
                 linewidth=2)
    else:
        # Particle-level for tissue task
        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
            'prob': agg_func,
            'label': 'first'
        }).reset_index()
        
        particle_fpr, particle_tpr, _ = roc_curve(particle_df['label'], particle_df['prob'])
        
        # Priority for particle AUC:
        # 1. Use optimal_aggregation test_confusion_matrix auc if available
        # 2. Fall back to particle_auroc
        # 3. Calculate from data as last resort
        if use_precalculated:
            if (optimal_aggregation and 
                'test_confusion_matrix' in optimal_aggregation and
                'auc' in optimal_aggregation['test_confusion_matrix']):
                particle_auc = optimal_aggregation['test_confusion_matrix']['auc']
                logging.info(f"Using optimal aggregation AUC: {particle_auc:.4f}")
            elif 'particle_auroc' in metrics:
                particle_auc = metrics['particle_auroc']
                logging.info(f"Using pre-calculated particle AUC: {particle_auc:.4f}")
            else:
                particle_auc = auc(particle_fpr, particle_tpr)
                logging.info(f"Calculated particle AUC: {particle_auc:.4f} for ROC curve")
        else:
            particle_auc = auc(particle_fpr, particle_tpr)
            logging.info(f"Calculated particle AUC: {particle_auc:.4f} for ROC curve")
        
        strategy_label = f" using {agg_strategy}" if agg_strategy else ""
        plt.plot(particle_fpr, particle_tpr, 
                 label=f'Particle-level{strategy_label} (AUC = {particle_auc:.3f})',
                 linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves for {task.capitalize()} Classification (Test Data)', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    try:
        save_path = output_dir / f'{task}_test_roc_curves.png'
        plt.savefig(save_path, dpi=300)
        logging.info(f"Successfully saved test ROC curve visualization to {save_path}")
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
                    pred_dict['inflammation_status'] = metadata.get('inflammation_status', 'unknown')
                
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
    
    validation_auc_metrics = {}
    
    # Calculate tile-level AUC
    if 'raw_pred' in val_df.columns:
        # Convert logits to probabilities if needed
        if val_df['raw_pred'].min() < 0 or val_df['raw_pred'].max() > 1:
            tile_probs = torch.sigmoid(torch.tensor(val_df['raw_pred'].values)).numpy()
        else:
            tile_probs = val_df['raw_pred'].values
            
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        try:
            validation_auc_metrics['validation_tile_auroc'] = roc_auc_score(val_df['label'].values, tile_probs)
        except Exception as e:
            logging.warning(f"Could not calculate validation tile-level AUC: {str(e)}")
            validation_auc_metrics['validation_tile_auroc'] = 0.5
    
    # Calculate higher-level AUC based on task
    if task == 'inflammation':
        # Calculate slide-level AUC
        slide_df = val_df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        if 'raw_pred' in slide_df.columns:
            # Convert logits to probabilities if needed
            if slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1:
                slide_probs = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
            else:
                slide_probs = slide_df['raw_pred'].values
                
            # Calculate AUC
            try:
                validation_auc_metrics['validation_slide_auroc'] = roc_auc_score(slide_df['label'].values, slide_probs)
            except Exception as e:
                logging.warning(f"Could not calculate validation slide-level AUC: {str(e)}")
                validation_auc_metrics['validation_slide_auroc'] = 0.5
    else:  # tissue task
        # Calculate particle-level AUC
        particle_df = val_df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        if 'raw_pred' in particle_df.columns:
            # Convert logits to probabilities if needed
            if particle_df['raw_pred'].min() < 0 or particle_df['raw_pred'].max() > 1:
                particle_probs = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
            else:
                particle_probs = particle_df['raw_pred'].values
                
            # Calculate AUC
            try:
                validation_auc_metrics['validation_particle_auroc'] = roc_auc_score(particle_df['label'].values, particle_probs)
            except Exception as e:
                logging.warning(f"Could not calculate validation particle-level AUC: {str(e)}")
                validation_auc_metrics['validation_particle_auroc'] = 0.5
    
    validation_optimized = {
        **optimal_thresholds,
        "optimal_aggregation": optimal_aggregation,
        **validation_auc_metrics
    }

    # Save thresholds to file if output directory is provided
    if output_dir:
        thresholds_file = output_dir / "validation_thresholds.json"
        with open(thresholds_file, 'w') as f:
            json.dump(validation_optimized, f, indent=2)
            
        logging.info(f"Validation-optimized thresholds and aggregation saved to: {thresholds_file}")
    
    return validation_optimized

def create_summary_file(metrics, output_path):
    """
    Write a summary file with performance metrics, ensuring correct slide/particle-level test metrics.
    
    Args:
        metrics (dict): Dictionary containing performance metrics and aggregation results.
        output_path (str or Path): Path to save the summary file.
    """
    with open(output_path, 'w') as f:
        # Header Information
        task = metrics.get('task', 'N/A')
        f.write("Histology Classification Metrics Summary\n")
        f.write("=====================================\n")
        f.write(f"Task: {task}\n")
        f.write(f"Split: {metrics.get('split', 'N/A')}\n")
        f.write(f"Model: {metrics.get('model_name', 'N/A')}\n\n")
        
        # Determine if this is inflammation or tissue task
        is_inflammation = task == 'inflammation'
        higher_level = "Slide" if is_inflammation else "Particle"
        auc_key = "slide_auroc" if is_inflammation else "particle_auroc"
        validation_auc_key = "validation_slide_auroc" if is_inflammation else "validation_particle_auroc"
        
        # Optimal Aggregation Strategy
        if 'optimal_aggregation' in metrics:
            agg = metrics['optimal_aggregation']
            f.write("OPTIMAL AGGREGATION STRATEGY\n")
            f.write("----------------------------\n")
            f.write(f"Strategy: {agg.get('best_strategy', 'N/A')}\n")
            f.write(f"Threshold: {agg.get('best_metrics', {}).get('threshold', 0):.3f}\n\n")
        
        # Validation Performance
        if 'optimal_thresholds' in metrics:
            tile_opt = metrics['optimal_thresholds'].get('tile', {})
            higher_level_opt = metrics['optimal_thresholds'].get('slide' if is_inflammation else 'particle', {})
            f.write("VALIDATION PERFORMANCE\n")
            f.write("-------------------------\n")
            f.write(f"Tile-Level Accuracy: {tile_opt.get('accuracy', 0):.2%}\n")
            f.write(f"Tile-Level Sensitivity: {tile_opt.get('sensitivity', 0):.2%}\n")
            f.write(f"Tile-Level Specificity: {tile_opt.get('specificity', 0):.2%}\n")
            f.write(f"Tile-Level F1 Score: {tile_opt.get('f1', 0):.2%}\n")
            f.write(f"Tile-Level AUC: {metrics.get('validation_tile_auroc', 0):.2%}\n")
            f.write(f"{higher_level}-Level Accuracy: {higher_level_opt.get('accuracy', 0):.2%}\n")
            f.write(f"{higher_level}-Level Sensitivity: {higher_level_opt.get('sensitivity', 0):.2%}\n")
            f.write(f"{higher_level}-Level Specificity: {higher_level_opt.get('specificity', 0):.2%}\n")
            f.write(f"{higher_level}-Level F1 Score: {higher_level_opt.get('f1', 0):.2%}\n")
            f.write(f"{higher_level}-Level AUC: {metrics.get(validation_auc_key, 0):.2%}\n")
            f.write("\n")
        
        # Test Performance
        f.write("TEST PERFORMANCE\n")
        f.write("--------------------\n")
        
        # Tile-Level Metrics
        f.write(f"Tile-Level Accuracy: {metrics.get('tile_accuracy', 0):.2%}\n")
        f.write(f"Tile-Level Sensitivity: {metrics.get('tile_sensitivity', 0):.2%}\n")
        f.write(f"Tile-Level Specificity: {metrics.get('tile_specificity', 0):.2%}\n")
        f.write(f"Tile-Level F1 Score: {metrics.get('tile_f1', 0):.2%}\n")
        f.write(f"Tile-Level AUC: {metrics.get('tile_auroc', 0):.2%}\n")
        
        # Higher Level Metrics (Slide or Particle) from Test Confusion Matrix
        if ('optimal_aggregation' in metrics and 
            'test_confusion_matrix' in metrics['optimal_aggregation']):
            agg_cm = metrics['optimal_aggregation']['test_confusion_matrix']
            if all(key in agg_cm for key in ['tp', 'tn', 'fp', 'fn']):
                # Extract confusion matrix values
                tp = agg_cm['tp']
                tn = agg_cm['tn']
                fp = agg_cm['fp']
                fn = agg_cm['fn']
                
                # Calculate metrics
                total = tp + tn + fp + fn
                accuracy = (tp + tn) / total if total > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = (2 * precision * sensitivity / (precision + sensitivity) 
                      if (precision + sensitivity) > 0 else 0)
                
                # Get AUC directly from the test_confusion_matrix if available
                if 'auc' in agg_cm:
                    auc = agg_cm['auc']
                else:
                    # Fallback to metrics dictionary
                    auc = metrics.get(auc_key, 0)
                
                # Write calculated metrics
                f.write(f"{higher_level}-Level Accuracy: {accuracy:.2%}\n")
                f.write(f"{higher_level}-Level Sensitivity: {sensitivity:.2%}\n")
                f.write(f"{higher_level}-Level Specificity: {specificity:.2%}\n")
                f.write(f"{higher_level}-Level F1 Score: {f1:.2%}\n")
                f.write(f"{higher_level}-Level AUC: {auc:.2%}\n")
            else:
                # Fallback if confusion matrix is incomplete
                f.write(f"{higher_level}-Level Accuracy: {metrics.get('slide_accuracy' if is_inflammation else 'particle_accuracy', 0):.2%}\n")
                f.write(f"{higher_level}-Level Sensitivity: {metrics.get('slide_sensitivity' if is_inflammation else 'particle_sensitivity', 0):.2%}\n")
                f.write(f"{higher_level}-Level Specificity: {metrics.get('slide_specificity' if is_inflammation else 'particle_specificity', 0):.2%}\n")
                f.write(f"{higher_level}-Level F1 Score: {metrics.get('slide_f1' if is_inflammation else 'particle_f1', 0):.2%}\n")
                f.write(f"{higher_level}-Level AUC: {metrics.get(auc_key, 0):.2%}\n")
        else:
            # Fallback if no confusion matrix is available
            f.write(f"{higher_level}-Level Accuracy: {metrics.get('slide_accuracy' if is_inflammation else 'particle_accuracy', 0):.2%}\n")
            f.write(f"{higher_level}-Level Sensitivity: {metrics.get('slide_sensitivity' if is_inflammation else 'particle_sensitivity', 0):.2%}\n")
            f.write(f"{higher_level}-Level Specificity: {metrics.get('slide_specificity' if is_inflammation else 'particle_specificity', 0):.2%}\n")
            f.write(f"{higher_level}-Level F1 Score: {metrics.get('slide_f1' if is_inflammation else 'particle_f1', 0):.2%}\n")
            f.write(f"{higher_level}-Level AUC: {metrics.get(auc_key, 0):.2%}\n")