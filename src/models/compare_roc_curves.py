"""
Script to combine ROC curves from multiple models for scientific comparison.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import json
from typing import List, Dict, Tuple
import seaborn as sns

def get_model_results(results_dir: Path, model_name: str) -> Tuple[pd.DataFrame, str]:
    """
    Load results for a specific model.
    
    Args:
        results_dir: Base directory containing model results
        model_name: Name of the model
    
    Returns:
        DataFrame with predictions and task name
    """
    model_dir = results_dir / model_name
    
    # Load predictions
    pred_path = model_dir / 'predictions.csv'
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    
    df = pd.read_csv(pred_path)
    
    # Determine task from stats file
    stats_path = model_dir / 'statistics.json'
    if not stats_path.exists():
        # Try to infer from predictions
        if 'particle_id' in df.columns:
            task = 'tissue'
        else:
            task = 'inflammation'
    else:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        task = stats.get('task', 'unknown')
        
    return df, task

def calculate_roc_data(df: pd.DataFrame, task: str, level: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate ROC curve data for a specific level (tile, particle, or slide).
    
    Args:
        df: DataFrame with predictions
        task: Classification task ('tissue' or 'inflammation')
        level: Level to calculate ROC for ('tile', 'particle', or 'slide')
    
    Returns:
        Tuple of (fpr, tpr, auc_score)
    """
    # Convert logits to probabilities if needed
    if 'raw_pred' in df.columns and (df['raw_pred'].min() < 0 or df['raw_pred'].max() > 1):
        probs = torch.sigmoid(torch.tensor(df['raw_pred'].values)).numpy()
    else:
        probs = df['raw_pred'].values
    
    if level == 'tile':
        fpr, tpr, _ = roc_curve(df['label'], probs)
        roc_auc = auc(fpr, tpr)
    elif level == 'particle' and task == 'tissue':
        particle_df = df.groupby(['slide_name', 'particle_id']).agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        if 'raw_pred' in particle_df.columns and (particle_df['raw_pred'].min() < 0 or particle_df['raw_pred'].max() > 1):
            particle_probs = torch.sigmoid(torch.tensor(particle_df['raw_pred'].values)).numpy()
        else:
            particle_probs = particle_df['raw_pred'].values
        
        fpr, tpr, _ = roc_curve(particle_df['label'], particle_probs)
        roc_auc = auc(fpr, tpr)
    elif level == 'slide' and task == 'inflammation':
        slide_df = df.groupby('slide_name').agg({
            'raw_pred': 'mean',
            'label': 'first'
        }).reset_index()
        
        if 'raw_pred' in slide_df.columns and (slide_df['raw_pred'].min() < 0 or slide_df['raw_pred'].max() > 1):
            slide_probs = torch.sigmoid(torch.tensor(slide_df['raw_pred'].values)).numpy()
        else:
            slide_probs = slide_df['raw_pred'].values
        
        fpr, tpr, _ = roc_curve(slide_df['label'], slide_probs)
        roc_auc = auc(fpr, tpr)
    else:
        raise ValueError(f"Invalid combination of task '{task}' and level '{level}'")
    
    return fpr, tpr, roc_auc

def plot_combined_roc_curves(model_data: Dict[str, Dict[str, Tuple]], 
                            task: str, 
                            level: str,
                            output_path: Path) -> None:
    """
    Plot combined ROC curves for multiple models.
    
    Args:
        model_data: Dictionary of model data
        task: Classification task
        level: Level to plot (tile, particle, or slide)
        output_path: Path to save the plot
    """
    # Create scientific plot with square format
    plt.figure(figsize=(8, 8))
    
    # Set seaborn style for scientific plotting
    sns.set_style("whitegrid")
    
    # Colors for different models
    colors = sns.color_palette("colorblind", n_colors=len(model_data))
    
    for (model_name, model_info), color in zip(model_data.items(), colors):
        if level in model_info:
            fpr, tpr, roc_auc = model_info[level]
            plt.plot(fpr, tpr, lw=2, color=color, 
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    
    # Configure plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    
    level_name = level.capitalize()
    if task == 'tissue':
        title = f'{level_name}-Level ROC Curves - Tissue Classification'
    else:  # inflammation
        title = f'{level_name}-Level ROC Curves - Inflammation Classification'
    
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add tick marks
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Ensure aspect ratio is equal (square plot)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Save figure with high resolution
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Combine ROC curves from multiple models')
    parser.add_argument('--results_dir', type=str, required=True, 
                        help='Base directory containing model results')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='List of model names to compare')
    parser.add_argument('--model_names', type=str, nargs='+',
                        help='Optional display names for models (must match order of --models)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save combined plots')
    parser.add_argument('--task', type=str, choices=['tissue', 'inflammation', 'both'], 
                        default='both', help='Task to plot')
    parser.add_argument('--level', type=str, 
                        choices=['tile', 'particle', 'slide', 'all'],
                        default='all', help='Level to plot')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data for all models
    tissue_models = {}
    inflammation_models = {}
    
    # Use display names if provided, otherwise use directory names
    display_names = args.model_names if args.model_names else args.models
    if args.model_names and len(args.model_names) != len(args.models):
        print("Warning: Number of model names does not match number of models. Using directory names instead.")
        display_names = args.models
    
    for i, model_dir in enumerate(args.models):
        display_name = display_names[i]
        try:
            df, task = get_model_results(results_dir, model_dir)
            
            if task == 'tissue':
                tissue_models[display_name] = {}
                # Calculate tile-level ROC
                tissue_models[display_name]['tile'] = calculate_roc_data(df, task, 'tile')
                # Calculate particle-level ROC
                tissue_models[display_name]['particle'] = calculate_roc_data(df, task, 'particle')
            elif task == 'inflammation':
                inflammation_models[display_name] = {}
                # Calculate tile-level ROC
                inflammation_models[display_name]['tile'] = calculate_roc_data(df, task, 'tile')
                # Calculate slide-level ROC
                inflammation_models[display_name]['slide'] = calculate_roc_data(df, task, 'slide')
            else:
                print(f"Unknown task '{task}' for model '{model_dir}' (display name: {display_name})")
        except Exception as e:
            print(f"Error processing model '{model_dir}' (display name: {display_name}): {str(e)}")
    
    # Generate plots based on task and level
    levels_to_plot = []
    if args.level == 'all':
        levels_to_plot = ['tile', 'particle', 'slide']
    else:
        levels_to_plot = [args.level]
    
    if args.task in ['tissue', 'both'] and tissue_models:
        for level in levels_to_plot:
            if level == 'slide':  # Skip invalid combination
                continue
                
            output_path = output_dir / f'tissue_{level}_roc_comparison.png'
            plot_combined_roc_curves(tissue_models, 'tissue', level, output_path)
            print(f"Generated tissue {level}-level ROC comparison: {output_path}")
    
    if args.task in ['inflammation', 'both'] and inflammation_models:
        for level in levels_to_plot:
            if level == 'particle':  # Skip invalid combination
                continue
                
            output_path = output_dir / f'inflammation_{level}_roc_comparison.png'
            plot_combined_roc_curves(inflammation_models, 'inflammation', level, output_path)
            print(f"Generated inflammation {level}-level ROC comparison: {output_path}")

if __name__ == "__main__":
    main()