import pandas as pd
import json
import os
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path
import sys

# Add the project root to the path if not already
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import path configuration
from src.config.paths import get_project_paths

# Get project paths
paths = get_project_paths()
BASE_DIR = paths["BASE_DIR"]

# Load the necessary data files
def load_csv_file(filepath):
    """Load a CSV file or return None if file doesn't exist"""
    full_path = os.path.join(BASE_DIR, filepath)
    try:
        return pd.read_csv(full_path)
    except Exception as e:
        print(f"Error loading {full_path}: {e}")
        return None

# Load all data files
# Metadata CSV files
tiles_per_particle = load_csv_file("results/metrics/tiles_per_particle.csv")
tissue_types = load_csv_file("results/metrics/tissue_types.csv")
inflammation_status = load_csv_file("results/metrics/inflammation_status.csv")

# Dataset split definitions
train_he = load_csv_file("data/splits/seed_42/train_HE.csv")
val_he = load_csv_file("data/splits/seed_42/val_HE.csv")
test_he = load_csv_file("data/splits/seed_42/test_HE.csv")
test_scanner2_he = load_csv_file("data/splits/seed_42/test_scanner2_HE.csv")

train_all = load_csv_file("data/splits/seed_42/train_all_stains.csv")
val_all = load_csv_file("data/splits/seed_42/val_all_stains.csv")
test_all = load_csv_file("data/splits/seed_42/test_all_stains.csv")
test_scanner2_all = load_csv_file("data/splits/seed_42/test_scanner2_all_stains.csv")

# Load statistics.json
with open(os.path.join(BASE_DIR, "results/metrics/statistics.json"), 'r') as f:
    statistics = json.load(f)

# Create mapping dictionaries to track which slides belong to which splits
def create_split_mappings():
    """Create mappings from slide name to dataset split"""
    # HE stain only splits
    split_mapping_he = {}
    for df, split_name in [
        (train_he, 'train'), 
        (val_he, 'val'), 
        (test_he, 'test'), 
        (test_scanner2_he, 'test_scanner2')
    ]:
        if df is not None:
            for slide_name in df['slide_name']:
                split_mapping_he[slide_name] = split_name
    
    # All stains splits
    split_mapping_all = {}
    for df, split_name in [
        (train_all, 'train'), 
        (val_all, 'val'), 
        (test_all, 'test'), 
        (test_scanner2_all, 'test_scanner2')
    ]:
        if df is not None:
            for slide_name in df['slide_name']:
                split_mapping_all[slide_name] = split_name
    
    return split_mapping_he, split_mapping_all

# Calculate tile distribution across dataset splits
def calculate_tile_distribution():
    """Calculate the distribution of tiles across dataset splits"""
    split_mapping_he, split_mapping_all = create_split_mappings()
    
    # Add split information to tiles_per_particle dataframe
    tiles_with_split = tiles_per_particle.copy()
    tiles_with_split['split_he'] = tiles_with_split['slide_name'].map(split_mapping_he)
    tiles_with_split['split_all'] = tiles_with_split['slide_name'].map(split_mapping_all)
    
    # Calculate HE stain distribution
    he_tiles = tiles_with_split[tiles_with_split['stain'] == 'HE']
    he_counts = he_tiles.groupby('split_he')['tiles_per_particle'].sum().reset_index()
    he_counts.columns = ['split', 'count']
    total_he = he_counts['count'].sum()
    he_counts['percentage'] = (he_counts['count'] / total_he * 100)
    
    # Calculate all stains distribution
    all_counts = tiles_with_split.groupby('split_all')['tiles_per_particle'].sum().reset_index()
    all_counts.columns = ['split', 'count']
    total_all = all_counts['count'].sum()
    all_counts['percentage'] = (all_counts['count'] / total_all * 100)
    
    return he_counts, all_counts, total_he, total_all

# Generate PNG tile distribution table (LaTeX format)
def generate_png_tile_distribution_table():
    """Generate LaTeX table for PNG tile distribution"""
    he_counts, all_counts, total_he, total_all = calculate_tile_distribution()
    
    # Map split names to display names
    split_display = {
        'train': 'Training',
        'val': 'Validation',
        'test': 'Test (Scanner 1)',
        'test_scanner2': 'Test (Scanner 2)'
    }
    
    # Start LaTeX table - use raw string for static parts
    latex = r"""\\begin{table}[htb!]
	\\centering
	\\caption{Distribution of PNG Tile Files Across Dataset Splits}
	\\label{tab:png_tile_distribution}
	\\begin{tabular}{lrrrr}
		\\toprule
		\\textbf{Split} & \\multicolumn{2}{c}{\\textbf{HE Stain Only}} & \\multicolumn{2}{c}{\\textbf{All Stains}} \\\\
		& \\textbf{Count} & \\textbf{\\%} & \\textbf{Count} & \\textbf{\\%} \\\\
		\\midrule"""
    
    # Add rows for each split
    for split_key, display_name in split_display.items():
        he_row = he_counts[he_counts['split'] == split_key]
        all_row = all_counts[all_counts['split'] == split_key]
        
        he_count = he_row['count'].values[0] if not he_row.empty else 0
        he_pct = he_row['percentage'].values[0] if not he_row.empty else 0
        
        all_count = all_row['count'].values[0] if not all_row.empty else 0
        all_pct = all_row['percentage'].values[0] if not all_row.empty else 0
        
        latex += f"""
		{display_name} & {int(he_count):,} & {he_pct:.1f}\\% & {int(all_count):,} & {all_pct:.1f}\\% \\\\"""
    
    # Add totals
    latex += f"""
		\\midrule
		\\textbf{{Total}} & {int(total_he):,} & 100\\% & {int(total_all):,} & 100\\% \\\\
		\\bottomrule
	\\end{{tabular}}
	\\begin{{tablenotes}}
		\\small
		\\item Note: The table shows the number of extracted PNG tile files (256×256 pixels) for each dataset split across different staining types. "All Stains" includes HE, PAS, and MG staining methods.
	\\end{{tablenotes}}
\\end{{table}}"""
    
    return latex

# Generate dataset summary statistics table
def generate_dataset_summary_table():
    """Generate LaTeX table for dataset summary statistics"""
    # Count unique patients, slides, and particles
    total_patients = len(set(tiles_per_particle['patient_id']))
    total_slides = len(set(tiles_per_particle['slide_name']))
    total_particles = len(tiles_per_particle)
    
    # Scanner distribution
    scanner1_patients = len(set(tiles_per_particle[tiles_per_particle['scanner_id'] == 1]['patient_id']))
    scanner2_patients = len(set(tiles_per_particle[tiles_per_particle['scanner_id'] == 2]['patient_id']))
    
    scanner1_slides = len(set(tiles_per_particle[tiles_per_particle['scanner_id'] == 1]['slide_name']))
    scanner2_slides = len(set(tiles_per_particle[tiles_per_particle['scanner_id'] == 2]['slide_name']))
    
    # Stain distribution
    he_slides = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'HE']['slide_name']))
    pas_slides = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'PAS']['slide_name']))
    mg_slides = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'MG']['slide_name']))
    
    # Tissue types
    corpus_particles = len(tiles_per_particle[tiles_per_particle['tissue_type'] == 'corpus'])
    antrum_particles = len(tiles_per_particle[tiles_per_particle['tissue_type'] == 'antrum'])
    intermediate_particles = len(tiles_per_particle[tiles_per_particle['tissue_type'] == 'intermediate'])
    other_particles = len(tiles_per_particle[tiles_per_particle['tissue_type'] == 'other'])
    
    # Inflammation status
    inflamed_slides = len(set(tiles_per_particle[tiles_per_particle['inflammation_status'] == 'inflamed']['slide_name']))
    noninflamed_slides = len(set(tiles_per_particle[tiles_per_particle['inflammation_status'] == 'noninflamed']['slide_name']))
    other_inflammation_slides = total_slides - inflamed_slides - noninflamed_slides
    
    # Generate LaTeX table
    latex = r"""\\begin{table}[ht]
	\\centering
	\\caption{Dataset Summary Statistics (Dataset$^{S12-HE,PAS,MG}_{IT,v}$)}
	\\label{tab:dataset_summary}
	\\begin{tabular}{lrr}
		\\toprule
		Category & Count & Percentage \\\\
		\\midrule"""
    
    # Add basic counts
    latex += f"""
		Total Patients & {total_patients} & 100\\% \\\\
		Total Slides & {total_slides} & 100\\% \\\\
		Total Tissue Particles & {total_particles} & 100\\% \\\\"""
    
    # Add scanner distribution
    latex += f"""
		\\midrule
		Scanner 1 Patients & {scanner1_patients} & {scanner1_patients/total_patients*100:.1f}\\% \\\\
		Scanner 2 Patients & {scanner2_patients} & {scanner2_patients/total_patients*100:.1f}\\% \\\\
		Scanner 1 Slides & {scanner1_slides} & {scanner1_slides/total_slides*100:.1f}\\% \\\\
		Scanner 2 Slides & {scanner2_slides} & {scanner2_slides/total_slides*100:.1f}\\% \\\\"""
    
    # Add stain distribution
    latex += f"""
		\\midrule
		HE Stained Slides & {he_slides} & {he_slides/total_slides*100:.1f}\\% \\\\
		PAS Stained Slides & {pas_slides} & {pas_slides/total_slides*100:.1f}\\% \\\\
		MG Stained Slides & {mg_slides} & {mg_slides/total_slides*100:.1f}\\% \\\\"""
    
    # Add tissue type distribution
    latex += f"""
		\\midrule
		Corpus Tissue Particles & {corpus_particles} & {corpus_particles/total_particles*100:.1f}\\% \\\\
		Antrum Tissue Particles & {antrum_particles} & {antrum_particles/total_particles*100:.1f}\\% \\\\
		Intermediate Tissue Particles & {intermediate_particles} & {intermediate_particles/total_particles*100:.1f}\\% \\\\
		Other Tissue Particles & {other_particles} & {other_particles/total_particles*100:.1f}\\% \\\\"""
    
    # Add inflammation status distribution
    latex += f"""
		\\midrule
		Inflamed Slides & {inflamed_slides} & {inflamed_slides/total_slides*100:.1f}\\% \\\\
		Non-inflamed Slides & {noninflamed_slides} & {noninflamed_slides/total_slides*100:.1f}\\% \\\\
		Other Inflammation Status & {other_inflammation_slides} & {other_inflammation_slides/total_slides*100:.1f}\\% \\\\
		\\bottomrule
	\\end{{tabular}}
\\end{{table}}"""
    
    return latex

# Generate stain type distribution table
def generate_stain_distribution_table():
    """Generate LaTeX table for stain type distribution"""
    # Count slides per stain type
    he_slides = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'HE']['slide_name']))
    pas_slides = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'PAS']['slide_name']))
    mg_slides = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'MG']['slide_name']))
    
    # Count patients per stain type
    he_patients = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'HE']['patient_id']))
    pas_patients = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'PAS']['patient_id']))
    mg_patients = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'MG']['patient_id']))
    
    # Count particles per tissue type and stain
    he_corpus = sum(1 for _, row in tiles_per_particle.iterrows() 
                   if row['stain'] == 'HE' and row['tissue_type'] == 'corpus')
    he_antrum = sum(1 for _, row in tiles_per_particle.iterrows() 
                   if row['stain'] == 'HE' and row['tissue_type'] == 'antrum')
    he_intermediate = sum(1 for _, row in tiles_per_particle.iterrows() 
                         if row['stain'] == 'HE' and row['tissue_type'] == 'intermediate')
    he_other = sum(1 for _, row in tiles_per_particle.iterrows() 
                  if row['stain'] == 'HE' and row['tissue_type'] == 'other')
    
    pas_corpus = sum(1 for _, row in tiles_per_particle.iterrows() 
                    if row['stain'] == 'PAS' and row['tissue_type'] == 'corpus')
    pas_antrum = sum(1 for _, row in tiles_per_particle.iterrows() 
                    if row['stain'] == 'PAS' and row['tissue_type'] == 'antrum')
    pas_intermediate = sum(1 for _, row in tiles_per_particle.iterrows() 
                          if row['stain'] == 'PAS' and row['tissue_type'] == 'intermediate')
    pas_other = sum(1 for _, row in tiles_per_particle.iterrows() 
                   if row['stain'] == 'PAS' and row['tissue_type'] == 'other')
    
    mg_corpus = sum(1 for _, row in tiles_per_particle.iterrows() 
                   if row['stain'] == 'MG' and row['tissue_type'] == 'corpus')
    mg_antrum = sum(1 for _, row in tiles_per_particle.iterrows() 
                   if row['stain'] == 'MG' and row['tissue_type'] == 'antrum')
    mg_intermediate = sum(1 for _, row in tiles_per_particle.iterrows() 
                         if row['stain'] == 'MG' and row['tissue_type'] == 'intermediate')
    mg_other = sum(1 for _, row in tiles_per_particle.iterrows() 
                  if row['stain'] == 'MG' and row['tissue_type'] == 'other')
    
    # Calculate totals
    he_total = he_corpus + he_antrum + he_intermediate + he_other
    pas_total = pas_corpus + pas_antrum + pas_intermediate + pas_other
    mg_total = mg_corpus + mg_antrum + mg_intermediate + mg_other
    
    total_slides = he_slides + pas_slides + mg_slides
    total_distinct_patients = len(set(tiles_per_particle['patient_id']))
    total_corpus = he_corpus + pas_corpus + mg_corpus
    total_antrum = he_antrum + pas_antrum + mg_antrum
    total_intermediate = he_intermediate + pas_intermediate + mg_intermediate
    total_other = he_other + pas_other + mg_other
    total_particles = he_total + pas_total + mg_total
    
    # Generate LaTeX table
    latex = r"""\\begin{table}[ht]
	\\centering
	\\caption{Stain Type Distribution (Dataset$^{S12-HE,PAS,MG}_{IT,v}$)}
	\\label{tab:stain_distribution}
	\\begin{tabular}{lrrrrrrr}
		\\toprule
		Stain & Slides & Patients & \\multicolumn{4}{c}{Tissue Particles} & Total \\\\
		\\cmidrule(lr){4-7}
		& & & Corpus & Antrum & Intermediate & Other & Particles \\\\
		\\midrule"""
    
    # Add rows for each stain type
    latex += f"""
		HE & {he_slides} & {he_patients} & {he_corpus} & {he_antrum} & {he_intermediate} & {he_other} & {he_total} \\\\
		PAS & {pas_slides} & {pas_patients} & {pas_corpus} & {pas_antrum} & {pas_intermediate} & {pas_other} & {pas_total} \\\\
		MG & {mg_slides} & {mg_patients} & {mg_corpus} & {mg_antrum} & {mg_intermediate} & {mg_other} & {mg_total} \\\\
		\\midrule
		Total & {total_slides} & {total_distinct_patients} & {total_corpus} & {total_antrum} & {total_intermediate} & {total_other} & {total_particles} \\\\
		\\bottomrule
	\\end{{tabular}}
\\end{{table}}"""
    
    return latex

# Generate tiles per particle distribution table
def generate_tiles_per_particle_table():
    """Generate LaTeX table for tiles per particle distribution"""
    # Calculate statistics by tissue type
    tissue_groups = {
        'Corpus': tiles_per_particle[tiles_per_particle['tissue_type'] == 'corpus']['tiles_per_particle'],
        'Antrum': tiles_per_particle[tiles_per_particle['tissue_type'] == 'antrum']['tiles_per_particle'],
        'Intermediate': tiles_per_particle[tiles_per_particle['tissue_type'] == 'intermediate']['tiles_per_particle'],
        'Other': tiles_per_particle[tiles_per_particle['tissue_type'] == 'other']['tiles_per_particle']
    }
    
    # Calculate statistics by inflammation status
    inflammation_groups = {
        'Inflamed': tiles_per_particle[tiles_per_particle['inflammation_status'] == 'inflamed']['tiles_per_particle'],
        'Non-inflamed': tiles_per_particle[tiles_per_particle['inflammation_status'] == 'noninflamed']['tiles_per_particle'],
        'Other': tiles_per_particle[(tiles_per_particle['inflammation_status'] != 'inflamed') & 
                                   (tiles_per_particle['inflammation_status'] != 'noninflamed')]['tiles_per_particle']
    }
    
    # Calculate statistics by stain type
    stain_groups = {
        'HE': tiles_per_particle[tiles_per_particle['stain'] == 'HE']['tiles_per_particle'],
        'PAS': tiles_per_particle[tiles_per_particle['stain'] == 'PAS']['tiles_per_particle'],
        'MG': tiles_per_particle[tiles_per_particle['stain'] == 'MG']['tiles_per_particle']
    }
    
    # Calculate total statistics
    total_stats = tiles_per_particle['tiles_per_particle']
    
    # Helper function to calculate statistics
    def calc_stats(series):
        if len(series) == 0:
            return {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'median': 0, 'max': 0}
        return {
            'count': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'median': series.median(),
            'max': series.max()
        }
    
    # Calculate statistics for each group
    tissue_stats = {name: calc_stats(group) for name, group in tissue_groups.items()}
    inflammation_stats = {name: calc_stats(group) for name, group in inflammation_groups.items()}
    stain_stats = {name: calc_stats(group) for name, group in stain_groups.items()}
    total_stat = calc_stats(total_stats)
    
    # Generate LaTeX table
    latex = r"""\\begin{table}[ht]
	\\centering
	\\caption{Tiles per Particle Distribution (Dataset$^{S12-HE,PAS,MG}_{IT,v}$)}
	\\label{tab:tiles_per_particle}
	\\begin{tabular}{lrrrrrr}
		\\toprule
		Category & Count & Mean & Std Dev & Min & Median & Max \\\\
		\\midrule
		\\textbf{Tissue Type} & & & & & & \\\\"""
    
    # Add tissue type statistics
    for name, stats in tissue_stats.items():
        latex += f"""
		{name} & {stats['count']} & {stats['mean']:.1f} & {stats['std']:.1f} & {stats['min']:.1f} & {stats['median']:.1f} & {stats['max']:.1f} \\\\"""
    
    # Add inflammation status statistics
    latex += r"""
		\\midrule
		\\textbf{Inflammation Status} & & & & & & \\\\"""
    
    for name, stats in inflammation_stats.items():
        latex += f"""
		{name} & {stats['count']} & {stats['mean']:.1f} & {stats['std']:.1f} & {stats['min']:.1f} & {stats['median']:.1f} & {stats['max']:.1f} \\\\"""
    
    # Add stain type statistics
    latex += r"""
		\\midrule
		\\textbf{Stain Type} & & & & & & \\\\"""
    
    for name, stats in stain_stats.items():
        latex += f"""
		{name} & {stats['count']} & {stats['mean']:.1f} & {stats['std']:.1f} & {stats['min']:.1f} & {stats['median']:.1f} & {stats['max']:.1f} \\\\"""
    
    # Add total statistics
    latex += f"""
		\\midrule
		Total & {total_stat['count']} & {total_stat['mean']:.1f} & {total_stat['std']:.1f} & {total_stat['min']:.1f} & {total_stat['median']:.1f} & {total_stat['max']:.1f} \\\\
		\\bottomrule
	\\end{{tabular}}
\\end{{table}}"""
    
    return latex

# Generate dataset splits distribution table
def generate_data_splits_table():
    """Generate LaTeX table for dataset splits distribution"""
    # Create mappings from slide name to split
    split_mapping_he, _ = create_split_mappings()
    
    # Update display names for splits
    split_display = {
        'train': 'Training',
        'val': 'Validation',
        'test': 'Test',
        'test_scanner2': 'Test Scanner2'
    }
    
    # Filter for HE stain slides and add split information
    he_slides = tiles_per_particle[tiles_per_particle['stain'] == 'HE'].copy()
    he_slides['split'] = he_slides['slide_name'].map(split_mapping_he)
    
    # Calculate counts per split
    splits_data = {}
    for split_key, display_name in split_display.items():
        split_df = he_slides[he_slides['split'] == split_key]
        
        # Skip if no slides in this split
        if len(split_df) == 0:
            continue
            
        # Count unique slides and patients
        splits_data[display_name] = {
            'slides': len(set(split_df['slide_name'])),
            'patients': len(set(split_df['patient_id'])),
            
            # Count inflammation status
            'inflamed': len(set(split_df[split_df['inflammation_status'] == 'inflamed']['slide_name'])),
            'noninflamed': len(set(split_df[split_df['inflammation_status'] == 'noninflamed']['slide_name'])),
            
            # Count tissue particles
            'corpus': len(split_df[split_df['tissue_type'] == 'corpus']),
            'antrum': len(split_df[split_df['tissue_type'] == 'antrum']),
            'intermediate': len(split_df[split_df['tissue_type'] == 'intermediate'])
        }
    
    # Calculate totals
    total_slides = sum(data['slides'] for data in splits_data.values())
    total_patients = sum(data['patients'] for data in splits_data.values())
    total_inflamed = sum(data['inflamed'] for data in splits_data.values())
    total_noninflamed = sum(data['noninflamed'] for data in splits_data.values())
    total_corpus = sum(data['corpus'] for data in splits_data.values())
    total_antrum = sum(data['antrum'] for data in splits_data.values())
    total_intermediate = sum(data['intermediate'] for data in splits_data.values())
    
    # Generate LaTeX table
    latex = r"""\\begin{table}[ht]
	\\centering
	\\caption{Dataset Splits Distribution (Dataset$^{S1-HE}_{IT,v}$)}
	\\label{tab:data_splits}
	\\begin{tabular}{lrrrrr}
		\\toprule
		Characteristic & Training & Validation & Test & Test Scanner2 & Total \\\\
		\\midrule
		\\textbf{HE Stain Only} & & & & & \\\\"""
    
    # Add slides and patients counts
    latex += f"""
		Slides & {splits_data.get('Training', {}).get('slides', 0)} & {splits_data.get('Validation', {}).get('slides', 0)} & {splits_data.get('Test', {}).get('slides', 0)} & {splits_data.get('Test Scanner2', {}).get('slides', 0)} & {total_slides} \\\\
		Patients & {splits_data.get('Training', {}).get('patients', 0)} & {splits_data.get('Validation', {}).get('patients', 0)} & {splits_data.get('Test', {}).get('patients', 0)} & {splits_data.get('Test Scanner2', {}).get('patients', 0)} & {total_patients} \\\\"""
    
    # Add inflammation status counts
    latex += r"""
		\\textbf{Inflammation Status (HE)} & & & & & \\\\"""
    
    latex += f"""
		Inflamed & {splits_data.get('Training', {}).get('inflamed', 0)} & {splits_data.get('Validation', {}).get('inflamed', 0)} & {splits_data.get('Test', {}).get('inflamed', 0)} & {splits_data.get('Test Scanner2', {}).get('inflamed', 0)} & {total_inflamed} \\\\
		Non-inflamed & {splits_data.get('Training', {}).get('noninflamed', 0)} & {splits_data.get('Validation', {}).get('noninflamed', 0)} & {splits_data.get('Test', {}).get('noninflamed', 0)} & {splits_data.get('Test Scanner2', {}).get('noninflamed', 0)} & {total_noninflamed} \\\\"""
    
    # Add tissue particles counts
    latex += r"""
		\\textbf{Tissue Particles (HE)} & & & & & \\\\"""
    
    latex += f"""
		Corpus & {splits_data.get('Training', {}).get('corpus', 0)} & {splits_data.get('Validation', {}).get('corpus', 0)} & {splits_data.get('Test', {}).get('corpus', 0)} & {splits_data.get('Test Scanner2', {}).get('corpus', 0)} & {total_corpus} \\\\
		Antrum & {splits_data.get('Training', {}).get('antrum', 0)} & {splits_data.get('Validation', {}).get('antrum', 0)} & {splits_data.get('Test', {}).get('antrum', 0)} & {splits_data.get('Test Scanner2', {}).get('antrum', 0)} & {total_antrum} \\\\
		Intermediate & {splits_data.get('Training', {}).get('intermediate', 0)} & {splits_data.get('Validation', {}).get('intermediate', 0)} & {splits_data.get('Test', {}).get('intermediate', 0)} & {splits_data.get('Test Scanner2', {}).get('intermediate', 0)} & {total_intermediate} \\\\
		\\bottomrule
	\\end{{tabular}}
\\end{{table}}"""
    
    return latex


def generate_he_dataset_composition_table():
    """Generate LaTeX table for detailed HE dataset composition"""
    # Create mappings from slide name to split
    split_mapping_he, _ = create_split_mappings()
    
    # Update display names for splits
    split_display = {
        'train': 'Training',
        'val': 'Validation',
        'test': 'Test',
        'test_scanner2': 'Test Scanner2'
    }
    
    # Filter for HE stain slides and add split information
    he_slides = tiles_per_particle[tiles_per_particle['stain'] == 'HE'].copy()
    he_slides['split'] = he_slides['slide_name'].map(split_mapping_he)
    
    # Initialize detailed data structure
    detailed_data = {}
    
    for split_key, display_name in split_display.items():
        split_df = he_slides[he_slides['split'] == split_key]
        
        # Skip if no slides in this split
        if len(split_df) == 0:
            continue
            
        # Count basic metrics
        slides = len(set(split_df['slide_name']))
        patients = len(set(split_df['patient_id']))
        
        # Count and sum tissue particles and tiles
        corpus_df = split_df[split_df['tissue_type'] == 'corpus']
        corpus_particles = len(corpus_df)
        corpus_tiles = corpus_df['tiles_per_particle'].sum()
        
        antrum_df = split_df[split_df['tissue_type'] == 'antrum']
        antrum_particles = len(antrum_df)
        antrum_tiles = antrum_df['tiles_per_particle'].sum()
        
        intermediate_df = split_df[split_df['tissue_type'] == 'intermediate']
        intermediate_particles = len(intermediate_df)
        intermediate_tiles = intermediate_df['tiles_per_particle'].sum()
        
        # Calculate inflammation status counts and percentages
        inflamed_slides = len(set(split_df[split_df['inflammation_status'] == 'inflamed']['slide_name']))
        noninflamed_slides = len(set(split_df[split_df['inflammation_status'] == 'noninflamed']['slide_name']))
        
        inflamed_pct = (inflamed_slides / slides * 100) if slides > 0 else 0
        noninflamed_pct = (noninflamed_slides / slides * 100) if slides > 0 else 0
        
        # Store data
        detailed_data[display_name] = {
            'slides': slides,
            'patients': patients,
            'corpus_particles': corpus_particles,
            'corpus_tiles': corpus_tiles,
            'antrum_particles': antrum_particles,
            'antrum_tiles': antrum_tiles,
            'intermediate_particles': intermediate_particles,
            'intermediate_tiles': intermediate_tiles,
            'inflamed_slides': inflamed_slides,
            'inflamed_pct': inflamed_pct,
            'noninflamed_slides': noninflamed_slides,
            'noninflamed_pct': noninflamed_pct
        }
    
    # Generate LaTeX table
    latex = r"""\\begin{table}[ht]
	\\centering
	\\caption{Detailed Composition of HE Dataset Partitions (Dataset$^{S1-HE}_{IT,v}$)}
	\\label{tab:train_val_test_composition}
	\\begin{tabular}{lrrrr}
		\\toprule
		Characteristic & Training & Validation & Test & Test Scanner2 \\\\
		\\midrule
		\\textbf{Basic Counts} & & & & \\\\"""
    
    # Add slides and patients counts
    latex += f"""
		Slides & {detailed_data.get('Training', {}).get('slides', 0)} & {detailed_data.get('Validation', {}).get('slides', 0)} & {detailed_data.get('Test', {}).get('slides', 0)} & {detailed_data.get('Test Scanner2', {}).get('slides', 0)} \\\\
		Patients & {detailed_data.get('Training', {}).get('patients', 0)} & {detailed_data.get('Validation', {}).get('patients', 0)} & {detailed_data.get('Test', {}).get('patients', 0)} & {detailed_data.get('Test Scanner2', {}).get('patients', 0)} \\\\
		\\midrule"""
    
    # Add tissue particles and tiles counts
    latex += r"""
		\\textbf{Tissue Particles} & & & & \\\\"""
    
    latex += f"""
		Corpus Particles & {detailed_data.get('Training', {}).get('corpus_particles', 0)} & {detailed_data.get('Validation', {}).get('corpus_particles', 0)} & {detailed_data.get('Test', {}).get('corpus_particles', 0)} & {detailed_data.get('Test Scanner2', {}).get('corpus_particles', 0)} \\\\
		Corpus Tiles & {int(detailed_data.get('Training', {}).get('corpus_tiles', 0))} & {int(detailed_data.get('Validation', {}).get('corpus_tiles', 0))} & {int(detailed_data.get('Test', {}).get('corpus_tiles', 0))} & {int(detailed_data.get('Test Scanner2', {}).get('corpus_tiles', 0))} \\\\
		Antrum Particles & {detailed_data.get('Training', {}).get('antrum_particles', 0)} & {detailed_data.get('Validation', {}).get('antrum_particles', 0)} & {detailed_data.get('Test', {}).get('antrum_particles', 0)} & {detailed_data.get('Test Scanner2', {}).get('antrum_particles', 0)} \\\\
		Antrum Tiles & {int(detailed_data.get('Training', {}).get('antrum_tiles', 0))} & {int(detailed_data.get('Validation', {}).get('antrum_tiles', 0))} & {int(detailed_data.get('Test', {}).get('antrum_tiles', 0))} & {int(detailed_data.get('Test Scanner2', {}).get('antrum_tiles', 0))} \\\\
		Intermediate Particles & {detailed_data.get('Training', {}).get('intermediate_particles', 0)} & {detailed_data.get('Validation', {}).get('intermediate_particles', 0)} & {detailed_data.get('Test', {}).get('intermediate_particles', 0)} & {detailed_data.get('Test Scanner2', {}).get('intermediate_particles', 0)} \\\\
		Intermediate Tiles & {int(detailed_data.get('Training', {}).get('intermediate_tiles', 0))} & {int(detailed_data.get('Validation', {}).get('intermediate_tiles', 0))} & {int(detailed_data.get('Test', {}).get('intermediate_tiles', 0))} & {int(detailed_data.get('Test Scanner2', {}).get('intermediate_tiles', 0))} \\\\
		\\midrule"""
    
    # Add inflammation status counts and percentages
    latex += r"""
		\\textbf{Inflammation Status} & & & & \\\\"""
    
    latex += f"""
		Inflamed Slides & {detailed_data.get('Training', {}).get('inflamed_slides', 0)} & {detailed_data.get('Validation', {}).get('inflamed_slides', 0)} & {detailed_data.get('Test', {}).get('inflamed_slides', 0)} & {detailed_data.get('Test Scanner2', {}).get('inflamed_slides', 0)} \\\\
		Inflamed (\\%) & {detailed_data.get('Training', {}).get('inflamed_pct', 0):.1f}\\% & {detailed_data.get('Validation', {}).get('inflamed_pct', 0):.1f}\\% & {detailed_data.get('Test', {}).get('inflamed_pct', 0):.1f}\\% & {detailed_data.get('Test Scanner2', {}).get('inflamed_pct', 0):.1f}\\% \\\\
		Non-inflamed Slides & {detailed_data.get('Training', {}).get('noninflamed_slides', 0)} & {detailed_data.get('Validation', {}).get('noninflamed_slides', 0)} & {detailed_data.get('Test', {}).get('noninflamed_slides', 0)} & {detailed_data.get('Test Scanner2', {}).get('noninflamed_slides', 0)} \\\\
		Non-inflamed (\\%) & {detailed_data.get('Training', {}).get('noninflamed_pct', 0):.1f}\\% & {detailed_data.get('Validation', {}).get('noninflamed_pct', 0):.1f}\\% & {detailed_data.get('Test', {}).get('noninflamed_pct', 0):.1f}\\% & {detailed_data.get('Test Scanner2', {}).get('noninflamed_pct', 0):.1f}\\% \\\\
		\\bottomrule
	\\end{{tabular}}
\\end{{table}}"""
    
    return latex

# Generate inflammation status distribution table
def generate_inflammation_distribution_table():
    """Generate LaTeX table for inflammation status distribution"""
    # Group by inflammation status
    inflammation_groups = tiles_per_particle.groupby('inflammation_status')
    
    # Count patients by inflammation status
    patients_by_inflammation = defaultdict(set)
    for _, row in tiles_per_particle.iterrows():
        patients_by_inflammation[row['inflammation_status']].add(row['patient_id'])
    
    # Count unique slides by inflammation status and stain
    slides_by_inflammation_stain = defaultdict(lambda: defaultdict(set))
    for _, row in tiles_per_particle.iterrows():
        slides_by_inflammation_stain[row['inflammation_status']][row['stain']].add(row['slide_name'])
    
    # Count tissue particles by inflammation status and tissue type
    particles_by_inflammation_tissue = defaultdict(lambda: defaultdict(int))
    for _, row in tiles_per_particle.iterrows():
        particles_by_inflammation_tissue[row['inflammation_status']][row['tissue_type']] += 1
    
    # Calculate totals
    total_patients = len(set(tiles_per_particle['patient_id']))
    total_slides = len(set(tiles_per_particle['slide_name']))
    
    total_he_slides = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'HE']['slide_name']))
    total_pas_slides = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'PAS']['slide_name']))
    total_mg_slides = len(set(tiles_per_particle[tiles_per_particle['stain'] == 'MG']['slide_name']))
    
    total_corpus = len(tiles_per_particle[tiles_per_particle['tissue_type'] == 'corpus'])
    total_antrum = len(tiles_per_particle[tiles_per_particle['tissue_type'] == 'antrum'])
    total_intermediate = len(tiles_per_particle[tiles_per_particle['tissue_type'] == 'intermediate'])
    
    # Generate LaTeX table
    latex = r"""\\begin{table}[ht]
	\\centering
	\\caption{Inflammation Status Distribution (Dataset$^{S1-HE}_{I,v}$)}
	\\label{tab:inflammation_distribution}
	\\begin{tabular}{lrrrrrrrr}
		\\toprule
		Inflammation & Patients & Total & \\multicolumn{3}{c}{Slides by Stain} & \\multicolumn{3}{c}{Tissue Particles} \\\\
		\\cmidrule(lr){4-6} \\cmidrule(lr){7-9}
		Status & & Slides & HE & PAS & MG & Corpus & Antrum & Intermediate \\\\
		\\midrule"""
    
    # Add rows for each inflammation status
    for status in ['inflamed', 'noninflamed', 'other']:
        # For 'other' inflammation status, combine all statuses that are not 'inflamed' or 'noninflamed'
        if status == 'other':
            patients = set().union(*[patients_by_inflammation[s] for s in patients_by_inflammation 
                                    if s not in ['inflamed', 'noninflamed']])
            
            total_slides_status = sum(len(slides_by_inflammation_stain[s]['HE']) + 
                                      len(slides_by_inflammation_stain[s]['PAS']) + 
                                      len(slides_by_inflammation_stain[s]['MG']) 
                                      for s in slides_by_inflammation_stain 
                                      if s not in ['inflamed', 'noninflamed'])
            
            he_slides = set().union(*[slides_by_inflammation_stain[s]['HE'] for s in slides_by_inflammation_stain 
                                     if s not in ['inflamed', 'noninflamed']])
            pas_slides = set().union(*[slides_by_inflammation_stain[s]['PAS'] for s in slides_by_inflammation_stain 
                                      if s not in ['inflamed', 'noninflamed']])
            mg_slides = set().union(*[slides_by_inflammation_stain[s]['MG'] for s in slides_by_inflammation_stain 
                                     if s not in ['inflamed', 'noninflamed']])
            
            corpus_particles = sum(particles_by_inflammation_tissue[s]['corpus'] for s in particles_by_inflammation_tissue 
                                  if s not in ['inflamed', 'noninflamed'])
            antrum_particles = sum(particles_by_inflammation_tissue[s]['antrum'] for s in particles_by_inflammation_tissue 
                                  if s not in ['inflamed', 'noninflamed'])
            intermediate_particles = sum(particles_by_inflammation_tissue[s]['intermediate'] for s in particles_by_inflammation_tissue 
                                         if s not in ['inflamed', 'noninflamed'])
        else:
            patients = patients_by_inflammation[status]
            
            total_slides_status = (len(slides_by_inflammation_stain[status]['HE']) + 
                                   len(slides_by_inflammation_stain[status]['PAS']) + 
                                   len(slides_by_inflammation_stain[status]['MG']))
            
            he_slides = slides_by_inflammation_stain[status]['HE']
            pas_slides = slides_by_inflammation_stain[status]['PAS']
            mg_slides = slides_by_inflammation_stain[status]['MG']
            
            corpus_particles = particles_by_inflammation_tissue[status]['corpus']
            antrum_particles = particles_by_inflammation_tissue[status]['antrum']
            intermediate_particles = particles_by_inflammation_tissue[status]['intermediate']
        
        # Format row
        status_display = 'Inflamed' if status == 'inflamed' else 'Non-inflamed' if status == 'noninflamed' else 'Other'
        
        latex += f"""
		{status_display} & {len(patients)} & {total_slides_status} & {len(he_slides)} & {len(pas_slides)} & {len(mg_slides)} & {corpus_particles} & {antrum_particles} & {intermediate_particles} \\\\"""
    
    # Add total row
    latex += f"""
		\\midrule
		Total & {total_patients} & {total_slides} & {total_he_slides} & {total_pas_slides} & {total_mg_slides} & {total_corpus} & {total_antrum} & {total_intermediate} \\\\
		\\bottomrule
	\\end{{tabular}}
\\end{{table}}"""
    
    return latex

# Generate patient selection flow table
def generate_patient_selection_flow_table():
    """Generate LaTeX table for patient selection flow"""
    # Count patients at each selection stage
    # These counts are based on the description in paste-2.txt
    initial_cohort = 274
    valid_annotation_cohort = 252
    scanner1_cohort = 222
    scanner1_he_cohort = 210
    scanner1_tissue_cohort = 201
    scanner1_inflammation_cohort = 204
    
    # Calculate overlap
    overlap_cohort = 195  # Patients in both tissue and inflammation cohorts
    
    # Generate LaTeX table
    latex = r"""\\begin{table}[ht]
	\\centering
	\\caption{Patient Selection Flow}
	\\label{tab:patient_selection_flow}
	\\begin{tabular}{lrr}
		\\toprule
		Selection Stage & Patient Count & Percentage \\\\
		\\midrule"""
    
    # Add rows for each selection stage
    latex += f"""
		Initial Cohort & {initial_cohort} & {100.0:.1f}\\% \\\\
		Valid Annotation Cohort & {valid_annotation_cohort} & {(valid_annotation_cohort/initial_cohort*100):.1f}\\% \\\\
		Scanner1 Cohort & {scanner1_cohort} & {(scanner1_cohort/initial_cohort*100):.1f}\\% \\\\
		Scanner1 HE Staining Cohort & {scanner1_he_cohort} & {(scanner1_he_cohort/initial_cohort*100):.1f}\\% \\\\
		Scanner1 Tissue Analysis Cohort & {scanner1_tissue_cohort} & {(scanner1_tissue_cohort/initial_cohort*100):.1f}\\% \\\\
		Scanner1 Inflammation Analysis Cohort & {scanner1_inflammation_cohort} & {(scanner1_inflammation_cohort/initial_cohort*100):.1f}\\% \\\\
		\\midrule
		Overlap (in both cohorts) & {overlap_cohort} & {(overlap_cohort/initial_cohort*100):.1f}\\% \\\\
		\\bottomrule
	\\end{{tabular}}
\\end{{table}}"""
    
    return latex

# Calculate and print tile counts from file paths
def calculate_tile_counts_detailed():
    """Calculate tile counts with detailed analysis of any discrepancies"""
    import glob
    import os
    from collections import defaultdict, Counter
    
    # Path to the tiles directory
    tiles_path = os.path.join(BASE_DIR, "data/processed/tiles")
    
    # Create mappings from slide name to split
    split_mapping_he, split_mapping_all = create_split_mappings()
    
    # Initialize counters
    he_counts = {'train': 0, 'val': 0, 'test': 0, 'test_scanner2': 0, 'unassigned': 0}
    all_counts = {'train': 0, 'val': 0, 'test': 0, 'test_scanner2': 0, 'unassigned': 0}
    
    # Dictionary to cache slide names to split mappings
    slide_to_split_he = {}
    slide_to_split_all = {}
    
    # For tracking unassigned files
    unassigned_slides = set()
    slide_counts = defaultdict(int)
    stain_counts = Counter()
    
    print(f"Scanning tiles directory: {tiles_path}")
    
    # Get all PNG files
    png_files = glob.glob(os.path.join(tiles_path, "*.png"))
    total_files = len(png_files)
    print(f"Found {total_files} PNG files")
    
    # Process file counts
    for file_path in png_files:
        # Extract slide name from the file path
        filename = os.path.basename(file_path)
        
        # Extract slide name (the first part before "_particle_")
        if "_particle_" in filename:
            slide_name = filename.split("_particle_")[0]
            slide_counts[slide_name] += 1
            
            # Determine stain type (HE, PAS, MG)
            if "HE" in slide_name:
                is_he = True
                stain_counts["HE"] += 1
            elif "PAS" in slide_name:
                is_he = False
                stain_counts["PAS"] += 1
            elif "MG" in slide_name:
                is_he = False
                stain_counts["MG"] += 1
            else:
                is_he = False
                stain_counts["Unknown"] += 1
            
            # Get the split for this slide
            if slide_name not in slide_to_split_he:
                # Cache the lookup results
                slide_to_split_he[slide_name] = split_mapping_he.get(slide_name)
                slide_to_split_all[slide_name] = split_mapping_all.get(slide_name)
            
            split_he = slide_to_split_he[slide_name]
            split_all = slide_to_split_all[slide_name]
            
            # Update counts
            if split_all:
                all_counts[split_all] += 1
            else:
                all_counts['unassigned'] += 1
                unassigned_slides.add(slide_name)
            
            if is_he:
                if split_he:
                    he_counts[split_he] += 1
                else:
                    he_counts['unassigned'] += 1
                    unassigned_slides.add(slide_name)
        else:
            # File doesn't match expected pattern
            all_counts['unassigned'] += 1
            if "HE" in filename:
                he_counts['unassigned'] += 1
            stain_counts["Pattern mismatch"] += 1
    
    # Print results
    print("\nTile counts per split:")
    print("=====================")
    print(f"{'Split':<20} {'HE Stain':<10} {'All Stains':<10}")
    print(f"{'-'*20} {'-'*10} {'-'*10}")
    
    total_he = 0
    total_all = 0
    
    for split in ['train', 'val', 'test', 'test_scanner2', 'unassigned']:
        if split == 'unassigned':
            split_display = 'Unassigned'
        else:
            split_display = {
                'train': 'Training',
                'val': 'Validation',
                'test': 'Test (Scanner 1)',
                'test_scanner2': 'Test (Scanner 2)'
            }[split]
        
        he_count = he_counts.get(split, 0)
        all_count = all_counts.get(split, 0)
        
        total_he += he_count
        total_all += all_count
        
        print(f"{split_display:<20} {int(he_count):<10,} {int(all_count):<10,}")
    
    print(f"{'-'*20} {'-'*10} {'-'*10}")
    print(f"{'Total':<20} {int(total_he):<10,} {int(total_all):<10,}")
    
    # Print summary of unassigned files if any
    if unassigned_slides:
        print("\nUnassigned slides analysis:")
        print("=========================")
        print(f"Found {len(unassigned_slides)} unassigned slide(s)")
        print("Unassigned slides:")
        for slide in sorted(unassigned_slides):
            print(f"  - {slide}: {slide_counts[slide]} tiles")
    
    # Print stain distribution
    print("\nStain distribution:")
    print("=================")
    for stain, count in stain_counts.most_common():
        print(f"{stain}: {count} tiles ({count/total_files*100:.1f}%)")
    
    # Compare with expected counts from tiles_per_particle.csv
    print("\nComparison with tiles_per_particle.csv:")
    print("=====================================")
    # Calculate expected counts from CSV
    expected_he_count = tiles_per_particle[tiles_per_particle['stain'] == 'HE']['tiles_per_particle'].sum()
    expected_all_count = tiles_per_particle['tiles_per_particle'].sum()
    
    print(f"{'Source':<20} {'HE Stain':<10} {'All Stains':<10}")
    print(f"{'-'*20} {'-'*10} {'-'*10}")
    print(f"{'Filesystem':<20} {total_he:<10,} {total_all:<10,}")
    print(f"{'tiles_per_particle':<20} {int(expected_he_count):<10,} {int(expected_all_count):<10,}")
    print(f"{'Difference':<20} {int(total_he - expected_he_count):<+10,} {int(total_all - expected_all_count):<+10,}")
    
    return he_counts, all_counts, unassigned_slides

# Main function to generate all tables
def main():
    """Generate all LaTeX tables and save to files"""
    # Create output directory with base path
    output_dir = os.path.join(BASE_DIR, "results/tables")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all tables
    tables = {
        "png_tile_distribution": generate_png_tile_distribution_table(),
        "dataset_summary": generate_dataset_summary_table(),
        "stain_distribution": generate_stain_distribution_table(),
        "tiles_per_particle": generate_tiles_per_particle_table(),
        "data_splits": generate_data_splits_table(),
        "train_val_test_composition": generate_he_dataset_composition_table(),
        "inflammation_distribution": generate_inflammation_distribution_table(),
        "patient_selection_flow": generate_patient_selection_flow_table()
    }
    
    # Save tables to files
    for name, latex in tables.items():
        with open(os.path.join(output_dir, f"{name}.tex"), "w") as f:
            f.write(latex)
        print(f"Generated {name}.tex")
    
    # Calculate and print tile counts
    calculate_tile_counts_detailed()
    
    print("\nAll tables have been generated successfully.")
    print(f"Tables saved to: {output_dir}")

if __name__ == "__main__":
    main()