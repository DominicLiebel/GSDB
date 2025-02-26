"""
Histology Dataset Module

This module implements a PyTorch Dataset for histological image analysis.
It handles both inflammation and tissue type classification tasks using HE-stained slides.

Key Features:
- Supports inflammation and tissue type classification tasks
- Handles data splits (train/val/test/test_scanner2)
- Validates tissue content in tiles
- Provides detailed dataset statistics
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Optional, Tuple

class HistologyDataset(Dataset):
    def __init__(self, split: str, transform=None, task: str = 'inflammation', 
                 he_only: bool = True, scanner1_only: bool = False):
        """Initialize the Histology Dataset.
        
        Args:
            split (str): Dataset split ('train', 'val', 'test', 'test_scanner2')
            transform: Optional transforms to apply to images
            task (str): Classification task ('inflammation' or 'tissue')
            he_only (bool): If True, use only HE stained slides (default: True)
            scanner1_only (bool): If True, use only scanner 1 slides (default: False)
        """
        self.split = split
        self.transform = transform
        self.task = task
        self.base_path = Path('/mnt/data/dliebel/2024_dliebel')
        self.tile_path = self.base_path / 'data/processed/tiles'
        
        # Load split information
        split_file = f"{split}_HE.csv"  # All splits use _HE suffix
        split_path = self.base_path / 'data/splits/seed_42' / split_file
        
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
            
        logging.info(f"Loading split file: {split_path}")
        self.split_df = pd.read_csv(split_path)
        
        # Load appropriate metadata based on task
        if task == 'inflammation':
            metadata_path = self.base_path / 'results/metrics/inflammation_status.csv'
            self.metadata_df = pd.read_csv(metadata_path)
        else:  # tissue task
            metadata_path = self.base_path / 'results/metrics/tissue_types.csv'
            self.metadata_df = pd.read_csv(metadata_path)
            
        # Load tile counts
        self.tiles_df = pd.read_csv(self.base_path / 'results/metrics/tiles_per_particle.csv')
        
        # Apply filters to metadata
        if he_only:
            self.metadata_df = self.metadata_df[self.metadata_df['stain'] == 'HE']
            self.tiles_df = self.tiles_df[self.tiles_df['stain'] == 'HE']
            
        if scanner1_only:
            self.metadata_df = self.metadata_df[self.metadata_df['scanner_id'] == 1]
            self.tiles_df = self.tiles_df[self.tiles_df['scanner_id'] == 1]
        
        # Filter by split
        split_slides = set(self.split_df['slide_name'])
        self.metadata_df = self.metadata_df[self.metadata_df['slide_name'].isin(split_slides)]
        self.tiles_df = self.tiles_df[self.tiles_df['slide_name'].isin(split_slides)]
        
        # Load valid samples for the task
        self.samples = self._load_samples()
        
        # Log dataset statistics
        self._log_statistics()

    def _extract_particle_id(self, filepath: str) -> str:
        """Extract particle ID from tile filename."""
        try:
            parts = str(filepath).split('particle_')
            if len(parts) > 1:
                # Get the part after 'particle_' and before next underscore
                particle_id = parts[1].split('_')[0]
                return particle_id
        except Exception as e:
            logging.error(f"Error extracting particle_id from {filepath}: {e}")
        return None
        
    def _load_samples(self):
        """Load valid samples based on task and filters."""
        samples = []
        
        if self.task == 'inflammation':
            # Get inflammation information per slide (only inflamed and noninflamed)
            valid_slides = self.metadata_df[self.metadata_df['inflammation_status'].isin(['inflamed', 'noninflamed'])]
            logging.info(f"Found {len(valid_slides)} valid slides for inflammation evaluation")
            
            # Load tiles_per_particle info which contains both particle and inflammation info
            for _, slide_row in tqdm(valid_slides.iterrows(), desc=f"Loading {self.task} {self.split} dataset"):
                slide_name = slide_row['slide_name']
                inflammation_status = slide_row['inflammation_status']
                
                # Get all particles for this slide
                slide_particles = self.tiles_df[self.tiles_df['slide_name'] == slide_name]
                
                for _, particle_row in slide_particles.iterrows():
                    if particle_row['tiles_per_particle'] > 0:  # Ensure particle has tiles
                        # Find tile files for this particle
                        tile_pattern = f"{slide_name}*particle_{particle_row['particle_id']}*.png"
                        matching_files = list(self.tile_path.glob(tile_pattern))
                        
                        for tile_path in matching_files:
                            if self._is_valid_tile(tile_path):
                                label = 1 if inflammation_status == 'inflamed' else 0
                                samples.append({
                                    'path': tile_path,
                                    'label': label,
                                    'slide_name': slide_name,
                                    'metadata': {
                                        'task': self.task,
                                        'split': self.split,
                                        'inflammation_status': inflammation_status,
                                        'particle_id': particle_row['particle_id'],
                                    }
                                })
        
            
        else:  # tissue task
            # Get valid particles (corpus or antrum)
            valid_particles = self.metadata_df[self.metadata_df['tissue_type'].isin(['corpus', 'antrum'])]
            
            for _, row in tqdm(valid_particles.iterrows(), desc=f"Loading {self.task} {self.split} dataset"):
                # Get tiles for this particle from tiles_df
                particle_tiles = self.tiles_df[
                    (self.tiles_df['slide_name'] == row['slide_name']) & 
                    (self.tiles_df['particle_id'] == row['particle_id'])
                ]
                
                if particle_tiles.empty:
                    logging.warning(f"No tile information found for particle {row['particle_id']} in slide {row['slide_name']}")
                    continue
                
                # Find matching tile files
                tile_pattern = f"{row['slide_name']}*particle_{row['particle_id']}*.png"
                matching_files = list(self.tile_path.glob(tile_pattern))
                
                if not matching_files:
                    logging.warning(f"No tile files found for particle {row['particle_id']} in slide {row['slide_name']}")
                    continue
                
                # Process each matching tile
                for tile_path in matching_files:
                    if self._is_valid_tile(tile_path):
                        label = 1 if row['tissue_type'] == 'corpus' else 0
                        samples.append({
                            'path': tile_path,
                            'label': label,
                            'slide_name': row['slide_name'],
                            'metadata': {
                                'task': self.task,
                                'split': self.split,
                                'tissue_type': row['tissue_type'],
                                'particle_id': row['particle_id'],
                                'cluster_id': row['cluster_id'],
                            }
                        })
        
        if len(samples) == 0:
            logging.warning(f"No valid samples found for {self.task} task in {self.split} split!")
            logging.warning("Please check:")
            logging.warning("1. Tile files exist in the expected location")
            logging.warning("2. Metadata has matching entries for the split")
            logging.warning("3. Particles have the expected tissue types")
            logging.warning("4. Tiles can be matched to particles")
            
        else:
            unique_slides = len(set(s['slide_name'] for s in samples))
            unique_particles = len(set(s['metadata']['particle_id'] for s in samples))
            logging.info(f"Loaded {len(samples)} tiles from {unique_particles} particles across {unique_slides} slides")
            
        return samples

    def _is_valid_tile(self, image_path: Path) -> bool:
        """Validate tile contains sufficient tissue content.
        
        Args:
            image_path (Path): Path to tile image
            
        Returns:
            bool: True if tile contains valid tissue (less than 90% white pixels)
        """
        try:
            with Image.open(image_path) as image:
                img_array = np.array(image.convert("L"))
                white_pixels = np.sum(img_array > 240) / img_array.size
                return white_pixels < 0.9
        except Exception as e:
            logging.warning(f"Error validating tile {image_path}: {e}")
            return False
    
    def _log_statistics(self):
        """Log detailed dataset statistics including class distribution and sample counts."""
        total = len(self.samples)
        if total == 0:
            logging.warning(f"No samples found for {self.task} task in {self.split} split!")
            return
            
        # Calculate basic statistics
        labels = [sample['label'] for sample in self.samples]
        pos_count = sum(labels)
        neg_count = total - pos_count
        unique_wsis = len({sample['slide_name'] for sample in self.samples})
        
        # Log general information
        logging.info(f"\n{self.split} Dataset Statistics ({self.task} task):")
        logging.info(f"Total samples: {total}")
        logging.info(f"Unique Slides: {unique_wsis}")
        
        # Log task-specific distribution
        if self.task == 'inflammation':
            logging.info(f"Inflamed: {pos_count} ({pos_count/total*100:.1f}%)")
            logging.info(f"Non-inflamed: {neg_count} ({neg_count/total*100:.1f}%)")
            
            # Detailed inflammation distribution
            inf_status_ = pd.Series([s['metadata']['inflammation_status'] for s in self.samples]).value_counts()
            logging.info("\nInflammation status distribution:")
            for inf_status, count in inf_status_.items():
                logging.info(f"{inf_status}: {count} ({count/total*100:.1f}%)")
        else:
            logging.info(f"Corpus: {pos_count} ({pos_count/total*100:.1f}%)")
            logging.info(f"Antrum: {neg_count} ({neg_count/total*100:.1f}%)")
            
            # Detailed tissue distribution
            tissue_types = pd.Series([s['metadata']['tissue_type'] for s in self.samples]).value_counts()
            logging.info("\nTissue type distribution:")
            for tissue_type, count in tissue_types.items():
                logging.info(f"{tissue_type}: {count} ({count/total*100:.1f}%)")
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        try:
            with Image.open(sample['path']) as image:
                image = image.convert('RGB')
                if self.transform:
                    image = self.transform(image)
                
                # Create metadata dictionary ensuring slide_name is always included
                metadata = sample['metadata'].copy()
                metadata['slide_name'] = sample['slide_name']  # Always include slide_name
                
                return image, torch.tensor(sample['label'], dtype=torch.float), metadata
                
        except Exception as e:
            logging.error(f"Error loading image {sample['path']}: {e}")
            metadata = sample['metadata'].copy()
            metadata['slide_name'] = sample['slide_name']  # Include slide_name even in error case
            return torch.zeros((3, 224, 224)), torch.tensor(sample['label'], dtype=torch.float), metadata