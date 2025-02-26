"""
Unified Dataset Processor

This script processes annotation files and generates consistent metadata files
for both HE-only and all-stains datasets, including detailed statistics.
"""

import json
import pandas as pd
import os
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict, List, Set
from collections import defaultdict
import argparse
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import path configuration
from src.config.paths import get_project_paths, add_path_args

class DatasetProcessor:
    def __init__(self, paths=None):
        """
        Initialize DatasetProcessor with project paths.
        
        Args:
            paths (Dict[str, Path], optional): Dictionary of project paths
        """
        if paths is None:
            paths = get_project_paths()
        
        self.base_dir = paths["BASE_DIR"]
        self.annotations_dir = paths["RAW_DIR"] / 'annotations'
        self.tiles_dir = paths["PROCESSED_DIR"] / 'tiles'
        self.metrics_dir = paths["METRICS_DIR"]
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def parse_slide_info(self, filename: str) -> Dict:
        parts = filename.split('_')
        return {
            'slide_name': '_'.join(parts[:4]),
            'slide_id': parts[0],
            'patient_id': parts[1],
            'scanner_id': parts[2],
            'stain': parts[3]
        }
    
    def process_annotation_file(self, file_path: Path) -> tuple[Dict, List[Dict]]:
        with open(file_path, 'r') as f:
            annotations = json.load(f)
        
        slide_info = self.parse_slide_info(file_path.name.split('_annotations.')[0])
        inflammation_status = None
        tissue_data = []
        
        # Even if annotations is empty, we still want to include this slide
        if not annotations:
            logging.info(f"Empty annotations file found for {slide_info['slide_name']}")
            inflammation_data = {
                **slide_info,
                'inflammation_status': None  # Use None for empty annotation files
            }
            return inflammation_data, tissue_data
            
        for annotation in annotations:
            if "properties" in annotation and "classification" in annotation["properties"]:
                classification = annotation["properties"]["classification"]
                
                if "inflammation_status" in classification and inflammation_status is None:
                    inflammation_status = classification["inflammation_status"]
                
                if "tissue_type" in classification:
                    particle_data = {
                        **slide_info,
                        'particle_id': annotation["id"],
                        'tissue_type': classification["tissue_type"],
                        'cluster_id': annotation["properties"].get("cluster_id", None)
                    }
                    tissue_data.append(particle_data)
        
        inflammation_data = {
            **slide_info,
            'inflammation_status': inflammation_status
        }
        
        return inflammation_data, tissue_data
    
    def find_all_slides(self) -> Set[str]:
        """Find all slides by analyzing both annotation files and tile directory."""
        all_slides = set()
        
        # Get slides from annotation files
        for json_file in self.annotations_dir.glob('*_annotations.json'):
            slide_name = json_file.name.split('_annotations.')[0]
            all_slides.add(slide_name)
        
        # Get slides from tiles directory to catch any without annotations
        for png_file in self.tiles_dir.glob('**/*.png'):
            filename = png_file.name
            try:
                parts = filename.split('_')
                if len(parts) >= 4:  # Make sure we have enough parts for a slide name
                    slide_name = '_'.join(parts[:4])
                    all_slides.add(slide_name)
            except Exception:
                continue
                
        return all_slides
    
    def count_tiles_per_particle(self) -> pd.DataFrame:
        particle_annotations = {}
        logging.info("Loading annotations...")
        for json_file in tqdm(list(self.annotations_dir.glob('*_annotations.json')), desc="Processing annotations"):
            try:
                with open(json_file, 'r') as f:
                    annotations = json.load(f)
                
                # Extract slide name for empty annotation files
                slide_name = json_file.name.split('_annotations.')[0]
                
                # Handle empty annotation files
                if not annotations:
                    continue
                    
                for annotation in annotations:
                    if "properties" in annotation and "classification" in annotation["properties"]:
                        particle_id = annotation["id"]
                        classification = annotation["properties"]["classification"]
                        particle_annotations[particle_id] = {
                            'tissue_type': classification.get('tissue_type'),
                            'inflammation_status': classification.get('inflammation_status')
                        }
            except Exception as e:
                logging.warning(f"Error reading annotation file {json_file}: {str(e)}")
        
        tiles_per_particle = defaultdict(int)
        slide_info = {}
        
        logging.info("Counting tiles per particle...")
        for png_file in tqdm(list(self.tiles_dir.glob('**/*.png')), desc="Processing tiles"):
            filename = png_file.name
            try:
                parts = filename.split('_')
                if len(parts) < 10:
                    continue
                
                slide_id = parts[0]
                patient_id = parts[1]
                scanner_id = parts[2]
                stain = parts[3]
                
                particle_idx = parts.index('particle')
                if particle_idx + 1 >= len(parts):
                    continue
                particle_id = parts[particle_idx + 1]
                
                tiles_per_particle[particle_id] += 1
                
                if particle_id not in slide_info:
                    slide_info[particle_id] = {
                        'slide_name': f"{slide_id}_{patient_id}_{scanner_id}_{stain}",
                        'slide_id': int(slide_id),
                        'patient_id': int(patient_id),
                        'scanner_id': int(scanner_id),
                        'stain': stain
                    }
                    
            except (ValueError, IndexError) as e:
                logging.warning(f"Could not parse filename: {filename} - {str(e)}")
                continue
        
        data = []
        for particle_id, tile_count in tiles_per_particle.items():
            if particle_id in slide_info:
                info = slide_info[particle_id]
                annotation_info = particle_annotations.get(particle_id, {})
                
                data.append({
                    'slide_name': info['slide_name'],
                    'slide_id': info['slide_id'],
                    'patient_id': info['patient_id'],
                    'scanner_id': info['scanner_id'],
                    'stain': info['stain'],
                    'particle_id': particle_id,
                    'tiles_per_particle': tile_count,
                    'tissue_type': annotation_info.get('tissue_type', 'unknown'),
                    'inflammation_status': annotation_info.get('inflammation_status', 'unknown')
                })
        
        return pd.DataFrame(data)
    
    def process_dataset(self):
        """Process the complete dataset and save single source metadata files."""
        # Initialize data collections
        inflammation_data = []
        tissue_data = []
        processed_slides = set()
        
        # Get all slides - both from annotations directory and tiles directory
        all_slides = self.find_all_slides()
        logging.info(f"Found {len(all_slides)} total slides")
        
        # Process annotation files
        annotation_files = list(self.annotations_dir.glob('*_annotations.json'))
        logging.info(f"Found {len(annotation_files)} annotation files")
        
        for file_path in tqdm(annotation_files, desc="Processing annotations"):
            try:
                infl_data, tiss_data = self.process_annotation_file(file_path)
                
                if infl_data['slide_name'] not in processed_slides:
                    inflammation_data.append(infl_data)
                    processed_slides.add(infl_data['slide_name'])
                
                tissue_data.extend(tiss_data)
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
        
        # Check for slides that might have no annotation files
        missing_slides = all_slides - processed_slides
        if missing_slides:
            logging.info(f"Found {len(missing_slides)} slides without processed annotations")
            for slide_name in missing_slides:
                try:
                    parts = slide_name.split('_')
                    if len(parts) >= 4:
                        slide_info = {
                            'slide_name': slide_name,
                            'slide_id': parts[0],
                            'patient_id': parts[1],
                            'scanner_id': parts[2],
                            'stain': parts[3],
                            'inflammation_status': None  # No annotation data available
                        }
                        inflammation_data.append(slide_info)
                        processed_slides.add(slide_name)
                except Exception as e:
                    logging.error(f"Error processing slide name {slide_name}: {str(e)}")
        
        # Create DataFrames with consistent column ordering
        inflammation_df = pd.DataFrame(inflammation_data)
        tissue_df = pd.DataFrame(tissue_data)
        
        # Define column order
        base_columns = ['slide_name', 'slide_id', 'patient_id', 'scanner_id', 'stain']
        inflammation_columns = base_columns + ['inflammation_status']
        tissue_columns = base_columns + ['particle_id', 'tissue_type', 'cluster_id']
        
        # Ensure all columns exist
        for col in inflammation_columns:
            if col not in inflammation_df.columns:
                inflammation_df[col] = None
                
        for col in tissue_columns:
            if col not in tissue_df.columns:
                tissue_df[col] = None
        
        inflammation_df = inflammation_df[inflammation_columns]
        tissue_df = tissue_df[tissue_columns]
        
        # Get tile counts
        particle_tiles_df = self.count_tiles_per_particle()
        
        # Save unfiltered metadata files
        inflammation_df.to_csv(self.metrics_dir / 'inflammation_status.csv', index=False)
        tissue_df.to_csv(self.metrics_dir / 'tissue_types.csv', index=False)
        particle_tiles_df.to_csv(self.metrics_dir / 'tiles_per_particle.csv', index=False)
        
        # Generate and save complete statistics
        stats = {
            'total_slides': len(processed_slides),
            'inflammation_distribution': inflammation_df['inflammation_status'].value_counts(dropna=False).to_dict(),
            'tissue_distribution': tissue_df['tissue_type'].value_counts(dropna=False).to_dict(),
            'slides_per_patient': inflammation_df.groupby('patient_id')['slide_name'].nunique().to_dict(),
            'particles_per_slide': tissue_df.groupby('slide_name')['particle_id'].nunique().to_dict(),
            'scanner_distribution': inflammation_df['scanner_id'].value_counts().to_dict(),
            'stain_distribution': inflammation_df['stain'].value_counts().to_dict()
        }
        
        with open(self.metrics_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Log statistics
        logging.info("\nDataset Statistics:")
        logging.info(f"Total slides processed: {len(processed_slides)}")
        logging.info("\nInflammation Status Distribution:")
        logging.info(inflammation_df['inflammation_status'].value_counts(dropna=False))
        logging.info("\nTissue Type Distribution:")
        logging.info(tissue_df['tissue_type'].value_counts(dropna=False))
        logging.info("\nScanner Distribution:")
        logging.info(inflammation_df['scanner_id'].value_counts())
        logging.info("\nStain Distribution:")
        logging.info(inflammation_df['stain'].value_counts())
        
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process dataset and generate metadata files")
    parser = add_path_args(parser)
    args = parser.parse_args()
    
    # Get project paths with any overrides from command line
    paths = get_project_paths(base_dir=args.base_dir)
    
    # Override specific directories if provided
    if args.data_dir:
        paths["DATA_DIR"] = args.data_dir
        paths["RAW_DIR"] = args.data_dir / "raw"
        paths["PROCESSED_DIR"] = args.data_dir / "processed" 
    
    if args.output_dir:
        paths["RESULTS_DIR"] = args.output_dir
        paths["METRICS_DIR"] = args.output_dir / "metrics"
    
    # Print paths
    print("Project paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    processor = DatasetProcessor(paths)
    processor.process_dataset()

if __name__ == "__main__":
    main()