from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import os
import random
from datetime import datetime
import seaborn as sns
from typing import Dict, List, Tuple, Set
import argparse
from sklearn.model_selection import train_test_split

# Import path configuration
import sys
# Add the project root to the path if not already
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from src.config.paths import get_project_paths, add_path_args

class DataSplitter:
    def __init__(self, paths=None, random_seed: int = 42):
        """Initialize DataSplitter with project paths and configuration.
        
        Args:
            paths (Dict[str, Path]): Dictionary of project paths
            random_seed (int): Random seed for reproducibility
        """
        if paths is None:
            paths = get_project_paths()
        
        self.base_dir = paths["BASE_DIR"]
        self.seed = random_seed
        
        # Setup project directories
        self.data_dir = paths["DATA_DIR"]
        self.raw_dir = paths["RAW_DIR"]
        self.processed_dir = paths["PROCESSED_DIR"]
        self.splits_dir = paths["SPLITS_DIR"]
        self.logs_dir = paths["LOGS_DIR"]
        self.figures_dir = paths["FIGURES_DIR"]
        self.tables_dir = paths["TABLES_DIR"]
        
        # Create necessary directories
        for dir_path in [self.splits_dir, self.logs_dir, self.figures_dir, self.tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging with timestamp and proper formatting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f'split_creation_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def create_slide_info_df(self) -> pd.DataFrame:
        """Create DataFrame with slide information from annotations.
        
        Returns:
            pd.DataFrame: DataFrame containing slide information
        """
        annotations_dir = self.raw_dir / "annotations"
        slides_data = []

        for root, _, files in os.walk(annotations_dir):
            for file in files:
                if not file.endswith('.json'):
                    continue
                    
                try:
                    file_path = Path(root) / file
                    with open(file_path) as f:
                        data = json.load(f)
                        
                        if not data:  # Skip empty files
                            continue
                            
                        base_name = file.replace('_annotations.json', '')
                        parts = base_name.split('_')
                        
                        # Extract metadata according to new naming convention:
                        # format: slide_id_patient_id_scanner_id_stain
                        # e.g., 241_157_2_HE
                        slide_id = parts[0]     # e.g., 241
                        patient_id = parts[1]   # e.g., 157
                        scanner_id = int(parts[2])  # e.g., 2
                        stain = parts[3]        # e.g., HE
                        
                        inflammation_status = data[0]['properties']['classification']['inflammation_status']
                        
                        slides_data.append({
                            'slide_name': f"{slide_id}_{patient_id}_{scanner_id}_{stain}",
                            'slide_id': slide_id,
                            'patient_id': patient_id,
                            'scanner_id': scanner_id,
                            'stain': stain,
                            'inflammation_status': inflammation_status
                        })
                        
                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")
                    continue

        df_slides = pd.DataFrame(slides_data)
        return df_slides

    def create_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create train/val/test splits based on patient_id."""
        # Set all random seeds for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)
        # Log seeding information
        logging.info(f"Set random seed to {self.seed} for reproducible data splitting")
        
        # Log initial statistics
        logging.info(f"\nInitial dataset statistics:")
        logging.info(f"Total patients before exclusion: {df['patient_id'].nunique()}")
        logging.info(f"Total slides before exclusion: {len(df)}")
        
        # Define and log excluded patients
        excluded_patients = {'4', '9', '14', '18', '24', '29', '34', '39', '44', '49', 
                            '54', '59', '64', '69', '74', '75', '79', '83', '180', '197', 
                             '213', '217'}
        logging.info(f"\nExcluding {len(excluded_patients)} patients:")
        logging.info(f"Excluded patient IDs: {sorted(excluded_patients)}")
        
        # Filter out excluded patients and log statistics
        df_filtered = df[~df['patient_id'].isin(excluded_patients)]
        logging.info(f"\nDataset statistics after exclusion:")
        logging.info(f"Remaining patients: {df_filtered['patient_id'].nunique()}")
        logging.info(f"Remaining slides: {len(df_filtered)}")
        logging.info(f"Removed slides: {len(df) - len(df_filtered)}")
        
        # Separate scanner 1 and scanner 2 data with logging
        df_scanner1 = df_filtered[df_filtered['scanner_id'] == 1]
        df_scanner2 = df_filtered[df_filtered['scanner_id'] == 2]
        
        logging.info(f"\nScanner distribution:")
        logging.info(f"Scanner 1 slides: {len(df_scanner1)}")
        logging.info(f"Scanner 1 patients: {df_scanner1['patient_id'].nunique()}")
        logging.info(f"Scanner 2 slides: {len(df_scanner2)}")
        logging.info(f"Scanner 2 patients: {df_scanner2['patient_id'].nunique()}")
        
        # Create splits with scikit-learn for more robust splitting
        unique_patients = df_scanner1['patient_id'].unique()
        
        # Split into train and temp (validation + test combined)
        train_patients, temp_patients = train_test_split(
            unique_patients, 
            test_size=0.4,  # 60% for training, 40% for val+test
            random_state=self.seed
        )
        
        # Split temp into validation and test (50% each of the 40%, resulting in 20% val, 20% test)
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=0.5,  # Half of temp goes to test
            random_state=self.seed
        )
        
        logging.info(f"\nSplit sizes (Scanner 1 patients):")
        logging.info(f"Training: {len(train_patients)} patients ({len(train_patients)/len(unique_patients):.1%})")
        logging.info(f"Validation: {len(val_patients)} patients ({len(val_patients)/len(unique_patients):.1%})")
        logging.info(f"Test: {len(test_patients)} patients ({len(test_patients)/len(unique_patients):.1%})")
        
        splits = {
            'train': df_scanner1[df_scanner1['patient_id'].isin(train_patients)],
            'val': df_scanner1[df_scanner1['patient_id'].isin(val_patients)],
            'test': df_scanner1[df_scanner1['patient_id'].isin(test_patients)],
            'test_scanner2': df_scanner2
        }
        
        # Log final split statistics
        for split_name, split_df in splits.items():
            logging.info(f"\n{split_name.upper()} split:")
            logging.info(f"Slides: {len(split_df)}")
            logging.info(f"Patients: {split_df['patient_id'].nunique()}")
            logging.info("Inflammation distribution:")
            logging.info(split_df['inflammation_status'].value_counts())
        
        self._create_split_variants(splits)
        self._validate_splits(splits)
        
        return splits
    
    def _create_split_variants(self, splits: Dict[str, pd.DataFrame]):
        """Create different variants of splits for different use cases.
        
        Args:
            splits (Dict[str, pd.DataFrame]): Original splits
        """
        for split_name, split_df in splits.items():
            base_path = self.splits_dir / f'seed_{self.seed}'
            base_path.mkdir(exist_ok=True)
            
            # Save HE-only version
            he_only = split_df[split_df['stain'] == 'HE']
            he_only.to_csv(base_path / f'{split_name}_HE.csv', index=False)
            
            # Save complete version with all stains
            split_df.to_csv(base_path / f'{split_name}_all_stains.csv', index=False)
            
    def _validate_splits(self, splits: Dict[str, pd.DataFrame]):
        """Validate the created splits.
        
        Args:
            splits (Dict[str, pd.DataFrame]): Dictionary containing the splits
        """
        logging.info("\nValidating splits...")
        
        # Check for patient_id overlap
        used_patients: Set[str] = set()
        for name, split_df in splits.items():
            if name != 'test_scanner2':  # Exclude scanner2 test set from overlap check
                current_patients = set(split_df['patient_id'])
                overlap = current_patients & used_patients
                assert not overlap, f"Found patient_id overlap in {name}: {overlap}"
                used_patients.update(current_patients)
        
        # Validate scanner separation
        for name, split_df in splits.items():
            if name == 'test_scanner2':
                assert all(split_df['scanner_id'] == 2), \
                    "Scanner 2 test set contains non-scanner-2 slides"
            else:
                assert all(split_df['scanner_id'] == 1), \
                    f"{name} split contains non-scanner-1 slides"
                
        # Log split statistics
        self._log_split_statistics(splits)
        
    def _log_split_statistics(self, splits: Dict[str, pd.DataFrame]):
        """Log detailed statistics about the splits.
        
        Args:
            splits (Dict[str, pd.DataFrame]): Dictionary containing the splits
        """
        for name, split_df in splits.items():
            logging.info(f"\n{name} split statistics:")
            logging.info(f"Total samples: {len(split_df)}")
            logging.info(f"Unique patient_ids: {split_df['patient_id'].nunique()}")
            
            # Stain distribution
            logging.info("\nStain distribution:")
            logging.info(split_df['stain'].value_counts())
            
            # Inflammation distribution
            logging.info("\nInflammation distribution:")
            logging.info(split_df['inflammation_status'].value_counts())
            
            # HE-only statistics
            he_only = split_df[split_df['stain'] == 'HE']
            logging.info(f"\nHE-only samples: {len(he_only)}")
            logging.info("HE inflammation distribution:")
            logging.info(he_only['inflammation_status'].value_counts())
            

def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create dataset splits for histology classification")
    parser = add_path_args(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
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
        paths["LOGS_DIR"] = args.output_dir / "logs"
        paths["FIGURES_DIR"] = args.output_dir / "figures"
        paths["TABLES_DIR"] = args.output_dir / "tables"
    
    start_time = datetime.now()
    
    try:
        # Initialize splitter with configured paths
        splitter = DataSplitter(paths=paths, random_seed=args.seed)
        
        logging.info(f"Starting split creation at {start_time}")
        logging.info(f"Using base directory: {paths['BASE_DIR']}")
        
        # Create slide info DataFrame
        df = splitter.create_slide_info_df()
        
        # Create splits
        splits = splitter.create_split(df)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info(f"Split creation completed successfully. Duration: {duration}")
        
    except Exception as e:
        logging.error(f"Error during split creation: {str(e)}", exc_info=True)
        raise
        
if __name__ == "__main__":
    main()