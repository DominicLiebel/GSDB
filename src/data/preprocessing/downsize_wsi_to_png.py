import openslide
import os
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image
import argparse
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import path configuration
from src.config.paths import get_project_paths, add_path_args

# Global constants
DOWNSAMPLE = 16

def setup_logging(log_dir: Path) -> None:
    """Set up logging configuration with rotating file handler and enhanced formatting"""
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f'downsample_wsi_to_png_{datetime.now():%Y%m%d_%H%M%S}.log'
    
    # Create a formatter with detailed information
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Set up console handler with less verbose formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")

def downsample_wsi(wsi_file: str, output_dir: str, downsample: int = DOWNSAMPLE) -> bool:
    """Downsample WSI and save the result"""
    wsi_name = Path(wsi_file).stem
    try:
        logging.debug(f"Opening WSI file: {wsi_file}")
        wsi = openslide.OpenSlide(wsi_file)
        
        # Log original dimensions and target dimensions
        logging.debug(f"Original dimensions: {wsi.dimensions}")
        new_width = wsi.dimensions[0] // downsample
        new_height = wsi.dimensions[1] // downsample
        logging.debug(f"Target dimensions: {new_width}x{new_height}")
        
        # Find the best level and log downsampling details
        best_level = wsi.get_best_level_for_downsample(downsample)
        level_downsample = wsi.level_downsamples[best_level]
        logging.debug(f"Selected level {best_level} with downsample factor {level_downsample}")
        
        # Calculate and log the region size at this level
        level_width = int(wsi.dimensions[0] / level_downsample)
        level_height = int(wsi.dimensions[1] / level_downsample)
        logging.debug(f"Reading region at level {best_level}: {level_width}x{level_height}")
        
        img = wsi.read_region((0, 0), best_level, (level_width, level_height))
        img = img.convert('RGB')
        
        # Additional downsampling if needed
        if level_downsample < downsample:
            additional_downsample = downsample / level_downsample
            new_width = int(level_width / additional_downsample)
            new_height = int(level_height / additional_downsample)
            logging.debug(f"Additional resize needed. New dimensions: {new_width}x{new_height}")
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save downsampled image
        output_file = Path(output_dir) / f"{wsi_name}_downsampled{DOWNSAMPLE}x.png"
        logging.debug(f"Saving image to: {output_file}")
        img.save(output_file, format="PNG", optimize=True)
        
        file_size = output_file.stat().st_size / (1024 * 1024)  # Size in MB
        logging.info(f"Successfully saved downsampled WSI: {output_file} (Size: {file_size:.2f} MB)")
        
        wsi.close()
        return True
        
    except Exception as e:
        logging.error(f"Error processing {wsi_name}: {str(e)}", exc_info=True)
        if 'wsi' in locals():
            wsi.close()
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Downsample WSI files to PNG format")
    parser = add_path_args(parser)
    parser.add_argument("--downsample", type=int, default=DOWNSAMPLE, help="Downsample factor")
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
        paths["LOGS_DIR"] = args.output_dir / "logs"
    
    # Setup paths using global directories
    wsi_base_dir = paths["RAW_DIR"] / "slides"
    downsized_dir = paths["PROCESSED_DIR"] / "downsized_slides"
    log_dir = paths["LOGS_DIR"]
    
    # Create output directory and setup logging
    downsized_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(log_dir)
    
    logging.info(f"Starting WSI downsampling process")
    logging.info(f"Input directory: {wsi_base_dir}")
    logging.info(f"Output directory: {downsized_dir}")
    logging.info(f"Downsample factor: {DOWNSAMPLE}")
    
    # Print paths
    logging.info("Project paths:")
    for name, path in paths.items():
        logging.info(f"  {name}: {path}")
    
    # Find all WSI files
    wsi_files = list(wsi_base_dir.rglob("*.mrxs"))
    total_wsis = len(wsi_files)
    logging.info(f"Found {total_wsis} WSI files to process")
    
    # Process each WSI file
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    for idx, wsi_file in enumerate(wsi_files, 1):
        wsi_name = wsi_file.stem
        output_pattern = f"{wsi_name}_downsampled{DOWNSAMPLE}x.png"
        output_path = downsized_dir / output_pattern
        
        if output_path.exists():
            logging.info(f"Skipping {wsi_name} ({idx}/{total_wsis}) - Already processed")
            skipped_count += 1
            continue
            
        logging.info(f"Processing {wsi_name} ({idx}/{total_wsis})")
        
        if downsample_wsi(str(wsi_file), str(downsized_dir), DOWNSAMPLE):
            successful_count += 1
        else:
            failed_count += 1
    
    # Log final statistics
    logging.info("Processing complete!")
    logging.info(f"Total files found: {total_wsis}")
    logging.info(f"Successfully processed: {successful_count}")
    logging.info(f"Failed: {failed_count}")
    logging.info(f"Skipped (already processed): {skipped_count}")

if __name__ == "__main__":
    main()