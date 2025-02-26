import os
import json
import openslide
from datetime import datetime
import time
from PIL import Image
from shapely.geometry import Polygon, Point
import math
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

@dataclass
class ExtractionConfig:
    tile_size: int = 256
    downsample: int = 10
    overlap: int = 64
    num_workers: int = 32
    batch_size: int = 512

def process_tile_batch(params: Tuple) -> List[Optional[Tuple]]:
    """Process a batch of tiles from a single WSI using read_region"""
    wsi_path, tile_coords, polygon, downsample, tile_size = params
    results = []
    
    try:
        wsi = openslide.OpenSlide(str(wsi_path))
        
        for orig_x, orig_y in tile_coords:
            try:
                # Scale coordinates down for polygon check
                ds_x = orig_x // downsample
                ds_y = orig_y // downsample
                
                # Create Point at the center of the downsampled tile
                tile_center = Point(ds_x + tile_size / 2, ds_y + tile_size / 2)
                
                if polygon.contains(tile_center):
                    # Read the region at full resolution
                    tile = wsi.read_region(
                        (orig_x, orig_y),
                        0,  # level 0 = full resolution
                        (tile_size * downsample, tile_size * downsample)
                    )
                    # Convert to RGB and resize
                    tile = tile.convert("RGB")
                    tile = tile.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
                    tile_array = np.asarray(tile, dtype=np.uint8, order='C')
                    results.append((orig_x, orig_y, tile_array))
                    
            except Exception as e:
                logging.error(f"Error processing tile at ({orig_x}, {orig_y}): {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Error in process_tile_batch: {str(e)}")
        return []
        
    finally:
        if 'wsi' in locals():
            wsi.close()
            
    return results

class WSITileExtractor:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.setup_logging()
        logging.info(f"Using PIL version {Image.__version__}")
        logging.info(f"Configuration: tile_size={config.tile_size}, "
                    f"downsample={config.downsample}, "
                    f"overlap={config.overlap}, "
                    f"num_workers={config.num_workers}, "
                    f"batch_size={config.batch_size}")

    def setup_logging(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def load_annotations(json_file: Path) -> Optional[List[dict]]:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Annotation data must be a list")
            return data
        except Exception as e:
            logging.error(f"Error loading {json_file}: {str(e)}")
            return None

    def is_processed(self, output_dir: Path, wsi_name: str, particle_id: str) -> bool:
        """Check if tiles for this WSI and particle_id combination have already been processed"""
        pattern = f"{wsi_name}_particle_{particle_id}_tile_*"
        existing_files = list(output_dir.glob(pattern))
        return len(existing_files) > 0

    def extract_tiles(self, wsi_path: Path, annotation_data: List[dict], output_dir: Path) -> None:
        try:
            with openslide.OpenSlide(str(wsi_path)) as wsi:
                wsi_name = wsi_path.stem
            
            self.logger.info(f"Processing {wsi_path.stem} from {wsi_path.parent}")

            for idx, annotation in enumerate(annotation_data):
                try:
                    particle_id = annotation['id']

                    # Check if this specific particle has already been processed
                    if self.is_processed(output_dir, wsi_name, particle_id):
                        self.logger.info(f"Skipping {wsi_name} particle {particle_id} - already processed")
                        continue

                    coords = annotation['geometry']['coordinates'][0]
                    
                    # Create polygon with original coordinates
                    original_polygon = Polygon(coords)
                    
                    # Create downsampled polygon for containment checks
                    polygon = Polygon([
                        (x // self.config.downsample, y // self.config.downsample) 
                        for x, y in original_polygon.exterior.coords
                    ])

                    min_x, min_y, max_x, max_y = original_polygon.bounds

                    # Calculate step size based on tile size and overlap in original coordinates
                    step_size = (self.config.tile_size - self.config.overlap) * self.config.downsample
                    tile_coords = []
                    
                    # Generate tile coordinates with overlap in original coordinates
                    current_x = int(min_x)
                    while current_x < max_x:
                        current_y = int(min_y)
                        while current_y < max_y:
                            tile_coords.append((current_x, current_y))
                            current_y += step_size
                        current_x += step_size

                    coord_batches = [
                        tile_coords[i:i + self.config.batch_size] 
                        for i in range(0, len(tile_coords), self.config.batch_size)
                    ]

                    with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                        try:
                            futures = []
                            for batch in coord_batches:
                                future = executor.submit(
                                    process_tile_batch,
                                    (
                                        wsi_path,
                                        batch,
                                        polygon,
                                        self.config.downsample,
                                        self.config.tile_size
                                    )
                                )
                                futures.append(future)

                            for future in tqdm(as_completed(futures), 
                                            total=len(futures),
                                            desc=f"Processing {wsi_name} particle {idx}"):
                                try:
                                    results = future.result()
                                    if results:
                                        for orig_x, orig_y, tile_array in results:
                                            try:
                                                tile = Image.fromarray(tile_array, mode='RGB')
                                                
                                                # Calculate original size of the extracted region
                                                original_size = self.config.tile_size * self.config.downsample
                                                
                                                # Updated filename pattern to use particle_id
                                                filename = (
                                                    f"{wsi_name}_particle_{particle_id}_"
                                                    f"tile_{orig_x}_{orig_y}_"
                                                    f"size_{original_size}x{original_size}.png"
                                                )
                                                
                                                tile.save(output_dir / filename, "PNG", quality=100)
                                                
                                            except Exception as e:
                                                self.logger.error(f"Error saving tile at ({orig_x}, {orig_y}): {str(e)}")
                                                continue
                                except Exception as e:
                                    self.logger.error(f"Error processing batch result: {str(e)}")
                                    continue
                                    
                        except KeyboardInterrupt:
                            self.logger.warning("Caught KeyboardInterrupt, cleaning up...")
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                            
                except Exception as e:
                    self.logger.error(f"Error processing particle {idx}: {str(e)}")
                    continue

        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by user")
            raise
        except Exception as e:
            self.logger.error(f"Error processing WSI {wsi_path}: {str(e)}")

def main():
    start_time = time.time()
    config = ExtractionConfig()
    
    # Updated paths according to new structure
    base_dir = Path('/mnt/data/dliebel/2024_dliebel')
    wsi_dir = base_dir / 'data/raw/slides'
    annotation_dir = base_dir / 'data/raw/annotations'
    output_dir = base_dir / 'data/processed/tiles'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging in results directory
    log_dir = base_dir / 'results/logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'tile_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    extractor = WSITileExtractor(config)
    wsi_files = list(wsi_dir.glob('**/*.mrxs'))
    
    for wsi_path in tqdm(wsi_files, desc="Processing WSIs"):
        annotation_path = annotation_dir / f"{wsi_path.stem}_annotations.json"
        
        if not annotation_path.exists():
            logging.warning(f"No annotation file found for {wsi_path.stem}")
            continue
            
        annotation_data = extractor.load_annotations(annotation_path)
        if not annotation_data:
            continue
            
        # Remove the create_output_directory call since we're using a fixed path
        extractor.extract_tiles(wsi_path, annotation_data, output_dir)
    
    total_wsi = len(wsi_files)
    for wsi_idx, wsi_path in enumerate(tqdm(wsi_files, desc="Processing WSIs"), 1):
        logging.info(f"Processing WSI {wsi_idx}/{total_wsi}: {wsi_path.stem}")
        
    execution_time = time.time() - start_time
    logging.info(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()