"""
Histology Auto-Classification Module

This module implements automatic classification of histology images using pre-trained models.
It supports multiple architectures and can automatically detect model paths.
"""

# Standard library imports
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import cv2
import numpy as np
import openslide
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Add project root to path if not already
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import path configuration, but make it optional to handle standalone usage
try:
    # Try to import directly
    from src.config.paths import get_project_paths, get_base_dir
except ImportError:
    # Try to load from specific location if direct import fails
    try:
        paths_module_path = Path("/mnt/data/dliebel/2024_dliebel/src/config/paths.py")
        if paths_module_path.exists():
            spec = importlib.util.spec_from_file_location("paths", paths_module_path)
            paths_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(paths_module)
            get_project_paths = paths_module.get_project_paths
            get_base_dir = paths_module.get_base_dir
        else:
            # Define a simple fallback if running standalone
            def get_base_dir():
                """Get base directory from current file path."""
                return Path(__file__).resolve().parent.parent.parent
                
            def get_project_paths(base_dir=None):
                """Get default project paths using the base directory."""
                if base_dir is None:
                    base_dir = get_base_dir()
                return {
                    "BASE_DIR": base_dir,
                    "MODELS_DIR": base_dir / "results" / "models"
                }
    except Exception:
        # Final fallback if all else fails
        def get_base_dir():
            """Get base directory from current file path."""
            return Path(__file__).resolve().parent.parent.parent
            
        def get_project_paths(base_dir=None):
            """Get default project paths using the base directory."""
            if base_dir is None:
                base_dir = get_base_dir()
            return {
                "BASE_DIR": base_dir,
                "MODELS_DIR": base_dir / "results" / "models"
            }


class AutoClassifier:
    """Automatic classifier for histology regions.
    
    This class handles the loading of pretrained tissue and inflammation models,
    extraction of tiles from regions, and classification of those regions.
    
    Attributes:
        ANNOTATION_OPTIONS: Dictionary containing annotation options for coloring
        tissue_model_arch: Model architecture for tissue classification
        inflammation_model_arch: Model architecture for inflammation classification
        paths: Dictionary with paths configuration
        TISSUE_MODEL_PATH: Path to the tissue model file
        INFLAMMATION_MODEL_PATH: Path to the inflammation model file
        TILE_SIZE: Size of tiles to extract from regions
        OVERLAP: Overlap between tiles
        DOWNSAMPLE: Downsample factor for coordinate conversion
        device: PyTorch device to use for inference
        tissue_model: Loaded tissue classification model
        inflammation_model: Loaded inflammation classification model
        transform: Image transform pipeline for model input
    """
    
    def __init__(
            self, 
            annotation_options: Optional[Dict] = None, 
            paths: Optional[Dict[str, Path]] = None, 
            custom_tissue_model: Optional[Union[str, Path]] = None, 
            custom_inflammation_model: Optional[Union[str, Path]] = None,
            tissue_model_arch: Optional[str] = None, 
            inflammation_model_arch: Optional[str] = None
    ):
        """Initialize classifier with model paths and configuration.
        
        Args:
            annotation_options: Dictionary of annotation options for coloring
            paths: Dictionary with paths configuration
            custom_tissue_model: Path to custom tissue model
            custom_inflammation_model: Path to custom inflammation model
            tissue_model_arch: Model architecture for tissue model
            inflammation_model_arch: Model architecture for inflammation model
            
        Raises:
            Exception: If models cannot be initialized properly
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.ANNOTATION_OPTIONS = annotation_options
        
        # Store model architecture information
        self.tissue_model_arch = tissue_model_arch or "resnet18"  # Default to ResNet18
        self.inflammation_model_arch = inflammation_model_arch or "resnet18"  # Default to ResNet18
        
        # Setup paths using the path configuration system
        if paths is None:
            try:
                self.paths = get_project_paths()
            except Exception as e:
                self.logger.warning(f"Could not get project paths: {e}. Using default paths.")
                self.paths = {
                    "MODELS_DIR": Path("./results/models")
                }
        else:
            self.paths = paths
        
        # Use custom paths if provided, otherwise find automatically
        if custom_tissue_model and Path(custom_tissue_model).exists():
            self.TISSUE_MODEL_PATH = Path(custom_tissue_model)
            self.logger.info(f"Using custom tissue model: {self.TISSUE_MODEL_PATH}")
        else:
            self.TISSUE_MODEL_PATH = self._find_model("tissue")
            self.logger.info(f"Using auto-detected tissue model: {self.TISSUE_MODEL_PATH}")
            
        if custom_inflammation_model and Path(custom_inflammation_model).exists():
            self.INFLAMMATION_MODEL_PATH = Path(custom_inflammation_model)
            self.logger.info(f"Using custom inflammation model: {self.INFLAMMATION_MODEL_PATH}")
        else:
            self.INFLAMMATION_MODEL_PATH = self._find_model("inflammation")
            self.logger.info(f"Using auto-detected inflammation model: {self.INFLAMMATION_MODEL_PATH}")
        
        # Tile parameters matching training
        self.TILE_SIZE = 256
        self.OVERLAP = 64
        self.DOWNSAMPLE = 10
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        try:
            # Try to use model_utils for loading if available
            model_utils = self._try_import_model_utils()
            
            if model_utils:
                self.logger.info("Using model_utils to load models")
                # Load tissue model
                self.tissue_model = model_utils.load_model(
                    self.TISSUE_MODEL_PATH, 
                    self.device, 
                    architecture=self.tissue_model_arch
                )
                
                # Load inflammation model
                self.inflammation_model = model_utils.load_model(
                    self.INFLAMMATION_MODEL_PATH, 
                    self.device, 
                    architecture=self.inflammation_model_arch
                )
                
                # Get appropriate transforms
                self.transform = model_utils.get_transforms(is_training=False)
                
            else:
                # Fall back to manual model loading
                self.logger.info("Falling back to manual model loading")
                # Initialize models with correct architectures
                self.tissue_model = self._create_model(self.tissue_model_arch)
                self.inflammation_model = self._create_model(self.inflammation_model_arch)
                
                # Load models
                self.tissue_model.load_state_dict(
                    torch.load(self.TISSUE_MODEL_PATH, map_location=self.device, weights_only=True)
                )
                self.inflammation_model.load_state_dict(
                    torch.load(self.INFLAMMATION_MODEL_PATH, map_location=self.device, weights_only=True)
                )
                
                # Setup transform to match validation transform from training
                self.transform = transforms.Compose([
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
            # Set both models to evaluation mode
            self.tissue_model.eval()
            self.inflammation_model.eval()
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def _try_import_model_utils(self) -> Optional[object]:
        """Try to import model_utils from different locations.
        
        Returns:
            The model_utils module if found, None otherwise
        """
        try:
            # First try from project structure using the paths module
            try:
                from src.config.paths import get_project_paths
                paths = get_project_paths()
                base_dir = paths.get("BASE_DIR")
                if base_dir:
                    model_utils_path = base_dir / "src" / "models" / "model_utils.py"
                    if model_utils_path.exists():
                        self.logger.info(f"Importing model_utils from {model_utils_path}")
                        spec = importlib.util.spec_from_file_location(
                            "model_utils", model_utils_path
                        )
                        model_utils = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(model_utils)
                        return model_utils
            except Exception as e:
                self.logger.warning(f"Could not import model_utils using paths module: {e}")
            
            # Then try from regular import path
            self.logger.info("Attempting to import model_utils from regular path")
            from src.models import model_utils
            return model_utils
            
        except Exception as e:
            self.logger.warning(f"Could not import model_utils: {e}")
            return None

    def _find_model(self, task: str) -> Path:
        """Find the latest model for the given task.
        
        Args:
            task: Task type (tissue or inflammation)
            
        Returns:
            Path to the model file
        """
        # First check models directory
        models_dir = self.paths.get("MODELS_DIR", Path("./results/models"))
        
        # Primary search patterns
        patterns = [
            f"best_model_{task}_*.pt",  # Standard format from train.py
            f"best_model_{task}_*.pth", # Alternative extension
            f"{task}_*.pt",             # Alternative format
            f"{task}_*.pth"             # Alternative format with different extension
        ]
        
        # Search for models using the patterns
        for pattern in patterns:
            model_files = list(models_dir.glob(pattern))
            if model_files:
                # Sort by modification time to get the latest model
                return sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        
        # If not found in models directory, try other common locations
        backup_dirs = [
            Path("./models"),
            Path("./weights"),
            Path("./trained_models")
        ]
        
        for backup_dir in backup_dirs:
            if not backup_dir.exists():
                continue
                
            for pattern in patterns:
                model_files = list(backup_dir.glob(pattern))
                if model_files:
                    return sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        
        # Fallback to the hard-coded paths if absolutely necessary
        # This maintains backward compatibility
        fallback_paths = {
            "tissue": Path("./best_model_tissue.pth"),
            "inflammation": Path("./best_model_inflammation.pth")
        }
        
        # Log warning about using fallback
        self.logger.warning(
            f"Could not find {task} model in standard locations. "
            f"Using fallback path: {fallback_paths[task]}"
        )
        
        return fallback_paths[task]

    def _create_model(self, architecture: str) -> nn.Module:
        """Initialize model with the specified architecture.
        
        Args:
            architecture: Model architecture to use (e.g., 'gigapath', 'resnet18', 'swin_v2_b')
            
        Returns:
            Initialized PyTorch model
        """
        # Normalize architecture string for comparison
        arch_lower = architecture.lower()
        
        # Create model based on architecture
        if 'gigapath' in arch_lower:
            try:
                # Attempt to import GigaPathClassifier - this is a fallback if model_utils failed
                try:
                    from src.models.architectures import GigaPathClassifier
                    model = GigaPathClassifier(num_classes=1)
                except ImportError:
                    # Simplified fallback model
                    self.logger.warning(
                        "Could not import GigaPathClassifier, using ResNet50 as fallback"
                    )
                    model = models.resnet50(weights=None)
                    model.fc = nn.Linear(model.fc.in_features, 1)
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize GigaPath model: {e}. Using ResNet18 fallback."
                )
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 1)
        
        elif 'resnet18' in arch_lower:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)
        
        elif 'swin_v2_b' in arch_lower:
            try:
                # Initialize Swin Transformer V2-B
                model = models.swin_v2_b(weights=None)
                # Replace the classifier head
                model.head = nn.Linear(model.head.in_features, 1)
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize Swin V2-B model: {e}. Using fallback."
                )
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 1)
        
        elif 'convnext_large' in arch_lower:
            try:
                # Initialize ConvNeXt Large
                model = models.convnext_large(weights=None)
                # Replace the classifier head
                model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize ConvNeXt Large model: {e}. Using fallback."
                )
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 1)
        
        elif 'densenet121' in arch_lower:
            model = models.densenet121(weights=None)
            # Replace classifier
            model.classifier = nn.Linear(model.classifier.in_features, 1)
        
        elif 'densenet169' in arch_lower:
            model = models.densenet169(weights=None)
            # Replace classifier
            model.classifier = nn.Linear(model.classifier.in_features, 1)
        
        else:
            # Default to ResNet18 if architecture not recognized
            self.logger.warning(
                f"Unrecognized architecture: {architecture}. Using ResNet18 as fallback."
            )
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)
            
        return model.to(self.device)

    def extract_tiles_from_region(
            self, 
            slide_path: str, 
            coords: List[List[float]]
    ) -> List[Image.Image]:
        """Extract overlapping tiles from a region in the WSI.
        
        Args:
            slide_path: Path to the slide file
            coords: List of coordinate pairs defining the region boundary
            
        Returns:
            List of extracted image tiles from the region
            
        Raises:
            Exception: If tile extraction fails
        """
        try:
            # Open the slide
            slide = openslide.OpenSlide(str(slide_path))
            
            # Get bounding box of region
            coords_np = np.array(coords)
            min_x, min_y = np.min(coords_np, axis=0)
            max_x, max_y = np.max(coords_np, axis=0)
            
            # Scale coordinates to level 0
            level = self.DOWNSAMPLE
            min_x, min_y = int(min_x * level), int(min_y * level)
            max_x, max_y = int(max_x * level), int(max_y * level)
            width = max_x - min_x
            height = max_y - min_y
            
            # Calculate tile grid
            tiles = []
            tile_size = self.TILE_SIZE * level
            overlap = self.OVERLAP * level
            
            for y in range(min_y, max_y, tile_size - overlap):
                for x in range(min_x, max_x, tile_size - overlap):
                    # Read tile
                    tile = slide.read_region(
                        (x, y), 
                        0,  # level 0
                        (tile_size, tile_size)
                    )
                    tile = tile.convert('RGB')
                    
                    # Create mask for the region
                    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
                    shifted_coords = coords_np * level - [min_x, min_y]
                    cv2.fillPoly(mask, [shifted_coords.astype(np.int32)], 255)
                    
                    # Only keep tile if it overlaps with the region
                    if np.sum(mask) > 0:
                        # Convert to numpy, apply mask, and back to PIL
                        tile_np = np.array(tile)
                        tile_np = cv2.bitwise_and(tile_np, tile_np, mask=mask)
                        tile = Image.fromarray(tile_np)
                        tiles.append(tile)
            
            slide.close()
            return tiles
            
        except Exception as e:
            self.logger.error(f"Error extracting tiles: {str(e)}")
            raise

    def classify_region(
            self, 
            tiles: List[Image.Image]
    ) -> Tuple[str, str, Dict[str, float], Dict[str, float]]:
        """Classify tissue type and inflammation status using voting from tiles.
        
        Args:
            tiles: List of image tiles from the region
            
        Returns:
            Tuple containing:
                - Tissue type prediction (corpus/antrum/other)
                - Inflammation status prediction (inflamed/noninflamed/other)
                - Dictionary of tissue type probabilities
                - Dictionary of inflammation status probabilities
                
        Raises:
            Exception: If classification fails
        """
        try:
            tissue_votes = {'corpus': 0, 'antrum': 0}
            inflammation_votes = {'noninflamed': 0, 'inflamed': 0}
            
            tissue_total_probs = {'corpus': 0.0, 'antrum': 0.0}
            inflammation_total_probs = {'noninflamed': 0.0, 'inflamed': 0.0}
            
            for tile in tiles:
                # Transform tile
                img_tensor = self.transform(tile).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # Get tissue prediction
                    tissue_output = torch.sigmoid(self.tissue_model(img_tensor)).item()
                    tissue_pred = "corpus" if tissue_output > 0.5 else "antrum"
                    tissue_votes[tissue_pred] += 1
                    tissue_total_probs['corpus'] += tissue_output
                    tissue_total_probs['antrum'] += (1 - tissue_output)
                    
                    # Get inflammation prediction
                    inflammation_output = torch.sigmoid(
                        self.inflammation_model(img_tensor)
                    ).item()
                    inflammation_pred = "inflamed" if inflammation_output > 0.5 else "noninflamed"
                    inflammation_votes[inflammation_pred] += 1
                    inflammation_total_probs['inflamed'] += inflammation_output
                    inflammation_total_probs['noninflamed'] += (1 - inflammation_output)
            
            # Get final predictions by majority voting
            n_tiles = len(tiles)
            if n_tiles == 0:
                # Default values if no tiles were valid
                return "other", "other", {'corpus': 0.5, 'antrum': 0.5}, {'inflamed': 0.5, 'noninflamed': 0.5}
                
            tissue_pred = max(tissue_votes.items(), key=lambda x: x[1])[0]
            inflammation_pred = max(inflammation_votes.items(), key=lambda x: x[1])[0]
            
            # Average probabilities
            tissue_probs = {k: v/n_tiles for k, v in tissue_total_probs.items()}
            inflammation_probs = {k: v/n_tiles for k, v in inflammation_total_probs.items()}
            
            return tissue_pred, inflammation_pred, tissue_probs, inflammation_probs
            
        except Exception as e:
            self.logger.error(f"Error classifying tiles: {str(e)}")
            raise

    def process_wsi(
            self, 
            mrxs_path: Path, 
            annotations: List[dict]
    ) -> List[dict]:
        """Process entire WSI and update annotations with classifications.
        
        Args:
            mrxs_path: Path to the MRXS slide file
            annotations: List of annotation dictionaries to classify
            
        Returns:
            Updated list of annotations with classification information
        """
        try:
            # Convert path to string if it's a Path object
            mrxs_path_str = str(mrxs_path)
            
            total_annotations = len(annotations)
            self.logger.info(f"Processing {total_annotations} regions from {mrxs_path_str}")
            
            # Process each annotation
            updated_annotations = []
            for idx, annotation in enumerate(annotations, 1):
                # Get coordinates
                coords = annotation["geometry"]["coordinates"][0]
                
                # Extract tiles from region
                try:
                    tiles = self.extract_tiles_from_region(mrxs_path_str, coords)
                except Exception as e:
                    self.logger.error(f"Error extracting tiles for annotation {idx}: {e}")
                    # Fall back to using the existing annotation
                    updated_annotations.append(annotation)
                    continue
                
                if not tiles:
                    self.logger.warning(f"No valid tiles extracted for annotation {idx}")
                    # Fall back to using the existing annotation
                    updated_annotations.append(annotation)
                    continue
                
                # Classify region using tiles
                try:
                    tissue_type, inflammation_type, tissue_probs, inflammation_probs = (
                        self.classify_region(tiles)
                    )
                except Exception as e:
                    self.logger.error(f"Error classifying tiles for annotation {idx}: {e}")
                    # Fall back to using the existing annotation
                    updated_annotations.append(annotation)
                    continue
                
                # Update annotation
                updated_annotation = annotation.copy()
                
                # Check if annotation structure has required fields
                if "properties" not in updated_annotation:
                    updated_annotation["properties"] = {}
                    
                if "classification" not in updated_annotation["properties"]:
                    updated_annotation["properties"]["classification"] = {}
                
                # Set color based on tissue type if annotation options are available
                color = None
                if self.ANNOTATION_OPTIONS and tissue_type in self.ANNOTATION_OPTIONS.get('tissue', {}):
                    color = self.ANNOTATION_OPTIONS['tissue'][tissue_type]
                
                # Update properties with classifications
                updated_annotation["properties"]["classification"].update({
                    "tissue_type": tissue_type,
                    "inflammation_status": inflammation_type
                })
                
                # Add color if available
                if color:
                    updated_annotation["properties"]["classification"]["color"] = color
                
                # Add probabilities if needed (optional)
                if tissue_probs and inflammation_probs:
                    updated_annotation["properties"]["classification"].update({
                        "tissue_probabilities": tissue_probs,
                        "inflammation_status_probabilities": inflammation_probs
                    })
                
                updated_annotations.append(updated_annotation)
                
                # Log progress
                if idx % 10 == 0 or idx == total_annotations:
                    self.logger.info(f"Processed {idx}/{total_annotations} regions")
            
            return updated_annotations
            
        except Exception as e:
            self.logger.error(f"Error processing WSI: {str(e)}")
            # Return original annotations as fallback
            return annotations