"""
Common model utility functions to break circular dependencies.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
import logging

def get_transforms(is_training: bool = False) -> transforms.Compose:
    """Get data transformations matching GigaPath's requirements."""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                               std=(0.229, 0.224, 0.225))
        ])
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                           std=(0.229, 0.224, 0.225))
    ])

def load_model(model_path: Path, device: torch.device, architecture: str = "gigapath") -> nn.Module:
    """Load model based on specified architecture with support for both PyTorch and pickle formats."""
    # Import here inside the function to avoid circular imports
    import timm
    
    try:
        # Do delayed imports to avoid circular dependencies
        from src.models.architectures import GigaPathClassifier, HistologyClassifier
        
        # Initialize the model architecture
        if architecture == "gigapath":
            model = GigaPathClassifier(num_classes=1)
        elif architecture in ["resnet18", "swin_v2_b", "convnext_large", "densenet121", "densenet169"]:
            # Use HistologyClassifier for these custom architectures
            model = HistologyClassifier(model_name=architecture, num_classes=1)
        else:
            # Try to use timm directly for other architectures
            try:
                model = timm.create_model(architecture, pretrained=True, num_classes=1)
            except Exception as e:
                raise ValueError(f"Unsupported architecture: {architecture}. Error: {str(e)}")
        
        # Check file extension to determine loading method
        file_extension = model_path.suffix.lower()
        
        if file_extension == '.pkl':
            import pickle
            logging.info(f"Loading pickle model from {model_path}")
            
            with open(model_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # Handle different pickle serialization formats
            if hasattr(loaded_data, 'state_dict'):  # If pickle contains the entire model
                model = loaded_data
            elif isinstance(loaded_data, dict):
                # Try to load as state dictionary
                if 'model_state_dict' in loaded_data:
                    model.load_state_dict(loaded_data['model_state_dict'], strict=False)
                else:
                    # Assume the dictionary is the state dict itself
                    model.load_state_dict(loaded_data, strict=False)
            else:
                raise ValueError(f"Unsupported pickle format. Expected model or state dict.")
                
        else:  # Default PyTorch loading (.pt, .pth)
            logging.info(f"Loading PyTorch model from {model_path}")
            try:
                # First try with weights_only=True (safer)
                checkpoint = torch.load(model_path, map_location=device)
            except Exception as e:
                logging.warning(f"Failed to load with default settings, trying with weights_only=False: {str(e)}")
                # If that fails, fall back to weights_only=False (less safe but more compatible)
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                logging.info("Successfully loaded model with weights_only=False")
            
            # Extract the model weights using standardized keys
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Standard format (from both train.py and tune.py)
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    
                    # If model comes from tuning, log relevant info
                    if 'trial_number' in checkpoint:
                        logging.info(f"Loaded model from trial {checkpoint['trial_number']}")
                    if 'val_loss' in checkpoint:
                        logging.info(f"Validation loss: {checkpoint['val_loss']:.4f}")
                        
                    # If model comes from training, log relevant info
                    if 'epoch' in checkpoint and checkpoint['epoch'] != -1:
                        logging.info(f"Trained for {checkpoint['epoch']} epochs")
                else:
                    # Try to load the dictionary as a direct state dict
                    logging.warning("No 'model_state_dict' key found, attempting to load as direct state dict")
                    model.load_state_dict(checkpoint, strict=False)
            else:
                raise ValueError(f"Unsupported model format. Expected dictionary with 'model_state_dict'.")
        
        # Set model name attribute if not already present
        if not hasattr(model, 'model_name'):
            model.model_name = architecture
            
        # Set to eval mode and return
        return model.to(device).eval()
    
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise