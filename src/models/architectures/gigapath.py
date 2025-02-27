"""
GigaPath Model Architecture

This module implements the GigaPathClassifier, which leverages pre-trained
Prov-GigaPath features for histology image classification.
"""

import torch
import torch.nn as nn
import logging

class GigaPathClassifier(nn.Module):
    """Classifier using Prov-GigaPath features for histology classification.
    
    This model uses a pre-trained GigaPath tile encoder to extract features
    from histology images, followed by a trainable classification head.
    The encoder weights are frozen during training to leverage the 
    pre-trained representations.
    
    Attributes:
        model_name (str): Identifier for the model architecture
        tile_encoder (nn.Module): Pre-trained GigaPath feature extractor
        classifier (nn.Sequential): Trainable classification head
    """
    
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.2):
        """Initialize GigaPath classifier with frozen encoder and trainable head.
        
        Args:
            num_classes (int): Number of output classes (default: 1 for binary)
            dropout_rate (float): Dropout probability for regularization
            
        Raises:
            ImportError: If the required 'timm' library is not installed
        """
        super().__init__()
        self.model_name = "gigapath"
        
        try:
            import timm
            # Load pretrained GigaPath tile encoder
            self.tile_encoder = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                pretrained=True,
                num_classes=0  # Disable classification head
            )
            
            # Freeze the encoder weights
            for param in self.tile_encoder.parameters():
                param.requires_grad = False
                
            # Add classification head
            self.classifier = nn.Sequential(
                nn.Linear(1536, 512),  # GigaPath outputs 1536-dim features
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
        except ImportError:
            logging.error("timm library not found. Please install with: pip install timm")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W]
                
        Returns:
            torch.Tensor: Model output logits
        """
        with torch.no_grad():  # Ensure encoder remains in inference mode
            features = self.tile_encoder(x)
        return self.classifier(features)
        
    def train(self, mode: bool = True) -> 'GigaPathClassifier':
        """Set training mode for classifier only, keeping encoder in eval mode.
        
        Args:
            mode (bool): Whether to set training mode
            
        Returns:
            GigaPathClassifier: The model with updated mode
        """
        super().train(mode)
        # Keep encoder in eval mode
        self.tile_encoder.eval()
        return self
        
    def configure_optimizers(self, lr: float, weight_decay: float):
        """Configure optimizer and related training components.
        
        Since the encoder is frozen, only the classifier parameters are optimized.
        
        Args:
            lr (float): Learning rate
            weight_decay (float): Weight decay regularization factor
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        self.optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = torch.amp.GradScaler()
        return self.optimizer