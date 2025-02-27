"""
Model Architecture Module

This package contains implementations of various model architectures
for histological image classification.
"""

from .gigapath import GigaPathClassifier
from .base import HistologyClassifier

__all__ = ['GigaPathClassifier', 'HistologyClassifier']