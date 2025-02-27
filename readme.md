# Gastric Slide Database - Histology Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive deep learning framework for the classification of histological patterns in gastric tissue slides. This project provides end-to-end tools for processing whole slide images, extracting regions of interest, and classifying tissue types and inflammation status using state-of-the-art deep learning techniques.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Reproducibility](#reproducibility)
- [Usage](#usage)
- [Data Structure](#data-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

The Gastric Slide Database (GSDB) Classification project is designed to process and analyze whole slide images (WSIs) from gastric tissue biopsies. It addresses two primary classification tasks:

1. **Inflammation Classification**: Detecting inflamed versus non-inflamed gastric tissue
2. **Tissue Type Classification**: Distinguishing between corpus and antrum tissue regions

The framework incorporates a full machine learning pipeline including:
- Data preprocessing and tile extraction from whole slide images
- Dataset creation with appropriate train/validation/test splits
- Model training with hyperparameter optimization
- Comprehensive evaluation at multiple hierarchical levels (tile, particle, slide)
- Tools for interpretability and visualization

## Installation

### Requirements

- Python 3.10+
- CUDA-compatible GPU with at least 8GB VRAM (for training)
- 16GB+ RAM
- 100GB+ disk space (for dataset storage)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/DominicLiebel/GSDB.git
   cd GSDB
   ```

2. Create and activate a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate gsdb
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Set the environment variable for the base directory:
   ```bash
   export GASTRIC_BASE_DIR="/path/to/your/data"
   ```

### Docker Installation (Alternative)

For complete reproducibility, we provide a Docker container:

```bash
docker pull liebeld/gsdb-classification:latest
docker run --gpus all -v /path/to/your/data:/data liebeld/gsdb-classification:latest
```

## Reproducibility

We provide several mechanisms to ensure reproducibility of our results:

### Fixed Random Seeds

All random processes (data splitting, model initialization, etc.) use a fixed seed of 42 by default:

```python
# From configs/model_config.yaml
common:
  seed: 42
  deterministic: true
```

### Configuration Files

All hyperparameters are stored in YAML files:

- `configs/data_config.yaml`: Data processing parameters
- `configs/model_config.yaml`: Model training hyperparameters

### Pre-trained Models

We provide pre-trained model weights for all reported experiments: TODO!

```bash
# Download pre-trained models
python scripts/download_models.py --task inflammation --model convnext_large
```

### Standard Evaluation Protocol

All models are evaluated using the same protocol with three test sets:
1. Internal validation set (`val` split)
2. Internal test set (`test` split)
3. External test set from different scanner (`test_scanner2` split)

## Usage

### Pre-processing Only

To preprocess slides without training:

```bash
# Downsample whole slide images
python src/data/preprocessing/downsize_wsi_to_png.py --base-dir /path/to/data

# Annotate regions of interest
python src/data/preprocessing/wsi_annotation.py

# Extract tiles from annotated regions
python src/data/extract_tiles.py --tile-size 256 --downsample 10 --overlap 64
```

### Complete Pipeline

To run the complete pipeline from raw slides to evaluation:

```bash
# Extract tiles
python src/data/extract_tiles.py --base-dir /path/to/data

# Create dataset splits
python src/data/create_splits.py --seed 42 --base-dir /path/to/data

# Train a model
python src/models/train.py --task inflammation --model convnext_large --base-dir /path/to/data

# Evaluate the model
python src/models/evaluate.py --task inflammation --test_split test --model_path /path/to/model.pt
```

### Hyperparameter Optimization

To optimize model hyperparameters:

```bash
python src/models/tune.py --task inflammation --study-name inflammation_study --n-trials 100
```

### Inference on New Data

To run inference on new slides:

```bash
python scripts/inference.py --slides_dir /path/to/new/slides --model_path /path/to/model.pt --output_dir /path/to/results
```

## Data Structure

The dataset is organized as follows:

```
data/
├── raw/
│   ├── slides/          # Original whole slide images (*.mrxs)
│   └── annotations/     # Manual annotations (*.json)
├── processed/
│   ├── tiles/           # Extracted tissue tiles (*.png)
│   └── downsized_slides/# Downsampled whole slides for visualization
└── splits/
    └── seed_42/         # Data splits with random seed 42
        ├── train_HE.csv        # HE-stained training set
        ├── val_HE.csv          # HE-stained validation set
        ├── test_HE.csv         # HE-stained test set
        ├── test_scanner2_HE.csv # Scanner 2 test set (HE-stained)
        └── [other split files]
```

### Dataset Statistics

Our dataset consists of:
- 360 total slides
- 198 inflamed samples, 129 non-inflamed samples
- 1704 corpus tissue regions, 1779 antrum tissue regions
- 260 HE-stained slides, 55 PAS-stained slides, 45 MG-stained slides
- 270 slides from scanner 1, 90 slides from scanner 2

## Model Architecture

We implement several deep learning architectures:

1. **ResNet18**: Baseline model from the ResNet family
2. **ConvNeXt Large**: Modern convolutional network with improved design
3. **Swin Transformer V2 Base**: Vision transformer with hierarchical structure
4. **GigaPath**: Foundation model pre-trained on histopathology images

Models are trained using:
- Binary cross-entropy loss with class weighting
- AdamW optimizer with cosine learning rate scheduling
- Mixed-precision training for faster computation
- Data augmentation including rotation, flipping, and color jittering

## Results

Our best model achieves:

| Model | Task | Test Accuracy | Test F1 | Test AUC |
|-------|------|--------------|---------|----------|
| ConvNeXt Large | Inflammation | 89.4% | 0.912 | 0.937 |
| ConvNeXt Large | Tissue Type | 93.2% | 0.936 | 0.952 |
| Swin V2 B | Inflammation | 87.2% | 0.893 | 0.926 |
| GigaPath | Inflammation | 91.5% | 0.925 | 0.945 |
...

For detailed performance metrics, see `results/metrics/` directory after running evaluation.

## Contributing

We welcome contributions to this project. Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

Please ensure your code follows our style guide and includes appropriate tests.

## Citation

If you use this codebase in your research, please cite our paper:

```bibtex
@article{liebel2025gastric,
  title={Developing a comprehensive dataset and baseline model for classification and generalizability testing of gastric whole slide images in computational pathology},
  author={Liebel, Dominic},
  pages={[Pages]},
  year={2025},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the pathology department for providing annotated histological samples.