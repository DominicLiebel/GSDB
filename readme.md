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
- [Documentation](#documentation)
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

- Python 3.11+
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

3. Set the environment variable for the base directory:
   ```bash
   export GASTRIC_BASE_DIR="/path/to/your/data"
   ```

### Docker Installation (Alternative) - TODO

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

You can also set the seed via command line in all scripts:

```bash
python src/models/train.py --task inflammation --model resnet18 --seed 42 --deterministic
```

**Important Note:** The data splitting script (create_splits.py) uses a different seed (44) to ensure balanced class distribution. This was necessary as seed 42 produced imbalanced splits.


### Configuration Files

All hyperparameters are stored in a YAML file:

- `configs/model_config.yaml`: Model training hyperparameters

### Pre-trained Models

We provide pre-trained model weights for all reported experiments: TODO

```bash
# Download pre-trained models
python scripts/download_models.py --task inflammation --model convnext_large
```

### Standard Evaluation Protocol

All models are evaluated using the same protocol with three test sets:
1. Internal validation set (`val` split)
2. Internal test set (`test` split)
3. External test set from different scanner (`test_scanner2` split)

Each evaluation ensures proper separation between validation and test data, with thresholds optimized only on validation data.

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

For detailed information on the dataset, see our [data documentation](docs/data_documentation.md).

### Dataset Statistics

Our dataset consists of:
- 360 total slides
- 198 inflamed samples, 129 non-inflamed samples
- 1704 corpus tissue regions, 1779 antrum tissue regions
- 260 HE-stained slides, 55 PAS-stained slides, 45 MG-stained slides
- 270 slides from scanner 1, 90 slides from scanner 2

## Model Architecture

We implement several deep learning architectures through our modular architecture system:

1. **ResNet18**: Baseline model from the ResNet family
2. **DenseNet121/169**: Higher capacity models with dense connections
3. **ConvNeXt Large**: Modern convolutional network with improved design
4. **Swin Transformer V2 Base**: Vision transformer with hierarchical structure
5. **GigaPath**: Foundation model pre-trained on histopathology images

Models are trained using:
- Binary cross-entropy loss with class weighting
- AdamW optimizer with cosine learning rate scheduling
- Mixed-precision training for faster computation
- Data augmentation including rotation, flipping, and color jittering

The model implementations can be found in `src/models/architectures/`.

## Results

### Antrum/Corpus Classification Results - Dataset: HE stained, Test on Scanner1

For classification of slides from the same scanner as the train and validation set (Scanner1) the best tissue models achieve:

| Model | Test Accuracy | Test F1 | Test AUC |
|-------|---------------|---------|----------|
| ResNet18 | xx.x% | 0.xx | 0.xx |
| Densenet121 | xx.x% | 0.xx | 0.xx |
| Densenet169 | xx.x% | 0.xx | 0.xx |
| ConvNeXt Large | xx.x% | 0.xx | 0.xx |
| Swin V2 B | x.x% | 0.xx | 0.xx |
| GigaPath | x.x% | 0.xx | 0.xx |

### Antrum/Corpus Generalizability Classification Results - Dataset: HE stained, Test on Scanner2

For classification of slides from a different scanner (Scanner2) as the train and validation set (Scanner1) the best tissue models achieve:

| Model | Test Accuracy | Test F1 | Test AUC |
|-------|---------------|---------|----------|
| ResNet18 | xx.x% | 0.xx | 0.xx |
| Densenet121 | xx.x% | 0.xx | 0.xx |
| Densenet169 | xx.x% | 0.xx | 0.xx |
| ConvNeXt Large | xx.x% | 0.xx | 0.xx |
| Swin V2 B | x.x% | 0.xx | 0.xx |
| GigaPath | x.x% | 0.xx | 0.xx |

### Non/Inflamed Classification Results - Dataset: HE stained, Test on Scanner1

For classification of slides from the same scanner as the train and validation set (Scanner1) the best inflammation models achieve:

| Model | Test Accuracy | Test F1 | Test AUC |
|-------|---------------|---------|----------|
| ResNet18 | xx.x% | 0.xx | 0.xx |
| Densenet121 | xx.x% | 0.xx | 0.xx |
| Densenet169 | xx.x% | 0.xx | 0.xx |
| ConvNeXt Large | xx.x% | 0.xx | 0.xx |
| Swin V2 B | x.x% | 0.xx | 0.xx |
| GigaPath | x.x% | 0.xx | 0.xx |

### Non/Inflamed Generalizability Classification Results - Dataset: HE stained, Test on Scanner2

For classification of slides from a different scanner (Scanner2) as the train and validation set (Scanner1) the best inflammation models achieve:

| Model | Test Accuracy | Test F1 | Test AUC |
|-------|---------------|---------|----------|
| ResNet18 | xx.x% | 0.xx | 0.xx |
| Densenet121 | xx.x% | 0.xx | 0.xx |
| Densenet169 | xx.x% | 0.xx | 0.xx |
| ConvNeXt Large | xx.x% | 0.xx | 0.xx |
| Swin V2 B | x.x% | 0.xx | 0.xx |
| GigaPath | x.x% | 0.xx | 0.xx |

For detailed performance metrics, see `results/metrics/` directory after running evaluation.

## Documentation

We provide detailed documentation to help you understand and use the project:

- [Data Documentation](docs/data_documentation.md): Detailed information about the dataset
- [Project Organization](docs/project_organization.md): Code structure overview
- [Style Guide](docs/style_guide.md): Coding conventions for contributors

## Contributing

We welcome contributions to this project. Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

Please ensure your code follows our [style guide](docs/style_guide.md) and includes appropriate tests.

## Citation
 
If you use this codebase in your research, please cite our paper:

```bibtex
@article{liebel2025gastric,
  title={Developing a comprehensive dataset and baseline model for classification and generalizability testing of gastric whole slide images in computational pathology},
  author={Liebel, Dominic},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the pathology department for providing annotated histological samples.