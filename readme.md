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

**Important Note:** The data splitting script (create_splits.py) uses the base seed plus 2 (seed 42+2=44) to ensure balanced class distribution. This offset was found to produce better balance between train/validation/test sets than the default seed alone, as described in the Master's Thesis.


### Configuration File

All hyperparameters are stored in a YAML file:

- `configs/model_config.yaml`: Model training hyperparameters


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
- 270 slides from scanner 1, 90 slides from scanner 2
- 274 HE-stained slides, 34 PAS-stained slides, 34 MG-stained slides
- 41,107 256x256 pixel tiles

## Model Architecture

We implement several deep learning architectures through our modular architecture system:

1. **ResNet18**: Baseline model from the ResNet family
2. **DenseNet121/169**: Higher capacity models with dense connections
3. **ConvNeXt Large**: Modern convolutional network with improved design
4. **Swin Transformer V2 Base**: Vision transformer with hierarchical structure
5. **GigaPath**: Foundation model pre-trained on histopathology images

Models are trained using:
- Binary cross-entropy loss with class weighting
- AdamW optimizer with cosine learning rate scheduling or SGD (ResNet18)
- Data augmentation including e.g., rotation, flipping

The model implementations can be found in `src/models/architectures/`.

## Results

### Antrum/Corpus Classification Results - Dataset: HE stained, Test on Scanner1
For classification of slides from the same scanner as the train and validation set (Scanner1) the best tissue models achieve:

| Model          | Test Accuracy | Test F1 | Test AUC |
|----------------|---------------|---------|----------|
| ConvNeXt Large | 88.71%        | 0.8827  | 0.9183   |
| Densenet121    | 87.07%        | 0.8646  | 0.9052   |
| Densenet169    | 84.23%        | 0.8389  | 0.8879   |
| GigaPath       | 84.30%        | 0.8262  | 0.8963   |
| ResNet18       | 85.20%        | 0.8491  | 0.8920   |
| Swin V2 B      | 78.41%        | 0.7569  | 0.8455   |



### Non/Inflamed Classification Results - Dataset: HE stained, Test on Scanner1

For classification of slides from the same scanner as the train and validation set (Scanner1) the best inflammation models achieve:

| Model          | Test Accuracy | Test F1 | Test AUC |
|----------------|---------------|---------|----------|
| ConvNeXt Large | 83.73%        | 0.8737  | 0.9193   |
| Densenet121    | 76.93%        | 0.7989  | 0.9120   |
| Densenet169    | 82.76%        | 0.8644  | 0.9136   |
| GigaPath       | 80.44%        | 0.8411  | 0.9029   |
| ResNet18       | 81.70%        | 0.8539  | 0.9043   |
| Swin V2 B      | 75.88%        | 0.8052  | 0.8363   |


### Antrum/Corpus Generalizability Classification Results - Dataset: HE stained, Test on Scanner2

For classification of slides from a different scanner (Scanner2) as the train and validation set (Scanner1) the best tissue models achieve:

| Model          | Test Accuracy | Test F1 | Test AUC |
|----------------|---------------|---------|----------|
| ConvNeXt Large | 79.80%        | 0.7177  | 0.9491   |
| Densenet121    | 80.40%        | 0.7370  | 0.9381   |
| Densenet169    | 66.65%        | 0.4678  | 0.8687   |
| GigaPath       | 79.92%        | 0.7233  | 0.9626   |
| ResNet18       | 66.18%        | 0.4233  | 0.8748   |
| Swin V2 B      | 74.47%        | 0.6365  | 0.8801   |


### Non/Inflamed Generalizability Classification Results - Dataset: HE stained, Test on Scanner2

For classification of slides from a different scanner (Scanner2) as the train and validation set (Scanner1) the best inflammation models achieve:

| Model          | Test Accuracy | Test F1 | Test AUC |
|----------------|---------------|---------|----------|
| ConvNeXt Large | 68.11%        | 0.8090  | 0.6797   |
| Densenet121    | 71.12%        | 0.8153  | 0.7631   |
| Densenet169    | 60.51%        | 0.6963  | 0.6288   |
| GigaPath       | 70.95%        | 0.8097  | 0.7029   |
| ResNet18       | 64.75%        | 0.7781  | 0.6185   |
| Swin V2 B      | 67.78%        | 0.7994  | 0.6987   |


For detailed performance metrics, see `results/evaluations` directory after running evaluation.

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
@mastersthesis{liebel2025gastric,
  title={Developing a comprehensive dataset and baseline model for classification and generalizability testing of gastric whole slide images in computational pathology},
  author={Liebel, Dominic},
  year={2025},
  school={University of Bamberg}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the pathology department for providing annotated histological samples.