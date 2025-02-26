# Gastric Slide Database - Classification Pipeline

A comprehensive pipeline for processing, analyzing, and classifying gastric whole slide images.

## Project Structure

```
project/
├── configs/                # Configuration files
│   ├── model_config.yaml   # Model training configuration
│   └── data_config.yaml    # Data processing configuration
├── data/                   # Data directory
│   ├── raw/                # Raw data (slides, annotations)
│   │   ├── slides/         # Raw slide images (.mrxs)
│   │   └── annotations/    # Annotation JSON files
│   ├── processed/          # Processed data
│   │   ├── tiles/          # Extracted tissue tiles
│   │   └── downsized_slides/ # Downsampled whole slides
│   └── splits/             # Train/val/test splits
│       └── seed_42/        # Splits with random seed 42
├── results/                # Results directory
│   ├── logs/               # Log files
│   ├── figures/            # Generated figures
│   ├── tables/             # Generated tables
│   ├── metrics/            # Performance metrics
│   ├── models/             # Trained models
│   └── evaluations/        # Model evaluation results
└── src/                    # Source code
    ├── config/             # Configuration utilities
    │   └── paths.py        # Path management
    ├── data/               # Data processing code
    │   ├── create_splits.py            # Create data splits
    │   ├── dataset_analysis.py         # Dataset analysis
    │   ├── extract_tiles.py            # Extract tiles from WSIs
    │   ├── preprocessing/              # Preprocessing scripts
    │   │   ├── downsize_wsi_to_png.py  # Downsize WSIs
    │   │   └── wsi_annotation.py       # WSI annotation tool
    │   └── process_dataset.py          # Dataset processing
    └── models/                         # Model-related code
        ├── dataset.py                  # Dataset classes
        ├── evaluate.py                 # Evaluation code
        ├── metrics_utils.py            # Metrics utilities
        ├── train.py                    # Training code
        ├── training_utils.py           # Training utilities
        └── tune.py                     # Hyperparameter tuning
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DominicLiebel/GSDB.git
   cd GSDB
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n GSBD python=3.10
   conda activate GSDB
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The project uses environment variables and command-line arguments for configuration. You can either:

1. Set the environment variable `GASTRIC_BASE_DIR` to specify the base directory:
   ```bash
   export GASTRIC_BASE_DIR="/path/to/project/data"
   ```

2. Pass the base directory via command line:
   ```bash
   python src/data/create_splits.py --base-dir /path/to/project/data
   ```

## Usage

### 1. Preprocess Data

First, downsize whole slide images for easier viewing:

```bash
python src/data/preprocessing/downsize_wsi_to_png.py --base-dir /path/to/project/data
```

### 2. Extract Tiles

Extract tiles from whole slide images:

```bash
python src/data/extract_tiles.py --base-dir /path/to/project/data
```

### 3. Create Splits

Create train/validation/test splits:

```bash
python src/data/create_splits.py --base-dir /path/to/project/data --seed 42
```

### 4. Analyze Dataset

Run dataset analysis to generate metrics and tables:

```bash
python src/data/dataset_analysis.py --base-dir /path/to/project/data
```

### 5. Train Model

Train a classifier for inflammation detection:

```bash
python src/models/train.py --task inflammation --model convnext_large --base-dir /path/to/project/data
```