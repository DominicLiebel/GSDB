# Data Structure

This dataset is split into train, validation, test, and test_scanner2 sets using a fixed random seed of 42. The inflammation status for each slide is provided in the `inflammation_type.csv` file.

## Dataset Splits

The dataset is divided into the following splits, with separate files for HE stain only and all stains:

1. **Train Set**: 
   - `train_HE.csv`: HE-stained training samples
   - `train_all_stains.csv`: Training samples for all stain types (HE, PAS, MG)

2. **Validation Set**: 
   - `val_HE.csv`: HE-stained validation samples
   - `val_all_stains.csv`: Validation samples for all stain types

3. **Test Set**: 
   - `test_HE.csv`: HE-stained test samples
   - `test_all_stains.csv`: Test samples for all stain types

4. **Test Scanner 2 Set**: 
   - `test_scanner2_HE.csv`: HE-stained samples from scanner 2
   - `test_scanner2_all_stains.csv`: All stain type samples from scanner 2

Each split file contains a list of slides corresponding to that particular split.

## Inflammation Status

The `inflammation_status.csv` file provides the inflammation type for each slide in the dataset. It has the following columns:

- `slide_name`: The unique identifier for each slide.
- `inflammation_status`: The inflammation type associated with each slide, which can be either "inflamed" or "noninflamed".

## Data Directory Structure

The actual data directory structure is as follows:

```
data/
│
├── raw/
│   ├── annotations/         # JSON annotation files
│   │   ├── 1_1_1_HE_annotations.json
│   │   └── ...
│   ├── clusters/            # Cluster files for tissue regions
│   │   ├── 1_1_1_HE_clusters.json
│   │   └── ...
│   ├── archive/             # Archives of expert annotations
│   │   └── expert_annotations/
│   │       ├── new_scanner_annotations.docx
│   │       ├── old_scanner_annotations.docx
│   │       └── png_annotations/
│   │           └── ...
│   └── metrics/             # Dataset metrics
│       ├── expert_annotations_summary.csv
│       ├── inflammation_type.csv
│       └── ...
│
├── splits/
│   └── seed_42/             # Data splits with random seed 42
│       ├── train_HE.csv
│       ├── train_all_stains.csv
│       ├── val_HE.csv
│       ├── val_all_stains.csv
│       ├── test_HE.csv
│       ├── test_all_stains.csv
│       ├── test_scanner2_HE.csv
│       └── test_scanner2_all_stains.csv
│
└── processed/               # Processed data (generated during pipeline execution)
    ├── tiles/               # Extracted tissue tiles (*.png)
    └── downsized_slides/    # Downsampled whole slides for visualization
```

- The `raw/annotations/` directory contains the JSON annotation files for each slide.
- The `raw/clusters/` directory contains the cluster files generated from the annotations.
- The `raw/archive/` directory contains original expert annotations and reference materials.
- The `raw/metrics/` directory contains dataset statistics, including the inflammation type information.
- The `splits/seed_42/` directory contains the CSV files defining the dataset splits.
- The `processed/` directory is generated during pipeline execution and contains extracted tiles and downsized slides.

Note: Whole slide images are not included in the repository due to size constraints, but will be provided separately in MRXS format.