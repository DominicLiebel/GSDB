# Data Structure

This dataset is split into train, validation, test, and test_new_scanner sets using a fixed random seed of 42. The inflammation type for each slide is provided in the `inflammation_type.csv` file.

## Dataset Splits

The dataset is divided into the following splits:

1. **Train Set**: `train_set_seed42.csv`
   - Contains the majority of the data used for training the models.

2. **Validation Set**: `val_set_seed42.csv`
   - Used for evaluating the model's performance during training and for hyperparameter tuning.

3. **Test Set**: `test_set_seed42.csv`
   - Used for evaluating the final performance of the trained models.

4. **Test New Scanner Set**: `test_new_scanner_set_seed42.csv`
   - Contains slides from a new scanner, used to assess the model's generalization ability to data from a different source.

Each split file contains a list of slide IDs corresponding to that particular split.

## Inflammation Type

The `inflammation_type.csv` file provides the inflammation type for each slide in the dataset. It has the following columns:

- `NEW_ID`: The unique identifier for each slide.
- `INFLAMMATION`: The inflammation type associated with each slide, which can be either "inflamed" or "noninflamed".

## Data Directory Structure

The data directory structure is as follows:

```
data/
│
├── raw/
│   └── slides/
│       ├── slide1.mrxs
│       ├── slide2.mrxs
│       └── ...
│
├── splits/
│   ├── train_set_seed42.csv
│   ├── val_set_seed42.csv
│   ├── test_set_seed42.csv
│   └── test_new_scanner_set_seed42.csv
│
└── processed/
│   └── statistics/
│       ├── inflammation_type.csv
```

- The `raw/slides/` directory contains the original whole slide images in the MRXS format.
- The `splits/` directory contains the CSV files defining the train, validation, test, and test_new_scanner splits.
- The `inflammation_type.csv` contains the inflammation type for each slide in the dataset. Slides with inflammation types other than 'inflamed' and 'noninflamed' are filtered out.

Note: The specific paths mentioned in the provided files (`/mnt/data/dliebel/2024_dliebel/data/...`) should be adjusted according to your local directory structure.