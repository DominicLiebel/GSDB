# Gastric Stained Dataset Documentation

## Dataset Overview

The Gastric Stained Dataset (GSDB) consists of histological images of gastric tissue samples 
stained with different techniques to highlight specific tissue components and inflammatory markers.
The dataset is designed for computational pathology research with a focus on inflammation detection
and tissue type classification in gastric histology.

## Data Sources

The dataset consists of digitized slides collected from pathology archives. Each slide is:
1. Sectioned from a gastric tissue sample
2. Stained using one of three staining methods (HE, PAS, MG)
3. Digitized using one of two scanners for multi-scanner studies
4. Annotated by expert pathologists for inflammation status and tissue type

### Staining Methods

The following stains are included in the dataset:
- **Hematoxylin and Eosin (HE)**: Standard stain showing general tissue morphology
- **Periodic Acid–Schiff (PAS)**: Highlights mucus-secreting cells and basement membranes
- **Modified Giemsa (MG)**: Particularly useful for identifying *Helicobacter pylori* bacteria

### Scanner Information

The slides were digitized using two different whole slide imaging (WSI) scanners:
- **Scanner 1**: Primary scanner (majority of the dataset)
- **Scanner 2**: Secondary scanner (subset for domain adaptation studies)

## Data Processing Pipeline

### Raw Data
The raw data consists of whole slide images (WSIs) in proprietary format, which are not included
in the public distribution due to privacy concerns and file size limitations.

### Preprocessing Steps

1. **Particle Extraction**: Individual tissue fragments (particles) were extracted from WSIs
2. **Quality Control**: Particles were filtered for tissue content and scan quality
3. **Tile Generation**: Particles were divided into fixed-size tiles (2560×2560 pixels)
4. **Metadata Extraction**: Slide information was parsed from filenames and annotations
5. **Split Creation**: Slides were divided into train/validation/test sets based on patient ID

### Tile Naming Convention

Tiles follow this naming convention:
```
<scanner_id>_<patient_id>_<stain>_particle_<particle_uuid>_tile_<x>_<y>_size_<size>.png
```

Where:
- `scanner_id`: Scanner identifier (1 or 271/272/273)
- `patient_id`: Anonymized patient identifier
- `stain`: Staining method (HE, PAS, MG)
- `particle_uuid`: Unique identifier for each tissue particle
- `x`, `y`: Pixel coordinates of the tile within the particle
- `size`: Tile dimensions (always 2560x2560)

## Dataset Splits

The dataset is divided into the following splits:

### Train/Validation/Test Splits
- **Train split**: ~60% of patients
- **Validation split**: ~20% of patients 
- **Test split**: ~20% of patients

### Cross-Scanner Testing
- **test_scanner2**: Subset of test data from Scanner2, for domain adaptation evaluation

## Data Statistics

### Class Distribution

#### Inflammation Status
- Positive (inflammation present): XX% of particles
- Negative (no inflammation): XX% of particles

#### Tissue Types
- Antrum: XX% of particles
- Corpus: XX% of particles
- Other: XX% of particles

### Sample Counts
- Total patients: XXX
- Total slides: XXX
- Total particles: XXX
- Total tiles: XXX

## Annotation Methodology

Annotations were performed by pathologists following a multi-stage review process:

1. Initial annotation by a pathologist
2. Review by a second pathologist if unclear
3. Consensus resolution for any discrepancies

## Usage Guidelines

### Recommended Tasks
- Binary inflammation classification
- Binary tissue classification
- Multi-scanner domain adaptation
- Multi-class tissue type classification 

### Evaluation Metrics
For scientific reproducibility, we recommend reporting:
- Area Under ROC Curve (AUC)
- F1 Score
- Sensitivity & Specificity
- Balanced Accuracy

Report these metrics at both tile and slide/particle levels using validation-optimized thresholds.

## Limitations and Ethical Considerations

- The dataset may contain some labeling noise inherent to histopathology
- Results should be interpreted within the context of the specific staining techniques used
- Domain gaps between scanners should be considered in performance evaluation
- Patient privacy has been preserved through anonymization and extraction of only diagnostic regions