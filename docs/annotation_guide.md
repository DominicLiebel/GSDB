# WSI Annotation Tool Guide

This guide explains how to effectively use the WSI Annotation Tool for annotating whole slide images (WSIs) of gastric tissue, including manual annotation, automatic detection, classification, and clustering.

## Setup

1. **Initial Configuration**:
   - Start the annotation tool (`annotation_tool.py`)
   - Select a folder for storing annotations when prompted (or use the default)
   - The tool will create necessary subdirectories for annotations, clusters, and backups

2. **Loading Slides**:
   - Click "Load Slide Directory" and select the folder containing downsampled PNG files
   - The tool expects files with the naming format `*_downsampled16x.png`
   - Slides will appear in the left panel with thumbnail previews and annotation counts

## Annotation Methods

### Manual Annotation

1. Click on a slide in the left panel to load it
2. Draw a polygon around a region of interest:
   - Left-click and drag to create a polygon outline
   - Release to complete the drawing
3. In the popup dialog:
   - Select the tissue type (corpus, antrum, intermediate, other)
   - Select the inflammation status (inflamed, noninflamed, unclear, other)
   - Optionally check "Use as default for future annotations"
   - Click "Save" to store the annotation

### Quick Annotation Shortcuts

- Press `1` to quickly save as Corpus
- Press `2` to quickly save as Antrum
- Press `3` to quickly save as Intermediate
- Press `4` to quickly save as Other
- Press `Enter` to quick-save using the default settings
- Press `Escape` to cancel the current drawing

### Automatic Annotation

1. Click "Detect Regions" to automatically identify regions of interest
   - The tool will detect distinct tissue regions based on color and morphology
   - This creates annotations with default type "other"
2. Click "Auto-classify Regions" to classify detected regions
   - This requires access to the original MRXS file and classification models
   - The tool will classify each region as corpus or antrum and determine inflammation status

### Batch Inflammation Status

- Use the "Quick Inflammation Status" section to set inflammation status for all annotations at once
- Options include: inflamed, noninflamed, unclear, other

## Working with Clusters

Clusters allow you to group annotations into meaningful regions.

1. **Creating Clusters**:
   - Press `C` to set the first corner of a cluster rectangle
   - Move to desired position
   - Press `C` again to complete the rectangle and save the cluster

2. **Managing Clusters**:
   - "Process Clusters" - Assign cluster IDs to annotations based on their position
   - "Delete All Clusters" - Remove all clusters for the current slide
   - "Hide/Show Clusters" - Toggle visibility of cluster rectangles

## Editing Annotations

- **Select**: Right-click an annotation to select it and edit its properties
- **Move**: Middle-click (or Ctrl+left-click) and drag to move an annotation
- **Edit**: Select an annotation and press `E` or click "Edit" to modify its properties
- **Delete**: Select an annotation and press `Delete` or click "Delete"
- **Multi-select**: In the annotation list, you can select multiple annotations to edit or delete them together

## Keyboard Shortcuts

- `1-4`: Quick-save with different tissue types
- `Enter`: Save current annotation with default settings
- `Escape`: Cancel current drawing
- `E`: Edit selected annotation(s)
- `Delete`: Delete selected annotation(s)
- `C`: Start/complete cluster (press twice)
- `F1`: Show keyboard shortcut help

## Navigation

- Mouse wheel: Scroll vertically
- Shift + Mouse wheel: Scroll horizontally
- Middle mouse button: Pan the view

## File Formats

### Annotation Format

Annotations are stored as GeoJSON Feature objects with the following structure:

```json
{
  "type": "Feature",
  "id": "<unique_id>",
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [<x_coord>, <y_coord>],
        ...
      ]
    ]
  },
  "properties": {
    "objectType": "annotation",
    "classification": {
      "tissue_type": "<corpus|antrum|intermediate|other>",
      "inflammation_status": "<inflamed|noninflamed|unclear|other>",
      "color": [R, G, B]
    },
    "cluster_id": <cluster_id_or_null>
  }
}
```

Coordinates are stored in full slide resolution space (not downsampled).

### Cluster Format

Clusters are stored as simple objects with a unique ID and bounding box coordinates:

```json
{
  "id": <cluster_id>,
  "bounds": {
    "left": <x_min>,
    "right": <x_max>,
    "top": <y_min>,
    "bottom": <y_max>
  }
}
```

## Auto-Classification

The auto-classification feature uses pretrained models to classify regions:

1. Tissue Type Classification:
   - Classifies regions as either corpus or antrum
   - Supports the following model architectures:
     - GigaPath
     - ResNet18
     - Swin Transformer V2-B
     - ConvNeXt Large
     - DenseNet121
     - DenseNet169

2. Inflammation Classification:
   - Determines if a region is inflamed or non-inflamed
   - Also supports multiple model architectures

When using "Auto-classify Regions", you'll be prompted to:
1. Select the model file paths for both tissue and inflammation classification
2. Choose the appropriate model architecture for each model
3. Click "Confirm" to proceed or "Use Default" to use ResNet18

The tool will remember your selected model paths and architectures for future sessions. The system uses an intelligent model loading mechanism that:

- Adapts to your environment's file structure
- Automatically locates model utility functions
- Uses the same transformations as during model training
- Provides robust fallbacks if specialized loading fails

## Additional Features

- **Auto-backup**: The tool automatically backs up annotations every 5 minutes
- **Viewport tracking**: The thumbnail shows your current view position on the slide
- **Annotation counter**: Shows the number of annotations and clusters for each slide
- **Multi-selection**: Select multiple annotations at once for batch editing
- **Lazy loading**: Thumbnails are loaded in the background for better performance