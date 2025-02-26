# Annotation Guide

This guide explains how to use the annotation tool to manually or automatically annotate whole slide images (slides) of gastric tissue.

## Setup

1. Downsample the MRXS files to a manageable size. The script expects PNG files with the naming format `*_downsampled16x.png`. 

2. Start the Python annotation script.

3. In the annotation tool, click "Load slide Directory" and select the folder containing the downsampled PNG files. Wait for the files to load in the slide list on the left.

## Annotating a slide

1. Click on the slide you want to annotate in the list on the left. The image will load in the main viewing area.

2. There are two ways to annotate regions of interest:

    a. Manual annotation: 
        - Click and drag the mouse over a region to draw a polygon around it. 
        - Complete the polygon by connecting back to the starting point.
        - In the popup dialog, select the tissue type and inflammation status for that annotation.
        - Click "Save" to store the annotation.

    b. Automatic particle detection:
        - Click the "Auto-detect Particles" button.
        - The script will automatically detect distinct tissue regions and classify them.
        - Review and edit the auto-generated annotations if needed.

3. To create annotation clusters:
    - Press 'C' to set one corner of the cluster rectangle. 
    - Move the mouse to cover the desired cluster region.
    - Press 'C' again to complete the rectangle and save the cluster.

4. All annotations and clusters are automatically saved in JSON format in the `annotations` and `clusters` directories respectively.

## Annotation File Format

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
      "inflammation_type": "<inflamed|noninflamed|unclear|other>"  
    }
  }
}
```

The coordinates are stored in the full slide resolution space.

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

## Additional Features

- Right-click an annotation to select it and edit its properties
- Ctrl+Left-click and drag an annotation to move it
- In the annotation list, select one or more annotations to highlight, edit or delete them together
- Use the "Process Clusters" button to assign cluster IDs to annotations inside each cluster's bounds
- Toggle cluster visibility with the "Hide/Show Clusters" button

The annotation tool auto-saves your work to prevent loss of data. Use the intuitive GUI to zoom, pan and efficiently annotate large slides for downstream analysis or model training.