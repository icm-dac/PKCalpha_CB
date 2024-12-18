# PKCα Junction Analysis
<img src="/figures/pkca-junction-analysis-logo.png" width="200" alt="PKCα Junction Analysis Logo">

# Cell Junction Analysis Pipeline

## Overview
This pipeline is designed to analyze cellular junctions in microscopy images, particularly focusing on the detection and quantification of junctional proteins. The pipeline processes paired images (a protein-of-interest channel and an Alexa channel) to identify cell boundaries, detect junctions, and measure their properties.

## Features
- Automated detection of cell boundaries and junctions
- Special handling for CTNND1 protein images
- Support for untransfected cell analysis
- Honeycomb pattern enhancement
- Edge detection and analysis
- Intensity measurement and normalization using moving average
- CSV output generation with detailed metrics

## Dependencies
```python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-image
- scipy
- networkx
- shapely
```

## File Structure Requirements
The pipeline expects three main directories:
- Path 1: Directory containing primary images
- Path 2: Directory containing Alexa channel images
- Path 3: CSV output directory
- Path 4: Processed images output directory

## Input Image Requirements
- Supported formats: .png, .jpg, .jpeg, .tif, .tiff, .bmp
- Expected filename format: `[Protein] [Condition] [Experiment] - [Image]`
  Example: "CTNND1 Control Exp1 - 1"
- Paired images must exist in both primary and Alexa directories

## Key Components

### 1. JunctionsAnalyzer Class
The main class responsible for image analysis with methods for:
- Image preprocessing
- Skeletonization
- Graph building
- Node and edge detection
- Visualization

### 2. Image Processing Functions
- `enhance_honeycomb()`: Enhances cellular honeycomb patterns
- `preprocess_alexa_for_ctnnd1()`: Special preprocessing for CTNND1 protein
- `mark_intersections()`: Identifies and marks junction intersections
- `connect_centroids_within_threshold()`: Links nearby cellular centroids

### 3. Analysis Functions
- `calculate_edge_properties()`: Measures edge length and intensity
- `extract_edges_intersecting_with_lines()`: Identifies relevant junctions
- `calculate_moving_average_intensity()`: Computes normalized intensities

## Output
The pipeline generates a CSV file containing:
- Image metadata (name, protein, condition, experiment number)
- Edge properties (length, intensity)
- Normalized measurements
- Spatial information (centroids)

## Usage

1. Set up directory paths in the script:
```python
dir_path = '/path_1'  # Primary images
alexa_dir_path = '/path_2'  # Alexa channel images
csv_output_dir = '/path_3'  # CSV output
images_output_dir = '/path_4'  # Processed images
```

2. Run the main execution loop:
```python
execute_main(edge_count)  # edge_count determines the number of edges to analyze
```

## Special Cases

### Untransfected Cells
- Automatically detected from filename
- Uses modified analysis pipeline
- Direct analysis of Alexa channel

### CTNND1 Processing
- Special preprocessing applied
- Modified HSV color space filtering
- Specific intensity thresholding

## Error Handling
- File existence verification
- Image format validation
- Error logging for failed processes
- Graceful handling of missing data

## Performance Considerations
- Processes images in pairs
- Supports batch processing
- Memory-efficient image handling
- Configurable edge count for analysis

## Limitations
- Requires specific file naming convention
- Paired images must be present
- Memory usage scales with image size
- Processing time depends on edge count

