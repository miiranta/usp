# Plant Growth Analysis Application

A comprehensive desktop application for analyzing plant growth through photo processing and data visualization. This Electron-based application combines Angular frontend with Node.js backend to provide an intuitive interface for tracking plant development over time.

## Overview

This application enables researchers and plant enthusiasts to:
- Upload and organize plant photos into collections
- Process images using computer vision algorithms to extract growth metrics
- Visualize growth patterns through interactive charts
- Track height, width, and area measurements over time
- Generate scientific reports and documentation

## Features

### 📸 Photo Management
- **Upload System**: Drag-and-drop or browse to upload multiple photos
- **Collection Organization**: Group related photos into named collections
- **Metadata Tracking**: Automatic timestamp detection with manual override capability
- **Batch Processing**: Process multiple photos simultaneously with consistent parameters

### 🔍 Image Processing
- **Advanced Computer Vision**: Uses OpenCV-based Python algorithms for plant detection
- **Customizable Parameters**:
  - **Granularity** (1-100): Controls analysis region size for detection precision
  - **Threshold** (0-1): Sets sensitivity for green vegetation detection
- **Real-time Preview**: Side-by-side comparison of original and processed images
- **Processing Pipeline**:
  - Gaussian blur for noise reduction
  - Scharr edge detection for plant boundary identification
  - Morphological operations for shape refinement
  - Watershed algorithm for precise plant segmentation

### 📊 Data Visualization
- **Interactive Charts**: Built with Chart.js for responsive data exploration
- **Multiple Metrics**: Track height, width, and area measurements
- **Time Series Analysis**: 
  - Real distance mode: True temporal spacing between measurements
  - Equal spacing mode: Uniform intervals for trend analysis
- **Growth Trends**: Visual representation of plant development patterns

### 🗂️ Data Management
- **Local Database**: NeDB-based storage for offline capability
- **Photo Collections**: Organize photos by plant, experiment, or time period
- **Backup & Recovery**: Undo/redo functionality for processing operations
- **Export Options**: Data and charts ready for scientific documentation

## Technology Stack

### Frontend
- **Angular 19**: Modern reactive framework with standalone components
- **TypeScript**: Type-safe development environment
- **SCSS**: Advanced styling with component-scoped styles
- **Chart.js**: Interactive data visualization library

### Backend
- **Electron**: Cross-platform desktop application framework
- **Express.js**: RESTful API server for data operations
- **Node.js**: Runtime environment with ES6 modules

### Image Processing
- **Python 3.11**: Portable Python environment included
- **OpenCV**: Computer vision library for image analysis
- **NumPy**: Numerical computing for image processing algorithms

### Database
- **NeDB**: Embedded JavaScript database for local storage
- **Collections.db**: Stores collection metadata and organization
- **Photos.db**: Manages photo data, parameters, and results

## Installation

### Prerequisites
- Node.js 18+ (for development)
- Git

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd plant-growth

# Navigate to application directory
cd src/app

# Install dependencies
npm install

# Build and run the application
npm run electron
```

### Development Mode
```bash
# Run Angular development server
npm start

# In another terminal, run Electron
npm run electron
```

## Application Structure

```
plant-growth/
├── docs/                      # Documentation and research papers
│   ├── data/                  # Sample images and test data
│   └── paper/                 # LaTeX research documentation
├── src/
│   ├── app/                   # Main Angular application
│   │   ├── electron/          # Electron main process
│   │   │   ├── express/       # Backend API server
│   │   │   │   ├── database/  # NeDB database management
│   │   │   │   ├── image-processing/  # Python CV algorithms
│   │   │   │   └── router/    # API endpoints
│   │   │   └── main.js        # Electron main process
│   │   └── src/               # Angular source code
│   │       ├── app/           # Application components
│   │       │   ├── components/  # Reusable UI components
│   │       │   ├── pages/     # Application pages/routes
│   │       │   ├── services/  # Data services and API clients
│   │       │   └── models/    # TypeScript interfaces
│   │       └── assets/        # Static resources
│   └── test/                  # Testing utilities and scripts
```

## Usage Guide

### 1. Initial Setup
1. Launch the application
2. The main dashboard displays existing collections and photos
3. Create your first collection or start uploading photos directly

### 2. Photo Upload
1. Click "Upload Photos" or drag files to the upload area
2. Select target collection (optional)
3. Photos are automatically processed with default parameters
4. Review uploaded photos in the collections view

### 3. Image Processing
1. Select a collection or individual photos
2. Open the photo editor
3. Adjust processing parameters:
   - **Granularity**: Lower values for detailed analysis, higher for speed
   - **Threshold**: Lower to detect subtle greens, higher for accuracy
4. Process individual photos or batch process all
5. Review results with side-by-side comparison

### 4. Data Analysis
1. Navigate to a collection's analysis view
2. View interactive charts for height, width, and area trends
3. Toggle between real-time and equal-spacing display modes
4. Export charts and data for research documentation

### 5. Collection Management
1. Create themed collections (e.g., "Tomato Week 1-8")
2. Add/remove photos from collections
3. Rename and organize collections
4. Delete collections (photos remain available)

## Image Processing Algorithm

The application uses a sophisticated computer vision pipeline:

1. **Preprocessing**:
   - Convert to grayscale
   - Apply Gaussian blur for noise reduction

2. **Edge Detection**:
   - Scharr operators for X and Y gradients
   - Magnitude calculation for edge strength

3. **Morphological Operations**:
   - Closing operations to fill gaps
   - Rectangular kernel for shape preservation

4. **Segmentation**:
   - Watershed algorithm with distance-based markers
   - Granularity parameter controls marker spacing

5. **Plant Detection**:
   - HSV color space analysis for green vegetation
   - Threshold parameter controls color sensitivity
   - Contour analysis for plant boundary detection

6. **Measurement Extraction**:
   - Bounding box calculation for height and width
   - Contour area calculation for total plant area
   - Pixel-based measurements (convertible to real units)

## Development

### Building
```bash
# Development build
npm run build

# Production build
npm run build -- --configuration production
```

### Testing
The application includes test utilities in `src/test/` for algorithm validation and performance testing.

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate tests
4. Submit a pull request with detailed description

## Research Applications

This application has been developed for academic research in plant growth analysis. Key research features include:

- **Temporal Analysis**: Track growth patterns over extended periods
- **Comparative Studies**: Analyze multiple plant specimens simultaneously
- **Quantitative Metrics**: Extract precise measurements for statistical analysis
- **Reproducible Results**: Consistent processing parameters across experiments
- **Export Capabilities**: Integration with research documentation workflows

## Troubleshooting

### Common Issues

**Image Processing Fails**:
- Ensure photos are clear and well-lit
- Adjust threshold parameter for better plant detection
- Try different granularity settings for your image resolution

**Performance Issues**:
- Process smaller batches of photos
- Increase granularity for faster processing
- Close other applications to free system resources

**Database Errors**:
- Restart the application to refresh database connections
- Check disk space availability
- Backup important collections before major operations

## License

This project is developed for academic research purposes. Please refer to the specific license file for usage terms and conditions.

## Citation

If you use this application in your research, please cite:

```
Cardoso dos Santos, L., & Miranda Mendonça Rezende, L. (2025). 
Plant Growth Analysis Application: Computer Vision-Based Plant Development Tracking. 
Ribeirão Preto: Universidade de São Paulo.
```

## Support

For issues, questions, or contributions:
- Create an issue in the repository
- Contact the development team
- Refer to the documentation in `docs/` for detailed technical information

---

**Note**: This application includes a portable Python environment for cross-platform compatibility. No additional Python installation is required.
