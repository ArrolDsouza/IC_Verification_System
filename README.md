# IC Detection System

A comprehensive system for detecting fake vs real integrated circuits (ICs) using image analysis and text extraction.

## Features

- **GUI Interface**: User-friendly Tkinter interface for easy image upload and analysis
- **Text Extraction**: Extracts text from IC images using advanced image processing
- **Database Verification**: Cross-references extracted text against IC_DATA.csv database
- **Fake Detection**: Advanced algorithms to detect fake ICs based on multiple quality indicators
- **Responsive Design**: Clean, responsive GUI that adapts to different screen sizes

## Files

- `ic_detector_gui_fixed.py` - **Main GUI application (recommended)**
- `ic_detector_gui_final.py` - GUI with multiple OCR methods
- `ic_detector_gui_fallback.py` - GUI with basic analysis (fallback)
- `ic_detector_gui.py` - GUI with PaddleOCR integration (requires PaddleOCR)
- `IC_DATA.csv` - Database of 60+ IC types from various manufacturers
- `Real_IC/` - Folder containing real IC images for testing
- `Fake_IC/` - Folder containing fake IC images for testing
- `IC.ipynb` - Jupyter notebook with original analysis code

## Installation

1. Install required Python packages:
```bash
pip install opencv-python numpy pandas pillow tkinter
```

2. For enhanced OCR capabilities (optional):
```bash
pip install easyocr pytesseract
```

## Usage

### Running the GUI Application

1. **Recommended (Fixed Version)**:
```bash
python ic_detector_gui_fixed.py
```

2. **Final Version**:
```bash
python ic_detector_gui_final.py
```

3. **Fallback Version**:
```bash
python ic_detector_gui_fallback.py
```

4. **With PaddleOCR** (if installed):
```bash
python ic_detector_gui.py
```

### Using the Interface

1. **Select Image**: Click "Select Image" to choose an IC image file
2. **Analyze**: Click "Analyze" to process the image
3. **View Results**: See the analysis results in the right panel

### Analysis Results

The system provides:
- **Prediction**: REAL or FAKE with confidence percentage
- **Extracted Text**: Text found on the IC
- **Database Match**: Information about matched IC from database
- **Quality Metrics**: Image sharpness, contrast, brightness, edge density
- **Fake Indicators**: Specific reasons why an IC might be fake

## Database

The `IC_DATA.csv` file contains information about 60+ IC types from manufacturers including:
- Texas Instruments
- Analog Devices
- Maxim Integrated
- Linear Technology
- National Semiconductor
- STMicroelectronics
- Microchip
- Intel
- Fairchild Semiconductor
- ON Semiconductor

## Detection Logic

The system uses multiple criteria to determine if an IC is fake:

1. **Image Quality**: Sharpness, contrast, brightness, edge density
2. **Text Extraction**: Number and quality of text elements
3. **Database Matching**: Comparison with known IC patterns
4. **Content Analysis**: Detection of unusual characters or patterns
5. **Consistency Checks**: Verification of text consistency

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## System Requirements

- Python 3.7+
- Windows/Linux/macOS
- Minimum 4GB RAM
- OpenCV, NumPy, Pandas, Pillow, Tkinter

## Troubleshooting

### Common Issues

1. **"No module named 'paddleocr'"**: Use the fallback version instead
2. **Poor text extraction**: Ensure images are well-lit and high resolution
3. **GUI not responding**: Check if image file is corrupted or too large

### Performance Tips

- Use high-resolution images (800x600 or higher)
- Ensure good lighting and contrast
- Avoid blurry or distorted images
- Use images with clear text markings

## Contributing

To add new IC types to the database:
1. Edit `IC_DATA.csv`
2. Add manufacturer, part number, and text patterns
3. Test with sample images

## License

This project is for educational and research purposes.
