# pdf-image-element-detection

A comprehensive Python project for extracting elements from PDF documents using multiple detection methods including PaddleOCR, Unstructured, OpenCV, YOLO, LayoutLM, and AWS Textract.

## Features

This project implements **5 different methods** for document element detection:

1. **PaddleOCR PP-StructureV3** - Advanced layout detection with 20+ element types
2. **Unstructured Library** - Document parsing (basic version)  
3. **OpenCV Computer Vision** - Traditional contour and rectangle detection
4. **YOLO Object Detection** - Custom element detection with Ultralytics
5. **LayoutLM Multimodal AI** - Document understanding with transformers

*Note: AWS Textract and Detectron2 are commented out to avoid installation issues*

## Setup

### 1. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install Dependencies
```bash
# Install all dependencies (may take several minutes)
pip install -r requirements.txt
```

**Note:** Some dependencies like Detectron2 and PyTorch are large and may take time to install.

### 3. Test Installation
```bash
python test_setup.py
```

### 4. Optional: Advanced Setup
To enable advanced features (commented out by default):
```bash
# For Detectron2 (requires PyTorch first)
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2'

# For AWS Textract
pip install boto3
aws configure  # Enter your AWS credentials
```

## Usage

### Quick Start
Run all methods on the sample PDF:
```bash
python run_all_methods.py
```

### Individual Methods
Test each method separately:
```bash
# Convert PDF to images first
python pdf_to_images.py

# Run individual methods
python method_paddleocr.py
python method_opencv.py
python method_yolo.py
# etc.
```

### Custom PDF
Edit the `pdf_file` variable in `run_all_methods.py` to process your own PDF.

## Output

Results are saved in the `temp/` directory:
- **Images:** `temp_<timestamp>_page_<n>.png` 
- **Detection Results:** `*_results.json` files
- **Visualizations:** `*_visualization.png` files  
- **Summary Report:** `summary_report_<timestamp>.txt`

## Method Comparison

Each method has different strengths:
- **PaddleOCR:** Best for document structure analysis
- **OpenCV:** Fast, good for simple rectangular elements
- **YOLO:** Excellent for object detection, customizable
- **LayoutLM:** Advanced document understanding with context
- **Unstructured:** Great for automatic document parsing
- **AWS Textract:** High accuracy, cloud-based (requires API key)

## Requirements

- Python 3.8+
- 4GB+ RAM (for deep learning models)
- Optional: GPU for faster processing
- Optional: AWS account for Textract
