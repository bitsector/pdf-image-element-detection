"""
Method 1: PaddleOCR PP-StructureV3 Layout Detection
Uses PaddleOCR's advanced layout detection to identify 20+ document element types
"""

import os
import json
import cv2
import numpy as np

try:
    from paddleocr import PPStructure, PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PaddleOCR not available: {e}")
    print("Install with: pip install paddlepaddle paddleocr")
    PADDLEOCR_AVAILABLE = False


def detect_elements_with_paddleocr(image_path, save_results=True):
    """
    Detect document elements using PaddleOCR PP-StructureV3
    
    Args:
        image_path (str): Path to the image file
        save_results (bool): Whether to save results to JSON
    
    Returns:
        dict: Detected elements with bounding boxes and text
    """
    if not PADDLEOCR_AVAILABLE:
        print("PaddleOCR not available - skipping")
        return []
        
    # Initialize PP-Structure for layout detection
    ocr = PPStructure(
        layout_model="PP-DocLayout-M",  # Use PP-DocLayout-M for better layout detection
        table=False,                    # Disable table recognition for now
        ocr=True,                      # Enable OCR
        show_log=False,                # Reduce logging
        recovery=True,                 # Enable reading order recovery
        use_gpu=False                  # Set to True if GPU available
    )
    
    results = []
    
    try:
        # Process the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Run structure analysis
        result = ocr(img)
        
        for idx, element in enumerate(result):
            if 'bbox' in element:
                bbox = element['bbox']
                element_type = element.get('type', 'unknown')
                text_content = element.get('text', '')
                
                # Convert bbox to standard format (x, y, width, height)
                x1, y1, x2, y2 = bbox
                x, y = int(x1), int(y1)
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                element_info = {
                    'method': 'PaddleOCR_PP-StructureV3',
                    'element_id': idx,
                    'type': element_type,
                    'bbox': {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height
                    },
                    'text': text_content,
                    'confidence': element.get('confidence', 0.0)
                }
                results.append(element_info)
        
        # Save results if requested
        if save_results:
            output_path = image_path.replace('.png', '_paddleocr_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"PaddleOCR results saved to: {output_path}")
        
        print(f"PaddleOCR detected {len(results)} elements")
        return results
        
    except Exception as e:
        print(f"Error in PaddleOCR detection: {str(e)}")
        return []


def visualize_paddleocr_results(image_path, results, output_path=None):
    """
    Visualize PaddleOCR detection results by drawing bounding boxes
    
    Args:
        image_path (str): Path to the original image
        results (list): Results from detect_elements_with_paddleocr
        output_path (str): Path to save the visualization
    """
    if not PADDLEOCR_AVAILABLE or not results:
        print("PaddleOCR not available or no results to visualize")
        return None
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image for visualization: {image_path}")
        return
    
    # Define colors for different element types
    colors = {
        'text': (0, 255, 0),      # Green
        'title': (255, 0, 0),     # Blue  
        'table': (0, 0, 255),     # Red
        'figure': (255, 255, 0),  # Cyan
        'unknown': (128, 128, 128) # Gray
    }
    
    for element in results:
        bbox = element['bbox']
        element_type = element['type']
        
        # Get color for element type
        color = colors.get(element_type, colors['unknown'])
        
        # Draw bounding box
        cv2.rectangle(img, 
                     (bbox['x'], bbox['y']), 
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                     color, 2)
        
        # Add label
        label = f"{element_type} ({element.get('confidence', 0):.2f})"
        cv2.putText(img, label, 
                   (bbox['x'], bbox['y'] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save visualization
    if output_path is None:
        output_path = image_path.replace('.png', '_paddleocr_visualization.png')
    
    cv2.imwrite(output_path, img)
    print(f"PaddleOCR visualization saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test with a sample image
    test_image = "temp/temp_1733875200_page_1.png"  # Example path
    
    if os.path.exists(test_image):
        print("Testing PaddleOCR PP-StructureV3...")
        results = detect_elements_with_paddleocr(test_image)
        
        if results:
            visualize_paddleocr_results(test_image, results)
            print("PaddleOCR test completed successfully!")
        else:
            print("No elements detected by PaddleOCR")
    else:
        print(f"Test image not found: {test_image}")
        print("Please run pdf_to_images.py first to generate test images")