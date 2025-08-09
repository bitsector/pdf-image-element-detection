"""
Method 2: Unstructured Library with Detectron2
Uses Unstructured library for automatic document segmentation and element extraction
"""

import os
import json
import cv2
import numpy as np
from PIL import Image


def detect_elements_with_unstructured(image_path, save_results=True):
    """
    Detect document elements using Unstructured library
    
    Args:
        image_path (str): Path to the image file
        save_results (bool): Whether to save results to JSON
    
    Returns:
        dict: Detected elements with metadata
    """
    try:
        from unstructured.partition.image import partition_image
    except ImportError:
        try:
            from unstructured.partition.auto import partition
            print("Warning: Using basic unstructured partition (Detectron2 not available)")
        except ImportError:
            print("Error: Unstructured library not installed")
            print("Install with: pip install unstructured")
            return []
    
    results = []
    
    try:
        # Use partition_image for direct image processing
        elements = partition_image(image_path)
        
        for idx, element in enumerate(elements):
            element_info = {
                'method': 'Unstructured_Detectron2',
                'element_id': idx,
                'type': type(element).__name__,
                'text': str(element.text) if hasattr(element, 'text') else '',
                'metadata': {}
            }
            
            # Extract coordinates if available
            if hasattr(element, 'metadata') and element.metadata:
                if hasattr(element.metadata, 'coordinates') and element.metadata.coordinates:
                    coords = element.metadata.coordinates
                    # Convert coordinates to bounding box format
                    if hasattr(coords, 'points') and coords.points:
                        points = [(p.x, p.y) for p in coords.points]
                        if points:
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            
                            x = min(x_coords)
                            y = min(y_coords)
                            width = max(x_coords) - x
                            height = max(y_coords) - y
                            
                            element_info['bbox'] = {
                                'x': int(x),
                                'y': int(y),
                                'width': int(width),
                                'height': int(height)
                            }
                
                # Extract other metadata
                element_info['metadata'] = {
                    'page_number': getattr(element.metadata, 'page_number', None),
                    'filename': getattr(element.metadata, 'filename', None),
                    'file_directory': getattr(element.metadata, 'file_directory', None)
                }
            
            results.append(element_info)
        
        # Save results if requested
        if save_results:
            output_path = image_path.replace('.png', '_unstructured_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Unstructured results saved to: {output_path}")
        
        print(f"Unstructured detected {len(results)} elements")
        return results
        
    except Exception as e:
        print(f"Error in Unstructured detection: {str(e)}")
        return []


def visualize_unstructured_results(image_path, results, output_path=None):
    """
    Visualize Unstructured detection results by drawing bounding boxes
    
    Args:
        image_path (str): Path to the original image
        results (list): Results from detect_elements_with_unstructured
        output_path (str): Path to save the visualization
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image for visualization: {image_path}")
        return
    
    # Define colors for different element types
    colors = {
        'Title': (255, 0, 0),        # Blue
        'NarrativeText': (0, 255, 0), # Green
        'Text': (0, 255, 0),         # Green
        'Table': (0, 0, 255),        # Red
        'Image': (255, 255, 0),      # Cyan
        'ListItem': (255, 0, 255),   # Magenta
        'Header': (0, 255, 255),     # Yellow
        'Footer': (128, 0, 128),     # Purple
        'UncategorizedText': (128, 128, 128)  # Gray
    }
    
    for element in results:
        if 'bbox' in element:
            bbox = element['bbox']
            element_type = element['type']
            
            # Get color for element type
            color = colors.get(element_type, colors['UncategorizedText'])
            
            # Draw bounding box
            cv2.rectangle(img, 
                         (bbox['x'], bbox['y']), 
                         (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                         color, 2)
            
            # Add label
            label = f"{element_type}"
            cv2.putText(img, label, 
                       (bbox['x'], bbox['y'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save visualization
    if output_path is None:
        output_path = image_path.replace('.png', '_unstructured_visualization.png')
    
    cv2.imwrite(output_path, img)
    print(f"Unstructured visualization saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test with a sample image
    test_image = "temp/temp_1733875200_page_1.png"  # Example path
    
    if os.path.exists(test_image):
        print("Testing Unstructured library...")
        results = detect_elements_with_unstructured(test_image)
        
        if results:
            visualize_unstructured_results(test_image, results)
            print("Unstructured test completed successfully!")
        else:
            print("No elements detected by Unstructured")
    else:
        print(f"Test image not found: {test_image}")
        print("Please run pdf_to_images.py first to generate test images")