"""
Method 3: OpenCV Traditional Computer Vision
Uses OpenCV contour detection and shape analysis to detect rectangular regions
"""

import os
import json
import cv2
import numpy as np
import shutil


def detect_elements_with_opencv(image_path, save_results=True):
    """
    Detect rectangular elements using OpenCV contour detection
    
    Args:
        image_path (str): Path to the image file
        save_results (bool): Whether to save results to JSON
    
    Returns:
        list: Detected rectangular elements with bounding boxes
    """
    results = []
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques to detect different types of rectangles
        methods = [
            ("threshold", lambda x: cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)[1]),
            ("adaptive_threshold", lambda x: cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("canny_edges", lambda x: cv2.Canny(x, 50, 150)),
            ("morph_gradient", lambda x: cv2.morphologyEx(x, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))))
        ]
        
        all_rectangles = []
        
        for method_name, preprocess_func in methods:
            # Apply preprocessing
            processed = preprocess_func(gray)
            
            # Find contours
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            method_rectangles = []
            for contour in contours:
                # Calculate contour area (filter out very small contours)
                area = cv2.contourArea(contour)
                if area < 100:  # Minimum area threshold
                    continue
                
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if shape has 4 vertices (rectangle/square)
                if len(approx) == 4:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter rectangles by size and aspect ratio
                    if w > 20 and h > 20 and w < img.shape[1] * 0.9 and h < img.shape[0] * 0.9:
                        aspect_ratio = w / h
                        
                        # Classify rectangle type based on aspect ratio
                        if 0.9 <= aspect_ratio <= 1.1:
                            rect_type = "square"
                        elif aspect_ratio > 3:
                            rect_type = "horizontal_rectangle"
                        elif aspect_ratio < 0.33:
                            rect_type = "vertical_rectangle"
                        else:
                            rect_type = "rectangle"
                        
                        rectangle_info = {
                            'method': f'OpenCV_{method_name}',
                            'type': rect_type,
                            'bbox': {
                                'x': int(x),
                                'y': int(y),
                                'width': int(w),
                                'height': int(h)
                            },
                            'area': int(area),
                            'aspect_ratio': round(aspect_ratio, 2),
                            'perimeter': int(cv2.arcLength(contour, True)),
                            'vertices': len(approx)
                        }
                        
                        method_rectangles.append(rectangle_info)
            
            all_rectangles.extend(method_rectangles)
        
        # Remove duplicate rectangles (with similar positions)
        unique_rectangles = []
        for rect in all_rectangles:
            is_duplicate = False
            for existing in unique_rectangles:
                # Check if rectangles overlap significantly
                x1, y1, w1, h1 = rect['bbox']['x'], rect['bbox']['y'], rect['bbox']['width'], rect['bbox']['height']
                x2, y2, w2, h2 = existing['bbox']['x'], existing['bbox']['y'], existing['bbox']['width'], existing['bbox']['height']
                
                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                total_area = w1 * h1 + w2 * h2 - overlap_area
                overlap_ratio = overlap_area / total_area if total_area > 0 else 0
                
                if overlap_ratio > 0.7:  # 70% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                rect['element_id'] = len(unique_rectangles)
                unique_rectangles.append(rect)
        
        results = unique_rectangles
        
        # Save results if requested
        if save_results:
            output_path = image_path.replace('.png', '_opencv_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"OpenCV results saved to: {output_path}")
        
        # Create overlay visualization on copied image
        if results:
            create_opencv_overlay(image_path, results)
        
        print(f"OpenCV detected {len(results)} unique rectangular elements")
        return results
        
    except Exception as e:
        print(f"Error in OpenCV detection: {str(e)}")
        return []


def create_opencv_overlay(image_path, results):
    """
    Create an overlay visualization on a copy of the original image
    
    Args:
        image_path (str): Path to the original image
        results (list): Results from detect_elements_with_opencv
    """
    # Create a copy of the original image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_name = os.path.dirname(image_path)
    overlay_path = os.path.join(dir_name, f"{base_name}_opencv_overlay.png")
    
    # Copy the original image
    shutil.copy2(image_path, overlay_path)
    
    # Load the copied image
    img = cv2.imread(overlay_path)
    if img is None:
        print(f"Could not load image for overlay: {overlay_path}")
        return
    
    # Define colors for different rectangle types
    colors = {
        'square': (255, 0, 0),              # Blue
        'rectangle': (0, 255, 0),           # Green
        'horizontal_rectangle': (0, 0, 255), # Red
        'vertical_rectangle': (255, 255, 0), # Cyan
        'unknown': (128, 128, 128)          # Gray
    }
    
    for element in results:
        bbox = element['bbox']
        rect_type = element['type']
        
        # Get color for rectangle type
        color = colors.get(rect_type, colors['unknown'])
        
        # Draw bounding box
        cv2.rectangle(img, 
                     (bbox['x'], bbox['y']), 
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                     color, 2)
        
        # Add label with type and dimensions
        label = f"{rect_type} ({bbox['width']}x{bbox['height']})"
        cv2.putText(img, label, 
                   (bbox['x'], bbox['y'] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save the overlay
    cv2.imwrite(overlay_path, img)
    print(f"OpenCV overlay created: {overlay_path}")
    return overlay_path


def visualize_opencv_results(image_path, results, output_path=None):
    """
    Visualize OpenCV detection results by drawing bounding boxes
    
    Args:
        image_path (str): Path to the original image
        results (list): Results from detect_elements_with_opencv
        output_path (str): Path to save the visualization
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image for visualization: {image_path}")
        return
    
    # Define colors for different rectangle types
    colors = {
        'square': (255, 0, 0),              # Blue
        'rectangle': (0, 255, 0),           # Green
        'horizontal_rectangle': (0, 0, 255), # Red
        'vertical_rectangle': (255, 255, 0), # Cyan
        'unknown': (128, 128, 128)          # Gray
    }
    
    for element in results:
        bbox = element['bbox']
        rect_type = element['type']
        
        # Get color for rectangle type
        color = colors.get(rect_type, colors['unknown'])
        
        # Draw bounding box
        cv2.rectangle(img, 
                     (bbox['x'], bbox['y']), 
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                     color, 2)
        
        # Add label with type and dimensions
        label = f"{rect_type} ({bbox['width']}x{bbox['height']})"
        cv2.putText(img, label, 
                   (bbox['x'], bbox['y'] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save visualization
    if output_path is None:
        output_path = image_path.replace('.png', '_opencv_visualization.png')
    
    cv2.imwrite(output_path, img)
    print(f"OpenCV visualization saved to: {output_path}")
    return output_path


def detect_text_regions_opencv(image_path, save_results=True):
    """
    Specialized function to detect text regions using OpenCV EAST text detector
    
    Args:
        image_path (str): Path to the image file
        save_results (bool): Whether to save results to JSON
    
    Returns:
        list: Detected text regions with bounding boxes
    """
    results = []
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use morphological operations to detect text regions
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))  # Horizontal kernel for text lines
        
        # Apply morphological dilation to connect text characters
        dilated = cv2.dilate(gray, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 200:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter based on text-like properties
                aspect_ratio = w / h
                if aspect_ratio > 1.5 and w > 50:  # Text regions are usually wider than tall
                    text_region_info = {
                        'method': 'OpenCV_TextRegion',
                        'element_id': idx,
                        'type': 'text_region',
                        'bbox': {
                            'x': int(x),
                            'y': int(y),
                            'width': int(w),
                            'height': int(h)
                        },
                        'area': int(area),
                        'aspect_ratio': round(aspect_ratio, 2)
                    }
                    results.append(text_region_info)
        
        # Save results if requested
        if save_results:
            output_path = image_path.replace('.png', '_opencv_text_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"OpenCV text region results saved to: {output_path}")
        
        # Create overlay visualization on copied image
        if results:
            create_opencv_text_overlay(image_path, results)
        
        print(f"OpenCV detected {len(results)} text regions")
        return results
        
    except Exception as e:
        print(f"Error in OpenCV text detection: {str(e)}")
        return []


def create_opencv_text_overlay(image_path, results):
    """
    Create an overlay visualization on a copy of the original image for text regions
    
    Args:
        image_path (str): Path to the original image
        results (list): Results from detect_text_regions_opencv
    """
    # Create a copy of the original image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_name = os.path.dirname(image_path)
    overlay_path = os.path.join(dir_name, f"{base_name}_opencv_text_overlay.png")
    
    # Copy the original image
    shutil.copy2(image_path, overlay_path)
    
    # Load the copied image
    img = cv2.imread(overlay_path)
    if img is None:
        print(f"Could not load image for overlay: {overlay_path}")
        return
    
    # Define color for text regions
    color = (0, 255, 255)  # Yellow for text regions
    
    for element in results:
        bbox = element['bbox']
        
        # Draw bounding box
        cv2.rectangle(img, 
                     (bbox['x'], bbox['y']), 
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                     color, 2)
        
        # Add label with dimensions
        label = f"text ({bbox['width']}x{bbox['height']})"
        cv2.putText(img, label, 
                   (bbox['x'], bbox['y'] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save the overlay
    cv2.imwrite(overlay_path, img)
    print(f"OpenCV text overlay created: {overlay_path}")
    return overlay_path


if __name__ == "__main__":
    # Test with a sample image
    test_image = "temp/temp_1733875200_page_1.png"  # Example path
    
    if os.path.exists(test_image):
        print("Testing OpenCV rectangle detection...")
        results = detect_elements_with_opencv(test_image)
        
        if results:
            visualize_opencv_results(test_image, results)
            
            # Also test text region detection
            print("\nTesting OpenCV text region detection...")
            text_results = detect_text_regions_opencv(test_image)
            
            print("OpenCV test completed successfully!")
        else:
            print("No elements detected by OpenCV")
    else:
        print(f"Test image not found: {test_image}")
        print("Please run pdf_to_images.py first to generate test images")