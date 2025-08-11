"""
Method 4: YOLO Object Detection for Custom Elements
Uses Ultralytics YOLO for detecting document elements and general objects
"""

import os
import json
import cv2
import numpy as np
import shutil


def detect_elements_with_yolo(image_path, save_results=True, model_name="yolo11n.pt"):
    """
    Detect document elements using YOLO object detection
    
    Args:
        image_path (str): Path to the image file
        save_results (bool): Whether to save results to JSON
        model_name (str): YOLO model to use (yolo11n.pt, yolo11s.pt, etc.)
    
    Returns:
        list: Detected elements with bounding boxes and confidence scores
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: Ultralytics YOLO not installed")
        print("Install with: pip install ultralytics")
        return []
    
    results = []
    
    try:
        # Load YOLO model
        model = YOLO(model_name)
        
        # Run inference
        detections = model(image_path, verbose=False)
        
        # Process results
        for detection in detections:
            if detection.boxes is not None:
                for idx, box in enumerate(detection.boxes):
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Extract confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id] if class_id < len(model.names) else "unknown"
                    
                    # Convert to standard bbox format
                    x = int(x1)
                    y = int(y1)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    # Only include detections with reasonable confidence
                    if confidence > 0.3:  # Confidence threshold
                        element_info = {
                            'method': f'YOLO_{model_name}',
                            'element_id': idx,
                            'type': class_name,
                            'bbox': {
                                'x': x,
                                'y': y,
                                'width': width,
                                'height': height
                            },
                            'confidence': round(confidence, 3),
                            'class_id': class_id
                        }
                        results.append(element_info)
        
        # Save results if requested
        if save_results:
            output_path = image_path.replace('.png', '_yolo_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"YOLO results saved to: {output_path}")
        
        # Create overlay visualization on copied image
        if results:
            create_yolo_overlay(image_path, results)
        
        print(f"YOLO detected {len(results)} elements")
        return results
        
    except Exception as e:
        print(f"Error in YOLO detection: {str(e)}")
        return []


def create_yolo_overlay(image_path, results):
    """
    Create an overlay visualization on a copy of the original image
    
    Args:
        image_path (str): Path to the original image
        results (list): Results from detect_elements_with_yolo
    """
    # Create a copy of the original image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_name = os.path.dirname(image_path)
    overlay_path = os.path.join(dir_name, f"{base_name}_yolo_overlay.png")
    
    # Copy the original image
    shutil.copy2(image_path, overlay_path)
    
    # Load the copied image
    img = cv2.imread(overlay_path)
    if img is None:
        print(f"Could not load image for overlay: {overlay_path}")
        return
    
    # Define colors for different object types (COCO classes)
    colors = {
        'person': (255, 0, 0),        # Blue
        'book': (0, 255, 0),          # Green
        'cell phone': (0, 0, 255),    # Red
        'laptop': (255, 255, 0),      # Cyan
        'mouse': (255, 0, 255),       # Magenta
        'keyboard': (0, 255, 255),    # Yellow
        'bottle': (128, 0, 128),      # Purple
        'cup': (255, 165, 0),         # Orange
        'chair': (0, 128, 255),       # Light Blue
        'couch': (128, 255, 0),       # Light Green
        'potted plant': (255, 192, 203), # Pink
        'bed': (165, 42, 42),         # Brown
        'dining table': (128, 128, 0), # Olive
        'toilet': (128, 0, 0),        # Maroon
        'tv': (0, 128, 0),            # Dark Green
        'scissors': (0, 0, 128),      # Navy
        'teddy bear': (255, 20, 147), # Deep Pink
        'hair drier': (255, 69, 0),   # Red Orange
        'toothbrush': (50, 205, 50)   # Lime Green
    }
    
    for element in results:
        bbox = element['bbox']
        element_type = element['type']
        confidence = element['confidence']
        
        # Get color for element type (default to gray if not found)
        color = colors.get(element_type, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(img, 
                     (bbox['x'], bbox['y']), 
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                     color, 2)
        
        # Add label with confidence
        label = f"{element_type} ({confidence:.2f})"
        cv2.putText(img, label, 
                   (bbox['x'], bbox['y'] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save the overlay
    cv2.imwrite(overlay_path, img)
    print(f"YOLO overlay created: {overlay_path}")
    return overlay_path


def visualize_yolo_results(image_path, results, output_path=None):
    """
    Visualize YOLO detection results by drawing bounding boxes
    
    Args:
        image_path (str): Path to the original image
        results (list): Results from detect_elements_with_yolo
        output_path (str): Path to save the visualization
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image for visualization: {image_path}")
        return
    
    # Define colors for different object types (COCO classes)
    colors = {
        'person': (255, 0, 0),        # Blue
        'book': (0, 255, 0),          # Green
        'cell phone': (0, 0, 255),    # Red
        'laptop': (255, 255, 0),      # Cyan
        'mouse': (255, 0, 255),       # Magenta
        'keyboard': (0, 255, 255),    # Yellow
        'bottle': (128, 0, 128),      # Purple
        'cup': (255, 165, 0),         # Orange
        'chair': (0, 128, 255),       # Light Blue
        'couch': (128, 255, 0),       # Light Green
        'potted plant': (255, 192, 203), # Pink
        'bed': (165, 42, 42),         # Brown
        'dining table': (128, 128, 0), # Olive
        'toilet': (128, 0, 0),        # Maroon
        'tv': (0, 128, 0),            # Dark Green
        'scissors': (0, 0, 128),      # Navy
        'teddy bear': (255, 20, 147), # Deep Pink
        'hair drier': (255, 69, 0),   # Red Orange
        'toothbrush': (50, 205, 50)   # Lime Green
    }
    
    for element in results:
        bbox = element['bbox']
        element_type = element['type']
        confidence = element['confidence']
        
        # Get color for element type (default to gray if not found)
        color = colors.get(element_type, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(img, 
                     (bbox['x'], bbox['y']), 
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                     color, 2)
        
        # Add label with confidence
        label = f"{element_type} ({confidence:.2f})"
        cv2.putText(img, label, 
                   (bbox['x'], bbox['y'] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save visualization
    if output_path is None:
        output_path = image_path.replace('.png', '_yolo_visualization.png')
    
    cv2.imwrite(output_path, img)
    print(f"YOLO visualization saved to: {output_path}")
    return output_path


def detect_custom_document_elements(image_path, save_results=True):
    """
    Detect document-specific elements using YOLO-World for open vocabulary detection
    
    Args:
        image_path (str): Path to the image file
        save_results (bool): Whether to save results to JSON
    
    Returns:
        list: Detected document elements
    """
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        print("YOLOWorld not available, falling back to regular YOLO")
        return detect_elements_with_yolo(image_path, save_results, "yolo11n.pt")
    
    results = []
    
    try:
        # Load YOLO-World model for open vocabulary detection
        model = YOLOWorld("yolov8s-worldv2.pt")
        
        # Define custom classes for document elements
        document_classes = [
            "text block", "paragraph", "title", "heading", 
            "table", "image", "figure", "logo", "signature",
            "price", "offer", "discount", "sale", "banner",
            "button", "form", "checkbox", "textbox"
        ]
        
        # Set custom vocabulary
        model.set_classes(document_classes)
        
        # Run inference
        detections = model(image_path, verbose=False)
        
        # Process results similar to regular YOLO
        for detection in detections:
            if detection.boxes is not None:
                for idx, box in enumerate(detection.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence > 0.2:  # Lower threshold for document elements
                        class_name = document_classes[class_id] if class_id < len(document_classes) else "unknown"
                        
                        element_info = {
                            'method': 'YOLO_World_Custom',
                            'element_id': idx,
                            'type': class_name,
                            'bbox': {
                                'x': int(x1),
                                'y': int(y1),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1)
                            },
                            'confidence': round(confidence, 3),
                            'class_id': class_id
                        }
                        results.append(element_info)
        
        # Save results if requested
        if save_results:
            output_path = image_path.replace('.png', '_yolo_world_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"YOLO-World results saved to: {output_path}")
        
        # Create overlay visualization on copied image
        if results:
            create_yolo_world_overlay(image_path, results)
        
        print(f"YOLO-World detected {len(results)} document elements")
        return results
        
    except Exception as e:
        print(f"Error in YOLO-World detection: {str(e)}")
        # Fallback to regular YOLO
        return detect_elements_with_yolo(image_path, save_results, "yolo11n.pt")


def create_yolo_world_overlay(image_path, results):
    """
    Create an overlay visualization on a copy of the original image for YOLO-World results
    
    Args:
        image_path (str): Path to the original image
        results (list): Results from detect_custom_document_elements
    """
    # Create a copy of the original image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    dir_name = os.path.dirname(image_path)
    overlay_path = os.path.join(dir_name, f"{base_name}_yolo_world_overlay.png")
    
    # Copy the original image
    shutil.copy2(image_path, overlay_path)
    
    # Load the copied image
    img = cv2.imread(overlay_path)
    if img is None:
        print(f"Could not load image for overlay: {overlay_path}")
        return
    
    # Define colors for different document element types
    colors = {
        'text block': (0, 255, 0),        # Green
        'paragraph': (0, 255, 128),       # Light Green
        'title': (0, 0, 255),             # Red
        'heading': (128, 0, 255),         # Purple
        'table': (255, 0, 0),             # Blue
        'image': (255, 255, 0),           # Cyan
        'figure': (255, 128, 0),          # Orange
        'logo': (128, 255, 255),          # Light Cyan
        'signature': (255, 0, 255),       # Magenta
        'price': (0, 255, 255),           # Yellow
        'offer': (255, 128, 128),         # Light Red
        'discount': (128, 255, 128),      # Light Green
        'sale': (255, 255, 128),          # Light Yellow
        'banner': (128, 128, 255),        # Light Blue
        'button': (255, 192, 203),        # Pink
        'form': (165, 42, 42),            # Brown
        'checkbox': (0, 128, 255),        # Light Blue
        'textbox': (128, 255, 0)          # Lime
    }
    
    for element in results:
        bbox = element['bbox']
        element_type = element['type']
        confidence = element['confidence']
        
        # Get color for element type (default to gray if not found)
        color = colors.get(element_type, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(img, 
                     (bbox['x'], bbox['y']), 
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                     color, 2)
        
        # Add label with confidence
        label = f"{element_type} ({confidence:.2f})"
        cv2.putText(img, label, 
                   (bbox['x'], bbox['y'] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save the overlay
    cv2.imwrite(overlay_path, img)
    print(f"YOLO-World overlay created: {overlay_path}")
    return overlay_path


if __name__ == "__main__":
    # Test with a sample image
    test_image = "temp/temp_1733875200_page_1.png"  # Example path
    
    if os.path.exists(test_image):
        print("Testing YOLO object detection...")
        results = detect_elements_with_yolo(test_image)
        
        if results:
            visualize_yolo_results(test_image, results)
            
            # Also test custom document element detection
            print("\nTesting YOLO-World custom document detection...")
            custom_results = detect_custom_document_elements(test_image)
            
            print("YOLO test completed successfully!")
        else:
            print("No elements detected by YOLO")
    else:
        print(f"Test image not found: {test_image}")
        print("Please run pdf_to_images.py first to generate test images")