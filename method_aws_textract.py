"""
Method 6: AWS Textract - Cloud Document Analysis
Uses AWS Textract for comprehensive document analysis and block detection
NOTE: Requires AWS credentials and API access
"""

import os
import json
import cv2
import numpy as np
from PIL import Image


def detect_elements_with_textract(image_path, save_results=True):
    """
    Detect document elements using AWS Textract
    
    Args:
        image_path (str): Path to the image file
        save_results (bool): Whether to save results to JSON
    
    Returns:
        list: Detected elements with AWS Textract analysis
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
    except ImportError:
        print("Error: AWS SDK (boto3) not installed")
        print("Install with: pip install boto3")
        return []
    
    results = []
    
    try:
        # Initialize Textract client
        try:
            textract = boto3.client('textract')
        except NoCredentialsError:
            print("Error: AWS credentials not configured")
            print("Configure with: aws configure")
            print("Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
            return []
        
        # Read image file
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
        
        # Call Textract analyze_document
        response = textract.analyze_document(
            Document={'Bytes': image_bytes},
            FeatureTypes=['TABLES', 'FORMS']  # Enable table and form detection
        )
        
        # Process the response blocks
        blocks = response.get('Blocks', [])
        
        for block in blocks:
            block_type = block.get('BlockType', '')
            
            if block_type in ['LINE', 'WORD', 'TABLE', 'CELL', 'KEY_VALUE_SET']:
                # Extract bounding box
                geometry = block.get('Geometry', {})
                bbox_info = geometry.get('BoundingBox', {})
                
                if bbox_info:
                    # Get image dimensions for coordinate conversion
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                        
                        # Convert normalized coordinates to pixel coordinates
                        x = int(bbox_info['Left'] * img_width)
                        y = int(bbox_info['Top'] * img_height)
                        width = int(bbox_info['Width'] * img_width)
                        height = int(bbox_info['Height'] * img_height)
                        
                        # Extract text content
                        text_content = block.get('Text', '')
                        confidence = block.get('Confidence', 0.0)
                        
                        # Determine element type based on block type and relationships
                        element_type = map_textract_block_type(block, blocks)
                        
                        element_info = {
                            'method': 'AWS_Textract',
                            'element_id': block.get('Id', ''),
                            'type': element_type,
                            'block_type': block_type,
                            'text': text_content,
                            'bbox': {
                                'x': x,
                                'y': y,
                                'width': width,
                                'height': height
                            },
                            'confidence': round(confidence / 100.0, 3),  # Convert to 0-1 scale
                            'relationships': block.get('Relationships', [])
                        }
                        
                        # Add additional metadata for specific block types
                        if block_type == 'TABLE':
                            element_info['row_count'] = get_table_row_count(block, blocks)
                            element_info['column_count'] = get_table_column_count(block, blocks)
                        elif block_type == 'KEY_VALUE_SET':
                            element_info['entity_type'] = block.get('EntityTypes', [])
                        
                        results.append(element_info)
        
        # Save results if requested
        if save_results:
            output_path = image_path.replace('.png', '_textract_results.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Textract results saved to: {output_path}")
        
        print(f"AWS Textract detected {len(results)} elements")
        return results
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'InvalidImageFormatException':
            print("Error: Invalid image format for Textract")
        elif error_code == 'UnsupportedDocumentException':
            print("Error: Unsupported document type for Textract")
        else:
            print(f"AWS Textract error: {str(e)}")
        return []
    except Exception as e:
        print(f"Error in AWS Textract detection: {str(e)}")
        return []


def map_textract_block_type(block, all_blocks):
    """
    Map Textract block type to more semantic element types
    
    Args:
        block (dict): Textract block
        all_blocks (list): All blocks for context
    
    Returns:
        str: Semantic element type
    """
    block_type = block.get('BlockType', '')
    
    if block_type == 'LINE':
        # Analyze text content to classify lines
        text = block.get('Text', '').strip()
        if text:
            # Check if it looks like a title (all caps, short)
            if text.isupper() and len(text.split()) <= 5:
                return 'title'
            # Check if it looks like a price
            elif '$' in text or '€' in text or '£' in text:
                return 'price'
            # Check if it contains numbers that might be quantities
            elif any(char.isdigit() for char in text):
                return 'numeric_text'
            else:
                return 'text_line'
    elif block_type == 'WORD':
        text = block.get('Text', '')
        if '$' in text or '€' in text or '£' in text:
            return 'price_word'
        else:
            return 'word'
    elif block_type == 'TABLE':
        return 'table'
    elif block_type == 'CELL':
        return 'table_cell'
    elif block_type == 'KEY_VALUE_SET':
        entity_types = block.get('EntityTypes', [])
        if 'KEY' in entity_types:
            return 'form_key'
        elif 'VALUE' in entity_types:
            return 'form_value'
        else:
            return 'form_field'
    else:
        return block_type.lower()


def get_table_row_count(table_block, all_blocks):
    """
    Count rows in a table block
    """
    row_ids = set()
    relationships = table_block.get('Relationships', [])
    
    for relationship in relationships:
        if relationship.get('Type') == 'CHILD':
            for child_id in relationship.get('Ids', []):
                child_block = next((b for b in all_blocks if b.get('Id') == child_id), None)
                if child_block and child_block.get('BlockType') == 'CELL':
                    row_index = child_block.get('RowIndex', 0)
                    if row_index > 0:
                        row_ids.add(row_index)
    
    return len(row_ids)


def get_table_column_count(table_block, all_blocks):
    """
    Count columns in a table block
    """
    col_ids = set()
    relationships = table_block.get('Relationships', [])
    
    for relationship in relationships:
        if relationship.get('Type') == 'CHILD':
            for child_id in relationship.get('Ids', []):
                child_block = next((b for b in all_blocks if b.get('Id') == child_id), None)
                if child_block and child_block.get('BlockType') == 'CELL':
                    col_index = child_block.get('ColumnIndex', 0)
                    if col_index > 0:
                        col_ids.add(col_index)
    
    return len(col_ids)


def visualize_textract_results(image_path, results, output_path=None):
    """
    Visualize AWS Textract detection results by drawing bounding boxes
    
    Args:
        image_path (str): Path to the original image
        results (list): Results from detect_elements_with_textract
        output_path (str): Path to save the visualization
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image for visualization: {image_path}")
        return
    
    # Define colors for different Textract element types
    colors = {
        'title': (255, 0, 0),        # Blue
        'text_line': (0, 255, 0),    # Green
        'price': (0, 0, 255),        # Red
        'price_word': (0, 0, 255),   # Red
        'numeric_text': (255, 255, 0), # Cyan
        'table': (255, 0, 255),      # Magenta
        'table_cell': (200, 0, 200), # Dark Magenta
        'form_key': (0, 255, 255),   # Yellow
        'form_value': (0, 200, 200), # Dark Yellow
        'word': (128, 128, 128),     # Gray
        'unknown': (64, 64, 64)      # Dark Gray
    }
    
    for element in results:
        bbox = element['bbox']
        element_type = element['type']
        confidence = element.get('confidence', 0)
        text = element.get('text', '')
        
        # Get color for element type
        color = colors.get(element_type, colors['unknown'])
        
        # Draw bounding box
        cv2.rectangle(img, 
                     (bbox['x'], bbox['y']), 
                     (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), 
                     color, 2)
        
        # Add label with confidence and text preview
        text_preview = text[:15] + "..." if len(text) > 15 else text
        label = f"{element_type} ({confidence:.2f}): {text_preview}"
        cv2.putText(img, label, 
                   (bbox['x'], bbox['y'] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Save visualization
    if output_path is None:
        output_path = image_path.replace('.png', '_textract_visualization.png')
    
    cv2.imwrite(output_path, img)
    print(f"Textract visualization saved to: {output_path}")
    return output_path


def check_aws_credentials():
    """
    Check if AWS credentials are configured
    
    Returns:
        bool: True if credentials are available
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
        
        # Try to create a client
        boto3.client('textract')
        return True
    except (NoCredentialsError, ImportError):
        return False


if __name__ == "__main__":
    # Check AWS credentials first
    if not check_aws_credentials():
        print("AWS credentials not configured.")
        print("To use AWS Textract, you need to:")
        print("1. Install boto3: pip install boto3")
        print("2. Configure credentials: aws configure")
        print("3. Or set environment variables: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        exit(1)
    
    # Test with a sample image
    test_image = "temp/temp_1733875200_page_1.png"  # Example path
    
    if os.path.exists(test_image):
        print("Testing AWS Textract document analysis...")
        results = detect_elements_with_textract(test_image)
        
        if results:
            visualize_textract_results(test_image, results)
            print("AWS Textract test completed successfully!")
        else:
            print("No elements detected by AWS Textract")
    else:
        print(f"Test image not found: {test_image}")
        print("Please run pdf_to_images.py first to generate test images")