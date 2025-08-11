"""
Method 6: AWS Textract - Cloud Document Analysis on PNG Images
Processes PNG images with AWS Textract and overlays bounding boxes
NOTE: Requires AWS credentials and API access
"""

import os
import json
import subprocess
import shutil
import time
from PIL import Image, ImageDraw


def detect_elements_with_textract_png_images(output_dir="temp", save_results=True):
    """
    Detect document elements using AWS Textract on hardcoded PNG images
    
    Args:
        output_dir (str): Directory to save results and overlaid images
        save_results (bool): Whether to save results to JSON
    
    Returns:
        dict: Results per page with AWS Textract analysis
    """
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
    except ImportError:
        print("Error: boto3 not installed")
        print("Install with: pip install boto3")
        return {}
    
    try:
        # Initialize AWS Textract client
        textract = boto3.client('textract', region_name='us-east-1')
        print(f"✅ AWS Textract client initialized")
    except NoCredentialsError:
        print("❌ Error: AWS credentials not configured")
        print("Configure with: aws configure")
        return {}
    
    # Hardcoded image files (your uploaded PNG files)
    image_files = [
        "temp_1754761991_page_1.png",
        "temp_1754761991_page_2.png"
    ]
    
    print(f"🔍 Processing {len(image_files)} PNG images with AWS Textract")
    os.makedirs(output_dir, exist_ok=True)
    
    page_results = {}
    
    for idx, image_file in enumerate(image_files):
        page_num = idx + 1
        print(f"\n📄 Processing Page {page_num}: {image_file}")
        
        if not os.path.exists(image_file):
            print(f"❌ Image file not found: {image_file}")
            continue
        
        try:
            # Read image file as bytes
            with open(image_file, 'rb') as f:
                image_bytes = f.read()
            
            print(f"📊 Image size: {len(image_bytes)} bytes")
            
            # Call AWS Textract analyze_document
            print("📡 Calling AWS Textract...")
            response = textract.analyze_document(
                Document={'Bytes': image_bytes},
                FeatureTypes=['TABLES', 'FORMS']  # Enable table and form detection
            )
            
            print(f"✅ Textract processed image successfully")
            blocks = response.get('Blocks', [])
            print(f"📦 Total blocks detected: {len(blocks)}")
            
            # Save raw response for this page
            raw_json_path = os.path.join(output_dir, f'textract_raw_page_{page_num}.json')
            with open(raw_json_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            print(f"💾 Raw response saved: {raw_json_path}")
            
            # Create overlay image (copy of original)
            overlay_image_path = os.path.join(output_dir, f'textract_overlay_page_{page_num}.png')
            shutil.copy2(image_file, overlay_image_path)
            
            # Load image for drawing overlays
            overlay_image = Image.open(overlay_image_path).convert('RGB')
            draw = ImageDraw.Draw(overlay_image)
            W, H = overlay_image.size
            print(f"🖼️  Image dimensions: {W}x{H}")
            
            # Define colors for different block types
            colors = {
                'LINE': (255, 0, 0),           # Red
                'WORD': (0, 255, 0),           # Green  
                'TABLE': (255, 0, 255),        # Magenta
                'CELL': (0, 255, 255),         # Cyan
                'KEY_VALUE_SET': (255, 255, 0), # Yellow
                'SELECTION_ELEMENT': (128, 0, 255) # Purple
            }
            
            # Process blocks and create overlays
            page_elements = []
            element_count = 0
            crop_dir = os.path.join(output_dir, f'crops_page_{page_num}')
            os.makedirs(crop_dir, exist_ok=True)
            
            # Filter for relevant block types
            target_block_types = ['LINE', 'WORD', 'TABLE', 'CELL', 'KEY_VALUE_SET', 'SELECTION_ELEMENT']
            
            for block in blocks:
                block_type = block.get('BlockType', '')
                
                if block_type in target_block_types:
                    geometry = block.get('Geometry', {})
                    bbox_info = geometry.get('BoundingBox', {})
                    
                    if bbox_info:
                        # Convert normalized coordinates (0-1) to pixel coordinates
                        left = bbox_info.get('Left', 0)
                        top = bbox_info.get('Top', 0)
                        width = bbox_info.get('Width', 0)
                        height = bbox_info.get('Height', 0)
                        
                        # Convert to pixel coordinates
                        x1 = int(left * W)
                        y1 = int(top * H)
                        x2 = int((left + width) * W)
                        y2 = int((top + height) * H)
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, W-1))
                        y1 = max(0, min(y1, H-1))
                        x2 = max(x1+1, min(x2, W))
                        y2 = max(y1+1, min(y2, H))
                        
                        color = colors.get(block_type, (128, 128, 128))
                        
                        # Draw bounding box
                        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                        
                        # Get text content and confidence
                        text_content = block.get('Text', '')
                        confidence = block.get('Confidence', 0)
                        
                        # Create label
                        label = f"{block_type}"
                        if text_content:
                            text_preview = text_content[:20].replace('\\n', ' ')
                            label += f": {text_preview}..."
                        if confidence > 0:
                            label += f" ({confidence:.1f}%)"
                        
                        # Draw label (with background for readability)
                        try:
                            # Try to draw text with background
                            label_y = max(10, y1 - 25)
                            draw.rectangle([(x1, label_y), (x1 + len(label) * 6, label_y + 15)], fill=(0, 0, 0, 128))
                            draw.text((x1, label_y), label, fill=color)
                        except:
                            # Fallback: simple text
                            draw.text((x1, max(10, y1-15)), label, fill=color)
                        
                        # Save individual crop
                        if x2 > x1 and y2 > y1:
                            try:
                                crop = overlay_image.crop((x1, y1, x2, y2))
                                crop_filename = f'crop_{element_count:04d}_{block_type}'
                                if text_content:
                                    # Add text preview to filename (sanitized)
                                    text_safe = ''.join(c for c in text_content[:10] if c.isalnum() or c in '-_')
                                    if text_safe:
                                        crop_filename += f'_{text_safe}'
                                crop_filename += '.png'
                                
                                crop_path = os.path.join(crop_dir, crop_filename)
                                crop.save(crop_path)
                            except Exception as e:
                                print(f"⚠️  Could not save crop {element_count}: {e}")
                                crop_path = None
                        else:
                            crop_path = None
                        
                        # Store element info
                        element_info = {
                            'method': 'AWS_Textract_PNG',
                            'element_id': block.get('Id', ''),
                            'type': block_type.lower(),
                            'text': text_content,
                            'bbox': {
                                'x': x1,
                                'y': y1,
                                'width': x2 - x1,
                                'height': y2 - y1
                            },
                            'confidence': confidence,
                            'page': page_num,
                            'crop_path': crop_path
                        }
                        page_elements.append(element_info)
                        element_count += 1
            
            # Save overlaid image
            overlay_image.save(overlay_image_path)
            print(f"🎨 Overlaid {element_count} elements on page {page_num}")
            print(f"💾 Overlay saved: {overlay_image_path}")
            print(f"✂️  Crops saved: {crop_dir} ({len([f for f in os.listdir(crop_dir) if f.endswith('.png')])} files)")
            
            # Store results for this page
            page_results[f'page_{page_num}'] = {
                'page_number': page_num,
                'source_image': image_file,
                'element_count': len(page_elements),
                'elements': page_elements,
                'overlay_image': overlay_image_path,
                'crops_directory': crop_dir,
                'raw_json': raw_json_path
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print(f"❌ AWS Textract error ({error_code}): {e.response['Error']['Message']}")
            page_results[f'page_{page_num}'] = {
                'page_number': page_num,
                'source_image': image_file,
                'error': f"{error_code}: {e.response['Error']['Message']}",
                'element_count': 0,
                'elements': []
            }
        except Exception as e:
            print(f"❌ Error processing {image_file}: {str(e)}")
            page_results[f'page_{page_num}'] = {
                'page_number': page_num,
                'source_image': image_file,
                'error': str(e),
                'element_count': 0,
                'elements': []
            }
    
    # Save consolidated results
    if save_results and page_results:
        results_path = os.path.join(output_dir, 'textract_png_analysis_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(page_results, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Consolidated results saved: {results_path}")
    
    # Generate summary
    total_elements = sum(page.get('element_count', 0) for page in page_results.values())
    successful_pages = len([p for p in page_results.values() if 'error' not in p])
    
    print(f"\n🎉 AWS Textract PNG Analysis Complete!")
    print(f"   📊 Total elements detected: {total_elements}")
    print(f"   📄 Pages processed successfully: {successful_pages}/{len(image_files)}")
    print(f"   🖼️  Check {output_dir} for:")
    print(f"      - textract_overlay_page_*.png (overlaid images)")
    print(f"      - crops_page_*/ (individual element crops)")
    print(f"      - textract_png_analysis_results.json (structured results)")
    
    return page_results


# Wrapper functions for compatibility with run_all_methods.py
def detect_elements_with_textract(image_path, save_results=True):
    """
    Wrapper function for compatibility with run_all_methods.py
    Runs AWS Textract S3 pipeline on hardcoded S3 images
    """
    print("⚠️  AWS Textract wrapper called - running S3 pipeline on hardcoded S3 images")
    
    # Run the S3 pipeline
    results = run_aws_textract_s3_pipeline()
    
    # Return simplified results for compatibility
    aws_results = results.get('aws_results', {})
    if aws_results:
        # Convert to element list format for compatibility
        all_elements = []
        for page_key, page_data in aws_results.items():
            if page_data.get('success'):
                element = {
                    'method': 'AWS_Textract_S3',
                    'page': page_key,
                    'blocks_count': page_data.get('blocks_count', 0),
                    'json_file': page_data.get('json_file', ''),
                    's3_key': page_data.get('s3_key', '')
                }
                all_elements.append(element)
        return all_elements
    return []


def visualize_textract_results(image_path, results, output_path=None):
    """
    Wrapper function for compatibility - visualization is built into S3 pipeline
    """
    if not results:
        print("No AWS Textract results to visualize")
        return None
    print("⚠️  AWS Textract visualization is built into the S3 pipeline")
    return None


def check_aws_credentials():
    """Check if AWS credentials are configured"""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
        boto3.client('textract')
        return True
    except (NoCredentialsError, ImportError):
        return False


def run_textract_on_png_images():
    """
    Run AWS Textract on the hardcoded PNG images
    """
    print("🚀 Running AWS Textract on PNG Images")
    print("   Images: temp_1754761991_page_1.png, temp_1754761991_page_2.png")
    
    # Check prerequisites
    if not check_aws_credentials():
        print("❌ AWS credentials not configured")
        print("Configure with: aws configure")
        return
    
    # Run analysis
    results = detect_elements_with_textract_png_images()
    
    if results:
        print("\n✅ Analysis completed successfully!")
        successful_pages = [p for p in results.values() if 'error' not in p]
        if successful_pages:
            print(f"🔍 Successfully processed {len(successful_pages)} pages")
            print("📂 Output files created in temp/ directory")
        else:
            print("⚠️  No pages processed successfully")
    else:
        print("❌ Analysis failed - no results generated")


def run_aws_textract_subprocess(output_dir="temp"):
    """
    Run AWS Textract using subprocess calls to AWS CLI on S3 images
    
    Args:
        output_dir (str): Directory to save results
    
    Returns:
        dict: Results from both pages
    """
    print("🚀 Running AWS Textract via subprocess on S3 images")
    print("=" * 50)
    
    # S3 bucket and image files (hardcoded)
    s3_bucket = "ocr-bucker"
    s3_images = [
        "temp_1754761991_page_1.png",
        "temp_1754761991_page_2.png"
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    timestamp = int(time.time())
    
    for i, s3_key in enumerate(s3_images, 1):
        page_num = i
        print(f"\n📄 Processing Page {page_num}: s3://{s3_bucket}/{s3_key}")
        
        # JSON output file
        json_file = os.path.join(output_dir, f'textract_s3_page_{page_num}.json')
        
        # Construct AWS CLI command
        aws_cmd = [
            'aws', 'textract', 'analyze-document',
            '--region', 'us-east-1',
            '--document', f'{{"S3Object":{{"Bucket":"{s3_bucket}","Name":"{s3_key}"}}}}',
            '--feature-types', '["TABLES","FORMS"]',
            '--output', 'json'
        ]
        
        print(f"📡 Running AWS CLI command...")
        print(f"   aws textract analyze-document --region us-east-1 --document '{{\"S3Object\":{{\"Bucket\":\"{s3_bucket}\",\"Name\":\"{s3_key}\"}}}}'")
        
        try:
            # Run the AWS CLI command
            result = subprocess.run(
                aws_cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode == 0:
                # Save JSON response
                with open(json_file, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                
                print(f"✅ AWS CLI command succeeded")
                print(f"💾 JSON saved: {json_file}")
                
                # Parse the JSON to get block count
                try:
                    response_data = json.loads(result.stdout)
                    blocks = response_data.get('Blocks', [])
                    print(f"📦 Total blocks detected: {len(blocks)}")
                    
                    results[f'page_{page_num}'] = {
                        'success': True,
                        'json_file': json_file,
                        'blocks_count': len(blocks),
                        's3_key': s3_key,
                        'timestamp': timestamp
                    }
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️  Could not parse JSON response: {e}")
                    results[f'page_{page_num}'] = {
                        'success': False,
                        'error': f"JSON parse error: {e}",
                        'json_file': json_file
                    }
            else:
                print(f"❌ AWS CLI command failed (exit code {result.returncode})")
                print(f"📄 stdout: {result.stdout}")
                print(f"📄 stderr: {result.stderr}")
                
                results[f'page_{page_num}'] = {
                    'success': False,
                    'error': f"AWS CLI failed: {result.stderr}",
                    'exit_code': result.returncode
                }
                
        except subprocess.TimeoutExpired:
            print(f"❌ AWS CLI command timed out")
            results[f'page_{page_num}'] = {
                'success': False,
                'error': "Command timed out"
            }
        except Exception as e:
            print(f"❌ Error running AWS CLI: {str(e)}")
            results[f'page_{page_num}'] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def create_textract_overlays_from_s3_json(aws_results, output_dir="temp"):
    """
    Create overlay images from AWS Textract JSON files with timestamp naming
    
    Args:
        aws_results (dict): Results from run_aws_textract_subprocess
        output_dir (str): Directory containing JSON files and to save overlays
    """
    print(f"\n🎨 Creating overlay images from S3 Textract JSON files")
    print("-" * 40)
    
    # Define colors for different block types
    colors = {
        'LINE': (255, 0, 0),           # Red - Text lines
        'WORD': (0, 255, 0),           # Green - Individual words
        'TABLE': (255, 0, 255),        # Magenta - Tables
        'CELL': (0, 255, 255),         # Cyan - Table cells
        'KEY_VALUE_SET': (255, 255, 0), # Yellow - Form fields
        'SELECTION_ELEMENT': (128, 0, 255), # Purple - Checkboxes
        'SIGNATURE': (255, 165, 0)     # Orange - Signatures
    }
    
    # Find corresponding image files in temp directory
    temp_images = []
    if os.path.exists(output_dir):
        png_files = [f for f in os.listdir(output_dir) if f.endswith('.png') and 'temp_' in f and '_page_' in f and 'overlay' not in f]
        temp_images = sorted(png_files)
    
    print(f"🖼️  Found {len(temp_images)} potential image files: {temp_images}")
    
    overlay_results = {}
    
    for page_key, page_data in aws_results.items():
        if not page_data.get('success'):
            print(f"❌ Skipping {page_key} - AWS command failed")
            continue
            
        page_num = page_key.split('_')[1]  # Extract page number
        json_file = page_data['json_file']
        timestamp = page_data.get('timestamp', int(time.time()))
        
        # Find matching image file
        matching_images = [img for img in temp_images if f'page_{page_num}' in img]
        if not matching_images:
            print(f"❌ No matching image found for {page_key}")
            continue
            
        image_file = os.path.join(output_dir, matching_images[0])
        overlay_file = os.path.join(output_dir, f'temp_aws_textract_overlay_{timestamp}_page_{page_num}.png')
        
        print(f"\n📄 Processing {page_key}:")
        print(f"   JSON: {json_file}")
        print(f"   Image: {image_file}")
        print(f"   Overlay: {overlay_file}")
        
        try:
            # Load JSON data
            with open(json_file, 'r', encoding='utf-8') as f:
                textract_data = json.load(f)
            
            # Load and copy image
            image = Image.open(image_file).convert('RGB')
            draw = ImageDraw.Draw(image)
            W, H = image.size
            print(f"   📐 Image dimensions: {W}x{H}")
            
            # Process blocks
            blocks = textract_data.get('Blocks', [])
            target_types = ['LINE', 'WORD', 'TABLE', 'CELL', 'KEY_VALUE_SET', 'SELECTION_ELEMENT', 'SIGNATURE']
            
            overlay_count = 0
            block_counts = {}
            
            for block in blocks:
                block_type = block.get('BlockType', '')
                
                if block_type in target_types:
                    # Count this block type
                    block_counts[block_type] = block_counts.get(block_type, 0) + 1
                    
                    # Get geometry
                    geometry = block.get('Geometry', {})
                    bbox = geometry.get('BoundingBox', {})
                    
                    if bbox:
                        # Convert normalized coordinates to pixel coordinates
                        left = bbox.get('Left', 0)
                        top = bbox.get('Top', 0)
                        width = bbox.get('Width', 0)
                        height = bbox.get('Height', 0)
                        
                        x1 = int(left * W)
                        y1 = int(top * H)
                        x2 = int((left + width) * W)
                        y2 = int((top + height) * H)
                        
                        # Ensure coordinates are within bounds
                        x1 = max(0, min(x1, W-1))
                        y1 = max(0, min(y1, H-1))
                        x2 = max(x1+1, min(x2, W))
                        y2 = max(y1+1, min(y2, H))
                        
                        color = colors.get(block_type, (128, 128, 128))
                        
                        # Draw bounding box
                        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                        
                        # Add text label for LINE blocks
                        if block_type == 'LINE':
                            text_content = block.get('Text', '')
                            if text_content:
                                preview = text_content[:25] + "..." if len(text_content) > 25 else text_content
                                confidence = block.get('Confidence', 0)
                                
                                label = f"{preview}"
                                if confidence > 0:
                                    label += f" ({confidence:.0f}%)"
                                
                                # Draw label with background
                                label_y = max(5, y1 - 20)
                                try:
                                    draw.rectangle([(x1, label_y), (x1 + len(label) * 6, label_y + 15)], 
                                                 fill=(0, 0, 0, 200))
                                    draw.text((x1 + 2, label_y + 1), label, fill=color)
                                except:
                                    draw.text((x1, label_y), label, fill=color)
                        
                        overlay_count += 1
            
            # Save overlay image
            image.save(overlay_file)
            
            print(f"   ✅ Overlay complete: {overlay_count} elements")
            print(f"   📊 Block types:")
            for block_type, count in block_counts.items():
                color_name = {
                    'LINE': '🔴', 'WORD': '🟢', 'TABLE': '🟣', 
                    'CELL': '🔵', 'KEY_VALUE_SET': '🟡'
                }.get(block_type, '⚪')
                print(f"      {color_name} {block_type}: {count}")
            
            overlay_results[page_key] = {
                'overlay_file': overlay_file,
                'overlay_count': overlay_count,
                'block_counts': block_counts
            }
            
        except Exception as e:
            print(f"   ❌ Error creating overlay: {str(e)}")
            overlay_results[page_key] = {
                'error': str(e)
            }
    
    return overlay_results


def run_aws_textract_s3_pipeline(output_dir="temp"):
    """
    Run the complete AWS Textract S3 pipeline
    """
    print("🎯 AWS Textract S3 Pipeline")
    print("=" * 50)
    
    # Step 1: Run AWS Textract via subprocess on S3 images
    aws_results = run_aws_textract_subprocess(output_dir)
    
    # Step 2: Create overlays from JSON files
    if aws_results:
        overlay_results = create_textract_overlays_from_s3_json(aws_results, output_dir)
        
        # Step 3: Summary
        print(f"\n🎉 AWS Textract S3 Pipeline Complete!")
        print("-" * 30)
        
        successful_aws = len([r for r in aws_results.values() if r.get('success')])
        successful_overlays = len([r for r in overlay_results.values() if 'overlay_file' in r])
        
        print(f"📊 AWS Textract calls: {successful_aws}/2 successful")
        print(f"🎨 Overlay images: {successful_overlays} created")
        print(f"📂 Check {output_dir}/ for:")
        print(f"   - textract_s3_page_*.json (AWS S3 responses)")
        print(f"   - temp_aws_textract_overlay_*_page_*.png (overlaid images)")
        
        return {'aws_results': aws_results, 'overlay_results': overlay_results}
    else:
        print("❌ No AWS results to process")
        return {}


if __name__ == "__main__":
    # Run S3 pipeline by default
    run_aws_textract_s3_pipeline()