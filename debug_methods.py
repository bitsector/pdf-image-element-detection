"""
Debug script to test individual methods and see their outputs
"""

import os
import cv2
from method_opencv import detect_elements_with_opencv, visualize_opencv_results
from method_paddleocr import detect_elements_with_paddleocr, visualize_paddleocr_results
from method_layoutlm import detect_elements_with_layoutlm, visualize_layoutlm_results


def debug_method(method_name, detection_func, visualization_func, image_path):
    """Debug a specific method"""
    print(f"\n{'='*50}")
    print(f"DEBUGGING: {method_name}")
    print(f"Image: {image_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return
    
    # Run detection
    results = detection_func(image_path, save_results=True)
    
    print(f"Results: {len(results)} elements detected")
    
    # Print first few results
    for i, result in enumerate(results[:5]):  # Show first 5
        print(f"Element {i+1}:")
        print(f"  Type: {result.get('type', 'unknown')}")
        print(f"  BBox: {result.get('bbox', 'no bbox')}")
        if 'text' in result:
            text_preview = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
            print(f"  Text: '{text_preview}'")
        print()
    
    if len(results) > 5:
        print(f"... and {len(results) - 5} more elements")
    
    # Run visualization
    if visualization_func:
        viz_path = visualization_func(image_path, results)
        if viz_path:
            print(f"Visualization saved: {viz_path}")
    
    return results


def main():
    """Debug all methods"""
    # First, check if we need to create test images
    temp_dir = "temp"
    test_images = []
    
    if os.path.exists(temp_dir):
        png_files = [f for f in os.listdir(temp_dir) if f.endswith('.png') and 'page' in f and 'temp_' in f]
        if png_files:
            png_files.sort()
            test_images = [os.path.join(temp_dir, f) for f in png_files[:2]]  # First 2 pages
    
    # If no images found, create them from PDF
    if not test_images:
        print("No test images found in temp/ directory")
        print("Creating images from PDF...")
        
        # Import and run PDF conversion
        try:
            from pdf_to_images import pdf_to_image_bytes, save_images_to_disk
            pdf_file = "brochour_with_prices-merged.pdf"
            
            if os.path.exists(pdf_file):
                # Convert PDF to images
                images = pdf_to_image_bytes(pdf_file)
                if images:
                    save_images_to_disk(images, temp_dir)
                    
                    # Find the newly created images
                    png_files = [f for f in os.listdir(temp_dir) if f.endswith('.png') and 'page' in f]
                    png_files.sort()
                    test_images = [os.path.join(temp_dir, f) for f in png_files[:2]]
                    print(f"✅ Created {len(test_images)} test images")
                else:
                    print("❌ Failed to extract images from PDF")
                    return
            else:
                print(f"❌ PDF file not found: {pdf_file}")
                return
        except Exception as e:
            print(f"❌ Error creating test images: {e}")
            return
    
    methods_to_debug = [
        ("OpenCV", detect_elements_with_opencv, visualize_opencv_results),
        ("PaddleOCR", detect_elements_with_paddleocr, visualize_paddleocr_results),
        ("LayoutLM", detect_elements_with_layoutlm, visualize_layoutlm_results),
    ]
    
    for image_path in test_images:
        print(f"\n🔍 TESTING IMAGE: {image_path}")
        
        # Check image exists
        if not os.path.exists(image_path):
            print(f"❌ Image not found, skipping: {image_path}")
            continue
            
        # Load and show image info
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            print(f"📐 Image dimensions: {w}x{h}")
        
        for method_name, detection_func, visualization_func in methods_to_debug:
            try:
                debug_method(method_name, detection_func, visualization_func, image_path)
            except Exception as e:
                print(f"❌ ERROR in {method_name}: {e}")
    
    print(f"\n{'='*50}")
    print("DEBUG COMPLETE")
    print("Check the temp/ directory for visualization images")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()