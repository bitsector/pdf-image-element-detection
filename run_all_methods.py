"""
Main runner script to test all element detection methods on PDF pages
"""

import os
import json
import time
from pdf_to_images import pdf_to_image_bytes, save_images_to_disk

# Import all detection methods
from method_paddleocr import detect_elements_with_paddleocr, visualize_paddleocr_results
from method_unstructured import detect_elements_with_unstructured, visualize_unstructured_results
from method_opencv import detect_elements_with_opencv, visualize_opencv_results, detect_text_regions_opencv
from method_yolo import detect_elements_with_yolo, visualize_yolo_results, detect_custom_document_elements
from method_layoutlm import detect_elements_with_layoutlm, visualize_layoutlm_results
from method_aws_textract import detect_elements_with_textract, visualize_textract_results, check_aws_credentials
from method_chatgpt import detect_elements_with_chatgpt, visualize_chatgpt_results


def run_all_methods_on_pdf(pdf_path, output_dir="temp"):
    """
    Run all element detection methods on every page of a PDF
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save images and results
    
    Returns:
        dict: Comprehensive results from all methods
    """
    print(f"Processing PDF: {pdf_path}")
    print("="*60)
    
    # Step 1: Convert PDF to images
    print("Step 1: Converting PDF pages to images...")
    images = pdf_to_image_bytes(pdf_path)
    
    if not images:
        print("Error: Could not extract images from PDF")
        return {}
    
    # Save images to disk
    save_images_to_disk(images, output_dir)
    
    # Step 2: Run all detection methods on each page
    all_results = {}
    
    # Get timestamp for consistent file naming
    timestamp = int(time.time())
    
    methods_to_run = [
        ("PaddleOCR", detect_elements_with_paddleocr, visualize_paddleocr_results),
        ("Unstructured", detect_elements_with_unstructured, visualize_unstructured_results),
        ("OpenCV", detect_elements_with_opencv, visualize_opencv_results),
        ("YOLO", detect_elements_with_yolo, visualize_yolo_results),
        ("LayoutLM", detect_elements_with_layoutlm, visualize_layoutlm_results),
        ("AWS_Textract", detect_elements_with_textract, visualize_textract_results),
        # ("ChatGPT_Vision", detect_elements_with_chatgpt, visualize_chatgpt_results)
    ]
    
    for page_num, img_bytes in images:
        page_key = f"page_{page_num}"
        all_results[page_key] = {}
        
        # Image file path
        image_path = f"{output_dir}/temp_{timestamp}_page_{page_num}.png"
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        print(f"\nProcessing Page {page_num}...")
        print("-" * 40)
        
        # Run each method
        for method_name, detection_func, visualization_func in methods_to_run:
            print(f"Running {method_name}...")
            
            try:
                # Note: AWS Textract handling removed (commented out)
                
                # Run detection
                start_time = time.time()
                results = detection_func(image_path, save_results=True)
                end_time = time.time()
                
                # Run visualization
                if results and visualization_func:
                    visualization_func(image_path, results)
                
                # Store results
                all_results[page_key][method_name] = {
                    'status': 'success',
                    'processing_time': round(end_time - start_time, 2),
                    'element_count': len(results),
                    'elements': results
                }
                
                print(f"  ✓ {method_name}: {len(results)} elements detected in {end_time - start_time:.2f}s")
                
            except Exception as e:
                print(f"  ✗ {method_name}: Error - {str(e)}")
                all_results[page_key][method_name] = {
                    'status': 'error',
                    'error': str(e),
                    'elements': []
                }
        
        # Check if OpenCV is enabled in methods_to_run
        opencv_enabled = any("OpenCV" in method[0] for method in methods_to_run)
        if opencv_enabled:
            # Run additional OpenCV method for text regions
            print("Running OpenCV Text Regions...")
            try:
                text_results = detect_text_regions_opencv(image_path, save_results=True)
                all_results[page_key]["OpenCV_TextRegions"] = {
                    'status': 'success',
                    'element_count': len(text_results),
                    'elements': text_results
                }
                print(f"  ✓ OpenCV Text Regions: {len(text_results)} regions detected")
            except Exception as e:
                print(f"  ✗ OpenCV Text Regions: Error - {str(e)}")
                all_results[page_key]["OpenCV_TextRegions"] = {
                    'status': 'error',
                    'error': str(e),
                    'elements': []
                }
        
        # Check if YOLO is enabled in methods_to_run
        yolo_enabled = any("YOLO" in method[0] for method in methods_to_run)
        if yolo_enabled:
            # Run additional YOLO method for custom document elements
            print("Running YOLO Custom Document Detection...")
            try:
                custom_results = detect_custom_document_elements(image_path, save_results=True)
                all_results[page_key]["YOLO_Custom"] = {
                    'status': 'success',
                    'element_count': len(custom_results),
                    'elements': custom_results
                }
                print(f"  ✓ YOLO Custom: {len(custom_results)} elements detected")
            except Exception as e:
                print(f"  ✗ YOLO Custom: Error - {str(e)}")
                all_results[page_key]["YOLO_Custom"] = {
                    'status': 'error',
                    'error': str(e),
                    'elements': []
                }
    
    # Step 3: Save comprehensive results
    results_path = f"{output_dir}/all_methods_results_{timestamp}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "="*60)
    print(f"All results saved to: {results_path}")
    
    # Step 4: Generate summary report
    generate_summary_report(all_results, f"{output_dir}/summary_report_{timestamp}.txt")
    
    return all_results


def generate_summary_report(all_results, report_path):
    """
    Generate a summary report of all detection methods
    
    Args:
        all_results (dict): Results from all methods
        report_path (str): Path to save the report
    """
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PDF ELEMENT DETECTION SUMMARY REPORT\\n")
        f.write("="*50 + "\\n\\n")
        
        total_pages = len(all_results)
        f.write(f"Total Pages Processed: {total_pages}\\n\\n")
        
        # Method performance summary
        method_summary = {}
        
        for page_key, page_results in all_results.items():
            f.write(f"Page: {page_key.upper()}\\n")
            f.write("-" * 30 + "\\n")
            
            for method_name, method_results in page_results.items():
                if method_name not in method_summary:
                    method_summary[method_name] = {
                        'success_count': 0,
                        'error_count': 0,
                        'total_elements': 0,
                        'avg_processing_time': 0,
                        'processing_times': []
                    }
                
                status = method_results.get('status', 'unknown')
                element_count = method_results.get('element_count', 0)
                processing_time = method_results.get('processing_time', 0)
                
                if status == 'success':
                    method_summary[method_name]['success_count'] += 1
                    method_summary[method_name]['total_elements'] += element_count
                    if processing_time > 0:
                        method_summary[method_name]['processing_times'].append(processing_time)
                elif status == 'error':
                    method_summary[method_name]['error_count'] += 1
                
                f.write(f"  {method_name}: {status}")
                if status == 'success':
                    f.write(f" - {element_count} elements")
                    if processing_time > 0:
                        f.write(f" ({processing_time}s)")
                elif status == 'error':
                    f.write(f" - {method_results.get('error', 'Unknown error')}")
                f.write("\\n")
            
            f.write("\\n")
        
        # Overall method performance
        f.write("OVERALL METHOD PERFORMANCE\\n")
        f.write("="*30 + "\\n")
        
        for method_name, summary in method_summary.items():
            success_rate = summary['success_count'] / total_pages * 100 if total_pages > 0 else 0
            avg_elements = summary['total_elements'] / summary['success_count'] if summary['success_count'] > 0 else 0
            avg_time = sum(summary['processing_times']) / len(summary['processing_times']) if summary['processing_times'] else 0
            
            f.write(f"\\n{method_name}:\\n")
            f.write(f"  Success Rate: {success_rate:.1f}% ({summary['success_count']}/{total_pages})\\n")
            f.write(f"  Avg Elements per Page: {avg_elements:.1f}\\n")
            if avg_time > 0:
                f.write(f"  Avg Processing Time: {avg_time:.2f}s\\n")
            if summary['error_count'] > 0:
                f.write(f"  Errors: {summary['error_count']}\\n")
        
        # Recommendations
        f.write("\\n" + "="*50 + "\\n")
        f.write("RECOMMENDATIONS\\n")
        f.write("="*50 + "\\n")
        
        # Find best performing method
        best_method = max(method_summary.items(), 
                         key=lambda x: (x[1]['success_count'], x[1]['total_elements']))
        
        f.write(f"\\nBest Overall Method: {best_method[0]}\\n")
        f.write(f"  - Highest success rate and element detection count\\n")
        
        # Find fastest method
        fastest_methods = [(name, sum(data['processing_times'])/len(data['processing_times'])) 
                          for name, data in method_summary.items() 
                          if data['processing_times']]
        
        if fastest_methods:
            fastest_method = min(fastest_methods, key=lambda x: x[1])
            f.write(f"\\nFastest Method: {fastest_method[0]} ({fastest_method[1]:.2f}s avg)\\n")
        
        f.write("\\n" + "="*50 + "\\n")
    
    print(f"Summary report saved to: {report_path}")


def main():
    """
    Main function to run all detection methods
    """
    # PDF file to process
    pdf_file = "brochour_with_prices-merged.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file not found: {pdf_file}")
        print("Available files:")
        for file in os.listdir("."):
            if file.endswith(".pdf"):
                print(f"  - {file}")
        return
    
    # Check dependencies
    print("Checking dependencies...")
    missing_deps = []
    
    try:
        import paddleocr
        print("✓ PaddleOCR available")
    except ImportError:
        missing_deps.append("paddleocr")
    
    try:
        import unstructured
        print("✓ Unstructured available")
    except ImportError:
        missing_deps.append("unstructured[local-inference]")
    
    try:
        import cv2
        print("✓ OpenCV available")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import ultralytics
        print("✓ Ultralytics YOLO available")
    except ImportError:
        missing_deps.append("ultralytics")
    
    try:
        import transformers
        print("✓ Transformers (LayoutLM) available")
    except ImportError:
        missing_deps.append("transformers torch")
    
    try:
        import openai
        print("✓ OpenAI (ChatGPT Vision) available")
    except ImportError:
        missing_deps.append("openai python-dotenv")
    
    # AWS Textract commented out
    # try:
    #     import boto3
    #     print("✓ boto3 (AWS Textract) available")
    # except ImportError:
    #     print("⚠ boto3 not available (AWS Textract will be skipped)")
    
    if missing_deps:
        print(f"\\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        print("Or run: pip install -r requirements.txt")
        return
    
    # Run all methods
    print(f"\\nStarting comprehensive element detection on {pdf_file}...")
    results = run_all_methods_on_pdf(pdf_file)
    
    if results:
        print("\\n🎉 All methods completed successfully!")
        print("Check the temp/ directory for:")
        print("  - Individual page images")
        print("  - Detection results (JSON files)")
        print("  - Visualization images")
        print("  - Summary report")
    else:
        print("\\n❌ Processing failed")


if __name__ == "__main__":
    main()