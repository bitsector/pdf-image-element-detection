"""
Test script to verify that all dependencies are properly installed
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False

def main():
    print("Testing PDF Element Detection Setup")
    print("=" * 40)
    
    # Core dependencies
    print("\\nCore Dependencies:")
    core_deps = [
        ("fitz", "PyMuPDF"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("pandas", "pandas")
    ]
    
    core_success = all(test_import(module, name) for module, name in core_deps)
    
    # Method-specific dependencies
    print("\\nMethod-Specific Dependencies:")
    method_deps = [
        ("paddleocr", "PaddleOCR"),
        ("easyocr", "EasyOCR"), 
        ("unstructured", "Unstructured"),
        ("ultralytics", "Ultralytics YOLO"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("boto3", "AWS Boto3")
    ]
    
    method_success = []
    for module, name in method_deps:
        success = test_import(module, name)
        method_success.append(success)
    
    # Optional dependencies
    print("\\nOptional Dependencies:")
    optional_deps = [
        ("pytesseract", "Tesseract OCR"),
        ("matplotlib", "Matplotlib"),
        ("scikit-image", "scikit-image")
    ]
    
    for module, name in optional_deps:
        test_import(module, name)
    
    # Summary
    print("\\n" + "=" * 40)
    print("SETUP SUMMARY")
    print("=" * 40)
    
    if core_success:
        print("✓ Core dependencies: ALL GOOD")
    else:
        print("✗ Core dependencies: MISSING SOME")
    
    successful_methods = sum(method_success)
    total_methods = len(method_success)
    print(f"📊 Method dependencies: {successful_methods}/{total_methods} available")
    
    print("\\nAvailable Methods:")
    method_names = ["PaddleOCR", "EasyOCR", "Unstructured", "YOLO", "LayoutLM", "AWS Textract"]
    for i, (success, name) in enumerate(zip(method_success, method_names)):
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    if successful_methods >= 3:
        print("\\n🎉 Setup looks good! You can run multiple detection methods.")
        print("\\nNext steps:")
        print("1. Run: python run_all_methods.py")
        print("2. Check results in the temp/ directory")
    else:
        print("\\n⚠️  Consider installing more dependencies for better results.")
        print("Run: pip install -r requirements.txt")
    
    return core_success and successful_methods >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)