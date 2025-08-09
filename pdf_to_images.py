import fitz  # PyMuPDF
import io
import os
import time
from PIL import Image

def pdf_to_image_bytes(pdf_path, dpi=150):
    """
    Convert PDF pages to image bytes representation.
    
    Args:
        pdf_path (str): Path to the PDF file
        dpi (int): Resolution for image conversion (default: 150)
    
    Returns:
        list: List of tuples (page_number, image_bytes)
    """
    doc = fitz.open(pdf_path)
    image_data_list = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Create transformation matrix for desired DPI
        zoom = dpi / 72.0  # 72 is default DPI
        mat = fitz.Matrix(zoom, zoom)
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to bytes
        img_bytes = pix.tobytes("png")
        
        # Store page number (1-based) and bytes
        image_data_list.append((page_num + 1, img_bytes))
        
        print(f"Page {page_num + 1}: {len(img_bytes)} bytes")
    
    doc.close()
    return image_data_list

def save_images_to_disk(image_data_list, output_dir="temp"):
    """
    Save image bytes to disk as PNG files with timestamp
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = int(time.time())
    
    for page_num, img_bytes in image_data_list:
        filename = f"{output_dir}/temp_{timestamp}_page_{page_num}.png"
        with open(filename, 'wb') as f:
            f.write(img_bytes)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    # Process the PDF
    pdf_file = "brochour_with_prices-merged.pdf"
    
    print(f"Processing: {pdf_file}")
    images = pdf_to_image_bytes(pdf_file)
    
    print(f"\nTotal pages processed: {len(images)}")
    
    # Optionally save to disk
    save_choice = input("Save images to disk? (y/n): ").lower()
    if save_choice == 'y':
        save_images_to_disk(images)