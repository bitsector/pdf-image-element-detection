<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# **PDF Multi-Page Element Extraction and OCR: Libraries and Solutions**

## **AWS Textract - Document Analysis and Block Detection**

**AWS Textract** provides comprehensive document analysis capabilities that can detect and extract elements from multi-page PDFs. The service returns structured data through **Block objects** that represent different elements:[^1][^2]

- **PAGE blocks**: Represent individual pages
- **LINE blocks**: Contain detected text lines
- **WORD blocks**: Individual words within lines
- **TABLE blocks**: Table structures with cell information
- **CELL blocks**: Individual table cells
- **KEY_VALUE_SET blocks**: Form field pairs

**Key capabilities:**

- Detects rectangular regions with bounding box coordinates (X, Y, Width, Height)[^2][^1]
- Provides confidence scores for detection accuracy
- Supports both synchronous and asynchronous processing
- Can handle multi-page documents through pagination

**Usage example:**

```bash
aws textract analyze-document --document '{"S3Object":{"Bucket":"bucket","Name":"document.pdf"}}' --feature-types TABLES FORMS --no-cli-pager
```


## **Python Libraries for PDF Element Detection and OCR**

### **1. PaddleOCR - Comprehensive Document Analysis**

**PaddleOCR** offers advanced layout detection and table recognition capabilities:[^3][^4][^5]

- **PP-StructureV3**: Complete document structure analysis pipeline
- **Layout Detection**: Identifies 23 document elements including tables, text, images, titles
- **Table Recognition V2**: Advanced table structure detection and cell extraction
- **Bounding box coordinates**: Provides precise element locations

**Installation and usage:**

```bash
python3 -m pip install paddleocr
```

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(layout_detection_model_name="PP-DocLayout-M")
output = pipeline.predict("document.pdf")

for res in output:
    res.print()  # Shows detected elements with coordinates
    res.save_to_json("output/")
```


### **2. Unstructured Library - Document Parsing**

**Unstructured** provides automatic document segmentation and element extraction:[^6][^7][^8]

- Automatically detects document elements (titles, paragraphs, tables, lists)
- Uses **Detectron2** for layout detection
- Supports multiple document formats including PDFs
- Returns structured element objects with metadata

**Installation and usage:**

```bash
python3 -m pip install "unstructured[local-inference]"
python3 -m pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"
```

```python
from unstructured.partition.auto import partition

elements = partition("multi_page_document.pdf")

for element in elements:
    print(f"Type: {type(element).__name__}")
    print(f"Text: {element.text}")
    if hasattr(element, 'metadata'):
        print(f"Coordinates: {element.metadata.coordinates}")
```


### **3. OpenCV + Traditional Computer Vision**

**OpenCV** can detect rectangular regions through contour detection and shape analysis:[^9][^10][^11]

```python
import cv2
import numpy as np

def detect_rectangles(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold and find contours
    ret, thresh = cv2.threshold(gray, 50, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for cnt in contours:
        # Approximate contour to polygon
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Check if shape has 4 vertices (rectangle)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            rectangles.append((x, y, w, h))
            
    return rectangles
```


### **4. YOLO Object Detection for Custom Elements**

**YOLO** can be trained to detect specific document elements like offer blocks:[^12][^13][^14]

```python
from ultralytics import YOLO

# Load pre-trained model or train custom model
model = YOLO("yolo11n.pt")

# Detect objects in document pages
results = model("document_page.jpg")

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy.tolist()
        confidence = box.conf.item()
        class_id = box.cls.item()
        
        # Extract detected region
        cropped_region = img[int(y1):int(y2), int(x1):int(x2)]
```


## **Multimodal Document AI Models**

### **LayoutLM Series - Layout-Aware Document Understanding**

**LayoutLM** models combine text, layout, and visual information for document analysis:[^15][^16][^17]

- Uses 2D position embeddings for spatial understanding
- Processes text and layout jointly in a single framework
- Excels at form understanding, receipt processing, and document classification

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Process document image with OCR
image = Image.open("document.jpg")
encoding = processor(image, return_tensors="pt")
outputs = model(**encoding)
```


## **Complete PDF Processing Pipeline**

Here's a comprehensive approach combining multiple tools:

```python
import fitz  # PyMuPDF
import cv2
import numpy as np
from paddleocr import PaddleOCR
import pandas as pd

def process_pdf_multipage(pdf_path):
    doc = fitz.open(pdf_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    all_elements = []
    
    for page_num in range(len(doc)):
        # Convert page to image
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom
        img_data = pix.tobytes("png")
        
        # Convert to numpy array for OpenCV
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect elements using PaddleOCR structure analysis
        result = ocr.ocr(img, cls=True)
        
        for line in result:
            bbox = line  # Bounding box coordinates
            text = line[^1]  # Extracted text
            confidence = line[^1][^1]  # Confidence score
            
            # Calculate bounding box dimensions
            x_coords = [point for point in bbox]
            y_coords = [point[^1] for point in bbox]
            x, y = min(x_coords), min(y_coords)
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            all_elements.append({
                'page': page_num + 1,
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'text': text,
                'confidence': confidence
            })
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(all_elements)
    df.to_csv('extracted_elements.csv', index=False)
    
    return df

# Usage
results = process_pdf_multipage("flyer.pdf")
```


## **Recommendations for Your Upwork Project**

Based on your requirements for extracting offer blocks from flyer PDFs:

1. **Primary Solution**: Use **PaddleOCR** with its layout detection capabilities - it provides excellent element segmentation and OCR in one package[^4][^3]
2. **Cloud Alternative**: **AWS Textract** for production workloads with high accuracy requirements[^1][^2]
3. **Custom Training**: Train a **YOLO model** specifically for your offer block detection if you have consistent flyer layouts[^13][^12]
4. **Hybrid Approach**: Combine **Unstructured** for initial document parsing with **EasyOCR** for text extraction from detected regions[^18][^6]

The key advantage of these solutions is they provide both **element detection with bounding boxes** and **OCR capabilities** in integrated pipelines, exactly matching your project requirements for extracting offer blocks with coordinates and text content from multi-page flyer PDFs.

<div style="text-align: center">⁂</div>

[^1]: https://docs.aws.amazon.com/textract/latest/dg/how-it-works-document-layout.html

[^2]: https://docs.aws.amazon.com/textract/latest/dg/API_Block.html

[^3]: https://huggingface.co/PaddlePaddle/SLANet_plus

[^4]: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/table_structure_recognition.html

[^5]: https://huggingface.co/PaddlePaddle/PP-DocLayout-M

[^6]: https://github.com/Unstructured-IO/unstructured

[^7]: https://pypi.org/project/unstructured/0.5.0/

[^8]: https://unstructured.io/blog/how-to-process-pdf-in-python

[^9]: https://stackoverflow.com/questions/61166180/detect-rectangles-in-opencv-4-2-0-using-python-3-7

[^10]: https://www.tutorialspoint.com/how-to-detect-a-rectangle-and-square-in-an-image-using-opencv-python

[^11]: https://github.com/RanaShankani/OpenCV-Camera-Rectangle-Detector

[^12]: https://www.robotexchange.io/t/how-to-make-a-yolo-model-with-orientated-bounding-boxes-to-find-the-center-and-angle-of-the-bounding-box/3322

[^13]: https://github.com/parasgulati8/Object-Detection

[^14]: https://www.v7labs.com/blog/yolo-object-detection

[^15]: https://www.klippa.com/en/blog/information/layoutlm-explained/

[^16]: https://huggingface.co/docs/transformers/en/model_doc/layoutlm

[^17]: https://arxiv.org/pdf/1912.13318.pdf

[^18]: https://stackoverflow.com/questions/73956962/finding-white-rectangular-areas-in-given-image-to-increase-ocr-results

[^19]: https://www.geeksforgeeks.org/python/python-reading-contents-of-pdf-using-ocr-optical-character-recognition/

[^20]: https://www.ripublication.com/acst18/acstv11n1_05.pdf

[^21]: https://github.com/ocrmypdf/OCRmyPDF

[^22]: https://www.scitepress.org/papers/2009/18067/18067.pdf

[^23]: https://ocrmypdf.readthedocs.io/en/latest/introduction.html

[^24]: https://www.fig.io/manual/aws/textract/analyze-document

[^25]: https://www.ayadata.ai/bounding-boxes-in-computer-vision-uses-best-practices-for-labeling-and-more/

[^26]: https://unstract.com/blog/evaluating-python-pdf-to-text-libraries/

[^27]: https://awscli.amazonaws.com/v2/documentation/api/latest/reference/textract/detect-document-text.html

[^28]: https://stackoverflow.com/questions/54593401/rectangular-region-detection-in-opencv-python

[^29]: https://ploomber.io/blog/pdf-ocr/

[^30]: https://awscli.amazonaws.com/v2/documentation/api/2.0.34/reference/textract/detect-document-text.html

[^31]: http://d2l.ai/chapter_computer-vision/bounding-box.html

[^32]: https://www.metriccoders.com/post/a-guide-to-pdf-extraction-libraries-in-python

[^33]: https://fig.io/manual/aws/textract/get-document-analysis

[^34]: https://www.sciencedirect.com/topics/computer-science/rectangle-feature

[^35]: https://www.reddit.com/r/Python/comments/1awc0hh/extracting_information_text_tables_layouts_from/

[^36]: https://docs.aws.amazon.com/textract/latest/dg/analyzing-document-text.html

[^37]: https://www.geeksforgeeks.org/python/python-opencv-cv2-rectangle-method/

[^38]: https://iphylo.blogspot.com/2023/08/document-layout-analysis.html

[^39]: https://docs.ultralytics.com/tasks/obb/

[^40]: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout?view=doc-intel-4.0.0

[^41]: https://github.com/tobiasctrl/Opencv-Rectangle-and-circle-detection

[^42]: https://blog.roboflow.com/how-to-draw-a-bounding-box-in-python/

[^43]: https://www.youtube.com/watch?v=Wl11eloYVm8

[^44]: https://docs.ultralytics.com/tasks/detect/

[^45]: https://github.com/Noba1anc3/Document-Analysis-Recognition/blob/master/LayoutLM: Pre-training of Text and Layout for Document Image Understanding.md

[^46]: https://learnopencv.com/contour-detection-using-opencv-python-c

[^47]: https://encord.com/blog/yolo-object-detection-guide/

[^48]: https://techcommunity.microsoft.com/blog/azurearchitectureblog/enhancing-document-extraction-with-azure-ai-document-intelligence-and-langchain-/4187387

[^49]: https://www.youtube.com/watch?v=s6k5OI7BQGg

[^50]: https://zilliz.com/ai-faq/how-do-i-use-langchain-for-automatic-document-processing

[^51]: https://www.youtube.com/watch?v=Dvm5Yu68jDY

[^52]: https://docs.unstructured.io

[^53]: https://blog.langchain.com/open-source-extraction-service/

[^54]: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/layout_detection.html

[^55]: https://www.tigerdata.com/blog/parsing-all-the-data-with-open-source-tools-unstructured-and-pgai

[^56]: https://python.langchain.com/docs/tutorials/extraction/

[^57]: https://github.com/PaddlePaddle/PaddleOCR/issues/9509

[^58]: https://docs.unstructured.io/open-source/introduction/quick-start

[^59]: https://python.langchain.com/docs/integrations/document_transformers/google_docai/

[^60]: https://github.com/PaddlePaddle/PaddleOCR/blob/main/README_en.md

[^61]: https://saeedesmaili.com/demystifying-text-data-with-the-unstructured-python-library/

[^62]: https://blog.roboflow.com/multimodal-vision-models/

[^63]: https://ironsoftware.com/csharp/ocr/blog/compare-to-other-components/paddle-ocr-vs-tesseract/

[^64]: https://wseas.com/journals/isa/2023/a465109-013(2023).pdf

[^65]: https://www.roots.ai/blog/segmenting-documents-with-llms-and-multimodal-document-ai-part-1

[^66]: https://ai.gopubby.com/should-you-use-textract-for-ocr-and-data-extraction-the-best-textract-alternative-d8c2a4675d62

[^67]: https://www.koyeb.com/blog/best-multimodal-vision-models-in-2025

[^68]: https://unstract.com/blog/best-pdf-ocr-software/

[^69]: https://python.langchain.com/docs/integrations/document_loaders/unstructured_pdfloader/

[^70]: https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/pre-intro

[^71]: https://news.ycombinator.com/item?id=32053525

[^72]: https://sreekartammana.hashnode.dev/text-detection-using-easyocr-python

[^73]: https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models

[^74]: https://pragmile.com/ocr-ranking-2025-comparison-of-the-best-text-recognition-and-document-structure-software/

[^75]: https://muegenai.com/docs/gen-ai/gen-ai-sub-topic/chapter-13-ocr-fundamentals/table-detection-and-structure-recognition/

[^76]: https://encord.com/blog/top-multimodal-models/

[^77]: https://www.reddit.com/r/LanguageTechnology/comments/wsolqa/open_source_equivalent_of_aws_textract/

[^78]: https://www.muegenai.com/docs/datascience/computer_vision_applications/ocr_fundamentals/table_recognition

[^79]: https://cloud.google.com/vision

[^80]: https://nanonets.com/blog/identifying-the-best-ocr-api/

