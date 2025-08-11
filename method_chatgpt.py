"""
Method 7: ChatGPT Vision API - AI-Powered Brochure Element Extraction
Uses OpenAI GPT Vision models to extract brochure elements with pricing information

Supported Models:
- gpt-5-mini: Good accuracy, cost-effective
  Input: $0.00025 per 1K tokens, Output: $0.00200 per 1K tokens
  
- gpt-5-nano: Basic accuracy, ultra-low cost
  Input: $0.00005 per 1K tokens, Output: $0.00040 per 1K tokens
  (5x cheaper input, 5x cheaper output vs gpt-5-mini)

Note: GPT-5 pricing as of August 2025. Aggressive pricing to compete with other providers.
"""

import os
import json
import base64
import time
from typing import List, Dict, Any
from PIL import Image
import requests
from dotenv import load_dotenv


def load_api_key():
    """Load OpenAI API key from .env file"""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    return api_key


def show_pricing_comparison():
    """Display pricing comparison between different GPT models"""
    models = {
        "gpt-5-mini": {"input": 0.00025, "output": 0.002, "name": "GPT-5-mini (Cost-effective)"},
        "gpt-5-nano": {"input": 0.00005, "output": 0.0004, "name": "GPT-5-nano (Ultra-low cost)"}
    }
    
    print("💰 OpenAI GPT Vision Model Pricing Comparison (per 1K tokens)")
    print("=" * 65)
    print(f"{'Model':<15} {'Input Rate':<12} {'Output Rate':<13} {'Description':<25}")
    print("-" * 65)
    
    for model, info in models.items():
        print(f"{model:<15} ${info['input']:<11.5f} ${info['output']:<12.5f} {info['name']:<25}")
    
    print("\n📊 Cost Examples (for 10K input + 2K output tokens):")
    print("-" * 50)
    
    example_input = 10000
    example_output = 2000
    
    for model, info in models.items():
        cost = (example_input / 1000) * info['input'] + (example_output / 1000) * info['output']
        print(f"{model:<15} ${cost:.6f}")
        
        # Show savings vs gpt-5-mini (if using nano)
        if model == "gpt-5-nano":
            mini_cost = (example_input / 1000) * 0.00025 + (example_output / 1000) * 0.002
            savings = mini_cost - cost
            savings_percent = (savings / mini_cost) * 100
            print(f"{'':15} (${savings:.6f} saved, {savings_percent:.1f}% cheaper)")
    
    print("\n💡 Recommendations:")
    print("- gpt-5-mini: Best balance of cost and performance (recommended)")
    print("- gpt-5-nano: Ultra-low cost for simple extraction tasks")
    print("\nNote: GPT-5 pricing as of August 2025. Aggressive pricing to compete with other providers.")


def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def create_brochure_extraction_prompt() -> str:
    """Create the prompt for brochure element extraction"""
    return """
    You are an expert document analyzer specializing in brochure and marketing material analysis. 
    
    Analyze this image which is a page from a converted PDF containing brochures, advertisements, or marketing materials.
    
    Your task is to identify and extract EACH INDIVIDUAL brochure element, product listing, or promotional item separately. 
    Do NOT provide a bulk text summary - instead, identify each distinct brochure/product element.
    
    For each element you find, extract:
    1. **Element Type**: (e.g., "product_brochure", "service_ad", "price_listing", "promotional_banner", "contact_info")
    2. **Title/Heading**: Main heading or product name
    3. **Description**: Brief description or key features
    4. **Price Information**: Any prices, costs, or monetary values mentioned
    5. **Contact Details**: Phone numbers, addresses, emails if present
    6. **Additional Text**: Any other relevant text content
    7. **Estimated Position**: Describe roughly where on the page this element appears (e.g., "top-left", "center", "bottom-right")
    
    Format your response as a JSON array where each object represents one brochure element:
    
    ```json
    [
        {
            "element_id": 1,
            "element_type": "product_brochure",
            "title": "Product Name Here",
            "description": "Product description and features",
            "price_info": "$99.99" or "Call for pricing" or null if no price,
            "contact_details": "Phone: 555-1234, Email: info@company.com" or null,
            "additional_text": "Any other relevant text",
            "estimated_position": "top-left quadrant",
            "confidence": "high/medium/low"
        }
    ]
    ```
    
    Important guidelines:
    - Extract each brochure/product element as a separate JSON object
    - Be thorough - don't miss any distinct elements
    - If you see pricing information, extract it exactly as shown
    - If no clear price is visible for an element, set price_info to null
    - Estimate the position on the page for each element
    - Rate your confidence in the extraction for each element
    
    Return ONLY the JSON array, no additional text or formatting.
    """


def extract_elements_with_chatgpt(image_path: str, output_dir: str = "temp", save_results: bool = True, model: str = "gpt-5-mini") -> List[Dict[str, Any]]:
    """
    Extract brochure elements using ChatGPT Vision API
    
    Args:
        image_path (str): Path to the image file
        output_dir (str): Directory to save results
        save_results (bool): Whether to save results to JSON
        model (str): Model to use - "gpt-5-mini" or "gpt-5-nano"
    
    Returns:
        List[Dict]: List of extracted brochure elements
    """
    print(f"🤖 Analyzing image with ChatGPT Vision: {image_path}")
    
    try:
        # Load API key
        api_key = load_api_key()
        print("✅ OpenAI API key loaded")
        
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Get image info
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"📐 Image dimensions: {width}x{height}")
        
        # Encode image to base64
        print("📄 Encoding image to base64...")
        base64_image = encode_image_to_base64(image_path)
        print(f"✅ Image encoded ({len(base64_image)} characters)")
        
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Model-specific pricing info
        model_pricing = {
            "gpt-5-mini": {"input": 0.00025, "output": 0.002, "name": "GPT-5-mini (Cost-effective)"},
            "gpt-5-nano": {"input": 0.00005, "output": 0.0004, "name": "GPT-5-nano (Ultra-low cost)"}
        }
        
        # Validate model
        if model not in model_pricing:
            print(f"⚠️  Unknown model '{model}', defaulting to gpt-5-mini")
            model = "gpt-5-mini"
        
        print(f"🤖 Using model: {model_pricing[model]['name']}")
        print(f"💰 Pricing: ${model_pricing[model]['input']:.5f}/1K input, ${model_pricing[model]['output']:.5f}/1K output")
        
        # Create the request payload
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": create_brochure_extraction_prompt()
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"  # High detail for better analysis
                            }
                        }
                    ]
                }
            ],
            "max_completion_tokens": 4000  # GPT-5 uses max_completion_tokens instead of max_tokens
            # Note: GPT-5 only supports default temperature (1), custom values not allowed
        }
        
        print("📡 Sending request to OpenAI API...")
        start_time = time.time()
        
        # Make the API request
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120  # 2 minute timeout
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ API response received in {processing_time:.2f} seconds")
        
        # Check if request was successful
        if response.status_code != 200:
            error_msg = f"API request failed with status {response.status_code}: {response.text}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
        
        # Parse the response
        response_data = response.json()
        
        # Extract usage information
        usage = response_data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        
        print(f"📊 Token usage:")
        print(f"   Prompt tokens: {prompt_tokens}")
        print(f"   Completion tokens: {completion_tokens}")
        print(f"   Total tokens: {total_tokens}")
        
        # Calculate estimated cost using model-specific pricing
        input_rate = model_pricing[model]['input']
        output_rate = model_pricing[model]['output']
        input_cost = (prompt_tokens / 1000) * input_rate
        output_cost = (completion_tokens / 1000) * output_rate
        total_cost = input_cost + output_cost
        
        print(f"💰 Estimated cost: ${total_cost:.6f}")
        print(f"   Input cost: ${input_cost:.6f} (${input_rate:.5f}/1K × {prompt_tokens:,} tokens)")
        print(f"   Output cost: ${output_cost:.6f} (${output_rate:.5f}/1K × {completion_tokens:,} tokens)")
        print(f"   Model: {model_pricing[model]['name']}")
        
        # Cost comparison with gpt-5-mini (if using nano)
        if model == "gpt-5-nano":
            mini_input_cost = (prompt_tokens / 1000) * 0.00025
            mini_output_cost = (completion_tokens / 1000) * 0.002
            mini_total_cost = mini_input_cost + mini_output_cost
            savings = mini_total_cost - total_cost
            savings_percent = (savings / mini_total_cost) * 100 if mini_total_cost > 0 else 0
            print(f"   💸 Savings vs GPT-5-mini: ${savings:.6f} ({savings_percent:.1f}% cheaper)")
        
        # Extract the response content
        message_content = response_data['choices'][0]['message']['content'].strip()
        
        # Try to parse the JSON response
        try:
            # Clean the response (remove any markdown formatting)
            if message_content.startswith('```json'):
                message_content = message_content.replace('```json', '').replace('```', '').strip()
            elif message_content.startswith('```'):
                message_content = message_content.replace('```', '').strip()
            
            extracted_elements = json.loads(message_content)
            
            if not isinstance(extracted_elements, list):
                raise ValueError("Response is not a JSON array")
            
            print(f"🎯 Successfully extracted {len(extracted_elements)} brochure elements")
            
            # Add metadata to each element
            for i, element in enumerate(extracted_elements):
                element.update({
                    'method': f'ChatGPT_Vision_{model}',
                    'model_used': model,
                    'source_image': image_path,
                    'processing_time': processing_time,
                    'api_cost': total_cost,
                    'tokens_used': total_tokens,
                    'extraction_timestamp': time.time()
                })
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON response: {e}")
            print(f"Raw response: {message_content[:500]}...")
            
            # Return a fallback result
            extracted_elements = [{
                'element_id': 1,
                'element_type': 'raw_text',
                'title': 'ChatGPT Analysis',
                'description': message_content,
                'price_info': None,
                'contact_details': None,
                'additional_text': None,
                'estimated_position': 'full_page',
                'confidence': 'medium',
                'method': f'ChatGPT_Vision_{model}',
                'model_used': model,
                'source_image': image_path,
                'processing_time': processing_time,
                'api_cost': total_cost,
                'tokens_used': total_tokens,
                'extraction_timestamp': time.time(),
                'json_parse_error': str(e)
            }]
        
        # Save results if requested
        if save_results and extracted_elements:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate unique filename
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            results_file = os.path.join(output_dir, f'chatgpt_results_{base_filename}.json')
            
            # Prepare results with metadata
            results_data = {
                'metadata': {
                    'source_image': image_path,
                    'processing_time': processing_time,
                    'total_cost': total_cost,
                    'tokens_used': total_tokens,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'model_used': model,
                    'extraction_timestamp': time.time(),
                    'element_count': len(extracted_elements)
                },
                'elements': extracted_elements
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Results saved to: {results_file}")
        
        return extracted_elements
        
    except Exception as e:
        print(f"❌ Error in ChatGPT extraction: {str(e)}")
        return []


def detect_elements_with_chatgpt_batch(output_dir: str = "temp", save_results: bool = True, model: str = "gpt-5-mini") -> Dict[str, Any]:
    """
    Run ChatGPT Vision on the hardcoded PNG images (batch processing)
    
    Args:
        output_dir (str): Directory containing images and to save results
        save_results (bool): Whether to save results
        model (str): Model to use - "gpt-5-mini" or "gpt-5-nano"
    
    Returns:
        Dict: Results from all processed images
    """
    print("🚀 ChatGPT Vision Batch Processing")
    print("=" * 50)
    
    # Hardcoded image files (matching AWS Textract method)
    image_files = [
        "temp_1754761991_page_1.png",
        "temp_1754761991_page_2.png"
    ]
    
    # Look for images in temp directory
    temp_image_paths = []
    if os.path.exists(output_dir):
        for img_file in image_files:
            img_path = os.path.join(output_dir, img_file)
            if os.path.exists(img_path):
                temp_image_paths.append(img_path)
            else:
                # Try to find similar files
                png_files = [f for f in os.listdir(output_dir) if f.endswith('.png') and 'page_' in f]
                matching = [f for f in png_files if f'page_{img_file.split("_page_")[1].split(".")[0]}' in f]
                if matching:
                    temp_image_paths.append(os.path.join(output_dir, matching[0]))
    
    if not temp_image_paths:
        print("❌ No matching image files found in temp directory")
        print(f"Looking for: {image_files}")
        return {}
    
    print(f"📁 Found {len(temp_image_paths)} images to process")
    for path in temp_image_paths:
        print(f"   - {path}")
    
    batch_results = {}
    total_cost = 0.0
    total_tokens = 0
    total_elements = 0
    
    for i, image_path in enumerate(temp_image_paths, 1):
        page_key = f'page_{i}'
        print(f"\n📄 Processing {page_key}: {os.path.basename(image_path)}")
        print("-" * 40)
        
        try:
            elements = extract_elements_with_chatgpt(image_path, output_dir, save_results, model)
            
            if elements:
                # Extract cost and token info from first element (they all have the same values)
                page_cost = elements[0].get('api_cost', 0)
                page_tokens = elements[0].get('tokens_used', 0)
                
                total_cost += page_cost
                total_tokens += page_tokens
                total_elements += len(elements)
                
                batch_results[page_key] = {
                    'success': True,
                    'image_path': image_path,
                    'element_count': len(elements),
                    'elements': elements,
                    'api_cost': page_cost,
                    'tokens_used': page_tokens,
                    'processing_time': elements[0].get('processing_time', 0)
                }
                
                print(f"✅ {page_key}: {len(elements)} elements extracted")
            else:
                batch_results[page_key] = {
                    'success': False,
                    'image_path': image_path,
                    'element_count': 0,
                    'elements': [],
                    'error': 'No elements extracted'
                }
                print(f"❌ {page_key}: No elements extracted")
                
        except Exception as e:
            batch_results[page_key] = {
                'success': False,
                'image_path': image_path,
                'element_count': 0,
                'elements': [],
                'error': str(e)
            }
            print(f"❌ {page_key}: Error - {str(e)}")
    
    # Save batch summary
    if save_results and batch_results:
        summary_file = os.path.join(output_dir, 'chatgpt_batch_summary.json')
        summary_data = {
            'batch_metadata': {
                'total_pages_processed': len(batch_results),
                'successful_pages': len([r for r in batch_results.values() if r.get('success')]),
                'total_elements_extracted': total_elements,
                'total_api_cost': total_cost,
                'total_tokens_used': total_tokens,
                'model_used': 'gpt-4o',
                'batch_timestamp': time.time()
            },
            'page_results': batch_results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Batch summary saved: {summary_file}")
    
    # Print final summary
    print(f"\n🎉 ChatGPT Batch Processing Complete!")
    print("=" * 50)
    successful_pages = len([r for r in batch_results.values() if r.get('success')])
    print(f"📊 Pages processed: {successful_pages}/{len(batch_results)} successful")
    print(f"🎯 Total elements extracted: {total_elements}")
    print(f"🪙 Total tokens used: {total_tokens:,}")
    print(f"💰 Total estimated cost: ${total_cost:.6f}")
    
    # Calculate average cost per page and per element
    if successful_pages > 0:
        avg_cost_per_page = total_cost / successful_pages
        print(f"📄 Average cost per page: ${avg_cost_per_page:.6f}")
    
    if total_elements > 0:
        avg_cost_per_element = total_cost / total_elements
        print(f"🎯 Average cost per element: ${avg_cost_per_element:.6f}")
    
    # Show model used and potential savings
    if batch_results:
        first_result = next(iter(batch_results.values()))
        if first_result.get('success') and first_result.get('elements'):
            model_used = first_result['elements'][0].get('model_used', 'unknown')
            print(f"🤖 Model used: {model_used}")
            
            # Calculate potential savings vs gpt-5-mini (if using nano)
            if model_used == "gpt-5-nano":
                mini_cost = (total_tokens / 1000) * (0.00025 + 0.002) / 2  # rough estimate
                if total_cost < mini_cost:
                    savings_percent = ((mini_cost - total_cost) / mini_cost) * 100
                    print(f"💸 Estimated savings vs GPT-5-mini: ~{savings_percent:.1f}% cheaper")
    
    print(f"📂 Check {output_dir}/ for:")
    print(f"   - chatgpt_results_*.json (per-page results)")
    print(f"   - chatgpt_batch_summary.json (batch summary)")
    
    return batch_results


# Wrapper functions for compatibility with run_all_methods.py
def detect_elements_with_chatgpt(image_path: str, save_results: bool = True, model: str = "gpt-5-mini") -> List[Dict[str, Any]]:
    """
    Compatibility wrapper for run_all_methods.py
    Runs ChatGPT analysis on a single image
    
    Args:
        image_path (str): Path to the image file
        save_results (bool): Whether to save results to JSON
        model (str): Model to use - "gpt-5-mini" or "gpt-5-nano"
    
    Returns:
        List[Dict]: List of extracted brochure elements
    """
    print("⚠️  ChatGPT Vision wrapper called for single image")
    elements = extract_elements_with_chatgpt(image_path, "temp", save_results, model)
    return elements


def visualize_chatgpt_results(image_path: str, results: List[Dict], output_path: str = None):
    """
    Compatibility wrapper - ChatGPT provides text analysis, no visual overlay needed
    """
    if not results:
        print("No ChatGPT results to visualize")
        return None
    
    print(f"ℹ️  ChatGPT Vision analysis complete: {len(results)} elements extracted")
    print("   (Results are text-based, no visual overlay created)")
    
    # Print summary of extracted elements
    for i, element in enumerate(results[:3]):  # Show first 3 elements
        print(f"   {i+1}. {element.get('element_type', 'unknown')}: {element.get('title', 'No title')}")
    
    if len(results) > 3:
        print(f"   ... and {len(results) - 3} more elements")
    
    return None


if __name__ == "__main__":
    # Run batch processing on hardcoded images with different models
    import sys
    
    # Show help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("🤖 ChatGPT Vision API for Brochure Element Extraction")
        print("\nUsage:")
        print("  python method_chatgpt.py [model]")
        print("  python method_chatgpt.py --pricing")
        print("\nAvailable models:")
        print("  gpt-5-mini    - Good accuracy, cost-effective (default)")
        print("  gpt-5-nano    - Basic accuracy, ultra-low cost")
        print("\nExamples:")
        print("  python method_chatgpt.py")
        print("  python method_chatgpt.py gpt-5-mini")
        print("  python method_chatgpt.py gpt-5-nano")
        print("  python method_chatgpt.py --pricing")
        exit(0)
    
    # Show pricing comparison if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['--pricing', '-p', 'pricing']:
        show_pricing_comparison()
        exit(0)
    
    # Allow model selection via command line argument
    model_to_use = "gpt-5-mini"  # Default to cost-effective option
    if len(sys.argv) > 1:
        model_to_use = sys.argv[1]
        
    print(f"🚀 Running ChatGPT batch processing with model: {model_to_use}")
    print("💡 Use 'python method_chatgpt.py --pricing' to see cost comparison")
    print("=" * 60)
    
    detect_elements_with_chatgpt_batch(model=model_to_use)