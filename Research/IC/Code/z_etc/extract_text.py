"""
Script to extract text from global_rmse_comparison_resize.png using OCR
and save the parsed data to a CSV file.
Uses EasyOCR which doesn't require separate Tesseract installation.
"""

import easyocr
from PIL import Image
import pandas as pd
import re
import os
import numpy as np

# Increase the image size limit to handle large images
Image.MAX_IMAGE_PIXELS = None  # Remove the limit entirely

def extract_text_from_image(image_path, output_file):
    """
    Extract text from an image using OCR.
    Rotates the image 90 degrees and splits it into chunks for better processing.
    Saves text progressively after each chunk.
    
    Args:
        image_path: Path to the image file
        output_file: Path to the output text file
        
    Returns:
        Extracted text as a string
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            print(f"Created directory: {temp_dir}")
        
        # Clear or create the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("")
        print(f"Output file created: {output_file}")
        
        print("Loading image...")
        # Open the image
        img = Image.open(image_path)
        print(f"Original image size: {img.size} (width x height)")
        
        # Rotate the image 90 degrees counter-clockwise (positive = counter-clockwise)
        print("Rotating image 90 degrees...")
        img_rotated = img.rotate(-90, expand=True)
        print(f"Rotated image size: {img_rotated.size}")
        
        # Save the rotated image to temp folder
        rotated_filename = os.path.basename(image_path).replace('.png', '_rotated.png')
        rotated_path = os.path.join(temp_dir, rotated_filename)
        print(f"Saving rotated image to {rotated_path}...")
        img_rotated.save(rotated_path, optimize=False)
        print(f"Rotated image saved successfully")
        
        # Split image into chunks
        width, height = img_rotated.size
        chunk_height = 4000  # Height of each chunk in pixels
        num_chunks = (height + chunk_height - 1) // chunk_height  # Ceiling division
        
        print(f"\nSplitting image into {num_chunks} chunks of {chunk_height}px height...")
        
        print("Initializing EasyOCR reader (this may take a moment on first run)...")
        # Initialize the OCR reader for English
        reader = easyocr.Reader(['en'], gpu=False)
        
        total_text_elements = 0
        
        for i in range(num_chunks):
            # Calculate chunk boundaries
            top = i * chunk_height
            bottom = min((i + 1) * chunk_height, height)
            
            print(f"\nProcessing chunk {i+1}/{num_chunks} (rows {top} to {bottom})...")
            
            # Crop the chunk
            chunk = img_rotated.crop((0, top, width, bottom))
            
            # Save chunk to temp folder
            chunk_filename = os.path.basename(image_path).replace('.png', f'_chunk_{i+1}.png')
            chunk_path = os.path.join(temp_dir, chunk_filename)
            #chunk.save(chunk_path)
            print(f"  Chunk saved to {chunk_path}")
            
            # Convert to numpy array
            chunk_array = np.array(chunk)
            print(f"  Chunk shape: {chunk_array.shape}")
            
            # Perform OCR on this chunk with improved parameters
            print(f"  Running OCR on chunk {i+1}...")
            result = reader.readtext(
                chunk_array,
                detail=1,             
                paragraph=False,       
                min_size=1,  
                text_threshold=0.4,
                low_text=0.2,
                link_threshold=0.2,
                canvas_size=chunk_height,
                mag_ratio=1.5 
            )
            
            print(f"  Found {len(result)} text elements in chunk {i+1}")
            total_text_elements += len(result)
            
            # Extract text from results and save immediately
            if result:
                chunk_text = '\n'.join([detection[1] for detection in result])
                
                # Append to output file after each chunk
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(chunk_text)
                    f.write('\n')  # Add newline after chunk
                
                print(f"  ✓ Text from chunk {i+1} saved to {output_file}")
                
                # Show sample detections
                print(f"  Sample text from chunk {i+1}:")
                for j, detection in enumerate(result[:5]):  # Show first 5
                    print(f"    - '{detection[1]}' (confidence: {detection[2]:.2f})")
            else:
                print(f"  No text found in chunk {i+1}")
        
        # Read the final combined text
        with open(output_file, 'r', encoding='utf-8') as f:
            extracted_text = f.read()
        
        print(f"\n{'='*50}")
        print(f"Total text elements found: {total_text_elements}")
        print(f"All text saved to: {output_file}")
        print(f"{'='*50}")
        
        return extracted_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_extracted_text(text):
    """
    Parse the extracted text and organize it into structured data.
    
    Args:
        text: Raw text extracted from the image
        
    Returns:
        List of dictionaries containing parsed data
    """
    if not text:
        return []
    
    # Split text into lines
    lines = text.strip().split('\n')
    
    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Parse the data - adjust this logic based on the actual structure of your image
    parsed_data = []
    
    for i, line in enumerate(lines):
        # Try to extract numbers (adjust regex pattern as needed)
        numbers = re.findall(r'-?\d+\.?\d*', line)
        
        parsed_data.append({
            'line_number': i + 1,
            'raw_text': line,
            'extracted_numbers': ', '.join(numbers) if numbers else ''
        })
    
    return parsed_data

def save_to_csv(data, output_path):
    """
    Save parsed data to a CSV file.
    
    Args:
        data: List of dictionaries containing parsed data
        output_path: Path for the output CSV file
    """
    if not data:
        print("No data to save.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Data saved to {output_path}")
    print(f"\nPreview of extracted data:")
    print(df.head(10))
    print(f"\nTotal rows: {len(df)}")

def main():
    """Main function to orchestrate the text extraction and save to txt."""
    
    # Create temp directory if it doesn't exist
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"Created directory: {temp_dir}")
    
    # Define paths
    image_path = "global_rmse_comparison_resize.png"
    output_text = os.path.join(temp_dir, "extracted_text.txt")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    print(f"Extracting text from {image_path}...")
    print(f"Saving text progressively to {output_text}")
    print("="*50)
    
    # Extract text from image (saves progressively)
    extracted_text = extract_text_from_image(image_path, output_text)
    
    if extracted_text:
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE!")
        print("="*50)
        print(f"All extracted text saved to: {output_text}")
        print(f"Total characters: {len(extracted_text)}")
        print(f"Total lines: {extracted_text.count(chr(10)) + 1}")
        
        # Show preview of extracted text
        print("\n" + "="*50)
        print("Preview of extracted text (first 1000 characters):")
        print("="*50)
        print(extracted_text[:1000])
        if len(extracted_text) > 1000:
            print("...")
            print(f"[{len(extracted_text) - 1000} more characters]")
    else:
        print("Failed to extract text from image.")

if __name__ == "__main__":
    main()
