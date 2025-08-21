import os
import argparse
import pandas as pd
from pdf2image import convert_from_path
import cv2
import numpy as np
from PIL import Image
import json

def split_staff_lines(image, min_line_height=100, padding=20):
    """
    Split music score image into individual staff lines
    :param image: PIL Image object
    :param min_line_height: minimum line height (pixels)
    :param padding: extra space added above and below each line (pixels)
    :return: list of split line images (PIL Image objects)
    """
    # Convert PIL image to OpenCV format
    img_np = np.array(image)
    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Binarization processing (inverted color: black background with white lines)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Horizontal projection: calculate number of black pixels per row
    horizontal_projection = np.sum(binary, axis=1)
    
    # Detect line regions
    peaks = []
    in_peak = False
    start_index = 0
    
    # Dynamic threshold (0.05% of maximum projection value)
    threshold = np.max(horizontal_projection) * 0.0005
    
    for i, value in enumerate(horizontal_projection):
        if value > threshold:
            if not in_peak:
                in_peak = True
                start_index = i
        elif in_peak:
            if (i - start_index) > min_line_height:
                # Add padding
                start = max(0, start_index - padding)
                end = min(len(horizontal_projection), i + padding)
                peaks.append((start, end))
            in_peak = False
    
    # Process last region
    if in_peak and (len(horizontal_projection) - start_index) > min_line_height:
        start = max(0, start_index - padding)
        end = min(len(horizontal_projection), len(horizontal_projection))
        peaks.append((start, end))
    
    # Split image
    staff_lines = []
    for start, end in peaks:
        line_img = img_np[start:end, :]
        staff_lines.append(Image.fromarray(line_img))
    
    return staff_lines

def process_pdf(pdf_path, sample_id, output_dir, max_pages):
    """
    Process PDF file and save split line images
    :return: metadata information
    """
    # Create output directory for current sample
    sample_output_dir = os.path.join(output_dir, sample_id)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    metadata = {
        "sample_id": sample_id,
        "pdf_path": pdf_path,
        "pages_processed": 0,
        "total_lines": 0,
        "line_images": []
    }
    
    # try:
    # Convert to images (only process first max_pages pages)
    pages = convert_from_path(pdf_path, first_page=1, last_page=max_pages)
    metadata["pages_processed"] = len(pages)

    for page_num, page in enumerate(pages):
        
        line_path = os.path.join(sample_output_dir, f"page{page_num}.jpg")
        page.save(line_path, "JPEG")
        
        # Split current page into individual lines
        staff_lines = split_staff_lines(page)
        metadata["total_lines"] += len(staff_lines)
        
        for line_num, line_img in enumerate(staff_lines):
            # Save line image to file
            line_filename = f"page{page_num+1}_line{line_num+1}.jpg"
            line_path = os.path.join(sample_output_dir, line_filename)
            line_img.save(line_path, "JPEG")
            
            
            # Record metadata
            metadata["line_images"].append({
                "page": page_num + 1,
                "line": line_num + 1,
                "file_path": line_path,
                "relative_path": os.path.join(sample_id, line_filename)
            })
    
    # Save metadata
    meta_path = os.path.join(sample_output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

    # except Exception as e:
    #     print(f"Error processing {pdf_path}: {str(e)}")
    #     return None

def main(csv_path, output_dir, max_pages, max_samples=None):
    """Main processing function"""
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # If maximum number of samples is set, truncate data
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
    
    total = len(df)
    print(f"Starting staff line splitting for {total} samples...")
    
    # Process each music score sample
    for i, row in enumerate(df.itertuples(), 1):
        sample_id = row.sample_id
        pdf_path = row.pdf_file
        
        print(f"\n[{i}/{total}] Splitting staff lines: {sample_id} | File: {pdf_path}")
        
        # Process PDF and split lines
        metadata = process_pdf(pdf_path, sample_id, output_dir, max_pages)
        
        if metadata:
            print(f"Split {metadata['pages_processed']} pages into {metadata['total_lines']} staff lines")
    
    print("\nSplitting complete!")
    print(f"All results saved to: {output_dir}")

if __name__ == "__main__":
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Split music scores into staff lines')
    parser.add_argument('--csv_path', type=str, required=True, 
                        help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for split lines')
    parser.add_argument('--max_pages', type=int, default=10, 
                        help='Maximum pages per score to process')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run main program
    main(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        max_samples=args.max_samples
    )