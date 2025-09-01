import csv
import os
import subprocess
from pathlib import Path

def convert_mxl_to_abc(csv_files, output_dir):
    """
    Convert all MXL files listed in CSV files to ABC format
    
    Args:
        csv_files: List of CSV file paths
        output_dir: Directory to save converted ABC files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"Processing CSV file: {csv_file}")
        
        # Open and read the CSV file
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Process each row in the CSV
            for row in reader:
                mxl_file = row['mxl_file']
                
                # Skip if MXL file path is empty or file doesn't exist
                if not mxl_file or not os.path.exists(mxl_file):
                    print(f"Warning: MXL file does not exist - {mxl_file}")
                    continue
                
                # Generate output filename (same name with .abc extension)
                filename = Path(mxl_file).stem
                output_path = os.path.join(output_dir, f"{filename}.abc")
                
                # Execute conversion command
                try:
                    cmd = [
                        'python', 'xml2abc.py',
                        mxl_file,
                        '-o', output_dir
                    ]
                    
                    print(f"Converting: {mxl_file} -> {output_path}")
                    # Run the conversion command
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    
                    if result.returncode == 0:
                        print(f"Successfully converted: {mxl_file}")
                    else:
                        print(f"Conversion failed: {mxl_file}")
                        print(f"Error: {result.stderr}")
                        
                except subprocess.CalledProcessError as e:
                    print(f"Conversion error: {mxl_file}")
                    print(f"Error details: {e.stderr}")
                except Exception as e:
                    print(f"Unexpected error: {mxl_file}")
                    print(f"Error details: {str(e)}")

if __name__ == "__main__":
    # Configuration - update these paths as needed
    csv_files = [
        'dataset/mini_train.csv',
        'dataset/mini_test.csv'  # Adjust filenames based on your actual data
    ]
    output_dir = 'gt_abc'  # Output directory for ABC files
    
    # Execute conversion
    convert_mxl_to_abc(csv_files, output_dir)
    
    print("All files converted successfully!")