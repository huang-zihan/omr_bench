#!/bin/bash

conda env create -f environment.yml
conda activate omr

# download
URLS=(
    "https://zenodo.org/records/15571083/files/data.tar.gz?download=1"
    "https://zenodo.org/records/15571083/files/metadata.tar.gz?download=1"
    "https://zenodo.org/records/15571083/files/mid.tar.gz?download=1"
    "https://zenodo.org/records/15571083/files/mxl.tar.gz?download=1"
    "https://zenodo.org/records/15571083/files/pdf.tar.gz?download=1"
    "https://zenodo.org/records/15571083/files/PDMX.csv?download=1"
    "https://zenodo.org/records/15571083/files/subset_paths.tar.gz?download=1"
)

FILES=(
    "data.tar.gz"
    "metadata.tar.gz"
    "mid.tar.gz"
    "mxl.tar.gz"
    "pdf.tar.gz"
    "PDMX.csv"
    "subset_paths.tar.gz"
)

echo "Starting file download process..."

# Loop through all URLs and download files
for i in "${!URLS[@]}"; do
    url="${URLS[$i]}"
    filename="${FILES[$i]}"
    
    echo "Downloading: $filename"
    # Download file with proper filename using -O option
    wget -O "$filename" "$url"
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "✓ Download completed: $filename"
        
        # Extract if file is a tar.gz archive
        if [[ "$filename" == *.tar.gz ]]; then
            echo "Extracting: $filename"
            # Extract tar.gz file
            tar -xzf "$filename"
            if [ $? -eq 0 ]; then
                echo "✓ Extraction completed: $filename"
                # Optional: remove the archive after extraction
                # rm "$filename"
            else
                echo "✗ Extraction failed: $filename"
            fi
        fi
    else
        echo "✗ Download failed: $filename"
    fi
    
    echo "----------------------------------------"
done

echo "All files processed successfully!"

# extract and annotate the track number metadata
python extract_track.py

# filtering only multi-track data (not compulsary)
python filter_multi_track.py

# read PDF music score files, transform them into fine-grained music score images
./score_split.sh