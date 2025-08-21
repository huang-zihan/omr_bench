import csv
import zipfile
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm

def extract_xml_from_mxl(mxl_path):
    """Extract the main XML file content from the .mxl zip package."""
    with zipfile.ZipFile(mxl_path, 'r') as zf:
        # Find container.xml to determine the main file path
        with zf.open('META-INF/container.xml') as container_file:
            container = ET.parse(container_file)
            rootfile = container.find('.//rootfile')
            if rootfile is None:
                raise ValueError("container.xml missing rootfile element")
            xml_path = rootfile.attrib['full-path']
        
        # Read the main XML file content
        with zf.open(xml_path) as xml_file:
            return xml_file.read().decode('utf-8')

def count_tracks_in_musicxml(xml_content):
    """Count the number of parts in MusicXML."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"XML parsing error: {str(e)}")
    
    # Handle XML namespaces
    if root.tag.startswith('{'):
        namespace = re.match(r'{(.*)}', root.tag).group(1)
        ns = {'ns': namespace}
        ns_key = 'ns:'
    else:
        ns = {}
        ns_key = ''
    
    # Search for the part list in score-partwise/score-timewise
    if root.tag.endswith('score-partwise') or root.tag.endswith('score-timewise'):
        part_list = root.find(f'.//{ns_key}part-list', ns)
        if part_list is not None:
            score_parts = part_list.findall(f'{ns_key}score-part', ns)
            if score_parts:
                return len(score_parts)
    
    # If part-list does not exist, directly count part elements
    parts = root.findall(f'.//{ns_key}part', ns)
    if parts:
        return len(parts)
    
    # If none, return 0
    return 0

def process_csv(csv_path, output_csv_path):
    """Process the CSV file and add a track_cnt column."""
    # Read the original data
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Add new column track_cnt
        fieldnames = reader.fieldnames + ['track_cnt'] if 'track_cnt' not in reader.fieldnames else reader.fieldnames
        
        for row in reader:
            rows.append(row)
    
    # Process each row and add track_cnt
    for row in tqdm(rows):
        mxl_path = row['mxl_file']
        
        # print("mxl_path:", mxl_path)
        
        # Skip empty paths
        if not mxl_path or not isinstance(mxl_path, str) or mxl_path.strip() == "":
            row['track_cnt'] = "Empty path"
            continue
            
        # Check if the file exists
        if not os.path.exists(mxl_path):
            row['track_cnt'] = "File not found"
            continue
            
        try:
            xml_content = extract_xml_from_mxl(mxl_path)
            track_count = count_tracks_in_musicxml(xml_content)
            row['track_cnt'] = str(track_count)
        except Exception as e:
            row['track_cnt'] = f"Error: {str(e)}"
    
    # Write to a new CSV file
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return output_csv_path

if __name__ == "__main__":
    # Configure input and output CSV file paths
    input_csv = "music_scores.csv"
    output_csv = "music_scores_with_track_count.csv"
    
    # Process and save the new file
    result_file = process_csv(input_csv, output_csv)
    print(f"Processing complete! Results saved to: {result_file}")
    print(f"New column added: track_cnt")