
import os
import pandas as pd
import tempfile
import time
import json
import argparse
import re
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Qwen2_5_VLForConditionalGeneration

# Configuration parameters
CSV_PATH = "filtered_music_scores.csv"  # CSV file containing music score information
OUTPUT_DIR = "qwen_results"             # Output directory
SPLIT_OUTPUT_DIR = "split_lines"        # Directory for split line images
MAX_SAMPLES = None                      # Maximum number of samples to process

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load Qwen-VL model and tokenizer"""
    global model, tokenizer
    if model is None or tokenizer is None:
        model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()
        
        # Configure generation parameters
        model.generation_config = GenerationConfig.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        model.generation_config.max_new_tokens = 8192
        model.generation_config.do_sample = False
        
        print("Model loaded successfully.")

def load_split_images(sample_id, split_dir):
    """
    Load split line images
    :return: (list of line images, metadata)
    """
    # Load metadata
    meta_path = os.path.join(split_dir, sample_id, "metadata.json")
    if not os.path.exists(meta_path):
        print(f"Metadata not found for sample: {sample_id}")
        return None, None
    
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    
    # Load all line images
    line_images = []
    for img_info in metadata["line_images"]:
        img_path = img_info["file_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            line_images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            continue
    
    return line_images, metadata

def parse_abc_voices(abc_text):
    """
    Parse ABC notation into a dictionary of voices and header
    """
    # Split into lines
    lines = abc_text.split('\n')
    
    # Extract header (everything before the first voice)
    header_lines = []
    voice_lines = {}
    current_voice = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a voice line
        voice_match = re.match(r'^\[V:([^\]]+)\]', line)
        if voice_match:
            current_voice = voice_match.group(1)
            if current_voice not in voice_lines:
                voice_lines[current_voice] = []
            # Add the voice line itself
            voice_lines[current_voice].append(line)
        elif current_voice:
            # This is a continuation of the current voice
            voice_lines[current_voice].append(line)
        else:
            # This is part of the header
            header_lines.append(line)
    
    # Join header and voice lines
    header = '\n'.join(header_lines)
    voices = {voice: '\n'.join(lines) for voice, lines in voice_lines.items()}
    
    return header, voices

def merge_abc_parts(abc_parts):
    """
    Merge multiple ABC notation parts into a single ABC notation
    """
    if not abc_parts:
        return ""
    
    # Parse the first part (contains header and initial voices)
    header, all_voices = parse_abc_voices(abc_parts[0])
    
    # Process subsequent parts
    for part in abc_parts[1:]:
        _, part_voices = parse_abc_voices(part)
        
        # Merge each voice from this part
        for voice, content in part_voices.items():
            if voice in all_voices:
                # Append to existing voice
                all_voices[voice] += " " + content
            else:
                # Create new voice
                all_voices[voice] = f"[V:{voice}] {content}"
    
    # Reconstruct the full ABC notation
    result = header
    for voice_content in all_voices.values():
        result += "\n" + voice_content
    
    return result

def qwen_omr_request(line_images, previous_context=None):
    """
    Perform music score recognition using Qwen-VL model with context from previous generations
    :param line_images: list of PIL.Image objects (in order)
    :param previous_context: ABC notation from previous generations (for context)
    :return: abc format music score
    """
    # System prompt
    system_prompt = (
        "You are a professional Optical Music Recognition (OMR) system. Convert music score images into precise ABC notation.\n\n"
    )
    
    all_abc_results = []
    
    # Process images line by line
    for i, img in enumerate(line_images):
        # Build context message if available
        context_message = []
        if previous_context and i == 0:
            context_message = [{
                "type": "text",
                "text": f"Here is the previous ABC notation for context:\n\n{previous_context}\n\n"
            }]
        
        # Choose different prompts based on whether it's the first line
        if i == 0 and not previous_context:
            user_prompt = [{
                "type": "text",
                "text": "Convert the staff lines to ABC notation. Include the header information (X, T, M, K, etc.) and the notes. Output ONLY the ABC code. Use [V:VoiceName] notation to mark each voice track."
            }]
        else:
            user_prompt = [{
                "type": "text",
                "text": "Convert the following staff lines to ABC notation, which are the continuation of the ABC notation above. Output ONLY the ABC code without repeating the header."
            }]

        # Build message list with context
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": context_message + user_prompt + [{"image": img}]
            }
        ]

        try:
            # Preprocess messages using tokenizer.apply_chat_template
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            # Convert input to the format required by the model
            model_inputs = tokenizer(
                [text_input],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True
            ).to(model.device)
            
            input_length = model_inputs.attention_mask.sum(dim=1).tolist()[0]

            # Generate response
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1000,
                do_sample=True
            )
            
            # Decode generation results
            response = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)

            # Extract abc content (remove possible extra text)
            if "**" in response:
                response = response.split("**", 1)[0]
            
            # Clean response, remove possible additional explanatory text
            response = response.strip()
            if response.startswith("ABC Notation:"):
                response = response[len("ABC Notation:"):].strip()
            
            all_abc_results.append(response)

        except Exception as e:
            print(f"Model inference failed for line {i}: {str(e)}")
            all_abc_results.append("")  # Add empty string as placeholder
    
    # Merge all ABC results for this set of line images
    if not all_abc_results:
        return None
    
    # For multiple line images in one generation, just concatenate them
    combined_abc = "\n".join(all_abc_results)
    return combined_abc

def save_results(sample_id, abc_code, metadata):
    """Save recognition results and metadata"""
    # Save abc code
    abc_path = os.path.join(OUTPUT_DIR, f"{sample_id}.abc")
    with open(abc_path, "w", encoding="utf-8") as f:
        f.write(abc_code)
    
    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, f"{sample_id}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

def main(csv_path, split_dir, max_samples=None):
    """Main processing function"""
    # Load model
    load_model()
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # If maximum number of samples is set, truncate data
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)
    
    total = len(df)
    print(f"Starting OMR processing for {total} samples...")
    
    # Process each music score sample
    for i, row in enumerate(df.itertuples(), 1):
        sample_id = row.sample_id
        
        print(f"\n[{i}/{total}] Processing sample: {sample_id}")
        
        # Step 1: Load split line images
        start_time = time.time()
        line_images, split_metadata = load_split_images(sample_id, split_dir)
        loading_time = time.time() - start_time
        
        if not line_images or not split_metadata:
            print(f"Failed to load split images for sample: {sample_id}")
            continue
        
        print(f"Loaded {len(line_images)} staff lines in {loading_time:.2f}s")
        
        # Step 2: Perform recognition using Qwen-VL with context
        # We need to process images in batches according to the original split
        # The split_metadata should contain information about how images are grouped
        
        # Get the grouping information from metadata
        # This assumes the metadata has a structure that tells us how to group images
        # For example, it might have a "groups" field with lists of image indices
        groups = split_metadata.get("groups", [[i] for i in range(len(line_images))])
        
        all_abc_parts = []
        previous_context = None
        
        for group_idx, image_indices in enumerate(groups):
            print(f"Processing group {group_idx+1}/{len(groups)} with {len(image_indices)} images")
            
            # Extract the images for this group
            group_images = [line_images[i] for i in image_indices]
            
            # Perform recognition with previous context
            start_time = time.time()
            abc_code = qwen_omr_request(group_images, previous_context)
            recognition_time = time.time() - start_time
            
            if not abc_code:
                print(f"Recognition failed for group {group_idx+1}")
                continue
            
            print(f"Group {group_idx+1} recognition completed! Time: {recognition_time:.2f}s | abc length: {len(abc_code)} characters")
            
            # Add this part to our collection
            all_abc_parts.append(abc_code)
            
            # Update context for next group (use the full merged result so far)
            previous_context = merge_abc_parts(all_abc_parts)
        
        # If we have multiple parts, merge them
        if len(all_abc_parts) > 1:
            final_abc = merge_abc_parts(all_abc_parts)
        elif all_abc_parts:
            final_abc = all_abc_parts[0]
        else:
            final_abc = ""
            
        if not final_abc:
            print("Recognition failed for all groups")
            continue
        
        print(f"Final merged ABC code length: {len(final_abc)} characters")
        
        # Step 3: Save results
        metadata = {
            "sample_id": sample_id,
            "source_pdf": row.pdf_file,
            "processing_time": recognition_time,
            "loading_time": loading_time,
            "pages_processed": split_metadata["pages_processed"],
            "total_lines": len(line_images),
            "model": "Qwen-VL-3B-Chat",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "title": row.title,
            "creator": row.creator,
            "key_signatures": row.key_signatures,
            "time_signatures": row.time_signatures,
            "tempo": row.tempo,
            "resolution": row.resolution,
            "parts_count": row.parts_count,
            "track_cnt": row.track_cnt,
            "split_metadata": split_metadata,
            "abc_parts_count": len(all_abc_parts)
        }
        
        save_results(sample_id, final_abc, metadata)
        print(f"Results saved to {OUTPUT_DIR}/{sample_id}.*")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Music Optical Recognition with Qwen-VL')
    parser.add_argument('--csv_path', type=str, default=CSV_PATH, 
                        help='Path to input CSV file')
    parser.add_argument('--split_dir', type=str, default=SPLIT_OUTPUT_DIR, 
                        help='Directory with split staff lines')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, 
                        help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Update configuration
    CSV_PATH = args.csv_path
    SPLIT_OUTPUT_DIR = args.split_dir
    OUTPUT_DIR = args.output_dir
    MAX_SAMPLES = args.max_samples
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run main program
    main(
        csv_path=CSV_PATH,
        split_dir=SPLIT_OUTPUT_DIR,
        max_samples=MAX_SAMPLES
    )

# import os
# import pandas as pd
# import tempfile
# import time
# import json
# import argparse
# from PIL import Image
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Qwen2_5_VLForConditionalGeneration

# # Configuration parameters
# CSV_PATH = "filtered_music_scores.csv"  # CSV file containing music score information
# OUTPUT_DIR = "qwen_results"             # Output directory
# SPLIT_OUTPUT_DIR = "split_lines"        # Directory for split line images
# MAX_SAMPLES = None                      # Maximum number of samples to process

# # Create output directory
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Global model and tokenizer
# model = None
# tokenizer = None

# def load_model():
#     """Load Qwen-VL model and tokenizer"""
#     global model, tokenizer
#     if model is None or tokenizer is None:
#         model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name, 
#             trust_remote_code=True
#         )
        
#         model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.float16,
#             trust_remote_code=True
#         ).eval()
        
#         # Configure generation parameters
#         model.generation_config = GenerationConfig.from_pretrained(
#             model_name, 
#             trust_remote_code=True
#         )
#         model.generation_config.max_new_tokens = 8192
#         model.generation_config.do_sample = False
        
#         print("Model loaded successfully.")

# def load_split_images(sample_id, split_dir):
#     """
#     Load split line images
#     :return: (list of line images, metadata)
#     """
#     # Load metadata
#     meta_path = os.path.join(split_dir, sample_id, "metadata.json")
#     if not os.path.exists(meta_path):
#         print(f"Metadata not found for sample: {sample_id}")
#         return None, None
    
#     with open(meta_path, "r") as f:
#         metadata = json.load(f)
    
#     # Load all line images
#     line_images = []
#     for img_info in metadata["line_images"]:
#         img_path = img_info["file_path"]
#         try:
#             img = Image.open(img_path).convert("RGB")
#             line_images.append(img)
#         except Exception as e:
#             print(f"Error loading image {img_path}: {str(e)}")
#             continue
    
#     return line_images, metadata

# def qwen_omr_request(line_images):
#     """
#     Perform music score recognition using Qwen-VL model (process line by line and concatenate results)
#     :param line_images: list of PIL.Image objects (in order)
#     :return: abc format music score
#     """
#     # System prompt - now requires output in abc format
#     system_prompt = (
#         "You are a professional Optical Music Recognition (OMR) system. Convert music score images into precise format.\n\n"
#     )
    
#     all_abc_results = []
    
#     # Process images line by line
#     for i, img in enumerate(line_images):
#         # Choose different prompts based on whether it's the first line
#         if i == 0:
#             user_prompt = [{
#                 "type": "text",
#                 "text": "Convert the staff lines to ABC notation. Include the header information (X, T, M, K, etc.) and the notes. Output ONLY the ABC code."
#             }]
#         else:
#             user_prompt = [{
#                 "type": "text",
#                 "text": "Convert the following staff lines to ABC notation, which are the continuation of the ABC notation above. Output ONLY the ABC code."
#             }]

#         # Build message list
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {
#                 "role": "user",
#                 "content": user_prompt + [{"image": img}]
#             }
#         ]

#         try:
#             # Preprocess messages using tokenizer.apply_chat_template
#             text_input = tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True,
#                 enable_thinking=False
#             )
            
#             # Convert input to the format required by the model
#             model_inputs = tokenizer(
#                 [text_input],
#                 padding=True,
#                 return_tensors="pt",
#                 return_attention_mask=True
#             ).to(model.device)
            
#             input_length = model_inputs.attention_mask.sum(dim=1).tolist()[0]

#             # Generate response
#             generated_ids = model.generate(
#                 **model_inputs,
#                 max_new_tokens=1000,  # Single line doesn't need too many tokens
#                 do_sample=True
#             )
            
#             # Decode generation results
#             response = tokenizer.decode(generated_ids[0][input_length:], skip_special_tokens=True)

#             # Extract abc content (remove possible extra text)
#             if "**" in response:
#                 response = response.split("**", 1)[0]
            
#             # Clean response, remove possible additional explanatory text
#             response = response.strip()
#             if response.startswith("ABC Notation:"):
#                 response = response[len("ABC Notation:"):].strip()
            
#             all_abc_results.append(response)

#         except Exception as e:
#             print(f"Model inference failed for line {i}: {str(e)}")
#             all_abc_results.append("")  # Add empty string as placeholder
    
#     # Merge all ABC results
#     if not all_abc_results:
#         return None
    
#     # First line should contain complete header information
#     full_abc = all_abc_results[0]
#     # print("start:", full_abc)
#     # Add note parts from subsequent lines
#     for i in range(1, len(all_abc_results)):
#         if all_abc_results[i]:
#             notes_only = all_abc_results[i]
#             if notes_only:
#                 full_abc += "\n" + notes_only
                
#             # Extract pure note parts (may require additional processing)
#             # notes_only = extract_notes_from_abc(all_abc_results[i])
#             # print("before filter:", all_abc_results[i])
#             # print("note only parts:", notes_only)
#             # if notes_only:
#             #     full_abc += "\n" + notes_only
    
#     return full_abc

# def extract_notes_from_abc(abc_text):
#     """
#     Extract pure note parts from ABC notation, remove possible duplicate header information
#     """
#     # Simple implementation: remove common header fields
#     lines = abc_text.split('\n')
#     note_lines = []
    
#     for line in lines:
#         line = line.strip()
#         # Skip empty lines and header information lines
#         if not line or line.startswith(('X:', 'T:', 'M:', 'L:', 'K:', '%')):
#             continue
#         note_lines.append(line)
    
#     return ' '.join(note_lines)

# def save_results(sample_id, abc_code, metadata):
#     """Save recognition results and metadata"""
#     # Save abc code
#     abc_path = os.path.join(OUTPUT_DIR, f"{sample_id}.abc")
#     with open(abc_path, "w", encoding="utf-8") as f:
#         f.write(abc_code)
    
#     # Save metadata
#     meta_path = os.path.join(OUTPUT_DIR, f"{sample_id}_metadata.json")
#     with open(meta_path, "w") as f:
#         json.dump(metadata, f, indent=2)

# def main(csv_path, split_dir, max_samples=None):
#     """Main processing function"""
#     # Load model
#     load_model()
    
#     # Read CSV file
#     df = pd.read_csv(csv_path)
    
#     # If maximum number of samples is set, truncate data
#     if max_samples is not None and max_samples > 0:
#         df = df.head(max_samples)
    
#     total = len(df)
#     print(f"Starting OMR processing for {total} samples...")
    
#     # Process each music score sample
#     for i, row in enumerate(df.itertuples(), 1):
#         sample_id = row.sample_id
        
#         print(f"\n[{i}/{total}] Processing sample: {sample_id}")
        
#         # Step 1: Load split line images
#         start_time = time.time()
#         line_images, split_metadata = load_split_images(sample_id, split_dir)
#         loading_time = time.time() - start_time
        
#         if not line_images or not split_metadata:
#             print(f"Failed to load split images for sample: {sample_id}")
#             continue
        
#         print(f"Loaded {len(line_images)} staff lines in {loading_time:.2f}s")
        
#         # Step 2: Perform recognition using Qwen-VL
#         start_time = time.time()
#         abc_code = qwen_omr_request(line_images)
#         recognition_time = time.time() - start_time
        
#         if not abc_code:
#             print("Recognition failed")
#             continue
        
#         print(f"Recognition completed! Time: {recognition_time:.2f}s | abc length: {len(abc_code)} characters")
        
#         # Step 3: Save results
#         metadata = {
#             "sample_id": sample_id,
#             "source_pdf": row.pdf_file,
#             "processing_time": recognition_time,
#             "loading_time": loading_time,
#             "pages_processed": split_metadata["pages_processed"],
#             "total_lines": len(line_images),
#             "model": "Qwen-VL-3B-Chat",
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "title": row.title,
#             "creator": row.creator,
#             "key_signatures": row.key_signatures,
#             "time_signatures": row.time_signatures,
#             "tempo": row.tempo,
#             "resolution": row.resolution,
#             "parts_count": row.parts_count,
#             "track_cnt": row.track_cnt,
#             "split_metadata": split_metadata
#         }
        
#         save_results(sample_id, abc_code, metadata)
#         print(f"Results saved to {OUTPUT_DIR}/{sample_id}.*")
    
#     print("\nProcessing complete!")

# if __name__ == "__main__":
#     # Set command line arguments
#     parser = argparse.ArgumentParser(description='Music Optical Recognition with Qwen-VL')
#     parser.add_argument('--csv_path', type=str, default=CSV_PATH, 
#                         help='Path to input CSV file')
#     parser.add_argument('--split_dir', type=str, default=SPLIT_OUTPUT_DIR, 
#                         help='Directory with split staff lines')
#     parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, 
#                         help='Output directory for results')
#     parser.add_argument('--max_samples', type=int, default=None, 
#                         help='Maximum number of samples to process')
    
#     args = parser.parse_args()
    
#     # Update configuration
#     CSV_PATH = args.csv_path
#     SPLIT_OUTPUT_DIR = args.split_dir
#     OUTPUT_DIR = args.output_dir
#     MAX_SAMPLES = args.max_samples
    
#     # Ensure output directory exists
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # Run main program
#     main(
#         csv_path=CSV_PATH,
#         split_dir=SPLIT_OUTPUT_DIR,
#         max_samples=MAX_SAMPLES
#     )