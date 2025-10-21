import os
import json
from .data_processing_utils import extract_text_from_PDF, split_text_into_chunks

def process_pdfs_to_chunks(input_dir: str, output_path: str):
    """
     Process PDF files by extracting text and splitting it in chunks per 400 char. Output is JSON file with chunked text
    """
    all_chunks = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(input_dir, filename)
            
            text = extract_text_from_PDF(filepath)
            
            if text:
                chunks = split_text_into_chunks(text, filename)
                all_chunks.extend(chunks)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    input_folder = "data/raw"
    output_file = "data/processed/processed_pdfs.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    process_pdfs_to_chunks(input_folder, output_file)