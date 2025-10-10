from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
import fitz
import os


def extract_text_from_PDF(filepath: str) -> str:
    try:
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"Error while extracting file {filepath}: {e}")
        return ""


def split_text_into_chunks(text: str, source: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[Dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "id": f"{os.path.splitext(source)[0]}_{i}",
            "text": chunk,
            "source": source
        })
        
    return documents


