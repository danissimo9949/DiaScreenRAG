import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

def create_vector_store(processed_data_path: str, vector_db_path: str):
    """
    This function create embeddings from processed data and save them in vector database (ChromaDB).
    
    :param processed_data_path: path to JSON file.
    :param vector_db_path: path for data storage.
    """
    try:
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        if not documents:
            print("No JSON files in directory")
            return

        langchain_documents = []
        for doc in documents:
            langchain_documents.append(Document(
                page_content=doc['text'],
                metadata={"source": doc['source'], "id": doc['id']}
            ))

        embedding_function = SentenceTransformerEmbeddings(
            model_name="intfloat/multilingual-e5-large"
        )
        
        print("Creating vector database with local model")

        vector_store = Chroma.from_documents(
            documents=langchain_documents,
            embedding=embedding_function,
            persist_directory=vector_db_path
        )
        
        vector_store.persist()
        
        print(f"Congrats! Database succesfully created at {vector_db_path}.")

    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    
    processed_json_path = os.path.join("data", "processed", "processed_pdfs.json")
    vector_db_folder = os.path.join("data", "vector_db")
    
    os.makedirs(vector_db_folder, exist_ok=True)
    
    create_vector_store(processed_json_path, vector_db_folder)
