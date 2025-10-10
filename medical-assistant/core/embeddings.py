import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

def create_vector_store(processed_data_path: str, vector_db_path: str):
    """
    Эта функция читает обработанные данные, создает эмбеддинги и сохраняет их в ChromaDB.
    
    :param processed_data_path: Путь к JSON-файлу с текстом.
    :param vector_db_path: Путь, где будет сохранена база данных.
    """
    try:
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        if not documents:
            print("В JSON-файле нет данных для обработки.")
            return

        langchain_documents = []
        for doc in documents:
            langchain_documents.append(Document(
                page_content=doc['text'],
                metadata={"source": doc['source'], "id": doc['id']}
            ))

        # Используем бесплатную локальную модель для создания эмбеддингов
        # Вы можете поменять модель на "intfloat/multilingual-e5-large"
        embedding_function = SentenceTransformerEmbeddings(
            model_name="intfloat/multilingual-e5-large"
        )
        
        print("Создаем векторную базу данных с локальной моделью...")

        # Создаем базу данных Chroma, используя документы и функцию для эмбеддингов
        vector_store = Chroma.from_documents(
            documents=langchain_documents,
            embedding=embedding_function,
            persist_directory=vector_db_path # Указываем, куда сохранять базу данных
        )
        
        # Сохраняем базу данных на диск
        vector_store.persist()
        
        print(f"Готово! База данных сохранена в {vector_db_path}.")

    except Exception as e:
        print(f"Произошла ошибка: {e}")



# 3. Проверяем, запущен ли скрипт напрямую
if __name__ == "__main__":
    # Указываем пути к файлам и папкам
    processed_json_path = os.path.join("data", "processed", "processed_pdfs.json")
    vector_db_folder = os.path.join("data", "vector_db")
    
    # Создаем папку для базы данных, если ее еще нет
    os.makedirs(vector_db_folder, exist_ok=True)
    
    # Запускаем нашу главную функцию
    create_vector_store(processed_json_path, vector_db_folder)
