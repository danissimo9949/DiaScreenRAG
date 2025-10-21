import os
import time
import psutil
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) Pipeline:
    1. Извлекает релевантные документы из Chroma.
    2. Формирует промпт.
    3. Отправляет его в LLM (Google Gemini).
    """

    def __init__(
        self,
        vector_db_path: str,
        model_name: str = "intfloat/multilingual-e5-large",
        llm_model: str = "models/gemini-2.5-pro",
        language: str = "en",
        debug: bool = False,
        relevance_threshold: float = 0.3,
    ):
        self.vector_db_path = vector_db_path
        self.language = language
        self.debug = debug
        self.relevance_threshold = relevance_threshold  # Порог: подберите под ваши embeddings (0.5-0.7 типично)
        self.embedding_model = self._load_embedding_model(model_name)
        self.vector_store = self._load_vector_store()
        self.llm = self._load_llm(llm_model)
        
    def _load_embedding_model(self, model_name: str):
        return SentenceTransformerEmbeddings(model_name=model_name)

    def _load_llm(self, model_name: str):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key didn't exist")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    def _load_vector_store(self):
        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(f"Vector database didn't exist: {self.vector_db_path}")

        vector_store = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embedding_model,
        )

        if self.debug:
            print(f"Load documents with Chroma: {len(vector_store.get()['ids'])}")
        return vector_store
    
    def query(self, user_question: str, k: int = 4) -> str:
        relevant_docs = self._retrieve_context(user_question, k)
        context = self._combine_context(relevant_docs)

        if not relevant_docs or context.strip() == "No relevant context found.":
            prompt = self._create_fallback_prompt()
            context = ""  # Без контекста для fallback
        else:
            prompt = self._create_prompt()

        return self._generate_answer(prompt, context, user_question)
    
    def _retrieve_context(self, question: str, k: int):
        # Изменено: используем similarity_search_with_score для получения score
        docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
        
        # Фильтруем по релевантности: score - это distance (меньше = лучше)
        relevant_docs = [doc for doc, score in docs_with_scores if score <= self.relevance_threshold]
        
        if self.debug:
            print(f"\nRetrieved {len(docs_with_scores)} docs, but after filtering by threshold {self.relevance_threshold}: {len(relevant_docs)} relevant")
            for i, (doc, score) in enumerate(docs_with_scores):
                print(f"Doc {i+1} score: {score:.4f} (relevant: {score <= self.relevance_threshold})")
        
        return relevant_docs  # Возвращаем только релевантные документы
    
    def _combine_context(self, docs) -> str:
        if not docs:
            return "No relevant context found."
        return "\n".join(doc.page_content for doc in docs)
    
    def _create_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
            You are a helpful medical assistant. You must only answer questions related to medicine, health and diabetus mellitus.

            Use the following context to answer the question as accurately as possible.
            If the context does not contain enough information to give a complete or confident answer,
            you may use your own medical knowledge to fill in the missing details.
            However, always prioritize and clearly reference the information found in the provided context.
            Also, do not mention or refer to any "provided context" or "background information" in your answer.
            Simply answer as if you already know these facts

            Context:
            {context}

            Question: {question}

            Answer (in {language}) please, in a natural and professional tone:

            At the end of your response, include this disclaimer:
            "⚠️ This information is not a substitute for professional medical advice. Always consult a healthcare provider."
            """
        )
    
    def _create_fallback_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """
            You are a helpful medical assistant. You must only answer questions related to medicine, health and diabetus mellitus.
            The system could not find any relevant documents,
            So please answer the following question using your general knowledge.
            
            Question: {question}

            Answer (in {language}) please:

            At the end of your response, include this disclaimer:
            "⚠️ This information is not a substitute for professional medical advice. Always consult a healthcare provider."
            """
        )
    
    def _generate_answer(self, prompt_template, context: str, question: str) -> str:
        chain = prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question, "language": self.language})
        
        if self.debug:
            print(f"\n🧠 Контекст, переданный в LLM:\n{context}...\n")
        return response
    
    def health_check(self) -> Dict[str, Any]:
        """
        Проверяет состояние всех компонентов системы
        """
        health_status = {
            "vector_database": self._check_vector_database(),
            "llm_service": self._check_llm_service(),
            "embedding_model": self._check_embedding_model(),
            "performance": self._get_performance_metrics(),
            "configuration": self._get_configuration_info()
        }
        
        # Определяем общий статус
        all_up = all(
            component["status"] == "up" 
            for component in health_status.values() 
            if isinstance(component, dict) and "status" in component
        )
        
        health_status["overall_status"] = "healthy" if all_up else "unhealthy"
        health_status["timestamp"] = datetime.now().isoformat()
        
        return health_status
    
    def _check_vector_database(self) -> Dict[str, Any]:
        """Проверяет состояние векторной базы данных"""
        try:
            # Проверяем существование директории
            if not os.path.exists(self.vector_db_path):
                return {
                    "status": "down",
                    "message": f"Vector database directory not found: {self.vector_db_path}",
                    "details": {"path": self.vector_db_path}
                }
            
            # Проверяем подключение к ChromaDB
            test_query = self.vector_store.similarity_search("test", k=1)
            
            # Получаем информацию о базе данных
            db_info = self.vector_store.get()
            documents_count = len(db_info['ids']) if db_info['ids'] else 0
            
            # Получаем размер директории
            db_size = self._get_directory_size(self.vector_db_path)
            
            return {
                "status": "up",
                "message": "Vector database is accessible",
                "details": {
                    "documents_count": documents_count,
                    "database_size_mb": round(db_size / (1024 * 1024), 2),
                    "path": self.vector_db_path
                }
            }
            
        except Exception as e:
            return {
                "status": "down",
                "message": f"Vector database error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _check_llm_service(self) -> Dict[str, Any]:
        """Проверяет состояние LLM сервиса"""
        try:
            # Простой тестовый запрос
            start_time = time.time()
            test_response = self.llm.invoke("Hello")
            response_time = time.time() - start_time
            
            return {
                "status": "up",
                "message": "LLM service is accessible",
                "details": {
                    "model": "models/gemini-2.5-pro",
                    "response_time_seconds": round(response_time, 2),
                    "test_response_length": len(str(test_response))
                }
            }
            
        except Exception as e:
            return {
                "status": "down",
                "message": f"LLM service error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _check_embedding_model(self) -> Dict[str, Any]:
        """Проверяет состояние embedding модели"""
        try:
            # Тестовое создание эмбеддинга
            start_time = time.time()
            test_embedding = self.embedding_model.embed_query("test")
            embedding_time = time.time() - start_time
            
            return {
                "status": "up",
                "message": "Embedding model is loaded and working",
                "details": {
                    "model": "intfloat/multilingual-e5-large",
                    "embedding_dimensions": len(test_embedding),
                    "embedding_time_seconds": round(embedding_time, 3)
                }
            }
            
        except Exception as e:
            return {
                "status": "down",
                "message": f"Embedding model error: {str(e)}",
                "details": {"error": str(e)}
            }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Получает метрики производительности"""
        try:
            # Получаем информацию о системе
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            return {
                "memory_usage_mb": round(memory.used / (1024 * 1024), 2),
                "memory_percent": memory.percent,
                "cpu_percent": cpu_percent,
                "available_memory_mb": round(memory.available / (1024 * 1024), 2)
            }
            
        except Exception as e:
            return {
                "error": f"Failed to get performance metrics: {str(e)}"
            }
    
    def _get_configuration_info(self) -> Dict[str, Any]:
        """Получает информацию о конфигурации"""
        return {
            "relevance_threshold": self.relevance_threshold,
            "language": self.language,
            "debug_mode": self.debug,
            "vector_db_path": self.vector_db_path
        }
    
    def _get_directory_size(self, path: str) -> int:
        """Вычисляет размер директории в байтах"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size


if __name__ == "__main__":
    try:
        base_dir = os.getcwd()
        vector_db_folder = os.path.join(base_dir, "data", "vector_db") 
        rag_system = RAGPipeline(vector_db_path=vector_db_folder, debug=True, relevance_threshold=0.35)
        
        user_question = "Who is Gamash, you know?"
        print(f"Question: {user_question}")
        answer = rag_system.query(user_question)
        print(f"Answer: {answer}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error occured: {e}")