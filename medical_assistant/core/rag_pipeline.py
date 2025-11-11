import hashlib
import os
import time
from copy import deepcopy
import psutil
from datetime import datetime
from typing import Dict, Any, List
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
        enable_cache: bool = True,
    ):
        self.vector_db_path = vector_db_path
        self.enable_cache = enable_cache
        self.cache = {} if enable_cache else None
        self.language = language
        self.debug = debug
        self.relevance_threshold = relevance_threshold
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
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            max_retries=1,
        )

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

    def _get_cache_key(self, question: str) -> str:
        return hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    def clear_cache(self):
        """Очищает весь кеш"""
        if self.enable_cache and self.cache:
            cache_size = len(self.cache)
            self.cache.clear()
            if self.debug:
                print(f"🗑️ Cache cleared. {cache_size} items removed.")
            return cache_size
        return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Получает статистику кеша"""
        if not self.enable_cache or self.cache is None:
            return {
                "enabled": False,
                "size": 0,
                "items": 0
            }
        
        # Вычисляем примерный размер кеша в байтах
        cache_size_bytes = sum(len(str(v).encode('utf-8')) for v in self.cache.values())
        
        return {
            "enabled": True,
            "items": len(self.cache),
            "size_kb": round(cache_size_bytes / 1024, 2),
            "size_mb": round(cache_size_bytes / (1024 * 1024), 2)
        }
    
    def query(self, user_question: str, k: int = 4) -> Dict[str, Any]:
        start_time = time.time()

        cache_key = None
        if self.enable_cache:
            cache_key = self._get_cache_key(user_question)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                if self.debug:
                    print(f"Cache hit for question: {user_question}")
                cached_copy = deepcopy(cached_result)
                metadata = cached_copy.setdefault("metadata", {})
                metadata["cache_hit"] = True
                metadata.setdefault("response_time_ms", 0)
                metadata.setdefault("retrieved_documents", len(cached_copy.get("sources", [])))
                metadata.setdefault("relevance_threshold", self.relevance_threshold)
                metadata.setdefault("language", self.language)
                return cached_copy
        
        relevant_docs = self._retrieve_context(user_question, k)
        context = self._combine_context(relevant_docs)

        if not relevant_docs or context.strip() == "No relevant context found.":
            prompt = self._create_fallback_prompt()
            prompt_type = "fallback"
            context = ""
        else:
            prompt = self._create_prompt()
            prompt_type = "standard"

        answer = self._generate_answer(prompt, context, user_question)
        response_time_ms = round((time.time() - start_time) * 1000)

        sources = self._format_sources(relevant_docs)
        metadata = {
            "cache_hit": False,
            "response_time_ms": response_time_ms,
            "retrieved_documents": len(sources),
            "relevance_threshold": self.relevance_threshold,
            "language": self.language,
            "prompt_type": prompt_type,
        }

        result = {
            "answer": answer,
            "sources": sources,
            "metadata": metadata,
        }
        
        if self.enable_cache and cache_key is not None:
            self.cache[cache_key] = deepcopy(result)
            if self.debug:
                print(f"💾 Answer cached. Total cached items: {len(self.cache)}")
        
        return result
    
    def _retrieve_context(self, question: str, k: int) -> List[Dict[str, Any]]:
        docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
        
        relevant_docs = []
        for doc, score in docs_with_scores:
            if score <= self.relevance_threshold:
                relevant_docs.append({
                    "document": doc,
                    "score": score
                })
        
        if self.debug:
            print(f"\nRetrieved {len(docs_with_scores)} docs, but after filtering by threshold {self.relevance_threshold}: {len(relevant_docs)} relevant")
            for i, (doc, score) in enumerate(docs_with_scores):
                print(f"Doc {i+1} score: {score:.4f} (relevant: {score <= self.relevance_threshold})")
        
        return relevant_docs
    
    def _combine_context(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return "No relevant context found."
        return "\n".join(entry["document"].page_content for entry in docs)

    def _format_sources(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_ids = set()
        sources = []
        for entry in docs:
            doc = entry["document"]
            score = entry["score"]
            metadata = doc.metadata or {}
            source_id = metadata.get("id") or metadata.get("source")
            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            sources.append({
                "id": metadata.get("id"),
                "source": metadata.get("source"),
                "score": score,
            })
        return sources
    
    def _create_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """You are a medical AI assistant specializing in diabetes mellitus. Your goal is to provide accurate, evidence-based information to patients and healthcare providers.

Context from medical literature:
{context}

Patient question: {question}

Instructions for your response:
1. **If the context fully answers the question**: Provide a confident, detailed answer based on the medical literature
2. **If the context is incomplete**: Start with "Based on available information and general medical knowledge:" and explain what you know
3. **If the context is not relevant**: Start with "I don't have specific information in my database about this, but based on general medical knowledge:"
4. **Use simple language**: Explain medical terms when you use them
5. **Be structured**: Use bullet points or sections for complex topics
6. **Be empathetic**: Use a supportive, professional tone

Answer in {language}:

---

Key takeaways:
• [Main point 1]
• [Main point 2]
• [Main point 3 if applicable]

⚠️ This information is educational and not a substitute for professional medical advice. Always consult your healthcare provider for personalized guidance.
"""
        )
    
    def _create_fallback_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            """You are a medical AI assistant specializing in diabetes mellitus.

⚠️ Note: No relevant information was found in the medical database for this question.

Question: {question}

Instructions:
- Answer ONLY if the question is related to medicine, health, or diabetes mellitus
- If the question is unrelated to these topics, politely decline and suggest asking a diabetes-related question
- Start your answer with: "I don't have specific information in my database, but based on general medical knowledge:"
- Use simple language in {language}
- Be transparent about limitations
- Keep the answer concise but helpful

Your answer:

⚠️ This information is based on general medical knowledge and not a substitute for professional medical advice. Please consult your healthcare provider.
"""
        )
    
    def _generate_answer(self, prompt_template, context: str, question: str) -> str:
        chain = prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question, "language": self.language})
        
        if self.debug:
            print(f"\n🧠 Контекст, переданный в LLM:\n{context}...\n")
        return response
    
    def health_check(self) -> Dict[str, Any]:
        health_status = {
            "vector_database": self._check_vector_database(),
            "llm_service": self._check_llm_service(),
            "embedding_model": self._check_embedding_model(),
            "performance": self._get_performance_metrics(),
            "cache": self.get_cache_stats(),
            "configuration": self._get_configuration_info()
        }
        
        all_up = all(
            component["status"] == "up" 
            for component in health_status.values() 
            if isinstance(component, dict) and "status" in component
        )
        
        health_status["overall_status"] = "healthy" if all_up else "unhealthy"
        health_status["timestamp"] = datetime.now().isoformat()
        
        return health_status
    
    def _check_vector_database(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.vector_db_path):
                return {
                    "status": "down",
                    "message": f"Vector database directory not found: {self.vector_db_path}",
                    "details": {"path": self.vector_db_path}
                }
            
            test_query = self.vector_store.similarity_search("test", k=1)
            
            db_info = self.vector_store.get()
            documents_count = len(db_info['ids']) if db_info['ids'] else 0
            
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
        return {
            "relevance_threshold": self.relevance_threshold,
            "language": self.language,
            "debug_mode": self.debug,
            "cache_enabled": self.enable_cache,
            "vector_db_path": self.vector_db_path
        }
    
    def _get_directory_size(self, path: str) -> int:
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