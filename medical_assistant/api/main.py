from typing import Union
from datetime import datetime
import logging
import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from medical_assistant.core.rag_pipeline import RAGPipeline
from medical_assistant.core.data_processing_utils import extract_text_from_PDF, split_text_into_chunks
from medical_assistant.core.embeddings import create_vector_store
from medical_assistant.api.models import HealthResponse, SimpleHealthResponse, HealthStatus, ComponentStatus, ComponentHealth

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize RAG on server start
try:
    base_dir = os.getcwd()
    vector_db_folder = os.path.join(base_dir, "data", "vector_db")
    logger.info(f"Initializing RAG pipeline with vector DB at: {vector_db_folder}")
    rag_pipeline = RAGPipeline(vector_db_path=vector_db_folder, relevance_threshold=0.35)
    logger.info("RAG pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG pipeline: {e}")
    raise

def add_documents_to_vector_store(chunks: list, vector_db_path: str):
    
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_core.documents import Document
    
    try:
        embedding_function = SentenceTransformerEmbeddings(
            model_name="intfloat/multilingual-e5-large"
        )
        
        new_documents = []
        for chunk in chunks:
            new_documents.append(Document(
                page_content=chunk['text'],
                metadata={"source": chunk['source'], "id": chunk['id']}
            ))
        
        if os.path.exists(vector_db_path):
            logger.info(f"Loading existing vector database from {vector_db_path}")
            vector_store = Chroma(
                persist_directory=vector_db_path,
                embedding_function=embedding_function
            )
            vector_store.add_documents(new_documents)
            logger.info(f"Added {len(new_documents)} new documents to existing database")
        else:
            logger.info(f"Creating new vector database at {vector_db_path}")
            vector_store = Chroma.from_documents(
                documents=new_documents,
                embedding=embedding_function,
                persist_directory=vector_db_path
            )
            logger.info(f"Created new database with {len(new_documents)} documents")
        
        vector_store.persist()
        logger.info("Vector database saved successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating vector database: {e}")
        return False

def process_single_pdf(file_path: str, filename: str):
    try:
        logger.info(f"Processing PDF file: {filename}")

        processed_file = os.path.join(base_dir, "data", "processed", "processed_pdfs.json")
        all_chunks = []

        if os.path.exists(processed_file):
            with open(processed_file, 'r', encoding='utf-8') as f:
                all_chunks = json.load(f)
                
            existing_filenames = {chunk["source"] for chunk in all_chunks if "source" in chunk}
            if filename in existing_filenames:
                logger.info(f"⚠️ Файл {filename} уже обработан — пропускаем")
                return False

        text = extract_text_from_PDF(file_path)
        if not text:
            logger.error(f"No text extracted from {filename}")
            return False

        chunks = split_text_into_chunks(text, filename)
        logger.info(f"Created {len(chunks)} chunks from {filename}")

        all_chunks.extend(chunks)

        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=4)

        vector_db_folder = os.path.join(base_dir, "data", "vector_db")
        success = add_documents_to_vector_store(chunks, vector_db_folder)
        
        if success:
            logger.info(f"Successfully added {filename} to vector database")
            return True
        else:
            logger.error(f"Failed to add {filename} to vector database")
            return False

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return False

# Get response from LLM on user question
@app.get("/get-response")
def get_response_from_LLM(question: str):
    """
    Получить ответ от медицинского ассистента на вопрос пользователя
    """
    try:
        logger.info(f"Received question: {question[:100]}...")
        
        if not question or not question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        answer = rag_pipeline.query(question)
        logger.info(f"Successfully generated answer for question")
        
        return {"answer": answer}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question '{question}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Health check endpoints
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Comprehensive health check endpoint
    """
    try:
        logger.info("Performing comprehensive health check")
        health_data = rag_pipeline.health_check()
        
        components = {}
        overall_status = HealthStatus.HEALTHY
        
        for component_name, component_data in health_data.items():
            if component_name in ["overall_status", "timestamp"]:
                continue
                
            if isinstance(component_data, dict) and "status" in component_data:
                status = ComponentStatus.UP if component_data["status"] == "up" else ComponentStatus.DOWN
                if status == ComponentStatus.DOWN:
                    overall_status = HealthStatus.UNHEALTHY
                    logger.warning(f"Component {component_name} is down: {component_data.get('message')}")
                    
                components[component_name] = ComponentHealth(
                    status=status,
                    message=component_data.get("message"),
                    details=component_data.get("details")
                )
        
        logger.info(f"Health check completed with status: {overall_status}")
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            components=components,
            performance=health_data.get("performance"),
            configuration=health_data.get("configuration")
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/health/live", response_model=SimpleHealthResponse)
def liveness_probe():
    """
    Kubernetes liveness probe - проверяет, что приложение запущено
    """
    return SimpleHealthResponse(
        status=HealthStatus.HEALTHY,
        timestamp=datetime.now(),
        message="Application is running"
    )

@app.get("/health/ready", response_model=SimpleHealthResponse)
def readiness_probe():
    """
    Kubernetes readiness probe - проверяет готовность к обработке запросов
    """
    try:
        logger.debug("Performing readiness probe")
        health_data = rag_pipeline.health_check()
        
        if health_data.get("overall_status") == "healthy":
            logger.debug("Application is ready")
            return SimpleHealthResponse(
                status=HealthStatus.HEALTHY,
                timestamp=datetime.now(),
                message="Application is ready to serve requests"
            )
        else:
            logger.warning("Application is not ready")
            return SimpleHealthResponse(
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                message="Application is not ready"
            )
            
    except Exception as e:
        logger.error(f"Readiness probe failed: {str(e)}")
        return SimpleHealthResponse(
            status=HealthStatus.UNHEALTHY,
            timestamp=datetime.now(),
            message=f"Readiness check failed: {str(e)}"
        )

@app.get("/health/detailed")
def detailed_health_check():
    """
    Расширенная диагностика системы
    """
    try:
        logger.info("Performing detailed health check")
        health_data = rag_pipeline.health_check()
        
        detailed_info = {
            **health_data,
            "system_info": {
                "python_version": os.sys.version,
                "working_directory": os.getcwd(),
                "environment_variables": {
                    "GEMINI_API_KEY": "***" if os.getenv("GEMINI_API_KEY") else "Not set"
                }
            },
            "api_info": {
                "endpoints": ["/get-response", "/health", "/health/live", "/health/ready", "/health/detailed"],
                "version": "1.0.0"
            }
        }
        
        logger.info("Detailed health check completed successfully")
        return detailed_info
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detailed health check failed: {str(e)}"
        )

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
   
    try:
        logger.info(f"Uploading document: {file.filename}")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        raw_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        file_path = os.path.join(raw_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        success = process_single_pdf(file_path, file.filename)
        
        if success:
            return {
                "message": f"Document {file.filename} uploaded and processed successfully",
                "filename": file.filename,
                "status": "success"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process document {file.filename}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document: {str(e)}"
        )


@app.post("/cache/clear")
def clear_cache():
    try:
        logger.info("Clearing cache")
        cleared_items = rag_pipeline.clear_cache()
        logger.info(f"Cache cleared: {cleared_items} items removed")
        return {
            "message": "Cache cleared successfully",
            "cleared_items": cleared_items,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )