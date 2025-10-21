from typing import Union
from datetime import datetime
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from medical_assistant.core.rag_pipeline import RAGPipeline
from medical_assistant.api.models import HealthResponse, SimpleHealthResponse, HealthStatus, ComponentStatus, ComponentHealth

import os

# Настройка логирования
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
        
        # Преобразуем данные в формат модели
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
        # Быстрая проверка основных компонентов
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
        
        # Добавляем дополнительную информацию
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