from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class ComponentStatus(str, Enum):
    UP = "up"
    DOWN = "down"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    status: ComponentStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: HealthStatus
    timestamp: datetime
    version: str = "1.0.0"
    components: Dict[str, ComponentHealth]
    performance: Optional[Dict[str, Any]] = None
    configuration: Optional[Dict[str, Any]] = None


class SimpleHealthResponse(BaseModel):
    status: HealthStatus
    timestamp: datetime
    message: str


class PersonalizedQueryRequest(BaseModel):
    question: str
    context: str
    mode: Optional[str] = None
    language: Optional[str] = None