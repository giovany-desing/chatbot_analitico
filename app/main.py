"""
Servidor FastAPI - Punto de entrada de la aplicación.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH
# Esto es necesario cuando se ejecuta directamente con python3 app/main.py
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import asyncio
from contextlib import asynccontextmanager

from app.config import settings
from app.agents.graph import chatbot_graph
from app.agents.state import create_initial_state
from app.db.connections import check_all_connections
from app.services.cache_service import cache_service

from .feedback.feedback_service import feedback_service
from pydantic import BaseModel, Field

class FeedbackRequest(BaseModel):
    feedback_id: int = Field(..., description="ID de la interacción")
    rating: int = Field(..., ge=1, le=5, description="Rating de 1 a 5")
    feedback_text: Optional[str] = Field(None, description="Comentario opcional")

class MetricsResponse(BaseModel):
    period_days: int
    general: Dict
    rating_distribution: Dict[int, int]
    top_charts: List[Dict]
    top_errors: List[Dict]

# Configurar logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ Lifecycle ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida de la app.
    Se ejecuta al inicio y al final.
    """
    # Startup
    logger.info("🚀 Starting Chatbot Analítico API...")

    # Verificar conexiones
    connections = check_all_connections()
    logger.info(f"Database connections: {connections}")

    if not all(connections.values()):
        logger.error("❌ Some database connections failed!")
    else:
        logger.info("✅ All database connections OK")

    yield

    # Shutdown
    logger.info("👋 Shutting down...")


# ============ FastAPI App ============

app = FastAPI(
    title=settings.APP_NAME,
    description="API para chatbot analítico con LangChain y LangGraph",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS (permitir frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Schemas ============

class ChatRequest(BaseModel):
    """Request para el endpoint /chat"""
    message: str = Field(..., min_length=1, max_length=1000, description="Pregunta del usuario")
    conversation_id: Optional[str] = Field(None, description="ID de conversación (para historial)")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "¿Cuántas ventas hubo en enero?",
                "conversation_id": "conv-123"
            }
        }


class ChatResponse(BaseModel):
    """Response del endpoint /chat"""
    response: str = Field(..., description="Respuesta del chatbot")
    intent: str = Field(..., description="Intención clasificada")
    sql_query: Optional[str] = Field(None, description="Query SQL generada (si aplica)")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="Resultados de la query SQL")
    kpis: Optional[Dict[str, Any]] = Field(None, description="KPIs calculados (si aplica)")
    chart_config: Optional[Dict[str, Any]] = Field(None, description="Configuración de gráfica Plotly (si aplica)")
    error: Optional[str] = Field(None, description="Error si ocurrió")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Aquí están las ventas del último mes con su gráfica.",
                "intent": "hybrid",
                "sql_query": "SELECT producto, SUM(total) as revenue FROM ventas_preventivas WHERE fecha_creacion >= DATE_SUB(NOW(), INTERVAL 1 MONTH) GROUP BY producto",
                "results": [
                    {"producto": "Tela A", "revenue": 15000},
                    {"producto": "Tela B", "revenue": 12000}
                ],
                "kpis": {
                    "revenue_total": 27000,
                    "productos_unicos": 2
                },
                "chart_config": {
                    "data": [{"type": "bar", "x": ["Tela A", "Tela B"], "y": [15000, 12000]}],
                    "layout": {"title": "Revenue por Producto", "xaxis": {"title": "Producto"}, "yaxis": {"title": "Revenue"}}
                },
                "error": None
            }
        }


class HealthResponse(BaseModel):
    """Response del endpoint /health"""
    status: str
    version: str
    databases: Dict[str, bool]


# ============ Endpoints ============

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """
    Endpoint de health check.
    Verifica que la API y las bases de datos funcionen.
    """
    connections = check_all_connections()

    return HealthResponse(
        status="healthy" if all(connections.values()) else "degraded",
        version=settings.APP_VERSION,
        databases=connections
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Endpoint principal del chatbot.
    
    Recibe una pregunta y retorna la respuesta usando el grafo LangGraph.
    
    Args:
        request: ChatRequest con el mensaje del usuario
    
    Returns:
        ChatResponse con la respuesta del chatbot
    
    Raises:
        HTTPException: Si ocurre un error procesando la pregunta
    """
    try:
        logger.info(f"Received query: {request.message}")

        # Crear estado inicial
        state = create_initial_state(request.message)

        # Ejecutar grafo con timeout para evitar que se cuelgue
        # Usar asyncio para ejecutar la función síncrona con timeout
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(chatbot_graph.invoke, state),
                timeout=120.0  # 120 segundos de timeout (aumentado para LLM)
            )
        except asyncio.TimeoutError:
            logger.error("Chatbot graph execution timed out after 120 seconds")
            raise HTTPException(
                status_code=504,
                detail="Request timed out. The chatbot is taking too long to respond. This may be due to LLM API delays or database connection issues."
            )

        # Extraer respuesta
        response_text = ""
        if result.get('messages'):
            response_text = result['messages'][-1].content

        # Construir response
        response = ChatResponse(
            response=response_text or "No pude generar una respuesta.",
            intent=result.get('intent', 'unknown'),
            sql_query=result.get('sql_query'),
            results=result.get('sql_results'),
            kpis=result.get('kpis'),
            chart_config=result.get('chart_config'),
            error=result.get('error')
        )

        logger.info(f"Response intent: {response.intent}")

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your question: {str(e)}"
        )


@app.get("/schema", tags=["Database"])
async def get_schema():
    """
    Retorna el schema de la base de datos.
    Útil para debugging y documentación.
    """
    try:
        from app.tools.sql_tool import mysql_tool
        schema = mysql_tool.get_schema_info()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples", tags=["Database"])
async def get_examples():
    """
    Retorna queries SQL de ejemplo.
    """
    try:
        from app.tools.sql_tool import mysql_tool
        examples = mysql_tool.get_sample_queries()
        return {"examples": examples}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cache", tags=["Admin"])
async def clear_cache():
    """
    Limpia todo el cache de queries.
    Útil después de modificar datos en la BD.
    """
    cleared = cache_service.clear_all()
    return {
        "message": f"Cache cleared: {cleared} queries",
        "cleared": cleared
    }
    
@app.get("/cache/stats", tags=["Admin"])
async def cache_stats():
    """
    Estadísticas del cache Redis.
    """
    try:
        redis_client = cache_service.redis.client
        info = redis_client.info('stats')

        return {
            "keyspace_hits": info.get('keyspace_hits', 0),
            "keyspace_misses": info.get('keyspace_misses', 0),
            "total_commands": info.get('total_commands_processed', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Enviar valoración de una interacción
    """
    try:
        success = feedback_service.update_rating(
            feedback_id=feedback.feedback_id,
            rating=feedback.rating,
            feedback_text=feedback.feedback_text
        )
        if success:
            return {"status": "success", "message": "Feedback guardado correctamente"}
        else:
            raise HTTPException(status_code=404, detail="Interacción no encontrada")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando feedback: {str(e)}")

@app.get("/metrics", response_model=MetricsResponse, tags=["Analytics"])
async def get_metrics(days: int = 7):
    """
    Obtener métricas de rendimiento de los últimos N días
    """
    try:
        metrics = feedback_service.get_metrics(days=days)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas: {str(e)}")

@app.get("/analytics/low-rated", tags=["Analytics"])
async def get_low_rated_queries(min_rating: int = 2, limit: int = 50):
    """
    Obtener queries con baja valoración para análisis
    """
    try:
        queries = feedback_service.get_low_rated_queries(min_rating=min_rating, limit=limit)
        return {"count": len(queries), "queries": queries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo queries: {str(e)}")

@app.post("/analytics/export-retraining", tags=["Analytics"])
async def export_retraining_data(max_rating: int = 3):
    """
    Exportar datos para reentrenamiento del modelo
    """
    try:
        count = feedback_service.export_for_retraining(max_rating=max_rating)
        return {
            "status": "success",
            "examples_exported": count,
            "file": "retraining_data.jsonl"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando datos: {str(e)}")


# ============ Main ============

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
