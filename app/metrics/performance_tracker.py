"""
Sistema de tracking de métricas automáticas para el chatbot analítico.
"""

import time
import logging
import uuid
import json
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta
from sqlalchemy import text
from app.db.connections import get_postgres

logger = logging.getLogger(__name__)

# ============ Configuración ============

# Pool de métricas para batch inserts (si hay >100 en cola)
_metrics_queue = []
_MAX_QUEUE_SIZE = 100
_BATCH_INSERT_INTERVAL = 5  # segundos

# ============ Creación de Tabla ============

def create_metrics_table() -> None:
    """
    Crea la tabla performance_metrics en PostgreSQL si no existe.
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            component VARCHAR(50) NOT NULL,
            session_id UUID,
            user_query TEXT NOT NULL,
            
            -- Métricas de Router
            intent_predicted VARCHAR(50),
            intent_confidence FLOAT,
            rag_similarity_avg FLOAT,
            
            -- Métricas de SQL
            sql_query TEXT,
            sql_success BOOLEAN,
            sql_error_type VARCHAR(100),
            sql_correction_attempts INT DEFAULT 0,
            rows_returned INT,
            
            -- Métricas de Visualización
            chart_type VARCHAR(50),
            viz_layer_used VARCHAR(20),
            
            -- Métricas generales
            success BOOLEAN NOT NULL,
            latency_ms INT NOT NULL,
            error_message TEXT,
            
            -- Metadata flexible
            metadata JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_metrics_component ON performance_metrics(component);
        CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_metrics_success ON performance_metrics(success);
        CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON performance_metrics(session_id);
        """
        
        session.execute(text(create_table_sql))
        session.commit()
        logger.info("✅ Tabla performance_metrics creada/verificada")
        
    except Exception as e:
        logger.error(f"❌ Error creando tabla performance_metrics: {e}", exc_info=True)
        raise
    finally:
        session.close()


def init_metrics() -> None:
    """
    Inicializa el sistema de métricas.
    Crea la tabla si no existe.
    """
    try:
        create_metrics_table()
        logger.info("✅ Sistema de métricas inicializado")
    except Exception as e:
        logger.error(f"❌ Error inicializando métricas: {e}", exc_info=True)
        # No lanzar excepción, permitir que la app continúe sin métricas


# ============ Funciones de Tracking ============

def _insert_metric(data: Dict[str, Any]) -> None:
    """
    Inserta una métrica en la base de datos.
    Fail-safe: no debe romper el flujo principal.
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        # Preparar datos para inserción
        insert_sql = """
        INSERT INTO performance_metrics (
            timestamp, component, session_id, user_query,
            intent_predicted, intent_confidence, rag_similarity_avg,
            sql_query, sql_success, sql_error_type, sql_correction_attempts, rows_returned,
            chart_type, viz_layer_used,
            success, latency_ms, error_message, metadata
        ) VALUES (
            :timestamp, :component, :session_id, :user_query,
            :intent_predicted, :intent_confidence, :rag_similarity_avg,
            :sql_query, :sql_success, :sql_error_type, :sql_correction_attempts, :rows_returned,
            :chart_type, :viz_layer_used,
            :success, :latency_ms, :error_message, :metadata
        )
        """
        
        # Convertir session_id a UUID si es string
        session_id = data.get('session_id')
        if session_id and isinstance(session_id, str):
            try:
                session_id = uuid.UUID(session_id)
            except ValueError:
                session_id = None
        
        params = {
            'timestamp': data.get('timestamp', datetime.now()),
            'component': data.get('component'),
            'session_id': session_id,
            'user_query': data.get('user_query', '')[:1000],  # Limitar longitud
            'intent_predicted': data.get('intent_predicted'),
            'intent_confidence': data.get('intent_confidence'),
            'rag_similarity_avg': data.get('rag_similarity_avg'),
            'sql_query': data.get('sql_query')[:5000] if data.get('sql_query') else None,  # Limitar longitud
            'sql_success': data.get('sql_success'),
            'sql_error_type': data.get('sql_error_type'),
            'sql_correction_attempts': data.get('sql_correction_attempts', 0),
            'rows_returned': data.get('rows_returned'),
            'chart_type': data.get('chart_type'),
            'viz_layer_used': data.get('viz_layer_used'),
            'success': data.get('success', True),
            'latency_ms': data.get('latency_ms', 0),
            'error_message': data.get('error_message')[:1000] if data.get('error_message') else None,
            'metadata': json.dumps(data.get('metadata') or {})  # Convertir dict a JSON string para JSONB
        }
        
        session.execute(text(insert_sql), params)
        session.commit()
        
    except Exception as e:
        logger.error(f"❌ Error insertando métrica: {e}", exc_info=True)
        # No lanzar excepción, permitir que el flujo continúe
    finally:
        if 'session' in locals():
            session.close()


def track_router_decision(
    query: str,
    intent: str,
    confidence: float,
    rag_similarity: float,
    latency_ms: int,
    session_id: Optional[str] = None,
    error_message: Optional[str] = None
) -> None:
    """
    Registra decisión del router.
    
    Args:
        query: Query del usuario
        intent: Intención predicha
        confidence: Confianza de la predicción (0-1)
        rag_similarity: Similitud promedio de RAG
        latency_ms: Latencia en milisegundos
        session_id: ID de sesión (opcional)
        error_message: Mensaje de error si hubo (opcional)
    """
    try:
        success = confidence > 0.7 if confidence else False
        if error_message:
            success = False
        
        data = {
            'component': 'router',
            'session_id': session_id,
            'user_query': query,
            'intent_predicted': intent,
            'intent_confidence': confidence,
            'rag_similarity_avg': rag_similarity,
            'success': success,
            'latency_ms': latency_ms,
            'error_message': error_message,
            'metadata': {
                'intent': intent,
                'confidence': confidence,
                'rag_similarity': rag_similarity
            }
        }
        
        _insert_metric(data)
        logger.debug(f"📊 Router metric tracked: intent={intent}, confidence={confidence:.2f}, latency={latency_ms}ms")
        
    except Exception as e:
        logger.error(f"❌ Error tracking router decision: {e}", exc_info=True)


def track_sql_execution(
    query: str,
    sql_query: str,
    success: bool,
    latency_ms: int,
    rows_returned: int = 0,
    error_type: Optional[str] = None,
    correction_attempts: int = 0,
    session_id: Optional[str] = None,
    error_message: Optional[str] = None
) -> None:
    """
    Registra ejecución de SQL.
    
    Args:
        query: Query original del usuario
        sql_query: Query SQL generada
        success: Si la ejecución fue exitosa
        latency_ms: Latencia en milisegundos
        rows_returned: Número de filas retornadas
        error_type: Tipo de error (opcional)
        correction_attempts: Número de intentos de corrección
        session_id: ID de sesión (opcional)
        error_message: Mensaje de error (opcional)
    """
    try:
        data = {
            'component': 'sql',
            'session_id': session_id,
            'user_query': query,
            'sql_query': sql_query,
            'sql_success': success,
            'sql_error_type': error_type,
            'sql_correction_attempts': correction_attempts,
            'rows_returned': rows_returned,
            'success': success,
            'latency_ms': latency_ms,
            'error_message': error_message,
            'metadata': {
                'correction_attempts': correction_attempts,
                'rows_returned': rows_returned,
                'error_type': error_type
            }
        }
        
        _insert_metric(data)
        logger.debug(f"📊 SQL metric tracked: success={success}, rows={rows_returned}, latency={latency_ms}ms")
        
    except Exception as e:
        logger.error(f"❌ Error tracking SQL execution: {e}", exc_info=True)


def track_viz_generation(
    query: str,
    chart_type: str,
    layer_used: str,
    success: bool,
    latency_ms: int,
    session_id: Optional[str] = None,
    error_message: Optional[str] = None
) -> None:
    """
    Registra generación de visualización.
    
    Args:
        query: Query original del usuario
        chart_type: Tipo de gráfica generada
        layer_used: Capa usada ('rules', 'finetuned', 'llm')
        success: Si la generación fue exitosa
        latency_ms: Latencia en milisegundos
        session_id: ID de sesión (opcional)
        error_message: Mensaje de error (opcional)
    """
    try:
        data = {
            'component': 'viz',
            'session_id': session_id,
            'user_query': query,
            'chart_type': chart_type,
            'viz_layer_used': layer_used,
            'success': success,
            'latency_ms': latency_ms,
            'error_message': error_message,
            'metadata': {
                'chart_type': chart_type,
                'layer_used': layer_used
            }
        }
        
        _insert_metric(data)
        logger.debug(f"📊 Viz metric tracked: chart={chart_type}, layer={layer_used}, latency={latency_ms}ms")
        
    except Exception as e:
        logger.error(f"❌ Error tracking viz generation: {e}", exc_info=True)


def track_hybrid_execution(
    query: str,
    success: bool,
    latency_ms: int,
    session_id: Optional[str] = None,
    sql_latency: Optional[int] = None,
    viz_latency: Optional[int] = None,
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Registra ejecución híbrida (SQL + KPI + Viz).
    
    Args:
        query: Query original del usuario
        success: Si la ejecución fue exitosa
        latency_ms: Latencia total en milisegundos
        session_id: ID de sesión (opcional)
        sql_latency: Latencia de SQL (opcional)
        viz_latency: Latencia de visualización (opcional)
        error_message: Mensaje de error (opcional)
        metadata: Metadata adicional (opcional)
    """
    try:
        data = {
            'component': 'hybrid',
            'session_id': session_id,
            'user_query': query,
            'success': success,
            'latency_ms': latency_ms,
            'error_message': error_message,
            'metadata': {
                'sql_latency': sql_latency,
                'viz_latency': viz_latency,
                **(metadata or {})
            }
        }
        
        _insert_metric(data)
        logger.debug(f"📊 Hybrid metric tracked: success={success}, latency={latency_ms}ms")
        
    except Exception as e:
        logger.error(f"❌ Error tracking hybrid execution: {e}", exc_info=True)


# ============ Funciones de Consulta de Métricas ============

def get_component_metrics(
    component: str,
    days: int = 7
) -> Dict[str, Any]:
    """
    Obtiene métricas agregadas de un componente.
    
    Args:
        component: Componente ('router', 'sql', 'viz', 'hybrid')
        days: Número de días hacia atrás
    
    Returns:
        Dict con métricas agregadas
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Query principal
        query_sql = """
        SELECT 
            COUNT(*) as total_requests,
            COUNT(*) FILTER (WHERE success = true) as successful_requests,
            AVG(latency_ms) as avg_latency_ms,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50_latency,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency
        FROM performance_metrics
        WHERE component = :component
        AND timestamp >= :cutoff_date
        """
        
        result = session.execute(
            text(query_sql),
            {'component': component, 'cutoff_date': cutoff_date}
        ).fetchone()
        
        if not result or result[0] is None:
            return {
                "component": component,
                "period_days": days,
                "total_requests": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0,
                "error_breakdown": {},
                "p50_latency": 0,
                "p95_latency": 0,
                "p99_latency": 0
            }
        
        total_requests = result[0] or 0
        successful_requests = result[1] or 0
        success_rate = (successful_requests / total_requests) if total_requests > 0 else 0.0
        
        # Error breakdown (solo para router y sql)
        error_breakdown = {}
        if component in ['router', 'sql']:
            error_query = """
            SELECT 
                CASE 
                    WHEN error_message IS NOT NULL THEN 'other'
                    WHEN component = 'router' AND intent_confidence < 0.7 THEN 'low_confidence'
                    WHEN component = 'sql' AND sql_error_type IS NOT NULL THEN sql_error_type
                    ELSE 'other'
                END as error_type,
                COUNT(*) as count
            FROM performance_metrics
            WHERE component = :component
            AND timestamp >= :cutoff_date
            AND success = false
            GROUP BY error_type
            """
            
            error_results = session.execute(
                text(error_query),
                {'component': component, 'cutoff_date': cutoff_date}
            ).fetchall()
            
            error_breakdown = {row[0]: row[1] for row in error_results if row[0]}
        
        metrics = {
            "component": component,
            "period_days": days,
            "total_requests": int(total_requests),
            "success_rate": round(success_rate, 3),
            "avg_latency_ms": int(result[2] or 0),
            "error_breakdown": error_breakdown,
            "p50_latency": int(result[3] or 0),
            "p95_latency": int(result[4] or 0),
            "p99_latency": int(result[5] or 0)
        }
        
        session.close()
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo métricas de {component}: {e}", exc_info=True)
        return {
            "component": component,
            "period_days": days,
            "error": str(e)
        }


def get_all_metrics_summary(days: int = 7) -> Dict[str, Any]:
    """
    Resumen de métricas de todos los componentes.
    
    Args:
        days: Número de días hacia atrás
    
    Returns:
        Dict con resumen de todos los componentes
    """
    try:
        components = ['router', 'sql', 'viz', 'hybrid']
        summary = {}
        
        for component in components:
            summary[component] = get_component_metrics(component, days)
        
        # Calcular métricas overall
        postgres = get_postgres()
        session = postgres.get_session()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        overall_query = """
        SELECT 
            COUNT(*) as total_queries,
            COUNT(*) FILTER (WHERE success = true) as successful_queries,
            AVG(latency_ms) as avg_latency
        FROM performance_metrics
        WHERE timestamp >= :cutoff_date
        """
        
        result = session.execute(
            text(overall_query),
            {'cutoff_date': cutoff_date}
        ).fetchone()
        
        total_queries = result[0] or 0
        successful_queries = result[1] or 0
        overall_success_rate = (successful_queries / total_queries) if total_queries > 0 else 0.0
        
        summary['overall'] = {
            "total_queries": int(total_queries),
            "overall_success_rate": round(overall_success_rate, 3),
            "avg_end_to_end_latency": int(result[2] or 0)
        }
        
        session.close()
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo resumen de métricas: {e}", exc_info=True)
        return {"error": str(e)}


# ============ Decorador para Tracking Automático ============

def track_performance(component: str):
    """
    Decorador que trackea automáticamente el rendimiento de una función.
    
    Args:
        component: Componente a trackear ('router', 'sql', 'viz', 'hybrid')
    
    Usage:
        @track_performance('router')
        def router_node(state) -> state:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state):
            start_time = time.time()
            success = True
            error_msg = None
            result = None
            
            try:
                result = func(state)
                
                # Detectar si hubo error en el state
                if result and result.get('error'):
                    success = False
                    error_msg = result['error']
                
                return result
                
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
                
            finally:
                try:
                    latency_ms = int((time.time() - start_time) * 1000)
                    session_id = state.get('session_id') if state else None
                    query = state.get('user_query', '') if state else ''
                    
                    # Trackear según componente
                    if component == "router":
                        intent = result.get('intent', 'unknown') if result else 'unknown'
                        confidence = result.get('confidence', 0.0) if result else 0.0
                        rag_similarity = result.get('rag_similarity', 0.0) if result else 0.0
                        
                        track_router_decision(
                            query=query,
                            intent=intent,
                            confidence=confidence,
                            rag_similarity=rag_similarity,
                            latency_ms=latency_ms,
                            session_id=session_id,
                            error_message=error_msg
                        )
                    
                    elif component == "sql":
                        sql_query = result.get('sql_query', '') if result else ''
                        rows_returned = len(result.get('sql_results', [])) if result else 0
                        correction_attempts = result.get('sql_correction_attempts', 0) if result else 0
                        
                        track_sql_execution(
                            query=query,
                            sql_query=sql_query,
                            success=success,
                            latency_ms=latency_ms,
                            rows_returned=rows_returned,
                            correction_attempts=correction_attempts,
                            session_id=session_id,
                            error_message=error_msg
                        )
                    
                    elif component == "viz":
                        chart_type = result.get('chart_config', {}).get('chart_type', 'unknown') if result else 'unknown'
                        layer_used = result.get('chart_config', {}).get('source', 'unknown') if result else 'unknown'
                        
                        track_viz_generation(
                            query=query,
                            chart_type=chart_type,
                            layer_used=layer_used,
                            success=success,
                            latency_ms=latency_ms,
                            session_id=session_id,
                            error_message=error_msg
                        )
                    
                    elif component == "hybrid":
                        track_hybrid_execution(
                            query=query,
                            success=success,
                            latency_ms=latency_ms,
                            session_id=session_id,
                            error_message=error_msg
                        )
                    
                except Exception as e:
                    # No debe romper el flujo principal
                    logger.error(f"❌ Error en tracking automático: {e}", exc_info=True)
        
        return wrapper
    return decorator


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing Performance Tracker ===\n")
    
    # Test 1: Inicializar
    print("1. Inicializando métricas...")
    init_metrics()
    
    # Test 2: Track router
    print("\n2. Tracking router decision...")
    track_router_decision(
        query="¿Cuántas ventas hay?",
        intent="sql",
        confidence=0.85,
        rag_similarity=0.78,
        latency_ms=1200
    )
    
    # Test 3: Track SQL
    print("\n3. Tracking SQL execution...")
    track_sql_execution(
        query="¿Cuántas ventas hay?",
        sql_query="SELECT COUNT(*) FROM ventas",
        success=True,
        latency_ms=800,
        rows_returned=1
    )
    
    # Test 4: Obtener métricas
    print("\n4. Obteniendo métricas...")
    router_metrics = get_component_metrics('router', days=1)
    print(f"   Router metrics: {router_metrics}")
    
    print("\n✅ Tests completados")

