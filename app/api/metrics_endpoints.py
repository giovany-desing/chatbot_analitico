"""
Endpoints REST para consultar métricas del sistema.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import text

from fastapi import HTTPException, Query
from app.db.connections import get_postgres
from app.metrics.performance_tracker import get_component_metrics
from app.metrics.alerts import (
    get_active_alerts,
    run_health_check,
    _get_sql_correction_rate,
    _get_viz_llm_fallback_rate
)

logger = logging.getLogger(__name__)


# ============ Funciones Helper ============

def get_viz_layer_distribution(days: int) -> Dict[str, int]:
    """
    Cuenta cuántas veces se usó cada capa del sistema híbrido de visualización.
    
    Args:
        days: Número de días hacia atrás
    
    Returns:
        Dict con distribución por capa (rules, finetuned, llm)
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        sql = """
        SELECT 
            viz_layer_used, 
            COUNT(*) as count
        FROM performance_metrics
        WHERE component IN ('viz', 'hybrid')
            AND timestamp >= :cutoff_date
            AND viz_layer_used IS NOT NULL
        GROUP BY viz_layer_used
        """
        
        result = session.execute(
            text(sql),
            {'cutoff_date': cutoff_date}
        ).fetchall()
        
        session.close()
        
        distribution = {row[0]: int(row[1]) for row in result}
        
        # Asegurar que todas las capas estén presentes
        for layer in ['rules', 'finetuned', 'llm']:
            if layer not in distribution:
                distribution[layer] = 0
        
        return distribution
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo distribución de capas: {e}", exc_info=True)
        return {'rules': 0, 'finetuned': 0, 'llm': 0}


def get_daily_aggregated_metrics(days: int) -> List[Dict]:
    """
    Métricas agregadas por día.
    
    Args:
        days: Número de días hacia atrás
    
    Returns:
        Lista de dicts con métricas diarias
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        sql = """
        SELECT
            DATE(timestamp) as date,
            COUNT(*) as total_requests,
            AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) as success_rate,
            AVG(latency_ms) as avg_latency,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency
        FROM performance_metrics
        WHERE timestamp >= :cutoff_date
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        """
        
        result = session.execute(
            text(sql),
            {'cutoff_date': cutoff_date}
        ).fetchall()
        
        session.close()
        
        daily_metrics = []
        for row in result:
            date_obj = row[0]
            if isinstance(date_obj, datetime):
                date_str = date_obj.isoformat()
            else:
                date_str = str(date_obj)
            
            daily_metrics.append({
                "date": date_str,
                "total_requests": int(row[1] or 0),
                "success_rate": float(row[2] or 0.0),
                "avg_latency": float(row[3] or 0.0),
                "p95_latency": float(row[4] or 0.0)
            })
        
        return daily_metrics
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo métricas diarias: {e}", exc_info=True)
        return []


def calculate_trend(daily_data: List[Dict], metric: str) -> str:
    """
    Calcula tendencia: 'improving' | 'stable' | 'degrading' | 'insufficient_data'
    
    Args:
        daily_data: Lista de métricas diarias
        metric: Nombre de la métrica a analizar
    
    Returns:
        String con la tendencia
    """
    if len(daily_data) < 3:
        return "insufficient_data"
    
    # Usar últimos 7 días para calcular tendencia
    recent_data = daily_data[:7] if len(daily_data) >= 7 else daily_data
    
    values = [d.get(metric, 0) for d in recent_data if d.get(metric) is not None]
    
    if len(values) < 2:
        return "insufficient_data"
    
    # Regresión lineal simple
    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    
    numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return "stable"
    
    slope = numerator / denominator
    
    # Normalizar slope por el valor promedio para obtener cambio porcentual
    if y_mean > 0:
        slope_pct = slope / y_mean
    else:
        slope_pct = 0
    
    # Determinar tendencia según métrica
    if metric in ['success_rate']:
        # Para success_rate, slope positivo es mejor
        if slope_pct > 0.01:  # Mejoró >1%
            return "improving"
        elif slope_pct < -0.01:  # Empeoró >1%
            return "degrading"
    else:  # latency
        # Para latency, slope negativo es mejor
        if slope_pct < -0.05:  # Mejoró >5%
            return "improving"
        elif slope_pct > 0.05:  # Empeoró >5%
            return "degrading"
    
    return "stable"


# ============ Funciones de Endpoints ============

async def get_router_metrics_endpoint(days: int = 7):
    """Métricas del componente router"""
    try:
        metrics = get_component_metrics("router", days)
        
        status = "healthy"
        if metrics.get('success_rate', 0) < 0.9:
            status = "degraded"
        elif metrics.get('p95_latency', 0) > 5000:
            status = "degraded"
        
        return {
            "component": "router",
            "period_days": days,
            "metrics": metrics,
            "status": status
        }
    except Exception as e:
        logger.error(f"Error obteniendo métricas de router: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas: {str(e)}")


async def get_sql_metrics_endpoint(days: int = 7):
    """Métricas del componente SQL"""
    try:
        metrics = get_component_metrics("sql", days)
        
        # Obtener métricas adicionales
        correction_rate = _get_sql_correction_rate("sql", days, offset_days=0)
        
        # Calcular promedio de filas retornadas
        try:
            postgres = get_postgres()
            session = postgres.get_session()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            sql = """
            SELECT AVG(rows_returned) as avg_rows
            FROM performance_metrics
            WHERE component = 'sql'
                AND timestamp >= :cutoff_date
                AND rows_returned IS NOT NULL
            """
            
            result = session.execute(
                text(sql),
                {'cutoff_date': cutoff_date}
            ).fetchone()
            
            session.close()
            
            avg_rows = float(result[0] or 0) if result else 0.0
        except Exception as e:
            logger.warning(f"Error calculando avg_rows: {e}")
            avg_rows = 0.0
        
        return {
            "component": "sql",
            "period_days": days,
            "metrics": metrics,
            "insights": {
                "correction_rate": correction_rate,
                "avg_rows_returned": avg_rows,
                "self_correction_working": correction_rate > 0 and correction_rate < 0.3
            }
        }
    except Exception as e:
        logger.error(f"Error obteniendo métricas de SQL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas: {str(e)}")


async def get_viz_metrics_endpoint(days: int = 7):
    """Métricas del componente visualización"""
    try:
        metrics = get_component_metrics("viz", days)
        
        # Breakdown por capa
        layer_distribution = get_viz_layer_distribution(days)
        
        total = sum(layer_distribution.values())
        
        insights = {}
        if total > 0:
            insights = {
                "finetuned_usage_rate": layer_distribution.get('finetuned', 0) / total,
                "llm_fallback_rate": layer_distribution.get('llm', 0) / total,
                "rules_usage_rate": layer_distribution.get('rules', 0) / total
            }
        else:
            insights = {
                "finetuned_usage_rate": 0.0,
                "llm_fallback_rate": 0.0,
                "rules_usage_rate": 0.0
            }
        
        return {
            "component": "viz",
            "period_days": days,
            "metrics": metrics,
            "layer_distribution": layer_distribution,
            "insights": insights
        }
    except Exception as e:
        logger.error(f"Error obteniendo métricas de Viz: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas: {str(e)}")


async def get_system_health_endpoint():
    """Estado general de salud del sistema"""
    try:
        health_report = run_health_check()
        return health_report
    except Exception as e:
        logger.error(f"Error ejecutando health check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error ejecutando health check: {str(e)}")


async def get_active_alerts_endpoint():
    """Alertas activas del sistema"""
    try:
        alerts = get_active_alerts()
        
        # Convertir Alert objects a dicts si es necesario
        alerts_dict = []
        for alert in alerts:
            if isinstance(alert, dict):
                alerts_dict.append(alert)
            else:
                alerts_dict.append({
                    "severity": alert.severity,
                    "component": alert.component,
                    "metric": alert.metric,
                    "message": alert.message,
                    "current_value": alert.current_value,
                    "previous_value": alert.previous_value,
                    "threshold": alert.threshold,
                    "recommendation": alert.recommendation,
                    "created_at": alert.created_at.isoformat() if hasattr(alert.created_at, 'isoformat') else str(alert.created_at)
                })
        
        return {
            "total_alerts": len(alerts_dict),
            "critical_count": sum(1 for a in alerts_dict if a.get('severity') == 'critical'),
            "warning_count": sum(1 for a in alerts_dict if a.get('severity') == 'warning'),
            "info_count": sum(1 for a in alerts_dict if a.get('severity') == 'info'),
            "alerts": alerts_dict
        }
    except Exception as e:
        logger.error(f"Error obteniendo alertas: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error obteniendo alertas: {str(e)}")


async def get_performance_trends_endpoint(days: int = 30):
    """Tendencias de performance en el tiempo"""
    try:
        daily_metrics = get_daily_aggregated_metrics(days)
        
        # Calcular tendencias
        success_rate_trend = "insufficient_data"
        latency_trend = "insufficient_data"
        
        if daily_metrics:
            success_rate_trend = calculate_trend(daily_metrics, 'success_rate')
            latency_trend = calculate_trend(daily_metrics, 'avg_latency')
        
        return {
            "period_days": days,
            "daily_metrics": daily_metrics,
            "trends": {
                "success_rate_trend": success_rate_trend,
                "latency_trend": latency_trend
            }
        }
    except Exception as e:
        logger.error(f"Error obteniendo tendencias: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error obteniendo tendencias: {str(e)}")


async def trigger_export_endpoint(days: int = 7, min_confidence: float = 0.8):
    """Trigger manual de exportación para reentrenamiento"""
    try:
        from scripts.auto_export_training_data import export_training_data
        
        output_file = export_training_data(
            days=days,
            min_confidence=min_confidence,
            output_dir="data/retraining"
        )
        
        # Contar líneas del archivo
        sample_count = 0
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                sample_count = sum(1 for _ in f)
        except Exception as e:
            logger.warning(f"Error contando muestras: {e}")
        
        return {
            "success": True,
            "output_file": output_file,
            "sample_count": sample_count,
            "period_days": days,
            "min_confidence": min_confidence
        }
    except Exception as e:
        logger.error(f"Error en exportación: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


async def get_notifications_history_endpoint():
    """Historial de notificaciones enviadas"""
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        # Total enviadas
        total_sql = """
        SELECT COUNT(*) as total
        FROM alert_history
        """
        total_result = session.execute(text(total_sql)).fetchone()
        total_sent = total_result[0] if total_result else 0
        
        # Por canal
        by_channel_sql = """
        SELECT 
            jsonb_array_elements_text(channels) as channel,
            COUNT(*) as count
        FROM alert_history
        GROUP BY channel
        """
        by_channel_results = session.execute(text(by_channel_sql)).fetchall()
        by_channel = {row[0]: int(row[1]) for row in by_channel_results}
        
        # Por severidad
        by_severity_sql = """
        SELECT 
            severity,
            COUNT(*) as count
        FROM alert_history
        GROUP BY severity
        """
        by_severity_results = session.execute(text(by_severity_sql)).fetchall()
        by_severity = {row[0]: int(row[1]) for row in by_severity_results}
        
        # Fallos de entrega
        failed_sql = """
        SELECT COUNT(*) as failed
        FROM alert_history
        WHERE delivery_status::text LIKE '%false%'
        """
        failed_result = session.execute(text(failed_sql)).fetchone()
        failed_deliveries = failed_result[0] if failed_result else 0
        
        # Últimas 24 horas
        last_24h_sql = """
        SELECT 
            sent_at,
            component,
            metric,
            severity,
            channels,
            delivery_status
        FROM alert_history
        WHERE sent_at >= NOW() - INTERVAL '24 hours'
        ORDER BY sent_at DESC
        LIMIT 50
        """
        last_24h_results = session.execute(text(last_24h_sql)).fetchall()
        
        last_24h = []
        for row in last_24h_results:
            import json
            last_24h.append({
                "sent_at": row[0].isoformat() if hasattr(row[0], 'isoformat') else str(row[0]),
                "component": row[1],
                "metric": row[2],
                "severity": row[3],
                "channels": json.loads(row[4]) if isinstance(row[4], str) else row[4],
                "delivery_status": json.loads(row[5]) if isinstance(row[5], str) else row[5]
            })
        
        session.close()
        
        return {
            "total_sent": total_sent,
            "by_channel": by_channel,
            "by_severity": by_severity,
            "failed_deliveries": failed_deliveries,
            "last_24h": last_24h
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo historial de notificaciones: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error obteniendo historial: {str(e)}")


# ============ Función para Registrar Endpoints ============

def register_metrics_endpoints(fastapi_app):
    """
    Registra todos los endpoints de métricas en la app FastAPI.
    Se llama desde app/main.py después de que 'app' esté definido.
    """
    @fastapi_app.get("/metrics/router", tags=["Metrics"])
    async def get_router_metrics(
        days: int = Query(7, ge=1, le=365, description="Número de días hacia atrás")
    ):
        """
        Métricas del componente router.
        
        Retorna información sobre el rendimiento del router de intenciones:
        - Success rate
        - Latencia (p50, p95, p99)
        - Distribución de errores
        """
        return await get_router_metrics_endpoint(days)
    
    @fastapi_app.get("/metrics/sql", tags=["Metrics"])
    async def get_sql_metrics(
        days: int = Query(7, ge=1, le=365, description="Número de días hacia atrás")
    ):
        """
        Métricas del componente SQL.
        
        Retorna información sobre el rendimiento de generación y ejecución SQL:
        - Success rate
        - Latencia
        - Tasa de corrección automática
        - Promedio de filas retornadas
        """
        return await get_sql_metrics_endpoint(days)
    
    @fastapi_app.get("/metrics/viz", tags=["Metrics"])
    async def get_viz_metrics(
        days: int = Query(7, ge=1, le=365, description="Número de días hacia atrás")
    ):
        """
        Métricas del componente visualización.
        
        Retorna información sobre el rendimiento de generación de visualizaciones:
        - Success rate
        - Latencia
        - Distribución por capa (rules, finetuned, llm)
        - Tasa de uso de cada capa
        """
        return await get_viz_metrics_endpoint(days)
    
    @fastapi_app.get("/metrics/health", tags=["Metrics"])
    async def get_system_health():
        """
        Estado general de salud del sistema.
        
        Ejecuta un health check completo y retorna:
        - Estado general (healthy, warning, critical)
        - Estado de cada componente
        - Alertas activas
        """
        return await get_system_health_endpoint()
    
    @fastapi_app.get("/metrics/alerts", tags=["Metrics"])
    async def get_active_alerts():
        """
        Alertas activas del sistema.
        
        Retorna todas las alertas actuales ordenadas por severidad:
        - Alertas críticas
        - Alertas de advertencia
        - Alertas informativas
        """
        return await get_active_alerts_endpoint()
    
    @fastapi_app.get("/metrics/trends", tags=["Metrics"])
    async def get_performance_trends(
        days: int = Query(30, ge=7, le=365, description="Número de días hacia atrás (mínimo 7)")
    ):
        """
        Tendencias de performance en el tiempo.
        
        Analiza métricas diarias y calcula tendencias:
        - Success rate trend (improving/stable/degrading)
        - Latency trend (improving/stable/degrading)
        - Métricas diarias agregadas
        """
        return await get_performance_trends_endpoint(days)
    
    @fastapi_app.get("/metrics/export", tags=["Metrics"])
    async def trigger_export(
        days: int = Query(7, ge=1, le=90, description="Número de días hacia atrás"),
        min_confidence: float = Query(0.8, ge=0.0, le=1.0, description="Confianza mínima requerida")
    ):
        """
        Trigger manual de exportación para reentrenamiento.
        
        Ejecuta el script de exportación de datos de entrenamiento y retorna:
        - Archivo generado
        - Número de ejemplos exportados
        - Período de datos
        
        **Nota:** Este endpoint puede tardar varios segundos en ejecutarse.
        """
        return await trigger_export_endpoint(days, min_confidence)
    
    @fastapi_app.get("/metrics/notifications/history", tags=["Metrics"])
    async def get_notifications_history():
        """
        Historial de notificaciones enviadas.
        
        Retorna estadísticas de notificaciones:
        - Total enviadas
        - Por canal (slack, email)
        - Por severidad (critical, warning)
        - Fallos de entrega
        - Últimas 24 horas
        """
        return await get_notifications_history_endpoint()
