"""
Sistema de alertas por degradación de métricas.
Detecta proactivamente problemas antes que afecten usuarios.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy import text

from app.metrics.performance_tracker import get_component_metrics
from app.db.connections import get_postgres

logger = logging.getLogger(__name__)

# ============ Umbrales de Salud ============

THRESHOLDS = {
    "router": {
        "success_rate_critical": 0.85,
        "success_rate_warning": 0.90,
        "latency_warning_ms": 3000,
        "latency_critical_ms": 5000,
    },
    "sql": {
        "success_rate_critical": 0.80,
        "success_rate_warning": 0.88,
        "latency_warning_ms": 5000,
        "latency_critical_ms": 10000,
        "correction_rate_warning": 0.30,  # >30% queries necesitan corrección
    },
    "viz": {
        "success_rate_critical": 0.85,
        "success_rate_warning": 0.92,
        "latency_warning_ms": 4000,
        "latency_critical_ms": 8000,
        "llm_fallback_rate_warning": 0.40,  # >40% usa LLM en vez de modelo
    },
    "hybrid": {
        "success_rate_critical": 0.80,
        "success_rate_warning": 0.88,
        "latency_warning_ms": 6000,
        "latency_critical_ms": 12000,
    }
}

# ============ Dataclasses ============

@dataclass
class Alert:
    """Representa una alerta del sistema."""
    severity: str  # 'critical' | 'warning' | 'info'
    component: str
    metric: str
    message: str
    current_value: float
    previous_value: Optional[float] = None
    threshold: Optional[float] = None
    recommendation: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HealthStatus:
    """Estado de salud de un componente."""
    component: str
    status: str  # 'healthy' | 'warning' | 'critical'
    alerts: List[Alert]
    metrics: Dict
    checked_at: datetime


# ============ Funciones Helper para Métricas Adicionales ============

def _get_sql_correction_rate(component: str, days: int, offset_days: int = 0) -> float:
    """
    Calcula el porcentaje de queries SQL que necesitaron corrección.
    
    Args:
        component: Componente ('sql')
        days: Número de días hacia atrás
        offset_days: Días de offset (para comparar períodos anteriores)
    
    Returns:
        Porcentaje de queries con correction_attempts > 0
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        end_date = datetime.now() - timedelta(days=offset_days)
        start_date = end_date - timedelta(days=days)
        
        query_sql = """
        SELECT 
            COUNT(*) as total_queries,
            COUNT(*) FILTER (WHERE sql_correction_attempts > 0) as corrected_queries
        FROM performance_metrics
        WHERE component = :component
        AND timestamp >= :start_date
        AND timestamp < :end_date
        """
        
        result = session.execute(
            text(query_sql),
            {
                'component': component,
                'start_date': start_date,
                'end_date': end_date
            }
        ).fetchone()
        
        session.close()
        
        if not result or result[0] is None or result[0] == 0:
            return 0.0
        
        total = result[0]
        corrected = result[1] or 0
        return corrected / total
        
    except Exception as e:
        logger.error(f"❌ Error calculando correction_rate: {e}", exc_info=True)
        return 0.0


def _get_viz_llm_fallback_rate(component: str, days: int, offset_days: int = 0) -> float:
    """
    Calcula el porcentaje de visualizaciones que usaron LLM (fallback).
    
    Args:
        component: Componente ('viz')
        days: Número de días hacia atrás
        offset_days: Días de offset (para comparar períodos anteriores)
    
    Returns:
        Porcentaje de visualizaciones con viz_layer_used = 'llm'
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        end_date = datetime.now() - timedelta(days=offset_days)
        start_date = end_date - timedelta(days=days)
        
        query_sql = """
        SELECT 
            COUNT(*) as total_viz,
            COUNT(*) FILTER (WHERE viz_layer_used = 'llm') as llm_viz
        FROM performance_metrics
        WHERE component = :component
        AND timestamp >= :start_date
        AND timestamp < :end_date
        """
        
        result = session.execute(
            text(query_sql),
            {
                'component': component,
                'start_date': start_date,
                'end_date': end_date
            }
        ).fetchone()
        
        session.close()
        
        if not result or result[0] is None or result[0] == 0:
            return 0.0
        
        total = result[0]
        llm_count = result[1] or 0
        return llm_count / total
        
    except Exception as e:
        logger.error(f"❌ Error calculando llm_fallback_rate: {e}", exc_info=True)
        return 0.0


def _get_component_metrics_with_offset(
    component: str,
    days: int,
    offset_days: int = 0
) -> Dict[str, Any]:
    """
    Obtiene métricas de un componente con soporte para offset.
    Similar a get_component_metrics pero permite especificar un offset.
    
    Args:
        component: Componente ('router', 'sql', 'viz', 'hybrid')
        days: Número de días hacia atrás
        offset_days: Días de offset (para comparar períodos anteriores)
    
    Returns:
        Dict con métricas agregadas
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()
        
        end_date = datetime.now() - timedelta(days=offset_days)
        start_date = end_date - timedelta(days=days)
        
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
        AND timestamp >= :start_date
        AND timestamp < :end_date
        """
        
        result = session.execute(
            text(query_sql),
            {
                'component': component,
                'start_date': start_date,
                'end_date': end_date
            }
        ).fetchone()
        
        if not result or result[0] is None:
            session.close()
            return {
                "component": component,
                "period_days": days,
                "total_requests": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0,
                "p50_latency": 0,
                "p95_latency": 0,
                "p99_latency": 0
            }
        
        total_requests = result[0] or 0
        successful_requests = result[1] or 0
        success_rate = (successful_requests / total_requests) if total_requests > 0 else 0.0
        
        metrics = {
            "component": component,
            "period_days": days,
            "total_requests": int(total_requests),
            "success_rate": round(success_rate, 3),
            "avg_latency_ms": int(result[2] or 0),
            "p50_latency": int(result[3] or 0),
            "p95_latency": int(result[4] or 0),
            "p99_latency": int(result[5] or 0)
        }
        
        # Agregar métricas específicas por componente
        if component == "sql":
            metrics['correction_rate'] = _get_sql_correction_rate(component, days, offset_days)
        
        if component == "viz":
            metrics['llm_fallback_rate'] = _get_viz_llm_fallback_rate(component, days, offset_days)
        
        session.close()
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo métricas de {component}: {e}", exc_info=True)
        return {
            "component": component,
            "period_days": days,
            "error": str(e)
        }


# ============ Función de Evaluación de Salud ============

def check_component_health(component: str, days: int = 1) -> HealthStatus:
    """
    Evalúa salud de un componente.
    
    Args:
        component: Componente a evaluar ('router', 'sql', 'viz', 'hybrid')
        days: Número de días hacia atrás para evaluar
    
    Returns:
        HealthStatus con estado y alertas
    """
    try:
        # Obtener métricas básicas
        metrics = get_component_metrics(component, days)
        
        # Agregar métricas adicionales específicas del componente
        if component == "sql":
            metrics['correction_rate'] = _get_sql_correction_rate(component, days, offset_days=0)
        
        if component == "viz":
            metrics['llm_fallback_rate'] = _get_viz_llm_fallback_rate(component, days, offset_days=0)
        
        # Verificar si el componente tiene umbrales definidos
        if component not in THRESHOLDS:
            logger.warning(f"⚠️ Componente {component} no tiene umbrales definidos")
            return HealthStatus(
                component=component,
                status="healthy",
                alerts=[],
                metrics=metrics,
                checked_at=datetime.now()
            )
        
        thresholds = THRESHOLDS[component]
        alerts = []
        status = "healthy"  # healthy | warning | critical
        
        # Check success rate
        success_rate = metrics.get('success_rate', 0.0)
        if success_rate < thresholds.get('success_rate_critical', 1.0):
            alerts.append(Alert(
                severity="critical",
                component=component,
                metric="success_rate",
                value=success_rate,
                threshold=thresholds['success_rate_critical'],
                message=f"{component} success rate crítico: {success_rate:.2%}",
                recommendation="Revisar logs de errores y validar cambios recientes"
            ))
            status = "critical"
        
        elif success_rate < thresholds.get('success_rate_warning', 1.0):
            alerts.append(Alert(
                severity="warning",
                component=component,
                metric="success_rate",
                value=success_rate,
                threshold=thresholds['success_rate_warning'],
                message=f"{component} success rate bajo: {success_rate:.2%}",
                recommendation="Monitorear tendencia y revisar posibles causas"
            ))
            if status != "critical":
                status = "warning"
        
        # Check latency (p95)
        p95_latency = metrics.get('p95_latency', 0)
        if p95_latency > thresholds.get('latency_critical_ms', float('inf')):
            alerts.append(Alert(
                severity="critical",
                component=component,
                metric="p95_latency",
                value=p95_latency,
                threshold=thresholds['latency_critical_ms'],
                message=f"{component} latencia crítica: {p95_latency}ms (p95)",
                recommendation="Optimizar queries o revisar carga del sistema"
            ))
            status = "critical"
        
        elif p95_latency > thresholds.get('latency_warning_ms', float('inf')):
            alerts.append(Alert(
                severity="warning",
                component=component,
                metric="p95_latency",
                value=p95_latency,
                threshold=thresholds['latency_warning_ms'],
                message=f"{component} latencia alta: {p95_latency}ms (p95)",
                recommendation="Monitorear tendencia de latencia"
            ))
            if status != "critical":
                status = "warning"
        
        # Component-specific checks
        if component == "sql":
            correction_rate = metrics.get('correction_rate', 0.0)
            correction_threshold = thresholds.get('correction_rate_warning', 1.0)
            if correction_rate > correction_threshold:
                alerts.append(Alert(
                    severity="warning",
                    component=component,
                    metric="correction_rate",
                    value=correction_rate,
                    threshold=correction_threshold,
                    message=f"SQL: {correction_rate:.1%} queries necesitan corrección",
                    recommendation="Revisar calidad de generación SQL y mejorar ejemplos de RAG"
                ))
                if status != "critical":
                    status = "warning"
        
        if component == "viz":
            llm_rate = metrics.get('llm_fallback_rate', 0.0)
            llm_threshold = thresholds.get('llm_fallback_rate_warning', 1.0)
            if llm_rate > llm_threshold:
                alerts.append(Alert(
                    severity="warning",
                    component=component,
                    metric="llm_fallback_rate",
                    value=llm_rate,
                    threshold=llm_threshold,
                    message=f"Viz: {llm_rate:.1%} usa LLM (modelo fine-tuned no usándose)",
                    recommendation="Revisar modelo fine-tuned y mejorar cobertura de reglas"
                ))
                if status != "critical":
                    status = "warning"
        
        return HealthStatus(
            component=component,
            status=status,
            alerts=alerts,
            metrics=metrics,
            checked_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"❌ Error evaluando salud de {component}: {e}", exc_info=True)
        return HealthStatus(
            component=component,
            status="healthy",
            alerts=[Alert(
                severity="warning",
                component=component,
                metric="health_check_error",
                value=0.0,
                message=f"Error evaluando salud de {component}: {str(e)}",
                recommendation="Revisar sistema de métricas"
            )],
            metrics={},
            checked_at=datetime.now()
        )


# ============ Función de Detección de Degradación ============

def detect_degradation(component: str) -> Optional[Alert]:
    """
    Compara métricas actuales vs semana pasada.
    Detecta degradación significativa.
    
    Args:
        component: Componente a evaluar
    
    Returns:
        Alert si hay degradación, None en caso contrario
    """
    try:
        # Métricas actuales (último día)
        current = _get_component_metrics_with_offset(component, days=1, offset_days=0)
        
        # Métricas de la semana pasada (días 7-14)
        previous = _get_component_metrics_with_offset(component, days=7, offset_days=7)
        
        # Validar que hay datos suficientes
        if current.get('total_requests', 0) == 0 or previous.get('total_requests', 0) == 0:
            return None
        
        # Comparar success rate
        current_success = current.get('success_rate', 0.0)
        previous_success = previous.get('success_rate', 0.0)
        success_delta = current_success - previous_success
        
        if success_delta < -0.05:  # Bajó >5%
            return Alert(
                severity="critical",
                component=component,
                metric="success_rate_degradation",
                message=f"{component}: Success rate bajó {abs(success_delta):.1%} vs semana pasada ({current_success:.1%} vs {previous_success:.1%})",
                current_value=current_success,
                previous_value=previous_success,
                recommendation="Revisar cambios recientes en código o datos"
            )
        
        # Comparar latency
        current_latency = current.get('avg_latency_ms', 0)
        previous_latency = previous.get('avg_latency_ms', 0)
        
        if previous_latency > 0:
            latency_delta_pct = (current_latency - previous_latency) / previous_latency
            
            if latency_delta_pct > 0.5:  # Subió >50%
                return Alert(
                    severity="warning",
                    component=component,
                    metric="latency_increase",
                    message=f"{component}: Latencia aumentó {latency_delta_pct:.0%} vs semana pasada ({current_latency}ms vs {previous_latency}ms)",
                    current_value=current_latency,
                    previous_value=previous_latency,
                    recommendation="Verificar carga del sistema o optimizar queries"
                )
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Error detectando degradación de {component}: {e}", exc_info=True)
        return None


# ============ Sistema de Alertas Activas ============

def get_active_alerts() -> List[Alert]:
    """
    Retorna todas las alertas activas del sistema.
    
    Returns:
        Lista de Alert ordenadas por severidad (critical primero)
    """
    all_alerts = []
    
    try:
        for component in ['router', 'sql', 'viz', 'hybrid']:
            # Health check
            health = check_component_health(component)
            all_alerts.extend(health.alerts)
            
            # Degradation check
            degradation = detect_degradation(component)
            if degradation:
                all_alerts.append(degradation)
    
    except Exception as e:
        logger.error(f"❌ Error obteniendo alertas activas: {e}", exc_info=True)
    
    # Ordenar por severidad: critical > warning > info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(all_alerts, key=lambda x: severity_order.get(x.severity, 99))


# ============ Función de Envío de Alertas ============

def send_alert(alert: Alert) -> None:
    """
    Loggea alerta y envía notificaciones externas.
    
    Args:
        alert: Alert a enviar
    """
    if alert.severity == "critical":
        logger.critical(f"🚨 ALERTA CRÍTICA: {alert.message}")
        logger.critical(f"   Componente: {alert.component}")
        logger.critical(f"   Métrica: {alert.metric}")
        if alert.current_value is not None:
            logger.critical(f"   Valor actual: {alert.current_value}")
        if alert.threshold is not None:
            logger.critical(f"   Umbral: {alert.threshold}")
        if alert.previous_value is not None:
            logger.critical(f"   Valor anterior: {alert.previous_value}")
        if alert.recommendation:
            logger.critical(f"   Recomendación: {alert.recommendation}")
        
        # Enviar notificaciones externas para alertas críticas
        try:
            from app.config import settings
            if getattr(settings, 'NOTIFICATIONS_ENABLED', False):
                from app.metrics.notifications import send_notification
                results = send_notification(alert)
                
                for channel, success in results.items():
                    if success:
                        logger.info(f"✅ Notificación enviada a {channel}")
                    else:
                        logger.warning(f"⚠️ Falló notificación a {channel}")
        except Exception as e:
            logger.error(f"❌ Error enviando notificaciones: {e}", exc_info=True)
            # No romper el flujo si fallan las notificaciones
    
    elif alert.severity == "warning":
        logger.warning(f"⚠️  ALERTA: {alert.message}")
        logger.warning(f"   Componente: {alert.component}")
        logger.warning(f"   Métrica: {alert.metric}")
        if alert.current_value is not None:
            logger.warning(f"   Valor actual: {alert.current_value}")
        if alert.recommendation:
            logger.warning(f"   Recomendación: {alert.recommendation}")
        
        # Warnings se agregan a digest (no enviar inmediatamente)
        # Solo logging por ahora
    
    else:  # info
        logger.info(f"ℹ️  ALERTA INFO: {alert.message}")
        # Info solo se loggea, no se envía notificación


# ============ Job Periódico de Monitoreo ============

def run_health_check() -> Dict:
    """
    Ejecuta health check completo del sistema.
    
    Returns:
        Dict con resultados del health check
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "components": {},
        "active_alerts": []
    }
    
    try:
        for component in ['router', 'sql', 'viz', 'hybrid']:
            health = check_component_health(component)
            results['components'][component] = {
                "status": health.status,
                "alerts_count": len(health.alerts),
                "success_rate": health.metrics.get('success_rate', 0.0),
                "p95_latency_ms": health.metrics.get('p95_latency', 0)
            }
            
            if health.status == "critical":
                results['overall_status'] = "critical"
            elif health.status == "warning" and results['overall_status'] != "critical":
                results['overall_status'] = "warning"
        
        # Obtener todas las alertas activas
        active_alerts = get_active_alerts()
        
        # Convertir alertas a dict para serialización
        results['active_alerts'] = [
            {
                "severity": alert.severity,
                "component": alert.component,
                "metric": alert.metric,
                "message": alert.message,
                "current_value": alert.current_value,
                "previous_value": alert.previous_value,
                "threshold": alert.threshold,
                "recommendation": alert.recommendation,
                "created_at": alert.created_at.isoformat()
            }
            for alert in active_alerts
        ]
        
        # Enviar alertas críticas
        for alert in active_alerts:
            if alert.severity == 'critical':
                send_alert(alert)
        
    except Exception as e:
        logger.error(f"❌ Error ejecutando health check: {e}", exc_info=True)
        results['error'] = str(e)
        results['overall_status'] = "error"
    
    return results


# ============ Para Testing ============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing Alert System ===\n")
    
    # Test 1: Health check de un componente
    print("1. Health check de router...")
    router_health = check_component_health('router', days=1)
    print(f"   Status: {router_health.status}")
    print(f"   Alerts: {len(router_health.alerts)}")
    
    # Test 2: Detección de degradación
    print("\n2. Detección de degradación...")
    degradation = detect_degradation('sql')
    if degradation:
        print(f"   Degradación detectada: {degradation.message}")
    else:
        print("   No se detectó degradación")
    
    # Test 3: Alertas activas
    print("\n3. Alertas activas...")
    alerts = get_active_alerts()
    print(f"   Total alertas: {len(alerts)}")
    for alert in alerts:
        print(f"   - [{alert.severity}] {alert.message}")
    
    # Test 4: Health check completo
    print("\n4. Health check completo...")
    health_results = run_health_check()
    print(f"   Overall status: {health_results['overall_status']}")
    print(f"   Total alertas: {len(health_results['active_alerts'])}")
    
    print("\n✅ Tests completados")

