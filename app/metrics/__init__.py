"""
Módulo de métricas y tracking de rendimiento.
"""

from app.metrics.performance_tracker import (
    create_metrics_table,
    init_metrics,
    track_router_decision,
    track_sql_execution,
    track_viz_generation,
    track_hybrid_execution,
    get_component_metrics,
    get_all_metrics_summary,
    track_performance
)

from app.metrics.alerts import (
    THRESHOLDS,
    Alert,
    HealthStatus,
    check_component_health,
    detect_degradation,
    get_active_alerts,
    send_alert,
    run_health_check
)

__all__ = [
    "create_metrics_table",
    "init_metrics",
    "track_router_decision",
    "track_sql_execution",
    "track_viz_generation",
    "track_hybrid_execution",
    "get_component_metrics",
    "get_all_metrics_summary",
    "track_performance",
    "THRESHOLDS",
    "Alert",
    "HealthStatus",
    "check_component_health",
    "detect_degradation",
    "get_active_alerts",
    "send_alert",
    "run_health_check"
]

