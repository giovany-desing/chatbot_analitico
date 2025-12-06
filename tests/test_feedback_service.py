import pytest
from app.feedback.feedback_service import feedback_service
import time

def test_save_and_update_interaction():
    """Test completo del ciclo de feedback"""

    # 1. Guardar interacción
    feedback_id = feedback_service.save_interaction(
        session_id="test_session_123",
        user_query="¿Cuántas ventas hay?",
        sql_generated="SELECT COUNT(*) FROM ordenes",
        chart_type="bar",
        chart_config={"type": "bar", "title": "Test"},
        response_time_ms=250,
        error_occurred=False
    )

    assert feedback_id > 0
    print(f"✅ Interacción guardada con ID: {feedback_id}")

    # 2. Actualizar con rating
    success = feedback_service.update_rating(
        feedback_id=feedback_id,
        rating=5,
        feedback_text="Excelente respuesta"
    )

    assert success is True
    print("✅ Rating actualizado correctamente")

    # 3. Verificar que no aparece en low-rated
    low_rated = feedback_service.get_low_rated_queries(min_rating=2, limit=10)
    assert not any(q['id'] == feedback_id for q in low_rated)
    print("✅ No aparece en queries de baja valoración")

def test_metrics():
    """Test de generación de métricas"""

    metrics = feedback_service.get_metrics(days=30)

    assert 'general' in metrics
    assert 'total_interactions' in metrics['general']
    assert 'rating_distribution' in metrics

    print("✅ Métricas generadas correctamente:")
    print(f"   Total interacciones: {metrics['general']['total_interactions']}")
    print(f"   Rating promedio: {metrics['general']['avg_rating']}")

def test_export_retraining():
    """Test de exportación para reentrenamiento"""

    # Primero crear algunos ejemplos con rating bajo
    for i in range(3):
        fid = feedback_service.save_interaction(
            session_id=f"test_{i}",
            user_query=f"Query de prueba {i}",
            chart_type="bar"
        )
        feedback_service.update_rating(fid, rating=2, feedback_text="Mejorable")

    # Exportar
    count = feedback_service.export_for_retraining(
        output_file='test_retraining.jsonl',
        max_rating=3
    )

    assert count >= 3
    print(f"✅ Exportados {count} ejemplos para reentrenamiento")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])