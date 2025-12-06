import pytest
from app.intelligence.rules_engine import DeterministicRulesEngine, ConfidenceLevel

def test_rules_engine_temporal():
    """Test: Detecta datos temporales"""
    engine = DeterministicRulesEngine()

    query = "Muestra ventas por mes"
    results = [
        {"mes": "2024-01", "ventas": 1000},
        {"mes": "2024-02", "ventas": 1500},
    ]

    result = engine.apply(query, results)

    assert result.chart_type == "line"
    assert result.confidence == ConfidenceLevel.HIGH
    assert "temporal" in result.reasoning.lower()
    print("✅ Test reglas temporales pasado")

def test_rules_engine_top_n():
    """Test: Detecta Top N"""
    engine = DeterministicRulesEngine()

    query = "Top 5 productos más vendidos"
    results = [
        {"producto": "A", "cantidad": 100},
        {"producto": "B", "cantidad": 90},
        {"producto": "C", "cantidad": 80},
    ]

    result = engine.apply(query, results)

    assert result.chart_type == "bar"
    assert result.confidence == ConfidenceLevel.HIGH
    print("✅ Test reglas Top N pasado")
