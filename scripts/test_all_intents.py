"""
Script para probar todos los intents del chatbot.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

test_cases = [
    # ===== SQL Tests =====
    {
        "name": "SQL - Count ventas preventivas",
        "query": "¿Cuántas ventas preventivas hay en total?",
        "expected_intent": "sql"
    },
    {
        "name": "SQL - Count ventas correctivas",
        "query": "¿Cuántas órdenes correctivas tenemos?",
        "expected_intent": "sql"
    },
    {
        "name": "SQL - Top productos preventivas",
        "query": "Muéstrame los 5 productos más vendidos en ventas preventivas",
        "expected_intent": "sql"
    },
    {
        "name": "SQL - Comparación preventivas vs correctivas",
        "query": "Compara el total de ventas preventivas con las correctivas",
        "expected_intent": "sql"
    },
    {
        "name": "SQL - Análisis temporal",
        "query": "Dame las ventas preventivas agrupadas por mes",
        "expected_intent": "sql"
    },

    # ===== KPI Tests =====
    {
        "name": "KPI - Revenue total preventivas",
        "query": "Calcula el revenue total de ventas preventivas",
        "expected_intent": "kpi"
    },
    {
        "name": "KPI - Ticket promedio",
        "query": "¿Cuál es el valor promedio de las órdenes preventivas?",
        "expected_intent": "kpi"
    },
    {
        "name": "KPI - Unidades vendidas",
        "query": "Dame el total de unidades vendidas en correctivas",
        "expected_intent": "kpi"
    },
    {
        "name": "KPI - Multiple metrics",
        "query": "Calcula revenue total, ticket promedio y unidades totales",
        "expected_intent": "kpi"
    },

    # ===== Visualización Tests =====
    {
        "name": "Viz - Gráfica productos",
        "query": "Muéstrame una gráfica de los productos más vendidos",
        "expected_intent": "viz"
    },
    {
        "name": "Viz - Tendencia temporal",
        "query": "Grafícame las ventas preventivas por mes",
        "expected_intent": "viz"
    },
    {
        "name": "Viz - Comparación barras",
        "query": "Haz un gráfico comparando preventivas vs correctivas",
        "expected_intent": "viz"
    },

    # ===== General Tests =====
    {
        "name": "General - Greeting",
        "query": "Hola, ¿qué puedes hacer?",
        "expected_intent": "general"
    },
    {
        "name": "General - Capabilities",
        "query": "¿Qué información tienes disponible?",
        "expected_intent": "general"
    },
    {
        "name": "General - Help",
        "query": "Ayúdame a entender qué son las ventas preventivas",
        "expected_intent": "general"
    },

    # ===== Hybrid Tests =====
    {
        "name": "Hybrid - Análisis completo preventivas",
        "query": "Analiza las ventas preventivas, calcula KPIs y muéstrame una gráfica",
        "expected_intent": "hybrid"
    },
    {
        "name": "Hybrid - Top productos con gráfica",
        "query": "Dame los top 10 productos y grafícalos",
        "expected_intent": "hybrid"
    },
    {
        "name": "Hybrid - Revenue y visualización",
        "query": "Muéstrame el revenue por producto en una gráfica de barras",
        "expected_intent": "hybrid"
    }
]


def test_chatbot():
    print("=" * 60)
    print("TESTING CHATBOT - ALL INTENTS")
    print("=" * 60)

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test['name']}")
        print(f"Query: {test['query']}")

        try:
            response = requests.post(
                f"{BASE_URL}/chat",
                json={"message": test['query']},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                intent = data['intent']

                success = intent == test['expected_intent']
                status = "✓ PASS" if success else "✗ FAIL"

                print(f"Expected: {test['expected_intent']}")
                print(f"Got: {intent}")
                print(f"Status: {status}")

                if data.get('sql_query'):
                    print(f"SQL: {data['sql_query'][:80]}...")

                if data.get('kpis'):
                    print(f"KPIs: {len(data['kpis'])} calculated")

                if data.get('chart_config'):
                    print(f"Chart: {data['chart_config']['chart_type']}")

                results.append({
                    'test': test['name'],
                    'success': success,
                    'intent': intent
                })
            else:
                print(f"✗ FAIL - HTTP {response.status_code}")
                results.append({
                    'test': test['name'],
                    'success': False,
                    'error': response.status_code
                })

        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append({
                'test': test['name'],
                'success': False,
                'error': str(e)
            })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.get('success'))
    total = len(results)

    print(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")

    for result in results:
        status = "✓" if result.get('success') else "✗"
        print(f"{status} {result['test']}")

    return passed == total


if __name__ == "__main__":
    import sys
    success = test_chatbot()
    sys.exit(0 if success else 1)