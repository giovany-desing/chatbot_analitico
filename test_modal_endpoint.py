#!/usr/bin/env python3
"""
Script para probar el endpoint de Modal y validar las predicciones del modelo.
"""

import requests
import json
from typing import Dict, Any

# URL del endpoint
ENDPOINT_URL = "https://egsamaca56--viz-expert-model-predict.modal.run"

def test_prediction(
    user_query: str,
    sql_query: str,
    columns: list,
    num_rows: int,
    data_preview: list,
    expected_chart_type: str = None
) -> Dict[str, Any]:
    """
    Prueba una predicción en el endpoint.
    
    Args:
        user_query: Consulta del usuario
        sql_query: Consulta SQL
        columns: Lista de columnas
        num_rows: Número de filas
        data_preview: Preview de los datos
        expected_chart_type: Tipo de gráfico esperado (opcional, para validación)
    
    Returns:
        Dict con el resultado de la prueba
    """
    payload = {
        "user_query": user_query,
        "sql_query": sql_query,
        "columns": columns,
        "num_rows": num_rows,
        "data_preview": data_preview
    }
    
    print(f"\n{'='*60}")
    print(f"📊 Probando predicción:")
    print(f"   Query: {user_query}")
    print(f"   SQL: {sql_query}")
    print(f"   Columnas: {columns}")
    print(f"   Filas: {num_rows}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(ENDPOINT_URL, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        print(f"\n✅ Respuesta recibida:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Validar estructura de respuesta
        validation = {
            "success": True,
            "has_error": "error" in result,
            "is_valid_json": not result.get("error"),
            "response_time": response.elapsed.total_seconds()
        }
        
        if expected_chart_type and not result.get("error"):
            validation["matches_expected"] = (
                result.get("chart_type", "").lower() == expected_chart_type.lower()
            )
        
        return {
            "payload": payload,
            "response": result,
            "validation": validation,
            "status_code": response.status_code
        }
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error en la petición: {e}")
        return {
            "payload": payload,
            "error": str(e),
            "success": False
        }
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parseando JSON: {e}")
        return {
            "payload": payload,
            "error": f"JSON decode error: {e}",
            "success": False
        }

def run_test_suite():
    """Ejecuta una suite de pruebas con diferentes casos."""
    
    test_cases = [
        {
            "name": "Top productos más vendidos",
            "user_query": "Muestra los 10 productos más vendidos",
            "sql_query": "SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 10",
            "columns": ["producto", "total"],
            "num_rows": 10,
            "data_preview": [
                {"producto": "Tela Algodón", "total": 5000},
                {"producto": "Tela Poliéster", "total": 4500},
                {"producto": "Tela Lino", "total": 3000}
            ],
            "expected_chart_type": "bar"  # Esperamos un gráfico de barras
        },
        {
            "name": "Ventas por mes",
            "user_query": "Muestra las ventas por mes del último año",
            "sql_query": "SELECT mes, SUM(ventas) as total_ventas FROM ventas WHERE fecha >= DATE_SUB(NOW(), INTERVAL 12 MONTH) GROUP BY mes ORDER BY mes",
            "columns": ["mes", "total_ventas"],
            "num_rows": 12,
            "data_preview": [
                {"mes": "2024-01", "total_ventas": 15000},
                {"mes": "2024-02", "total_ventas": 18000},
                {"mes": "2024-03", "total_ventas": 22000}
            ],
            "expected_chart_type": "line"  # Esperamos un gráfico de línea
        },
        {
            "name": "Distribución de categorías",
            "user_query": "Muestra la distribución de productos por categoría",
            "sql_query": "SELECT categoria, COUNT(*) as cantidad FROM productos GROUP BY categoria",
            "columns": ["categoria", "cantidad"],
            "num_rows": 5,
            "data_preview": [
                {"categoria": "Telas", "cantidad": 45},
                {"categoria": "Hilos", "cantidad": 30},
                {"categoria": "Accesorios", "cantidad": 25}
            ],
            "expected_chart_type": "pie"  # Esperamos un gráfico de pastel
        },
        {
            "name": "Comparación de ventas por región",
            "user_query": "Compara las ventas por región",
            "sql_query": "SELECT region, SUM(ventas) as total FROM ventas GROUP BY region ORDER BY total DESC",
            "columns": ["region", "total"],
            "num_rows": 4,
            "data_preview": [
                {"region": "Norte", "total": 50000},
                {"region": "Sur", "total": 45000},
                {"region": "Este", "total": 40000}
            ],
            "expected_chart_type": "bar"  # Esperamos un gráfico de barras
        }
    ]
    
    results = []
    
    print("🚀 Iniciando suite de pruebas...")
    print(f"📍 Endpoint: {ENDPOINT_URL}\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'#'*60}")
        print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
        print(f"{'#'*60}")
        
        result = test_prediction(
            user_query=test_case["user_query"],
            sql_query=test_case["sql_query"],
            columns=test_case["columns"],
            num_rows=test_case["num_rows"],
            data_preview=test_case["data_preview"],
            expected_chart_type=test_case.get("expected_chart_type")
        )
        
        result["test_name"] = test_case["name"]
        results.append(result)
    
    # Resumen
    print(f"\n\n{'='*60}")
    print("📊 RESUMEN DE PRUEBAS")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r.get("success", False))
    total = len(results)
    
    print(f"\n✅ Pruebas exitosas: {successful}/{total}")
    print(f"❌ Pruebas fallidas: {total - successful}/{total}")
    
    for i, result in enumerate(results, 1):
        status = "✅" if result.get("success", False) else "❌"
        test_name = result.get("test_name", f"Test {i}")
        validation = result.get("validation", {})
        
        print(f"\n{status} {test_name}:")
        if result.get("success"):
            print(f"   - Tiempo de respuesta: {validation.get('response_time', 0):.2f}s")
            print(f"   - JSON válido: {'✅' if validation.get('is_valid_json') else '❌'}")
            if "matches_expected" in validation:
                match = validation["matches_expected"]
                print(f"   - Coincide con esperado: {'✅' if match else '❌'}")
        else:
            print(f"   - Error: {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    # Instalar requests si no está instalado
    try:
        import requests
    except ImportError:
        print("❌ Por favor instala requests: pip install requests")
        exit(1)
    
    # Ejecutar suite de pruebas
    results = run_test_suite()
    
    # Guardar resultados en archivo
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: test_results.json")

