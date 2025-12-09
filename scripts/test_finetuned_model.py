"""
Script para probar el endpoint del modelo fine-tuned en Modal.
Verifica que el endpoint responda correctamente y en tiempo razonable.
"""

import requests
import json
import time
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings

ENDPOINT_URL = settings.FINETUNED_MODEL_ENDPOINT or "https://egsamaca56--viz-expert-model-predict.modal.run"
TIMEOUT_SECONDS = 10
MAX_RESPONSE_TIME = 5  # Esperamos respuesta en menos de 5 segundos


def test_finetuned_endpoint():
    """
    Prueba el endpoint del modelo fine-tuned.
    
    Returns:
        bool: True si todas las pruebas pasan
    """
    print("🧪 Testing Fine-tuned Model Endpoint")
    print("=" * 60)
    print(f"Endpoint: {ENDPOINT_URL}")
    print(f"Timeout: {TIMEOUT_SECONDS}s")
    print()
    
    if not ENDPOINT_URL:
        print("❌ ERROR: FINETUNED_MODEL_ENDPOINT no está configurado en .env")
        print("   Agrega: FINETUNED_MODEL_ENDPOINT=https://egsamaca56--viz-expert-model-predict.modal.run")
        return False
    
    # Casos de prueba
    test_cases = [
        {
            "name": "Top N productos (Bar Chart)",
            "payload": {
                "user_query": "Muestra los 10 productos más vendidos",
                "sql_query": "SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 10",
                "columns": ["producto", "total"],
                "num_rows": 10,
                "data_preview": [
                    {"producto": "Tela Algodón", "total": 5000},
                    {"producto": "Tela Poliéster", "total": 4500},
                    {"producto": "Tela Lino", "total": 3000}
                ]
            },
            "expected_chart_type": "bar"
        },
        {
            "name": "Serie temporal (Line Chart)",
            "payload": {
                "user_query": "Muestra las ventas por mes del último año",
                "sql_query": "SELECT mes, SUM(ventas) as total_ventas FROM ventas WHERE fecha >= DATE_SUB(NOW(), INTERVAL 12 MONTH) GROUP BY mes ORDER BY mes",
                "columns": ["mes", "total_ventas"],
                "num_rows": 12,
                "data_preview": [
                    {"mes": "2024-01", "total_ventas": 15000},
                    {"mes": "2024-02", "total_ventas": 18000},
                    {"mes": "2024-03", "total_ventas": 16000}
                ]
            },
            "expected_chart_type": "line"
        },
        {
            "name": "Distribución (Pie Chart)",
            "payload": {
                "user_query": "Muestra la distribución de productos por categoría",
                "sql_query": "SELECT categoria, COUNT(*) as cantidad FROM productos GROUP BY categoria",
                "columns": ["categoria", "cantidad"],
                "num_rows": 5,
                "data_preview": [
                    {"categoria": "Telas", "cantidad": 45},
                    {"categoria": "Hilos", "cantidad": 30},
                    {"categoria": "Accesorios", "cantidad": 25}
                ]
            },
            "expected_chart_type": "pie"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}/{len(test_cases)}: {test_case['name']}")
        print("-" * 60)
        
        try:
            # Medir tiempo de respuesta
            start_time = time.time()
            
            response = requests.post(
                ENDPOINT_URL,
                json=test_case['payload'],
                timeout=TIMEOUT_SECONDS
            )
            
            elapsed_time = time.time() - start_time
            
            # Verificar status code
            if response.status_code != 200:
                print(f"❌ FAILED: HTTP {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                results.append(False)
                continue
            
            # Verificar tiempo de respuesta
            if elapsed_time > MAX_RESPONSE_TIME:
                print(f"⚠️  WARNING: Response time {elapsed_time:.2f}s exceeds {MAX_RESPONSE_TIME}s")
            else:
                print(f"✅ Response time: {elapsed_time:.2f}s (< {MAX_RESPONSE_TIME}s)")
            
            # Parsear respuesta
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                print(f"❌ FAILED: Invalid JSON response")
                print(f"   Error: {e}")
                print(f"   Response: {response.text[:200]}")
                results.append(False)
                continue
            
            # Validar estructura de respuesta
            required_fields = ['chart_type']
            missing_fields = [field for field in required_fields if field not in response_data]
            
            if missing_fields:
                print(f"❌ FAILED: Missing required fields: {missing_fields}")
                print(f"   Response: {json.dumps(response_data, indent=2)}")
                results.append(False)
                continue
            
            # Validar chart_type
            chart_type = response_data.get('chart_type')
            if not chart_type:
                print(f"❌ FAILED: chart_type is empty")
                results.append(False)
                continue
            
            print(f"✅ Chart type: {chart_type}")
            
            # Validar reasoning si existe
            if 'reasoning' in response_data:
                reasoning = response_data['reasoning']
                print(f"✅ Reasoning: {reasoning[:100]}...")
            
            # Validar config si existe
            if 'config' in response_data:
                config = response_data['config']
                print(f"✅ Config: {json.dumps(config, indent=2)}")
            
            # Validar que el chart_type sea razonable (opcional)
            valid_chart_types = ['bar', 'line', 'pie', 'scatter', 'area', 'histogram']
            if chart_type.lower() not in valid_chart_types:
                print(f"⚠️  WARNING: chart_type '{chart_type}' not in common types: {valid_chart_types}")
            
            print(f"✅ Test passed!")
            results.append(True)
            
        except requests.exceptions.Timeout:
            print(f"❌ FAILED: Request timed out after {TIMEOUT_SECONDS}s")
            results.append(False)
        except requests.exceptions.RequestException as e:
            print(f"❌ FAILED: Request error: {e}")
            results.append(False)
        except Exception as e:
            print(f"❌ FAILED: Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Resumen
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return True
    else:
        print(f"❌ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = test_finetuned_endpoint()
    sys.exit(0 if success else 1)

