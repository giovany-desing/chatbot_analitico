#!/usr/bin/env python3
"""
Script simple para probar rápidamente el endpoint de Modal.
"""

import requests
import json

ENDPOINT_URL = "https://egsamaca56--viz-expert-model-predict.modal.run"

# Datos de prueba
payload = {
    "user_query": "Muestra los 10 productos más vendidos",
    "sql_query": "SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 10",
    "columns": ["producto", "total"],
    "num_rows": 10,
    "data_preview": [
        {"producto": "Tela Algodón", "total": 5000},
        {"producto": "Tela Poliéster", "total": 4500},
        {"producto": "Tela Lino", "total": 3000}
    ]
}

print("🚀 Enviando petición al endpoint...")
print(f"📍 URL: {ENDPOINT_URL}")
print(f"📦 Payload:")
print(json.dumps(payload, indent=2, ensure_ascii=False))

try:
    # Timeout de 15 minutos (900 segundos) para permitir carga inicial del modelo
    print("⏳ Esperando respuesta (puede tardar varios minutos en la primera carga)...")
    response = requests.post(ENDPOINT_URL, json=payload, timeout=900)
    response.raise_for_status()
    
    result = response.json()
    
    print(f"\n✅ Respuesta recibida (Status: {response.status_code}):")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Validar respuesta
    if "error" in result:
        print(f"\n❌ Error en la respuesta: {result['error']}")
    else:
        print(f"\n✅ Predicción exitosa!")
        if "chart_type" in result:
            print(f"   📊 Tipo de gráfico: {result['chart_type']}")
        if "reasoning" in result:
            print(f"   💭 Razón: {result['reasoning']}")
    
    print(f"\n⏱️  Tiempo de respuesta: {response.elapsed.total_seconds():.2f}s")
    
except requests.exceptions.RequestException as e:
    print(f"\n❌ Error en la petición: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"   Status code: {e.response.status_code}")
        print(f"   Response: {e.response.text}")

