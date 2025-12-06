import sys
import os
from pathlib import Path

# Agregar el directorio raíz al PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import time
from app.intelligence.hybrid_system import HybridVizSystem

# Inicializar
system = HybridVizSystem(
    finetuned_endpoint="https://egsamaca56--viz-expert-model-predict.modal.run"
)

# Test cases
test_cases = [
    {
        "query": "Top 10 productos",
        "results": [{"producto": f"P{i}", "ventas": 100-i} for i in range(10)],
        "expected_type": "bar",
        "expected_source": "rules"
    },
    {
        "query": "Evolución de ventas por mes",
        "results": [{"mes": f"2024-{i:02d}", "ventas": 1000+i*100} for i in range(1, 13)],
        "expected_type": "line",
        "expected_source": "rules"
    },
    {
        "query": "Distribución de ventas por región",
        "results": [{"region": f"R{i}", "total": 1000*i} for i in range(1, 6)],
        "expected_type": "pie",
        "expected_source": "rules"
    },
]

# Ejecutar tests
results = []
for i, test in enumerate(test_cases, 1):
    start = time.time()

    result = system.decide_chart(test["query"], test["results"])

    elapsed = time.time() - start

    success = (
        result.get("chart_type") == test["expected_type"] and
        result.get("source") == test["expected_source"]
    )

    results.append({
        "test": i,
        "success": success,
        "time": elapsed,
        "source": result.get("source")
    })

    status = "✅" if success else "❌"
    print(f"{status} Test {i}: {test['query'][:30]}...")
    print(f"   Type: {result.get('chart_type')} | Source: {result.get('source')} | Time: {elapsed*1000:.0f}ms")

# Summary
total = len(results)
passed = sum(1 for r in results if r["success"])
avg_time = sum(r["time"] for r in results) / total

print(f"\n📊 Resultados:")
print(f"   Precisión: {passed}/{total} ({passed/total*100:.1f}%)")
print(f"   Tiempo promedio: {avg_time*1000:.0f}ms")