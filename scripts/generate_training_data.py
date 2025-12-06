#!/usr/bin/env python3
"""
Script para generar dataset de entrenamiento completo (500 ejemplos)
para fine-tuning de modelo de decisión de gráficos
"""

import json
import random
from typing import List, Dict

# Configuración
TOTAL_EXAMPLES = 500
OUTPUT_FILE = "training_data_complete.jsonl"

# Productos de la industria textil
PRODUCTOS = [
    "Tela Algodón", "Tela Poliéster", "Tela Lycra", "Tela Seda",
    "Tela Denim", "Tela Lino", "Tela Cashmere", "Tela Nylon",
    "Tela Rayón", "Tela Lana", "Tela Spandex", "Tela Chenille",
    "Tela Terciopelo", "Tela Satén", "Tela Franela", "Tela Gamuza",
    "Tela Organza", "Tela Tul", "Tela Crepe", "Tela Gabardina"
]

SYSTEM_PROMPT = "Eres un experto en visualización de datos para análisis de ventas textiles. Debes elegir el mejor tipo de gráfico basándote en la query del usuario y los datos SQL disponibles."

def generate_bar_example() -> Dict:
    """Genera ejemplo para gráfico de barras"""
    templates = [
        {
            "query": f"Muestra los {random.randint(5, 20)} productos más vendidos",
            "sql": f"SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT {random.randint(5, 20)}",
            "chart_type": "bar",
            "reasoning": "Query de ranking (top N). Bar chart es ideal para comparar cantidades entre categorías y mostrar orden.",
            "confidence": round(random.uniform(0.95, 0.99), 2)
        },
        {
            "query": "Compara las ventas entre productos",
            "sql": f"SELECT producto, SUM(total) as revenue FROM ventas_preventivas WHERE producto IN {tuple(random.sample(PRODUCTOS, 5))} GROUP BY producto",
            "chart_type": "bar",
            "reasoning": "Comparación entre múltiples categorías. Bar chart permite ver diferencias claramente.",
            "confidence": round(random.uniform(0.94, 0.98), 2)
        },
        {
            "query": "Revenue por día de la semana",
            "sql": "SELECT DAYNAME(fecha_creacion) as dia, SUM(total) as revenue FROM ventas_preventivas GROUP BY dia",
            "chart_type": "bar",
            "reasoning": "Comparación categórica (días de semana). Bar chart muestra qué días tienen mayor actividad.",
            "confidence": round(random.uniform(0.95, 0.98), 2)
        }
    ]

    template = random.choice(templates)
    return create_message(template)

def generate_line_example() -> Dict:
    """Genera ejemplo para gráfico de línea"""
    templates = [
        {
            "query": f"Evolución de ventas por {random.choice(['mes', 'trimestre', 'año'])}",
            "sql": "SELECT DATE_FORMAT(fecha_creacion, '%Y-%m') as periodo, SUM(total) as revenue FROM ventas_preventivas GROUP BY periodo ORDER BY periodo",
            "chart_type": "line",
            "reasoning": "Serie temporal con datos de fechas. Line chart es perfecto para mostrar tendencias y evolución temporal.",
            "confidence": round(random.uniform(0.96, 0.99), 2)
        },
        {
            "query": "Tendencia mensual del último año",
            "sql": "SELECT MONTH(fecha_creacion) as mes, SUM(total) as revenue FROM ventas_preventivas WHERE fecha_creacion >= DATE_SUB(NOW(), INTERVAL 1 YEAR) GROUP BY mes",
            "chart_type": "line",
            "reasoning": "Datos temporales mensuales. Line chart permite visualizar patrones estacionales y tendencias.",
            "confidence": round(random.uniform(0.97, 0.99), 2)
        },
        {
            "query": "Ventas diarias de la última semana",
            "sql": "SELECT DATE(fecha_creacion) as fecha, SUM(total) as revenue FROM ventas_preventivas WHERE fecha_creacion >= DATE_SUB(NOW(), INTERVAL 7 DAY) GROUP BY fecha",
            "chart_type": "line",
            "reasoning": "Serie temporal diaria. Line chart muestra evolución día a día.",
            "confidence": round(random.uniform(0.96, 0.98), 2)
        }
    ]

    template = random.choice(templates)
    return create_message(template)

def generate_pie_example() -> Dict:
    """Genera ejemplo para gráfico de pie"""
    templates = [
        {
            "query": f"Distribución de ventas por {random.choice(['producto', 'tipo', 'categoría'])}",
            "sql": f"SELECT producto, SUM(total) as revenue FROM ventas_preventivas GROUP BY producto LIMIT {random.randint(3, 7)}",
            "chart_type": "pie",
            "reasoning": "Pocas categorías mostrando distribución porcentual. Pie chart es ideal para mostrar partes de un todo.",
            "confidence": round(random.uniform(0.93, 0.97), 2)
        },
        {
            "query": "Proporción preventivas vs correctivas",
            "sql": "SELECT tipo, COUNT(*) as cantidad FROM (SELECT 'Preventivas' as tipo FROM ventas_preventivas UNION ALL SELECT 'Correctivas' FROM ventas_correctivas) t GROUP BY tipo",
            "chart_type": "pie",
            "reasoning": "Distribución entre dos categorías. Pie chart muestra claramente las proporciones.",
            "confidence": round(random.uniform(0.95, 0.98), 2)
        },
        {
            "query": "Participación de mercado por producto",
            "sql": "SELECT producto, SUM(total) as revenue FROM ventas_preventivas GROUP BY producto ORDER BY revenue DESC LIMIT 5",
            "chart_type": "pie",
            "reasoning": "Top productos mostrando participación. Pie chart visualiza la concentración de mercado.",
            "confidence": round(random.uniform(0.94, 0.97), 2)
        }
    ]

    template = random.choice(templates)
    return create_message(template)

def generate_scatter_example() -> Dict:
    """Genera ejemplo para gráfico de scatter"""
    templates = [
        {
            "query": "Relación entre cantidad y precio",
            "sql": f"SELECT cantidad, total FROM ventas_preventivas WHERE producto = '{random.choice(PRODUCTOS)}'",
            "chart_type": "scatter",
            "reasoning": "Dos variables numéricas continuas. Scatter plot permite visualizar correlación entre cantidad y precio.",
            "confidence": round(random.uniform(0.92, 0.96), 2)
        },
        {
            "query": "Correlación precio unitario vs volumen",
            "sql": "SELECT cantidad, total/cantidad as precio_unitario FROM ventas_preventivas WHERE cantidad > 0",
            "chart_type": "scatter",
            "reasoning": "Análisis de correlación entre dos variables numéricas. Scatter muestra si hay descuentos por volumen.",
            "confidence": round(random.uniform(0.93, 0.96), 2)
        },
        {
            "query": "Relación items por orden vs total",
            "sql": "SELECT COUNT(*) as items, SUM(total) as total_orden FROM ventas_preventivas GROUP BY orden_compra",
            "chart_type": "scatter",
            "reasoning": "Dos métricas numéricas para detectar patrones. Scatter plot permite ver correlación.",
            "confidence": round(random.uniform(0.91, 0.95), 2)
        }
    ]

    template = random.choice(templates)
    return create_message(template)

def generate_histogram_example() -> Dict:
    """Genera ejemplo para histograma"""
    templates = [
        {
            "query": "Distribución de precios",
            "sql": "SELECT total FROM ventas_preventivas WHERE total > 0",
            "chart_type": "histogram",
            "reasoning": "Muchos valores numéricos para análisis de distribución. Histogram muestra frecuencia de rangos de precios.",
            "confidence": round(random.uniform(0.92, 0.96), 2)
        },
        {
            "query": "Distribución de cantidades vendidas",
            "sql": "SELECT cantidad FROM ventas_preventivas WHERE cantidad > 0",
            "chart_type": "histogram",
            "reasoning": "Gran volumen de datos numéricos. Histogram muestra cómo se distribuyen las cantidades.",
            "confidence": round(random.uniform(0.93, 0.96), 2)
        },
        {
            "query": "Análisis de distribución de valores de órdenes",
            "sql": "SELECT SUM(total) as total_orden FROM ventas_preventivas GROUP BY orden_compra",
            "chart_type": "histogram",
            "reasoning": "Distribución de valores agregados. Histogram muestra rangos más frecuentes de totales de órdenes.",
            "confidence": round(random.uniform(0.91, 0.95), 2)
        }
    ]

    template = random.choice(templates)
    return create_message(template)

def generate_none_example() -> Dict:
    """Genera ejemplo para sin gráfico (KPI)"""
    templates = [
        {
            "query": random.choice([
                "¿Cuántas ventas hay?",
                "Total de órdenes",
                "Dame el revenue total",
                "¿Cuál es el ticket promedio?",
                "Promedio de cantidad por orden"
            ]),
            "sql": random.choice([
                "SELECT COUNT(*) as total FROM ventas_preventivas",
                "SELECT SUM(total) as revenue FROM ventas_preventivas",
                "SELECT AVG(total) as promedio FROM ventas_preventivas"
            ]),
            "chart_type": "none",
            "reasoning": "Métrica única (KPI). Se muestra mejor como valor destacado, no requiere gráfico.",
            "confidence": round(random.uniform(0.97, 0.99), 2)
        }
    ]

    template = random.choice(templates)
    return create_message(template)

def create_message(template: Dict) -> Dict:
    """Crea mensaje en formato chat"""
    # Simular preview de datos
    num_rows = random.randint(2, 100)

    user_content = f"""Query: {template['query']}
SQL: {template['sql']}
Columnas: {['producto', 'total'] if 'producto' in template['sql'] else ['fecha', 'revenue']}
Filas: {num_rows}
Data preview: [{{"producto": "{random.choice(PRODUCTOS)}", "total": {random.randint(1000, 50000)}}}]"""

    assistant_content = json.dumps({
        "chart_type": template['chart_type'],
        "reasoning": template['reasoning'],
        "confidence": template['confidence'],
        "config": {
            "x_axis": "producto" if "producto" in template['sql'] else "fecha",
            "y_axis": "total" if "total" in template['sql'] else "revenue",
            "title": template['query'][:50]
        }
    }, ensure_ascii=False)

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }

def generate_dataset(total: int) -> List[Dict]:
    """Genera dataset completo con distribución balanceada"""

    # Distribución objetivo
    distribution = {
        "bar": int(total * 0.40),      # 40%
        "line": int(total * 0.30),     # 30%
        "pie": int(total * 0.20),      # 20%
        "scatter": int(total * 0.06),  # 6%
        "histogram": int(total * 0.04) # 4%
    }

    examples = []

    # Generar ejemplos por tipo
    for _ in range(distribution["bar"]):
        examples.append(generate_bar_example())

    for _ in range(distribution["line"]):
        examples.append(generate_line_example())

    for _ in range(distribution["pie"]):
        examples.append(generate_pie_example())

    for _ in range(distribution["scatter"]):
        examples.append(generate_scatter_example())

    for _ in range(distribution["histogram"]):
        examples.append(generate_histogram_example())

    # Completar hasta total con KPIs
    remaining = total - len(examples)
    for _ in range(remaining):
        examples.append(generate_none_example())

    # Mezclar aleatoriamente
    random.shuffle(examples)

    return examples

def main():
    print(f"🚀 Generando dataset de {TOTAL_EXAMPLES} ejemplos...")
    print(f"📊 Distribución:")
    print(f"   - Bar charts: {int(TOTAL_EXAMPLES * 0.40)} (40%)")
    print(f"   - Line charts: {int(TOTAL_EXAMPLES * 0.30)} (30%)")
    print(f"   - Pie charts: {int(TOTAL_EXAMPLES * 0.20)} (20%)")
    print(f"   - Scatter plots: {int(TOTAL_EXAMPLES * 0.06)} (6%)")
    print(f"   - Histograms: {int(TOTAL_EXAMPLES * 0.04)} (4%)")
    print()

    dataset = generate_dataset(TOTAL_EXAMPLES)

    # Guardar en archivo JSONL
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"✅ Dataset generado exitosamente: {OUTPUT_FILE}")
    print(f"📝 Total de ejemplos: {len(dataset)}")
    print()
    print("🎯 Siguiente paso:")
    print(f"   Sube {OUTPUT_FILE} a Google Colab para fine-tuning")
    print("   Ver: FASE_1_FINE_TUNING.md")

if __name__ == "__main__":
    main()
