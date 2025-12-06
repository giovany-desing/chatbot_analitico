# 📊 Datos de Entrenamiento - Fine-Tuning

## ✅ Archivos Generados

### 1. `training_data.jsonl` (100 ejemplos manuales)
- **Contenido**: Ejemplos curados manualmente con alta calidad
- **Uso**: Ideal para testing inicial y validación
- **Características**:
  - Queries reales basadas en tu schema
  - Casos edge incluidos (datos vacíos, 1 fila, etc.)
  - Razonamientos detallados

### 2. `training_data_complete.jsonl` (500 ejemplos)
- **Contenido**: Dataset completo generado programáticamente
- **Uso**: Recomendado para fine-tuning en producción
- **Distribución**:
  - 200 ejemplos (40%) → Bar charts
  - 150 ejemplos (30%) → Line charts
  - 100 ejemplos (20%) → Pie charts
  - 30 ejemplos (6%) → Scatter plots
  - 20 ejemplos (4%) → Histograms

## 📋 Formato del Dataset

Cada línea es un ejemplo en formato JSONL:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Eres un experto en visualización de datos..."
    },
    {
      "role": "user",
      "content": "Query: ...\nSQL: ...\nColumnas: ...\nFilas: ...\nData preview: ..."
    },
    {
      "role": "assistant",
      "content": "{\"chart_type\": \"bar\", \"reasoning\": \"...\", \"confidence\": 0.95}"
    }
  ]
}
```

## 🎯 Tipos de Gráficos Cubiertos

| Tipo | Descripción | Casos de Uso |
|------|-------------|--------------|
| `bar` | Gráfico de barras | Rankings, comparaciones, categorías |
| `line` | Gráfico de línea | Series temporales, tendencias, evolución |
| `pie` | Gráfico de pastel | Distribuciones, porcentajes, partes de un todo |
| `scatter` | Gráfico de dispersión | Correlaciones, relaciones entre 2 variables |
| `histogram` | Histograma | Distribuciones de frecuencia, rangos |
| `none` | Sin gráfico (KPI) | Métricas únicas, conteos simples |

## 🚀 Cómo Usar

### Opción 1: Dataset Manual (100 ejemplos)
```bash
# Para pruebas rápidas o fine-tuning ligero
cp training_data.jsonl mi_dataset.jsonl
```

### Opción 2: Dataset Completo (500 ejemplos) - RECOMENDADO
```bash
# Para fine-tuning en producción
cp training_data_complete.jsonl mi_dataset.jsonl
```

### Opción 3: Combinar ambos
```bash
# Mejores resultados: manual (calidad) + generado (volumen)
cat training_data.jsonl training_data_complete.jsonl > mi_dataset_full.jsonl
```

### Opción 4: Generar más datos
```bash
# Modificar TOTAL_EXAMPLES en el script
python3 scripts/generate_training_data.py
```

## 📤 Subir a Google Colab

1. Abre Google Colab: https://colab.research.google.com
2. Sube el archivo:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Selecciona training_data_complete.jsonl
   ```
3. Continúa con FASE_1_FINE_TUNING.md paso 3

## 🔍 Validar Dataset

Antes de entrenar, valida que el formato sea correcto:

```python
import json

# Contar ejemplos por tipo
chart_types = {}
with open('training_data_complete.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        assistant_msg = data['messages'][2]['content']
        chart_type = json.loads(assistant_msg)['chart_type']
        chart_types[chart_type] = chart_types.get(chart_type, 0) + 1

print("Distribución:")
for chart_type, count in sorted(chart_types.items()):
    print(f"  {chart_type}: {count}")
```

**Output esperado:**
```
Distribución:
  bar: 200
  histogram: 20
  line: 150
  none: ~50
  pie: 100
  scatter: 30
```

## ✨ Características del Dataset

### ✅ Basado en tu Schema Real
- Tablas: `ventas_preventivas`, `ventas_correctivas`
- Columnas: `id`, `orden_compra`, `producto`, `fecha_creacion`, `cantidad`, `total`
- Productos: 20 productos textiles realistas

### ✅ Queries Variadas
- Simples: `SELECT COUNT(*) ...`
- Complejas: `UNION`, `JOIN`, `CASE WHEN`, ventanas analíticas
- Agregaciones: `SUM`, `AVG`, `COUNT`, `GROUP BY`
- Temporales: fechas, meses, trimestres, años

### ✅ Casos Edge Incluidos
- Datos vacíos (0 filas)
- 1 sola fila
- Muchas filas (>10,000)
- Pocas categorías (2-3)
- Muchas categorías (>15)
- Valores nulos
- Outliers

### ✅ Lenguaje Natural
- Queries en español como usuarios reales escribirían
- Variaciones: "dame", "muéstrame", "cuántos", "qué", etc.
- Contexto de negocio textil

## 📊 Métricas Esperadas Post-Entrenamiento

Después de fine-tuning con 500 ejemplos:

| Métrica | Valor Esperado |
|---------|----------------|
| **Training Loss** | <0.10 |
| **Validation Accuracy** | >85% |
| **Precision (bar)** | >90% |
| **Precision (line)** | >88% |
| **Precision (pie)** | >85% |
| **Precision (scatter)** | >80% |
| **Recall promedio** | >82% |

## 🔄 Agregar Datos Reales

Cuando tengas queries reales de usuarios:

1. Exporta desde el sistema de feedback (Fase 4)
2. Revisa y corrige manualmente
3. Agrega al dataset:
   ```bash
   cat training_data_complete.jsonl feedback_queries.jsonl > training_v2.jsonl
   ```
4. Re-entrena mensualmente

## 🎓 Ejemplo Completo

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Eres un experto en visualización de datos para análisis de ventas textiles. Debes elegir el mejor tipo de gráfico basándote en la query del usuario y los datos SQL disponibles."
    },
    {
      "role": "user",
      "content": "Query: Muestra los 10 productos más vendidos\nSQL: SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 10\nColumnas: [producto, total]\nFilas: 10\nData preview: [{\"producto\": \"Tela Algodón\", \"total\": 5000}, {\"producto\": \"Tela Poliéster\", \"total\": 4200}]"
    },
    {
      "role": "assistant",
      "content": "{\"chart_type\": \"bar\", \"reasoning\": \"Top 10 implica ranking de productos. Bar chart es ideal para comparar cantidades entre categorías discretas y mostrar claramente el orden de mayor a menor.\", \"confidence\": 0.98, \"config\": {\"x_axis\": \"producto\", \"y_axis\": \"total\", \"title\": \"Top 10 Productos Más Vendidos\", \"sort\": \"descending\"}}"
    }
  ]
}
```

## 🛠️ Personalización

### Cambiar distribución de tipos de gráfico

Edita `scripts/generate_training_data.py`:

```python
distribution = {
    "bar": int(total * 0.35),      # Reducir bar
    "line": int(total * 0.35),     # Aumentar line
    "pie": int(total * 0.20),      # Mantener
    "scatter": int(total * 0.06),  # Mantener
    "histogram": int(total * 0.04) # Mantener
}
```

### Agregar nuevos productos

```python
PRODUCTOS = [
    "Tela Algodón", "Tela Poliéster",
    # Agregar los tuyos:
    "Tela Nueva A", "Tela Nueva B"
]
```

### Agregar nuevos tipos de queries

```python
def generate_bar_example() -> Dict:
    templates = [
        # ... existentes ...
        {
            "query": "TU NUEVA QUERY",
            "sql": "TU SQL",
            "chart_type": "bar",
            "reasoning": "TU RAZONAMIENTO",
            "confidence": 0.95
        }
    ]
```

## 📚 Referencias

- **FASE_1_FINE_TUNING.md**: Guía completa de fine-tuning
- **FASE_2_SISTEMA_HIBRIDO.md**: Cómo integrar el modelo entrenado
- **FASE_4_FEEDBACK_MEJORA_CONTINUA.md**: Cómo recolectar datos para reentrenamiento

## ❓ FAQ

**P: ¿Necesito 500 ejemplos obligatoriamente?**
R: No, puedes empezar con 100-200. Más ejemplos = mejor accuracy, pero 200+ ya da buenos resultados.

**P: ¿Puedo mezclar inglés y español?**
R: Sí, pero es mejor mantener un solo idioma consistente. Para modelos multilingües, necesitas ejemplos en ambos idiomas.

**P: ¿Cada cuánto debo reentrenar?**
R: Con feedback activo: cada mes. Sin feedback: cada 3-6 meses o cuando notes degradación.

**P: ¿Puedo usar este dataset con otros modelos?**
R: Sí, el formato es compatible con: Llama, Mistral, GPT (con mínimas modificaciones), Gemini.

---

**¡Dataset listo para entrenar!** 🚀

Siguiente paso: Ver **FASE_1_FINE_TUNING.md** sección 3.
