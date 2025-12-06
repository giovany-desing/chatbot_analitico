# 🧪 Guía para Probar el Endpoint de Modal

## 📍 URL del Endpoint
```
https://egsamaca56--viz-expert-model-predict.modal.run
```

## 🚀 Métodos de Prueba

### 1. Script Python Simple (Recomendado)

```bash
# Instalar dependencias si es necesario
pip install requests

# Ejecutar prueba simple
python test_simple.py
```

### 2. Script Python Completo (Suite de Pruebas)

```bash
# Ejecutar suite completa de pruebas
python test_modal_endpoint.py
```

Esto ejecutará múltiples casos de prueba y generará un resumen con:
- ✅ Pruebas exitosas
- ⏱️ Tiempos de respuesta
- 📊 Validación de respuestas
- 💾 Resultados guardados en `test_results.json`

### 3. Usando cURL (Terminal)

```bash
curl -X POST https://egsamaca56--viz-expert-model-predict.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "Muestra los 10 productos más vendidos",
    "sql_query": "SELECT producto, SUM(cantidad) as total FROM ventas_preventivas GROUP BY producto ORDER BY total DESC LIMIT 10",
    "columns": ["producto", "total"],
    "num_rows": 10,
    "data_preview": [
      {"producto": "Tela Algodón", "total": 5000},
      {"producto": "Tela Poliéster", "total": 4500}
    ]
  }'
```

### 4. Usando Python Interactivo

```python
import requests
import json

url = "https://egsamaca56--viz-expert-model-predict.modal.run"

payload = {
    "user_query": "Muestra las ventas por mes",
    "sql_query": "SELECT mes, SUM(ventas) as total FROM ventas GROUP BY mes",
    "columns": ["mes", "total"],
    "num_rows": 12,
    "data_preview": [
        {"mes": "2024-01", "total": 15000},
        {"mes": "2024-02", "total": 18000}
    ]
}

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

## 📋 Formato del Request

El endpoint espera un JSON con los siguientes campos:

```json
{
  "user_query": "string - Consulta del usuario",
  "sql_query": "string - Consulta SQL ejecutada",
  "columns": ["array", "de", "columnas"],
  "num_rows": 10,
  "data_preview": [
    {"columna1": "valor1", "columna2": "valor2"}
  ]
}
```

## 📊 Formato de la Respuesta

El endpoint devuelve un JSON con la predicción del tipo de gráfico:

```json
{
  "chart_type": "bar|line|pie|scatter|etc",
  "reasoning": "Explicación de por qué se eligió este gráfico",
  "config": {
    "x_axis": "columna_x",
    "y_axis": "columna_y"
  }
}
```

O en caso de error:

```json
{
  "error": "Mensaje de error",
  "raw": "Respuesta cruda del modelo"
}
```

## ✅ Criterios de Validación

Una buena predicción debe:

1. ✅ **Responder en menos de 60 segundos** (timeout del endpoint)
2. ✅ **Devolver JSON válido** sin errores de parsing
3. ✅ **Incluir `chart_type`** con un tipo de gráfico válido
4. ✅ **Ser coherente** con el tipo de datos y la consulta
5. ✅ **Incluir razonamiento** (si está disponible) que explique la elección

## 🎯 Casos de Prueba Sugeridos

### Caso 1: Top N elementos (Gráfico de barras)
- **Query**: "Muestra los 10 productos más vendidos"
- **Esperado**: `chart_type: "bar"`

### Caso 2: Serie temporal (Gráfico de línea)
- **Query**: "Muestra las ventas por mes"
- **Esperado**: `chart_type: "line"`

### Caso 3: Distribución (Gráfico de pastel)
- **Query**: "Muestra la distribución por categoría"
- **Esperado**: `chart_type: "pie"`

### Caso 4: Comparación (Gráfico de barras)
- **Query**: "Compara ventas por región"
- **Esperado**: `chart_type: "bar"`

## 🔍 Debugging

Si el endpoint no responde correctamente:

1. **Verificar que el endpoint esté desplegado:**
   ```bash
   curl https://egsamaca56--viz-expert-model-predict.modal.run
   ```

2. **Revisar logs en Modal:**
   ```bash
   modal app logs viz-expert-model
   ```

3. **Probar localmente primero:**
   ```bash
   modal run modal_viz_model.py
   ```

4. **Verificar el formato del JSON** enviado

## 📝 Notas

- El endpoint tiene un timeout de 60 segundos
- La primera llamada puede tardar más (cold start) mientras carga el modelo
- Las siguientes llamadas serán más rápidas (modelo cacheado)
- El modelo usa quantización 4-bit para optimizar memoria

