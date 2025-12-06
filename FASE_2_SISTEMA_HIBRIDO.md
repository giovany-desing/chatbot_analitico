# 🔄 FASE 2: Sistema Híbrido Inteligente

## Objetivo
Implementar un sistema de 3 capas (Reglas + Modelo Fine-tuned + LLM) para decisiones inteligentes de gráficas con máxima precisión y mínimo costo.

---

## 📦 Prerequisitos

- Fase 1 completada (modelo fine-tuned deployado)
- URL del endpoint de Modal.com
- Proyecto actual funcionando

---

## 🔧 Paso 1: Crear Estructura de Archivos

```bash
# Crear nuevos archivos
mkdir -p app/intelligence
touch app/intelligence/__init__.py
touch app/intelligence/hybrid_system.py
touch app/intelligence/rules_engine.py
touch app/intelligence/finetuned_client.py
```

---

## 📝 Paso 2: Implementar Rules Engine

### 2.1 Crear motor de reglas determinísticas

```python
# app/intelligence/rules_engine.py

import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RuleResult:
    chart_type: Optional[str]
    confidence: ConfidenceLevel
    config: Optional[Dict[str, Any]]
    reasoning: str


class DeterministicRulesEngine:
    """
    Motor de reglas determinísticas para decisiones rápidas.
    Cubre ~70% de casos comunes sin usar IA.
    """

    def apply(self, query: str, results: List[Dict]) -> RuleResult:
        """
        Aplica reglas determinísticas.

        Returns:
            RuleResult con chart_type, confidence y config
        """
        if not results:
            return RuleResult(None, ConfidenceLevel.LOW, None, "Sin datos")

        df = pd.DataFrame(results)
        query_lower = query.lower()

        # REGLA 1: Keywords explícitas de tipo de gráfica
        explicit_result = self._check_explicit_keywords(query_lower, df)
        if explicit_result:
            return explicit_result

        # REGLA 2: Detectar columnas de fecha (temporal)
        date_result = self._check_date_columns(df, query_lower)
        if date_result:
            return date_result

        # REGLA 3: Top N / Ranking
        topn_result = self._check_top_n(query_lower, df)
        if topn_result:
            return topn_result

        # REGLA 4: Pocas categorías (pie chart)
        pie_result = self._check_pie_chart(query_lower, df)
        if pie_result:
            return pie_result

        # REGLA 5: Comparación explícita
        comparison_result = self._check_comparison(query_lower, df)
        if comparison_result:
            return comparison_result

        # REGLA 6: Distribución
        distribution_result = self._check_distribution(query_lower, df)
        if distribution_result:
            return distribution_result

        # REGLA 7: Scatter (correlación)
        scatter_result = self._check_scatter(query_lower, df)
        if scatter_result:
            return scatter_result

        # Sin match
        return RuleResult(None, ConfidenceLevel.LOW, None, "No rule matched")

    def _check_explicit_keywords(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta keywords explícitas del tipo de gráfica"""
        keywords_map = {
            'line': ['línea', 'line', 'tendencia', 'evolución', 'temporal'],
            'bar': ['barra', 'barras', 'bar'],
            'pie': ['pastel', 'pie', 'torta', 'circular'],
            'scatter': ['dispersión', 'scatter', 'puntos'],
            'histogram': ['histogram', 'histograma', 'distribución']
        }

        for chart_type, keywords in keywords_map.items():
            if any(kw in query for kw in keywords):
                config = self._auto_config(chart_type, df)
                return RuleResult(
                    chart_type=chart_type,
                    confidence=ConfidenceLevel.HIGH,
                    config=config,
                    reasoning=f"Keyword explícito detectado: {chart_type}"
                )
        return None

    def _check_date_columns(self, df: pd.DataFrame, query: str) -> Optional[RuleResult]:
        """Detecta columnas de fecha para line chart"""
        date_keywords = ['fecha', 'date', 'mes', 'month', 'año', 'year', 'trimestre', 'quarter']

        for col in df.columns:
            if any(kw in col.lower() for kw in date_keywords):
                config = self._line_chart_config(df, col)
                return RuleResult(
                    chart_type='line',
                    confidence=ConfidenceLevel.HIGH,
                    config=config,
                    reasoning=f"Columna temporal detectada: {col}"
                )
        return None

    def _check_top_n(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta queries de tipo Top N"""
        top_keywords = ['top', 'mejores', 'más vendidos', 'ranking', 'primeros', 'mayores']

        if any(kw in query for kw in top_keywords) and len(df) <= 20:
            config = self._bar_chart_config(df)
            return RuleResult(
                chart_type='bar',
                confidence=ConfidenceLevel.HIGH,
                config=config,
                reasoning="Top N detectado con cantidad razonable de items"
            )
        return None

    def _check_pie_chart(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta casos apropiados para pie chart"""
        pie_keywords = ['distribución', 'proporción', 'porcentaje', 'reparto']

        if (any(kw in query for kw in pie_keywords) and 2 <= len(df) <= 7):
            config = self._pie_chart_config(df)
            return RuleResult(
                chart_type='pie',
                confidence=ConfidenceLevel.HIGH,
                config=config,
                reasoning="Proporción con pocas categorías (2-7)"
            )
        return None

    def _check_comparison(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta comparaciones explícitas"""
        comparison_keywords = ['vs', 'versus', 'compara', 'comparación', 'diferencia']

        if any(kw in query for kw in comparison_keywords):
            config = self._bar_chart_config(df)
            return RuleResult(
                chart_type='bar',
                confidence=ConfidenceLevel.HIGH,
                config=config,
                reasoning="Comparación explícita detectada"
            )
        return None

    def _check_distribution(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta análisis de distribución"""
        if 'histogram' in query or 'frecuencia' in query:
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                config = self._histogram_config(df, numeric_cols[0])
                return RuleResult(
                    chart_type='histogram',
                    confidence=ConfidenceLevel.HIGH,
                    config=config,
                    reasoning="Distribución solicitada"
                )
        return None

    def _check_scatter(self, query: str, df: pd.DataFrame) -> Optional[RuleResult]:
        """Detecta necesidad de scatter plot"""
        scatter_keywords = ['correlación', 'relación', 'scatter']
        numeric_cols = df.select_dtypes(include=['number']).columns

        if any(kw in query for kw in scatter_keywords) and len(numeric_cols) >= 2:
            config = self._scatter_config(df, numeric_cols[0], numeric_cols[1])
            return RuleResult(
                chart_type='scatter',
                confidence=ConfidenceLevel.HIGH,
                config=config,
                reasoning="Correlación entre dos variables numéricas"
            )
        return None

    def _auto_config(self, chart_type: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Genera configuración automática según tipo"""
        if chart_type == 'line':
            return self._line_chart_config(df, df.columns[0])
        elif chart_type == 'bar':
            return self._bar_chart_config(df)
        elif chart_type == 'pie':
            return self._pie_chart_config(df)
        elif chart_type == 'scatter':
            numeric_cols = df.select_dtypes(include=['number']).columns
            return self._scatter_config(df, numeric_cols[0], numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
        elif chart_type == 'histogram':
            numeric_cols = df.select_dtypes(include=['number']).columns
            return self._histogram_config(df, numeric_cols[0])
        return {}

    def _line_chart_config(self, df: pd.DataFrame, x_col: str) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include=['number']).columns
        y_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]
        return {
            'chart_type': 'line',
            'x_column': x_col,
            'y_column': y_col,
            'title': f'Evolución de {y_col}',
            'x_label': x_col,
            'y_label': y_col
        }

    def _bar_chart_config(self, df: pd.DataFrame) -> Dict[str, Any]:
        x_col = df.columns[0]
        numeric_cols = df.select_dtypes(include=['number']).columns
        y_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]
        return {
            'chart_type': 'bar',
            'x_column': x_col,
            'y_column': y_col,
            'title': f'{y_col} por {x_col}',
            'x_label': x_col,
            'y_label': y_col
        }

    def _pie_chart_config(self, df: pd.DataFrame) -> Dict[str, Any]:
        x_col = df.columns[0]
        numeric_cols = df.select_dtypes(include=['number']).columns
        y_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[1]
        return {
            'chart_type': 'pie',
            'x_column': x_col,
            'y_column': y_col,
            'title': f'Distribución de {y_col}',
            'x_label': x_col,
            'y_label': y_col
        }

    def _histogram_config(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        return {
            'chart_type': 'histogram',
            'x_column': col,
            'y_column': None,
            'title': f'Distribución de {col}',
            'x_label': col,
            'y_label': 'Frecuencia'
        }

    def _scatter_config(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        return {
            'chart_type': 'scatter',
            'x_column': x_col,
            'y_column': y_col,
            'title': f'{y_col} vs {x_col}',
            'x_label': x_col,
            'y_label': y_col
        }
```

---

## 📝 Paso 3: Implementar Cliente del Modelo Fine-tuned

```python
# app/intelligence/finetuned_client.py

import requests
import json
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FineTunedResult:
    chart_type: Optional[str]
    confidence: float
    config: Optional[Dict[str, Any]]
    reasoning: str


class FineTunedVizClient:
    """
    Cliente para el modelo fine-tuned deployado en Modal.com
    """

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url

    def predict(self, query: str, results: List[Dict]) -> FineTunedResult:
        """
        Consulta el modelo fine-tuned.

        Returns:
            FineTunedResult con predicción y confianza
        """
        if not results:
            return FineTunedResult(None, 0.0, None, "Sin datos")

        try:
            # Preparar prompt
            prompt = self._format_prompt(query, results)

            # Llamar API
            response = requests.post(
                self.endpoint_url,
                json={"prompt": prompt},
                timeout=5
            )

            if response.status_code != 200:
                logger.error(f"Fine-tuned model error: {response.status_code}")
                return FineTunedResult(None, 0.0, None, f"HTTP {response.status_code}")

            # Parsear respuesta
            prediction = response.json().get("prediction", "")

            # Extraer JSON de la respuesta
            config = self._extract_json(prediction)

            if not config:
                return FineTunedResult(None, 0.0, None, "No se pudo parsear respuesta")

            return FineTunedResult(
                chart_type=config.get('chart_type'),
                confidence=config.get('confidence', 0.9),
                config=config,
                reasoning=config.get('reasoning', '')
            )

        except requests.exceptions.Timeout:
            logger.warning("Fine-tuned model timeout")
            return FineTunedResult(None, 0.0, None, "Timeout")
        except Exception as e:
            logger.error(f"Fine-tuned model error: {e}")
            return FineTunedResult(None, 0.0, None, str(e))

    def _format_prompt(self, query: str, results: List[Dict]) -> str:
        """
        Formatea prompt en el mismo formato usado en entrenamiento.
        """
        df = pd.DataFrame(results)

        # Análisis de datos
        num_rows = len(df)
        columns = list(df.columns)
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        categorical_cols = list(df.select_dtypes(include=['object']).columns)

        # Detectar tipos de columnas
        col_types = []
        for col in columns:
            if col in numeric_cols:
                col_types.append(f"Tipo_{col}: numérico")
            elif col in categorical_cols:
                col_types.append(f"Tipo_{col}: categórico")

        input_text = f"""Query: {query}
Datos: {num_rows} filas
Columnas: {columns}
{chr(10).join(col_types)}"""

        prompt = f"""A continuación hay una instrucción que describe una tarea, junto con una entrada que proporciona más contexto. Escribe una respuesta que complete apropiadamente la solicitud.

### Instrucción:
Decide la gráfica apropiada

### Entrada:
{input_text}

### Respuesta:
"""
        return prompt

    def _extract_json(self, text: str) -> Optional[Dict]:
        """
        Extrae JSON de la respuesta del modelo.
        """
        try:
            # Buscar JSON en la respuesta
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        return None
```

---

## 📝 Paso 4: Implementar Sistema Híbrido

```python
# app/intelligence/hybrid_system.py

from typing import Dict, Any, List
import logging
from app.intelligence.rules_engine import DeterministicRulesEngine, ConfidenceLevel
from app.intelligence.finetuned_client import FineTunedVizClient
from app.llm.models import get_llama_model, invoke_llm_with_retry, get_viz_prompt

logger = logging.getLogger(__name__)


class HybridVizSystem:
    """
    Sistema híbrido de 3 capas para decisión inteligente de gráficas.

    Flujo:
    1. Reglas determinísticas (instantáneo, gratis)
    2. Modelo fine-tuned (rápido, casi gratis)
    3. LLM grande (preciso, más caro)
    """

    def __init__(self, finetuned_endpoint: str):
        self.rules_engine = DeterministicRulesEngine()
        self.finetuned_client = FineTunedVizClient(finetuned_endpoint)
        self.llm_fallback = get_llama_model()

    def decide_chart(self, query: str, sql_results: List[Dict]) -> Dict[str, Any]:
        """
        Decide la mejor gráfica usando sistema híbrido.

        Returns:
            Dict con configuración de gráfica
        """
        logger.info(f"HybridVizSystem: Processing query")

        # CAPA 1: Reglas Determinísticas
        logger.info("Layer 1: Applying deterministic rules")
        rule_result = self.rules_engine.apply(query, sql_results)

        if rule_result.confidence == ConfidenceLevel.HIGH:
            logger.info(f"✓ Rule match: {rule_result.chart_type} ({rule_result.reasoning})")
            return {
                **rule_result.config,
                'source': 'rules',
                'reasoning': rule_result.reasoning
            }

        # CAPA 2: Modelo Fine-tuned
        logger.info("Layer 2: Consulting fine-tuned model")
        finetuned_result = self.finetuned_client.predict(query, sql_results)

        if finetuned_result.confidence >= 0.85:
            logger.info(f"✓ Fine-tuned: {finetuned_result.chart_type} (confidence: {finetuned_result.confidence})")
            return {
                **finetuned_result.config,
                'source': 'finetuned',
                'reasoning': finetuned_result.reasoning
            }

        # CAPA 3: LLM Grande (Llama 90B)
        logger.info("Layer 3: Fallback to LLM")
        llm_result = self._llm_decide(query, sql_results)

        return {
            **llm_result,
            'source': 'llm',
            'reasoning': llm_result.get('reasoning', 'LLM decision')
        }

    def _llm_decide(self, query: str, results: List[Dict]) -> Dict[str, Any]:
        """
        Usa LLM grande como fallback.
        """
        import json

        prompt = get_viz_prompt()

        formatted_prompt = prompt.format(
            sql_results=json.dumps(results[:5], indent=2),
            user_query=query
        )

        response = invoke_llm_with_retry(
            self.llm_fallback,
            [{"role": "user", "content": formatted_prompt}]
        )

        # Limpiar y parsear
        response = response.strip()
        if response.startswith('```json'):
            response = response.split('```json')[1].split('```')[0].strip()
        elif response.startswith('```'):
            response = response.split('```')[1].split('```')[0].strip()

        try:
            config = json.loads(response)
            return config
        except json.JSONDecodeError:
            # Fallback básico
            import pandas as pd
            df = pd.DataFrame(results)
            return {
                'chart_type': 'bar',
                'x_column': df.columns[0],
                'y_column': df.columns[1] if len(df.columns) > 1 else df.columns[0],
                'title': 'Análisis de Datos'
            }
```

---

## 📝 Paso 5: Integrar con viz_node

```python
# app/agents/nodes.py

# Al inicio del archivo, agregar:
from app.intelligence.hybrid_system import HybridVizSystem
from app.config import settings

# Crear instancia global (después de los imports)
hybrid_viz = HybridVizSystem(
    finetuned_endpoint=settings.FINETUNED_MODEL_ENDPOINT
)

# Modificar viz_node:
def viz_node(state: AgentState) -> AgentState:
    """
    Genera visualizaciones usando sistema híbrido.
    """
    logger.info("=== Viz Node (Hybrid) ===")

    try:
        # Si no hay datos, ejecutar SQL primero
        if not state.get('sql_results'):
            logger.info("No data available, executing SQL first")
            state = sql_node(state)

        results = state.get('sql_results', [])

        if not results:
            raise ValueError("No data available for visualization")

        # USAR SISTEMA HÍBRIDO
        logger.info("Using HybridVizSystem for chart decision")
        chart_config = hybrid_viz.decide_chart(
            query=state['user_query'],
            sql_results=results
        )

        logger.info(f"Chart decided: {chart_config.get('chart_type')} (source: {chart_config.get('source')})")

        # Generar gráfica con viz_tool
        from app.tools.viz_tool import viz_tool

        chart_result = viz_tool._run(
            data=results,
            chart_type=chart_config.get('chart_type'),
            x_column=chart_config.get('x_column'),
            y_column=chart_config.get('y_column'),
            title=chart_config.get('title'),
            x_label=chart_config.get('x_label'),
            y_label=chart_config.get('y_label')
        )

        # Actualizar estado
        state['chart_config'] = chart_result
        state['intermediate_steps'].append({
            'node': 'viz',
            'chart_type': chart_config.get('chart_type'),
            'decision_source': chart_config.get('source'),
            'reasoning': chart_config.get('reasoning')
        })

        # Mensaje
        response = f"📊 He generado una gráfica de tipo **{chart_config.get('chart_type')}**.\n\n"
        response += f"**Fuente de decisión:** {chart_config.get('source')}\n"
        response += f"**Razonamiento:** {chart_config.get('reasoning')}\n"

        from langchain_core.messages import AIMessage
        state['messages'].append(AIMessage(content=response))

        logger.info("✓ Viz node completed with hybrid system")

        return state

    except Exception as e:
        logger.error(f"Error in viz_node: {e}")
        state['error'] = str(e)
        return state
```

---

## 📝 Paso 6: Configurar Variables de Entorno

```bash
# .env

# Agregar URL del modelo fine-tuned
FINETUNED_MODEL_ENDPOINT=https://tu-usuario--viz-expert-model-predict.modal.run
```

```python
# app/config.py

# Agregar en la clase Settings:
class Settings(BaseSettings):
    # ... otros campos ...

    # Fine-tuned model
    FINETUNED_MODEL_ENDPOINT: str = Field(
        default="",
        description="URL del modelo fine-tuned en Modal.com"
    )
```

---

## 🧪 Plan de Pruebas

### Test 1: Reglas Determinísticas

```python
# tests/test_hybrid_system.py

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

# Ejecutar:
# pytest tests/test_hybrid_system.py -v
```

**✅ Output esperado:**
```
test_rules_engine_temporal PASSED
test_rules_engine_top_n PASSED
```

---

### Test 2: Sistema Híbrido Completo

```bash
# Crear test manual
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Gráfica de ventas mensuales del último año"
  }'
```

**✅ Output esperado:**
```json
{
  "response": "📊 He generado una gráfica de tipo **line**...",
  "intent": "viz",
  "chart_config": {
    "chart_type": "line",
    "source": "rules",
    "reasoning": "Columna temporal detectada: mes"
  }
}
```

---

### Test 3: Benchmark de Performance

```python
# scripts/benchmark_hybrid.py

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
```

**✅ Output esperado:**
```
✅ Test 1: Top 10 productos...
   Type: bar | Source: rules | Time: 5ms
✅ Test 2: Evolución de ventas por mes...
   Type: line | Source: rules | Time: 3ms
✅ Test 3: Distribución de ventas por re...
   Type: pie | Source: rules | Time: 4ms

📊 Resultados:
   Precisión: 3/3 (100.0%)
   Tiempo promedio: 4ms
```

---

## 📋 Checklist de Fase 2

- [ ] Archivos de estructura creados
- [ ] `rules_engine.py` implementado
- [ ] `finetuned_client.py` implementado
- [ ] `hybrid_system.py` implementado
- [ ] `viz_node` actualizado para usar sistema híbrido
- [ ] Variables de entorno configuradas
- [ ] Test 1 pasado (reglas determinísticas)
- [ ] Test 2 pasado (sistema completo)
- [ ] Test 3 pasado (benchmark ≥90% precisión, <100ms promedio)

---

## 🎯 Siguientes Pasos

Continúa con **FASE_3_GRAFICAS_PROFESIONALES.md**

---

## 💰 Costos Fase 2

- Todo gratis (usa infraestructura existente)
- Ahorro estimado: ~70% de llamadas a LLM (resueltas por reglas)
