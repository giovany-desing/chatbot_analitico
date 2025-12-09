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

    def predict(self, query: str, sql_query: str, results: List[Dict]) -> FineTunedResult:
        """
        Consulta el modelo fine-tuned usando el formato del endpoint Modal.
        
        Args:
            query: Query del usuario
            sql_query: Query SQL ejecutada
            results: Resultados de la query SQL
        
        Returns:
            FineTunedResult con predicción y confianza
        """
        if not self.endpoint_url:
            logger.warning("Fine-tuned model endpoint not configured, skipping")
            return FineTunedResult(None, 0.0, None, "Endpoint not configured")
        
        if not results:
            return FineTunedResult(None, 0.0, None, "Sin datos")

        try:
            # Preparar payload según formato del endpoint Modal
            columns = list(results[0].keys()) if results else []
            data_preview = results[:10]  # Primeras 10 filas como preview
            
            payload = {
                "user_query": query,
                "sql_query": sql_query,
                "columns": columns,
                "num_rows": len(results),
                "data_preview": data_preview
            }
            
            logger.info(f"Calling fine-tuned model at {self.endpoint_url}")
            logger.debug(f"Payload: {payload}")

            # Llamar API con timeout de 10 segundos
            response = requests.post(
                self.endpoint_url,
                json=payload,
                timeout=10
            )

            if response.status_code != 200:
                logger.error(f"Fine-tuned model error: HTTP {response.status_code} - {response.text}")
                return FineTunedResult(None, 0.0, None, f"HTTP {response.status_code}")

            # Parsear respuesta
            response_data = response.json()
            
            # El endpoint retorna: {"chart_type": "...", "reasoning": "...", "config": {...}}
            chart_type = response_data.get("chart_type")
            reasoning = response_data.get("reasoning", "")
            config = response_data.get("config", {})
            
            if not chart_type:
                logger.warning(f"Fine-tuned model returned invalid response: {response_data}")
                return FineTunedResult(None, 0.0, None, "Invalid response format")

            # Construir config completo
            full_config = {
                'chart_type': chart_type,
                'x_column': config.get('x_axis'),
                'y_column': config.get('y_axis'),
                'title': config.get('title', query),
                'x_label': config.get('x_label', config.get('x_axis')),
                'y_label': config.get('y_label', config.get('y_axis'))
            }

            logger.info(f"✓ Fine-tuned model prediction: {chart_type} (reasoning: {reasoning[:50]}...)")
            
            return FineTunedResult(
                chart_type=chart_type,
                confidence=0.9,  # Alta confianza si el modelo responde
                config=full_config,
                reasoning=reasoning
            )

        except requests.exceptions.Timeout:
            logger.warning("Fine-tuned model timeout after 10 seconds")
            return FineTunedResult(None, 0.0, None, "Timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Fine-tuned model request error: {e}")
            return FineTunedResult(None, 0.0, None, f"Request error: {str(e)}")
        except Exception as e:
            logger.error(f"Fine-tuned model error: {e}", exc_info=True)
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