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