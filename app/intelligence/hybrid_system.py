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

    def decide_chart(self, query: str, sql_results: List[Dict], sql_query: str = "") -> Dict[str, Any]:
        """
        Decide la mejor gráfica usando sistema híbrido.
        
        Args:
            query: Query del usuario
            sql_results: Resultados de la query SQL
            sql_query: Query SQL ejecutada (opcional, necesario para fine-tuned model)

        Returns:
            Dict con configuración de gráfica
        """
        logger.info(f"🔍 HybridVizSystem: Processing query")

        # CAPA 1: Reglas Determinísticas
        logger.info("🔹 Capa 1: Aplicando reglas determinísticas...")
        rule_result = self.rules_engine.apply(query, sql_results)

        if rule_result.confidence == ConfidenceLevel.HIGH:
            logger.info(f"✅ Capa 1 (Reglas) activada: {rule_result.chart_type}")
            logger.info(f"   Reasoning: {rule_result.reasoning}")
            return {
                **rule_result.config,
                'source': 'rules',
                'reasoning': rule_result.reasoning
            }
        
        logger.info(f"⚠️ Capa 1 (Reglas) no aplicable, continuando a Capa 2...")

        # CAPA 2: Modelo Fine-tuned
        logger.info("🔹 Capa 2: Consultando modelo fine-tuned...")
        
        finetuned_result = self.finetuned_client.predict(query, sql_query, sql_results)

        if finetuned_result.confidence >= 0.85 and finetuned_result.chart_type:
            logger.info(f"✅ Capa 2 (Fine-tuned) activada: {finetuned_result.chart_type} (confidence: {finetuned_result.confidence:.2f})")
            logger.info(f"   Reasoning: {finetuned_result.reasoning[:100]}...")
            return {
                **finetuned_result.config,
                'source': 'finetuned',
                'reasoning': finetuned_result.reasoning
            }
        else:
            logger.info(f"⚠️ Capa 2 (Fine-tuned) no aplicable o falló: {finetuned_result.reasoning}")

        # CAPA 3: LLM Grande (Llama 3.3 70B)
        logger.info("🔹 Capa 3: Fallback a LLM (Llama 3.3 70B)...")
        llm_result = self._llm_decide(query, sql_results)
        logger.info(f"✅ Capa 3 (LLM) activada: {llm_result.get('chart_type', 'unknown')}")

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