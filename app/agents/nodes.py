"""
Nodos del grafo LangGraph.
Cada nodo es una función que recibe y retorna AgentState.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from pathlib import Path
from app.tools.viz_tool import viz_tool
from app.rag.vectorstore import vectorstore
from app.intelligence.hybrid_system import HybridVizSystem
from app.config import settings
from app.tools.professional_viz import professional_viz_tool

hybrid_viz = HybridVizSystem(
    finetuned_endpoint=settings.FINETUNED_MODEL_ENDPOINT
)

# Agregar el directorio raíz del proyecto al PYTHONPATH si se ejecuta directamente
if __name__ == "__main__":
    # Obtener el directorio raíz del proyecto (2 niveles arriba de este archivo)
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
import logging
import json
import time
import uuid

from app.agents.state import AgentState
from app.llm.models import (
    get_llama_model,
    invoke_llm_with_retry,
    get_router_prompt,
    get_sql_prompt,
    get_sql_correction_prompt,
    get_kpi_prompt,
    get_viz_prompt,
    get_general_prompt,
    get_embedding_model
)
from app.tools.sql_tool import mysql_tool
from app.db.connections import get_postgres
from app.agents.sql_validator import validate_and_execute_sql, get_table_schema
from app.validators.data_validator import (
    validate_user_query,
    validate_sql_results,
    validate_data_for_chart,
    validate_rag_context,
    ValidationResult
)
from app.metrics.performance_tracker import (
    track_router_decision,
    track_sql_execution,
    track_viz_generation,
    track_hybrid_execution
)

logger = logging.getLogger(__name__)

# ============ Helper Functions para Estados de Error ============

def error_state_with_message(msg: str, state: AgentState) -> AgentState:
    """
    Crea un estado de error con mensaje técnico.
    
    Args:
        msg: Mensaje de error técnico
        state: Estado actual del agente
    
    Returns:
        Estado con error configurado
    """
    state['error'] = msg
    state['status'] = 'error'
    logger.error(f"❌ Error state: {msg}")
    return state


def friendly_error_state(user_msg: str, state: AgentState) -> AgentState:
    """
    Crea un estado de error con mensaje amigable para el usuario.
    
    Args:
        user_msg: Mensaje amigable para el usuario (no técnico)
        state: Estado actual del agente
    
    Returns:
        Estado con respuesta amigable
    """
    from langchain_core.messages import AIMessage
    
    state['response'] = user_msg
    state['status'] = 'completed'
    state['messages'].append(AIMessage(content=user_msg))
    logger.info(f"ℹ️ Friendly error message: {user_msg}")
    return state


def load_kpi_definitions() -> dict:
    """Carga definiciones de KPIs desde el archivo JSON"""
    json_path = Path(__file__).parent.parent.parent / "data" / "sql_examples.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('kpi_definitions', {})


def hybrid_node(state: AgentState) -> AgentState:
    """
    Nodo híbrido que combina SQL + KPI + Visualización.
    
    Ejecuta secuencialmente:
    1. SQL node para obtener datos
    2. KPI node para calcular métricas
    3. Viz node para generar gráfica
    
    Args:
        state: Estado actual
    
    Returns:
        Estado con todos los resultados combinados
    """
    start_time = time.time()
    logger.info("=== Hybrid Node ===")

    # Variables para tracking
    query = state.get('user_query', '')
    success = False
    error_message = None
    sql_latency = None
    viz_latency = None

    try:
        # 1. Ejecutar SQL
        logger.info("Step 1: Executing SQL")
        state = sql_node(state)

        if state.get('error'):
            return state

        # 2. Calcular KPIs (solo si NO es una query de visualización)
        # Si el usuario pide gráfica, NO calcular KPIs porque sobrescribe los datos
        query_lower = state['user_query'].lower()
        is_viz_query = any(kw in query_lower for kw in ['grafica', 'gráfica', 'chart', 'plot', 'visualiza', 'muestra gráfica'])

        if 'sql_results' in state and state['sql_results'] and not is_viz_query:
            logger.info("Step 2: Calculating KPIs")
            state = kpi_node(state)
        elif is_viz_query:
            logger.info("Step 2: Skipping KPIs (visualization query detected)")

        # 3. Generar visualización
        if state.get('sql_results'):
            logger.info("Step 3: Generating visualization")
            logger.info(f"   SQL Query: {state.get('sql_query', 'N/A')}")
            logger.info(f"   Results: {len(state.get('sql_results', []))} rows")
            
            # VALIDACIÓN ESPECÍFICA PARA DATOS HÍBRIDOS
            sql_results = state.get('sql_results', [])
            kpis = state.get('kpis', {})
            chart_decision = state.get('chart_config', {})
            
            if sql_results and chart_decision:
                # Validar datos para el tipo de gráfica decidido
                chart_type = chart_decision.get('chart_type', 'bar')
                try:
                    hybrid_validation = validate_data_for_chart(sql_results, chart_type)
                    if not hybrid_validation.is_valid:
                        logger.warning(f"⚠️ Hybrid data validation failed: {hybrid_validation.error_msg}")
                        # No abortar, dejar que viz_node maneje la corrección
                    else:
                        logger.info(f"✅ Hybrid data validation passed | Chart type: {chart_type}")
                except Exception as e:
                    logger.warning(f"⚠️ Error validando datos híbridos: {e}. Continuando...", exc_info=True)
            
            if sql_results and kpis:
                logger.info("✅ Hybrid data validation: Both SQL results and KPIs available")
            elif not sql_results:
                logger.warning("⚠️ Hybrid data validation: No SQL results available")
            
            state = viz_node(state)

        # 4. Generar resumen combinado
        llm = get_llama_model()

        summary_parts = []

        if state.get('sql_results'):
            summary_parts.append(f"**Datos:** {len(state['sql_results'])} registros obtenidos")

        if state.get('kpis'):
            kpi_summary = "\n".join([
                f"- {kpi['name']}: {kpi['formatted_value']}"
                for kpi in state['kpis'].values()
            ])
            summary_parts.append(f"**KPIs:**\n{kpi_summary}")

        if state.get('chart_config'):
            summary_parts.append(f"**Visualización:** Gráfica {state['chart_config']['chart_type']} generada")

        combined_summary = "\n\n".join(summary_parts)

        # Generar explicación narrativa con LLM
        explanation_prompt = f"""Eres un analista de datos. Resume estos resultados de forma clara y concisa:

{combined_summary}

Pregunta original del usuario: {state['user_query']}

Genera un resumen ejecutivo de 2-3 párrafos explicando los hallazgos principales."""

        explanation = invoke_llm_with_retry(
            llm,
            [{"role": "user", "content": explanation_prompt}]
        )

        # Actualizar mensajes
        final_response = f"{explanation}\n\n---\n\n{combined_summary}"
        state['messages'].append(AIMessage(content=final_response))

        state['intermediate_steps'].append({
            'node': 'hybrid',
            'steps_executed': ['sql', 'kpi', 'viz'],
            'success': True
        })

        logger.info("✓ Hybrid node completed successfully")
        success = True

        return state

    except Exception as e:
        logger.error(f"Error in hybrid_node: {e}")
        success = False
        error_message = str(e)
        state['error'] = error_message
        return state
    
    finally:
        # Tracking al final (siempre se ejecuta)
        try:
            latency_ms = int((time.time() - start_time) * 1000)
            track_hybrid_execution(
                query=query,
                success=success,
                latency_ms=latency_ms,
                session_id=state.get('session_id'),
                sql_latency=sql_latency,
                viz_latency=viz_latency,
                error_message=error_message
            )
        except Exception as e:
            logger.debug(f"Error tracking hybrid execution: {e}")




# ============ Router Node ============

def search_router_examples(query: str, top_k: int = 3) -> list:
    """
    Busca ejemplos similares en router_examples usando búsqueda vectorial.
    
    Args:
        query: Query del usuario
        top_k: Número de ejemplos a recuperar
    
    Returns:
        Lista de tuplas (query, intent, reasoning, similarity)
    """
    try:
        embedding_model = get_embedding_model()
        postgres = get_postgres()
        
        # Generar embedding del query
        query_embedding = embedding_model.encode(query).tolist()
        query_embedding_str = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"
        
        # Buscar ejemplos similares
        search_sql = """
        SELECT
            query,
            intent,
            reasoning,
            1 - (embedding <=> %s::vector) as similarity
        FROM router_examples
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        with postgres.get_session() as session:
            # Obtener conexión raw de psycopg2
            conn = session.connection()
            cursor = conn.connection.cursor()
            cursor.execute(search_sql, (query_embedding_str, query_embedding_str, top_k))
            results = cursor.fetchall()
            cursor.close()
        
        return results
        
    except Exception as e:
        logger.warning(f"Error searching router examples: {e}")
        return []


def router_node(state: AgentState) -> AgentState:
    """
    Clasifica la intención del usuario usando RAG + Few-Shot classification.
    
    Mejoras:
    - Usa búsqueda vectorial en router_examples para encontrar ejemplos similares
    - Votación mayoritaria de los top 3 ejemplos más similares
    - Fallback a LLM si hay empate o baja confianza
    - Mantiene detección de saludos/general para queries obvias
    
    Args:
        state: Estado actual
    
    Returns:
        Estado con 'intent' actualizado
    """
    start_time = time.time()
    logger.info("=== Router Node (RAG + Few-Shot) ===")
    
    # Variables para tracking
    query = state.get('user_query') or ''
    intent = 'unknown'
    confidence = 0.0
    rag_similarity_avg = 0.0
    error_msg = None
    
    # Obtener user_query del estado - puede estar en diferentes lugares
    user_query = query
    
    # Si está vacío, intentar obtenerlo de los mensajes
    if not user_query and state.get('messages'):
        for msg in state['messages']:
            if hasattr(msg, 'content') and msg.content:
                user_query = msg.content
                break
    
    # Normalizar (pero mantener original para búsqueda vectorial)
    user_query_original = str(user_query).strip() if user_query else ''
    user_query_lower = user_query_original.lower().strip() if user_query_original else ''
    
    logger.info(f"user_query extracted: '{user_query_original}'")
    
    if not user_query_original:
        logger.error(f"user_query is empty! State: {list(state.keys())}, messages: {len(state.get('messages', []))}")
        # Fallback: intentar obtener del estado original
        user_query_original = str(state.get('user_query', '')).strip()
        user_query_lower = user_query_original.lower()

    try:
        # PRIORIDAD 1: Detectar saludos y preguntas generales PRIMERO
        # Estas deben tener la máxima prioridad para evitar clasificación incorrecta
        general_keywords = [
            'hola', 'hi', 'hello', 'buenos días', 'buenas tardes', 'buenas noches',
            'ayuda', 'help', 'qué puedes', 'que puedes', 'qué puedes hacer', 'que puedes hacer',
            'qué es', 'que es', 'explícame', 'explicame', 'cómo funciona', 'como funciona',
            'gracias', 'thanks', 'thank you', 'adios', 'adiós', 'bye', 'hasta luego',
            'qué haces', 'que haces', 'para qué sirves', 'para que sirves'
        ]
        
        general_matches = [word for word in general_keywords if word in user_query_lower]
        
        # Si es claramente un saludo/general, clasificar directamente
        if general_matches:
            intent = 'general'
            confidence = 1.0  # Máxima confianza para keywords
            logger.info(f"General keyword detection: {intent} (matched: {general_matches})")
            state['intent'] = intent
            state['confidence'] = confidence
            state['intermediate_steps'].append({
                'node': 'router',
                'intent': intent,
                'method': 'keyword',
                'query': user_query_original
            })
            
            # Tracking
            latency_ms = int((time.time() - start_time) * 1000)
            try:
                track_router_decision(
                    query=user_query_original,
                    intent=intent,
                    confidence=confidence,
                    rag_similarity=0.0,
                    latency_ms=latency_ms,
                    session_id=state.get('session_id'),
                    error_message=None
                )
            except Exception as e:
                logger.debug(f"Error tracking router decision: {e}")
            
            return state
        
        # PRIORIDAD 2: Usar RAG + Few-Shot classification
        logger.info("Using RAG + Few-Shot classification")
        
        # Buscar los 3 ejemplos más similares
        rag_results = search_router_examples(user_query_original, top_k=3)
        
        if rag_results and len(rag_results) > 0:
            logger.info(f"Found {len(rag_results)} similar examples from RAG")
            
            # Extraer intents y similitudes
            intent_votes = {}
            similarities = []
            
            for query_ex, intent_ex, reasoning_ex, similarity in rag_results:
                intent_votes[intent_ex] = intent_votes.get(intent_ex, 0) + 1
                similarities.append(similarity)
                logger.info(f"  [{intent_ex}] (sim: {similarity:.3f}) - '{query_ex}'")
            
            # Calcular similitud promedio (confianza)
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            rag_similarity_avg = avg_similarity
            logger.info(f"Average similarity: {avg_similarity:.3f}")
            
            # Votación mayoritaria
            if intent_votes:
                # Ordenar por votos (y luego por similitud si hay empate)
                sorted_intents = sorted(
                    intent_votes.items(),
                    key=lambda x: (x[1], max([s for q, i, r, s in rag_results if i == x[0]], default=0)),
                    reverse=True
                )
                
                predicted_intent = sorted_intents[0][0]
                votes = sorted_intents[0][1]
                total_votes = sum(intent_votes.values())
                
                logger.info(f"Voting result: {predicted_intent} ({votes}/{total_votes} votes, confidence: {avg_similarity:.3f})")
                
                # Verificar condiciones para usar RAG o fallback a LLM
                has_tie = len([v for v in intent_votes.values() if v == votes]) > 1
                low_confidence = avg_similarity < 0.6  # Umbral de confianza
                
                if has_tie or low_confidence:
                    logger.info(f"Tie detected or low confidence (tie: {has_tie}, confidence: {avg_similarity:.3f}), using LLM fallback")
                    # Fallback a LLM
                    llm = get_llama_model()
                    prompt = get_router_prompt()
                    
                    # Incluir ejemplos RAG en el prompt para few-shot
                    examples_text = "\n".join([
                        f"- Query: '{q}' → Intent: {i} (sim: {s:.3f})"
                        for q, i, r, s in rag_results
                    ])
                    
                    formatted_prompt = f"""{prompt.format(user_query=user_query_original)}

Ejemplos similares encontrados:
{examples_text}

Basándote en estos ejemplos, clasifica la query del usuario."""
                    
                    llm_intent = invoke_llm_with_retry(
                        llm,
                        [{"role": "user", "content": formatted_prompt}]
                    )
                    
                    llm_intent = llm_intent.strip().lower()
                    logger.info(f"LLM classification (with RAG examples): {llm_intent}")
                    intent = llm_intent
                else:
                    # Usar resultado de votación
                    intent = predicted_intent
                    logger.info(f"Using RAG voting result: {intent}")
            else:
                # No se encontraron ejemplos, usar LLM
                logger.info("No examples found in RAG, using LLM")
                llm = get_llama_model()
                prompt = get_router_prompt()
                formatted_prompt = prompt.format(user_query=user_query_original)
                
                llm_intent = invoke_llm_with_retry(
                    llm,
                    [{"role": "user", "content": formatted_prompt}]
                )
                
                llm_intent = llm_intent.strip().lower()
                logger.info(f"LLM classification: {llm_intent}")
                intent = llm_intent
        else:
            # No se encontraron ejemplos en RAG, usar LLM directamente
            logger.info("No RAG results, using LLM for classification")
            llm = get_llama_model()
            prompt = get_router_prompt()
            formatted_prompt = prompt.format(user_query=user_query_original)
            
            llm_intent = invoke_llm_with_retry(
                llm,
                [{"role": "user", "content": formatted_prompt}]
            )
            
            llm_intent = llm_intent.strip().lower()
            logger.info(f"LLM classification: {llm_intent}")
            intent = llm_intent

        # Validar intent
        valid_intents = ['sql', 'kpi', 'viz', 'general', 'hybrid']
        if intent not in valid_intents:
            logger.warning(f"Invalid intent '{intent}', defaulting to 'sql'")
            intent = 'sql'  # Default a SQL si hay duda (más útil que general)

        logger.info(f"✓ Final intent: {intent}")

        # Calcular confidence basado en similitud promedio
        if rag_results and len(rag_results) > 0:
            confidence = rag_similarity_avg
        else:
            # Si usó LLM sin RAG, confianza media
            confidence = 0.5

        # Actualizar estado
        state['intent'] = intent
        state['confidence'] = confidence
        state['rag_similarity'] = rag_similarity_avg
        state['intermediate_steps'].append({
            'node': 'router',
            'intent': intent,
            'method': 'rag_fewshot' if rag_results else 'llm',
            'query': user_query_original,
            'rag_results_count': len(rag_results) if rag_results else 0
        })

        return state

    except Exception as e:
        logger.error(f"Error in router_node: {e}", exc_info=True)
        error_msg = str(e)
        state['error'] = error_msg
        state['intent'] = 'sql'  # Fallback seguro
        intent = 'sql'
        confidence = 0.0
        return state
    
    finally:
        # Tracking al final (siempre se ejecuta)
        try:
            latency_ms = int((time.time() - start_time) * 1000)
            track_router_decision(
                query=user_query_original,
                intent=intent,
                confidence=confidence,
                rag_similarity=rag_similarity_avg,
                latency_ms=latency_ms,
                session_id=state.get('session_id'),
                error_message=error_msg
            )
        except Exception as e:
            logger.debug(f"Error tracking router decision: {e}")


# ============ SQL Node ============

def sql_node(state: AgentState) -> AgentState:
    """
    Genera y ejecuta query SQL con RAG y Self-Correction.
    
    Características:
    - Usa RAG para obtener ejemplos relevantes
    - Self-correction: intenta corregir errores SQL automáticamente
    - Máximo 2 reintentos de corrección
    - Logging detallado de cada intento
    """
    start_time = time.time()
    logger.info("=== SQL Node (with RAG + Self-Correction) ===")

    # Variables para tracking
    query = state.get('user_query', '')
    sql_query = ""
    success = False
    error_type = None
    correction_attempts = 0
    rows_returned = 0
    error_message = None

    try:
        # VALIDACIÓN 1: Validar user_query ANTES de generar SQL
        try:
            validation = validate_user_query(state['user_query'])
            if not validation.is_valid:
                error_msg = validation.error_msg or "Query inválida"
                logger.error(f"❌ Query inválida: {error_msg} | Query: {state['user_query'][:50]}...")
                return error_state_with_message(
                    f"Query inválida: {error_msg}. Por favor, reformula tu pregunta.",
                    state
                )
            logger.info(f"✅ User query validation passed | Length: {validation.metadata.get('query_length', 0)}")
        except Exception as e:
            logger.error(f"❌ Error en validación de query: {e}", exc_info=True)
            # Fail-safe: continuar si el validador falla
            logger.warning("⚠️ Continuando sin validación de query debido a error en validador")
        
        llm = get_llama_model()
        sql_prompt = get_sql_prompt()
        correction_prompt = get_sql_correction_prompt()

        # Obtener schema info
        schema_info = mysql_tool.get_schema_info()
        detailed_schema = get_table_schema()

        # **Usar RAG para obtener ejemplos relevantes (con timeout)**
        try:
            # Obtener ejemplos usando search_similar para tener acceso a la lista completa
            rag_examples_list = []
            try:
                rag_examples_list = vectorstore.search_similar(
                    state['user_query'],
                    top_k=3
                )
                logger.info(f"Retrieved {len(rag_examples_list)} relevant examples from RAG")
                
                # VALIDACIÓN 2: Validar contexto RAG DESPUÉS de obtener ejemplos
                if rag_examples_list:
                    try:
                        rag_validation = validate_rag_context(rag_examples_list)
                        if not rag_validation.is_valid:
                            warning_msg = rag_validation.error_msg or "RAG context inválido"
                            logger.warning(f"⚠️ RAG falló: {warning_msg} | Continuando sin contexto RAG")
                        else:
                            avg_sim = rag_validation.metadata.get('avg_similarity', 0.0)
                            logger.info(f"✅ RAG context validation passed | Avg similarity: {avg_sim:.3f}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error validando RAG context: {e}. Continuing anyway.", exc_info=True)
                
                # Formatear ejemplos para el prompt
                relevant_examples = vectorstore.get_relevant_examples(
                    state['user_query'],
                    top_k=3
                )
            except Exception as e:
                logger.warning(f"RAG search_similar failed: {e}. Trying get_relevant_examples...")
                relevant_examples = vectorstore.get_relevant_examples(
                    state['user_query'],
                    top_k=3
                )
                logger.info("Retrieved relevant examples from RAG (formatted)")
        except Exception as e:
            logger.warning(f"RAG search failed or timed out: {e}. Continuing without examples.")
            relevant_examples = "No hay ejemplos similares disponibles."

        # Máximo de reintentos (incluyendo el intento inicial)
        max_retries = 2
        sql_query = ""
        results = None
        error_message = None
        correction_attempts_list = []

        for attempt in range(max_retries + 1):  # 0, 1, 2 (3 intentos totales)
            try:
                if attempt == 0:
                    # Primer intento: generar SQL desde cero
                    logger.info(f"🔄 Attempt {attempt + 1}/{max_retries + 1}: Generating initial SQL query...")
                    
                    # Formatear prompt con ejemplos relevantes
                    formatted_prompt = sql_prompt.format(
                        schema_info=schema_info,
                        examples=relevant_examples,
                        user_query=state['user_query']
                    )
                    
                    sql_query = invoke_llm_with_retry(
                        llm,
                        [{"role": "user", "content": formatted_prompt}]
                    )
                    
                    # Limpiar query (remover markdown, etc.)
                    sql_query = sql_query.strip()
                    if sql_query.startswith('```sql'):
                        sql_query = sql_query.split('```sql')[1].split('```')[0].strip()
                    elif sql_query.startswith('```'):
                        sql_query = sql_query.split('```')[1].split('```')[0].strip()
                    
                    logger.info(f"Generated SQL (attempt {attempt + 1}): {sql_query}")
                else:
                    # Intentos de corrección
                    logger.warning(f"🔄 Attempt {attempt + 1}/{max_retries + 1}: Attempting SQL self-correction...")
                    logger.warning(f"   Previous error: {error_message}")
                    logger.warning(f"   Previous SQL: {sql_query}")
                    
                    # Formatear prompt de corrección
                    formatted_correction_prompt = correction_prompt.format(
                        schema_info=detailed_schema,
                        original_sql=sql_query,
                        error_message=error_message,
                        user_query=state['user_query']
                    )
                    
                    corrected_sql = invoke_llm_with_retry(
                        llm,
                        [{"role": "user", "content": formatted_correction_prompt}]
                    )
                    
                    # Limpiar query corregida
                    corrected_sql = corrected_sql.strip()
                    if corrected_sql.startswith('```sql'):
                        corrected_sql = corrected_sql.split('```sql')[1].split('```')[0].strip()
                    elif corrected_sql.startswith('```'):
                        corrected_sql = corrected_sql.split('```')[1].split('```')[0].strip()
                    
                    sql_query = corrected_sql
                    correction_attempts = attempt  # Número de intentos de corrección
                    logger.info(f"Corrected SQL (attempt {attempt + 1}): {sql_query}")
                    correction_attempts_list.append({
                        'attempt': attempt,
                        'original_error': error_message,
                        'corrected_sql': sql_query
                    })

                # Validar y ejecutar SQL
                logger.info(f"Validating and executing SQL (attempt {attempt + 1})...")
                success, result = validate_and_execute_sql(sql_query)
                
                if success:
                    results = result
                    rows_returned = len(results)
                    success = True
                    logger.info(f"✅ SQL executed successfully on attempt {attempt + 1}: {rows_returned} rows returned")
                    
                    # VALIDACIÓN 3: Validar resultados SQL DESPUÉS de ejecutar
                    try:
                        results_validation = validate_sql_results(results, min_rows=1)
                        
                        if not results_validation.is_valid:
                            error_msg = results_validation.error_msg or "Resultados SQL inválidos"
                            logger.error(f"❌ Sin resultados: {error_msg} | Query: {sql_query[:50]}...")
                            success = False
                            error_message = error_msg
                            error_type = "validation_error"
                            return friendly_error_state(
                                "No encontré datos para tu consulta. Verifica los filtros o intenta con otra pregunta.",
                                state
                            )
                        
                        if results_validation.warnings:
                            for warning in results_validation.warnings:
                                logger.warning(f"⚠️ Advertencia datos: {warning}")
                            # Agregar warnings al estado para referencia
                            state['sql_validation_warnings'] = results_validation.warnings
                        
                        row_count = results_validation.metadata.get('row_count', len(results))
                        rows_returned = row_count
                        logger.info(f"✅ SQL results validation passed | Rows: {row_count}")
                    except Exception as e:
                        logger.error(f"❌ Error validando resultados SQL: {e}", exc_info=True)
                        # Fail-safe: continuar si el validador falla
                        logger.warning("⚠️ Continuando sin validación de resultados debido a error en validador")
                    
                    if attempt > 0:
                        correction_attempts = attempt
                        logger.info(f"🎉 Self-correction successful after {attempt} correction attempt(s)")
                    
                    # Guardar resultados en estado
                    state['sql_query'] = sql_query
                    state['sql_results'] = results
                    state['sql_correction_attempts'] = correction_attempts
                    
                    break  # Éxito, salir del loop
                else:
                    error_message = result
                    error_type = "sql_execution_error"
                    logger.warning(f"❌ SQL execution failed on attempt {attempt + 1}: {error_message}")
                    
                    if attempt < max_retries:
                        logger.info(f"   Will retry with correction (attempts remaining: {max_retries - attempt})")
                    else:
                        logger.error(f"   Max retries ({max_retries + 1}) reached. Giving up.")
                        break

            except Exception as e:
                error_message = f"Unexpected error during SQL generation/execution: {str(e)}"
                logger.error(f"❌ Error in attempt {attempt + 1}: {error_message}", exc_info=True)
                
                if attempt < max_retries:
                    logger.info(f"   Will retry with correction (attempts remaining: {max_retries - attempt})")
                else:
                    logger.error(f"   Max retries ({max_retries + 1}) reached. Giving up.")
                    break

        # Verificar si tuvimos éxito
        if results is None:
            # Todos los intentos fallaron
            logger.error(f"❌ All SQL execution attempts failed. Final error: {error_message}")
            state['error'] = f"Error ejecutando query SQL después de {max_retries + 1} intentos: {error_message}"
            state['sql_query'] = sql_query  # Guardar la última query intentada
            state['sql_results'] = None
            state['intermediate_steps'].append({
                'node': 'sql',
                'query': sql_query,
                'error': error_message,
                'correction_attempts': correction_attempts,
                'num_attempts': max_retries + 1
            })
            return state

        # Éxito: actualizar estado
        state['sql_query'] = sql_query
        state['sql_results'] = results
        state['intermediate_steps'].append({
            'node': 'sql',
            'query': sql_query,
            'num_results': len(results),
            'correction_attempts': correction_attempts if correction_attempts else None,
            'num_attempts': correction_attempts + 1  # 1 intento inicial + N correcciones
        })

        # Añadir mensaje
        if correction_attempts > 0:
            state['messages'].append(
                AIMessage(content=f"Ejecuté la query (corregida después de {correction_attempts} intento(s)): {sql_query}")
            )
        else:
            state['messages'].append(
                AIMessage(content=f"Ejecuté la query: {sql_query}")
            )

        return state

    except Exception as e:
        logger.error(f"Error in sql_node: {e}", exc_info=True)
        success = False
        error_message = str(e)
        error_type = type(e).__name__
        state['error'] = error_message
        state['sql_query'] = None
        state['sql_results'] = []
        return state
    
    finally:
        # Tracking al final (siempre se ejecuta)
        try:
            latency_ms = int((time.time() - start_time) * 1000)
            track_sql_execution(
                query=query,
                sql_query=sql_query,
                success=success,
                latency_ms=latency_ms,
                rows_returned=rows_returned,
                error_type=error_type,
                correction_attempts=correction_attempts,
                session_id=state.get('session_id'),
                error_message=error_message
            )
        except Exception as e:
            logger.debug(f"Error tracking SQL execution: {e}")


# ============ Format Results Node ============

def format_results(state: AgentState) -> AgentState:
    """
    Formatea los resultados en lenguaje natural.
    
    Args:
        state: Estado actual
    
    Returns:
        Estado con respuesta final formateada
    """
    logger.info("=== Format Results Node ===")

    try:
        # Si hay error, reportarlo
        if state.get('error'):
            response = f"Lo siento, ocurrió un error: {state['error']}"
            state['messages'].append(AIMessage(content=response))
            return state

        # Si es intent general, ya está manejado
        if state['intent'] == 'general':
            return state

        # Para SQL, formatear resultados
        if state['intent'] == 'sql' and state.get('sql_results'):
            results = state['sql_results']

            # Si hay muchos resultados, resumir
            if len(results) > 10:
                summary = f"La query retornó {len(results)} resultados. Aquí los primeros 10:\n\n"
                results_to_show = results[:10]
            else:
                summary = f"La query retornó {len(results)} resultado(s):\n\n"
                results_to_show = results

            # Formatear como tabla simple
            if results_to_show:
                # Headers
                headers = list(results_to_show[0].keys())
                summary += " | ".join(headers) + "\n"
                summary += "-" * (len(summary) - 2) + "\n"

                # Rows
                for row in results_to_show:
                    summary += " | ".join(str(v) for v in row.values()) + "\n"

            state['messages'].append(AIMessage(content=summary))

        return state

    except Exception as e:
        logger.error(f"Error in format_results: {e}")
        state['error'] = str(e)
        return state


# ============ General Node (placeholder) ============

def general_node(state: AgentState) -> AgentState:
    """
    Responde preguntas generales sin datos.
    
    Args:
        state: Estado actual
    
    Returns:
        Estado con respuesta general
    """
    logger.info("=== General Node ===")

    try:
        llm = get_llama_model()
        prompt = get_general_prompt()

        formatted_prompt = prompt.format(user_query=state['user_query'])

        response = invoke_llm_with_retry(
            llm,
            [{"role": "user", "content": formatted_prompt}]
        )

        state['messages'].append(AIMessage(content=response))

        return state

    except Exception as e:
        logger.error(f"Error in general_node: {e}")
        state['error'] = str(e)
        return state


# Placeholders para otros nodos (implementaremos después)
def kpi_node(state: AgentState) -> AgentState:
    """
    Calcula KPIs (Key Performance Indicators).
    
    Puede:
    1. Usar datos de sql_results si ya existen
    2. Ejecutar queries propias para calcular KPIs
    3. Aplicar fórmulas predefinidas
    
    Args:
        state: Estado actual
    
    Returns:
        Estado con 'kpis' actualizados
    """
    logger.info("=== KPI Node ===")

    try:
        user_query = state['user_query'].lower()

        # Cargar definiciones de KPIs
        kpi_defs = load_kpi_definitions()

        # Detectar qué KPI(s) solicita el usuario
        requested_kpis = []
        for kpi_key, kpi_info in kpi_defs.items():
            kpi_name = kpi_info['name'].lower()
            if kpi_name in user_query or kpi_key in user_query:
                requested_kpis.append((kpi_key, kpi_info))

        # Si no se detecta KPI específico, calcular los principales
        if not requested_kpis:
            logger.info("No specific KPI requested, calculating main KPIs")
            main_kpis = ['revenue_total', 'ticket_promedio', 'num_ventas', 'unidades_vendidas']
            requested_kpis = [(k, kpi_defs[k]) for k in main_kpis if k in kpi_defs]

        # Calcular cada KPI
        calculated_kpis = {}

        for kpi_key, kpi_info in requested_kpis:
            try:
                logger.info(f"Calculating KPI: {kpi_info['name']}")

                # Ejecutar query del KPI
                result = mysql_tool._run(kpi_info['sql'])

                if result:
                    # Extraer valor (primera fila, primer campo)
                    value = list(result[0].values())[0]

                    # Formatear según tipo
                    if kpi_info['format'] == 'currency':
                        formatted_value = f"${value:,.2f}" if value else "$0.00"
                    elif kpi_info['format'] == 'number':
                        formatted_value = f"{int(value):,}" if value else "0"
                    elif kpi_info['format'] == 'percentage':
                        formatted_value = f"{value:.2f}%" if value else "0%"
                    else:
                        formatted_value = str(value)

                    calculated_kpis[kpi_key] = {
                        'name': kpi_info['name'],
                        'value': value,
                        'formatted_value': formatted_value,
                        'description': kpi_info['description']
                    }

                    logger.info(f"  {kpi_info['name']}: {formatted_value}")

            except Exception as e:
                logger.error(f"Error calculating KPI {kpi_key}: {e}")
                calculated_kpis[kpi_key] = {
                    'name': kpi_info['name'],
                    'value': None,
                    'formatted_value': 'N/A',
                    'description': kpi_info['description'],
                    'error': str(e)
                }

        # Si usamos datos existentes de sql_results, calcular KPIs adicionales
        if state.get('sql_results'):
            logger.info("Using existing sql_results for additional KPIs")
            df = pd.DataFrame(state['sql_results'])

            # Calcular estadísticas básicas si hay columnas numéricas
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                calculated_kpis[f'stats_{col}'] = {
                    'name': f'Estadísticas de {col}',
                    'value': {
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'std': float(df[col].std())
                    },
                    'formatted_value': f"Promedio: {df[col].mean():.2f}",
                    'description': f'Estadísticas descriptivas de {col}'
                }

        # Generar explicación con LLM
        llm = get_llama_model()
        prompt = get_kpi_prompt()

        kpi_summary = "\n".join([
            f"- {kpi['name']}: {kpi['formatted_value']} ({kpi['description']})"
            for kpi in calculated_kpis.values()
        ])

        formatted_prompt = prompt.format(
            sql_results=json.dumps(calculated_kpis, indent=2),
            user_query=state['user_query']
        )

        explanation = invoke_llm_with_retry(
            llm,
            [{"role": "user", "content": formatted_prompt}]
        )

        # Verificar si el usuario pidió una gráfica de KPIs
        user_query_lower = state['user_query'].lower()
        wants_chart = any(word in user_query_lower for word in ['gráfica', 'grafica', 'chart', 'plot', 'visualiza', 'gráfico', 'grafico'])
        
        # Si pidió gráfica y tenemos KPIs, preparar datos para visualización
        if wants_chart and calculated_kpis:
            logger.info("User requested chart for KPIs, preparing visualization data")
            # Convertir KPIs a formato de datos para graficar
            kpi_chart_data = []
            for kpi_key, kpi_info in calculated_kpis.items():
                # Solo incluir KPIs con valores numéricos simples
                if isinstance(kpi_info.get('value'), (int, float)):
                    kpi_chart_data.append({
                        'kpi': kpi_info['name'],
                        'valor': kpi_info['value'],
                        'formatted': kpi_info.get('formatted_value', str(kpi_info['value']))
                    })
            
            if kpi_chart_data:
                # Guardar datos para que el nodo VIZ los use
                state['sql_results'] = kpi_chart_data
                logger.info(f"Prepared {len(kpi_chart_data)} KPIs for chart visualization")
        
        # Actualizar estado
        state['kpis'] = calculated_kpis
        state['intermediate_steps'].append({
            'node': 'kpi',
            'kpis_calculated': list(calculated_kpis.keys()),
            'num_kpis': len(calculated_kpis)
        })

        # Añadir mensaje con resumen
        response = f"**KPIs Calculados:**\n\n{kpi_summary}\n\n{explanation}"
        state['messages'].append(AIMessage(content=response))

        logger.info(f"✓ Calculated {len(calculated_kpis)} KPIs")

        return state

    except Exception as e:
        logger.error(f"Error in kpi_node: {e}")
        state['error'] = str(e)
        state['kpis'] = {}
        return state


def viz_node(state: AgentState) -> AgentState:
    """
    Genera visualizaciones usando sistema híbrido.
    Puede graficar:
    - Datos SQL (sql_results)
    - KPIs estadísticos (kpis)
    - Combinación de ambos
    """
    start_time = time.time()
    logger.info("=== Viz Node (Hybrid) ===")

    # Variables para tracking
    query = state.get('user_query', '')
    chart_type = 'unknown'
    layer_used = 'unknown'
    success = False
    error_message = None

    try:
        # Verificar si hay KPIs para graficar
        kpis = state.get('kpis', {})
        has_kpis = bool(kpis)
        
        # Si no hay datos SQL pero hay KPIs, crear datos para graficar KPIs
        if not state.get('sql_results') and has_kpis:
            logger.info("No SQL data but KPIs available, creating KPI chart data")
            # Convertir KPIs a formato de datos para graficar
            kpi_data = []
            for kpi_key, kpi_info in kpis.items():
                if isinstance(kpi_info.get('value'), (int, float)):
                    kpi_data.append({
                        'kpi': kpi_info['name'],
                        'valor': kpi_info['value'],
                        'formatted': kpi_info.get('formatted_value', str(kpi_info['value']))
                    })
            
            if kpi_data:
                state['sql_results'] = kpi_data
                logger.info(f"Created {len(kpi_data)} KPI data points for visualization")
        
        # Si no hay datos, ejecutar SQL primero
        if not state.get('sql_results'):
            logger.info("No data available, executing SQL first")
            state = sql_node(state)

        results = state.get('sql_results', [])

        if not results:
            raise ValueError("No data available for visualization")

        # VALIDACIÓN: Validar datos ANTES de crear gráfica
        # Primero necesitamos saber qué tipo de gráfica se va a usar
        # Por ahora validamos con el tipo que decide el sistema híbrido
        
        # USAR SISTEMA HÍBRIDO
        logger.info("Using HybridVizSystem for chart decision")
        sql_query = state.get('sql_query', '')
        chart_config = hybrid_viz.decide_chart(
            query=state['user_query'],
            sql_results=results,
            sql_query=sql_query
        )
        
        chart_type = chart_config.get('chart_type', 'bar')
        
        # VALIDACIÓN: Validar datos ANTES de crear gráfica
        try:
            chart_validation = validate_data_for_chart(
                data=results,
                chart_type=chart_type
            )
            
            if not chart_validation.is_valid:
                error_msg = chart_validation.error_msg or "Datos incompatibles con el tipo de gráfica"
                suggestions = chart_validation.metadata.get('suggestions', [])
                alternative_chart = chart_validation.metadata.get('alternative_chart')
                
                logger.error(f"❌ Datos incompatibles: {error_msg} | Chart type: {chart_type}")
                
                if alternative_chart:
                    logger.info(f"ℹ️ Sugerencia: usar {alternative_chart}")
                    # Auto-corregir a chart alternativo
                    chart_config['chart_type'] = alternative_chart
                    chart_config['reasoning'] = f"Tipo de gráfica cambiado a {alternative_chart} debido a: {error_msg}"
                    
                    # Validar nuevamente con el tipo alternativo
                    alt_validation = validate_data_for_chart(results, alternative_chart)
                    if alt_validation.is_valid:
                        logger.info(f"✅ Chart alternativo '{alternative_chart}' es válido")
                        chart_type = alternative_chart
                    else:
                        # Si el alternativo tampoco funciona, retornar error
                        return friendly_error_state(
                            "No puedo generar una gráfica con estos datos. " + 
                            (suggestions[0] if suggestions else "Intenta con otra consulta."),
                            state
                        )
                else:
                    # No hay alternativa, retornar error con sugerencias
                    suggestion_msg = suggestions[0] if suggestions else "Intenta con otra consulta."
                    return friendly_error_state(
                        f"No puedo generar una gráfica con estos datos. {suggestion_msg}",
                        state
                    )
            
            # Si hay sugerencias, loguearlas
            suggestions = chart_validation.metadata.get('suggestions', [])
            if suggestions:
                for suggestion in suggestions:
                    logger.info(f"ℹ️ Sugerencia de mejora: {suggestion}")
                chart_config['validation_suggestions'] = suggestions
            
        except Exception as e:
            logger.error(f"❌ Error validando datos para gráfica: {e}", exc_info=True)
            # Fail-safe: continuar si el validador falla
            logger.warning("⚠️ Continuando sin validación de datos debido a error en validador")
        logger.info("Generating professional chart")
        chart_result = professional_viz_tool.create_chart(
            data=results,
            chart_type=chart_config.get('chart_type'),
            config=chart_config
        )

        logger.info(f"Chart decided: {chart_config.get('chart_type')} (source: {chart_config.get('source')})")

        # Extraer información para tracking
        chart_type = chart_config.get('chart_type', 'unknown')
        layer_used = chart_config.get('source', 'unknown')  # 'rules', 'finetuned', 'llm'

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
        success = True

        return state

    except Exception as e:
        logger.error(f"Error in viz_node: {e}")
        success = False
        error_message = str(e)
        state['error'] = error_message
        return state
    
    finally:
        # Tracking al final (siempre se ejecuta)
        try:
            latency_ms = int((time.time() - start_time) * 1000)
            track_viz_generation(
                query=query,
                chart_type=chart_type,
                layer_used=layer_used,
                success=success,
                latency_ms=latency_ms,
                session_id=state.get('session_id'),
                error_message=error_message
            )
        except Exception as e:
            logger.debug(f"Error tracking viz generation: {e}")


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from app.agents.state import create_initial_state

    print("=== Testing Nodes ===\n")

    # Test 1: Router node
    print("1. Testing router_node:")
    state = create_initial_state("¿Cuántas ventas hay?")
    state = router_node(state)
    print(f"   Intent: {state['intent']}\n")

    # Test 2: SQL node
    print("2. Testing sql_node:")
    state = create_initial_state("¿Cuántas ventas hay en total?")
    state['intent'] = 'sql'
    state = sql_node(state)
    print(f"   SQL: {state['sql_query']}")
    print(f"   Results: {state['sql_results']}\n")
    
    # Test 3: Format results
    print("3. Testing format_results:")
    state = format_results(state)
    print(f"   Response: {state['messages'][-1].content[:200]}...")
    
from ..feedback.feedback_service import feedback_service
import time


def track_interaction_node(state: AgentState) -> AgentState:
    """
    Nodo para rastrear la interacción y guardar métricas
    Se ejecuta al final del workflow
    """
    logger.info("=== Track Interaction Node ===")
    
    try:
        # Calcular tiempo de respuesta
        start_time = state.get('start_time')
        if start_time:
            response_time_ms = int((time.time() - start_time) * 1000)
        else:
            logger.warning("start_time not found in state, using 0")
            response_time_ms = 0

        session_id = state.get('session_id', 'unknown')
        user_query = state.get('user_query', '')
        
        logger.info(f"Tracking interaction: session_id={session_id}, query_length={len(user_query)}, response_time={response_time_ms}ms")

        # Guardar interacción
        feedback_id = feedback_service.save_interaction(
            session_id=session_id,
            user_query=user_query,
            sql_generated=state.get('sql_query'),
            chart_type=state.get('chart_config', {}).get('type') if state.get('chart_config') else None,
            chart_config=state.get('chart_config'),
            response_time_ms=response_time_ms,
            error_occurred=bool(state.get('error')),
            error_message=state.get('error')
        )

        # Agregar feedback_id al state para el frontend
        state['feedback_id'] = feedback_id
        logger.info(f"✅ Feedback saved with ID: {feedback_id}")

    except Exception as e:
        logger.error(f"❌ Error guardando feedback: {e}", exc_info=True)
        # No fallar el workflow por error de tracking, pero registrar el error
        state['feedback_id'] = None
        state['error'] = state.get('error') or f"Error tracking interaction: {str(e)}"

    return state