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

from app.agents.state import AgentState
from app.llm.models import (
    get_llama_model,
    invoke_llm_with_retry,
    get_router_prompt,
    get_sql_prompt,
    get_kpi_prompt,
    get_viz_prompt,
    get_general_prompt
)
from app.tools.sql_tool import mysql_tool

logger = logging.getLogger(__name__)

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
    logger.info("=== Hybrid Node ===")

    try:
        # 1. Ejecutar SQL
        logger.info("Step 1: Executing SQL")
        state = sql_node(state)

        if state.get('error'):
            return state

        # 2. Calcular KPIs (si tiene sentido)
        if state.get('sql_results'):
            logger.info("Step 2: Calculating KPIs")
            state = kpi_node(state)

        # 3. Generar visualización
        if state.get('sql_results'):
            logger.info("Step 3: Generating visualization")
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

        return state

    except Exception as e:
        logger.error(f"Error in hybrid_node: {e}")
        state['error'] = str(e)
        return state




# ============ Router Node ============

def router_node(state: AgentState) -> AgentState:
    """
    Clasifica la intención del usuario (versión mejorada).
    
    Mejoras:
    - Mejor detección de keywords
    - Fallback a SQL si hay duda
    - Logging más detallado
    
    Args:
        state: Estado actual
    
    Returns:
        Estado con 'intent' actualizado
    """
    logger.info("=== Router Node ===")
    
    # Obtener user_query del estado - puede estar en diferentes lugares
    user_query = state.get('user_query') or ''
    
    # Si está vacío, intentar obtenerlo de los mensajes
    if not user_query and state.get('messages'):
        for msg in state['messages']:
            if hasattr(msg, 'content') and msg.content:
                user_query = msg.content
                break
    
    # Normalizar
    user_query = str(user_query).lower().strip() if user_query else ''
    
    logger.info(f"user_query extracted: '{user_query}'")
    
    if not user_query:
        logger.error(f"user_query is empty! State: {list(state.keys())}, messages: {len(state.get('messages', []))}")
        # Fallback: intentar obtener del estado original
        user_query = str(state.get('user_query', '')).strip().lower()

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
        
        # PRIORIDAD 2: Detectar visualizaciones
        viz_keywords = ['gráfica', 'grafica', 'chart', 'plot', 'visualiza', 'visualización', 
                       'visualizacion', 'gráfico', 'grafico', 'diagrama', 'mostrar gráfica']
        
        # PRIORIDAD 3: Detectar KPIs (incluye palabras de negocio que son KPIs)
        kpi_keywords = [
            'kpi', 'métrica', 'metrica', 'revenue', 'ticket promedio', 
            'indicador', 'indicadores clave', 'ingresos', 'promedio',
            'ratio', 'porcentaje', 'estadística', 'estadisticas', 'estadísticas'
        ]
        
        # Detectar si se pide graficar KPIs específicos (debe ser hybrid)
        kpi_business_terms = ['revenue', 'ingresos', 'ticket', 'promedio', 'ratio', 'ventas totales']
        wants_kpi_chart = any(word in user_query for word in viz_keywords) and any(word in user_query for word in kpi_business_terms)
        
        # PRIORIDAD 4: Detectar queries SQL (solo si hay palabras específicas de datos)
        # Estas palabras SOLO deben activar SQL si están en contexto de datos
        sql_keywords = [
            # Palabras de cantidad
            'cuántas', 'cuantos', 'cuántos', 'cuanto', 'cuánto', 'cuantas', 'cuantas',
            # Palabras de datos/entidades
            'producto', 'productos', 'ventas', 'venta', 'ventas', 'correctivas', 'preventivas',
            'orden', 'ordenes', 'órdenes', 'compra', 'compras', 'cliente', 'clientes',
            # Palabras de acción
            'muestra', 'muéstrame', 'lista', 'listar', 'mostrar', 'dame', 'dame',
            # Palabras de existencia
            'hay', 'existen', 'existe', 'tiene', 'tienen',
            # Palabras de agregación
            'total', 'suma', 'promedio', 'máximo', 'mínimo', 'contar', 'count',
            # Palabras genéricas de datos
            'datos', 'registros', 'filas', 'tabla', 'tablas'
        ]
        
        # PRIORIDAD: Las intenciones específicas (viz, kpi, sql) tienen prioridad sobre general
        # Solo clasificar como "general" si NO hay indicadores de intenciones específicas
        
        logger.info(f"Analyzing query: '{user_query}'")
        
        # Detectar todas las keywords encontradas
        viz_matches = [word for word in viz_keywords if word in user_query]
        kpi_matches = [word for word in kpi_keywords if word in user_query]
        sql_matches = [word for word in sql_keywords if word in user_query]
        general_matches = [word for word in general_keywords if word in user_query]
        
        logger.info(f"Keyword matches - viz: {viz_matches}, kpi: {kpi_matches}, sql: {sql_matches}, general: {general_matches}")
        
        # 0. Verificar si se pide graficar KPIs (debe ser hybrid)
        # Si hay palabras de visualización Y palabras de KPI/negocio, es hybrid
        kpi_business_terms = ['revenue', 'ingresos', 'ticket', 'promedio', 'ratio', 'ventas totales', 'kpi', 'métrica']
        has_viz = bool(viz_matches)
        has_kpi_term = any(word in user_query for word in kpi_business_terms)
        
        if has_viz and has_kpi_term:
            intent = 'hybrid'
            logger.info(f"Hybrid detection (KPI chart): {intent} (viz: {viz_matches}, kpi terms: {has_kpi_term})")
        # 1. Verificar visualizaciones PRIMERO (tiene máxima prioridad para queries de datos)
        elif viz_matches:
            intent = 'viz'
            logger.info(f"Viz keyword detection: {intent} (matched: {viz_matches})")
        # 2. Verificar KPIs
        elif kpi_matches or has_kpi_term:
            intent = 'kpi'
            logger.info(f"KPI keyword detection: {intent} (matched: {kpi_matches or 'business terms'})")
        # 3. Verificar SQL (queries de datos) - PRIORIDAD sobre general
        elif sql_matches:
            # Si tiene palabras de datos, es SQL aunque tenga palabras generales
            intent = 'sql'
            logger.info(f"SQL keyword detection: {intent} (matched: {sql_matches})")
        # 4. Verificar si es un saludo o pregunta general (solo si no hay intenciones específicas)
        elif general_matches:
            intent = 'general'
            logger.info(f"General keyword detection: {intent} (matched: {general_matches})")
        else:
            # Usar LLM para clasificar (solo si no detectamos keywords)
            logger.info("No keywords detected, using LLM for classification")
            llm = get_llama_model()
            prompt = get_router_prompt()

            formatted_prompt = prompt.format(user_query=state['user_query'])

            llm_intent = invoke_llm_with_retry(
                llm,
                [{"role": "user", "content": formatted_prompt}]
            )

            llm_intent = llm_intent.strip().lower()
            logger.info(f"LLM classification: {llm_intent}")
            
            # Validar que el LLM no sobreescriba con "general" si hay indicadores de datos
            if llm_intent == 'general' and (sql_matches or kpi_matches or viz_matches):
                logger.warning(f"LLM classified as 'general' but found data keywords, overriding to 'sql'")
                intent = 'sql'
            else:
                intent = llm_intent

        # Validar intent
        valid_intents = ['sql', 'kpi', 'viz', 'general', 'hybrid']
        if intent not in valid_intents:
            logger.warning(f"Invalid intent '{intent}', defaulting to 'general'")
            intent = 'general'  # Default a general si hay duda (más seguro que SQL)

        logger.info(f"✓ Final intent: {intent}")

        # Actualizar estado
        state['intent'] = intent
        state['intermediate_steps'].append({
            'node': 'router',
            'intent': intent,
            'query': state['user_query']
        })

        return state

    except Exception as e:
        logger.error(f"Error in router_node: {e}")
        state['error'] = str(e)
        state['intent'] = 'sql'  # Fallback seguro
        return state


# ============ SQL Node ============

def sql_node(state: AgentState) -> AgentState:
    """
    Genera y ejecuta query SQL (versión con RAG).
    """
    logger.info("=== SQL Node (with RAG) ===")

    try:
        llm = get_llama_model()
        prompt = get_sql_prompt()

        # Obtener schema info
        schema_info = mysql_tool.get_schema_info()

        # **NUEVO: Usar RAG para obtener ejemplos relevantes (con timeout)**
        try:
            relevant_examples = vectorstore.get_relevant_examples(
                  state['user_query'],
                  top_k=3
            )
            logger.info("Retrieved relevant examples from RAG")
        except Exception as e:
            logger.warning(f"RAG search failed or timed out: {e}. Continuing without examples.")
            relevant_examples = "No hay ejemplos similares disponibles."

        # Formatear prompt con ejemplos relevantes
        formatted_prompt = prompt.format(
            schema_info=schema_info,
            examples=relevant_examples,  # Usar ejemplos del RAG
            user_query=state['user_query']
        )
        #####
        # Generar SQL
        logger.info("Generating SQL query...")
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

        logger.info(f"Generated SQL: {sql_query}")

        # Ejecutar query con manejo de errores
        logger.info("Executing SQL query...")
        try:
            results = mysql_tool._run(sql_query)
            logger.info(f"Query returned {len(results)} rows")
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            state['error'] = f"Error ejecutando query SQL: {str(e)}. Verifica la conexión a la base de datos."
            state['sql_results'] = None
            # Continuar sin resultados, el format_results manejará el error
            return state

        # Actualizar estado
        state['sql_query'] = sql_query
        state['sql_results'] = results
        state['intermediate_steps'].append({
            'node': 'sql',
            'query': sql_query,
            'num_results': len(results)
        })

        # Añadir mensaje
        state['messages'].append(
            AIMessage(content=f"Ejecuté la query: {sql_query}")
        )

        return state

    except Exception as e:
        logger.error(f"Error in sql_node: {e}")
        state['error'] = str(e)
        state['sql_query'] = None
        state['sql_results'] = []
        return state


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
    logger.info("=== Viz Node (Hybrid) ===")

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

        # USAR SISTEMA HÍBRIDO
        logger.info("Using HybridVizSystem for chart decision")
        chart_config = hybrid_viz.decide_chart(
            query=state['user_query'],
            sql_results=results
        )
        logger.info("Generating professional chart")
        chart_result = professional_viz_tool.create_chart(
            data=results,
            chart_type=chart_config.get('chart_type'),
            config=chart_config
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