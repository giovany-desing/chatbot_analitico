"""
Grafo LangGraph que orquesta el flujo del chatbot.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH si se ejecuta directamente
if __name__ == "__main__":
    # Obtener el directorio raíz del proyecto (2 niveles arriba de este archivo)
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from langgraph.graph import StateGraph, END
from typing import Literal
import logging

from app.agents.state import AgentState
from app.agents.nodes import (
    router_node,
    sql_node,
    kpi_node,
    viz_node,
    general_node,
    hybrid_node,
    format_results
)

logger = logging.getLogger(__name__)


def route_by_intent(state: AgentState) -> Literal["sql", "kpi", "viz", "general", "hybrid"]:
    """
    Función de routing condicional.
    Decide a qué nodo ir basado en el intent.
    
    Args:
        state: Estado actual
    
    Returns:
        Nombre del siguiente nodo
    """
    intent = state.get('intent', 'general')
    logger.info(f"Routing to: {intent}")
    return intent


def create_chatbot_graph():
    """
    Crea y compila el grafo del chatbot.
    
    Flujo:
    START → router → [sql|kpi|viz|general|hybrid] → format_results → END
    
    Returns:
        Grafo compilado listo para ejecutar
    """
    # Crear grafo
    workflow = StateGraph(AgentState)

    # Añadir nodos
    workflow.add_node("router", router_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("kpi", kpi_node)
    workflow.add_node("viz", viz_node)
    workflow.add_node("general", general_node)
    workflow.add_node("hybrid", hybrid_node)
    workflow.add_node("format_results", format_results)

    # Punto de entrada
    workflow.set_entry_point("router")

    # Conditional edges desde router
    workflow.add_conditional_edges(
        "router",
        route_by_intent,
        {
            "sql": "sql",
            "kpi": "kpi",
            "viz": "viz",
            "general": "general",
            "hybrid": "hybrid"
        }
    )

    # Todos los nodos van a format_results
    workflow.add_edge("sql", "format_results")
    workflow.add_edge("kpi", "format_results")
    workflow.add_edge("viz", "format_results")
    workflow.add_edge("general", "format_results")
    workflow.add_edge("hybrid", "format_results")

    # format_results va a END
    workflow.add_edge("format_results", END)

    # Compilar
    app = workflow.compile()

    logger.info("✅ Chatbot graph compiled")

    return app


# Instancia global del grafo
chatbot_graph = create_chatbot_graph()


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from app.agents.state import create_initial_state

    print("=== Testing Chatbot Graph ===\n")

    # Test 1: Query SQL
    print("1. Testing SQL flow:")
    state = create_initial_state("¿Cuántas ventas hay en total?")
    result = chatbot_graph.invoke(state)

    print(f"   Intent: {result['intent']}")
    print(f"   SQL: {result['sql_query']}")
    print(f"   Results: {len(result.get('sql_results', []))} rows")
    print(f"   Response: {result['messages'][-1].content[:200]}...\n")

    # Test 2: Query general
    print("2. Testing general flow:")
    state = create_initial_state("¿Qué puedes hacer?")
    result = chatbot_graph.invoke(state)

    print(f"   Intent: {result['intent']}")
    print(f"   Response: {result['messages'][-1].content[:200]}...\n")

    # Test 3: Query compleja
    print("3. Testing complex SQL:")
    state = create_initial_state("Muéstrame los 5 productos más vendidos")
    result = chatbot_graph.invoke(state)

    print(f"   Intent: {result['intent']}")
    print(f"   SQL: {result['sql_query']}")
    print(f"   Results: {result.get('sql_results', [])}")