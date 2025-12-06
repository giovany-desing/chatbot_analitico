"""
Definición del estado compartido del grafo.
Esta estructura se pasa entre todos los nodos.
"""

from typing import TypedDict, List, Dict, Optional, Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


def keep_first(left: str, right: str) -> str:
    """Reducer para user_query: mantiene el primer valor no vacío"""
    # Si left está vacío pero right tiene valor, usar right
    if not left or left.strip() == '':
        return right if right else left
    # Si left tiene valor, mantenerlo (prioridad)
    return left


class AgentState(TypedDict):
    """
    Estado del agente que fluye a través del grafo.
    
    Cada nodo puede leer y modificar este estado.
    LangGraph gestiona automáticamente las actualizaciones.
    
    Attributes:
        messages: Historial de conversación (gestionado por add_messages)
        user_query: Pregunta original del usuario
        intent: Intención clasificada (sql|kpi|viz|general|hybrid)
        sql_query: Query SQL generada
        sql_results: Resultados de la query (lista de dicts)
        kpis: KPIs calculados {nombre: valor}
        chart_config: Configuración de la gráfica (JSON de Plotly)
        context: Metadata adicional
        error: Mensaje de error si algo falla
        intermediate_steps: Pasos intermedios para debugging
    """

    # Historial de mensajes (LangGraph gestiona automáticamente)
    messages: Annotated[List[BaseMessage], add_messages]

    # Query del usuario (no debería cambiar, pero usamos reducer para evitar conflictos)
    user_query: Annotated[str, keep_first]

    # Intención clasificada por router_node
    intent: Optional[str]

    # ===== Para flujo SQL =====
    sql_query: Optional[str]
    sql_results: Optional[List[Dict[str, Any]]]

    # ===== Para flujo KPI =====
    kpis: Optional[Dict[str, Any]]

    # ===== Para flujo Visualización =====
    chart_config: Optional[Dict[str, Any]]

    # ===== Contexto y debugging =====
    context: Dict[str, Any]
    error: Optional[str]
    intermediate_steps: List[Dict[str, Any]]
    
    # ===== Tracking =====
    session_id: Optional[str]
    start_time: Optional[float]
    feedback_id: Optional[int]


# Función helper para crear estado inicial
def create_initial_state(user_query: str) -> AgentState:
    """
    Crea un estado inicial a partir de una query del usuario.
    
    Args:
        user_query: Pregunta del usuario
    
    Returns:
        AgentState con valores iniciales
    """
    import time
    import uuid
    
    # Asegurar que user_query no esté vacío
    if not user_query or not str(user_query).strip():
        raise ValueError(f"user_query cannot be empty. Received: '{user_query}'")
    
    return AgentState(
        messages=[],
        user_query=str(user_query).strip(),  # Asegurar que sea string y no vacío
        intent=None,
        sql_query=None,
        sql_results=None,
        kpis=None,
        chart_config=None,
        context={},
        error=None,
        intermediate_steps=[],
        session_id=str(uuid.uuid4()),
        start_time=time.time(),
        feedback_id=None
    )


# Para testing
if __name__ == "__main__":
    print("=== Testing AgentState ===\n")

    # Test 1: Crear estado inicial
    state = create_initial_state("¿Cuántas ventas hay?")
    print("1. Estado inicial:")
    print(f"   user_query: {state['user_query']}")
    print(f"   intent: {state['intent']}")
    print(f"   messages: {state['messages']}\n")

    # Test 2: Simular actualización de nodo
    state['intent'] = 'sql'
    state['sql_query'] = 'SELECT COUNT(*) FROM ventas'
    print("2. Después de router_node:")
    print(f"   intent: {state['intent']}")
    print(f"   sql_query: {state['sql_query']}\n")

    # Test 3: Simular resultados
    state['sql_results'] = [{'count': 42}]
    print("3. Después de sql_node:")
    print(f"   sql_results: {state['sql_results']}\n")
