"""
Tests para nodos del grafo.
"""

import pytest
from app.agents.state import create_initial_state
from app.agents.nodes import router_node, sql_node


class TestRouterNode:

    def test_classifies_sql(self):
        """Test clasificación de SQL"""
        state = create_initial_state("¿Cuántas ventas hay?")
        state = router_node(state)
        assert state['intent'] == 'sql'

    def test_classifies_kpi(self):
        """Test clasificación de KPI"""
        state = create_initial_state("Calcula el revenue total")
        state = router_node(state)
        assert state['intent'] == 'kpi'

    def test_classifies_viz(self):
        """Test clasificación de visualización"""
        state = create_initial_state("Muéstrame una gráfica")
        state = router_node(state)
        assert state['intent'] == 'viz'

    def test_classifies_general(self):
        """Test clasificación general"""
        state = create_initial_state("Hola")
        state = router_node(state)
        assert state['intent'] == 'general'


class TestSQLNode:

    def test_generates_and_executes_sql(self):
        """Test que genera y ejecuta SQL"""
        state = create_initial_state("¿Cuántas ventas hay?")
        state['intent'] = 'sql'
        state = sql_node(state)

        assert state['sql_query'] is not None
        assert state['sql_results'] is not None
        assert len(state['sql_results']) > 0

    def test_handles_complex_query(self):
        """Test query compleja"""
        state = create_initial_state("Muéstrame los productos más vendidos")
        state['intent'] = 'sql'
        state = sql_node(state)

        assert 'SELECT' in state['sql_query'].upper()
        assert state['sql_results'] is not None