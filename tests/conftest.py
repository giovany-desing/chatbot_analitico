"""
Configuración de pytest.
"""

import pytest
import sys
from pathlib import Path

# Añadir app al path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def test_query():
    """Query de prueba"""
    return "¿Cuántas ventas hay?"


@pytest.fixture
def expected_sql():
    """SQL esperado"""
    return "SELECT COUNT(*) FROM ventas"


"""
Tests para herramientas (SQL, Viz).
"""

import pytest
from app.tools.sql_tool import mysql_tool
from app.tools.viz_tool import viz_tool


class TestMySQLTool:

    def test_simple_query(self):
        """Test query simple"""
        result = mysql_tool._run("SELECT COUNT(*) as total FROM ventas")
        assert isinstance(result, list)
        assert len(result) > 0
        assert 'total' in result[0]

    def test_security_blocks_delete(self):
        """Test que bloquea DELETE"""
        with pytest.raises(ValueError, match="Solo se permiten queries SELECT"):
            mysql_tool._run("DELETE FROM ventas")

    def test_security_blocks_drop(self):
        """Test que bloquea DROP"""
        with pytest.raises(ValueError, match="Solo se permiten queries SELECT"):
            mysql_tool._run("DROP TABLE ventas")

    def test_adds_limit(self):
        """Test que añade LIMIT automáticamente"""
        query = "SELECT * FROM productos"
        limited = mysql_tool._ensure_limit(query)
        assert "LIMIT" in limited.upper()


class TestVizTool:

    def test_bar_chart(self):
        """Test gráfica de barras"""
        data = [
            {"producto": "A", "ventas": 100},
            {"producto": "B", "ventas": 200}
        ]

        result = viz_tool._run(
            data=data,
            chart_type="bar",
            x_column="producto",
            y_column="ventas"
        )

        assert result['chart_type'] == 'bar'
        assert result['data_points'] == 2
        assert 'config' in result

    def test_line_chart(self):
        """Test gráfica de línea"""
        data = [
            {"mes": "Ene", "revenue": 1000},
            {"mes": "Feb", "revenue": 1500}
        ]

        result = viz_tool._run(
            data=data,
            chart_type="line",
            x_column="mes",
            y_column="revenue"
        )

        assert result['chart_type'] == 'line'