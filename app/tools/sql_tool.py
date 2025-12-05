"""
Herramienta para ejecutar queries SQL de forma segura.
Solo permite SELECT queries.
"""

import sys
import os
from pathlib import Path
from app.services.cache_service import cache_service

# Agregar el directorio raíz del proyecto al PYTHONPATH si se ejecuta directamente
if __name__ == "__main__":
    # Obtener el directorio raíz del proyecto (2 niveles arriba de este archivo)
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from langchain_core.tools import BaseTool
from typing import List, Dict, Any, Optional
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DML
import logging

from app.db.connections import get_mysql
from app.config import settings

logger = logging.getLogger(__name__)


class MySQLTool(BaseTool):
    """
    Herramienta de LangChain para ejecutar queries SQL.
    
    Características:
    - Solo permite SELECT queries
    - Valida sintaxis SQL
    - Limita número de resultados
    - Logging de todas las queries
    """

    name: str = "mysql_query"
    description: str = """
    Ejecuta una query SQL SELECT en la base de datos textil.
    
    Input: Query SQL válida (solo SELECT)
    Output: Lista de resultados en formato JSON
    
    Tablas disponibles:
    - ventas_preventivas: id, orden_compra, producto, fecha_creacion, cantidad, total
    - ventas_correctivas: id, orden_compra, producto, fecha_creacion, cantidad, total
    
    Contexto del negocio:
    - Base de datos de industria textil
    - Ventas preventivas: órdenes planificadas con anticipación
    - Ventas correctivas: órdenes de corrección o ajuste
    """

    def __init__(self):
        super().__init__()
        # Usar object.__setattr__ para asignar atributos en clases Pydantic
        object.__setattr__(self, 'mysql', get_mysql())

    def _run(self, query: str) -> List[Dict[str, Any]]:
        """
        Ejecuta la query SQL (versión con cache).
        """
        try:
            # 1. Intentar obtener del cache
            cached_results = cache_service.get(query)
            if cached_results is not None:
                logger.info("Returning cached results")
                return cached_results

            # 2. Validar sintaxis
            if not self._is_valid_sql(query):
                raise ValueError("Query SQL inválida")

            # 3. Validar que sea solo SELECT
            if not self._is_safe_query(query):
                raise ValueError("Solo se permiten queries SELECT")

            # 4. Añadir LIMIT si no existe
            query = self._ensure_limit(query)

            # 5. Ejecutar query
            logger.info(f"Executing SQL: {query}")
            results = self.mysql.execute_query(query)

            logger.info(f"Query returned {len(results)} rows")

            # 6. Guardar en cache
            cache_service.set(query, results)

            return results

        except Exception as e:
            logger.error(f"Error executing SQL: {e}")
            raise

    def _is_valid_sql(self, query: str) -> bool:
        """
        Valida que la query tenga sintaxis SQL correcta.
        
        Args:
            query: Query SQL
        
        Returns:
            True si es válida
        """
        try:
            parsed = sqlparse.parse(query)
            return len(parsed) > 0 and parsed[0].is_group
        except:
            return False

    def _is_safe_query(self, query: str) -> bool:
        """
        Verifica que la query sea segura (solo SELECT).
        
        Args:
            query: Query SQL
        
        Returns:
            True si es segura
        """
        # Normalizar query
        query_upper = query.strip().upper()

        # Blacklist de comandos peligrosos
        dangerous_keywords = [
            'DELETE', 'DROP', 'UPDATE', 'INSERT',
            'ALTER', 'CREATE', 'TRUNCATE', 'REPLACE',
            'EXEC', 'EXECUTE', 'CALL'
        ]

        for keyword in dangerous_keywords:
            if keyword in query_upper:
                logger.warning(f"Blocked dangerous keyword: {keyword}")
                return False

        # Debe empezar con SELECT
        if not query_upper.startswith('SELECT'):
            logger.warning("Query must start with SELECT")
            return False

        return True

    def _ensure_limit(self, query: str, default_limit: int = 100) -> str:
        """
        Asegura que la query tenga un LIMIT.
        Si no tiene, añade uno por defecto.
        
        Args:
            query: Query SQL
            default_limit: Límite por defecto
        
        Returns:
            Query con LIMIT
        """
        query_upper = query.upper()

        # Si ya tiene LIMIT, retornar sin cambios
        if 'LIMIT' in query_upper:
            return query

        # Añadir LIMIT
        query = query.rstrip(';').strip()
        return f"{query} LIMIT {default_limit}"

    def get_schema_info(self) -> str:
        """
        Obtiene información del schema en formato texto.
        Útil para incluir en prompts del LLM.
        
        Returns:
            String con información de tablas y columnas
        """
        schema = self.mysql.get_schema_info()

        schema_text = []
        for table, columns in schema.items():
            cols_text = []
            for col in columns:
                col_str = f"{col['column']} ({col['type']})"
                if col['key'] == 'PRI':
                    col_str += " PRIMARY KEY"
                elif col['key'] == 'MUL':
                    col_str += " FOREIGN KEY"
                cols_text.append(col_str)

            schema_text.append(f"{table}:\n  " + "\n  ".join(cols_text))

        return "\n\n".join(schema_text)

    def get_sample_queries(self) -> List[str]:
        """
        Retorna queries de ejemplo para few-shot learning.
        Adaptado al schema de la BD textil.
        
        Returns:
            Lista de queries SQL de ejemplo
        """
        return [
            # Queries básicas de conteo
            "SELECT COUNT(*) as total_preventivas FROM ventas_preventivas",
            "SELECT COUNT(*) as total_correctivas FROM ventas_correctivas",

            # Análisis por producto
            "SELECT producto, SUM(cantidad) as unidades, SUM(total) as revenue FROM ventas_preventivas GROUP BY producto ORDER BY revenue DESC LIMIT 10",

            # Análisis temporal
            "SELECT DATE_FORMAT(fecha_creacion, '%Y-%m') as mes, SUM(total) as revenue FROM ventas_preventivas GROUP BY mes ORDER BY mes",

            # Comparación preventivas vs correctivas
            """
            SELECT 
                'Preventivas' as tipo,
                COUNT(*) as num_ordenes,
                SUM(cantidad) as unidades,
                SUM(total) as revenue
            FROM ventas_preventivas
            UNION ALL
            SELECT 
                'Correctivas' as tipo,
                COUNT(*) as num_ordenes,
                SUM(cantidad) as unidades,
                SUM(total) as revenue
            FROM ventas_correctivas
            """,

            # Top productos en ambas tablas
            """
            SELECT producto, SUM(cantidad) as total_unidades, SUM(total) as total_revenue
            FROM (
                SELECT producto, cantidad, total FROM ventas_preventivas
                UNION ALL
                SELECT producto, cantidad, total FROM ventas_correctivas
            ) as todas_ventas
            GROUP BY producto
            ORDER BY total_revenue DESC
            LIMIT 10
            """,

            # Análisis por orden de compra
            "SELECT orden_compra, COUNT(*) as items, SUM(total) as total_orden FROM ventas_preventivas GROUP BY orden_compra ORDER BY total_orden DESC LIMIT 10"
        ]


# Instancia global de la herramienta
mysql_tool = MySQLTool()


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testing MySQLTool ===\n")

    tool = MySQLTool()

    # Test 1: Query simple - Contar ventas preventivas
    print("1. Count ventas preventivas:")
    try:
        result = tool._run("SELECT COUNT(*) as total FROM ventas_preventivas")
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    # Test 2: Query simple - Contar ventas correctivas
    print("2. Count ventas correctivas:")
    try:
        result = tool._run("SELECT COUNT(*) as total FROM ventas_correctivas")
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    # Test 3: Top productos en ventas preventivas
    print("3. Top 5 productos (ventas preventivas):")
    try:
        result = tool._run("""
            SELECT producto, SUM(cantidad) as unidades, SUM(total) as revenue
            FROM ventas_preventivas
            GROUP BY producto
            ORDER BY revenue DESC
            LIMIT 5
        """)
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    # Test 4: Comparación preventivas vs correctivas
    print("4. Comparison preventivas vs correctivas:")
    try:
        result = tool._run("""
            SELECT 
                'Preventivas' as tipo,
                COUNT(*) as num_ordenes,
                SUM(total) as revenue
            FROM ventas_preventivas
            UNION ALL
            SELECT 
                'Correctivas' as tipo,
                COUNT(*) as num_ordenes,
                SUM(total) as revenue
            FROM ventas_correctivas
        """)
        print(f"   Result: {result}\n")
    except Exception as e:
        print(f"   Error: {e}\n")

    # Test 5: Validación de seguridad - Debe bloquear DELETE
    print("5. Testing security (should block DELETE):")
    try:
        tool._run("DELETE FROM ventas_preventivas")
        print("   ✗ Security FAILED (allowed DELETE)")
    except ValueError as e:
        print(f"   ✓ Security passed: {e}\n")

    # Test 6: Schema info
    print("6. Schema info:")
    schema = tool.get_schema_info()
    print(f"{schema}\n")

    # Test 7: Sample queries
    print("7. Sample queries:")
    for i, query in enumerate(tool.get_sample_queries(), 1):
        # Mostrar solo las primeras 80 caracteres de cada query
        query_preview = query.replace('\n', ' ').strip()
        if len(query_preview) > 80:
            query_preview = query_preview[:77] + "..."
        print(f"   {i}. {query_preview}")
