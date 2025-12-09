"""
Módulo para validación y autocorrección de SQL.
Incluye funciones para ejecutar SQL de forma segura y obtener información del schema.
"""

import logging
from typing import Tuple, List, Dict, Any, Optional
from app.tools.sql_tool import mysql_tool
from app.db.connections import get_mysql

logger = logging.getLogger(__name__)


def validate_and_execute_sql(sql_query: str) -> Tuple[bool, Any]:
    """
    Valida y ejecuta una query SQL.
    
    Args:
        sql_query: Query SQL a ejecutar
    
    Returns:
        Tupla (success: bool, data/error_message)
        - Si success=True: retorna (True, lista de resultados)
        - Si success=False: retorna (False, mensaje de error)
    """
    try:
        # Validar sintaxis básica
        if not mysql_tool._is_valid_sql(sql_query):
            error_msg = "Query SQL inválida: sintaxis incorrecta"
            logger.warning(f"SQL validation failed: {error_msg}")
            return False, error_msg
        
        # Validar que sea segura (solo SELECT)
        if not mysql_tool._is_safe_query(sql_query):
            error_msg = "Query SQL no segura: solo se permiten queries SELECT"
            logger.warning(f"SQL validation failed: {error_msg}")
            return False, error_msg
        
        # Añadir LIMIT si no existe
        sql_query = mysql_tool._ensure_limit(sql_query)
        
        # Intentar ejecutar
        logger.info(f"Executing SQL: {sql_query}")
        results = mysql_tool._run(sql_query)
        
        logger.info(f"✅ SQL executed successfully: {len(results)} rows returned")
        return True, results
        
    except ValueError as e:
        # Error de validación (sintaxis, seguridad)
        error_msg = str(e)
        logger.warning(f"❌ SQL validation error: {error_msg}")
        return False, error_msg
        
    except Exception as e:
        # Error de ejecución (MySQL)
        error_msg = str(e)
        logger.warning(f"❌ SQL execution error: {error_msg}")
        
        # Extraer mensaje de error más específico si es posible
        error_str = str(e)
        if "Unknown column" in error_str:
            # Extraer nombre de columna no encontrada
            parts = error_str.split("'")
            if len(parts) >= 2:
                column = parts[1]
                error_msg = f"Columna desconocida: '{column}'. Verifica el nombre de la columna en el schema."
        elif "Table" in error_str and "doesn't exist" in error_str:
            # Extraer nombre de tabla no encontrada
            parts = error_str.split("'")
            if len(parts) >= 2:
                table = parts[1]
                error_msg = f"Tabla desconocida: '{table}'. Las tablas disponibles son: ventas_preventivas, ventas_correctivas."
        elif "syntax error" in error_str.lower() or "SQL syntax" in error_str:
            error_msg = f"Error de sintaxis SQL: {error_str}"
        else:
            error_msg = f"Error MySQL: {error_str}"
        
        return False, error_msg


def get_table_schema() -> str:
    """
    Obtiene la estructura detallada de las tablas principales.
    Incluye tipos de datos, claves primarias y foreign keys.
    
    Returns:
        String formateado con información del schema para usar en prompts
    """
    try:
        mysql = get_mysql()
        schema_dict = mysql.get_schema_info()
        
        schema_text = []
        
        for table_name, columns in schema_dict.items():
            # Filtrar solo las tablas de ventas
            if table_name not in ['ventas_preventivas', 'ventas_correctivas']:
                continue
            
            table_info = [f"**{table_name}:**"]
            
            for col in columns:
                col_info = f"  - {col['column']}: {col['type']}"
                
                # Agregar información adicional
                if col['key'] == 'PRI':
                    col_info += " (PRIMARY KEY)"
                elif col['key'] == 'MUL':
                    col_info += " (INDEXED)"
                
                if not col['nullable']:
                    col_info += " (NOT NULL)"
                
                table_info.append(col_info)
            
            schema_text.append("\n".join(table_info))
        
        return "\n\n".join(schema_text)
        
    except Exception as e:
        logger.error(f"Error getting table schema: {e}")
        # Fallback a schema básico
        return """**ventas_preventivas:**
  - id: int (PRIMARY KEY, NOT NULL)
  - orden_compra: varchar (NOT NULL)
  - producto: varchar (NOT NULL)
  - fecha_creacion: datetime (NOT NULL)
  - cantidad: decimal (NOT NULL)
  - total: decimal (NOT NULL)

**ventas_correctivas:**
  - id: int (PRIMARY KEY, NOT NULL)
  - orden_compra: varchar (NOT NULL)
  - producto: varchar (NOT NULL)
  - fecha_creacion: datetime (NOT NULL)
  - cantidad: decimal (NOT NULL)
  - total: decimal (NOT NULL)"""


def get_detailed_schema_info() -> Dict[str, List[Dict[str, Any]]]:
    """
    Obtiene información detallada del schema en formato dict.
    Útil para análisis programático.
    
    Returns:
        Dict con estructura {table_name: [column_info, ...]}
    """
    try:
        mysql = get_mysql()
        schema_dict = mysql.get_schema_info()
        
        # Filtrar solo tablas de ventas
        filtered_schema = {
            table: columns
            for table, columns in schema_dict.items()
            if table in ['ventas_preventivas', 'ventas_correctivas']
        }
        
        return filtered_schema
        
    except Exception as e:
        logger.error(f"Error getting detailed schema info: {e}")
        return {}

