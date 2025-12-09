"""
Módulo de validación de datos para el chatbot analítico.
Incluye validación de queries, resultados SQL, datos para gráficas, y más.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import date, datetime
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de una validación"""
    is_valid: bool
    error_msg: str = ""
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# ============ SQL Injection Patterns ============

SQL_INJECTION_PATTERNS = [
    r"(\bOR\b|\bAND\b)\s*\d+\s*=\s*\d+",  # OR 1=1, AND 1=1
    r"(\bUNION\b|\bSELECT\b).*(\bFROM\b|\bWHERE\b)",  # UNION SELECT
    r"(\bDROP\b|\bDELETE\b|\bINSERT\b|\bUPDATE\b|\bALTER\b|\bCREATE\b)",  # Comandos peligrosos
    r"(--|#|/\*|\*/)",  # Comentarios SQL
    r"(\bEXEC\b|\bEXECUTE\b|\bCALL\b)",  # Ejecución de comandos
    r"(\bSCRIPT\b|\bJAVASCRIPT\b|\bVBSCRIPT\b)",  # Scripts
    r"(\bXP_\w+|sp_\w+)",  # Stored procedures peligrosas
    r"(\bLOAD_FILE\b|\bINTO\s+OUTFILE\b)",  # Lectura/escritura de archivos
]


# ============ Validación de User Query ============

def validate_user_query(query: str) -> bool:
    """
    Valida una query del usuario.
    
    Args:
        query: Query del usuario
    
    Returns:
        True si es válida, False si no
    
    Raises:
        ValueError: Si la query es inválida (con mensaje descriptivo)
    """
    if not query:
        raise ValueError("La query no puede estar vacía")
    
    # Normalizar
    query = query.strip()
    
    if not query:
        raise ValueError("La query no puede contener solo espacios")
    
    # Validar longitud
    if len(query) < 3:
        raise ValueError(f"La query es muy corta (mínimo 3 caracteres, tienes {len(query)})")
    
    if len(query) > 500:
        raise ValueError(f"La query es muy larga (máximo 500 caracteres, tienes {len(query)})")
    
    # Detectar SQL injection básico
    query_upper = query.upper()
    
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, query_upper, re.IGNORECASE):
            logger.warning(f"Posible SQL injection detectado en query: {query[:50]}...")
            raise ValueError("La query contiene patrones sospechosos que no están permitidos")
    
    # Verificar caracteres peligrosos excesivos
    dangerous_chars = [';', '--', '/*', '*/', 'xp_', 'sp_']
    char_count = sum(query_upper.count(char) for char in dangerous_chars)
    
    if char_count > 3:
        logger.warning(f"Query con muchos caracteres sospechosos: {query[:50]}...")
        raise ValueError("La query contiene demasiados caracteres sospechosos")
    
    return True


# ============ Validación de Resultados SQL ============

def validate_sql_results(results: List[Dict]) -> ValidationResult:
    """
    Valida resultados de una query SQL.
    
    Args:
        results: Lista de diccionarios con los resultados
    
    Returns:
        ValidationResult con is_valid, error_msg y warnings
    """
    warnings = []
    
    # Verificar que no esté vacío
    if not results:
        return ValidationResult(
            is_valid=False,
            error_msg="Los resultados están vacíos",
            warnings=warnings
        )
    
    if len(results) == 0:
        return ValidationResult(
            is_valid=False,
            error_msg="No se encontraron resultados",
            warnings=warnings
        )
    
    # Validar que todas las filas tengan las mismas columnas
    if not results:
        return ValidationResult(
            is_valid=False,
            error_msg="Lista de resultados vacía",
            warnings=warnings
        )
    
    # Obtener columnas de la primera fila
    first_row_keys = set(results[0].keys())
    
    if not first_row_keys:
        return ValidationResult(
            is_valid=False,
            error_msg="La primera fila no tiene columnas",
            warnings=warnings
        )
    
    # Verificar consistencia de columnas
    inconsistent_rows = []
    for i, row in enumerate(results[1:], start=1):
        row_keys = set(row.keys())
        if row_keys != first_row_keys:
            inconsistent_rows.append(i)
            if len(inconsistent_rows) > 5:  # Limitar reporte
                break
    
    if inconsistent_rows:
        return ValidationResult(
            is_valid=False,
            error_msg=f"Filas con columnas inconsistentes: {inconsistent_rows[:5]}",
            warnings=warnings
        )
    
    # Detectar valores None excesivos
    total_cells = len(results) * len(first_row_keys)
    none_count = sum(
        1 for row in results
        for value in row.values()
        if value is None
    )
    
    none_percentage = (none_count / total_cells) * 100 if total_cells > 0 else 0
    
    if none_percentage > 80:
        return ValidationResult(
            is_valid=False,
            error_msg=f"Demasiados valores None ({none_percentage:.1f}% > 80%)",
            warnings=warnings
        )
    elif none_percentage > 50:
        warnings.append(f"Alto porcentaje de valores None ({none_percentage:.1f}%)")
    
    # Advertencias adicionales
    if len(results) > 1000:
        warnings.append(f"Gran cantidad de resultados ({len(results)}), considerar agregar LIMIT")
    
    if len(first_row_keys) > 20:
        warnings.append(f"Muchas columnas ({len(first_row_keys)}), puede ser difícil de visualizar")
    
    return ValidationResult(
        is_valid=True,
        error_msg="",
        warnings=warnings
    )


# ============ Validación de Datos para Gráficas ============

def validate_data_for_chart(data: List[Dict], chart_type: str) -> ValidationResult:
    """
    Valida que los datos sean apropiados para el tipo de gráfica.
    
    Args:
        data: Lista de diccionarios con los datos
        chart_type: Tipo de gráfica (bar, line, pie, scatter, etc.)
    
    Returns:
        ValidationResult con validación y sugerencias
    """
    warnings = []
    
    if not data:
        return ValidationResult(
            is_valid=False,
            error_msg="No hay datos para graficar",
            warnings=warnings
        )
    
    # Convertir a DataFrame para análisis
    try:
        df = pd.DataFrame(data)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_msg=f"Error convirtiendo datos a DataFrame: {e}",
            warnings=warnings
        )
    
    if df.empty:
        return ValidationResult(
            is_valid=False,
            error_msg="DataFrame vacío",
            warnings=warnings
        )
    
    # Identificar tipos de columnas
    numeric_cols = list(df.select_dtypes(include=['number']).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    datetime_cols = list(df.select_dtypes(include=['datetime64']).columns)
    
    chart_type_lower = chart_type.lower()
    
    # Validación específica por tipo de gráfica
    if chart_type_lower in ['bar', 'column']:
        if len(categorical_cols) == 0:
            return ValidationResult(
                is_valid=False,
                error_msg="Gráfica de barras requiere al menos 1 columna categórica",
                warnings=warnings
            )
        
        if len(numeric_cols) == 0:
            return ValidationResult(
                is_valid=False,
                error_msg="Gráfica de barras requiere al menos 1 columna numérica",
                warnings=warnings
            )
        
        if len(categorical_cols) > 1:
            warnings.append(f"Múltiples columnas categóricas ({len(categorical_cols)}), considerar agrupar")
        
        if len(numeric_cols) > 3:
            warnings.append(f"Múltiples columnas numéricas ({len(numeric_cols)}), considerar gráfica de barras agrupadas")
        
        # Verificar valores negativos
        for col in numeric_cols:
            if (df[col] < 0).any():
                warnings.append(f"Columna '{col}' contiene valores negativos, verificar si es correcto para barras")
    
    elif chart_type_lower in ['line', 'area']:
        # Para líneas, necesitamos una columna temporal/secuencial
        has_temporal = len(datetime_cols) > 0
        
        # Verificar si hay columna que parezca temporal (fechas, números secuenciales)
        potential_temporal = []
        for col in df.columns:
            if col.lower() in ['fecha', 'date', 'time', 'mes', 'año', 'year', 'month', 'dia', 'day']:
                potential_temporal.append(col)
            elif col in numeric_cols and df[col].is_monotonic_increasing:
                potential_temporal.append(col)
        
        if not has_temporal and not potential_temporal:
            return ValidationResult(
                is_valid=False,
                error_msg="Gráfica de línea requiere una columna temporal o secuencial",
                warnings=warnings
            )
        
        if len(numeric_cols) == 0:
            return ValidationResult(
                is_valid=False,
                error_msg="Gráfica de línea requiere al menos 1 columna numérica",
                warnings=warnings
            )
        
        if len(numeric_cols) > 5:
            warnings.append(f"Múltiples series ({len(numeric_cols)}), puede ser difícil de leer")
    
    elif chart_type_lower in ['pie', 'donut']:
        if len(categorical_cols) == 0:
            return ValidationResult(
                is_valid=False,
                error_msg="Gráfica de pastel requiere al menos 1 columna categórica",
                warnings=warnings
            )
        
        if len(numeric_cols) == 0:
            return ValidationResult(
                is_valid=False,
                error_msg="Gráfica de pastel requiere al menos 1 columna numérica",
                warnings=warnings
            )
        
        # Verificar valores positivos
        for col in numeric_cols:
            if (df[col] < 0).any():
                return ValidationResult(
                    is_valid=False,
                    error_msg=f"Gráfica de pastel no puede tener valores negativos en '{col}'",
                    warnings=warnings
                )
            
            # Verificar que la suma no sea cero
            total = df[col].sum()
            if total == 0:
                return ValidationResult(
                    is_valid=False,
                    error_msg=f"La suma de valores en '{col}' es cero",
                    warnings=warnings
                )
        
        # Advertencia si hay muchas categorías
        if len(df) > 10:
            warnings.append(f"Muchas categorías ({len(df)}), considerar agrupar las menores o usar otro tipo de gráfica")
        
        # Advertencia si hay valores muy pequeños
        for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if min_val > 0 and (min_val / max_val) < 0.01:
                warnings.append(f"Hay valores muy pequeños en '{col}', pueden ser difíciles de ver en el pastel")
    
    elif chart_type_lower in ['scatter', 'bubble']:
        if len(numeric_cols) < 2:
            return ValidationResult(
                is_valid=False,
                error_msg="Gráfica de dispersión requiere al menos 2 columnas numéricas",
                warnings=warnings
            )
        
        if len(numeric_cols) > 3:
            warnings.append(f"Múltiples columnas numéricas ({len(numeric_cols)}), considerar usar solo 2-3 para scatter")
        
        # Verificar que haya suficientes puntos
        if len(df) < 3:
            warnings.append(f"Pocos puntos de datos ({len(df)}), scatter puede no ser informativo")
    
    elif chart_type_lower in ['histogram', 'hist']:
        if len(numeric_cols) == 0:
            return ValidationResult(
                is_valid=False,
                error_msg="Histograma requiere al menos 1 columna numérica",
                warnings=warnings
            )
        
        if len(df) < 10:
            warnings.append(f"Pocos datos ({len(df)}), histograma puede no ser representativo")
    
    else:
        # Tipo de gráfica desconocido o genérico
        warnings.append(f"Tipo de gráfica '{chart_type}' no tiene validación específica")
    
    # Validaciones generales
    if len(df) > 10000:
        warnings.append(f"Muchos datos ({len(df)}), considerar agregar o filtrar para mejor rendimiento")
    
    if len(df.columns) > 15:
        warnings.append(f"Muchas columnas ({len(df.columns)}), puede ser difícil de visualizar")
    
    return ValidationResult(
        is_valid=True,
        error_msg="",
        warnings=warnings
    )


# ============ Sanitización de Rango de Fechas ============

def sanitize_date_range(
    start_date: Optional[Any],
    end_date: Optional[Any],
    min_year: int = 1900,
    max_year: int = 2030,
    max_range_years: int = 10
) -> Tuple[date, date]:
    """
    Sanitiza y valida un rango de fechas.
    
    Args:
        start_date: Fecha de inicio (date, datetime, str, o None)
        end_date: Fecha de fin (date, datetime, str, o None)
        min_year: Año mínimo permitido
        max_year: Año máximo permitido
        max_range_years: Rango máximo en años
    
    Returns:
        Tupla (start_date, end_date) como objetos date
    
    Raises:
        ValueError: Si las fechas son inválidas
    """
    def parse_date(d) -> date:
        """Convierte diferentes formatos a date"""
        if d is None:
            raise ValueError("Fecha no puede ser None")
        
        if isinstance(d, date):
            return d
        
        if isinstance(d, datetime):
            return d.date()
        
        if isinstance(d, str):
            # Intentar parsear diferentes formatos
            formats = ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']
            for fmt in formats:
                try:
                    dt = datetime.strptime(d, fmt)
                    return dt.date()
                except ValueError:
                    continue
            raise ValueError(f"No se pudo parsear la fecha: {d}")
        
        raise ValueError(f"Tipo de fecha no soportado: {type(d)}")
    
    # Parsear fechas
    try:
        start = parse_date(start_date)
        end = parse_date(end_date)
    except ValueError as e:
        raise ValueError(f"Error parseando fechas: {e}")
    
    # Validar años
    if start.year < min_year or start.year > max_year:
        raise ValueError(f"Año de inicio ({start.year}) fuera del rango permitido ({min_year}-{max_year})")
    
    if end.year < min_year or end.year > max_year:
        raise ValueError(f"Año de fin ({end.year}) fuera del rango permitido ({min_year}-{max_year})")
    
    # Corregir orden si end < start
    if end < start:
        logger.warning(f"Fechas invertidas: end ({end}) < start ({start}), invirtiendo orden")
        start, end = end, start
    
    # Validar rango máximo
    from datetime import timedelta
    range_days = (end - start).days
    range_years = range_days / 365.25
    
    if range_years > max_range_years:
        logger.warning(f"Rango de fechas muy grande ({range_years:.1f} años > {max_range_years}), limitando")
        max_days = int(max_range_years * 365.25)
        end = start + timedelta(days=max_days)
    
    return start, end


# ============ Validación de Contexto RAG ============

def validate_rag_context(rag_examples: List) -> bool:
    """
    Valida que el contexto RAG sea útil.
    
    Args:
        rag_examples: Lista de ejemplos retornados por RAG
    
    Returns:
        True si el contexto es válido, False si no
    
    Raises:
        ValueError: Si el contexto es inválido (con mensaje descriptivo)
    """
    if not rag_examples:
        raise ValueError("RAG no retornó ejemplos")
    
    if len(rag_examples) == 0:
        raise ValueError("Lista de ejemplos RAG vacía")
    
    # Verificar estructura de ejemplos
    valid_examples = 0
    low_similarity_count = 0
    
    for example in rag_examples:
        if not example:
            continue
        
        valid_examples += 1
        
        # Verificar similarity score si está disponible
        if isinstance(example, dict):
            similarity = example.get('similarity', example.get('score', 1.0))
            
            if similarity < 0.5:
                low_similarity_count += 1
                logger.warning(f"Ejemplo RAG con baja similitud: {similarity:.3f}")
    
    if valid_examples == 0:
        raise ValueError("No se encontraron ejemplos RAG válidos")
    
    # Alerta si muchos ejemplos tienen baja similitud
    if low_similarity_count > len(rag_examples) / 2:
        logger.warning(f"Muchos ejemplos RAG con baja similitud ({low_similarity_count}/{len(rag_examples)})")
        # No fallar, solo advertir
    
    return True


# ============ Funciones Auxiliares ============

def get_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Obtiene los tipos de columnas de un DataFrame.
    
    Args:
        df: DataFrame
    
    Returns:
        Dict con {columna: tipo}
    """
    types = {}
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        
        if 'int' in dtype or 'float' in dtype:
            types[col] = 'numeric'
        elif 'datetime' in dtype:
            types[col] = 'datetime'
        elif 'bool' in dtype:
            types[col] = 'boolean'
        else:
            types[col] = 'categorical'
    
    return types


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing Data Validator ===\n")
    
    # Test 1: validate_user_query
    print("1. Testing validate_user_query:")
    try:
        validate_user_query("¿Cuántas ventas hay?")
        print("   ✅ Query válida")
    except ValueError as e:
        print(f"   ❌ Error: {e}")
    
    try:
        validate_user_query("SELECT * FROM ventas; DROP TABLE ventas;")
        print("   ❌ Debería haber fallado")
    except ValueError as e:
        print(f"   ✅ SQL injection detectado: {e}")
    
    # Test 2: validate_sql_results
    print("\n2. Testing validate_sql_results:")
    results = [
        {"producto": "A", "cantidad": 10},
        {"producto": "B", "cantidad": 20}
    ]
    result = validate_sql_results(results)
    print(f"   Valid: {result.is_valid}, Warnings: {result.warnings}")
    
    # Test 3: validate_data_for_chart
    print("\n3. Testing validate_data_for_chart:")
    data = [
        {"producto": "A", "ventas": 100},
        {"producto": "B", "ventas": 200}
    ]
    result = validate_data_for_chart(data, "bar")
    print(f"   Valid: {result.is_valid}, Warnings: {result.warnings}")
    
    print("\n✅ Tests completados")

