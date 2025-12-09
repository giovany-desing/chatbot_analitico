"""
Módulo de validación de datos para el chatbot analítico.
Incluye validación de queries, resultados SQL, datos para gráficas, y más.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import pandas as pd

# ============ Constantes de Configuración ============

MAX_QUERY_LENGTH = 500
MIN_QUERY_LENGTH = 3
SQL_INJECTION_PATTERNS = ["DROP", "DELETE", "UPDATE", "INSERT", "--", ";", "/*"]
MIN_DATE_YEAR = 1900
MAX_DATE_YEAR = 2030
MAX_DATE_RANGE_YEARS = 10
MIN_SIMILARITY_THRESHOLD = 0.5
MAX_NULL_PERCENTAGE = 0.8

# ============ Clase ValidationResult ============

@dataclass
class ValidationResult:
    """
    Resultado de una validación.
    
    Attributes:
        is_valid: Si la validación pasó
        error_msg: Mensaje de error si falló (opcional)
        warnings: Lista de advertencias
        metadata: Diccionario con metadatos adicionales (opcional)
    """
    is_valid: bool
    error_msg: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        """Permite usar ValidationResult como booleano"""
        return self.is_valid
    
    def __str__(self) -> str:
        """Representación legible para logging"""
        status = "✅ VÁLIDO" if self.is_valid else "❌ INVÁLIDO"
        msg = f"{status}"
        
        if self.error_msg:
            msg += f" | Error: {self.error_msg}"
        
        if self.warnings:
            msg += f" | Warnings: {len(self.warnings)}"
        
        if self.metadata:
            metadata_str = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            msg += f" | Metadata: {metadata_str}"
        
        return msg


# ============ Loggers por Función ============

_logger_validate_query = logging.getLogger(f"{__name__}.validate_user_query")
_logger_validate_sql = logging.getLogger(f"{__name__}.validate_sql_results")
_logger_validate_chart = logging.getLogger(f"{__name__}.validate_data_for_chart")
_logger_sanitize_date = logging.getLogger(f"{__name__}.sanitize_date_range")
_logger_validate_rag = logging.getLogger(f"{__name__}.validate_rag_context")


# ============ Validación de User Query ============

def validate_user_query(query: str) -> ValidationResult:
    """
    Valida una query del usuario.
    
    Args:
        query: Query del usuario a validar
    
    Returns:
        ValidationResult con:
        - is_valid: True si es válida
        - error_msg: Mensaje de error si no es válida
        - sanitized_query: Query sanitizada (sin espacios extra)
        - metadata: Información adicional (longitud, etc.)
    """
    logger = _logger_validate_query
    
    # Verificar que no sea None
    if query is None:
        error_msg = "La query no puede ser None"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"query_length": 0}
        )
    
    # Normalizar: strip y verificar
    sanitized_query = query.strip()
    
    # Verificar vacía o solo espacios
    if not sanitized_query:
        error_msg = "La query no puede estar vacía o contener solo espacios"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"query_length": 0, "sanitized_query": ""}
        )
    
    # Verificar solo caracteres especiales (sin letras ni números)
    if not re.search(r'[a-zA-Z0-9]', sanitized_query):
        error_msg = "La query no puede contener solo caracteres especiales"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"query_length": len(sanitized_query), "sanitized_query": sanitized_query}
        )
    
    # Validar longitud
    query_length = len(sanitized_query)
    
    if query_length < MIN_QUERY_LENGTH:
        error_msg = f"La query es muy corta (mínimo {MIN_QUERY_LENGTH} caracteres, tienes {query_length})"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"query_length": query_length, "sanitized_query": sanitized_query}
        )
    
    if query_length > MAX_QUERY_LENGTH:
        error_msg = f"La query es muy larga (máximo {MAX_QUERY_LENGTH} caracteres, tienes {query_length})"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"query_length": query_length, "sanitized_query": sanitized_query}
        )
    
    # Detectar SQL injection
    query_upper = sanitized_query.upper()
    detected_patterns = []
    
    for pattern in SQL_INJECTION_PATTERNS:
        # Buscar patrón como palabra completa o como parte de comando SQL
        if pattern in ["--", ";", "/*"]:
            # Caracteres especiales: buscar directamente
            if pattern in query_upper:
                detected_patterns.append(pattern)
        else:
            # Comandos SQL: buscar como palabra completa
            pattern_regex = r'\b' + re.escape(pattern) + r'\b'
            if re.search(pattern_regex, query_upper, re.IGNORECASE):
                detected_patterns.append(pattern)
    
    if detected_patterns:
        error_msg = f"La query contiene patrones sospechosos de SQL injection: {', '.join(detected_patterns)}"
        logger.warning(f"❌ {error_msg} | Query: {sanitized_query[:50]}...")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={
                "query_length": query_length,
                "detected_patterns": detected_patterns,
                "sanitized_query": sanitized_query
            }
        )
    
    # Query válida
    logger.info(f"✅ Query válida (longitud: {query_length})")
    return ValidationResult(
        is_valid=True,
        error_msg=None,
        metadata={"query_length": query_length, "sanitized_query": sanitized_query}
    )


# ============ Validación de Resultados SQL ============

def validate_sql_results(
    results: List[Dict],
    min_rows: int = 1
) -> ValidationResult:
    """
    Valida resultados de una query SQL.
    
    Args:
        results: Lista de diccionarios con los resultados
        min_rows: Número mínimo de filas esperadas (default: 1)
    
    Returns:
        ValidationResult con:
        - is_valid: True si los resultados son válidos
        - error_msg: Mensaje de error si no son válidos
        - warnings: Lista de advertencias
        - row_count: Número de filas retornadas
        - metadata: Información adicional
    """
    logger = _logger_validate_sql
    
    # Verificar que no sea None
    if results is None:
        error_msg = "Los resultados no pueden ser None"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            warnings=[],
            metadata={"row_count": 0}
        )
    
    # Verificar que no esté vacío
    if not results:
        error_msg = f"Los resultados están vacíos (mínimo esperado: {min_rows} filas)"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            warnings=[],
            metadata={"row_count": 0}
        )
    
    row_count = len(results)
    
    # Verificar mínimo de filas
    if row_count < min_rows:
        error_msg = f"Resultados insuficientes: {row_count} filas (mínimo esperado: {min_rows})"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            warnings=[],
            metadata={"row_count": row_count}
        )
    
    # Validar que todas las filas tengan las mismas claves
    if not results:
        error_msg = "Lista de resultados vacía"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            warnings=[],
            metadata={"row_count": 0}
        )
    
    first_row_keys = set(results[0].keys())
    
    if not first_row_keys:
        error_msg = "La primera fila no tiene columnas"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            warnings=[],
            metadata={"row_count": row_count}
        )
    
    # Verificar consistencia de columnas
    inconsistent_rows = []
    for i, row in enumerate(results[1:], start=1):
        row_keys = set(row.keys())
        if row_keys != first_row_keys:
            inconsistent_rows.append(i)
            if len(inconsistent_rows) > 10:  # Limitar reporte
                break
    
    if inconsistent_rows:
        error_msg = f"Filas con columnas inconsistentes: {inconsistent_rows[:10]}"
        logger.warning(f"❌ {error_msg} | Total de filas inconsistentes: {len(inconsistent_rows)}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            warnings=[],
            metadata={
                "row_count": row_count,
                "inconsistent_rows": inconsistent_rows[:10],
                "expected_columns": list(first_row_keys)
            }
        )
    
    # Detectar exceso de valores None por columna
    warnings = []
    column_null_percentages = {}
    
    for col in first_row_keys:
        none_count = sum(1 for row in results if row.get(col) is None)
        null_percentage = none_count / row_count if row_count > 0 else 0
        column_null_percentages[col] = null_percentage
        
        if null_percentage > MAX_NULL_PERCENTAGE:
            warning_msg = f"Columna '{col}' tiene {null_percentage:.1%} de valores None (> {MAX_NULL_PERCENTAGE:.0%})"
            warnings.append(warning_msg)
            logger.warning(f"⚠️ {warning_msg}")
    
    # Advertencias adicionales
    if row_count > 1000:
        warnings.append(f"Gran cantidad de resultados ({row_count}), considerar agregar LIMIT")
        logger.info(f"ℹ️ Gran cantidad de resultados: {row_count}")
    
    if len(first_row_keys) > 20:
        warnings.append(f"Muchas columnas ({len(first_row_keys)}), puede ser difícil de visualizar")
        logger.info(f"ℹ️ Muchas columnas: {len(first_row_keys)}")
    
    logger.info(f"✅ Resultados SQL válidos: {row_count} filas, {len(first_row_keys)} columnas")
    
    return ValidationResult(
        is_valid=True,
        error_msg=None,
        warnings=warnings,
        metadata={
            "row_count": row_count,
            "column_count": len(first_row_keys),
            "columns": list(first_row_keys),
            "null_percentages": column_null_percentages
        }
    )


# ============ Validación de Datos para Gráficas ============

def validate_data_for_chart(
    data: List[Dict],
    chart_type: str
) -> ValidationResult:
    """
    Valida que los datos sean apropiados para el tipo de gráfica.
    
    Args:
        data: Lista de diccionarios con los datos
        chart_type: Tipo de gráfica (bar, line, pie, scatter, heatmap, etc.)
    
    Returns:
        ValidationResult con:
        - is_valid: True si los datos son válidos para el tipo de gráfica
        - suggestions: Lista de sugerencias
        - alternative_chart: Tipo de gráfica alternativo sugerido (si aplica)
        - metadata: Información sobre tipos de columnas detectadas
    """
    logger = _logger_validate_chart
    
    if not data:
        error_msg = "No hay datos para graficar"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"suggestions": [], "alternative_chart": None}
        )
    
    # Convertir a DataFrame para análisis
    try:
        df = pd.DataFrame(data)
    except Exception as e:
        error_msg = f"Error convirtiendo datos a DataFrame: {e}"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"suggestions": [], "alternative_chart": None}
        )
    
    if df.empty:
        error_msg = "DataFrame vacío"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"suggestions": [], "alternative_chart": None}
        )
    
    # Inferir tipos de columnas
    numeric_cols = list(df.select_dtypes(include=['number']).columns)
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    datetime_cols = list(df.select_dtypes(include=['datetime64']).columns)
    bool_cols = list(df.select_dtypes(include=['bool']).columns)
    
    chart_type_lower = chart_type.lower()
    suggestions = []
    alternative_chart = None
    
    # Validación específica por tipo de gráfica
    if chart_type_lower in ['bar', 'column']:
        if len(categorical_cols) == 0:
            error_msg = "Gráfica de barras requiere al menos 1 columna categórica"
            logger.warning(f"❌ {error_msg}")
            alternative_chart = "line" if len(numeric_cols) >= 2 else None
            return ValidationResult(
                is_valid=False,
                error_msg=error_msg,
                metadata={
                    "suggestions": [f"Usar gráfica de tipo '{alternative_chart}'" if alternative_chart else "Agregar columna categórica"],
                    "alternative_chart": alternative_chart,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols,
                    "datetime_cols": datetime_cols
                }
            )
        
        if len(numeric_cols) == 0:
            error_msg = "Gráfica de barras requiere al menos 1 columna numérica"
            logger.warning(f"❌ {error_msg}")
            return ValidationResult(
                is_valid=False,
                error_msg=error_msg,
                metadata={
                    "suggestions": ["Agregar columna numérica para valores"],
                    "alternative_chart": None,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols
                }
            )
        
        if len(categorical_cols) > 1:
            suggestions.append(f"Múltiples columnas categóricas ({len(categorical_cols)}), considerar agrupar")
        
        if len(numeric_cols) > 3:
            suggestions.append(f"Múltiples columnas numéricas ({len(numeric_cols)}), considerar gráfica de barras agrupadas")
        
        # Verificar valores negativos
        for col in numeric_cols:
            if (df[col] < 0).any():
                suggestions.append(f"Columna '{col}' contiene valores negativos, verificar si es correcto para barras")
    
    elif chart_type_lower in ['line', 'area']:
        # Para líneas, necesitamos una columna temporal/secuencial
        has_temporal = len(datetime_cols) > 0
        
        # Verificar si hay columna que parezca temporal
        potential_temporal = []
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['fecha', 'date', 'time', 'mes', 'año', 'year', 'month', 'dia', 'day', 'timestamp']:
                potential_temporal.append(col)
            elif col in numeric_cols and df[col].is_monotonic_increasing:
                potential_temporal.append(col)
        
        if not has_temporal and not potential_temporal:
            error_msg = "Gráfica de línea requiere una columna temporal o secuencial"
            logger.warning(f"❌ {error_msg}")
            alternative_chart = "bar" if len(categorical_cols) > 0 and len(numeric_cols) > 0 else None
            return ValidationResult(
                is_valid=False,
                error_msg=error_msg,
                metadata={
                    "suggestions": [f"Usar gráfica de tipo '{alternative_chart}'" if alternative_chart else "Agregar columna temporal"],
                    "alternative_chart": alternative_chart,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols,
                    "datetime_cols": datetime_cols
                }
            )
        
        if len(numeric_cols) == 0:
            error_msg = "Gráfica de línea requiere al menos 1 columna numérica"
            logger.warning(f"❌ {error_msg}")
            return ValidationResult(
                is_valid=False,
                error_msg=error_msg,
                metadata={
                    "suggestions": ["Agregar columna numérica"],
                    "alternative_chart": None,
                    "numeric_cols": numeric_cols,
                    "datetime_cols": datetime_cols
                }
            )
        
        if len(numeric_cols) > 5:
            suggestions.append(f"Múltiples series ({len(numeric_cols)}), puede ser difícil de leer")
    
    elif chart_type_lower in ['pie', 'donut']:
        if len(categorical_cols) == 0:
            error_msg = "Gráfica de pastel requiere al menos 1 columna categórica"
            logger.warning(f"❌ {error_msg}")
            alternative_chart = "bar" if len(numeric_cols) > 0 else None
            return ValidationResult(
                is_valid=False,
                error_msg=error_msg,
                metadata={
                    "suggestions": [f"Usar gráfica de tipo '{alternative_chart}'" if alternative_chart else "Agregar columna categórica"],
                    "alternative_chart": alternative_chart,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols
                }
            )
        
        if len(numeric_cols) == 0:
            error_msg = "Gráfica de pastel requiere al menos 1 columna numérica"
            logger.warning(f"❌ {error_msg}")
            return ValidationResult(
                is_valid=False,
                error_msg=error_msg,
                metadata={
                    "suggestions": ["Agregar columna numérica"],
                    "alternative_chart": None,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols
                }
            )
        
        # Verificar valores positivos y suma > 0
        for col in numeric_cols:
            if (df[col] < 0).any():
                error_msg = f"Gráfica de pastel no puede tener valores negativos en '{col}'"
                logger.warning(f"❌ {error_msg}")
                alternative_chart = "bar"
                return ValidationResult(
                    is_valid=False,
                    error_msg=error_msg,
                    metadata={
                        "suggestions": [f"Usar gráfica de tipo '{alternative_chart}'"],
                        "alternative_chart": alternative_chart,
                        "numeric_cols": numeric_cols,
                        "categorical_cols": categorical_cols,
                        "problematic_column": col
                    }
                )
            
            total = df[col].sum()
            if total == 0:
                error_msg = f"La suma de valores en '{col}' es cero"
                logger.warning(f"❌ {error_msg}")
                return ValidationResult(
                    is_valid=False,
                    error_msg=error_msg,
                    metadata={
                        "suggestions": ["Verificar datos o usar otro tipo de gráfica"],
                        "alternative_chart": None,
                        "numeric_cols": numeric_cols,
                        "categorical_cols": categorical_cols,
                        "problematic_column": col
                    }
                )
        
        # Advertencias
        if len(df) > 10:
            suggestions.append(f"Muchas categorías ({len(df)}), considerar agrupar las menores o usar otro tipo de gráfica")
    
    elif chart_type_lower in ['scatter', 'bubble']:
        if len(numeric_cols) < 2:
            error_msg = "Gráfica de dispersión requiere al menos 2 columnas numéricas"
            logger.warning(f"❌ {error_msg}")
            alternative_chart = "bar" if len(categorical_cols) > 0 and len(numeric_cols) > 0 else None
            return ValidationResult(
                is_valid=False,
                error_msg=error_msg,
                metadata={
                    "suggestions": [f"Usar gráfica de tipo '{alternative_chart}'" if alternative_chart else "Agregar más columnas numéricas"],
                    "alternative_chart": alternative_chart,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols
                }
            )
        
        if len(numeric_cols) > 3:
            suggestions.append(f"Múltiples columnas numéricas ({len(numeric_cols)}), considerar usar solo 2-3 para scatter")
        
        if len(df) < 3:
            suggestions.append(f"Pocos puntos de datos ({len(df)}), scatter puede no ser informativo")
    
    elif chart_type_lower in ['heatmap']:
        if len(categorical_cols) < 2:
            error_msg = "Heatmap requiere al menos 2 columnas categóricas"
            logger.warning(f"❌ {error_msg}")
            return ValidationResult(
                is_valid=False,
                error_msg=error_msg,
                metadata={
                    "suggestions": ["Agregar más columnas categóricas"],
                    "alternative_chart": None,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols
                }
            )
        
        if len(numeric_cols) == 0:
            error_msg = "Heatmap requiere al menos 1 columna numérica"
            logger.warning(f"❌ {error_msg}")
            return ValidationResult(
                is_valid=False,
                error_msg=error_msg,
                metadata={
                    "suggestions": ["Agregar columna numérica"],
                    "alternative_chart": None,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols
                }
            )
    
    elif chart_type_lower in ['histogram', 'hist']:
        if len(numeric_cols) == 0:
            error_msg = "Histograma requiere al menos 1 columna numérica"
            logger.warning(f"❌ {error_msg}")
            return ValidationResult(
                is_valid=False,
                error_msg=error_msg,
                metadata={
                    "suggestions": ["Agregar columna numérica"],
                    "alternative_chart": None,
                    "numeric_cols": numeric_cols,
                    "categorical_cols": categorical_cols
                }
            )
        
        if len(df) < 10:
            suggestions.append(f"Pocos datos ({len(df)}), histograma puede no ser representativo")
    
    else:
        # Tipo de gráfica desconocido
        suggestions.append(f"Tipo de gráfica '{chart_type}' no tiene validación específica")
        logger.info(f"ℹ️ Tipo de gráfica '{chart_type}' sin validación específica")
    
    # Validaciones generales
    if len(df) > 10000:
        suggestions.append(f"Muchos datos ({len(df)}), considerar agregar o filtrar para mejor rendimiento")
    
    if len(df.columns) > 15:
        suggestions.append(f"Muchas columnas ({len(df.columns)}), puede ser difícil de visualizar")
    
    if suggestions:
        logger.info(f"ℹ️ Sugerencias para gráfica '{chart_type}': {len(suggestions)}")
    
    logger.info(f"✅ Datos válidos para gráfica '{chart_type}': {len(df)} filas, {len(df.columns)} columnas")
    
    return ValidationResult(
        is_valid=True,
        error_msg=None,
        warnings=suggestions if suggestions else [],
        metadata={
            "suggestions": suggestions,
            "alternative_chart": alternative_chart,
            "chart_type": chart_type,
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "bool_cols": bool_cols
        }
    )


# ============ Sanitización de Rango de Fechas ============

def sanitize_date_range(
    start_date: Any,
    end_date: Any
) -> Tuple[datetime, datetime]:
    """
    Sanitiza y valida un rango de fechas.
    
    Args:
        start_date: Fecha de inicio (date, datetime, str ISO, o None)
        end_date: Fecha de fin (date, datetime, str ISO, o None)
    
    Returns:
        Tupla (start_date, end_date) como objetos datetime
    
    Raises:
        ValueError: Si las fechas son inválidas o están fuera del rango permitido
    """
    logger = _logger_sanitize_date
    
    def parse_date(d: Any) -> datetime:
        """Convierte diferentes formatos a datetime"""
        if d is None:
            raise ValueError("Fecha no puede ser None")
        
        if isinstance(d, datetime):
            return d
        
        if isinstance(d, date):
            return datetime.combine(d, datetime.min.time())
        
        if isinstance(d, str):
            # Intentar parsear diferentes formatos ISO
            formats = [
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%d/%m/%Y'
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(d, fmt)
                except ValueError:
                    continue
            raise ValueError(f"No se pudo parsear la fecha: {d}")
        
        raise ValueError(f"Tipo de fecha no soportado: {type(d)}")
    
    # Parsear fechas
    try:
        start = parse_date(start_date)
        end = parse_date(end_date)
    except ValueError as e:
        error_msg = f"Error parseando fechas: {e}"
        logger.warning(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # Validar años
    if start.year < MIN_DATE_YEAR or start.year > MAX_DATE_YEAR:
        error_msg = f"Año de inicio ({start.year}) fuera del rango permitido ({MIN_DATE_YEAR}-{MAX_DATE_YEAR})"
        logger.warning(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    if end.year < MIN_DATE_YEAR or end.year > MAX_DATE_YEAR:
        error_msg = f"Año de fin ({end.year}) fuera del rango permitido ({MIN_DATE_YEAR}-{MAX_DATE_YEAR})"
        logger.warning(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
    # Corregir orden si end < start
    if end < start:
        logger.warning(f"⚠️ Fechas invertidas: end ({end}) < start ({start}), invirtiendo orden")
        start, end = end, start
    
    # Validar rango máximo
    range_days = (end - start).days
    range_years = range_days / 365.25
    
    if range_years > MAX_DATE_RANGE_YEARS:
        logger.warning(f"⚠️ Rango de fechas muy grande ({range_years:.1f} años > {MAX_DATE_RANGE_YEARS}), limitando")
        max_days = int(MAX_DATE_RANGE_YEARS * 365.25)
        end = start + timedelta(days=max_days)
        logger.info(f"ℹ️ Rango limitado a {MAX_DATE_RANGE_YEARS} años: {start.date()} - {end.date()}")
    
    logger.info(f"✅ Rango de fechas válido: {start.date()} - {end.date()} ({range_days} días)")
    
    return start, end


# ============ Validación de Contexto RAG ============

def validate_rag_context(
    rag_examples: List[Dict],
    min_similarity: float = MIN_SIMILARITY_THRESHOLD
) -> ValidationResult:
    """
    Valida que el contexto RAG sea útil.
    
    Args:
        rag_examples: Lista de ejemplos retornados por RAG
        min_similarity: Umbral mínimo de similitud (default: 0.5)
    
    Returns:
        ValidationResult con:
        - is_valid: True si el contexto es válido
        - warning_msg: Mensaje de advertencia si hay problemas
        - avg_similarity: Promedio de similitud de los ejemplos
        - metadata: Información sobre los ejemplos
    """
    logger = _logger_validate_rag
    
    if not rag_examples:
        error_msg = "RAG no retornó ejemplos"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"example_count": 0, "avg_similarity": 0.0}
        )
    
    if len(rag_examples) == 0:
        error_msg = "Lista de ejemplos RAG vacía"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"example_count": 0, "avg_similarity": 0.0}
        )
    
    # Verificar estructura de ejemplos y calcular similitud promedio
    valid_examples = 0
    similarity_scores = []
    low_similarity_count = 0
    
    for i, example in enumerate(rag_examples):
        if not example:
            continue
        
        valid_examples += 1
        
        # Extraer similarity score si está disponible
        similarity = None
        if isinstance(example, dict):
            similarity = example.get('similarity') or example.get('score')
            if similarity is None and 'metadata' in example:
                similarity = example['metadata'].get('similarity')
        
        if similarity is not None:
            try:
                similarity = float(similarity)
                similarity_scores.append(similarity)
                
                if similarity < min_similarity:
                    low_similarity_count += 1
                    logger.warning(f"⚠️ Ejemplo RAG {i+1} con baja similitud: {similarity:.3f} < {min_similarity}")
            except (ValueError, TypeError):
                pass
    
    if valid_examples == 0:
        error_msg = "No se encontraron ejemplos RAG válidos"
        logger.warning(f"❌ {error_msg}")
        return ValidationResult(
            is_valid=False,
            error_msg=error_msg,
            metadata={"example_count": len(rag_examples), "avg_similarity": 0.0}
        )
    
    # Calcular promedio de similitud
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    # Verificar si al menos un ejemplo tiene similitud suficiente
    has_good_similarity = any(score >= min_similarity for score in similarity_scores) if similarity_scores else True
    
    warning_msg = None
    if not has_good_similarity and similarity_scores:
        warning_msg = f"Ningún ejemplo RAG tiene similitud >= {min_similarity} (máxima: {max(similarity_scores):.3f})"
        logger.warning(f"⚠️ {warning_msg}")
    elif low_similarity_count > len(rag_examples) / 2:
        warning_msg = f"Muchos ejemplos RAG con baja similitud ({low_similarity_count}/{len(rag_examples)})"
        logger.warning(f"⚠️ {warning_msg}")
    
    logger.info(f"✅ Contexto RAG válido: {valid_examples} ejemplos, similitud promedio: {avg_similarity:.3f}")
    
    return ValidationResult(
        is_valid=True,
        error_msg=None,
        warnings=[warning_msg] if warning_msg else [],
        metadata={
            "example_count": valid_examples,
            "total_examples": len(rag_examples),
            "avg_similarity": avg_similarity,
            "similarity_scores": similarity_scores,
            "low_similarity_count": low_similarity_count,
            "has_good_similarity": has_good_similarity
        }
    )


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Testing Data Validator ===\n")
    
    # Test 1: validate_user_query
    print("1. Testing validate_user_query:")
    result = validate_user_query("¿Cuántas ventas hay?")
    print(f"   {result}")
    
    result = validate_user_query("SELECT * FROM ventas; DROP TABLE ventas;")
    print(f"   {result}")
    
    # Test 2: validate_sql_results
    print("\n2. Testing validate_sql_results:")
    results = [
        {"producto": "A", "cantidad": 10},
        {"producto": "B", "cantidad": 20}
    ]
    result = validate_sql_results(results)
    print(f"   {result}")
    
    # Test 3: validate_data_for_chart
    print("\n3. Testing validate_data_for_chart:")
    data = [
        {"producto": "A", "ventas": 100},
        {"producto": "B", "ventas": 200}
    ]
    result = validate_data_for_chart(data, "bar")
    print(f"   {result}")
    
    print("\n✅ Tests completados")
