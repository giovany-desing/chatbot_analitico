"""
Script de exportación automática de datos de entrenamiento.
Exporta datos de alta calidad para reentrenamiento del modelo de visualización.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import logging
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
from sqlalchemy import text

from app.db.connections import get_postgres

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ Función Principal de Exportación ============

def export_training_data(
    days: int = 7,
    min_confidence: float = 0.8,
    output_dir: str = "data/retraining",
    min_rating: int = 4
) -> str:
    """
    Exporta datos de calidad para reentrenamiento del modelo.

    Criterios de selección:
    - Queries exitosas (success=True)
    - Alta confianza (confidence >= min_confidence)
    - Sin errores ni correcciones
    - Feedback positivo si disponible (rating >= min_rating)
    - Balance entre tipos de visualización

    Args:
        days: Número de días hacia atrás
        min_confidence: Confianza mínima requerida
        output_dir: Directorio de salida
        min_rating: Rating mínimo para feedback positivo

    Returns:
        Path al archivo generado
    """
    logger.info(f"📊 Exportando datos de entrenamiento (últimos {days} días)")

    # Consultar performance_metrics
    successful_queries = get_successful_queries(days, min_confidence)
    logger.info(f"   Queries exitosas encontradas: {len(successful_queries)}")

    # Consultar feedback positivo
    positive_feedback = get_positive_feedback(days, min_rating)
    logger.info(f"   Feedback positivo encontrado: {len(positive_feedback)}")

    # Combinar y deduplicar
    training_samples = merge_and_deduplicate(successful_queries, positive_feedback)
    logger.info(f"   Ejemplos únicos después de deduplicar: {len(training_samples)}")

    # Balancear clases
    balanced_samples = balance_chart_types(training_samples)
    logger.info(f"   Ejemplos después de balancear: {len(balanced_samples)}")

    # Convertir a formato de entrenamiento
    training_data = format_for_training(balanced_samples)

    # Guardar
    filename = f"training_data_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in training_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    logger.info(f"✅ Exportados {len(training_data)} ejemplos a {output_path}")

    return str(output_path)


# ============ Función para Obtener Queries Exitosas ============

def get_successful_queries(days: int, min_confidence: float) -> List[Dict]:
    """
    Obtiene queries de alta calidad de performance_metrics.

    Args:
        days: Número de días hacia atrás
        min_confidence: Confianza mínima requerida

    Returns:
        Lista de dicts con user_query, sql_query, chart_type, intent_confidence, metadata
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()

        cutoff_date = datetime.now() - timedelta(days=days)

        # Query para obtener queries exitosas con alta confianza
        # Usamos DISTINCT ON para evitar duplicados por (user_query, chart_type)
        sql = """
        SELECT DISTINCT ON (pm.user_query, pm.chart_type)
            pm.user_query,
            pm.sql_query,
            pm.chart_type,
            pm.intent_confidence,
            pm.metadata,
            pm.timestamp
        FROM performance_metrics pm
        WHERE pm.component IN ('viz', 'hybrid')
            AND pm.success = TRUE
            AND pm.intent_confidence >= :min_confidence
            AND (pm.sql_correction_attempts = 0 OR pm.sql_correction_attempts IS NULL)
            AND pm.timestamp >= :cutoff_date
            AND pm.chart_type IS NOT NULL
            AND pm.user_query IS NOT NULL
            AND pm.sql_query IS NOT NULL
        ORDER BY pm.user_query, pm.chart_type, pm.timestamp DESC
        """

        result = session.execute(
            text(sql),
            {
                'min_confidence': min_confidence,
                'cutoff_date': cutoff_date
            }
        ).fetchall()

        session.close()

        # Convertir a lista de dicts
        queries = []
        for row in result:
            queries.append({
                'user_query': row[0],
                'sql_query': row[1],
                'chart_type': row[2],
                'intent_confidence': float(row[3]) if row[3] else 0.0,
                'metadata': row[4] if isinstance(row[4], dict) else json.loads(row[4]) if row[4] else {},
                'timestamp': row[5]
            })

        return queries

    except Exception as e:
        logger.error(f"❌ Error obteniendo queries exitosas: {e}", exc_info=True)
        return []


# ============ Función para Obtener Feedback Positivo ============

def get_positive_feedback(days: int, min_rating: int = 4) -> List[Dict]:
    """
    Obtiene ejemplos con feedback positivo del usuario.

    Args:
        days: Número de días hacia atrás
        min_rating: Rating mínimo requerido (default: 4)

    Returns:
        Lista de dicts con user_query, sql_query, chart_type, rating
    """
    try:
        postgres = get_postgres()
        session = postgres.get_session()

        cutoff_date = datetime.now() - timedelta(days=days)

        sql = """
        SELECT
            f.user_query,
            f.sql_generated as sql_query,
            f.chart_type,
            f.chart_config,
            f.user_rating,
            f.created_at
        FROM user_feedback f
        WHERE f.user_rating >= :min_rating
            AND f.created_at >= :cutoff_date
            AND f.chart_type IS NOT NULL
            AND f.user_query IS NOT NULL
            AND f.sql_generated IS NOT NULL
            AND f.error_occurred = FALSE
        ORDER BY f.user_rating DESC, f.created_at DESC
        """

        result = session.execute(
            text(sql),
            {
                'min_rating': min_rating,
                'cutoff_date': cutoff_date
            }
        ).fetchall()

        session.close()

        # Convertir a lista de dicts
        feedback_list = []
        for row in result:
            chart_config = row[3]
            if isinstance(chart_config, str):
                try:
                    chart_config = json.loads(chart_config)
                except:
                    chart_config = {}

            feedback_list.append({
                'user_query': row[0],
                'sql_query': row[1],
                'chart_type': row[2],
                'chart_config': chart_config,
                'rating': row[4],
                'timestamp': row[5],
                'source': 'feedback'  # Marcar como proveniente de feedback
            })

        return feedback_list

    except Exception as e:
        logger.error(f"❌ Error obteniendo feedback positivo: {e}", exc_info=True)
        return []


# ============ Función de Combinación y Deduplicación ============

def merge_and_deduplicate(
    successful_queries: List[Dict],
    positive_feedback: List[Dict]
) -> List[Dict]:
    """
    Combina queries exitosas y feedback positivo, eliminando duplicados.

    Args:
        successful_queries: Lista de queries exitosas
        positive_feedback: Lista de feedback positivo

    Returns:
        Lista deduplicada de ejemplos
    """
    # Usar un set para trackear combinaciones únicas (user_query + chart_type)
    seen = set()
    merged = []

    # Agregar feedback positivo primero (tiene prioridad)
    for item in positive_feedback:
        key = (item['user_query'].lower().strip(), item['chart_type'])
        if key not in seen:
            seen.add(key)
            merged.append(item)

    # Agregar queries exitosas que no estén ya en el set
    for item in successful_queries:
        key = (item['user_query'].lower().strip(), item['chart_type'])
        if key not in seen:
            seen.add(key)
            merged.append(item)

    return merged


# ============ Función de Balanceo de Clases ============

def balance_chart_types(samples: List[Dict], max_per_type: int = 50) -> List[Dict]:
    """
    Balancea tipos de gráficas para evitar sesgo.

    Args:
        samples: Lista de ejemplos
        max_per_type: Máximo de ejemplos por tipo de gráfica

    Returns:
        Lista balanceada de ejemplos
    """
    by_chart_type = defaultdict(list)
    for sample in samples:
        chart_type = sample.get('chart_type', 'unknown')
        by_chart_type[chart_type].append(sample)

    balanced = []
    for chart_type, items in by_chart_type.items():
        # Si hay más de max_per_type, samplear aleatoriamente
        if len(items) > max_per_type:
            items = random.sample(items, max_per_type)
        balanced.extend(items)

        logger.info(f"  {chart_type}: {len(items)} ejemplos")

    return balanced


# ============ Función de Formateo para Entrenamiento ============

def format_for_training(samples: List[Dict]) -> List[Dict]:
    """
    Convierte a formato chat de FASE_1 (compatible con training_data.jsonl).

    Args:
        samples: Lista de ejemplos en formato raw

    Returns:
        Lista de ejemplos en formato de entrenamiento
    """
    training_data = []

    SYSTEM_PROMPT = """Eres un experto en visualización de datos para análisis de ventas textiles. Debes elegir el mejor tipo de gráfico basándote en la query del usuario y los datos SQL disponibles."""

    for sample in samples:
        user_query = sample.get('user_query', '')
        sql_query = sample.get('sql_query', '')
        chart_type = sample.get('chart_type', 'bar')
        chart_config = sample.get('chart_config', {})
        metadata = sample.get('metadata', {})

        # Extraer información de columnas y datos si está disponible en metadata
        columns = metadata.get('columns', [])
        rows_count = metadata.get('rows_count', metadata.get('rows_returned', 10))
        data_preview = metadata.get('data_preview', [])
        
        # Si no hay columnas en metadata, intentar extraer del SQL (básico)
        if not columns and sql_query:
            # Intentar extraer columnas del SELECT (muy básico, solo para casos simples)
            select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql_query, re.IGNORECASE)
            if select_match:
                select_clause = select_match.group(1)
                # Extraer nombres de columnas (muy básico)
                cols = [col.strip().split(' as ')[-1].split(' AS ')[-1].strip() 
                       for col in select_clause.split(',')]
                columns = [col for col in cols if col and not col.startswith('(')]
        
        # Si aún no hay columnas, usar valores por defecto
        if not columns:
            columns = ['columna1', 'columna2']

        # Formatear preview de datos
        if data_preview and isinstance(data_preview, list):
            data_preview_str = json.dumps(data_preview[:2], ensure_ascii=False)
        else:
            data_preview_str = "[]"

        # Construir contenido del usuario
        user_content = f"Query: {user_query}\nSQL: {sql_query}\nColumnas: {json.dumps(columns, ensure_ascii=False)}\nFilas: {rows_count}\nData preview: {data_preview_str}"

        # Construir respuesta del asistente
        assistant_config = {
            "chart_type": chart_type,
            "reasoning": _generate_reasoning(chart_type, user_query),
            "confidence": sample.get('intent_confidence', 0.95),
        }

        # Agregar configuración si está disponible
        if chart_config:
            if isinstance(chart_config, dict):
                assistant_config["config"] = chart_config
            else:
                assistant_config["config"] = {}

        # Si hay rating, agregarlo como metadata
        if 'rating' in sample:
            assistant_config["user_rating"] = sample['rating']
            assistant_config["source"] = "user_feedback"

        assistant_content = json.dumps(assistant_config, ensure_ascii=False)

        training_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })

    return training_data


def _generate_reasoning(chart_type: str, user_query: str) -> str:
    """
    Genera reasoning básico basado en el tipo de gráfica y la query.

    Args:
        chart_type: Tipo de gráfica
        user_query: Query del usuario

    Returns:
        String con el reasoning
    """
    reasoning_map = {
        "bar": "Bar chart es ideal para comparar valores entre categorías discretas",
        "line": "Line chart es perfecto para mostrar tendencias y evolución a lo largo del tiempo",
        "pie": "Pie chart es ideal para mostrar distribución y partes de un todo",
        "scatter": "Scatter plot es útil para mostrar relaciones entre dos variables numéricas",
        "area": "Area chart muestra la evolución acumulativa de valores a lo largo del tiempo",
        "table": "Tabla es apropiada para mostrar datos detallados en formato estructurado"
    }

    base_reasoning = reasoning_map.get(chart_type, f"{chart_type} chart es apropiado para esta visualización")
    
    # Agregar contexto de la query si es relevante
    query_lower = user_query.lower()
    if "tendencia" in query_lower or "evolución" in query_lower or "tiempo" in query_lower:
        if chart_type == "line":
            base_reasoning += ". La query menciona tendencias temporales."
    elif "comparar" in query_lower or "vs" in query_lower:
        if chart_type == "bar":
            base_reasoning += ". La query requiere comparación entre categorías."
    elif "distribución" in query_lower or "porcentaje" in query_lower:
        if chart_type == "pie":
            base_reasoning += ". La query busca mostrar distribución o proporciones."

    return base_reasoning


# ============ Script Ejecutable ============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Exporta datos de alta calidad para reentrenamiento del modelo"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Número de días hacia atrás para exportar (default: 7)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Confianza mínima requerida (default: 0.8)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/retraining",
        help="Directorio de salida (default: data/retraining)"
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=4,
        help="Rating mínimo para feedback positivo (default: 4)"
    )

    args = parser.parse_args()

    try:
        output_file = export_training_data(
            days=args.days,
            min_confidence=args.min_confidence,
            output_dir=args.output_dir,
            min_rating=args.min_rating
        )

        logger.info(f"🎉 Exportación completada: {output_file}")

        # Estadísticas
        with open(output_file, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        logger.info(f"📈 Total de ejemplos exportados: {count}")

        # Mostrar distribución por tipo de gráfica
        logger.info("\n📊 Distribución por tipo de gráfica:")
        chart_type_counts = defaultdict(int)
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                assistant_content = data['messages'][2]['content']
                assistant_data = json.loads(assistant_content)
                chart_type = assistant_data.get('chart_type', 'unknown')
                chart_type_counts[chart_type] += 1

        for chart_type, count in sorted(chart_type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"   {chart_type}: {count}")

    except Exception as e:
        logger.error(f"❌ Error en exportación: {e}", exc_info=True)
        sys.exit(1)

