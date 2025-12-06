from typing import Optional, Dict, List
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from ..config import settings

class FeedbackService:
    """Servicio para gestionar feedback de usuarios y métricas"""

    def __init__(self):
        self.conn_params = {
            'host': settings.POSTGRES_HOST,
            'port': settings.POSTGRES_PORT,
            'user': settings.POSTGRES_USER,
            'password': settings.POSTGRES_PASSWORD,
            'database': settings.POSTGRES_DB
        }

    def _get_connection(self):
        """Obtiene conexión a PostgreSQL"""
        return psycopg2.connect(**self.conn_params)

    def save_interaction(
        self,
        session_id: str,
        user_query: str,
        sql_generated: Optional[str] = None,
        chart_type: Optional[str] = None,
        chart_config: Optional[Dict] = None,
        response_time_ms: Optional[int] = None,
        error_occurred: bool = False,
        error_message: Optional[str] = None
    ) -> int:
        """
        Guarda una interacción del usuario
        Returns: ID del registro creado
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_feedback (
                        session_id, user_query, sql_generated, chart_type,
                        chart_config, response_time_ms, error_occurred, error_message
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    session_id, user_query, sql_generated, chart_type,
                    json.dumps(chart_config) if chart_config else None,
                    response_time_ms, error_occurred, error_message
                ))
                feedback_id = cur.fetchone()[0]
                conn.commit()
                return feedback_id

    def update_rating(
        self,
        feedback_id: int,
        rating: int,
        feedback_text: Optional[str] = None
    ) -> bool:
        """
        Actualiza la valoración de una interacción
        Returns: True si se actualizó correctamente
        """
        if rating < 1 or rating > 5:
            raise ValueError("Rating debe estar entre 1 y 5")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE user_feedback
                    SET user_rating = %s, feedback_text = %s
                    WHERE id = %s
                """, (rating, feedback_text, feedback_id))
                conn.commit()
                return cur.rowcount > 0

    def get_low_rated_queries(
        self,
        min_rating: int = 2,
        limit: int = 100
    ) -> List[Dict]:
        """
        Obtiene queries con baja valoración para reentrenamiento
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        id, user_query, sql_generated, chart_type,
                        user_rating, feedback_text, created_at
                    FROM user_feedback
                    WHERE user_rating <= %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (min_rating, limit))
                return [dict(row) for row in cur.fetchall()]

    def get_metrics(self, days: int = 7) -> Dict:
        """
        Calcula métricas de rendimiento de los últimos N días
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Métricas generales
                cur.execute("""
                    SELECT
                        COUNT(*) as total_interactions,
                        COUNT(user_rating) as rated_interactions,
                        ROUND(AVG(user_rating)::NUMERIC, 2) as avg_rating,
                        COUNT(CASE WHEN error_occurred THEN 1 END) as errors,
                        ROUND(AVG(response_time_ms)::NUMERIC, 0) as avg_response_time_ms,
                        ROUND((PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms))::NUMERIC, 0) as p95_response_time_ms
                    FROM user_feedback
                    WHERE created_at >= %s
                """, (cutoff_date,))
                general_metrics = dict(cur.fetchone())

                # Distribución de ratings
                cur.execute("""
                    SELECT
                        user_rating,
                        COUNT(*) as count
                    FROM user_feedback
                    WHERE created_at >= %s AND user_rating IS NOT NULL
                    GROUP BY user_rating
                    ORDER BY user_rating
                """, (cutoff_date,))
                rating_distribution = {row['user_rating']: row['count'] for row in cur.fetchall()}

                # Charts más usados
                cur.execute("""
                    SELECT
                        chart_type,
                        COUNT(*) as count,
                        ROUND(AVG(user_rating)::NUMERIC, 2) as avg_rating
                    FROM user_feedback
                    WHERE created_at >= %s AND chart_type IS NOT NULL
                    GROUP BY chart_type
                    ORDER BY count DESC
                    LIMIT 10
                """, (cutoff_date,))
                top_charts = [dict(row) for row in cur.fetchall()]

                # Errores más comunes
                cur.execute("""
                    SELECT
                        error_message,
                        COUNT(*) as count
                    FROM user_feedback
                    WHERE created_at >= %s AND error_occurred = TRUE
                    GROUP BY error_message
                    ORDER BY count DESC
                    LIMIT 10
                """, (cutoff_date,))
                top_errors = [dict(row) for row in cur.fetchall()]

                return {
                    'period_days': days,
                    'general': general_metrics,
                    'rating_distribution': rating_distribution,
                    'top_charts': top_charts,
                    'top_errors': top_errors
                }

    def export_for_retraining(
        self,
        output_file: str = 'retraining_data.jsonl',
        max_rating: int = 3
    ) -> int:
        """
        Exporta datos de baja calidad para reentrenamiento
        Formato compatible con training_data_complete.jsonl
        Returns: Número de ejemplos exportados
        """
        queries = self.get_low_rated_queries(min_rating=max_rating, limit=1000)

        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for query in queries:
                # Formato idéntico al dataset de entrenamiento original
                example = {
                    'messages': [
                        {
                            'role': 'system',
                            'content': 'Eres un experto en visualización de datos para análisis de ventas textiles. Debes elegir el mejor tipo de gráfico basándote en la query del usuario y los datos SQL disponibles.'
                        },
                        {
                            'role': 'user',
                            'content': f"Query: {query['user_query']}\nSQL: {query['sql_generated']}\nColumnas: {['producto', 'total']}\nFilas: 10\nData preview: []"
                        },
                        {
                            'role': 'assistant',
                            'content': json.dumps({
                                'chart_type': query['chart_type'],
                                'reasoning': query['feedback_text'] or 'Necesita mejora según feedback de usuario',
                                'confidence': 0.70,  # Baja confianza por rating bajo
                                'user_rating': query['user_rating'],
                                'needs_review': True
                            }, ensure_ascii=False)
                        }
                    ]
                }
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                count += 1

        return count

# Singleton instance
feedback_service = FeedbackService()