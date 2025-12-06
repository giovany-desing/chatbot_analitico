# Fase 4: Sistema de Feedback y Mejora Continua

> **📌 ACTUALIZADO:** La función `export_for_retraining()` genera datos en formato **chat** (compatible con `training_data_complete.jsonl`). El formato exportado incluye los 3 roles: system, user, assistant.

## 🎯 Objetivo

Implementar un sistema de feedback que permita:
- Capturar valoraciones de usuarios sobre respuestas del chatbot
- Almacenar interacciones problemáticas para reentrenamiento
- Generar métricas de rendimiento en tiempo real
- Crear pipeline automático de mejora continua
- Dashboard de monitoreo y analytics

## ✅ Prerrequisitos

- [x] Fase 1, 2 y 3 completadas
- [x] Docker Compose funcionando
- [x] PostgreSQL con pgvector activo
- [x] Acceso a la base de datos MySQL (lecturas/escrituras)

## 📋 Cambios a Implementar

### 1. Crear Tabla de Feedback en PostgreSQL

**Archivo nuevo:** `migrations/001_create_feedback_table.sql`

```sql
-- Tabla para almacenar feedback de usuarios
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_query TEXT NOT NULL,
    sql_generated TEXT,
    chart_type VARCHAR(50),
    chart_config JSONB,
    user_rating INTEGER CHECK (user_rating BETWEEN 1 AND 5),
    feedback_text TEXT,
    error_occurred BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Índices para búsquedas rápidas
    CONSTRAINT valid_rating CHECK (user_rating IS NULL OR user_rating BETWEEN 1 AND 5)
);

CREATE INDEX idx_feedback_rating ON user_feedback(user_rating);
CREATE INDEX idx_feedback_session ON user_feedback(session_id);
CREATE INDEX idx_feedback_created ON user_feedback(created_at DESC);
CREATE INDEX idx_feedback_errors ON user_feedback(error_occurred) WHERE error_occurred = TRUE;

-- Tabla para métricas agregadas (cache de analytics)
CREATE TABLE IF NOT EXISTS analytics_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    metadata JSONB,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_name ON analytics_metrics(metric_name);
CREATE INDEX idx_metrics_date ON analytics_metrics(calculated_at DESC);
```

**Ejecutar migración:**

```bash
# Desde el directorio raíz del proyecto
docker-compose exec postgres psql -U postgres -d vectordb -f /migrations/001_create_feedback_table.sql
```

### 2. Crear Sistema de Feedback

**Archivo nuevo:** `app/feedback/feedback_service.py`

```python
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
                        ROUND(AVG(user_rating), 2) as avg_rating,
                        COUNT(CASE WHEN error_occurred THEN 1 END) as errors,
                        ROUND(AVG(response_time_ms), 0) as avg_response_time_ms,
                        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms), 0) as p95_response_time_ms
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
                        ROUND(AVG(user_rating), 2) as avg_rating
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
```

### 3. Integrar Feedback en el Workflow

**Modificar:** `app/agents/nodes.py`

Agregar al final del archivo:

```python
from ..feedback.feedback_service import feedback_service
import time

def track_interaction_node(state: State) -> State:
    """
    Nodo para rastrear la interacción y guardar métricas
    Se ejecuta al final del workflow
    """
    start_time = time.time()

    try:
        # Calcular tiempo de respuesta
        response_time_ms = int((time.time() - state.get('start_time', start_time)) * 1000)

        # Guardar interacción
        feedback_id = feedback_service.save_interaction(
            session_id=state.get('session_id', 'unknown'),
            user_query=state['user_query'],
            sql_generated=state.get('sql_query'),
            chart_type=state.get('chart_config', {}).get('type'),
            chart_config=state.get('chart_config'),
            response_time_ms=response_time_ms,
            error_occurred=bool(state.get('error')),
            error_message=state.get('error')
        )

        # Agregar feedback_id al state para el frontend
        state['feedback_id'] = feedback_id

    except Exception as e:
        logger.error(f"Error guardando feedback: {e}")
        # No fallar el workflow por error de tracking

    return state
```

**Modificar:** `app/agents/graph.py`

Agregar el nodo de tracking:

```python
from .nodes import track_interaction_node

# En la función create_graph(), agregar después de todos los nodos:

# Nodo de tracking (al final)
workflow.add_node("track", track_interaction_node)

# Conectar todos los nodos finales al tracking
workflow.add_edge("sql", "track")
workflow.add_edge("hybrid", "track")
workflow.add_edge("error", "track")

# Track es el nodo final
workflow.set_finish_point("track")
```

### 4. Agregar Endpoints de Feedback a la API

**Modificar:** `app/main.py`

Agregar después de los imports existentes:

```python
from .feedback.feedback_service import feedback_service
from pydantic import BaseModel, Field

class FeedbackRequest(BaseModel):
    feedback_id: int = Field(..., description="ID de la interacción")
    rating: int = Field(..., ge=1, le=5, description="Rating de 1 a 5")
    feedback_text: Optional[str] = Field(None, description="Comentario opcional")

class MetricsResponse(BaseModel):
    period_days: int
    general: Dict
    rating_distribution: Dict[int, int]
    top_charts: List[Dict]
    top_errors: List[Dict]
```

Agregar los endpoints al final del archivo antes de `if __name__ == "__main__"`:

```python
@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Enviar valoración de una interacción
    """
    try:
        success = feedback_service.update_rating(
            feedback_id=feedback.feedback_id,
            rating=feedback.rating,
            feedback_text=feedback.feedback_text
        )
        if success:
            return {"status": "success", "message": "Feedback guardado correctamente"}
        else:
            raise HTTPException(status_code=404, detail="Interacción no encontrada")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando feedback: {str(e)}")

@app.get("/metrics", response_model=MetricsResponse, tags=["Analytics"])
async def get_metrics(days: int = 7):
    """
    Obtener métricas de rendimiento de los últimos N días
    """
    try:
        metrics = feedback_service.get_metrics(days=days)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas: {str(e)}")

@app.get("/analytics/low-rated", tags=["Analytics"])
async def get_low_rated_queries(min_rating: int = 2, limit: int = 50):
    """
    Obtener queries con baja valoración para análisis
    """
    try:
        queries = feedback_service.get_low_rated_queries(min_rating=min_rating, limit=limit)
        return {"count": len(queries), "queries": queries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo queries: {str(e)}")

@app.post("/analytics/export-retraining", tags=["Analytics"])
async def export_retraining_data(max_rating: int = 3):
    """
    Exportar datos para reentrenamiento del modelo
    """
    try:
        count = feedback_service.export_for_retraining(max_rating=max_rating)
        return {
            "status": "success",
            "examples_exported": count,
            "file": "retraining_data.jsonl"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando datos: {str(e)}")
```

### 5. Actualizar Frontend con Sistema de Rating

**Modificar:** `front_app.py`

Agregar después de los imports:

```python
import uuid

# Generar session_id persistente
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
```

Agregar función de rating después de mostrar cada respuesta (buscar donde se muestra `response['chart']` y agregar después):

```python
# Sistema de rating (agregar después de mostrar la respuesta)
if 'feedback_id' in response:
    st.markdown("---")
    st.markdown("**¿Qué te pareció esta respuesta?**")

    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 3])

    rating = None
    with col1:
        if st.button("⭐", key=f"rate1_{response['feedback_id']}"):
            rating = 1
    with col2:
        if st.button("⭐⭐", key=f"rate2_{response['feedback_id']}"):
            rating = 2
    with col3:
        if st.button("⭐⭐⭐", key=f"rate3_{response['feedback_id']}"):
            rating = 3
    with col4:
        if st.button("⭐⭐⭐⭐", key=f"rate4_{response['feedback_id']}"):
            rating = 4
    with col5:
        if st.button("⭐⭐⭐⭐⭐", key=f"rate5_{response['feedback_id']}"):
            rating = 5

    if rating:
        # Opcional: pedir comentario para ratings bajos
        feedback_text = None
        if rating <= 3:
            feedback_text = st.text_input(
                "¿Qué podemos mejorar?",
                key=f"feedback_text_{response['feedback_id']}"
            )

        # Enviar feedback
        try:
            feedback_response = requests.post(
                f"{API_URL}/feedback",
                json={
                    "feedback_id": response['feedback_id'],
                    "rating": rating,
                    "feedback_text": feedback_text
                },
                timeout=5
            )
            if feedback_response.status_code == 200:
                st.success(f"¡Gracias por tu valoración de {rating} estrellas!")
        except Exception as e:
            st.error(f"Error enviando feedback: {e}")
```

Agregar página de métricas en el sidebar:

```python
# En el sidebar, agregar después de los ejemplos:
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analytics")

if st.sidebar.button("Ver Métricas"):
    try:
        metrics_response = requests.get(f"{API_URL}/metrics?days=7", timeout=10)
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()

            st.markdown("## 📈 Métricas de los Últimos 7 Días")

            # Métricas generales
            gen = metrics['general']
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Interacciones", gen['total_interactions'])
            col2.metric("Rating Promedio", f"{gen['avg_rating']}/5.0")
            col3.metric("Tiempo Respuesta (avg)", f"{gen['avg_response_time_ms']}ms")

            # Distribución de ratings
            if metrics['rating_distribution']:
                st.markdown("### Distribución de Ratings")
                rating_df = pd.DataFrame([
                    {'Rating': f"{k}⭐", 'Cantidad': v}
                    for k, v in metrics['rating_distribution'].items()
                ])
                st.bar_chart(rating_df.set_index('Rating'))

            # Charts más usados
            if metrics['top_charts']:
                st.markdown("### Gráficos Más Usados")
                charts_df = pd.DataFrame(metrics['top_charts'])
                st.dataframe(charts_df)

            # Errores comunes
            if metrics['top_errors']:
                st.markdown("### Errores Más Comunes")
                errors_df = pd.DataFrame(metrics['top_errors'])
                st.dataframe(errors_df)
    except Exception as e:
        st.error(f"Error obteniendo métricas: {e}")
```

### 6. Actualizar docker-compose.yml

**Modificar:** `docker-compose.yml`

Agregar volumen para migraciones en el servicio `postgres`. Busca la sección `volumes:` del servicio `postgres` y agrega la línea para montar el directorio de migraciones:

**Antes:**
```yaml
  postgres:
    image: ankane/pgvector:latest
    container_name: chatbot_postgres
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_postgres.sql:/docker-entrypoint-initdb.d/init.sql
```

**Después:**
```yaml
  postgres:
    image: ankane/pgvector:latest
    container_name: chatbot_postgres
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_postgres.sql:/docker-entrypoint-initdb.d/init.sql
      - ./migrations:/migrations  # ← Agregar esta línea para montar el directorio de migraciones
```

**Explicación:**
- `./migrations:/migrations` monta el directorio local `./migrations` en `/migrations` dentro del contenedor
- Esto permite ejecutar migraciones SQL directamente desde el contenedor usando la ruta `/migrations/`
- Después de agregar esta línea, reinicia el contenedor: `docker-compose restart postgres`

### 7. Crear Script de Reentrenamiento Automático

**Archivo nuevo:** `scripts/auto_retrain.py`

```python
#!/usr/bin/env python3
"""
Script para reentrenamiento automático basado en feedback
Ejecutar semanalmente via cron o scheduler
"""
import sys
import os
from pathlib import Path

# Agregar path del proyecto
sys.path.append(str(Path(__file__).parent.parent))

from app.feedback.feedback_service import feedback_service
from datetime import datetime

def main():
    print("🔄 Iniciando proceso de reentrenamiento automático")
    print(f"📅 Fecha: {datetime.now().isoformat()}")

    # 1. Obtener métricas
    print("\n📊 Obteniendo métricas...")
    metrics = feedback_service.get_metrics(days=7)

    avg_rating = metrics['general']['avg_rating']
    total_interactions = metrics['general']['total_interactions']

    print(f"   Total interacciones: {total_interactions}")
    print(f"   Rating promedio: {avg_rating}/5.0")

    # 2. Decidir si reentrenar
    THRESHOLD_RATING = 3.5
    THRESHOLD_INTERACTIONS = 100

    if avg_rating < THRESHOLD_RATING and total_interactions >= THRESHOLD_INTERACTIONS:
        print(f"\n⚠️  Rating bajo ({avg_rating}) - Iniciando reentrenamiento...")

        # 3. Exportar datos
        output_file = f"retraining_data_{datetime.now().strftime('%Y%m%d')}.jsonl"
        count = feedback_service.export_for_retraining(
            output_file=output_file,
            max_rating=3
        )

        print(f"✅ Exportados {count} ejemplos a {output_file}")
        print(f"📝 Siguiente paso: Combinar con dataset original y subir a Google Colab")
        print(f"💡 Comando: cat training_data_complete.jsonl {output_file} > training_v2.jsonl")
        print(f"📖 Ver FASE_1_FINE_TUNING_ACTUALIZADO.md para reentrenamiento")

        return count
    else:
        print(f"\n✅ Sistema funcionando bien (rating: {avg_rating})")
        print("   No es necesario reentrenar")
        return 0

if __name__ == "__main__":
    exported = main()
    sys.exit(0 if exported >= 0 else 1)
```

Hacer ejecutable:

```bash
chmod +x scripts/auto_retrain.py
```

## 📦 Librerías Adicionales

Agregar a `requirements.txt`:

```txt
# Ya existentes (verificar que estén)
psycopg2-binary>=2.9.9
```

No se requieren librerías adicionales, todo usa dependencias existentes.

## 🧪 Plan de Pruebas (Reproducible)

### Prueba 1: Verificar Migración de Base de Datos

**Objetivo:** Confirmar que las tablas de feedback se crearon correctamente

```bash
# 1. Crear directorio de migraciones
mkdir -p migrations

# 2. Copiar el SQL de la sección 1 a migrations/001_create_feedback_table.sql

# 3. Ejecutar migración
docker-compose exec postgres psql -U postgres -d vectordb -f /migrations/001_create_feedback_table.sql

# 4. Verificar tablas creadas
docker-compose exec postgres psql -U postgres -d vectordb -c "\dt"
```

**Output esperado:**

```
              List of relations
 Schema |       Name        | Type  |  Owner
--------+-------------------+-------+----------
 public | analytics_metrics | table | postgres
 public | user_feedback     | table | postgres
(2 rows)
```

### Prueba 2: Test del FeedbackService

**Crear archivo:** `tests/test_feedback_service.py`

```python
import pytest
from app.feedback.feedback_service import feedback_service
import time

def test_save_and_update_interaction():
    """Test completo del ciclo de feedback"""

    # 1. Guardar interacción
    feedback_id = feedback_service.save_interaction(
        session_id="test_session_123",
        user_query="¿Cuántas ventas hay?",
        sql_generated="SELECT COUNT(*) FROM ordenes",
        chart_type="bar",
        chart_config={"type": "bar", "title": "Test"},
        response_time_ms=250,
        error_occurred=False
    )

    assert feedback_id > 0
    print(f"✅ Interacción guardada con ID: {feedback_id}")

    # 2. Actualizar con rating
    success = feedback_service.update_rating(
        feedback_id=feedback_id,
        rating=5,
        feedback_text="Excelente respuesta"
    )

    assert success is True
    print("✅ Rating actualizado correctamente")

    # 3. Verificar que no aparece en low-rated
    low_rated = feedback_service.get_low_rated_queries(min_rating=2, limit=10)
    assert not any(q['id'] == feedback_id for q in low_rated)
    print("✅ No aparece en queries de baja valoración")

def test_metrics():
    """Test de generación de métricas"""

    metrics = feedback_service.get_metrics(days=30)

    assert 'general' in metrics
    assert 'total_interactions' in metrics['general']
    assert 'rating_distribution' in metrics

    print("✅ Métricas generadas correctamente:")
    print(f"   Total interacciones: {metrics['general']['total_interactions']}")
    print(f"   Rating promedio: {metrics['general']['avg_rating']}")

def test_export_retraining():
    """Test de exportación para reentrenamiento"""

    # Primero crear algunos ejemplos con rating bajo
    for i in range(3):
        fid = feedback_service.save_interaction(
            session_id=f"test_{i}",
            user_query=f"Query de prueba {i}",
            chart_type="bar"
        )
        feedback_service.update_rating(fid, rating=2, feedback_text="Mejorable")

    # Exportar
    count = feedback_service.export_for_retraining(
        output_file='test_retraining.jsonl',
        max_rating=3
    )

    assert count >= 3
    print(f"✅ Exportados {count} ejemplos para reentrenamiento")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Ejecutar:**

```bash
# Con flag -s para mostrar los prints de los tests
docker-compose exec app pytest tests/test_feedback_service.py -v -s
```

**Output esperado:**

```
============================= test session starts ==============================
platform linux -- Python 3.11.14, pytest-9.0.1, pluggy-1.6.0 -- /usr/local/bin/python
cachedir: .pytest_cache
rootdir: /app
plugins: timeout-2.4.0, asyncio-1.3.0, anyio-4.12.0, cov-7.0.0, langsmith-0.4.56
collecting ... collected 3 items

tests/test_feedback_service.py::test_save_and_update_interaction ✅ Interacción guardada con ID: 1
✅ Rating actualizado correctamente
✅ No aparece en queries de baja valoración
PASSED
tests/test_feedback_service.py::test_metrics ✅ Métricas generadas correctamente:
   Total interacciones: 15
   Rating promedio: 4.2
PASSED
tests/test_feedback_service.py::test_export_retraining ✅ Exportados 5 ejemplos para reentrenamiento
PASSED

============================== 3 passed in 1.23s ===============================
```

**Nota:** El flag `-s` (o `--capture=no`) es necesario para ver los mensajes de `print()` dentro de los tests. Sin este flag, pytest captura la salida y solo la muestra si hay errores.

### Prueba 3: Test de API Endpoints

```bash
# 1. Hacer una query para obtener un feedback_id
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "¿Cuántas ventas hay?"}' | jq '.feedback_id'

# Supongamos que devuelve: 42

# 2. Enviar rating
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "feedback_id": 42,
    "rating": 5,
    "feedback_text": "Muy buena respuesta"
  }'

# 3. Ver métricas
curl http://localhost:8000/metrics?days=7 | jq
```

**Output esperado del rating:**

```json
{
  "status": "success",
  "message": "Feedback guardado correctamente"
}
```

**Output esperado de métricas:**

```json
{
  "period_days": 7,
  "general": {
    "total_interactions": 45,
    "rated_interactions": 23,
    "avg_rating": 4.35,
    "errors": 2,
    "avg_response_time_ms": 320,
    "p95_response_time_ms": 850
  },
  "rating_distribution": {
    "1": 1,
    "2": 2,
    "3": 5,
    "4": 8,
    "5": 7
  },
  "top_charts": [
    {"chart_type": "bar", "count": 18, "avg_rating": 4.5},
    {"chart_type": "line", "count": 12, "avg_rating": 4.2}
  ],
  "top_errors": []
}
```

### Prueba 4: Test de Frontend con Rating

**Pasos manuales:**

1. Abrir http://localhost:8501
2. Hacer una pregunta: "¿Cuántas ventas hay?"
3. Esperar respuesta
4. Verificar que aparecen botones de rating (⭐ a ⭐⭐⭐⭐⭐)
5. Click en "⭐⭐⭐⭐" (4 estrellas)
6. Verificar mensaje: "¡Gracias por tu valoración de 4 estrellas!"
7. Click en "Ver Métricas" en sidebar
8. Verificar que se muestra dashboard con métricas

**Output esperado en Streamlit:**

```
📈 Métricas de los Últimos 7 Días

Total Interacciones: 45
Rating Promedio: 4.35/5.0
Tiempo Respuesta (avg): 320ms

[Gráfico de barras con distribución de ratings]
[Tabla de gráficos más usados]
```

### Prueba 5: Script de Reentrenamiento Automático

```bash
# Ejecutar script
docker-compose exec app python scripts/auto_retrain.py
```

**Output esperado (caso 1: rating bajo):**

```
🔄 Iniciando proceso de reentrenamiento automático
📅 Fecha: 2025-12-05T10:30:00

📊 Obteniendo métricas...
   Total interacciones: 150
   Rating promedio: 3.2/5.0

⚠️  Rating bajo (3.2) - Iniciando reentrenamiento...
✅ Exportados 45 ejemplos a retraining_data_20251205.jsonl
📝 Siguiente paso: Subir retraining_data_20251205.jsonl a Google Colab para reentrenamiento
📖 Ver FASE_1_FINE_TUNING.md sección 'Reentrenamiento'
```

**Output esperado (caso 2: sistema OK):**

```
🔄 Iniciando proceso de reentrenamiento automático
📅 Fecha: 2025-12-05T10:30:00

📊 Obteniendo métricas...
   Total interacciones: 150
   Rating promedio: 4.5/5.0

✅ Sistema funcionando bien (rating: 4.5)
   No es necesario reentrenar
```

## ✅ Checklist de Completitud

- [ ] Migración SQL ejecutada y tablas creadas
- [ ] `feedback_service.py` creado y funcionando
- [ ] Nodo `track_interaction_node` agregado al workflow
- [ ] Endpoints `/feedback` y `/metrics` funcionando en API
- [ ] Frontend muestra botones de rating después de cada respuesta
- [ ] Dashboard de métricas visible en sidebar
- [ ] Script `auto_retrain.py` ejecutable y funcionando
- [ ] Tests unitarios pasando (3/3 ✅)
- [ ] Tests de API devuelven status 200
- [ ] Datos se guardan correctamente en PostgreSQL
- [ ] Métricas se calculan sin errores

## 💰 Costos

- **PostgreSQL storage**: Incluido en Docker local (0€)
- **CPU para analytics**: Mínimo, queries optimizadas con índices (0€)
- **Todo el sistema de feedback**: 100% GRATIS

## 🎯 Próximos Pasos

Una vez completada esta fase, tienes un sistema completo de mejora continua:

1. **Monitoreo continuo**: Dashboard de métricas en tiempo real
2. **Feedback loop**: Usuarios valoran respuestas automáticamente
3. **Detección de problemas**: Queries con baja valoración se exportan
4. **Reentrenamiento automático**: Script detecta cuándo es necesario mejorar
5. **Ciclo completo**: Nuevos datos → Reentrenamiento → Mejora → Más feedback

### Automatización Opcional (Cron)

Para ejecutar el script semanalmente:

```bash
# Editar crontab
crontab -e

# Agregar (ejecutar todos los lunes a las 2 AM)
0 2 * * 1 cd /path/to/chatbot_analitico && docker-compose exec app python scripts/auto_retrain.py >> logs/retrain.log 2>&1
```

## 🎉 ¡Proyecto Completo!

Has implementado exitosamente:

1. ✅ **Fase 1**: Fine-tuning de modelo especializado
2. ✅ **Fase 2**: Sistema híbrido de 3 capas
3. ✅ **Fase 3**: Gráficos profesionales de nivel enterprise
4. ✅ **Fase 4**: Sistema de feedback y mejora continua

**Tu chatbot ahora tiene:**
- Inteligencia híbrida (reglas + IA + LLM)
- Visualizaciones profesionales
- Sistema de feedback integrado
- Métricas en tiempo real
- Pipeline de mejora continua
- Todo sin costo adicional

---

**¡Disfruta de tu chatbot analítico de nivel empresarial!** 🚀📊
