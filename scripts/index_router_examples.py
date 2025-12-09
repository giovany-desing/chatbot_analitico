"""
Script para indexar ejemplos de clasificación de intents en PostgreSQL/pgvector.
Esto permite al router usar RAG para few-shot classification.
"""
import json
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.connections import get_postgres
from app.llm.models import get_embedding_model
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_router_examples_table():
    """Crea tabla específica para ejemplos del router si no existe."""
    postgres = get_postgres()

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS router_examples (
        id SERIAL PRIMARY KEY,
        query TEXT NOT NULL,
        intent VARCHAR(50) NOT NULL,
        reasoning TEXT,
        embedding vector(768),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Índice para búsqueda vectorial eficiente
    CREATE INDEX IF NOT EXISTS router_examples_embedding_idx
    ON router_examples USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

    -- Índice para búsqueda por intent
    CREATE INDEX IF NOT EXISTS router_examples_intent_idx
    ON router_examples(intent);
    """

    try:
        with postgres.get_session() as session:
            # Ejecutar cada statement por separado
            statements = create_table_sql.strip().split(';')
            for statement in statements:
                statement = statement.strip()
                if statement:
                    session.execute(text(statement))
            session.commit()
        logger.info("✅ Tabla router_examples creada/verificada")
    except Exception as e:
        logger.error(f"❌ Error creando tabla: {e}")
        raise


def index_router_examples(examples_file: str):
    """
    Indexa los ejemplos de clasificación en PostgreSQL.

    Args:
        examples_file: Path al archivo JSON con ejemplos
    """
    # Cargar ejemplos
    with open(examples_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    logger.info(f"📚 Cargados {len(examples)} ejemplos")

    # Inicializar modelo de embeddings
    embedding_model = get_embedding_model()
    postgres = get_postgres()

    # Limpiar tabla anterior (opcional - comentar si quieres mantener datos)
    try:
        with postgres.get_session() as session:
            session.execute(text("DELETE FROM router_examples"))
            session.commit()
        logger.info("🧹 Tabla router_examples limpiada")
    except Exception as e:
        logger.warning(f"⚠️  No se pudo limpiar tabla: {e}")

    # Indexar cada ejemplo
    indexed_count = 0
    for idx, example in enumerate(examples, 1):
        try:
            query = example['query']
            intent = example['intent']
            reasoning = example.get('reasoning', '')

            # Generar embedding
            embedding = embedding_model.encode(query).tolist()

            # Convertir embedding a formato pgvector: lista entre corchetes como string
            embedding_str = "[" + ",".join(str(float(v)) for v in embedding) + "]"

            # Insertar en BD usando conexión raw de psycopg2
            insert_sql = """
            INSERT INTO router_examples (query, intent, reasoning, embedding)
            VALUES (:query, :intent, :reasoning, :embedding::vector)
            """

            with postgres.get_session() as session:
                # Obtener conexión raw de psycopg2
                conn = session.connection()
                cursor = conn.connection.cursor()
                cursor.execute(
                    "INSERT INTO router_examples (query, intent, reasoning, embedding) VALUES (%s, %s, %s, %s::vector)",
                    (query, intent, reasoning, embedding_str)
                )
                session.commit()
                cursor.close()

            indexed_count += 1
            logger.info(f"✅ [{idx}/{len(examples)}] Indexado: '{query}' → {intent}")

        except Exception as e:
            logger.error(f"❌ Error indexando ejemplo {idx}: {e}")
            continue

    logger.info(f"\n🎉 Indexación completada: {indexed_count}/{len(examples)} ejemplos")


def test_router_rag(test_query: str, top_k: int = 3):
    """
    Prueba la búsqueda RAG para el router.

    Args:
        test_query: Query de prueba
        top_k: Número de ejemplos similares a recuperar
    """
    logger.info(f"\n🔍 Probando RAG con query: '{test_query}'")

    embedding_model = get_embedding_model()
    postgres = get_postgres()

    # Generar embedding del query
    query_embedding = embedding_model.encode(test_query).tolist()
    query_embedding_str = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"

    # Buscar ejemplos similares
    search_sql = """
    SELECT
        query,
        intent,
        reasoning,
        1 - (embedding <=> %s::vector) as similarity
    FROM router_examples
    ORDER BY embedding <=> %s::vector
    LIMIT %s
    """

    try:
        with postgres.get_session() as session:
            # Obtener conexión raw de psycopg2
            conn = session.connection()
            cursor = conn.connection.cursor()
            cursor.execute(search_sql, (query_embedding_str, query_embedding_str, top_k))
            results = cursor.fetchall()
            cursor.close()

        logger.info(f"\n📋 Top {top_k} ejemplos similares:\n")
        for i, (q, intent, reasoning, similarity) in enumerate(results, 1):
            logger.info(f"{i}. [{intent}] (sim: {similarity:.3f})")
            logger.info(f"   Query: {q}")
            logger.info(f"   Razón: {reasoning}\n")

        # Votar por intent más frecuente
        intent_votes = {}
        for _, intent, _, _ in results:
            intent_votes[intent] = intent_votes.get(intent, 0) + 1

        predicted_intent = max(intent_votes, key=intent_votes.get)
        logger.info(f"🎯 Intent predicho: {predicted_intent} (votos: {intent_votes})")

        return predicted_intent

    except Exception as e:
        logger.error(f"❌ Error en búsqueda RAG: {e}")
        return None


if __name__ == "__main__":
    logger.info("🚀 Iniciando indexación de ejemplos del router\n")

    # Paso 1: Crear tabla
    create_router_examples_table()

    # Paso 2: Indexar ejemplos
    examples_path = Path(__file__).parent.parent / "data" / "router_examples.json"
    index_router_examples(str(examples_path))

    # Paso 3: Probar con queries ambiguas
    logger.info("\n" + "="*60)
    logger.info("🧪 PRUEBAS CON QUERIES AMBIGUAS")
    logger.info("="*60)

    test_queries = [
        "Dame estadísticas",  # Ambiguo: ¿SQL, KPI o hybrid?
        "¿Cómo van las ventas?",  # Ambiguo: ¿SQL o hybrid?
        "Muestra el rendimiento",  # Ambiguo: ¿KPI o viz?
        "Top productos",  # Ambiguo: ¿SQL o viz?
    ]

    for test_q in test_queries:
        test_router_rag(test_q, top_k=3)
        logger.info("\n" + "-"*60 + "\n")
