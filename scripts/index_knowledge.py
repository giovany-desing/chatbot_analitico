"""
Script para indexar todos los ejemplos SQL en el vectorstore.
"""

import json
import logging
from pathlib import Path
import sys

# Añadir parent directory al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.vectorstore import vectorstore
from app.tools.sql_tool import mysql_tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def index_sql_examples():
    """Indexa ejemplos SQL desde el archivo JSON"""

    # Cargar ejemplos
    json_path = Path(__file__).parent.parent / "data" / "sql_examples.json"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sql_examples = data.get('sql_examples', [])

    logger.info(f"Indexing {len(sql_examples)} SQL examples...")

    for example in sql_examples:
        # Combinar pregunta y SQL en un texto
        text = f"{example['question']} {example['sql']}"

        # Indexar
        vectorstore.index_example(
            text=text,
            metadata={
                "question": example['question'],
                "sql": example['sql'],
                "description": example['description'],
                "type": "sql_example"
            }
        )

    logger.info("✓ SQL examples indexed")


def index_schema_info():
    """Indexa información del schema"""

    # Usar get_schema_info() directamente de la conexión MySQL, no del tool
    # porque el tool retorna un string, pero necesitamos el diccionario
    from app.db.connections import get_mysql
    mysql = get_mysql()
    schema = mysql.get_schema_info()

    logger.info(f"Indexing schema for {len(schema)} tables...")

    for table_name, columns in schema.items():
        # Crear descripción del schema
        col_descriptions = []
        for col in columns:
            col_str = f"{col['column']} ({col['type']})"
            if col['key'] == 'PRI':
                col_str += " [PRIMARY KEY]"
            col_descriptions.append(col_str)

        text = f"Tabla {table_name}: " + ", ".join(col_descriptions)

        # Indexar
        vectorstore.index_example(
            text=text,
            metadata={
                "table": table_name,
                "columns": [col['column'] for col in columns],
                "description": f"Schema de la tabla {table_name}",
                "type": "schema"
            }
        )

    logger.info("✓ Schema indexed")


def index_kpi_definitions():
    """Indexa definiciones de KPIs"""

    json_path = Path(__file__).parent.parent / "data" / "sql_examples.json"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    kpi_defs = data.get('kpi_definitions', {})

    logger.info(f"Indexing {len(kpi_defs)} KPI definitions...")

    for kpi_key, kpi_info in kpi_defs.items():
        text = f"{kpi_info['name']} {kpi_info['description']} {kpi_info['formula']}"

        vectorstore.index_example(
            text=text,
            metadata={
                "kpi_key": kpi_key,
                "name": kpi_info['name'],
                "formula": kpi_info['formula'],
                "sql": kpi_info['sql'],
                "description": kpi_info['description'],
                "type": "kpi"
            }
        )

    logger.info("✓ KPI definitions indexed")


def main():
    """Indexa todo el conocimiento"""

    print("=" * 60)
    print("INDEXING KNOWLEDGE BASE")
    print("=" * 60)

    # Limpiar vectorstore existente
    logger.info("Clearing existing embeddings...")
    vectorstore.clear_all()

    # Indexar todo
    index_sql_examples()
    index_schema_info()
    index_kpi_definitions()

    print("\n" + "=" * 60)
    print("✓ INDEXING COMPLETE")
    print("=" * 60)

    # Test de búsqueda
    print("\nTesting search:")
    results = vectorstore.search_similar("ventas totales", top_k=3)

    for result in results:
        print(f"\nSimilarity: {result['similarity']:.3f}")
        print(f"Type: {result['metadata']['type']}")
        print(f"Text: {result['text'][:100]}...")


if __name__ == "__main__":
    main()