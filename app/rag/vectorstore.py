"""
Sistema RAG con pgvector para mejorar generación de SQL.
"""

from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import json
from sqlalchemy import text

from app.db.connections import get_postgres
from app.llm.models import generate_single_embedding, generate_embeddings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Gestor de vectorstore para RAG.
    Indexa y recupera ejemplos SQL semánticamente.
    """

    def __init__(self):
        self.postgres = get_postgres()

    def index_example(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> int:
        """
        Indexa un ejemplo en el vectorstore.
        
        Args:
            text: Texto del ejemplo (pregunta + SQL)
            metadata: Metadata (question, sql, description, etc.)
        
        Returns:
            ID del embedding insertado
        """
        try:
            # Generar embedding
            embedding = generate_single_embedding(text)

            # Insertar en BD
            embedding_id = self.postgres.insert_embedding(
                text=text,
                vector=embedding,
                metadata=metadata
            )

            logger.info(f"Indexed example: {metadata.get('question', 'N/A')}")

            return embedding_id

        except Exception as e:
            logger.error(f"Error indexing example: {e}")
            raise

    def search_similar(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Busca ejemplos similares a la query.
        
        Args:
            query: Pregunta del usuario
            top_k: Número de resultados
            threshold: Umbral de similitud (0-1)
        
        Returns:
            Lista de ejemplos similares con metadata
        """
        try:
            # Generar embedding de la query
            query_embedding = generate_single_embedding(query)

            # Buscar en BD
            results = self.postgres.similarity_search(
                query_vector=query_embedding,
                top_k=top_k,
                threshold=threshold
            )

            logger.info(f"Found {len(results)} similar examples for: {query[:50]}")

            return results

        except Exception as e:
            logger.error(f"Error searching similar examples: {e}")
            return []

    def get_relevant_examples(
        self,
        user_query: str,
        top_k: int = 3
    ) -> str:
        """
        Obtiene ejemplos relevantes formateados para inyectar en prompts.
        
        Args:
            user_query: Pregunta del usuario
            top_k: Número de ejemplos
        
        Returns:
            String con ejemplos formateados
        """
        results = self.search_similar(user_query, top_k=top_k)

        if not results:
            return "No hay ejemplos similares disponibles."

        examples = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            similarity = result['similarity']

            example = f"""Ejemplo {i} (similitud: {similarity:.2f}):
Pregunta: {metadata.get('question', 'N/A')}
SQL: {metadata.get('sql', 'N/A')}
Descripción: {metadata.get('description', 'N/A')}
"""
            examples.append(example)

        return "\n".join(examples)

    def clear_all(self):
        """Elimina todos los embeddings (útil para re-indexar)"""
        try:
            with self.postgres.get_session() as session:
                session.execute(text("DELETE FROM embeddings"))
                session.commit()
            logger.info("✓ Cleared all embeddings")
        except Exception as e:
            logger.error(f"Error clearing embeddings: {e}")
            raise


# Instancia global
vectorstore = VectorStore()


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testing VectorStore ===\n")

    # Test 1: Index example
    print("1. Indexing example:")
    embedding_id = vectorstore.index_example(
        text="¿Cuántas ventas hay? SELECT COUNT(*) FROM ventas",
        metadata={
            "question": "¿Cuántas ventas hay?",
            "sql": "SELECT COUNT(*) FROM ventas",
            "description": "Conteo total de ventas",
            "type": "count"
        }
    )
    print(f"   Indexed with ID: {embedding_id}\n")

    # Test 2: Search similar
    print("2. Searching similar:")
    results = vectorstore.search_similar("cuántas transacciones tenemos")

    for result in results:
        print(f"   Similarity: {result['similarity']:.3f}")
        print(f"   Question: {result['metadata']['question']}")
        print(f"   SQL: {result['metadata']['sql']}\n")

    # Test 3: Get formatted examples
    print("3. Get formatted examples:")
    formatted = vectorstore.get_relevant_examples("número de ventas")
    print(formatted)