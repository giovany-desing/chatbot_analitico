-- Script de inicialización de PostgreSQL + pgvector
-- Habilita la extensión pgvector para almacenar embeddings

-- Crear extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla para almacenar embeddings de conocimiento
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    vector vector(768),  -- Dimension de sentence-transformers
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Índice para búsqueda rápida de similitud coseno
CREATE INDEX IF NOT EXISTS embeddings_vector_idx
ON embeddings
USING ivfflat (vector vector_cosine_ops)
WITH (lists = 100);

-- Índice para búsqueda por metadata
CREATE INDEX IF NOT EXISTS embeddings_metadata_idx
ON embeddings
USING GIN (metadata);

-- Función helper para búsqueda de similitud
CREATE OR REPLACE FUNCTION search_similar_embeddings(
    query_vector vector(768),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id int,
    text text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        embeddings.id,
        embeddings.text,
        embeddings.metadata,
        1 - (embeddings.vector <=> query_vector) AS similarity
    FROM embeddings
    WHERE 1 - (embeddings.vector <=> query_vector) > match_threshold
    ORDER BY embeddings.vector <=> query_vector
    LIMIT match_count;
END;
$$;

