"""
Gestión de conexiones a bases de datos.
Implementa connection pooling, retry logic y health checks.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz del proyecto al PYTHONPATH si se ejecuta directamente
if __name__ == "__main__":
    # Obtener el directorio raíz del proyecto (2 niveles arriba de este archivo)
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text as sql_text, pool, bindparam
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import redis
from redis.exceptions import RedisError
import logging
import json
from typing import Optional, Dict, Any
from functools import lru_cache
import time

from app.config import settings

# Configurar logging
logger = logging.getLogger(__name__)


# ============ MySQL Connection ============

class MySQLConnection:
    """
    Gestor de conexión a MySQL con connection pooling.
    Usa SQLAlchemy para gestionar el pool de conexiones.
    """

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize()

    def _initialize(self):
        """Inicializa el engine y sessionmaker"""
        try:
            # Log de conexión (sin password)
            mysql_url_safe = settings.MYSQL_URL.split('@')[1] if '@' in settings.MYSQL_URL else settings.MYSQL_HOST
            logger.info(f"Connecting to MySQL: {settings.MYSQL_USER}@{mysql_url_safe}")
            
            self.engine = create_engine(
                settings.MYSQL_URL,
                poolclass=pool.QueuePool,
                pool_size=10,              # Conexiones permanentes
                max_overflow=20,           # Conexiones extra bajo demanda
                pool_timeout=60,           # Timeout para obtener conexión (aumentado para RDS)
                pool_recycle=3600,         # Reciclar conexiones cada hora
                pool_pre_ping=True,        # Verificar conexión antes de usar
                echo=settings.DEBUG,       # Log de queries en debug mode
                connect_args={
                    "connect_timeout": 30,  # Timeout de conexión en segundos
                    "autocommit": False
                }
            )

            # Crear sessionmaker
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            # Test de conexión con retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if self.health_check():
                        logger.info("✅ MySQL connection pool initialized")
                        return
                    else:
                        raise Exception("Health check failed")
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"MySQL connection attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        raise

        except SQLAlchemyError as e:
            logger.error(f"❌ Error initializing MySQL: {e}")
            logger.error(f"   Host: {settings.MYSQL_HOST}:{settings.MYSQL_PORT}")
            logger.error(f"   Database: {settings.MYSQL_DATABASE}")
            logger.error(f"   User: {settings.MYSQL_USER}")
            logger.error("   Troubleshooting:")
            logger.error("   1. Verify RDS Security Group allows connections from your IP/container")
            logger.error("   2. Check if RDS is in a VPC and network configuration is correct")
            logger.error("   3. Verify credentials in .env file")
            logger.error("   4. Check RDS endpoint is correct and accessible")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error initializing MySQL: {e}")
            raise

    def get_session(self) -> Session:
        """
        Obtiene una sesión de SQLAlchemy.
        Usar con context manager:
        
        with mysql.get_session() as session:
            result = session.execute(sql_text("SELECT * FROM ventas"))
        """
        return self.SessionLocal()

    def execute_query(self, query: str, params: Optional[Dict] = None) -> list:
        """
        Ejecuta una query SQL y retorna resultados como lista de dicts.
        
        Args:
            query: Query SQL (puede usar :param para parametrización)
            params: Diccionario de parámetros
        
        Returns:
            Lista de diccionarios con los resultados
        """
        try:
            with self.get_session() as session:
                result = session.execute(sql_text(query), params or {})

                # Convertir a lista de dicts
                columns = result.keys()
                rows = result.fetchall()

                return [dict(zip(columns, row)) for row in rows]

        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {e}")
            raise

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Obtiene información del schema de la BD.
        Útil para que el LLM genere SQL correctamente.
        
        Returns:
            Dict con información de tablas y columnas
        """
        schema_query = """
        SELECT 
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :db_name
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """

        results = self.execute_query(
            schema_query,
            {"db_name": settings.MYSQL_DATABASE}
        )

        # Agrupar por tabla
        schema = {}
        for row in results:
            table = row['TABLE_NAME']
            if table not in schema:
                schema[table] = []
            schema[table].append({
                'column': row['COLUMN_NAME'],
                'type': row['DATA_TYPE'],
                'nullable': row['IS_NULLABLE'] == 'YES',
                'key': row['COLUMN_KEY']
            })

        return schema

    def health_check(self) -> bool:
        """Verifica que la conexión funcione"""
        try:
            # Usar timeout más corto para health check
            result = self.execute_query("SELECT 1 as test")
            return result[0]['test'] == 1
        except Exception as e:
            logger.error(f"MySQL health check failed: {e}")
            logger.error(f"   This may indicate RDS is not accessible. Check:")
            logger.error(f"   1. RDS Security Group allows connections")
            logger.error(f"   2. RDS is publicly accessible")
            logger.error(f"   3. Network routing is correct")
            return False


# ============ Redis Connection ============

class RedisConnection:
    """
    Gestor de conexión a Redis con reconnect automático.
    """

    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self._initialize()

    def _initialize(self):
        """Inicializa el cliente de Redis"""
        try:
            self.client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,      # Retorna strings en vez de bytes
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test de conexión
            self.health_check()
            logger.info("✅ Redis connection initialized")

        except RedisError as e:
            logger.error(f"❌ Error initializing Redis: {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        """Obtiene un valor del cache"""
        try:
            return self.client.get(key)
        except RedisError as e:
            logger.error(f"Redis GET error: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Guarda un valor en el cache.
        
        Args:
            key: Clave
            value: Valor (string o JSON serializado)
            ttl: Time to live en segundos (default: settings.REDIS_TTL)
        """
        try:
            ttl = ttl or settings.REDIS_TTL
            return self.client.setex(key, ttl, value)
        except RedisError as e:
            logger.error(f"Redis SET error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Elimina una clave del cache"""
        try:
            return bool(self.client.delete(key))
        except RedisError as e:
            logger.error(f"Redis DELETE error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Verifica si una clave existe"""
        try:
            return bool(self.client.exists(key))
        except RedisError as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False

    def health_check(self) -> bool:
        """Verifica que Redis funcione"""
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


# ============ PostgreSQL + pgvector Connection ============

class PostgresConnection:
    """
    Gestor de conexión a PostgreSQL con soporte para pgvector.
    Usado para almacenar y buscar embeddings (RAG).
    """

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize()

    def _initialize(self):
        """Inicializa el engine"""
        try:
            self.engine = create_engine(
                settings.POSTGRES_URL,
                poolclass=pool.QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=settings.DEBUG
            )

            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            # Test de conexión
            self.health_check()
            logger.info("✅ PostgreSQL connection pool initialized")

        except SQLAlchemyError as e:
            logger.error(f"❌ Error initializing PostgreSQL: {e}")
            raise

    def get_session(self) -> Session:
        """Obtiene una sesión"""
        return self.SessionLocal()

    def insert_embedding(
        self, 
        text: str, 
        vector: list, 
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Inserta un embedding en la BD.
        
        Args:
            text: Texto original
            vector: Lista de floats (embedding)
            metadata: Metadata adicional (JSON)
        
        Returns:
            ID del embedding insertado
        """
        # Convertir vector a formato pgvector: lista entre corchetes como string
        vector_str = "[" + ",".join(str(float(v)) for v in vector) + "]"
        metadata_json = json.dumps(metadata or {})
        
        # Usar parámetros posicionales con psycopg2 para evitar conflictos con : en el texto
        query = """
        INSERT INTO embeddings (text, vector, metadata)
        VALUES (%s, %s::vector, %s::jsonb)
        RETURNING id
        """

        try:
            with self.get_session() as session:
                # Obtener la conexión raw de psycopg2
                conn = session.connection()
                # Usar execute directamente con parámetros posicionales
                cursor = conn.connection.cursor()
                cursor.execute(query, (text, vector_str, metadata_json))
                result = cursor.fetchone()
                session.commit()
                cursor.close()
                return result[0] if result else None

        except SQLAlchemyError as e:
            logger.error(f"Error inserting embedding: {e}")
            raise

    def similarity_search(
        self, 
        query_vector: list, 
        top_k: int = 5,
        threshold: float = 0.7
    ) -> list:
        """
        Búsqueda de similitud coseno.
        
        Args:
            query_vector: Vector de la query
            top_k: Número de resultados
            threshold: Umbral de similitud (0-1)
        
        Returns:
            Lista de dicts con text, metadata y similarity
        """
        query = """
        SELECT 
            id,
            text,
            metadata,
            1 - (vector <=> :query_vector::vector) AS similarity
        FROM embeddings
        WHERE 1 - (vector <=> :query_vector::vector) > :threshold
        ORDER BY vector <=> :query_vector::vector
        LIMIT :top_k
        """

        try:
            with self.get_session() as session:
                result = session.execute(
                    sql_text(query),
                    {
                        "query_vector": str(query_vector),
                        "threshold": threshold,
                        "top_k": top_k
                    }
                )

                return [
                    {
                        "id": row.id,
                        "text": row.text,
                        "metadata": row.metadata,
                        "similarity": float(row.similarity)
                    }
                    for row in result
                ]

        except SQLAlchemyError as e:
            logger.error(f"Error in similarity search: {e}")
            raise

    def health_check(self) -> bool:
        """Verifica que PostgreSQL + pgvector funcionen"""
        try:
            with self.get_session() as session:
                # Verificar que pgvector esté habilitado
                result = session.execute(
                    sql_text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                )
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False


# ============ Singleton Instances ============

@lru_cache()
def get_mysql() -> MySQLConnection:
    """Retorna instancia singleton de MySQL"""
    return MySQLConnection()

@lru_cache()
def get_redis() -> RedisConnection:
    """Retorna instancia singleton de Redis"""
    return RedisConnection()

@lru_cache()
def get_postgres() -> PostgresConnection:
    """Retorna instancia singleton de PostgreSQL"""
    return PostgresConnection()


# ============ Health Check General ============

def check_all_connections() -> Dict[str, bool]:
    """
    Verifica el estado de todas las conexiones.
    Útil para el endpoint /health de la API.
    
    Returns:
        Dict con el estado de cada BD
    """
    return {
        "mysql": get_mysql().health_check(),
        "redis": get_redis().health_check(),
        "postgres": get_postgres().health_check()
    }


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testing Connections ===\n")

    # Test MySQL
    print("1. MySQL:")
    mysql = get_mysql()
    print(f"   Health: {mysql.health_check()}")
    result = mysql.execute_query("SELECT COUNT(*) as total FROM ventas_preventivas")
    print(f"   Total ventas: {result[0]['total']}")

    # Test Redis
    print("\n2. Redis:")
    redis_conn = get_redis()
    print(f"   Health: {redis_conn.health_check()}")
    redis_conn.set("test_key", "test_value", ttl=60)
    print(f"   Test value: {redis_conn.get('test_key')}")

    # Test PostgreSQL
    print("\n3. PostgreSQL:")
    postgres = get_postgres()
    print(f"   Health: {postgres.health_check()}")

    # Test general
    print("\n4. All connections:")
    print(check_all_connections())
