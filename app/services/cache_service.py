"""
Servicio de cache para queries SQL.
"""

import hashlib
import json
import logging
from typing import Optional, List, Dict, Any

from app.db.connections import get_redis
from app.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """
    Servicio de cache con Redis.
    Cachea resultados de queries SQL.
    """

    def __init__(self):
        self.redis = get_redis()
        self.prefix = "chatbot:query:"
        self.ttl = settings.REDIS_TTL

    def _make_key(self, query: str) -> str:
        """
        Genera key de cache desde la query.
        Usa hash MD5 para keys consistentes.
        
        Args:
            query: Query SQL
        
        Returns:
            Key de Redis
        """
        # Normalizar query (lowercase, strip whitespace)
        normalized = " ".join(query.lower().split())

        # Hash MD5
        query_hash = hashlib.md5(normalized.encode()).hexdigest()

        return f"{self.prefix}{query_hash}"

    def get(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Obtiene resultados cacheados.
        
        Args:
            query: Query SQL
        
        Returns:
            Lista de resultados o None si no existe
        """
        try:
            key = self._make_key(query)
            cached = self.redis.get(key)

            if cached:
                logger.info(f"✓ Cache HIT for query: {query[:50]}...")
                return json.loads(cached)

            logger.info(f"✗ Cache MISS for query: {query[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    def set(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Guarda resultados en cache.
        
        Args:
            query: Query SQL
            results: Resultados a cachear
            ttl: Time to live (default: settings.REDIS_TTL)
        
        Returns:
            True si se guardó correctamente
        """
        try:
            key = self._make_key(query)
            value = json.dumps(results, default=str)  # default=str para fechas

            ttl = ttl or self.ttl
            success = self.redis.set(key, value, ttl=ttl)

            if success:
                logger.info(f"✓ Cached query results (TTL: {ttl}s): {query[:50]}...")

            return success

        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    def invalidate(self, query: str) -> bool:
        """
        Invalida cache de una query específica.
        
        Args:
            query: Query SQL
        
        Returns:
            True si se eliminó
        """
        try:
            key = self._make_key(query)
            deleted = self.redis.delete(key)

            if deleted:
                logger.info(f"✓ Cache invalidated: {query[:50]}...")

            return deleted

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False

    def clear_all(self) -> int:
        """
        Limpia todo el cache de queries.
        
        Returns:
            Número de keys eliminadas
        """
        try:
            # Buscar todas las keys con el prefix
            pattern = f"{self.prefix}*"
            keys = self.redis.client.keys(pattern)

            if keys:
                deleted = self.redis.client.delete(*keys)
                logger.info(f"✓ Cleared {deleted} cached queries")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0


# Instancia global
cache_service = CacheService()


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testing CacheService ===\n")

    # Test 1: Set and get
    print("1. Set and get:")
    query = "SELECT COUNT(*) FROM ventas"
    results = [{"count": 42}]

    cache_service.set(query, results)
    cached = cache_service.get(query)

    print(f"   Original: {results}")
    print(f"   Cached: {cached}")
    print(f"   Match: {results == cached}\n")

    # Test 2: Cache miss
    print("2. Cache miss:")
    cached = cache_service.get("SELECT * FROM productos")
    print(f"   Result: {cached}\n")

    # Test 3: Invalidate
    print("3. Invalidate:")
    cache_service.invalidate(query)
    cached = cache_service.get(query)
    print(f"   After invalidation: {cached}\n")

    # Test 4: Clear all
    print("4. Clear all:")
    cache_service.set("query1", [{"a": 1}])
    cache_service.set("query2", [{"b": 2}])
    cleared = cache_service.clear_all()
    print(f"   Cleared: {cleared} keys")