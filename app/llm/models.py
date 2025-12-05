"""
Configuración de LLMs y embeddings.
Gestiona modelos, prompts, retry logic y fallbacks.
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

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sentence_transformers import SentenceTransformer
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from typing import Optional, List
import logging
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)


# ============ LLM Configuration ============

@lru_cache()
def get_llama_model() -> ChatGroq:
    """
    Retorna instancia de Llama 3.3 70B via Groq.
    Usa cache para reutilizar la misma instancia.
    
    Configuración:
    - Temperature: 0.1 (preciso para SQL y análisis)
    - Max tokens: 2000 (suficiente para queries complejas)
    - Timeout: 30s
    
    Returns:
        ChatGroq configurado
    """
    try:
        llm = ChatGroq(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
            groq_api_key=settings.GROQ_API_KEY,
            max_retries=3
        )

        logger.info(f"✅ Llama model initialized: {settings.LLM_MODEL}")
        return llm

    except Exception as e:
        logger.error(f"❌ Error initializing Llama model: {e}")
        raise


@lru_cache()
def get_fallback_model() -> Optional[ChatOpenAI]:
    """
    Retorna GPT-4 como fallback si Groq falla.
    Solo se inicializa si OPENAI_API_KEY está configurada.
    
    Returns:
        ChatOpenAI o None si no hay API key
    """
    if not settings.OPENAI_API_KEY:
        logger.warning("⚠️ No OPENAI_API_KEY configured, no fallback available")
        return None

    try:
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            timeout=settings.LLM_TIMEOUT,
            openai_api_key=settings.OPENAI_API_KEY
        )

        logger.info("✅ GPT-4 fallback model initialized")
        return llm

    except Exception as e:
        logger.error(f"❌ Error initializing fallback model: {e}")
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
def invoke_llm_with_retry(llm, messages: list) -> str:
    """
    Invoca el LLM con retry automático.
    Si falla 3 veces, intenta con fallback.
    
    Args:
        llm: Instancia del LLM
        messages: Lista de mensajes
    
    Returns:
        Respuesta del LLM
    """
    try:
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")

        # Intentar fallback
        fallback = get_fallback_model()
        if fallback:
            logger.info("🔄 Using GPT-4 fallback")
            response = fallback.invoke(messages)
            return response.content

        raise


# ============ Embeddings Configuration ============

@lru_cache()
def get_embedding_model() -> SentenceTransformer:
    """
    Retorna modelo de embeddings multilingüe.
    
    Modelo: paraphrase-multilingual-mpnet-base-v2
    - Dimensión: 768
    - Idiomas: 50+ incluyendo español
    - Optimizado para similitud semántica
    
    Returns:
        SentenceTransformer
    """
    try:
        model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            device='cpu'  # Cambiar a 'cuda' si tienes GPU
        )

        logger.info("✅ Embedding model initialized (768 dim)")
        return model

    except Exception as e:
        logger.error(f"❌ Error initializing embedding model: {e}")
        raise


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Genera embeddings para una lista de textos.
    
    Args:
        texts: Lista de strings
    
    Returns:
        Lista de embeddings (cada uno es una lista de 768 floats)
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def generate_single_embedding(text: str) -> List[float]:
    """
    Genera embedding para un solo texto.
    
    Args:
        text: String a embedder
    
    Returns:
        Lista de 768 floats
    """
    return generate_embeddings([text])[0]


# ============ Prompts Templates ============

def get_router_prompt() -> ChatPromptTemplate:
    """
    Prompt para clasificar la intención del usuario.
    
    Clasifica en:
    - sql: Necesita datos de la BD
    - kpi: Necesita calcular métricas
    - viz: Necesita gráficas
    - general: Pregunta general sin datos
    - hybrid: Combinación de las anteriores
    
    Returns:
        ChatPromptTemplate
    """
    template = """Eres un clasificador de intenciones para un chatbot analítico de ventas.

Tu tarea es clasificar la pregunta del usuario en UNA de estas categorías:

1. **sql**: El usuario pide datos específicos de la base de datos
   Ejemplos: "¿Cuántas ventas hubo en enero?", "Muéstrame los productos más vendidos"

2. **kpi**: El usuario pide calcular métricas o KPIs
   Ejemplos: "¿Cuál es el revenue total?", "Calcula el ticket promedio"

3. **viz**: El usuario pide una visualización o gráfica
   Ejemplos: "Muéstrame una gráfica de ventas", "Grafícame los productos"

4. **general**: Pregunta general sin necesidad de datos
   Ejemplos: "¿Qué puedes hacer?", "Hola", "Explícame qué es un KPI"

5. **hybrid**: Necesita combinar SQL + KPIs + Visualización
   Ejemplos: "Dame las ventas del último mes y grafícalas", "Calcula el revenue y muéstralo en gráfica"

Pregunta del usuario: {user_query}

Responde ÚNICAMENTE con una palabra: sql, kpi, viz, general, o hybrid
No des explicaciones, solo la categoría."""

    return ChatPromptTemplate.from_template(template)


def get_sql_prompt() -> ChatPromptTemplate:
    """
    Prompt para generar queries SQL desde lenguaje natural.
    
    Returns:
        ChatPromptTemplate
    """
    template = """Eres un experto en SQL especializado en bases de datos de ventas.

**ESQUEMA DE LA BASE DE DATOS:**

{schema_info}

**EJEMPLOS DE QUERIES:**

{examples}

**REGLAS IMPORTANTES:**

1. Genera SOLO la query SQL, sin explicaciones
2. Usa sintaxis MySQL 8.0
3. NO uses DELETE, UPDATE, DROP, ALTER, CREATE
4. Usa ONLY SELECT queries
5. Parametriza fechas en formato 'YYYY-MM-DD'
6. Usa JOINs cuando sea necesario
7. Agrega LIMIT 100 si no especifican límite
8. Comenta queries complejas con -- comentarios

**PREGUNTA DEL USUARIO:**

{user_query}

**QUERY SQL:**"""

    return ChatPromptTemplate.from_template(template)


def get_kpi_prompt() -> ChatPromptTemplate:
    """
    Prompt para calcular KPIs desde resultados SQL.
    
    Returns:
        ChatPromptTemplate
    """
    template = """Eres un analista de datos especializado en KPIs de ventas.

**DATOS DISPONIBLES:**

{sql_results}

**KPIs COMUNES:**

- Revenue Total: SUM(total)
- Ticket Promedio: SUM(total) / COUNT(DISTINCT venta_id)
- Unidades Vendidas: SUM(cantidad)
- Productos Únicos: COUNT(DISTINCT producto_id)
- Clientes Únicos: COUNT(DISTINCT cliente_id)
- Crecimiento: ((periodo_actual - periodo_anterior) / periodo_anterior) * 100

**PREGUNTA DEL USUARIO:**

{user_query}

**INSTRUCCIONES:**

1. Calcula los KPIs solicitados
2. Explica brevemente qué significa cada KPI
3. Incluye el valor numérico formateado
4. Si hay comparaciones, calcula el % de cambio

**RESPUESTA:**"""

    return ChatPromptTemplate.from_template(template)


def get_viz_prompt() -> ChatPromptTemplate:
    """
    Prompt para decidir qué tipo de gráfica crear.
    
    Returns:
        ChatPromptTemplate
    """
    template = """Eres un experto en visualización de datos.

**DATOS DISPONIBLES:**

{sql_results}

**TIPOS DE GRÁFICAS DISPONIBLES:**

1. **bar**: Comparar categorías (productos, ciudades, etc.)
2. **line**: Tendencias en el tiempo (ventas por mes, etc.)
3. **pie**: Proporciones (% de ventas por categoría)
4. **scatter**: Relaciones entre variables (precio vs cantidad)
5. **histogram**: Distribución de valores (distribución de precios)

**PREGUNTA DEL USUARIO:**

{user_query}

**INSTRUCCIONES:**

Analiza los datos y decide:
1. Tipo de gráfica más apropiado
2. Columna para eje X
3. Columna para eje Y
4. Título de la gráfica
5. Etiquetas de ejes

Responde en formato JSON:

{{
    "chart_type": "bar|line|pie|scatter|histogram",
    "x_column": "nombre_columna",
    "y_column": "nombre_columna",
    "title": "Título descriptivo",
    "x_label": "Etiqueta eje X",
    "y_label": "Etiqueta eje Y"
}}

**RESPUESTA JSON:**"""

    return ChatPromptTemplate.from_template(template)


def get_general_prompt() -> ChatPromptTemplate:
    """
    Prompt para respuestas generales sin acceso a datos.
    
    Returns:
        ChatPromptTemplate
    """
    template = """Eres un asistente virtual especializado en análisis de ventas.

**TUS CAPACIDADES:**

- Responder preguntas sobre ventas en la base de datos
- Generar queries SQL para analizar datos
- Calcular KPIs (revenue, ticket promedio, etc.)
- Crear gráficas y visualizaciones
- Explicar conceptos de análisis de datos

**CONTEXTO:**

Tienes acceso a una base de datos con tablas: productos, clientes, ventas.

**PREGUNTA DEL USUARIO:**

{user_query}

**INSTRUCCIONES:**

1. Responde de forma concisa y amigable
2. Si preguntan qué puedes hacer, lista tus capacidades
3. Si preguntan por datos específicos, explica que necesitas más detalles
4. Mantén un tono profesional pero cercano

**RESPUESTA:**"""

    return ChatPromptTemplate.from_template(template)


# ============ Helper Functions ============

def test_llm_connection() -> bool:
    """
    Test básico de conectividad con el LLM.
    
    Returns:
        True si funciona, False si falla
    """
    try:
        llm = get_llama_model()
        response = invoke_llm_with_retry(
            llm,
            [{"role": "user", "content": "Responde solo 'OK' si me entiendes"}]
        )

        success = "ok" in response.lower()
        if success:
            logger.info("✅ LLM connection test passed")
        else:
            logger.warning(f"⚠️ Unexpected LLM response: {response}")

        return success

    except Exception as e:
        logger.error(f"❌ LLM connection test failed: {e}")
        return False


def test_embeddings() -> bool:
    """
    Test básico de generación de embeddings.
    
    Returns:
        True si funciona, False si falla
    """
    try:
        embedding = generate_single_embedding("test")

        # Verificar dimensión correcta
        if len(embedding) == 768:
            logger.info("✅ Embeddings test passed (768 dim)")
            return True
        else:
            logger.error(f"❌ Wrong embedding dimension: {len(embedding)}")
            return False

    except Exception as e:
        logger.error(f"❌ Embeddings test failed: {e}")
        return False


# Para testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testing LLM Configuration ===\n")

    # Test 1: Inicialización
    print("1. Initializing models...")
    llm = get_llama_model()
    embedding_model = get_embedding_model()
    print("   ✓ Models loaded\n")

    # Test 2: LLM básico
    print("2. Testing LLM:")
    response = invoke_llm_with_retry(
        llm,
        [{"role": "user", "content": "¿Cuál es la capital de Colombia?"}]
    )
    print(f"   Question: ¿Cuál es la capital de Colombia?")
    print(f"   Answer: {response}\n")

    # Test 3: Embeddings
    print("3. Testing embeddings:")
    text = "Este es un texto de prueba"
    embedding = generate_single_embedding(text)
    print(f"   Text: {text}")
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}\n")

    # Test 4: Prompts
    print("4. Testing prompts:")
    router_prompt = get_router_prompt()
    print(f"   Router prompt loaded: {len(router_prompt.messages)} messages")

    sql_prompt = get_sql_prompt()
    print(f"   SQL prompt loaded: {len(sql_prompt.messages)} messages\n")

    # Test 5: Router con LLM
    print("5. Testing router classification:")
    test_queries = [
        "¿Cuántas ventas hubo en enero?",
        "Calcula el revenue total",
        "Muéstrame una gráfica de ventas",
        "Hola, ¿cómo estás?"
    ]

    for query in test_queries:
        prompt = router_prompt.format(user_query=query)
        intent = invoke_llm_with_retry(llm, [{"role": "user", "content": prompt}])
        print(f"   Query: {query}")
        print(f"   Intent: {intent.strip()}\n")

    # Test 6: Connection tests
    print("6. Running connection tests:")
    print(f"   LLM: {'✓' if test_llm_connection() else '✗'}")
    print(f"   Embeddings: {'✓' if test_embeddings() else '✗'}")
