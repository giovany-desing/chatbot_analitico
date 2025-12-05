"""
Inicializador LLM
Expone modelos configurados
"""
from app.llm.models import (
    get_llama_model,
    get_embedding_model,
    get_sql_prompt,
    get_kpi_prompt,
    get_router_prompt
)

__all__ = [
    "get_llama_model",
    "get_embedding_model",
    "get_sql_prompt",
    "get_kpi_prompt",
    "get_router_prompt"
]

