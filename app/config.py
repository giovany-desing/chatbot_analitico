from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional


class Settings(BaseSettings):
    """
    Configuración centralizada de la aplicación.
    Carga variables desde .env y las valida.
    """

    # ============ LLM APIs ============
    GROQ_API_KEY: str = Field(..., description="API key de Groq")
    OPENAI_API_KEY: Optional[str] = Field(None, description="API key OpenAI (fallback)")

    # ============ MySQL ============
    MYSQL_HOST: str = Field(default="localhost")
    MYSQL_PORT: int = Field(default=3306)
    MYSQL_USER: str = Field(...)
    MYSQL_PASSWORD: str = Field(...)
    MYSQL_DATABASE: str = Field(...)

    @property
    def MYSQL_URL(self) -> str:
        """Construye connection string de MySQL"""
        return (
            f"mysql+mysqlconnector://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    # ============ Redis ============
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_TTL: int = Field(default=3600, description="TTL en segundos")

    # ============ PostgreSQL ============
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_USER: str = Field(...)
    POSTGRES_PASSWORD: str = Field(...)
    POSTGRES_DB: str = Field(...)

    @property
    def POSTGRES_URL(self) -> str:
        """Construye connection string de PostgreSQL"""
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ============ LLM Configuration ============
    LLM_MODEL: str = Field(default="llama-3.3-70b-versatile")
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2000, gt=0)
    LLM_TIMEOUT: int = Field(default=30, description="Timeout en segundos")

    # ============ App Configuration ============
    APP_NAME: str = Field(default="Chatbot Analitico")
    APP_VERSION: str = Field(default="1.0.0")
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")

    # Configuración de Pydantic Settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignora variables extra en .env
    )

    @field_validator("GROQ_API_KEY")
    @classmethod
    def validate_groq_key(cls, v):
        if not v.startswith("gsk_"):
            raise ValueError("GROQ_API_KEY debe empezar con 'gsk_'")
        return v

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError("Temperature debe estar entre 0 y 2")
        return v
    
    # ============ Fine-tuned Model ============
    FINETUNED_MODEL_ENDPOINT: str = Field(
        default="",
        description="URL del modelo fine-tuned en Modal.com"
    )
    


# Singleton: una sola instancia en toda la app
settings = Settings()

# Para debugging (opcional)
if __name__ == "__main__":
    print("=== Configuración Cargada ===")
    print(f"App: {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"MySQL URL: {settings.MYSQL_URL}")
    print(f"Redis URL: {settings.REDIS_URL}")
    print(f"Postgres URL: {settings.POSTGRES_URL}")
    print(f"LLM Model: {settings.LLM_MODEL}")
    print(f"Debug: {settings.DEBUG}")