from pathlib import Path
from functools import lru_cache
from typing import List, ClassVar
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = BASE_DIR / ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False,
        extra = "ignore"
    )

    DATA_RAW: ClassVar[Path] = BASE_DIR / "data" / "data_raw"
    DATA_PROCESSED: ClassVar[Path] = BASE_DIR / "data" / "data_processed"

    CORS_ORIGINS: List[str] = ["*"]

    APP_NAME: str = "Movie Recommendation System"
    APP_VERSION: str = "2.0.0"

    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY: str = Field(default="", description="LangSmith API Key")
    LANGCHAIN_PROJECT: str = "movie_recommender"

    GROQ_API_KEY: str = Field(default="sua_chave_aqui", description="Groq API Key for LLM")
    GROQ_MODEL: str = "llama3-70b-8192"
    GROQ_TEMPERATURE: float = 0.7
    GROQ_MAX_TOKENS: int = 2048

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    PINECONE_API_KEY: str = Field(..., description="Pinecone API Key")
    PINECONE_ENVIRONMENT: str = Field(..., description="Pinecone Environment")
    PINECONE_INDEX_NAME: str = "movie-recommendation"
    PINECONE_NAMESPACE: str = "default"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    RETRIEVAL_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    USE_EMBEDDINGS: bool = True
    ENABLE_EXPLANATIONS: bool = True

    # paths for storing numpy arrays; use Path so we can access .parent etc.
    EMBEDDINGS_PATH: ClassVar[Path] = BASE_DIR / "data" / "processed" / "movies_embeddings.npy"
    MOVIE_IDS_PATH: ClassVar[Path] = BASE_DIR / "data" / "processed" / "movie_ids.npy"

@lru_cache()
def get_settings():
    return Settings() # type: ignore

settings = get_settings()