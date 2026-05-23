from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM (mimo uses Anthropic-compatible API)
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://token-plan-cn.xiaomimimo.com/anthropic"
    model_name: str = "mimo-v2.5-pro"
    max_tokens: int = 4096

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    memory_db_path: str = "./data/memory_db"
    checkpoint_db_path: str = "./data/checkpoints/checkpoints.db"

    # Workflow
    max_iterations: int = 50
    max_retries_per_subtask: int = 3

    # Memory
    short_term_ttl: float = 3600.0
    short_term_max_entries: int = 1000
    retrieval_top_n: int = 5
    similarity_threshold: float = 0.7

    # Logging
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
