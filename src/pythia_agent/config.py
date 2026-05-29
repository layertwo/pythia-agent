from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class AgentConfig(BaseModel):
    name: str = "Pythia"
    system_prompt: str = "You are Pythia, a self-hosted AI agent with persistent memory and tool access. Be direct and concise."


class ModelConfig(BaseModel):
    provider: str = "ollama"
    model_id: str = "llama3.1"


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"


class OpenAIConfig(BaseModel):
    model_id: str = "gpt-4o"


class AnthropicConfig(BaseModel):
    model_id: str = "claude-sonnet-4-20250514"


class BedrockConfig(BaseModel):
    model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0"
    region: str = "us-east-1"


class MemoryLLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 2000


class MemoryEmbedderConfig(BaseModel):
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    embedding_dims: int = 768
    ollama_base_url: str = "http://localhost:11434"


class MemoryVectorStoreConfig(BaseModel):
    provider: str = "pgvector"
    collection_name: str = "pythia_memories"
    host: str = "localhost"
    port: int = 5432
    user: str = "pythia"
    password: str = "pythia"
    dbname: str = "pythia"


class MemoryConfig(BaseModel):
    enabled: bool = True
    auto_inject: bool = True
    auto_inject_top_k: int = 5
    auto_store: bool = True
    llm: MemoryLLMConfig = Field(default_factory=MemoryLLMConfig)
    embedder: MemoryEmbedderConfig = Field(default_factory=MemoryEmbedderConfig)
    vector_store: MemoryVectorStoreConfig = Field(default_factory=MemoryVectorStoreConfig)


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class _YamlSource(PydanticBaseSettingsSource):
    """Settings source that reads from a pre-loaded YAML dict.

    Sits below env-var sources in the priority chain, so env vars override it.
    """

    def __init__(self, settings_cls: type[BaseSettings], data: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:  # type: ignore[override]
        value = self._data.get(field_name)
        return value, field_name, value is not None

    def field_is_complex(self, field: Any) -> bool:  # type: ignore[override]
        return self.field_is_complex(field)

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in self._data.items() if v is not None}


def _make_settings_class(yaml_data: dict[str, Any]) -> type[BaseSettings]:
    """Return a Settings subclass whose lowest-priority source is *yaml_data*."""

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(env_prefix="PYTHIA_", env_nested_delimiter="__")

        agent: AgentConfig = Field(default_factory=AgentConfig)
        model: ModelConfig = Field(default_factory=ModelConfig)
        ollama: OllamaConfig = Field(default_factory=OllamaConfig)
        openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
        anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
        bedrock: BedrockConfig = Field(default_factory=BedrockConfig)
        memory: MemoryConfig = Field(default_factory=MemoryConfig)
        server: ServerConfig = Field(default_factory=ServerConfig)

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                file_secret_settings,
                _YamlSource(settings_cls, yaml_data),
            )

    return Settings


# Public type alias — import this elsewhere in the codebase.
Settings = _make_settings_class({})


def load_settings(config_path: str | Path | None = None) -> Any:
    """Load settings from config.yaml merged with environment variable overrides.

    Environment variables always take priority over values in config.yaml.
    """
    yaml_data: dict[str, Any] = {}

    if config_path is None:
        config_path = Path("/app/config.yaml")
        if not config_path.exists():
            config_path = Path("config.yaml")

    if isinstance(config_path, str):
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}

    return _make_settings_class(yaml_data)()
