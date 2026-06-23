from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class AgentConfig(BaseModel):
    name: str = "Pythia"
    system_prompt: str = (
        "You are Pythia, a self-hosted AI agent with persistent memory and tool access. Be direct and concise."
    )


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
    auto_inject_min_score: float = 0.5
    auto_store: bool = True
    llm: MemoryLLMConfig = Field(default_factory=MemoryLLMConfig)
    embedder: MemoryEmbedderConfig = Field(default_factory=MemoryEmbedderConfig)
    vector_store: MemoryVectorStoreConfig = Field(default_factory=MemoryVectorStoreConfig)


class ContextConfig(BaseModel):
    """Native conversation-window management (replaces the custom ContextPlugin).

    Maps onto strands' SummarizingConversationManager:
    - pin_first             <- the old protect_first_n (system prompt + setup)
    - preserve_recent_messages <- the old protect_last_n (active work)
    Summarization uses the agent's own model unless a separate one is wired in.
    """

    enabled: bool = True
    pin_first: int = 3
    preserve_recent_messages: int = 6
    summary_ratio: float = 0.3


class LimitsConfig(BaseModel):
    """Per-invocation hard backstop passed to the Strands agent loop.

    Complements (does not replace) SafetyPlugin's soft loop/repeat detection.
    A breached limit stops the loop gracefully via stop_reason rather than raising.
    Any field left as None is omitted from the limits dict sent to Strands.
    """

    enabled: bool = True
    turns: int | None = 50
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_limits(self) -> dict | None:
        """Render to the Strands `limits` dict, or None when disabled/empty.

        Omits unset fields so Strands receives only the caps actually set
        (its Limits is a total=False TypedDict).
        """
        if not self.enabled:
            return None
        built = {k: v for k, v in self.model_dump(exclude={"enabled"}).items() if v is not None}
        return built or None


class DreamsConfig(BaseModel):
    """Memory consolidation ("dreams") settings.

    When enabled, a cron job periodically iterates every user that has
    conversation activity, reads each user's recent sessions plus their
    current memory store, and asks an LLM to consolidate duplicates,
    resolve contradictions, and surface new insights — then atomically
    replaces that user's memories with the consolidated set.

    Auto-fire is gated by BOTH the cron schedule AND minimum activity since
    the user's last dream — `min_hours_between_dreams` and
    `min_sessions_between_dreams` prevent dreaming on a quiet user (wasted
    LLM calls) or on a single long session split across days (cron fires
    but nothing has changed for that user).
    """

    enabled: bool = False
    cron: str = "0 3 * * *"           # 3 AM UTC daily — only fires when dual gate passes
    max_sessions: int = 10            # recent sessions pulled when sessions are included
    max_messages_per_session: int = 50
    max_drop_ratio: float = 0.5       # skip swap if > this fraction dropped
    max_rewrite_ratio: float = 0.7    # skip swap if > this fraction replaced
    min_retention_ratio: float = 0.3  # skip swap if count_after/count_before < this
    retain_runs: int = 7              # number of historical Dream snapshots kept per user
    # Dual-gate trigger (only applies to cron-fired dreams, not on-demand):
    min_hours_between_dreams: int = 24       # cron skips if last dream is newer than this
    min_sessions_between_dreams: int = 5     # cron skips if fewer distinct sessions since last dream


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
        # YAML data is already-parsed Python (dicts, lists, scalars); no string
        # decoding needed. Returning False prevents pydantic-settings from trying
        # to JSON-decode values that came in already typed.
        return False

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
        context: ContextConfig = Field(default_factory=ContextConfig)
        limits: LimitsConfig = Field(default_factory=LimitsConfig)
        dreams: DreamsConfig = Field(default_factory=DreamsConfig)
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
