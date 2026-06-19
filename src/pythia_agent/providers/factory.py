import os
import logging

from pythia_agent.config import Settings

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic", "bedrock", "litellm")


class ModelFactory:
    """Creates Strands model instances from configuration."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def create(self):
        """Create and return a Strands model based on configured provider."""
        provider = self.settings.model.provider.lower()
        logger.info(
            "Creating model provider: %s (model_id=%s)",
            provider,
            self.settings.model.model_id,
        )

        method = getattr(self, f"_create_{provider}", None)
        if method is None:
            raise ValueError(f"Unsupported model provider: '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}")
        return method()

    def _create_ollama(self):
        from pythia_agent.providers.pooled_ollama import PooledOllamaModel

        return PooledOllamaModel(
            host=self.settings.ollama.host,
            model_id=self.settings.model.model_id,
        )

    def _create_openai(self):
        from strands.models.openai import OpenAIModel

        return OpenAIModel(
            client_args={"api_key": os.environ.get("OPENAI_API_KEY")},
            model_id=self.settings.openai.model_id,
        )

    def _create_anthropic(self):
        from strands.models.anthropic import AnthropicModel

        return AnthropicModel(
            client_args={"api_key": os.environ.get("ANTHROPIC_API_KEY")},
            model_id=self.settings.anthropic.model_id,
        )

    def _create_bedrock(self):
        from strands.models.bedrock import BedrockModel

        return BedrockModel(
            model_id=self.settings.bedrock.model_id,
            region_name=self.settings.bedrock.region,
        )

    def _create_litellm(self):
        from strands.models.litellm import LiteLLMModel

        return LiteLLMModel(model_id=self.settings.model.model_id)


def create_model(settings: Settings):
    """Convenience function to create a model from settings."""
    return ModelFactory(settings).create()
