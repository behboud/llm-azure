import os
from typing import Iterable, Iterator, List, Union

import llm
import yaml
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from llm import hookimpl
from llm.default_plugins.openai_models import (AsyncChat, Chat, _Shared,
                                               not_nulls)
from llm.models import EmbeddingModel
from llm.utils import logging_client
from openai import AsyncAzureOpenAI, AzureOpenAI


def simplify_usage_dict(d):
    # Recursively remove keys with value 0 and empty dictionaries
    def remove_empty_and_zero(obj):
        if isinstance(obj, dict):
            cleaned = {
                k: remove_empty_and_zero(v)
                for k, v in obj.items()
                if v != 0 and v != {}
            }
            return {k: v for k, v in cleaned.items() if v is not None and v != {}}
        return obj

    return remove_empty_and_zero(d) or {}

def validate_model_config(model):
    """Validate Azure model configuration for authentication method.
    
    Ensures that exactly one authentication method is specified: either
    'azure_resource_name' (for Entra ID) or 'api_base' (for API key),
    but not both or neither.
    
    Args:
        model: Dictionary containing model configuration with keys like
               'model_id', 'azure_resource_name', and/or 'api_base'.
    
    Raises:
        ValueError: If both authentication methods are specified or if
                   neither authentication method is specified.
    """
    has_resource_name = 'azure_resource_name' in model
    has_api_base = 'api_base' in model
    
    if has_resource_name and has_api_base:
        raise ValueError(
            f"Model '{model.get('model_id', 'unknown')}' has both 'azure_resource_name' "
            f"and 'api_base' configured. Please specify only one authentication method."
        )
    
    if not has_resource_name and not has_api_base:
        raise ValueError(
            f"Model '{model.get('model_id', 'unknown')}' must specify either "
            f"'azure_resource_name' (for Entra ID) or 'api_base' (for API key)."
        )


@hookimpl
def register_models(register):
    """Register Azure OpenAI chat models with the LLM plugin system.
    
    Loads model configurations from the user's azure/config.yaml file and
    registers both synchronous (AzureChat) and asynchronous (AzureAsyncChat)
    versions of each chat model.
    
    Args:
        register: Registration function provided by the LLM plugin system.
    """
    azure_path = llm.user_dir() / "azure"
    azure_path.mkdir(exist_ok=True)
    azure_path = azure_path / "config.yaml"

    with open(azure_path, encoding='utf-8') as f:
        azure_models = yaml.safe_load(f)

    for model in azure_models:
        validate_model_config(model)
        
        if model.get('embedding_model'):
            continue

        aliases = model.pop("aliases", [])

        register(
            AzureChat(**model),
            AzureAsyncChat(**model),
            aliases=aliases,
        )


@hookimpl
def register_embedding_models(register):
    """Register Azure OpenAI embedding models with the LLM plugin system.
    
    Loads model configurations from the user's azure/config.yaml file and
    registers embedding models (AzureEmbedding) for text vectorization.
    
    Args:
        register: Registration function provided by the LLM plugin system.
    """
    azure_path = llm.user_dir() / "azure"
    azure_path.mkdir(exist_ok=True)
    azure_path = azure_path / "config.yaml"

    with open(azure_path, encoding='utf-8') as f:
        azure_models = yaml.safe_load(f)

    for model in azure_models:
        validate_model_config(model)
        
        if not model.get('embedding_model'):
            continue

        aliases = model.pop("aliases", [])
        model.pop('embedding_model')

        register(
            AzureEmbedding(**model),
            aliases=aliases,
        )


class AzureShared(_Shared):
    """Shared functionality for Azure OpenAI chat models.
    
    Provides common authentication and client creation logic for both
    synchronous and asynchronous Azure OpenAI chat models. Supports two
    authentication methods:
    - Entra ID (Azure AD) via DefaultAzureCredential when azure_resource_name is provided
    - API key authentication when api_base is provided
    
    Attributes:
        key_env_var: Environment variable name for API key (AZURE_OPENAI_API_KEY).
        default_max_tokens: Default maximum tokens for completions (None).
        azure_resource_name: Azure resource name for Entra ID authentication.
        needs_key: Whether API key is required ("azure" or None).
    """
    key_env_var = "AZURE_OPENAI_API_KEY"
    default_max_tokens = None

    def __init__(self, *args, **kwargs):
        """Initialize AzureShared with optional Entra ID authentication.
        
        Args:
            *args: Positional arguments passed to parent _Shared class.
            **kwargs: Keyword arguments including optional 'azure_resource_name'
                     for Entra ID authentication. Other kwargs passed to parent.
        """
        self.azure_resource_name = kwargs.pop('azure_resource_name', None)
        super().__init__(*args, **kwargs)
        if self.azure_resource_name:
            self.needs_key = None
        else:
            self.needs_key = "azure"

    def get_key(self, key):
        """Get the Azure OpenAI API key from parameter or environment variable.
        
        Args:
            key: API key provided as parameter, or None to use environment variable.
        
        Returns:
            The API key string from parameter or AZURE_OPENAI_API_KEY env var.
        """
        if key:
            return key
        return os.environ.get(self.key_env_var)

    def logging_client(self):
        """Get an HTTP client configured for response logging.
        
        Returns:
            An httpx.Client configured to log HTTP requests/responses when
            LLM_OPENAI_SHOW_RESPONSES environment variable is set.
        """
        
        return logging_client()

    def get_client(self, key, *, async_=False):
        """Create an Azure OpenAI client with appropriate authentication.
        
        Supports two authentication methods:
        - Entra ID: Uses DefaultAzureCredential when azure_resource_name is set
        - API Key: Uses provided key or AZURE_OPENAI_API_KEY env var
        
        Args:
            key: API key for authentication (ignored if using Entra ID).
            async_: If True, returns AsyncAzureOpenAI client; otherwise AzureOpenAI.
        
        Returns:
            AzureOpenAI or AsyncAzureOpenAI client instance configured with
            appropriate authentication and endpoint.
        """
        if self.azure_resource_name:
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default"
            )
            kwargs = {
                "azure_ad_token_provider": token_provider,
                "api_version": self.api_version,
                "azure_endpoint": f"https://{self.azure_resource_name}.openai.azure.com/",
            }
        else:
            kwargs = {
                "api_key": self.get_key(key),
                "api_version": self.api_version,
                "azure_endpoint": self.api_base,
            }
        
        if os.environ.get("LLM_OPENAI_SHOW_RESPONSES"):
            kwargs["http_client"] = self.logging_client()
        if async_:
            return AsyncAzureOpenAI(**kwargs)
        else:
            return AzureOpenAI(**kwargs)

    def build_kwargs(self, prompt, stream):
        """Build keyword arguments for Azure OpenAI API completion request.
        
        Constructs the kwargs dict from prompt options, handling special cases
        like JSON output formatting and default max_tokens.
        
        Args:
            prompt: Prompt object containing user options and configuration.
            stream: Boolean indicating if response should be streamed.
        
        Returns:
            Dictionary of keyword arguments to pass to the API client.
        """
        kwargs = dict(not_nulls(prompt.options))
        json_object = kwargs.pop("json_object", None)
        if "max_tokens" not in kwargs and self.default_max_tokens is not None:
            kwargs["max_tokens"] = self.default_max_tokens
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs
    
    def set_usage(self, response, usage):
        if not usage:
            return
        input_tokens = usage.pop("prompt_tokens")
        output_tokens = usage.pop("completion_tokens")
        usage.pop("total_tokens")
        response.set_usage(
            input=input_tokens, output=output_tokens, details=simplify_usage_dict(usage)
        )


class AzureChat(AzureShared, Chat):
    """Synchronous Azure OpenAI chat model.
    
    Inherits from AzureShared for Azure-specific authentication and client
    creation, and from Chat for synchronous chat completion functionality.
    """


class AzureAsyncChat(AzureShared, AsyncChat):
    """Asynchronous Azure OpenAI chat model.
    
    Inherits from AzureShared for Azure-specific authentication and client
    creation, and from AsyncChat for asynchronous chat completion functionality.
    """


class AzureEmbedding(EmbeddingModel):
    """Azure OpenAI embedding model for text vectorization.
    
    Provides functionality to generate embeddings (vector representations)
    for text inputs using Azure OpenAI's embedding models.
    
    Attributes:
        needs_key: Indicates API key is required ("azure").
        key_env_var: Environment variable name for API key (AZURE_OPENAI_API_KEY).
        batch_size: Number of items to embed in a single batch (100).
        model_id: Identifier for this model in the LLM system.
        model_name: Actual model name to pass to Azure OpenAI API.
        api_base: Azure OpenAI endpoint URL.
        api_version: Azure OpenAI API version string.
    """
    needs_key = "azure"
    key_env_var = "AZURE_OPENAI_API_KEY"
    batch_size = 100

    def __init__(self, model_id, model_name, api_base, api_version):
        """Initialize Azure embedding model.
        
        Args:
            model_id: Identifier for this model in the LLM system.
            model_name: Actual Azure OpenAI model name (e.g., 'text-embedding-3-small').
            api_base: Azure OpenAI endpoint URL.
            api_version: Azure OpenAI API version string (e.g., '2023-05-14').
        """
        self.model_id = model_id
        self.model_name = model_name
        self.api_base = api_base
        self.api_version = api_version

    def embed_batch(self, items: Iterable[Union[str, bytes]]) -> Iterator[List[float]]:
        """Generate embeddings for a batch of text items.
        
        Creates vector embeddings for the provided text items using the
        Azure OpenAI embeddings API.
        
        Args:
            items: Iterable of strings or bytes to generate embeddings for.
        
        Returns:
            Iterator yielding lists of floats, where each list is the embedding
            vector for the corresponding input item.
        """
        client = AzureOpenAI(
            api_key=self.get_key(),
            api_version=self.api_version,
            azure_endpoint=self.api_base,
        )
        kwargs = {
            "input": items,
            "model": self.model_name,
        }
        results = client.embeddings.create(**kwargs).data
        return ([float(r) for r in result.embedding] for result in results)
