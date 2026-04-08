"""Abstract base class for runtime providers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class BaseRuntimeProvider(ABC):
    """Abstract base class for LLM runtime providers.
    
    All runtime providers (llama.cpp, vLLM, etc.) must inherit from this
    class and implement the required async methods.
    """

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        """Initialize the provider with a base URL.
        
        Args:
            base_url: The base URL of the runtime server (e.g., "http://127.0.0.1:8081")
            **kwargs: Additional provider-specific configuration
        """
        self.base_url = base_url.rstrip("/")
        self.config = kwargs

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Yields:
            String chunks of the generated content
        """
        raise NotImplementedError("Subclasses must implement generate_stream")

    @abstractmethod
    async def check_health(self) -> bool:
        """Check if the runtime server is healthy and ready.
        
        Returns:
            True if the server is healthy, False otherwise
        """
        raise NotImplementedError("Subclasses must implement check_health")

    async def close(self) -> None:
        """Clean up any resources. Override if needed."""
        pass
