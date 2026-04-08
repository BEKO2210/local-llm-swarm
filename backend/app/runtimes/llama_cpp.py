"""Llama.cpp server runtime provider implementation."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from backend.app.runtimes.base import BaseRuntimeProvider

logger = logging.getLogger(__name__)


class LlamaCppProvider(BaseRuntimeProvider):
    """Provider for llama.cpp server (llama-server) OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        max_retries: int = 30,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the llama.cpp provider.

        Args:
            base_url: The base URL of the llama-server instance
            timeout: Request timeout in seconds (default: 120s for long generations)
            max_retries: Max retries to wait for server readiness (default: 30)
            retry_delay: Delay between readiness checks in seconds (default: 1.0)
            **kwargs: Additional configuration options
        """
        super().__init__(base_url, **kwargs)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client with limited connections."""
        if self._client is None:
            # Limit connections to prevent Windows socket exhaustion
            limits = httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            )
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
                limits=limits,
            )
        return self._client

    async def check_health(self) -> bool:
        """Check if the llama-server is healthy.

        Returns:
            True if the server responds with 200, False otherwise
        """
        try:
            response = await self.client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {self.base_url}: {e}")
            return False

    async def wait_for_ready(self) -> bool:
        """Wait for the server to be ready to accept requests.

        Polls the /health endpoint until the server responds with 200
        or max_retries is reached.

        Returns:
            True if server is ready, False if timed out
        """
        # Use a separate client with shorter timeout for health checks
        # to avoid blocking the main client
        health_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=5.0,
            limits=httpx.Limits(max_connections=3, max_keepalive_connections=1),
        )
        
        try:
            for attempt in range(self.max_retries):
                try:
                    response = await health_client.get("/health")
                    if response.status_code == 200:
                        logger.debug(f"Server {self.base_url} is ready")
                        return True
                except Exception:
                    pass

                if attempt < self.max_retries - 1:
                    logger.debug(
                        f"Server not ready, waiting... ({attempt + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(self.retry_delay)
        finally:
            await health_client.aclose()

        return False

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        top_p: float = 0.9,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model identifier (optional, llama-server ignores this)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: List of stop sequences
            **kwargs: Additional parameters to pass to the API

        Yields:
            String chunks of the generated content
        """
        # Wait for server to be ready first
        if not await self.wait_for_ready():
            raise RuntimeError(
                f"Server at {self.base_url} did not become ready after "
                f"{self.max_retries * self.retry_delay}s"
            )

        payload: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
        }

        # Add optional parameters
        if model:
            payload["model"] = model
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        # Merge any additional kwargs
        payload.update(kwargs)

        logger.debug(f"Sending streaming request to {self.base_url}/v1/chat/completions")

        try:
            # Use a fresh client for streaming to avoid issues
            async with httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            ) as client:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=payload,
                ) as response:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        # Read the error response before raising
                        error_text = await response.aread()
                        logger.error(
                            f"HTTP error {e.response.status_code}: {error_text.decode()[:500]}"
                        )
                        raise

                    async for line in response.aiter_lines():
                        line = line.strip()

                        # Skip empty lines
                        if not line:
                            continue

                        # Skip lines that don't start with "data: "
                        if not line.startswith("data: "):
                            continue

                        # Extract the data portion
                        data = line[6:]  # Remove "data: " prefix

                        # Check for stream end
                        if data == "[DONE]":
                            break

                        # Parse the JSON chunk
                        try:
                            chunk = json.loads(data)

                            # Extract content from choices
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue

                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                yield content

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse SSE chunk: {e}")
                            continue

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from llama-server: {e.response.status_code}"
            )
            raise RuntimeError(
                f"LLM server error: {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Request error connecting to llama-server: {e}")
            raise RuntimeError(
                f"Failed to connect to LLM server at {self.base_url}"
            ) from e

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
