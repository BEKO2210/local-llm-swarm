"""Base agent implementation for the swarm system."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from backend.app.core.config import get_settings
from backend.app.database.db import AgentTrace, ConversationTrace, async_session_maker
from backend.app.runtimes.llama_cpp import LlamaCppProvider
from backend.app.runtimes.process_manager import get_process_manager

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base agent class for managing LLM interactions.
    
    Each agent manages its own model lifecycle through the ProcessManager
    and provides a simple interface for text generation.
    """

    def __init__(
        self,
        pool_name: str,
        role_prompt: str = "",
        agent_type: str | None = None,
    ):
        """Initialize the base agent.
        
        Args:
            pool_name: Name of the model pool to use (e.g., 'worker_pool')
            role_prompt: System prompt defining the agent's role
            agent_type: Optional identifier for the agent type
        """
        self.pool_name = pool_name
        self.role_prompt = role_prompt
        self.agent_type = agent_type or self.__class__.__name__
        self.agent_id = f"{self.agent_type}_{uuid.uuid4().hex[:8]}"
        
        # These will be set when the model is started
        self._model_id: str | None = None
        self._process = None
        self._provider: LlamaCppProvider | None = None
        self._base_url: str | None = None

    async def _ensure_model_running(self) -> tuple[str, LlamaCppProvider]:
        """Ensure the model for this pool is running.
        
        Agents using the same pool share the same process to save VRAM.
        
        Returns:
            Tuple of (base_url, provider)
        """
        settings = get_settings()
        process_manager = get_process_manager()
        
        # Get pool configuration
        pool = settings.get_pool(self.pool_name)
        if not pool.models:
            raise RuntimeError(f"Pool '{self.pool_name}' has no models configured")
        
        # Get the first model from the pool
        model_id = pool.models[0]
        model_config = settings.get_model(model_id)
        self._model_id = model_id
        
        # Use pool_name as instance_id so agents in the same pool share the process
        # This prevents VRAM overload from multiple model instances
        pool_instance_id = f"pool_{self.pool_name}"
        
        # Check if this pool's model is already running
        existing_process = process_manager.get_process(pool_instance_id)
        if existing_process and existing_process.is_running:
            self._process = existing_process
            self._base_url = f"http://127.0.0.1:{existing_process.port}"
            logger.info(
                f"Agent {self.agent_id}: Reusing existing pool {self.pool_name} on {self._base_url}"
            )
        else:
            # Start the model (respects VRAM budget)
            try:
                self._process = await process_manager.start_model(model_id, pool_instance_id)
                self._base_url = f"http://127.0.0.1:{self._process.port}"
                logger.info(
                    f"Agent {self.agent_id}: Started new pool {self.pool_name} on {self._base_url}"
                )
            except RuntimeError as e:
                logger.error(f"Failed to start model for agent {self.agent_id}: {e}")
                raise
        
        # Initialize provider
        self._provider = LlamaCppProvider(base_url=self._base_url)
        
        return self._base_url, self._provider

    async def _save_to_db(
        self,
        user_prompt: str,
        assistant_response: str,
        trace_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save agent interaction to database.
        
        Args:
            user_prompt: The user's input
            assistant_response: The model's response
            trace_metadata: Optional metadata for debugging
        """
        if not self._model_id or not self._process:
            logger.warning(f"Agent {self.agent_id}: Cannot save to DB, model not running")
            return
        
        try:
            async with async_session_maker() as session:
                # Create agent trace
                agent_trace = AgentTrace(
                    id=self.agent_id,
                    agent_type=self.agent_type,
                    model_id=self._model_id,
                    pool_id=self.pool_name,
                    status="running",
                    vram_cost_mb=self._process.vram_cost_mb,
                    port=self._process.port,
                    pid=self._process.pid,
                    started_at=datetime.now(timezone.utc),
                )
                session.add(agent_trace)
                
                # Create conversation trace for user message
                user_trace = ConversationTrace(
                    id=f"{self.agent_id}_user_{uuid.uuid4().hex[:8]}",
                    agent_id=self.agent_id,
                    role="user",
                    content=user_prompt,
                    trace_metadata=trace_metadata,
                )
                session.add(user_trace)
                
                # Create conversation trace for assistant response
                assistant_trace = ConversationTrace(
                    id=f"{self.agent_id}_assistant_{uuid.uuid4().hex[:8]}",
                    agent_id=self.agent_id,
                    role="assistant",
                    content=assistant_response,
                    trace_metadata=trace_metadata,
                )
                session.add(assistant_trace)
                
                await session.commit()
                logger.debug(f"Agent {self.agent_id}: Saved interaction to DB")
                
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Failed to save to DB: {e}")
            # Don't raise - DB failure shouldn't break the flow

    def _build_messages(
        self,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build the message list for the LLM.
        
        Args:
            user_prompt: The current user prompt
            history: Optional conversation history (OpenAI format)
            
        Returns:
            List of messages ready for the LLM
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.role_prompt},
        ]
        
        # Add history if provided
        if history:
            messages.extend(history)
        
        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})
        
        return messages

    async def generate(
        self,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion for the given prompt.
        
        This method:
        1. Ensures the model is running
        2. Sends the prompt with role context and optional history
        3. Collects all chunks into a full response
        4. Saves the interaction to the database
        
        Args:
            user_prompt: The user's input prompt
            history: Optional conversation history in OpenAI format
            **kwargs: Additional generation parameters
            
        Returns:
            The complete generated response as a string
        """
        # Ensure model is running
        base_url, provider = await self._ensure_model_running()
        
        # Build messages with system prompt, history, and user prompt
        messages = self._build_messages(user_prompt, history)
        
        logger.info(f"Agent {self.agent_id}: Generating response...")
        start_time = datetime.now(timezone.utc)
        
        try:
            # Collect all chunks into a full response
            chunks: list[str] = []
            async for chunk in provider.generate_stream(messages=messages, **kwargs):
                chunks.append(chunk)
            
            full_response = "".join(chunks)
            
            elapsed_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            
            logger.info(
                f"Agent {self.agent_id}: Generated {len(full_response)} chars "
                f"in {elapsed_ms:.0f}ms"
            )
            
            # Save to database
            await self._save_to_db(
                user_prompt=user_prompt,
                assistant_response=full_response,
                trace_metadata={
                    "base_url": base_url,
                    "model_id": self._model_id,
                    "elapsed_ms": elapsed_ms,
                    "history_length": len(history) if history else 0,
                },
            )
            
            return full_response
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Generation failed: {e}")
            raise
        finally:
            # Don't close provider here - we might reuse it
            pass

    async def generate_stream(
        self,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ):
        """Generate a streaming completion for the given prompt.
        
        This is a generator that yields chunks as they arrive.
        The final response is saved to DB after streaming completes.
        
        Args:
            user_prompt: The user's input prompt
            history: Optional conversation history in OpenAI format
            **kwargs: Additional generation parameters
            
        Yields:
            String chunks of the generated content
        """
        # Ensure model is running
        base_url, provider = await self._ensure_model_running()
        
        # Build messages with system prompt, history, and user prompt
        messages = self._build_messages(user_prompt, history)
        
        logger.info(f"Agent {self.agent_id}: Generating streaming response...")
        start_time = datetime.now(timezone.utc)
        
        chunks: list[str] = []
        
        try:
            async for chunk in provider.generate_stream(messages=messages, **kwargs):
                chunks.append(chunk)
                yield chunk
            
            full_response = "".join(chunks)
            elapsed_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            
            logger.info(
                f"Agent {self.agent_id}: Streamed {len(full_response)} chars "
                f"in {elapsed_ms:.0f}ms"
            )
            
            # Save to database after streaming completes
            await self._save_to_db(
                user_prompt=user_prompt,
                assistant_response=full_response,
                trace_metadata={
                    "base_url": base_url,
                    "model_id": self._model_id,
                    "elapsed_ms": elapsed_ms,
                    "streamed": True,
                    "history_length": len(history) if history else 0,
                },
            )
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Streaming generation failed: {e}")
            raise

    async def close(self) -> None:
        """Clean up resources."""
        if self._provider:
            await self._provider.close()
            self._provider = None
