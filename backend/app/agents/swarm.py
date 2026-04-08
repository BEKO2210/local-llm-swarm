"""Hierarchical swarm pipeline for multi-agent text refinement with memory."""

import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, desc

from backend.app.agents.base import BaseAgent
from backend.app.database.db import AgentTrace, ConversationTrace, async_session_maker

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages conversation history and context retrieval from the database."""

    @staticmethod
    async def fetch_history(
        conversation_id: str,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Fetch the last N messages for a conversation.
        
        Args:
            conversation_id: The unique conversation identifier
            limit: Maximum number of messages to fetch (default: 10)
            
        Returns:
            List of messages in OpenAI format [{"role": "user", "content": "..."}, ...]
        """
        try:
            async with async_session_maker() as session:
                # Query conversation traces for this conversation_id
                # We look for traces that have this conversation_id in their metadata
                stmt = (
                    select(ConversationTrace)
                    .where(
                        ConversationTrace.trace_metadata["conversation_id"].as_string() == conversation_id
                    )
                    .order_by(desc(ConversationTrace.created_at))
                    .limit(limit)
                )
                
                result = await session.execute(stmt)
                traces = result.scalars().all()
                
                # Convert to OpenAI format and reverse to get chronological order
                history = [
                    {"role": trace.role, "content": trace.content}
                    for trace in reversed(traces)
                ]
                
                logger.debug(
                    f"ContextManager: Fetched {len(history)} messages for {conversation_id}"
                )
                return history
                
        except Exception as e:
            logger.error(f"ContextManager: Failed to fetch history: {e}")
            return []

    @staticmethod
    async def save_message(
        conversation_id: str,
        agent_id: str,
        role: str,
        content: str,
        step_name: str = "",
        trace_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a message to the conversation history.
        
        Args:
            conversation_id: The unique conversation identifier
            agent_id: ID of the agent that generated this message
            role: Message role (user/assistant/system)
            content: Message content
            step_name: Optional step identifier (e.g., "planner", "executor", "critic")
            trace_metadata: Additional metadata
        """
        try:
            async with async_session_maker() as session:
                conversation_trace = ConversationTrace(
                    id=f"{conversation_id}_{step_name}_{uuid.uuid4().hex[:6]}",
                    agent_id=agent_id,
                    role=role,
                    content=content,
                    trace_metadata={
                        "conversation_id": conversation_id,
                        "step": step_name,
                        **(trace_metadata or {}),
                    },
                )
                session.add(conversation_trace)
                await session.commit()
                
        except Exception as e:
            logger.error(f"ContextManager: Failed to save message: {e}")


class SwarmPipeline:
    """A hierarchical pipeline with persistent memory and optional deep thinking mode.
    
    Standard mode: Planner -> Executor -> Critic
    Deep Thinking mode: Planner -> Executor -> Critic -> Heavy Critic (Gemma 26B)
    
    The planner creates a strategy, the executor builds a response, the critic polishes,
    and optionally the heavy critic provides a final deep-dive analysis.
    The final output is streamed directly to the user for better UX.
    """

    def __init__(
        self,
        planner_pool: str = "worker_pool",
        executor_pool: str = "worker_pool",
        critic_pool: str = "worker_pool",
        heavy_critic_pool: str = "heavy_pool",
        deep_thinking: bool = False,
    ):
        """Initialize the swarm pipeline with specialized agents.
        
        Args:
            planner_pool: Pool name for the planner agent
            executor_pool: Pool name for the executor agent
            critic_pool: Pool name for the critic agent
            heavy_critic_pool: Pool name for the heavy critic (Gemma 26B)
            deep_thinking: If True, enables 4-stage pipeline with heavy critic
        """
        self.pipeline_id = f"swarm_{uuid.uuid4().hex[:8]}"
        self.context_manager = ContextManager()
        self.deep_thinking = deep_thinking
        
        # Initialize planner agent (creates strategy)
        self.planner = BaseAgent(
            pool_name=planner_pool,
            role_prompt=(
                "You are a strategist. Create a concise plan to solve the user's request. "
                "Respond in the user's language. Keep your plan brief and actionable."
            ),
            agent_type="PlannerAgent",
        )
        
        # Initialize executor agent (builds response based on plan + history)
        self.executor = BaseAgent(
            pool_name=executor_pool,
            role_prompt=(
                "You are a builder. Follow the provided plan to create a response. "
                "Use the conversation history for context. Be thorough but concise."
            ),
            agent_type="ExecutorAgent",
        )
        
        # Initialize critic agent (initial polish)
        self.critic = BaseAgent(
            pool_name=critic_pool,
            role_prompt=(
                "You are a perfectionist. Review the executor's work against the plan "
                "and history. Provide a polished version. Do not add conversational filler."
            ),
            agent_type="CriticAgent",
        )
        
        # Initialize heavy critic agent (super-intelligent deep analysis)
        # Only used in deep thinking mode
        self.heavy_critic = BaseAgent(
            pool_name=heavy_critic_pool,
            role_prompt=(
                "You are a super-intelligent AI auditor with vast knowledge. "
                "Provide a final deep-dive polish of the response. "
                "Enhance depth, accuracy, and insight while maintaining the original language. "
                "Be thorough but concise. No meta-commentary."
            ),
            agent_type="HeavyCriticAgent",
        )
        
        mode_str = "DEEP THINKING" if deep_thinking else "STANDARD"
        logger.info(
            f"SwarmPipeline {self.pipeline_id}: Initialized in {mode_str} mode with "
            f"planner={planner_pool}, executor={executor_pool}, critic={critic_pool}, "
            f"heavy_critic={heavy_critic_pool if deep_thinking else 'disabled'}"
        )

    async def _save_pipeline_step(
        self,
        conversation_id: str,
        agent_id: str,
        agent_type: str,
        model_id: str | None,
        pool_id: str,
        role: str,
        content: str,
        step_name: str,
        trace_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a pipeline step to the database.
        
        Args:
            conversation_id: The conversation identifier
            agent_id: ID of the agent
            agent_type: Type of agent (PlannerAgent/ExecutorAgent/CriticAgent)
            model_id: ID of the model used
            pool_id: Pool used
            role: Role in conversation (user/assistant/system)
            content: The content
            step_name: Name of the pipeline step
            trace_metadata: Optional metadata
        """
        try:
            async with async_session_maker() as session:
                # Create or update agent trace
                agent_trace = AgentTrace(
                    id=agent_id,
                    agent_type=agent_type,
                    model_id=model_id or "unknown",
                    pool_id=pool_id,
                    status="completed",
                    vram_cost_mb=0,
                    port=None,
                    pid=None,
                    started_at=datetime.now(timezone.utc),
                    stopped_at=datetime.now(timezone.utc),
                )
                await session.merge(agent_trace)
                await session.commit()
                
                # Save conversation trace via context manager
                await self.context_manager.save_message(
                    conversation_id=conversation_id,
                    agent_id=agent_id,
                    role=role,
                    content=content,
                    step_name=step_name,
                    trace_metadata={
                        "pipeline_id": self.pipeline_id,
                        "agent_type": agent_type,
                        **(trace_metadata or {}),
                    },
                )
                
                logger.debug(f"Pipeline {self.pipeline_id}: Saved {step_name} to DB")
                
        except Exception as e:
            logger.error(f"Pipeline {self.pipeline_id}: Failed to save step: {e}")

    async def run_pipeline(
        self,
        user_prompt: str,
        conversation_id: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Run the full 3-stage swarm pipeline with memory.
        
        Workflow:
        1. Fetch conversation history for context
        2. Planner generates a hidden strategy plan
        3. Executor builds a response based on plan + history
        4. Critic streams the final polished version
        5. All steps are saved to database with conversation_id
        
        Args:
            user_prompt: The user's original request
            conversation_id: Unique identifier for this conversation
            **kwargs: Additional generation parameters
            
        Yields:
            String chunks from the critic's final polished response
        """
        logger.info(
            f"Pipeline {self.pipeline_id}: Starting for conversation {conversation_id}, "
            f"prompt: {user_prompt[:50]}..."
        )
        pipeline_start = datetime.now(timezone.utc)
        
        # ========================================================================
        # STEP 0: Fetch conversation history
        # ========================================================================
        logger.info(f"Pipeline {self.pipeline_id}: Fetching conversation history...")
        history = await self.context_manager.fetch_history(conversation_id, limit=10)
        logger.info(f"Pipeline {self.pipeline_id}: Loaded {len(history)} history messages")
        
        # ========================================================================
        # STAGE 1: Planner creates strategy (hidden step, saved to DB)
        # ========================================================================
        logger.info(f"Pipeline {self.pipeline_id}: Stage 1 - Planner creating strategy...")
        
        try:
            plan = await self.planner.generate(
                user_prompt=user_prompt,
                history=history if history else None,
                **kwargs,
            )
            logger.info(
                f"Pipeline {self.pipeline_id}: Planner created plan ({len(plan)} chars)"
            )
        except Exception as e:
            logger.error(f"Pipeline {self.pipeline_id}: Planner failed: {e}")
            yield f"[ERROR: Planner failed: {e}]"
            return
        
        # Save planner output
        if self.planner._model_id:
            await self._save_pipeline_step(
                conversation_id=conversation_id,
                agent_id=self.planner.agent_id,
                agent_type="PlannerAgent",
                model_id=self.planner._model_id,
                pool_id=self.planner.pool_name,
                role="assistant",
                content=plan,
                step_name="planner",
                trace_metadata={"user_prompt": user_prompt},
            )
        
        # ========================================================================
        # STAGE 2: Executor builds response (hidden step, saved to DB)
        # ========================================================================
        logger.info(f"Pipeline {self.pipeline_id}: Stage 2 - Executor building response...")
        
        executor_prompt = (
            f"Original User Request: {user_prompt}\n\n"
            f"Strategy Plan:\n{plan}\n\n"
            f"Create a response following the plan above."
        )
        
        try:
            draft = await self.executor.generate(
                user_prompt=executor_prompt,
                history=history if history else None,
                **kwargs,
            )
            logger.info(
                f"Pipeline {self.pipeline_id}: Executor built draft ({len(draft)} chars)"
            )
        except Exception as e:
            logger.error(f"Pipeline {self.pipeline_id}: Executor failed: {e}")
            yield f"[ERROR: Executor failed: {e}]"
            return
        
        # Save executor output
        if self.executor._model_id:
            await self._save_pipeline_step(
                conversation_id=conversation_id,
                agent_id=self.executor.agent_id,
                agent_type="ExecutorAgent",
                model_id=self.executor._model_id,
                pool_id=self.executor.pool_name,
                role="assistant",
                content=draft,
                step_name="executor",
                trace_metadata={"user_prompt": user_prompt, "plan": plan},
            )
        
        # ========================================================================
        # STAGE 3: Critic polishes (internal step when deep thinking)
        # ========================================================================
        logger.info(f"Pipeline {self.pipeline_id}: Stage 3 - Critic polishing...")
        
        critic_prompt = (
            f"Original User Request: {user_prompt}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Executor's Draft:\n{draft}\n\n"
            f"Provide the final polished version."
        )
        
        try:
            if self.deep_thinking:
                # In deep thinking mode, critic runs internally
                critic_draft = await self.critic.generate(
                    user_prompt=critic_prompt,
                    history=history if history else None,
                    **kwargs,
                )
                logger.info(
                    f"Pipeline {self.pipeline_id}: Critic completed internal polish "
                    f"({len(critic_draft)} chars)"
                )
            else:
                # In standard mode, critic streams to user
                critic_chunks: list[str] = []
                async for chunk in self.critic.generate_stream(
                    user_prompt=critic_prompt,
                    history=history if history else None,
                    **kwargs,
                ):
                    critic_chunks.append(chunk)
                    yield chunk
                
                critic_draft = "".join(critic_chunks)
                logger.info(
                    f"Pipeline {self.pipeline_id}: Critic streamed "
                    f"({len(critic_draft)} chars)"
                )
                
                # Save critic output
                elapsed_ms = (
                    datetime.now(timezone.utc) - pipeline_start
                ).total_seconds() * 1000
                logger.info(f"Pipeline {self.pipeline_id}: Completed in {elapsed_ms:.0f}ms")
                return  # End here for standard mode
                
        except Exception as e:
            logger.error(f"Pipeline {self.pipeline_id}: Critic failed: {e}")
            yield f"[ERROR: Critic failed: {e}]"
            return
        
        # Save critic output (for deep thinking mode)
        if self.deep_thinking and self.critic._model_id:
            await self._save_pipeline_step(
                conversation_id=conversation_id,
                agent_id=self.critic.agent_id,
                agent_type="CriticAgent",
                model_id=self.critic._model_id,
                pool_id=self.critic.pool_name,
                role="assistant",
                content=critic_draft,
                step_name="critic",
                trace_metadata={"user_prompt": user_prompt, "plan": plan, "draft": draft},
            )
        
        # ========================================================================
        # STAGE 4: Heavy Critic deep analysis (DEEP THINKING mode only)
        # ========================================================================
        if not self.deep_thinking:
            return
            
        logger.info(f"Pipeline {self.pipeline_id}: Stage 4 - Heavy Critic deep analysis...")
        
        heavy_critic_prompt = (
            f"Original User Request: {user_prompt}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Executor's Draft:\n{draft}\n\n"
            f"Critic's Polish:\n{critic_draft}\n\n"
            f"Provide the final deep-dive analysis and enhancement."
        )
        
        heavy_critic_chunks: list[str] = []
        
        try:
            async for chunk in self.heavy_critic.generate_stream(
                user_prompt=heavy_critic_prompt,
                history=history if history else None,
                **kwargs,
            ):
                heavy_critic_chunks.append(chunk)
                yield chunk
            
            full_heavy_response = "".join(heavy_critic_chunks)
            logger.info(
                f"Pipeline {self.pipeline_id}: Heavy Critic completed "
                f"({len(full_heavy_response)} chars)"
            )
            
            elapsed_ms = (
                datetime.now(timezone.utc) - pipeline_start
            ).total_seconds() * 1000
            logger.info(
                f"Pipeline {self.pipeline_id}: Deep thinking completed in {elapsed_ms:.0f}ms"
            )
            
        except Exception as e:
            logger.error(f"Pipeline {self.pipeline_id}: Heavy Critic failed: {e}")
            yield f"[ERROR: Heavy Critic failed: {e}]"
            return

    async def close(self) -> None:
        """Clean up all agents."""
        await self.planner.close()
        await self.executor.close()
        await self.critic.close()
        await self.heavy_critic.close()
        logger.info(f"Pipeline {self.pipeline_id}: Closed")
