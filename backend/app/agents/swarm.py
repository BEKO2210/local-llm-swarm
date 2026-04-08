"""Hierarchical swarm pipeline for multi-agent text refinement with memory."""

import logging
import uuid
import json
import re
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, desc

from backend.app.agents.base import BaseAgent
from backend.app.database.db import AgentTrace, ConversationTrace, async_session_maker
from backend.app.core.brain import brain

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages conversation history and context retrieval from the database."""

    @staticmethod
    async def fetch_history(
        conversation_id: str,
        limit: int = 10,
        max_tokens: int = 2000,
    ) -> list[dict[str, str]]:
        """Fetch the last N messages for a conversation, limited by token count."""
        try:
            async with async_session_maker() as session:
                stmt = (
                    select(ConversationTrace)
                    .where(ConversationTrace.trace_metadata["conversation_id"].as_string() == conversation_id)
                    .order_by(desc(ConversationTrace.created_at))
                    .limit(limit)
                )

                result = await session.execute(stmt)
                traces = result.scalars().all()

                history = []
                current_tokens = 0
                # Simple token estimation: 4 chars per token
                for trace in traces:
                    content_len = len(trace.content)
                    est_tokens = content_len // 4
                    if current_tokens + est_tokens > max_tokens:
                        break
                    history.append({"role": trace.role, "content": trace.content})
                    current_tokens += est_tokens

                history.reverse()
                logger.debug(f"ContextManager: Fetched {len(history)} messages ({current_tokens} est. tokens)")
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
    """A hierarchical pipeline with persistent memory and reflection step."""

    def __init__(
        self,
        planner_pool: str = "worker_pool",
        executor_pool: str = "worker_pool",
        critic_pool: str = "worker_pool",
        heavy_critic_pool: str = "heavy_pool",
        deep_thinking: bool = False,
    ):
        self.pipeline_id = f"swarm_{uuid.uuid4().hex[:8]}"
        self.context_manager = ContextManager()
        self.deep_thinking = deep_thinking

        self.planner = BaseAgent(pool_name=planner_pool, agent_type="PlannerAgent")
        self.executor = BaseAgent(pool_name=executor_pool, agent_type="ExecutorAgent")
        self.critic = BaseAgent(pool_name=critic_pool, agent_type="CriticAgent")
        self.heavy_critic = BaseAgent(pool_name=heavy_critic_pool, agent_type="HeavyCriticAgent")

    async def _save_pipeline_step(self, conversation_id, agent, role, content, step_name, metadata=None):
        if agent._model_id:
            try:
                async with async_session_maker() as session:
                    agent_trace = AgentTrace(
                        id=agent.agent_id,
                        agent_type=agent.agent_type,
                        model_id=agent._model_id,
                        pool_id=agent.pool_name,
                        status="completed",
                        vram_cost_mb=0,
                        port=None,
                        pid=None,
                        started_at=datetime.now(timezone.utc),
                        stopped_at=datetime.now(timezone.utc),
                    )
                    await session.merge(agent_trace)
                    await session.commit()
                    await self.context_manager.save_message(
                        conversation_id=conversation_id,
                        agent_id=agent.agent_id,
                        role=role,
                        content=content,
                        step_name=step_name,
                        trace_metadata={
                            "pipeline_id": self.pipeline_id,
                            "agent_type": agent.agent_type,
                            **(metadata or {}),
                        },
                    )
            except Exception as e:
                logger.error(f"Failed to save step {step_name}: {e}")

    async def run_pipeline(
        self,
        user_prompt: str,
        conversation_id: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        history = await self.context_manager.fetch_history(conversation_id, limit=10, max_tokens=2000)
        history_str = json.dumps(history)

        # STAGE 1: Planner
        planner_prompt = brain.get_prompt("planner", history=history_str, user_input=user_prompt)
        plan = await self.planner.generate(user_prompt=planner_prompt, **kwargs)
        await self._save_pipeline_step(conversation_id, self.planner, "assistant", plan, "planner")

        # STAGE 2: Executor + Reflection Loop
        executor_input = f"User: {user_prompt}\nPlan: {plan}"
        executor_prompt = brain.get_prompt("executor", history=history_str, user_input=executor_input)
        draft = await self.executor.generate(user_prompt=executor_prompt, **kwargs)

        # REFLECTION STEP
        critic_prompt = brain.get_prompt("critic", user_input=draft)
        critic_feedback_json = await self.critic.generate(user_prompt=critic_prompt, **kwargs)
        
        try:
            # Extract JSON from potential markdown
            match = re.search(r'\{.*\}', critic_feedback_json, re.DOTALL)
            if match:
                feedback_data = json.loads(match.group())
                quality = feedback_data.get("quality", 10)
                feedback = feedback_data.get("feedback", "")
            else:
                quality = 10
                feedback = ""
        except:
            quality = 10
            feedback = ""

        if quality < 7:
            logger.info(f"Quality {quality} < 7. Retrying execution with feedback.")
            executor_prompt = brain.get_prompt("executor", history=history_str, user_input=executor_input, feedback=feedback)
            draft = await self.executor.generate(user_prompt=executor_prompt, **kwargs)

        await self._save_pipeline_step(conversation_id, self.executor, "assistant", draft, "executor")

        # STAGE 3: Final Stream (Critic or Heavy Critic)
        if self.deep_thinking:
            heavy_prompt = brain.get_prompt("heavy_critic", history=history_str, user_input=draft)
            async for chunk in self.heavy_critic.generate_stream(user_prompt=heavy_prompt, **kwargs):
                yield chunk
        else:
            # Simple pass-through or final polish stream
            async for chunk in self.critic.generate_stream(user_prompt=f"Final polish of: {draft}", **kwargs):
                yield chunk

    async def close(self) -> None:
        await self.planner.close()
        await self.executor.close()
        await self.critic.close()
        await self.heavy_critic.close()
