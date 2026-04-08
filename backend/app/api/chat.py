"""Chat API endpoints for streaming LLM interactions."""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.agents.swarm import SwarmPipeline
from backend.app.core.config import get_settings
from backend.app.runtimes.llama_cpp import LlamaCppProvider
from backend.app.runtimes.process_manager import get_process_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role of the message sender (system/user/assistant)")
    content: str = Field(..., description="Content of the message")


class ChatStreamRequest(BaseModel):
    """Request body for streaming chat completion."""
    messages: list[ChatMessage] = Field(
        ...,
        description="List of conversation messages",
        min_length=1,
    )
    pool_name: str = Field(
        ...,
        description="Name of the model pool to use (e.g., 'worker_pool', 'orchestrator_pool')",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )


@router.post("/stream")
async def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
    """Stream a chat completion from an LLM.
    
    This endpoint:
    1. Looks up the pool configuration
    2. Starts the first model in the pool (respecting VRAM limits)
    3. Streams the LLM response via SSE
    
    Args:
        request: Chat stream request with messages and pool name
        
    Returns:
        StreamingResponse with text/event-stream content type
    """
    settings = get_settings()
    process_manager = get_process_manager()
    
    # Step 1: Validate and get pool configuration
    try:
        pool = settings.get_pool(request.pool_name)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pool not found: {request.pool_name}",
        ) from e
    
    if not pool.models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pool '{request.pool_name}' has no models configured",
        )
    
    # Step 2: Get the first model from the pool
    model_id = pool.models[0]
    try:
        model_config = settings.get_model(model_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model '{model_id}' from pool not found in configuration",
        ) from e
    
    # Step 3: Start the model process (respects VRAM budget)
    agent_id = f"chat_{model_id}_{uuid.uuid4().hex[:8]}"
    
    try:
        process = await process_manager.start_model(model_id, agent_id)
        logger.info(
            f"Started model {model_id} for chat session {agent_id} "
            f"on port {process.port}"
        )
    except RuntimeError as e:
        if "Insufficient VRAM" in str(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Insufficient VRAM to start model. {e}",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start model: {e}",
        ) from e
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing required file: {e}",
        ) from e
    
    # Step 4: Initialize the provider with the dynamic URL
    base_url = f"http://127.0.0.1:{process.port}"
    provider = LlamaCppProvider(base_url=base_url)
    
    # Step 5: Prepare messages for the LLM
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages
    ]
    
    # Step 6: Define the streaming generator
    async def stream_generator():
        """Generate SSE stream from LLM provider."""
        try:
            async for chunk in provider.generate_stream(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            ):
                # SSE format: data: <content>\n\n
                # Escape newlines in chunk for SSE format
                safe_chunk = chunk.replace("\n", "\\n").replace("\r", "\\r")
                yield f"data: {safe_chunk}\n\n"
            
            # Send completion marker
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            error_msg = str(e).replace("\n", "\\n")
            yield f"data: [ERROR] {error_msg}\n\n"
        finally:
            # Clean up provider
            await provider.close()
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


class SwarmRequest(BaseModel):
    """Request body for swarm pipeline."""
    prompt: str = Field(
        ...,
        description="The user's prompt to process through the swarm pipeline",
        min_length=1,
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation ID for continuity. New UUID generated if not provided.",
    )
    deep_thinking: bool = Field(
        default=False,
        description="Enable 4-stage deep thinking mode with Gemma 26B heavy critic",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )


@router.post("/swarm")
async def swarm_chat(request: SwarmRequest) -> StreamingResponse:
    """Process a prompt through the swarm pipeline with memory.
    
    Standard Mode (3-stage):
    1. Planner Agent: Creates a strategy plan
    2. Executor Agent: Builds a response based on plan and history
    3. Critic Agent: Polishes and streams the final result
    
    Deep Thinking Mode (4-stage, deep_thinking=true):
    1. Planner Agent: Creates a strategy plan
    2. Executor Agent: Builds a response based on plan and history
    3. Critic Agent: Initial polish (internal)
    4. Heavy Critic Agent (Gemma 26B): Final deep-dive analysis (streamed)
    
    The final output is streamed directly to the user for optimal UX.
    All steps are saved to the database with the conversation_id for persistence.
    
    Args:
        request: Swarm request with prompt, optional conversation_id, and deep_thinking flag
        
    Returns:
        StreamingResponse with the final polished output.
        The conversation_id is returned in the X-Conversation-ID header.
    """
    # Generate or use provided conversation_id
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    
    logger.info(
        f"Swarm API: Processing prompt for conversation {conversation_id}: "
        f"{request.prompt[:50]}..."
    )
    
    # Initialize the swarm pipeline
    # Light agents use worker_pool; heavy_critic uses heavy_pool (Gemma 26B)
    pipeline = SwarmPipeline(
        planner_pool="worker_pool",
        executor_pool="worker_pool",
        critic_pool="worker_pool",
        heavy_critic_pool="heavy_pool",
        deep_thinking=request.deep_thinking,
    )
    
    async def stream_generator():
        """Generate stream from the swarm pipeline."""
        try:
            # Yield conversation_id as first chunk for client to capture
            yield f"data: [CONVERSATION_ID:{conversation_id}]\n\n"
            
            async for chunk in pipeline.run_pipeline(
                user_prompt=request.prompt,
                conversation_id=conversation_id,
                temperature=request.temperature,
            ):
                # SSE format
                safe_chunk = chunk.replace("\n", "\\n").replace("\r", "\\r")
                yield f"data: {safe_chunk}\n\n"
            
            # Send completion marker
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Swarm API: Pipeline error: {e}")
            error_msg = str(e).replace("\n", "\\n")
            yield f"data: [ERROR] {error_msg}\n\n"
        finally:
            # Clean up pipeline resources
            await pipeline.close()
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Conversation-ID": conversation_id,
        },
    )


@router.get("/conversations", include_in_schema=True)
async def list_conversations(limit: int = 20) -> dict[str, Any]:
    """List recent conversations from the database.
    
    Args:
        limit: Maximum number of conversations to return
        
    Returns:
        Dictionary with conversation list
    """
    from backend.app.database.db import async_session_maker, ConversationTrace
    from sqlalchemy import select, func, desc
    
    try:
        async with async_session_maker() as session:
            # Get unique conversation_ids with their latest message
            stmt = (
                select(
                    ConversationTrace.trace_metadata["conversation_id"].as_string().label("conv_id"),
                    func.max(ConversationTrace.created_at).label("last_updated"),
                    func.count(ConversationTrace.id).label("message_count"),
                )
                .where(ConversationTrace.trace_metadata["conversation_id"].is_not(None))
                .group_by(ConversationTrace.trace_metadata["conversation_id"].as_string())
                .order_by(desc(func.max(ConversationTrace.created_at)))
                .limit(limit)
            )
            
            result = await session.execute(stmt)
            conversations = []
            
            for row in result:
                # Get the first user message as title
                title_stmt = (
                    select(ConversationTrace.content)
                    .where(
                        ConversationTrace.trace_metadata["conversation_id"].as_string() == row.conv_id,
                        ConversationTrace.role == "user"
                    )
                    .order_by(ConversationTrace.created_at)
                    .limit(1)
                )
                title_result = await session.execute(title_stmt)
                first_message = title_result.scalar()
                
                title = first_message[:50] + "..." if first_message and len(first_message) > 50 else (first_message or "Untitled")
                
                conversations.append({
                    "id": row.conv_id,
                    "title": title,
                    "last_updated": row.last_updated.isoformat() if row.last_updated else None,
                    "message_count": row.message_count,
                })
            
            return {
                "conversations": conversations,
                "total": len(conversations),
            }
            
    except Exception as e:
        logger.error(f"Failed to fetch conversations: {e}")
        return {"conversations": [], "total": 0, "error": str(e)}


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    """Get all messages for a specific conversation.
    
    Args:
        conversation_id: The conversation ID
        
    Returns:
        Dictionary with conversation messages
    """
    from backend.app.database.db import async_session_maker, ConversationTrace
    from sqlalchemy import select, desc
    
    try:
        async with async_session_maker() as session:
            # Get all messages for this conversation
            stmt = (
                select(ConversationTrace)
                .where(
                    ConversationTrace.trace_metadata["conversation_id"].as_string() == conversation_id
                )
                .where(ConversationTrace.role.in_(["user", "assistant"]))
                .order_by(ConversationTrace.created_at)
            )
            
            result = await session.execute(stmt)
            traces = result.scalars().all()
            
            messages = []
            for trace in traces:
                # Skip internal pipeline steps (planner/executor/critic)
                step = trace.trace_metadata.get("step", "") if trace.trace_metadata else ""
                if step in ["planner", "executor", "critic"]:
                    continue
                    
                messages.append({
                    "id": trace.id,
                    "role": trace.role,
                    "content": trace.content,
                    "timestamp": trace.created_at.isoformat() if trace.created_at else None,
                })
            
            return {
                "conversation_id": conversation_id,
                "messages": messages,
                "total": len(messages),
            }
            
    except Exception as e:
        logger.error(f"Failed to fetch conversation {conversation_id}: {e}")
        return {"conversation_id": conversation_id, "messages": [], "total": 0, "error": str(e)}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, Any]:
    """Delete a conversation and all its messages.
    
    Args:
        conversation_id: The conversation ID to delete
        
    Returns:
        Dictionary with deletion status
    """
    from backend.app.database.db import async_session_maker, ConversationTrace
    from sqlalchemy import delete
    
    try:
        async with async_session_maker() as session:
            # Delete all traces for this conversation
            stmt = (
                delete(ConversationTrace)
                .where(
                    ConversationTrace.trace_metadata["conversation_id"].as_string() == conversation_id
                )
            )
            
            result = await session.execute(stmt)
            await session.commit()
            
            deleted_count = result.rowcount
            logger.info(f"Deleted conversation {conversation_id}: {deleted_count} messages removed")
            
            return {
                "success": True,
                "conversation_id": conversation_id,
                "deleted_messages": deleted_count,
            }
            
    except Exception as e:
        logger.error(f"Failed to delete conversation {conversation_id}: {e}")
        return {"success": False, "conversation_id": conversation_id, "error": str(e)}


@router.get("/pools")
async def list_pools() -> dict[str, Any]:
    """List available model pools.
    
    Returns:
        Dictionary of pool configurations
    """
    settings = get_settings()
    
    return {
        "pools": {
            pool_id: {
                "name": pool.name,
                "description": pool.description,
                "models": pool.models,
                "max_concurrent": pool.max_concurrent,
            }
            for pool_id, pool in settings.pools.pools.items()
        }
    }
