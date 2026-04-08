"""SQLAlchemy 2.0 async database setup for swarm tracing."""

from collections.abc import AsyncGenerator
from datetime import datetime

from sqlalchemy import String, Float, DateTime, ForeignKey, JSON, func
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DATABASE_URL = "sqlite+aiosqlite:///swarm_traces.db"


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class AgentTrace(Base):
    """Table for tracking agent instances and their lifecycle."""
    
    __tablename__ = "agent_traces"
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)
    model_id: Mapped[str] = mapped_column(String(50), nullable=False)
    pool_id: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="starting")
    
    # Resource tracking
    vram_cost_mb: Mapped[int] = mapped_column(nullable=False)
    port: Mapped[int | None] = mapped_column(nullable=True)
    pid: Mapped[int | None] = mapped_column(nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    conversations: Mapped[list["ConversationTrace"]] = relationship(
        back_populates="agent",
        cascade="all, delete-orphan"
    )


class ConversationTrace(Base):
    """Table for tracking conversations/interactions."""
    
    __tablename__ = "conversation_traces"
    
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("agent_traces.id"),
        nullable=False
    )
    
    # Conversation metadata
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    
    # Performance metrics
    tokens_prompt: Mapped[int | None] = mapped_column()
    tokens_completion: Mapped[int | None] = mapped_column()
    latency_ms: Mapped[float | None] = mapped_column(Float)
    
    # Raw request/response for debugging
    trace_metadata: Mapped[dict | None] = mapped_column(JSON)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    
    # Relationships
    agent: Mapped["AgentTrace"] = relationship(back_populates="conversations")


class VramAllocationLog(Base):
    """Table for tracking VRAM allocation/deallocation events."""
    
    __tablename__ = "vram_allocation_logs"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_id: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Allocation details
    action: Mapped[str] = mapped_column(String(20), nullable=False)  # "allocate" or "free"
    vram_amount_mb: Mapped[int] = mapped_column(nullable=False)
    vram_total_used_mb: Mapped[int] = mapped_column(nullable=False)
    vram_budget_mb: Mapped[int] = mapped_column(nullable=False)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )


# Database engine and session factory
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
