"""Pydantic-based configuration management for the swarm system."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Configuration for a single GGUF model."""
    name: str
    gguf_path: str
    vram_estimate_mb: int
    context_length: int
    chat_format: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    """Container for all model configurations."""
    models: dict[str, ModelConfig]


class PoolConfig(BaseModel):
    """Configuration for a logical model pool."""
    name: str
    description: str
    models: list[str]
    priority: int = 1
    max_concurrent: int = 1


class PoolsConfig(BaseModel):
    """Container for all pool configurations."""
    pools: dict[str, PoolConfig]


class LlamaServerConfig(BaseModel):
    """Configuration for llama-server binary."""
    binary_path: str
    host: str = "127.0.0.1"
    port_range: dict[str, int]


class VramConfig(BaseModel):
    """VRAM management configuration."""
    max_system_budget_mb: int
    safety_margin_mb: int = 500


class ProcessConfig(BaseModel):
    """Process management configuration."""
    startup_timeout_seconds: int = 30
    shutdown_timeout_seconds: int = 10
    health_check_interval_seconds: int = 5


class RuntimeConfig(BaseModel):
    """Container for runtime configuration."""
    llama_server: LlamaServerConfig
    vram: VramConfig
    process: ProcessConfig


class Settings(BaseSettings):
    """Main application settings loaded from YAML files."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # App settings
    app_name: str = "Multi-Agent Swarm"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Paths (relative to project root)
    models_config_path: Path = Path("configs/models.yaml")
    pools_config_path: Path = Path("configs/pools.yaml")
    runtime_config_path: Path = Path("configs/runtime.yaml")
    
    # Loaded configs (populated on init)
    models: ModelsConfig | None = None
    pools: PoolsConfig | None = None
    runtime: RuntimeConfig | None = None
    
    def load_yaml_configs(self) -> None:
        """Load all YAML configuration files."""
        self.models = self._load_models()
        self.pools = self._load_pools()
        self.runtime = self._load_runtime()
    
    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load a YAML file and return its contents."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def _load_models(self) -> ModelsConfig:
        """Load model configurations from YAML."""
        data = self._load_yaml(self.models_config_path)
        models = {
            key: ModelConfig(**value)
            for key, value in data.get("models", {}).items()
        }
        return ModelsConfig(models=models)
    
    def _load_pools(self) -> PoolsConfig:
        """Load pool configurations from YAML."""
        data = self._load_yaml(self.pools_config_path)
        pools = {
            key: PoolConfig(**value)
            for key, value in data.get("pools", {}).items()
        }
        return PoolsConfig(pools=pools)
    
    def _load_runtime(self) -> RuntimeConfig:
        """Load runtime configuration from YAML."""
        data = self._load_yaml(self.runtime_config_path)
        return RuntimeConfig(**data.get("runtime", {}))
    
    def get_model(self, model_id: str) -> ModelConfig:
        """Get a model configuration by ID."""
        if self.models is None:
            raise RuntimeError("Models config not loaded")
        
        if model_id not in self.models.models:
            raise ValueError(f"Unknown model: {model_id}")
        
        return self.models.models[model_id]
    
    def get_pool(self, pool_id: str) -> PoolConfig:
        """Get a pool configuration by ID."""
        if self.pools is None:
            raise RuntimeError("Pools config not loaded")
        
        if pool_id not in self.pools.pools:
            raise ValueError(f"Unknown pool: {pool_id}")
        
        return self.pools.pools[pool_id]


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.load_yaml_configs()
    return _settings
