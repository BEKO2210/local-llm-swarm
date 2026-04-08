"""Process manager for self-managed llama-server runtimes."""

import asyncio
import logging
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings, ModelConfig

logger = logging.getLogger(__name__)


class LlamaServerProcess:
    """Manages a single llama-server subprocess instance."""

    def __init__(
        self,
        model_id: str,
        model_config: ModelConfig,
        port: int,
        binary_path: str,
    ):
        self.model_id = model_id
        self.model_config = model_config
        self.port = port
        self.binary_path = Path(binary_path)
        
        self.process: subprocess.Popen | None = None
        self.pid: int | None = None
        self.started_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def vram_cost_mb(self) -> int:
        """Return the VRAM cost for this model instance."""
        return self.model_config.vram_estimate_mb

    @property
    def is_running(self) -> bool:
        """Check if the process is currently running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def _calculate_gpu_layers(self) -> int:
        """Calculate optimal number of GPU layers based on VRAM constraints.
        
        For models that exceed available VRAM, we calculate how many layers
        can fit and offload those to GPU, running the rest on CPU.
        
        Returns:
            Number of layers to offload to GPU (-1 means all layers)
        """
        from backend.app.runtimes.process_manager import get_process_manager
        
        process_manager = get_process_manager()
        available_vram = process_manager.available_vram_mb
        model_vram = self.model_config.vram_estimate_mb
        
        # Get requested GPU layers from config (default to -1 = all)
        requested_layers = self.model_config.parameters.get("n_gpu_layers", -1)
        
        # If model fits comfortably in VRAM, use requested layers
        if model_vram <= available_vram:
            logger.info(
                f"Model {self.model_id} fits in VRAM "
                f"({model_vram}MB <= {available_vram}MB), "
                f"offloading {requested_layers} layers"
            )
            return requested_layers
        
        # Model exceeds VRAM - need to calculate partial offloading
        # Estimate: typical models have ~30-40 layers, VRAM usage is roughly proportional
        # We'll estimate ~250-300MB per layer for quantized models
        estimated_layers = 35  # Conservative estimate for 26B model
        vram_per_layer = model_vram / estimated_layers
        
        # Calculate how many layers fit (leave 500MB buffer for context overhead)
        safe_vram = available_vram - 500
        layers_that_fit = max(0, int(safe_vram / vram_per_layer))
        
        # Target 70-80% GPU utilization for overflow models
        target_layers = int(layers_that_fit * 0.75)
        
        logger.warning(
            f"Model {self.model_id} exceeds VRAM "
            f"({model_vram}MB > {available_vram}MB). "
            f"Offloading {target_layers}/{estimated_layers} layers to GPU, "
            f"rest will run on CPU."
        )
        
        return target_layers

    def _build_args(self) -> list[str]:
        """Build command-line arguments for llama-server."""
        args = [
            str(self.binary_path),
            "--model", self.model_config.gguf_path,
            "--port", str(self.port),
            "--host", "127.0.0.1",
        ]
        
        # Calculate optimal GPU layers
        gpu_layers = self._calculate_gpu_layers()
        if gpu_layers != 0:
            args.extend(["-ngl", str(gpu_layers)])
        
        # Add model-specific parameters
        params = self.model_config.parameters
        if params.get("n_ctx"):
            args.extend(["--ctx-size", str(params["n_ctx"])])
        
        return args

    async def start(self, startup_timeout: int = 30) -> bool:
        """Start the llama-server process."""
        async with self._lock:
            if self.is_running:
                logger.warning(f"Process for {self.model_id} is already running")
                return True

            if not self.binary_path.exists():
                raise FileNotFoundError(
                    f"llama-server binary not found: {self.binary_path}"
                )

            gguf_full_path = Path(self.model_config.gguf_path)
            if not gguf_full_path.exists():
                raise FileNotFoundError(
                    f"GGUF model not found: {gguf_full_path}"
                )

            args = self._build_args()
            logger.info(f"Starting llama-server for {self.model_id} on port {self.port}")
            logger.debug(f"Command: {' '.join(args)}")

            try:
                self.process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
                )
                self.pid = self.process.pid
                self.started_at = time.time()

                # Wait briefly to catch immediate failures
                await asyncio.sleep(0.5)
                
                if not self.is_running:
                    stdout, stderr = self.process.communicate(timeout=5)
                    raise RuntimeError(
                        f"llama-server failed to start: {stderr or stdout}"
                    )

                logger.info(
                    f"llama-server started for {self.model_id} "
                    f"(PID: {self.pid}, Port: {self.port}, "
                    f"VRAM: {self.vram_cost_mb} MB)"
                )
                return True

            except Exception as e:
                logger.error(f"Failed to start llama-server for {self.model_id}: {e}")
                self.process = None
                self.pid = None
                raise

    async def stop(self, shutdown_timeout: int = 10) -> bool:
        """Stop the llama-server process and free VRAM."""
        async with self._lock:
            if self.process is None:
                return True

            if not self.is_running:
                logger.debug(f"Process for {self.model_id} is not running")
                self.process = None
                self.pid = None
                return True

            logger.info(f"Stopping llama-server for {self.model_id} (PID: {self.pid})")

            try:
                # Try graceful termination first
                if hasattr(signal, 'SIGTERM'):
                    self.process.terminate()
                else:
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)

                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=shutdown_timeout)
                    logger.info(f"llama-server for {self.model_id} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"llama-server for {self.model_id} did not stop gracefully, "
                        f"forcing kill"
                    )
                    self.process.kill()
                    self.process.wait(timeout=5)
                    logger.info(f"llama-server for {self.model_id} killed")

                return True

            except Exception as e:
                logger.error(f"Error stopping llama-server for {self.model_id}: {e}")
                return False

            finally:
                self.process = None
                self.pid = None
                self.started_at = None

    def get_status(self) -> dict[str, Any]:
        """Get current status of the process."""
        return {
            "model_id": self.model_id,
            "is_running": self.is_running,
            "pid": self.pid,
            "port": self.port,
            "vram_cost_mb": self.vram_cost_mb,
            "started_at": self.started_at,
            "uptime_seconds": time.time() - self.started_at if self.started_at else None,
        }


class ProcessManager:
    """Manages multiple llama-server processes with VRAM tracking."""

    def __init__(self):
        self._processes: dict[str, LlamaServerProcess] = {}
        self._port_counter = 8081
        self._lock = asyncio.Lock()
        
        settings = get_settings()
        self._runtime_config = settings.runtime
        self._port_range = self._runtime_config.llama_server.port_range
        self._binary_path = self._runtime_config.llama_server.binary_path
        self._startup_timeout = self._runtime_config.process.startup_timeout_seconds
        self._shutdown_timeout = self._runtime_config.process.shutdown_timeout_seconds

    @property
    def total_vram_used_mb(self) -> int:
        """Calculate total VRAM currently used by all running processes.
        
        Each unique process is counted only once, even if multiple agents
        reference it.
        """
        seen_pids = set()
        total = 0
        for proc in self._processes.values():
            if proc.is_running and proc.pid not in seen_pids:
                seen_pids.add(proc.pid)
                total += proc.vram_cost_mb
        return total

    @property
    def available_vram_mb(self) -> int:
        """Calculate available VRAM based on budget."""
        budget = self._runtime_config.vram.max_system_budget_mb
        safety = self._runtime_config.vram.safety_margin_mb
        return budget - safety - self.total_vram_used_mb

    def _get_next_port(self) -> int:
        """Get the next available port in the configured range."""
        port = self._port_counter
        self._port_counter += 1
        
        if self._port_counter > self._port_range["max"]:
            self._port_counter = self._port_range["min"]
        
        return port

    async def start_model(
        self,
        model_id: str,
        agent_id: str | None = None,
        allow_overflow: bool = True,
    ) -> LlamaServerProcess:
        """Start a model instance, with optional VRAM overflow handling.
        
        Args:
            model_id: The model to start
            agent_id: Optional custom agent ID
            allow_overflow: If True, allows models larger than VRAM to start
                          with partial GPU offloading (CPU+GPU hybrid)
        """
        from backend.app.core.config import get_settings
        
        settings = get_settings()
        model_config = settings.get_model(model_id)
        
        # Check VRAM availability
        if model_config.vram_estimate_mb > self.available_vram_mb:
            if not allow_overflow:
                raise RuntimeError(
                    f"Insufficient VRAM to start {model_id}. "
                    f"Required: {model_config.vram_estimate_mb} MB, "
                    f"Available: {self.available_vram_mb} MB"
                )
            logger.warning(
                f"Model {model_id} ({model_config.vram_estimate_mb}MB) exceeds "
                f"available VRAM ({self.available_vram_mb}MB). "
                f"Starting with partial GPU offloading (CPU+GPU hybrid mode)."
            )

        instance_id = agent_id or f"{model_id}_{int(time.time())}"
        
        async with self._lock:
            if instance_id in self._processes:
                existing = self._processes[instance_id]
                if existing.is_running:
                    logger.info(f"Process for {instance_id} already running")
                    return existing

            port = self._get_next_port()
            process = LlamaServerProcess(
                model_id=model_id,
                model_config=model_config,
                port=port,
                binary_path=self._binary_path,
            )

            await process.start(startup_timeout=self._startup_timeout)
            self._processes[instance_id] = process

            logger.info(
                f"Started {model_id} as {instance_id}. "
                f"VRAM used: {self.total_vram_used_mb} MB"
            )
            return process

    async def stop_model(self, instance_id: str) -> bool:
        """Stop a model instance and free its VRAM."""
        async with self._lock:
            if instance_id not in self._processes:
                logger.warning(f"No process found for instance {instance_id}")
                return False

            process = self._processes[instance_id]
            success = await process.stop(shutdown_timeout=self._shutdown_timeout)
            
            if success:
                del self._processes[instance_id]
                logger.info(
                    f"Stopped {instance_id}. "
                    f"VRAM used: {self.total_vram_used_mb} MB"
                )
            
            return success

    async def stop_all(self) -> None:
        """Stop all running model instances."""
        logger.info(f"Stopping all {len(self._processes)} model instances")
        
        # Stop all processes concurrently
        tasks = [
            self.stop_model(instance_id)
            for instance_id in list(self._processes.keys())
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("All model instances stopped")

    def get_process(self, instance_id: str) -> LlamaServerProcess | None:
        """Get a process by instance ID if it exists."""
        return self._processes.get(instance_id)

    def get_status(self) -> dict[str, Any]:
        """Get overall process manager status."""
        return {
            "total_processes": len(self._processes),
            "running_processes": sum(
                1 for p in self._processes.values() if p.is_running
            ),
            "total_vram_used_mb": self.total_vram_used_mb,
            "available_vram_mb": self.available_vram_mb,
            "processes": {
                instance_id: proc.get_status()
                for instance_id, proc in self._processes.items()
            },
        }


# Global process manager instance
_process_manager: ProcessManager | None = None


def get_process_manager() -> ProcessManager:
    """Get or create the global process manager instance."""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager
