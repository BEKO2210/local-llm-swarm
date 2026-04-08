"""Process manager for self-managed llama-server runtimes."""

import asyncio
import logging
import signal
import subprocess
import time
import os
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
        self.last_healthy_at = time.time()

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
        """Calculate optimal number of GPU layers based on VRAM constraints."""
        from backend.app.runtimes.process_manager import get_process_manager

        process_manager = get_process_manager()
        available_vram = process_manager.available_vram_mb
        model_vram = self.model_config.vram_estimate_mb

        requested_layers = self.model_config.parameters.get("n_gpu_layers", -1)

        if model_vram <= available_vram:
            logger.info(
                f"Model {self.model_id} fits in VRAM "
                f"({model_vram}MB <= {available_vram}MB), "
                f"offloading {requested_layers} layers"
            )
            return requested_layers

        estimated_layers = 35 
        vram_per_layer = model_vram / estimated_layers

        safe_vram = available_vram - 500
        layers_that_fit = max(0, int(safe_vram / vram_per_layer))

        target_layers = int(layers_that_fit * 0.75)

        logger.warning(
            f"Model {self.model_id} exceeds VRAM "
            f"({model_vram}MB > {available_vram}MB). "
            f"Offloading {target_layers}/{estimated_layers} layers to GPU."
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

        gpu_layers = self._calculate_gpu_layers()
        if gpu_layers != 0:
            args.extend(["-ngl", str(gpu_layers)])

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
                raise FileNotFoundError(f"llama-server binary not found: {self.binary_path}")

            args = self._build_args()
            logger.info(f"Starting llama-server for {self.model_id} on port {self.port}")

            try:
                self.process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )
                self.pid = self.process.pid
                self.started_at = time.time()
                self.last_healthy_at = time.time()

                await asyncio.sleep(0.5)

                if not self.is_running:
                    stdout, stderr = self.process.communicate(timeout=5)
                    raise RuntimeError(f"llama-server failed to start: {stderr or stdout}")

                logger.info(f"llama-server started for {self.model_id} (PID: {self.pid}, Port: {self.port})")
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
                self.process = None
                self.pid = None
                return True

            logger.info(f"Stopping llama-server for {self.model_id} (PID: {self.pid})")

            try:
                if os.name == 'nt':
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self.process.terminate()

                try:
                    self.process.wait(timeout=shutdown_timeout)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)

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
        seen_pids = set()
        total = 0
        for proc in self._processes.values():
            if proc.is_running and proc.pid not in seen_pids:
                seen_pids.add(proc.pid)
                total += proc.vram_cost_mb
        return total

    @property
    def available_vram_mb(self) -> int:
        budget = self._runtime_config.vram.max_system_budget_mb
        safety = self._runtime_config.vram.safety_margin_mb
        return budget - safety - self.total_vram_used_mb

    def _get_next_port(self) -> int:
        port = self._port_counter
        self._port_counter += 1
        if self._port_counter > self._port_range["max"]:
            self._port_counter = self._port_range["min"]
        return port

    def get_process(self, instance_id: str) -> LlamaServerProcess | None:
        """Get a process by instance ID if it exists."""
        return self._processes.get(instance_id)

    async def start_model(
        self,
        model_id: str,
        agent_id: str | None = None,
        allow_overflow: bool = True,
    ) -> LlamaServerProcess:
        """Start a model instance, with optional VRAM overflow handling."""
        async with self._lock:
            from backend.app.core.config import get_settings
            settings = get_settings()
            model_config = settings.get_model(model_id)

            if model_config.vram_estimate_mb > self.available_vram_mb:
                if not allow_overflow:
                    raise RuntimeError(f"Insufficient VRAM to start {model_id}.")
                logger.warning(f"Model {model_id} exceeds available VRAM. Partial offloading enabled.")

            instance_id = agent_id or f"{model_id}_{int(time.time())}"

            if instance_id in self._processes:
                existing = self._processes[instance_id]
                if existing.is_running:
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
            return process

    async def stop_model(self, instance_id: str) -> bool:
        async with self._lock:
            if instance_id not in self._processes:
                return False
            process = self._processes[instance_id]
            success = await process.stop(shutdown_timeout=self._shutdown_timeout)
            if success:
                del self._processes[instance_id]
            return success

    async def stop_all(self) -> None:
        tasks = [self.stop_model(instance_id) for instance_id in list(self._processes.keys())]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.cleanup_zombies()

    def cleanup_zombies(self):
        """Forceful cleanup of zombie llama-server processes on Windows."""
        if os.name == 'nt':
            logger.info("Cleaning up zombie llama-server processes...")
            try:
                subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe", "/T"], 
                               capture_output=True, check=False)
            except Exception as e:
                logger.error(f"Failed to cleanup zombies: {e}")

    async def monitor_heartbeat(self):
        """Monitor llama-server instances for health."""
        while True:
            for instance_id, proc in list(self._processes.items()):
                if proc.is_running:
                    # In a real scenario, we would check /health endpoint here
                    # If health check fails for 60s, restart
                    if time.time() - proc.last_healthy_at > 60:
                        logger.warning(f"Instance {instance_id} unresponsive for 60s. Restarting...")
                        await self.stop_model(instance_id)
                        await self.start_model(proc.model_id, agent_id=instance_id)
            await asyncio.sleep(10)

def get_process_manager() -> ProcessManager:
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager

_process_manager: ProcessManager | None = None

