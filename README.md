# Multi-Agent Swarm Intelligence System (GGUF Optimized)

[![Status](https://img.shields.io/badge/Status-Operational-brightgreen)]()
[![Backend](https://img.shields.io/badge/Backend-FastAPI-blue)]()
[![Frontend](https://img.shields.io/badge/Frontend-Next.js-black)]()
[![Engine](https://img.shields.io/badge/Engine-llama.cpp-orange)]()

## ?? Overview
This is a high-performance, local-first multi-agent swarm system designed to squeeze maximum intelligence out of constrained hardware. By orchestrating multiple GGUF models (like Gemma, Mistral, Llama) through a hierarchical pipeline, we achieve reasoning capabilities that far exceed a single small model.

## ?? Key Features

### 1. **Intelligence Overhaul: The Reflection Loop**
The system is now 'self-aware'. Every output goes through a mandatory **Reflection Step**:
- **Executor** creates a draft.
- **Critic** evaluates the draft (JSON scoring 1-10 + feedback).
- **Auto-Retry**: If the score is < 7, the Executor receives the feedback and *must* regenerate the response once more with the corrections.

### 2. **Token-Aware Swarm Memory**
Forget simple message limits. The ContextManager now monitors actual token density:
- **Dynamic Context**: It loads the most recent history up to a strict token limit (default: 2000 tokens).
- **Stability**: Prevents exceeding the GGUF model's context window, ensuring no "hallucination loops" due to truncated context.

### 3. **PromptManager (Brain Core)**
Centralized prompt logic inspired by industry standards:
- **Role-Based instructions**: All agent personalities are defined in configs/agents.yaml.
- **Dynamic Injection**: Uses template placeholders {{history}} and {{input}} for surgical prompt construction.

### 4. **VRAM Protection & Zombie Killer**
- **Concurrency Lock**: syncio.Lock() ensures that VRAM isn't double-allocated if multiple requests hit the server simultaneously.
- **Windows Zombie Killer**: Automatic forceful cleanup of orphaned llama-server.exe processes on shutdown or failure using 	askkill.
- **Heartbeat Monitor**: Self-healing runtime that detects unresponsive models and restarts them automatically.

## ?? Tech Stack
- **Core**: Python 3.10+ / FastAPI
- **LLM Engine**: llama.cpp (llama-server)
- **Database**: SQLite with SQLAlchemy (Async)
- **Frontend**: Next.js 14, TailwindCSS, TypeScript

## ?? Setup & Installation

### Backend
1. Install dependencies: pip install -r requirements.txt
2. Configure models in ackend/app/core/config.py
3. Start API: python -m backend.main

### Frontend
1. Install dependencies: 
pm install
2. Start Dev: 
pm run dev

## ?? Agent Roles
- **Planner**: Architect of the solution.
- **Executor**: The builder. Implements the code/text.
- **Critic**: The perfectionist. Quality assurance & feedback loop.
- **Heavy Critic**: (Deep Thinking Mode) 26B+ model for final philosophical/technical deep-dives.

---
**Build for the future. Run on your desk.** ?
