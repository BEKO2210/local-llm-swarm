# Swarm Intelligence: The GGUF-First Orchestrator

<p align="center">
  <img src="https://raw.githubusercontent.com/BEKO2210/local-llm-swarm/main/assets/brain.png" width="120" alt="Swarm Logo" />
</p>

<p align="center">
  <a href="https://github.com/BEKO2210/local-llm-swarm/stargazers"><img src="https://img.shields.io/github/stars/BEKO2210/local-llm-swarm?style=for-the-badge&color=yellow" alt="Stars"></a>
  <a href="https://github.com/BEKO2210/local-llm-swarm/network/members"><img src="https://img.shields.io/github/forks/BEKO2210/local-llm-swarm?style=for-the-badge&color=blue" alt="Forks"></a>
  <a href="https://github.com/BEKO2210/local-llm-swarm/issues"><img src="https://img.shields.io/github/issues/BEKO2210/local-llm-swarm?style=for-the-badge&color=red" alt="Issues"></a>
</p>

---

##  Overview

**Gemini Swarm Intelligence** is a professional-grade, local-first multi-agent orchestration system. It is engineered to extract maximum cognitive performance from constrained consumer hardware by leveraging a **hierarchical swarm pipeline** of GGUF-quantized models.

Unlike standard LLM interfaces, this system treats AI as a **unified processor**, where specialized agents (Planner, Executor, Critic) work in a synchronized loop to ensure high-fidelity output, security, and logical consistency.

---

## ?? System Architecture

`mermaid
graph TD
    A[User Prompt] --> B{Planner}
    B -->|Strategy| C[Executor]
    C -->|Draft| D{Critic Loop}
    D -->|Score < 7| C
    D -->|Score >= 7| E[Final Output]
    E --> F((User))
    
    subgraph "VRAM Management"
    G[Process Manager] --- H[llama-server 1]
    G --- I[llama-server 2]
    G --- J[VRAM Lock]
    end
`

---

##  Advanced Features

###  **The Reflection Step (Self-Correction)**
The system is now **fully self-aware**. We implemented a "Logic Guard" where the **Critic Agent** evaluates the **Executor's** work using structured JSON feedback. 
- **Quality Scoring:** 1-10 scale.
- **Auto-Regeneration:** If a draft scores below 7, the Executor is automatically re-invoked with the specific feedback to fix errors before the user even sees it.

###  **Token-Aware Context Memory**
Traditional "last 10 messages" memory is obsolete. Our ContextManager uses **dynamic token density monitoring**:
- It calculates the estimated token load and fills the context window up to a strict limit (e.g., 2000 tokens).
- Prevents model degradation and "forgetting" by ensuring the prompt always fits perfectly within the GGUF model's 
_ctx.

###  **VRAM Concurrency Protection**
Running multiple LLMs locally is dangerous for your GPU. We solved this with:
- **syncio.Lock() System:** Prevents simultaneous model allocations that would crash your drivers.
- **Hybrid Offloading:** Automatic calculation of how many layers fit in your VRAM vs. System RAM.

###  **Windows Zombie Killer**
Local development often leaves orphaned processes. Our **Lifespan Manager** detects and kills "zombie" llama-server.exe instances automatically on shutdown, keeping your system clean and your VRAM free.

---

##  Tech Stack

| Component | Technology |
| :--- | :--- |
| **Backend** | FastAPI (Python 3.10+) |
| **Inference** | llama.cpp (GGUF Optimized) |
| **Database** | SQLAlchemy 2.0 + SQLite (Async) |
| **Frontend** | Next.js 14 (App Router) + Tailwind CSS |
| **Logic** | Custom Swarm Orchestrator |

---

##  Setup & Installation

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- [llama-server.exe](https://github.com/ggerganov/llama.cpp/releases) (ensure it's in your PATH or configured in core/config.py)

### 2. Backend Installation
`ash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
`

### 3. Frontend Installation
`ash
cd frontend
npm install
npm run dev
`

---

##  Configurable Agents (configs/agents.yaml)

You can define your own agent personalities and logic flows:

`yaml
critic:
  role: "QA Lead"
  goal: "Logic & Security Verification"
  prompt_template: |
    Role: {{role}}
    Evaluate the following: {{input}}
    Provide JSON: { "quality": 1-10, "feedback": "..." }
`

---

##  License
Distributed under the MIT License. See LICENSE for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

---
**Build for the future. Run on your desk.** ?
