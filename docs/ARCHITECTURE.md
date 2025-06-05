# AGI Project Architecture

This document summarises the key modules of the codebase and how the agent executes tasks.

## Core Modules
- **core/agent_loop.py** – Main agent class handling task lifecycle, planning, execution and memory integration. It uses `GoalRouter`, `TaskOrchestrator`, `Executor`, and `Memory`.
- **core/memory.py** & **core/vector_memory.py** – Lightweight JSONL memory store with optional vector search for similarity lookup.
- **core/evolver.py** – Handles code evolution of project files via LLMs with safety checks and scoring.
- **core/tools/** – Collection of callable tools exposed to the planner (shell, evolution, browser, etc.) with a global registry in `tool_registry.py`.
- **core/chat.py** – Helper functions to call language models with response sanitisation, scoring and retry logic.
- **core/goal_router.py** – Routes goals to relevant files using tag heuristics and vector memory.
- **core/task_orchestrator.py** – Clusters and chains tasks based on metadata and dependencies.

Entry point scripts such as `main.py` and `agents/sentient_duo.py` create an `Agent` instance and start the main loop.

## Execution Flow
1. **Task Creation** – Goals are loaded from `tasks.jsonl` or generated dynamically. Each is converted into a `Task` object with metadata and tags.
2. **Planning** – `Agent._generate_plan` queries the LLM to map a goal into a sequence of tool invocations.
3. **Tool Execution** – Steps are executed via `tool_registry.run_tool` which validates arguments and logs calls. Results are normalised.
4. **Memory Update** – Outcomes are appended to `Memory`; vector embeddings enable similarity search for future retries.
5. **Retry & Reflection** – Failed tasks are retried using `Agent.retry_task` which can consult past successful entries to adjust the goal. Reflection agents score evolutions and update rewards.
6. **Reward** – `RewardCalculator` computes scores for evolutions which influence pruning and future planning.

## Asynchronous Behaviour
The main agent loop can run continuously (see `Agent.start` in `agent_loop.py`). Some tools such as chat streaming or watchers spawn background threads, but the core task processing is synchronous.

## Available Tools
The tool registry exposes several helper functions used by the planner:

- `file_read(path)` – return the contents of a text file.
- `internet_fetch(url)` – download raw text from a URL.
- `os_metrics()` – gather basic CPU, memory and disk metrics.
- `repo_scan(pattern="*")` – list repository files matching a glob.

Additional utilities like `run_shell`, `evolve_file` and `web_search` are also registered and can be called by the agent.
