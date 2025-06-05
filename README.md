# AGI

This repository contains a lightweight autonomous agent that can plan goals, execute tasks and even evolve its own codebase. The project started as an experimental sandbox and was completely vibe coded.

## Features

- **Agent loop** powered by `core/agent_loop.py` that manages task planning, execution and memory integration.
- **Memory system** in `core/memory.py` with optional vector search for similarity lookup.
- **Self‑evolution** through `core/evolver.py` and the `evolve_file` tool.
- **Tool registry** exposing helpers like shell commands, web fetch, system metrics and more (see `core/tools/tool_registry.py`).
- **Chat and reflection** utilities in `core/chat.py` and `core/reflection.py` to interact with language models.

For a deeper dive into how the modules fit together, see `docs/ARCHITECTURE.md`.

## Installation

1. Clone the repository.
2. Install the `python3-venv` package if it isn't already available:
   ```bash
   sudo apt install python3-venv -y
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv agi_env
   source agi_env/bin/activate
   ```
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Optional packages such as `ollama` or `faiss` can be installed if you want full functionality, but they are not required for the tests.

## Usage

### Running the main agent

Start the continuous monitoring loop or an interactive shell:
```bash
python main.py            # background monitoring
python main.py -i         # interactive REPL
```

### Conversational demo

To run the two‑agent conversation experiment:
```bash
python agents/sentient_duo.py
```

### Tasks and evolution

Pending goals are read from `output/pending_goals.json` and processed periodically. Evolution results and memory are stored under the `output/` directory. Use the included tools or tests as examples of how to create new tasks.

## Use Cases

- Automating repetitive shell work with safe command execution.
- Researching topics via LLM prompts and storing insights.
- Self‑evolving Python files to improve quality or fix bugs.
- Experimenting with conversational agents that collaborate with each other.

Because the codebase was vibe coded, expect some rough edges, but it provides a solid foundation for building autonomous agents that learn from their actions.
