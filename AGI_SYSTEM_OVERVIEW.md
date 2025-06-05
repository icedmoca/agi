# AGI System Overview

## Architecture

```
User / External Goals  -->  GoalReceiver  -->  pending_goals.jsonl
                                     ↓
                                PlannerAgent
                                     ↓
                                  Tasks.jsonl
                                     ↓
       ┌───────────┬────────────┬───────────────┐
       │ CoderAgent│ HealerAgent│  Chat (LLM)   │
       └───────────┴────────────┴───────────────┘
                                     ↓
                                  Memory
                                     ↓
                               Reflection
                                     ↓
                           New self-improvement goals
```

* **PlannerAgent** – decomposes high-level goals into actionable tasks.
* **CoderAgent** – evolves code, runs tests & canary, commits & pushes.
* **HealerAgent** – monitors host alerts and fixes common issues.
* **BrowserAgent** – fetches pages and summarises content.

## Tool Registry
Tools are registered dynamically and governed by `ALLOWED_TOOLS` in `core.config`.

```
run_shell, evolve_file, git_commit, git_push, browse_page, submit_external_goal, describe_self, …
```

## Goal Flow
1. Goal arrives (user, reflection, external API).
2. PlannerAgent expands goal → tasks (chat / evolution / heal).
3. CoderAgent executes evolution tasks; HealerAgent resolves alerts.
4. Results & scores stored in `memory.jsonl`; reward influences future planning.
5. Reflection analyses memory + logs → generates meta-goals.

## Control & Monitoring
* `goal_receiver.py --serve 8000` exposes `/submit`, `/status`, `/pause`, `/resume`, `/stop`.
* `state.json` controls runtime state.
* `output/audit.log` records critical events.

---
For details see `README.md`. Thank you for improving Autonomo 🤖. 