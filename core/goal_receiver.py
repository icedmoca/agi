"""core.goal_receiver – lightweight entry-point for external goal intake.

Run as CLI:
    python -m core.goal_receiver --goal "Improve caching layer"

Run as HTTP server:
    python -m core.goal_receiver --serve 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn  # type: ignore
except Exception:  # pragma: no cover – optional deps
    FastAPI = None  # type: ignore

PENDING = Path("pending_goals.jsonl")


def _append_goal(goal: str, origin: str = "external") -> None:
    PENDING.touch(exist_ok=True)
    entry: dict[str, Any] = {
        "goal": goal,
        "origin": origin,
        "created": datetime.now().isoformat(),
    }
    with PENDING.open("a") as fp:
        fp.write(json.dumps(entry) + "\n")


# ------------------------------------------------------------------ #
# CLI entry
# ------------------------------------------------------------------ #

def _cli(goal_text: str) -> None:
    _append_goal(goal_text)
    print("[✓] Goal submitted")


# ------------------------------------------------------------------ #
# FastAPI
# ------------------------------------------------------------------ #

def _serve(port: int = 8000) -> None:
    if FastAPI is None:
        print("FastAPI not installed – cannot serve API")
        return

    app = FastAPI()

    @app.post("/submit")
    def submit(payload: dict):  # type: ignore
        goal = payload.get("goal")
        if not goal:
            return JSONResponse({"error": "Missing goal"}, status_code=400)
        _append_goal(goal)
        return {"status": "accepted"}

    # ------------------------- STATUS / CONTROL -------------------- #
    state_path = Path("state.json")

    def _read_state():
        if state_path.exists():
            try:
                return json.loads(state_path.read_text())
            except Exception:
                pass
        return {"status": "unknown"}

    @app.get("/status")
    def status():  # type: ignore
        return _read_state()

    @app.post("/pause")
    def pause():  # type: ignore
        state = _read_state()
        state["status"] = "paused"
        state_path.write_text(json.dumps(state))
        return {"status": "paused"}

    @app.post("/resume")
    def resume():  # type: ignore
        state = _read_state()
        state["status"] = "running"
        state_path.write_text(json.dumps(state))
        return {"status": "running"}

    @app.post("/stop")
    def stop():  # type: ignore
        state = _read_state()
        state["status"] = "stopped"
        state_path.write_text(json.dumps(state))
        return {"status": "stopped"}

    # intent test
    from core.intent_classifier import classify_intent as _ci

    @app.post("/intent")
    def intent(payload: dict):  # type: ignore
        goal = payload.get("goal", "")
        return _ci(goal)

    @app.post("/simulate")
    def simulate(payload: dict):  # type: ignore
        goal = payload.get("goal", "")
        from core.intent_classifier import classify_intent as ci
        return {"intent": ci(goal), "plan": "simulation placeholder"}

    uvicorn.run(app, host="0.0.0.0", port=port)


# ------------------------------------------------------------------ #
# Main dispatcher
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External goal receiver")
    parser.add_argument("--goal", help="Goal text")
    parser.add_argument("--serve", type=int, help="Start HTTP server on port")
    parser.add_argument("--status", action="store_true", help="Show current agent status")
    parser.add_argument("--pause", action="store_true", help="Pause agent")
    parser.add_argument("--resume", action="store_true", help="Resume agent")
    parser.add_argument("--stop", action="store_true", help="Stop agent")
    args = parser.parse_args()

    if args.serve:
        _serve(args.serve)
    elif args.status:
        print(json.dumps(_read_state(), indent=2))
    elif args.pause:
        _append_goal("", origin="cmd_pause")  # noop just ensure file exists
        state = {"status": "paused", "crash_count": 0}
        Path("state.json").write_text(json.dumps(state))
        print("Agent paused")
    elif args.resume:
        state = {"status": "running", "crash_count": 0}
        Path("state.json").write_text(json.dumps(state))
        print("Agent resumed")
    elif args.stop:
        state = {"status": "stopped", "crash_count": 0}
        Path("state.json").write_text(json.dumps(state))
        print("Agent stopped")
    elif args.goal:
        _cli(args.goal)
    else:
        parser.print_help() 