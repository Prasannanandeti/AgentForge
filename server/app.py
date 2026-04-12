"""
Customer Support RL Environment — FastAPI Server
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uvicorn

from customer_support_env import CustomerSupportEnv

app = FastAPI(
    title="Customer Support RL Environment",
    description="OpenEnv-compliant RL environment for customer support tasks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session Store ─────────────────────────────────────────────────────────────
SESSIONS: Dict[str, CustomerSupportEnv] = {}


# ── SAFE CLAMP ────────────────────────────────────────────────────────────────
def _safe(value: float) -> float:
    try:
        v = float(value)
    except:
        v = 0.10
    return round(max(0.01, min(0.99, v)), 4)


# ── Request Models ────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task: str = "classify-ticket"
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    task: str = "classify-ticket"
    action: Dict[str, Any]
    session_id: Optional[str] = None


class StateRequest(BaseModel):
    task: Optional[str] = None
    session_id: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "classify-ticket",
                "reward_range": [0.01, 0.99],
            },
            {
                "name": "draft-response",
                "reward_range": [0.01, 0.99],
            },
            {
                "name": "resolve-escalation",
                "reward_range": [0.01, 0.99],
            },
        ]
    }


# FIXED RESET (IMPORTANT)
@app.post("/reset")
def reset(req: Optional[ResetRequest] = Body(default=None)):
    if req is None:
        req = ResetRequest()

    env = CustomerSupportEnv(task=req.task)
    obs = env.reset()

    session_id = req.session_id or env._session_id
    SESSIONS[session_id] = env

    return {
        "observation": obs,
        "session_id": session_id,
        "task": req.task,
    }


@app.post("/step")
def step(req: StepRequest):
    env = None

    if req.session_id and req.session_id in SESSIONS:
        env = SESSIONS[req.session_id]

    if env is None:
        env = CustomerSupportEnv(task=req.task)
        env.reset()
        session_id = env._session_id
        SESSIONS[session_id] = env

    result = env.step(req.action)

    raw_reward = result.get("reward", 0.10)
    raw_score  = result.get("score", raw_reward)

    safe_reward = _safe(raw_reward)
    safe_score  = _safe(raw_score)

    return {
        "observation": result.get("observation", {}),
        "reward": safe_reward,
        "score": safe_score,
        "done": bool(result.get("done", False)),
        "info": result.get("info", {}),
    }


@app.get("/state")
@app.post("/state")
def state(req: Optional[StateRequest] = None):
    env = None

    if req and req.session_id and req.session_id in SESSIONS:
        env = SESSIONS[req.session_id]

    if env is None and SESSIONS:
        env = list(SESSIONS.values())[-1]

    if env is None:
        return {
            "reward_sum": 0.1,
            "reward_avg": 0.1,
            "steps": 0,
        }

    s = env.state()

    if "reward_avg" in s:
        s["reward_avg"] = _safe(s["reward_avg"])

    return s


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
