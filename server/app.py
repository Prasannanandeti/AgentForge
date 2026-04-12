"""
FastAPI server for Customer Support RL Environment
===================================================
All endpoints comply with OpenEnv spec:
  POST /reset   — start a new episode
  POST /step    — take an action
  GET  /state   — query current state
  GET  /tasks   — list available tasks
  GET  /health  — health check
"""

import json
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from customer_support_env import CustomerSupportEnv, CATEGORIES

app = FastAPI(
    title="Customer Support RL Environment",
    description="OpenEnv-compliant customer support simulation for RL agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (task → env instance)
_sessions: Dict[str, CustomerSupportEnv] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = "classify-ticket"


class StepRequest(BaseModel):
    action: Dict[str, Any]
    task: Optional[str] = "classify-ticket"


class StateRequest(BaseModel):
    task: Optional[str] = "classify-ticket"


# ── Helper ────────────────────────────────────────────────────────────────────

def get_or_create_env(task: str) -> CustomerSupportEnv:
    if task not in _sessions:
        _sessions[task] = CustomerSupportEnv(task=task)
    return _sessions[task]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "customer-support-env"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "classify-ticket",
                "description": "Classify a support ticket into the correct category",
                "difficulty": "easy",
                "categories": CATEGORIES,
            },
            {
                "name": "draft-response",
                "description": "Draft a professional response to a customer complaint",
                "difficulty": "medium",
            },
            {
                "name": "resolve-escalation",
                "description": "Handle a complex multi-turn escalation scenario",
                "difficulty": "hard",
            },
        ]
    }


# ── CRITICAL: reset must be POST ──────────────────────────────────────────────
@app.post("/reset")
def reset(body: ResetRequest = None):
    task = (body.task if body else None) or "classify-ticket"
    try:
        env = CustomerSupportEnv(task=task)
        _sessions[task] = env
        observation = env.reset()
        return {"observation": observation, "task": task}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(body: StepRequest):
    task = body.task or "classify-ticket"
    env = _sessions.get(task)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session for task '{task}'. Call /reset first.",
        )
    result = env.step(body.action)
    return result


@app.get("/state")
def state(task: str = "classify-ticket"):
    env = _sessions.get(task)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session for task '{task}'. Call /reset first.",
        )
    return env.state()


# Also accept POST /state for compatibility
@app.post("/state")
def state_post(body: StateRequest = None):
    task = (body.task if body else None) or "classify-ticket"
    env = _sessions.get(task)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session for task '{task}'. Call /reset first.",
        )
    return env.state()


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "customer-support-env",
        "version": "1.0.0",
        "endpoints": ["/health", "/tasks", "/reset (POST)", "/step (POST)", "/state (GET/POST)"],
    }
