---
title: Customer Support RL Environment
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🎧 Customer Support RL Environment

An OpenEnv-compliant reinforcement learning environment that simulates a real-world customer support desk.

## Overview

An AI agent learns to handle customer support workflows across three progressively harder tasks:

| Task | Difficulty | Description |
|------|-----------|-------------|
| `classify-ticket` | Easy | Assign a support ticket to the correct category |
| `draft-response` | Medium | Write a professional empathetic response |
| `resolve-escalation` | Hard | Handle a multi-turn escalation sequence |

## Action Space

Each action is a JSON object:
```json
{
  "action_type": "classify | draft | acknowledge_urgency | escalate_manager | ...",
  "content": "<text content of the action>"
}
```

## Observation Space

Each observation is a JSON object containing:
- `task` — current task name
- `session_id` — unique episode identifier
- `instruction` — what the agent should do
- `ticket` or `conversation` — the support scenario
- `action_format` — expected format of the action

## Reward Function

| Task | Reward Signal |
|------|--------------|
| classify-ticket | 1.0 correct, 0.5 related category, 0.0 wrong |
| draft-response | 0.0–1.0 based on required elements present, forbidden elements absent, response length |
| resolve-escalation | 1.0 correct action per step, 0.4 related action, 0.0 wrong |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Start a new episode |
| POST | `/step` | Take an action |
| GET/POST | `/state` | Query current episode state |

### Example: Reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "classify-ticket"}'
```

### Example: Step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task": "classify-ticket", "action": {"action_type": "classify", "content": "billing"}}'
```

## Setup & Running

### Local

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t customer-support-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  customer-support-env
```

### Run Inference

```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_BASE_URL=http://localhost:7860

python inference.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API key | — |
| `MODEL_NAME` | LLM model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `ENV_BASE_URL` | This environment's URL | `http://localhost:7860` |
| `TASK` | Run a specific task only | all tasks |

## Team

**AgentForge** — OpenEnv Hackathon 2026
