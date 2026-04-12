"""
Customer Support RL Environment
"""

import random
import uuid
from typing import Any, Dict, List, Optional


CATEGORIES = [
    "billing",
    "technical",
    "shipping",
    "account",
    "product",
    "refund",
    "general",
]


CLASSIFY_TICKETS = [
    {
        "id": "T001",
        "subject": "I was charged twice for my order",
        "body": "Hi, I noticed two identical charges on my credit card statement for order #8821. Please help.",
        "correct_category": "billing",
    },
    {
        "id": "T002",
        "subject": "App keeps crashing on iPhone 14",
        "body": "Every time I open the app it crashes within 5 seconds. I'm on iOS 17.2.",
        "correct_category": "technical",
    },
]


DRAFT_SCENARIOS = [
    {
        "ticket": {
            "id": "D001",
            "customer_name": "Sarah",
            "subject": "Refund request for broken item",
            "body": "The laptop stand I ordered arrived with a cracked base. I want a full refund.",
            "sentiment": "angry",
        },
        "required_elements": ["apologize", "refund", "timeline"],
        "forbidden_elements": ["blame", "ignore"],
    }
]


ESCALATION_SCENARIOS = [
    {
        "ticket_id": "E001",
        "history": [
            {"role": "customer", "text": "My account has been hacked and money is missing!"}
        ],
        "correct_actions_sequence": ["acknowledge_urgency"],
        "resolution": "security_team_contacted",
    }
]


# ─────────────── GRADERS ───────────────

def normalize_score(score: float) -> float:
    """Ensure score is strictly between (0,1)"""
    return max(0.01, min(0.99, float(score)))


def grade_classify(action: Dict, ticket: Dict) -> float:
    chosen = (action.get("content") or "").strip().lower()
    correct = ticket["correct_category"].lower()

    if chosen == correct:
        score = 0.95
    else:
        score = 0.05

    return normalize_score(score)


def grade_draft(action: Dict, scenario: Dict) -> float:
    response = (action.get("content") or "").lower()

    if not response:
        return 0.05

    score = 0.5

    if "sorry" in response:
        score += 0.2
    if "refund" in response:
        score += 0.2

    return normalize_score(score)


def grade_escalation(action: Dict, scenario: Dict, step_index: int) -> float:
    action_type = (action.get("action_type") or "").lower()
    expected = scenario["correct_actions_sequence"]

    if step_index < len(expected) and action_type == expected[step_index]:
        score = 0.95
    else:
        score = 0.05

    return normalize_score(score)


# ─────────────── ENV ───────────────

class CustomerSupportEnv:
    TASKS = ["classify-ticket", "draft-response", "resolve-escalation"]

    def __init__(self, task: str = "classify-ticket"):
        self.task = task
        self._scenario = None
        self._done = False
        self._step_count = 0
        self._rewards = []
        self._session_id = str(uuid.uuid4())

    def reset(self):
        self._done = False
        self._step_count = 0
        self._rewards = []

        if self.task == "classify-ticket":
            self._scenario = random.choice(CLASSIFY_TICKETS)

        elif self.task == "draft-response":
            self._scenario = random.choice(DRAFT_SCENARIOS)

        else:
            self._scenario = random.choice(ESCALATION_SCENARIOS)

        return {
            "task": self.task,
            "session_id": self._session_id
        }

    def step(self, action: Dict[str, Any]):

        if self._done:
            return {
                "observation": {},
                "reward": 0.05,
                "score": 0.05,
                "done": True,
                "info": {}
            }

        self._step_count += 1

        # ───── TASK LOGIC ─────
        if self.task == "classify-ticket":
            reward = grade_classify(action, self._scenario)
            self._done = True

            obs = {
                "result": f"{round(reward,2)} confidence",
                "correct_category": self._scenario["correct_category"]
            }

        elif self.task == "draft-response":
            reward = grade_draft(action, self._scenario)
            self._done = True

            obs = {
                "score": reward
            }

        else:
            reward = grade_escalation(action, self._scenario, self._step_count - 1)
            self._done = True

            obs = {
                "resolution": self._scenario["resolution"]
            }

        # CRITICAL FIX
        reward = normalize_score(reward)

        self._rewards.append(reward)

        return {
            "observation": obs,
            "reward": reward,
            "score": reward, 
            "done": self._done,
            "info": {}
        }

    def state(self):
        return {
            "reward_sum": sum(self._rewards)
        }
