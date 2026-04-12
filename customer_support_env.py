"""
Customer Support RL Environment
================================
Simulates a real-world customer support desk where an AI agent must:
- Classify support tickets
- Draft appropriate responses
- Handle escalations
"""

import random
import uuid
from typing import Any, Dict, List, Optional


# ── Ticket categories ─────────────────────────────────────────────────────────
CATEGORIES = [
    "billing",
    "technical",
    "shipping",
    "account",
    "product",
    "refund",
    "general",
]

# ── Sample tickets per task ──────────────────────────────────────────────────
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
    {
        "id": "T003",
        "subject": "Package not delivered after 2 weeks",
        "body": "My order #5544 was supposed to arrive 14 days ago. Tracking shows it left the warehouse.",
        "correct_category": "shipping",
    },
    {
        "id": "T004",
        "subject": "Cannot reset my password",
        "body": "The reset email never arrives. I've checked spam. My email is user@example.com",
        "correct_category": "account",
    },
    {
        "id": "T005",
        "subject": "Product looks different from website photo",
        "body": "The color and size are completely different from what was shown on the product page.",
        "correct_category": "product",
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
    },
    {
        "ticket": {
            "id": "D002",
            "customer_name": "James",
            "subject": "Billing overcharge",
            "body": "I was billed $149 but the website clearly showed $99 for the annual plan.",
            "sentiment": "frustrated",
        },
        "required_elements": ["apologize", "investigate", "resolution"],
        "forbidden_elements": ["deny", "rude"],
    },
    {
        "ticket": {
            "id": "D003",
            "customer_name": "Priya",
            "subject": "Feature request — dark mode",
            "body": "Love your product! Any plans to add dark mode? My eyes hurt at night.",
            "sentiment": "positive",
        },
        "required_elements": ["acknowledge", "feedback", "thank"],
        "forbidden_elements": ["dismiss", "ignore"],
    },
]

ESCALATION_SCENARIOS = [
    {
        "ticket_id": "E001",
        "history": [
            {"role": "customer", "text": "My account has been hacked and money is missing!"},
            {"role": "agent", "text": "I'm sorry to hear that. Can you verify your identity?"},
            {"role": "customer", "text": "I've verified already! This is urgent!"},
        ],
        "correct_actions_sequence": ["acknowledge_urgency", "escalate_security", "provide_eta"],
        "resolution": "security_team_contacted",
    },
    {
        "ticket_id": "E002",
        "history": [
            {"role": "customer", "text": "I've contacted support 5 times about my refund. Nobody helps!"},
            {"role": "agent", "text": "I apologize for the inconvenience."},
            {"role": "customer", "text": "Apologies aren't enough! I want a manager!"},
        ],
        "correct_actions_sequence": ["acknowledge_frustration", "escalate_manager", "confirm_callback"],
        "resolution": "manager_callback_scheduled",
    },
]


# ── Graders ──────────────────────────────────────────────────────────────────

def grade_classify(action: Dict, ticket: Dict) -> float:
    """Score strictly between 0 and 1 for ticket classification."""
    chosen = (action.get("content") or "").strip().lower()
    correct = ticket["correct_category"].lower()

    if chosen == correct:
        return 0.95
    # Partial credit for closely related categories
    related = {
        "billing": ["refund"],
        "refund": ["billing"],
        "technical": ["product"],
        "account": ["technical"],
    }
    if chosen in related.get(correct, []):
        return 0.5
    return 0.05


def grade_draft(action: Dict, scenario: Dict) -> float:
    """Score strictly between 0 and 1 for response quality."""
    response = (action.get("content") or "").lower()
    required = scenario["required_elements"]
    forbidden = scenario["forbidden_elements"]

    if not response:
        return 0.05

    # Check forbidden words first
    for word in forbidden:
        if word in response:
            return 0.05

    # Score based on required element coverage
    score = 0.0
    per_element = 0.85 / len(required)  # max from elements = 0.85
    keyword_map = {
        "apologize": ["sorry", "apologize", "apologi"],
        "refund": ["refund", "reimburse", "return"],
        "timeline": ["within", "days", "hours", "soon", "24", "48"],
        "investigate": ["investigate", "look into", "check", "review"],
        "resolution": ["resolve", "fix", "solution", "address"],
        "acknowledge": ["understand", "hear", "noted", "appreciate"],
        "feedback": ["feedback", "suggestion", "feature", "note"],
        "thank": ["thank", "grateful", "appreciate"],
    }
    for element in required:
        keywords = keyword_map.get(element, [element])
        if any(kw in response for kw in keywords):
            score += per_element

    # Bonus for length
    words = len(response.split())
    if words >= 40:
        score += 0.1

    # Clamp strictly between 0 and 1
    return round(min(max(score, 0.05), 0.95), 2)


def grade_escalation(action: Dict, scenario: Dict, step_index: int) -> float:
    """Score strictly between 0 and 1 for each escalation step."""
    action_type = (action.get("action_type") or "").lower()
    expected = scenario["correct_actions_sequence"]

    if step_index >= len(expected):
        return 0.05

    if action_type == expected[step_index]:
        return 0.95

    # Partial: related action
    related_map = {
        "acknowledge_urgency": ["acknowledge_frustration"],
        "acknowledge_frustration": ["acknowledge_urgency"],
        "escalate_security": ["escalate_manager"],
        "escalate_manager": ["escalate_security"],
    }
    if action_type in related_map.get(expected[step_index], []):
        return 0.4

    return 0.05


# ── Environment class ────────────────────────────────────────────────────────

class CustomerSupportEnv:
    TASKS = ["classify-ticket", "draft-response", "resolve-escalation"]

    def __init__(self, task: str = "classify-ticket"):
        if task not in self.TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose from {self.TASKS}")
        self.task = task
        self._scenario: Optional[Dict] = None
        self._step_count = 0
        self._done = False
        self._rewards: List[float] = []
        self._session_id = str(uuid.uuid4())

    # ── reset ─────────────────────────────────────────────────────────────────
    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._done = False
        self._rewards = []
        self._session_id = str(uuid.uuid4())

        if self.task == "classify-ticket":
            self._scenario = random.choice(CLASSIFY_TICKETS)
            obs = {
                "task": self.task,
                "session_id": self._session_id,
                "instruction": (
                    "Classify the following support ticket into one of these categories: "
                    + ", ".join(CATEGORIES)
                ),
                "ticket": {
                    "id": self._scenario["id"],
                    "subject": self._scenario["subject"],
                    "body": self._scenario["body"],
                },
                "action_format": {
                    "action_type": "classify",
                    "content": "<category>",
                },
            }

        elif self.task == "draft-response":
            self._scenario = random.choice(DRAFT_SCENARIOS)
            obs = {
                "task": self.task,
                "session_id": self._session_id,
                "instruction": (
                    "Draft a professional, empathetic customer support response "
                    "to the following ticket."
                ),
                "ticket": self._scenario["ticket"],
                "action_format": {
                    "action_type": "draft",
                    "content": "<your full response here>",
                },
            }

        else:  # resolve-escalation
            self._scenario = random.choice(ESCALATION_SCENARIOS)
            obs = {
                "task": self.task,
                "session_id": self._session_id,
                "instruction": (
                    "Handle this escalated support conversation. "
                    "At each step, pick the best action_type from: "
                    "acknowledge_urgency, acknowledge_frustration, escalate_security, "
                    "escalate_manager, provide_eta, confirm_callback."
                ),
                "conversation": self._scenario["history"],
                "step_index": self._step_count,
                "action_format": {
                    "action_type": "<action_type>",
                    "content": "<optional message to customer>",
                },
            }

        return obs

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self._done:
            return {
                "observation": {},
                "reward": 0.05,
                "done": True,
                "info": {"error": "Episode already finished. Call reset() first."},
            }

        self._step_count += 1

        if self.task == "classify-ticket":
            reward = grade_classify(action, self._scenario)
            self._done = True
            obs = {
                "task": self.task,
                "session_id": self._session_id,
                "result": "correct" if reward >= 0.9 else "incorrect",
                "correct_category": self._scenario["correct_category"],
                "your_answer": action.get("content", ""),
            }

        elif self.task == "draft-response":
            reward = grade_draft(action, self._scenario)
            self._done = True
            obs = {
                "task": self.task,
                "session_id": self._session_id,
                "score": reward,
                "feedback": (
                    "Good response." if reward >= 0.7
                    else "Response missing required elements or contained forbidden words."
                ),
            }

        else:  # resolve-escalation
            seq = self._scenario["correct_actions_sequence"]
            step_idx = self._step_count - 1
            reward = grade_escalation(action, self._scenario, step_idx)

            if self._step_count >= len(seq):
                self._done = True
                obs = {
                    "task": self.task,
                    "session_id": self._session_id,
                    "step": self._step_count,
                    "resolution": self._scenario["resolution"],
                    "message": "Escalation resolved.",
                }
            else:
                obs = {
                    "task": self.task,
                    "session_id": self._session_id,
                    "step": self._step_count,
                    "next_instruction": (
                        f"Continue handling the escalation. Next expected action #{self._step_count + 1}."
                    ),
                    "conversation": self._scenario["history"],
                    "step_index": self._step_count,
                }

        self._rewards.append(reward)
        return {
            "observation": obs,
            "reward": reward,
            "done": self._done,
            "info": {},
        }

    # ── state ─────────────────────────────────────────────────────────────────
    def state(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "session_id": self._session_id,
            "step_count": self._step_count,
            "done": self._done,
            "cumulative_reward": round(sum(self._rewards), 4),
            "rewards_history": self._rewards,
        }
