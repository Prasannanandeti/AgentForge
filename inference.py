import os
import sys
import json
from openai import OpenAI
from customer_support_env import CustomerSupportEnv

# ── ENV VARIABLES (injected by validator) ──────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# ── LLM CLIENT (must use API_BASE_URL + API_KEY) ───────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def clamp(reward: float) -> float:
    """Ensure reward is strictly between 0 and 1."""
    return round(min(max(float(reward), 0.01), 0.99), 2)

def ask_llm(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=256,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def run_inference():
    tasks = ["classify-ticket", "draft-response", "resolve-escalation"]

    for tid in tasks:
        env = CustomerSupportEnv(task=tid)
        obs = env.reset()
        print(f"[START] task={tid} env=customer-support model={MODEL_NAME}")

        done     = False
        step_idx = 0
        rewards  = []
        success  = "false"

        system_prompt = (
            "You are a customer support AI agent. "
            "Given the current ticket observation, decide the next action. "
            "Reply with a JSON object only, with keys: action_type, content. "
            "action_type must be one of: classify, draft, acknowledge_urgency, "
            "acknowledge_frustration, escalate_security, escalate_manager, "
            "provide_eta, confirm_callback. "
            "Example: {\"action_type\": \"classify\", \"content\": \"billing\"}"
        )

        while not done and step_idx < 8:
            step_idx += 1
            try:
                user_prompt = f"Current observation:\n{json.dumps(obs, default=str)}"
                llm_reply   = ask_llm(system_prompt, user_prompt)

                # parse LLM JSON safely
                try:
                    action = json.loads(llm_reply)
                except Exception:
                    action = {"action_type": "classify", "content": llm_reply}

                result = env.step(action)
                reward = clamp(result.get("reward", 0.05) if isinstance(result, dict) else 0.05)
                done   = result.get("done", False) if isinstance(result, dict) else True
                obs    = result.get("observation", obs) if isinstance(result, dict) else obs
                rewards.append(reward)

                if done:
                    success = "true"

                print(
                    f"[STEP] step={step_idx} "
                    f"action={action.get('action_type', 'unknown')} "
                    f"reward={reward:.2f} "
                    f"done={str(done).lower()} error=null"
                )

            except Exception as e:
                r = 0.05  # never print 0.00
                rewards.append(r)
                print(
                    f"[STEP] step={step_idx} action=error "
                    f"reward={r:.2f} done=true error={str(e)}"
                )
                done = True

        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={success} steps={step_idx} rewards={rewards_str}")


if __name__ == "__main__":
    run_inference()
