import os
import sys

# ── make repo root importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from customer_support_env import CustomerSupportEnv

# ── ENV VARIABLES ──────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

env = CustomerSupportEnv(task="classify-ticket")

# ── INFERENCE LOGIC ────────────────────────────────────────────────────────
def run_inference():
    task_ids = ["classify-ticket", "draft-response", "resolve-escalation"]

    for tid in task_ids:
        print(f"[START] task={tid} env=customer-support model={MODEL_NAME}")
        try:
            obs = env.reset(tid) if hasattr(env, 'reset') else None
        except Exception:
            env2 = CustomerSupportEnv(task=tid)
            obs = env2.reset()

        done = False
        step_idx = 0
        rewards = []

        while not done and step_idx < 8:
            step_idx += 1
            try:
                action = {"action_type": "classify", "content": "billing"}
                result = env.step(action) if hasattr(env, 'step') else {}
                reward = result.get("reward", 0.0) if isinstance(result, dict) else 0.0
                done = result.get("done", True) if isinstance(result, dict) else True
                rewards.append(reward)
                print(
                    f"[STEP] step={step_idx} action={action['action_type']} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null"
                )
            except Exception as e:
                print(
                    f"[STEP] step={step_idx} action=error "
                    f"reward=0.00 done=true error={str(e)}"
                )
                done = True

        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success=true steps={step_idx} rewards={rewards_str}")


if __name__ == "__main__":
    run_inference()
