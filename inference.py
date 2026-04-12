import os
import json
from openai import OpenAI
from customer_support_env import CustomerSupportEnv

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def clamp(r):
    return round(min(max(float(r), 0.01), 0.99), 2)


def ask_llm(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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

        done = False
        step_idx = 0
        rewards = []
        success = "false"

        system_prompt = (
            "You are a customer support AI agent. "
            "Reply ONLY in valid JSON with keys: action_type, content. "
            "No explanations. No extra text. "
            "Example: {\"action_type\": \"classify\", \"content\": \"billing\"}"
        )

        while not done and step_idx < 8:
            step_idx += 1

            try:
                user_prompt = f"Observation:\n{json.dumps(obs, default=str)}"
                llm_reply = ask_llm(system_prompt, user_prompt)

                try:
                    action = json.loads(llm_reply)
                except Exception:
                    action = {"action_type": "classify", "content": llm_reply}

                result = env.step(action)

                # CRITICAL FIX: prefer score over reward
                raw_score = result.get("score", result.get("reward", 0.05))

                reward = clamp(raw_score)

                done = result.get("done", False)
                obs = result.get("observation", obs)

                rewards.append(reward)

                # avoid binary success inference
                if done and reward > 0.5:
                    success = "true"
                else:
                    success = "false"

                print(
                    f"[STEP] step={step_idx} action={action.get('action_type','unknown')} "
                    f"reward={reward:.2f} done={str(done).lower()} error=null"
                )

            except Exception as e:
                r = 0.05
                rewards.append(r)
                print(
                    f"[STEP] step={step_idx} action=error reward={r:.2f} done=true error={str(e)}"
                )
                done = True

        rewards_str = ",".join([f"{r:.2f}" for r in rewards])

        print(
            f"[END] success={success} steps={step_idx} rewards={rewards_str}"
        )


if __name__ == "__main__":
    run_inference()
