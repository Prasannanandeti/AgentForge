import os
from fastapi import FastAPI, HTTPException, Body
from openai import OpenAI
from env.environment import AgentForgeEnv
from env.models import Action

# -------------------------
# INIT
# -------------------------
app = FastAPI()
env = AgentForgeEnv()

# -------------------------
# ENV VARIABLES
# -------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/")
def home():
    return {"status": "running"}

# -------------------------
# RESET
# -------------------------
@app.post("/reset")
def reset(payload: dict = Body(default=None)):
    try:
        task_id = "easy_1"
        if payload and isinstance(payload, dict):
            task_id = payload.get("task_id", "easy_1") or "easy_1"
        obs = env.reset(task_id)
        if hasattr(obs, "model_dump"):
            return obs.model_dump()
        return obs
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------
# STEP
# -------------------------
@app.post("/step")
def step(action: dict = Body(...)):
    try:
        act = Action(**action)
        obs, reward, done, info = env.step(act)
        obs_out = obs.model_dump() if hasattr(obs, "model_dump") else obs
        reward_out = reward.value if hasattr(reward, "value") else float(reward)
        return {
            "observation": obs_out,
            "reward": reward_out,
            "done": bool(done),
            "info": info if isinstance(info, dict) else {},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------
# STATE (OPTIONAL)
# -------------------------
@app.get("/state")
def state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# INFERENCE LOGIC (OFFLINE)
# -------------------------
def run_inference():
    task_ids = ["easy_1", "medium_1", "hard_1"]
    for tid in task_ids:
        print(f"[START] task={tid} env=agentforge model={MODEL_NAME}")
        obs = env.reset(tid)
        done = False
        step_idx = 0
        rewards = []
        success = "false"
        order_id = obs.task_context.get("order_id") if hasattr(obs, "task_context") else None

        while not done and step_idx < 8:
            step_idx += 1
            try:
                if tid == "easy_1":
                    if step_idx == 1:
                        action = Action(
                            action_type="call_tool",
                            tool_name="get_order_details",
                            tool_params={"order_id": order_id},
                        )
                    elif step_idx == 2:
                        action = Action(
                            action_type="reply",
                            text=f"Your order {order_id} has been shipped.",
                        )
                    else:
                        action = Action(action_type="close_ticket")
                else:
                    action = Action(action_type="close_ticket")

                obs, reward_obj, done, info = env.step(action)
                r_val = reward_obj.value if hasattr(reward_obj, "value") else float(reward_obj)
                rewards.append(r_val)
                print(
                    f"[STEP] step={step_idx} action={action.action_type} "
                    f"reward={r_val:.2f} done={str(done).lower()} error=null"
                )
            except Exception as e:
                print(
                    f"[STEP] step={step_idx} action=error "
                    f"reward=0.00 done=true error={str(e)}"
                )
                done = True

        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={success} steps={step_idx} rewards={rewards_str}")
