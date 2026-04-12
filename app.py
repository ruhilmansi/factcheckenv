from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from environment.env import OpenEnv, FactCheckAction, ClaimObservation
from environment.tasks import TASKS

app = FastAPI()
env = OpenEnv()

@app.get("/")
def root():
    return {"status": "ok", "env": "factcheck-env"}

@app.get("/health")
def health():
    return {"status": "ok", "env": "factcheck-env"}

@app.get("/reset")
def reset_get():
    obs = env.reset()
    return obs.model_dump()

@app.post("/reset")
def reset_post():
    obs = env.reset()
    return obs.model_dump()

@app.get("/state")
def state():
    return env.state()

@app.post("/step")
def step(action: FactCheckAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.post("/run_task/{task_id}")
def run_task(task_id: str):
    try:
        env.load_task(task_id)
        obs = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 10:
            if steps == 0:
                action = FactCheckAction(action_type="search", query="baseline check")
            else:
                action = FactCheckAction(action_type="verdict", verdict="FALSE", reasoning="dummy reasoning")

            obs, reward, done, info = env.step(action)
            total_reward = reward
            steps += 1

        total_reward = max(0.01, min(0.99, total_reward))

        return {"task_id": task_id, "total_reward": total_reward, "final_status": "done" if done else "incomplete"}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
