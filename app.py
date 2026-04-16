import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from environment.env import OpenEnv, FactCheckAction, ClaimObservation
from environment.tasks import TASKS
from environment.graders import FactCheckGrader
from inference import get_action

app = FastAPI()
env = OpenEnv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        done = False
        
        task_config = next(t for t in TASKS if t.task_id == task_id)
        ground_truth = {
            "verdict": task_config.ground_truth_verdict, 
            "explanation": " ".join(task_config.grader_config.get("keywords", []))
        }
        
        steps_log = []
        final_prediction = {"verdict": "", "reasoning": ""}
        
        while not done:
            action = get_action(obs)
            obs, reward, done, info = env.step(action)
            steps_log.append({"action": action.model_dump(), "reward": reward})
            
            if action.action_type == "verdict":
                final_prediction["verdict"] = action.verdict or ""
                final_prediction["reasoning"] = action.reasoning or ""
                
            # safety break in case model loops endlessly
            if obs.steps_taken >= obs.max_steps:
                done = True
        
        # using actual grader func (req natively strictly bw 0 and 1)
        final_score = FactCheckGrader.score(final_prediction, ground_truth)

        return {
            "task_id": task_id, 
            "total_score": final_score, 
            "final_status": "done",
            "steps": steps_log
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/stream_task/{task_id}")
async def stream_task(task_id: str):
    """
    Real-time streaming (Server-Sent Events) of the AI Agent reasoning
    and searching steps throughout the fact-checking process.
    """
    def event_stream():
        try:
            env.load_task(task_id)
            obs = env.reset()
            done = False
            
            task_config = next(t for t in TASKS if t.task_id == task_id)
            ground_truth = {
                "verdict": task_config.ground_truth_verdict, 
                "explanation": " ".join(task_config.grader_config.get("keywords", []))
            }
            
            yield f"data: {json.dumps({'event': 'start', 'task_id': task_id, 'claim': obs.claim_text})}\n\n"
            
            final_prediction = {"verdict": "", "reasoning": ""}
            
            while not done:
                action = get_action(obs)
                obs, reward, done, info = env.step(action)
                
                step_data = {
                    "event": "step",
                    "action": action.model_dump(),
                    "reward": reward,
                    "done": done,
                    "snippets_count": len(obs.evidence_snippets),
                    "steps_taken": obs.steps_taken
                }
                yield f"data: {json.dumps(step_data)}\n\n"
                
                if action.action_type == "verdict":
                    final_prediction["verdict"] = action.verdict or ""
                    final_prediction["reasoning"] = action.reasoning or ""
                    
                if obs.steps_taken >= obs.max_steps:
                    done = True
            
            final_score = FactCheckGrader.score(final_prediction, ground_truth)
            yield f"data: {json.dumps({'event': 'done', 'final_score': final_score, 'prediction': final_prediction})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
