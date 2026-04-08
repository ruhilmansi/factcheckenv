import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from environment.env import OpenEnv, FactCheckAction
from environment.tasks import TASKS

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN")
)

def build_context(obs):
    prompt = f"fact check this: {obs.claim_text}\n\n"
    
    if obs.search_history:
        prompt += "your search history:\n"
        for q in obs.search_history:
            prompt += f"- {q}\n"
    
    if obs.evidence_snippets:
        prompt += "\nEvidence Found:\n"
        for s in obs.evidence_snippets:
            prompt += f"- {s}\n"
            
    prompt += f"\nsteps: {obs.steps_taken}/{obs.max_steps}\n"
    prompt += "\nresponse must be JSON with: action_type (search/verdict), reasoning, query (for search), verdict (TRUE/FALSE/MISLEADING)."
    return prompt

def get_action(obs):
    res = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "You are a professional fact-checker. Output JSON only."},
            {"role": "user", "content": build_context(obs)}
        ]
    )
    
    txt = res.choices[0].message.content.strip()
    if "```" in txt:
        txt = txt.split("```")[1].replace("json", "").strip()
        
    try:
        data = json.loads(txt)
        data = {k: v for k, v in data.items() if v and str(v).lower() != "null"}
        return FactCheckAction(**data)
    except Exception as e:
        return FactCheckAction(action_type="verdict", verdict="FALSE", reasoning=f"Model error: {e}")

def run_evaluation():
    print("--- starting AI fact check benchmark ---")
    env = OpenEnv()
    
    total_score = 0
    for task in TASKS:
        print(f"\ntask: {task.task_id} ({task.difficulty})")
        
        env.load_task(task.task_id)
        obs = env.reset()
        
        done = False
        while not done:
            action = get_action(obs)
            
            if action.action_type == "search":
                print(f"  -> searching: {action.query}")
            else:
                print(f"  -> verdict: {action.verdict} | {action.reasoning}")
                
            obs, reward, done, _ = env.step(action)
            
        print(f"score: {reward:.2f}")
        total_score += reward
        
    print(f"\navg score: {total_score/len(TASKS):.2f}")

if __name__ == "__main__":
    run_evaluation()
