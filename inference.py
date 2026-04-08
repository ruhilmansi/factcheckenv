import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from environment.env import OpenEnv, FactCheckAction
from environment.tasks import TASKS

load_dotenv()

api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
hf_token = os.getenv("HF_TOKEN")

client = OpenAI(base_url=api_base_url, api_key=hf_token)

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
        model=model_name,
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
    env = OpenEnv()
    total_score = 0

    for task in TASKS:
        env.load_task(task.task_id)
        obs = env.reset()
        print(f"[START] task_id={task.task_id} difficulty={task.difficulty}")

        done = False
        step_num = 0
        while not done:
            action = get_action(obs)
            obs, reward, done, info = env.step(action)
            step_num += 1
            print(f"[STEP] task_id={task.task_id} step={step_num} action={action.action_type} reward={reward:.2f} done={done}")

        print(f"[END] task_id={task.task_id} final_reward={reward:.2f}")
        total_score += reward

    print(f"[END] overall_avg_score={total_score/len(TASKS):.2f}")

if __name__ == "__main__":
    run_evaluation()
