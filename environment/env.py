from typing import List, Literal, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field
from .evidence_store import EvidenceStore
from .tasks import TASKS, TaskConfig

class ClaimObservation(BaseModel):
    claim_text: str
    claim_id: str
    evidence_snippets: List[str]
    search_history: List[str]
    steps_taken: int
    max_steps: int

class FactCheckAction(BaseModel):
    action_type: Literal["search", "verdict", "skip"]
    query: Optional[str] = None
    verdict: Optional[Literal["TRUE", "MISLEADING", "FALSE"]] = None
    reasoning: Optional[str] = None

class OpenEnv:
    def __init__(self):
        self.evidence_store = EvidenceStore()
        self.current_task_idx = 0
        self.steps = 0
        self.history = []
        self.snippets = []
        self.done = False

    def reset(self) -> ClaimObservation:
        self.steps = 0
        self.history = []
        self.snippets = []
        self.done = False
        self.evidence_store.reset_session()
        return self._get_observation()

    def state(self) -> Dict[str, Any]:
        task = TASKS[self.current_task_idx]
        return {
            "task_id": task.task_id,
            "steps": self.steps,
            "done": self.done,
            "history": self.history,
            "snippets": self.snippets
        }

    def _get_observation(self) -> ClaimObservation:
        task = TASKS[self.current_task_idx]
        return ClaimObservation(
            claim_text=task.claim,
            claim_id=task.task_id,
            evidence_snippets=self.snippets,
            search_history=self.history,
            steps_taken=self.steps,
            max_steps=task.max_steps
        )

    def state(self) -> Dict[str, Any]:
        return {
            "current_task_idx": self.current_task_idx,
            "steps": self.steps,
            "done": self.done,
            "history": self.history
        }

    def step(self, action: FactCheckAction) -> Tuple[ClaimObservation, float, bool, Dict[str, Any]]:
        if self.done:
            return self._get_observation(), 0.01, True, {"info": "Env already done"}

        task = TASKS[self.current_task_idx]
        self.steps += 1
        reward = -0.05  

        if self.steps > task.max_steps:
            self.done = True
            return self._get_observation(), 0.01, True, {"info": "Max steps exceeded"}

        if action.action_type == "search":
            if action.query:
                self.history.append(action.query)
                new_snippets = self.evidence_store.search(action.query, task.task_id)
                if not any("no relevant evidence found" in s for s in new_snippets):
                    reward += 0.1 
                    self.snippets.extend(new_snippets)
                else:
                    reward -= 0.05 
            else:
                reward -= 0.05 

        elif action.action_type == "verdict":
            self.done = True
            if action.verdict == task.ground_truth_verdict:
                reward += 1.0 
            else:
                reward -= 0.3 

        elif action.action_type == "skip":
            self.done = True
            reward = 0.5 # neutral score
            
        reward = max(0.01, min(0.99, reward))

        return self._get_observation(), reward, self.done, {"info": "step executed"}

    def load_task(self, task_id: str):
        for i, t in enumerate(TASKS):
            if t.task_id == task_id:
                self.current_task_idx = i
                return
        raise ValueError(f"task {task_id} not found")
