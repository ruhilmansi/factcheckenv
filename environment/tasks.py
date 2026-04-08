from pydantic import BaseModel
from typing import List, Callable, Dict, Any, Literal

class TaskConfig(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    claim: str
    ground_truth_verdict: Literal["TRUE", "MISLEADING", "FALSE"]
    max_steps: int
    grader_config: Dict[str, Any]

TASKS = [
    TaskConfig(
        task_id="health_claim_simple",
        difficulty="easy",
        claim="Drinking lemon water every morning boosts immunity by 40%",
        ground_truth_verdict="FALSE",
        max_steps=5,
        grader_config={
            "exact_verdict_weight": 1.0,
            "partial_reasoning_weight": 0.5
        }
    ),
    TaskConfig(
        task_id="election_claim_nuanced",
        difficulty="medium",
        claim="Voter turnout in the 2020 US election was the lowest in 50 years",
        ground_truth_verdict="MISLEADING",
        max_steps=8,
        grader_config={
            "verdict_weight": 0.6,
            "keywords": ["highest", "record", "66.7%", "increase"],
            "keyword_weight_total": 0.4
        }
    ),
    TaskConfig(
        task_id="scientific_consensus_complex",
        difficulty="hard",
        claim="Scientists have reversed their position on coffee being harmful — it is now proven to prevent cancer",
        ground_truth_verdict="MISLEADING",
        max_steps=12,
        grader_config={
            "verdict_weight": 0.4,
            "keywords": ["correlation", "causation", "possible", "downgraded", "observed"],
            "keyword_weight_total": 0.6
        }
    )
]
