from typing import Dict, Any

class FactCheckGrader:
    @staticmethod
    def score(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        
        \"\"\"
        Calculates a score between 0 and 1.
        Criteria:
        - Verdict match (0.7 weight)
        - Reasoning quality (simulated, 0.3 weight)
        \"\"\"
        verdict_score = 1.0 if prediction.get("verdict", "").lower() == ground_truth.get("verdict", "").lower() else 0.0
        reasoning = prediction.get("reasoning", "").lower()
        explanation = ground_truth.get("explanation", "").lower()
        
        keywords = set(explanation.split())
        match_count = sum(1 for word in keywords if word in reasoning and len(word) > 4)
        reasoning_score = min(1.0, match_count / 5.0) if keywords else 1.0
    
        score = (verdict_score * 0.7) + (reasoning_score * 0.3)
        return max(0.01, min(0.99, score))
