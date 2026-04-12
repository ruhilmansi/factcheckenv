from typing import Dict, Any

class FactCheckGrader:
    @staticmethod
    def score(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        pred_verdict = prediction.get("verdict", "").strip().lower()
        true_verdict = ground_truth.get("verdict", "").strip().lower()

        verdict_score = 0.98 if pred_verdict == true_verdict else 0.02

        reasoning = prediction.get("reasoning", "").lower()
        explanation = ground_truth.get("explanation", "").lower()

        keywords = {w for w in explanation.split() if len(w) > 4}
        if keywords:
            match_count = sum(1 for word in keywords if word in reasoning)
            reasoning_score = min(0.98, max(0.02, match_count / 5.0))
        else:
            reasoning_score = 0.5

        score = (verdict_score * 0.7) + (reasoning_score * 0.3)
        return max(0.01, min(0.99, score))
