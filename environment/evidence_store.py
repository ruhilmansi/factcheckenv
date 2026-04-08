import os
import requests
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class EvidenceStore:
    def __init__(self):
        self.url = os.environ.get("SUPABASE_URL", "")
        self.key = os.environ.get("SUPABASE_KEY", "")
        self.session_retrieved = set()

    def reset_session(self):
        self.session_retrieved.clear()

    def _compute_score(self, query: str, doc_text: str, keywords: List[str]) -> float:
        query_words = set(query.lower().split())
        doc_words = set(doc_text.lower().split())
        score = 0.0
        for word in query_words:
            if word in doc_words:
                score += 1.0
            if word in keywords:
                score += 2.0
        return score

    def search(self, query: str, claim_id: str) -> List[str]:
        if not self.url or not self.key:
            return ["Credentials mismatch."]

        # Accessing Supabase via standard HTTP
        endpoint = f"{self.url.rstrip('/')}/rest/v1/claims"
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}"
        }
        params = {"claim_id": f"eq.{claim_id}", "select": "*"}

        try:
            response = requests.get(endpoint, headers=headers, params=params, verify=False)
            if response.status_code != 200:
                return [f"Network Error: {response.status_code}"]

            data = response.json()
            if not data:
                return ["Claim not found in cloud storage."]

            claim = data[0]
            scored_docs = []
            for doc in claim.get("evidence_pool", []):
                if doc["text"] in self.session_retrieved:
                    continue
                score = self._compute_score(query, doc["text"], doc.get("relevance_keywords", []))
                if score > 0:
                    scored_docs.append((doc["text"], score))
            
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            results = [doc[0] for doc in scored_docs[:2]]
            
            if not results:
                return ["No relevant evidence found for this query."]
            
            for res in results:
                self.session_retrieved.add(res)
            return results
            
        except Exception as e:
            return [f"System Error: {str(e)}"]
