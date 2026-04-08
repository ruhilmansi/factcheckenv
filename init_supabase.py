import os
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

def sync_data():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        print("missing credentials")
        return

    api_url = f"{url.rstrip('/')}/rest/v1/claims"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    try:
        with open("environment/data/claims.json", "r") as f:
            claims = json.load(f)
    except:
        return

    print(f"Syncing {len(claims)} claims...")

    for claim in claims:
        response = requests.post(api_url, headers=headers, json=claim, verify=False)
        if response.status_code in [200, 201]:
            print(f"done: {claim['claim_id']}")
        else:
            print(f"error {claim['claim_id']}: {response.text}")

if __name__ == "__main__":
    sync_data()
