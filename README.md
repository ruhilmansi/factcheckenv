**factcheckenv**

an OpenEnv compliant reinforcement learning environment that simulates the workflow of a newsroom fact checker. the agent receives a viral claim, searches for evidence in a database and issues a verdict (TRUE, FALSE or MISLEADING). misinformation is a real problem in journalism and automated fact checking is an active area of research. this environment is designed around how real newsrooms approach claim verification like search, gather, analyze, decide.

**why this environment**

most fact checking benchmarks hand the model a claim and ask for a label. that skips the hard part, the information retrieval. in a real newsroom, a journalist has to figure out what to search for, evaluate what comes back and then make a judgment call. this environment forces the agent to do all three.

the agent gets penalized for wasting steps on bad searches, rewarded for finding relevant evidence and scored on whether the final verdict is correct. the reward function gives signal throughout the trajectory, not just at the end.

**observation space**

the agent receives a `ClaimObservation` (Pydantic model) at each step:

| field | type | description |
|---|---|---|
| claim_text | str | the raw claim to verify |
| claim_id | str | unique identifier for the claim |
| evidence_snippets | list[str] | evidence found so far from searches |
| search_history | list[str] | previous search queries made |
| steps_taken | int | current step number |
| max_steps | int | maximum steps allowed for this task |

**action space**

the agent returns a `FactCheckAction` (pydantic model):

| field | type | description |
|---|---|---|
| action_type | "search" / "verdict" / "skip" | what the agent wants to do |
| query | str (optional) | search query, required if action_type is "search" |
| verdict | "TRUE" / "FALSE" / "MISLEADING" (optional) | final verdict, required if action_type is "verdict" |
| reasoning | str (optional) | explanation for the verdict |

**reward function**

the reward is designed to give meaningful feedback at every step, not just at the end:

- +0.10 for a search that returns relevant evidence
- -0.05 base penalty per step (encourages efficiency)
- -0.05 extra penalty for irrelevant or empty searches
- +1.00 for a correct final verdict
- -0.30 for an incorrect final verdict
- -0.50 for exceeding max steps

this means the agent cannot just guess, it needs to actually gather evidence efficiently before making a call

**tasks**

three tasks with increasing difficulty. each has a programmatic grader that produces a score between 0.0 and 1.0

| task ID | topic | difficulty | description |
|---|---|---|---|
| health_001 | health | easy | verify a straightforward claim about lemon water and immunity. ground truth is clearly FALSE. |
| election_001 | politics | medium | check a nuanced claim about 2020 US election voter turnout. requires careful interpretation, the claim is MISLEADING, not outright false. |
| science_001 | science | hard | evaluate a complex claim about coffee and cancer prevention. scientific evidence is mixed, making this genuinely difficult. |

the grading logic weighs verdict accuracy (0.7) against reasoning quality (0.3). a correct verdict with poor reasoning scores lower than a correct verdict with good reasoning.

**evidence store**

claims and evidence are stored in a **Supabase PostgreSQL** database with RLS policies enabled. the evidence store uses keyword based scoring to return relevant snippets for a given search query. this is intentionally not a perfect retrieval system, the agent has to learn what queries produce useful results.

**setup**

```bash
pip install -r requirements.txt
```

create a `.env` file:
```
SUPABASE_URL=
SUPABASE_KEY=
HF_TOKEN=
API_BASE_URL=
MODEL_NAME=
```

sync claims to your database:
```bash
python init_supabase.py
```

run the baseline evaluation:
```bash
python inference.py
```

**docker**

```bash
docker build -t factcheckenv 
docker run -p 7860:7860 --env-file .env factcheckenv
```

**API endpoints**

| method | endpoint | description |
|---|---|---|
| GET | / | health check, returns 200 |
| GET | /reset | reset environment, returns initial observation |
| POST | /reset | reset environment, returns initial observation |
| GET | /state | returns current environment state |
| POST | /step | execute an action, returns observation, reward, done, info |

**baseline results**

| task | verdict Given | correct | score |
|---|---|---|---|
| health_001 | FALSE | Yes | 0.85 |
| election_001 | FALSE | No (should be MISLEADING) | -0.15 |
| science_001 | FALSE | No | -0.20 |
| **average** | | | **~0.17** |

the baseline model struggles with nuanced claims that require MISLEADING verdicts. this is expected, distinguishing "false" from "misleading" requires genuine reasoning about evidence quality.

**file structure**

```
factcheckenv/
  app.py                  — FastAPI server
  inference.py            — Baseline evaluation script
  openenv.yaml            — OpenEnv metadata
  Dockerfile              — Container config
  requirements.txt        — Dependencies
  environment/
    env.py                — Core OpenEnv implementation
    tasks.py              — Task definitions
    graders.py            — Grading logic
    evidence_store.py     — Supabase evidence retrieval
    data/
      claims.json         — Claim dataset
```

**OpenEnv compliance**

- typed Pydantic models for observation and action
- step(action) returns (observation, reward, done, info)
- reset() returns initial observation
- state() returns current state
- openenv.yaml with metadata
- minimum 3 tasks with graders scoring 0.0 to 1.0
- reward function with partial progress signals
- baseline inference script with reproducible scores
