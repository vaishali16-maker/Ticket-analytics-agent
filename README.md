# Ticket SLA Analytics Agent 🎫🤖

An AI agent for IT support ticket analytics — ask questions about your ticket data in plain English and get real SQL-backed answers, and predict whether a new ticket is likely to breach its SLA before it even happens.

**Live app:** https://ticket-analytics-agent.onrender.com/docs
**Live UI:** run `streamlit run streamlitapp.py` (or point `API_URL` at the Render URL)

## What it does

Given a ticket dataset (priority, category, team, timestamps, SLA plan, resolution time, CSAT), this project answers two kinds of questions:

1. **"What happened?"** — Ask in plain English ("what's the SLA breach rate by priority?") and get a real, verified SQL query run against a live Postgres database, with the actual results.
2. **"What's likely to happen?"** — Enter a new ticket's details and get a predicted probability that it will breach its SLA, before it's even assigned.

## Architecture

```
User question ──► NL-to-SQL Agent ──► Postgres (Supabase) ──► Answer
                   (generates, validates,
                    retries on failure,
                    remembers conversation)

New ticket ──► Feature prep ──► ML Model (Logistic Regression) ──► Risk %
```

- **NL-to-SQL Agent**: not a single LLM call — a genuine agent loop. Generates SQL, executes it, validates the result, and retries (up to 3x) with the previous error fed back into the prompt if something goes wrong. Includes short-term conversation memory (follow-up questions like "now break that down by region" work without repeating context) and a safety check that rejects any non-`SELECT` query.
- **ML Classifier**: predicts SLA breach risk from ticket features known at creation time (priority, channel, segment, product area, region, plan) — deliberately excludes fields only known after resolution, to avoid data leakage.
- **Fine-tuned Transformer**: DistilBERT + LoRA, fine-tuned on ticket text — see findings below.

## Key findings (the actual analysis, not just the build)

- **SLA thresholds weren't provided** by the dataset. I checked whether resolution time actually varied by plan tier — it didn't (gold, platinum, and standard tickets take roughly the same time to resolve). So I set deadlines using real percentiles of the resolution-time distribution instead of arbitrary round numbers, and documented this as a dataset limitation rather than hiding it.
- **`priority` is the dominant predictive signal** — breach rate ranges from 16% (urgent) to 51% (low priority). Every other dimension I tested (channel, region, customer segment, product area) was flat, with no meaningful signal.
- **Counterintuitive finding**: platinum-plan tickets breach *more* often than gold (65.8% vs 47.4%), despite being the "premium" tier. This isn't because platinum service is slower — it's because platinum has the tightest deadline (20hrs vs 32hrs) while actual resolution speed doesn't differ by plan. Premium tiers promise more than operations actually delivers.

## ML model comparison

| Model | Precision (Breached) | Recall (Breached) | F1 |
|---|---|---|---|
| Logistic Regression (unbalanced) | 0.72 | 0.40 | 0.52 |
| Logistic Regression (balanced) | 0.51 | **0.81** | **0.62** |
| Random Forest (balanced) | 0.54 | 0.52 | 0.53 |

**Chose balanced Logistic Regression** — for an early-warning system, missing a real breach (false negative) is worse than a false alarm, so I prioritized recall over raw accuracy. Random Forest didn't outperform logistic regression, likely because the data has one dominant feature (`priority`) rather than complex interactions a tree model could exploit.

## Fine-tuning results (DistilBERT + LoRA)

Fine-tuned the same base model on the same ticket text, on two different targets, training only **1.1% of parameters** (739K / 67.7M):

| Target | Accuracy | What it proves |
|---|---|---|
| Predict SLA breach from ticket text | ~50% (chance level) | Ticket text doesn't encode breach outcome in this dataset |
| Predict issue category from ticket text | 100% | Ticket text is template-generated per category, so the model correctly learns this near-instantly |

This contrast is the actual point: the model correctly found strong signal where it genuinely exists (category) and correctly found none where it doesn't (breach risk) — proving the fine-tuning pipeline works, and that structured fields, not free text, carry the real breach signal for this dataset.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| Database | Postgres (Supabase) |
| LLM | Groq (`openai/gpt-oss-120b`) |
| Classical ML | scikit-learn (Logistic Regression, Random Forest) |
| Fine-tuning | HuggingFace Transformers + PEFT/LoRA (DistilBERT) |
| Frontend | Streamlit + Plotly |
| Deployment | Docker, Render |

## Known limitations & next steps

- **Production observability is basic** — logging and a SQL safety check exist, but no structured monitoring/alerting stack (e.g. Sentry).
- **Conversation memory is session-only** — no persistent history across sessions or user accounts (a deliberate scope decision, not an oversight).
- **No automated test suite for the agent** — testing was thorough but manual; a `pytest` suite covering `generate_sql()` and `agent_answer()` would be the natural next step.
- **Fine-tuning was on synthetic, template-generated text** — real-world ticket text is messier; I'd expect the breach-prediction task to have more (though likely still modest) signal on real data.
- **Ambiguous follow-up questions can be interpreted multiple valid ways** (e.g. "break that down by region as well" was read as "instead of" rather than "in addition to" in one test) — a known, human-language limitation, not a bug.

## Local setup

```bash
git clone https://github.com/vaishali16-maker/Ticket-analytics-agent.git
cd Ticket-analytics-agent
pip install -r requirements.txt
```

Create a `.env` file:
```
SUPABASE_CONNECTION_STRING=your_connection_string
GROQ_API_KEY=your_groq_key
```

Run the backend:
```bash
uvicorn main:app --reload --port 8000
```

Run the UI (separate terminal):
```bash
streamlit run streamlitapp.py
```

Or run in Docker:
```bash
docker build -t ticket-agent .
docker run -p 8000:8000 --env-file .env ticket-agent
```