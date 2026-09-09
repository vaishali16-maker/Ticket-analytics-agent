import os
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from groq import Groq
from dotenv import load_dotenv
import joblib
import pandas as pd

load_dotenv()
app = FastAPI()

engine = create_engine(os.getenv("SUPABASE_CONNECTION_STRING"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
sla_model = joblib.load("sla_model.pkl")

SCHEMA_DESCRIPTION = """
Table: tickets
Columns:
- ticket_id (TEXT)
- created_at (TIMESTAMP)
- customer_segment (TEXT): education, enterprise, individual, non_profit, small_business
- channel (TEXT): chat, email, in_app, phone_transcript, web_form
- product_area (TEXT): billing, login_auth, api_integration, analytics_dashboard, mobile_app, notifications, data_export
- issue_type (TEXT)
- priority (TEXT): low, medium, high, urgent
- status (TEXT): resolved, in_progress, closed_no_response, on_hold
- sla_plan (TEXT): standard, gold, platinum
- resolution_time_hours (FLOAT, nullable)
- sla_deadline_hours (FLOAT)
- sla_breached (BOOLEAN, nullable): TRUE if breached, FALSE if not, NULL if unresolved.
  IMPORTANT: when calculating breach rate or percentage, always filter WHERE
  sla_breached IS NOT NULL first, so unresolved tickets don't distort the denominator.
- reopened (INTEGER): 0 or 1
- customer_sentiment (TEXT): very_negative, negative, neutral, positive, very_positive
- csat_score (INTEGER): 1-5
- region (TEXT, nullable): EU, NA, APAC, LATAM, MEA
"""

MAX_ATTEMPTS = 3


def generate_sql(question: str, previous_error: str = None, previous_sql: str = None) -> str:
    prompt = f"""You are a SQL expert. Given this table schema:
{SCHEMA_DESCRIPTION}
Convert the following question into a single valid PostgreSQL SELECT query.
Only return the raw SQL query, nothing else — no explanation, no markdown, no backticks.
Question: {question}
"""

    if previous_error:
        prompt += f"""
Your previous attempt failed. Fix the error and try again.
Previous SQL: {previous_sql}
Error: {previous_error}
"""

    # This call now runs on EVERY attempt, not just retries
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def run_sql(sql: str):
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = [dict(row._mapping) for row in result]
    return rows


def validate_result(rows: list, question: str) -> tuple[bool, str]:
    if len(rows) == 0:
        return False, "Query returned zero rows — this may indicate incorrect filtering logic."
    return True, ""


def agent_answer(question: str):
    """The agent loop: generate SQL, run it, validate, retry on failure — up to MAX_ATTEMPTS."""
    attempts = []
    sql = None
    error = None

    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        sql = generate_sql(question, previous_error=error, previous_sql=sql)
        try:
            rows = run_sql(sql)
            is_valid, validation_msg = validate_result(rows, question)

            attempts.append({
                "attempt": attempt_num,
                "sql": sql,
                "row_count": len(rows),
                "valid": is_valid,
            })
            if is_valid:
                return {
                    "question": question,
                    "final_sql": sql,
                    "result": rows,
                    "attempts_needed": attempt_num,
                    "attempt_log": attempts,
                }
            else:
                error = validation_msg

        except Exception as e:
            error = str(e)
            attempts.append({
                "attempt": attempt_num,
                "sql": sql,
                "error": error,
            })

    return {
        "question": question,
        "error": "Could not produce a valid answer after multiple attempts.",
        "attempt_log": attempts,
    }


class QuestionRequest(BaseModel):
    question: str


class TicketFeatures(BaseModel):
    priority: str
    channel: str
    customer_segment: str
    product_area: str
    issue_type: str
    region: str
    sla_plan: str


@app.post("/ask")
def ask_question(request: QuestionRequest):
    return agent_answer(request.question)


@app.post("/predict")
def predict_breach(ticket: TicketFeatures):
    input_df = pd.DataFrame([ticket.dict()])
    probability = sla_model.predict_proba(input_df)[0][1]
    prediction = sla_model.predict(input_df)[0]

    return {
        "breach_probability": round(float(probability), 3),
        "predicted_breach": bool(prediction),
    }