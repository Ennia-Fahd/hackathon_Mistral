import os
import json
import pandas as pd
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from risk_engine import detect_anomalies

# -------- Prompts (IMPORTANT: double braces {{ }} in JSON schema) ----------
SYSTEM_PROMPT = """You are a financial crime & AML analyst assistant.
Be precise, structured, and evidence-based. Avoid hallucinations.
If information is missing, say "unknown".
"""

USER_PROMPT_TEMPLATE = """
You are given a dataset summary and a list of top anomalies.
Your task:
1) Provide a risk assessment (0-100).
2) Identify top suspicious patterns.
3) Provide an explainable narrative (clear, non-technical).
4) Recommend next steps for an investigator.
5) Generate a SAR-style summary.

Return JSON with this schema:
{{
  "overall_risk_score": number,
  "top_findings": [{{"title": string, "why_suspicious": string, "evidence": string}}],
  "recommended_actions": [string],
  "sar_summary": {{
    "subject": string,
    "timeline": string,
    "suspicious_activity": string,
    "supporting_details": string,
    "recommendation": string
  }}
}}

RULES:
- Evidence MUST reference identifiers when available (transaction_id/account_id).
- If uncertain, set verdict as "uncertain" in your language but keep JSON schema.

DATASET_SUMMARY:
{dataset_summary}

TOP_ANOMALIES (rows JSON):
{anomalies_table}
"""

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

app = FastAPI(title="Mistral Risk Copilot", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_mistral_client = None


def get_mistral_client():
    global _mistral_client
    if _mistral_client is None:
        if not API_KEY:
            raise HTTPException(status_code=500, detail="MISTRAL_API_KEY missing in backend/.env")
        from mistralai import Mistral  # lazy import
        _mistral_client = Mistral(api_key=API_KEY)
    return _mistral_client


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "has_api_key": bool(API_KEY)}


@app.get("/ping")
def ping():
    return {"pong": True}


@app.post("/analyze_fast")
async def analyze_fast(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    content = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty.")

    pack = detect_anomalies(df, max_rows=12)
    return {
        "meta": pack["meta"],
        "dataset_summary": pack["dataset_summary"],
        "top_anomalies": pack["anomalies"],
        "llm_result": {"note": "Fast mode (no LLM)."},
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    content = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty.")

    pack = detect_anomalies(df, max_rows=12)

    dataset_summary = pack["dataset_summary"]
    anomalies = pack["anomalies"]
    anomalies_table = json.dumps(anomalies, ensure_ascii=False, indent=2)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        dataset_summary=dataset_summary,
        anomalies_table=anomalies_table,
    )

    client = get_mistral_client()
    try:
        resp = client.chat.complete(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\nReturn STRICT JSON only."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=650,
        )
        raw = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mistral API error: {e}")

    try:
        llm_json = json.loads(raw)
    except Exception:
        llm_json = {"raw_model_output": raw}

    return {
        "meta": pack["meta"],
        "dataset_summary": dataset_summary,
        "top_anomalies": anomalies,
        "llm_result": llm_json,
    }


# ---------- WOW BOOST ENDPOINTS ----------

class ExecutiveSummaryIn(BaseModel):
    dataset_summary: str
    top_anomalies: list


@app.post("/executive_summary")
def executive_summary(payload: ExecutiveSummaryIn):
    client = get_mistral_client()
    prompt = f"""
Write ONE executive summary paragraph (5-7 sentences) for a compliance manager.
Mention: overall risk level, top patterns, evidence highlights (use transaction_id/account_id if present), and next steps.
No bullet points.

DATASET_SUMMARY:
{payload.dataset_summary}

TOP_ANOMALIES (JSON):
{json.dumps(payload.top_anomalies, ensure_ascii=False)}
"""
    resp = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a senior AML compliance officer. Be concise and evidence-based."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=220,
    )
    return {"executive_summary": resp.choices[0].message.content.strip()}


class ExplainAnomalyIn(BaseModel):
    dataset_summary: str
    row: dict


@app.post("/explain_anomaly")
def explain_anomaly(payload: ExplainAnomalyIn):
    client = get_mistral_client()
    prompt = f"""
Explain why this specific transaction row is suspicious or not.
Return STRICT JSON only:
{{
  "verdict": "suspicious" | "not_suspicious" | "uncertain",
  "why": string,
  "evidence": string,
  "follow_up_checks": [string]
}}

DATASET_SUMMARY:
{payload.dataset_summary}

ROW (JSON):
{json.dumps(payload.row, ensure_ascii=False)}
"""
    resp = client.chat.complete(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an AML investigator. Strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=350,
    )
    raw = resp.choices[0].message.content
    try:
        return {"explanation": json.loads(raw)}
    except Exception:
        return {"explanation": {"raw_model_output": raw}}