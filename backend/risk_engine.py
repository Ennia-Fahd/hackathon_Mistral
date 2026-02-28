import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering minimaliste (robuste à CSV variés).
    Attend idéalement: amount, channel, country, merchant_category, account_id.
    """
    work = df.copy()

    # amount
    if "amount" in work.columns:
        work["amount_num"] = _safe_to_numeric(work["amount"])
    else:
        # fallback: chercher une colonne montant plausible
        candidates = [c for c in work.columns if "amount" in c.lower() or "montant" in c.lower()]
        if candidates:
            work["amount_num"] = _safe_to_numeric(work[candidates[0]])
        else:
            work["amount_num"] = np.nan

    work["amount_num"] = work["amount_num"].fillna(work["amount_num"].median() if work["amount_num"].notna().any() else 0.0)

    # Timestamp -> hour feature (si présent)
    ts_col = None
    for c in work.columns:
        if c.lower() in ["timestamp", "date", "datetime", "transaction_date", "created_at"]:
            ts_col = c
            break

    if ts_col:
        try:
            t = pd.to_datetime(work[ts_col], errors="coerce")
            work["hour"] = t.dt.hour.fillna(-1).astype(int)
        except Exception:
            work["hour"] = -1
    else:
        work["hour"] = -1

    # Categorical hashing via category codes
    for col in ["channel", "country", "merchant_category", "currency"]:
        if col in work.columns:
            work[f"{col}_code"] = work[col].astype("category").cat.codes
        else:
            work[f"{col}_code"] = -1

    feat = work[["amount_num", "hour", "channel_code", "country_code", "merchant_category_code", "currency_code"]].copy()
    return feat


def detect_anomalies(df: pd.DataFrame, max_rows: int = 12) -> dict:
    """
    Retourne:
    - meta
    - dataset_summary (texte)
    - anomalies (liste dict)
    """
    if df is None or df.empty:
        return {
            "meta": {"n_rows": 0, "n_cols": 0},
            "dataset_summary": "Empty dataset.",
            "anomalies": [],
        }

    features = _build_features(df)

    # IsolationForest
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(features)
    scores = -iso.decision_function(features)  # plus grand = plus anomal
    df_out = df.copy()
    df_out["_anomaly_score"] = scores

    # Top anomalies
    top = df_out.sort_values("_anomaly_score", ascending=False).head(max_rows)

    # Build summary
    n_rows, n_cols = df.shape
    amount_col = "amount" if "amount" in df.columns else None
    amount_stats = ""
    if amount_col:
        a = pd.to_numeric(df[amount_col], errors="coerce")
        if a.notna().any():
            amount_stats = f"Amount: min={a.min():,.2f}, median={a.median():,.2f}, max={a.max():,.2f}."
    summary_parts = [
        f"Rows={n_rows}, Columns={n_cols}.",
        amount_stats,
    ]

    # Simple distribution hints
    for col in ["country", "channel", "merchant_category", "currency"]:
        if col in df.columns:
            vc = df[col].astype(str).value_counts().head(3)
            top_str = ", ".join([f"{k}({v})" for k, v in vc.items()])
            summary_parts.append(f"Top {col}: {top_str}.")

    dataset_summary = " ".join([p for p in summary_parts if p])

    anomalies = top.replace({np.nan: None}).to_dict(orient="records")
    meta = {"n_rows": n_rows, "n_cols": n_cols}

    return {"meta": meta, "dataset_summary": dataset_summary, "anomalies": anomalies}