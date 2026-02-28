import os
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# -------------------- Page + Style --------------------
st.set_page_config(page_title="Mistral Risk Copilot", layout="wide")

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px; }
.small-muted { color: rgba(255,255,255,0.65); font-size: 0.92rem; }

.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.03);
}
.card-title { font-weight: 650; margin-bottom: 6px; }
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px; font-size:0.85rem;
  border:1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.04);
  margin-right: 6px; margin-top: 4px;
}
.b-green{border-color:rgba(0,255,160,0.35);}
.b-red{border-color:rgba(255,60,60,0.35);}
.b-blue{border-color:rgba(80,160,255,0.45);}
.b-yellow{border-color:rgba(255,200,0,0.45);}
.b-purple{border-color:rgba(190,120,255,0.45);}
hr { border: none; height: 1px; background: rgba(255,255,255,0.10); margin: 14px 0; }

div.stButton > button, div.stDownloadButton > button { border-radius: 12px; }
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,0.10); }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Session state --------------------
def init_state():
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = DEFAULT_BACKEND_URL
    if "result" not in st.session_state:
        st.session_state.result = None
    if "anoms_df" not in st.session_state:
        st.session_state.anoms_df = None
    if "csv_df" not in st.session_state:
        st.session_state.csv_df = None
    if "exec_summary" not in st.session_state:
        st.session_state.exec_summary = None
    if "explain_result" not in st.session_state:
        st.session_state.explain_result = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "filters" not in st.session_state:
        st.session_state.filters = {
            "country": [],
            "channel": [],
            "merchant_category": [],
            "currency": [],
            "account_id": [],
        }
    # ✅ Persist active tab across reruns (default set ONCE before widget is created)
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "📌 Overview"

init_state()

# -------------------- Helpers --------------------
def ping_backend(url: str):
    try:
        r = requests.get(f"{url}/health", timeout=3)
        if r.status_code == 200:
            return True, r.json()
        return False, None
    except Exception:
        return False, None

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def risk_label(score: float):
    if score >= 75:
        return "HIGH", "b-red"
    if score >= 40:
        return "MEDIUM", "b-yellow"
    return "LOW", "b-green"

def df_apply_filters(df: pd.DataFrame, filters: dict):
    """Apply multi-select filters robustly (string match), returns filtered df."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col, selected in filters.items():
        if selected and col in out.columns:
            out = out[out[col].astype(str).isin([str(x) for x in selected])]
    return out

def llm_get_score(llm_result: dict) -> float | None:
    if not isinstance(llm_result, dict):
        return None
    v = llm_result.get("overall_risk_score")
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def render_llm_pretty(llm: dict):
    """Pretty LLM output: score + findings + actions + SAR summary."""
    if not isinstance(llm, dict):
        st.info("No LLM result.")
        return

    score = llm_get_score(llm)
    if score is not None:
        label, klass = risk_label(score)
        st.markdown(
            f"<span class='badge {klass}'>Risk: {label}</span>"
            f"<span class='badge b-purple'>Score: {score:.1f}/100</span>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(score / 100.0, 0.0), 1.0))
    else:
        st.markdown("<span class='badge b-yellow'>Risk: Unknown</span>", unsafe_allow_html=True)

    findings = llm.get("top_findings", [])
    if isinstance(findings, list) and findings:
        st.markdown("### 🔎 Top Findings")
        for i, f in enumerate(findings[:6], start=1):
            title = (f or {}).get("title", f"Finding {i}")
            why = (f or {}).get("why_suspicious", "")
            ev = (f or {}).get("evidence", "")
            with st.expander(f"{i}. {title}", expanded=(i <= 2)):
                st.markdown("**Why**")
                st.write(why if why else "—")
                st.markdown("**Evidence**")
                st.code(ev if ev else "—")
    else:
        st.markdown("### 🔎 Top Findings")
        st.info("No structured findings available (model may have returned raw text).")

    actions = llm.get("recommended_actions", [])
    st.markdown("### ✅ Recommended Actions")
    if isinstance(actions, list) and actions:
        for a in actions[:10]:
            st.write(f"• {a}")
    else:
        st.write("—")

    sar = llm.get("sar_summary", {})
    st.markdown("### 🧾 SAR-style Summary")
    if isinstance(sar, dict) and sar:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='card'><div class='card-title'>Subject</div>", unsafe_allow_html=True)
            st.write(sar.get("subject", "—"))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'><div class='card-title'>Timeline</div>", unsafe_allow_html=True)
            st.write(sar.get("timeline", "—"))
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card'><div class='card-title'>Suspicious activity</div>", unsafe_allow_html=True)
            st.write(sar.get("suspicious_activity", "—"))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'><div class='card-title'>Supporting details</div>", unsafe_allow_html=True)
            st.write(sar.get("supporting_details", "—"))
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><div class='card-title'>Recommendation</div>", unsafe_allow_html=True)
        st.write(sar.get("recommendation", "—"))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("—")

# -------------------- Header --------------------
ok, health = ping_backend(st.session_state.backend_url)

h1, h2 = st.columns([1.2, 0.8], vertical_alignment="center")
with h1:
    st.markdown("## 🧠 Mistral Risk Copilot")
    st.markdown("<div class='small-muted'>CSV → anomaly detection → LLM copilot (summary + explainability)</div>", unsafe_allow_html=True)

with h2:
    badges = []
    if ok:
        badges.append("<span class='badge b-green'>Backend: OK</span>")
        if health and health.get("has_api_key"):
            badges.append("<span class='badge b-blue'>LLM: Ready</span>")
        else:
            badges.append("<span class='badge b-yellow'>LLM: No key</span>")
    else:
        badges.append("<span class='badge b-red'>Backend: Offline</span>")

    st.markdown("".join(badges), unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>URL: {st.session_state.backend_url}</div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- Sidebar --------------------
st.sidebar.header("⚙️ Settings")
st.session_state.backend_url = st.sidebar.text_input("BACKEND_URL", st.session_state.backend_url)

fast_mode = st.sidebar.toggle("Skip LLM (Fast mode)", value=False)
timeout_s = st.sidebar.slider("Timeout (seconds)", 30, 600, 180, 30)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear session (restart)"):
    st.session_state.result = None
    st.session_state.anoms_df = None
    st.session_state.csv_df = None
    st.session_state.exec_summary = None
    st.session_state.explain_result = None
    st.session_state.filters = {
        "country": [],
        "channel": [],
        "merchant_category": [],
        "currency": [],
        "account_id": [],
    }
    # ✅ IMPORTANT: do NOT set st.session_state.active_tab here (radio owns it once created)
    # Let it keep the current value; and we reset uploader so app looks "fresh"
    st.session_state.uploader_key += 1
    st.rerun()

# -------------------- Upload + Preview --------------------
uploaded = st.file_uploader("📤 Upload a CSV file", type=["csv"], key=f"uploader_{st.session_state.uploader_key}")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.session_state.csv_df = df
    except Exception as e:
        st.error(f"CSV preview error: {e}")
        st.stop()

    st.markdown("### 📌 File Overview")
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Rows", f"{len(df):,}")
    cB.metric("Columns", f"{len(df.columns):,}")
    cC.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    amount_col = "amount" if "amount" in df.columns else None
    if amount_col:
        amt = pd.to_numeric(df[amount_col], errors="coerce")
        cD.metric("Max amount", f"{amt.max():,.2f}" if amt.notna().any() else "—")
    else:
        cD.metric("Max amount", "—")

    with st.expander("🔎 Preview CSV (top 50 rows)", expanded=True):
        st.dataframe(df.head(50), width="stretch")
        st.caption("Columns:")
        st.code(", ".join(df.columns.tolist()))

    run_cols = st.columns([0.26, 0.44, 0.30])
    with run_cols[0]:
        run_clicked = st.button("🚀 Run Analysis", use_container_width=True)
    with run_cols[1]:
        st.markdown("<div class='small-muted'>Fast mode = anomaly engine only. LLM mode = adds narrative + SAR + explain.</div>", unsafe_allow_html=True)
    with run_cols[2]:
        st.markdown("<div class='small-muted'>Tip: run fast first to validate the CSV, then run with LLM.</div>", unsafe_allow_html=True)
else:
    run_clicked = False

# -------------------- Run Analysis --------------------
if uploaded is not None and run_clicked:
    endpoint = "analyze_fast" if fast_mode else "analyze"
    files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}

    with st.spinner(f"Running `{endpoint}`..."):
        try:
            r = requests.post(f"{st.session_state.backend_url}/{endpoint}", files=files, timeout=timeout_s)
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach backend. Ensure FastAPI is running and BACKEND_URL is correct.")
            st.stop()
        except requests.exceptions.ReadTimeout:
            st.error("Backend timed out. Increase timeout or use Fast mode.")
            st.stop()

    if r.status_code != 200:
        st.error(f"Backend error ({r.status_code}): {r.text}")
        st.stop()

    st.session_state.result = r.json()
    st.session_state.anoms_df = pd.DataFrame(st.session_state.result.get("top_anomalies", []))
    st.session_state.exec_summary = None
    st.session_state.explain_result = None
    st.success("✅ Analysis completed")

# -------------------- Results --------------------
result = st.session_state.result
anoms_df = st.session_state.anoms_df
csv_df = st.session_state.csv_df

if result is not None:
    llm = result.get("llm_result", {})
    score = llm_get_score(llm)

    strip = st.columns([0.62, 0.38], vertical_alignment="center")
    with strip[0]:
        st.markdown("### 🧾 Investigation Dashboard")
        st.markdown("<div class='small-muted'>Triage → anomalies → account risk → LLM copilot</div>", unsafe_allow_html=True)
    with strip[1]:
        if score is not None:
            label, klass = risk_label(score)
            st.markdown(
                f"<span class='badge {klass}'>Overall: {label}</span>"
                f"<span class='badge b-purple'>{score:.1f}/100</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<span class='badge b-yellow'>Overall: Unknown</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ✅ Radio navigation (DO NOT assign into session_state after creation)
    tabs = ["📌 Overview", "🚨 Anomalies", "👤 Accounts", "🤖 Copilot (LLM)"]
    active_tab = st.radio(
        "Navigation",
        tabs,
        index=tabs.index(st.session_state.active_tab) if st.session_state.active_tab in tabs else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="active_tab",
    )

    st.markdown("---")

    # ---------- Overview ----------
    if active_tab == "📌 Overview":
        left, right = st.columns([1.05, 0.95])

        with left:
            st.markdown("<div class='card'><div class='card-title'>Dataset Summary</div>", unsafe_allow_html=True)
            st.write(result.get("dataset_summary", ""))
            st.markdown("</div>", unsafe_allow_html=True)

            if anoms_df is not None and (not anoms_df.empty) and "_anomaly_score" in anoms_df.columns:
                st.markdown("#### Anomaly score (top anomalies)")
                st.area_chart(anoms_df["_anomaly_score"], height=220)

        with right:
            st.markdown("<div class='card'><div class='card-title'>Artifacts</div>", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download full result (JSON)",
                data=json.dumps(result, ensure_ascii=False, indent=2),
                file_name="risk_copilot_result.json",
                mime="application/json",
                use_container_width=True,
            )
            if anoms_df is not None and not anoms_df.empty:
                st.download_button(
                    "⬇️ Download anomalies (CSV)",
                    data=anoms_df.to_csv(index=False).encode("utf-8"),
                    file_name="top_anomalies.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Anomalies ----------
    elif active_tab == "🚨 Anomalies":
        if anoms_df is None or anoms_df.empty:
            st.warning("No anomalies returned.")
        else:
            st.markdown("### Filters")
            fcols = st.columns(5)
            for i, col in enumerate(["country", "channel", "merchant_category", "currency", "account_id"]):
                with fcols[i]:
                    if col in anoms_df.columns:
                        opts = sorted(anoms_df[col].dropna().astype(str).unique().tolist())
                        selected = st.multiselect(
                            col,
                            options=opts,
                            default=st.session_state.filters.get(col, []),
                            key=f"flt_{col}",
                        )
                        st.session_state.filters[col] = selected
                    else:
                        st.session_state.filters[col] = []
                        st.caption(f"{col} (missing)")

            filtered = df_apply_filters(anoms_df, st.session_state.filters)

            st.markdown("### Top Anomalies (filtered)")
            st.caption(f"Showing {len(filtered):,} / {len(anoms_df):,} anomalies")
            st.dataframe(filtered, width="stretch")

    # ---------- Accounts ----------
    elif active_tab == "👤 Accounts":
        if anoms_df is None or anoms_df.empty or "account_id" not in anoms_df.columns:
            st.info("account_id not found in anomalies. Add account_id column to CSV to enable this view.")
        else:
            df_acc = anoms_df.copy()
            if "_anomaly_score" not in df_acc.columns:
                df_acc["_anomaly_score"] = 0.0
            if "amount" not in df_acc.columns:
                df_acc["amount"] = 0.0

            grp = (
                df_acc.groupby("account_id")
                .agg(
                    anomalies_count=("account_id", "count"),
                    max_anomaly_score=("_anomaly_score", "max"),
                    total_anomal_amount=("amount", "sum"),
                )
                .reset_index()
                .sort_values(["max_anomaly_score", "anomalies_count"], ascending=False)
            )
            denom = grp["anomalies_count"].max() if grp["anomalies_count"].max() else 1
            grp["risk_score_0_100"] = (
                100 * (0.7 * grp["max_anomaly_score"] + 0.3 * (grp["anomalies_count"] / denom))
            ).clip(0, 100).round(1)

            st.markdown("### Risk by account_id")
            st.dataframe(grp, width="stretch")
            st.markdown("#### Top 15 accounts by risk score")
            st.bar_chart(grp.set_index("account_id")["risk_score_0_100"].head(15), height=260)

    # ---------- Copilot ----------
    elif active_tab == "🤖 Copilot (LLM)":
        c1, c2 = st.columns([0.58, 0.42])

        with c1:
            st.markdown("<div class='card'><div class='card-title'>LLM Report</div>", unsafe_allow_html=True)
            if isinstance(llm, dict) and "raw_model_output" in llm:
                st.warning("LLM returned unparsed text. Showing raw output.")
                st.write(llm.get("raw_model_output", ""))
            else:
                render_llm_pretty(llm)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card'><div class='card-title'>✍️ Executive Summary (1 paragraph)</div>", unsafe_allow_html=True)
            if st.button("Generate Executive Summary", use_container_width=True):
                payload = {
                    "dataset_summary": result.get("dataset_summary", ""),
                    "top_anomalies": result.get("top_anomalies", []),
                }
                with st.spinner("Generating executive summary..."):
                    r2 = requests.post(
                        f"{st.session_state.backend_url}/executive_summary",
                        json=payload,
                        timeout=timeout_s,
                    )
                if r2.status_code == 200:
                    st.session_state.exec_summary = r2.json().get("executive_summary", "")
                else:
                    st.error(f"Executive summary error ({r2.status_code}): {r2.text}")

            if st.session_state.exec_summary:
                st.write(st.session_state.exec_summary)
                st.download_button(
                    "⬇️ Download summary (txt)",
                    data=st.session_state.exec_summary.encode("utf-8"),
                    file_name="executive_summary.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                st.caption("Click to generate a jury-ready paragraph.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<hr/>", unsafe_allow_html=True)

            st.markdown("<div class='card'><div class='card-title'>🔍 Explain a specific anomaly row</div>", unsafe_allow_html=True)
            if anoms_df is None or anoms_df.empty:
                st.info("Run analysis first.")
            else:
                base_df = df_apply_filters(anoms_df, st.session_state.filters) if isinstance(st.session_state.filters, dict) else anoms_df
                if base_df.empty:
                    st.warning("No anomalies match current filters.")
                else:
                    key_col = "transaction_id" if "transaction_id" in base_df.columns else None

                    if key_col:
                        choice = st.selectbox(
                            "Select transaction_id",
                            base_df[key_col].astype(str).tolist(),
                            key="explain_txid",
                        )
                        selected_row = base_df[base_df[key_col].astype(str) == str(choice)].iloc[0].to_dict()
                    else:
                        idx = st.selectbox("Select row index", base_df.index.tolist(), key="explain_idx")
                        selected_row = base_df.loc[idx].to_dict()

                    st.caption("Selected row")
                    st.json(selected_row)

                    if st.button("Explain row", use_container_width=True):
                        payload = {"dataset_summary": result.get("dataset_summary", ""), "row": selected_row}
                        with st.spinner("Explaining row..."):
                            r3 = requests.post(
                                f"{st.session_state.backend_url}/explain_anomaly",
                                json=payload,
                                timeout=timeout_s,
                            )
                        if r3.status_code == 200:
                            st.session_state.explain_result = r3.json().get("explanation", {})
                        else:
                            st.error(f"Explain error ({r3.status_code}): {r3.text}")

                    if st.session_state.explain_result:
                        st.markdown("**Explanation**")
                        st.json(st.session_state.explain_result)

            st.markdown("</div>", unsafe_allow_html=True)