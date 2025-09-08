"""
app.py
Streamlit UI for AI-Powered Snowflake Schema Evolution Assistant (robust, tested)
"""

import os
import json
import pandas as pd
import streamlit as st
import altair as alt
import warnings
import logging


logging.getLogger("torch").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from schema_utils import (
    infer_schema_from_df, compare_schemas,
    explain_changes_with_ai, generate_sql_with_ai,
    fetch_schema_from_snowflake, make_snowflake_connection,
)
from chatbot import answer_question
from faq import FAQS
from logging_utils import log_user_query


# -------------------------
# Helper / config
# -------------------------
def _DEF(k, d=None):
    try:
        return st.secrets.get(k, os.getenv(k, d))
    except Exception:
        return os.getenv(k, d)


st.set_page_config(page_title="Snowflake Schema Evolution Assistant", layout="wide")
st.title("🤖 AI-Powered Snowflake Schema Evolution Assistant")

# initialize session state
if "latest_context" not in st.session_state:
    st.session_state.latest_context = ""
if "live_schema" not in st.session_state:
    st.session_state.live_schema = None
if "prev_schema_text" not in st.session_state:
    st.session_state.prev_schema_text = "{}"
if "prev_schema" not in st.session_state:
    st.session_state.prev_schema = {}

# -------------------------
# Layout: tabs
# -------------------------
schema_tab, chat_tab, faq_tab = st.tabs(["📊 Schema Analysis", "💬 Chatbot", "❓ FAQ"])


# -------------------------
# Chatbot Tab
# -------------------------
with chat_tab:
    st.subheader("Interactive Q&A")
    question = st.text_input("Ask anything about Snowflake schema evolution or your latest table changes:")
    if st.button("Get Answer", key="chat_get_answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            extra_ctx = st.session_state.latest_context or None
            with st.spinner("Thinking..."):
                try:
                    answer = answer_question(question, extra_context=extra_ctx)
                except Exception as e:
                    answer = f"Error while calling LLM: {e}"
            st.markdown(answer)
            log_user_query("guest", question, answer)


# -------------------------
# Schema Analysis Tab
# -------------------------
with schema_tab:
    st.subheader("Analyze Table Schema Changes")
    mode = st.radio("Mode", ["Upload CSV/JSON (Local Demo)", "Snowflake Live (Optional)"], index=0)

    # -------------------------
    # Local Demo mode: upload file
    # -------------------------
    if mode == "Upload CSV/JSON (Local Demo)":
        existing_text = st.text_area(
            "Existing schema (JSON: { column: TYPE })",
            value=json.dumps({
                "FIRST_NAME": "TEXT",
                "LAST_NAME": "TEXT",
                "EMAIL": "TEXT",
                "ADDRESS": "TEXT",
                "CITY": "TEXT",
                "JOB_START_DATE": "TIMESTAMP_NTZ"
            }, indent=2),
            height=150,
            key="existing_schema_text"
        )
        uploaded = st.file_uploader("Upload CSV or JSON (rows)", type=["csv", "json"], key="upload_demo")
        table_name = st.text_input("Target table name", value="EMPLOYEE", key="local_table_name")

        if uploaded:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_json(uploaded)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                df = None

            if df is not None:
                new_schema = infer_schema_from_df(df)
                st.write("### Detected schema (from file)")
                st.json(new_schema)

                try:
                    existing_schema = json.loads(existing_text)
                except Exception as e:
                    st.error(f"Invalid existing schema JSON: {e}")
                    existing_schema = {}

                new_cols, missing_cols, conflicts = compare_schemas(existing_schema, new_schema)
                st.write("**Added columns:**", new_cols)
                st.write("**Missing columns:**", missing_cols)
                st.write("**Type conflicts:**", conflicts)

                if st.button("Ask AI to explain & generate SQL", key="local_ai_explain"):
                    with st.spinner("AI analyzing changes..."):
                        explanation = explain_changes_with_ai(existing_schema, new_schema, table_name)
                        sql = generate_sql_with_ai(existing_schema, new_schema, table_name)
                    st.markdown("### AI Explanation")
                    st.markdown(explanation)
                    st.markdown("### AI-Generated SQL")
                    st.code(sql, language="sql")

                    # Line chart visualization
                    df_chart = pd.DataFrame({
                        "ChangeType": ["New Columns", "Missing Columns", "Conflicts"],
                        "Count": [len(new_cols), len(missing_cols), len(conflicts)]
                    })
                    st.markdown("### Schema Changes Visualization")
                    line_chart = (
                        alt.Chart(df_chart)
                        .mark_line(color="#ff2b2b", point=alt.OverlayMarkDef(color="#ff2b2b"))
                        .encode(x="ChangeType", y="Count")
                        .properties(width="container", height=300)
                    )
                    st.altair_chart(line_chart, use_container_width=True)

                    st.session_state.latest_context = f"Existing: {existing_schema}\nNew: {new_schema}\nSQL: {sql}"

    # -------------------------
    # Snowflake Live Mode
    # -------------------------
    else:
        st.info("🔗 Connect to Snowflake (account credentials come from .streamlit/secrets.toml)")

        cfg = {k: _DEF(k, "") for k in [
            "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_WAREHOUSE"
        ]}

        missing = [k for k, v in cfg.items() if not v]
        if missing:
            st.error(f"❌ Missing Snowflake credentials in secrets.toml: {', '.join(missing)}")
            st.stop()

        try:
            conn = make_snowflake_connection(cfg)
        except Exception as e:
            st.error(f"❌ Could not connect to Snowflake: {e}")
            st.stop()

        # Databases
        db_list = []
        try:
            cur = conn.cursor()
            cur.execute("SHOW DATABASES")
            db_list = [r[1] for r in cur.fetchall()]
            cur.close()
        except Exception as e:
            st.error(f"Failed to fetch databases: {e}")

        db = st.selectbox("Database", db_list if db_list else ["(no databases found)"], key="sf_db")
        if db == "(no databases found)":
            db = None

        # Schemas
        schema_list = []
        if db:
            try:
                cur = conn.cursor()
                cur.execute(f"SHOW SCHEMAS IN DATABASE {db}")
                schema_list = [r[1] for r in cur.fetchall()]
                cur.close()
            except Exception as e:
                st.error(f"Failed to fetch schemas for {db}: {e}")

        sc = st.selectbox("Schema", schema_list if schema_list else ["(no schemas found)"], key="sf_schema")
        if sc == "(no schemas found)":
            sc = None

        # Tables
        table_list = []
        if db and sc:
            try:
                cur = conn.cursor()
                cur.execute(f"SHOW TABLES IN SCHEMA {db}.{sc}")
                table_list = [r[1] for r in cur.fetchall()]
                cur.close()
            except Exception as e:
                st.error(f"Failed to fetch tables for {db}.{sc}: {e}")

        table = st.selectbox("Table", table_list if table_list else ["(no tables found)"], key="sf_table")
        if table == "(no tables found)":
            table = None

        # Fetch schema
        if st.button("Fetch live schema", key="sf_fetch_schema") and db and sc and table:
            try:
                with st.spinner(f"Fetching schema for {db}.{sc}.{table} ..."):
                    live_schema = fetch_schema_from_snowflake(conn, db, sc, table)
                st.success(f"✅ Live schema fetched for {db}.{sc}.{table}")
                st.json(live_schema)
                st.session_state.live_schema = live_schema
            except Exception as e:
                st.error(f"❌ Snowflake error: {e}")
                st.session_state.live_schema = None

        prev_text = st.text_area("Paste previous schema snapshot (dict {col:type} OR row-array JSON)",
                                 value=st.session_state.prev_schema_text, height=180, key="prev_schema_text")

        # Validate schema
        if st.button("Validate previous schema JSON", key="validate_prev"):
            try:
                parsed = json.loads(prev_text)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                parsed = None

            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                df = pd.DataFrame(parsed)
                inferred = infer_schema_from_df(df)
                st.success("Detected row-array JSON. Inferred schema:")
                st.json(inferred)
                st.session_state.prev_schema = inferred
            elif isinstance(parsed, dict):
                normalized = {k.upper(): v for k, v in parsed.items()}
                st.success("Valid schema JSON (dict).")
                st.json(normalized)
                st.session_state.prev_schema = normalized
            else:
                st.error("Unsupported JSON structure.")

        # Analyze
        if st.button("Analyze changes with AI", key="analyze_live"):
            live_schema = st.session_state.live_schema
            prev_schema = st.session_state.prev_schema

            if not live_schema:
                st.error("No live schema present. Fetch it first.")
            else:
                try:
                    with st.spinner("AI analyzing live schema changes..."):
                        explanation = explain_changes_with_ai(prev_schema, live_schema, f"{db}.{sc}.{table}")
                        sql = generate_sql_with_ai(prev_schema, live_schema, f"{db}.{sc}.{table}")

                    st.markdown("### AI Explanation")
                    st.markdown(explanation)
                    st.markdown("### AI-Generated SQL")
                    st.code(sql, language="sql")

                    # Visualization
                    new_cols, missing_cols, conflicts = compare_schemas(prev_schema, live_schema)
                    df_chart = pd.DataFrame({
                        "ChangeType": ["New Columns", "Missing Columns", "Conflicts"],
                        "Count": [len(new_cols), len(missing_cols), len(conflicts)]
                    })
                    st.markdown("### Schema Changes Visualization")
                    line_chart = (
                        alt.Chart(df_chart)
                        .mark_line(color="#ff2b2b", point=alt.OverlayMarkDef(color="#ff2b2b"))
                        .encode(x="ChangeType", y="Count")
                        .properties(width="container", height=300)
                    )
                    st.altair_chart(line_chart, use_container_width=True)

                    st.session_state.latest_context = f"Prev: {prev_schema}\nLive: {live_schema}\nSQL: {sql}"
                except Exception as e:
                    st.error(f"Error while calling AI: {e}")


# -------------------------
# FAQ Tab
# -------------------------
with faq_tab:
    st.subheader("Frequently Asked Questions")
    selected = st.selectbox("Choose a question:", [""] + FAQS, key="faq_select")
    if selected and st.button("Answer with AI", key="faq_answer"):
        try:
            ans = answer_question(selected)
            st.markdown(ans)
        except Exception as e:
            st.error(f"Error while calling AI: {e}")
