"""
schema_utils.py
Schema inference (for local CSV/JSON), schema comparison utilities,
and AI-driven explanations & SQL generation (via llm_utils.chat_llm).
Also optional Snowflake fetch helpers.
"""

import json
from typing import Dict, Tuple, List
import pandas as pd

from llm_utils import chat_llm


# -------------------------
# Schema inference from DataFrame
# -------------------------
def infer_schema_from_df(df: pd.DataFrame) -> Dict[str, str]:
    schema = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if 'int' in dtype:
            schema[col.upper()] = 'NUMBER'
        elif 'float' in dtype:
            schema[col.upper()] = 'FLOAT'
        elif 'bool' in dtype:
            schema[col.upper()] = 'BOOLEAN'
        elif 'datetime' in dtype or 'date' in dtype:
            schema[col.upper()] = 'TIMESTAMP_NTZ'
        else:
            schema[col.upper()] = 'TEXT'
    return schema


# -------------------------
# Schema comparison
# -------------------------
def compare_schemas(existing: Dict[str, str], new: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Tuple[str, str]]]:
    """Return dicts of new cols, missing cols, and conflicts."""
    existing = {k.upper(): v for k, v in existing.items()}
    new = {k.upper(): v for k, v in new.items()}

    new_cols = {col: new[col] for col in new if col not in existing}
    missing_cols = {col: existing[col] for col in existing if col not in new}
    conflicts = {col: (existing[col], new[col]) for col in existing if col in new and existing[col] != new[col]}

    return new_cols, missing_cols, conflicts


def _format_schema(s: Dict[str, str]) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in sorted(s.items())])


# -------------------------
# AI Explanation
# -------------------------
def explain_changes_with_ai(existing: Dict[str, str], new: Dict[str, str], table_name: str) -> str:
    new_cols, missing_cols, conflicts = compare_schemas(existing, new)

    # If no changes at all → static safe message
    if not new_cols and not missing_cols and not conflicts:
        return f"""✅ No schema changes detected for table `{table_name}`.  
Both existing and candidate schemas are identical.  
No risks, conflicts, or SQL migrations are required."""

    # Otherwise → call LLM to explain
    prompt = f"""
You are an expert Snowflake engineer. Compare the two table schemas and explain the changes clearly.

Table: {table_name}

Existing schema:
{_format_schema(existing)}

New schema (candidate):
{_format_schema(new)}

Describe:
1) Added columns (with types)
2) Removed or missing columns and their impact
3) Data type conflicts and safe migration advice
4) Risks (NULLability, backfills, ingestion issues)
Keep it concise with bullet points.
"""
    return chat_llm(prompt)


# -------------------------
# AI SQL Generation (diff-aware)
# -------------------------
def generate_sql_with_ai(existing: Dict[str, str], new: Dict[str, str], table_name: str) -> str:
    new_cols, missing_cols, conflicts = compare_schemas(existing, new)

    sql_statements = []

    # Add new columns
    for col, dtype in new_cols.items():
        sql_statements.append(f"ALTER TABLE {table_name} ADD COLUMN {col} {dtype} NULL;")

    # Warn about missing cols
    for col, dtype in missing_cols.items():
        sql_statements.append(f"-- NOTE: Column {col} ({dtype}) exists in old schema but not in new. Drop only if intentional.")

    # Handle conflicts
    for col, (old_type, new_type) in conflicts.items():
        sql_statements.append(f"ALTER TABLE {table_name} ALTER COLUMN {col} SET DATA TYPE {new_type};")

    # No changes
    if not sql_statements:
        return "-- No schema changes detected. No ALTER TABLE needed."

    return "\n".join(sql_statements)


# -------------------------
# Snowflake helpers
# -------------------------
def make_snowflake_connection(cfg: dict):
    import snowflake.connector
    conn = snowflake.connector.connect(
        account=cfg.get("SNOWFLAKE_ACCOUNT"),
        user=cfg.get("SNOWFLAKE_USER"),
        password=cfg.get("SNOWFLAKE_PASSWORD"),
        warehouse=cfg.get("SNOWFLAKE_WAREHOUSE"),
        database=cfg.get("SNOWFLAKE_DATABASE"),
        schema=cfg.get("SNOWFLAKE_SCHEMA"),
    )
    return conn


def fetch_schema_from_snowflake(conn, database: str, schema: str, table: str) -> Dict[str, str]:
    q = f"""
    SELECT COLUMN_NAME, DATA_TYPE
    FROM {database}.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
    ORDER BY ORDINAL_POSITION
    """
    cur = conn.cursor()
    try:
        cur.execute(q, (schema.upper(), table.upper()))
        rows = cur.fetchall()
        return {r[0].upper(): r[1].upper() for r in rows}
    finally:
        cur.close()
