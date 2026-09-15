import pandas as pd

REQUIRED_COLUMNS = [
    "request_id", "created_at", "completed_at", "caisse", "service",
    "request_type", "sla_days", "processing_days", "is_completed",
    "is_success", "team_size_fte", "productivity_target", "within_sla"
]

def quality_report(df: pd.DataFrame) -> dict:
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    duplicates = int(df.duplicated(subset=["request_id"]).sum()) if "request_id" in df.columns else len(df)
    missing_values = int(df[REQUIRED_COLUMNS].isna().sum().sum()) if not missing_columns else 0
    negative_delays = int((df["processing_days"] < 0).sum()) if "processing_days" in df.columns else 0
    invalid_completion = 0
    if {"created_at", "completed_at", "is_completed"}.issubset(df.columns):
        invalid_completion = int((
            df["is_completed"] & df["completed_at"].notna() & (df["completed_at"] < df["created_at"])
        ).sum())
    total_cells = max(1, len(df) * len(REQUIRED_COLUMNS))
    penalty = duplicates + missing_values + negative_delays + invalid_completion + len(missing_columns) * len(df)
    score = max(0.0, 1 - penalty / total_cells)
    return {
        "score": score,
        "duplicates": duplicates,
        "missing_values": missing_values,
        "negative_delays": negative_delays,
        "invalid_completion_dates": invalid_completion,
        "missing_columns": missing_columns,
    }

def top_missing_columns(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    return (
        df.isna().sum().rename("missing").to_frame()
        .assign(rate=lambda x: x["missing"] / max(1, len(df)))
        .sort_values("missing", ascending=False).head(top_n)
        .reset_index().rename(columns={"index": "column"})
    )
