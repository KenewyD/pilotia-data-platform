import pandas as pd

def compute_kpis(df: pd.DataFrame) -> dict:
    total = len(df)
    completed = int(df["is_completed"].sum())
    success = int(df["is_success"].sum())
    backlog = total - completed
    cdf = df[df["is_completed"]]
    return {
        "demandes": total,
        "traitees": completed,
        "backlog": backlog,
        "taux_realisation": completed / total if total else 0.0,
        "taux_succes": success / completed if completed else 0.0,
        "taux_echec": 1 - (success / completed) if completed else 0.0,
        "delai_median": float(cdf["processing_days"].median()) if not cdf.empty else 0.0,
        "taux_sla": float(cdf["within_sla"].mean()) if not cdf.empty else 0.0,
    }

def daily_activity(df: pd.DataFrame) -> pd.DataFrame:
    received = (
        df.assign(date=df["created_at"].dt.floor("D"))
        .groupby("date").size().rename("received")
    )
    completed_df = df[df["is_completed"]].copy()
    completed_df["date"] = completed_df["completed_at"].dt.floor("D")
    completed = completed_df.groupby("date").size().rename("completed")
    daily = pd.concat([received, completed], axis=1).fillna(0).reset_index()
    daily[["received", "completed"]] = daily[["received", "completed"]].astype(int)
    daily["net_flow"] = daily["received"] - daily["completed"]
    daily["backlog_proxy"] = daily["net_flow"].cumsum().clip(lower=0)
    return daily.sort_values("date")
