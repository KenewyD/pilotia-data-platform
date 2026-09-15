from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import DATA_PATH, DATA_DIR, RANDOM_SEED, DEFAULT_ROWS, SERVICES, CAISSES, REQUEST_TYPES

def generate_synthetic_data(n_rows: int = DEFAULT_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-14")
    days = (end - start).days + 1
    created_at = start + pd.to_timedelta(rng.integers(0, days, size=n_rows), unit="D")

    service = rng.choice(SERVICES, n_rows, p=[0.28, 0.27, 0.18, 0.12, 0.15])
    caisse = rng.choice(CAISSES, n_rows)
    request_type = rng.choice(REQUEST_TYPES, n_rows, p=[0.44, 0.11, 0.18, 0.17, 0.10])

    base_delay = np.select(
        [
            service == "Indemnités journalières",
            service == "Remboursements",
            service == "Affiliation",
            service == "Complémentaire santé",
        ],
        [5.5, 2.8, 6.0, 4.2],
        default=7.2,
    )
    request_penalty = np.where(request_type == "Réclamation", 2.8, 0.0)
    weekday = pd.DatetimeIndex(created_at).weekday.to_numpy()
    monday_effect = np.where(weekday == 0, 0.9, 0.0)
    season = np.sin((pd.DatetimeIndex(created_at).dayofyear.to_numpy() / 365.25) * 2 * np.pi)

    delay = rng.gamma(shape=2.2, scale=(base_delay + request_penalty + monday_effect + 0.8 * season) / 2.2)
    delay = np.clip(delay, 0.15, 35)

    incident_mask = (
        (pd.DatetimeIndex(created_at) >= pd.Timestamp("2026-05-10"))
        & (pd.DatetimeIndex(created_at) <= pd.Timestamp("2026-05-25"))
        & (caisse == "Caisse 07")
    )
    delay[incident_mask] *= 1.85

    sla_days = np.select(
        [
            service == "Remboursements",
            service == "Complémentaire santé",
            service == "Indemnités journalières",
        ],
        [5, 7, 8],
        default=10,
    )

    success_prob = np.where(request_type == "Réclamation", 0.83, 0.94)
    success_prob -= np.where(delay > sla_days, 0.07, 0.0)
    is_success = rng.random(n_rows) < success_prob

    completion_prob = 0.965 - np.where(created_at > pd.Timestamp("2026-08-25"), 0.12, 0.0)
    is_completed = rng.random(n_rows) < completion_prob

    completed_at = pd.Series(created_at + pd.to_timedelta(delay, unit="D"))
    completed_at = completed_at.where(is_completed, pd.NaT)

    df = pd.DataFrame({
        "request_id": [f"REQ-{i:08d}" for i in range(1, n_rows + 1)],
        "created_at": created_at,
        "completed_at": completed_at,
        "caisse": caisse,
        "service": service,
        "request_type": request_type,
        "sla_days": sla_days,
        "processing_days": np.round(delay, 2),
        "is_completed": is_completed,
        "is_success": is_success & is_completed,
        "team_size_fte": rng.integers(8, 31, size=n_rows),
        "productivity_target": np.round(rng.normal(34, 5, size=n_rows).clip(18, 50), 2),
    })
    df["within_sla"] = df["is_completed"] & (df["processing_days"] <= df["sla_days"])

    duplicate_sample = df.sample(frac=0.001, random_state=seed)
    df = pd.concat([df, duplicate_sample], ignore_index=True)
    missing_idx = rng.choice(df.index, size=max(10, int(len(df) * 0.0015)), replace=False)
    df.loc[missing_idx, "request_type"] = None
    return df.sort_values("created_at").reset_index(drop=True)

def ensure_data(path: Path = DATA_PATH, n_rows: int = DEFAULT_ROWS) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df = generate_synthetic_data(n_rows=n_rows)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path, parse_dates=["created_at", "completed_at"])

if __name__ == "__main__":
    df = ensure_data()
    print(f"Dataset prêt : {len(df):,} lignes -> {DATA_PATH}")
