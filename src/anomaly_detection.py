import pandas as pd
from sklearn.ensemble import IsolationForest

def build_daily_service_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["date"] = work["created_at"].dt.floor("D")
    return (
        work.groupby(["date", "caisse", "service"])
        .agg(
            volume=("request_id", "count"),
            median_delay=("processing_days", "median"),
            success_rate=("is_success", "mean"),
            sla_rate=("within_sla", "mean"),
        )
        .reset_index()
    )

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    features = build_daily_service_features(df)
    if len(features) < 30:
        features["anomaly_score"] = 0.0
        features["is_anomaly"] = False
        return features
    X = features[["volume", "median_delay", "success_rate", "sla_rate"]].fillna(0)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    features["is_anomaly"] = model.fit_predict(X) == -1
    features["anomaly_score"] = -model.score_samples(X)
    return features.sort_values(["is_anomaly", "anomaly_score"], ascending=[False, False])
