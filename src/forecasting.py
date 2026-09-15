import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from src.metrics import daily_activity

def _features(dates, origin):
    dates = pd.DatetimeIndex(dates)
    t = (dates - origin).days.astype(float)
    return pd.DataFrame({
        "trend": t,
        "dow_sin": np.sin(2 * np.pi * dates.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * dates.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * dates.month / 12),
        "month_cos": np.cos(2 * np.pi * dates.month / 12),
    })

def fit_and_forecast(df: pd.DataFrame, horizon: int = 30):
    daily = daily_activity(df)[["date", "received"]].sort_values("date")
    if len(daily) < 30:
        raise ValueError("Pas assez d'historique pour la prévision.")
    origin = daily["date"].min()
    split = max(14, int(len(daily) * 0.85))
    X = _features(daily["date"], origin)
    y = daily["received"].astype(float)
    model = Ridge(alpha=3.0)
    model.fit(X.iloc[:split], y.iloc[:split])
    mae = mean_absolute_error(y.iloc[split:], model.predict(X.iloc[split:]))
    model.fit(X, y)
    future_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    pred = np.clip(model.predict(_features(future_dates, origin)), 0, None)
    forecast = pd.DataFrame({
        "date": future_dates,
        "forecast": np.round(pred).astype(int),
        "lower": np.round(np.clip(pred - 1.64 * mae, 0, None)).astype(int),
        "upper": np.round(pred + 1.64 * mae).astype(int),
    })
    metrics = {
        "mae": float(mae),
        "historical_daily_mean": float(y.mean()),
        "forecast_daily_mean": float(forecast["forecast"].mean()),
    }
    return forecast, metrics
