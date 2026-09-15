from src.data_generator import generate_synthetic_data
from src.metrics import compute_kpis
from src.data_quality import quality_report
from src.capacity import simulate_capacity
from src.forecasting import fit_and_forecast
from src.anomaly_detection import detect_anomalies

def test_generation_and_kpis():
    df = generate_synthetic_data(n_rows=5000, seed=1)
    kpi = compute_kpis(df)
    assert kpi["demandes"] >= 5000
    assert 0 <= kpi["taux_realisation"] <= 1
    assert 0 <= kpi["taux_sla"] <= 1

def test_quality():
    df = generate_synthetic_data(n_rows=3000, seed=2)
    report = quality_report(df)
    assert 0 <= report["score"] <= 1

def test_capacity():
    result = simulate_capacity(30, 0.08, 35, 5, 6000)
    assert result["capacity"] > 0
    assert 0 <= result["risk"] <= 1

def test_forecast():
    df = generate_synthetic_data(n_rows=15000, seed=3)
    forecast, metrics = fit_and_forecast(df, horizon=7)
    assert len(forecast) == 7
    assert metrics["mae"] >= 0

def test_anomalies():
    df = generate_synthetic_data(n_rows=10000, seed=4)
    out = detect_anomalies(df, contamination=0.02)
    assert "is_anomaly" in out.columns
