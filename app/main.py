from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import ensure_data
from src.metrics import compute_kpis, daily_activity
from src.data_quality import quality_report, top_missing_columns
from src.forecasting import fit_and_forecast
from src.anomaly_detection import detect_anomalies
from src.capacity import simulate_capacity

st.set_page_config(page_title="PILOT'IA", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.20);
    padding: 14px;
    border-radius: 12px;
}
.small-note {font-size: 0.88rem; opacity: 0.75;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    return ensure_data()

@st.cache_data(show_spinner=False)
def cached_forecast(df, horizon):
    return fit_and_forecast(df, horizon=horizon)

@st.cache_data(show_spinner=False)
def cached_anomalies(df):
    return detect_anomalies(df)

df = load_data()

st.sidebar.title("PILOT'IA")
st.sidebar.caption("Pilotage intelligent de l'activité")

min_date = df["created_at"].min().date()
max_date = df["created_at"].max().date()
date_range = st.sidebar.date_input(
    "Période", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    d1 = pd.Timestamp(date_range[0])
    d2 = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    d1 = pd.Timestamp(min_date)
    d2 = pd.Timestamp(max_date) + pd.Timedelta(days=1)

caisse_options = sorted(df["caisse"].dropna().unique())
service_options = sorted(df["service"].dropna().unique())

selected_caisses = st.sidebar.multiselect("Caisses", caisse_options, default=caisse_options)
selected_services = st.sidebar.multiselect("Services", service_options, default=service_options)

filtered = df[
    (df["created_at"] >= d1)
    & (df["created_at"] < d2)
    & (df["caisse"].isin(selected_caisses))
    & (df["service"].isin(selected_services))
].copy()

page = st.sidebar.radio(
    "Navigation",
    [
        "Vue exécutive",
        "Délais & qualité de service",
        "Charge & ressources",
        "Prévision",
        "Anomalies",
        "Qualité des données",
        "Gouvernance",
    ],
)

st.title("PILOT'IA")
st.caption("Prototype de pilotage opérationnel — données 100 % synthétiques.")

if filtered.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

if page == "Vue exécutive":
    kpi = compute_kpis(filtered)
    cols = st.columns(6)
    cols[0].metric("Demandes", f"{kpi['demandes']:,}".replace(",", " "))
    cols[1].metric("Traitées", f"{kpi['traitees']:,}".replace(",", " "))
    cols[2].metric("Backlog", f"{kpi['backlog']:,}".replace(",", " "))
    cols[3].metric("Taux réalisation", f"{kpi['taux_realisation']:.1%}")
    cols[4].metric("Dans le SLA", f"{kpi['taux_sla']:.1%}")
    cols[5].metric("Délai médian", f"{kpi['delai_median']:.1f} j")

    daily = daily_activity(filtered)
    c1, c2 = st.columns([2, 1])

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["received"], name="Reçues"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["completed"], name="Traitées"))
        fig.update_layout(title="Flux quotidien", height=420, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_service = (
            filtered.groupby("service")
            .agg(demandes=("request_id", "count"))
            .reset_index()
            .sort_values("demandes", ascending=True)
        )
        fig = px.bar(by_service, x="demandes", y="service", orientation="h", title="Volume par service")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Synthèse par caisse")
    table = (
        filtered.groupby("caisse")
        .agg(
            demandes=("request_id", "count"),
            traitees=("is_completed", "sum"),
            delai_median=("processing_days", "median"),
            taux_sla=("within_sla", "mean"),
            taux_succes=("is_success", "mean"),
        )
        .reset_index()
    )
    table["taux_sla"] = (table["taux_sla"] * 100).round(1)
    table["taux_succes"] = (table["taux_succes"] * 100).round(1)
    st.dataframe(table, use_container_width=True, hide_index=True)

elif page == "Délais & qualité de service":
    st.subheader("Délais de traitement et qualité de service")
    completed = filtered[filtered["is_completed"]].copy()
    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(
            completed, x="service", y="processing_days",
            points=False, title="Distribution des délais par service"
        )
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        weekly = (
            completed.assign(week=completed["created_at"].dt.to_period("W").dt.start_time)
            .groupby("week")
            .agg(taux_sla=("within_sla", "mean"), delai_median=("processing_days", "median"))
            .reset_index()
        )
        fig = px.line(weekly, x="week", y="taux_sla", title="Évolution du respect du SLA")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    service_perf = (
        completed.groupby(["caisse", "service"])
        .agg(
            dossiers=("request_id", "count"),
            delai_median=("processing_days", "median"),
            taux_sla=("within_sla", "mean"),
            taux_succes=("is_success", "mean"),
        )
        .reset_index()
    )
    st.dataframe(service_perf, use_container_width=True, hide_index=True)

elif page == "Charge & ressources":
    st.subheader("Simulation charge ↔ ressources")
    st.write("Testez un scénario opérationnel et mesurez le risque de saturation.")

    c1, c2, c3, c4 = st.columns(4)
    agents = c1.number_input("ETP planifiés", 1.0, 500.0, 35.0, 1.0)
    absence = c2.slider("Taux d'absence", 0.0, 0.40, 0.08, 0.01)
    productivity = c3.number_input("Productivité / ETP / jour", 5.0, 100.0, 35.0, 1.0)
    workdays = c4.number_input("Jours ouvrés", 1, 31, 5)

    daily = daily_activity(filtered)
    baseline_volume = float(daily["received"].tail(28).mean() * workdays)
    expected_growth = st.slider("Variation attendue de la charge", -0.30, 0.80, 0.10, 0.05)
    expected_volume = baseline_volume * (1 + expected_growth)

    result = simulate_capacity(
        agents_fte=agents,
        absence_rate=absence,
        productivity_per_fte_day=productivity,
        workdays=workdays,
        expected_volume=expected_volume,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Charge estimée", f"{expected_volume:,.0f}".replace(",", " "))
    m2.metric("Capacité", f"{result['capacity']:,.0f}".replace(",", " "))
    m3.metric("Utilisation", f"{result['utilization']:.1%}")
    m4.metric("Risque saturation", f"{result['risk']:.1%}")

    if result["gap"] < 0:
        st.error(result["recommendation"])
    else:
        st.success(result["recommendation"])

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result["utilization"] * 100,
        title={"text": "Taux d'utilisation de la capacité"},
        gauge={"axis": {"range": [0, 130]}},
    ))
    gauge.update_layout(height=350)
    st.plotly_chart(gauge, use_container_width=True)

elif page == "Prévision":
    st.subheader("Prévision des demandes")
    horizon = st.radio("Horizon", [7, 30], horizontal=True, index=1)

    forecast, model_metrics = cached_forecast(filtered, horizon)
    daily = daily_activity(filtered)[["date", "received"]].tail(90)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["received"], name="Historique"))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["forecast"], name="Prévision"))
    fig.add_trace(go.Scatter(
        x=list(forecast["date"]) + list(forecast["date"][::-1]),
        y=list(forecast["upper"]) + list(forecast["lower"][::-1]),
        fill="toself", name="Intervalle indicatif", line={"width": 0}
    ))
    fig.update_layout(height=460, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE validation", f"{model_metrics['mae']:.1f} dossiers/j")
    c2.metric("Moyenne historique", f"{model_metrics['historical_daily_mean']:.1f}/j")
    c3.metric("Moyenne prévisionnelle", f"{model_metrics['forecast_daily_mean']:.1f}/j")

    delta = model_metrics["forecast_daily_mean"] / max(model_metrics["historical_daily_mean"], 1e-9) - 1
    if delta > 0.08:
        st.warning(f"Hausse prévisionnelle de l'activité : {delta:.1%}.")
    elif delta < -0.08:
        st.info(f"Baisse prévisionnelle de l'activité : {delta:.1%}.")
    else:
        st.success("Activité prévisionnelle relativement stable.")

elif page == "Anomalies":
    st.subheader("Détection d'anomalies")
    anomalies = cached_anomalies(filtered)
    flagged = anomalies[anomalies["is_anomaly"]].copy()

    c1, c2 = st.columns([1, 2])
    c1.metric("Signaux détectés", len(flagged))
    c1.metric("Part des observations", f"{len(flagged) / max(1, len(anomalies)):.2%}")

    with c2:
        fig = px.scatter(
            anomalies, x="volume", y="median_delay", color="is_anomaly",
            hover_data=["date", "caisse", "service", "sla_rate", "success_rate"],
            title="Volume vs délai médian"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Un signal d'anomalie doit être investigué. Il ne constitue pas à lui seul une conclusion métier.")
    st.dataframe(
        flagged[["date", "caisse", "service", "volume", "median_delay", "sla_rate", "success_rate", "anomaly_score"]].head(50),
        use_container_width=True, hide_index=True
    )

elif page == "Qualité des données":
    st.subheader("Contrôle qualité")
    report = quality_report(filtered)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score qualité", f"{report['score']:.1%}")
    c2.metric("Doublons", f"{report['duplicates']:,}".replace(",", " "))
    c3.metric("Valeurs manquantes", f"{report['missing_values']:,}".replace(",", " "))
    c4.metric("Délais négatifs", report["negative_delays"])

    missing = top_missing_columns(filtered)
    fig = px.bar(missing, x="column", y="missing", title="Valeurs manquantes par variable")
    st.plotly_chart(fig, use_container_width=True)

    if report["missing_columns"]:
        st.error("Colonnes obligatoires manquantes : " + ", ".join(report["missing_columns"]))
    else:
        st.success("Toutes les colonnes obligatoires sont présentes.")

    st.markdown("""
    **Règles contrôlées**
    - unicité de l'identifiant de demande ;
    - présence des colonnes attendues ;
    - valeurs manquantes ;
    - délai de traitement non négatif ;
    - cohérence entre date de création et date de clôture.
    """)

elif page == "Gouvernance":
    st.subheader("Gouvernance & sécurité")
    st.info("Cette démonstration fonctionne exclusivement avec des données synthétiques.")
    st.markdown("""
    ### Principes appliqués
    - aucun nom ou prénom ;
    - aucun numéro de sécurité sociale ;
    - aucune donnée médicale individuelle ;
    - identifiants techniques fictifs ;
    - agrégation dans les vues de pilotage ;
    - minimisation des données ;
    - reproductibilité du pipeline ;
    - séparation entre données, logique métier et interface.

    ### Utilisation responsable
    Le modèle d'anomalies sert à prioriser des investigations.
    Il ne prend aucune décision sur une personne.

    Les prévisions sont des aides au pilotage et doivent être confrontées
    aux événements métier, réglementaires et organisationnels.

    ### Positionnement
    PILOT'IA est un prototype professionnel sur données synthétiques.
    Il ne représente pas l'architecture réelle d'un organisme existant.
    """)

st.divider()
st.markdown(
    '<div class="small-note">PILOT’IA — démonstrateur Data Analytics sur données 100 % synthétiques.</div>',
    unsafe_allow_html=True,
)
