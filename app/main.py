from pathlib import Path
import sys

# ============================================================
# 1. CONFIGURATION DU PROJET
# ============================================================

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


# ============================================================
# 2. CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="PILOT'IA | Pilotage intelligent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. STYLE GLOBAL
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }

    h1 {
        font-size: 2.55rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2 {
        font-weight: 700 !important;
    }

    h3 {
        font-weight: 650 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.55);
        border: 1px solid rgba(100,100,100,0.15);
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.90rem;
        opacity: 0.80;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
        font-weight: 750;
    }

    .pilotia-subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 10px;
    }

    .pilotia-card {
        padding: 18px 20px;
        border-radius: 14px;
        border: 1px solid rgba(100,100,100,0.15);
        background: rgba(255,255,255,0.55);
        margin-bottom: 12px;
    }

    .pilotia-card-title {
        font-size: 0.85rem;
        opacity: 0.70;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .pilotia-card-value {
        font-size: 1.4rem;
        font-weight: 750;
        margin-top: 4px;
    }

    .pilotia-info {
        padding: 14px 16px;
        border-left: 4px solid #4c78a8;
        background: rgba(76,120,168,0.08);
        border-radius: 8px;
        margin-top: 8px;
        margin-bottom: 14px;
    }

    .pilotia-success {
        padding: 14px 16px;
        border-left: 4px solid #2ca02c;
        background: rgba(44,160,44,0.08);
        border-radius: 8px;
        margin-top: 8px;
        margin-bottom: 14px;
    }

    .pilotia-warning {
        padding: 14px 16px;
        border-left: 4px solid #ff9800;
        background: rgba(255,152,0,0.08);
        border-radius: 8px;
        margin-top: 8px;
        margin-bottom: 14px;
    }

    .pilotia-danger {
        padding: 14px 16px;
        border-left: 4px solid #d62728;
        background: rgba(214,39,40,0.08);
        border-radius: 8px;
        margin-top: 8px;
        margin-bottom: 14px;
    }

    .small-note {
        font-size: 0.84rem;
        opacity: 0.72;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(100,100,100,0.10);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():
    return ensure_data()


@st.cache_data(show_spinner=False)
def cached_forecast(df, horizon):
    return fit_and_forecast(df, horizon=horizon)


@st.cache_data(show_spinner=False)
def cached_anomalies(df):
    return detect_anomalies(df)


try:
    df = load_data()
except Exception as exc:
    st.error("Impossible de charger les données.")
    st.exception(exc)
    st.stop()


# ============================================================
# 5. EN-TÊTE
# ============================================================

st.title("PILOT'IA")

st.markdown(
    """
    <div class="pilotia-subtitle">
        Plateforme intelligente de pilotage opérationnel et d'aide à la décision
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pilotia-info">
        <b>Objectif :</b> suivre la performance opérationnelle, anticiper la charge,
        identifier les points de tension et soutenir l'allocation des ressources
        à partir de données 100 % synthétiques.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. BARRE LATÉRALE
# ============================================================

st.sidebar.title("PILOT'IA")
st.sidebar.caption("Pilotage intelligent de l'activité")

st.sidebar.divider()

st.sidebar.markdown("### Filtres")

min_date = df["created_at"].min().date()
max_date = df["created_at"].max().date()

date_range = st.sidebar.date_input(
    "Période d'analyse",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    d1 = pd.Timestamp(date_range[0])
    d2 = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
else:
    d1 = pd.Timestamp(min_date)
    d2 = pd.Timestamp(max_date) + pd.Timedelta(days=1)

caisse_options = sorted(df["caisse"].dropna().unique())
service_options = sorted(df["service"].dropna().unique())

selected_caisses = st.sidebar.multiselect(
    "Caisses",
    caisse_options,
    default=caisse_options,
)

selected_services = st.sidebar.multiselect(
    "Services",
    service_options,
    default=service_options,
)

filtered = df[
    (df["created_at"] >= d1)
    & (df["created_at"] < d2)
    & (df["caisse"].isin(selected_caisses))
    & (df["service"].isin(selected_services))
].copy()

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Vue exécutive",
        "⏱ Délais & qualité de service",
        "👥 Charge & ressources",
        "📈 Prévision",
        "🚨 Anomalies",
        "🧪 Qualité des données",
        "🔐 Gouvernance",
    ],
)

st.sidebar.divider()

st.sidebar.caption(
    "Démonstrateur professionnel utilisant uniquement des données synthétiques."
)


# ============================================================
# 7. CONTRÔLE DES FILTRES
# ============================================================

if filtered.empty:
    st.warning(
        "Aucune donnée ne correspond aux filtres sélectionnés. "
        "Modifiez la période, les caisses ou les services."
    )
    st.stop()


# ============================================================
# 8. FONCTIONS D'AIDE
# ============================================================

def format_int(value):
    return f"{int(value):,}".replace(",", " ")


def severity_text(value):
    if value < 0.80:
        return "Maîtrisé"
    elif value < 1.00:
        return "Sous surveillance"
    elif value < 1.15:
        return "Tension"
    return "Critique"


def period_delta(current_df):
    """
    Compare la période sélectionnée avec une période précédente
    de même durée.
    """

    days = max(1, (d2 - d1).days)

    previous_end = d1
    previous_start = d1 - pd.Timedelta(days=days)

    previous = df[
        (df["created_at"] >= previous_start)
        & (df["created_at"] < previous_end)
        & (df["caisse"].isin(selected_caisses))
        & (df["service"].isin(selected_services))
    ].copy()

    if previous.empty:
        return None

    return compute_kpis(previous)


# ============================================================
# 9. VUE EXÉCUTIVE
# ============================================================

if page == "🏠 Vue exécutive":

    st.header("Vue exécutive")

    st.caption(
        "Lecture synthétique de l'activité, de la qualité de service "
        "et des principaux signaux opérationnels."
    )

    current_kpi = compute_kpis(filtered)
    previous_kpi = period_delta(filtered)

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    if previous_kpi:
        demandes_delta = (
            current_kpi["demandes"] / max(previous_kpi["demandes"], 1) - 1
        )
        sla_delta = (
            current_kpi["taux_sla"] - previous_kpi["taux_sla"]
        )
        delay_delta = (
            current_kpi["delai_median"] - previous_kpi["delai_median"]
        )
    else:
        demandes_delta = None
        sla_delta = None
        delay_delta = None

    col1.metric(
        "Demandes reçues",
        format_int(current_kpi["demandes"]),
        None if demandes_delta is None else f"{demandes_delta:+.1%}",
    )

    col2.metric(
        "Demandes traitées",
        format_int(current_kpi["traitees"]),
    )

    col3.metric(
        "Backlog",
        format_int(current_kpi["backlog"]),
    )

    col4.metric(
        "Taux de réalisation",
        f"{current_kpi['taux_realisation']:.1%}",
    )

    col5.metric(
        "Respect du SLA",
        f"{current_kpi['taux_sla']:.1%}",
        None if sla_delta is None else f"{sla_delta:+.1%}",
    )

    col6.metric(
        "Délai médian",
        f"{current_kpi['delai_median']:.1f} j",
        None if delay_delta is None else f"{delay_delta:+.1f} j",
        delta_color="inverse",
    )

    st.divider()

    # --------------------------------------------------------
    # Synthèse automatique
    # --------------------------------------------------------

    st.subheader("Synthèse opérationnelle")

    messages = []

    if current_kpi["taux_sla"] >= 0.90:
        messages.append(
            "✅ Le niveau global de respect des délais est satisfaisant."
        )
    elif current_kpi["taux_sla"] >= 0.80:
        messages.append(
            "⚠️ Le respect des délais doit être surveillé."
        )
    else:
        messages.append(
            "🔴 Le niveau de respect des délais nécessite une attention prioritaire."
        )

    if current_kpi["backlog"] > current_kpi["demandes"] * 0.10:
        messages.append(
            "⚠️ Le volume de dossiers non clôturés représente une part notable de l'activité."
        )
    else:
        messages.append(
            "✅ Le backlog reste contenu par rapport au volume global."
        )

    if current_kpi["taux_succes"] >= 0.90:
        messages.append(
            "✅ Le taux de succès des dossiers traités est élevé."
        )

    for msg in messages:
        st.write(msg)

    st.divider()

    daily = daily_activity(filtered)

    col_left, col_right = st.columns([2, 1])

    # --------------------------------------------------------
    # Flux quotidien
    # --------------------------------------------------------

    with col_left:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["received"],
                mode="lines",
                name="Demandes reçues",
                line=dict(width=2.2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["completed"],
                mode="lines",
                name="Demandes traitées",
                line=dict(width=2.2),
            )
        )

        fig.update_layout(
            title="Évolution quotidienne de l'activité",
            height=430,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=55, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # Volume par service
    # --------------------------------------------------------

    with col_right:

        by_service = (
            filtered.groupby("service")
            .agg(
                demandes=("request_id", "count"),
            )
            .reset_index()
            .sort_values("demandes", ascending=True)
        )

        fig = px.bar(
            by_service,
            x="demandes",
            y="service",
            orientation="h",
            title="Répartition des demandes",
        )

        fig.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=55, b=20),
            yaxis_title=None,
            xaxis_title="Nombre de demandes",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --------------------------------------------------------
    # Performance par caisse
    # --------------------------------------------------------

    st.subheader("Performance par caisse")

    caisse_perf = (
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

    caisse_perf["taux_realisation"] = (
        caisse_perf["traitees"] / caisse_perf["demandes"]
    )

    caisse_perf = caisse_perf.sort_values(
        "taux_sla",
        ascending=False,
    )

    display_perf = caisse_perf.copy()

    display_perf["taux_realisation"] = (
        display_perf["taux_realisation"] * 100
    ).round(1)

    display_perf["taux_sla"] = (
        display_perf["taux_sla"] * 100
    ).round(1)

    display_perf["taux_succes"] = (
        display_perf["taux_succes"] * 100
    ).round(1)

    display_perf["delai_median"] = (
        display_perf["delai_median"].round(1)
    )

    st.dataframe(
        display_perf.rename(
            columns={
                "caisse": "Caisse",
                "demandes": "Demandes",
                "traitees": "Traitées",
                "delai_median": "Délai médian (j)",
                "taux_realisation": "Réalisation (%)",
                "taux_sla": "SLA (%)",
                "taux_succes": "Succès (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = display_perf.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Télécharger la synthèse",
        data=csv_data,
        file_name="pilotia_synthese_caisses.csv",
        mime="text/csv",
    )


# ============================================================
# 10. DÉLAIS & QUALITÉ DE SERVICE
# ============================================================

elif page == "⏱ Délais & qualité de service":

    st.header("Délais & qualité de service")

    st.caption(
        "Analyse du respect des délais, de la dispersion des temps de traitement "
        "et des différences entre services."
    )

    completed = filtered[
        filtered["is_completed"]
    ].copy()

    kpi = compute_kpis(filtered)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Délai médian",
        f"{kpi['delai_median']:.1f} jours",
    )

    c2.metric(
        "Respect du SLA",
        f"{kpi['taux_sla']:.1%}",
    )

    c3.metric(
        "Taux de succès",
        f"{kpi['taux_succes']:.1%}",
    )

    c4.metric(
        "Taux d'échec",
        f"{kpi['taux_echec']:.1%}",
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            completed,
            x="service",
            y="processing_days",
            points=False,
            title="Distribution des délais par service",
        )

        fig.update_xaxes(tickangle=-20)

        fig.update_layout(
            height=450,
            xaxis_title=None,
            yaxis_title="Délai de traitement (jours)",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:

        weekly = (
            completed.assign(
                week=completed["created_at"]
                .dt.to_period("W")
                .dt.start_time
            )
            .groupby("week")
            .agg(
                taux_sla=("within_sla", "mean"),
                delai_median=("processing_days", "median"),
            )
            .reset_index()
        )

        fig = px.line(
            weekly,
            x="week",
            y="taux_sla",
            markers=True,
            title="Évolution du respect du SLA",
        )

        fig.update_yaxes(
            tickformat=".0%",
            range=[0, 1],
        )

        fig.update_layout(
            height=450,
            xaxis_title=None,
            yaxis_title="Taux de respect",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader("Analyse par service")

    service_perf = (
        completed.groupby(["caisse", "service"])
        .agg(
            dossiers=("request_id", "count"),
            delai_median=("processing_days", "median"),
            taux_sla=("within_sla", "mean"),
            taux_succes=("is_success", "mean"),
        )
        .reset_index()
        .sort_values("taux_sla")
    )

    st.dataframe(
        service_perf,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 11. CHARGE & RESSOURCES
# ============================================================

elif page == "👥 Charge & ressources":

    st.header("Pilotage charge ↔ ressources")

    st.caption(
        "Simuler un scénario opérationnel et mesurer le risque de saturation."
    )

    st.markdown(
        """
        <div class="pilotia-info">
            Ce module permet d'évaluer si les ressources disponibles
            sont cohérentes avec la charge attendue et d'estimer
            un besoin éventuel en ETP supplémentaires.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    agents = col1.number_input(
        "ETP planifiés",
        min_value=1.0,
        max_value=500.0,
        value=35.0,
        step=1.0,
    )

    absence = col2.slider(
        "Taux d'absence",
        min_value=0.0,
        max_value=0.40,
        value=0.08,
        step=0.01,
    )

    productivity = col3.number_input(
        "Productivité / ETP / jour",
        min_value=5.0,
        max_value=100.0,
        value=35.0,
        step=1.0,
    )

    workdays = col4.number_input(
        "Nombre de jours ouvrés",
        min_value=1,
        max_value=31,
        value=5,
    )

    daily = daily_activity(filtered)

    recent_daily_volume = (
        daily["received"]
        .tail(28)
        .mean()
    )

    baseline_volume = (
        recent_daily_volume * workdays
    )

    expected_growth = st.slider(
        "Évolution anticipée de la charge",
        min_value=-0.30,
        max_value=0.80,
        value=0.10,
        step=0.05,
        format="%0.0f%%",
    )

    expected_volume = (
        baseline_volume
        * (1 + expected_growth)
    )

    result = simulate_capacity(
        agents_fte=agents,
        absence_rate=absence,
        productivity_per_fte_day=productivity,
        workdays=workdays,
        expected_volume=expected_volume,
    )

    st.divider()

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric(
        "Charge estimée",
        format_int(expected_volume),
    )

    m2.metric(
        "Capacité théorique",
        format_int(result["capacity"]),
    )

    m3.metric(
        "ETP disponibles",
        f"{result['available_fte']:.1f}",
    )

    m4.metric(
        "Utilisation capacité",
        f"{result['utilization']:.1%}",
    )

    m5.metric(
        "Risque saturation",
        f"{result['risk']:.1%}",
    )

    st.divider()

    col_left, col_right = st.columns([1.15, 1])

    with col_left:

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=result["utilization"] * 100,
                delta={"reference": 100},
                title={
                    "text": "Utilisation de la capacité (%)"
                },
                gauge={
                    "axis": {
                        "range": [0, 140]
                    },
                    "steps": [
                        {
                            "range": [0, 80],
                            "color": "rgba(44,160,44,0.15)",
                        },
                        {
                            "range": [80, 100],
                            "color": "rgba(255,193,7,0.18)",
                        },
                        {
                            "range": [100, 115],
                            "color": "rgba(255,152,0,0.20)",
                        },
                        {
                            "range": [115, 140],
                            "color": "rgba(214,39,40,0.20)",
                        },
                    ],
                    "threshold": {
                        "line": {
                            "color": "red",
                            "width": 3,
                        },
                        "thickness": 0.75,
                        "value": 100,
                    },
                },
            )
        )

        gauge.update_layout(
            height=390,
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )

    with col_right:

        st.subheader("Diagnostic")

        level = severity_text(
            result["utilization"]
        )

        st.write(
            f"**Niveau de tension : {level}**"
        )

        st.write(
            f"Capacité estimée : **{format_int(result['capacity'])} dossiers**"
        )

        st.write(
            f"Charge attendue : **{format_int(expected_volume)} dossiers**"
        )

        if result["gap"] >= 0:

            st.success(
                "Les ressources disponibles couvrent la charge estimée."
            )

        else:

            st.error(
                f"Déficit de capacité estimé : "
                f"{format_int(abs(result['gap']))} dossiers."
            )

            st.warning(
                result["recommendation"]
            )

            st.metric(
                "Besoin supplémentaire estimé",
                f"{result['additional_fte']:.1f} ETP",
            )

    st.divider()

    st.markdown(
        """
        **Lecture métier**

        Ce simulateur permet de tester rapidement différentes hypothèses :

        - hausse ou baisse des volumes ;
        - évolution de l'absentéisme ;
        - modification de la productivité ;
        - changement des ressources disponibles.

        L'objectif est d'anticiper les tensions avant qu'elles ne se traduisent
        par une dégradation du service rendu.
        """
    )


# ============================================================
# 12. PRÉVISION
# ============================================================

elif page == "📈 Prévision":

    st.header("Prévision de l'activité")

    st.caption(
        "Anticiper les volumes futurs afin de mieux préparer les capacités de traitement."
    )

    horizon = st.segmented_control(
        "Horizon de prévision",
        options=[7, 30],
        default=30,
        format_func=lambda x: f"{x} jours",
    )

    if horizon is None:
        horizon = 30

    try:

        forecast, model_metrics = cached_forecast(
            filtered,
            horizon,
        )

    except Exception as exc:

        st.warning(
            "L'historique disponible n'est pas suffisant pour calculer une prévision fiable."
        )

        st.stop()

    daily = (
        daily_activity(filtered)[
            ["date", "received"]
        ]
        .tail(120)
    )

    fig = go.Figure()

    # Historique
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["received"],
            name="Historique",
            mode="lines",
            line=dict(
                width=2.2,
            ),
        )
    )

    # Intervalle AVANT la prévision pour qu'il reste en arrière-plan
    fig.add_trace(
        go.Scatter(
            x=list(forecast["date"])
            + list(forecast["date"][::-1]),
            y=list(forecast["upper"])
            + list(forecast["lower"][::-1]),
            fill="toself",
            fillcolor="rgba(76,120,168,0.12)",
            line=dict(
                color="rgba(255,255,255,0)"
            ),
            hoverinfo="skip",
            name="Intervalle indicatif",
        )
    )

    # Prévision
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["forecast"],
            name="Prévision",
            mode="lines+markers",
            line=dict(
                width=3,
            ),
            marker=dict(
                size=5,
            ),
        )
    )

    fig.update_layout(
        height=510,
        hovermode="x unified",
        title="Historique et activité prévisionnelle",
        xaxis_title=None,
        yaxis_title="Nombre de demandes",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "MAE validation",
        f"{model_metrics['mae']:.1f}",
    )

    m2.metric(
        "Moyenne historique",
        f"{model_metrics['historical_daily_mean']:.1f}/jour",
    )

    m3.metric(
        "Moyenne prévisionnelle",
        f"{model_metrics['forecast_daily_mean']:.1f}/jour",
    )

    delta = (
        model_metrics["forecast_daily_mean"]
        / max(
            model_metrics["historical_daily_mean"],
            1e-9,
        )
        - 1
    )

    m4.metric(
        "Évolution attendue",
        f"{delta:+.1%}",
    )

    st.divider()

    st.subheader("Interprétation")

    if delta > 0.15:

        st.error(
            f"Hausse importante attendue : {delta:.1%}. "
            "Une revue des capacités disponibles est recommandée."
        )

    elif delta > 0.05:

        st.warning(
            f"Hausse modérée de l'activité attendue : {delta:.1%}."
        )

    elif delta < -0.10:

        st.info(
            f"Baisse de l'activité attendue : {delta:.1%}."
        )

    else:

        st.success(
            "Le niveau d'activité prévisionnel reste globalement stable."
        )

    st.caption(
        "Les prévisions sont une aide au pilotage et doivent être interprétées "
        "avec les informations métier, réglementaires et organisationnelles."
    )


# ============================================================
# 13. ANOMALIES
# ============================================================

elif page == "🚨 Anomalies":

    st.header("Détection des anomalies")

    st.caption(
        "Identifier automatiquement des comportements atypiques dans les volumes, "
        "délais ou niveaux de service."
    )

    anomalies = cached_anomalies(
        filtered
    )

    flagged = anomalies[
        anomalies["is_anomaly"]
    ].copy()

    anomaly_rate = (
        len(flagged)
        / max(1, len(anomalies))
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Signaux détectés",
        format_int(len(flagged)),
    )

    c2.metric(
        "Observations analysées",
        format_int(len(anomalies)),
    )

    c3.metric(
        "Taux de signaux",
        f"{anomaly_rate:.2%}",
    )

    st.divider()

    fig = px.scatter(
        anomalies,
        x="volume",
        y="median_delay",
        color="is_anomaly",
        size="anomaly_score",
        hover_data=[
            "date",
            "caisse",
            "service",
            "sla_rate",
            "success_rate",
        ],
        title="Détection des situations atypiques",
        labels={
            "volume": "Volume journalier",
            "median_delay": "Délai médian",
            "is_anomaly": "Anomalie",
        },
    )

    fig.update_layout(
        height=500,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="pilotia-warning">
        <b>Important :</b> une anomalie est un signal statistique à investiguer.
        Elle ne constitue pas une conclusion métier ni une preuve d'anomalie de gestion.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader(
        "Signaux prioritaires"
    )

    if flagged.empty:

        st.success(
            "Aucune anomalie significative n'a été détectée sur la période."
        )

    else:

        top_flagged = (
            flagged[
                [
                    "date",
                    "caisse",
                    "service",
                    "volume",
                    "median_delay",
                    "sla_rate",
                    "success_rate",
                    "anomaly_score",
                ]
            ]
            .sort_values(
                "anomaly_score",
                ascending=False,
            )
            .head(50)
        )

        st.dataframe(
            top_flagged,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 14. QUALITÉ DES DONNÉES
# ============================================================

elif page == "🧪 Qualité des données":

    st.header("Qualité des données")

    st.caption(
        "Contrôler la fiabilité des données avant toute analyse ou prise de décision."
    )

    report = quality_report(
        filtered
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Score qualité",
        f"{report['score']:.1%}",
    )

    c2.metric(
        "Doublons",
        format_int(report["duplicates"]),
    )

    c3.metric(
        "Valeurs manquantes",
        format_int(report["missing_values"]),
    )

    c4.metric(
        "Délais négatifs",
        format_int(report["negative_delays"]),
    )

    c5.metric(
        "Dates incohérentes",
        format_int(
            report["invalid_completion_dates"]
        ),
    )

    st.divider()

    if report["score"] >= 0.99:

        st.success(
            "La qualité globale des données est élevée."
        )

    elif report["score"] >= 0.95:

        st.warning(
            "La qualité est globalement satisfaisante, mais certaines anomalies doivent être corrigées."
        )

    else:

        st.error(
            "Le niveau de qualité nécessite une investigation avant utilisation analytique."
        )

    missing = top_missing_columns(
        filtered
    )

    fig = px.bar(
        missing,
        x="column",
        y="missing",
        title="Valeurs manquantes par variable",
        labels={
            "column": "Variable",
            "missing": "Nombre de valeurs manquantes",
        },
    )

    fig.update_layout(
        height=420,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader(
        "Règles de qualité contrôlées"
    )

    st.markdown(
        """
        - présence des variables obligatoires ;
        - unicité de l'identifiant de demande ;
        - détection des doublons ;
        - contrôle des valeurs manquantes ;
        - détection des délais négatifs ;
        - cohérence entre date de création et date de clôture ;
        - contrôle de la complétude nécessaire au calcul des KPI.
        """
    )

    if report["missing_columns"]:

        st.error(
            "Variables obligatoires absentes : "
            + ", ".join(
                report["missing_columns"]
            )
        )

    else:

        st.success(
            "Toutes les variables obligatoires sont présentes."
        )


# ============================================================
# 15. GOUVERNANCE
# ============================================================

elif page == "🔐 Gouvernance":

    st.header("Gouvernance & sécurité")

    st.markdown(
        """
        <div class="pilotia-success">
        <b>Principe fondamental :</b> cette démonstration fonctionne exclusivement
        avec des données artificielles générées pour le projet.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Protection des données"
        )

        st.markdown(
            """
            - aucun nom ou prénom ;
            - aucun numéro de sécurité sociale ;
            - aucun identifiant réel ;
            - aucune donnée médicale individuelle ;
            - identifiants techniques fictifs ;
            - minimisation des variables ;
            - agrégation des résultats ;
            - absence de données personnelles réelles.
            """
        )

    with col2:

        st.subheader(
            "Utilisation responsable"
        )

        st.markdown(
            """
            - les anomalies sont des signaux à investiguer ;
            - aucune décision individuelle automatisée ;
            - les prévisions sont des aides au pilotage ;
            - les résultats doivent être interprétés avec l'expertise métier ;
            - les règles organisationnelles et réglementaires restent prioritaires.
            """
        )

    st.divider()

    st.subheader(
        "Architecture logique"
    )

    st.code(
        """
Données synthétiques
        │
        ▼
Contrôles qualité
        │
        ▼
Préparation / SQL
        │
        ├───────────────┐
        ▼               ▼
Calcul des KPI      Data Science
        │          ├─ Prévision
        │          └─ Anomalies
        │               │
        └───────┬───────┘
                ▼
         Application Streamlit
                │
                ▼
      Pilotage / Aide à la décision
        """,
        language="text",
    )

    st.subheader(
        "Positionnement du prototype"
    )

    st.write(
        """
        PILOT'IA n'a pas vocation à reproduire l'architecture ou les systèmes
        d'information d'un organisme existant.

        Le projet démontre une démarche Data complète :

        **comprendre le besoin métier → contrôler les données → construire les indicateurs →
        analyser → anticiper → restituer → aider à la décision.**
        """
    )


# ============================================================
# 16. PIED DE PAGE
# ============================================================

st.divider()

st.markdown(
    """
    <div class="small-note">
        <b>PILOT'IA</b> — Démonstrateur Data Analytics & Decision Support<br>
        Données 100 % synthétiques · Python · SQL · Machine Learning · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
