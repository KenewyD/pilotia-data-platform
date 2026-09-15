CREATE TABLE IF NOT EXISTS fact_activity (
    request_id VARCHAR(32) PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP NULL,
    caisse VARCHAR(100) NOT NULL,
    service VARCHAR(150) NOT NULL,
    request_type VARCHAR(150) NULL,
    sla_days INTEGER NOT NULL,
    processing_days NUMERIC(10,2) NOT NULL,
    is_completed BOOLEAN NOT NULL,
    is_success BOOLEAN NOT NULL,
    team_size_fte INTEGER NOT NULL,
    productivity_target NUMERIC(10,2) NOT NULL,
    within_sla BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_activity_created_at ON fact_activity(created_at);
CREATE INDEX IF NOT EXISTS idx_activity_caisse ON fact_activity(caisse);
CREATE INDEX IF NOT EXISTS idx_activity_service ON fact_activity(service);

CREATE OR REPLACE VIEW vw_service_kpi AS
SELECT
    caisse,
    service,
    COUNT(*) AS demandes,
    SUM(CASE WHEN is_completed THEN 1 ELSE 0 END) AS traitees,
    SUM(CASE WHEN NOT is_completed THEN 1 ELSE 0 END) AS backlog,
    AVG(CASE WHEN is_completed THEN processing_days END) AS delai_moyen,
    AVG(CASE WHEN within_sla THEN 1.0 ELSE 0.0 END) AS taux_sla,
    AVG(CASE WHEN is_success THEN 1.0 ELSE 0.0 END) AS taux_succes
FROM fact_activity
GROUP BY caisse, service;
