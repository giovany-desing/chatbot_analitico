CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_query TEXT NOT NULL,
    sql_generated TEXT,
    chart_type VARCHAR(50),
    chart_config JSONB,
    user_rating INTEGER CHECK (user_rating BETWEEN 1 AND 5),
    feedback_text TEXT,
    error_occurred BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Índices para búsquedas rápidas
    CONSTRAINT valid_rating CHECK (user_rating IS NULL OR user_rating BETWEEN 1 AND 5)
);

CREATE INDEX idx_feedback_rating ON user_feedback(user_rating);
CREATE INDEX idx_feedback_session ON user_feedback(session_id);
CREATE INDEX idx_feedback_created ON user_feedback(created_at DESC);
CREATE INDEX idx_feedback_errors ON user_feedback(error_occurred) WHERE error_occurred = TRUE;

-- Tabla para métricas agregadas (cache de analytics)
CREATE TABLE IF NOT EXISTS analytics_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    metadata JSONB,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_name ON analytics_metrics(metric_name);
CREATE INDEX idx_metrics_date ON analytics_metrics(calculated_at DESC);