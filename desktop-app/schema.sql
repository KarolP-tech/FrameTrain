-- FrameTrain Desktop App Database Schema
-- SQLite Database für lokale Modell- und Trainingsverwaltung

-- Models Table
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    base_model TEXT,
    model_path TEXT,
    status TEXT NOT NULL DEFAULT 'created',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name)
);

-- Model Versions Table
CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    version_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
    UNIQUE(model_id, version)
);

-- Training Configurations Table
CREATE TABLE IF NOT EXISTS training_configs (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    epochs INTEGER NOT NULL DEFAULT 10,
    batch_size INTEGER NOT NULL DEFAULT 32,
    learning_rate REAL NOT NULL DEFAULT 0.001,
    optimizer TEXT NOT NULL DEFAULT 'adam',
    loss_function TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (version_id) REFERENCES model_versions(id) ON DELETE CASCADE
);

-- Training Metrics Table
CREATE TABLE IF NOT EXISTS training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    loss REAL NOT NULL,
    accuracy REAL,
    val_loss REAL,
    val_accuracy REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (version_id) REFERENCES model_versions(id) ON DELETE CASCADE
);

-- Training Sessions Table
CREATE TABLE IF NOT EXISTS training_sessions (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at DATETIME,
    completed_at DATETIME,
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (version_id) REFERENCES model_versions(id) ON DELETE CASCADE
);

-- Datasets Table
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    size_bytes INTEGER,
    rows_count INTEGER,
    columns_count INTEGER,
    validated BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name)
);

-- App Configuration Table
CREATE TABLE IF NOT EXISTS app_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
CREATE INDEX IF NOT EXISTS idx_model_versions_model_id ON model_versions(model_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_status ON model_versions(status);
CREATE INDEX IF NOT EXISTS idx_training_metrics_version_id ON training_metrics(version_id);
CREATE INDEX IF NOT EXISTS idx_training_metrics_epoch ON training_metrics(epoch);
CREATE INDEX IF NOT EXISTS idx_training_sessions_version_id ON training_sessions(version_id);
CREATE INDEX IF NOT EXISTS idx_training_sessions_status ON training_sessions(status);
CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name);

-- Trigger: Update updated_at on models
CREATE TRIGGER IF NOT EXISTS update_models_timestamp 
AFTER UPDATE ON models
BEGIN
    UPDATE models SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Trigger: Update updated_at on app_config
CREATE TRIGGER IF NOT EXISTS update_config_timestamp 
AFTER UPDATE ON app_config
BEGIN
    UPDATE app_config SET updated_at = CURRENT_TIMESTAMP WHERE key = NEW.key;
END;

-- Insert default config values
INSERT OR IGNORE INTO app_config (key, value) VALUES 
    ('api_key', ''),
    ('api_url', 'https://frametrain.ai/api'),
    ('auto_update', 'true'),
    ('theme', 'dark');
