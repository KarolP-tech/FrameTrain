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

-- ============================================
-- NEW VERSION SYSTEM TABLES
-- ============================================

-- Model Versions New (Enhanced Version System)
-- Foreign Keys removed for better compatibility and initialization order
CREATE TABLE IF NOT EXISTS model_versions_new (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    version_name TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    file_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    is_root INTEGER NOT NULL DEFAULT 0,
    parent_version_id TEXT
);

-- Training Metrics New (Enhanced Metrics System)
-- Foreign Keys removed for better compatibility and initialization order
CREATE TABLE IF NOT EXISTS training_metrics_new (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL UNIQUE,
    final_train_loss REAL NOT NULL,
    final_val_loss REAL,
    total_epochs INTEGER NOT NULL,
    total_steps INTEGER NOT NULL,
    best_epoch INTEGER,
    training_duration_seconds INTEGER,
    created_at TEXT NOT NULL
);

-- Indices for new version system
CREATE INDEX IF NOT EXISTS idx_versions_model ON model_versions_new(model_id);
CREATE INDEX IF NOT EXISTS idx_versions_number ON model_versions_new(version_number);
CREATE INDEX IF NOT EXISTS idx_versions_root ON model_versions_new(is_root);
CREATE INDEX IF NOT EXISTS idx_metrics_version ON training_metrics_new(version_id);

-- ============================================
-- TEST RESULTS TABLES
-- ============================================

-- Test Results (for model evaluation)
CREATE TABLE IF NOT EXISTS test_results (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    total_samples INTEGER NOT NULL,
    correct_predictions INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    average_loss REAL NOT NULL,
    average_inference_time REAL NOT NULL,
    results_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Indices for test results
CREATE INDEX IF NOT EXISTS idx_test_results_version ON test_results(version_id);
CREATE INDEX IF NOT EXISTS idx_test_results_created ON test_results(created_at);
