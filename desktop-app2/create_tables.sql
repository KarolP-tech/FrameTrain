-- Erstelle Version-Tabellen für desktop-app2

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
    parent_version_id TEXT,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS training_metrics_new (
    id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL UNIQUE,
    final_train_loss REAL NOT NULL,
    final_val_loss REAL,
    total_epochs INTEGER NOT NULL,
    total_steps INTEGER NOT NULL,
    best_epoch INTEGER,
    training_duration_seconds INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (version_id) REFERENCES model_versions_new(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_versions_model ON model_versions_new(model_id);
CREATE INDEX IF NOT EXISTS idx_metrics_version ON training_metrics_new(version_id);
