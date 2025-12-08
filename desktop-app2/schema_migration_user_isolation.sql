-- FrameTrain User Data Isolation Migration
-- Adds user_id to all tables for multi-user support

-- Add user_id to models table
ALTER TABLE models ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
CREATE INDEX IF NOT EXISTS idx_models_user_id ON models(user_id);

-- Add user_id to model_versions table  
ALTER TABLE model_versions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
CREATE INDEX IF NOT EXISTS idx_model_versions_user_id ON model_versions(user_id);

-- Add user_id to training_configs table
ALTER TABLE training_configs ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
CREATE INDEX IF NOT EXISTS idx_training_configs_user_id ON training_configs(user_id);

-- Add user_id to training_metrics table
ALTER TABLE training_metrics ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
CREATE INDEX IF NOT EXISTS idx_training_metrics_user_id ON training_metrics(user_id);

-- Add user_id to training_sessions table
ALTER TABLE training_sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
CREATE INDEX IF NOT EXISTS idx_training_sessions_user_id ON training_sessions(user_id);

-- Add user_id to datasets table
ALTER TABLE datasets ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON datasets(user_id);

-- Add user_id to model_versions_new table
ALTER TABLE model_versions_new ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
CREATE INDEX IF NOT EXISTS idx_model_versions_new_user_id ON model_versions_new(user_id);

-- Add user_id to training_metrics_new table
ALTER TABLE training_metrics_new ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
CREATE INDEX IF NOT EXISTS idx_training_metrics_new_user_id ON training_metrics_new(user_id);

-- Add user_id to test_results table  
ALTER TABLE test_results ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous';
CREATE INDEX IF NOT EXISTS idx_test_results_user_id ON test_results(user_id);

-- Create user_sessions table for auth state persistence
CREATE TABLE IF NOT EXISTS user_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL,
    api_key_hash TEXT NOT NULL,
    last_login TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions(expires_at);
