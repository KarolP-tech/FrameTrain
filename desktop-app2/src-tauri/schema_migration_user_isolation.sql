-- Migration: Add user_id to all tables for multi-user isolation
-- This ensures that each user only sees their own data

-- Step 1: Add user_id column to models table
ALTER TABLE models ADD COLUMN user_id TEXT;

-- Step 2: Add user_id column to model_versions table  
ALTER TABLE model_versions ADD COLUMN user_id TEXT;

-- Step 3: Add user_id column to training_configs table
ALTER TABLE training_configs ADD COLUMN user_id TEXT;

-- Step 4: Add user_id column to datasets table
ALTER TABLE datasets ADD COLUMN user_id TEXT;

-- Step 5: Add user_id column to test_results table
ALTER TABLE test_results ADD COLUMN user_id TEXT;

-- Step 6: Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_models_user_id ON models(user_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_user_id ON model_versions(user_id);
CREATE INDEX IF NOT EXISTS idx_training_configs_user_id ON training_configs(user_id);
CREATE INDEX IF NOT EXISTS idx_datasets_user_id ON datasets(user_id);
CREATE INDEX IF NOT EXISTS idx_test_results_user_id ON test_results(user_id);

-- Note: Existing data without user_id will have NULL values
-- These records will not be accessible until assigned to a user
-- This is intentional for security - no orphaned data access
