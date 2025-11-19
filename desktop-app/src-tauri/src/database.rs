// Database Module für FrameTrain Desktop App
// SQLite Integration für lokale Datenverwaltung

use rusqlite::{Connection, params, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
// use chrono::{DateTime, Utc};

#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub base_model: Option<String>,
    pub model_path: Option<String>,
    pub status: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelVersion {
    pub id: String,
    pub model_id: String,
    pub version: i32,
    pub version_path: Option<String>,
    pub status: String,
    pub created_at: String,
    pub completed_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub id: String,
    pub version_id: String,
    pub dataset_path: String,
    pub epochs: i32,
    pub batch_size: i32,
    pub learning_rate: f64,
    pub optimizer: String,
    pub loss_function: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub id: Option<i64>,
    pub version_id: String,
    pub epoch: i32,
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub val_loss: Option<f64>,
    pub val_accuracy: Option<f64>,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub file_path: String,
    pub file_type: String,
    pub size_bytes: Option<i64>,
    pub rows_count: Option<i32>,
    pub columns_count: Option<i32>,
    pub validated: bool,
    pub created_at: String,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    /// Erstellt oder öffnet die SQLite Datenbank
    pub fn new(db_path: PathBuf) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        
        // Initialisiere Schema
        Self::init_schema(&conn)?;
        
        Ok(Database { conn })
    }
    
    /// Initialisiert das Datenbank-Schema
    fn init_schema(conn: &Connection) -> Result<()> {
        let schema = include_str!("../../schema.sql");
        conn.execute_batch(schema)?;
        Ok(())
    }
    
    // ==================== MODELS ====================
    
    pub fn create_model(&self, model: &Model) -> Result<()> {
        self.conn.execute(
            "INSERT INTO models (id, name, description, base_model, model_path, status) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &model.id,
                &model.name,
                &model.description,
                &model.base_model,
                &model.model_path,
                &model.status
            ],
        )?;
        Ok(())
    }
    
    pub fn get_model(&self, id: &str) -> Result<Model> {
        self.conn.query_row(
            "SELECT id, name, description, base_model, model_path, status, created_at, updated_at 
             FROM models WHERE id = ?1",
            params![id],
            |row| {
                Ok(Model {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    description: row.get(2)?,
                    base_model: row.get(3)?,
                    model_path: row.get(4)?,
                    status: row.get(5)?,
                    created_at: row.get(6)?,
                    updated_at: row.get(7)?,
                })
            },
        )
    }
    
    pub fn list_models(&self) -> Result<Vec<Model>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, description, base_model, model_path, status, created_at, updated_at 
             FROM models ORDER BY created_at DESC"
        )?;
        
        let models = stmt.query_map([], |row| {
            Ok(Model {
                id: row.get(0)?,
                name: row.get(1)?,
                description: row.get(2)?,
                base_model: row.get(3)?,
                model_path: row.get(4)?,
                status: row.get(5)?,
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
            })
        })?;
        
        models.collect()
    }
    
    pub fn update_model_status(&self, id: &str, status: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE models SET status = ?1 WHERE id = ?2",
            params![status, id],
        )?;
        Ok(())
    }
    
    pub fn delete_model(&self, id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM models WHERE id = ?1", params![id])?;
        Ok(())
    }
    
    // ==================== MODEL VERSIONS ====================
    
    pub fn create_version(&self, version: &ModelVersion) -> Result<()> {
        self.conn.execute(
            "INSERT INTO model_versions (id, model_id, version, version_path, status) 
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                &version.id,
                &version.model_id,
                version.version,
                &version.version_path,
                &version.status
            ],
        )?;
        Ok(())
    }
    
    pub fn get_model_versions(&self, model_id: &str) -> Result<Vec<ModelVersion>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, model_id, version, version_path, status, created_at, completed_at 
             FROM model_versions WHERE model_id = ?1 ORDER BY version DESC"
        )?;
        
        let versions = stmt.query_map(params![model_id], |row| {
            Ok(ModelVersion {
                id: row.get(0)?,
                model_id: row.get(1)?,
                version: row.get(2)?,
                version_path: row.get(3)?,
                status: row.get(4)?,
                created_at: row.get(5)?,
                completed_at: row.get(6)?,
            })
        })?;
        
        versions.collect()
    }
    
    pub fn get_next_version_number(&self, model_id: &str) -> Result<i32> {
        let version: Option<i32> = self.conn.query_row(
            "SELECT MAX(version) FROM model_versions WHERE model_id = ?1",
            params![model_id],
            |row| row.get(0),
        )?;
        
        Ok(version.unwrap_or(0) + 1)
    }
    
    // ==================== TRAINING CONFIG ====================
    
    pub fn save_training_config(&self, config: &TrainingConfig) -> Result<()> {
        self.conn.execute(
            "INSERT INTO training_configs 
             (id, version_id, dataset_path, epochs, batch_size, learning_rate, optimizer, loss_function) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                &config.id,
                &config.version_id,
                &config.dataset_path,
                config.epochs,
                config.batch_size,
                config.learning_rate,
                &config.optimizer,
                &config.loss_function
            ],
        )?;
        Ok(())
    }
    
    pub fn get_training_config(&self, version_id: &str) -> Result<TrainingConfig> {
        self.conn.query_row(
            "SELECT id, version_id, dataset_path, epochs, batch_size, learning_rate, optimizer, loss_function 
             FROM training_configs WHERE version_id = ?1",
            params![version_id],
            |row| {
                Ok(TrainingConfig {
                    id: row.get(0)?,
                    version_id: row.get(1)?,
                    dataset_path: row.get(2)?,
                    epochs: row.get(3)?,
                    batch_size: row.get(4)?,
                    learning_rate: row.get(5)?,
                    optimizer: row.get(6)?,
                    loss_function: row.get(7)?,
                })
            },
        )
    }
    
    // ==================== TRAINING METRICS ====================
    
    pub fn save_training_metric(&self, metric: &TrainingMetrics) -> Result<()> {
        self.conn.execute(
            "INSERT INTO training_metrics 
             (version_id, epoch, loss, accuracy, val_loss, val_accuracy) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                &metric.version_id,
                metric.epoch,
                metric.loss,
                metric.accuracy,
                metric.val_loss,
                metric.val_accuracy
            ],
        )?;
        Ok(())
    }
    
    pub fn get_training_metrics(&self, version_id: &str) -> Result<Vec<TrainingMetrics>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, version_id, epoch, loss, accuracy, val_loss, val_accuracy, timestamp 
             FROM training_metrics WHERE version_id = ?1 ORDER BY epoch ASC"
        )?;
        
        let metrics = stmt.query_map(params![version_id], |row| {
            Ok(TrainingMetrics {
                id: Some(row.get(0)?),
                version_id: row.get(1)?,
                epoch: row.get(2)?,
                loss: row.get(3)?,
                accuracy: row.get(4)?,
                val_loss: row.get(5)?,
                val_accuracy: row.get(6)?,
                timestamp: row.get(7)?,
            })
        })?;
        
        metrics.collect()
    }
    
    // ==================== DATASETS ====================
    
    pub fn save_dataset(&self, dataset: &Dataset) -> Result<()> {
        self.conn.execute(
            "INSERT INTO datasets 
             (id, name, file_path, file_type, size_bytes, rows_count, columns_count, validated) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                &dataset.id,
                &dataset.name,
                &dataset.file_path,
                &dataset.file_type,
                dataset.size_bytes,
                dataset.rows_count,
                dataset.columns_count,
                dataset.validated
            ],
        )?;
        Ok(())
    }
    
    pub fn list_datasets(&self) -> Result<Vec<Dataset>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, file_path, file_type, size_bytes, rows_count, columns_count, validated, created_at 
             FROM datasets ORDER BY created_at DESC"
        )?;
        
        let datasets = stmt.query_map([], |row| {
            Ok(Dataset {
                id: row.get(0)?,
                name: row.get(1)?,
                file_path: row.get(2)?,
                file_type: row.get(3)?,
                size_bytes: row.get(4)?,
                rows_count: row.get(5)?,
                columns_count: row.get(6)?,
                validated: row.get(7)?,
                created_at: row.get(8)?,
            })
        })?;
        
        datasets.collect()
    }
    
    // ==================== APP CONFIG ====================
    
    pub fn set_config(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO app_config (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }
    
    pub fn get_config(&self, key: &str) -> Result<String> {
        self.conn.query_row(
            "SELECT value FROM app_config WHERE key = ?1",
            params![key],
            |row| row.get(0),
        )
    }
}

/// Gibt den Pfad zur Datenbank zurück
pub fn get_database_path() -> PathBuf {
    let data_dir = get_data_directory();
    data_dir.join("frametrain.db")
}

fn get_data_directory() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA").unwrap_or_else(|_| String::from("."));
        PathBuf::from(appdata).join("FrameTrain")
    }
    
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
        PathBuf::from(home).join("Library/Application Support/FrameTrain")
    }
    
    #[cfg(target_os = "linux")]
    {
        let home = std::env::var("HOME").unwrap_or_else(|_| String::from("."));
        PathBuf::from(home).join(".local/share/frametrain")
    }
}
