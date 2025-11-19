// Training Manager für FrameTrain
// Orchestriert ML-Training mit Checkpoint-Management und Live-Updates

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::fs;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tauri::{Manager, State};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub dataset_ids: Vec<String>,
    
    // Basic Training
    pub epochs: u32,
    pub batch_size: u32,
    pub eval_batch_size: u32,
    pub learning_rate: f64,
    pub optimizer: String,
    
    // Regularization
    pub weight_decay: f64,
    pub max_grad_norm: f64,
    pub dropout: f64,
    pub attention_dropout: f64,
    
    // Learning Rate Schedule
    pub warmup_ratio: f64,
    pub warmup_steps: u32,
    pub lr_scheduler_type: String,
    
    // Training Strategy
    pub gradient_accumulation_steps: u32,
    pub fp16: bool,
    pub bf16: bool,
    
    // Checkpointing
    pub save_strategy: String,
    pub save_total_limit: u32,
    pub checkpoint_interval: u32,
    
    // Evaluation
    pub eval_strategy: String,
    pub eval_interval: u32,
    pub metric_for_best_model: String,
    pub greater_is_better: bool,
    pub load_best_model_at_end: bool,
    
    // Early Stopping
    pub early_stopping_patience: Option<u32>,
    pub early_stopping_threshold: f64,
    
    // Logging
    pub logging_steps: u32,
    pub logging_strategy: String,
    
    // Generation (for Seq2Seq)
    pub predict_with_generate: bool,
    pub generation_max_length: Option<u32>,
    pub generation_num_beams: Option<u32>,
    
    // Advanced
    pub seed: u32,
    pub resume_from_checkpoint: Option<String>,
    pub dataloader_num_workers: u32,
    pub dataloader_pin_memory: bool,
    pub group_by_length: bool,
    pub length_column_name: Option<String>,
    pub label_smoothing_factor: f64,
    
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub training_id: String,
    pub status: TrainingStatus,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub current_step: u32,
    pub total_steps: u32,
    pub progress_percentage: f32,
    pub train_loss: f64,
    pub train_accuracy: Option<f64>,
    pub val_loss: Option<f64>,
    pub val_accuracy: Option<f64>,
    pub learning_rate: f64,
    pub elapsed_time_seconds: u64,
    pub estimated_time_remaining_seconds: Option<u64>,
    pub throughput_samples_per_second: Option<f64>,
    pub checkpoints: Vec<CheckpointInfo>,
    pub last_updated: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TrainingStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub id: String,
    pub epoch: u32,
    pub step: u32,
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub path: String,
    pub size_bytes: u64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSession {
    pub id: String,
    pub config: TrainingConfig,
    pub progress: TrainingProgress,
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigValidation {
    pub is_valid: bool,
    pub overall_score: f32,  // 0-100
    pub quality_level: String,  // "poor", "fair", "good", "excellent"
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
    pub estimated_training_time: Option<String>,
    pub estimated_memory_usage: Option<String>,
    pub issues: Vec<ConfigIssue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigIssue {
    pub severity: String,  // "error", "warning", "info"
    pub category: String,  // "batch_size", "learning_rate", etc.
    pub message: String,
    pub suggestion: String,
}

// Global Training State
pub struct TrainingManager {
    pub active_trainings: Arc<Mutex<std::collections::HashMap<String, TrainingSession>>>,
    pub process_handles: Arc<Mutex<std::collections::HashMap<String, Child>>>,
}

impl Default for TrainingManager {
    fn default() -> Self {
        Self {
            active_trainings: Arc::new(Mutex::new(std::collections::HashMap::new())),
            process_handles: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
}

/// Startet ein neues Training
#[tauri::command]
pub async fn start_training_session(
    app_handle: tauri::AppHandle,
    config: TrainingConfig,
) -> Result<String, String> {
    println!("🚀 Starting training session: {}", config.model_name);
    
    // Erstelle Training-ID
    let training_id = Uuid::new_v4().to_string();
    
    // Erstelle Training-Verzeichnis
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let training_dir = app_data_dir.join("trainings").join(&training_id);
    fs::create_dir_all(&training_dir)
        .map_err(|e| format!("Konnte Training-Verzeichnis nicht erstellen: {}", e))?;
    
    // Speichere Config
    save_training_config(&training_dir, &config)?;
    
    // Initialisiere Progress
    let progress = TrainingProgress {
        training_id: training_id.clone(),
        status: TrainingStatus::Pending,
        current_epoch: 0,
        total_epochs: config.epochs,
        current_step: 0,
        total_steps: 0,
        progress_percentage: 0.0,
        train_loss: 0.0,
        train_accuracy: None,
        val_loss: None,
        val_accuracy: None,
        learning_rate: config.learning_rate,
        elapsed_time_seconds: 0,
        estimated_time_remaining_seconds: None,
        throughput_samples_per_second: None,
        checkpoints: vec![],
        last_updated: Utc::now().to_rfc3339(),
    };
    
    save_training_progress(&training_dir, &progress)?;
    
    // Starte Python Training-Prozess
    let python_script = get_training_script_path(&app_handle)?;
    let config_path = training_dir.join("config.json");
    
    println!("   Training script: {:?}", python_script);
    println!("   Training dir: {:?}", training_dir);
    
    let mut child = Command::new("python3")
        .arg(&python_script)
        .arg("--config")
        .arg(&config_path)
        .arg("--output-dir")
        .arg(&training_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Konnte Training nicht starten: {}", e))?;
    
    // Speichere Session
    let session = TrainingSession {
        id: training_id.clone(),
        config,
        progress: progress.clone(),
        logs: vec![],
    };
    
    // Store in manager (würde normalerweise über State funktionieren)
    save_training_session(&training_dir, &session)?;
    
    println!("✅ Training started: {}", training_id);
    
    Ok(training_id)
}

/// Lädt Training-Progress
#[tauri::command]
pub async fn get_training_status(
    app_handle: tauri::AppHandle,
    training_id: String,
) -> Result<TrainingProgress, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let training_dir = app_data_dir.join("trainings").join(&training_id);
    let progress_file = training_dir.join("progress.json");
    
    if !progress_file.exists() {
        return Err("Training nicht gefunden".to_string());
    }
    
    let content = fs::read_to_string(&progress_file)
        .map_err(|e| format!("Konnte Progress nicht lesen: {}", e))?;
    
    let progress: TrainingProgress = serde_json::from_str(&content)
        .map_err(|e| format!("Konnte Progress nicht parsen: {}", e))?;
    
    Ok(progress)
}

/// Pausiert Training
#[tauri::command]
pub async fn pause_training(
    app_handle: tauri::AppHandle,
    training_id: String,
) -> Result<(), String> {
    println!("⏸️  Pausing training: {}", training_id);
    
    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let training_dir = app_data_dir.join("trainings").join(&training_id);
    let control_file = training_dir.join("control.json");
    
    // Schreibe Pause-Signal
    let control = serde_json::json!({
        "action": "pause",
        "timestamp": Utc::now().to_rfc3339()
    });
    
    fs::write(&control_file, control.to_string())
        .map_err(|e| format!("Konnte Control-File nicht schreiben: {}", e))?;
    
    Ok(())
}

/// Resümiert Training
#[tauri::command]
pub async fn resume_training(
    app_handle: tauri::AppHandle,
    training_id: String,
) -> Result<(), String> {
    println!("▶️  Resuming training: {}", training_id);
    
    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let training_dir = app_data_dir.join("trainings").join(&training_id);
    let control_file = training_dir.join("control.json");
    
    // Schreibe Resume-Signal
    let control = serde_json::json!({
        "action": "resume",
        "timestamp": Utc::now().to_rfc3339()
    });
    
    fs::write(&control_file, control.to_string())
        .map_err(|e| format!("Konnte Control-File nicht schreiben: {}", e))?;
    
    Ok(())
}

/// Stoppt Training
#[tauri::command]
pub async fn stop_training_session(
    app_handle: tauri::AppHandle,
    training_id: String,
) -> Result<(), String> {
    println!("🛑 Stopping training: {}", training_id);
    
    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let training_dir = app_data_dir.join("trainings").join(&training_id);
    let control_file = training_dir.join("control.json");
    
    // Schreibe Stop-Signal
    let control = serde_json::json!({
        "action": "stop",
        "timestamp": Utc::now().to_rfc3339()
    });
    
    fs::write(&control_file, control.to_string())
        .map_err(|e| format!("Konnte Control-File nicht schreiben: {}", e))?;
    
    Ok(())
}

/// Liste alle verfügbaren Checkpoints für ein Training
#[tauri::command]
pub async fn list_checkpoints(
    app_handle: tauri::AppHandle,
    training_id: String,
) -> Result<Vec<CheckpointInfo>, String> {
    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let checkpoints_dir = app_data_dir
        .join("trainings")
        .join(&training_id)
        .join("checkpoints");
    
    if !checkpoints_dir.exists() {
        return Ok(vec![]);
    }
    
    let mut checkpoints = Vec::new();
    
    for entry in fs::read_dir(&checkpoints_dir)
        .map_err(|e| format!("Konnte Checkpoints nicht lesen: {}", e))? 
    {
        let entry = entry.map_err(|e| format!("Fehler beim Lesen: {}", e))?;
        let path = entry.path();
        
        if path.is_dir() {
            let metadata_file = path.join("metadata.json");
            if metadata_file.exists() {
                if let Ok(content) = fs::read_to_string(&metadata_file) {
                    if let Ok(checkpoint) = serde_json::from_str::<CheckpointInfo>(&content) {
                        checkpoints.push(checkpoint);
                    }
                }
            }
        }
    }
    
    // Sortiere nach Epoch/Step
    checkpoints.sort_by(|a, b| {
        a.epoch.cmp(&b.epoch).then(a.step.cmp(&b.step))
    });
    
    Ok(checkpoints)
}

/// Liste alle Trainings für ein Modell
#[tauri::command]
pub async fn list_model_trainings(
    app_handle: tauri::AppHandle,
    model_id: String,
) -> Result<Vec<TrainingSession>, String> {
    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let trainings_dir = app_data_dir.join("trainings");
    
    if !trainings_dir.exists() {
        return Ok(vec![]);
    }
    
    let mut sessions = Vec::new();
    
    for entry in fs::read_dir(&trainings_dir)
        .map_err(|e| format!("Konnte Trainings nicht lesen: {}", e))? 
    {
        let entry = entry.map_err(|e| format!("Fehler beim Lesen: {}", e))?;
        let path = entry.path();
        
        if path.is_dir() {
            let session_file = path.join("session.json");
            if session_file.exists() {
                if let Ok(content) = fs::read_to_string(&session_file) {
                    if let Ok(session) = serde_json::from_str::<TrainingSession>(&content) {
                        if session.config.model_id == model_id {
                            sessions.push(session);
                        }
                    }
                }
            }
        }
    }
    
    // Sortiere nach Datum (neueste zuerst)
    sessions.sort_by(|a, b| b.config.created_at.cmp(&a.config.created_at));
    
    Ok(sessions)
}

/// Lädt Training-Logs
#[tauri::command]
pub async fn get_training_logs(
    app_handle: tauri::AppHandle,
    training_id: String,
    lines: Option<usize>,
) -> Result<Vec<String>, String> {
    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let log_file = app_data_dir
        .join("trainings")
        .join(&training_id)
        .join("train.log");
    
    if !log_file.exists() {
        return Ok(vec![]);
    }
    
    let content = fs::read_to_string(&log_file)
        .map_err(|e| format!("Konnte Logs nicht lesen: {}", e))?;
    
    let all_lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    
    if let Some(n) = lines {
        Ok(all_lines.into_iter().rev().take(n).rev().collect())
    } else {
        Ok(all_lines)
    }
}

/// Listet alle aktiven Trainings auf (für globale Progress Bar)
#[tauri::command]
pub async fn list_active_trainings(
    app_handle: tauri::AppHandle,
) -> Result<Vec<TrainingProgress>, String> {
    let app_data_dir = app_handle.path().app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let trainings_dir = app_data_dir.join("trainings");
    
    if !trainings_dir.exists() {
        return Ok(vec![]);
    }
    
    let mut active_trainings = Vec::new();
    
    // Durchsuche alle Training-Ordner
    if let Ok(entries) = fs::read_dir(trainings_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let progress_file = path.join("progress.json");
                if progress_file.exists() {
                    if let Ok(content) = fs::read_to_string(&progress_file) {
                        if let Ok(progress) = serde_json::from_str::<TrainingProgress>(&content) {
                            // Nur aktive Trainings zurückgeben
                            if progress.status == TrainingStatus::Running || progress.status == TrainingStatus::Paused {
                                active_trainings.push(progress);
                            }
                        }
                    }
                }
            }
        }
    }
    
    Ok(active_trainings)
}

// ==================== Helper Functions ====================

fn get_training_script_path(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    // In production: bundled mit der App
    // In development: aus ml_backend Ordner
    
    #[cfg(debug_assertions)]
    {
        let current_dir = std::env::current_dir()
            .map_err(|e| format!("Konnte current_dir nicht ermitteln: {}", e))?;
        Ok(current_dir.join("ml_backend").join("train.py"))
    }
    
    #[cfg(not(debug_assertions))]
    {
        let resource_dir = app_handle
            .path()
            .resource_dir()
            .map_err(|e| format!("Konnte resource_dir nicht ermitteln: {}", e))?;
        Ok(resource_dir.join("ml_backend").join("train.py"))
    }
}

fn save_training_config(training_dir: &Path, config: &TrainingConfig) -> Result<(), String> {
    let config_file = training_dir.join("config.json");
    let json = serde_json::to_string_pretty(config)
        .map_err(|e| format!("Konnte Config nicht serialisieren: {}", e))?;
    fs::write(config_file, json)
        .map_err(|e| format!("Konnte Config nicht speichern: {}", e))?;
    Ok(())
}

fn save_training_progress(training_dir: &Path, progress: &TrainingProgress) -> Result<(), String> {
    let progress_file = training_dir.join("progress.json");
    let json = serde_json::to_string_pretty(progress)
        .map_err(|e| format!("Konnte Progress nicht serialisieren: {}", e))?;
    fs::write(progress_file, json)
        .map_err(|e| format!("Konnte Progress nicht speichern: {}", e))?;
    Ok(())
}

fn save_training_session(training_dir: &Path, session: &TrainingSession) -> Result<(), String> {
    let session_file = training_dir.join("session.json");
    let json = serde_json::to_string_pretty(session)
        .map_err(|e| format!("Konnte Session nicht serialisieren: {}", e))?;
    fs::write(session_file, json)
        .map_err(|e| format!("Konnte Session nicht speichern: {}", e))?;
    Ok(())
}
