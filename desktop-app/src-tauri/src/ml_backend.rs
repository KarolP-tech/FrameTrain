// Tauri Command: ML Training Backend
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model_name: String,
    pub dataset_path: String,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
    pub optimizer: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub epoch: u32,
    pub loss: f32,
    pub accuracy: f32,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub version: u32,
    pub status: String,
    pub created_at: String,
}

// Startet Training-Prozess
#[tauri::command]
pub async fn start_training(config: TrainingConfig) -> Result<String, String> {
    // Python Training Script aufrufen
    let python_script = get_training_script_path();
    
    let output = Command::new("python")
        .arg(python_script)
        .arg("--model").arg(&config.model_name)
        .arg("--dataset").arg(&config.dataset_path)
        .arg("--epochs").arg(config.epochs.to_string())
        .arg("--batch-size").arg(config.batch_size.to_string())
        .arg("--learning-rate").arg(config.learning_rate.to_string())
        .arg("--optimizer").arg(&config.optimizer)
        .output()
        .map_err(|e| format!("Training fehlgeschlagen: {}", e))?;

    if output.status.success() {
        Ok("Training gestartet".to_string())
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        Err(format!("Training-Fehler: {}", error))
    }
}

// Lädt Trainingsfortschritt
#[tauri::command]
pub async fn get_training_progress(training_id: String) -> Result<TrainingProgress, String> {
    let progress_file = get_progress_file_path(&training_id);
    
    let content = fs::read_to_string(&progress_file)
        .map_err(|e| format!("Fortschritt konnte nicht geladen werden: {}", e))?;
    
    let progress: TrainingProgress = serde_json::from_str(&content)
        .map_err(|e| format!("JSON Parse-Fehler: {}", e))?;
    
    Ok(progress)
}

// Stoppt Training
#[tauri::command]
pub async fn stop_training(training_id: String) -> Result<String, String> {
    // Signal an Python-Prozess senden
    // TODO: Implementierung mit Process-Management
    Ok("Training gestoppt".to_string())
}

// Lädt Modell von HuggingFace
#[tauri::command]
pub async fn download_model(model_name: String) -> Result<String, String> {
    let python_script = get_download_script_path();
    
    let output = Command::new("python")
        .arg(python_script)
        .arg("--model").arg(&model_name)
        .output()
        .map_err(|e| format!("Download fehlgeschlagen: {}", e))?;

    if output.status.success() {
        Ok(format!("Modell {} heruntergeladen", model_name))
    } else {
        let error = String::from_utf8_lossy(&output.stderr);
        Err(format!("Download-Fehler: {}", error))
    }
}

// Lädt Liste aller lokalen Modelle
#[tauri::command]
pub async fn get_local_models() -> Result<Vec<ModelInfo>, String> {
    let models_dir = get_models_directory();
    
    let mut models = Vec::new();
    
    if let Ok(entries) = fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            if let Ok(metadata_path) = entry.path().join("metadata.json").canonicalize() {
                if let Ok(content) = fs::read_to_string(metadata_path) {
                    if let Ok(model_info) = serde_json::from_str::<ModelInfo>(&content) {
                        models.push(model_info);
                    }
                }
            }
        }
    }
    
    Ok(models)
}

// Validiert Datensatz
#[tauri::command]
pub async fn validate_dataset(dataset_path: String) -> Result<bool, String> {
    let path = PathBuf::from(&dataset_path);
    
    if !path.exists() {
        return Err("Datensatz existiert nicht".to_string());
    }
    
    // Prüfe Dateiformat
    let extension = path.extension()
        .and_then(|s| s.to_str())
        .ok_or("Ungültiges Dateiformat")?;
    
    match extension.to_lowercase().as_str() {
        "csv" | "json" | "txt" => Ok(true),
        _ => Err("Nicht unterstütztes Format. Verwende CSV, JSON oder TXT".to_string()),
    }
}

// Hilfsfunktionen
fn get_training_script_path() -> PathBuf {
    // TODO: Pfad zum Python Training Script
    PathBuf::from("./ml_backend/train.py")
}

fn get_download_script_path() -> PathBuf {
    PathBuf::from("./ml_backend/download_model.py")
}

fn get_progress_file_path(training_id: &str) -> PathBuf {
    get_data_directory().join("trainings").join(format!("{}_progress.json", training_id))
}

fn get_models_directory() -> PathBuf {
    get_data_directory().join("models")
}

fn get_data_directory() -> PathBuf {
    // Platform-spezifischer Daten-Ordner
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
