// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ml_backend;
mod database;
mod db_commands;

use tauri::Manager;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;
use database::{Database, get_database_path};

// Global Database State
pub struct AppState {
    db: Mutex<Database>,
}

#[tauri::command]
fn verify_api_key(api_key: String) -> Result<bool, String> {
    // Hier würde die tatsächliche Verifikation gegen das Backend erfolgen
    // Für jetzt eine einfache Überprüfung
    if api_key.starts_with("ft_") && api_key.len() > 20 {
        Ok(true)
    } else {
        Err("Ungültiger API-Key".to_string())
    }
}

#[tauri::command]
fn save_config(app_handle: tauri::AppHandle, api_key: String) -> Result<(), String> {
    let config_dir = app_handle
        .path_resolver()
        .app_config_dir()
        .ok_or("Konnte Config-Verzeichnis nicht finden")?;
    
    fs::create_dir_all(&config_dir)
        .map_err(|e| format!("Konnte Config-Verzeichnis nicht erstellen: {}", e))?;
    
    let config_path = config_dir.join("config.json");
    let config = serde_json::json!({
        "api_key": api_key
    });
    
    fs::write(config_path, config.to_string())
        .map_err(|e| format!("Konnte Config nicht speichern: {}", e))?;
    
    Ok(())
}

#[tauri::command]
fn load_config(app_handle: tauri::AppHandle) -> Result<String, String> {
    let config_dir = app_handle
        .path_resolver()
        .app_config_dir()
        .ok_or("Konnte Config-Verzeichnis nicht finden")?;
    
    let config_path = config_dir.join("config.json");
    
    if !config_path.exists() {
        return Err("Keine Konfiguration gefunden".to_string());
    }
    
    let config_str = fs::read_to_string(config_path)
        .map_err(|e| format!("Konnte Config nicht lesen: {}", e))?;
    
    let config: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| format!("Ungültige Config: {}", e))?;
    
    config["api_key"]
        .as_str()
        .ok_or("API-Key nicht gefunden".to_string())
        .map(|s| s.to_string())
}

#[tauri::command]
fn get_app_data_dir(app_handle: tauri::AppHandle) -> Result<String, String> {
    app_handle
        .path_resolver()
        .app_data_dir()
        .ok_or("Konnte App-Daten-Verzeichnis nicht finden".to_string())
        .map(|p| p.to_string_lossy().to_string())
}

fn main() {
    // Initialisiere Datenbank
    let db_path = get_database_path();
    
    // Erstelle Verzeichnis falls nötig
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent).expect("Konnte Datenbank-Verzeichnis nicht erstellen");
    }
    
    let db = Database::new(db_path).expect("Konnte Datenbank nicht öffnen");
    
    tauri::Builder::default()
        .manage(AppState {
            db: Mutex::new(db),
        })
        .invoke_handler(tauri::generate_handler![
            verify_api_key,
            save_config,
            load_config,
            get_app_data_dir,
            ml_backend::start_training,
            ml_backend::get_training_progress,
            ml_backend::stop_training,
            ml_backend::download_model,
            ml_backend::get_local_models,
            ml_backend::validate_dataset,
            // Database commands
            db_commands::db_create_model,
            db_commands::db_list_models,
            db_commands::db_get_model,
            db_commands::db_delete_model,
            db_commands::db_list_datasets,
            db_commands::db_save_dataset
        ])
        .run(tauri::generate_context!())
        .expect("Fehler beim Starten der Tauri-Anwendung");
}
