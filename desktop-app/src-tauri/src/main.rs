// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod ml_backend;
mod database;
mod db_commands;
mod auth;

use std::fs;
use std::sync::Mutex;
use database::{Database, get_database_path};
use tauri::Manager; // wichtig für .path()

// GLOBAL DB STATE
pub struct AppState {
    db: Mutex<Database>,
}

#[tauri::command]
fn verify_api_key(api_key: String) -> Result<bool, String> {
    if api_key.starts_with("ft_") && api_key.len() > 20 {
        Ok(true)
    } else {
        Err("Ungültiger API-Key".to_string())
    }
}

#[tauri::command]
fn save_config(app_handle: tauri::AppHandle, api_key: String, config: String) -> Result<(), String> {
    let config_dir = app_handle
        .path()
        .app_config_dir()
        .map_err(|e| format!("Konnte Config-Verzeichnis nicht finden: {}", e))?;

    fs::create_dir_all(&config_dir)
        .map_err(|e| format!("Konnte Config-Verzeichnis nicht erstellen: {}", e))?;

    let config_path = config_dir.join("config.json");

    fs::write(config_path, config)
        .map_err(|e| format!("Konnte Config nicht speichern: {}", e))?;

    Ok(())
}

#[tauri::command]
fn load_config(app_handle: tauri::AppHandle) -> Result<String, String> {
    let config_dir = app_handle
        .path()
        .app_config_dir()
        .map_err(|e| format!("Konnte Config-Verzeichnis nicht finden: {}", e))?;

    let config_path = config_dir.join("config.json");

    if !config_path.exists() {
        return Err("Keine Konfiguration gefunden".to_string());
    }

    let config_str = fs::read_to_string(config_path)
        .map_err(|e| format!("Konnte Config nicht lesen: {}", e))?;

    let config: serde_json::Value =
        serde_json::from_str(&config_str).map_err(|e| format!("Ungültige Config: {}", e))?;

    config["api_key"]
        .as_str()
        .ok_or("API-Key nicht gefunden".to_string())
        .map(|s| s.to_string())
}

#[tauri::command]
fn get_app_data_dir(app_handle: tauri::AppHandle) -> Result<String, String> {
    app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("Konnte App-Daten-Verzeichnis nicht finden: {}", e))
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn clear_config(app_handle: tauri::AppHandle) -> Result<(), String> {
    let config_dir = app_handle
        .path()
        .app_config_dir()
        .map_err(|e| format!("Konnte Config-Verzeichnis nicht finden: {}", e))?;

    let config_path = config_dir.join("config.json");

    if config_path.exists() {
        fs::remove_file(config_path)
            .map_err(|e| format!("Konnte Config nicht l\u{00f6}schen: {}", e))?;
    }

    Ok(())
}

fn main() {
    // Datenbank vorbereiten
    let db_path = get_database_path();

    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent).expect("Konnte Datenbank-Verzeichnis nicht erstellen");
    }

    let db = Database::new(db_path).expect("Konnte Datenbank nicht öffnen");

    // Wichtig: Tauri 2: generate_context! nur ohne ()
    let context = tauri::generate_context!();

    tauri::Builder::default()
        .manage(AppState {
            db: Mutex::new(db),
        })
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![
            verify_api_key,
            save_config,
            load_config,
            get_app_data_dir,
            clear_config,
            auth::validate_credentials,
            ml_backend::start_training,
            ml_backend::get_training_progress,
            ml_backend::stop_training,
            ml_backend::download_model,
            ml_backend::get_local_models,
            ml_backend::validate_dataset,
            db_commands::db_create_model,
            db_commands::db_list_models,
            db_commands::db_get_model,
            db_commands::db_delete_model,
            db_commands::db_list_datasets,
            db_commands::db_save_dataset
        ])
        .run(context)
        .expect("Fehler beim Starten der Tauri-Anwendung");
}