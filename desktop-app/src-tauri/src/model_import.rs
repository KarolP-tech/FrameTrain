// Model Import Commands für Tauri 2.x

use std::path::{Path, PathBuf};
use std::fs;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use crate::AppState;
use crate::database::Model;
use tauri::State;
use tauri::Manager;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelImportInfo {
    pub name: String,
    pub path: String,
    pub size_bytes: u64,
    pub file_count: usize,
    pub model_type: String,
}

/// Öffnet einen Ordner-Auswahldialog und gibt den Pfad zurück
#[tauri::command]
pub async fn select_model_folder(app_handle: tauri::AppHandle) -> Result<String, String> {
    use tauri_plugin_dialog::DialogExt;
    
    let file_path = app_handle
        .dialog()
        .file()
        .blocking_pick_folder();
    
    match file_path {
        Some(path) => {
            // FilePath.as_path() gibt Option<&Path> zurück
            match path.as_path() {
                Some(p) => Ok(p.to_string_lossy().to_string()),
                None => Err("Ungültiger Pfad".to_string()),
            }
        }
        None => Err("Kein Ordner ausgewählt".to_string()),
    }
}

/// Validiert einen Modell-Ordner und gibt Infos zurück
#[tauri::command]
pub async fn validate_model_folder(folder_path: String) -> Result<ModelImportInfo, String> {
    let path = Path::new(&folder_path);
    
    // 1. Prüfe ob Ordner existiert
    if !path.exists() {
        return Err("Ordner existiert nicht".to_string());
    }
    
    if !path.is_dir() {
        return Err("Pfad ist kein Ordner".to_string());
    }
    
    // 2. Zähle Dateien und berechne Größe
    let (file_count, total_size) = count_files_and_size(path)
        .map_err(|e| format!("Fehler beim Analysieren: {}", e))?;
    
    if file_count == 0 {
        return Err("Ordner ist leer".to_string());
    }
    
    // 3. Erkenne Modell-Typ (PyTorch, TensorFlow, ONNX, etc.)
    let model_type = detect_model_type(path);
    
    // 4. Hole Ordner-Name als Modell-Name
    let name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("Unbekanntes Modell")
        .to_string();
    
    Ok(ModelImportInfo {
        name,
        path: folder_path,
        size_bytes: total_size,
        file_count,
        model_type,
    })
}

/// Importiert ein Modell in die App-Datenbank und kopiert es ins App-Verzeichnis
#[tauri::command]
pub async fn import_model(
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
    source_path: String,
    model_name: String,
    description: Option<String>,
) -> Result<Model, String> {
    let source = Path::new(&source_path);
    
    // 1. Validiere Quelle
    if !source.exists() || !source.is_dir() {
        return Err("Ungültiger Quellordner".to_string());
    }
    
    // 2. Erstelle Ziel-Verzeichnis in App-Daten (Tauri 2.x)
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let models_dir = app_data_dir.join("models");
    fs::create_dir_all(&models_dir)
        .map_err(|e| format!("Konnte Modell-Verzeichnis nicht erstellen: {}", e))?;
    
    // 3. Erstelle eindeutigen Ordner-Namen mit UUID
    let model_id = Uuid::new_v4().to_string();
    let dest_dir = models_dir.join(&model_id);
    
    // 4. Kopiere Modell-Ordner
    println!("📦 Copying model from {} to {:?}", source_path, dest_dir);
    copy_dir_recursive(source, &dest_dir)
        .map_err(|e| format!("Fehler beim Kopieren: {}", e))?;
    
    println!("✅ Model copied successfully");
    
    // 5. Erkenne Modell-Typ
    let model_type = detect_model_type(&dest_dir);
    
    // 6. Speichere in Datenbank
    let db = state.db.lock().map_err(|e| e.to_string())?;
    
    let model = Model {
        id: model_id,
        name: model_name,
        description,
        base_model: Some(model_type),
        model_path: Some(dest_dir.to_string_lossy().to_string()),
        status: "imported".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        updated_at: chrono::Utc::now().to_rfc3339(),
    };
    
    db.create_model(&model)
        .map_err(|e| format!("Fehler beim Speichern in DB: {}", e))?;
    
    println!("✅ Model saved to database: {}", model.id);
    
    Ok(model)
}

/// Zählt Dateien und berechnet Gesamtgröße rekursiv
fn count_files_and_size(path: &Path) -> std::io::Result<(usize, u64)> {
    let mut file_count = 0;
    let mut total_size = 0;
    
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        
        if metadata.is_file() {
            file_count += 1;
            total_size += metadata.len();
        } else if metadata.is_dir() {
            let (sub_count, sub_size) = count_files_and_size(&entry.path())?;
            file_count += sub_count;
            total_size += sub_size;
        }
    }
    
    Ok((file_count, total_size))
}

/// Erkenne Modell-Typ anhand vorhandener Dateien
fn detect_model_type(path: &Path) -> String {
    let has_pytorch = path.join("pytorch_model.bin").exists() 
        || path.join("model.safetensors").exists()
        || find_files_with_extension(path, "pt").is_some()
        || find_files_with_extension(path, "pth").is_some();
    
    let has_tensorflow = path.join("saved_model.pb").exists()
        || path.join("model.h5").exists()
        || find_files_with_extension(path, "h5").is_some();
    
    let has_onnx = find_files_with_extension(path, "onnx").is_some();
    
    let has_config = path.join("config.json").exists();
    
    if has_pytorch && has_config {
        "PyTorch (Transformers)".to_string()
    } else if has_pytorch {
        "PyTorch".to_string()
    } else if has_tensorflow {
        "TensorFlow".to_string()
    } else if has_onnx {
        "ONNX".to_string()
    } else {
        "Unbekannt".to_string()
    }
}

/// Sucht Dateien mit bestimmter Extension rekursiv
fn find_files_with_extension(path: &Path, extension: &str) -> Option<PathBuf> {
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == extension {
                        return Some(path);
                    }
                }
            } else if path.is_dir() {
                if let Some(found) = find_files_with_extension(&path, extension) {
                    return Some(found);
                }
            }
        }
    }
    None
}

/// Kopiert einen Ordner rekursiv
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        
        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    
    Ok(())
}
