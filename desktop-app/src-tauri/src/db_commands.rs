// Database Commands für Tauri Frontend

use crate::AppState;
use crate::database::{Model, Dataset};
use tauri::State;
use uuid::Uuid;

#[tauri::command]
pub fn db_create_model(
    state: State<AppState>,
    name: String,
    description: Option<String>,
    base_model: Option<String>,
) -> Result<Model, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    
    let model = Model {
        id: Uuid::new_v4().to_string(),
        name,
        description,
        base_model,
        model_path: None,
        status: "created".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        updated_at: chrono::Utc::now().to_rfc3339(),
    };
    
    db.create_model(&model)
        .map_err(|e| format!("Fehler beim Erstellen des Modells: {}", e))?;
    
    Ok(model)
}

#[tauri::command]
pub fn db_list_models(state: State<AppState>) -> Result<Vec<Model>, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    
    db.list_models()
        .map_err(|e| format!("Fehler beim Laden der Modelle: {}", e))
}

#[tauri::command]
pub fn db_get_model(state: State<AppState>, id: String) -> Result<Model, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    
    db.get_model(&id)
        .map_err(|e| format!("Modell nicht gefunden: {}", e))
}

#[tauri::command]
pub fn db_delete_model(state: State<AppState>, id: String) -> Result<(), String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    
    db.delete_model(&id)
        .map_err(|e| format!("Fehler beim Löschen: {}", e))
}

#[tauri::command]
pub fn db_save_dataset(
    state: State<AppState>,
    name: String,
    file_path: String,
    file_type: String,
) -> Result<Dataset, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    
    let dataset = Dataset {
        id: Uuid::new_v4().to_string(),
        name,
        file_path,
        file_type,
        size_bytes: None,
        rows_count: None,
        columns_count: None,
        validated: false,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    
    db.save_dataset(&dataset)
        .map_err(|e| format!("Fehler beim Speichern: {}", e))?;
    
    Ok(dataset)
}

#[tauri::command]
pub fn db_list_datasets(state: State<AppState>) -> Result<Vec<Dataset>, String> {
    let db = state.db.lock().map_err(|e| e.to_string())?;
    
    db.list_datasets()
        .map_err(|e| format!("Fehler beim Laden: {}", e))
}
