// Dataset Management für FrameTrain
// Verwaltet Datasets pro Modell mit Validierung und Split-Management

use std::path::{Path, PathBuf};
use std::fs;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use crate::AppState;
use crate::database::Model;
use tauri::{State, Manager};

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DatasetType {
    Image,
    Video,
    Text,
    Audio,
    Mixed,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SplitType {
    Train,
    Val,
    Test,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DatasetStatus {
    Unused,      // Noch nicht im Training verwendet
    InUse,       // Aktuell im Training
    Used,        // Bereits verwendet
    HardExample, // Als schwieriges Beispiel markiert
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub id: String,
    pub model_id: String,
    pub name: String,
    pub dataset_type: DatasetType,
    pub file_path: String,
    pub file_count: usize,
    pub total_size_bytes: u64,
    pub split_type: SplitType,
    pub status: DatasetStatus,
    pub formats: Vec<String>,
    pub is_hard_example: bool,
    pub created_at: String,
    pub last_used_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetValidation {
    pub is_valid: bool,
    pub detected_type: DatasetType,
    pub file_count: usize,
    pub formats: Vec<String>,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub model_compatible: bool,
    pub expected_formats: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SplitConfig {
    pub train_ratio: f32,
    pub val_ratio: f32,
    pub test_ratio: f32,
    pub shuffle: bool,
}

// Unterstützte Formate pro Datentyp
const IMAGE_FORMATS: &[&str] = &["jpg", "jpeg", "png", "bmp", "gif", "webp", "tiff", "tif"];
const VIDEO_FORMATS: &[&str] = &["mp4", "avi", "mov", "mkv", "webm", "flv", "m4v"];
const TEXT_FORMATS: &[&str] = &["txt", "csv", "json", "jsonl", "tsv", "parquet"];
const AUDIO_FORMATS: &[&str] = &["mp3", "wav", "flac", "ogg", "m4a", "aac"];

// Modell-spezifische Anforderungen
fn get_model_requirements(model: &Model) -> (Vec<DatasetType>, Vec<String>) {
    let model_name_lower = model.name.to_lowercase();
    let base_model_lower = model.base_model.as_ref()
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    
    // Kombiniere Name und base_model für bessere Erkennung
    let full_name = format!("{} {}", model_name_lower, base_model_lower);
    
    // Text-Modelle (NLP, Language Models)
    if full_name.contains("bert") || full_name.contains("gpt") || full_name.contains("t5") 
        || full_name.contains("mt5") || full_name.contains("roberta") || full_name.contains("xlm")
        || full_name.contains("llama") || full_name.contains("mistral") || full_name.contains("claude")
        || full_name.contains("gemini") || full_name.contains("transformer") 
        || full_name.contains("language") || full_name.contains("nlp")
        || full_name.contains("spellcheck") || full_name.contains("grammar") {
        return (
            vec![DatasetType::Text],
            vec!["txt".to_string(), "csv".to_string(), "json".to_string(), "jsonl".to_string(), "tsv".to_string(), "parquet".to_string()]
        );
    }
    
    // Vision-Modelle (Computer Vision)
    if full_name.contains("resnet") || full_name.contains("vgg") || full_name.contains("inception")
        || full_name.contains("yolo") || full_name.contains("efficientnet") 
        || full_name.contains("mobilenet") || full_name.contains("vision") 
        || full_name.contains("cnn") || full_name.contains("convnet")
        || full_name.contains("image") || full_name.contains("detection")
        || full_name.contains("segmentation") || full_name.contains("classification") {
        return (
            vec![DatasetType::Image],
            vec!["jpg".to_string(), "jpeg".to_string(), "png".to_string(), "bmp".to_string(), "webp".to_string()]
        );
    }
    
    // Multimodal-Modelle (CLIP, ALIGN, etc.)
    if full_name.contains("clip") || full_name.contains("align") || full_name.contains("blip")
        || full_name.contains("flamingo") || full_name.contains("multimodal") {
        return (
            vec![DatasetType::Image, DatasetType::Text, DatasetType::Mixed],
            vec!["jpg".to_string(), "png".to_string(), "txt".to_string(), "json".to_string(), "csv".to_string()]
        );
    }
    
    // Video-Modelle
    if full_name.contains("video") || full_name.contains("temporal") 
        || full_name.contains("action") || full_name.contains("i3d") {
        return (
            vec![DatasetType::Video],
            vec!["mp4".to_string(), "avi".to_string(), "mov".to_string(), "mkv".to_string()]
        );
    }
    
    // Audio-Modelle (Speech, ASR, TTS)
    if full_name.contains("wav2vec") || full_name.contains("whisper") || full_name.contains("audio")
        || full_name.contains("speech") || full_name.contains("sound") || full_name.contains("asr")
        || full_name.contains("tts") || full_name.contains("voice") {
        return (
            vec![DatasetType::Audio],
            vec!["wav".to_string(), "mp3".to_string(), "flac".to_string(), "m4a".to_string()]
        );
    }
    
    // PyTorch Generic
    if full_name.contains("pytorch") {
        return (
            vec![DatasetType::Mixed],
            vec!["pt".to_string(), "pth".to_string(), "pkl".to_string(), "pickle".to_string()]
        );
    }
    
    // TensorFlow Generic  
    if full_name.contains("tensorflow") || full_name.contains("keras") {
        return (
            vec![DatasetType::Mixed],
            vec!["tfrecord".to_string(), "h5".to_string(), "npy".to_string()]
        );
    }
    
    // Default: Alle Typen erlaubt
    (
        vec![DatasetType::Image, DatasetType::Video, DatasetType::Text, DatasetType::Audio, DatasetType::Mixed],
        TEXT_FORMATS.iter().chain(IMAGE_FORMATS.iter()).chain(VIDEO_FORMATS.iter()).chain(AUDIO_FORMATS.iter())
            .map(|s| s.to_string())
            .collect()
    )
}

/// Öffnet Datei/Ordner-Dialog für Dataset-Upload
#[tauri::command]
pub async fn select_dataset_path(app_handle: tauri::AppHandle, allow_multiple: bool) -> Result<Vec<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    
    if allow_multiple {
        let files = app_handle
            .dialog()
            .file()
            .blocking_pick_files();
        
        match files {
            Some(paths) => {
                let path_strings: Vec<String> = paths.iter()
                    .filter_map(|p| p.as_path())
                    .map(|p| p.to_string_lossy().to_string())
                    .collect();
                Ok(path_strings)
            }
            None => Err("Keine Dateien ausgewählt".to_string()),
        }
    } else {
        let folder = app_handle
            .dialog()
            .file()
            .blocking_pick_folder();
        
        match folder {
            Some(path) => {
                match path.as_path() {
                    Some(p) => Ok(vec![p.to_string_lossy().to_string()]),
                    None => Err("Ungültiger Pfad".to_string()),
                }
            }
            None => Err("Kein Ordner ausgewählt".to_string()),
        }
    }
}

/// Validiert Dataset und erkennt Typ (mit Modell-Kompatibilität)
#[tauri::command]
pub async fn validate_dataset_path(
    state: State<'_, AppState>,
    path: String,
    model_id: String,
) -> Result<DatasetValidation, String> {
    let dataset_path = Path::new(&path);
    
    if !dataset_path.exists() {
        return Err("Pfad existiert nicht".to_string());
    }
    
    // Lade Modell-Info
    let db = state.db.lock().map_err(|e| e.to_string())?;
    let model = db.get_model(&model_id)
        .map_err(|e| format!("Modell nicht gefunden: {}", e))?;
    
    // Hole Modell-Anforderungen
    let (allowed_types, expected_formats) = get_model_requirements(&model);
    
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let mut all_formats = Vec::new();
    let mut file_count = 0;
    
    // Sammle alle Dateien
    let files = if dataset_path.is_dir() {
        collect_files_recursive(dataset_path, &mut file_count)?
    } else {
        file_count = 1;
        vec![dataset_path.to_path_buf()]
    };
    
    if files.is_empty() {
        errors.push("Keine Dateien gefunden".to_string());
        return Ok(DatasetValidation {
            is_valid: false,
            detected_type: DatasetType::Mixed,
            file_count: 0,
            formats: vec![],
            errors,
            warnings,
            model_compatible: false,
            expected_formats: expected_formats.clone(),
        });
    }
    
    // Sammle alle Formate
    for file in &files {
        if let Some(ext) = file.extension() {
            if let Some(ext_str) = ext.to_str() {
                let ext_lower = ext_str.to_lowercase();
                if !all_formats.contains(&ext_lower) {
                    all_formats.push(ext_lower);
                }
            }
        }
    }
    
    // Erkenne Datentyp
    let detected_type = detect_dataset_type(&all_formats);
    
    // Prüfe Modell-Kompatibilität
    let model_compatible = allowed_types.contains(&detected_type);
    
    if !model_compatible {
        errors.push(format!(
            "Dataset-Typ {:?} ist nicht kompatibel mit Modell '{}'. Erwartet: {:?}",
            detected_type, model.name, allowed_types
        ));
    }
    
    // Validiere Format-Kompatibilität gegen Modell-Anforderungen
    let mut incompatible_formats = Vec::new();
    for format in &all_formats {
        if !expected_formats.contains(format) {
            incompatible_formats.push(format.clone());
        }
    }
    
    if !incompatible_formats.is_empty() {
        errors.push(format!(
            "Inkompatible Formate für Modell '{}': {}. Erwartet: {}",
            model.name,
            incompatible_formats.join(", "),
            expected_formats.join(", ")
        ));
    }
    
    // Warnungen
    if file_count < 10 {
        warnings.push("Sehr kleiner Dataset (< 10 Dateien)".to_string());
    }
    
    if all_formats.len() > 3 && detected_type != DatasetType::Mixed {
        warnings.push("Viele verschiedene Formate - eventuell gemischter Dataset".to_string());
    }
    
    let is_valid = model_compatible && incompatible_formats.is_empty() && errors.is_empty();
    
    Ok(DatasetValidation {
        is_valid,
        detected_type,
        file_count,
        formats: all_formats,
        errors,
        warnings,
        model_compatible,
        expected_formats,
    })
}

/// Importiert Dataset und kopiert in App-Verzeichnis
#[tauri::command]
pub async fn import_dataset(
    app_handle: tauri::AppHandle,
    model_id: String,
    source_path: String,
    dataset_name: String,
    dataset_type: DatasetType,
    split_type: SplitType,
) -> Result<DatasetInfo, String> {
    let source = Path::new(&source_path);
    
    if !source.exists() {
        return Err("Quellpfad existiert nicht".to_string());
    }
    
    // App-Daten-Verzeichnis
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    // Datasets-Ordner pro Modell
    let datasets_dir = app_data_dir.join("datasets").join(&model_id);
    fs::create_dir_all(&datasets_dir)
        .map_err(|e| format!("Konnte Dataset-Verzeichnis nicht erstellen: {}", e))?;
    
    // Eindeutige Dataset-ID
    let dataset_id = Uuid::new_v4().to_string();
    let dest_dir = datasets_dir.join(&dataset_id);
    
    println!("📦 Importing dataset {} for model {}", dataset_name, model_id);
    println!("   Source: {}", source_path);
    println!("   Destination: {:?}", dest_dir);
    
    // Kopiere Dataset
    let (file_count, total_size) = if source.is_dir() {
        copy_dir_recursive(source, &dest_dir)?;
        count_files_and_size(&dest_dir)?
    } else {
        // Einzelne Datei
        fs::create_dir_all(&dest_dir)
            .map_err(|e| format!("Konnte Verzeichnis erstellen: {}", e))?;
        let dest_file = dest_dir.join(source.file_name().unwrap());
        fs::copy(source, &dest_file)
            .map_err(|e| format!("Konnte Datei kopieren: {}", e))?;
        (1, fs::metadata(source).map(|m| m.len()).unwrap_or(0))
    };
    
    // Sammle Formate
    let formats = collect_formats(&dest_dir)?;
    
    let dataset_info = DatasetInfo {
        id: dataset_id,
        model_id: model_id.clone(),
        name: dataset_name,
        dataset_type,
        file_path: dest_dir.to_string_lossy().to_string(),
        file_count,
        total_size_bytes: total_size,
        split_type,
        status: DatasetStatus::Unused,
        formats,
        is_hard_example: false,
        created_at: chrono::Utc::now().to_rfc3339(),
        last_used_at: None,
    };
    
    // Speichere Metadaten
    save_dataset_metadata(&dest_dir, &dataset_info)?;
    
    println!("✅ Dataset imported: {}", dataset_info.id);
    
    Ok(dataset_info)
}

/// Liste alle Datasets für ein Modell
#[tauri::command]
pub async fn list_datasets_for_model(
    app_handle: tauri::AppHandle,
    model_id: String,
) -> Result<Vec<DatasetInfo>, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let datasets_dir = app_data_dir.join("datasets").join(&model_id);
    
    if !datasets_dir.exists() {
        return Ok(vec![]);
    }
    
    let mut datasets = Vec::new();
    
    for entry in fs::read_dir(&datasets_dir)
        .map_err(|e| format!("Konnte Datasets-Verzeichnis nicht lesen: {}", e))? 
    {
        let entry = entry.map_err(|e| format!("Fehler beim Lesen: {}", e))?;
        let path = entry.path();
        
        if path.is_dir() {
            if let Ok(dataset_info) = load_dataset_metadata(&path) {
                datasets.push(dataset_info);
            }
        }
    }
    
    // Sortiere nach Erstellungsdatum (neueste zuerst)
    datasets.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    
    Ok(datasets)
}

/// Automatischer Train/Val/Test Split
#[tauri::command]
pub async fn auto_split_dataset(
    app_handle: tauri::AppHandle,
    model_id: String,
    dataset_id: String,
    config: SplitConfig,
) -> Result<Vec<DatasetInfo>, String> {
    // Validiere Ratios
    let total = config.train_ratio + config.val_ratio + config.test_ratio;
    if (total - 1.0).abs() > 0.01 {
        return Err(format!("Ratios müssen 1.0 ergeben (aktuell: {})", total));
    }
    
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let dataset_path = app_data_dir
        .join("datasets")
        .join(&model_id)
        .join(&dataset_id);
    
    if !dataset_path.exists() {
        return Err("Dataset nicht gefunden".to_string());
    }
    
    // Lade Original-Metadata
    let original_dataset = load_dataset_metadata(&dataset_path)?;
    
    // Sammle alle Dateien
    let mut all_files = Vec::new();
    collect_files_for_split(&dataset_path, &mut all_files)?;
    
    if all_files.is_empty() {
        return Err("Keine Dateien zum Splitten gefunden".to_string());
    }
    
    // Shuffle wenn gewünscht
    if config.shuffle {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        all_files.shuffle(&mut rng);
    }
    
    let total_files = all_files.len();
    let train_count = (total_files as f32 * config.train_ratio) as usize;
    let val_count = (total_files as f32 * config.val_ratio) as usize;
    
    // Split in drei Teile
    let train_files = &all_files[..train_count];
    let val_files = &all_files[train_count..train_count + val_count];
    let test_files = &all_files[train_count + val_count..];
    
    let datasets_dir = app_data_dir.join("datasets").join(&model_id);
    
    // Erstelle drei neue Datasets
    let mut result_datasets = Vec::new();
    
    // Train Dataset
    if !train_files.is_empty() {
        let train_dataset = create_split_dataset(
            &datasets_dir,
            &original_dataset,
            train_files,
            SplitType::Train,
            "train",
        )?;
        result_datasets.push(train_dataset);
    }
    
    // Val Dataset
    if !val_files.is_empty() {
        let val_dataset = create_split_dataset(
            &datasets_dir,
            &original_dataset,
            val_files,
            SplitType::Val,
            "val",
        )?;
        result_datasets.push(val_dataset);
    }
    
    // Test Dataset
    if !test_files.is_empty() {
        let test_dataset = create_split_dataset(
            &datasets_dir,
            &original_dataset,
            test_files,
            SplitType::Test,
            "test",
        )?;
        result_datasets.push(test_dataset);
    }
    
    println!("✅ Dataset split complete:");
    println!("   Train: {} files", train_files.len());
    println!("   Val: {} files", val_files.len());
    println!("   Test: {} files", test_files.len());
    
    Ok(result_datasets)
}

/// Markiere Dataset als Hard Example
#[tauri::command]
pub async fn toggle_hard_example(
    app_handle: tauri::AppHandle,
    model_id: String,
    dataset_id: String,
) -> Result<DatasetInfo, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let dataset_path = app_data_dir
        .join("datasets")
        .join(&model_id)
        .join(&dataset_id);
    
    let mut dataset_info = load_dataset_metadata(&dataset_path)?;
    dataset_info.is_hard_example = !dataset_info.is_hard_example;
    
    if dataset_info.is_hard_example {
        dataset_info.status = DatasetStatus::HardExample;
    } else {
        dataset_info.status = DatasetStatus::Unused;
    }
    
    save_dataset_metadata(&dataset_path, &dataset_info)?;
    
    Ok(dataset_info)
}

/// Update Dataset Status
#[tauri::command]
pub async fn update_dataset_status(
    app_handle: tauri::AppHandle,
    model_id: String,
    dataset_id: String,
    status: DatasetStatus,
) -> Result<DatasetInfo, String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let dataset_path = app_data_dir
        .join("datasets")
        .join(&model_id)
        .join(&dataset_id);
    
    let mut dataset_info = load_dataset_metadata(&dataset_path)?;
    dataset_info.status = status;
    
    if matches!(status, DatasetStatus::InUse | DatasetStatus::Used) {
        dataset_info.last_used_at = Some(chrono::Utc::now().to_rfc3339());
    }
    
    save_dataset_metadata(&dataset_path, &dataset_info)?;
    
    Ok(dataset_info)
}

/// Lösche Dataset
#[tauri::command]
pub async fn delete_dataset(
    app_handle: tauri::AppHandle,
    model_id: String,
    dataset_id: String,
) -> Result<(), String> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| format!("App-Datenverzeichnis nicht gefunden: {}", e))?;
    
    let dataset_path = app_data_dir
        .join("datasets")
        .join(&model_id)
        .join(&dataset_id);
    
    if dataset_path.exists() {
        fs::remove_dir_all(&dataset_path)
            .map_err(|e| format!("Konnte Dataset nicht löschen: {}", e))?;
    }
    
    Ok(())
}

// ==================== Helper Functions ====================

fn detect_dataset_type(formats: &[String]) -> DatasetType {
    let image_count = formats.iter().filter(|f| IMAGE_FORMATS.contains(&f.as_str())).count();
    let video_count = formats.iter().filter(|f| VIDEO_FORMATS.contains(&f.as_str())).count();
    let text_count = formats.iter().filter(|f| TEXT_FORMATS.contains(&f.as_str())).count();
    let audio_count = formats.iter().filter(|f| AUDIO_FORMATS.contains(&f.as_str())).count();
    
    let total = formats.len();
    
    if image_count == total {
        DatasetType::Image
    } else if video_count == total {
        DatasetType::Video
    } else if text_count == total {
        DatasetType::Text
    } else if audio_count == total {
        DatasetType::Audio
    } else {
        DatasetType::Mixed
    }
}

fn validate_formats(formats: &[String], expected_type: &DatasetType) -> (bool, Vec<String>) {
    let mut errors = Vec::new();
    
    let valid_formats: &[&str] = match expected_type {
        DatasetType::Image => IMAGE_FORMATS,
        DatasetType::Video => VIDEO_FORMATS,
        DatasetType::Text => TEXT_FORMATS,
        DatasetType::Audio => AUDIO_FORMATS,
        DatasetType::Mixed => return (true, errors), // Mixed erlaubt alles
    };
    
    for format in formats {
        if !valid_formats.contains(&format.as_str()) {
            errors.push(format!("Ungültiges Format für {:?}: {}", expected_type, format));
        }
    }
    
    (errors.is_empty(), errors)
}

fn collect_files_recursive(path: &Path, count: &mut usize) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    
    for entry in fs::read_dir(path).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        
        if path.is_file() {
            files.push(path);
            *count += 1;
        } else if path.is_dir() {
            files.extend(collect_files_recursive(&path, count)?);
        }
    }
    
    Ok(files)
}

fn collect_formats(path: &Path) -> Result<Vec<String>, String> {
    let mut formats = Vec::new();
    let mut count = 0;
    let files = collect_files_recursive(path, &mut count)?;
    
    for file in files {
        if let Some(ext) = file.extension() {
            if let Some(ext_str) = ext.to_str() {
                let ext_lower = ext_str.to_lowercase();
                if !formats.contains(&ext_lower) {
                    formats.push(ext_lower);
                }
            }
        }
    }
    
    Ok(formats)
}

fn count_files_and_size(path: &Path) -> Result<(usize, u64), String> {
    let mut file_count = 0;
    let mut total_size = 0;
    
    for entry in fs::read_dir(path).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let metadata = entry.metadata().map_err(|e| e.to_string())?;
        
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

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    fs::create_dir_all(dst).map_err(|e| e.to_string())?;
    
    for entry in fs::read_dir(src).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let file_type = entry.file_type().map_err(|e| e.to_string())?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        
        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path).map_err(|e| e.to_string())?;
        }
    }
    
    Ok(())
}

fn save_dataset_metadata(dataset_path: &Path, info: &DatasetInfo) -> Result<(), String> {
    let metadata_path = dataset_path.join("metadata.json");
    let json = serde_json::to_string_pretty(info)
        .map_err(|e| format!("Konnte Metadata nicht serialisieren: {}", e))?;
    fs::write(metadata_path, json)
        .map_err(|e| format!("Konnte Metadata nicht speichern: {}", e))?;
    Ok(())
}

fn load_dataset_metadata(dataset_path: &Path) -> Result<DatasetInfo, String> {
    let metadata_path = dataset_path.join("metadata.json");
    let json = fs::read_to_string(metadata_path)
        .map_err(|e| format!("Konnte Metadata nicht lesen: {}", e))?;
    serde_json::from_str(&json)
        .map_err(|e| format!("Konnte Metadata nicht deserialisieren: {}", e))
}

fn collect_files_for_split(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(path).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        
        // Skip metadata.json
        if path.file_name().and_then(|n| n.to_str()) == Some("metadata.json") {
            continue;
        }
        
        if path.is_file() {
            files.push(path);
        } else if path.is_dir() {
            collect_files_for_split(&path, files)?;
        }
    }
    
    Ok(())
}

fn create_split_dataset(
    datasets_dir: &Path,
    original: &DatasetInfo,
    files: &[PathBuf],
    split_type: SplitType,
    suffix: &str,
) -> Result<DatasetInfo, String> {
    let dataset_id = Uuid::new_v4().to_string();
    let dest_dir = datasets_dir.join(&dataset_id);
    
    fs::create_dir_all(&dest_dir)
        .map_err(|e| format!("Konnte Split-Verzeichnis nicht erstellen: {}", e))?;
    
    let mut total_size = 0u64;
    
    // Kopiere Dateien
    for file in files {
        let filename = file.file_name().unwrap();
        let dest_file = dest_dir.join(filename);
        fs::copy(file, &dest_file)
            .map_err(|e| format!("Konnte Datei kopieren: {}", e))?;
        total_size += fs::metadata(file).map(|m| m.len()).unwrap_or(0);
    }
    
    let dataset_info = DatasetInfo {
        id: dataset_id,
        model_id: original.model_id.clone(),
        name: format!("{} ({})", original.name, suffix),
        dataset_type: original.dataset_type.clone(),
        file_path: dest_dir.to_string_lossy().to_string(),
        file_count: files.len(),
        total_size_bytes: total_size,
        split_type,
        status: DatasetStatus::Unused,
        formats: original.formats.clone(),
        is_hard_example: false,
        created_at: chrono::Utc::now().to_rfc3339(),
        last_used_at: None,
    };
    
    save_dataset_metadata(&dest_dir, &dataset_info)?;
    
    Ok(dataset_info)
}
