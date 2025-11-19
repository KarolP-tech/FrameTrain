// Config Validation für Training
// Analysiert Training-Config und gibt Empfehlungen

use serde::{Deserialize, Serialize};
use super::training_manager::TrainingConfig;

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
    pub category: String,
    pub message: String,
    pub suggestion: String,
}

/// Validiert Training-Config und gibt detaillierte Empfehlungen
#[tauri::command]
pub async fn validate_training_config(
    config: TrainingConfig,
    total_samples: u32,
) -> Result<ConfigValidation, String> {
    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();
    let mut score = 100.0f32;
    
    let effective_batch = config.batch_size * config.gradient_accumulation_steps;
    let total_steps = if effective_batch > 0 {
        (total_samples / effective_batch) * config.epochs
    } else {
        0
    };
    
    // 1. ==================== BATCH SIZE ====================
    if config.batch_size < 1 {
        issues.push(ConfigIssue {
            severity: "error".to_string(),
            category: "batch_size".to_string(),
            message: "Batch size muss mindestens 1 sein".to_string(),
            suggestion: "Setze batch_size auf 4-8 für optimale Performance".to_string(),
        });
        score -= 20.0;
    } else if config.batch_size > 128 {
        issues.push(ConfigIssue {
            severity: "warning".to_string(),
            category: "batch_size".to_string(),
            message: "Sehr große Batch Size → Speicherprobleme möglich".to_string(),
            suggestion: "M1 Mac: 4-16, moderne GPU: 16-64".to_string(),
        });
        score -= 10.0;
    } else if config.batch_size >= 4 && config.batch_size <= 32 {
        recommendations.push("✅ Batch Size ist optimal".to_string());
    }
    
    // Effektive Batch Size
    if effective_batch < 8 {
        warnings.push(format!(
            "⚠️ Effektive Batch Size ({}) ist klein → Training könnte instabil sein",
            effective_batch
        ));
        score -= 5.0;
    } else if effective_batch >= 16 && effective_batch <= 64 {
        recommendations.push("✅ Effektive Batch Size ist perfekt".to_string());
    }
    
    // 2. ==================== LEARNING RATE ====================
    if config.learning_rate <= 0.0 {
        issues.push(ConfigIssue {
            severity: "error".to_string(),
            category: "learning_rate".to_string(),
            message: "Learning Rate muss positiv sein".to_string(),
            suggestion: "Fine-Tuning: 1e-5 bis 5e-5, Training from scratch: 1e-4 bis 1e-3".to_string(),
        });
        score -= 20.0;
    } else if config.learning_rate > 0.001 {
        issues.push(ConfigIssue {
            severity: "warning".to_string(),
            category: "learning_rate".to_string(),
            message: "Learning Rate sehr hoch → Instabiles Training".to_string(),
            suggestion: "Reduziere auf 2e-5 für stabiles Fine-Tuning".to_string(),
        });
        score -= 15.0;
    } else if config.learning_rate < 0.00001 {
        warnings.push("⚠️ Learning Rate sehr niedrig → Sehr langsames Training".to_string());
        score -= 5.0;
    } else if config.learning_rate >= 0.00001 && config.learning_rate <= 0.0001 {
        recommendations.push("✅ Learning Rate ist optimal für Fine-Tuning".to_string());
    }
    
    // 3. ==================== EPOCHS ====================
    if config.epochs == 0 {
        issues.push(ConfigIssue {
            severity: "error".to_string(),
            category: "epochs".to_string(),
            message: "Mindestens 1 Epoch erforderlich".to_string(),
            suggestion: "Setze epochs auf 3-5 für Fine-Tuning".to_string(),
        });
        score -= 20.0;
    } else if config.epochs > 20 {
        warnings.push("⚠️ Viele Epochs → Risiko für Overfitting".to_string());
        recommendations.push("💡 Aktiviere Early Stopping (patience: 2-3)".to_string());
        score -= 5.0;
    } else if config.epochs >= 3 && config.epochs <= 10 {
        recommendations.push("✅ Epoch-Anzahl ist ideal".to_string());
    } else if config.epochs == 1 {
        warnings.push("⚠️ Nur 1 Epoch → Modell lernt möglicherweise nicht genug".to_string());
        score -= 10.0;
    }
    
    // 4. ==================== WARMUP ====================
    if config.warmup_ratio < 0.0 || config.warmup_ratio > 1.0 {
        issues.push(ConfigIssue {
            severity: "error".to_string(),
            category: "warmup".to_string(),
            message: "Warmup ratio muss zwischen 0 und 1 liegen".to_string(),
            suggestion: "Empfohlen: 0.05 (5%) für stabileren Start".to_string(),
        });
        score -= 15.0;
    } else if config.warmup_ratio > 0.3 {
        warnings.push("⚠️ Warmup ratio sehr hoch → Verschwendete Trainingszeit".to_string());
        score -= 5.0;
    } else if config.warmup_ratio >= 0.03 && config.warmup_ratio <= 0.1 {
        recommendations.push("✅ Warmup ist optimal konfiguriert".to_string());
    } else if config.warmup_ratio == 0.0 {
        warnings.push("⚠️ Kein Warmup → Instabiler Trainingsstart möglich".to_string());
        score -= 3.0;
    }
    
    // 5. ==================== WEIGHT DECAY ====================
    if config.weight_decay < 0.0 {
        issues.push(ConfigIssue {
            severity: "error".to_string(),
            category: "weight_decay".to_string(),
            message: "Weight Decay darf nicht negativ sein".to_string(),
            suggestion: "Empfohlen: 0.01 für Regularisierung".to_string(),
        });
        score -= 10.0;
    } else if config.weight_decay > 0.1 {
        warnings.push("⚠️ Weight Decay sehr hoch → Risiko für Underfitting".to_string());
        score -= 5.0;
    } else if config.weight_decay >= 0.001 && config.weight_decay <= 0.01 {
        recommendations.push("✅ Weight Decay ist gut gewählt".to_string());
    }
    
    // 6. ==================== GRADIENT CLIPPING ====================
    if config.max_grad_norm <= 0.0 {
        warnings.push("⚠️ Gradient Clipping deaktiviert → Exploding Gradients möglich".to_string());
        score -= 8.0;
    } else if config.max_grad_norm > 10.0 {
        warnings.push("⚠️ Max Grad Norm sehr hoch → Wenig Schutz vor Exploding Gradients".to_string());
        score -= 3.0;
    } else if config.max_grad_norm >= 0.5 && config.max_grad_norm <= 2.0 {
        recommendations.push("✅ Gradient Clipping ist optimal".to_string());
    }
    
    // 7. ==================== PRECISION ====================
    if config.fp16 && config.bf16 {
        issues.push(ConfigIssue {
            severity: "warning".to_string(),
            category: "precision".to_string(),
            message: "FP16 und BF16 gleichzeitig aktiviert".to_string(),
            suggestion: "Wähle nur eins: BF16 für bessere Stabilität, FP16 für Geschwindigkeit".to_string(),
        });
        score -= 10.0;
    } else if !config.fp16 && !config.bf16 {
        warnings.push("💡 Mixed Precision (FP16/BF16) deaktiviert → Langsameres Training".to_string());
    } else {
        recommendations.push("✅ Mixed Precision aktiv → Schnelleres Training".to_string());
    }
    
    // 8. ==================== EVALUATION ====================
    if config.eval_strategy == "no" {
        warnings.push("⚠️ Keine Evaluation → Keine Überwachung der Performance".to_string());
        score -= 5.0;
    } else {
        if config.eval_interval == 0 || config.eval_interval > 1000 {
            warnings.push("⚠️ Eval Interval zu groß → Seltenes Feedback".to_string());
            score -= 3.0;
        } else if config.eval_interval <= 100 {
            recommendations.push("✅ Regelmäßige Evaluation konfiguriert".to_string());
        }
    }
    
    // 9. ==================== DROPOUT ====================
    if config.dropout < 0.0 || config.dropout > 1.0 {
        issues.push(ConfigIssue {
            severity: "error".to_string(),
            category: "dropout".to_string(),
            message: "Dropout muss zwischen 0 und 1 liegen".to_string(),
            suggestion: "Empfohlen: 0.1 für leichte Regularisierung".to_string(),
        });
        score -= 10.0;
    } else if config.dropout > 0.5 {
        warnings.push("⚠️ Dropout sehr hoch → Könnte zu viel Information verwerfen".to_string());
        score -= 5.0;
    } else if config.dropout >= 0.1 && config.dropout <= 0.3 {
        recommendations.push("✅ Dropout ist gut eingestellt".to_string());
    }
    
    // 10. ==================== EARLY STOPPING ====================
    if let Some(patience) = config.early_stopping_patience {
        if patience == 0 {
            warnings.push("⚠️ Early Stopping Patience ist 0 → Stoppt sofort".to_string());
            score -= 5.0;
        } else if patience >= 2 && patience <= 5 {
            recommendations.push("✅ Early Stopping gut konfiguriert".to_string());
        }
    } else if config.epochs > 10 {
        warnings.push("💡 Viele Epochs ohne Early Stopping → Könnte Overfitting".to_string());
    }
    
    // 11. ==================== CHECKPOINTING ====================
    if config.save_total_limit == 0 {
        warnings.push("⚠️ Keine Checkpoints werden gespeichert".to_string());
        score -= 5.0;
    } else if config.save_total_limit >= 2 && config.save_total_limit <= 5 {
        recommendations.push("✅ Checkpoint-Strategie ist sinnvoll".to_string());
    }
    
    // 12. ==================== OPTIMIZER ====================
    match config.optimizer.to_lowercase().as_str() {
        "adamw" => recommendations.push("✅ AdamW ist die beste Wahl für Transformers".to_string()),
        "adam" => warnings.push("💡 AdamW ist besser als Adam für Transformers".to_string()),
        "sgd" => {
            warnings.push("⚠️ SGD ist nicht optimal für Transformers → Verwende AdamW".to_string());
            score -= 10.0;
        }
        _ => {}
    }
    
    // 13. ==================== LABEL SMOOTHING ====================
    if config.label_smoothing_factor < 0.0 || config.label_smoothing_factor > 1.0 {
        issues.push(ConfigIssue {
            severity: "error".to_string(),
            category: "label_smoothing".to_string(),
            message: "Label Smoothing muss zwischen 0 und 1 liegen".to_string(),
            suggestion: "Empfohlen: 0.1 für bessere Generalisierung".to_string(),
        });
        score -= 5.0;
    } else if config.label_smoothing_factor > 0.3 {
        warnings.push("⚠️ Label Smoothing sehr hoch → Könnte zu unsicheren Predictions führen".to_string());
    }
    
    // 14. ==================== ZEIT & SPEICHER SCHÄTZUNG ====================
    let steps_per_epoch = if effective_batch > 0 {
        total_samples / effective_batch
    } else {
        0
    };
    
    // Geschätzte Zeit (sehr grob)
    let estimated_seconds_per_step = if config.batch_size <= 8 {
        0.5
    } else if config.batch_size <= 32 {
        1.0
    } else {
        2.0
    };
    
    let total_training_seconds = total_steps as f64 * estimated_seconds_per_step;
    let hours = (total_training_seconds / 3600.0) as u32;
    let minutes = ((total_training_seconds % 3600.0) / 60.0) as u32;
    
    let estimated_time = if hours > 0 {
        format!("~{} Stunden {} Minuten", hours, minutes)
    } else {
        format!("~{} Minuten", minutes)
    };
    
    // Geschätzter Speicher
    let estimated_memory_mb = (config.batch_size * 512) as f64;  // Sehr grobe Schätzung
    let estimated_memory = if estimated_memory_mb > 1024.0 {
        format!("~{:.1} GB", estimated_memory_mb / 1024.0)
    } else {
        format!("~{} MB", estimated_memory_mb as u32)
    };
    
    // 15. ==================== GESAMTBEWERTUNG ====================
    score = score.max(0.0).min(100.0);
    
    let quality_level = if score >= 85.0 {
        "excellent".to_string()
    } else if score >= 70.0 {
        "good".to_string()
    } else if score >= 50.0 {
        "fair".to_string()
    } else {
        "poor".to_string()
    };
    
    let is_valid = !issues.iter().any(|i| i.severity == "error");
    
    // Zusammenfassung
    if score >= 85.0 {
        recommendations.insert(0, "🎯 Exzellente Konfiguration! Training sollte gut verlaufen.".to_string());
    } else if score >= 70.0 {
        recommendations.insert(0, "👍 Gute Konfiguration mit kleinen Verbesserungsmöglichkeiten.".to_string());
    } else if score >= 50.0 {
        warnings.insert(0, "⚠️ Konfiguration funktioniert, aber nicht optimal.".to_string());
    } else {
        warnings.insert(0, "❌ Problematische Konfiguration - Training wird wahrscheinlich suboptimal verlaufen.".to_string());
    }
    
    Ok(ConfigValidation {
        is_valid,
        overall_score: score,
        quality_level,
        warnings,
        recommendations,
        estimated_training_time: Some(estimated_time),
        estimated_memory_usage: Some(estimated_memory),
        issues,
    })
}
