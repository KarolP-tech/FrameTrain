-- Test-Skript für Analysis Panel
-- Dieses Skript fügt Test-Daten in die Datenbank ein, um die Analysis-Funktionalität zu testen

-- ============================================
-- SCHRITT 1: Test-Daten für Metriken einfügen
-- ============================================

-- Angenommen, du hast bereits eine Version in model_versions_new
-- Ersetze 'DEINE_VERSION_ID' mit einer echten Version-ID aus deiner Datenbank

-- Beispiel: Finde alle Versionen
SELECT id, version_name, model_id FROM model_versions_new LIMIT 5;

-- Füge Test-Metriken für eine Version ein
-- WICHTIG: Ersetze 'ver_xxxxxxxxxxxx' mit einer echten Version-ID
INSERT INTO training_metrics_new (
    id,
    version_id,
    final_train_loss,
    final_val_loss,
    total_epochs,
    total_steps,
    best_epoch,
    training_duration_seconds,
    created_at
) VALUES (
    'test_metrics_' || hex(randomblob(6)),
    'ver_xxxxxxxxxxxx',  -- HIER DEINE VERSION-ID EINFÜGEN
    0.245,
    0.289,
    3,
    150,
    2,
    3600,
    datetime('now')
);

-- Überprüfe, ob die Metriken eingefügt wurden
SELECT * FROM training_metrics_new ORDER BY created_at DESC LIMIT 1;

-- ============================================
-- SCHRITT 2: Training Logs manuell erstellen
-- ============================================

-- Training Logs werden als JSON-Datei im Versions-Ordner gespeichert
-- Finde zuerst den Pfad zur Version:
SELECT id, version_name, path FROM model_versions_new WHERE id = 'ver_xxxxxxxxxxxx';

-- Dann erstelle manuell eine Datei: {path}/logs/training_logs.json
-- Mit folgendem Inhalt:

/*
[
  {
    "epoch": 1,
    "step": 50,
    "train_loss": 0.8,
    "val_loss": 0.85,
    "learning_rate": 0.0001,
    "timestamp": "2025-01-15T10:00:00Z"
  },
  {
    "epoch": 1,
    "step": 100,
    "train_loss": 0.6,
    "val_loss": 0.65,
    "learning_rate": 0.0001,
    "timestamp": "2025-01-15T10:05:00Z"
  },
  {
    "epoch": 2,
    "step": 50,
    "train_loss": 0.4,
    "val_loss": 0.45,
    "learning_rate": 0.00009,
    "timestamp": "2025-01-15T10:10:00Z"
  },
  {
    "epoch": 2,
    "step": 100,
    "train_loss": 0.3,
    "val_loss": 0.35,
    "learning_rate": 0.00008,
    "timestamp": "2025-01-15T10:15:00Z"
  },
  {
    "epoch": 3,
    "step": 50,
    "train_loss": 0.25,
    "val_loss": 0.29,
    "learning_rate": 0.00007,
    "timestamp": "2025-01-15T10:20:00Z"
  },
  {
    "epoch": 3,
    "step": 100,
    "train_loss": 0.245,
    "val_loss": 0.289,
    "learning_rate": 0.00006,
    "timestamp": "2025-01-15T10:25:00Z"
  }
]
*/

-- ============================================
-- SCHRITT 3: Komplettes Test-Setup
-- ============================================

-- Diese Query zeigt alle Informationen für eine Version:
SELECT 
    v.id as version_id,
    v.version_name,
    v.path,
    m.final_train_loss,
    m.final_val_loss,
    m.total_epochs,
    m.total_steps,
    m.training_duration_seconds
FROM model_versions_new v
LEFT JOIN training_metrics_new m ON v.id = m.version_id
ORDER BY v.created_at DESC
LIMIT 10;

-- ============================================
-- SCHRITT 4: Test mit mehreren Versionen
-- ============================================

-- Füge Metriken für mehrere Versionen ein, um verschiedene Szenarien zu testen

-- Szenario 1: Gutes Training (geringe Loss-Verbesserung, kein Overfitting)
INSERT INTO training_metrics_new VALUES (
    'test_good_' || hex(randomblob(6)),
    'ver_good_training',  -- Ersetze mit echter ID
    0.15,    -- final_train_loss
    0.18,    -- final_val_loss (nur 20% höher)
    5,       -- total_epochs
    250,     -- total_steps
    4,       -- best_epoch
    1800,    -- 30 Minuten
    datetime('now')
);

-- Szenario 2: Overfitting (große Differenz zwischen Train und Val)
INSERT INTO training_metrics_new VALUES (
    'test_overfit_' || hex(randomblob(6)),
    'ver_overfitting',  -- Ersetze mit echter ID
    0.12,    -- final_train_loss (sehr niedrig)
    0.45,    -- final_val_loss (viel höher = Overfitting)
    10,      -- total_epochs
    500,     -- total_steps
    3,       -- best_epoch
    3600,    -- 1 Stunde
    datetime('now')
);

-- Szenario 3: Schnelles Training
INSERT INTO training_metrics_new VALUES (
    'test_fast_' || hex(randomblob(6)),
    'ver_fast_training',  -- Ersetze mit echter ID
    0.35,    -- final_train_loss
    0.38,    -- final_val_loss
    2,       -- total_epochs (wenige)
    100,     -- total_steps
    2,       -- best_epoch
    600,     -- 10 Minuten (schnell)
    datetime('now')
);

-- ============================================
-- NÜTZLICHE QUERIES ZUM DEBUGGEN
-- ============================================

-- Alle Metriken anzeigen
SELECT * FROM training_metrics_new;

-- Alle Versionen mit ihren Metriken
SELECT 
    v.version_name,
    v.created_at as version_created,
    m.final_train_loss,
    m.total_epochs,
    m.training_duration_seconds / 60.0 as duration_minutes
FROM model_versions_new v
LEFT JOIN training_metrics_new m ON v.id = m.version_id
ORDER BY v.created_at DESC;

-- Metriken löschen (falls du neu starten willst)
-- DELETE FROM training_metrics_new WHERE id LIKE 'test_%';

-- Alle Tabellen-Schemas anzeigen
SELECT sql FROM sqlite_master WHERE type='table' AND name LIKE '%metrics%';

-- ============================================
-- ERWARTETE ERGEBNISSE IM FRONTEND
-- ============================================

/*
Wenn die Test-Daten korrekt eingefügt wurden, sollte das Analysis Panel zeigen:

1. Übersichts-Karten:
   - Status: Abgeschlossen
   - Final Loss: Der Wert aus final_train_loss
   - Epochen: Der Wert aus total_epochs
   - Dauer: Formatierte Zeit aus training_duration_seconds

2. Loss-Graphen:
   - Werden aus training_logs.json generiert
   - Zeigen Train/Val Loss über Epochen und Steps

3. Performance-Metriken:
   - Loss Verbesserung in %
   - Konvergenz-Geschwindigkeit
   - Overfitting-Indikator (Val-Train Gap)
   - Training-Geschwindigkeit (Steps/Sekunde)

4. Empfehlungen:
   - Automatische Analyse basierend auf den Metriken
   - Warnung bei Overfitting (Val Loss >> Train Loss)
   - Info bei Loss-Plateau
   - Bestätigung bei gutem Training
*/
