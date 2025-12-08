-- Fix für bereits erstellte Version ver_f16a0756302a
-- Füge die Metriken manuell hinzu

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
    'metrics_' || hex(randomblob(6)),
    'ver_f16a0756302a',
    0.6898145511530448,
    0.0,
    1,
    138,
    1,
    NULL,
    datetime('now')
);

-- Überprüfe, ob die Metriken eingefügt wurden
SELECT * FROM training_metrics_new WHERE version_id = 'ver_f16a0756302a';
