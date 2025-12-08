#!/usr/bin/env python3
"""
Patch Script für train_engine.py
Fügt Training-Logs-Funktionalität hinzu
"""

import re
from pathlib import Path

TRAIN_ENGINE_PATH = Path(__file__).parent / "src-tauri" / "python" / "train_engine.py"

def apply_patch():
    print(f"📝 Patching {TRAIN_ENGINE_PATH}...")
    
    with open(TRAIN_ENGINE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add training_logs and start_time to __init__
    init_patch = '''        self.is_stopped = False
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_logs = []  # NEW: Collect all training logs
        self.start_time = None  # NEW: Track training duration'''
    
    content = content.replace(
        '''        self.is_stopped = False
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')''',
        init_patch
    )
    
    # 2. Add log collection in train_epoch
    logging_patch = '''            # Logging
            if step % self.config.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # NEW: Collect log entry
                log_entry = {
                    "epoch": epoch,
                    "step": step + 1,
                    "train_loss": total_loss / (step + 1),
                    "val_loss": None,  # Will be filled during evaluation
                    "learning_rate": current_lr,
                    "timestamp": datetime.now().isoformat()
                }
                self.training_logs.append(log_entry)
                
                send_progress(
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    step=step + 1,
                    total_steps=num_batches,
                    train_loss=total_loss / (step + 1),
                    learning_rate=current_lr
                )'''
    
    content = content.replace(
        '''            # Logging
            if step % self.config.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                send_progress(
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    step=step + 1,
                    total_steps=num_batches,
                    train_loss=total_loss / (step + 1),
                    learning_rate=current_lr
                )''',
        logging_patch
    )
    
    # 3. Add start_time tracking and val_loss update in train()
    train_start_patch = '''    def train(self):
        """Haupttrainingsschleife."""
        self.start_time = time.time()  # NEW: Track training start
        send_status("starting", "Training wird gestartet...")'''
    
    content = content.replace(
        '''    def train(self):
        """Haupttrainingsschleife."""
        send_status("starting", "Training wird gestartet...")''',
        train_start_patch
    )
    
    # 4. Update val_loss in logs after evaluation
    val_loss_patch = '''            # Validation
            val_loss = self.evaluate()
            val_losses.append(val_loss)
            
            # NEW: Update logs with val_loss for this epoch
            if self.training_logs:
                for log in reversed(self.training_logs):
                    if log["epoch"] == epoch:
                        log["val_loss"] = val_loss'''
    
    content = content.replace(
        '''            # Validation
            val_loss = self.evaluate()
            val_losses.append(val_loss)''',
        val_loss_patch
    )
    
    # 5. Save logs and metrics at the end
    complete_patch = '''            # Training abgeschlossen
            if not self.is_stopped:
                final_path = self.save_final_model()
                
                # NEW: Save training logs
                logs_dir = Path(final_path) / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                logs_file = logs_dir / "training_logs.json"
                
                with open(logs_file, 'w', encoding='utf-8') as f:
                    json.dump(self.training_logs, f, indent=2)
                
                send_status("logs_saved", f"Training logs gespeichert: {logs_file}")
                
                # NEW: Save metrics.json
                training_duration = int(time.time() - self.start_time) if self.start_time else None
                
                metrics_file = Path(final_path) / "metrics.json"
                final_metrics = {
                    "final_train_loss": train_losses[-1] if train_losses else 0,
                    "final_val_loss": val_losses[-1] if val_losses else 0,
                    "best_val_loss": self.best_val_loss,
                    "total_epochs": len(train_losses),
                    "total_steps": self.global_step,
                    "best_epoch": train_losses.index(min(train_losses)) + 1 if train_losses else None,
                    "training_duration_seconds": training_duration
                }
                
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(final_metrics, f, indent=2)
                
                send_status("metrics_saved", f"Training metrics gespeichert: {metrics_file}")
                
                send_complete(final_path, final_metrics)'''
    
    content = content.replace(
        '''            # Training abgeschlossen
            if not self.is_stopped:
                final_path = self.save_final_model()
                
                final_metrics = {
                    "final_train_loss": train_losses[-1] if train_losses else 0,
                    "final_val_loss": val_losses[-1] if val_losses else 0,
                    "best_val_loss": self.best_val_loss,
                    "total_epochs": len(train_losses),
                    "total_steps": self.global_step
                }
                
                send_complete(final_path, final_metrics)''',
        complete_patch
    )
    
    # Write patched file
    with open(TRAIN_ENGINE_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Patch erfolgreich angewendet!")
    print("\nÄnderungen:")
    print("  1. ✅ training_logs Liste zur Log-Sammlung hinzugefügt")
    print("  2. ✅ start_time für Duration-Tracking hinzugefügt")
    print("  3. ✅ Log-Entries werden bei jedem logging_step gesammelt")
    print("  4. ✅ val_loss wird nach Evaluation in Logs eingetragen")
    print("  5. ✅ training_logs.json wird am Ende gespeichert")
    print("  6. ✅ metrics.json wird mit allen Metriken gespeichert")
    print("\n🎉 Training wird jetzt vollständige Logs und Metriken speichern!")

if __name__ == "__main__":
    if not TRAIN_ENGINE_PATH.exists():
        print(f"❌ Fehler: {TRAIN_ENGINE_PATH} nicht gefunden!")
        print(f"   Aktuelles Verzeichnis: {Path.cwd()}")
        print(f"   Erwarteter Pfad: {TRAIN_ENGINE_PATH}")
        exit(1)
    
    apply_patch()
