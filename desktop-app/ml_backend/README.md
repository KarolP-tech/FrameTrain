# FrameTrain ML Backend

Python-Backend für lokales Machine Learning Training in der FrameTrain Desktop-App.

## Installation

```bash
pip install torch torchvision huggingface-hub transformers
```

## Skripte

### train.py
Führt lokales Training mit PyTorch durch.

**Verwendung:**
```bash
python train.py --model bert-base --dataset ./data/train.csv --epochs 10 --batch-size 32 --learning-rate 0.001 --optimizer adam
```

**Parameter:**
- `--model`: Name des Modells
- `--dataset`: Pfad zum Trainingsdatensatz
- `--epochs`: Anzahl der Trainings-Epochen
- `--batch-size`: Batch-Größe
- `--learning-rate`: Lernrate
- `--optimizer`: Optimizer (adam/sgd)

### download_model.py
Lädt Modelle von HuggingFace herunter.

**Verwendung:**
```bash
python download_model.py --model bert-base-uncased
```

**Parameter:**
- `--model`: HuggingFace Model-Name (z.B. `bert-base-uncased`, `gpt2`)
- `--cache-dir`: Optional, Cache-Verzeichnis

## Datenstruktur

```
~/Library/Application Support/FrameTrain/  (macOS)
%APPDATA%/FrameTrain/                       (Windows)
~/.local/share/frametrain/                  (Linux)
├── models/
│   ├── cache/              # HuggingFace Model Cache
│   ├── bert-base/
│   │   ├── metadata.json
│   │   ├── model.pt
│   │   └── versions/
│   └── gpt2/
└── trainings/
    ├── training_123_progress.json
    └── training_456_progress.json
```

## Progress Format

Training-Fortschritt wird als JSON gespeichert:

```json
{
  "epoch": 5,
  "loss": 0.234,
  "accuracy": 92.5,
  "status": "training",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Status-Werte

- `training`: Training läuft
- `completed`: Training abgeschlossen
- `failed`: Training fehlgeschlagen
- `stopped`: Training gestoppt

## Anforderungen

- Python 3.8+
- PyTorch 2.0+
- HuggingFace Hub
- Transformers (optional)

## Support

Bei Problemen siehe Hauptdokumentation: https://docs.frametrain.ai
