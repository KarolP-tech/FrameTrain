"""
FrameTrain Model Downloader
Downloads models from HuggingFace Hub
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("ERROR: huggingface_hub nicht installiert. Bitte installiere: pip install huggingface-hub")
    sys.exit(1)

def get_data_dir():
    """Gibt platform-spezifisches Daten-Verzeichnis zurück"""
    import os
    if sys.platform == 'win32':
        base = Path(os.environ.get('APPDATA', '.'))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path.home() / '.local' / 'share'
    
    return base / 'FrameTrain'

def download_model(model_name, cache_dir=None):
    """
    Lädt ein Modell von HuggingFace herunter
    
    Args:
        model_name: Name des Modells (z.B. 'bert-base-uncased')
        cache_dir: Optionales Cache-Verzeichnis
    """
    if cache_dir is None:
        cache_dir = get_data_dir() / 'models' / 'cache'
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Lade Modell '{model_name}' herunter...")
    
    try:
        # Lade komplettes Modell herunter
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            resume_download=True
        )
        
        print(f"Modell erfolgreich heruntergeladen: {model_path}")
        
        # Speichere Metadata
        metadata = {
            'id': model_name.replace('/', '_'),
            'name': model_name,
            'version': 1,
            'status': 'downloaded',
            'created_at': datetime.now().isoformat(),
            'path': model_path
        }
        
        models_dir = get_data_dir() / 'models' / metadata['id']
        models_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = models_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
        
    except Exception as e:
        print(f"ERROR: Download fehlgeschlagen: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='FrameTrain Model Downloader')
    parser.add_argument('--model', required=True, help='Model name from HuggingFace (e.g., bert-base-uncased)')
    parser.add_argument('--cache-dir', help='Optional cache directory')
    
    args = parser.parse_args()
    
    download_model(args.model, args.cache_dir)
    print("Download abgeschlossen!")

if __name__ == '__main__':
    main()
