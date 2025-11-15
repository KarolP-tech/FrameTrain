"""
FrameTrain ML Training Backend
Handles local model training with PyTorch
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch nicht installiert. Bitte installiere: pip install torch torchvision")
    sys.exit(1)

class CustomDataset(Dataset):
    """Custom Dataset für verschiedene Datenformate"""
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.data = self.load_data()
    
    def load_data(self):
        # TODO: Implementiere verschiedene Datenformate (CSV, JSON, etc.)
        # Placeholder
        return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class TrainingManager:
    """Verwaltet den Training-Prozess"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.progress_file = None
        
    def setup(self):
        """Initialisiert Modell, Optimizer, etc."""
        print(f"Device: {self.device}")
        
        # Lade oder erstelle Modell
        # TODO: Implementiere Modell-Loading
        
        # Setup Optimizer
        optimizer_name = self.config['optimizer'].lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate']
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup DataLoader
        dataset = CustomDataset(self.config['dataset_path'])
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # Progress file
        self.progress_file = self.get_progress_file()
        
    def train(self):
        """Führt Training durch"""
        self.model.train()
        
        for epoch in range(self.config['epochs']):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # Berechne Metriken
            avg_loss = epoch_loss / len(self.train_loader)
            accuracy = 100. * correct / total
            
            # Update Progress
            progress = {
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': accuracy,
                'status': 'training',
                'timestamp': datetime.now().isoformat()
            }
            self.save_progress(progress)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Training abgeschlossen
        final_progress = {
            'epoch': self.config['epochs'],
            'loss': avg_loss,
            'accuracy': accuracy,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        self.save_progress(final_progress)
        
        # Speichere finales Modell
        self.save_model()
        
    def save_progress(self, progress):
        """Speichert Trainingsfortschritt"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)
    
    def save_model(self):
        """Speichert trainiertes Modell"""
        model_dir = self.get_model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / 'model.pt'
        torch.save(self.model.state_dict(), model_path)
        
        # Speichere Metadata
        metadata = {
            'name': self.config['model_name'],
            'version': self.get_next_version(),
            'created_at': datetime.now().isoformat(),
            'config': self.config
        }
        
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_progress_file(self):
        """Gibt Pfad zur Progress-Datei zurück"""
        data_dir = self.get_data_dir()
        trainings_dir = data_dir / 'trainings'
        trainings_dir.mkdir(parents=True, exist_ok=True)
        
        training_id = f"{self.config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return trainings_dir / f'{training_id}_progress.json'
    
    def get_model_dir(self):
        """Gibt Pfad zum Modell-Verzeichnis zurück"""
        data_dir = self.get_data_dir()
        return data_dir / 'models' / self.config['model_name']
    
    def get_next_version(self):
        """Gibt nächste Versionsnummer zurück"""
        model_dir = self.get_model_dir()
        if not model_dir.exists():
            return 1
        
        versions = [int(d.name) for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        return max(versions, default=0) + 1
    
    @staticmethod
    def get_data_dir():
        """Gibt platform-spezifisches Daten-Verzeichnis zurück"""
        if sys.platform == 'win32':
            base = Path(os.environ.get('APPDATA', '.'))
        elif sys.platform == 'darwin':
            base = Path.home() / 'Library' / 'Application Support'
        else:
            base = Path.home() / '.local' / 'share'
        
        return base / 'FrameTrain'

def main():
    parser = argparse.ArgumentParser(description='FrameTrain ML Training')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--dataset', required=True, help='Dataset path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', default='adam', help='Optimizer (adam/sgd)')
    
    args = parser.parse_args()
    
    config = {
        'model_name': args.model,
        'dataset_path': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer
    }
    
    try:
        trainer = TrainingManager(config)
        trainer.setup()
        trainer.train()
        print("Training erfolgreich abgeschlossen!")
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
