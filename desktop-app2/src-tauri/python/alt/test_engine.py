"""
FrameTrain - Model Testing Engine
==================================
Dieses Modul testet trainierte Modelle auf Test-Datensätzen.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback

# PyTorch imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Transformers
try:
    from transformers import (
        AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ============ Message Protocol ============

def send_message(msg_type: str, data: Dict[str, Any]):
    """Sendet eine JSON-Nachricht an das Rust-Backend."""
    message = {
        "type": msg_type,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    print(json.dumps(message), flush=True)

def send_progress(current: int, total: int, samples_per_second: float = 0.0):
    """Sendet Test-Progress Update."""
    # Calculate estimated time remaining
    remaining_samples = total - current
    estimated_time_remaining = None
    if samples_per_second > 0:
        estimated_time_remaining = remaining_samples / samples_per_second
    
    send_message("progress", {
        "current_sample": current,
        "total_samples": total,
        "progress_percent": (current / total) * 100 if total > 0 else 0,
        "samples_per_second": samples_per_second,
        "estimated_time_remaining": estimated_time_remaining,
    })

def send_status(status: str, message: str = ""):
    """Sendet Status-Update."""
    send_message("status", {
        "status": status,
        "message": message
    })

def send_error(error: str, details: str = ""):
    """Sendet Fehler-Nachricht."""
    send_message("error", {
        "error": error,
        "details": details
    })

def send_complete(results: Dict[str, Any]):
    """Sendet Completion-Nachricht."""
    send_message("complete", results)

# ============ Test Engine ============

class TestEngine:
    """Engine für Modell-Testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.model_path = config.get("model_path")
        self.dataset_path = config.get("dataset_path")
        self.output_path = config.get("output_path")
        self.batch_size = config.get("batch_size", 8)
        self.max_samples = config.get("max_samples")
        
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.model_class = None
        
    def _get_device(self) -> torch.device:
        """Ermittelt das beste verfügbare Device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self):
        """Lädt das Modell."""
        send_status("loading", f"Lade Modell von {self.model_path}...")
        
        try:
            # Load config to determine model type
            model_config = AutoConfig.from_pretrained(self.model_path)
            model_type = model_config.model_type
            architectures = getattr(model_config, 'architectures', [])
            
            # Determine model class
            if architectures:
                arch_name = architectures[0]
                if any(x in arch_name for x in ['T5', 'MT5', 'Bart', 'Pegasus', 'Marian']):
                    self.model_class = 'seq2seq'
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
                elif 'ForSequenceClassification' in arch_name:
                    self.model_class = 'seq_classification'
                    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                else:
                    self.model_class = 'causal_lm'
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            else:
                # Fallback
                self.model_class = 'causal_lm'
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            send_status("loaded", f"Modell geladen ({self.model_class})")
            
        except Exception as e:
            send_error("Model loading failed", str(e))
            raise
    
    def load_test_data(self) -> List[Dict[str, str]]:
        """Lädt Test-Daten."""
        send_status("loading", "Lade Test-Daten...")
        
        test_path = Path(self.dataset_path) / "test"
        if not test_path.exists():
            test_path = Path(self.dataset_path) / "val"
        if not test_path.exists():
            raise ValueError(f"Test data not found: {self.dataset_path}")
        
        test_data = []
        
        # Load files
        for file_path in test_path.glob("*"):
            if file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            test_data.append({"text": line, "expected": None})
            
            elif file_path.suffix == '.csv':
                import csv
                with open(file_path, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        if len(row) >= 2:
                            test_data.append({
                                "text": row[0].strip(),
                                "expected": row[1].strip()
                            })
                        elif len(row) == 1:
                            test_data.append({
                                "text": row[0].strip(),
                                "expected": None
                            })
            
            elif file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            # Extract text and expected fields
                            text = data.get('input') or data.get('text') or data.get('prompt') or ''
                            expected = data.get('target') or data.get('output') or data.get('label')
                            test_data.append({"text": text, "expected": expected})
                        except:
                            pass
            
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                text = item.get('input') or item.get('text') or item.get('prompt') or ''
                                expected = item.get('target') or item.get('output') or item.get('label')
                                test_data.append({"text": text, "expected": expected})
                        else:
                            text = data.get('input') or data.get('text') or data.get('prompt') or ''
                            expected = data.get('target') or data.get('output') or data.get('label')
                            test_data.append({"text": text, "expected": expected})
                    except:
                        pass
        
        if not test_data:
            raise ValueError("No test data found")
        
        # Limit samples if specified
        if self.max_samples and len(test_data) > self.max_samples:
            test_data = test_data[:self.max_samples]
        
        send_status("loaded", f"{len(test_data)} Test-Samples geladen")
        return test_data
    
    @torch.no_grad()
    def test_sample(self, text: str, expected: Optional[str] = None) -> Dict[str, Any]:
        """Testet ein einzelnes Sample."""
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Generate prediction
        if self.model_class == 'seq2seq':
            # Seq2Seq models
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate loss if expected output is available
            loss = None
            if expected:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        expected,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).input_ids.to(self.device)
                
                outputs_with_loss = self.model(**inputs, labels=labels)
                loss = outputs_with_loss.loss.item()
        
        elif self.model_class == 'seq_classification':
            # Classification models
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=-1).item()
            predicted_text = str(predicted_label)
            
            # Calculate loss if expected label available
            loss = None
            if expected is not None:
                try:
                    expected_label = int(expected)
                    labels = torch.tensor([expected_label]).to(self.device)
                    outputs_with_loss = self.model(**inputs, labels=labels)
                    loss = outputs_with_loss.loss.item()
                except:
                    pass
        
        else:
            # Causal LM models
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs.input_ids[0]) + 100,
                num_beams=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input from output
            if predicted_text.startswith(text):
                predicted_text = predicted_text[len(text):].strip()
            
            # Calculate loss if expected available
            loss = None
            if expected:
                full_text = text + " " + expected
                full_inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs_with_loss = self.model(**full_inputs, labels=full_inputs.input_ids)
                loss = outputs_with_loss.loss.item()
        
        inference_time = time.time() - start_time
        
        # Check if correct
        is_correct = False
        if expected:
            # Normalize for comparison
            pred_norm = predicted_text.strip().lower()
            exp_norm = expected.strip().lower()
            is_correct = pred_norm == exp_norm
        
        return {
            "predicted_output": predicted_text,
            "is_correct": is_correct,
            "loss": loss,
            "inference_time": inference_time
        }
    
    def run_test(self):
        """Führt den kompletten Test aus."""
        send_status("starting", "Test wird gestartet...")
        
        try:
            self.load_model()
            test_data = self.load_test_data()
            
            send_status("testing", "Testing läuft...")
            
            predictions = []
            correct_count = 0
            total_loss = 0.0
            loss_count = 0
            total_inference_time = 0.0
            
            start_time = time.time()
            
            for i, sample in enumerate(test_data):
                text = sample["text"]
                expected = sample["expected"]
                
                # Test sample
                result = self.test_sample(text, expected)
                
                # Create prediction result
                prediction = {
                    "sample_id": i,
                    "input_text": text,
                    "expected_output": expected,
                    "predicted_output": result["predicted_output"],
                    "is_correct": result["is_correct"],
                    "loss": result["loss"],
                    "confidence": None,  # Could calculate from logits
                    "inference_time": result["inference_time"],
                    "error_type": None if result["is_correct"] or not expected else "mismatch"
                }
                
                predictions.append(prediction)
                
                if result["is_correct"]:
                    correct_count += 1
                
                if result["loss"] is not None:
                    total_loss += result["loss"]
                    loss_count += 1
                
                total_inference_time += result["inference_time"]
                
                # Progress update
                elapsed = time.time() - start_time
                samples_per_second = (i + 1) / elapsed if elapsed > 0 else 0
                send_progress(i + 1, len(test_data), samples_per_second)
            
            # Calculate final metrics
            accuracy = (correct_count / len(test_data)) * 100 if test_data else 0
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            avg_inference_time = total_inference_time / len(test_data) if test_data else 0
            
            results = {
                "total_samples": len(test_data),
                "correct_predictions": correct_count,
                "incorrect_predictions": len(test_data) - correct_count,
                "accuracy": accuracy,
                "average_loss": avg_loss,
                "average_inference_time": avg_inference_time,
                "predictions": predictions,
                "metrics": {
                    "samples_per_second": len(test_data) / (time.time() - start_time),
                    "total_time": time.time() - start_time
                }
            }
            
            # Save results
            output_path = Path(self.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / "test_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            send_status("complete", f"Test abgeschlossen: {accuracy:.2f}% Accuracy")
            send_complete(results)
            
        except Exception as e:
            send_error("Test failed", traceback.format_exc())
            raise

# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description="FrameTrain Test Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    
    args = parser.parse_args()
    
    # Check requirements
    if not TORCH_AVAILABLE:
        send_error("PyTorch not installed", "Please install PyTorch")
        sys.exit(1)
    
    if not TRANSFORMERS_AVAILABLE:
        send_error("Transformers not installed", "Please install transformers")
        sys.exit(1)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Run test
    engine = TestEngine(config)
    engine.run_test()

if __name__ == "__main__":
    main()
