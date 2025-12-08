"""
FrameTrain - PyTorch Training Engine
=====================================
Dieses Modul führt das eigentliche ML-Training durch.
Es kommuniziert via stdout JSON-Messages mit dem Rust-Backend.
"""

import os
import sys
import json
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import traceback

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch.optim.lr_scheduler import (
        StepLR, ExponentialLR, CosineAnnealingLR, 
        ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts,
        LinearLR, PolynomialLR
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Transformers für LLM Fine-tuning
try:
    from transformers import (
        AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoModelForQuestionAnswering,
        AutoTokenizer, AutoConfig,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
        DataCollatorWithPadding
    )
    from datasets import load_dataset, Dataset as HFDataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# PEFT für LoRA/QLoRA
try:
    from peft import (
        get_peft_model, LoraConfig, TaskType,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# BitsAndBytes für Quantisierung
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


# ============ Message Protocol ============

def send_message(msg_type: str, data: Dict[str, Any]):
    """Sendet eine JSON-Nachricht an das Rust-Backend via stdout."""
    message = {
        "type": msg_type,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    print(json.dumps(message), flush=True)


def send_progress(epoch: int, total_epochs: int, step: int, total_steps: int,
                  train_loss: float, val_loss: Optional[float] = None,
                  learning_rate: float = 0.0, metrics: Dict[str, float] = None):
    """Sendet Training-Progress Update."""
    send_message("progress", {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": learning_rate,
        "metrics": metrics or {},
        "progress_percent": ((epoch - 1) * total_steps + step) / (total_epochs * total_steps) * 100
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


def send_complete(model_path: str, metrics: Dict[str, Any]):
    """Sendet Completion-Nachricht."""
    send_message("complete", {
        "model_path": model_path,
        "output_path": model_path,  # Add output_path for version creation
        "final_metrics": metrics
    })


def send_checkpoint(checkpoint_path: str, epoch: int, metrics: Dict[str, float]):
    """Sendet Checkpoint-Saved Nachricht."""
    send_message("checkpoint", {
        "path": checkpoint_path,
        "epoch": epoch,
        "metrics": metrics
    })


# ============ Training Configuration ============

class TrainingConfig:
    """Zentrale Konfigurationsklasse für alle Training-Parameter."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        # Pfade
        self.model_path = config_dict.get("model_path", "")
        self.dataset_path = config_dict.get("dataset_path", "")
        self.output_path = config_dict.get("output_path", "")
        self.checkpoint_dir = config_dict.get("checkpoint_dir", "")
        
        # Training Basics
        self.epochs = config_dict.get("epochs", 3)
        self.batch_size = config_dict.get("batch_size", 8)
        self.gradient_accumulation_steps = config_dict.get("gradient_accumulation_steps", 1)
        self.max_steps = config_dict.get("max_steps", -1)  # -1 = use epochs
        
        # Learning Rate
        self.learning_rate = config_dict.get("learning_rate", 5e-5)
        self.weight_decay = config_dict.get("weight_decay", 0.01)
        self.warmup_steps = config_dict.get("warmup_steps", 0)
        self.warmup_ratio = config_dict.get("warmup_ratio", 0.0)
        
        # Optimizer
        self.optimizer = config_dict.get("optimizer", "adamw")  # adam, adamw, sgd, adagrad, rmsprop
        self.adam_beta1 = config_dict.get("adam_beta1", 0.9)
        self.adam_beta2 = config_dict.get("adam_beta2", 0.999)
        self.adam_epsilon = config_dict.get("adam_epsilon", 1e-8)
        self.sgd_momentum = config_dict.get("sgd_momentum", 0.9)
        
        # Scheduler
        self.scheduler = config_dict.get("scheduler", "linear")  # linear, cosine, constant, polynomial, etc.
        self.scheduler_step_size = config_dict.get("scheduler_step_size", 1)
        self.scheduler_gamma = config_dict.get("scheduler_gamma", 0.1)
        self.cosine_min_lr = config_dict.get("cosine_min_lr", 0.0)
        
        # Regularization
        self.dropout = config_dict.get("dropout", 0.1)
        self.max_grad_norm = config_dict.get("max_grad_norm", 1.0)
        self.label_smoothing = config_dict.get("label_smoothing", 0.0)
        
        # Mixed Precision
        self.fp16 = config_dict.get("fp16", False)
        self.bf16 = config_dict.get("bf16", False)
        
        # LoRA / PEFT
        self.use_lora = config_dict.get("use_lora", False)
        self.lora_r = config_dict.get("lora_r", 8)
        self.lora_alpha = config_dict.get("lora_alpha", 32)
        self.lora_dropout = config_dict.get("lora_dropout", 0.1)
        self.lora_target_modules = config_dict.get("lora_target_modules", ["q_proj", "v_proj"])
        
        # Quantization
        self.load_in_8bit = config_dict.get("load_in_8bit", False)
        self.load_in_4bit = config_dict.get("load_in_4bit", False)
        
        # Data
        self.max_seq_length = config_dict.get("max_seq_length", 512)
        self.num_workers = config_dict.get("num_workers", 4)
        self.pin_memory = config_dict.get("pin_memory", True)
        
        # Evaluation
        self.eval_steps = config_dict.get("eval_steps", 500)
        self.eval_strategy = config_dict.get("eval_strategy", "steps")  # steps, epoch
        self.save_steps = config_dict.get("save_steps", 500)
        self.save_strategy = config_dict.get("save_strategy", "steps")  # steps, epoch
        self.save_total_limit = config_dict.get("save_total_limit", 3)
        
        # Logging
        self.logging_steps = config_dict.get("logging_steps", 10)  # Häufigere Updates (alle 10 steps statt 100)
        
        # Advanced
        self.seed = config_dict.get("seed", 42)
        self.dataloader_drop_last = config_dict.get("dataloader_drop_last", False)
        self.group_by_length = config_dict.get("group_by_length", False)
        
        # Training Type
        self.training_type = config_dict.get("training_type", "fine_tuning")  # fine_tuning, lora, qlora, full
        self.task_type = config_dict.get("task_type", "causal_lm")  # causal_lm, seq_classification, token_classification, etc.
        
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert Config zurück zu Dict."""
        return vars(self)


# ============ Optimizer Factory ============

def create_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """Erstellt den Optimizer basierend auf der Konfiguration."""
    
    # Parameter die trainiert werden sollen
    params = [p for p in model.parameters() if p.requires_grad]
    
    if config.optimizer == "adam":
        return optim.Adam(
            params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        return optim.AdamW(
            params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "sgd":
        return optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "adagrad":
        return optim.Adagrad(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "rmsprop":
        return optim.RMSprop(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "adadelta":
        return optim.Adadelta(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


# ============ Scheduler Factory ============

def create_scheduler(optimizer: optim.Optimizer, config: TrainingConfig, 
                     num_training_steps: int) -> Optional[object]:
    """Erstellt den Learning Rate Scheduler."""
    
    num_warmup_steps = config.warmup_steps
    if config.warmup_ratio > 0:
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    if config.scheduler == "constant":
        return None
    elif config.scheduler == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=num_training_steps
        )
    elif config.scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=config.cosine_min_lr
        )
    elif config.scheduler == "cosine_warmup":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_training_steps // 4,
            eta_min=config.cosine_min_lr
        )
    elif config.scheduler == "step":
        return StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
    elif config.scheduler == "exponential":
        return ExponentialLR(
            optimizer,
            gamma=config.scheduler_gamma
        )
    elif config.scheduler == "polynomial":
        return PolynomialLR(
            optimizer,
            total_iters=num_training_steps,
            power=2.0
        )
    elif config.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_gamma,
            patience=5
        )
    elif config.scheduler == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=num_training_steps
        )
    else:
        return None


# ============ Generic Dataset ============

class GenericFileDataset(Dataset):
    """Generisches Dataset für Dateien aus einem Ordner."""
    
    def __init__(self, data_dir: str, transform: Callable = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.files = list(self.data_dir.glob("*"))
        self.files = [f for f in self.files if f.is_file()]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # Lade basierend auf Dateiendung
        if file_path.suffix in ['.json', '.jsonl']:
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.suffix == '.txt':
            with open(file_path, 'r') as f:
                data = {"text": f.read()}
        else:
            # Für Bilder etc. - später erweitern
            data = {"path": str(file_path)}
        
        if self.transform:
            data = self.transform(data)
            
        return data


# ============ Training Engine ============

class TrainingEngine:
    """Haupt-Training-Engine für alle Modelltypen."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.model_class = None  # Speichert den Modelltyp
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.is_stopped = False
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_logs = []  # NEW: Collect all training logs
        self.start_time = None  # NEW: Track training duration
        
        # Signal Handler für graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler für Stop-Signale."""
        send_status("stopping", "Training wird gestoppt...")
        self.is_stopped = True
        
    def _get_device(self) -> torch.device:
        """Ermittelt das beste verfügbare Device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            send_status("device", f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            send_status("device", "Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device("cpu")
            send_status("device", "Using CPU")
        return device
    
    def _forward(self, batch):
        """Führt Forward-Pass durch und berechnet Loss."""
        # Handle different batch formats
        if not isinstance(batch, dict):
            # If batch is not a dict, it might be a tuple or tensor
            # This shouldn't happen with our dataloader, but let's be safe
            raise ValueError(f"Unexpected batch type: {type(batch)}. Expected dict with 'input_ids' and 'labels'.")
        
        # Check if batch has required keys
        if 'input_ids' not in batch:
            raise KeyError(f"Batch missing 'input_ids'. Available keys: {list(batch.keys())}")
        if 'labels' not in batch:
            raise KeyError(f"Batch missing 'labels'. Available keys: {list(batch.keys())}")
        
        # Move to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss
    

    def _make_tensors_contiguous(self):
        """Makes all model tensors contiguous to fix safetensors saving issues."""
        for name, param in self.model.named_parameters():
            if param.data is not None and not param.is_contiguous():
                param.data = param.data.contiguous()
        
        for name, buffer in self.model.named_buffers():
            if buffer.data is not None and not buffer.is_contiguous():
                buffer.data = buffer.data.contiguous()
    
    def load_model(self):
        """Lädt das Modell basierend auf Konfiguration."""
        send_status("loading", "Lade Modell...")
        
        model_path = self.config.model_path
        
        try:
            # Versuche als Transformers Modell zu laden
            if TRANSFORMERS_AVAILABLE:
                quantization_config = None
                
                # BitsAndBytes ist NUR auf CUDA verfügbar, nicht auf MPS oder CPU
                use_quantization = BITSANDBYTES_AVAILABLE and torch.cuda.is_available()
                
                if self.config.load_in_4bit and use_quantization:
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        send_status("loading", "Verwende 4-bit Quantisierung")
                    except Exception as e:
                        send_status("loading", f"4-bit Quantisierung fehlgeschlagen, nutze FP16: {e}")
                        quantization_config = None
                        self.config.load_in_4bit = False
                        self.config.fp16 = True
                elif self.config.load_in_8bit and use_quantization:
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                        send_status("loading", "Verwende 8-bit Quantisierung")
                    except Exception as e:
                        send_status("loading", f"8-bit Quantisierung fehlgeschlagen, nutze FP16: {e}")
                        quantization_config = None
                        self.config.load_in_8bit = False
                        self.config.fp16 = True
                elif (self.config.load_in_4bit or self.config.load_in_8bit) and not torch.cuda.is_available():
                    send_status("loading", "Quantisierung nur auf CUDA verfügbar. Nutze FP16 auf MPS/CPU.")
                    self.config.load_in_4bit = False
                    self.config.load_in_8bit = False
                    self.config.fp16 = True
                
                # Lade Config zuerst um Modelltyp zu erkennen
                model_config = AutoConfig.from_pretrained(model_path)
                model_type = model_config.model_type
                architectures = getattr(model_config, 'architectures', [])
                original_name = getattr(model_config, '_name_or_path', model_path)
                
                send_status("loading", f"Erkannter Modelltyp: {model_type}")
                
                # Bestimme den richtigen Auto-Klasse basierend auf Architektur und Config
                model_class = None
                
                # Prüfe zuerst die Architectures (am zuverlässigsten)
                if architectures:
                    arch_name = architectures[0]
                    
                    # Seq2Seq Modelle (Encoder-Decoder)
                    if any(x in arch_name for x in ['T5', 'MT5', 'Bart', 'Pegasus', 'Marian', 'MBart', 'Blenderbot']):
                        model_class = 'seq2seq'
                        send_status("loading", "Erkannt als Seq2Seq Modell (Encoder-Decoder)")
                    
                    # Causal LM Modelle
                    elif any(x in arch_name for x in ['GPT', 'Llama', 'Mistral', 'Falcon', 'Bloom', 'OPT', 'CodeGen']):
                        model_class = 'causal_lm'
                        send_status("loading", "Erkannt als Causal Language Model")
                    
                    # Masked LM Modelle
                    elif any(x in arch_name for x in ['Bert', 'RoBERTa', 'Albert', 'Electra', 'DistilBert']):
                        model_class = 'masked_lm'
                        send_status("loading", "Erkannt als Masked Language Model")
                    
                    # Classification Modelle
                    elif 'ForSequenceClassification' in arch_name:
                        model_class = 'seq_classification'
                        send_status("loading", "Erkannt als Sequence Classification Model")
                    
                    elif 'ForTokenClassification' in arch_name:
                        model_class = 'token_classification'
                        send_status("loading", "Erkannt als Token Classification Model")
                
                # Fallback: Prüfe model_type
                if not model_class:
                    if model_type in ['t5', 'mt5', 'bart', 'pegasus', 'marian', 'mbart', 'blenderbot', 'blenderbot-small']:
                        model_class = 'seq2seq'
                    elif model_type in ['gpt2', 'gpt_neo', 'gpt_neox', 'gptj', 'llama', 'mistral', 'falcon', 'bloom', 'opt', 'codegen']:
                        model_class = 'causal_lm'
                    elif model_type in ['bert', 'roberta', 'albert', 'electra', 'distilbert']:
                        model_class = 'masked_lm'
                    else:
                        # Default: Versuche Causal LM
                        model_class = 'causal_lm'
                        send_status("loading", f"Unbekannter Typ {model_type}, versuche Causal LM")
                
                # Task Type aus Config überschreiben wenn explizit gesetzt
                if self.config.task_type != 'causal_lm':
                    if self.config.task_type == 'seq_classification':
                        model_class = 'seq_classification'
                    elif self.config.task_type == 'token_classification':
                        model_class = 'token_classification'
                    elif self.config.task_type == 'seq2seq':
                        model_class = 'seq2seq'
                
                # Bestimme dtype basierend auf Device
                # CRITICAL FIX: MPS hat Probleme mit float16 bei bestimmten Operationen
                if self.device.type == 'mps':
                    torch_dtype = torch.float32
                    send_status("loading", "Nutze FP32 für MPS (Apple Silicon) für Stabilität")
                    self.config.fp16 = False
                    self.config.bf16 = False
                elif self.config.fp16:
                    torch_dtype = torch.float16
                elif self.config.bf16:
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float32
                
                # Lade Modell mit der richtigen Klasse
                try:
                    if model_class == 'seq2seq':
                        from transformers import AutoModelForSeq2SeqLM
                        self.model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            device_map="auto" if quantization_config else None,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True
                        )
                    elif model_class == 'causal_lm':
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            device_map="auto" if quantization_config else None,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True
                        )
                    elif model_class == 'masked_lm':
                        from transformers import AutoModelForMaskedLM
                        self.model = AutoModelForMaskedLM.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            device_map="auto" if quantization_config else None,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True
                        )
                    elif model_class == 'seq_classification':
                        self.model = AutoModelForSequenceClassification.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            device_map="auto" if quantization_config else None,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True
                        )
                    elif model_class == 'token_classification':
                        self.model = AutoModelForTokenClassification.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            device_map="auto" if quantization_config else None,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True
                        )
                    else:
                        # Ultimate Fallback: Lade als Auto ohne spezifischen Task
                        self.model = AutoModel.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            device_map="auto" if quantization_config else None,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True
                        )
                    
                    self.model_class = model_class
                    
                except Exception as e:
                    # Wenn spezifische Klasse fehlschlägt, versuche AutoModel
                    send_status("loading", f"Spezifische Klasse fehlgeschlagen, versuche AutoModel: {e}")
                    self.model = AutoModel.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        device_map="auto" if quantization_config else None,
                        torch_dtype=torch.float16 if self.config.fp16 else torch.float32
                    )
                    self.model_class = 'auto'
                
                # LoRA anwenden falls aktiviert
                if self.config.use_lora and PEFT_AVAILABLE:
                    send_status("loading", "Konfiguriere LoRA...")
                    
                    if quantization_config:
                        try:
                            self.model = prepare_model_for_kbit_training(self.model)
                        except Exception as e:
                            send_status("loading", f"kbit_training prep fehlgeschlagen, fahre trotzdem fort: {e}")
                    
                    # Bestimme Task Type für PEFT
                    if model_class == 'seq2seq':
                        peft_task_type = TaskType.SEQ_2_SEQ_LM
                    elif model_class == 'seq_classification':
                        peft_task_type = TaskType.SEQ_CLS
                    elif model_class == 'token_classification':
                        peft_task_type = TaskType.TOKEN_CLS
                    else:
                        peft_task_type = TaskType.CAUSAL_LM
                    
                    peft_config = LoraConfig(
                        r=self.config.lora_r,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        target_modules=self.config.lora_target_modules,
                        bias="none",
                        task_type=peft_task_type
                    )
                    
                    self.model = get_peft_model(self.model, peft_config)
                    self.model.print_trainable_parameters()
                
                # Tokenizer laden
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                except Exception as tokenizer_error:
                    # Fallback: Versuche mit use_fast=False
                    send_status("loading", f"Fast tokenizer fehlgeschlagen, versuche slow tokenizer")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    except Exception as slow_error:
                        # Letzter Versuch: Lade von Original HuggingFace Hub
                        send_status("loading", f"Lade Tokenizer von Original-Quelle: {original_name}")
                        self.tokenizer = AutoTokenizer.from_pretrained(original_name, use_fast=False)
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            else:
                # Fallback: Lade als PyTorch State Dict
                self.model = torch.load(model_path, map_location=self.device)
                self.model_class = 'pytorch'
            
            if not quantization_config:
                self.model = self.model.to(self.device)
            
            # MEMORY FIX: Enable gradient checkpointing on MPS
            if self.device.type == 'mps' and hasattr(self.model, 'gradient_checkpointing_enable'):
                try:
                    self.model.gradient_checkpointing_enable()
                    send_status("loaded", "Gradient Checkpointing aktiviert (spart 40-50% Speicher)")
                except:
                    pass
                
            send_status("loaded", f"Modell geladen: {model_path}")
            
        except Exception as e:
            send_error("Model loading failed", str(e))
            raise
    
    def load_data(self):
        """Lädt die Trainingsdaten."""
        send_status("loading", "Lade Datensätze...")
        
        dataset_path = Path(self.config.dataset_path)
        train_path = dataset_path / "train"
        val_path = dataset_path / "val"
        
        if not train_path.exists():
            raise ValueError(f"Training data not found: {train_path}")
        
        try:
            if TRANSFORMERS_AVAILABLE and hasattr(self, 'tokenizer'):
                # Für Text-Daten mit Tokenizer
                train_files = list(train_path.glob("*.json")) + list(train_path.glob("*.jsonl")) + list(train_path.glob("*.txt")) + list(train_path.glob("*.csv"))
                
                # DEBUG: Zeige alle Dateien im Verzeichnis
                all_files = list(train_path.glob("*"))
                send_status("loading", f"Alle Dateien in {train_path}: {[f.name for f in all_files]}")
                send_status("loading", f"Gefundene Trainingsdateien (.txt/.json/.jsonl/.csv): {[f.name for f in train_files]}")
                
                # CRITICAL: Log wie viele samples pro Datei
                send_status("loading", f"Anzahl gefundener Dateien: {len(train_files)}")
                
                if train_files:
                    # Lade als HuggingFace Dataset
                    train_data = []
                    for f in train_files:
                        if f.suffix == '.txt':
                            with open(f, 'r', encoding='utf-8') as file:
                                for line in file:
                                    line = line.strip()
                                    if line:
                                        train_data.append({"text": line})
                        elif f.suffix == '.csv':
                            # CSV Support für noisy,clean Paare
                            import csv
                            with open(f, 'r', encoding='utf-8') as file:
                                csv_reader = csv.reader(file)
                                for row in csv_reader:
                                    if len(row) >= 2:
                                        # Für Seq2Seq: input -> target
                                        train_data.append({
                                            "input": row[0].strip(),
                                            "target": row[1].strip()
                                        })
                                    elif len(row) == 1:
                                        # Fallback: nur Text
                                        train_data.append({"text": row[0].strip()})
                        elif f.suffix == '.jsonl':
                            with open(f, 'r', encoding='utf-8') as file:
                                for line in file:
                                    try:
                                        data = json.loads(line)
                                        train_data.append(data)
                                    except:
                                        pass
                        else:  # .json
                            with open(f, 'r', encoding='utf-8') as file:
                                try:
                                    data = json.load(file)
                                    if isinstance(data, list):
                                        train_data.extend(data)
                                    else:
                                        train_data.append(data)
                                except:
                                    pass
                    
                    if not train_data:
                        raise ValueError("Keine gültigen Daten gefunden")
                    
                    # CRITICAL: Log total samples geladen
                    send_status("loading", f"Total {len(train_data)} Trainingssamples aus {len(train_files)} Dateien geladen")
                    
                    # Prüfe welche Felder vorhanden sind
                    sample = train_data[0]
                    available_fields = list(sample.keys())
                    send_status("loading", f"Gefundene Felder: {available_fields}")
                    
                    # Bestimme Text-Feld und Target-Feld (für Seq2Seq)
                    text_field = None
                    target_field = None
                    
                    # Suche Input-Feld
                    if 'input' in available_fields:
                        text_field = 'input'
                    elif 'text' in available_fields:
                        text_field = 'text'
                    elif 'content' in available_fields:
                        text_field = 'content'
                    elif 'sentence' in available_fields:
                        text_field = 'sentence'
                    elif 'prompt' in available_fields:
                        text_field = 'prompt'
                    else:
                        # Nimm das erste String-Feld
                        for field in available_fields:
                            if isinstance(sample.get(field), str):
                                text_field = field
                                break
                    
                    # Suche Target-Feld (für Seq2Seq)
                    if 'target' in available_fields:
                        target_field = 'target'
                    elif 'output' in available_fields:
                        target_field = 'output'
                    elif 'label' in available_fields:
                        target_field = 'label'
                    elif 'clean' in available_fields:
                        target_field = 'clean'
                    
                    if not text_field:
                        raise ValueError(f"Kein Text-Feld gefunden in: {available_fields}")
                    
                    if target_field:
                        send_status("loading", f"Nutze '{text_field}' -> '{target_field}' für Seq2Seq Training")
                    else:
                        send_status("loading", f"Nutze Feld '{text_field}' für Training")
                    
                    train_dataset = HFDataset.from_list(train_data)
                    
                    # Tokenize - unterschiedlich für verschiedene Modelltypen
                    def tokenize_function(examples):
                        try:
                            # Extrahiere Text aus dem identifizierten Feld
                            texts = examples.get(text_field)
                            if texts is None:
                                raise ValueError(f"Text-Feld '{text_field}' fehlt in Batch: {list(examples.keys())}")
                            if not isinstance(texts, list):
                                texts = [texts]
                            
                            # Debug: Zeige ersten Text
                            if len(texts) > 0:
                                send_status("loading", f"Tokenisiere {len(texts)} Texte. Erster Text: {texts[0][:100]}...")
                            
                            # Für Seq2Seq Modelle (T5, MT5, etc.)
                            if self.model_class == 'seq2seq':
                                # Seq2Seq braucht input_ids und labels
                                # Wenn es target/output Feld gibt, nutze es, sonst self-supervised
                                if target_field and target_field in examples:
                                    targets = examples.get(target_field, texts)
                                    if not isinstance(targets, list):
                                        targets = [targets]
                                    
                                    model_inputs = self.tokenizer(
                                        texts,
                                        truncation=True,
                                        max_length=self.config.max_seq_length,
                                        padding="max_length"
                                    )
                                    
                                    # Wichtig: Labels für Seq2Seq models
                                    with self.tokenizer.as_target_tokenizer():
                                        labels = self.tokenizer(
                                            targets,
                                            truncation=True,
                                            max_length=self.config.max_seq_length,
                                            padding="max_length"
                                        )
                                    
                                    # CRITICAL FIX: Maskiere Padding-Tokens mit -100
                                    # Ohne dies wird Loss extrem hoch (40-80 statt 0.5-3)
                                    labels_with_ignore = []
                                    for label_seq in labels["input_ids"]:
                                        masked_labels = [
                                            token if token != self.tokenizer.pad_token_id else -100
                                            for token in label_seq
                                        ]
                                        labels_with_ignore.append(masked_labels)
                                    
                                    model_inputs["labels"] = labels_with_ignore
                                    return model_inputs
                                else:
                                    # Self-supervised: nutze Text als Input und Output
                                    model_inputs = self.tokenizer(
                                        texts,
                                        truncation=True,
                                        max_length=self.config.max_seq_length,
                                        padding="max_length"
                                    )
                                    # Für self-supervised müssen wir auch as_target_tokenizer nutzen
                                    with self.tokenizer.as_target_tokenizer():
                                        labels_tok = self.tokenizer(
                                            texts,
                                            truncation=True,
                                            max_length=self.config.max_seq_length,
                                            padding="max_length"
                                        )
                                    
                                    # CRITICAL FIX: Maskiere Padding auch hier
                                    labels_with_ignore = []
                                    for label_seq in labels_tok["input_ids"]:
                                        masked_labels = [
                                            token if token != self.tokenizer.pad_token_id else -100
                                            for token in label_seq
                                        ]
                                        labels_with_ignore.append(masked_labels)
                                    
                                    model_inputs["labels"] = labels_with_ignore
                                    return model_inputs
                            else:
                                # Für Causal LM und andere (GPT, LLaMA, Mistral, Phi, Qwen ...)
                                model_inputs = self.tokenizer(
                                    texts,
                                    truncation=True,
                                    max_length=self.config.max_seq_length,
                                    padding="max_length"
                                )
                                # Self-supervised labels = input_ids kopieren
                                model_inputs["labels"] = model_inputs["input_ids"].copy()
                                return model_inputs
                        except Exception as e:
                            send_error("Tokenization failed", f"Error: {str(e)}")
                            raise
                    
                    train_dataset = train_dataset.map(tokenize_function, batched=True)
                    
                    # IMPORTANT: Bestätige dass Labels maskiert wurden
                    if self.model_class == 'seq2seq':
                        send_status("loading", "Labels maskiert: Padding-Tokens auf -100 gesetzt (essentiell für Seq2Seq!)")
                    
                    # Debug: Prüfe was im Dataset ist
                    send_status("loading", f"Dataset Spalten nach Tokenisierung: {train_dataset.column_names}")
                    if len(train_dataset) > 0:
                        send_status("loading", f"Erstes Sample Keys: {list(train_dataset[0].keys())}")
                    
                    # Entferne Original-Felder, behalte nur Tokenizer-Outputs
                    columns_to_remove = [col for col in train_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
                    if columns_to_remove:
                        send_status("loading", f"Entferne Spalten: {columns_to_remove}")
                        train_dataset = train_dataset.remove_columns(columns_to_remove)
                    
                    # Setze Format zu PyTorch Tensoren (nur vorhandene Spalten)
                    available_columns = [col for col in ['input_ids', 'attention_mask', 'labels'] if col in train_dataset.column_names]
                    train_dataset.set_format(type='torch', columns=available_columns)
                    
                    self.train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=True,
                        num_workers=0,  # Setze auf 0 für Kompatibilität
                        pin_memory=False  # Deaktiviere für MPS
                    )
                else:
                    # Fallback für andere Dateitypen
                    send_status("loading", "WARNUNG: Keine .txt/.json/.jsonl Dateien gefunden! Nutze GenericFileDataset (erzeugt nur 'path' Keys)")
                    train_dataset = GenericFileDataset(str(train_path))
                    self.train_dataloader = DataLoader(
                        train_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=True
                    )
            else:
                send_status("loading", "WARNUNG: Kein Tokenizer verfügbar! Nutze GenericFileDataset")
                train_dataset = GenericFileDataset(str(train_path))
                self.train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True
                )
            
            # Validation Data
            if val_path.exists():
                if TRANSFORMERS_AVAILABLE and hasattr(self, 'tokenizer'):
                    val_files = list(val_path.glob("*.json")) + list(val_path.glob("*.jsonl")) + list(val_path.glob("*.txt")) + list(val_path.glob("*.csv"))
                    if val_files:
                        val_data = []
                        for f in val_files:
                            if f.suffix == '.txt':
                                with open(f, 'r', encoding='utf-8') as file:
                                    for line in file:
                                        line = line.strip()
                                        if line:
                                            val_data.append({"text": line})
                            elif f.suffix == '.csv':
                                import csv
                                with open(f, 'r', encoding='utf-8') as file:
                                    csv_reader = csv.reader(file)
                                    for row in csv_reader:
                                        if len(row) >= 2:
                                            val_data.append({
                                                "input": row[0].strip(),
                                                "target": row[1].strip()
                                            })
                                        elif len(row) == 1:
                                            val_data.append({"text": row[0].strip()})
                            elif f.suffix == '.jsonl':
                                with open(f, 'r', encoding='utf-8') as file:
                                    for line in file:
                                        try:
                                            data = json.loads(line)
                                            val_data.append(data)
                                        except:
                                            pass
                            else:
                                with open(f, 'r', encoding='utf-8') as file:
                                    try:
                                        data = json.load(file)
                                        if isinstance(data, list):
                                            val_data.extend(data)
                                        else:
                                            val_data.append(data)
                                    except:
                                        pass
                        
                        if val_data:
                            val_dataset = HFDataset.from_list(val_data)
                            val_dataset = val_dataset.map(tokenize_function, batched=True)
                            
                            # Entferne Original-Felder
                            columns_to_remove = [col for col in val_dataset.column_names if col not in ['input_ids', 'attention_mask', 'labels']]
                            if columns_to_remove:
                                val_dataset = val_dataset.remove_columns(columns_to_remove)
                            
                            # Setze Format zu PyTorch Tensoren (nur vorhandene Spalten)
                            available_columns = [col for col in ['input_ids', 'attention_mask', 'labels'] if col in val_dataset.column_names]
                            val_dataset.set_format(type='torch', columns=available_columns)
                            
                            self.val_dataloader = DataLoader(
                                val_dataset,
                                batch_size=self.config.batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=False
                            )
                else:
                    val_dataset = GenericFileDataset(str(val_path))
                    self.val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False
                    )
            
            send_status("loaded", f"Daten geladen: {len(self.train_dataloader)} Training-Batches")
            
        except Exception as e:
            send_error("Data loading failed", str(e))
            raise
    
    def setup_training(self):
        """Initialisiert Optimizer und Scheduler."""
        send_status("setup", "Konfiguriere Training...")
        
        # MEMORY FIX: Use SGD instead of AdamW on MPS to save memory
        # ABER: User-Choice respektieren! Nur warnen, nicht überschreiben
        if self.device.type == 'mps' and self.config.optimizer == 'adamw':
            send_status("setup", "WARNUNG: AdamW verbraucht viel Speicher auf MPS. Wenn OOM auftritt, nutze SGD.")
            # NUR überschreiben wenn Memory-Problem wahrscheinlich ist
            # Prüfe Batch Size und Model Size
            # Für jetzt: NICHT überschreiben, User entscheidet
        
        # Optimizer
        self.optimizer = create_optimizer(self.model, self.config)
        
        # Scheduler
        num_training_steps = len(self.train_dataloader) * self.config.epochs
        if self.config.max_steps > 0:
            num_training_steps = self.config.max_steps
            
        self.scheduler = create_scheduler(
            self.optimizer, 
            self.config, 
            num_training_steps
        )
        
        # Mixed Precision Scaler
        self.scaler = None
        if self.config.fp16 and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
        
        send_status("ready", "Training bereit")
    
    def train_epoch(self, epoch: int) -> float:
        """Trainiert eine Epoche."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)

        for step, batch in enumerate(self.train_dataloader):
            if self.is_stopped:
                break

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            # MEMORY FIX: Clear MPS cache before each step
            if self.device.type == 'mps':
                torch.mps.empty_cache()

            # ----------- FIXED FORWARD PASS -----------
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss = self._forward(batch)
            else:
                loss = self._forward(batch)
            # ------------------------------------------

            # Safety check
            if not loss.requires_grad:
                send_error("Training Error", "Loss requires no grad - möglicherweise sind alle Parameter eingefroren")
                raise RuntimeError("Loss requires no grad")

            # Gradient Accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer / Scheduler
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()

                if self.scheduler and self.config.scheduler != "plateau":
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1
                
                # MEMORY FIX: Clear cache after optimizer step
                if self.device.type == 'mps':
                    torch.mps.empty_cache()

            total_loss += loss.item() * self.config.gradient_accumulation_steps

            # Logging
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
                )

            # Checkpoint speichern
            if self.config.save_strategy == "steps" and \
            self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(epoch, total_loss / (step + 1))

        return total_loss / num_batches
        
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluiert auf Validation Set."""
        if self.val_dataloader is None:
            return 0.0
            
        self.model.eval()
        total_loss = 0.0
        
        for batch in self.val_dataloader:
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            if isinstance(batch, dict) and 'input_ids' in batch:
                outputs = self.model(**batch)
                loss = outputs.loss
            else:
                outputs = self.model(batch)
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            
            total_loss += loss.item()
        
        return total_loss / len(self.val_dataloader)
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Speichert einen Checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint-{self.global_step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        if TRANSFORMERS_AVAILABLE and hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_path)
            if hasattr(self, 'tokenizer'):
                self.tokenizer.save_pretrained(checkpoint_path)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'global_step': self.global_step
            }, checkpoint_path / "pytorch_model.bin")
        
        send_checkpoint(str(checkpoint_path), epoch, {"loss": loss})
        
        # Alte Checkpoints löschen
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Löscht alte Checkpoints über dem Limit."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda x: int(x.name.split("-")[1])
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
    
    def save_final_model(self):
        """Speichert das finale Modell."""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if TRANSFORMERS_AVAILABLE and hasattr(self.model, 'save_pretrained'):
            # Merge LoRA weights if applicable
            if self.config.use_lora and PEFT_AVAILABLE:
                self.model = self.model.merge_and_unload()

                # CRITICAL FIX: Make all tensors contiguous before saving
                # This prevents "non contiguous tensor" errors with safetensors
                for name, param in self.model.named_parameters():
                    if param.data is not None and not param.is_contiguous():
                        param.data = param.data.contiguous()
                
                for name, buffer in self.model.named_buffers():
                    if buffer.data is not None and not buffer.is_contiguous():
                        buffer.data = buffer.data.contiguous()
                
            self.model.save_pretrained(output_path)
            if hasattr(self, 'tokenizer'):
                self.tokenizer.save_pretrained(output_path)
        else:
            torch.save(self.model.state_dict(), output_path / "pytorch_model.bin")
        
        return str(output_path)
    
    def train(self):
        """Haupttrainingsschleife."""
        self.start_time = time.time()  # NEW: Track training start
        send_status("starting", "Training wird gestartet...")
        
        try:
            self.load_model()
            self.load_data()
            self.setup_training()
            
            send_status("training", "Training läuft...")
            
            train_losses = []
            val_losses = []
            
            for epoch in range(1, self.config.epochs + 1):
                if self.is_stopped:
                    break
                
                self.current_epoch = epoch
                send_status("epoch", f"Epoche {epoch}/{self.config.epochs}")
                
                # Training
                train_loss = self.train_epoch(epoch)
                train_losses.append(train_loss)
                
                # Validation
                val_loss = self.evaluate()
                val_losses.append(val_loss)
                
                # Update scheduler on plateau
                if self.scheduler and self.config.scheduler == "plateau":
                    self.scheduler.step(val_loss)
                
                # Progress update
                send_progress(
                    epoch=epoch,
                    total_epochs=self.config.epochs,
                    step=len(self.train_dataloader),
                    total_steps=len(self.train_dataloader),
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=self.optimizer.param_groups[0]['lr']
                )
                
                # Checkpoint am Ende der Epoche
                if self.config.save_strategy == "epoch":
                    self.save_checkpoint(epoch, train_loss)
                
                # Best Model speichern
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = Path(self.config.checkpoint_dir) / "best_model"
                    best_path.mkdir(parents=True, exist_ok=True)
                    if TRANSFORMERS_AVAILABLE and hasattr(self.model, 'save_pretrained'):
                        
                        # CRITICAL FIX: Make all tensors contiguous before saving
                        # This prevents "non contiguous tensor" errors with safetensors
                        for name, param in self.model.named_parameters():
                            if param.data is not None and not param.is_contiguous():
                                param.data = param.data.contiguous()
                        
                        for name, buffer in self.model.named_buffers():
                            if buffer.data is not None and not buffer.is_contiguous():
                                buffer.data = buffer.data.contiguous()
                        
                        self.model.save_pretrained(best_path)
                    else:
                        torch.save(self.model.state_dict(), best_path / "pytorch_model.bin")
            
            # Training abgeschlossen
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
                
                send_complete(final_path, final_metrics)
            else:
                send_status("stopped", "Training wurde gestoppt")
                
        except Exception as e:
            send_error("Training failed", traceback.format_exc())
            raise


# ============ Parameter Rating System ============

class ParameterRatingSystem:
    """Bewertet die Qualität der Training-Parameter-Kombination."""
    
    RATINGS = {
        "excellent": {"score": 5, "label": "Exzellent", "color": "green"},
        "good": {"score": 4, "label": "Gut", "color": "blue"},
        "okay": {"score": 3, "label": "Okay", "color": "yellow"},
        "risky": {"score": 2, "label": "Riskant", "color": "orange"},
        "bad": {"score": 1, "label": "Schlecht", "color": "red"}
    }
    
    @staticmethod
    def rate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Bewertet eine Konfiguration und gibt Feedback."""
        issues = []
        warnings = []
        tips = []
        score = 100
        
        lr = config.get("learning_rate", 5e-5)
        batch_size = config.get("batch_size", 8)
        epochs = config.get("epochs", 3)
        optimizer = config.get("optimizer", "adamw")
        use_lora = config.get("use_lora", False)
        load_in_4bit = config.get("load_in_4bit", False)
        max_grad_norm = config.get("max_grad_norm", 1.0)
        warmup_ratio = config.get("warmup_ratio", 0.0)
        weight_decay = config.get("weight_decay", 0.01)
        
        # Learning Rate Check
        if lr > 1e-3:
            issues.append("Learning Rate ist sehr hoch (>1e-3). Das kann zu instabilem Training führen.")
            score -= 25
        elif lr > 5e-4:
            warnings.append("Learning Rate ist relativ hoch. Beobachte den Loss genau.")
            score -= 10
        elif lr < 1e-6:
            warnings.append("Learning Rate ist sehr niedrig. Training könnte sehr langsam sein.")
            score -= 10
        else:
            tips.append("Learning Rate ist im guten Bereich.")
        
        # Batch Size Check
        if batch_size < 4:
            warnings.append("Kleine Batch Size kann zu verrauschten Gradienten führen.")
            score -= 5
        if batch_size > 64 and not use_lora:
            warnings.append("Große Batch Size benötigt viel GPU-Speicher.")
            score -= 5
        
        # Epochs Check
        if epochs > 10:
            warnings.append("Viele Epochen können zu Overfitting führen.")
            score -= 5
        elif epochs < 2:
            warnings.append("Wenige Epochen könnten nicht ausreichen für gute Konvergenz.")
            score -= 5
        
        # Optimizer + LR Kombination
        if optimizer == "sgd" and lr > 1e-2:
            warnings.append("SGD mit hoher LR kann instabil sein.")
            score -= 10
        if optimizer in ["adam", "adamw"] and lr > 1e-3:
            warnings.append("Adam-Optimizer funktionieren meist besser mit LR < 1e-3.")
            score -= 10
        
        # LoRA Checks
        if use_lora:
            lora_r = config.get("lora_r", 8)
            lora_alpha = config.get("lora_alpha", 32)
            
            if lora_alpha < lora_r:
                warnings.append("LoRA Alpha sollte >= LoRA R sein für stabiles Training.")
                score -= 10
            if lora_r > 64:
                warnings.append("Sehr hoher LoRA Rank kann zu Overfitting führen.")
                score -= 5
            
            tips.append("LoRA ist aktiviert - gute Wahl für effizientes Fine-Tuning!")
        
        # 4-bit Quantization
        if load_in_4bit:
            if not use_lora:
                issues.append("4-bit Quantisierung sollte mit LoRA/QLoRA verwendet werden.")
                score -= 20
            else:
                tips.append("QLoRA-Setup erkannt - sehr speichereffizient!")
        
        # Warmup
        if warmup_ratio == 0 and epochs >= 3:
            tips.append("Tipp: Ein Warmup von 0.03-0.1 kann Training stabilisieren.")
        elif warmup_ratio > 0.2:
            warnings.append("Sehr langes Warmup kann effektive Trainingszeit reduzieren.")
            score -= 5
        
        # Weight Decay
        if weight_decay > 0.1:
            warnings.append("Hoher Weight Decay kann zu Underfitting führen.")
            score -= 10
        elif weight_decay == 0 and optimizer == "adamw":
            tips.append("Tipp: AdamW profitiert von leichtem Weight Decay (0.01-0.1).")
        
        # Gradient Clipping
        if max_grad_norm > 5.0:
            warnings.append("Hohes Gradient Clipping kann explodierende Gradienten nicht verhindern.")
            score -= 5
        
        # Determine rating
        if score >= 90:
            rating = "excellent"
        elif score >= 75:
            rating = "good"
        elif score >= 60:
            rating = "okay"
        elif score >= 40:
            rating = "risky"
        else:
            rating = "bad"
        
        return {
            "score": max(0, min(100, score)),
            "rating": rating,
            "rating_info": ParameterRatingSystem.RATINGS[rating],
            "issues": issues,
            "warnings": warnings,
            "tips": tips
        }


# ============ Preset Configurations ============

PRESET_CONFIGS = {
    "llm_qlora_efficient": {
        "name": "LLM QLoRA (Speichereffizient)",
        "description": "Optimiert für Fine-Tuning großer Sprachmodelle mit wenig VRAM",
        "config": {
            "learning_rate": 2e-4,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "epochs": 3,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "load_in_4bit": True,
            "fp16": True,
            "max_grad_norm": 0.3,
            "max_seq_length": 512
        }
    },
    "llm_lora_balanced": {
        "name": "LLM LoRA (Ausgewogen)",
        "description": "Gute Balance zwischen Qualität und Geschwindigkeit",
        "config": {
            "learning_rate": 1e-4,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "epochs": 3,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "use_lora": True,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "load_in_4bit": False,
            "fp16": True,
            "max_grad_norm": 1.0,
            "max_seq_length": 1024
        }
    },
    "llm_full_finetuning": {
        "name": "LLM Full Fine-Tuning",
        "description": "Vollständiges Fine-Tuning - benötigt viel VRAM",
        "config": {
            "learning_rate": 2e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "epochs": 3,
            "optimizer": "adamw",
            "scheduler": "linear",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "use_lora": False,
            "load_in_4bit": False,
            "fp16": True,
            "max_grad_norm": 1.0,
            "max_seq_length": 512
        }
    },
    "classification_standard": {
        "name": "Text-Klassifikation Standard",
        "description": "Für Sentiment, Topic Classification etc.",
        "config": {
            "learning_rate": 5e-5,
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "epochs": 5,
            "optimizer": "adamw",
            "scheduler": "linear",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "use_lora": False,
            "fp16": True,
            "max_grad_norm": 1.0,
            "max_seq_length": 256,
            "task_type": "seq_classification"
        }
    },
    "image_classification": {
        "name": "Bild-Klassifikation",
        "description": "Für Image Classification Tasks",
        "config": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "epochs": 10,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_ratio": 0.05,
            "weight_decay": 0.05,
            "use_lora": False,
            "fp16": True,
            "max_grad_norm": 1.0
        }
    },
    "quick_test": {
        "name": "Schnelltest",
        "description": "Zum schnellen Testen ob alles funktioniert",
        "config": {
            "learning_rate": 1e-4,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
            "optimizer": "adam",
            "scheduler": "constant",
            "warmup_ratio": 0,
            "weight_decay": 0,
            "use_lora": False,
            "fp16": False,
            "max_grad_norm": 1.0,
            "max_seq_length": 128,
            "logging_steps": 10,
            "eval_steps": 50
        }
    },
    "stable_conservative": {
        "name": "Stabil & Konservativ",
        "description": "Langsames aber stabiles Training",
        "config": {
            "learning_rate": 1e-5,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "epochs": 5,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "use_lora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "fp16": True,
            "max_grad_norm": 0.5,
            "max_seq_length": 512
        }
    },
    "aggressive_fast": {
        "name": "Aggressiv & Schnell",
        "description": "Schnelles Training mit höherem Risiko",
        "config": {
            "learning_rate": 5e-4,
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "epochs": 2,
            "optimizer": "adam",
            "scheduler": "one_cycle",
            "warmup_ratio": 0,
            "weight_decay": 0,
            "use_lora": True,
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "fp16": True,
            "max_grad_norm": 1.0,
            "max_seq_length": 256
        }
    }
}


# ============ CLI Entry Point ============

def main():
    parser = argparse.ArgumentParser(description="FrameTrain Training Engine")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--rate-only", action="store_true", help="Only rate the config, don't train")
    parser.add_argument("--list-presets", action="store_true", help="List available presets")
    
    args = parser.parse_args()
    
    if args.list_presets:
        presets = [
            {"id": k, "name": v["name"], "description": v["description"]}
            for k, v in PRESET_CONFIGS.items()
        ]
        print(json.dumps({"presets": presets}))
        return
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    if args.rate_only:
        rating = ParameterRatingSystem.rate_config(config_dict)
        print(json.dumps({"type": "rating", "data": rating}))
        return
    
    # Check requirements
    if not TORCH_AVAILABLE:
        send_error("PyTorch not installed", "Please install PyTorch: pip install torch")
        sys.exit(1)
    
    # Create and run training
    config = TrainingConfig(config_dict)
    engine = TrainingEngine(config)
    engine.train()


if __name__ == "__main__":
    main()