import { useState, useEffect } from 'react';
import { Play, Pause, Square, AlertCircle, CheckCircle, Info, TrendingUp, Clock, HardDrive, Settings, ChevronDown, ChevronUp, Database, FileText } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { invoke } from '@tauri-apps/api/core';

interface TrainingPanelProps {
  selectedModel: any;
}

interface TrainingConfig {
  learning_rate: number;
  batch_size: number;
  eval_batch_size: number;
  epochs: number;
  optimizer: string;
  weight_decay: number;
  dropout: number;
  max_grad_norm: number;
  warmup_ratio: number;
  warmup_steps: number | null;
  lr_scheduler_type: string;
  gradient_accumulation_steps: number;
  fp16: boolean;
  bf16: boolean;
  save_strategy: string;
  save_steps: number | null;
  save_total_limit: number;
  eval_strategy: string;
  eval_interval: number;
  metric_for_best_model: string;
  load_best_model_at_end: boolean;
  early_stopping_patience: number | null;
  early_stopping_threshold: number;
  logging_steps: number;
  logging_strategy: string;
  predict_with_generate: boolean;
  generation_max_length: number | null;
  generation_num_beams: number | null;
  dataloader_num_workers: number;
  group_by_length: boolean;
  label_smoothing_factor: number;
  seed: number;
}

interface ConfigValidation {
  is_valid: boolean;
  quality_level: string;
  overall_score: number;
  warnings: string[];
  recommendations: string[];
  issues: ConfigIssue[];
  estimated_training_time: string | null;
  estimated_memory_usage: string | null;
}

interface ConfigIssue {
  severity: string;
  category: string;
  message: string;
  suggestion: string;
}

interface DatasetInfo {
  id: string;
  model_id: string;
  name: string;
  dataset_type: string;
  file_path: string;
  file_count: number;
  total_size_bytes: number;
  split_type: string;
  status: string;
  formats: string[];
  is_hard_example: boolean;
  created_at: string;
  last_used_at: string | null;
}

export default function TrainingPanel({ selectedModel }: TrainingPanelProps) {
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [trainingData, setTrainingData] = useState<any[]>([]);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [validation, setValidation] = useState<ConfigValidation | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasets, setSelectedDatasets] = useState<Set<string>>(new Set());
  const [totalSamples, setTotalSamples] = useState(0);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  
  // Training Configuration
  const [config, setConfig] = useState<TrainingConfig>({
    learning_rate: 0.00002,
    batch_size: 8,
    eval_batch_size: 8,
    epochs: 3,
    optimizer: 'adamw',
    weight_decay: 0.01,
    dropout: 0.1,
    max_grad_norm: 1.0,
    warmup_ratio: 0.05,
    warmup_steps: null,
    lr_scheduler_type: 'linear',
    gradient_accumulation_steps: 1,
    fp16: false,
    bf16: false,
    save_strategy: 'epoch',
    save_steps: null,
    save_total_limit: 3,
    eval_strategy: 'epoch',
    eval_interval: 100,
    metric_for_best_model: 'accuracy',
    load_best_model_at_end: true,
    early_stopping_patience: 3,
    early_stopping_threshold: 0.001,
    logging_steps: 50,
    logging_strategy: 'steps',
    predict_with_generate: false,
    generation_max_length: null,
    generation_num_beams: null,
    dataloader_num_workers: 0,
    group_by_length: false,
    label_smoothing_factor: 0.0,
    seed: 42,
  });

  // Lade Datasets wenn Modell ausgewählt wird
  useEffect(() => {
    if (selectedModel?.id) {
      loadDatasets();
    }
  }, [selectedModel?.id]);

  // Berechne totale Samples basierend auf ausgewählten Datasets
  useEffect(() => {
    const total = datasets
      .filter(ds => selectedDatasets.has(ds.id) && ds.split_type === 'train')
      .reduce((sum, ds) => sum + ds.file_count, 0);
    setTotalSamples(total);
  }, [selectedDatasets, datasets]);

  // Validiere Config bei jeder Änderung
  useEffect(() => {
    if (totalSamples > 0) {
      validateConfig();
    }
  }, [config, totalSamples]);

  const loadDatasets = async () => {
    try {
      const result = await invoke<DatasetInfo[]>('list_datasets_for_model', {
        modelId: selectedModel.id,
      });
      setDatasets(result);
      
      // Auto-select train datasets
      const trainDatasets = result.filter(ds => ds.split_type === 'train');
      if (trainDatasets.length > 0) {
        setSelectedDatasets(new Set(trainDatasets.map(ds => ds.id)));
      }
    } catch (error) {
      console.error('Fehler beim Laden der Datasets:', error);
    }
  };

  const validateConfig = async () => {
    try {
      const result = await invoke<ConfigValidation>('validate_training_config', {
        config,
        totalSamples,
      });
      setValidation(result);
    } catch (error) {
      console.error('Validation error:', error);
    }
  };

  const updateConfig = (key: keyof TrainingConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  const toggleDataset = (datasetId: string) => {
    setSelectedDatasets(prev => {
      const newSet = new Set(prev);
      if (newSet.has(datasetId)) {
        newSet.delete(datasetId);
      } else {
        newSet.add(datasetId);
      }
      return newSet;
    });
  };

  const getQualityColor = (level: string) => {
    switch (level) {
      case 'excellent': return 'text-green-400 bg-green-500/20 border-green-500/30';
      case 'good': return 'text-blue-400 bg-blue-500/20 border-blue-500/30';
      case 'fair': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
      case 'poor': return 'text-red-400 bg-red-500/20 border-red-500/30';
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30';
    }
  };

  const getQualityBadgeColor = (level: string) => {
    switch (level) {
      case 'excellent': return 'bg-green-500/30 text-green-300 border-green-500/50';
      case 'good': return 'bg-blue-500/30 text-blue-300 border-blue-500/50';
      case 'fair': return 'bg-yellow-500/30 text-yellow-300 border-yellow-500/50';
      case 'poor': return 'bg-red-500/30 text-red-300 border-red-500/50';
      default: return 'bg-gray-500/30 text-gray-300 border-gray-500/50';
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  const getQualityColorSimple = (level: string) => {
    switch (level) {
      case 'excellent': return 'bg-green-500/20 border-green-500/50';
      case 'good': return 'bg-yellow-500/20 border-yellow-500/50';
      case 'fair': return 'bg-orange-500/20 border-orange-500/50';
      case 'poor': return 'bg-red-500/20 border-red-500/50';
      default: return 'bg-gray-500/20 border-gray-500/50';
    }
  };

  const getQualityIcon = (level: string) => {
    switch (level) {
      case 'excellent': return '🟢';
      case 'good': return '🟡';
      case 'fair': return '🟠';
      case 'poor': return '🔴';
      default: return '⚪';
    }
  };

  const getQualityText = (level: string) => {
    switch (level) {
      case 'excellent': return 'Perfekt';
      case 'good': return 'Paar kleine Sachen';
      case 'fair': return 'Nochmal prüfen';
      case 'poor': return 'Fehler gefunden';
      default: return 'Unbekannt';
    }
  };

  const handleStartTraining = () => {
    if (selectedDatasets.size === 0) {
      alert('Bitte wähle mindestens ein Dataset aus!');
      return;
    }

    // Bei roter Ampel: Bestätigung erforderlich
    if (validation?.quality_level === 'poor') {
      setShowConfirmDialog(true);
      return;
    }

    // Sonst direkt starten
    startTraining();
  };

  const startTraining = async () => {
    try {
      setIsTraining(true);
      
      // Erstelle vollständiges TrainingConfig Objekt für Rust
      const fullConfig = {
        id: crypto.randomUUID(),
        model_id: selectedModel.id,
        model_name: selectedModel.name,
        dataset_ids: Array.from(selectedDatasets),
        
        // Basic Training
        epochs: config.epochs,
        batch_size: config.batch_size,
        eval_batch_size: config.eval_batch_size,
        learning_rate: config.learning_rate,
        optimizer: config.optimizer,
        
        // Regularization
        weight_decay: config.weight_decay,
        max_grad_norm: config.max_grad_norm,
        dropout: config.dropout,
        attention_dropout: config.dropout, // Use same as dropout
        
        // Learning Rate Schedule
        warmup_ratio: config.warmup_ratio,
        warmup_steps: config.warmup_steps || 0,
        lr_scheduler_type: config.lr_scheduler_type,
        
        // Training Strategy
        gradient_accumulation_steps: config.gradient_accumulation_steps,
        fp16: config.fp16,
        bf16: config.bf16,
        
        // Checkpointing
        save_strategy: config.save_strategy,
        save_total_limit: config.save_total_limit,
        checkpoint_interval: config.save_steps || 500,
        
        // Evaluation
        eval_strategy: config.eval_strategy,
        eval_interval: config.eval_interval,
        metric_for_best_model: config.metric_for_best_model,
        greater_is_better: true,
        load_best_model_at_end: config.load_best_model_at_end,
        
        // Early Stopping
        early_stopping_patience: config.early_stopping_patience,
        early_stopping_threshold: config.early_stopping_threshold,
        
        // Logging
        logging_steps: config.logging_steps,
        logging_strategy: config.logging_strategy,
        
        // Generation (for Seq2Seq)
        predict_with_generate: config.predict_with_generate,
        generation_max_length: config.generation_max_length,
        generation_num_beams: config.generation_num_beams,
        
        // Advanced
        seed: config.seed,
        resume_from_checkpoint: null,
        dataloader_num_workers: config.dataloader_num_workers,
        dataloader_pin_memory: false,
        group_by_length: config.group_by_length,
        length_column_name: null,
        label_smoothing_factor: config.label_smoothing_factor,
        
        created_at: new Date().toISOString(),
      };
      
      console.log('Starting training with config:', fullConfig);
      
      const trainingId = await invoke<string>('start_training_session', {
        config: fullConfig,
      });
      
      console.log('✅ Training started with ID:', trainingId);
      alert(`Training erfolgreich gestartet! ID: ${trainingId}`);
      
    } catch (error) {
      console.error('❌ Fehler beim Start des Trainings:', error);
      alert('Fehler beim Start des Trainings: ' + error);
      setIsTraining(false);
    }
  };

  return (
    <div>
      <h2 className="text-3xl font-bold text-white mb-8">Training Konfiguration</h2>

      {!selectedModel ? (
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-12 border border-white/20 text-center">
          <div className="text-6xl mb-4">🎯</div>
          <h3 className="text-2xl font-bold text-white mb-2">Kein Modell ausgewählt</h3>
          <p className="text-gray-300">
            Wähle zuerst ein Modell in der Modellverwaltung aus.
          </p>
        </div>
      ) : (
        <>
          {/* Modell-Info */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
            <h3 className="text-xl font-bold text-white mb-2">Ausgewähltes Modell</h3>
            <p className="text-gray-300">{selectedModel.name}</p>
            {selectedModel.base_model && (
              <p className="text-sm text-gray-400 mt-1">Typ: {selectedModel.base_model}</p>
            )}
          </div>

          {/* Dataset Auswahl */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Database className="w-6 h-6 text-purple-400" />
                <h3 className="text-xl font-bold text-white">Datasets auswählen</h3>
              </div>
              <div className="text-sm text-gray-400">
                {selectedDatasets.size} ausgewählt · {totalSamples} Samples
              </div>
            </div>

            {datasets.length === 0 ? (
              <div className="text-center py-8">
                <FileText className="w-12 h-12 text-gray-500 mx-auto mb-3" />
                <p className="text-gray-400 mb-2">Keine Datasets verfügbar</p>
                <p className="text-sm text-gray-500">
                  Lade zuerst Datasets in der Datensätze-Sektion hoch
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {datasets.map((dataset) => (
                  <div
                    key={dataset.id}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedDatasets.has(dataset.id)
                        ? 'bg-purple-500/20 border-purple-500/50'
                        : 'bg-white/5 border-white/10 hover:bg-white/10'
                    }`}
                    onClick={() => toggleDataset(dataset.id)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <input
                            type="checkbox"
                            checked={selectedDatasets.has(dataset.id)}
                            onChange={() => toggleDataset(dataset.id)}
                            className="w-4 h-4 text-purple-600 bg-white/5 border-white/10 rounded focus:ring-purple-500"
                          />
                          <h4 className="text-white font-semibold">{dataset.name}</h4>
                          <span className={`px-2 py-0.5 text-xs rounded ${
                            dataset.split_type === 'train' ? 'bg-blue-500/20 text-blue-300' :
                            dataset.split_type === 'val' ? 'bg-yellow-500/20 text-yellow-300' :
                            'bg-purple-500/20 text-purple-300'
                          }`}>
                            {dataset.split_type}
                          </span>
                        </div>
                        <div className="ml-6 space-y-1">
                          <p className="text-sm text-gray-400">
                            {dataset.file_count} Dateien · {formatBytes(dataset.total_size_bytes)}
                          </p>
                          <p className="text-xs text-gray-500">
                            Typ: {dataset.dataset_type} · Formate: {dataset.formats.join(', ')}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Validation Score */}
          {validation && (
            <div className={`backdrop-blur-lg rounded-2xl p-6 border-2 mb-6 ${getQualityColor(validation.quality_level)}`}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-white">Konfigurations-Qualität</h3>
                <div className={`px-6 py-2 rounded-lg font-bold text-2xl border-2 ${getQualityBadgeColor(validation.quality_level)}`}>
                  {validation.overall_score.toFixed(0)}/100
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="flex items-center space-x-3">
                  <Clock className="w-6 h-6 text-blue-400" />
                  <div>
                    <p className="text-sm text-gray-400">Geschätzte Zeit</p>
                    <p className="text-white font-medium">{validation.estimated_training_time}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <HardDrive className="w-6 h-6 text-purple-400" />
                  <div>
                    <p className="text-sm text-gray-400">Geschätzter Speicher</p>
                    <p className="text-white font-medium">{validation.estimated_memory_usage}</p>
                  </div>
                </div>
              </div>

              {/* Quality Level Indicator */}
              <div className="mb-4">
                <div className="flex items-center justify-between text-sm mb-2">
                  <span className="text-gray-400">Bewertung</span>
                  <span className="text-white font-semibold uppercase">
                    {validation.quality_level === 'excellent' && '🏆 Exzellent'}
                    {validation.quality_level === 'good' && '✨ Gut'}
                    {validation.quality_level === 'fair' && '⚡ OK'}
                    {validation.quality_level === 'poor' && '⚠️ Verbesserungswürdig'}
                  </span>
                </div>
                <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      validation.quality_level === 'excellent' ? 'bg-gradient-to-r from-green-500 to-emerald-500' :
                      validation.quality_level === 'good' ? 'bg-gradient-to-r from-blue-500 to-cyan-500' :
                      validation.quality_level === 'fair' ? 'bg-gradient-to-r from-yellow-500 to-orange-500' :
                      'bg-gradient-to-r from-red-500 to-pink-500'
                    }`}
                    style={{ width: `${validation.overall_score}%` }}
                  />
                </div>
              </div>

              {/* Recommendations */}
              {validation.recommendations.length > 0 && (
                <div className="space-y-2 mb-3">
                  {validation.recommendations.map((rec, i) => (
                    <div key={i} className="flex items-start space-x-2 text-sm bg-green-500/10 rounded-lg p-2">
                      <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0 text-green-400" />
                      <span className="text-green-200">{rec}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Warnings */}
              {validation.warnings.length > 0 && (
                <div className="space-y-2 mb-3">
                  {validation.warnings.map((warn, i) => (
                    <div key={i} className="flex items-start space-x-2 text-sm bg-yellow-500/10 rounded-lg p-2">
                      <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0 text-yellow-400" />
                      <span className="text-yellow-200">{warn}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Issues */}
              {validation.issues.length > 0 && (
                <div className="space-y-2">
                  {validation.issues.map((issue, i) => (
                    <div key={i} className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                      <div className="flex items-start space-x-2">
                        <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
                        <div className="flex-1">
                          <p className="text-sm text-red-300 font-medium">{issue.message}</p>
                          <p className="text-xs text-red-400 mt-1">💡 {issue.suggestion}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Basic Parameters */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6">
            <h3 className="text-lg font-bold text-white mb-4">Basis-Parameter</h3>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Learning Rate
                </label>
                <input
                  type="number"
                  value={config.learning_rate}
                  onChange={(e) => updateConfig('learning_rate', parseFloat(e.target.value))}
                  step="0.000001"
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={config.batch_size}
                  onChange={(e) => updateConfig('batch_size', parseInt(e.target.value))}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Epochs
                </label>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => updateConfig('epochs', parseInt(e.target.value))}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Gradient Accumulation Steps
                </label>
                <input
                  type="number"
                  value={config.gradient_accumulation_steps}
                  onChange={(e) => updateConfig('gradient_accumulation_steps', parseInt(e.target.value))}
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
                <p className="text-xs text-gray-400 mt-1">
                  Effektive Batch Size: {config.batch_size * config.gradient_accumulation_steps}
                </p>
              </div>
            </div>
          </div>

          {/* Advanced Parameters Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full bg-white/10 backdrop-blur-lg rounded-2xl p-4 border border-white/20 mb-6 flex items-center justify-between hover:bg-white/15 transition-all"
          >
            <div className="flex items-center space-x-2">
              <Settings className="w-5 h-5 text-purple-400" />
              <span className="text-white font-semibold">Erweiterte Parameter</span>
            </div>
            {showAdvanced ? (
              <ChevronUp className="w-5 h-5 text-gray-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-400" />
            )}
          </button>

          {/* Advanced Parameters */}
          {showAdvanced && (
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-6 space-y-6">
              {/* Regularization */}
              <div>
                <h4 className="text-md font-bold text-white mb-3">Regularisierung</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Weight Decay
                    </label>
                    <input
                      type="number"
                      value={config.weight_decay}
                      onChange={(e) => updateConfig('weight_decay', parseFloat(e.target.value))}
                      step="0.001"
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Dropout
                    </label>
                    <input
                      type="number"
                      value={config.dropout}
                      onChange={(e) => updateConfig('dropout', parseFloat(e.target.value))}
                      step="0.05"
                      max="1"
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Max Grad Norm
                    </label>
                    <input
                      type="number"
                      value={config.max_grad_norm}
                      onChange={(e) => updateConfig('max_grad_norm', parseFloat(e.target.value))}
                      step="0.1"
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Label Smoothing
                    </label>
                    <input
                      type="number"
                      value={config.label_smoothing_factor}
                      onChange={(e) => updateConfig('label_smoothing_factor', parseFloat(e.target.value))}
                      step="0.05"
                      max="1"
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>
              </div>

              {/* Learning Rate Schedule */}
              <div>
                <h4 className="text-md font-bold text-white mb-3">Learning Rate Schedule</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Warmup Ratio
                    </label>
                    <input
                      type="number"
                      value={config.warmup_ratio}
                      onChange={(e) => updateConfig('warmup_ratio', parseFloat(e.target.value))}
                      step="0.01"
                      max="1"
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Scheduler Type
                    </label>
                    <select
                      value={config.lr_scheduler_type}
                      onChange={(e) => updateConfig('lr_scheduler_type', e.target.value)}
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="linear">Linear</option>
                      <option value="cosine">Cosine</option>
                      <option value="constant">Constant</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Early Stopping */}
              <div>
                <h4 className="text-md font-bold text-white mb-3">Early Stopping</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Patience (Epochs)
                    </label>
                    <input
                      type="number"
                      value={config.early_stopping_patience || ''}
                      onChange={(e) => updateConfig('early_stopping_patience', parseInt(e.target.value) || null)}
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Metric for Best Model
                    </label>
                    <input
                      type="text"
                      value={config.metric_for_best_model}
                      onChange={(e) => updateConfig('metric_for_best_model', e.target.value)}
                      placeholder="accuracy"
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>
              </div>

              {/* Precision */}
              <div>
                <h4 className="text-md font-bold text-white mb-3">Precision</h4>
                <div className="flex space-x-4">
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.fp16}
                      onChange={(e) => updateConfig('fp16', e.target.checked)}
                      className="w-4 h-4 text-purple-600 bg-white/5 border-white/10 rounded focus:ring-purple-500"
                    />
                    <span className="text-white">FP16 (Mixed Precision)</span>
                  </label>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.bf16}
                      onChange={(e) => updateConfig('bf16', e.target.checked)}
                      className="w-4 h-4 text-purple-600 bg-white/5 border-white/10 rounded focus:ring-purple-500"
                    />
                    <span className="text-white">BF16 (Bfloat16)</span>
                  </label>
                </div>
              </div>

              {/* Logging */}
              <div>
                <h4 className="text-md font-bold text-white mb-3">Logging & Checkpoints</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Logging Steps
                    </label>
                    <input
                      type="number"
                      value={config.logging_steps}
                      onChange={(e) => updateConfig('logging_steps', parseInt(e.target.value))}
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Save Strategy
                    </label>
                    <select
                      value={config.save_strategy}
                      onChange={(e) => updateConfig('save_strategy', e.target.value)}
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="epoch">Nach Epoch</option>
                      <option value="steps">Nach Steps</option>
                      <option value="no">Kein Auto-Save</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Save Total Limit
                    </label>
                    <input
                      type="number"
                      value={config.save_total_limit}
                      onChange={(e) => updateConfig('save_total_limit', parseInt(e.target.value))}
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Eval Interval (Steps)
                    </label>
                    <input
                      type="number"
                      value={config.eval_interval}
                      onChange={(e) => updateConfig('eval_interval', parseInt(e.target.value))}
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                </div>
              </div>

              {/* Other Options */}
              <div>
                <h4 className="text-md font-bold text-white mb-3">Sonstige Optionen</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Optimizer
                    </label>
                    <select
                      value={config.optimizer}
                      onChange={(e) => updateConfig('optimizer', e.target.value)}
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="adamw">AdamW</option>
                      <option value="adam">Adam</option>
                      <option value="sgd">SGD</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Random Seed
                    </label>
                    <input
                      type="number"
                      value={config.seed}
                      onChange={(e) => updateConfig('seed', parseInt(e.target.value))}
                      className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>
                  <div className="col-span-2">
                    <label className="flex items-center space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={config.group_by_length}
                        onChange={(e) => updateConfig('group_by_length', e.target.checked)}
                        className="w-4 h-4 text-purple-600 bg-white/5 border-white/10 rounded focus:ring-purple-500"
                      />
                      <span className="text-white">Group by Length (Effizienter)</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Validation Score - Vereinfacht als Ampel */}
          {validation && (
            <div className={`backdrop-blur-lg rounded-2xl p-6 border-2 mb-6 ${getQualityColorSimple(validation.quality_level)}`}>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <span className="text-5xl">{getQualityIcon(validation.quality_level)}</span>
                  <div>
                    <h3 className="text-2xl font-bold text-white">{getQualityText(validation.quality_level)}</h3>
                    <p className="text-sm text-gray-300">Trainingsparameter-Bewertung</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-4xl font-bold text-white">{validation.overall_score.toFixed(0)}/100</div>
                  <div className="text-xs text-gray-400 uppercase tracking-wider">{validation.quality_level}</div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="flex items-center space-x-3 bg-white/5 rounded-lg p-3">
                  <Clock className="w-5 h-5 text-blue-400" />
                  <div>
                    <p className="text-xs text-gray-400">Geschätzte Zeit</p>
                    <p className="text-white font-medium">{validation.estimated_training_time || 'N/A'}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3 bg-white/5 rounded-lg p-3">
                  <HardDrive className="w-5 h-5 text-purple-400" />
                  <div>
                    <p className="text-xs text-gray-400">Geschätzter Speicher</p>
                    <p className="text-white font-medium">{validation.estimated_memory_usage || 'N/A'}</p>
                  </div>
                </div>
              </div>

              {/* Quality Bar */}
              <div className="mt-4">
                <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      validation.quality_level === 'excellent' ? 'bg-gradient-to-r from-green-500 to-emerald-500' :
                      validation.quality_level === 'good' ? 'bg-gradient-to-r from-yellow-500 to-yellow-400' :
                      validation.quality_level === 'fair' ? 'bg-gradient-to-r from-orange-500 to-orange-400' :
                      'bg-gradient-to-r from-red-500 to-red-400'
                    }`}
                    style={{ width: `${validation.overall_score}%` }}
                  />
                </div>
              </div>

              {/* Kompakte Hinweise */}
              {validation.recommendations.length > 0 && (
                <div className="mt-4 space-y-1">
                  {validation.recommendations.slice(0, 2).map((rec, i) => (
                    <div key={i} className="flex items-start space-x-2 text-sm">
                      <span className="text-green-400">✓</span>
                      <span className="text-gray-300">{rec}</span>
                    </div>
                  ))}
                </div>
              )}

              {validation.warnings.length > 0 && (
                <div className="mt-4 space-y-1">
                  {validation.warnings.slice(0, 2).map((warn, i) => (
                    <div key={i} className="flex items-start space-x-2 text-sm">
                      <span className="text-yellow-400">⚠</span>
                      <span className="text-gray-300">{warn}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Dataset Warning */}
          {selectedDatasets.size === 0 && (
            <div className="bg-yellow-500/10 backdrop-blur-lg rounded-2xl p-6 border-2 border-yellow-500/30 mb-6">
              <div className="flex items-start space-x-3">
                <AlertCircle className="w-6 h-6 text-yellow-400 flex-shrink-0" />
                <div>
                  <h3 className="text-lg font-bold text-yellow-300 mb-1">Keine Datasets ausgewählt</h3>
                  <p className="text-sm text-yellow-200/80 mb-3">
                    Du musst mindestens ein Dataset auswählen um das Training zu starten.
                  </p>
                  <p className="text-xs text-yellow-200/60">
                    💡 Scrolle nach oben zum "Datasets auswählen" Abschnitt und wähle die Trainingsdaten aus.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Training Actions */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <button
                  onClick={handleStartTraining}
                  disabled={selectedDatasets.size === 0 || isTraining}
                  className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
                >
                  <Play className="w-5 h-5" />
                  <span>Training starten</span>
                </button>

                {isTraining && (
                  <div className="flex items-center space-x-2 text-green-400">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    <span className="text-sm font-medium">Training läuft...</span>
                  </div>
                )}
              </div>

              {/* Status Messages */}
              <div className="flex items-center space-x-2">
                {selectedDatasets.size === 0 && (
                  <div className="flex items-center space-x-2 text-yellow-400 bg-yellow-500/10 px-3 py-1.5 rounded-lg">
                    <AlertCircle className="w-4 h-4" />
                    <span className="text-sm font-medium">Keine Datasets ausgewählt</span>
                  </div>
                )}

                {selectedDatasets.size > 0 && !isTraining && validation && (
                  <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg ${
                    validation.quality_level === 'excellent' ? 'text-green-400 bg-green-500/10' :
                    validation.quality_level === 'good' ? 'text-yellow-400 bg-yellow-500/10' :
                    validation.quality_level === 'fair' ? 'text-orange-400 bg-orange-500/10' :
                    'text-red-400 bg-red-500/10'
                  }`}>
                    <span className="text-lg">{getQualityIcon(validation.quality_level)}</span>
                    <span className="text-sm font-medium">{getQualityText(validation.quality_level)}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Confirm Dialog für rote Ampel */}
          {showConfirmDialog && (
            <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
              <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl p-8 border-2 border-red-500/50 max-w-md shadow-2xl">
                <div className="flex items-start space-x-4 mb-6">
                  <span className="text-6xl">🔴</span>
                  <div>
                    <h3 className="text-2xl font-bold text-white mb-2">Fehler in Konfiguration</h3>
                    <p className="text-gray-300">
                      Die Trainingsparameter haben Probleme. Bist du sicher, dass du trotzdem starten möchtest?
                    </p>
                  </div>
                </div>

                {validation?.warnings.length > 0 && (
                  <div className="bg-red-500/10 rounded-lg p-4 mb-6 border border-red-500/30">
                    <p className="text-sm font-semibold text-red-300 mb-2">Probleme:</p>
                    <ul className="space-y-1">
                      {validation.warnings.slice(0, 3).map((warn, i) => (
                        <li key={i} className="text-sm text-red-200 flex items-start space-x-2">
                          <span>•</span>
                          <span>{warn}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="flex space-x-3">
                  <button
                    onClick={() => {
                      setShowConfirmDialog(false);
                      startTraining();
                    }}
                    className="flex-1 px-4 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-all"
                  >
                    Ja, trotzdem starten
                  </button>
                  <button
                    onClick={() => setShowConfirmDialog(false)}
                    className="flex-1 px-4 py-3 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-semibold transition-all"
                  >
                    Abbrechen
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}



