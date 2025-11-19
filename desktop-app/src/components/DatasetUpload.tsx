import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { 
  Upload, 
  Folder, 
  Trash2, 
  Split, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  Image,
  Video,
  FileText,
  Music,
  Layers,
  Clock,
  HardDrive,
  Filter
} from 'lucide-react';

interface Model {
  id: string;
  name: string;
  base_model: string | null;
}

interface DatasetInfo {
  id: string;
  model_id: string;
  name: string;
  dataset_type: 'image' | 'video' | 'text' | 'audio' | 'mixed';
  file_path: string;
  file_count: number;
  total_size_bytes: number;
  split_type: 'train' | 'val' | 'test';
  status: 'unused' | 'inuse' | 'used' | 'hardexample';
  formats: string[];
  is_hard_example: boolean;
  created_at: string;
  last_used_at: string | null;
}

interface DatasetValidation {
  is_valid: boolean;
  detected_type: string;
  file_count: number;
  formats: string[];
  errors: string[];
  warnings: string[];
  model_compatible: boolean;
  expected_formats: string[];
}

interface SplitConfig {
  train_ratio: number;
  val_ratio: number;
  test_ratio: number;
  shuffle: boolean;
}

export default function DatasetUpload() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [showSplitDialog, setShowSplitDialog] = useState(false);
  const [selectedDatasetForSplit, setSelectedDatasetForSplit] = useState<DatasetInfo | null>(null);
  
  // Upload state
  const [uploadPath, setUploadPath] = useState('');
  const [uploadName, setUploadName] = useState('');
  const [validation, setValidation] = useState<DatasetValidation | null>(null);
  const [splitType, setSplitType] = useState<'train' | 'val' | 'test'>('train');
  
  // Split config
  const [splitConfig, setSplitConfig] = useState<SplitConfig>({
    train_ratio: 0.7,
    val_ratio: 0.2,
    test_ratio: 0.1,
    shuffle: true,
  });

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      loadDatasets(selectedModel.id);
    }
  }, [selectedModel]);

  const loadModels = async () => {
    try {
      const modelList = await invoke<Model[]>('db_list_models');
      setModels(modelList);
      if (modelList.length > 0) {
        setSelectedModel(modelList[0]);
      }
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  const loadDatasets = async (modelId: string) => {
    try {
      const datasetList = await invoke<DatasetInfo[]>('list_datasets_for_model', { modelId });
      setDatasets(datasetList);
    } catch (error) {
      console.error('Failed to load datasets:', error);
      setDatasets([]);
    }
  };

  const selectPath = async () => {
    try {
      const paths = await invoke<string[]>('select_dataset_path', { allowMultiple: false });
      if (paths && paths.length > 0) {
        const path = paths[0];
        setUploadPath(path);
        
        // Auto-validate (mit Modell-ID)
        if (selectedModel) {
          const result = await invoke<DatasetValidation>('validate_dataset_path', { 
            path,
            modelId: selectedModel.id 
          });
          setValidation(result);
        }
        
        // Auto-set name from folder
        const folderName = path.split('/').pop() || 'Dataset';
        setUploadName(folderName);
      }
    } catch (error: any) {
      console.error('Failed to select path:', error);
    }
  };

  const handleUpload = async () => {
    if (!selectedModel || !uploadPath || !uploadName) return;
    
    setLoading(true);
    try {
      await invoke('import_dataset', {
        modelId: selectedModel.id,
        sourcePath: uploadPath,
        datasetName: uploadName,
        datasetType: validation?.detected_type || 'mixed',
        splitType,
      });
      
      // Reset & reload
      setShowUploadDialog(false);
      setUploadPath('');
      setUploadName('');
      setValidation(null);
      await loadDatasets(selectedModel.id);
    } catch (error: any) {
      alert('Upload fehlgeschlagen: ' + error);
    } finally {
      setLoading(false);
    }
  };

  const handleAutoSplit = async () => {
    if (!selectedModel || !selectedDatasetForSplit) return;
    
    setLoading(true);
    try {
      await invoke('auto_split_dataset', {
        modelId: selectedModel.id,
        datasetId: selectedDatasetForSplit.id,
        config: splitConfig,
      });
      
      setShowSplitDialog(false);
      setSelectedDatasetForSplit(null);
      await loadDatasets(selectedModel.id);
    } catch (error: any) {
      alert('Split fehlgeschlagen: ' + error);
    } finally {
      setLoading(false);
    }
  };

  const toggleHardExample = async (datasetId: string) => {
    if (!selectedModel) return;
    
    try {
      await invoke('toggle_hard_example', {
        modelId: selectedModel.id,
        datasetId,
      });
      await loadDatasets(selectedModel.id);
    } catch (error: any) {
      console.error('Toggle failed:', error);
    }
  };

  const deleteDataset = async (datasetId: string) => {
    if (!selectedModel) return;
    if (!confirm('Dataset wirklich löschen?')) return;
    
    try {
      await invoke('delete_dataset', {
        modelId: selectedModel.id,
        datasetId,
      });
      await loadDatasets(selectedModel.id);
    } catch (error: any) {
      alert('Löschen fehlgeschlagen: ' + error);
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'image': return <Image className="w-5 h-5" />;
      case 'video': return <Video className="w-5 h-5" />;
      case 'text': return <FileText className="w-5 h-5" />;
      case 'audio': return <Music className="w-5 h-5" />;
      default: return <Layers className="w-5 h-5" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'unused': return 'bg-gray-500/20 text-gray-300';
      case 'inuse': return 'bg-blue-500/20 text-blue-300';
      case 'used': return 'bg-green-500/20 text-green-300';
      case 'hardexample': return 'bg-red-500/20 text-red-300';
      default: return 'bg-gray-500/20 text-gray-300';
    }
  };

  const getSplitColor = (split: string) => {
    switch (split) {
      case 'train': return 'bg-purple-500/20 text-purple-300';
      case 'val': return 'bg-yellow-500/20 text-yellow-300';
      case 'test': return 'bg-pink-500/20 text-pink-300';
      default: return 'bg-gray-500/20 text-gray-300';
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  };

  // Group datasets by split type
  const trainDatasets = datasets.filter(d => d.split_type === 'train');
  const valDatasets = datasets.filter(d => d.split_type === 'val');
  const testDatasets = datasets.filter(d => d.split_type === 'test');

  return (
    <div>
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Datensätze</h2>
          <p className="text-gray-400">Verwalte Trainings-, Validierungs- und Test-Daten</p>
        </div>
        <button
          onClick={() => setShowUploadDialog(true)}
          disabled={!selectedModel}
          className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all shadow-lg disabled:opacity-50"
        >
          <Upload className="w-5 h-5" />
          <span>Dataset hochladen</span>
        </button>
      </div>

      {/* Model Selection */}
      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-8">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Modell auswählen
        </label>
        <select
          value={selectedModel?.id || ''}
          onChange={(e) => {
            const model = models.find(m => m.id === e.target.value);
            setSelectedModel(model || null);
          }}
          className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
        >
          {models.map(model => (
            <option key={model.id} value={model.id}>
              {model.name} {model.base_model && `(${model.base_model})`}
            </option>
          ))}
        </select>
      </div>

      {/* Dataset Statistics */}
      {selectedModel && (
        <div className="grid grid-cols-4 gap-6 mb-8">
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
            <div className="text-gray-400 text-sm mb-2">Gesamt</div>
            <div className="text-3xl font-bold text-white">{datasets.length}</div>
          </div>
          <div className="bg-purple-500/10 backdrop-blur-lg rounded-xl p-6 border border-purple-500/20">
            <div className="text-purple-300 text-sm mb-2">Training</div>
            <div className="text-3xl font-bold text-white">{trainDatasets.length}</div>
          </div>
          <div className="bg-yellow-500/10 backdrop-blur-lg rounded-xl p-6 border border-yellow-500/20">
            <div className="text-yellow-300 text-sm mb-2">Validierung</div>
            <div className="text-3xl font-bold text-white">{valDatasets.length}</div>
          </div>
          <div className="bg-pink-500/10 backdrop-blur-lg rounded-xl p-6 border border-pink-500/20">
            <div className="text-pink-300 text-sm mb-2">Test</div>
            <div className="text-3xl font-bold text-white">{testDatasets.length}</div>
          </div>
        </div>
      )}

      {/* Datasets List */}
      {selectedModel ? (
        datasets.length === 0 ? (
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-12 border border-white/20 text-center">
            <div className="text-6xl mb-4">📊</div>
            <h3 className="text-2xl font-bold text-white mb-2">Keine Datensätze</h3>
            <p className="text-gray-300 mb-6">
              Lade deinen ersten Datensatz für {selectedModel.name} hoch.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20 hover:border-purple-500/50 transition-all"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start space-x-4">
                    <div className="p-3 bg-white/5 rounded-lg">
                      {getTypeIcon(dataset.dataset_type)}
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-white mb-1">{dataset.name}</h3>
                      <div className="flex items-center space-x-2 text-sm text-gray-400">
                        <span>{dataset.file_count} Dateien</span>
                        <span>•</span>
                        <span>{formatBytes(dataset.total_size_bytes)}</span>
                        <span>•</span>
                        <span className="capitalize">{dataset.dataset_type}</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => {
                        setSelectedDatasetForSplit(dataset);
                        setShowSplitDialog(true);
                      }}
                      className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                      title="Dataset splitten"
                    >
                      <Split className="w-5 h-5 text-gray-400 hover:text-purple-400" />
                    </button>
                    <button
                      onClick={() => toggleHardExample(dataset.id)}
                      className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                      title="Als Hard Example markieren"
                    >
                      <Filter className={`w-5 h-5 ${dataset.is_hard_example ? 'text-red-400' : 'text-gray-400'}`} />
                    </button>
                    <button
                      onClick={() => deleteDataset(dataset.id)}
                      className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
                    >
                      <Trash2 className="w-5 h-5 text-gray-400 hover:text-red-400" />
                    </button>
                  </div>
                </div>

                <div className="flex items-center space-x-2 mb-3">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSplitColor(dataset.split_type)}`}>
                    {dataset.split_type.toUpperCase()}
                  </span>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(dataset.status)}`}>
                    {dataset.status === 'unused' && 'Unbenutzt'}
                    {dataset.status === 'inuse' && 'In Verwendung'}
                    {dataset.status === 'used' && 'Verwendet'}
                    {dataset.status === 'hardexample' && 'Hard Example'}
                  </span>
                  {dataset.is_hard_example && (
                    <span className="px-3 py-1 rounded-full text-sm font-medium bg-red-500/20 text-red-300">
                      Schwierige Beispiele
                    </span>
                  )}
                </div>

                <div className="text-sm text-gray-400">
                  <div className="flex items-center space-x-4">
                    <span>Formate: {dataset.formats.join(', ')}</span>
                    {dataset.last_used_at && (
                      <>
                        <span>•</span>
                        <span>Zuletzt verwendet: {new Date(dataset.last_used_at).toLocaleDateString('de-DE')}</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )
      ) : (
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-12 border border-white/20 text-center">
          <div className="text-6xl mb-4">🎯</div>
          <h3 className="text-2xl font-bold text-white mb-2">Kein Modell ausgewählt</h3>
          <p className="text-gray-300">
            Wähle zuerst ein Modell aus, um Datensätze zu verwalten.
          </p>
        </div>
      )}

      {/* Upload Dialog */}
      {showUploadDialog && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-900 rounded-2xl p-8 max-w-2xl w-full mx-4 border border-white/20">
            <h3 className="text-2xl font-bold text-white mb-6">Dataset hochladen</h3>

            <div className="space-y-6">
              {/* Path Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Ordner auswählen
                </label>
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={uploadPath}
                    readOnly
                    placeholder="Kein Ordner ausgewählt"
                    className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white"
                  />
                  <button
                    onClick={selectPath}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center space-x-2"
                  >
                    <Folder className="w-5 h-5" />
                    <span>Auswählen</span>
                  </button>
                </div>
              </div>

              {/* Validation Result */}
              {validation && (
                <div className={`p-4 rounded-lg border ${
                  validation.is_valid 
                    ? 'bg-green-500/10 border-green-500/20' 
                    : 'bg-red-500/10 border-red-500/20'
                }`}>
                  <div className="flex items-start space-x-2 mb-2">
                    {validation.is_valid ? (
                      <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    )}
                    <div className="flex-1">
                      <div className="font-semibold text-white mb-1">
                        {validation.is_valid ? 'Dataset gültig ✅' : 'Dataset ungültig ❌'}
                      </div>
                      <div className="text-sm text-gray-300 space-y-1">
                        <div>Typ: <span className="font-medium">{validation.detected_type}</span> • {validation.file_count} Dateien</div>
                        <div>Gefundene Formate: <span className="font-medium">{validation.formats.join(', ')}</span></div>
                        <div className={validation.model_compatible ? 'text-green-300' : 'text-red-300'}>
                          {validation.model_compatible ? '✅' : '❌'} Modell-Kompatibilität: {validation.model_compatible ? 'Kompatibel' : 'Nicht kompatibel'}
                        </div>
                        <div>Erwartete Formate: <span className="font-medium text-purple-300">{validation.expected_formats.join(', ')}</span></div>
                      </div>
                    </div>
                  </div>

                  {validation.errors.length > 0 && (
                    <div className="mt-3 space-y-1 border-t border-red-500/20 pt-3">
                      <div className="text-sm font-semibold text-red-300 mb-1">Fehler:</div>
                      {validation.errors.map((error, i) => (
                        <div key={i} className="text-sm text-red-300">❌ {error}</div>
                      ))}
                    </div>
                  )}

                  {validation.warnings.length > 0 && (
                    <div className="mt-3 space-y-1 border-t border-yellow-500/20 pt-3">
                      <div className="text-sm font-semibold text-yellow-300 mb-1">Warnungen:</div>
                      {validation.warnings.map((warning, i) => (
                        <div key={i} className="text-sm text-yellow-300">⚠️ {warning}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Name */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Dataset-Name
                </label>
                <input
                  type="text"
                  value={uploadName}
                  onChange={(e) => setUploadName(e.target.value)}
                  placeholder="z.B. Training Data V1"
                  className="w-full px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>

              {/* Split Type */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Split-Typ
                </label>
                <div className="grid grid-cols-3 gap-4">
                  {['train', 'val', 'test'].map((type) => (
                    <button
                      key={type}
                      onClick={() => setSplitType(type as any)}
                      className={`px-4 py-2 rounded-lg border transition-colors ${
                        splitType === type
                          ? 'bg-purple-600 border-purple-500 text-white'
                          : 'bg-white/5 border-white/10 text-gray-300 hover:bg-white/10'
                      }`}
                    >
                      {type.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end space-x-4 mt-8">
              <button
                onClick={() => {
                  setShowUploadDialog(false);
                  setUploadPath('');
                  setUploadName('');
                  setValidation(null);
                }}
                className="px-6 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg transition-colors"
              >
                Abbrechen
              </button>
              <button
                onClick={handleUpload}
                disabled={!validation?.is_valid || !uploadName || loading}
                className="px-6 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50"
              >
                {loading ? 'Wird hochgeladen...' : 'Hochladen'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Split Dialog */}
      {showSplitDialog && selectedDatasetForSplit && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-900 rounded-2xl p-8 max-w-2xl w-full mx-4 border border-white/20">
            <h3 className="text-2xl font-bold text-white mb-2">Dataset aufteilen</h3>
            <p className="text-gray-400 mb-6">
              Teile {selectedDatasetForSplit.name} in Train/Val/Test auf
            </p>

            <div className="space-y-6">
              {/* Ratios */}
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm text-gray-300 mb-2">
                    <span>Training</span>
                    <span>{(splitConfig.train_ratio * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={splitConfig.train_ratio}
                    onChange={(e) => setSplitConfig({...splitConfig, train_ratio: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-sm text-gray-300 mb-2">
                    <span>Validierung</span>
                    <span>{(splitConfig.val_ratio * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={splitConfig.val_ratio}
                    onChange={(e) => setSplitConfig({...splitConfig, val_ratio: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-sm text-gray-300 mb-2">
                    <span>Test</span>
                    <span>{(splitConfig.test_ratio * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={splitConfig.test_ratio}
                    onChange={(e) => setSplitConfig({...splitConfig, test_ratio: parseFloat(e.target.value)})}
                    className="w-full"
                  />
                </div>
              </div>

              {/* Shuffle */}
              <label className="flex items-center space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={splitConfig.shuffle}
                  onChange={(e) => setSplitConfig({...splitConfig, shuffle: e.target.checked})}
                  className="w-5 h-5 rounded"
                />
                <span className="text-white">Daten mischen (empfohlen)</span>
              </label>

              {/* Preview */}
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <div className="text-sm text-gray-300 space-y-1">
                  <div>Training: ~{Math.round(selectedDatasetForSplit.file_count * splitConfig.train_ratio)} Dateien</div>
                  <div>Validierung: ~{Math.round(selectedDatasetForSplit.file_count * splitConfig.val_ratio)} Dateien</div>
                  <div>Test: ~{Math.round(selectedDatasetForSplit.file_count * splitConfig.test_ratio)} Dateien</div>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex justify-end space-x-4 mt-8">
              <button
                onClick={() => {
                  setShowSplitDialog(false);
                  setSelectedDatasetForSplit(null);
                }}
                className="px-6 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg transition-colors"
              >
                Abbrechen
              </button>
              <button
                onClick={handleAutoSplit}
                disabled={loading}
                className="px-6 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50"
              >
                {loading ? 'Wird aufgeteilt...' : 'Aufteilen'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
