import { useState, useEffect, useRef } from 'react';
import { Plus, Upload, Folder, Trash2, FileCode, HardDrive, Download } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';

interface Model {
  id: string;
  name: string;
  description: string | null;
  base_model: string | null;
  model_path: string | null;
  status: string;
  created_at: string;
  updated_at: string;
}

interface ModelImportInfo {
  name: string;
  path: string;
  size_bytes: number;
  file_count: number;
  model_type: string;
}

interface ModelManagerProps {
  onSelectModel: (model: Model) => void;
}

export default function ModelManager({ onSelectModel }: ModelManagerProps) {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelImportInfo | null>(null);
  const [importName, setImportName] = useState('');
  const [importDescription, setImportDescription] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Lade Modelle beim Start
  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const result = await invoke<Model[]>('db_list_models');
      setModels(result);
    } catch (error) {
      console.error('Fehler beim Laden der Modelle:', error);
    }
  };

  // Öffne Ordner-Auswahl-Dialog
  const selectFolder = async () => {
    try {
      setLoading(true);
      const folderPath = await invoke<string>('select_model_folder');
      await validateAndSetFolder(folderPath);
    } catch (error: any) {
      if (error !== 'Keine Ordner ausgewählt') {
        alert(`Fehler: ${error}`);
      }
    } finally {
      setLoading(false);
    }
  };

  // Validiere Ordner und zeige Infos
  const validateAndSetFolder = async (folderPath: string) => {
    try {
      setLoading(true);
      const info = await invoke<ModelImportInfo>('validate_model_folder', {
        folderPath,
      });
      
      setSelectedFolder(folderPath);
      setModelInfo(info);
      setImportName(info.name);
      
    } catch (error: any) {
      alert(`Ungültiger Modell-Ordner: ${error}`);
      setSelectedFolder(null);
      setModelInfo(null);
    } finally {
      setLoading(false);
    }
  };

  // Importiere Modell in die App
  const importModel = async () => {
    if (!selectedFolder || !importName) {
      alert('Bitte wähle einen Ordner und gib einen Namen ein');
      return;
    }

    try {
      setLoading(true);
      
      const newModel = await invoke<Model>('import_model', {
        sourcePath: selectedFolder,
        modelName: importName,
        description: importDescription || null,
      });

      setModels([...models, newModel]);
      
      // Reset
      setShowImport(false);
      setSelectedFolder(null);
      setModelInfo(null);
      setImportName('');
      setImportDescription('');
      
      alert('✅ Modell erfolgreich importiert!');
      
    } catch (error: any) {
      alert(`Fehler beim Import: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // Lösche Modell
  const deleteModel = async (id: string) => {
    if (!confirm('Modell wirklich löschen?')) return;

    try {
      await invoke('db_delete_model', { id });
      setModels(models.filter((m) => m.id !== id));
    } catch (error: any) {
      alert(`Fehler beim Löschen: ${error}`);
    }
  };

  // Formatiere Dateigröße
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  // Drag & Drop Handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    // Hinweis: Tauri unterstützt Drag & Drop von Ordnern derzeit nicht direkt
    // Nutzer müssen den Dialog verwenden
    alert('Bitte nutze den "Ordner auswählen" Button, um ein Modell zu importieren.');
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-bold text-white">Modellverwaltung</h2>
        <button
          onClick={() => setShowImport(!showImport)}
          className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all shadow-lg"
        >
          <Plus className="w-5 h-5" />
          <span>Modell importieren</span>
        </button>
      </div>

      {/* Import Panel */}
      {showImport && (
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-8">
          <h3 className="text-xl font-bold text-white mb-4">Lokales Modell importieren</h3>
          
          {/* Drag & Drop Area */}
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-all ${
              dragActive
                ? 'border-purple-500 bg-purple-500/10'
                : 'border-white/20 bg-white/5'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <Upload className="w-16 h-16 mx-auto mb-4 text-gray-400" />
            <p className="text-white mb-2 font-semibold">
              Modell-Ordner auswählen
            </p>
            <p className="text-gray-400 text-sm mb-4">
              Wähle einen Ordner mit deinem trainierten Modell
            </p>
            <button
              onClick={selectFolder}
              disabled={loading}
              className="inline-flex items-center space-x-2 px-6 py-3 bg-purple-600 text-white font-semibold rounded-lg hover:bg-purple-700 transition-all disabled:opacity-50"
            >
              <Folder className="w-5 h-5" />
              <span>{loading ? 'Lädt...' : 'Ordner auswählen'}</span>
            </button>
          </div>

          {/* Modell Info */}
          {modelInfo && (
            <div className="mt-6 space-y-4">
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-3">Modell-Informationen</h4>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <p className="text-gray-400">Typ:</p>
                    <p className="text-white font-medium">{modelInfo.model_type}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Dateien:</p>
                    <p className="text-white font-medium">{modelInfo.file_count}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Größe:</p>
                    <p className="text-white font-medium">{formatFileSize(modelInfo.size_bytes)}</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Pfad:</p>
                    <p className="text-white font-medium text-xs truncate">{modelInfo.path}</p>
                  </div>
                </div>
              </div>

              {/* Name & Beschreibung */}
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Modell-Name *
                  </label>
                  <input
                    type="text"
                    value={importName}
                    onChange={(e) => setImportName(e.target.value)}
                    placeholder="z.B. BERT Base German"
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Beschreibung (optional)
                  </label>
                  <textarea
                    value={importDescription}
                    onChange={(e) => setImportDescription(e.target.value)}
                    placeholder="Füge eine Beschreibung hinzu..."
                    rows={3}
                    className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
              </div>

              {/* Import Button */}
              <div className="flex space-x-3">
                <button
                  onClick={importModel}
                  disabled={loading || !importName}
                  className="flex-1 px-6 py-3 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Importiere...' : 'Modell importieren'}
                </button>
                <button
                  onClick={() => {
                    setSelectedFolder(null);
                    setModelInfo(null);
                    setImportName('');
                    setImportDescription('');
                  }}
                  className="px-6 py-3 bg-white/10 text-white font-semibold rounded-lg hover:bg-white/20 transition-all"
                >
                  Abbrechen
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Modell-Liste */}
      {models.length === 0 ? (
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-12 border border-white/20 text-center">
          <div className="text-6xl mb-4">📦</div>
          <h3 className="text-2xl font-bold text-white mb-2">Noch keine Modelle</h3>
          <p className="text-gray-300 mb-6">
            Importiere dein erstes lokales Modell, um zu beginnen.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {models.map((model) => (
            <div
              key={model.id}
              className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 hover:border-purple-500/50 transition-all cursor-pointer group"
              onClick={() => onSelectModel(model)}
            >
              <div className="flex justify-between items-start mb-4">
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-white mb-2 truncate">{model.name}</h3>
                  <span
                    className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                      model.status === 'imported'
                        ? 'bg-green-500/20 text-green-300'
                        : model.status === 'training'
                        ? 'bg-blue-500/20 text-blue-300'
                        : 'bg-gray-500/20 text-gray-300'
                    }`}
                  >
                    {model.status}
                  </span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteModel(model.id);
                  }}
                  className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
                >
                  <Trash2 className="w-5 h-5 text-gray-400 hover:text-red-400" />
                </button>
              </div>

              {model.description && (
                <p className="text-sm text-gray-400 mb-3 line-clamp-2">{model.description}</p>
              )}

              <div className="space-y-2 text-sm text-gray-400">
                {model.base_model && (
                  <div className="flex items-center space-x-2">
                    <FileCode className="w-4 h-4" />
                    <span>{model.base_model}</span>
                  </div>
                )}
                {model.model_path && (
                  <div className="flex items-center space-x-2">
                    <HardDrive className="w-4 h-4" />
                    <span className="text-xs truncate">{model.model_path}</span>
                  </div>
                )}
                <div className="text-xs text-gray-500">
                  Erstellt: {new Date(model.created_at).toLocaleDateString('de-DE')}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
