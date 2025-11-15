import React, { useState } from 'react';
import { Plus, Download, Trash2, ExternalLink } from 'lucide-react';

interface ModelManagerProps {
  onSelectModel: (model: any) => void;
}

export default function ModelManager({ onSelectModel }: ModelManagerProps) {
  const [models, setModels] = useState<any[]>([]);
  const [showImport, setShowImport] = useState(false);
  const [huggingFaceUrl, setHuggingFaceUrl] = useState('');

  const importFromHuggingFace = async () => {
    // Hier würde die HuggingFace API-Integration erfolgen
    const newModel = {
      id: Date.now().toString(),
      name: huggingFaceUrl.split('/').pop() || 'Neues Modell',
      source: 'huggingface',
      url: huggingFaceUrl,
      createdAt: new Date().toISOString(),
      status: 'ready',
    };
    
    setModels([...models, newModel]);
    setHuggingFaceUrl('');
    setShowImport(false);
  };

  const deleteModel = (id: string) => {
    setModels(models.filter(m => m.id !== id));
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

      {showImport && (
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-8">
          <h3 className="text-xl font-bold text-white mb-4">Von HuggingFace importieren</h3>
          <div className="flex space-x-4">
            <input
              type="text"
              value={huggingFaceUrl}
              onChange={(e) => setHuggingFaceUrl(e.target.value)}
              placeholder="z.B. bert-base-uncased oder https://huggingface.co/..."
              className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <button
              onClick={importFromHuggingFace}
              disabled={!huggingFaceUrl}
              className="px-6 py-3 bg-purple-600 text-white font-semibold rounded-lg hover:bg-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Importieren
            </button>
          </div>
        </div>
      )}

      {models.length === 0 ? (
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-12 border border-white/20 text-center">
          <div className="text-6xl mb-4">📦</div>
          <h3 className="text-2xl font-bold text-white mb-2">Noch keine Modelle</h3>
          <p className="text-gray-300 mb-6">
            Importiere dein erstes Modell von HuggingFace oder lade eine lokale Datei hoch.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {models.map((model) => (
            <div
              key={model.id}
              className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 hover:border-purple-500/50 transition-all cursor-pointer group"
              onClick={() => onSelectModel(model)}
            >
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-xl font-bold text-white mb-2">{model.name}</h3>
                  <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-500/20 text-green-300">
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
              
              <div className="text-sm text-gray-400 space-y-1">
                <p>Quelle: {model.source}</p>
                <p>Erstellt: {new Date(model.createdAt).toLocaleDateString('de-DE')}</p>
              </div>

              {model.url && (
                <a
                  href={model.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  className="mt-4 inline-flex items-center text-purple-400 hover:text-purple-300 text-sm"
                >
                  <ExternalLink className="w-4 h-4 mr-2" />
                  HuggingFace ansehen
                </a>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
