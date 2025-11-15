import React from 'react';
import { GitBranch, Calendar, Settings } from 'lucide-react';

interface VersionManagerProps {
  selectedModel: any;
}

export default function VersionManager({ selectedModel }: VersionManagerProps) {
  // Mock-Versionen
  const versions = [
    {
      id: '1',
      version: 3,
      date: '2025-11-10',
      parameters: { epochs: 10, batch_size: 32, learning_rate: 0.001 },
      metrics: { loss: 0.3, accuracy: 0.92 },
      status: 'completed'
    },
    {
      id: '2',
      version: 2,
      date: '2025-11-09',
      parameters: { epochs: 8, batch_size: 32, learning_rate: 0.002 },
      metrics: { loss: 0.5, accuracy: 0.87 },
      status: 'completed'
    },
    {
      id: '3',
      version: 1,
      date: '2025-11-08',
      parameters: { epochs: 5, batch_size: 16, learning_rate: 0.001 },
      metrics: { loss: 0.8, accuracy: 0.75 },
      status: 'completed'
    },
  ];

  return (
    <div>
      <h2 className="text-3xl font-bold text-white mb-8">Versionsverwaltung</h2>

      {!selectedModel ? (
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-12 border border-white/20 text-center">
          <div className="text-6xl mb-4">📚</div>
          <h3 className="text-2xl font-bold text-white mb-2">Kein Modell ausgewählt</h3>
          <p className="text-gray-300">
            Wähle ein Modell aus, um dessen Versionen zu sehen.
          </p>
        </div>
      ) : (
        <>
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-8">
            <h3 className="text-xl font-bold text-white mb-2">{selectedModel.name}</h3>
            <p className="text-gray-300">{versions.length} Versionen verfügbar</p>
          </div>

          <div className="space-y-4">
            {versions.map((version, index) => (
              <div
                key={version.id}
                className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 hover:border-purple-500/50 transition-all"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-start space-x-4">
                    <div className="p-3 bg-purple-600/20 rounded-lg">
                      <GitBranch className="w-6 h-6 text-purple-400" />
                    </div>
                    <div>
                      <div className="flex items-center space-x-3 mb-2">
                        <h3 className="text-xl font-bold text-white">
                          Version {version.version}
                        </h3>
                        {index === 0 && (
                          <span className="px-3 py-1 bg-green-500/20 text-green-300 text-sm font-medium rounded-full">
                            Latest
                          </span>
                        )}
                        <span className="px-3 py-1 bg-blue-500/20 text-blue-300 text-sm font-medium rounded-full">
                          {version.status}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 text-sm text-gray-400">
                        <span className="flex items-center">
                          <Calendar className="w-4 h-4 mr-1" />
                          {new Date(version.date).toLocaleDateString('de-DE')}
                        </span>
                      </div>
                    </div>
                  </div>

                  <button className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors">
                    Laden
                  </button>
                </div>

                <div className="grid grid-cols-2 gap-6 mt-6">
                  <div>
                    <div className="flex items-center space-x-2 mb-3">
                      <Settings className="w-4 h-4 text-gray-400" />
                      <h4 className="text-sm font-semibold text-gray-300">Parameter</h4>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Epochs:</span>
                        <span className="text-white font-medium">{version.parameters.epochs}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Batch Size:</span>
                        <span className="text-white font-medium">{version.parameters.batch_size}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Learning Rate:</span>
                        <span className="text-white font-medium">{version.parameters.learning_rate}</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center space-x-2 mb-3">
                      <GitBranch className="w-4 h-4 text-gray-400" />
                      <h4 className="text-sm font-semibold text-gray-300">Metriken</h4>
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Loss:</span>
                        <span className="text-white font-medium">{version.metrics.loss.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Accuracy:</span>
                        <span className="text-white font-medium">{(version.metrics.accuracy * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
