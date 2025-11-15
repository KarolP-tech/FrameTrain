import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, XCircle } from 'lucide-react';
import { open } from '@tauri-apps/api/dialog';
import { readTextFile } from '@tauri-apps/api/fs';

export default function DatasetUpload() {
  const [dataset, setDataset] = useState<any>(null);
  const [preview, setPreview] = useState<any[]>([]);
  const [isValid, setIsValid] = useState<boolean | null>(null);

  const handleFileSelect = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [{
          name: 'Data',
          extensions: ['csv', 'json', 'txt']
        }]
      });

      if (selected && typeof selected === 'string') {
        const content = await readTextFile(selected);
        
        // Einfache CSV-Parsing-Simulation
        const lines = content.split('\n').slice(0, 10);
        const previewData = lines.map(line => line.split(','));
        
        setDataset({ path: selected, name: selected.split('/').pop() });
        setPreview(previewData);
        setIsValid(true);
      }
    } catch (error) {
      console.error('Fehler beim Laden der Datei:', error);
      setIsValid(false);
    }
  };

  return (
    <div>
      <h2 className="text-3xl font-bold text-white mb-8">Datensatz hochladen</h2>

      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 border border-white/20 mb-8">
        <button
          onClick={handleFileSelect}
          className="w-full border-2 border-dashed border-white/20 rounded-xl p-12 hover:border-purple-500/50 transition-all cursor-pointer group"
        >
          <div className="text-center">
            <Upload className="w-16 h-16 text-gray-400 group-hover:text-purple-400 mx-auto mb-4" />
            <p className="text-xl font-semibold text-white mb-2">
              Klicke hier, um eine Datei auszuwählen
            </p>
            <p className="text-gray-400">
              Unterstützt: CSV, JSON, TXT
            </p>
          </div>
        </button>
      </div>

      {dataset && (
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20 mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <FileText className="w-6 h-6 text-purple-400" />
              <div>
                <h3 className="text-lg font-bold text-white">{dataset.name}</h3>
                <p className="text-sm text-gray-400">{dataset.path}</p>
              </div>
            </div>
            {isValid !== null && (
              <div className="flex items-center space-x-2">
                {isValid ? (
                  <>
                    <CheckCircle className="w-6 h-6 text-green-400" />
                    <span className="text-green-400 font-medium">Gültig</span>
                  </>
                ) : (
                  <>
                    <XCircle className="w-6 h-6 text-red-400" />
                    <span className="text-red-400 font-medium">Fehler</span>
                  </>
                )}
              </div>
            )}
          </div>

          {preview.length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-semibold text-gray-300 mb-2">Vorschau (erste 10 Zeilen):</h4>
              <div className="bg-black/30 rounded-lg p-4 overflow-auto max-h-64">
                <table className="w-full text-sm text-gray-300">
                  <tbody>
                    {preview.map((row, i) => (
                      <tr key={i} className="border-b border-white/10">
                        {row.map((cell, j) => (
                          <td key={j} className="py-2 px-3">{cell}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
