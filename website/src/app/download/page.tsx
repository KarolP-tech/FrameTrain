'use client';

import { useState, useEffect } from 'react';
import { Download, Loader2, CheckCircle, XCircle, Terminal, Globe } from 'lucide-react';

type Platform = 'windows' | 'mac' | 'linux';

interface DownloadInfo {
  version: string;
  platform: Platform;
  download_url: string;
  filename: string;
  size_mb: string;
}

export default function DownloadPage() {
  const [platform, setPlatform] = useState<Platform>('windows');
  const [apiKey, setApiKey] = useState('');
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadInfo, setDownloadInfo] = useState<DownloadInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Detect user's platform
  useEffect(() => {
    const userAgent = window.navigator.userAgent.toLowerCase();
    if (userAgent.includes('win')) {
      setPlatform('windows');
    } else if (userAgent.includes('mac')) {
      setPlatform('mac');
    } else if (userAgent.includes('linux')) {
      setPlatform('linux');
    }
  }, []);

  const handleDownload = async () => {
    if (!apiKey.trim()) {
      setError('Bitte gib deinen API-Key ein');
      return;
    }

    setIsDownloading(true);
    setError(null);
    setSuccessMessage(null);
    setDownloadInfo(null);

    try {
      // Get download info from API
      const response = await fetch(
        `/api/download-app?platform=${platform}&key=${encodeURIComponent(apiKey)}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || data.error || 'Download fehlgeschlagen');
      }

      setDownloadInfo(data);
      setSuccessMessage('Download wird gestartet...');

      // Start download by opening the URL
      window.location.href = data.download_url;

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ein Fehler ist aufgetreten');
    } finally {
      setIsDownloading(false);
    }
  };

  const getPlatformIcon = (p: Platform) => {
    const icons = {
      windows: '🪟',
      mac: '🍎',
      linux: '🐧'
    };
    return icons[p];
  };

  const getPlatformName = (p: Platform) => {
    const names = {
      windows: 'Windows',
      mac: 'macOS',
      linux: 'Linux'
    };
    return names[p];
  };

  const getInstallInstructions = () => {
    const instructions = {
      windows: [
        'Download die .msi Datei',
        'Doppelklick auf die Datei',
        'Folge dem Installations-Wizard',
        'FrameTrain wird automatisch gestartet'
      ],
      mac: [
        'Download die .dmg Datei',
        'Öffne die heruntergeladene Datei',
        'Ziehe FrameTrain in den Applications Ordner',
        'Starte FrameTrain aus Applications'
      ],
      linux: [
        'Download die .AppImage Datei',
        'Mache sie ausführbar: chmod +x FrameTrain*.AppImage',
        'Führe aus: ./FrameTrain*.AppImage',
        'Optional: Erstelle Desktop-Verknüpfung'
      ]
    };
    return instructions[platform];
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Navigation */}
      <nav className="border-b border-white/10 bg-white/5 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg" />
              <span className="text-xl font-bold text-white">FrameTrain</span>
            </div>
            <div className="flex items-center space-x-4">
              <a href="/" className="text-gray-300 hover:text-white transition-colors">
                Home
              </a>
              <a href="/docs" className="text-gray-300 hover:text-white transition-colors">
                Docs
              </a>
              <a href="/dashboard" className="text-gray-300 hover:text-white transition-colors">
                Dashboard
              </a>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-4 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">
            Download FrameTrain
          </h1>
          <p className="text-xl text-gray-300">
            Lokales ML-Training für Windows, macOS und Linux
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 p-8 mb-8">
          {/* Platform Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-200 mb-3">
              Wähle deine Plattform
            </label>
            <div className="grid grid-cols-3 gap-4">
              {(['windows', 'mac', 'linux'] as Platform[]).map((p) => (
                <button
                  key={p}
                  onClick={() => setPlatform(p)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    platform === p
                      ? 'border-purple-500 bg-purple-500/20'
                      : 'border-white/20 bg-white/5 hover:border-white/40'
                  }`}
                >
                  <div className="text-4xl mb-2">{getPlatformIcon(p)}</div>
                  <div className="text-white font-medium">{getPlatformName(p)}</div>
                </button>
              ))}
            </div>
          </div>

          {/* API Key Input */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-200 mb-2">
              API-Key
            </label>
            <input
              type="text"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Gib deinen API-Key ein..."
              className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <p className="mt-2 text-sm text-gray-400">
              Du erhältst deinen API-Key nach dem Kauf auf der{' '}
              <a href="/payment" className="text-purple-400 hover:text-purple-300">
                Payment-Seite
              </a>
            </p>
          </div>

          {/* Download Button */}
          <button
            onClick={handleDownload}
            disabled={isDownloading || !apiKey.trim()}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-4 rounded-lg font-semibold hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {isDownloading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Wird heruntergeladen...</span>
              </>
            ) : (
              <>
                <Download className="w-5 h-5" />
                <span>Download für {getPlatformName(platform)}</span>
              </>
            )}
          </button>

          {/* Success Message */}
          {successMessage && (
            <div className="mt-4 p-4 bg-green-500/20 border border-green-500/50 rounded-lg flex items-start space-x-3">
              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-green-300 font-medium">{successMessage}</p>
                {downloadInfo && (
                  <p className="text-green-400 text-sm mt-1">
                    Version: {downloadInfo.version} • Größe: {downloadInfo.size_mb} MB
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-start space-x-3">
              <XCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-red-300">{error}</p>
            </div>
          )}
        </div>

        {/* Installation Instructions */}
        <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 p-8 mb-8">
          <h2 className="text-2xl font-bold text-white mb-4 flex items-center space-x-2">
            <Globe className="w-6 h-6" />
            <span>Installationsanleitung</span>
          </h2>
          <ol className="space-y-3">
            {getInstallInstructions().map((step, index) => (
              <li key={index} className="flex items-start space-x-3">
                <span className="flex-shrink-0 w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
                  {index + 1}
                </span>
                <span className="text-gray-300">{step}</span>
              </li>
            ))}
          </ol>
        </div>

        {/* CLI Alternative */}
        <div className="bg-white/10 backdrop-blur-md rounded-2xl border border-white/20 p-8">
          <h2 className="text-2xl font-bold text-white mb-4 flex items-center space-x-2">
            <Terminal className="w-6 h-6" />
            <span>Installation per Terminal</span>
          </h2>
          <p className="text-gray-300 mb-4">
            Für Entwickler: Installiere FrameTrain direkt über die Kommandozeile
          </p>
          <div className="bg-black/30 rounded-lg p-4 font-mono text-sm">
            <div className="text-gray-400 mb-2"># Installiere die CLI</div>
            <div className="text-green-400 mb-4">pip install frametrain-cli</div>
            
            <div className="text-gray-400 mb-2"># Installiere die Desktop-App</div>
            <div className="text-green-400 mb-4">frametrain install --key YOUR_API_KEY</div>
            
            <div className="text-gray-400 mb-2"># Starte FrameTrain</div>
            <div className="text-green-400">frametrain start</div>
          </div>
        </div>

        {/* Support */}
        <div className="mt-8 text-center text-gray-400">
          <p>
            Probleme beim Download?{' '}
            <a href="mailto:support@frametrain.ai" className="text-purple-400 hover:text-purple-300">
              Kontaktiere unseren Support
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
