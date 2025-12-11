import { useEffect, useState } from 'react';
import { check } from '@tauri-apps/plugin-updater';
import { relaunch } from '@tauri-apps/plugin-process';
import { Download, X, RefreshCw, CheckCircle, AlertTriangle } from 'lucide-react';

interface UpdateInfo {
  version: string;
  date: string;
  body?: string;
}

export function UpdateChecker() {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [checkingForUpdates, setCheckingForUpdates] = useState(false);

  // Check for updates on mount
  useEffect(() => {
    checkForUpdates();
  }, []);

  const checkForUpdates = async () => {
    setCheckingForUpdates(true);
    setError(null);
    
    try {
      console.log('[Updater] Checking for updates...');
      const update = await check();

      if (update) {
        console.log('[Updater] Update available:', update.version);
        setUpdateInfo({
          version: update.version,
          date: update.date,
          body: update.body,
        });
        setUpdateAvailable(true);
        setShowModal(true);
      } else {
        console.log('[Updater] No updates available');
        setUpdateAvailable(false);
      }
    } catch (err) {
      console.error('[Updater] Error checking for updates:', err);
      setError(err instanceof Error ? err.message : 'Fehler beim Prüfen auf Updates');
    } finally {
      setCheckingForUpdates(false);
    }
  };

  const installUpdate = async () => {
    if (!updateAvailable) return;

    setIsDownloading(true);
    setError(null);
    setDownloadProgress(0);

    try {
      console.log('[Updater] Starting update download...');
      const update = await check();
      
      if (!update) {
        throw new Error('Update nicht mehr verfügbar');
      }

      // Download and install
      console.log('[Updater] Downloading update...');
      
      let totalDownloaded = 0;
      const contentLength = update.contentLength || 10 * 1024 * 1024; // Fallback: 10MB
      
      await update.downloadAndInstall((event) => {
        switch (event.event) {
          case 'Started':
            console.log('[Updater] Download started');
            setDownloadProgress(0);
            totalDownloaded = 0;
            break;
          case 'Progress':
            // Accumulate chunks
            totalDownloaded += event.data.chunkLength || 0;
            const progress = (totalDownloaded / contentLength) * 100;
            console.log(`[Updater] Download progress: ${progress.toFixed(0)}%`);
            setDownloadProgress(Math.min(progress, 99)); // Cap at 99% until finished
            break;
          case 'Finished':
            console.log('[Updater] Download finished');
            setDownloadProgress(100);
            break;
        }
      });

      console.log('[Updater] Update installed, restarting app...');
      
      // Relaunch the app
      await relaunch();
    } catch (err) {
      console.error('[Updater] Error installing update:', err);
      setError(err instanceof Error ? err.message : 'Fehler beim Installieren des Updates');
      setIsDownloading(false);
    }
  };

  const dismissUpdate = () => {
    setShowModal(false);
  };

  if (!showModal) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in">
      <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl shadow-2xl border border-white/10 max-w-md w-full overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-6 relative">
          <button
            onClick={dismissUpdate}
            disabled={isDownloading}
            className="absolute top-4 right-4 p-2 hover:bg-white/10 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <X className="w-5 h-5 text-white" />
          </button>
          
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center">
              <Download className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Update verfügbar</h2>
              <p className="text-white/80 text-sm">Version {updateInfo?.version}</p>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-4">
          {/* Update Info */}
          <div className="bg-white/5 border border-white/10 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle className="w-5 h-5 text-green-400" />
              <span className="text-white font-semibold">Neue Features & Verbesserungen</span>
            </div>
            {updateInfo?.body ? (
              <p className="text-gray-300 text-sm whitespace-pre-line">{updateInfo.body}</p>
            ) : (
              <p className="text-gray-300 text-sm">
                Diese Version enthält wichtige Verbesserungen und Fehlerbehebungen.
              </p>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-red-300 font-medium">Fehler</p>
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            </div>
          )}

          {/* Download Progress */}
          {isDownloading && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-300">Download läuft...</span>
                <span className="text-purple-400 font-semibold">{downloadProgress.toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300"
                  style={{ width: `${downloadProgress}%` }}
                />
              </div>
              <p className="text-gray-400 text-xs text-center">
                Die App wird nach dem Download automatisch neu gestartet.
              </p>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={dismissUpdate}
              disabled={isDownloading}
              className="flex-1 px-4 py-3 bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Später
            </button>
            <button
              onClick={installUpdate}
              disabled={isDownloading}
              className="flex-1 px-4 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isDownloading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  <span>Installiere...</span>
                </>
              ) : (
                <>
                  <Download className="w-5 h-5" />
                  <span>Jetzt updaten</span>
                </>
              )}
            </button>
          </div>

          {/* Info */}
          <p className="text-gray-500 text-xs text-center">
            Das Update wird heruntergeladen und automatisch installiert.
          </p>
        </div>
      </div>
    </div>
  );
}
