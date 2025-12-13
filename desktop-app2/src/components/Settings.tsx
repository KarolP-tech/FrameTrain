import { useState, useEffect } from 'react';
import { User, Key, Shield, Bell, Palette, Info, ExternalLink, LogOut, AlertCircle, CheckCircle, Check, Download } from 'lucide-react';
import { useTheme, ThemeId } from '../contexts/ThemeContext';
import { getVersion } from '@tauri-apps/api/app';
import { check } from '@tauri-apps/plugin-updater';
import { relaunch } from '@tauri-apps/plugin-process';

interface UserData {
  apiKey: string;
  password: string;
  userId: string;
  email: string;
}

interface SettingsProps {
  userData: UserData;
  onLogout: () => void;
}

type SettingsTab = 'account' | 'appearance' | 'notifications' | 'updates' | 'about';

export default function Settings({ userData, onLogout }: SettingsProps) {
  const [activeTab, setActiveTab] = useState<SettingsTab>('account');
  const [showApiKey, setShowApiKey] = useState(false);
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);
  const { currentTheme, setTheme, themes: allThemes } = useTheme();
  const [appVersion, setAppVersion] = useState<string>('Loading...');
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const [updateVersion, setUpdateVersion] = useState<string>('');
  const [checkingForUpdates, setCheckingForUpdates] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);

  useEffect(() => {
    loadAppVersion();
    checkForUpdates();
  }, []);

  const loadAppVersion = async () => {
    try {
      const version = await getVersion();
      setAppVersion(version);
    } catch (error) {
      console.error('Failed to load app version:', error);
      setAppVersion('Unknown');
    }
  };

  const checkForUpdates = async () => {
    setCheckingForUpdates(true);
    try {
      console.log('[Settings/Updates] ========================================'  );
      console.log('[Settings/Updates] Checking for updates...');
      console.log('[Settings/Updates] Current version:', appVersion);
      
      const update = await check();
      
      console.log('[Settings/Updates] Check result:', update);
      
      if (update) {
        console.log('[Settings/Updates] ✅ Update available!');
        console.log('[Settings/Updates] New version:', update.version);
        console.log('[Settings/Updates] ========================================');
        
        setUpdateAvailable(true);
        setUpdateVersion(update.version);
        setNotification({ type: 'success', message: `Update verfügbar: ${update.version}` });
        setTimeout(() => setNotification(null), 3000);
      } else {
        console.log('[Settings/Updates] ℹ️ No updates available');
        console.log('[Settings/Updates] ========================================');
        
        setUpdateAvailable(false);
        setNotification({ type: 'success', message: 'Du bist auf dem neuesten Stand!' });
        setTimeout(() => setNotification(null), 3000);
      }
    } catch (error) {
      console.error('[Settings/Updates] ❌ Error checking for updates:', error);
      console.error('[Settings/Updates] Error details:', JSON.stringify(error, null, 2));
      console.error('[Settings/Updates] ========================================');
      
      setNotification({ type: 'error', message: 'Fehler beim Prüfen auf Updates' });
      setTimeout(() => setNotification(null), 3000);
    } finally {
      setCheckingForUpdates(false);
    }
  };

  const installUpdate = async () => {
    if (!updateAvailable) return;

    setIsDownloading(true);
    setDownloadProgress(0);

    try {
      const update = await check();
      if (!update) {
        throw new Error('Update nicht mehr verfügbar');
      }

      let totalDownloaded = 0;
      const estimatedSize = 10 * 1024 * 1024;

      await update.downloadAndInstall((event) => {
        switch (event.event) {
          case 'Started':
            setDownloadProgress(0);
            totalDownloaded = 0;
            break;
          case 'Progress':
            totalDownloaded += event.data.chunkLength || 0;
            const progress = (totalDownloaded / estimatedSize) * 100;
            setDownloadProgress(Math.min(progress, 99));
            break;
          case 'Finished':
            setDownloadProgress(100);
            break;
        }
      });

      await relaunch();
    } catch (error) {
      console.error('Failed to install update:', error);
      setNotification({ type: 'error', message: 'Fehler beim Installieren des Updates' });
      setTimeout(() => setNotification(null), 3000);
      setIsDownloading(false);
    }
  };

  const tabs = [
    { id: 'account' as SettingsTab, label: 'Konto', icon: User },
    { id: 'appearance' as SettingsTab, label: 'Darstellung', icon: Palette },
    { id: 'notifications' as SettingsTab, label: 'Benachrichtigungen', icon: Bell },
    { id: 'updates' as SettingsTab, label: 'Updates', icon: Download },
    { id: 'about' as SettingsTab, label: 'Über', icon: Info },
  ];

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setNotification({ type: 'success', message: 'In Zwischenablage kopiert!' });
      setTimeout(() => setNotification(null), 3000);
    } catch (error) {
      setNotification({ type: 'error', message: 'Kopieren fehlgeschlagen' });
      setTimeout(() => setNotification(null), 3000);
    }
  };

  const renderAccountTab = () => (
    <div className="space-y-6">
      {/* User Info Card */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Benutzerinformationen</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">E-Mail</label>
            <div className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white">
              {userData.email}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">User ID</label>
            <div className="flex items-center space-x-2">
              <div className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white font-mono text-sm truncate">
                {userData.userId}
              </div>
              <button
                onClick={() => copyToClipboard(userData.userId)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
              >
                Kopieren
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* API Key Card */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">API-Key</h3>
          <Key className="w-5 h-5 text-purple-400" />
        </div>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Dein API-Key</label>
            <div className="flex items-center space-x-2">
              <div className="flex-1 px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-white font-mono text-sm">
                {showApiKey ? userData.apiKey : '••••••••••••••••••••'}
              </div>
              <button
                onClick={() => setShowApiKey(!showApiKey)}
                className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white rounded-lg transition-colors border border-white/10"
              >
                {showApiKey ? 'Verbergen' : 'Anzeigen'}
              </button>
              <button
                onClick={() => copyToClipboard(userData.apiKey)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
              >
                Kopieren
              </button>
            </div>
          </div>

          <div className="flex items-start space-x-2 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
            <Shield className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-yellow-300">
              Teile deinen API-Key niemals mit anderen. Er gewährt vollen Zugriff auf deinen Account.
            </p>
          </div>
        </div>
      </div>

      {/* Account Management */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Kontoverwaltung</h3>
        
        <div className="space-y-3">
          <a
            href="https://frame-train.vercel.app/dashboard"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>Dashboard öffnen</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <a
            href="https://frame-train.vercel.app/dashboard"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>Passwort ändern</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <button
            onClick={onLogout}
            className="w-full flex items-center justify-between px-4 py-3 bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 rounded-lg text-red-300 hover:text-red-200 transition-colors"
          >
            <span>Abmelden</span>
            <LogOut className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );

  const handleThemeChange = async (themeId: ThemeId) => {
    setTheme(themeId);
    setNotification({ type: 'success', message: 'Theme erfolgreich geändert!' });
    setTimeout(() => setNotification(null), 3000);
  };

  const renderAppearanceTab = () => (
    <div className="space-y-6">
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Farbschema</h3>
        <p className="text-gray-400 mb-6">Wähle dein bevorzugtes Farbschema für die Desktop-App</p>
        
        <div className="grid grid-cols-3 gap-4 max-h-[500px] overflow-y-auto pr-2">
          {Object.values(allThemes).map((theme) => (
            <button
              key={theme.id}
              onClick={() => handleThemeChange(theme.id)}
              className={`relative p-5 bg-gradient-to-br ${theme.colors.background} border-2 rounded-xl transition-all hover:scale-105 ${
                currentTheme.id === theme.id
                  ? 'border-white/40 shadow-lg ring-2 ring-white/20'
                  : 'border-white/10 hover:border-white/20'
              }`}
            >
              {/* Checkmark for active theme */}
              {currentTheme.id === theme.id && (
                <div className="absolute top-2 right-2 w-6 h-6 bg-white rounded-full flex items-center justify-center shadow-lg">
                  <Check className="w-4 h-4 text-slate-900" />
                </div>
              )}
              
              {/* Color preview */}
              <div className="flex justify-center mb-3 space-x-2">
                <div className={`w-7 h-7 rounded-full bg-gradient-to-br ${theme.colors.gradient} shadow-md`} />
                <div className="w-7 h-7 rounded-full shadow-md" style={{ backgroundColor: theme.colors.accent }} />
              </div>
              
              {/* Theme info */}
              <div className="text-center">
                <div className="text-white font-semibold text-sm mb-1">{theme.name}</div>
                <div className="text-xs text-gray-400">{theme.description}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Preview Section */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Vorschau</h3>
        <div className={`p-6 bg-gradient-to-br ${currentTheme.colors.background} rounded-xl border border-white/10`}>
          <div className="flex items-center space-x-4 mb-4">
            <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${currentTheme.colors.gradient} flex items-center justify-center`}>
              <Palette className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="text-white font-semibold">Beispiel Button</div>
              <div className="text-gray-400 text-sm">So sieht dein Theme aus</div>
            </div>
          </div>
          <button className={`w-full py-3 px-4 bg-gradient-to-r ${currentTheme.colors.gradient} text-white font-semibold rounded-lg hover:opacity-90 transition-opacity`}>
            Beispiel Button
          </button>
        </div>
      </div>
    </div>
  );

  const renderNotificationsTab = () => (
    <div className="space-y-6">
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Desktop-Benachrichtigungen</h3>
        
        <div className="space-y-4">
          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <div className="text-white font-medium">Training abgeschlossen</div>
              <div className="text-sm text-gray-400">Benachrichtigung wenn Training fertig ist</div>
            </div>
            <input type="checkbox" className="w-5 h-5 rounded bg-white/5 border-white/10" defaultChecked />
          </label>

          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <div className="text-white font-medium">Fehler und Warnungen</div>
              <div className="text-sm text-gray-400">Benachrichtigung bei Problemen</div>
            </div>
            <input type="checkbox" className="w-5 h-5 rounded bg-white/5 border-white/10" defaultChecked />
          </label>

          <label className="flex items-center justify-between cursor-pointer">
            <div>
              <div className="text-white font-medium">Updates verfügbar</div>
              <div className="text-sm text-gray-400">Benachrichtigung über neue Versionen</div>
            </div>
            <input type="checkbox" className="w-5 h-5 rounded bg-white/5 border-white/10" defaultChecked />
          </label>
        </div>
      </div>
    </div>
  );

  const renderUpdatesTab = () => (
    <div className="space-y-6">
      {/* Current Version Card */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Aktuelle Version</h3>
          <Download className="w-5 h-5 text-purple-400" />
        </div>
        
        <div className="flex items-center justify-between p-4 bg-white/5 border border-white/10 rounded-lg">
          <div>
            <div className="text-white font-semibold text-lg">FrameTrain Desktop {appVersion}</div>
            <div className="text-sm text-gray-400 mt-1">
              {updateAvailable ? 'Update verfügbar' : 'Du bist auf dem neuesten Stand'}
            </div>
          </div>
          {updateAvailable && (
            <div className="px-3 py-1 bg-green-500/20 border border-green-500/30 rounded-full">
              <span className="text-green-400 text-sm font-medium">Update verfügbar</span>
            </div>
          )}
          {!updateAvailable && !checkingForUpdates && (
            <div className="px-3 py-1 bg-purple-500/20 border border-purple-500/30 rounded-full">
              <span className="text-purple-400 text-sm font-medium">✓ Aktuell</span>
            </div>
          )}
        </div>
      </div>

      {/* Update Available Card */}
      {updateAvailable && (
        <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-xl p-6">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-pink-600 rounded-xl flex items-center justify-center flex-shrink-0">
              <Download className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-bold text-white mb-2">Neue Version verfügbar!</h3>
              <p className="text-gray-300 mb-4">
                Version <span className="font-semibold text-purple-400">{updateVersion}</span> ist jetzt verfügbar. 
                Aktualisiere, um die neuesten Features und Verbesserungen zu erhalten.
              </p>
              
              {isDownloading ? (
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
                  <p className="text-gray-400 text-xs">
                    Die App wird nach dem Download automatisch neu gestartet.
                  </p>
                </div>
              ) : (
                <button
                  onClick={installUpdate}
                  className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white rounded-lg font-semibold transition-all flex items-center gap-2"
                >
                  <Download className="w-5 h-5" />
                  <span>Jetzt aktualisieren</span>
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Check for Updates Button */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Nach Updates suchen</h3>
        <p className="text-gray-400 mb-4 text-sm">
          Prüfe manuell, ob eine neue Version von FrameTrain verfügbar ist.
        </p>
        <button
          onClick={checkForUpdates}
          disabled={checkingForUpdates || isDownloading}
          className="px-4 py-2 bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {checkingForUpdates ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Prüfe...</span>
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              <span>Nach Updates suchen</span>
            </>
          )}
        </button>
      </div>

      {/* Auto-Update Info */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Automatische Updates</h3>
        <p className="text-gray-400 text-sm">
          FrameTrain prüft automatisch beim Start auf neue Versionen. Du wirst benachrichtigt, 
          wenn ein Update verfügbar ist.
        </p>
      </div>
    </div>
  );

  const renderAboutTab = () => (
    <div className="space-y-6">
      <div className="bg-white/5 rounded-xl p-6 border border-white/10 text-center">
        <div className="inline-block p-4 bg-gradient-to-br from-purple-600 to-pink-600 rounded-2xl mb-4">
          <svg
            className="w-12 h-12 text-white"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
            />
          </svg>
        </div>
        <h3 className="text-2xl font-bold text-white mb-2">FrameTrain Desktop</h3>
        <p className="text-gray-400 mb-4">Version {appVersion}</p>
        <p className="text-sm text-gray-400 max-w-md mx-auto">
          Trainiere Machine Learning Modelle lokal auf deinem Computer mit der Leistung von PyTorch.
        </p>
      </div>

      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <h3 className="text-lg font-semibold text-white mb-4">Links</h3>
        
        <div className="space-y-3">
          <a
            href="https://frame-train.vercel.app/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>Website</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <a
            href="https://frame-train.vercel.app/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>Dokumentation</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>

          <a
            href="https://github.com/KarolP-tech/FrameTrain/releases"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-white transition-colors group"
          >
            <span>GitHub</span>
            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-purple-400" />
          </a>
        </div>
      </div>

      <div className="bg-white/5 rounded-xl p-6 border border-white/10">
        <p className="text-sm text-gray-400 text-center">
          © 2024 FrameTrain. Alle Rechte vorbehalten.
        </p>
      </div>
    </div>
  );

  return (
    <div>
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">Einstellungen</h2>
        <p className="text-gray-400">Verwalte dein Konto und App-Einstellungen</p>
      </div>

      {/* Notification */}
      {notification && (
        <div className={`mb-6 flex items-start space-x-2 p-4 rounded-lg border ${
          notification.type === 'success'
            ? 'bg-green-500/10 border-green-500/20'
            : 'bg-red-500/10 border-red-500/20'
        }`}>
          {notification.type === 'success' ? (
            <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
          ) : (
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
          )}
          <p className={`text-sm ${
            notification.type === 'success' ? 'text-green-300' : 'text-red-300'
          }`}>
            {notification.message}
          </p>
        </div>
      )}

      <div className="grid grid-cols-4 gap-6">
        {/* Sidebar Tabs */}
        <div className="space-y-2">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg'
                    : 'bg-white/5 text-gray-300 hover:bg-white/10 hover:text-white border border-white/10'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Content Area */}
        <div className="col-span-3">
          {activeTab === 'account' && renderAccountTab()}
          {activeTab === 'appearance' && renderAppearanceTab()}
          {activeTab === 'notifications' && renderNotificationsTab()}
          {activeTab === 'updates' && renderUpdatesTab()}
          {activeTab === 'about' && renderAboutTab()}
        </div>
      </div>
    </div>
  );
}
