import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import GlobalTrainingProgress from './components/GlobalTrainingProgress';
import './App.css';

interface ApiKeyValidation {
  user_id: string;
  email: string;
  is_valid: boolean;
}

interface UserData {
  apiKey: string;
  password: string;
  userId: string;
  email: string;
}

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [userData, setUserData] = useState<UserData | null>(null);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      // Versuche gespeicherte Credentials zu laden
      const savedConfig = await invoke<string>('load_config');
      const config = JSON.parse(savedConfig);
      
      if (config.api_key && config.password) {
        // Validiere gegen Datenbank
        const validation = await invoke<ApiKeyValidation>('validate_credentials', {
          apiKey: config.api_key,
          password: config.password
        });
        
        if (validation.is_valid) {
          setUserData({
            apiKey: config.api_key,
            password: config.password,
            userId: validation.user_id,
            email: validation.email
          });
          setIsAuthenticated(true);
        } else {
          // Ungültige Credentials → Config löschen
          await invoke('clear_config');
        }
      }
    } catch (error) {
      console.log('Keine gültige Authentifizierung gefunden:', error);
      // Config löschen falls vorhanden aber ungültig
      try {
        await invoke('clear_config');
      } catch (e) {
        // Ignore
      }
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async (apiKey: string, password: string) => {
    try {
      // Validiere gegen Datenbank
      const validation = await invoke<ApiKeyValidation>('validate_credentials', {
        apiKey,
        password
      });
      
      if (!validation.is_valid) {
        throw new Error('Authentifizierung fehlgeschlagen');
      }

      // Speichere Credentials
      const config = {
        api_key: apiKey,
        password: password
      };
      
      await invoke('save_config', { 
        apiKey: apiKey,  // Benötigt für backward compatibility
        config: JSON.stringify(config)
      });

      setUserData({
        apiKey,
        password,
        userId: validation.user_id,
        email: validation.email
      });
      setIsAuthenticated(true);
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  };

  const handleLogout = async () => {
    try {
      // Lösche gespeicherte Config
      await invoke('clear_config');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setIsAuthenticated(false);
      setUserData(null);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <div className="text-center">
          <svg
            className="animate-spin h-12 w-12 text-purple-500 mx-auto mb-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
          <div className="text-white text-xl">Lädt...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {isAuthenticated && userData ? (
        <>
          <Dashboard 
            userData={userData}
            onLogout={handleLogout} 
          />
          <GlobalTrainingProgress />
        </>
      ) : (
        <Login onLogin={handleLogin} />
      )}
    </div>
  );
}

export default App;
