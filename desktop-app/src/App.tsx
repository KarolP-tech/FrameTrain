import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import './App.css';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [apiKey, setApiKey] = useState('');

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const savedKey = await invoke<string>('load_config');
      const isValid = await invoke<boolean>('verify_api_key', { apiKey: savedKey });
      
      if (isValid) {
        setApiKey(savedKey);
        setIsAuthenticated(true);
      }
    } catch (error) {
      console.log('Keine gültige Authentifizierung gefunden');
    } finally {
      setLoading(false);
    }
  };

  const handleLogin = async (key: string) => {
    try {
      const isValid = await invoke<boolean>('verify_api_key', { apiKey: key });
      
      if (isValid) {
        await invoke('save_config', { apiKey: key });
        setApiKey(key);
        setIsAuthenticated(true);
      } else {
        throw new Error('Ungültiger API-Key');
      }
    } catch (error) {
      throw error;
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setApiKey('');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <div className="text-white text-xl">Lädt...</div>
      </div>
    );
  }

  return (
    <div className="app">
      {isAuthenticated ? (
        <Dashboard apiKey={apiKey} onLogout={handleLogout} />
      ) : (
        <Login onLogin={handleLogin} />
      )}
    </div>
  );
}

export default App;
