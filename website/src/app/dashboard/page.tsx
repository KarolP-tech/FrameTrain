'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import { Download, Key, Copy, Check, ExternalLink, X, Lightbulb } from 'lucide-react'

interface ApiKey {
  id: string
  key: string
  createdAt: string
  lastUsedAt: string | null
  isActive: boolean
}

export default function DashboardPage() {
  const router = useRouter()
  const { user, isAuthenticated, loading: authLoading, logout } = useAuth()
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([])
  const [copiedKey, setCopiedKey] = useState<string | null>(null)
  const [dataLoading, setDataLoading] = useState(true)

  // Redirect wenn nicht authentifiziert
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login')
    }
  }, [authLoading, isAuthenticated, router])

  // Lade Dashboard-Daten
  useEffect(() => {
    if (isAuthenticated && user) {
      fetchDashboardData()
    }
  }, [isAuthenticated, user])

  const fetchDashboardData = async () => {
    try {
      // TODO: Echten API-Call implementieren
      // const res = await fetch('/api/keys', { credentials: 'include' })
      // const data = await res.json()
      
      // Dummy-Daten für jetzt
      const mockApiKey: ApiKey = {
        id: '1',
        key: 'ft_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15),
        createdAt: new Date().toISOString(),
        lastUsedAt: null,
        isActive: true,
      }
      
      setApiKeys([mockApiKey])
    } catch (error) {
      console.error('Fehler beim Laden der Dashboard-Daten:', error)
    } finally {
      setDataLoading(false)
    }
  }

  const copyToClipboard = (key: string) => {
    navigator.clipboard.writeText(key)
    setCopiedKey(key)
    setTimeout(() => setCopiedKey(null), 2000)
  }

  const downloadApp = async (platform: 'windows' | 'mac' | 'linux') => {
    window.location.href = `/api/download-app?platform=${platform}`
  }

  // Loading state
  if (authLoading || dataLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <div className="text-white text-xl">Lädt...</div>
      </div>
    )
  }

  // Nicht authentifiziert (wird umgeleitet)
  if (!isAuthenticated) {
    return null
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 px-4 py-12">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-white mb-2">Dashboard</h1>
            <p className="text-gray-400">Willkommen, {user?.email}</p>
          </div>

          {/* API Keys Section */}
          <div className="glass-strong rounded-2xl shadow-lg p-8 mb-8 border border-white/10">
            <div className="flex items-center mb-6">
              <Key className="w-6 h-6 text-purple-400 mr-3" />
              <h2 className="text-2xl font-bold text-white">Deine API-Keys</h2>
            </div>

            {apiKeys.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-400 mb-4">Du hast noch keine API-Keys.</p>
                <button className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-lg hover:shadow-lg hover:shadow-purple-500/50 transition-all">
                  Ersten Key erstellen
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                {apiKeys.map((apiKey) => (
                  <div key={apiKey.id} className="glass rounded-lg p-4 border border-white/10">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <code className="text-purple-400 font-mono text-sm bg-black/30 px-3 py-1 rounded border border-white/10">
                            {apiKey.key}
                          </code>
                          <button
                            onClick={() => copyToClipboard(apiKey.key)}
                            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                            title={copiedKey === apiKey.key ? 'Kopiert!' : 'Kopieren'}
                          >
                            {copiedKey === apiKey.key ? (
                              <Check className="w-4 h-4 text-green-400" />
                            ) : (
                              <Copy className="w-4 h-4 text-gray-400" />
                            )}
                          </button>
                        </div>
                        <div className="flex items-center space-x-4 text-sm text-gray-400">
                          <span>Erstellt: {new Date(apiKey.createdAt).toLocaleDateString('de-DE')}</span>
                          {apiKey.lastUsedAt && (
                            <span>Zuletzt genutzt: {new Date(apiKey.lastUsedAt).toLocaleDateString('de-DE')}</span>
                          )}
                          <span className={`flex items-center gap-1 ${apiKey.isActive ? 'text-green-400' : 'text-red-400'}`}>
                            {apiKey.isActive ? (
                              <>
                                <Check className="w-4 h-4" />
                                <span>Aktiv</span>
                              </>
                            ) : (
                              <>
                                <X className="w-4 h-4" />
                                <span>Inaktiv</span>
                              </>
                            )}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Download Section */}
          <div className="glass-strong rounded-2xl shadow-lg p-8 mb-8 border border-white/10">
            <div className="flex items-center mb-6">
              <Download className="w-6 h-6 text-purple-400 mr-3" />
              <h2 className="text-2xl font-bold text-white">Desktop-App herunterladen</h2>
            </div>

            <div className="grid md:grid-cols-3 gap-6">
              <button
                onClick={() => downloadApp('windows')}
                className="glass-strong hover:bg-white/10 border border-white/10 rounded-xl p-6 transition-all group"
              >
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center">
                    <Download className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Windows</h3>
                  <p className="text-gray-400 text-sm mb-4">Version 1.0.0</p>
                  <div className="flex items-center justify-center text-purple-400 group-hover:text-purple-300">
                    <Download className="w-4 h-4 mr-2" />
                    <span className="font-semibold">Download .exe</span>
                  </div>
                </div>
              </button>

              <button
                onClick={() => downloadApp('mac')}
                className="glass-strong hover:bg-white/10 border border-white/10 rounded-xl p-6 transition-all group"
              >
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center">
                    <Download className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">macOS</h3>
                  <p className="text-gray-400 text-sm mb-4">Version 1.0.0</p>
                  <div className="flex items-center justify-center text-purple-400 group-hover:text-purple-300">
                    <Download className="w-4 h-4 mr-2" />
                    <span className="font-semibold">Download .dmg</span>
                  </div>
                </div>
              </button>

              <button
                onClick={() => downloadApp('linux')}
                className="glass-strong hover:bg-white/10 border border-white/10 rounded-xl p-6 transition-all group"
              >
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-green-500 to-emerald-500 rounded-2xl flex items-center justify-center">
                    <Download className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Linux</h3>
                  <p className="text-gray-400 text-sm mb-4">Version 1.0.0</p>
                  <div className="flex items-center justify-center text-purple-400 group-hover:text-purple-300">
                    <Download className="w-4 h-4 mr-2" />
                    <span className="font-semibold">Download .AppImage</span>
                  </div>
                </div>
              </button>
            </div>

            <div className="mt-8 p-4 glass border border-blue-400/20 rounded-lg">
              <p className="text-gray-300 text-sm flex items-start gap-2">
                <Lightbulb className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                <span><strong className="text-blue-400">Tipp:</strong> Nach der Installation kannst du dich mit deinem API-Key in der App anmelden.</span>
              </p>
            </div>
          </div>

          {/* CLI Section */}
          <div className="glass-strong rounded-2xl shadow-lg p-8 border border-white/10">
            <h2 className="text-2xl font-bold text-white mb-4">CLI Installation</h2>
            <p className="text-gray-400 mb-4">Installiere die FrameTrain CLI für erweiterte Funktionen:</p>
            
            <div className="bg-gray-900 rounded-lg p-4 mb-4">
              <code className="text-green-400 font-mono text-sm">
                pip install frametrain-cli
              </code>
            </div>

            <div className="bg-gray-900 rounded-lg p-4 mb-4">
              <code className="text-green-400 font-mono text-sm">
                frametrain verify-key {apiKeys[0]?.key || 'YOUR_KEY'}
              </code>
            </div>

            <a
              href="/docs"
              className="inline-flex items-center text-purple-400 hover:text-purple-300"
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              Zur vollständigen Dokumentation
            </a>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}
