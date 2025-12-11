'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import { Download, Key, Copy, Check, ExternalLink, X, RefreshCw, Lightbulb } from 'lucide-react'

interface ApiKey {
  id: string
  key: string
  createdAt: string
  lastUsedAt: string | null
  isActive: boolean
}

export default function DashboardPage() {
  const router = useRouter()
  const { user, isAuthenticated, loading: authLoading } = useAuth()
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([])
  const [copiedKey, setCopiedKey] = useState<string | null>(null)
  const [dataLoading, setDataLoading] = useState(true)
  const [regenerating, setRegenerating] = useState(false)
  const [redirectingToPayment, setRedirectingToPayment] = useState(false)
  const [appVersion, setAppVersion] = useState('...')

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

  // Lade App-Version
  useEffect(() => {
    fetchAppVersion()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const res = await fetch('/api/keys', { 
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      
      if (!res.ok) {
        throw new Error('Fehler beim Laden der Keys')
      }

      const data = await res.json()
      setApiKeys(data.apiKeys || [])
    } catch (error) {
      console.error('Fehler beim Laden der Dashboard-Daten:', error)
    } finally {
      setDataLoading(false)
    }
  }

  const fetchAppVersion = async () => {
    try {
      const res = await fetch('/api/app-version')
      if (res.ok) {
        const data = await res.json()
        setAppVersion(data.version || '1.0.0')
      }
    } catch (error) {
      console.error('Fehler beim Laden der App-Version:', error)
      setAppVersion('1.0.0') // Fallback
    }
  }

  const regenerateKey = async () => {
    if (!confirm('Möchtest du wirklich einen neuen API-Key generieren? Der alte wird ungültig!')) {
      return
    }

    setRegenerating(true)
    try {
      const res = await fetch('/api/keys/regenerate', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!res.ok) {
        throw new Error('Fehler beim Regenerieren')
      }

      const data = await res.json()
      
      // Aktualisiere die Keys-Liste
      await fetchDashboardData()
      
      alert('✅ Neuer API-Key erfolgreich generiert!')
    } catch (error) {
      console.error('Fehler beim Regenerieren:', error)
      alert('❌ Fehler beim Generieren des neuen Keys')
    } finally {
      setRegenerating(false)
    }
  }

  const copyToClipboard = (key: string) => {
    navigator.clipboard.writeText(key)
    setCopiedKey(key)
    setTimeout(() => setCopiedKey(null), 2000)
  }

  const handlePayment = async () => {
    setRedirectingToPayment(true)
    try {
      const res = await fetch('/api/payment/create-checkout', {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!res.ok) {
        throw new Error('Fehler beim Erstellen der Checkout-Session')
      }

      const data = await res.json()
      
      if (data.url) {
        window.location.href = data.url
      }
    } catch (error) {
      console.error('Fehler beim Weiterleiten zur Zahlung:', error)
      alert('❌ Fehler beim Weiterleiten zur Zahlung')
      setRedirectingToPayment(false)
    }
  }

  const downloadApp = async (platform: 'windows' | 'mac' | 'linux') => {
    const activeKey = apiKeys.find(k => k.isActive)
    if (!activeKey) {
      alert('❌ Du brauchst einen aktiven API-Key um die App herunterzuladen')
      return
    }

    try {
      // Fetch download info from API
      const res = await fetch(`/api/download-app?platform=${platform}&key=${activeKey.key}`, {
        credentials: 'include',
      })

      if (!res.ok) {
        const error = await res.json()
        alert(`❌ ${error.message || 'Download fehlgeschlagen'}`)
        return
      }

      const data = await res.json()
      
      if (data.download_url) {
        // Redirect to actual download URL
        window.location.href = data.download_url
      } else {
        alert('❌ Download-URL nicht gefunden')
      }
    } catch (error) {
      console.error('Download-Fehler:', error)
      alert('❌ Fehler beim Herunterladen')
    }
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

  const activeKey = apiKeys.find(k => k.isActive)

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
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center">
                <Key className="w-6 h-6 text-purple-400 mr-3" />
                <h2 className="text-2xl font-bold text-white">Deine API-Keys</h2>
              </div>
              {activeKey && (
                <button
                  onClick={regenerateKey}
                  disabled={regenerating}
                  className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <RefreshCw className={`w-4 h-4 ${regenerating ? 'animate-spin' : ''}`} />
                  {regenerating ? 'Generiere...' : 'Neuen Key generieren'}
                </button>
              )}
            </div>

            {apiKeys.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-400 mb-4">Du hast noch keine API-Keys.</p>
                <p className="text-gray-500 text-sm mb-6">API-Keys werden automatisch nach erfolgreicher Zahlung erstellt.</p>
                <button
                  onClick={handlePayment}
                  disabled={redirectingToPayment}
                  className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 mx-auto"
                >
                  {redirectingToPayment ? (
                    <>
                      <RefreshCw className="w-5 h-5 animate-spin" />
                      <span>Weiterleitung...</span>
                    </>
                  ) : (
                    <>
                      <Key className="w-5 h-5" />
                      <span>Jetzt bezahlen & API-Key erhalten</span>
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                {apiKeys.map((apiKey) => (
                  <div key={apiKey.id} className={`glass rounded-lg p-4 border ${apiKey.isActive ? 'border-green-400/20' : 'border-white/10'}`}>
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
                disabled={!activeKey}
                className="glass-strong hover:bg-white/10 border border-white/10 rounded-xl p-6 transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center">
                    <Download className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Windows</h3>
                  <p className="text-gray-400 text-sm mb-4">Version {appVersion}</p>
                  <div className="flex items-center justify-center text-purple-400 group-hover:text-purple-300">
                    <Download className="w-4 h-4 mr-2" />
                    <span className="font-semibold">Download .exe</span>
                  </div>
                </div>
              </button>

              <button
                onClick={() => downloadApp('mac')}
                disabled={!activeKey}
                className="glass-strong hover:bg-white/10 border border-white/10 rounded-xl p-6 transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center">
                    <Download className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">macOS</h3>
                  <p className="text-gray-400 text-sm mb-4">Version {appVersion}</p>
                  <div className="flex items-center justify-center text-purple-400 group-hover:text-purple-300">
                    <Download className="w-4 h-4 mr-2" />
                    <span className="font-semibold">Download .dmg</span>
                  </div>
                </div>
              </button>

              <button
                onClick={() => downloadApp('linux')}
                disabled={!activeKey}
                className="glass-strong hover:bg-white/10 border border-white/10 rounded-xl p-6 transition-all group disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="text-center">
                  <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-green-500 to-emerald-500 rounded-2xl flex items-center justify-center">
                    <Download className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Linux</h3>
                  <p className="text-gray-400 text-sm mb-4">Version {appVersion}</p>
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
                <span>
                  <strong className="text-blue-400"></strong> Beim ersten Öffnen könnte eine Sicherheitswarnung erscheinen. 
                  <a href="/install" className="text-purple-400 hover:text-purple-300 underline"> Siehe Installations-Guide</a> für Details.
                </span>
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

            {activeKey && (
              <div className="bg-gray-900 rounded-lg p-4 mb-4">
                <code className="text-green-400 font-mono text-sm">
                  frametrain verify-key {activeKey.key}
                </code>
              </div>
            )}

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
