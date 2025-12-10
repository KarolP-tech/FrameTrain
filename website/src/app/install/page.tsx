'use client'

import { useState } from 'react'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import { Download, AlertCircle, Terminal, FileCheck } from 'lucide-react'

type Platform = 'macos' | 'windows' | 'linux'

export default function InstallPage() {
  const [selectedPlatform, setSelectedPlatform] = useState<Platform>('macos')

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 px-4 py-12">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold text-white mb-4">
              Installations-Anleitung
            </h1>
            <p className="text-xl text-gray-400">
              So installierst du FrameTrain auf deinem System
            </p>
          </div>

          {/* Platform Tabs */}
          <div className="flex gap-4 mb-8 justify-center flex-wrap">
            <button 
              onClick={() => setSelectedPlatform('macos')}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                selectedPlatform === 'macos' 
                  ? 'bg-purple-600 text-white' 
                  : 'glass text-white hover:bg-white/10'
              }`}
            >
              macOS
            </button>
            <button 
              onClick={() => setSelectedPlatform('windows')}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                selectedPlatform === 'windows' 
                  ? 'bg-purple-600 text-white' 
                  : 'glass text-white hover:bg-white/10'
              }`}
            >
              Windows
            </button>
            <button 
              onClick={() => setSelectedPlatform('linux')}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                selectedPlatform === 'linux' 
                  ? 'bg-purple-600 text-white' 
                  : 'glass text-white hover:bg-white/10'
              }`}
            >
              Linux
            </button>
          </div>

          {/* macOS Installation */}
          {selectedPlatform === 'macos' && (
            <div className="glass-strong rounded-2xl p-8 mb-8 border border-white/10">
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
                <Download className="w-8 h-8 text-purple-400" />
                macOS Installation
              </h2>

              {/* Warning */}
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4 mb-6">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-amber-400 font-semibold mb-1">
                      Sicherheitswarnung beim ersten Öffnen
                    </h3>
                    <p className="text-gray-300 text-sm">
                      Da FrameTrain noch nicht von Apple signiert ist, erscheint eine Warnung. 
                      Die App ist vollständig sicher - folge einfach den Schritten unten.
                    </p>
                  </div>
                </div>
              </div>

              {/* Steps */}
              <div className="space-y-6">
                {/* Step 1 */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      1
                    </div>
                    <h3 className="text-xl font-semibold text-white">Download & Installation</h3>
                  </div>
                  <div className="ml-11 space-y-2 text-gray-300">
                    <p>• Lade die <code className="text-purple-400 bg-black/30 px-2 py-1 rounded">.dmg</code> Datei herunter</p>
                    <p>• Öffne die <code className="text-purple-400 bg-black/30 px-2 py-1 rounded">.dmg</code> Datei</p>
                    <p>• Ziehe <strong>FrameTrain 2</strong> in den <strong>Applications</strong> Ordner</p>
                  </div>
                </div>

                {/* Step 2 - Terminal Method */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      2
                    </div>
                    <h3 className="text-xl font-semibold text-white">Quarantäne entfernen (Terminal)</h3>
                  </div>
                  
                  <div className="ml-11 space-y-3">
                    <p className="text-gray-300">
                      Öffne <strong>Terminal</strong> (Programme → Dienstprogramme → Terminal) und führe diesen Befehl aus:
                    </p>
                    
                    <div className="bg-gray-900 rounded-lg p-4 border border-white/10">
                      <code className="text-green-400 text-sm font-mono block">
                        sudo xattr -cr "/Applications/FrameTrain 2.app"
                      </code>
                    </div>

                    <p className="text-gray-400 text-sm">
                      Gib dein macOS Passwort ein (wird nicht angezeigt beim Tippen) und drücke Enter.
                    </p>
                  </div>
                </div>

                {/* Step 3 */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      3
                    </div>
                    <h3 className="text-xl font-semibold text-white">App starten</h3>
                  </div>
                  <div className="ml-11 text-gray-300">
                    <p>Jetzt kannst du FrameTrain normal öffnen - Doppelklick auf die App im Applications Ordner!</p>
                  </div>
                </div>
              </div>

              {/* Why */}
              <div className="mt-6 bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                <h4 className="text-blue-400 font-semibold mb-2">Warum diese Schritte?</h4>
                <p className="text-gray-300 text-sm">
                  macOS blockiert Apps von nicht-verifizierten Entwicklern. FrameTrain ist Open Source und vollständig sicher. 
                  Wir arbeiten an der offiziellen Apple-Signierung für zukünftige Versionen.
                </p>
              </div>
            </div>
          )}

          {/* Windows Installation */}
          {selectedPlatform === 'windows' && (
            <div className="glass-strong rounded-2xl p-8 mb-8 border border-white/10">
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
                <Download className="w-8 h-8 text-purple-400" />
                Windows Installation
              </h2>

              {/* Warning */}
              <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4 mb-6">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-amber-400 font-semibold mb-1">
                      SmartScreen Warnung
                    </h3>
                    <p className="text-gray-300 text-sm">
                      Windows SmartScreen zeigt eine Warnung, da die App noch nicht von Microsoft verifiziert ist. 
                      Die App ist sicher - folge den Schritten unten.
                    </p>
                  </div>
                </div>
              </div>

              {/* Steps */}
              <div className="space-y-6">
                {/* Step 1 */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      1
                    </div>
                    <h3 className="text-xl font-semibold text-white">Download</h3>
                  </div>
                  <div className="ml-11 space-y-2 text-gray-300">
                    <p>• Lade die <code className="text-purple-400 bg-black/30 px-2 py-1 rounded">.msi</code> oder <code className="text-purple-400 bg-black/30 px-2 py-1 rounded">.exe</code> Datei herunter</p>
                  </div>
                </div>

                {/* Step 2 */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      2
                    </div>
                    <h3 className="text-xl font-semibold text-white">Datei-Eigenschaften öffnen</h3>
                  </div>
                  <div className="ml-11 space-y-2 text-gray-300">
                    <p>• <strong>Rechtsklick</strong> auf die heruntergeladene Datei</p>
                    <p>• Wähle <strong>"Eigenschaften"</strong></p>
                    <p>• Gehe zum Tab <strong>"Allgemein"</strong></p>
                  </div>
                </div>

                {/* Step 3 */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      3
                    </div>
                    <h3 className="text-xl font-semibold text-white">Blockierung aufheben</h3>
                  </div>
                  <div className="ml-11 space-y-3 text-gray-300">
                    <p>Im Bereich <strong>"Sicherheit"</strong> (unten im Fenster):</p>
                    <p>• Aktiviere die Checkbox <strong>"Zulassen"</strong> oder <strong>"Unblock"</strong></p>
                    <p>• Klicke <strong>"Übernehmen"</strong> → <strong>"OK"</strong></p>
                  </div>
                </div>

                {/* Step 4 */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      4
                    </div>
                    <h3 className="text-xl font-semibold text-white">Installation starten</h3>
                  </div>
                  <div className="ml-11 space-y-2 text-gray-300">
                    <p>• Doppelklick auf die Datei</p>
                    <p>• Falls SmartScreen noch erscheint: Klicke <strong>"Weitere Informationen"</strong> → <strong>"Trotzdem ausführen"</strong></p>
                    <p>• Folge dem Installations-Assistenten</p>
                  </div>
                </div>
              </div>

              {/* Why */}
              <div className="mt-6 bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                <h4 className="text-blue-400 font-semibold mb-2">Warum diese Warnung?</h4>
                <p className="text-gray-300 text-sm">
                  Windows SmartScreen schützt vor unbekannten Apps. FrameTrain ist sicher und Open Source. 
                  Wir arbeiten an der offiziellen Microsoft-Signierung für zukünftige Versionen.
                </p>
              </div>
            </div>
          )}

          {/* Linux Installation */}
          {selectedPlatform === 'linux' && (
            <div className="glass-strong rounded-2xl p-8 mb-8 border border-white/10">
              <h2 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
                <Download className="w-8 h-8 text-purple-400" />
                Linux Installation
              </h2>

              {/* Info */}
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4 mb-6">
                <div className="flex items-start gap-3">
                  <FileCheck className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="text-blue-400 font-semibold mb-1">
                      AppImage Format
                    </h3>
                    <p className="text-gray-300 text-sm">
                      FrameTrain wird als AppImage bereitgestellt - portabel und funktioniert auf den meisten Linux-Distributionen.
                    </p>
                  </div>
                </div>
              </div>

              {/* Steps */}
              <div className="space-y-6">
                {/* Step 1 */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      1
                    </div>
                    <h3 className="text-xl font-semibold text-white">FUSE installieren (falls nötig)</h3>
                  </div>
                  <div className="ml-11 space-y-3">
                    <p className="text-gray-300">Auf Ubuntu 22.04+:</p>
                    <div className="bg-gray-900 rounded-lg p-4 border border-white/10">
                      <code className="text-green-400 text-sm font-mono block">
                        sudo apt install libfuse2t64
                      </code>
                    </div>
                    <p className="text-gray-300">Auf Ubuntu 20.04:</p>
                    <div className="bg-gray-900 rounded-lg p-4 border border-white/10">
                      <code className="text-green-400 text-sm font-mono block">
                        sudo apt install libfuse2
                      </code>
                    </div>
                  </div>
                </div>

                {/* Step 2 */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      2
                    </div>
                    <h3 className="text-xl font-semibold text-white">Download</h3>
                  </div>
                  <div className="ml-11 text-gray-300">
                    <p>Lade die <code className="text-purple-400 bg-black/30 px-2 py-1 rounded">.AppImage</code> Datei herunter</p>
                  </div>
                </div>

                {/* Step 3 - GUI Method */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      3a
                    </div>
                    <h3 className="text-xl font-semibold text-white">Ausführbar machen (GUI)</h3>
                  </div>
                  <div className="ml-11 space-y-2 text-gray-300">
                    <p>• <strong>Rechtsklick</strong> auf die <code className="text-purple-400 bg-black/30 px-2 py-1 rounded">.AppImage</code> Datei</p>
                    <p>• Wähle <strong>"Eigenschaften"</strong> oder <strong>"Properties"</strong></p>
                    <p>• Gehe zum Tab <strong>"Berechtigungen"</strong> oder <strong>"Permissions"</strong></p>
                    <p>• Aktiviere <strong>"Als Programm ausführen"</strong> oder <strong>"Allow executing file as program"</strong></p>
                    <p>• Klicke <strong>"OK"</strong></p>
                  </div>
                </div>

                {/* Step 3 - Terminal Method */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      3b
                    </div>
                    <h3 className="text-xl font-semibold text-white">Ausführbar machen (Terminal)</h3>
                  </div>
                  <div className="ml-11 space-y-3">
                    <div className="bg-gray-900 rounded-lg p-4 border border-white/10">
                      <code className="text-green-400 text-sm font-mono block">
                        chmod +x FrameTrain.2_1.0.1_amd64.AppImage
                      </code>
                    </div>
                    <p className="text-gray-400 text-sm">
                      Passe den Dateinamen an deine heruntergeladene Datei an.
                    </p>
                  </div>
                </div>

                {/* Step 4 */}
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                      4
                    </div>
                    <h3 className="text-xl font-semibold text-white">App starten</h3>
                  </div>
                  <div className="ml-11 space-y-2 text-gray-300">
                    <p>• Doppelklick auf die <code className="text-purple-400 bg-black/30 px-2 py-1 rounded">.AppImage</code> Datei</p>
                    <p className="text-sm text-gray-400">Oder im Terminal:</p>
                    <div className="bg-gray-900 rounded-lg p-4 border border-white/10">
                      <code className="text-green-400 text-sm font-mono block">
                        ./FrameTrain.2_1.0.1_amd64.AppImage
                      </code>
                    </div>
                  </div>
                </div>
              </div>

              {/* Troubleshooting */}
              <div className="mt-6 bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
                <h4 className="text-amber-400 font-semibold mb-2">Problem: "Permission denied"?</h4>
                <p className="text-gray-300 text-sm mb-2">
                  Falls die AppImage auf einer externen Festplatte liegt, muss diese mit <code className="text-purple-400 bg-black/30 px-2 py-1 rounded">exec</code> mountet sein:
                </p>
                <div className="bg-gray-900 rounded-lg p-3 border border-white/10">
                  <code className="text-green-400 text-xs font-mono block">
                    sudo mount -o remount,exec /dev/sdX /mount/point
                  </code>
                </div>
              </div>
            </div>
          )}

          {/* System Requirements */}
          <div className="glass rounded-xl p-6 border border-white/10 mb-8">
            <h3 className="text-xl font-bold text-white mb-4">Systemanforderungen</h3>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div>
                <p className="text-gray-400 mb-1">Betriebssystem</p>
                <p className="text-white font-semibold">
                  {selectedPlatform === 'macos' && 'macOS 11.0+'}
                  {selectedPlatform === 'windows' && 'Windows 10/11'}
                  {selectedPlatform === 'linux' && 'Ubuntu 20.04+ (oder äquivalent)'}
                </p>
              </div>
              <div>
                <p className="text-gray-400 mb-1">Arbeitsspeicher</p>
                <p className="text-white font-semibold">8GB (16GB empfohlen)</p>
              </div>
              <div>
                <p className="text-gray-400 mb-1">Speicherplatz</p>
                <p className="text-white font-semibold">2GB frei</p>
              </div>
            </div>
          </div>

          {/* Help */}
          <div className="text-center">
            <p className="text-gray-400 mb-4">Benötigst du Hilfe?</p>
            <div className="flex gap-4 justify-center flex-wrap">
              <a href="mailto:support@frametrain.ai" className="text-purple-400 hover:text-purple-300">
                📧 Email Support
              </a>
              <a href="/docs" className="text-purple-400 hover:text-purple-300">
                📚 Dokumentation
              </a>
              <a href="https://github.com/KarolP-tech/FrameTrain" target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:text-purple-300">
                💻 GitHub
              </a>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}
