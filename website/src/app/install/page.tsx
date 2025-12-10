'use client'

import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import { Download, AlertCircle, CheckCircle, Terminal, MousePointer } from 'lucide-react'

export default function InstallPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 px-4 py-12">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold text-white mb-4">
              Installation Guide
            </h1>
            <p className="text-xl text-gray-400">
              So installierst du FrameTrain auf deinem System
            </p>
          </div>

          {/* macOS Installation */}
          <div className="glass-strong rounded-2xl p-8 mb-8 border border-white/10">
            <h2 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
              <Download className="w-8 h-8 text-purple-400" />
              macOS Installation
            </h2>

            {/* Warning Alert */}
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4 mb-6">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-amber-400 font-semibold mb-1">
                    Wichtiger Hinweis
                  </h3>
                  <p className="text-gray-300 text-sm">
                    Da FrameTrain noch nicht von Apple signiert ist, zeigt macOS beim ersten Öffnen eine Sicherheitswarnung an. 
                    Die App ist vollständig sicher - folge einfach den Schritten unten.
                  </p>
                </div>
              </div>
            </div>

            {/* Step 1 */}
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                  1
                </div>
                <h3 className="text-xl font-semibold text-white">Download & Installation</h3>
              </div>
              <div className="ml-11 space-y-2 text-gray-300">
                <p>• Lade die <code className="text-purple-400 bg-black/30 px-2 py-1 rounded">.dmg</code> Datei herunter</p>
                <p>• Öffne die heruntergeladene Datei</p>
                <p>• Ziehe <strong>FrameTrain 2</strong> in den <strong>Applications</strong> Ordner</p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold">
                  2
                </div>
                <h3 className="text-xl font-semibold text-white">App das erste Mal öffnen</h3>
              </div>
              
              <div className="ml-11 space-y-4">
                <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
                  <p className="text-green-400 font-semibold mb-3 flex items-center gap-2">
                    <CheckCircle className="w-5 h-5" />
                    So geht's richtig:
                  </p>
                  <div className="space-y-2 text-gray-300">
                    <p>✓ Öffne den <strong>Applications</strong> Ordner</p>
                    <p>✓ <strong>Rechtsklick</strong> auf <strong>FrameTrain 2</strong></p>
                    <p>✓ Wähle <strong>"Öffnen"</strong></p>
                    <p>✓ Klicke im Dialog auf <strong>"Öffnen"</strong></p>
                  </div>
                </div>
              </div>
            </div>

            {/* Alternative */}
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-4">
                <Terminal className="w-6 h-6 text-purple-400" />
                <h3 className="text-xl font-semibold text-white">Alternative: Terminal</h3>
              </div>
              
              <div className="ml-9 space-y-3">
                <div className="bg-gray-900 rounded-lg p-4 border border-white/10">
                  <code className="text-green-400 text-sm font-mono">
                    sudo xattr -cr "/Applications/FrameTrain 2.app"
                  </code>
                </div>
              </div>
            </div>
          </div>

          {/* Help */}
          <div className="mt-8 text-center">
            <p className="text-gray-400 mb-4">Benötigst du Hilfe?</p>
            <a href="/docs" className="text-purple-400 hover:text-purple-300">
              📚 Zur Dokumentation
            </a>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}
