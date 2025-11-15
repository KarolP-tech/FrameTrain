'use client'

import { useState } from 'react'
import Link from 'next/link'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import {
  Book, Code2, Zap, Download, Key, Settings, Database,
  Terminal, Rocket, ChevronRight, FileCode, Cpu, Cloud,
  Shield, Package, BarChart3, Play, CheckCircle2, AlertCircle
} from 'lucide-react'

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState('getting-started')

  const navigation = [
    {
      title: 'Getting Started',
      id: 'getting-started',
      icon: <Rocket className="w-4 h-4" />,
      items: [
        { id: 'installation', title: 'Installation' },
        { id: 'quick-start', title: 'Quick Start' },
        { id: 'configuration', title: 'Konfiguration' },
      ]
    },
    {
      title: 'Core Concepts',
      id: 'core-concepts',
      icon: <Book className="w-4 h-4" />,
      items: [
        { id: 'architecture', title: 'Architektur' },
        { id: 'data-flow', title: 'Datenfluss' },
        { id: 'models', title: 'Modelle' },
      ]
    },
    {
      title: 'API Reference',
      id: 'api',
      icon: <Code2 className="w-4 h-4" />,
      items: [
        { id: 'authentication', title: 'Authentifizierung' },
        { id: 'endpoints', title: 'Endpoints' },
        { id: 'cli', title: 'CLI Commands' },
      ]
    },
    {
      title: 'Training',
      id: 'training',
      icon: <Cpu className="w-4 h-4" />,
      items: [
        { id: 'datasets', title: 'Datasets' },
        { id: 'hyperparameters', title: 'Hyperparameter' },
        { id: 'monitoring', title: 'Monitoring' },
      ]
    },
    {
      title: 'Deployment',
      id: 'deployment',
      icon: <Cloud className="w-4 h-4" />,
      items: [
        { id: 'export', title: 'Model Export' },
        { id: 'versioning', title: 'Versionierung' },
        { id: 'optimization', title: 'Optimierung' },
      ]
    },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative py-20 px-4 border-b border-white/10">
          <div className="max-w-7xl mx-auto">
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-500 rounded-3xl mb-6">
                <Book className="w-10 h-10 text-white" />
              </div>
              <h1 className="text-5xl md:text-6xl font-black text-white mb-6">
                Dokumentation
              </h1>
              <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                Alles was du wissen musst, um mit FrameTrain zu starten und professionelle ML-Modelle zu trainieren
              </p>
            </div>
          </div>
        </section>

        {/* Main Content */}
        <section className="py-12 px-4">
          <div className="max-w-7xl mx-auto">
            <div className="flex flex-col lg:flex-row gap-8">
              {/* Sidebar Navigation */}
              <aside className="lg:w-64 flex-shrink-0">
                <div className="glass-strong rounded-2xl p-6 border border-white/10 sticky top-24">
                  <nav className="space-y-1">
                    {navigation.map((section) => (
                      <div key={section.id} className="mb-4">
                        <div className="flex items-center gap-2 text-gray-400 font-semibold mb-2 px-3">
                          {section.icon}
                          <span className="text-sm">{section.title}</span>
                        </div>
                        <div className="space-y-1">
                          {section.items.map((item) => (
                            <button
                              key={item.id}
                              onClick={() => setActiveSection(item.id)}
                              className={`w-full text-left px-3 py-2 rounded-lg transition-all ${
                                activeSection === item.id
                                  ? 'bg-purple-500/20 text-purple-400 font-medium'
                                  : 'text-gray-400 hover:bg-white/5 hover:text-white'
                              }`}
                            >
                              {item.title}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </nav>
                </div>
              </aside>

              {/* Content Area */}
              <div className="flex-1 max-w-4xl">
                <div className="glass-strong rounded-2xl p-8 border border-white/10">
                  {activeSection === 'installation' && <InstallationSection />}
                  {activeSection === 'quick-start' && <QuickStartSection />}
                  {activeSection === 'configuration' && <ConfigurationSection />}
                  {activeSection === 'architecture' && <ArchitectureSection />}
                  {activeSection === 'data-flow' && <DataFlowSection />}
                  {activeSection === 'models' && <ModelsSection />}
                  {activeSection === 'authentication' && <AuthenticationSection />}
                  {activeSection === 'endpoints' && <EndpointsSection />}
                  {activeSection === 'cli' && <CLISection />}
                  {activeSection === 'datasets' && <DatasetsSection />}
                  {activeSection === 'hyperparameters' && <HyperparametersSection />}
                  {activeSection === 'monitoring' && <MonitoringSection />}
                  {activeSection === 'export' && <ExportSection />}
                  {activeSection === 'versioning' && <VersioningSection />}
                  {activeSection === 'optimization' && <OptimizationSection />}
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  )
}

// Section Components
function InstallationSection() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
        <Download className="w-8 h-8 text-purple-400" />
        Installation
      </h1>
      
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-4">1. Account erstellen</h2>
          <p className="text-gray-400 mb-4">
            Registriere dich auf unserer Website und schließe die einmalige Zahlung von 1,99€ ab.
          </p>
          <Link href="/register" className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-semibold hover:shadow-lg hover:shadow-purple-500/50 transition-all">
            <Rocket className="w-4 h-4" />
            Jetzt registrieren
          </Link>
        </div>

        <div>
          <h2 className="text-2xl font-bold text-white mb-4">2. Desktop-App herunterladen</h2>
          <p className="text-gray-400 mb-4">
            Lade die FrameTrain Desktop-App für dein Betriebssystem herunter:
          </p>
          
          <div className="grid md:grid-cols-3 gap-4">
            <div className="glass border border-white/10 rounded-lg p-4">
              <h3 className="text-white font-semibold mb-2">Windows</h3>
              <p className="text-gray-400 text-sm mb-3">Windows 10/11 (64-bit)</p>
              <button className="text-purple-400 hover:text-purple-300 text-sm font-medium flex items-center gap-1">
                <Download className="w-4 h-4" />
                frametrain-setup.exe
              </button>
            </div>
            
            <div className="glass border border-white/10 rounded-lg p-4">
              <h3 className="text-white font-semibold mb-2">macOS</h3>
              <p className="text-gray-400 text-sm mb-3">macOS 11+ (Intel/Apple Silicon)</p>
              <button className="text-purple-400 hover:text-purple-300 text-sm font-medium flex items-center gap-1">
                <Download className="w-4 h-4" />
                frametrain.dmg
              </button>
            </div>
            
            <div className="glass border border-white/10 rounded-lg p-4">
              <h3 className="text-white font-semibold mb-2">Linux</h3>
              <p className="text-gray-400 text-sm mb-3">Ubuntu 20.04+, Fedora 35+</p>
              <button className="text-purple-400 hover:text-purple-300 text-sm font-medium flex items-center gap-1">
                <Download className="w-4 h-4" />
                frametrain.AppImage
              </button>
            </div>
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-bold text-white mb-4">3. CLI Installation (Optional)</h2>
          <p className="text-gray-400 mb-4">
            Für erweiterte Funktionen und Automatisierung kannst du die CLI installieren:
          </p>
          
          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <code className="text-green-400 font-mono text-sm">
              pip install frametrain-cli
            </code>
          </div>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <code className="text-green-400 font-mono text-sm">
              frametrain --version
            </code>
          </div>
        </div>

        <div className="glass border border-blue-400/20 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-blue-400 font-semibold mb-1">Systemanforderungen</p>
              <ul className="text-gray-400 text-sm space-y-1">
                <li>• 8GB RAM (16GB empfohlen)</li>
                <li>• NVIDIA GPU mit CUDA 11.0+ (optional, aber empfohlen)</li>
                <li>• 5GB freier Festplattenspeicher</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function QuickStartSection() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
        <Zap className="w-8 h-8 text-purple-400" />
        Quick Start
      </h1>
      
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-4">Dein erstes Modell in 5 Minuten</h2>
          <p className="text-gray-400 mb-6">
            Folge diesen Schritten, um dein erstes ML-Modell mit FrameTrain zu trainieren:
          </p>
        </div>

        <div className="space-y-4">
          <StepCard 
            number="1"
            title="Desktop-App starten"
            description="Öffne die FrameTrain App und melde dich mit deinem API-Key an"
            icon={<Play className="w-6 h-6" />}
          />
          
          <StepCard 
            number="2"
            title="Modell auswählen"
            description="Wähle ein vortrainiertes Modell von HuggingFace oder importiere dein eigenes"
            icon={<Package className="w-6 h-6" />}
            code="bert-base-uncased"
          />
          
          <StepCard 
            number="3"
            title="Dataset hochladen"
            description="Lade deinen Datensatz hoch (CSV, JSON, oder TXT)"
            icon={<Database className="w-6 h-6" />}
          />
          
          <StepCard 
            number="4"
            title="Training konfigurieren"
            description="Passe Hyperparameter an (Epochs, Batch Size, Learning Rate)"
            icon={<Settings className="w-6 h-6" />}
            code="epochs: 3
batch_size: 16
learning_rate: 2e-5"
          />
          
          <StepCard 
            number="5"
            title="Training starten"
            description="Klicke auf 'Start Training' und beobachte den Fortschritt live"
            icon={<BarChart3 className="w-6 h-6" />}
          />
        </div>

        <div className="glass border border-green-400/20 rounded-lg p-4 mt-8">
          <div className="flex items-start gap-3">
            <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-green-400 font-semibold mb-1">Gratuliere!</p>
              <p className="text-gray-400 text-sm">
                Dein trainiertes Modell ist nun verfügbar und kann exportiert oder direkt in der App verwendet werden.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function ConfigurationSection() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
        <Settings className="w-8 h-8 text-purple-400" />
        Konfiguration
      </h1>
      
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-4">API-Key Konfiguration</h2>
          <p className="text-gray-400 mb-4">
            Dein API-Key authentifiziert dich bei der FrameTrain API. Du findest ihn im Dashboard.
          </p>
          
          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <code className="text-green-400 font-mono text-sm">
              frametrain config set-key YOUR_API_KEY
            </code>
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-bold text-white mb-4">Umgebungsvariablen</h2>
          <p className="text-gray-400 mb-4">
            Alternativ kannst du Umgebungsvariablen verwenden:
          </p>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <code className="text-green-400 font-mono text-sm block mb-2">
              export FRAMETRAIN_API_KEY=&quot;your_api_key&quot;
            </code>
            <code className="text-green-400 font-mono text-sm block">
              export FRAMETRAIN_DATA_DIR=&quot;./data&quot;
            </code>
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-bold text-white mb-4">GPU Konfiguration</h2>
          <p className="text-gray-400 mb-4">
            FrameTrain erkennt automatisch verfügbare GPUs. Du kannst die Nutzung konfigurieren:
          </p>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <code className="text-green-400 font-mono text-sm block mb-2">
              # Spezifische GPU auswählen
            </code>
            <code className="text-green-400 font-mono text-sm block mb-2">
              frametrain config set-gpu 0
            </code>
            <code className="text-green-400 font-mono text-sm block mb-2">
              
            </code>
            <code className="text-green-400 font-mono text-sm block">
              # Nur CPU verwenden
            </code>
            <code className="text-green-400 font-mono text-sm block">
              frametrain config set-device cpu
            </code>
          </div>
        </div>
      </div>
    </div>
  )
}

function AuthenticationSection() {
  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
        <Key className="w-8 h-8 text-purple-400" />
        Authentifizierung
      </h1>
      
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-4">API-Key Verwaltung</h2>
          <p className="text-gray-400 mb-4">
            Jeder API-Key ist einzigartig und mit deinem Account verknüpft. Behandle ihn wie ein Passwort.
          </p>
        </div>

        <div>
          <h3 className="text-xl font-bold text-white mb-3">Key-Verifikation</h3>
          <p className="text-gray-400 mb-4">
            Teste deinen API-Key:
          </p>
          
          <div className="bg-gray-900 rounded-lg p-4 mb-4">
            <code className="text-green-400 font-mono text-sm">
              frametrain verify-key YOUR_API_KEY
            </code>
          </div>
        </div>

        <div>
          <h3 className="text-xl font-bold text-white mb-3">HTTP Headers</h3>
          <p className="text-gray-400 mb-4">
            Bei direkten API-Calls verwende den Key im Header:
          </p>
          
          <div className="bg-gray-900 rounded-lg p-4">
            <code className="text-green-400 font-mono text-sm block">
              curl -H &quot;Authorization: Bearer YOUR_API_KEY&quot; \
            </code>
            <code className="text-green-400 font-mono text-sm block">
              &nbsp;&nbsp;https://api.frametrain.ai/models
            </code>
          </div>
        </div>

        <div className="glass border border-yellow-400/20 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-yellow-400 font-semibold mb-1">Sicherheitshinweis</p>
              <p className="text-gray-400 text-sm">
                Teile deinen API-Key niemals öffentlich. Wenn er kompromittiert wurde, erstelle einen neuen im Dashboard.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Placeholder sections (du kannst diese nach Bedarf erweitern)
function ArchitectureSection() {
  return <SectionPlaceholder title="Architektur" icon={<Code2 className="w-8 h-8 text-purple-400" />} />
}

function DataFlowSection() {
  return <SectionPlaceholder title="Datenfluss" icon={<Database className="w-8 h-8 text-purple-400" />} />
}

function ModelsSection() {
  return <SectionPlaceholder title="Modelle" icon={<Package className="w-8 h-8 text-purple-400" />} />
}

function EndpointsSection() {
  return <SectionPlaceholder title="API Endpoints" icon={<Terminal className="w-8 h-8 text-purple-400" />} />
}

function CLISection() {
  return <SectionPlaceholder title="CLI Commands" icon={<Terminal className="w-8 h-8 text-purple-400" />} />
}

function DatasetsSection() {
  return <SectionPlaceholder title="Datasets" icon={<Database className="w-8 h-8 text-purple-400" />} />
}

function HyperparametersSection() {
  return <SectionPlaceholder title="Hyperparameter" icon={<Settings className="w-8 h-8 text-purple-400" />} />
}

function MonitoringSection() {
  return <SectionPlaceholder title="Monitoring" icon={<BarChart3 className="w-8 h-8 text-purple-400" />} />
}

function ExportSection() {
  return <SectionPlaceholder title="Model Export" icon={<Download className="w-8 h-8 text-purple-400" />} />
}

function VersioningSection() {
  return <SectionPlaceholder title="Versionierung" icon={<FileCode className="w-8 h-8 text-purple-400" />} />
}

function OptimizationSection() {
  return <SectionPlaceholder title="Optimierung" icon={<Zap className="w-8 h-8 text-purple-400" />} />
}

// Helper Components
function StepCard({ number, title, description, icon, code }: any) {
  return (
    <div className="glass border border-white/10 rounded-lg p-6">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0">
          <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center text-white font-bold text-lg">
            {number}
          </div>
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <div className="text-purple-400">{icon}</div>
            <h3 className="text-lg font-bold text-white">{title}</h3>
          </div>
          <p className="text-gray-400 text-sm">{description}</p>
          {code && (
            <div className="bg-gray-900 rounded-lg p-3 mt-3">
              <code className="text-green-400 font-mono text-xs whitespace-pre-wrap">{code}</code>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function SectionPlaceholder({ title, icon }: any) {
  return (
    <div className="prose prose-invert max-w-none">
      <h1 className="text-3xl font-bold text-white mb-6 flex items-center gap-3">
        {icon}
        {title}
      </h1>
      
      <div className="glass border border-white/10 rounded-lg p-8 text-center">
        <p className="text-gray-400 mb-4">
          Dieser Abschnitt wird bald erweitert.
        </p>
        <p className="text-gray-500 text-sm">
          Weitere Informationen findest du auf unserer Community-Seite oder im Discord.
        </p>
      </div>
    </div>
  )
}
