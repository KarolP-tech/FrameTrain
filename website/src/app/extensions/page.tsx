'use client'

import { useState } from 'react'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import { 
  Package, Search, Star, Download, TrendingUp, 
  Zap, Brain, Database, Code2, Image, FileText,
  Video, Music, Globe, Shield, CheckCircle2,
  ExternalLink, Filter, ArrowUpDown
} from 'lucide-react'

interface Extension {
  id: string
  name: string
  description: string
  longDescription: string
  icon: any
  category: string
  price: number
  rating: number
  downloads: number
  tags: string[]
  features: string[]
  isPremium: boolean
  isPopular: boolean
  isNew: boolean
}

const extensions: Extension[] = [
  {
    id: 'nlp-advanced',
    name: 'Advanced NLP Suite',
    description: 'Erweiterte Natural Language Processing Tools mit BERT, GPT & Transformer Support',
    longDescription: 'Professionelle NLP-Extension mit voller Unterstützung für moderne Transformer-Modelle. Inkludiert Pre-Training, Fine-Tuning und Deployment-Tools.',
    icon: Brain,
    category: 'NLP',
    price: 4.99,
    rating: 4.8,
    downloads: 2834,
    tags: ['NLP', 'Transformer', 'BERT', 'GPT'],
    features: [
      'BERT & GPT Integration',
      'Multi-Language Support',
      'Fine-Tuning Workflows',
      'Real-time Inference',
      'Custom Tokenizer'
    ],
    isPremium: true,
    isPopular: true,
    isNew: false
  },
  {
    id: 'computer-vision',
    name: 'Computer Vision Pro',
    description: 'Komplette CV Suite mit YOLO, ResNet & Vision Transformer Support',
    longDescription: 'Alles was du für Computer Vision brauchst: Object Detection, Image Classification, Segmentation und mehr.',
    icon: Image,
    category: 'Computer Vision',
    price: 5.99,
    rating: 4.9,
    downloads: 3521,
    tags: ['Vision', 'YOLO', 'ResNet', 'Detection'],
    features: [
      'Object Detection (YOLO)',
      'Image Segmentation',
      'Face Recognition',
      'Vision Transformers',
      'Real-time Processing'
    ],
    isPremium: true,
    isPopular: true,
    isNew: false
  },
  {
    id: 'time-series',
    name: 'Time Series Analytics',
    description: 'Spezialisierte Tools für Zeitreihen-Analyse und Forecasting',
    longDescription: 'Professionelle Zeitreihen-Analyse mit LSTM, GRU und Attention-Mechanismen für präzise Vorhersagen.',
    icon: TrendingUp,
    category: 'Analytics',
    price: 3.99,
    rating: 4.7,
    downloads: 1823,
    tags: ['Time Series', 'LSTM', 'Forecasting'],
    features: [
      'LSTM & GRU Networks',
      'Attention Mechanisms',
      'Multi-variate Support',
      'Anomaly Detection',
      'Auto-tuning'
    ],
    isPremium: true,
    isPopular: false,
    isNew: true
  },
  {
    id: 'data-augmentation',
    name: 'Smart Data Augmentation',
    description: 'Intelligente Datenerweiterung für bessere Modell-Performance',
    longDescription: 'Erweitere deine Trainingsdaten automatisch mit intelligenten Augmentation-Techniken.',
    icon: Database,
    category: 'Data Processing',
    price: 2.99,
    rating: 4.6,
    downloads: 4123,
    tags: ['Augmentation', 'Preprocessing', 'Data'],
    features: [
      'Auto Augmentation',
      'Mixup & CutMix',
      'Generative Augmentation',
      'Domain Adaptation',
      'Balance Datasets'
    ],
    isPremium: false,
    isPopular: true,
    isNew: false
  },
  {
    id: 'model-compression',
    name: 'Model Compression Toolkit',
    description: 'Quantisierung, Pruning & Distillation für kleinere, schnellere Modelle',
    longDescription: 'Optimiere deine Modelle für Production mit State-of-the-Art Compression-Techniken.',
    icon: Zap,
    category: 'Optimization',
    price: 4.49,
    rating: 4.8,
    downloads: 2156,
    tags: ['Compression', 'Quantization', 'Optimization'],
    features: [
      'Quantization (INT8/INT4)',
      'Neural Pruning',
      'Knowledge Distillation',
      'ONNX Export',
      'Mobile Deployment'
    ],
    isPremium: true,
    isPopular: false,
    isNew: true
  },
  {
    id: 'audio-processing',
    name: 'Audio ML Suite',
    description: 'Speech Recognition, Audio Classification & Generation',
    longDescription: 'Komplette Audio-ML Lösung mit Whisper, Wav2Vec und Audio-Generierungs-Tools.',
    icon: Music,
    category: 'Audio',
    price: 3.99,
    rating: 4.5,
    downloads: 1456,
    tags: ['Audio', 'Speech', 'Whisper'],
    features: [
      'Whisper Integration',
      'Wav2Vec Models',
      'Audio Classification',
      'Voice Cloning',
      'Noise Reduction'
    ],
    isPremium: true,
    isPopular: false,
    isNew: true
  },
  {
    id: 'reinforcement-learning',
    name: 'RL Gymnasium',
    description: 'Reinforcement Learning Environment mit PPO, DQN & Actor-Critic',
    longDescription: 'Trainiere RL-Agenten mit modernen Algorithmen in Custom Environments.',
    icon: Shield,
    category: 'RL',
    price: 5.49,
    rating: 4.7,
    downloads: 987,
    tags: ['RL', 'PPO', 'DQN', 'Gym'],
    features: [
      'OpenAI Gym Integration',
      'PPO & DQN Algorithms',
      'Custom Environments',
      'Multi-Agent Support',
      'Reward Shaping'
    ],
    isPremium: true,
    isPopular: false,
    isNew: false
  },
  {
    id: 'text-generation',
    name: 'Text Generation Studio',
    description: 'Fine-tune & Deploy Language Models für Text Generation',
    longDescription: 'Erstelle eigene Text-Generierungs-Modelle mit modernen LLM-Techniken.',
    icon: FileText,
    category: 'NLP',
    price: 4.99,
    rating: 4.9,
    downloads: 3234,
    tags: ['LLM', 'Generation', 'Fine-tuning'],
    features: [
      'LLM Fine-Tuning',
      'LoRA & QLoRA',
      'Prompt Engineering',
      'RLHF Support',
      'Local Inference'
    ],
    isPremium: true,
    isPopular: true,
    isNew: false
  },
  {
    id: 'video-analysis',
    name: 'Video Understanding',
    description: 'Video Classification, Action Recognition & Tracking',
    longDescription: 'Analysiere Videos mit Deep Learning für Classification, Tracking und mehr.',
    icon: Video,
    category: 'Computer Vision',
    price: 6.99,
    rating: 4.6,
    downloads: 1234,
    tags: ['Video', 'Tracking', 'Action Recognition'],
    features: [
      'Action Recognition',
      'Object Tracking',
      'Video Classification',
      'Temporal Modeling',
      'Real-time Analysis'
    ],
    isPremium: true,
    isPopular: false,
    isNew: true
  },
  {
    id: 'deployment-tools',
    name: 'Deployment Toolkit',
    description: 'One-Click Deployment für ONNX, TensorRT & Mobile',
    longDescription: 'Deploye deine Modelle mit einem Klick auf verschiedene Plattformen.',
    icon: Globe,
    category: 'Deployment',
    price: 3.49,
    rating: 4.8,
    downloads: 2567,
    tags: ['Deployment', 'ONNX', 'TensorRT'],
    features: [
      'ONNX Conversion',
      'TensorRT Optimization',
      'Mobile Export (iOS/Android)',
      'Docker Containers',
      'REST API Generation'
    ],
    isPremium: false,
    isPopular: true,
    isNew: false
  },
  {
    id: 'code-generation',
    name: 'Code ML Assistant',
    description: 'Code Generation & Completion mit ML Models',
    longDescription: 'Trainiere eigene Code-Completion Modelle oder nutze vorgefertigte.',
    icon: Code2,
    category: 'NLP',
    price: 4.49,
    rating: 4.7,
    downloads: 1890,
    tags: ['Code', 'Generation', 'Completion'],
    features: [
      'Code Completion',
      'Documentation Generation',
      'Bug Detection',
      'Code Translation',
      'Multi-Language Support'
    ],
    isPremium: true,
    isPopular: false,
    isNew: true
  },
  {
    id: 'anomaly-detection',
    name: 'Anomaly Detection Pro',
    description: 'Unsupervised Learning für Anomalie-Erkennung',
    longDescription: 'Erkenne Anomalien in Daten mit Autoencoder, Isolation Forest und mehr.',
    icon: Shield,
    category: 'Analytics',
    price: 3.99,
    rating: 4.5,
    downloads: 1567,
    tags: ['Anomaly', 'Unsupervised', 'Detection'],
    features: [
      'Autoencoder Networks',
      'Isolation Forest',
      'One-Class SVM',
      'Real-time Monitoring',
      'Custom Thresholds'
    ],
    isPremium: false,
    isPopular: false,
    isNew: false
  }
]

const categories = ['Alle', 'NLP', 'Computer Vision', 'Analytics', 'Audio', 'RL', 'Deployment', 'Data Processing', 'Optimization']

export default function ExtensionsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('Alle')
  const [sortBy, setSortBy] = useState<'popular' | 'rating' | 'newest' | 'price'>('popular')
  const [showFilters, setShowFilters] = useState(false)
  const [selectedExtension, setSelectedExtension] = useState<Extension | null>(null)

  // Filter extensions
  const filteredExtensions = extensions
    .filter(ext => {
      const matchesSearch = ext.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          ext.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          ext.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
      const matchesCategory = selectedCategory === 'Alle' || ext.category === selectedCategory
      return matchesSearch && matchesCategory
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'popular': return b.downloads - a.downloads
        case 'rating': return b.rating - a.rating
        case 'newest': return (b.isNew ? 1 : 0) - (a.isNew ? 1 : 0)
        case 'price': return a.price - b.price
        default: return 0
      }
    })

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 px-4 py-12">
        <div className="max-w-7xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 glass px-4 py-2 rounded-full text-sm text-purple-400 mb-6">
              <Package className="w-4 h-4" />
              <span>Erweitere deine ML-Pipeline</span>
            </div>
            <h1 className="text-5xl md:text-6xl font-black mb-6 bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
              Extensions Marketplace
            </h1>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Professionelle Extensions für jede ML-Aufgabe. Von NLP bis Computer Vision.
              <span className="text-purple-400"> Installiere mit einem Klick.</span>
            </p>
          </div>

          {/* Search & Filter Bar */}
          <div className="glass-strong rounded-2xl p-6 mb-8 border border-white/10">
            <div className="flex flex-col md:flex-row gap-4">
              {/* Search */}
              <div className="flex-1 relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Extension suchen..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-12 pr-4 py-3 bg-black/30 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 transition-colors"
                />
              </div>

              {/* Sort */}
              <div className="flex gap-2">
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="flex items-center gap-2 px-4 py-3 glass hover:bg-white/10 rounded-xl transition-colors border border-white/10"
                >
                  <Filter className="w-5 h-5 text-gray-400" />
                  <span className="text-gray-300 hidden sm:inline">Filter</span>
                </button>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  className="px-4 py-3 bg-black/30 border border-white/10 rounded-xl text-white focus:outline-none focus:border-purple-500 transition-colors"
                >
                  <option value="popular">Beliebt</option>
                  <option value="rating">Best bewertet</option>
                  <option value="newest">Neueste</option>
                  <option value="price">Preis</option>
                </select>
              </div>
            </div>

            {/* Category Filter */}
            {showFilters && (
              <div className="mt-4 pt-4 border-t border-white/10">
                <div className="flex flex-wrap gap-2">
                  {categories.map(cat => (
                    <button
                      key={cat}
                      onClick={() => setSelectedCategory(cat)}
                      className={`px-4 py-2 rounded-lg transition-all ${
                        selectedCategory === cat
                          ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white'
                          : 'glass hover:bg-white/10 text-gray-300'
                      }`}
                    >
                      {cat}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Stats Bar */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div className="glass-strong rounded-xl p-4 border border-white/10">
              <div className="text-3xl font-bold text-purple-400 mb-1">{extensions.length}</div>
              <div className="text-sm text-gray-400">Extensions</div>
            </div>
            <div className="glass-strong rounded-xl p-4 border border-white/10">
              <div className="text-3xl font-bold text-blue-400 mb-1">{extensions.filter(e => e.isNew).length}</div>
              <div className="text-sm text-gray-400">Neu</div>
            </div>
            <div className="glass-strong rounded-xl p-4 border border-white/10">
              <div className="text-3xl font-bold text-pink-400 mb-1">{extensions.filter(e => e.isPopular).length}</div>
              <div className="text-sm text-gray-400">Beliebt</div>
            </div>
            <div className="glass-strong rounded-xl p-4 border border-white/10">
              <div className="text-3xl font-bold text-green-400 mb-1">{extensions.reduce((acc, e) => acc + e.downloads, 0)}</div>
              <div className="text-sm text-gray-400">Downloads</div>
            </div>
          </div>

          {/* Extensions Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredExtensions.map(ext => (
              <ExtensionCard
                key={ext.id}
                extension={ext}
                onSelect={() => setSelectedExtension(ext)}
              />
            ))}
          </div>

          {filteredExtensions.length === 0 && (
            <div className="text-center py-16">
              <Package className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <p className="text-xl text-gray-400">Keine Extensions gefunden</p>
              <p className="text-gray-500 mt-2">Versuche einen anderen Suchbegriff</p>
            </div>
          )}
        </div>
      </main>

      {/* Extension Detail Modal */}
      {selectedExtension && (
        <ExtensionModal
          extension={selectedExtension}
          onClose={() => setSelectedExtension(null)}
        />
      )}

      <Footer />
    </div>
  )
}

function ExtensionCard({ extension, onSelect }: { extension: Extension; onSelect: () => void }) {
  const Icon = extension.icon

  return (
    <div 
      onClick={onSelect}
      className="glass-strong rounded-2xl p-6 border border-white/10 hover:border-purple-500/50 transition-all duration-300 cursor-pointer group hover:scale-105"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className={`w-14 h-14 bg-gradient-to-br ${
          extension.isPremium 
            ? 'from-purple-600 to-pink-600' 
            : 'from-blue-600 to-cyan-600'
        } rounded-xl flex items-center justify-center`}>
          <Icon className="w-7 h-7 text-white" />
        </div>
        <div className="flex gap-2">
          {extension.isNew && (
            <span className="px-2 py-1 text-xs font-bold bg-green-500/20 text-green-400 rounded-lg border border-green-500/30">
              NEU
            </span>
          )}
          {extension.isPopular && (
            <span className="px-2 py-1 text-xs font-bold bg-purple-500/20 text-purple-400 rounded-lg border border-purple-500/30">
              BELIEBT
            </span>
          )}
        </div>
      </div>

      {/* Content */}
      <h3 className="text-xl font-bold text-white mb-2 group-hover:text-purple-400 transition-colors">
        {extension.name}
      </h3>
      <p className="text-gray-400 text-sm mb-4 line-clamp-2">
        {extension.description}
      </p>

      {/* Tags */}
      <div className="flex flex-wrap gap-2 mb-4">
        {extension.tags.slice(0, 3).map(tag => (
          <span key={tag} className="px-2 py-1 text-xs bg-black/30 text-gray-400 rounded-lg border border-white/10">
            {tag}
          </span>
        ))}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-4 border-t border-white/10">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1 text-yellow-400">
            <Star className="w-4 h-4 fill-current" />
            <span className="text-sm font-semibold">{extension.rating}</span>
          </div>
          <div className="flex items-center gap-1 text-gray-400">
            <Download className="w-4 h-4" />
            <span className="text-sm">{extension.downloads.toLocaleString()}</span>
          </div>
        </div>
        <div className="text-lg font-bold text-purple-400">
          {extension.price.toFixed(2)}€
        </div>
      </div>
    </div>
  )
}

function ExtensionModal({ extension, onClose }: { extension: Extension; onClose: () => void }) {
  const Icon = extension.icon

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm" onClick={onClose}>
      <div 
        className="glass-strong rounded-3xl max-w-3xl w-full max-h-[90vh] overflow-y-auto border border-white/10"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-8 border-b border-white/10">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-4">
              <div className={`w-16 h-16 bg-gradient-to-br ${
                extension.isPremium 
                  ? 'from-purple-600 to-pink-600' 
                  : 'from-blue-600 to-cyan-600'
              } rounded-xl flex items-center justify-center`}>
                <Icon className="w-8 h-8 text-white" />
              </div>
              <div>
                <h2 className="text-3xl font-bold text-white mb-2">{extension.name}</h2>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1 text-yellow-400">
                    <Star className="w-5 h-5 fill-current" />
                    <span className="font-semibold">{extension.rating}</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-400">
                    <Download className="w-5 h-5" />
                    <span>{extension.downloads.toLocaleString()} Downloads</span>
                  </div>
                </div>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            >
              <ExternalLink className="w-6 h-6 text-gray-400 rotate-45" />
            </button>
          </div>
          
          <p className="text-gray-300 text-lg">{extension.longDescription}</p>
        </div>

        {/* Content */}
        <div className="p-8">
          {/* Tags */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-400 mb-3">KATEGORIE</h3>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1.5 text-sm bg-purple-500/20 text-purple-400 rounded-lg border border-purple-500/30">
                {extension.category}
              </span>
              {extension.tags.map(tag => (
                <span key={tag} className="px-3 py-1.5 text-sm bg-black/30 text-gray-400 rounded-lg border border-white/10">
                  {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Features */}
          <div className="mb-8">
            <h3 className="text-sm font-semibold text-gray-400 mb-4">FEATURES</h3>
            <div className="space-y-3">
              {extension.features.map((feature, i) => (
                <div key={i} className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-green-400 flex-shrink-0" />
                  <span className="text-gray-300">{feature}</span>
                </div>
              ))}
            </div>
          </div>

          {/* CTA */}
          <div className="flex items-center justify-between p-6 glass rounded-2xl border border-white/10">
            <div>
              <div className="text-3xl font-bold text-purple-400 mb-1">
                {extension.price.toFixed(2)}€
              </div>
              <div className="text-sm text-gray-400">Einmaliger Kauf</div>
            </div>
            <button className="group relative px-8 py-4 rounded-xl overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 animate-gradient" />
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 opacity-0 group-hover:opacity-100 blur-xl transition-opacity" />
              <div className="relative flex items-center gap-2 text-white font-bold">
                <Download className="w-5 h-5" />
                <span>Jetzt kaufen</span>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
