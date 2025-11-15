'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import { 
  Sparkles, Lock, Rocket, Zap, Code2, Database, 
  BarChart3, Package, Shield, ArrowRight, Check,
  Brain, Cpu, Cloud, Download
} from 'lucide-react'

export default function HomePage() {
  const router = useRouter()
  const { isAuthenticated, loading } = useAuth()
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY })
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="glass px-8 py-4 rounded-2xl">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-gray-300">Loading...</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Animated background gradient */}
      <div 
        className="fixed inset-0 opacity-30 pointer-events-none"
        style={{
          background: `radial-gradient(600px at ${mousePosition.x}px ${mousePosition.y}px, rgba(168, 85, 247, 0.15), transparent 80%)`
        }}
      />
      
      {/* Grid background */}
      <div className="fixed inset-0 grid-bg opacity-50 pointer-events-none" />
      
      <Header />
      
      <main className="flex-1 relative">
        {/* Hero Section */}
        <section className="relative pt-20 pb-32 px-4">
          <div className="max-w-7xl mx-auto">
            {/* Floating badges */}
            <div className="flex justify-center gap-4 mb-8 flex-wrap">
              <div className="glass px-4 py-2 rounded-full text-sm text-gray-300 animate-float flex items-center gap-2">
                <Rocket className="w-4 h-4" />
                <span>ML Training reimagined</span>
              </div>
              <div className="glass px-4 py-2 rounded-full text-sm text-gray-300 animate-float flex items-center gap-2" style={{ animationDelay: '1s' }}>
                <Zap className="w-4 h-4" />
                <span>Lightning fast</span>
              </div>
              <div className="glass px-4 py-2 rounded-full text-sm text-gray-300 animate-float flex items-center gap-2" style={{ animationDelay: '2s' }}>
                <Lock className="w-4 h-4" />
                <span>100% Local</span>
              </div>
            </div>

            {/* Main headline */}
            <div className="text-center max-w-5xl mx-auto">
              <h1 className="text-6xl md:text-8xl font-black mb-8 leading-tight">
                <span className="block text-glow-purple bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                  Train ML Models
                </span>
                <span className="block text-glow-blue bg-gradient-to-r from-blue-400 via-cyan-400 to-green-400 bg-clip-text text-transparent">
                  Locally. Securely.
                </span>
              </h1>
              
              <p className="text-xl md:text-2xl text-gray-400 mb-12 max-w-3xl mx-auto leading-relaxed">
                Die ultimative Desktop-App für professionelles Machine Learning Training. 
                <span className="text-purple-400"> Deine Daten, deine Kontrolle, deine Modelle.</span>
              </p>

              {/* CTA Buttons */}
              <div className="flex gap-4 justify-center flex-wrap">
                {isAuthenticated ? (
                  <Link
                    href="/dashboard"
                    className="group relative px-8 py-4 rounded-2xl overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 animate-gradient" />
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 opacity-0 group-hover:opacity-100 blur-xl transition-opacity" />
                    <div className="relative flex items-center gap-2 text-white font-bold text-lg">
                      <Sparkles className="w-5 h-5" />
                      <span>Zum Dashboard</span>
                      <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </Link>
                ) : (
                  <Link
                    href="/register"
                    className="group relative px-8 py-4 rounded-2xl overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 animate-gradient" />
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 opacity-0 group-hover:opacity-100 blur-xl transition-opacity" />
                    <div className="relative flex items-center gap-2 text-white font-bold text-lg">
                      <Sparkles className="w-5 h-5" />
                      <span>Get Started</span>
                      <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </Link>
                )}
                
                <Link
                  href="#features"
                  className="glass-strong px-8 py-4 rounded-2xl hover:bg-white/10 transition-all duration-300 group"
                >
                  <div className="flex items-center gap-2 text-gray-200 font-semibold text-lg">
                    <span>Explore Features</span>
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </div>
                </Link>
              </div>

              {/* Stats */}
              <div className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
                <StatCard number="100%" label="Local" color="purple" />
                <StatCard number="0ms" label="Latency" color="blue" />
                <StatCard number="∞" label="Models" color="pink" />
                <StatCard number="1,99€" label="One-time" color="green" />
              </div>
            </div>
          </div>

          {/* Animated orbs */}
          <div className="absolute top-1/4 left-10 w-72 h-72 bg-purple-600 rounded-full blur-[128px] opacity-20 animate-pulse-slow" />
          <div className="absolute bottom-1/4 right-10 w-72 h-72 bg-blue-600 rounded-full blur-[128px] opacity-20 animate-pulse-slow" style={{ animationDelay: '2s' }} />
        </section>

        {/* Features Section */}
        <section className="py-32 px-4 relative" id="features">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-20">
              <h2 className="text-5xl md:text-6xl font-black mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Powerful Features
              </h2>
              <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                Alles was du brauchst für professionelles ML-Training in einer App
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              <FeatureCard
                icon={<Lock className="w-6 h-6" />}
                title="100% Local Training"
                description="Deine Daten verlassen niemals dein Gerät. Maximale Privatsphäre und Kontrolle."
                color="purple"
              />
              <FeatureCard
                icon={<Brain className="w-6 h-6" />}
                title="HuggingFace Integration"
                description="Direkter Zugriff auf tausende vortrainierte Modelle. Ein Klick zum Import."
                color="pink"
              />
              <FeatureCard
                icon={<BarChart3 className="w-6 h-6" />}
                title="Live Monitoring"
                description="Verfolge Training in Echtzeit mit interaktiven Charts und Metriken."
                color="blue"
              />
              <FeatureCard
                icon={<Package className="w-6 h-6" />}
                title="Smart Versioning"
                description="Automatische Versionierung aller Modelle mit kompletter Historie."
                color="green"
              />
              <FeatureCard
                icon={<Zap className="w-6 h-6" />}
                title="GPU Accelerated"
                description="Nutze die volle Power deiner GPU für blitzschnelles Training."
                color="yellow"
              />
              <FeatureCard
                icon={<Shield className="w-6 h-6" />}
                title="DSGVO Compliant"
                description="Keine Cloud, keine Telemetrie, keine Datenübertragung. Period."
                color="cyan"
              />
            </div>
          </div>
        </section>

        {/* How it works */}
        <section className="py-32 px-4 relative">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-20">
              <h2 className="text-5xl md:text-6xl font-black mb-6 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                How It Works
              </h2>
              <p className="text-xl text-gray-400 max-w-2xl mx-auto">
                Von Installation bis zum Training in wenigen Minuten
              </p>
            </div>

            <div className="grid md:grid-cols-3 gap-8">
              <StepCard
                number="01"
                icon={<Download className="w-8 h-8" />}
                title="Install & Register"
                description="Registriere dich, zahle einmalig 1,99€ und lade die Desktop-App herunter."
              />
              <StepCard
                number="02"
                icon={<Database className="w-8 h-8" />}
                title="Import Model & Data"
                description="Wähle ein Modell von HuggingFace oder lokal und lade deinen Datensatz hoch."
              />
              <StepCard
                number="03"
                icon={<Rocket className="w-8 h-8" />}
                title="Train & Deploy"
                description="Konfiguriere Parameter, starte das Training und beobachte den Fortschritt live."
              />
            </div>
          </div>
        </section>

        {/* Pricing Section */}
        <section className="py-32 px-4 relative" id="pricing">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-5xl md:text-6xl font-black mb-6 bg-gradient-to-r from-pink-400 to-purple-400 bg-clip-text text-transparent">
              Simple Pricing
            </h2>
            <p className="text-xl text-gray-400 mb-16">
              Einmalige Zahlung. Lebenslanger Zugang. Keine Abos.
            </p>

            <div className="glass-strong neon-border rounded-3xl p-12 max-w-md mx-auto hover:scale-105 transition-transform duration-500">
              <div className="mb-8">
                <div className="text-7xl font-black bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
                  1,99€
                </div>
                <div className="text-gray-400">One-time payment</div>
              </div>

              <ul className="space-y-4 mb-8 text-left">
                {[
                  'Voller Zugang zur Desktop-App',
                  'Unbegrenzte Modelle & Trainings',
                  'Alle zukünftigen Updates',
                  'Community Support',
                  'GPU Acceleration',
                  'Lokale Datenverarbeitung'
                ].map((feature, i) => (
                  <li key={i} className="flex items-center gap-3 text-gray-300">
                    <div className="flex-shrink-0 w-6 h-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                      <Check className="w-4 h-4 text-white" />
                    </div>
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>

              <Link
                href={isAuthenticated ? "/dashboard" : "/register"}
                className="group relative block w-full py-4 rounded-2xl overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 animate-gradient" />
                <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 opacity-0 group-hover:opacity-100 blur-xl transition-opacity" />
                <span className="relative text-white font-bold text-lg">
                  {isAuthenticated ? 'Zum Dashboard' : 'Get FrameTrain Now'}
                </span>
              </Link>
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <section className="py-32 px-4 relative">
          <div className="max-w-4xl mx-auto text-center">
            <div className="glass-dark rounded-3xl p-16 relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-purple-600/20 to-blue-600/20" />
              <div className="relative">
                <h2 className="text-4xl md:text-5xl font-black mb-6 text-white">
                  Ready to transform your ML workflow?
                </h2>
                <p className="text-xl text-gray-300 mb-8">
                  Join hundreds of ML engineers training locally with FrameTrain
                </p>
                <Link
                  href={isAuthenticated ? "/dashboard" : "/register"}
                  className="inline-block group relative px-10 py-5 rounded-2xl overflow-hidden"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 animate-gradient" />
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 opacity-0 group-hover:opacity-100 blur-xl transition-opacity" />
                  <div className="relative flex items-center gap-3 text-white font-bold text-xl">
                    <Sparkles className="w-6 h-6" />
                    <span>{isAuthenticated ? 'Zum Dashboard' : 'Start Training Now'}</span>
                    <ArrowRight className="w-6 h-6 group-hover:translate-x-1 transition-transform" />
                  </div>
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  )
}

function StatCard({ number, label, color }: { number: string; label: string; color: string }) {
  const colors = {
    purple: 'from-purple-500 to-pink-500',
    blue: 'from-blue-500 to-cyan-500',
    pink: 'from-pink-500 to-rose-500',
    green: 'from-green-500 to-emerald-500',
  }

  return (
    <div className="glass-strong rounded-2xl p-6 hover:scale-105 transition-transform">
      <div className={`text-4xl font-black bg-gradient-to-r ${colors[color as keyof typeof colors]} bg-clip-text text-transparent mb-2`}>
        {number}
      </div>
      <div className="text-sm text-gray-400 font-medium">{label}</div>
    </div>
  )
}

function FeatureCard({ icon, title, description, color }: any) {
  const colors = {
    purple: 'from-purple-500 to-pink-500',
    pink: 'from-pink-500 to-rose-500',
    blue: 'from-blue-500 to-cyan-500',
    green: 'from-green-500 to-emerald-500',
    yellow: 'from-yellow-500 to-orange-500',
    cyan: 'from-cyan-500 to-blue-500',
  }

  return (
    <div className="glass-strong rounded-2xl p-8 hover:scale-105 transition-all duration-300 group cursor-pointer">
      <div className={`w-14 h-14 bg-gradient-to-br ${colors[color as keyof typeof colors]} rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
        <div className="text-white">
          {icon}
        </div>
      </div>
      <h3 className="text-xl font-bold text-white mb-3">{title}</h3>
      <p className="text-gray-400 leading-relaxed">{description}</p>
    </div>
  )
}

function StepCard({ number, icon, title, description }: any) {
  return (
    <div className="relative">
      <div className="glass-strong rounded-2xl p-8 hover:scale-105 transition-all duration-300">
        <div className="text-6xl font-black text-purple-500/20 mb-4">{number}</div>
        <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mb-6">
          <div className="text-white">
            {icon}
          </div>
        </div>
        <h3 className="text-2xl font-bold text-white mb-3">{title}</h3>
        <p className="text-gray-400 leading-relaxed">{description}</p>
      </div>
    </div>
  )
}
