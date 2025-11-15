'use client'

import Link from 'next/link'
import { Home, Sparkles } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'
import { useEffect, useState } from 'react'

export function Header() {
  const { user, logout, isAuthenticated, loading } = useAuth()
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <>
      {/* Spacer for fixed header */}
      <div className="h-20" />
      
      {/* Dynamic Island Header */}
      <header 
        className={`fixed top-4 left-1/2 -translate-x-1/2 z-50 transition-all duration-500 ${
          scrolled ? 'scale-95' : 'scale-100'
        }`}
      >
        <div className={`
          glass-strong rounded-[2rem] px-6 py-3.5 
          transition-all duration-500
          border border-white/10
          shadow-2xl shadow-purple-500/20
          ${scrolled ? 'w-[95vw] max-w-7xl' : 'w-[90vw] max-w-6xl'}
        `}>
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Link 
              href="/" 
              className="flex items-center gap-3 group"
            >
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl blur-md opacity-75 group-hover:opacity-100 transition" />
                <div className="relative bg-gradient-to-br from-purple-600 to-pink-600 p-2 rounded-xl">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                FrameTrain
              </span>
            </Link>
            
            {/* Navigation */}
            <nav className="flex items-center gap-2">
              {!loading && (
                <>
                  {isAuthenticated ? (
                    <>
                      <Link
                        href="/"
                        className="flex items-center gap-2 px-4 py-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-xl transition-all duration-300"
                        title="Zur Startseite"
                      >
                        <Home className="w-4 h-4" />
                        <span className="hidden sm:inline">Home</span>
                      </Link>
                      <Link
                        href="/dashboard"
                        className="px-4 py-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-xl transition-all duration-300"
                      >
                        Dashboard
                      </Link>
                      <Link
                        href="/docs"
                        className="px-4 py-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-xl transition-all duration-300 hidden md:block"
                      >
                        Docs
                      </Link>
                      <div className="hidden md:flex items-center px-3 py-1.5 glass rounded-lg text-sm text-gray-400 ml-2">
                        {user?.email}
                      </div>
                      <button
                        onClick={logout}
                        className="px-4 py-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-xl transition-all duration-300"
                      >
                        Abmelden
                      </button>
                    </>
                  ) : (
                    <>
                      <Link
                        href="/#features"
                        className="px-4 py-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-xl transition-all duration-300 hidden sm:block"
                      >
                        Features
                      </Link>
                      <Link
                        href="/#pricing"
                        className="px-4 py-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-xl transition-all duration-300 hidden sm:block"
                      >
                        Pricing
                      </Link>
                      <Link
                        href="/login"
                        className="px-4 py-2 text-gray-300 hover:text-white hover:bg-white/5 rounded-xl transition-all duration-300"
                      >
                        Login
                      </Link>
                      <Link
                        href="/register"
                        className="relative group px-5 py-2 rounded-xl overflow-hidden"
                      >
                        <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 animate-gradient" />
                        <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 opacity-0 group-hover:opacity-100 blur transition-opacity" />
                        <span className="relative text-white font-semibold">
                          Starten
                        </span>
                      </Link>
                    </>
                  )}
                </>
              )}
            </nav>
          </div>
        </div>
      </header>
    </>
  )
}
