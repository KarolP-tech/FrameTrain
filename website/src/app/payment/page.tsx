'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import { CreditCard, Shield, CheckCircle } from 'lucide-react'

export default function PaymentPage() {
  const router = useRouter()
  const { user, isAuthenticated, loading: authLoading } = useAuth()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login')
    }
  }, [authLoading, isAuthenticated, router])

  const handlePayment = async () => {
    setLoading(true)
    setError('')

    try {
      const res = await fetch('/api/payment/create-checkout', {
        method: 'POST',
        credentials: 'include',
      })

      const data = await res.json()

      if (!res.ok) {
        throw new Error(data.error || 'Fehler beim Erstellen der Checkout-Session')
      }

      // Weiterleitung zu Stripe Checkout
      if (data.url) {
        window.location.href = data.url
      }
    } catch (err: any) {
      setError(err.message)
      setLoading(false)
    }
  }

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">Lädt...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return null
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 flex items-center justify-center px-4 py-12 bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="w-full max-w-2xl">
          <div className="bg-white rounded-2xl shadow-xl p-8 md:p-12">
            {/* Header */}
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold mb-4">Zahlung abschließen</h1>
              <p className="text-gray-600">
                Einmalige Zahlung für lebenslangen Zugang
              </p>
            </div>

            {/* Pricing Card */}
            <div className="bg-gradient-to-br from-primary-50 to-blue-50 rounded-xl p-8 mb-8">
              <div className="text-center">
                <div className="text-5xl font-bold text-primary-600 mb-2">
                  1,99 €
                </div>
                <p className="text-gray-600 mb-6">Einmalige Zahlung</p>
                
                <div className="space-y-3 text-left">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                    <span>Voller Zugang zur Desktop-App</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                    <span>Unbegrenzte Modelle & Trainings</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                    <span>Lebenslange Updates</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                    <span>API-Key sofort nach Zahlung</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0" />
                    <span>Community Support</span>
                  </div>
                </div>
              </div>
            </div>

            {/* User Info */}
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Zahlung für:</span>
                <span className="font-semibold">{user?.email}</span>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="mb-6 p-4 bg-red-50 border border-red-200 text-red-700 rounded-lg">
                {error}
              </div>
            )}

            {/* Payment Button */}
            <button
              onClick={handlePayment}
              disabled={loading}
              className="w-full py-4 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {loading ? (
                'Wird geladen...'
              ) : (
                <>
                  <CreditCard className="w-5 h-5" />
                  Jetzt sicher bezahlen
                </>
              )}
            </button>

            {/* Security Note */}
            <div className="mt-6 flex items-center justify-center gap-2 text-sm text-gray-600">
              <Shield className="w-4 h-4 text-green-600" />
              <span>Sichere Zahlung über Stripe</span>
            </div>

            {/* Cancel Link */}
            <div className="mt-6 text-center">
              <button
                onClick={() => router.push('/dashboard')}
                className="text-gray-600 hover:text-gray-900 text-sm"
              >
                Später bezahlen
              </button>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}
