'use client'

import { useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import { CheckCircle, Download, Key } from 'lucide-react'

export default function PaymentSuccessPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const sessionId = searchParams.get('session_id')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Hier könntest du die Session verifizieren
    // und den API-Key generieren
    setLoading(false)
  }, [sessionId])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-xl">Lädt...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 flex items-center justify-center px-4 py-12 bg-gradient-to-br from-green-50 to-blue-50">
        <div className="w-full max-w-2xl">
          <div className="bg-white rounded-2xl shadow-xl p-8 md:p-12 text-center">
            {/* Success Icon */}
            <div className="flex justify-center mb-6">
              <div className="bg-green-100 rounded-full p-4">
                <CheckCircle className="w-16 h-16 text-green-600" />
              </div>
            </div>

            {/* Success Message */}
            <h1 className="text-4xl font-bold mb-4">Zahlung erfolgreich! 🎉</h1>
            <p className="text-xl text-gray-600 mb-8">
              Vielen Dank für deinen Kauf! Du hast jetzt vollen Zugang zu FrameTrain.
            </p>

            {/* Next Steps */}
            <div className="bg-blue-50 rounded-xl p-6 mb-8 text-left">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Key className="w-5 h-5 text-primary-600" />
                Nächste Schritte
              </h2>
              <ol className="space-y-3">
                <li className="flex items-start gap-3">
                  <span className="bg-primary-600 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 text-sm font-semibold">
                    1
                  </span>
                  <span>Gehe zum Dashboard um deinen API-Key zu sehen</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="bg-primary-600 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 text-sm font-semibold">
                    2
                  </span>
                  <span>Lade die Desktop-App für dein Betriebssystem herunter</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="bg-primary-600 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 text-sm font-semibold">
                    3
                  </span>
                  <span>Melde dich in der App mit deinem API-Key an</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="bg-primary-600 text-white rounded-full w-6 h-6 flex items-center justify-center flex-shrink-0 text-sm font-semibold">
                    4
                  </span>
                  <span>Starte dein erstes ML-Training! 🚀</span>
                </li>
              </ol>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/dashboard"
                className="px-8 py-3 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition flex items-center justify-center gap-2"
              >
                <Key className="w-5 h-5" />
                Zum Dashboard
              </Link>
              <Link
                href="/dashboard#download"
                className="px-8 py-3 border-2 border-primary-600 text-primary-600 rounded-lg font-semibold hover:bg-primary-50 transition flex items-center justify-center gap-2"
              >
                <Download className="w-5 h-5" />
                App herunterladen
              </Link>
            </div>

            {/* Email Note */}
            <div className="mt-8 p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">
                📧 Du erhältst in Kürze eine Bestätigungs-E-Mail mit allen Details.
              </p>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}
