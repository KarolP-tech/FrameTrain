'use client'

import Link from 'next/link'
import { Header } from '@/components/Header'
import { Footer } from '@/components/Footer'
import { XCircle, ArrowLeft } from 'lucide-react'

export default function PaymentCancelPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      
      <main className="flex-1 flex items-center justify-center px-4 py-12 bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="w-full max-w-2xl">
          <div className="bg-white rounded-2xl shadow-xl p-8 md:p-12 text-center">
            {/* Cancel Icon */}
            <div className="flex justify-center mb-6">
              <div className="bg-gray-100 rounded-full p-4">
                <XCircle className="w-16 h-16 text-gray-600" />
              </div>
            </div>

            {/* Cancel Message */}
            <h1 className="text-4xl font-bold mb-4">Zahlung abgebrochen</h1>
            <p className="text-xl text-gray-600 mb-8">
              Die Zahlung wurde nicht abgeschlossen. Keine Sorge, es wurden keine Gebühren erhoben.
            </p>

            {/* Info Box */}
            <div className="bg-blue-50 rounded-xl p-6 mb-8 text-left">
              <h2 className="text-lg font-semibold mb-3">Was passiert jetzt?</h2>
              <ul className="space-y-2 text-gray-700">
                <li>• Du kannst die Zahlung jederzeit wiederholen</li>
                <li>• Dein Account bleibt aktiv (ohne API-Key)</li>
                <li>• Keine automatischen Abbuchungen</li>
              </ul>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/payment"
                className="px-8 py-3 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition"
              >
                Zahlung erneut versuchen
              </Link>
              <Link
                href="/dashboard"
                className="px-8 py-3 border-2 border-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-50 transition flex items-center justify-center gap-2"
              >
                <ArrowLeft className="w-5 h-5" />
                Zurück zum Dashboard
              </Link>
            </div>

            {/* Support Note */}
            <div className="mt-8 text-sm text-gray-600">
              <p>
                Probleme bei der Zahlung?{' '}
                <a href="mailto:support@frametrain.ai" className="text-primary-600 hover:text-primary-700 underline">
                  Kontaktiere unseren Support
                </a>
              </p>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}
