'use client';

import { useState } from 'react';

interface PaymentFormProps {
  email: string;
  onSuccess: () => void;
}

export function PaymentForm({ email, onSuccess }: PaymentFormProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePayment = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/payment/create-checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        setError(data.error || 'Fehler beim Erstellen der Checkout-Session');
        setLoading(false);
        return;
      }

      // Redirect to Stripe Checkout
      if (data.url) {
        window.location.href = data.url;
      } else {
        setError('Keine Checkout-URL erhalten');
        setLoading(false);
      }
    } catch (err) {
      console.error('Payment error:', err);
      setError('Ein Fehler ist aufgetreten. Bitte versuche es erneut.');
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h3 className="text-xl font-semibold mb-4">Zahlung</h3>
      
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-gray-600">FrameTrain Lizenz</span>
          <span className="font-semibold">2,00 €</span>
        </div>
        <p className="text-sm text-gray-500">Einmalige Zahlung, lebenslanger Zugang</p>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-600 text-sm">
          {error}
        </div>
      )}

      <button
        onClick={handlePayment}
        disabled={loading}
        className="w-full py-3 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition disabled:bg-gray-400 disabled:cursor-not-allowed"
      >
        {loading ? 'Wird verarbeitet...' : 'Jetzt bezahlen mit Stripe'}
      </button>

      <p className="text-xs text-gray-500 text-center mt-4">
        Sichere Zahlung über Stripe. Deine Zahlungsdaten werden verschlüsselt übertragen.
      </p>
    </div>
  );
}
