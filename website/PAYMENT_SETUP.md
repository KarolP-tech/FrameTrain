# Stripe Payment Integration - Setup Guide

## ✅ Was wurde implementiert:

### 1. Payment Flow
```
Registrierung → Auto-Login → Payment-Seite → Stripe Checkout → Success/Cancel
```

### 2. API-Endpunkte
- ✅ `/api/payment/create-checkout` - Erstellt Stripe Checkout Session
- ✅ `/api/payment/webhook` - Empfängt Stripe Webhooks
- ✅ `/payment` - Payment-Seite
- ✅ `/payment/success` - Erfolgs-Seite nach Zahlung
- ✅ `/payment/cancel` - Abbruch-Seite

### 3. Features
- ✅ Sichere Zahlung über Stripe Checkout
- ✅ Webhook für Payment-Bestätigung
- ✅ Payment-Tracking in Datenbank
- ✅ Redirect nach erfolgreicher Zahlung

## 🔧 Konfiguration

### Environment Variables (bereits konfiguriert):
```bash
STRIPE_SECRET_KEY="sk_test_XXXXXXXX..."
STRIPE_PUBLISHABLE_KEY="pk_test_XXXXXXXX..."
STRIPE_WEBHOOK_SECRET="whsec_XXXXXXXX..."
STRIPE_PRICE_ID="price_XXXXXXXX..."
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY="pk_test_XXXXXXXX..."
```

### Stripe Dashboard Setup:

1. **Webhook einrichten:**
   ```
   URL: http://localhost:5001/api/payment/webhook
   Events: checkout.session.completed, checkout.session.expired
   ```

2. **Webhook Secret kopieren:**
   - Gehe zu Stripe Dashboard → Developers → Webhooks
   - Klicke auf deinen Webhook
   - Kopiere den "Signing secret"
   - Füge ihn als `STRIPE_WEBHOOK_SECRET` in .env.local ein

## 🧪 Testen (Lokal):

### 1. Stripe CLI installieren:
```bash
brew install stripe/stripe-cli/stripe
# oder
curl -s https://packages.stripe.com/api/security/keypair/stripe-cli-gpg/public | gpg --dearmor | sudo tee /usr/share/keyrings/stripe.gpg
echo "deb [signed-by=/usr/share/keyrings/stripe.gpg] https://packages.stripe.com/stripe-cli-debian-local stable main" | sudo tee -a /etc/apt/sources.list.d/stripe.list
sudo apt update
sudo apt install stripe
```

### 2. Stripe CLI Login:
```bash
stripe login
```

### 3. Webhook Forwarding starten:
```bash
stripe listen --forward-to localhost:5001/api/payment/webhook
```

Das gibt dir einen Webhook-Secret wie:
```
whsec_XXXXXXXX...
```

Füge diesen in `.env.local` als `STRIPE_WEBHOOK_SECRET` ein.

### 4. Test-Payment durchführen:

1. Registriere dich mit neuer E-Mail
2. Du wirst zur Payment-Seite weitergeleitet
3. Klicke "Jetzt sicher bezahlen"
4. Verwende Stripe Test-Karte:
   ```
   Karte: 4242 4242 4242 4242
   Ablaufdatum: 12/34
   CVC: 123
   PLZ: 12345
   ```
5. Nach erfolgreicher Zahlung: Weiterleitung zu `/payment/success`

## 📊 Payment-Flow Details:

### User Journey:
1. **Registrierung** (`/register`)
   - User gibt E-Mail + Passwort ein
   - Account wird erstellt
   - Auto-Login mit Cookie
   - Redirect zu `/payment`

2. **Payment-Seite** (`/payment`)
   - Zeigt Preis (1,99€)
   - Button "Jetzt sicher bezahlen"
   - Klick erstellt Stripe Checkout Session
   - Redirect zu Stripe Checkout

3. **Stripe Checkout**
   - User gibt Kartendaten ein
   - Stripe verarbeitet Zahlung
   - Bei Erfolg: Redirect zu `/payment/success?session_id=xxx`
   - Bei Abbruch: Redirect zu `/payment/cancel`

4. **Webhook** (`/api/payment/webhook`)
   - Stripe sendet Event `checkout.session.completed`
   - Payment wird in DB gespeichert
   - (TODO: API-Key generieren & per E-Mail senden)

5. **Success-Seite** (`/payment/success`)
   - Bestätigung der Zahlung
   - Anleitung für nächste Schritte
   - Link zum Dashboard

## 🔐 Sicherheit:

- ✅ Stripe Checkout (PCI-compliant)
- ✅ Webhook-Signatur-Verifizierung
- ✅ HTTPS in Production
- ✅ Keine Kartendaten im Code
- ✅ Session-basierte Auth

## 🚀 Production Deployment:

### 1. Webhook URL ändern:
```
https://your-domain.com/api/payment/webhook
```

### 2. Live-Keys verwenden:
```bash
STRIPE_SECRET_KEY="sk_live_XXXXXXXX..."
STRIPE_PUBLISHABLE_KEY="pk_live_XXXXXXXX..."
STRIPE_PRICE_ID="price_XXXXXXXX..." # Live Price ID
```

### 3. Webhook neu einrichten:
- In Stripe Dashboard mit Production URL
- Neuen Webhook Secret in Production .env setzen

## 📝 TODO:

- [ ] API-Key Generierung nach erfolgreicher Zahlung
- [ ] E-Mail-Versand mit API-Key
- [ ] Payment-Status im Dashboard anzeigen
- [ ] Mehrfache Zahlungen verhindern
- [ ] Rückerstattungs-Logik

## 🐛 Troubleshooting:

### Webhook wird nicht empfangen:
```bash
# Prüfe ob Stripe CLI läuft
stripe listen --forward-to localhost:5001/api/payment/webhook

# Logs anschauen
tail -f .pids/website.log
```

### Payment funktioniert nicht:
```bash
# Prüfe Environment Variables
cd website
cat .env.local | grep STRIPE

# Prüfe Stripe Dashboard
# https://dashboard.stripe.com/test/payments
```

### Test-Karten:
```
Erfolg: 4242 4242 4242 4242
3D Secure: 4000 0027 6000 3184
Abgelehnt: 4000 0000 0000 0002
```

## 📚 Links:

- [Stripe Checkout Docs](https://stripe.com/docs/checkout)
- [Stripe Webhooks](https://stripe.com/docs/webhooks)
- [Stripe Test Cards](https://stripe.com/docs/testing)
