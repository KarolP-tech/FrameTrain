# FrameTrain - Was du für Payments brauchst

## 🎯 Schnelle Antwort

Du brauchst **3 Dinge**:

### 1. Stripe Account (Payment Processing) 💳
- **Website:** https://stripe.com
- **Kostenlos:** Account erstellen
- **Was du brauchst:**
  - Publishable Key (`pk_test_...`)
  - Secret Key (`sk_test_...`)  
  - Webhook Secret (`whsec_...`)

### 2. Datenbank (User & Payments speichern) 🗄️
- **Option A:** PostgreSQL lokal (kostenlos)
- **Option B:** Supabase (kostenlos)
- **Option C:** Railway (kostenlos)

### 3. JWT Secret (User Login) 🔐
- Einfach random generieren:
  ```bash
  openssl rand -base64 32
  ```

---

## ⚡ Schnellste Methode

```bash
# 1. Interaktiver Setup (macht alles für dich!)
chmod +x setup-env.sh
./setup-env.sh
```

**Der Script fragt dich nach allem und erstellt die .env.local automatisch!**

---

## 📋 Schritt-für-Schritt

### Option 1: Interaktiv (EMPFOHLEN)

```bash
./setup-env.sh
```

Beantworte die Fragen:
1. Datenbank wählen
2. JWT Secret generieren
3. Stripe Keys eingeben
4. Fertig! ✅

### Option 2: Manuell

1. **Stripe Keys holen:**
   - Gehe zu https://dashboard.stripe.com/test/apikeys
   - Kopiere beide Keys

2. **Datenbank wählen:**
   - Lokal: `postgresql://localhost:5432/frametrain`
   - Oder Supabase/Railway URL

3. **JWT Secret generieren:**
   ```bash
   openssl rand -base64 32
   ```

4. **Datei erstellen:**
   ```bash
   cd website
   cp .env.local.example .env.local
   nano .env.local  # Oder dein Editor
   ```

5. **Ausfüllen:**
   ```bash
   DATABASE_URL="deine-db-url"
   JWT_SECRET="dein-jwt-secret"
   STRIPE_SECRET_KEY="sk_test_..."
   STRIPE_PUBLISHABLE_KEY="pk_test_..."
   STRIPE_WEBHOOK_SECRET="whsec_..."
   ```

---

## 💰 Kosten

### Development (Test-Modus):
- **Stripe:** Kostenlos, unbegrenzt testen
- **Datenbank:** Kostenlos (lokal oder Supabase/Railway Free Tier)
- **Total:** 0€

### Production (Live-Modus):
- **Stripe:** 2.9% + 0.30€ pro Transaktion
- **Datenbank:** ~5-25€/Monat (je nach Anbieter)
- **Hosting:** ~10-50€/Monat

---

## 🧪 Test-Kreditkarten

Wenn alles läuft, teste mit:

**Erfolgreiche Zahlung:**
```
Kartennummer: 4242 4242 4242 4242
Ablaufdatum:  12/34
CVC:          123
PLZ:          12345
```

**Fehlgeschlagene Zahlung:**
```
Kartennummer: 4000 0000 0000 0002
```

---

## ✅ Nach dem Setup

```bash
# 1. Datenbank initialisieren
cd website
npx prisma generate
npx prisma db push

# 2. Services starten  
cd ..
./start.sh

# 3. Website öffnen
open http://localhost:3000

# 4. Test-Registrierung durchführen
# - Registriere einen Account
# - Zahle mit Test-Kreditkarte
# - Erhalte API Key
```

---

## 📚 Detaillierte Anleitungen

- **Komplette Setup-Anleitung:** `PAYMENT_SETUP.md`
- **Development Guide:** `docs/DEVELOPMENT.md`
- **Quick Start:** `QUICKSTART_GUIDE.md`

---

## 🆘 Hilfe?

**Interaktiven Setup verwenden:**
```bash
./setup-env.sh
```

**Oder Dokumentation lesen:**
```bash
cat PAYMENT_SETUP.md
```

**Status prüfen:**
```bash
./status.sh
```

---

## 🎉 Das war's!

Nach dem Setup:
```bash
./start.sh
```

Website: http://localhost:3000

**Viel Erfolg! 🚀**
