# FrameTrain - Payment & API Setup Guide

## 🎯 Was du konfigurieren musst

### 1️⃣ Stripe Account (Payment Processing)

#### Account erstellen:
1. Gehe zu: https://stripe.com
2. Klicke auf "Sign up"
3. Erstelle einen Account (kostenlos)
4. Bestätige deine Email

#### API Keys holen:

**Für Development (Test-Modus):**

1. Gehe zu: https://dashboard.stripe.com/test/apikeys
2. Du siehst zwei Keys:

   **Publishable Key** (beginnt mit `pk_test_...`)
   ```
   Beispiel: pk_test_51234567890abcdefghijklmnop
   ```
   → Wird im Frontend verwendet (öffentlich)

   **Secret Key** (beginnt mit `sk_test_...`)
   ```
   Beispiel: sk_test_51234567890abcdefghijklmnop
   ```
   → Wird im Backend verwendet (geheim!)

3. Kopiere beide Keys

#### Webhook Secret einrichten:

1. Gehe zu: https://dashboard.stripe.com/test/webhooks
2. Klicke "Add endpoint"
3. Endpoint URL: `http://localhost:3000/api/payment/webhook`
4. Events auswählen:
   - `checkout.session.completed`
   - `payment_intent.succeeded`
5. Klicke "Add endpoint"
6. Kopiere den **Webhook Signing Secret** (beginnt mit `whsec_...`)
   ```
   Beispiel: whsec_1234567890abcdefghijklmnop
   ```

---

### 2️⃣ PostgreSQL Datenbank

#### Option A: Lokal (Empfohlen für Development)

```bash
# macOS
brew install postgresql@15
brew services start postgresql@15
createdb frametrain

# Connection String:
DATABASE_URL="postgresql://localhost:5432/frametrain"
```

#### Option B: Supabase (Kostenlos)

1. Gehe zu: https://supabase.com
2. "Start your project" → Sign up (kostenlos)
3. Erstelle neues Projekt:
   - Name: `frametrain`
   - Password: Wähle ein sicheres Passwort
   - Region: Nächste zu dir
4. Warte auf Projekt-Setup (1-2 Minuten)
5. Gehe zu "Project Settings" → "Database"
6. Kopiere "Connection string" (Pooler)
   ```
   Beispiel:
   postgresql://postgres.abcdefghijk:[DEIN-PASSWORD]@aws-0-eu-central-1.pooler.supabase.com:6543/postgres
   ```

#### Option C: Railway (Kostenlos)

1. Gehe zu: https://railway.app
2. Sign up mit GitHub
3. "New Project" → "Provision PostgreSQL"
4. Klicke auf PostgreSQL → "Connect"
5. Kopiere "Postgres Connection URL"
   ```
   Beispiel:
   postgresql://postgres:password@containers-us-west-123.railway.app:5432/railway
   ```

---

### 3️⃣ JWT Secret (für Authentication)

**Generiere einen zufälligen String:**

```bash
# Auf macOS/Linux:
openssl rand -base64 32

# Oder online:
# https://www.random.org/strings/
```

Beispiel Output:
```
Kx9mP2nQ5rT8vW1yZ4aC6dF7gH0jK3lM5nP8qR1sT4u
```

→ Das ist dein `JWT_SECRET`

---

## 📝 .env.local Konfiguration

Erstelle die Datei:
```bash
cd website
cp .env.local.example .env.local
nano .env.local  # oder code .env.local
```

Fülle sie so aus:

```bash
# ============================================
# DATABASE
# ============================================
# Wähle EINE der Optionen oben
DATABASE_URL="postgresql://localhost:5432/frametrain"
# ODER
# DATABASE_URL="postgresql://postgres.xyz:[PASSWORD]@...supabase.com:6543/postgres"

# ============================================
# JWT SECRET (Authentication)
# ============================================
# Generiere mit: openssl rand -base64 32
JWT_SECRET="Kx9mP2nQ5rT8vW1yZ4aC6dF7gH0jK3lM5nP8qR1sT4u"

# ============================================
# STRIPE (Payment)
# ============================================
# Von https://dashboard.stripe.com/test/apikeys
STRIPE_SECRET_KEY="sk_test_51234567890abcdefghijklmnop"
STRIPE_PUBLISHABLE_KEY="pk_test_51234567890abcdefghijklmnop"

# Von https://dashboard.stripe.com/test/webhooks
STRIPE_WEBHOOK_SECRET="whsec_1234567890abcdefghijklmnop"

# ============================================
# API CONFIGURATION
# ============================================
# Für Development immer localhost
API_URL="http://localhost:3000"
NEXT_PUBLIC_API_URL="http://localhost:3000"

# ============================================
# APP DOWNLOAD (Für später in Production)
# ============================================
# Für Development kannst du das leer lassen
APP_DOWNLOAD_BASE_URL="http://localhost:3000/downloads"

# ============================================
# ENVIRONMENT
# ============================================
NODE_ENV="development"
```

---

## ✅ Checkliste

Gehe durch und hake ab:

### Stripe (https://stripe.com)
- [ ] Account erstellt
- [ ] Test-Modus aktiviert (sollte default sein)
- [ ] **Publishable Key** kopiert (`pk_test_...`)
- [ ] **Secret Key** kopiert (`sk_test_...`)
- [ ] Webhook endpoint erstellt
- [ ] **Webhook Secret** kopiert (`whsec_...`)

### Datenbank
- [ ] PostgreSQL installiert ODER
- [ ] Supabase Account erstellt ODER
- [ ] Railway Account erstellt
- [ ] **Connection String** kopiert

### JWT Secret
- [ ] Random String generiert (32+ Zeichen)

### .env.local Datei
- [ ] Datei erstellt: `website/.env.local`
- [ ] `DATABASE_URL` eingetragen
- [ ] `JWT_SECRET` eingetragen
- [ ] `STRIPE_SECRET_KEY` eingetragen
- [ ] `STRIPE_PUBLISHABLE_KEY` eingetragen
- [ ] `STRIPE_WEBHOOK_SECRET` eingetragen

---

## 🧪 Testen

Nachdem alles konfiguriert ist:

```bash
# 1. Datenbank initialisieren
cd website
npx prisma generate
npx prisma db push

# 2. Services starten
cd ..
./start.sh

# 3. Status prüfen
./status.sh

# 4. Website öffnen
open http://localhost:3000
```

---

## 🔍 Stripe Test-Zahlung

Wenn alles läuft, teste die Zahlung:

### Test-Kreditkarten (funktionieren nur im Test-Modus):

**Erfolgreiche Zahlung:**
- Kartennummer: `4242 4242 4242 4242`
- Ablaufdatum: Beliebig in der Zukunft (z.B. `12/34`)
- CVC: Beliebig (z.B. `123`)
- PLZ: Beliebig (z.B. `12345`)

**Fehlgeschlagene Zahlung (zum Testen):**
- Kartennummer: `4000 0000 0000 0002`

**3D Secure erforderlich:**
- Kartennummer: `4000 0027 6000 3184`

Mehr Test-Karten: https://stripe.com/docs/testing

---

## ⚠️ Wichtig

### Development vs Production

**Test-Modus (Development):**
- Keys beginnen mit `pk_test_...` und `sk_test_...`
- Keine echten Zahlungen
- Nur Test-Kreditkarten funktionieren
- Kostenlos unbegrenzt testen

**Live-Modus (Production):**
- Keys beginnen mit `pk_live_...` und `sk_live_...`
- Echte Zahlungen
- Stripe nimmt Gebühren (2.9% + 0.30€ pro Transaktion)
- Braucht verifiziertes Business-Konto

→ **Für Development immer Test-Modus verwenden!**

### Sicherheit

❌ **NIE committen:**
- `.env.local` Datei
- Secret Keys
- Webhook Secrets
- Passwörter

✅ **Immer in .gitignore:**
```
.env.local
.env*.local
.env.production
```

---

## 🆘 Troubleshooting

### "Invalid API key"
→ Prüfe ob du Test-Keys verwendest (`pk_test_...` / `sk_test_...`)

### "Webhook signature verification failed"
→ Prüfe ob `STRIPE_WEBHOOK_SECRET` korrekt ist

### "Database connection failed"
→ Prüfe `DATABASE_URL` Format und ob DB läuft

### Stripe Dashboard zeigt keine Events
→ Normal! Im Development werden Webhooks nur bei echten Zahlungen getriggert.
   Für lokales Testing: https://stripe.com/docs/stripe-cli

---

## 📚 Weitere Ressourcen

- **Stripe Docs:** https://stripe.com/docs
- **Stripe Testing:** https://stripe.com/docs/testing
- **Supabase Docs:** https://supabase.com/docs
- **Prisma Docs:** https://www.prisma.io/docs

---

## 🎉 Fertig!

Nachdem alles konfiguriert ist:

```bash
./start.sh
```

Website: http://localhost:3000

Viel Erfolg! 🚀
