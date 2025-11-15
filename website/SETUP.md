# FrameTrain Website - Setup & Troubleshooting

## 🚀 Quick Setup

### 1. Datenbank initialisieren (WICHTIG!)

Bevor du die Website startest, **muss** die Datenbank initialisiert werden:

```bash
cd website
chmod +x init-db.sh
./init-db.sh
```

Dieser Befehl:
- Generiert den Prisma Client
- Erstellt die SQLite Datenbank (`dev.db`)
- Wendet das Schema an

### 2. Website starten

```bash
npm run dev
```

Die Website läuft dann auf: http://localhost:5001

## 🔧 Troubleshooting

### Error 500 bei `/api/auth/register` oder `/api/auth/login`

**Problem:** Datenbank wurde nicht initialisiert

**Lösung:**
```bash
cd website
./init-db.sh
```

### "PrismaClient is unable to run in this browser environment"

**Problem:** Prisma Client wurde nicht generiert

**Lösung:**
```bash
cd website
npx prisma generate
```

### CORS Errors

CORS ist jetzt automatisch konfiguriert über `src/middleware.ts`. 

Falls du dennoch Probleme hast:
1. Prüfe ob der Server läuft: http://localhost:5001
2. Prüfe die Browser-Konsole für Details
3. Die Middleware erlaubt alle Origins im Development-Mode

### Datenbank zurücksetzen

Falls du die Datenbank komplett neu aufsetzen willst:

```bash
cd website
rm -f prisma/dev.db prisma/dev.db-journal
./init-db.sh
```

## 📊 Datenbank verwalten

Prisma Studio öffnen (GUI für die Datenbank):

```bash
cd website
npx prisma studio
```

## 🔍 Logs anschauen

Während dem Entwickeln siehst du detaillierte Error-Logs in der Konsole wo `npm run dev` läuft.

Bei 500 Errors werden jetzt **detaillierte Fehlermeldungen** ausgegeben:
- Error Message
- Stack Trace (im Development Mode)
- Prisma Error Codes (falls vorhanden)

## 📝 Environment Variables

Stelle sicher dass `.env.local` existiert und folgende Variablen enthält:

```bash
# Database
DATABASE_URL="file:./dev.db"

# JWT
JWT_SECRET="dein-geheimer-jwt-key"

# Stripe
STRIPE_SECRET_KEY="sk_test_XXXXXXXX..."
STRIPE_PUBLISHABLE_KEY="pk_test_XXXXXXXX..."
STRIPE_WEBHOOK_SECRET="whsec_XXXXXXXX..."

# API
API_URL="http://localhost:5001"
NEXT_PUBLIC_API_URL="http://localhost:5001"
```

## ✅ Setup-Checkliste

- [ ] `.env.local` existiert und ist korrekt konfiguriert
- [ ] `./init-db.sh` wurde ausgeführt
- [ ] `dev.db` existiert im `website/prisma` Ordner
- [ ] `npm install` wurde ausgeführt
- [ ] Server läuft auf Port 5001

## 🆘 Immer noch Probleme?

1. **Logs prüfen:**
   ```bash
   tail -f .pids/website.log
   ```

2. **Komplett neu starten:**
   ```bash
   cd website
   rm -rf node_modules package-lock.json
   npm install
   ./init-db.sh
   npm run dev
   ```

3. **Prisma neu generieren:**
   ```bash
   cd website
   npx prisma generate
   npx prisma db push
   ```
