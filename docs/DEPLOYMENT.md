# FrameTrain Deployment Guide

Vollständiger Guide für das Deployment von FrameTrain in Production.

## Übersicht

FrameTrain besteht aus mehreren Komponenten:
1. **Website** (Next.js) - Landing, Registration, Payment, Dashboard
2. **API Backend** (Next.js API Routes) - Authentication, Key Management
3. **Desktop-App** (Tauri) - Lokale ML-Trainings-Anwendung
4. **CLI** (Python) - Installation & Management Tool
5. **Datenbank** (PostgreSQL) - User, Keys, Models

## Deployment-Architektur

```
┌─────────────────┐
│   CloudFlare    │  CDN & DDoS Protection
│   (Optional)    │
└────────┬────────┘
         │
┌────────▼────────┐
│   Vercel/       │  Website + API
│   Render        │  (Next.js)
└────────┬────────┘
         │
┌────────▼────────┐
│   PostgreSQL    │  Datenbank
│   (Managed)     │
└─────────────────┘

┌─────────────────┐
│   CDN/Storage   │  Desktop-App Downloads
│   (S3/R2)       │  (.exe, .dmg, .AppImage)
└─────────────────┘

┌─────────────────┐
│   PyPI/GitHub   │  CLI Distribution
│   Releases      │
└─────────────────┘
```

## 1. Website Deployment

### Option A: Vercel (Empfohlen)

**Vorteile:** Zero-Config, Auto-Deploy, Serverless

```bash
# Vercel CLI installieren
npm i -g vercel

# Projekt deployen
cd website
vercel

# Production Deployment
vercel --prod
```

**Environment Variables in Vercel:**
- `DATABASE_URL`
- `JWT_SECRET`
- `STRIPE_SECRET_KEY`
- `STRIPE_PUBLISHABLE_KEY`
- `STRIPE_WEBHOOK_SECRET`
- `NEXT_PUBLIC_API_URL`
- `APP_DOWNLOAD_BASE_URL`

**vercel.json:**
```json
{
  "buildCommand": "npm run build",
  "devCommand": "npm run dev",
  "installCommand": "npm install",
  "framework": "nextjs",
  "regions": ["fra1"]
}
```

### Option B: Render

```bash
# render.yaml erstellen
cd website
```

**render.yaml:**
```yaml
services:
  - type: web
    name: frametrain-web
    env: node
    buildCommand: npm install && npm run build
    startCommand: npm start
    envVars:
      - key: DATABASE_URL
        sync: false
      - key: JWT_SECRET
        generateValue: true
      - key: NODE_ENV
        value: production
```

### Option C: Docker (VPS)

**Dockerfile:**
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV production
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - db
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=frametrain
      - POSTGRES_USER=frametrain
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

## 2. Datenbank Setup

### Managed PostgreSQL (Empfohlen)

**Optionen:**
- **Vercel Postgres** (integriert)
- **Supabase** (kostenloser Tier)
- **Railway** (Developer-friendly)
- **AWS RDS** (Enterprise)

**Setup mit Vercel Postgres:**
```bash
# In Vercel Dashboard:
# 1. Storage → Create Database → Postgres
# 2. DATABASE_URL wird automatisch gesetzt

# Prisma Migration durchführen:
cd website
npx prisma migrate deploy
```

**Setup mit Supabase:**
```bash
# 1. Projekt auf supabase.com erstellen
# 2. Connection String kopieren
# 3. In .env setzen:
DATABASE_URL="postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres"

# Migration durchführen
npx prisma migrate deploy
```

## 3. Desktop-App Distribution

### Build für alle Plattformen

**GitHub Actions (.github/workflows/build-app.yml):**
```yaml
name: Build Desktop App

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: 18
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Install dependencies
        run: |
          cd desktop-app
          npm install
      
      - name: Build App
        run: |
          cd desktop-app
          npm run tauri:build
      
      - name: Upload Artifacts
        uses: actions/upload-artifact@v3
        with:
          name: app-${{ matrix.os }}
          path: desktop-app/src-tauri/target/release/bundle/
```

### App-Hosting

**Option A: GitHub Releases**
```bash
# Release erstellen
gh release create v1.0.0 \
  desktop-app/src-tauri/target/release/bundle/dmg/*.dmg \
  desktop-app/src-tauri/target/release/bundle/msi/*.msi \
  desktop-app/src-tauri/target/release/bundle/appimage/*.AppImage
```

**Option B: CloudFlare R2 / AWS S3**
```bash
# Upload zu R2
aws s3 cp \
  desktop-app/src-tauri/target/release/bundle/ \
  s3://frametrain-downloads/ \
  --recursive

# CDN URL:
# https://downloads.frametrain.ai/v1.0.0/FrameTrain-1.0.0.dmg
```

**Download-Endpoint aktualisieren:**
```typescript
// website/src/app/api/download-app/route.ts
const DOWNLOAD_URLS = {
  windows: 'https://downloads.frametrain.ai/latest/FrameTrain-Setup.exe',
  macos: 'https://downloads.frametrain.ai/latest/FrameTrain.dmg',
  linux: 'https://downloads.frametrain.ai/latest/FrameTrain.AppImage',
};
```

## 4. CLI Distribution

### PyPI veröffentlichen

```bash
cd cli

# Build
python -m build

# Upload zu PyPI
python -m twine upload dist/*

# Installation für Nutzer:
# pip install frametrain-cli
```

### GitHub Releases (Alternative)

```bash
# Nutzer installieren via:
pip install git+https://github.com/yourusername/frametrain-cli.git
```

## 5. Stripe Payment Setup

### Webhook konfigurieren

1. **Stripe Dashboard:** Developers → Webhooks → Add endpoint
2. **Endpoint URL:** `https://yourdomain.com/api/payment/webhook`
3. **Events auswählen:**
   - `checkout.session.completed`
   - `payment_intent.succeeded`

4. **Webhook Secret kopieren** → zu Environment Variables hinzufügen

### Webhook testen

```bash
# Stripe CLI installieren
brew install stripe/stripe-cli/stripe

# Login
stripe login

# Webhook forwarden (Development)
stripe listen --forward-to localhost:3000/api/payment/webhook
```

## 6. Monitoring & Logging

### Vercel Analytics
```bash
# In package.json hinzufügen:
npm install @vercel/analytics

# In layout.tsx:
import { Analytics } from '@vercel/analytics/react'
```

### Sentry (Error Tracking)
```bash
npm install @sentry/nextjs

# sentry.client.config.js erstellen
```

### Logging
```typescript
// lib/logger.ts
import pino from 'pino'

export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    target: 'pino-pretty',
  },
})
```

## 7. Security Checklist

- [ ] Environment Variables korrekt gesetzt
- [ ] JWT_SECRET stark und zufällig
- [ ] HTTPS aktiviert
- [ ] CORS richtig konfiguriert
- [ ] Rate Limiting implementiert
- [ ] SQL Injection Prevention (Prisma macht das)
- [ ] XSS Prevention (React macht das)
- [ ] CSRF Protection
- [ ] Stripe Webhook Signatures verifiziert
- [ ] API Keys verschlüsselt gespeichert
- [ ] Sensitive Logs entfernt

## 8. Performance Optimization

### Next.js Optimierungen
```javascript
// next.config.js
module.exports = {
  images: {
    domains: ['your-cdn.com'],
  },
  compress: true,
  poweredByHeader: false,
  generateEtags: true,
}
```

### Database Indexing
```prisma
// Indizes in schema.prisma bereits definiert
@@index([key])
@@index([userId])
```

### CDN Setup
- Statische Assets über CDN ausliefern
- CloudFlare kostenlos nutzen
- Cache-Control Headers setzen

## 9. Backup & Recovery

### Database Backups
```bash
# Automatische Backups bei Managed Services
# Oder manuell:
pg_dump $DATABASE_URL > backup-$(date +%Y%m%d).sql

# Wiederherstellen:
psql $DATABASE_URL < backup-20240115.sql
```

### Code Backups
- Git Repository auf GitHub/GitLab
- Regelmäßige Tags für Releases
- CI/CD Artifacts aufbewahren

## 10. CI/CD Pipeline

**GitHub Actions (.github/workflows/deploy.yml):**
```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy-website:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: cd website && npm ci
      - run: cd website && npm run build
      - run: cd website && vercel --prod --token=${{ secrets.VERCEL_TOKEN }}
  
  deploy-app:
    runs-on: ubuntu-latest
    steps:
      # Build & Upload Desktop App
      # ... (siehe Build-App Workflow oben)
```

## Support & Wartung

### Monitoring
- Uptime Monitoring: UptimeRobot, Pingdom
- Error Tracking: Sentry
- Analytics: Vercel Analytics, Plausible

### Updates
- Regelmäßige Dependency Updates
- Security Patches sofort einspielen
- Desktop-App Auto-Update Mechanismus

### Scaling
- Vercel skaliert automatisch
- Database Connection Pooling bei Bedarf
- CDN für statische Assets

## Checkliste für Go-Live

- [ ] Domain registriert & DNS konfiguriert
- [ ] SSL Zertifikat (automatisch bei Vercel)
- [ ] Database Production-ready
- [ ] Environment Variables gesetzt
- [ ] Stripe Production Keys
- [ ] Monitoring aktiv
- [ ] Backups konfiguriert
- [ ] Desktop-App gebaut & hochgeladen
- [ ] CLI auf PyPI veröffentlicht
- [ ] Dokumentation aktualisiert
- [ ] Legal Pages (Impressum, Datenschutz)

## Kosten (geschätzt)

- **Vercel:** Free Tier ausreichend für Start
- **Database:** Supabase Free / $25-50/Monat
- **Storage:** CloudFlare R2 ~$5/Monat
- **Domain:** ~$10/Jahr
- **Total:** ~$10-60/Monat (abhängig von Nutzerzahlen)

## Support

Bei Fragen zum Deployment:
- Dokumentation: https://docs.frametrain.ai
- GitHub Issues: https://github.com/frametrain/frametrain/issues
