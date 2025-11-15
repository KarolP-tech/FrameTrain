# FrameTrain 🚀

<div align="center">
  <h3>Professionelle Plattform für lokales Machine Learning Training</h3>
  <p>Train ML models locally with full control over your data</p>
  
  [![License](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
  [![Build](https://github.com/KarolP-tech/FrameTrain/actions/workflows/build-desktop.yml/badge.svg)](https://github.com/KarolP-tech/FrameTrain/actions)
  [![Website](https://img.shields.io/badge/Website-frametrain.ai-purple)](https://frametrain.ai)
  [![Downloads](https://img.shields.io/github/downloads/KarolP-tech/FrameTrain/total)](https://github.com/KarolP-tech/FrameTrain/releases)
</div>

---

## 📋 Überblick

FrameTrain ist eine vollständige Plattform für lokales Machine Learning Training. Sie besteht aus:

- 🌐 **Website** - Landing, Registration, Payment & Dashboard
- 🖥️ **Desktop-App** - Lokales ML-Training mit GUI
- ⌨️ **CLI** - Command-line Tool für Installation & Management
- 📦 **Shared** - Gemeinsame TypeScript Module

### ✨ Features

- 🔒 **100% Lokal** - Alle Daten bleiben auf deinem Gerät
- 🤗 **HuggingFace Integration** - Modelle direkt importieren
- 📊 **Live Monitoring** - Training in Echtzeit verfolgen
- 📦 **Versionsverwaltung** - Modellversionen verwalten & vergleichen
- ⚡ **GPU Support** - PyTorch mit CUDA
- 🛡️ **DSGVO-konform** - Keine Cloud, keine Datenübertragung

## 🏗️ Architektur

```
FrameTrain/
├── website/              # Next.js 14 + Prisma + Stripe
│   ├── src/app/         # Pages & API Routes
│   ├── src/components/  # React Components
│   └── prisma/          # Database Schema
│
├── desktop-app/          # Tauri + React + PyTorch
│   ├── src/             # React Frontend
│   ├── src-tauri/       # Rust Backend
│   └── ml_backend/      # Python ML Scripts
│
├── cli/                  # Python Click CLI
│   └── frametrain/      # CLI Commands
│
├── shared/               # Gemeinsame TypeScript Module
│   └── src/             # Types & Utils
│
└── docs/                 # Dokumentation
    ├── DEVELOPMENT.md   # Development Guide
    ├── DEPLOYMENT.md    # Deployment Guide
    └── API.md           # API Documentation
```

## 🚀 Quick Start

### ⚡ One-Command Setup (Einfachste Methode)

```bash
chmod +x quickstart.sh
./quickstart.sh
```

Dieser Script:
1. Installiert Rust automatisch
2. Führt System-Check aus
3. Installiert alle Dependencies
4. Erstellt .env.local
5. Startet alle Services

### Voraussetzungen prüfen

```bash
chmod +x test.sh
./test.sh
```

Falls Rust fehlt:
```bash
chmod +x install-rust.sh
./install-rust.sh
```

### Automatisches Setup (Empfohlen)

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

### Manuelles Setup

#### 1. Website

```bash
cd website
npm install
cp .env.local.example .env.local
# Bearbeite .env.local mit deinen Credentials
npx prisma generate
npx prisma db push
npm run dev
```

→ http://localhost:5001

#### 2. Desktop-App

```bash
cd desktop-app
npm install

# ML Backend Setup
cd ml_backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..

# App starten
npm run tauri:dev
```

#### 3. CLI

```bash
cd cli
pip install -e .
frametrain --help
```

## 💻 Technologie-Stack

### Frontend
- **Next.js 14** - React Framework mit App Router
- **React 18** - UI Library
- **TailwindCSS** - Utility-First CSS
- **Recharts** - Charting Library

### Backend
- **Next.js API Routes** - RESTful API
- **Prisma** - ORM für PostgreSQL
- **JWT** - Authentication
- **Stripe** - Payment Processing

### Desktop
- **Tauri** - Cross-Platform Desktop Framework (Rust)
- **React** - Frontend
- **PyTorch** - ML Training Backend
- **HuggingFace Hub** - Model Repository

### CLI
- **Python 3.8+** - Runtime
- **Click** - CLI Framework
- **Requests** - HTTP Client

### Database
- **PostgreSQL** - Relational Database
- **Prisma** - ORM & Migrations

## 📚 Dokumentation

- 📖 [Development Guide](./docs/DEVELOPMENT.md) - Setup & Entwicklung
- 🚀 [Deployment Guide](./docs/DEPLOYMENT.md) - Production Deployment
- 🔌 [API Documentation](./docs/API.md) - REST API Referenz
- 🖥️ [ML Backend Guide](./desktop-app/ml_backend/README.md) - Training Backend
- ⌨️ [CLI Documentation](./cli/README.md) - CLI Befehle

## 🎯 Verwendung

### Für Nutzer

1. Auf [frametrain.ai](https://frametrain.ai) registrieren
2. 2€ bezahlen → API Key erhalten
3. CLI installieren: `pip install frametrain-cli`
4. Desktop-App installieren: `frametrain install`
5. App starten: `frametrain start`
6. Modelle trainieren! 🎉

### Für Entwickler

```bash
# Repository klonen
git clone https://github.com/KarolP-tech/FrameTrain.git
cd FrameTrain

# Setup ausführen
./setup.sh  # oder setup.bat auf Windows

# Development starten
cd website && npm run dev          # Website
cd desktop-app && npm run tauri:dev  # Desktop-App
frametrain --help                   # CLI
```

Siehe [DEVELOPMENT.md](./docs/DEVELOPMENT.md) für Details.

## 🏃‍♂️ Development Workflow

### Mit Management Scripts (Empfohlen)

```bash
# Alle Services starten
./start.sh

# Status prüfen
./status.sh

# Services stoppen
./stop.sh

# Neu starten
./restart.sh

# Logs verfolgen
tail -f .pids/*.log
```

### Manuell

```bash
# Website entwickeln
cd website && npm run dev

# Desktop-App entwickeln
cd desktop-app && npm run tauri:dev

# CLI testen
cd cli && frametrain verify-key

# Database GUI
cd website && npx prisma studio

# Tests
npm test  # In jedem Modul
```

## 🔐 Environment Variables

Beispiel `.env.local` für Website:

```bash
DATABASE_URL="postgresql://user:pass@localhost:5432/frametrain"
JWT_SECRET="your-super-secret-jwt-key"
STRIPE_SECRET_KEY="sk_test_XXXXXXXX..."
STRIPE_PUBLISHABLE_KEY="pk_test_XXXXXXXX..."
STRIPE_WEBHOOK_SECRET="whsec_XXXXXXXX..."
NEXT_PUBLIC_API_URL="http://localhost:5001"
APP_DOWNLOAD_BASE_URL="https://downloads.frametrain.ai"
```

## 🧪 Testing

```bash
# Unit Tests
npm test

# E2E Tests
npm run test:e2e

# Coverage
npm run test:coverage
```

## 📦 Building

```bash
# Website Production Build
cd website && npm run build

# Desktop-App Build
cd desktop-app && npm run tauri:build
# Output: src-tauri/target/release/bundle/

# CLI Distribution
cd cli && python -m build
```

## 🤝 Contributing

Wir freuen uns über Beiträge! 🎉

**Erlaubte Contributions:**
- 🐛 Bug Fixes
- 📝 Dokumentation
- ✨ Feature Requests
- 🧪 Tests
- 🎨 UI Verbesserungen

**Workflow:**
1. Fork das Repository
2. Feature Branch erstellen: `git checkout -b feature/amazing-feature`
3. Änderungen committen: `git commit -m 'Add amazing feature'`
4. Branch pushen: `git push origin feature/amazing-feature`
5. Pull Request öffnen

**Code of Conduct:** Sei respektvoll und konstruktiv.

**Lizenz:** Alle Contributions fallen unter die BSL 1.1 Lizenz.

Für größere Features: Öffne erst ein Issue zur Diskussion!

## 📄 Lizenz

**Business Source License 1.1**

Dieser Code ist unter der Business Source License 1.1 lizenziert.

**Das bedeutet:**
- ✅ Du kannst den Code ansehen und lernen
- ✅ Du kannst für persönliche Zwecke nutzen
- ✅ Du kannst Bugs melden und beitragen
- ❌ Keine kommerzielle Nutzung ohne Lizenz
- ❌ Keine Forks für konkurrierende Produkte

**Kommerzielle Nutzung:** Kaufe eine Lizenz für 2€ auf [frametrain.ai](https://frametrain.ai)

**Open Source Future:** Nach 4 Jahren (2028) wird der Code unter Apache 2.0 verfügbar.

Details siehe [LICENSE](./LICENSE)

## 🆘 Support

- 📧 Email: support@frametrain.ai
- 📖 Dokumentation: https://docs.frametrain.ai
- 🐛 Issues: https://github.com/frametrain/frametrain/issues

## 🙏 Credits

Erstellt mit:
- [Next.js](https://nextjs.org/)
- [Tauri](https://tauri.app/)
- [PyTorch](https://pytorch.org/)
- [HuggingFace](https://huggingface.co/)
- [Stripe](https://stripe.com/)

---

<div align="center">
  Made with ❤️ by FrameTrain Team
</div>
