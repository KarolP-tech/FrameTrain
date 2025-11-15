# FrameTrain Development Setup

Vollständiger Guide zur Einrichtung der Entwicklungsumgebung.

## Voraussetzungen

- **Node.js** 18+ und npm/pnpm
- **Python** 3.8+
- **Rust** (für Tauri)
- **PostgreSQL** (für Website)
- **Git**

## Projekt-Struktur

```
FrameTrain/
├── website/          # Next.js Frontend & Backend
├── desktop-app/      # Tauri Desktop-App
├── cli/             # Python CLI
├── shared/          # Gemeinsame TypeScript-Module
└── docs/            # Dokumentation
```

## Setup

### 1. Repository klonen

```bash
git clone https://github.com/yourusername/frametrain.git
cd frametrain
```

### 2. Website Setup

```bash
cd website

# Dependencies installieren
npm install

# Environment Variables
cp .env.local.example .env.local
# Bearbeite .env.local und füge deine Credentials ein

# Datenbank aufsetzen
npx prisma generate
npx prisma db push

# Development Server starten
npm run dev
```

Die Website läuft auf `http://localhost:3000`

**Environment Variables (.env.local):**
- `DATABASE_URL`: PostgreSQL Connection String
- `JWT_SECRET`: Geheimer Schlüssel für JWT
- `STRIPE_SECRET_KEY`: Stripe Secret Key
- `STRIPE_PUBLISHABLE_KEY`: Stripe Public Key
- `STRIPE_WEBHOOK_SECRET`: Stripe Webhook Secret

### 3. Desktop-App Setup

```bash
cd desktop-app

# Dependencies installieren
npm install

# Python ML Backend Setup
cd ml_backend
pip install torch torchvision huggingface-hub transformers
cd ..

# Tauri Development starten
npm run tauri:dev
```

**Hinweis:** Beim ersten Start wird Tauri's Rust-Code kompiliert, das kann einige Minuten dauern.

### 4. CLI Setup

```bash
cd cli

# In Development-Mode installieren
pip install -e .

# Testen
frametrain --help
```

### 5. Shared Module Setup

```bash
cd shared

# Dependencies installieren
npm install

# TypeScript kompilieren
npm run build
```

## Development Workflow

### Website entwickeln

```bash
cd website
npm run dev
```

- Hot-Reload aktiviert
- API Routes unter `/api/*`
- Prisma Studio: `npm run db:studio`

### Desktop-App entwickeln

```bash
cd desktop-app
npm run tauri:dev
```

- Hot-Reload für React-Frontend
- Rust-Backend kompiliert bei Änderungen neu

### CLI testen

```bash
cd cli

# Befehle testen
frametrain install
frametrain verify-key
frametrain start
```

## Datenbank

### Prisma Migrations

```bash
cd website

# Schema ändern in prisma/schema.prisma, dann:
npx prisma db push

# Oder Migration erstellen:
npx prisma migrate dev --name migration_name
```

### Prisma Studio (GUI)

```bash
npx prisma studio
```

Öffnet GUI auf `http://localhost:5555`

## Testing

### Website Tests

```bash
cd website
npm run test        # Unit Tests
npm run test:e2e    # E2E Tests
```

### Desktop-App Tests

```bash
cd desktop-app
npm run test
```

### CLI Tests

```bash
cd cli
pytest
```

## Building

### Website Production Build

```bash
cd website
npm run build
npm run start  # Production server
```

### Desktop-App Build

```bash
cd desktop-app

# Build für aktuelles OS
npm run tauri:build

# Fertige Installer in:
# src-tauri/target/release/bundle/
```

### CLI Distribution

```bash
cd cli

# Wheel erstellen
python -m build

# Oder direkt installieren
pip install .
```

## Troubleshooting

### Port bereits belegt
```bash
# Website Port ändern
PORT=3001 npm run dev
```

### PostgreSQL Verbindungsfehler
- Prüfe ob PostgreSQL läuft: `pg_ctl status`
- Prüfe DATABASE_URL in .env.local
- Erstelle DB wenn nötig: `createdb frametrain`

### Tauri Build-Fehler
- Stelle sicher, dass Rust installiert ist: `rustc --version`
- Update Rust: `rustup update`
- Clean build: `cargo clean` im src-tauri/ Ordner

### Python Import-Fehler
- Virtuelle Umgebung aktivieren
- Dependencies neu installieren: `pip install -r requirements.txt`

## VS Code Setup

Empfohlene Extensions:
- ESLint
- Prisma
- Tailwind CSS IntelliSense
- rust-analyzer
- Python

`.vscode/settings.json`:
```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "typescript.tsdk": "node_modules/typescript/lib"
}
```

## Git Workflow

```bash
# Feature Branch erstellen
git checkout -b feature/my-feature

# Änderungen committen
git add .
git commit -m "feat: add new feature"

# Push und Pull Request erstellen
git push origin feature/my-feature
```

## Nützliche Befehle

```bash
# Alle Dependencies aktualisieren
npm run update-deps  # In website/ und desktop-app/

# Projekt komplett neu aufsetzen
npm run clean && npm install

# Logs anzeigen
npm run logs

# Production Build testen
npm run build && npm run start
```

## Support

Bei Fragen oder Problemen:
- Issues auf GitHub erstellen
- Dokumentation: https://docs.frametrain.ai
- Slack Channel: #frametrain-dev

## Lizenz

Proprietär - Alle Rechte vorbehalten
