# FrameTrain Release & Distribution Guide

## 📦 Release Workflow

### 1. Erstelle einen neuen Release

```bash
# Stelle sicher, dass alle Änderungen committed sind
git add .
git commit -m "Release v1.0.0"

# Erstelle einen Tag für die Version
git tag v1.0.0

# Pushe den Tag zu GitHub
git push origin v1.0.0
```

**Was passiert dann automatisch:**
1. 🤖 GitHub Actions startet automatisch
2. 🏗️ Baut Desktop-App für Windows, macOS, Linux
3. 📦 Erstellt Installer (.msi, .dmg, .AppImage)
4. 🚀 Lädt alles zu GitHub Releases hoch
5. ✅ Release ist verfügbar für Downloads

### 2. Verifiziere den Build

1. Gehe zu: `https://github.com/YourUsername/FrameTrain/actions`
2. Prüfe ob der "Build Desktop App" Workflow erfolgreich war
3. Prüfe die Artifacts:
   - ✅ Windows: `.msi` oder `.exe`
   - ✅ macOS: `.dmg`
   - ✅ Linux: `.AppImage`

### 3. Release veröffentlichen

1. Gehe zu: `https://github.com/YourUsername/FrameTrain/releases`
2. Der Release wurde automatisch erstellt
3. Optional: Füge Release Notes hinzu
4. Klicke "Publish release"

---

## 🌐 Website Download Setup

### Environment Variables

Füge zu `.env.local` hinzu:

```bash
# GitHub Configuration
GITHUB_OWNER="YourUsername"
GITHUB_REPO="FrameTrain"
GITHUB_TOKEN=""  # Optional: nur für private repos
```

**GitHub Token erstellen (für private Repos):**

1. Gehe zu: `https://github.com/settings/tokens`
2. "Generate new token" → "Generate new token (classic)"
3. Scopes auswählen: `repo` (full control)
4. Token kopieren und in `.env.local` einfügen

### Download-Seite testen

```bash
cd website
npm run dev
```

Öffne: `http://localhost:5001/download`

**Test-Ablauf:**
1. ✅ Plattform-Erkennung funktioniert
2. ✅ API-Key eingeben
3. ✅ Download startet
4. ✅ Redirect zu GitHub Release URL

---

## ⌨️ CLI Distribution

### CLI auf PyPI veröffentlichen

```bash
cd cli

# Build erstellen
python -m pip install --upgrade build twine
python -m build

# Upload zu PyPI (Test)
python -m twine upload --repository testpypi dist/*

# Upload zu PyPI (Production)
python -m twine upload dist/*
```

### CLI lokal testen

```bash
cd cli
pip install -e .

# Teste Commands
frametrain --help
frametrain install --key test123
frametrain start
```

---

## 🔒 Zugriffskontrolle

### Wie Paid Access funktioniert

```
User Flow:
1. User kauft auf Website → erhält API Key
2. API Key wird in Datenbank gespeichert (hasPaid=true)

Download Flow:
3. User ruft /api/download-app?platform=windows&key=ABC auf
4. Backend prüft API Key in Datenbank
5. Wenn gültig → GitHub Release URL zurückgeben
6. Wenn ungültig → 403 Forbidden
```

### API Key verifizieren

Die Download-API prüft automatisch:
- ✅ API Key existiert in DB
- ✅ API Key ist aktiv (`isActive=true`)
- ✅ User hat bezahlt (`user.hasPaid=true`)

### Downloads tracken

Optional: Implementiere Download-Tracking:

```typescript
// In /api/download-app/route.ts
async function logDownload(platform: string, apiKey: string) {
  await prisma.downloadLog.create({
    data: {
      platform,
      apiKeyId: apiKey,
      timestamp: new Date(),
    }
  });
}
```

---

## 🚀 Deployment Checklist

### Vor dem ersten Release

- [ ] GitHub Actions Workflow getestet
- [ ] Environment Variables in `.env.local` gesetzt
- [ ] Database Migrations ausgeführt
- [ ] Stripe Payment konfiguriert
- [ ] API Key System funktioniert

### Release erstellen

- [ ] Code committed und gepusht
- [ ] Version in `package.json` und `Cargo.toml` aktualisiert
- [ ] Git Tag erstellt: `git tag v1.0.0`
- [ ] Tag gepusht: `git push origin v1.0.0`
- [ ] GitHub Actions Build erfolgreich
- [ ] Release auf GitHub veröffentlicht

### Nach dem Release

- [ ] Download-Seite getestet
- [ ] CLI Installation getestet
- [ ] Windows Installation verifiziert
- [ ] macOS Installation verifiziert
- [ ] Linux Installation verifiziert
- [ ] Release Notes geschrieben
- [ ] Dokumentation aktualisiert

---

## 🔄 Update Process

### Neue Version veröffentlichen

```bash
# 1. Version erhöhen
# In desktop-app/package.json und src-tauri/Cargo.toml

# 2. Änderungen committen
git add .
git commit -m "Bump version to v1.1.0"

# 3. Tag erstellen und pushen
git tag v1.1.0
git push origin v1.1.0

# 4. GitHub Actions baut automatisch
```

### CLI Update Command

User können updaten mit:

```bash
frametrain update
```

CLI macht dann:
1. ✅ Prüft ob neue Version verfügbar
2. ✅ Lädt neue Version herunter
3. ✅ Installiert automatisch
4. ✅ Startet neu

---

## 🐛 Troubleshooting

### GitHub Actions Build schlägt fehl

**Rust nicht installiert:**
- GitHub Actions installiert Rust automatisch
- Prüfe ob `dtolnay/rust-toolchain@stable` im Workflow ist

**Frontend Build Error:**
```bash
# Lokal testen:
cd desktop-app
npm ci
npm run build
npm run tauri:build
```

**Platform-spezifische Fehler:**
- Windows: Prüfe ob Windows SDK installiert ist
- macOS: Prüfe Xcode Command Line Tools
- Linux: Prüfe System Dependencies

### Download-API Fehler

**404 - Release nicht gefunden:**
- Prüfe ob Tag in GitHub gepusht wurde
- Prüfe `GITHUB_OWNER` und `GITHUB_REPO` in `.env.local`

**403 - Forbidden:**
- Bei private Repos: `GITHUB_TOKEN` setzen
- Bei public Repos: Token nicht nötig

**API Key invalid:**
- Prüfe ob Key in Datenbank existiert
- Prüfe `isActive=true` und `user.hasPaid=true`

### CLI Installation Fehler

**Cannot find download:**
```bash
# Prüfe API_URL in CLI config
frametrain config show

# Setze URL manuell
frametrain config set-url --url https://your-website.com
```

**Permission denied (Linux/macOS):**
```bash
# AppImage ausführbar machen
chmod +x ~/.local/share/frametrain/FrameTrain.AppImage
```

---

## 📊 Analytics & Monitoring

### Download Statistiken

Implementiere Download-Tracking in der API:

```typescript
// Schema hinzufügen in prisma/schema.prisma
model DownloadLog {
  id        String   @id @default(cuid())
  platform  String
  version   String
  apiKeyId  String
  apiKey    ApiKey   @relation(fields: [apiKeyId], references: [id])
  createdAt DateTime @default(now())
}
```

### Dashboard für Downloads

Erstelle Admin-Seite für Statistiken:
- Total Downloads
- Downloads pro Platform
- Downloads pro Version
- Downloads pro User/API Key

---

## 🎯 Best Practices

### Versioning

Nutze Semantic Versioning: `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking Changes
- **MINOR**: Neue Features
- **PATCH**: Bug Fixes

Beispiel: `v1.2.3`

### Release Notes

Immer Release Notes schreiben:

```markdown
## What's New in v1.2.0

### ✨ New Features
- Added model export to ONNX format
- Improved training speed by 30%

### 🐛 Bug Fixes
- Fixed crash on large datasets
- Resolved memory leak in training loop

### 📚 Documentation
- Updated getting started guide
- Added video tutorials
```

### Testing vor Release

1. ✅ Teste alle Features lokal
2. ✅ Teste Build auf allen Plattformen
3. ✅ Teste Installation & Update Process
4. ✅ Teste mit echtem API Key
5. ✅ Beta Test mit kleiner User-Gruppe

---

## 📞 Support

Bei Fragen oder Problemen:
- 📖 Docs: https://docs.frametrain.ai
- 📧 Email: support@frametrain.ai
- 💬 Discord: https://discord.gg/frametrain
