# GitHub Repository Setup Guide

## 🚀 Schritt-für-Schritt Anleitung

### 1️⃣ Repository erstellen

1. Gehe zu [GitHub](https://github.com) und logge dich ein
2. Klicke auf **"New Repository"** (grüner Button oben rechts)
3. Fülle die Felder aus:

```
Repository name: FrameTrain
Description: Professional platform for local ML training
Visibility: ✅ PUBLIC (empfohlen für kostenlose GitHub Actions)
Initialize: ❌ NICHT initialisieren (wir haben schon Code)
```

4. Klicke **"Create Repository"**

---

### 2️⃣ Lokales Repository verbinden

```bash
cd /Users/karol/Desktop/Laufende_Projekte/FrameTrain

# Git initialisieren (falls noch nicht geschehen)
git init

# Remote Repository hinzufügen (ersetze YourUsername!)
git remote add origin https://github.com/YourUsername/FrameTrain.git

# Alle Dateien adden
git add .

# Ersten Commit erstellen
git commit -m "Initial commit: FrameTrain v1.0.0"

# Branch umbenennen zu main (wenn noch master)
git branch -M main

# Zum GitHub pushen
git push -u origin main
```

---

### 3️⃣ Environment Variables in GitHub setzen

Für **Private Repository** (falls du dich dafür entscheidest):

1. Gehe zu: `Settings` → `Secrets and variables` → `Actions`
2. Klicke **"New repository secret"**
3. Füge hinzu:

```
Name: GITHUB_TOKEN
Value: [Dein GitHub Personal Access Token]
```

**Token erstellen:**
1. `Settings` → `Developer settings` → `Personal access tokens` → `Tokens (classic)`
2. **"Generate new token (classic)"**
3. Scopes: ✅ `repo` (full control)
4. Token kopieren und als Secret speichern

---

### 4️⃣ Branch Protection Rules (Optional)

Schütze deinen `main` Branch:

1. `Settings` → `Branches`
2. **"Add branch protection rule"**
3. Branch name pattern: `main`
4. Aktiviere:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass
   - ✅ Require branches to be up to date

---

### 5️⃣ GitHub Actions aktivieren

1. Gehe zu `Actions` Tab
2. Falls disabled: Klicke **"Enable GitHub Actions"**
3. Workflow sollte automatisch erkannt werden

Teste den Workflow:

```bash
# Erstelle einen Test-Tag
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions startet automatisch den Build! 🎉

Prüfe: `Actions` Tab → `Build Desktop App`

---

### 6️⃣ README Badges aktualisieren

Ersetze in `README.md`:

```markdown
[![Build](https://github.com/YourUsername/FrameTrain/actions/workflows/build-desktop.yml/badge.svg)](https://github.com/YourUsername/FrameTrain/actions)
[![Downloads](https://img.shields.io/github/downloads/YourUsername/FrameTrain/total)](https://github.com/YourUsername/FrameTrain/releases)
```

Mit deinem echten Username!

---

### 7️⃣ GitHub Pages (Optional)

Für Dokumentation:

1. `Settings` → `Pages`
2. Source: **Deploy from a branch**
3. Branch: `main` → `/docs`
4. **Save**

Deine Docs sind dann unter: `https://YourUsername.github.io/FrameTrain/`

---

### 8️⃣ Website .env.local aktualisieren

```bash
cd website
nano .env.local
```

Füge hinzu:

```bash
# GitHub Configuration
GITHUB_OWNER="YourUsername"  # ← Dein GitHub Username
GITHUB_REPO="FrameTrain"
# GITHUB_TOKEN nur für private Repos nötig
```

---

### 9️⃣ Ersten Release erstellen

```bash
# Stelle sicher, alles ist committed
git status

# Tag erstellen
git tag -a v1.0.0 -m "Release v1.0.0: Initial public release"

# Tag pushen
git push origin v1.0.0
```

**Was passiert dann:**
1. 🤖 GitHub Actions startet
2. 🏗️ Baut für Windows, macOS, Linux
3. 📦 Erstellt Release auf GitHub
4. ✅ Installables sind downloadbar

Prüfe: `https://github.com/YourUsername/FrameTrain/releases`

---

### 🔟 Download-Seite testen

```bash
cd website
npm run dev
```

Öffne: `http://localhost:5001/download`

**Test-Workflow:**
1. Plattform wählen (Windows/Mac/Linux)
2. API-Key eingeben (nutze Test-Key aus DB)
3. Download klicken
4. Sollte zu GitHub Release URL redirecten

---

## ✅ Checkliste

Nach Setup solltest du haben:

- [ ] GitHub Repository erstellt (Public empfohlen)
- [ ] Code gepusht zu GitHub
- [ ] LICENSE Datei committed (BSL 1.1)
- [ ] SECURITY.md committed
- [ ] README.md mit Badges
- [ ] GitHub Actions aktiviert
- [ ] Branch Protection Rules (optional)
- [ ] Erster Release Tag erstellt (`v1.0.0`)
- [ ] GitHub Actions Build erfolgreich
- [ ] Release auf GitHub sichtbar
- [ ] Download-Seite funktioniert
- [ ] `GITHUB_OWNER` und `GITHUB_REPO` in `.env.local`

---

## 🐛 Troubleshooting

### GitHub Actions schlägt fehl

**Problem: "Rust not found"**
```yaml
# Workflow hat bereits:
- uses: dtolnay/rust-toolchain@stable
# Sollte automatisch klappen
```

**Problem: "Permission denied"**
```yaml
# Füge zu Workflow hinzu:
permissions:
  contents: write
```

**Problem: "Release creation failed"**
- Prüfe ob Tag existiert: `git tag`
- Prüfe ob Tag gepusht: `git ls-remote --tags origin`
- Tag Format muss sein: `v1.0.0` (mit v!)

### Download-API 404

**Problem: "Release not found"**
```bash
# Prüfe .env.local:
GITHUB_OWNER="DeinUsername"  # Korrekt?
GITHUB_REPO="FrameTrain"     # Korrekt?

# Prüfe ob Release existiert:
curl https://api.github.com/repos/DeinUsername/FrameTrain/releases/latest
```

**Problem: "API rate limit"**
- Bei public repo: Kein Token nötig
- Bei private repo: `GITHUB_TOKEN` in `.env.local` setzen

### CLI Installation schlägt fehl

**Problem: "Cannot download app"**
```bash
# Prüfe CLI config:
frametrain config show

# URL manuell setzen:
frametrain config set-url --url https://frametrain.ai
```

---

## 🔒 Sicherheits-Tipps

### ❌ NIEMALS committen:

```bash
# Diese Dateien MÜSSEN in .gitignore sein:
.env.local
.env
*.key
*.pem
.github-token
.stripe-*
```

### ✅ Secrets Management:

```bash
# Lokale Entwicklung: .env.local
DATABASE_URL="..."
STRIPE_SECRET_KEY="..."

# GitHub Actions: Repository Secrets
# Settings → Secrets and variables → Actions
```

---

## 📞 Support

Bei Fragen zum GitHub Setup:

- 📖 [GitHub Docs](https://docs.github.com)
- 💬 [GitHub Community](https://github.community)
- 📧 FrameTrain Support: support@frametrain.ai

---

**Ready to go! 🚀**

Nach diesem Setup ist dein FrameTrain Repository production-ready!
