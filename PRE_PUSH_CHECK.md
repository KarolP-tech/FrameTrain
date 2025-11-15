# 🔍 Pre-Push Security Check für FrameTrain

## ✅ SICHERHEITS-CHECKLISTE

### 1. .gitignore Status
- [x] `.env.local` wird ignoriert
- [x] `.next/` wird ignoriert  
- [x] `node_modules/` wird ignoriert
- [x] Alle Secrets-Pattern sind abgedeckt

### 2. Dateien die NICHT committed werden sollten
Diese Dateien existieren lokal, werden aber von Git ignoriert:
- `website/.env.local` (enthält echte Stripe Keys)
- `website/.next/` (Build-Output mit hardcoded Keys)
- `website/node_modules/` (Dependencies)

### 3. Username Updates notwendig
Folgende Dateien müssen noch `YourUsername` → `KarolP-tech` ersetzen:

- [ ] `README.md` (4 Stellen)
- [ ] `website/.env.local.example` (1 Stelle)
- [ ] GitHub Actions Workflows (falls vorhanden)

## 🚀 PUSH-VORBEREITUNG

### Schritt 1: Username aktualisieren
```bash
# In README.md
sed -i '' 's/YourUsername/KarolP-tech/g' README.md

# In .env.local.example
sed -i '' 's/YourUsername/KarolP-tech/g' website/.env.local.example
```

### Schritt 2: Git Status prüfen
```bash
git status

# Sollte NICHT zeigen:
# - website/.env.local
# - website/.next/
# - website/node_modules/
```

### Schritt 3: Sicherheitscheck
```bash
# Prüfe dass keine Secrets committed werden
git diff --cached | grep -E "sk_test|pk_test|whsec_"

# Sollte LEER sein oder nur Beispiel-Keys zeigen (.example Dateien)
```

### Schritt 4: Commit & Push
```bash
# Alle Dateien hinzufügen
git add .

# Commit erstellen
git commit -m "Initial commit: FrameTrain v1.0.0 with BSL 1.1 license"

# Push zu GitHub
git push -u origin main
```

## 🔐 NACH DEM PUSH

### GitHub Repository Settings

1. **Secrets hinzufügen** (für GitHub Actions):
   - Settings → Secrets and variables → Actions
   - `STRIPE_SECRET_KEY` hinzufügen
   - `DATABASE_URL` hinzufügen
   - `JWT_SECRET` hinzufügen

2. **Branch Protection** (optional):
   - Settings → Branches → Add rule
   - Branch name pattern: `main`
   - ✅ Require pull request reviews

3. **About Section aktualisieren**:
   - Description: "Professional platform for local ML model training"
   - Website: https://frametrain.ai (falls du eine hast)
   - Topics: `machine-learning`, `pytorch`, `tauri`, `nextjs`, `stripe`

## ⚠️ WICHTIGE HINWEISE

### Warum .env.local lokal behalten?
Du hattest Recht zu fragen! Die Datei wird NICHT gelöscht, sondern:
- ✅ Bleibt lokal auf deinem Mac
- ✅ Wird von Git ignoriert (via .gitignore)
- ✅ Wird NICHT zu GitHub gepusht
- ✅ Jeder Entwickler muss seine eigene erstellen

### Was wenn ich später neue Secrets brauche?
1. In `.env.local.example` dokumentieren (ohne echte Werte!)
2. In `.env.local` hinzufügen (wird nicht committed)
3. Andere Entwickler updaten ihre lokale `.env.local`

### Build-Dateien (.next/)
- Werden bei jedem `npm run build` neu erstellt
- Enthalten temporär hardcoded Werte aus .env
- Werden von Git ignoriert
- Werden auf Server/Vercel neu gebaut

## ✨ ALLES KLAR?

Deine `.gitignore` ist **perfekt konfiguriert**!
Die sensiblen Dateien werden **automatisch ignoriert**.
Du musst sie **nicht löschen** - Git wird sie einfach nicht pushen.

Bereit für den Push? Führe die Schritte oben aus! 🚀
