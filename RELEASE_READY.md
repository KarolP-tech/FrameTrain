# 🚀 FrameTrain - GitHub Release Setup

## ✅ BEREIT FÜR RELEASE

Dein FrameTrain-Projekt ist jetzt vorbereitet für den ersten GitHub Release!

## 📦 Was wurde erstellt?

### Scripts
- ✅ `make-executable.sh` - Macht alle Scripts ausführbar
- ✅ `pre-release-check.sh` - Prüft ob alles bereit ist
- ✅ `create-release.sh` - Automatisiert den Release-Prozess
- ✅ `desktop-app/src-tauri/icons/generate-placeholder.sh` - Bash Icon Generator
- ✅ `desktop-app/src-tauri/icons/generate-icons.py` - Python Icon Generator

### Dokumentation
- ✅ `desktop-app/src-tauri/icons/README.md` - Icon-Anleitung
- ✅ Dieser Guide

### GitHub Actions (bereits vorhanden)
- ✅ `.github/workflows/release.yml` - Build & Release Workflow
- ✅ `.github/workflows/build-desktop.yml` - Desktop Build Workflow

## 🎯 3-SCHRITT QUICK START

```bash
# 1. Scripts ausführbar machen
cd /Users/karol/Desktop/Laufende_Projekte/FrameTrain
chmod +x make-executable.sh
./make-executable.sh

# 2. Icons generieren
cd desktop-app/src-tauri/icons
./generate-placeholder.sh
cd ../../..

# 3. Release erstellen
./create-release.sh
```

**Das war's!** 🎉

## 📊 Was passiert beim Release?

1. **Git Tag wird erstellt:** `v1.0.0`
2. **GitHub Actions startet automatisch**
3. **Builds für 3 Platforms:**
   - 🪟 Windows (MSI + EXE)
   - 🍎 macOS (DMG + APP)
   - 🐧 Linux (AppImage + DEB)
4. **GitHub Release wird erstellt**
5. **Installer werden hochgeladen**

**Dauer:** 15-30 Minuten

## 📍 Wichtige Links

Nach dem Release:

- **Actions:** https://github.com/KarolP-tech/FrameTrain/actions
- **Releases:** https://github.com/KarolP-tech/FrameTrain/releases
- **Latest:** https://github.com/KarolP-tech/FrameTrain/releases/latest

## 🔧 Nächste Schritte

### Während Build läuft (15-30 Min):

1. ☕ Kaffee holen
2. 📊 GitHub Actions beobachten
3. 📖 Dokumentation verbessern
4. 🎨 Screenshots vorbereiten

### Nach erfolgreichem Build:

1. **Download testen:**
   ```bash
   curl -L -o FrameTrain.dmg \
     https://github.com/KarolP-tech/FrameTrain/releases/download/v1.0.0/frametrain-macos-universal.dmg
   ```

2. **Website aktualisieren:**
   - Vercel Environment Variable: `APP_DOWNLOAD_BASE_URL`
   - Dashboard: Download-Links anpassen

3. **README updaten:**
   - Download-Links hinzufügen
   - Badge hinzufügen
   - Screenshots einbinden

4. **Marketing:**
   - Social Media Post
   - ProductHunt (optional)
   - HackerNews (optional)

## 🚨 Häufige Probleme

### "ImageMagick not found"
```bash
brew install imagemagick
```

### "Python version too old"
```bash
brew install python3
```

### "Rust not found"
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### "Tag already exists"
```bash
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0
```

### "Build failed on GitHub"
- Prüfe GitHub Actions Logs
- Teste lokal: `npm run tauri:build`
- Öffne ein Issue falls nötig

## 📞 Support

Falls etwas nicht funktioniert:

1. **Prüfe Logs:**
   - GitHub Actions
   - Terminal Output
   - Browser Console

2. **Teste lokal:**
   ```bash
   cd desktop-app
   npm run build
   npm run tauri:build
   ```

3. **Frag nach Hilfe:**
   - GitHub Issues
   - Oder direkt bei mir 😊

## 🎉 SUCCESS CRITERIA

✅ GitHub Actions Build erfolgreich
✅ Release auf GitHub sichtbar
✅ Alle 5 Installer verfügbar
✅ Download funktioniert
✅ App startet ohne Fehler
✅ API Key Verifikation funktioniert

**Dann: RELEASE IST LIVE! 🚀**

---

## 📖 Weitere Dokumentation

- [Pre-Release Checklist](./pre-release-check.sh)
- [Icon Generator Guide](./desktop-app/src-tauri/icons/README.md)
- [GitHub Actions Workflows](./.github/workflows/)
- [Complete Plan](./RELEASE_PLAN.md) - Siehe Anhang für Details

---

**Bereit? Los geht's!** 🚀

```bash
./create-release.sh
```
