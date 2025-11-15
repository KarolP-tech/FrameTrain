# FrameTrain Desktop App Downloads

## 🎯 Schnellstart

### Option 1: Website Download (Empfohlen)

1. Besuche [frametrain.ai/download](https://frametrain.ai/download)
2. Wähle deine Plattform (Windows/macOS/Linux)
3. Gib deinen API-Key ein
4. Download & Installation

### Option 2: CLI Installation

```bash
# Installiere die CLI
pip install frametrain-cli

# Installiere die Desktop-App
frametrain install --key YOUR_API_KEY

# Starte FrameTrain
frametrain start
```

---

## 💻 Plattform-spezifische Anleitungen

### Windows

**Download:**
- Datei: `FrameTrain_x.x.x_x64.msi`
- Größe: ~80-120 MB

**Installation:**
1. Doppelklick auf `.msi` Datei
2. Folge dem Installations-Wizard
3. FrameTrain wird automatisch gestartet
4. Desktop-Verknüpfung wird erstellt

**Alternative Installation (CLI):**
```bash
pip install frametrain-cli
frametrain install --key YOUR_KEY
```

### macOS

**Download:**
- Datei: `FrameTrain_x.x.x_x64.dmg`
- Größe: ~90-130 MB

**Installation:**
1. Öffne die `.dmg` Datei
2. Ziehe FrameTrain in den Applications Ordner
3. Starte aus Applications oder Spotlight
4. Bei Sicherheitswarnung: Systemeinstellungen → Sicherheit → "Trotzdem öffnen"

**Alternative Installation (CLI):**
```bash
pip3 install frametrain-cli
frametrain install --key YOUR_KEY
```

### Linux

**Download:**
- Datei: `FrameTrain_x.x.x_amd64.AppImage`
- Größe: ~100-140 MB

**Installation:**
```bash
# Download (via Website oder CLI)
chmod +x FrameTrain*.AppImage

# Ausführen
./FrameTrain*.AppImage

# Optional: Desktop-Integration
./FrameTrain*.AppImage --appimage-install
```

**Alternative Installation (CLI):**
```bash
pip3 install frametrain-cli
frametrain install --key YOUR_KEY
```

**System Requirements:**
- Ubuntu 20.04+ / Debian 11+ / Fedora 35+
- FUSE (für AppImage): `sudo apt install fuse libfuse2`

---

## 🔑 API Key erhalten

1. Registriere dich auf [frametrain.ai/register](https://frametrain.ai/register)
2. Gehe zur [Payment-Seite](https://frametrain.ai/payment)
3. Bezahle 2€ (einmalig)
4. Erhalte deinen API-Key per Email & Dashboard
5. Nutze den Key für Download & Installation

---

## 📦 Verfügbare Versionen

Alle Releases findest du auf:
- **GitHub Releases**: [github.com/YourUsername/FrameTrain/releases](https://github.com/YourUsername/FrameTrain/releases)
- **Website**: [frametrain.ai/download](https://frametrain.ai/download)

---

## 🔄 Updates

### Automatische Updates (CLI)

```bash
frametrain update
```

### Manuelle Updates

1. Besuche [Download-Seite](https://frametrain.ai/download)
2. Lade neueste Version herunter
3. Installiere über alte Version (überschreibt automatisch)

### Update-Benachrichtigungen

FrameTrain prüft automatisch auf Updates beim Start.

---

## 🐛 Troubleshooting

### Windows

**"Windows hat Ihren PC geschützt"**
- Klicke "Weitere Informationen"
- Klicke "Trotzdem ausführen"
- Grund: Neue App ohne teures Code-Signing Zertifikat

**Installation schlägt fehl**
- Prüfe Admin-Rechte
- Deaktiviere temporär Antivirus
- Nutze `.exe` Installer statt `.msi`

### macOS

**"App kann nicht geöffnet werden"**
```bash
# Terminal-Lösung:
xattr -cr /Applications/FrameTrain.app
```

Oder: Systemeinstellungen → Sicherheit → "Trotzdem öffnen"

**"Beschädigter Download"**
- Re-Download die App
- Prüfe Speicherplatz (min. 500 MB frei)

### Linux

**AppImage startet nicht**
```bash
# FUSE installieren
sudo apt install fuse libfuse2

# Ausführbar machen
chmod +x FrameTrain*.AppImage
```

**"Permission denied"**
```bash
# Rechte setzen
chmod +x FrameTrain*.AppImage

# Als Root ausführen (nicht empfohlen)
sudo ./FrameTrain*.AppImage --no-sandbox
```

---

## 💡 CLI Commands

### Installation
```bash
frametrain install --key YOUR_KEY
frametrain install --key YOUR_KEY --path /custom/path
```

### Start & Stop
```bash
frametrain start
frametrain start --no-verify  # Skip key verification
```

### Updates
```bash
frametrain update
frametrain update --force  # Force update even if up-to-date
```

### Configuration
```bash
frametrain config show
frametrain config set-key --key NEW_KEY
frametrain config set-url --url https://api.frametrain.ai
```

### Verification
```bash
frametrain verify-key --key YOUR_KEY
frametrain info  # Show installation info
```

### Uninstall
```bash
frametrain uninstall
```

---

## 🖥️ System Requirements

### Minimum

- **CPU**: Intel Core i5 / AMD Ryzen 5 (4 Kerne)
- **RAM**: 8 GB
- **GPU**: Integrierte Grafik
- **Speicher**: 2 GB freier Platz
- **OS**: 
  - Windows 10 (64-bit) oder neuer
  - macOS 11 Big Sur oder neuer
  - Ubuntu 20.04 / Debian 11 oder neuer

### Empfohlen

- **CPU**: Intel Core i7 / AMD Ryzen 7 (8 Kerne)
- **RAM**: 16 GB oder mehr
- **GPU**: NVIDIA GPU mit CUDA Support (RTX 3060+ empfohlen)
- **Speicher**: 10 GB freier Platz (für Modelle & Datasets)

---

## 🎓 Erste Schritte nach Installation

1. **Starte FrameTrain**
   - Windows: Desktop Icon oder Startmenü
   - macOS: Applications Ordner
   - Linux: `./FrameTrain.AppImage` oder App Menu

2. **Importiere ein Modell**
   - Klicke "New Project"
   - Wähle HuggingFace Modell oder lokales Modell
   - Konfiguriere Training-Parameter

3. **Starte Training**
   - Wähle Dataset aus
   - Klicke "Start Training"
   - Beobachte Live-Metriken

4. **Exportiere Modell**
   - Training abgeschlossen
   - Exportiere als PyTorch, ONNX oder TensorFlow

---

## 📚 Weiterführende Ressourcen

- 📖 **Dokumentation**: [docs.frametrain.ai](https://docs.frametrain.ai)
- 🎥 **Video Tutorials**: [youtube.com/@frametrain](https://youtube.com/@frametrain)
- 💬 **Community**: [discord.gg/frametrain](https://discord.gg/frametrain)
- 📧 **Support**: support@frametrain.ai

---

## 🔒 Sicherheit & Datenschutz

✅ **100% Lokal** - Alle Daten bleiben auf deinem Gerät  
✅ **Keine Cloud** - Keine Uploads, keine Tracking  
✅ **DSGVO-konform** - Keine Datenspeicherung auf Servern  
✅ **Open Training** - Volle Kontrolle über deine Modelle  

---

## 🤝 Support & Hilfe

**Probleme beim Download?**
- 📧 Email: support@frametrain.ai
- 💬 Discord: [discord.gg/frametrain](https://discord.gg/frametrain)
- 📝 GitHub Issues: [github.com/YourUsername/FrameTrain/issues](https://github.com/YourUsername/FrameTrain/issues)

**Feedback & Feature Requests:**
- 🐛 Bug Report: [GitHub Issues](https://github.com/YourUsername/FrameTrain/issues/new?template=bug_report.md)
- 💡 Feature Request: [GitHub Discussions](https://github.com/YourUsername/FrameTrain/discussions)

---

## 📄 Lizenz

Proprietär - Alle Rechte vorbehalten  
Nutzung erfordert gültigen API-Key (2€ einmalig)

---

<div align="center">
  <b>Made with ❤️ by FrameTrain Team</b>
</div>
