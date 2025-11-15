# FrameTrain CLI

Command-line interface für die FrameTrain Desktop-App.

## Installation

```bash
pip install -e .
```

## Verwendung

### App installieren
```bash
frametrain install
```
Lädt die Desktop-App herunter und installiert sie nach erfolgreicher Key-Verifizierung.

### App starten
```bash
frametrain start
```
Startet die FrameTrain Desktop-App.

### Key verifizieren
```bash
frametrain verify-key
```
Überprüft, ob dein API-Key gültig ist.

### App aktualisieren
```bash
frametrain update
```
Aktualisiert die Desktop-App auf die neueste Version.

### Konfiguration anzeigen
```bash
frametrain config
```
Zeigt die aktuelle Konfiguration an.

## Konfiguration

Die CLI speichert Konfigurationsdaten in:
- **Windows**: `%APPDATA%\FrameTrain\config.json`
- **macOS**: `~/Library/Application Support/FrameTrain/config.json`
- **Linux**: `~/.config/frametrain/config.json`

## API-Key

Deinen API-Key erhältst du nach der Registrierung und Bezahlung auf [frametrain.ai](https://frametrain.ai).

## Support

Bei Problemen oder Fragen:
- Website: https://frametrain.ai
- Dokumentation: https://docs.frametrain.ai
- Support: support@frametrain.ai

## Lizenz

Proprietär - Alle Rechte vorbehalten
