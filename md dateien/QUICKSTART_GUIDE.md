# FrameTrain - Rust Installation Problem gelöst! ✅

## Problem
Nach Installation von Rust war `rustc` nicht im PATH verfügbar.

## Lösung

### Option 1: Quickstart verwenden (EMPFOHLEN)

```bash
chmod +x quickstart.sh
./quickstart.sh
```

Der `quickstart.sh` Script macht ALLES automatisch:
- ✅ Installiert Rust
- ✅ Lädt Rust automatisch
- ✅ Führt System-Check aus
- ✅ Installiert Dependencies
- ✅ Startet Services

### Option 2: Manuell - Rust laden

**Nach der Installation von Rust:**

```bash
# Lade Rust in aktuelle Shell
source ~/.cargo/env

# Dann Setup ausführen
./setup.sh
```

### Option 3: Neues Terminal öffnen

Öffne einfach ein neues Terminal-Fenster. Rust ist dann automatisch verfügbar.

## Aktualisierte Scripts

Alle Scripts laden jetzt automatisch Rust aus `~/.cargo/env`:
- ✅ `setup.sh` - Lädt Rust automatisch
- ✅ `start.sh` - Lädt Rust automatisch
- ✅ `test.sh` - Lädt Rust automatisch
- ✅ `quickstart.sh` - Neu! Macht alles auf einmal

## Sofort starten

```bash
# 1. Quickstart ausführen (macht alles)
chmod +x quickstart.sh
./quickstart.sh

# ODER manuell:

# 1. Rust laden
source ~/.cargo/env

# 2. Setup ausführen
chmod +x setup.sh
./setup.sh

# 3. Services starten
./start.sh

# 4. Status prüfen
./status.sh
```

## Was der Quickstart macht

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FrameTrain Quick Start
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Schritt 1: Rust Installation ✓
Schritt 2: System Check ✓
Schritt 3: Setup ✓
Schritt 4: Environment konfigurieren ✓
Schritt 5: Services starten ✓

🎉 Quick Start abgeschlossen!
```

## Nächste Schritte

Nach erfolgreichem Setup:

1. **Status prüfen:**
```bash
./status.sh
```

2. **Website öffnen:**
```
http://localhost:5001
```

3. **Logs verfolgen:**
```bash
tail -f .pids/*.log
```

4. **Services stoppen:**
```bash
./stop.sh
```

## Troubleshooting

### "command not found: rustc"

**Lösung 1: Rust laden**
```bash
source ~/.cargo/env
```

**Lösung 2: Neues Terminal**
Öffne ein neues Terminal-Fenster.

**Lösung 3: Quickstart verwenden**
```bash
./quickstart.sh
```

### Setup fragt immer noch nach Rust

```bash
# Prüfe ob Rust verfügbar ist
source ~/.cargo/env
rustc --version

# Sollte ausgeben:
# rustc 1.91.1 (ed61e7d7e 2025-11-07)
```

### Services starten nicht

```bash
# 1. Status prüfen
./status.sh

# 2. Logs anschauen
tail -f .pids/*.log

# 3. Services stoppen & neu starten
./stop.sh
./restart.sh
```

## Alle verfügbaren Scripts

| Script | Funktion | Lädt Rust? |
|--------|----------|-----------|
| `quickstart.sh` | Alles auf einmal | ✅ |
| `setup.sh` | Installation | ✅ |
| `start.sh` | Services starten | ✅ |
| `stop.sh` | Services stoppen | ❌ |
| `restart.sh` | Neu starten | ✅ |
| `status.sh` | Status anzeigen | ❌ |
| `test.sh` | System prüfen | ✅ |
| `install-rust.sh` | Rust installieren | ✅ |

## Ready to go! 🚀

Du kannst jetzt mit der Entwicklung starten!
