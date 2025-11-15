# FrameTrain - Management Scripts

Scripts zum einfachen Verwalten der FrameTrain Services.

## 🚀 Verfügbare Scripts

### `start.sh` - Services starten

Startet alle oder einzelne Services im Development-Modus.

```bash
# Alle Services starten
./start.sh

# Nur Website starten
./start.sh website

# Nur Desktop-App starten
./start.sh desktop
```

**Was wird gestartet:**
- Website: Next.js Dev Server auf Port 3000
- Desktop-App: Tauri Development Modus

**Logs:** Alle Logs werden in `.pids/*.log` gespeichert

### `stop.sh` - Services stoppen

Stoppt alle laufenden Services sicher.

```bash
./stop.sh
```

**Funktionsweise:**
1. Versucht graceful shutdown (SIGTERM)
2. Wartet bis zu 10 Sekunden
3. Force kill wenn nötig (SIGKILL)
4. Räumt PID-Dateien auf

### `restart.sh` - Services neu starten

Stoppt und startet Services neu.

```bash
# Alle neu starten
./restart.sh

# Nur Website neu starten
./restart.sh website
```

### `status.sh` - Status anzeigen

Zeigt Status aller Services an.

```bash
./status.sh
```

**Ausgabe:**
- Service-Status (läuft/gestoppt)
- Process ID (PID)
- Uptime
- RAM-Verbrauch
- Letzte Log-Zeile
- URLs

**Beispiel:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FrameTrain Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ website: Läuft (PID: 12345, Uptime: 00:15:30, RAM: 256MB)
    └─ Ready in 1.2s

✓ desktop-app: Läuft (PID: 12346, Uptime: 00:15:25, RAM: 512MB)
    └─ Compiled successfully

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Services: 2/2 laufen

🌐 Website: http://localhost:3000
🖥️  Desktop-App: Dev-Modus aktiv
```

## 📝 Logs verfolgen

### Alle Logs gleichzeitig
```bash
tail -f .pids/*.log
```

### Einzelne Logs
```bash
# Website
tail -f .pids/website.log

# Desktop-App
tail -f .pids/desktop-app.log
```

## 🔧 Troubleshooting

### Services starten nicht

**Problem:** Port bereits belegt
```bash
# Prüfe welcher Prozess Port 3000 nutzt
lsof -i :3000

# Beende Prozess
kill -9 <PID>
```

**Problem:** PID-Dateien inkonsistent
```bash
# Lösche alte PID-Dateien
rm -rf .pids/

# Starte neu
./start.sh
```

### Services stoppen nicht

```bash
# Force stop aller Node/Rust Prozesse (VORSICHT!)
pkill -9 node
pkill -9 frametrain

# Räume auf
rm -rf .pids/
```

### Logs sind zu groß

```bash
# Lösche alte Logs
rm -f .pids/*.log

# Oder rotiere Logs
for log in .pids/*.log; do
    if [ -f "$log" ]; then
        mv "$log" "$log.old"
    fi
done
```

## 📊 Process Management

### PID-Dateien

PID-Dateien werden in `.pids/` gespeichert:
```
.pids/
├── website.pid
├── website.log
├── desktop-app.pid
└── desktop-app.log
```

### Manuelles Management

```bash
# Lese PID
cat .pids/website.pid

# Prüfe ob Prozess läuft
ps -p $(cat .pids/website.pid)

# Stoppe Prozess
kill $(cat .pids/website.pid)

# Force kill
kill -9 $(cat .pids/website.pid)
```

## 🎯 Typische Workflows

### Development starten
```bash
./start.sh
./status.sh
# Entwickle...
./stop.sh
```

### Nur Website testen
```bash
./start.sh website
# Browser öffnen: http://localhost:3000
./stop.sh
```

### Nach Code-Änderungen
```bash
./restart.sh
# oder einzeln:
./restart.sh website
```

### Logs während Entwicklung
```bash
# Terminal 1
./start.sh

# Terminal 2
tail -f .pids/website.log

# Terminal 3
tail -f .pids/desktop-app.log
```

## 🔐 Sicherheit

**Wichtig:**
- PID-Dateien sind in `.gitignore`
- Logs können sensible Daten enthalten
- Teile `.pids/` niemals öffentlich

## 📦 Integration mit anderen Tools

### VS Code Tasks

`.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Start All Services",
      "type": "shell",
      "command": "./start.sh",
      "problemMatcher": []
    },
    {
      "label": "Stop All Services",
      "type": "shell",
      "command": "./stop.sh",
      "problemMatcher": []
    },
    {
      "label": "Show Status",
      "type": "shell",
      "command": "./status.sh",
      "problemMatcher": []
    }
  ]
}
```

### npm Scripts

`package.json` im Root:
```json
{
  "scripts": {
    "start": "./start.sh",
    "stop": "./stop.sh",
    "restart": "./restart.sh",
    "status": "./status.sh"
  }
}
```

Dann verwendbar mit:
```bash
npm run start
npm run status
npm run stop
```

## 🐛 Debug-Modus

Aktiviere Debug-Ausgaben:
```bash
# Setze Debug-Flag
export DEBUG=1

# Starte Services
./start.sh

# Services geben nun mehr Logs aus
```

## 📈 Performance Monitoring

### Resource Usage anzeigen
```bash
# Während Services laufen
watch -n 1 './status.sh'

# Detaillierte Info
ps aux | grep -E "(node|frametrain)"
```

### Memory Leaks erkennen
```bash
# Überwache RAM über Zeit
while true; do
    ./status.sh | grep RAM
    sleep 60
done
```

## 🚨 Production

**Hinweis:** Diese Scripts sind für Development.

Für Production verwende:
- PM2 (Node.js)
- systemd (Linux Services)
- Docker Compose
- Kubernetes

Siehe `docs/DEPLOYMENT.md` für Details.

## 📚 Weitere Ressourcen

- [Development Guide](./DEVELOPMENT.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Troubleshooting](./TROUBLESHOOTING.md)

## 💡 Tips

1. **Immer Status prüfen** vor Start/Stop
2. **Logs regelmäßig leeren** bei langen Dev-Sessions
3. **Port-Konflikte vermeiden** durch Status-Check
4. **Graceful Shutdown** nutzen (nicht force-kill)
5. **Backups machen** vor großen Änderungen
