# FrameTrain Desktop App - Authentifizierungs-Update

## 🔒 Neu implementierte Features

### 1. Sichere Authentifizierung
- **API-Key + Password**: Beide Credentials werden gegen Supabase-Datenbank validiert
- **User-Verifizierung**: Nur wenn Key und Password zur gleichen User-ID gehören, ist Login erfolgreich
- **Zahlungsstatus-Check**: Nur Benutzer mit aktivem Abonnement (has_paid = true) können sich anmelden
- **Password-Hashing**: bcrypt-Verifizierung wie auf der Website

### 2. Session-Management
- **Kein Auto-Login mehr**: Nach Logout oder App-Schließen ist der User abgemeldet
- **Config wird gelöscht**: `clear_config` Command entfernt gespeicherte Credentials beim Logout
- **Re-Validierung**: Bei App-Start werden gespeicherte Credentials gegen DB validiert

### 3. Settings-Bereich
- **Neue View**: Zahnrad-Icon in Sidebar führt zu Einstellungen
- **4 Tabs**:
  - **Konto**: User-Info, API-Key Management, Kontoverwaltung
  - **Darstellung**: Theme-Einstellungen (Dark Mode aktiv)
  - **Benachrichtigungen**: Desktop-Benachrichtigungs-Einstellungen
  - **Über**: App-Info, Version, Links

### 4. Verbesserte UI
- User-Email wird in Sidebar angezeigt
- Settings-Button mit eigenem Icon
- Kopieren-Funktion für User-ID und API-Key
- Responsive Notifications

## 📦 Neue Dependencies (Rust)

```toml
# HTTP Client für Supabase
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1", features = ["full"] }
tokio-postgres = { version = "0.7", features = ["with-chrono-0_4"] }

# Password Hashing
bcrypt = "0.15"
```

## 🏗️ Architektur-Änderungen

### Backend (Rust)
**Neue Dateien:**
- `src-tauri/src/auth.rs` - Authentifizierungs-Logik mit Supabase

**Erweiterte Dateien:**
- `main.rs`:
  - Neues Modul `auth` importiert
  - `save_config` nimmt jetzt komplettes Config-JSON
  - `clear_config` Command hinzugefügt
  - `validate_credentials` Command registriert

### Frontend (React)
**Neue Komponenten:**
- `Settings.tsx` - Einstellungen-View mit 4 Tabs

**Aktualisierte Komponenten:**
- `Login.tsx`:
  - Password-Feld hinzugefügt
  - Verbesserte Validierung
  - Besseres Error-Handling
  
- `App.tsx`:
  - `userData` State mit userId und email
  - Validierung gegen DB bei App-Start
  - Config-Cleanup bei ungültigen Credentials
  
- `Dashboard.tsx`:
  - `userData` wird weitergegeben
  - Settings-View hinzugefügt
  
- `Sidebar.tsx`:
  - User-Email-Anzeige
  - Settings-Button
  - Verbessertes Layout

## 🔐 Sicherheitsverbesserungen

1. **Doppelte Verifizierung**: API-Key UND Password müssen stimmen
2. **User-ID-Matching**: Beide müssen zum gleichen User gehören
3. **Payment-Check**: Nur zahlende Kunden können sich anmelden
4. **Session-Cleanup**: Credentials werden bei Logout vollständig gelöscht
5. **Re-Validierung**: Bei jedem App-Start wird gegen DB geprüft
6. **Password wird niemals angezeigt**: Nur im Login-Feld sichtbar

## 🚀 Installation & Build

```bash
# Dependencies installieren
cd Desktop-app
npm install

# Rust Dependencies (automatisch beim Build)
cd src-tauri
cargo build

# Development
npm run tauri:dev

# Production Build
npm run tauri:build
```

## 🔄 Datenbank-Schema

Die App greift auf folgende Supabase-Tabellen zu:

### `users`
- `id` (String, Primary Key)
- `email` (String, Unique)
- `password_hash` (String, bcrypt)
- `has_paid` (Boolean)
- Timestamps

### `api_keys`
- `id` (String, Primary Key)
- `user_id` (String, Foreign Key)
- `key` (String, Unique, starts with "ft_")
- `is_active` (Boolean)
- `last_used_at` (DateTime, wird bei Login aktualisiert)
- Timestamps

## 📝 Login-Flow

```
1. User gibt API-Key + Password ein
   ↓
2. Frontend validiert Format (ft_*, min. 24 Zeichen)
   ↓
3. Rust Backend: validate_credentials()
   ↓
4. Verbindung zu Supabase Postgres
   ↓
5. API-Key aus Datenbank laden
   ↓
6. Prüfen: is_active = true?
   ↓
7. User-Daten laden (via user_id)
   ↓
8. bcrypt Password-Verifizierung
   ↓
9. Prüfen: has_paid = true?
   ↓
10. last_used_at aktualisieren
    ↓
11. Success → UserData zurückgeben
    ↓
12. Config speichern (API-Key + Password)
    ↓
13. Dashboard anzeigen
```

## 🐛 Fehlerbehebung

### Problem: "Datenbankverbindung fehlgeschlagen"
**Lösung**: Prüfe Supabase-URL und Netzwerkverbindung

### Problem: "API-Key nicht gefunden"
**Lösung**: Key existiert nicht in DB oder Tippfehler

### Problem: "Falsches Passwort"
**Lösung**: Password stimmt nicht mit DB-Hash überein

### Problem: "Account ist nicht aktiv"
**Lösung**: has_paid = false → User muss Abo abschließen

## 📚 Weitere Infos

- Supabase Connection String ist URL-encoded (Sonderzeichen im Passwort)
- Async/Await mit Tokio für Datenbank-Calls
- Fehlerbehandlung mit Result<T, String>
- Frontend verwendet TypeScript für Type-Safety

## ✅ Getestet

- [x] Login mit gültigem Key + Password
- [x] Login-Fehler bei falschem Password
- [x] Login-Fehler bei nicht existierendem Key
- [x] Login-Fehler bei inaktivem Account (has_paid = false)
- [x] Logout löscht Config
- [x] App-Neustart ohne Auto-Login
- [x] Settings-View öffnen
- [x] API-Key kopieren
- [x] User-ID kopieren

## 🎯 Nächste Schritte

- [ ] Environment Variables für Supabase-URL
- [ ] Token-basierte Auth statt Password-Speicherung
- [ ] 2FA-Support
- [ ] Session-Timeout
- [ ] Offline-Modus mit gespeicherten Credentials (optional)
