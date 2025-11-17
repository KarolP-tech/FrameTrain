# 🚀 FrameTrain Desktop App - Schnellreferenz

## ✅ Was wurde implementiert?

### 1️⃣ Sichere Authentifizierung
- ✅ API-Key + Password Login
- ✅ Validierung gegen Supabase PostgreSQL
- ✅ bcrypt Password-Verifizierung
- ✅ User-ID-Matching (Key und Password müssen zum gleichen User gehören)
- ✅ Payment-Check (nur has_paid = true)
- ✅ last_used_at Update bei Login

### 2️⃣ Session-Management
- ✅ Logout löscht Config komplett
- ✅ Kein Auto-Login nach App-Neustart
- ✅ Re-Validierung bei App-Start
- ✅ `clear_config()` Command

### 3️⃣ Settings-Bereich
- ✅ Neuer Tab "Einstellungen" in Sidebar
- ✅ 4 Tabs: Konto, Darstellung, Benachrichtigungen, Über
- ✅ User-Info mit E-Mail und User-ID
- ✅ API-Key Management (anzeigen/kopieren)
- ✅ Links zu Dashboard & Passwort-Änderung

### 4️⃣ UI-Verbesserungen
- ✅ User-Email in Sidebar
- ✅ Zahnrad-Icon für Settings
- ✅ Kopieren-Funktion mit Notifications
- ✅ Verbessertes Login-Design

## 📁 Geänderte/Neue Dateien

### Backend (Rust)
```
src-tauri/
├── Cargo.toml                    # ✏️ Dependencies hinzugefügt
├── src/
│   ├── main.rs                   # ✏️ auth-Modul, Commands erweitert
│   └── auth.rs                   # 🆕 Supabase-Authentifizierung
```

### Frontend (React)
```
src/
└── components/
    ├── Login.tsx                 # ✏️ Password-Feld, bessere UX
    ├── Dashboard.tsx             # ✏️ userData, Settings-View
    ├── Sidebar.tsx               # ✏️ User-Email, Settings-Button
    ├── Settings.tsx              # 🆕 Einstellungen-Komponente
    └── App.tsx                   # ✏️ Authentifizierungs-Logik
```

## 🔑 Neue Tauri Commands

```rust
// Validiert API-Key + Password gegen Datenbank
validate_credentials(api_key: String, password: String) 
  -> Result<ApiKeyValidation, String>

// Löscht gespeicherte Config
clear_config() -> Result<(), String>

// save_config wurde erweitert
save_config(api_key: String, config: String) -> Result<(), String>
```

## 🎯 Testing-Checklist

### ✅ Login
- [ ] Mit gültigem Key + Password → erfolgreicher Login
- [ ] Mit ungültigem Key → Fehlermeldung
- [ ] Mit falschem Password → Fehlermeldung
- [ ] Mit inaktivem Account (has_paid=false) → Fehlermeldung
- [ ] Key zu kurz → Fehlermeldung
- [ ] Key ohne "ft_" Prefix → Fehlermeldung

### ✅ Session
- [ ] Nach Login → Dashboard sichtbar
- [ ] App schließen + neu öffnen → Login-Screen (kein Auto-Login)
- [ ] Logout klicken → zurück zum Login
- [ ] Config-Datei nach Logout → nicht vorhanden

### ✅ Settings
- [ ] Settings-Button in Sidebar klickbar
- [ ] Alle 4 Tabs funktionieren
- [ ] User-Email korrekt angezeigt
- [ ] User-ID kopieren funktioniert
- [ ] API-Key anzeigen/verbergen funktioniert
- [ ] API-Key kopieren funktioniert
- [ ] Links öffnen korrekt
- [ ] Logout aus Settings funktioniert

## 🐛 Bekannte Probleme & Lösungen

### Problem: Build-Fehler bei tokio-postgres
```bash
# Lösung: Rust neu kompilieren
cd src-tauri
cargo clean
cargo build
```

### Problem: Supabase Connection Error
```
Prüfen:
1. Internet-Verbindung aktiv?
2. Supabase-URL korrekt?
3. Firewall blockiert Port 6543?
```

### Problem: Password-Verifizierung schlägt fehl
```
Mögliche Ursachen:
1. Password in DB ist nicht bcrypt-gehasht
2. Falsches bcrypt-Salt
3. Password-Hash ist null/leer in DB
```

## 🔒 Sicherheitshinweise

⚠️ **WICHTIG**: 
- Supabase Connection String enthält Credentials
- Sollte in `.env` ausgelagert werden (nicht in Code)
- Für Production: Environment Variables nutzen

```rust
// Besser für Production:
const SUPABASE_URL: &str = env!("DATABASE_URL");
```

## 📊 Datenbank-Queries

Die App führt folgende Queries aus:

```sql
-- Login: API-Key laden
SELECT id, user_id, key, is_active 
FROM api_keys 
WHERE key = $1;

-- Login: User-Daten laden
SELECT id, email, password_hash, has_paid 
FROM users 
WHERE id = $1;

-- Login: last_used_at aktualisieren
UPDATE api_keys 
SET last_used_at = NOW() 
WHERE id = $1;
```

## 🚀 Build Commands

```bash
# Development mit Hot-Reload
npm run tauri:dev

# Production Build
npm run tauri:build

# Nur Rust Backend testen
cd src-tauri
cargo run

# Frontend ohne Tauri
npm run dev
```

## 📦 Dependencies

### Rust (src-tauri/Cargo.toml)
```toml
tokio-postgres = "0.7"    # PostgreSQL Client
bcrypt = "0.15"           # Password Hashing
tokio = "1"               # Async Runtime
reqwest = "0.11"          # HTTP Client (falls später benötigt)
```

### TypeScript (package.json)
```json
"react": "^18.3.0"
"lucide-react": "^0.263.1"
"recharts": "^2.12.0"
```

## 💡 Tipps

1. **Entwicklung**: Nutze `npm run tauri:dev` für schnelles Testing
2. **Logs**: Rust-Fehler in Terminal, Frontend-Errors in DevTools
3. **Database**: Teste Queries zuerst in Supabase SQL Editor
4. **UI**: Tailwind DevTools für schnelles Styling

## 📞 Support

Bei Problemen:
1. Prüfe Console-Logs (Frontend & Backend)
2. Teste Datenbank-Verbindung separat
3. Validiere bcrypt-Hashes mit Online-Tool
4. Prüfe ob User in DB existiert und has_paid = true

## ✨ Features für Zukunft

- [ ] Token-basierte Auth (JWT)
- [ ] 2FA Support
- [ ] Session-Timeout
- [ ] Remember Me (optional)
- [ ] Biometrische Auth (Touch ID/Face ID)
- [ ] Multi-Account-Support
