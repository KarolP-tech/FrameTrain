# 🔐 Desktop App - Authentication Setup

## ✅ WAS DIE APP BEREITS KANN:

Die Desktop App hat **VOLLSTÄNDIGE** Authentifizierung implementiert:

### 1. **Login-Validierung** ✅
- Prüft API-Key Format (`ft_xxxxxx`)
- Validiert gegen Supabase Database
- Prüft ob API-Key aktiv ist
- Prüft ob API-Key zum User gehört

### 2. **Password-Validierung** ✅
- Vergleicht Password mit bcrypt Hash
- Nur User mit korrektem Password + API-Key kommen rein

### 3. **Payment-Check** ✅
- Prüft ob `has_paid = true` in Database
- Nur zahlende User haben Zugang

### 4. **Session Management** ✅
- Speichert Credentials lokal (verschlüsselt)
- Auto-Login beim nächsten Start
- Logout löscht gespeicherte Daten

---

## 🔧 SETUP FÜR DEVELOPMENT:

### Schritt 1: Supabase Connection String setzen

Die Connection String muss in `src-tauri/src/auth.rs` gesetzt werden.

**Aktuell:**
```rust
const SUPABASE_URL: &str = "postgresql://postgres.pmilxbuzfghbphjjaiar:YOUR_PASSWORD_HERE@aws-1-eu-west-1.pooler.supabase.com:6543/postgres?pgbouncer=true&connection_limit=1";
```

**Ersetze `YOUR_PASSWORD_HERE` mit deinem echten Supabase Passwort!**

### Schritt 2: Desktop App bauen

```bash
cd desktop-app

# Dependencies installieren
npm install

# Dev Mode starten
npm run tauri:dev

# Production Build
npm run tauri:build
```

---

## 🎯 WIE DIE AUTHENTIFIZIERUNG FUNKTIONIERT:

### Login Flow:

```
1. User gibt ein:
   - API-Key: ft_xxxxxxxxxx
   - Password: ********

2. App prüft Format:
   ✓ API-Key beginnt mit "ft_"
   ✓ API-Key ist lang genug
   ✓ Password min. 6 Zeichen

3. App fragt Supabase:
   SELECT * FROM api_keys WHERE key = 'ft_xxxxx'
   
4. App prüft:
   ✓ API-Key existiert in Database?
   ✓ is_active = true?
   
5. App holt User-Daten:
   SELECT * FROM users WHERE id = user_id
   
6. App validiert:
   ✓ Password Hash stimmt mit bcrypt.verify?
   ✓ has_paid = true?
   
7. Wenn alles OK:
   ✅ Login erfolgreich
   ✅ Credentials werden lokal gespeichert
   ✅ User kommt ins Dashboard
   ✅ last_used_at wird aktualisiert

8. Wenn etwas falsch:
   ❌ Error-Message wird angezeigt
   ❌ User bleibt auf Login-Screen
```

---

## 🔒 SICHERHEIT:

### Was ist sicher:

- ✅ Password wird nie im Klartext gespeichert
- ✅ Password wird als bcrypt Hash in Database gespeichert
- ✅ Connection zu Supabase ist verschlüsselt (TLS)
- ✅ API-Keys sind eindeutig und nicht wiederverwendbar
- ✅ Lokale Credentials sind verschlüsselt gespeichert

### Was User sehen:

- ❌ API-Key Format ungültig
- ❌ API-Key nicht gefunden
- ❌ API-Key ist deaktiviert
- ❌ Falsches Passwort
- ❌ Account nicht aktiv (nicht bezahlt)
- ✅ Login erfolgreich!

---

## 🧪 TESTING:

### Lokaler Test:

1. **Starte Dev Server:**
   ```bash
   npm run tauri:dev
   ```

2. **Teste Login mit echten Credentials:**
   - API-Key aus Website Dashboard kopieren
   - Password von Website-Account nutzen

3. **Erwartetes Verhalten:**
   - ✅ Login erfolgreich → Dashboard öffnet
   - ✅ Logout → zurück zu Login
   - ✅ App neu starten → Auto-Login (gespeicherte Session)

---

## 📋 ZUSAMMENFASSUNG:

**Die Desktop App hat ALLES was sie braucht:**

✅ API-Key Validierung gegen Database  
✅ Password Check mit bcrypt  
✅ Payment Status Check (`has_paid`)  
✅ User-Zuordnung (API-Key gehört zu User)  
✅ Session Management  
✅ Auto-Login  
✅ Logout Funktion  

**Das einzige was fehlt:**
- ⚠️ Supabase Connection String muss mit echtem Password in `auth.rs` gesetzt werden

---

## 🎉 READY FOR PRODUCTION!

Nach dem Setzen der Connection String ist die App produktionsbereit für:
- ✅ Windows Build
- ✅ macOS Build  
- ✅ Linux Build

Alle Builds werden automatisch via GitHub Actions erstellt wenn ein Tag gepusht wird!
