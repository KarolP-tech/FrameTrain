// Authentifizierungs-Modul für Supabase Integration

use serde::{Deserialize, Serialize};
use bcrypt;
use std::env;
use tokio_postgres::NoTls;

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiKeyValidation {
    pub user_id: String,
    pub email: String,
    pub is_valid: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct DbApiKey {
    id: String,
    user_id: String,
    key: String,
    is_active: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct DbUser {
    id: String,
    email: String,
    password_hash: String,
    has_paid: bool,
}

/// Holt die Supabase Connection String aus Environment Variable
fn get_database_url() -> Result<String, String> {
    env::var("SUPABASE_URL")
        .or_else(|_| env::var("DATABASE_URL"))
        .map_err(|_| "SUPABASE_URL environment variable not set".to_string())
}

/// Validiert API-Key und Password gegen Supabase-Datenbank
#[tauri::command]
pub async fn validate_credentials(api_key: String, password: String) -> Result<ApiKeyValidation, String> {
    println!("🔐 validate_credentials called");
    println!("   API Key: {}...", &api_key[..10]);
    println!("   Password length: {}", password.len());

    // 1. Prüfe API-Key Format
    if !api_key.starts_with("ft_") || api_key.len() < 24 {
        return Err("Ungültiges API-Key Format".to_string());
    }

    // 2. Hole Database URL
    let database_url = get_database_url()?;
    println!("🔌 Database URL: {}...", &database_url[..50]);

    // 3. Erstelle NEUE Connection für jede Anfrage (verhindert prepared statement conflicts)
    println!("📦 Creating new database connection...");
    let (client, connection) = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio_postgres::connect(&database_url, NoTls)
    )
    .await
    .map_err(|_| "Connection timeout".to_string())?
    .map_err(|e| format!("Connection failed: {}", e))?;

    // Spawn connection handler
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("❌ Connection error: {}", e);
        }
    });

    println!("   ✅ Connection established");

    // 4. API-Key aus Datenbank abrufen
    println!("🔑 Fetching API key from database...");
    let api_key_record = match fetch_api_key(&client, &api_key).await {
        Ok(record) => {
            println!("   ✅ API Key found: {}", record.id);
            record
        }
        Err(e) => {
            eprintln!("❌ API Key not found: {}", e);
            return Err("API-Key nicht gefunden oder ungültig".to_string());
        }
    };

    // 5. Prüfe ob Key aktiv ist
    if !api_key_record.is_active {
        println!("   ❌ API Key is inactive");
        return Err("API-Key ist deaktiviert".to_string());
    }

    println!("   ✅ API Key is active");

    // 6. User-Daten abrufen - MIT TIMEOUT!
    println!("👤 Fetching user data for ID: {}", api_key_record.user_id);
    
    let user = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        fetch_user(&client, &api_key_record.user_id)
    ).await {
        Ok(Ok(u)) => {
            println!("   ✅ User found: {}", u.email);
            u
        }
        Ok(Err(e)) => {
            eprintln!("❌ User fetch error: {}", e);
            return Err(format!("Benutzer nicht gefunden: {}", e));
        }
        Err(_) => {
            eprintln!("❌ User fetch TIMEOUT after 10 seconds");
            return Err("Datenbank-Timeout: Benutzer konnte nicht geladen werden".to_string());
        }
    };

    // 7. Password validieren
    println!("🔒 Verifying password...");
    let password_valid = match bcrypt::verify(&password, &user.password_hash) {
        Ok(valid) => {
            println!("   Password check: {}", if valid { "✅ valid" } else { "❌ invalid" });
            valid
        }
        Err(e) => {
            eprintln!("❌ Password verification error: {}", e);
            return Err("Password-Validierung fehlgeschlagen".to_string());
        }
    };

    if !password_valid {
        return Err("Falsches Passwort".to_string());
    }

    // 8. Prüfe ob User bezahlt hat
    if !user.has_paid {
        println!("   ⚠️  User has not paid");
        return Err("Account ist nicht aktiv. Bitte schließe eine Lizenz ab.".to_string());
    }

    println!("   ✅ User has paid");

    // 9. Update last_used_at
    update_api_key_usage(&client, &api_key_record.id).await.ok();

    // 10. Erfolgreiche Validierung
    println!("✅ Login successful for: {}", user.email);
    
    Ok(ApiKeyValidation {
        user_id: user.id,
        email: user.email,
        is_valid: true,
    })
}

/// Ruft API-Key aus Datenbank ab - OHNE prepared statements
async fn fetch_api_key(
    client: &tokio_postgres::Client,
    key: &str,
) -> Result<DbApiKey, String> {
    // Verwende format! um prepared statement conflicts zu vermeiden
    let query = format!(
        "SELECT id, user_id, key, is_active FROM api_keys WHERE key = '{}'",
        key.replace("'", "''") // SQL-Injection-Schutz
    );
    
    println!("   [DEBUG] API Key Query: {}", query);
    
    let rows = client.simple_query(&query).await
        .map_err(|e| format!("Query failed: {}", e))?;
    
    println!("   [DEBUG] Query returned {} messages", rows.len());
    
    // Finde die erste Row (nicht die erste Message!)
    let row = rows.iter()
        .find_map(|msg| {
            if let tokio_postgres::SimpleQueryMessage::Row(r) = msg {
                Some(r)
            } else {
                None
            }
        })
        .ok_or_else(|| "API key not found".to_string())?;
    
    println!("   [DEBUG] Found row with {} columns", row.len());
    for col_idx in 0..row.len() {
        if let Some(val) = row.get(col_idx) {
            println!("      Col {}: {}", col_idx, val);
        }
    }
    
    Ok(DbApiKey {
        id: row.get("id")
                .ok_or_else(|| "Column 'id' not found".to_string())?
                .to_string(),
            user_id: row.get("user_id")
                .ok_or_else(|| "Column 'user_id' not found".to_string())?
                .to_string(),
            key: row.get("key")
                .ok_or_else(|| "Column 'key' not found".to_string())?
                .to_string(),
            is_active: row.get("is_active")
                .ok_or_else(|| "Column 'is_active' not found".to_string())? == "t",
        })
}

/// Ruft User-Daten aus Datenbank ab - OHNE prepared statements
async fn fetch_user(
    client: &tokio_postgres::Client,
    user_id: &str,
) -> Result<DbUser, String> {
    println!("   [DEBUG] Executing query for user_id: '{}'", user_id);
    
    // Verwende format! um prepared statement conflicts zu vermeiden
    let query = format!(
        "SELECT id, email, password_hash, has_paid FROM users WHERE id = '{}'",
        user_id.replace("'", "''") // SQL-Injection-Schutz
    );
    
    println!("   [DEBUG] Query: {}", query);
    
    let rows = client.simple_query(&query).await
        .map_err(|e| {
            eprintln!("   [SQL ERROR] Message: {}", e);
            format!("Query failed: {}", e)
        })?;
    
    println!("   [DEBUG] Query returned {} messages", rows.len());
    
    // Finde die erste Row (nicht die erste Message!)
    let row = rows.iter()
        .find_map(|msg| {
            if let tokio_postgres::SimpleQueryMessage::Row(r) = msg {
                Some(r)
            } else {
                None
            }
        })
        .ok_or_else(|| "User not found".to_string())?;
    
    Ok(DbUser {
            id: row.get("id")
                .ok_or_else(|| "Column 'id' not found".to_string())?
                .to_string(),
            email: row.get("email")
                .ok_or_else(|| "Column 'email' not found".to_string())?
                .to_string(),
            password_hash: row.get("password_hash")
                .ok_or_else(|| "Column 'password_hash' not found".to_string())?
                .to_string(),
            has_paid: row.get("has_paid")
                .ok_or_else(|| "Column 'has_paid' not found".to_string())? == "t",
        })
}

/// Updated last_used_at für API-Key - OHNE prepared statements
async fn update_api_key_usage(
    client: &tokio_postgres::Client,
    key_id: &str,
) -> Result<(), String> {
    let query = format!(
        "UPDATE api_keys SET last_used_at = NOW() WHERE id = '{}'",
        key_id.replace("'", "''") // SQL-Injection-Schutz
    );
    
    client.simple_query(&query).await
        .map_err(|e| format!("Update failed: {}", e))?;
    Ok(())
}
