// Authentifizierungs-Modul für Supabase Integration

use serde::{Deserialize, Serialize};
use bcrypt;
use std::env;

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
    // Bei Development: Aus .env laden
    // Bei Production Build: Wird beim Kompilieren eingebettet via --env flag
    env::var("SUPABASE_URL")
        .or_else(|_| env::var("DATABASE_URL"))
        .map_err(|_| "SUPABASE_URL environment variable not set".to_string())
}

/// Validiert API-Key und Password gegen Supabase-Datenbank
#[tauri::command]
pub async fn validate_credentials(api_key: String, password: String) -> Result<ApiKeyValidation, String> {
    // 1. Prüfe API-Key Format
    if !api_key.starts_with("ft_") || api_key.len() < 24 {
        return Err("Ungültiges API-Key Format".to_string());
    }

    // 2. Hole Database URL
    let database_url = get_database_url()?;

    // 3. Verbindung zur Datenbank aufbauen
    let client = create_db_client(&database_url).await
        .map_err(|e| format!("Datenbankverbindung fehlgeschlagen: {}", e))?;

    // 4. API-Key aus Datenbank abrufen
    let api_key_record = fetch_api_key(&client, &api_key).await
        .map_err(|e| format!("API-Key nicht gefunden: {}", e))?;

    // 5. Prüfe ob Key aktiv ist
    if !api_key_record.is_active {
        return Err("API-Key ist deaktiviert".to_string());
    }

    // 6. User-Daten abrufen
    let user = fetch_user(&client, &api_key_record.user_id).await
        .map_err(|e| format!("Benutzer nicht gefunden: {}", e))?;

    // 7. Password validieren
    let password_valid = bcrypt::verify(&password, &user.password_hash)
        .map_err(|e| format!("Password-Validierung fehlgeschlagen: {}", e))?;

    if !password_valid {
        return Err("Falsches Passwort".to_string());
    }

    // 8. Prüfe ob User bezahlt hat
    if !user.has_paid {
        return Err("Account ist nicht aktiv. Bitte schließe eine Lizenz ab.".to_string());
    }

    // 9. Update last_used_at
    update_api_key_usage(&client, &api_key_record.id).await.ok();

    // 10. Erfolgreiche Validierung
    Ok(ApiKeyValidation {
        user_id: user.id,
        email: user.email,
        is_valid: true,
    })
}

/// Erstellt einen Datenbank-Client
async fn create_db_client(database_url: &str) -> Result<tokio_postgres::Client, tokio_postgres::Error> {
    let (client, connection) = tokio_postgres::connect(database_url, tokio_postgres::NoTls).await?;

    // Spawn connection handler
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("Database connection error: {}", e);
        }
    });

    Ok(client)
}

/// Ruft API-Key aus Datenbank ab
async fn fetch_api_key(
    client: &tokio_postgres::Client,
    key: &str,
) -> Result<DbApiKey, tokio_postgres::Error> {
    let row = client
        .query_one(
            "SELECT id, user_id, key, is_active FROM api_keys WHERE key = $1",
            &[&key],
        )
        .await?;

    Ok(DbApiKey {
        id: row.get(0),
        user_id: row.get(1),
        key: row.get(2),
        is_active: row.get(3),
    })
}

/// Ruft User-Daten aus Datenbank ab
async fn fetch_user(
    client: &tokio_postgres::Client,
    user_id: &str,
) -> Result<DbUser, tokio_postgres::Error> {
    let row = client
        .query_one(
            "SELECT id, email, password_hash, has_paid FROM users WHERE id = $1",
            &[&user_id],
        )
        .await?;

    Ok(DbUser {
        id: row.get(0),
        email: row.get(1),
        password_hash: row.get(2),
        has_paid: row.get(3),
    })
}

/// Updated last_used_at für API-Key
async fn update_api_key_usage(
    client: &tokio_postgres::Client,
    key_id: &str,
) -> Result<(), tokio_postgres::Error> {
    client
        .execute(
            "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1",
            &[&key_id],
        )
        .await?;

    Ok(())
}
