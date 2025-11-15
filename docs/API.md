# FrameTrain API Documentation

Vollständige API-Referenz für FrameTrain Backend.

## Base URL

```
Development: http://localhost:3000/api
Production: https://frametrain.ai/api
```

## Authentication

Die meisten Endpoints benötigen Authentifizierung via JWT Token.

**Header:**
```
Authorization: Bearer <jwt_token>
```

**Token erhalten:**
- Nach Login wird Token im Cookie `auth-token` gesetzt
- Oder im Response Body bei `/api/auth/login`

---

## Authentication Endpoints

### POST /api/auth/register

Registriert einen neuen Benutzer.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response:** `201 Created`
```json
{
  "success": true,
  "message": "Registrierung erfolgreich",
  "userId": "clx1234567890"
}
```

**Errors:**
- `400`: Ungültige Eingabe
- `409`: Email bereits registriert

---

### POST /api/auth/login

Meldet Benutzer an und gibt JWT Token zurück.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "id": "clx1234567890",
    "email": "user@example.com"
  }
}
```

**Errors:**
- `400`: Ungültige Eingabe
- `401`: Falsche Credentials

---

## API Key Endpoints

### POST /api/keys/create

Erstellt einen neuen API Key nach erfolgreicher Zahlung.

**Auth:** Required

**Request Body:**
```json
{
  "paymentId": "pi_1234567890"
}
```

**Response:** `201 Created`
```json
{
  "success": true,
  "apiKey": "ft_1234567890abcdef",
  "expiresAt": null
}
```

**Errors:**
- `401`: Nicht authentifiziert
- `400`: Ungültige Payment-ID
- `409`: Key existiert bereits für diesen User

---

### POST /api/keys/verify

Verifiziert ob ein API Key gültig ist.

**Auth:** Not required (Public endpoint für CLI/App)

**Request Body:**
```json
{
  "apiKey": "ft_1234567890abcdef"
}
```

**Response:** `200 OK`
```json
{
  "valid": true,
  "userId": "clx1234567890",
  "expiresAt": null
}
```

**Errors:**
- `400`: Key fehlt
- `404`: Key nicht gefunden oder ungültig

---

### GET /api/keys/list

Listet alle API Keys des eingeloggten Users.

**Auth:** Required

**Response:** `200 OK`
```json
{
  "keys": [
    {
      "id": "clx1234567890",
      "key": "ft_****7890abcdef",
      "isValid": true,
      "createdAt": "2024-01-15T10:00:00.000Z",
      "lastUsedAt": "2024-01-16T14:30:00.000Z"
    }
  ]
}
```

---

## Payment Endpoints

### POST /api/payment/create-checkout

Erstellt Stripe Checkout Session.

**Auth:** Not required

**Request Body:**
```json
{
  "email": "user@example.com"
}
```

**Response:** `200 OK`
```json
{
  "sessionId": "cs_test_1234567890",
  "url": "https://checkout.stripe.com/pay/cs_test_..."
}
```

**Errors:**
- `400`: Email fehlt oder ungültig
- `500`: Stripe Fehler

---

### POST /api/payment/webhook

Stripe Webhook Endpoint (nur für Stripe).

**Auth:** Stripe Signature

**Headers:**
```
Stripe-Signature: t=1234567890,v1=...
```

**Response:** `200 OK`
```json
{
  "received": true
}
```

**Events:**
- `checkout.session.completed`: Erstellt API Key für User

---

## Download Endpoints

### GET /api/download-app

Gibt Download-URL für Desktop-App zurück.

**Auth:** Required

**Query Parameters:**
```
?platform=windows|macos|linux
```

**Response:** `200 OK`
```json
{
  "downloadUrl": "https://downloads.frametrain.ai/v1.0.0/FrameTrain-Setup.exe",
  "version": "1.0.0",
  "platform": "windows",
  "checksum": "sha256:abc123..."
}
```

**Errors:**
- `401`: Nicht authentifiziert oder kein gültiger Key
- `400`: Ungültige Platform

---

## Model Endpoints (Optional - für zukünftige Features)

### POST /api/models/create

Erstellt neues Modell-Projekt.

**Auth:** Required

**Request Body:**
```json
{
  "name": "My Model",
  "description": "Description here"
}
```

**Response:** `201 Created`
```json
{
  "id": "clx1234567890",
  "name": "My Model",
  "status": "created",
  "createdAt": "2024-01-15T10:00:00.000Z"
}
```

---

### GET /api/models/list

Listet alle Modelle des Users.

**Auth:** Required

**Response:** `200 OK`
```json
{
  "models": [
    {
      "id": "clx1234567890",
      "name": "My Model",
      "status": "completed",
      "createdAt": "2024-01-15T10:00:00.000Z",
      "versions": 3
    }
  ]
}
```

---

### GET /api/models/:id

Gibt Details zu einem Modell zurück.

**Auth:** Required

**Response:** `200 OK`
```json
{
  "id": "clx1234567890",
  "name": "My Model",
  "description": "Description",
  "status": "completed",
  "createdAt": "2024-01-15T10:00:00.000Z",
  "versions": [
    {
      "version": 1,
      "parameters": {
        "epochs": 10,
        "batchSize": 32
      },
      "metrics": {
        "loss": 0.234,
        "accuracy": 0.925
      },
      "status": "completed",
      "createdAt": "2024-01-15T11:00:00.000Z"
    }
  ]
}
```

---

## Error Responses

Alle Endpoints können folgende Fehler zurückgeben:

### 400 Bad Request
```json
{
  "error": "Validation error",
  "details": {
    "email": "Invalid email format"
  }
}
```

### 401 Unauthorized
```json
{
  "error": "Unauthorized",
  "message": "Invalid or missing authentication token"
}
```

### 403 Forbidden
```json
{
  "error": "Forbidden",
  "message": "You don't have permission to access this resource"
}
```

### 404 Not Found
```json
{
  "error": "Not Found",
  "message": "Resource not found"
}
```

### 409 Conflict
```json
{
  "error": "Conflict",
  "message": "Resource already exists"
}
```

### 429 Too Many Requests
```json
{
  "error": "Rate Limit Exceeded",
  "message": "Too many requests, please try again later",
  "retryAfter": 60
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal Server Error",
  "message": "Something went wrong"
}
```

---

## Rate Limiting

API Endpoints sind rate-limited:

- **Authentication:** 5 requests / minute
- **Payment:** 3 requests / minute
- **Other:** 100 requests / minute

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642342800
```

---

## Pagination

Endpoints die Listen zurückgeben unterstützen Pagination:

**Query Parameters:**
```
?page=1&limit=20
```

**Response:**
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 100,
    "pages": 5
  }
}
```

---

## Webhook Events

### checkout.session.completed

Wird getriggert wenn Stripe Checkout abgeschlossen ist.

**Payload:**
```json
{
  "type": "checkout.session.completed",
  "data": {
    "object": {
      "id": "cs_test_...",
      "customer_email": "user@example.com",
      "payment_status": "paid"
    }
  }
}
```

**Action:** Erstellt API Key für User

---

## SDKs & Clients

### JavaScript/TypeScript

```typescript
import { FrameTrainClient } from '@frametrain/sdk'

const client = new FrameTrainClient({
  apiKey: 'ft_1234567890abcdef',
  baseUrl: 'https://frametrain.ai/api'
})

// Verify Key
const isValid = await client.verifyKey()

// Get Models
const models = await client.models.list()
```

### Python (CLI)

```python
from frametrain import FrameTrainAPI

api = FrameTrainAPI(api_key='ft_1234567890abcdef')

# Verify Key
is_valid = api.verify_key()

# Download App
app_url = api.download_app(platform='windows')
```

---

## Support

Bei Fragen zur API:
- Dokumentation: https://docs.frametrain.ai
- GitHub Issues: https://github.com/frametrain/frametrain/issues
- Email: support@frametrain.ai
