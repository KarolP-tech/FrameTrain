/**
 * Constants used across FrameTrain
 */

// ============================================
// API Configuration
// ============================================

export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000/api',
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3,
} as const;

// ============================================
// App Configuration
// ============================================

export const APP_CONFIG = {
  NAME: 'FrameTrain',
  VERSION: '1.0.0',
  DESCRIPTION: 'Professional Platform for Local Machine Learning Training',
  SUPPORT_EMAIL: 'support@frametrain.ai',
  WEBSITE: 'https://frametrain.ai',
} as const;

// ============================================
// Key Configuration
// ============================================

export const KEY_CONFIG = {
  PREFIX: 'ft',
  LENGTH: 64,
  EXPIRY_DAYS: null, // null = never expires
} as const;

// ============================================
// Payment Configuration
// ============================================

export const PAYMENT_CONFIG = {
  PRICE_EUR: 2,
  PRICE_CENTS: 200,
  CURRENCY: 'eur',
} as const;

// ============================================
// Training Configuration
// ============================================

export const TRAINING_CONFIG = {
  DEFAULT_EPOCHS: 10,
  DEFAULT_BATCH_SIZE: 32,
  DEFAULT_LEARNING_RATE: 0.001,
  DEFAULT_OPTIMIZER: 'adam',
  MAX_EPOCHS: 1000,
  MIN_EPOCHS: 1,
  MAX_BATCH_SIZE: 512,
  MIN_BATCH_SIZE: 1,
} as const;

// ============================================
// File Upload Configuration
// ============================================

export const UPLOAD_CONFIG = {
  MAX_FILE_SIZE: 100 * 1024 * 1024, // 100 MB
  ALLOWED_FORMATS: ['csv', 'json', 'png', 'jpg', 'jpeg', 'txt'],
  MAX_FILES: 10,
} as const;

// ============================================
// UI Constants
// ============================================

export const UI_CONFIG = {
  ITEMS_PER_PAGE: 10,
  TOAST_DURATION: 3000,
  DEBOUNCE_DELAY: 300,
  CHART_COLORS: {
    loss: '#ef4444',
    accuracy: '#10b981',
    valLoss: '#f97316',
    valAccuracy: '#3b82f6',
  },
} as const;

// ============================================
// Status Messages
// ============================================

export const STATUS_MESSAGES = {
  TRAINING: {
    PENDING: 'Training wird vorbereitet...',
    TRAINING: 'Training läuft...',
    COMPLETED: 'Training abgeschlossen',
    FAILED: 'Training fehlgeschlagen',
  },
  MODEL: {
    CREATED: 'Modell erstellt',
    TRAINING: 'Modell wird trainiert',
    COMPLETED: 'Modell fertig',
    FAILED: 'Modell-Training fehlgeschlagen',
  },
  KEY: {
    VALID: 'Gültiger API-Key',
    INVALID: 'Ungültiger API-Key',
    EXPIRED: 'API-Key abgelaufen',
  },
} as const;

// ============================================
// Error Messages
// ============================================

export const ERROR_MESSAGES = {
  AUTH: {
    INVALID_CREDENTIALS: 'Ungültige Anmeldedaten',
    UNAUTHORIZED: 'Nicht autorisiert',
    SESSION_EXPIRED: 'Sitzung abgelaufen',
  },
  KEY: {
    INVALID: 'Ungültiger API-Key',
    EXPIRED: 'API-Key ist abgelaufen',
    NOT_FOUND: 'API-Key nicht gefunden',
  },
  MODEL: {
    NOT_FOUND: 'Modell nicht gefunden',
    TRAINING_FAILED: 'Training fehlgeschlagen',
    INVALID_PARAMS: 'Ungültige Trainingsparameter',
  },
  DATASET: {
    INVALID_FORMAT: 'Ungültiges Dateiformat',
    TOO_LARGE: 'Datei zu groß',
    EMPTY: 'Datei ist leer',
  },
  PAYMENT: {
    FAILED: 'Zahlung fehlgeschlagen',
    CANCELLED: 'Zahlung abgebrochen',
  },
  GENERAL: {
    SERVER_ERROR: 'Serverfehler',
    NETWORK_ERROR: 'Netzwerkfehler',
    UNKNOWN: 'Unbekannter Fehler',
  },
} as const;

// ============================================
// Routes
// ============================================

export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  REGISTER: '/register',
  DASHBOARD: '/dashboard',
  API: {
    AUTH: {
      LOGIN: '/api/auth/login',
      REGISTER: '/api/auth/register',
      LOGOUT: '/api/auth/logout',
    },
    KEYS: {
      CREATE: '/api/keys/create',
      VERIFY: '/api/keys/verify',
      LIST: '/api/keys/list',
    },
    PAYMENT: {
      CREATE_INTENT: '/api/payment/create-intent',
      WEBHOOK: '/api/payment/webhook',
    },
    DOWNLOAD: '/api/download-app',
  },
} as const;

// ============================================
// Model Types
// ============================================

export const MODEL_TYPES = [
  'classification',
  'regression',
  'nlp',
  'computer-vision',
  'custom',
] as const;

export const OPTIMIZERS = [
  'adam',
  'sgd',
  'rmsprop',
  'adagrad',
] as const;

// ============================================
// Dataset Formats
// ============================================

export const DATASET_FORMATS = [
  'csv',
  'json',
  'images',
  'text',
] as const;
