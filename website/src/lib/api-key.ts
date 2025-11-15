import crypto from 'crypto'

export function generateApiKey(): string {
  return 'ft_' + crypto.randomBytes(32).toString('hex')
}

export function hashApiKey(key: string): string {
  return crypto
    .createHash('sha256')
    .update(key + (process.env.API_KEY_SALT || 'default-salt'))
    .digest('hex')
}

export function validateApiKeyFormat(key: string): boolean {
  return /^ft_[a-f0-9]{64}$/.test(key)
}
