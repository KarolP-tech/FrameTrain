import { NextRequest, NextResponse } from 'next/server'
import bcrypt from 'bcryptjs'
import { prisma } from '@/lib/prisma'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

// Rate limiting map (in production, use Redis or similar)
const rateLimitMap = new Map<string, { count: number; resetTime: number }>()

const MAX_ATTEMPTS = 5
const WINDOW_MS = 15 * 60 * 1000 // 15 minutes

function checkRateLimit(identifier: string): { allowed: boolean; retryAfter?: number } {
  const now = Date.now()
  const record = rateLimitMap.get(identifier)

  if (!record || now > record.resetTime) {
    rateLimitMap.set(identifier, { count: 1, resetTime: now + WINDOW_MS })
    return { allowed: true }
  }

  if (record.count >= MAX_ATTEMPTS) {
    const retryAfter = Math.ceil((record.resetTime - now) / 1000)
    return { allowed: false, retryAfter }
  }

  record.count++
  return { allowed: true }
}

export async function POST(req: NextRequest) {
  try {
    // Get client IP for rate limiting
    const clientIp = req.headers.get('x-forwarded-for') || 
                     req.headers.get('x-real-ip') || 
                     'unknown'

    // Parse request body
    const body = await req.json()
    const { apiKey, password } = body

    // Validate input
    if (!apiKey || !password) {
      return NextResponse.json(
        { 
          error: 'API-Key und Passwort sind erforderlich',
          isValid: false 
        },
        { status: 400 }
      )
    }

    // Check rate limit
    const rateLimitCheck = checkRateLimit(clientIp)
    if (!rateLimitCheck.allowed) {
      return NextResponse.json(
        { 
          error: `Zu viele Versuche. Bitte warte ${rateLimitCheck.retryAfter} Sekunden.`,
          isValid: false 
        },
        { 
          status: 429,
          headers: {
            'Retry-After': String(rateLimitCheck.retryAfter)
          }
        }
      )
    }

    // 1. Validate API-Key format
    if (!apiKey.startsWith('ft_') || apiKey.length < 24) {
      return NextResponse.json(
        { 
          error: 'Ungültiges API-Key Format',
          isValid: false 
        },
        { status: 400 }
      )
    }

    // 2. Find API key in database
    const apiKeyRecord = await prisma.apiKey.findUnique({
      where: { key: apiKey },
      include: {
        user: {
          select: {
            id: true,
            email: true,
            passwordHash: true,
            hasPaid: true,
          }
        }
      }
    })

    if (!apiKeyRecord) {
      return NextResponse.json(
        { 
          error: 'API-Key nicht gefunden oder ungültig',
          isValid: false 
        },
        { status: 401 }
      )
    }

    // 3. Check if key is active
    if (!apiKeyRecord.isActive) {
      return NextResponse.json(
        { 
          error: 'API-Key ist deaktiviert',
          isValid: false 
        },
        { status: 401 }
      )
    }

    // 4. Check if key is expired
    if (apiKeyRecord.expiresAt && apiKeyRecord.expiresAt < new Date()) {
      return NextResponse.json(
        { 
          error: 'API-Key ist abgelaufen',
          isValid: false 
        },
        { status: 401 }
      )
    }

    // 5. Verify password
    const passwordValid = await bcrypt.compare(password, apiKeyRecord.user.passwordHash)
    
    if (!passwordValid) {
      return NextResponse.json(
        { 
          error: 'Falsches Passwort',
          isValid: false 
        },
        { status: 401 }
      )
    }

    // 6. Check if user has paid
    if (!apiKeyRecord.user.hasPaid) {
      return NextResponse.json(
        { 
          error: 'Account ist nicht aktiv. Bitte schließe eine Lizenz ab.',
          isValid: false 
        },
        { status: 403 }
      )
    }

    // 7. Update last_used_at
    await prisma.apiKey.update({
      where: { id: apiKeyRecord.id },
      data: { lastUsedAt: new Date() }
    })

    // 8. Return success
    return NextResponse.json({
      isValid: true,
      userId: apiKeyRecord.user.id,
      email: apiKeyRecord.user.email,
      message: 'Validierung erfolgreich'
    })

  } catch (error) {
    console.error('Credential validation error:', error)
    
    // Don't expose internal errors to client
    return NextResponse.json(
      { 
        error: 'Ein interner Fehler ist aufgetreten',
        isValid: false
      },
      { status: 500 }
    )
  }
}

// Health check endpoint
export async function GET() {
  return NextResponse.json({
    status: 'ok',
    service: 'desktop-credential-validation',
    timestamp: new Date().toISOString()
  })
}
