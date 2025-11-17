import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'
import { hashApiKey, validateApiKeyFormat } from '@/lib/api-key'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { key } = body

    if (!key) {
      return NextResponse.json(
        { error: 'API-Key ist erforderlich', valid: false },
        { status: 400 }
      )
    }

    // Validate format
    if (!validateApiKeyFormat(key)) {
      return NextResponse.json(
        { error: 'Ungültiges Key-Format', valid: false },
        { status: 400 }
      )
    }

    // Hash the key
    const hashedKey = hashApiKey(key)

    // Check if key exists and is valid
    const apiKey = await prisma.apiKey.findUnique({
      where: { key: hashedKey },
      include: {
        user: {
          select: {
            id: true,
            email: true,
          },
        },
      },
    })

    if (!apiKey || !apiKey.isActive) {
      return NextResponse.json(
        { error: 'Ungültiger oder abgelaufener Key', valid: false },
        { status: 401 }
      )
    }

    // Update last used timestamp
    await prisma.apiKey.update({
      where: { id: apiKey.id },
      data: { lastUsedAt: new Date() },
    })

    return NextResponse.json({
      valid: true,
      message: 'Key ist gültig',
      user: apiKey.user,
    })
  } catch (error) {
    console.error('Verify key error:', error)
    return NextResponse.json(
      { error: 'Ein Fehler ist aufgetreten', valid: false },
      { status: 500 }
    )
  }
}
