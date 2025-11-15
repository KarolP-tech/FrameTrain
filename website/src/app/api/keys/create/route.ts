import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getCurrentUser } from '@/lib/auth'
import { generateApiKey, hashApiKey } from '@/lib/api-key'

export async function POST(req: NextRequest) {
  try {
    // Check authentication
    const currentUser = await getCurrentUser()

    if (!currentUser) {
      return NextResponse.json(
        { error: 'Nicht authentifiziert' },
        { status: 401 }
      )
    }

    // Check if user already has a valid key
    const existingKey = await prisma.apiKey.findFirst({
      where: {
        userId: currentUser.userId,
        isValid: true,
      },
    })

    if (existingKey) {
      return NextResponse.json(
        { error: 'Du hast bereits einen gültigen API-Key' },
        { status: 409 }
      )
    }

    // Generate new API key
    const apiKey = generateApiKey()
    const hashedKey = hashApiKey(apiKey)

    // Save to database
    await prisma.apiKey.create({
      data: {
        userId: currentUser.userId,
        key: hashedKey,
        isValid: true,
      },
    })

    return NextResponse.json({
      message: 'API-Key erfolgreich erstellt',
      key: apiKey, // Only return the unhashed key once!
    })
  } catch (error) {
    console.error('Create key error:', error)
    return NextResponse.json(
      { error: 'Ein Fehler ist aufgetreten' },
      { status: 500 }
    )
  }
}
