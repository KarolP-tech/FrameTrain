import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getCurrentUser } from '@/lib/auth'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export async function POST(req: NextRequest) {
  try {
    const user = await getCurrentUser()

    if (!user) {
      return NextResponse.json(
        { error: 'Nicht authentifiziert' },
        { status: 401 }
      )
    }

    // Lösche alle alten Keys komplett aus der Datenbank
    await prisma.apiKey.deleteMany({
      where: {
        userId: user.userId,
      },
    })

    // Generiere neuen Key
    const newKey = `ft_${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`

    // Erstelle neuen Key
    const apiKey = await prisma.apiKey.create({
      data: {
        userId: user.userId,
        key: newKey,
        isActive: true,
      },
      select: {
        id: true,
        key: true,
        isActive: true,
        createdAt: true,
        lastUsedAt: true,
      },
    })

    return NextResponse.json({
      message: 'API-Key erfolgreich regeneriert',
      apiKey,
    })
  } catch (error) {
    console.error('Regenerate key error:', error)
    return NextResponse.json(
      { error: 'Ein Fehler ist aufgetreten' },
      { status: 500 }
    )
  }
}
