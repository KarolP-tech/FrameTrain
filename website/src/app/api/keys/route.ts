import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'
import { getCurrentUser } from '@/lib/auth'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

export async function GET(req: NextRequest) {
  try {
    const user = await getCurrentUser()

    if (!user) {
      return NextResponse.json(
        { error: 'Nicht authentifiziert' },
        { status: 401 }
      )
    }

    // Hole alle API Keys des Users
    const apiKeys = await prisma.apiKey.findMany({
      where: {
        userId: user.userId,
      },
      orderBy: {
        createdAt: 'desc',
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
      apiKeys,
      hasPaid: true, // User ist eingeloggt, also hat bezahlt
    })
  } catch (error) {
    console.error('Get keys error:', error)
    return NextResponse.json(
      { error: 'Ein Fehler ist aufgetreten' },
      { status: 500 }
    )
  }
}
