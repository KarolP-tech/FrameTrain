import { PrismaClient } from '@prisma/client'

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log: ['query', 'info', 'warn', 'error'], // wichtig: verhindert lazy client caching
  })

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma