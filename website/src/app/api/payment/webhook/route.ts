import { NextRequest, NextResponse } from 'next/server'
import Stripe from 'stripe'
import { prisma } from '@/lib/prisma'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

// @ts-ignore - Stripe API version compatibility
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  // @ts-ignore
  apiVersion: '2024-06-20',
})

export async function POST(req: NextRequest) {
  const body = await req.text()
  const sig = req.headers.get('stripe-signature')!

  let event: Stripe.Event

  try {
    event = stripe.webhooks.constructEvent(
      body,
      sig,
      process.env.STRIPE_WEBHOOK_SECRET!
    )
  } catch (err: any) {
    console.error('Webhook signature verification failed:', err.message)
    return NextResponse.json(
      { error: `Webhook Error: ${err.message}` },
      { status: 400 }
    )
  }

  // Handle the event
  switch (event.type) {
    case 'checkout.session.completed':
      const session = event.data.object as Stripe.Checkout.Session

      try {
        // Find user by email
        const user = await prisma.user.findUnique({
          where: { email: session.customer_email! },
        })

        if (user) {
          // Update user: mark as paid
          await prisma.user.update({
            where: { id: user.id },
            data: { 
              hasPaid: true,
              updatedAt: new Date(),
            },
          })

          // Create payment record
          await prisma.payment.create({
            data: {
              userId: user.id,
              email: session.customer_email!,
              amount: session.amount_total!,
              currency: session.currency!,
              stripePaymentId: session.payment_intent as string || session.id,
              stripeSessionId: session.id,
              status: 'completed',
              completedAt: new Date(),
            },
          })

          // Generate API Key
          const apiKeyString = `ft_${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}${Math.random().toString(36).substring(2, 15)}`
          
          await prisma.apiKey.create({
            data: {
              userId: user.id,
              key: apiKeyString,
              isActive: true,
              createdAt: new Date(),
            },
          })

          console.log('✅ Payment completed & API Key created for:', session.customer_email)
        } else {
          console.error('❌ User not found for email:', session.customer_email)
        }
      } catch (dbError) {
        console.error('❌ Database error in webhook:', dbError)
      }
      break

    case 'checkout.session.expired':
      console.log('⚠️ Checkout session expired:', event.data.object.id)
      break

    default:
      console.log(`ℹ️ Unhandled event type ${event.type}`)
  }

  return NextResponse.json({ received: true })
}
