import { NextRequest, NextResponse } from 'next/server'
import Stripe from 'stripe'
import { prisma } from '@/lib/prisma'

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2024-11-20.acacia',
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

      // Speichere Payment in Datenbank
      await prisma.payment.create({
        data: {
          email: session.customer_email!,
          amount: session.amount_total!,
          currency: session.currency!,
          stripePaymentId: session.id,
          status: 'completed',
        },
      })

      // Hier könntest du auch einen API-Key generieren
      // und dem User per E-Mail zusenden

      console.log('Payment completed for:', session.customer_email)
      break

    case 'checkout.session.expired':
      console.log('Checkout session expired:', event.data.object.id)
      break

    default:
      console.log(`Unhandled event type ${event.type}`)
  }

  return NextResponse.json({ received: true })
}
