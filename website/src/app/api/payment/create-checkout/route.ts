import { NextRequest, NextResponse } from 'next/server'
import Stripe from 'stripe'
import { getCurrentUser } from '@/lib/auth'

// Force dynamic rendering
export const dynamic = 'force-dynamic'

// @ts-ignore - Stripe API version compatibility
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  // @ts-ignore
  apiVersion: '2024-06-20',
})

export async function POST(req: NextRequest) {
  try {
    const user = await getCurrentUser()

    if (!user) {
      return NextResponse.json(
        { error: 'Nicht authentifiziert' },
        { status: 401 }
      )
    }

    const priceId = process.env.STRIPE_PRICE_ID!
    
    // Get the base URL from the request or environment variable
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || 
                    `${req.nextUrl.protocol}//${req.nextUrl.host}`
    
    // Ensure the URL has a proper schema
    const fullBaseUrl = baseUrl.startsWith('http') ? baseUrl : `https://${baseUrl}`
    
    // Erstelle Stripe Checkout Session
    const session = await stripe.checkout.sessions.create({
      customer_email: user.email,
      payment_method_types: ['card'],
      line_items: [
        {
          price: priceId,
          quantity: 1,
        },
      ],
      mode: 'payment',
      success_url: `${fullBaseUrl}/payment/success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${fullBaseUrl}/payment/cancel`,
      metadata: {
        userId: user.userId,
        email: user.email,
      },
    })

    return NextResponse.json({ 
      sessionId: session.id,
      url: session.url,
    })
  } catch (error: any) {
    console.error('Stripe checkout error:', error)
    return NextResponse.json(
      { error: error.message || 'Fehler beim Erstellen der Checkout-Session' },
      { status: 500 }
    )
  }
}
