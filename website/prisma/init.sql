-- FrameTrain Database Schema
-- Generated from Prisma Schema
-- Execute this in Supabase SQL Editor

-- Create users table
CREATE TABLE IF NOT EXISTS "users" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "password_hash" TEXT NOT NULL,
    "has_paid" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- Create api_keys table
CREATE TABLE IF NOT EXISTS "api_keys" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "key" TEXT NOT NULL,
    "is_active" BOOLEAN NOT NULL DEFAULT true,
    "expires_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "last_used_at" TIMESTAMP(3),

    CONSTRAINT "api_keys_pkey" PRIMARY KEY ("id")
);

-- Create models table
CREATE TABLE IF NOT EXISTS "models" (
    "id" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "status" TEXT NOT NULL DEFAULT 'created',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "models_pkey" PRIMARY KEY ("id")
);

-- Create model_versions table
CREATE TABLE IF NOT EXISTS "model_versions" (
    "id" TEXT NOT NULL,
    "model_id" TEXT NOT NULL,
    "version" INTEGER NOT NULL,
    "parameters" TEXT NOT NULL,
    "metrics" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completed_at" TIMESTAMP(3),

    CONSTRAINT "model_versions_pkey" PRIMARY KEY ("id")
);

-- Create payments table
CREATE TABLE IF NOT EXISTS "payments" (
    "id" TEXT NOT NULL,
    "user_id" TEXT,
    "email" TEXT NOT NULL,
    "amount" INTEGER NOT NULL,
    "currency" TEXT NOT NULL DEFAULT 'eur',
    "stripe_payment_id" TEXT,
    "stripe_session_id" TEXT,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completed_at" TIMESTAMP(3),

    CONSTRAINT "payments_pkey" PRIMARY KEY ("id")
);

-- Create unique indexes
CREATE UNIQUE INDEX IF NOT EXISTS "users_email_key" ON "users"("email");
CREATE UNIQUE INDEX IF NOT EXISTS "api_keys_key_key" ON "api_keys"("key");
CREATE UNIQUE INDEX IF NOT EXISTS "payments_stripe_payment_id_key" ON "payments"("stripe_payment_id");
CREATE UNIQUE INDEX IF NOT EXISTS "payments_stripe_session_id_key" ON "payments"("stripe_session_id");
CREATE UNIQUE INDEX IF NOT EXISTS "model_versions_model_id_version_key" ON "model_versions"("model_id", "version");

-- Create regular indexes for performance
CREATE INDEX IF NOT EXISTS "api_keys_key_idx" ON "api_keys"("key");
CREATE INDEX IF NOT EXISTS "api_keys_user_id_idx" ON "api_keys"("user_id");
CREATE INDEX IF NOT EXISTS "models_user_id_idx" ON "models"("user_id");
CREATE INDEX IF NOT EXISTS "model_versions_model_id_idx" ON "model_versions"("model_id");
CREATE INDEX IF NOT EXISTS "payments_email_idx" ON "payments"("email");
CREATE INDEX IF NOT EXISTS "payments_user_id_idx" ON "payments"("user_id");

-- Add foreign key constraints
ALTER TABLE "api_keys" DROP CONSTRAINT IF EXISTS "api_keys_user_id_fkey";
ALTER TABLE "api_keys" ADD CONSTRAINT "api_keys_user_id_fkey" 
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "models" DROP CONSTRAINT IF EXISTS "models_user_id_fkey";
ALTER TABLE "models" ADD CONSTRAINT "models_user_id_fkey" 
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "model_versions" DROP CONSTRAINT IF EXISTS "model_versions_model_id_fkey";
ALTER TABLE "model_versions" ADD CONSTRAINT "model_versions_model_id_fkey" 
    FOREIGN KEY ("model_id") REFERENCES "models"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "payments" DROP CONSTRAINT IF EXISTS "payments_user_id_fkey";
ALTER TABLE "payments" ADD CONSTRAINT "payments_user_id_fkey" 
    FOREIGN KEY ("user_id") REFERENCES "users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- Success message
SELECT 'FrameTrain database schema created successfully!' AS status;
