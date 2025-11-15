import { z } from 'zod';

// ============================================
// User & Auth Types
// ============================================

export const UserSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  createdAt: z.date(),
  updatedAt: z.date(),
});

export type User = z.infer<typeof UserSchema>;

export const LoginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

export const RegisterSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  confirmPassword: z.string().min(8),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

export type LoginData = z.infer<typeof LoginSchema>;
export type RegisterData = z.infer<typeof RegisterSchema>;

// ============================================
// API Key Types
// ============================================

export const ApiKeySchema = z.object({
  id: z.string(),
  userId: z.string(),
  key: z.string(),
  isValid: z.boolean(),
  expiresAt: z.date().nullable(),
  createdAt: z.date(),
  lastUsedAt: z.date().nullable(),
});

export type ApiKey = z.infer<typeof ApiKeySchema>;

export const VerifyKeyRequestSchema = z.object({
  key: z.string(),
});

export const VerifyKeyResponseSchema = z.object({
  valid: z.boolean(),
  message: z.string(),
  userId: z.string().optional(),
});

export type VerifyKeyRequest = z.infer<typeof VerifyKeyRequestSchema>;
export type VerifyKeyResponse = z.infer<typeof VerifyKeyResponseSchema>;

// ============================================
// Model & Training Types
// ============================================

export const ModelStatusEnum = z.enum(['created', 'training', 'completed', 'failed']);
export type ModelStatus = z.infer<typeof ModelStatusEnum>;

export const ModelSchema = z.object({
  id: z.string(),
  userId: z.string(),
  name: z.string(),
  description: z.string().nullable(),
  status: ModelStatusEnum,
  createdAt: z.date(),
  updatedAt: z.date(),
});

export type Model = z.infer<typeof ModelSchema>;

export const TrainingParametersSchema = z.object({
  epochs: z.number().int().positive(),
  batchSize: z.number().int().positive(),
  learningRate: z.number().positive(),
  optimizer: z.enum(['adam', 'sgd', 'rmsprop', 'adagrad']),
  lossFunction: z.string().optional(),
  validationSplit: z.number().min(0).max(1).optional(),
});

export type TrainingParameters = z.infer<typeof TrainingParametersSchema>;

export const MetricsSchema = z.object({
  loss: z.array(z.number()),
  accuracy: z.array(z.number()).optional(),
  valLoss: z.array(z.number()).optional(),
  valAccuracy: z.array(z.number()).optional(),
  timestamp: z.array(z.string()),
});

export type Metrics = z.infer<typeof MetricsSchema>;

export const VersionStatusEnum = z.enum(['pending', 'training', 'completed', 'failed']);
export type VersionStatus = z.infer<typeof VersionStatusEnum>;

export const ModelVersionSchema = z.object({
  id: z.string(),
  modelId: z.string(),
  version: z.number().int(),
  parameters: TrainingParametersSchema,
  metrics: MetricsSchema.nullable(),
  status: VersionStatusEnum,
  createdAt: z.date(),
  completedAt: z.date().nullable(),
});

export type ModelVersion = z.infer<typeof ModelVersionSchema>;

// ============================================
// Dataset Types
// ============================================

export const DatasetSchema = z.object({
  name: z.string(),
  path: z.string(),
  size: z.number(),
  format: z.enum(['csv', 'json', 'images', 'text']),
  columns: z.array(z.string()).optional(),
  rowCount: z.number().optional(),
});

export type Dataset = z.infer<typeof DatasetSchema>;

// ============================================
// Payment Types
// ============================================

export const PaymentSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  amount: z.number().int().positive(),
  currency: z.string().default('eur'),
  stripePaymentId: z.string().nullable(),
  status: z.enum(['pending', 'completed', 'failed']),
  createdAt: z.date(),
});

export type Payment = z.infer<typeof PaymentSchema>;

export const CreatePaymentIntentSchema = z.object({
  email: z.string().email(),
  amount: z.number().int().positive().default(200), // 2 EUR in cents
});

export type CreatePaymentIntentData = z.infer<typeof CreatePaymentIntentSchema>;

// ============================================
// API Response Types
// ============================================

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

// ============================================
// Training Progress Types
// ============================================

export const TrainingProgressSchema = z.object({
  modelId: z.string(),
  versionId: z.string(),
  epoch: z.number().int(),
  totalEpochs: z.number().int(),
  batch: z.number().int().optional(),
  totalBatches: z.number().int().optional(),
  loss: z.number(),
  accuracy: z.number().optional(),
  valLoss: z.number().optional(),
  valAccuracy: z.number().optional(),
  eta: z.string().optional(), // Estimated time of arrival
  timestamp: z.string(),
});

export type TrainingProgress = z.infer<typeof TrainingProgressSchema>;

// ============================================
// Error Types
// ============================================

export class FrameTrainError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number = 500
  ) {
    super(message);
    this.name = 'FrameTrainError';
  }
}

export const ErrorCodes = {
  UNAUTHORIZED: 'UNAUTHORIZED',
  INVALID_KEY: 'INVALID_KEY',
  KEY_EXPIRED: 'KEY_EXPIRED',
  PAYMENT_FAILED: 'PAYMENT_FAILED',
  MODEL_NOT_FOUND: 'MODEL_NOT_FOUND',
  VERSION_NOT_FOUND: 'VERSION_NOT_FOUND',
  TRAINING_FAILED: 'TRAINING_FAILED',
  INVALID_DATASET: 'INVALID_DATASET',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  SERVER_ERROR: 'SERVER_ERROR',
} as const;

export type ErrorCode = typeof ErrorCodes[keyof typeof ErrorCodes];
