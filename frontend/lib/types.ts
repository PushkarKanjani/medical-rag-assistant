import { z } from 'zod';

export const CitationSchema = z.object({
  source_uri: z.string().default('local://unknown'),
  page_number: z.number().default(1),
  bbox: z.tuple([z.number(), z.number(), z.number(), z.number()]).default([0.0, 0.0, 1.0, 1.0]),
  authority_level: z.enum(["regulatory", "guideline", "textbook", "label", "journal"]).catch("guideline"),
});

export const ConfidenceVectorSchema = z.object({
  faithfulness: z.number().default(0.85),
  context_relevance: z.number().default(0.85),
});

export const ChatRequestSchema = z.object({
  query: z.string(),
  user_id: z.string(),
  abha_id: z.string().optional(),
  max_iterations: z.number().optional(),
});

export const ChatResponseSchema = z.object({
  final_answer: z.string(),
  citations: z.array(CitationSchema).default([]),
  confidence_vector: ConfidenceVectorSchema.default({ faithfulness: 0.85, context_relevance: 0.85 }),
  audit_id: z.string().default(''),
});

export type Citation = z.infer<typeof CitationSchema>;
export type ConfidenceVector = z.infer<typeof ConfidenceVectorSchema>;
export type ChatRequest = z.infer<typeof ChatRequestSchema>;
export type ChatResponse = z.infer<typeof ChatResponseSchema>;