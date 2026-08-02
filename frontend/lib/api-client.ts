import { ChatRequest, ChatResponse, ChatResponseSchema } from './types';

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function sendChatMessage(req: ChatRequest): Promise<ChatResponse> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080';

  const response = await fetch(`${baseUrl}/v1/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(req),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new ApiError(response.status, `API Request failed: ${errorText}`);
  }

  const data = await response.json();
  const result = ChatResponseSchema.safeParse(data);

  if (!result.success) {
    console.error('Schema validation failed:', result.error.format());
    throw new ApiError(response.status, 'API response did not match expected schema');
  }

  return result.data;
}