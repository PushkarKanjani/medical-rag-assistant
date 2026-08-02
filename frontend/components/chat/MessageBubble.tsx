import { ChatResponse, Citation } from '@/lib/types';
import { CitationCard } from './CitationCard';
import { ConfidenceBadge } from './ConfidenceBadge';

interface MessageBubbleProps {
  role: 'user' | 'assistant';
  content: string;
  response?: ChatResponse;
}

export function MessageBubble({ role, content, response }: MessageBubbleProps) {
  const isAssistant = role === 'assistant';
  
  return (
    <div className={`flex ${isAssistant ? 'justify-start' : 'justify-end'} mb-6`}>
      <div className={`max-w-[80%] p-4 rounded-2xl ${
        isAssistant ? 'bg-white border shadow-sm text-slate-800' : 'bg-blue-600 text-white'
      }`}>
        <div className='whitespace-pre-wrap leading-relaxed'>{content}</div>
        
        {isAssistant && response && (
          <div className='mt-4 pt-4 border-t border-slate-100'>
            <ConfidenceBadge vector={response.confidence_vector} />
            <div className='mt-3 grid grid-cols-1 gap-2'>
              {response.citations.map((cite, idx) => (
                <CitationCard key={idx} citation={cite} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}