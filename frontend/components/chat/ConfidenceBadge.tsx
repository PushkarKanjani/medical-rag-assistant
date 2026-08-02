import { ConfidenceVector } from '@/lib/types';

export function ConfidenceBadge({ vector }: { vector: ConfidenceVector }) {
  return (
    <div className='flex gap-4 text-xs font-mono'>
      <div className='flex flex-col'>
        <span className='text-slate-500'>Faithfulness</span>
        <span className='font-bold text-slate-900'>{(vector.faithfulness * 100).toFixed(1)}%</span>
      </div>
      <div className='flex flex-col'>
        <span className='text-slate-500'>Relevance</span>
        <span className='font-bold text-slate-900'>{(vector.context_relevance * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}