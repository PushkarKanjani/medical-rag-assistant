import { Citation } from '@/lib/types';
import { Card } from '@/components/ui/card';

const authorityColors: Record<Citation['authority_level'], string> = {
  regulatory: 'bg-red-100 text-red-800 border-red-200',
  guideline: 'bg-blue-100 text-blue-800 border-blue-200',
  textbook: 'bg-green-100 text-green-800 border-green-200',
  label: 'bg-purple-100 text-purple-800 border-purple-200',
  journal: 'bg-yellow-100 text-yellow-800 border-yellow-200',
};

export function CitationCard({ citation }: { citation: Citation }) {
  return (
    <Card className='p-3 text-sm hover:bg-slate-50 transition-colors cursor-pointer'>
      <div className='flex justify-between items-start mb-2'>
        <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${authorityColors[citation.authority_level]}`}>
          {citation.authority_level.toUpperCase()}
        </span>
        <span className='text-slate-500 text-xs'>Page {citation.page_number}</span>
      </div>
      <div className='truncate text-slate-700 font-medium'>
        {citation.source_uri}
      </div>
    </Card>
  );
}