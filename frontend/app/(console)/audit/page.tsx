import { notFound } from 'next/navigation';

interface AuditEntry {
  node: string;
  timestamp: string;
  input: any;
  output: any;
  verdict?: string;
}

async function getAuditTrail(jobId: string) {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (!baseUrl) throw new Error('NEXT_PUBLIC_API_BASE_URL is not defined');

  const res = await fetch(`${baseUrl}/v1/audit/${jobId}`, {
    cache: 'no-store'
  });

  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Audit fetch failed: ${res.statusText}`);

  return res.json() as Promise<AuditEntry[]>;
}

export default async function AuditPage({
  searchParams,
}: {
  searchParams: Promise<{ job_id?: string }>;
}) {
  const { job_id } = await searchParams;

  if (!job_id) {
    return (
      <div className='p-8 text-center'>
        <h1 className='text-xl font-bold text-slate-800'>No Audit ID Provided</h1>
        <p className='text-slate-500'>Please provide a job_id in the query parameters.</p>
      </div>
    );
  }

  const trail = await getAuditTrail(job_id);

  if (!trail) {
    notFound();
  }

  return (
    <div className='max-w-4xl mx-auto p-6 bg-slate-50 min-h-screen'>
      <header className='mb-8'>
        <h1 className='text-2xl font-bold text-slate-900'>Clinical Audit Trail</h1>
        <p className='text-slate-500 font-mono text-sm'>Job ID: {job_id}</p>
      </header>

      <div className='relative border-l-2 border-slate-200 ml-3 space-y-8'>
        {trail.map((entry, idx) => (
          <div key={idx} className='relative pl-6'>
            <div className='absolute -left-[9px] top-1 w-4 h-4 rounded-full bg-blue-600 border-2 border-white shadow-sm' />
            
            <div className='bg-white p-4 rounded-lg border shadow-sm'>
              <div className='flex justify-between items-center mb-3'>
                <span className='font-bold text-slate-900 uppercase text-xs tracking-wider'>
                  {entry.node}
                </span>
                <span className='text-slate-400 text-xs font-mono'>
                  {new Date(entry.timestamp).toLocaleString()}
                </span>
              </div>
              
              <div className='space-y-3'>
                <div>
                  <span className='text-[10px] font-bold text-slate-400 uppercase'>Input</span>
                  <pre className='text-xs bg-slate-50 p-2 rounded border overflow-auto max-h-40'>
                    {JSON.stringify(entry.input, null, 2)}
                  </pre>
                </div>
                <div>
                  <span className='text-[10px] font-bold text-slate-400 uppercase'>Output</span>
                  <pre className='text-xs bg-slate-50 p-2 rounded border overflow-auto max-h-40'>
                    {JSON.stringify(entry.output, null, 2)}
                  </pre>
                </div>
                {entry.verdict && (
                  <div className='mt-2 p-2 rounded bg-blue-50 text-blue-700 text-xs font-bold border border-blue-100'>
                    Verdict: {entry.verdict}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}