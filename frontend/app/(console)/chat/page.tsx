import { ChatWindow } from '@/components/chat/ChatWindow';

export default function ChatPage() {
  return (
    <div className='min-h-[calc(100vh-64px)] bg-[radial-gradient(circle_at_top,_rgba(15,23,42,0.05),_transparent_38%),linear-gradient(180deg,_#f8fafc_0%,_#eef2ff_100%)]'>
      <main className='mx-auto flex min-h-[calc(100vh-64px)] w-full max-w-6xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8'>
        <section className='rounded-3xl border border-slate-200/80 bg-white/80 px-6 py-5 shadow-[0_20px_60px_-30px_rgba(15,23,42,0.35)] backdrop-blur'>
          <div className='flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between'>
            <div>
              <p className='text-xs font-semibold uppercase tracking-[0.24em] text-slate-500'>Clinician console</p>
              <h1 className='mt-1 text-2xl font-semibold text-slate-950 sm:text-3xl'>Pushkar MedAssist</h1>
              <p className='mt-2 max-w-2xl text-sm leading-6 text-slate-600'>
                Ask a clinical question, review grounded citations, and inspect the audit trail when you need the reasoning path.
              </p>
            </div>
            <div className='flex flex-wrap gap-2 text-xs font-medium text-slate-600'>
              <span className='rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-emerald-700'>
                Citations required
              </span>
              <span className='rounded-full border border-sky-200 bg-sky-50 px-3 py-1 text-sky-700'>
                Audit-ready output
              </span>
              <span className='rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-amber-700'>
                Safety-checked
              </span>
            </div>
          </div>
        </section>

        <section className='min-h-0 flex-1 overflow-hidden rounded-3xl border border-slate-200/80 bg-white/90 shadow-[0_24px_80px_-40px_rgba(15,23,42,0.45)]'>
        <ChatWindow />
        </section>
      </main>
    </div>
  );
}