"use client";

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { sendChatMessage } from '@/lib/api-client';
import { ChatResponse } from '@/lib/types';
import { MessageBubble } from './MessageBubble';
import { motion, AnimatePresence } from 'motion/react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  response?: ChatResponse;
}

export function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');

  const mutation = useMutation({
    mutationFn: sendChatMessage,
    onSuccess: (data) => {
      setMessages(prev => [...prev, { role: 'assistant', content: data.final_answer, response: data }]);
    },
    onError: (error) => {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}` }]);
    }
  });

  const isEmpty = messages.length === 0 && !mutation.isPending;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);

    mutation.mutate({
      query: userMsg,
      user_id: 'clinician-01',
    });
  };

  return (
    <div className='flex h-full min-h-[70vh] flex-col'>
      <div className='flex-1 overflow-y-auto px-4 py-5 sm:px-6'>
        {isEmpty && (
          <div className='mb-6 rounded-3xl border border-dashed border-slate-300 bg-slate-50/80 p-6 text-slate-600'>
            <div className='max-w-2xl'>
              <p className='text-xs font-semibold uppercase tracking-[0.22em] text-slate-500'>Ready to begin</p>
              <h2 className='mt-2 text-xl font-semibold text-slate-950'>Ask for a differential, interaction check, dosage review, or guideline summary.</h2>
              <p className='mt-2 text-sm leading-6'>
                The assistant will answer with citations and an audit id once the backend returns a validated response.
              </p>
            </div>
            <div className='mt-5 flex flex-wrap gap-2 text-sm'>
              {['Acute fever with rash', 'Pediatric amoxicillin dose', 'Drug interaction with warfarin', 'Hypertension management steps'].map((label) => (
                <button
                  key={label}
                  type='button'
                  onClick={() => setInput(label)}
                  className='rounded-full border border-slate-300 bg-white px-4 py-2 text-slate-700 transition-colors hover:border-slate-400 hover:bg-slate-100'
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className='space-y-4'>
          <AnimatePresence>
            {messages.map((msg, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3 }}
              >
                <MessageBubble {...msg} />
              </motion.div>
            ))}
          </AnimatePresence>
          {mutation.isPending && (
            <div className='mb-6 flex justify-start'>
              <Card className='p-4 animate-pulse text-slate-400'>
                Consulting clinical knowledge base...
              </Card>
            </div>
          )}
        </div>
      </div>

      <form onSubmit={handleSubmit} className='border-t border-slate-200 bg-white/95 p-4 sm:p-6'>
        <div className='relative'>
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
            placeholder='Enter clinical query...'
            className='h-24 resize-none pr-20'
          />
          <Button
            type='submit'
            disabled={mutation.isPending || !input.trim()}
            className='absolute right-3 bottom-3 rounded-lg bg-blue-600 px-4 py-2 font-medium text-white transition-colors hover:bg-blue-700 disabled:bg-slate-300'
          >
            Send
          </Button>
        </div>
      </form>
    </div>
  );
}