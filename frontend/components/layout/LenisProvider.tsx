"use client";

import { ReactLenis } from 'lenis/react';
import { ReactNode } from 'react';

interface LenisProviderProps {
  children: ReactNode;
}

export function LenisProvider({ children }: LenisProviderProps) {
  return (
    <ReactLenis root options={{ autoRaf: true, anchors: true }}>
      {children}
    </ReactLenis>
  );
}