import type { Metadata } from 'next';
import './globals.css';
import { LenisProvider } from '@/components/layout/LenisProvider';
import Providers from '@/components/layout/Providers';

export const metadata: Metadata = {
  title: 'Pushkar MedAssist',
  description: 'Clinical Decision Support Platform',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang='en'>
      <body>
        <Providers>
          <LenisProvider>
            {children}
          </LenisProvider>
        </Providers>
      </body>
    </html>
  );
}
