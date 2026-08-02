import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { CitationCard } from '@/components/chat/CitationCard';
import { ConfidenceBadge } from '@/components/chat/ConfidenceBadge';
import type { Citation } from '@/lib/types';

describe('Chat Components', () => {
  it('renders CitationCard with correct authority level', () => {
    const citation: Citation = {
      source_uri: 'https://fda.gov/drug-x',
      page_number: 12,
      bbox: [0, 0, 100, 100],
      authority_level: 'regulatory',
    };
    render(<CitationCard citation={citation} />);
    expect(screen.getByText('REGULATORY')).toBeInTheDocument();
    expect(screen.getByText('Page 12')).toBeInTheDocument();
  });

  it('renders ConfidenceBadge with formatted percentages', () => {
    const vector = { faithfulness: 0.954, context_relevance: 0.881 };
    render(<ConfidenceBadge vector={vector} />);
    expect(screen.getByText('95.4%')).toBeInTheDocument();
    expect(screen.getByText('88.1%')).toBeInTheDocument();
  });
});