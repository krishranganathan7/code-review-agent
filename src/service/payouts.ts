import { Claim } from '../types';
import { centsToDisplay, sumCents } from '../util/money';
import * as claims from '../repo/claims';

export interface PayoutSummary {
  claimantId: string;
  claimCount: number;
  total: string;
  average: string;
}

/** Summarise what a claimant is owed across all of their claims. */
export async function summarise(claimantId: string): Promise<PayoutSummary> {
  const rows = await claims.forClaimant(claimantId);
  const amounts = rows.map((c) => c.estimateCents / 100);
  const total = sumCents(amounts);

  return {
    claimantId,
    claimCount: rows.length,
    total: centsToDisplay(total),
    average: centsToDisplay(rows.length ? total / rows.length : 0),
  };
}

/** The largest single estimate in a set of claims. */
export function largest(rows: Claim[]): number {
  return rows.reduce((best, c) => (c.estimateCents > best ? c.estimateCents : best), 0);
}
