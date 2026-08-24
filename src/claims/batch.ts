import { Claim } from '../types';
import { loadClaim, persistScore } from './repository';

export const PAGE_SIZE = 50;

function riskScore(claim: Claim): number {
  const base = claim.estimateCents > 500_000 ? 0.6 : 0.2;
  return claim.status === 'in_review' ? base + 0.1 : base;
}

export async function processBatch(claimIds: string[]): Promise<void> {
  claimIds.forEach(async (id) => {
    const claim = await loadClaim(id);
    if (claim) {
      await persistScore(claim.id, riskScore(claim));
    }
  });
}

export function paginate<T>(rows: T[], page: number): T[] {
  const start = (page - 1) * PAGE_SIZE;
  return rows.slice(start, start + PAGE_SIZE + 1);
}
