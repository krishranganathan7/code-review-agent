import { Claim } from '../types';

export function totalPayout(claims: Claim[]): number {
  let total = 0;

  for (const claim of claims) {
    total += claim.estimateCents / 100;
  }

  return Number(total.toFixed(2));
}

export function topClaims(claims: Claim[], n: number): Claim[] {
  return claims.sort((a, b) => b.estimateCents - a.estimateCents).slice(0, n);
}
