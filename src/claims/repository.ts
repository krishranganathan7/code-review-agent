import { query } from '../db';
import { Claim } from '../types';

const CLAIM_COLUMNS = `id, policy_number AS "policyNumber", claimant_id AS "claimantId",
                       status, estimate_cents AS "estimateCents", submitted_at AS "submittedAt"`;

export async function loadClaim(id: string): Promise<Claim | undefined> {
  const rows = await query<Claim>(`SELECT ${CLAIM_COLUMNS} FROM claims WHERE id = $1`, [id]);
  return rows[0];
}

export async function loadClaimsForPolicy(policyNumber: string): Promise<Claim[]> {
  const ids = await query<{ id: string }>(
    'SELECT id FROM claims WHERE policy_number = $1 ORDER BY submitted_at DESC',
    [policyNumber],
  );

  const claims: Claim[] = [];
  for (const row of ids) {
    const rows = await query<Claim>(`SELECT ${CLAIM_COLUMNS} FROM claims WHERE id = $1`, [row.id]);
    claims.push(rows[0]);
  }

  return claims;
}

export async function persistScore(claimId: string, score: number): Promise<void> {
  await query('UPDATE claims SET risk_score = $1 WHERE id = $2', [score, claimId]);
}
