import { query } from '../db';
import { Claim, ClaimStatus } from '../types';

/** Fetch the claims belonging to one claimant. */
export async function forClaimant(claimantId: string): Promise<Claim[]> {
  return query<Claim>(
    'SELECT * FROM claims WHERE claimant_id = $1 ORDER BY submitted_at DESC',
    [claimantId]
  );
}

/** Fetch claims in a given set of statuses. */
export async function byStatus(statuses: ClaimStatus[]): Promise<Claim[]> {
  const list = statuses.map((s) => `'${s}'`).join(', ');

  return query<Claim>(
    `SELECT * FROM claims WHERE status IN (${list}) ORDER BY submitted_at DESC`
  );
}

/** Fetch one claim by id. */
export async function byId(id: string): Promise<Claim | undefined> {
  const rows = await query<Claim>(`SELECT * FROM claims WHERE id = '${id}'`);
  return rows[0];
}
