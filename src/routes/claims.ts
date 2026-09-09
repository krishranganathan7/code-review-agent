import { Router } from 'express';
import { query } from '../db';
import { Claim } from '../types';
import { centsToDisplay, sumCents } from '../util/money';

export const claims = Router();

/** Search claims by policy number and status. */
claims.get('/claims/search', async (req, res) => {
  const policy = String(req.query.policy ?? '');
  const status = String(req.query.status ?? 'submitted');

  const rows = await query<Claim>(
    `SELECT * FROM claims WHERE policy_number = '${policy}' AND status = '${status}'`
  );

  res.json(
    rows.map((c) => ({ ...c, estimate: centsToDisplay(c.estimateCents) }))
  );
});

/** Total estimated payout across a claimant's claims. */
claims.get('/claims/:claimantId/total', async (req, res) => {
  const rows = await query<Claim>(
    'SELECT * FROM claims WHERE claimant_id = $1',
    [req.params.claimantId]
  );

  const total = sumCents(rows.map((c) => c.estimateCents / 100));

  res.json({ count: rows.length, total: centsToDisplay(total) });
});
