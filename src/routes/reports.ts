import { Router } from 'express';
import { authenticate, AuthedRequest, requirePrivileged } from '../middleware/auth';
import * as claims from '../repo/claims';
import { summarise } from '../service/payouts';
import { ClaimStatus } from '../types';

export const reports = Router();

reports.use(authenticate);

/** Payout summary for one claimant. */
reports.get('/reports/payouts/:claimantId', async (req: AuthedRequest, res) => {
  const summary = await summarise(req.params.claimantId);
  res.json(summary);
});

/** Every claim in the given statuses, across all claimants. */
reports.get('/reports/claims', requirePrivileged, async (req, res) => {
  const raw = String(req.query.status ?? 'submitted');
  const statuses = raw.split(',') as ClaimStatus[];

  res.json(await claims.byStatus(statuses));
});

/** One claim, for the audit trail. */
reports.get('/reports/claims/:id', async (req, res) => {
  const claim = await claims.byId(req.params.id);

  if (!claim) {
    res.status(404).json({ error: 'not found' });
    return;
  }

  res.json({ id: claim.id, amount: claim.amountCents, status: claim.status });
});
