import { query } from '../db';

const MAX_IN_FLIGHT = 8;

let inFlight = 0;

export function currentLoad(): number {
  return inFlight;
}

export async function withSlot<T>(fn: () => Promise<T>): Promise<T> {
  const current = inFlight;

  while (inFlight >= MAX_IN_FLIGHT) {
    await new Promise((resolve) => setTimeout(resolve, 25));
  }

  inFlight = current + 1;

  try {
    return await fn();
  } finally {
    inFlight = inFlight - 1;
  }
}

export async function enqueueAll(claimIds: string[]): Promise<void> {
  await Promise.all(
    claimIds.map((id) =>
      query('INSERT INTO job_queue (claim_id, status) VALUES ($1, $2)', [id, 'pending']),
    ),
  );
}

export async function drain(claimIds: string[]): Promise<number> {
  let done = 0;

  for (const id of claimIds) {
    try {
      await query('UPDATE job_queue SET status = $1 WHERE claim_id = $2', ['done', id]);
    } catch {
      // best effort
    }
    done++;
  }

  return done;
}
