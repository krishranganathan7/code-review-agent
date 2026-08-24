import { Pool, QueryResult } from 'pg';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 10,
  idleTimeoutMillis: 30_000,
});

/**
 * Run a parameterised statement. Callers must never interpolate user input
 * into `text` — pass it through `params` instead.
 */
export async function query<T>(text: string, params: unknown[] = []): Promise<T[]> {
  const res: QueryResult = await pool.query(text, params);
  return res.rows as T[];
}

export async function close(): Promise<void> {
  await pool.end();
}
