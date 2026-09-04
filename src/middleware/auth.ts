import { NextFunction, Request, Response } from 'express';
import { query } from '../db';
import { User } from '../types';

export interface AuthedRequest extends Request {
  user?: User;
}

/** Resolve the caller from the bearer token and attach it to the request. */
export async function authenticate(
  req: AuthedRequest,
  _res: Response,
  next: NextFunction
): Promise<void> {
  const header = req.headers.authorization ?? '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : '';

  if (!token) {
    return next();
  }

  const rows = await query<User>(
    'SELECT * FROM users WHERE session_token = $1',
    [token]
  );
  req.user = rows[0];
  return next();
}

/** Allow only callers who may see other claimants' data. */
export function requirePrivileged(
  req: AuthedRequest,
  res: Response,
  next: NextFunction
): void {
  const role = req.user?.role;

  if (role === 'superadmin' || role === 'adjuster') {
    return next();
  }

  res.status(403).json({ error: 'forbidden' });
}
