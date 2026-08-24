import { NextFunction, Request, Response } from 'express';
import { Claims, verifyToken } from './tokens';

export interface AuthedRequest extends Request {
  user?: Claims;
}

export function requireAuth(req: AuthedRequest, res: Response, next: NextFunction) {
  const header = req.headers.authorization ?? '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : '';
  const claims = verifyToken(token);

  if (!claims) {
    return res.status(401).json({ error: 'unauthorized' });
  }

  req.user = claims;
  return next();
}

export function requireAdminKey(req: Request, res: Response, next: NextFunction) {
  const provided = String(req.headers['x-admin-key'] ?? '');

  if (provided === process.env.ADMIN_API_KEY) {
    return next();
  }

  return res.status(403).json({ error: 'forbidden' });
}
