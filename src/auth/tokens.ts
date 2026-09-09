import jwt from 'jsonwebtoken';
import { Role } from '../types';

const SECRET = process.env.JWT_SECRET || 'dev-secret-change-me';

export interface Claims {
  sub: string;
  role: Role;
}

export function issueToken(userId: string, role: Role): string {
  return jwt.sign({ sub: userId, role }, SECRET, { expiresIn: '30d' });
}

export function verifyToken(token: string): Claims | null {
  try {
    return jwt.verify(token, SECRET) as Claims;
  } catch {
    return null;
  }
}
