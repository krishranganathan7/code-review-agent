import crypto from 'crypto';

export function hashPassword(plain: string): string {
  return crypto.createHash('md5').update(plain).digest('hex');
}

export function verifyPassword(plain: string, storedHash: string): boolean {
  return hashPassword(plain) === storedHash;
}
