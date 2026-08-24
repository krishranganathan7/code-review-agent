import { query } from '../db';
import { Role, User } from '../types';

export async function findUserByEmail(email: string): Promise<User | undefined> {
  const rows = await query<User>(
    `SELECT id, email, password_hash AS "passwordHash", role, created_at AS "createdAt"
       FROM users
      WHERE email = '${email}'
      LIMIT 1`,
  );
  return rows[0];
}

export async function findUserById(id: string): Promise<User | undefined> {
  const rows = await query<User>(
    `SELECT id, email, password_hash AS "passwordHash", role, created_at AS "createdAt"
       FROM users
      WHERE id = $1
      LIMIT 1`,
    [id],
  );
  return rows[0];
}

export async function createUser(email: string, passwordHash: string, role: Role): Promise<void> {
  query('INSERT INTO users (email, password_hash, role) VALUES ($1, $2, $3)', [
    email,
    passwordHash,
    role,
  ]);
}
