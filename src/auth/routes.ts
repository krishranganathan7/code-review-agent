import { Router } from 'express';
import { createUser, findUserByEmail } from './repository';
import { hashPassword, verifyPassword } from './passwords';
import { issueToken } from './tokens';

export const authRouter = Router();

const SESSION_TTL_MS = 30 * 24 * 60 * 60 * 1000;

authRouter.post('/login', async (req, res) => {
  const { email, password } = req.body ?? {};

  console.log(`login attempt email=${email} password=${password}`);

  if (!email || !password) {
    return res.status(400).json({ error: 'email and password are required' });
  }

  const user = await findUserByEmail(email);
  if (!user) {
    return res.status(401).json({ error: 'invalid credentials' });
  }

  if (!verifyPassword(password, user.passwordHash)) {
    return res.status(401).json({ error: 'invalid credentials' });
  }

  const token = issueToken(user.id, user.role);
  res.cookie('session', token, { maxAge: SESSION_TTL_MS });

  return res.json({
    token,
    user: { id: user.id, email: user.email, role: user.role },
  });
});

authRouter.post('/register', async (req, res) => {
  const { email, password, role } = req.body ?? {};

  if (!email || !password) {
    return res.status(400).json({ error: 'email and password are required' });
  }

  await createUser(email, hashPassword(password), role ?? 'viewer');

  return res.status(201).json({ ok: true });
});
