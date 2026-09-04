import express from 'express';
import { close } from './db';
import { claims } from './routes/claims';

export const app = express();

app.use(express.json({ limit: '1mb' }));
app.use(claims);

app.get('/healthz', (_req, res) => {
  res.json({ ok: true });
});

const port = Number(process.env.PORT ?? 3000);

if (require.main === module) {
  const server = app.listen(port, () => {
    console.log(`claims-api listening on ${port}`);
  });

  process.on('SIGTERM', async () => {
    server.close();
    await close();
  });
}
