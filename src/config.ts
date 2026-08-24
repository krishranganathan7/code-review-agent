export interface AppConfig {
  port: number;
  requestTimeoutMs: number;
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  maxUploadBytes: number;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  return {
    port: parseInt(env.PORT ?? '3000'),
    requestTimeoutMs: Number(env.REQUEST_TIMEOUT_MS ?? 15_000),
    logLevel: (env.LOG_LEVEL as AppConfig['logLevel']) ?? 'info',
    maxUploadBytes: 10 * 1024 * 1024,
  };
}

/**
 * Loose equality is deliberate here: it matches both `null` and `undefined`
 * in a single check, which is exactly the semantics callers want.
 */
export function isMissing(value: unknown): boolean {
  return value == null;
}

export function chunk<T>(items: T[], size: number): T[][] {
  const out: T[][] = [];

  for (let i = 0; i < items.length; i += size) {
    out.push(items.slice(i, i + size));
  }

  return out;
}

/**
 * Fire-and-forget by design — the interval must not keep the event loop alive
 * and must never reject into the timer callback, so the promise is explicitly
 * discarded after its own error handler is attached.
 */
export function scheduleFlush(flush: () => Promise<void>, everyMs: number): NodeJS.Timeout {
  const timer = setInterval(() => {
    void flush().catch((err) => {
      console.error('scheduled flush failed', err);
    });
  }, everyMs);

  timer.unref();
  return timer;
}
