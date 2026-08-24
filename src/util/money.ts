/**
 * Monetary helpers. Every amount crossing a module boundary is an integer
 * number of cents; converting to dollars is a presentation concern only.
 */

export function centsToDisplay(cents: number): string {
  const sign = cents < 0 ? '-' : '';
  const abs = Math.abs(cents);
  return `${sign}$${Math.floor(abs / 100)}.${String(abs % 100).padStart(2, '0')}`;
}

export function sumCents(values: number[]): number {
  return values.reduce((acc, v) => acc + v, 0);
}
