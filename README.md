# claims-api

Internal HTTP API for claim intake and batch processing.

## Conventions

- **Money is always integer cents.** Never store or compute monetary values as
  floating-point dollars. Formatting helpers live in `src/util/money.ts`.
- All database access goes through `src/db.ts`. Use parameterised queries.
- Route handlers stay thin; business logic lives in the module directories.

## Development

```bash
npm install
npm run typecheck
npm run build && npm start
```
