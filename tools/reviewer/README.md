# reviewer — vendored

The AI code-review agent, vendored here so `.github/workflows/ai-review.yml`
can install it without cross-repo authentication. Upstream source lives outside
this repository; **edit it there and re-vendor**, do not patch this copy.

Tests, `CLAUDE.md` and `PROGRESS.md` are deliberately not vendored — only what
`pip install` needs.

## Required repository secrets

Settings → Secrets and variables → Actions → *New repository secret*:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | access key id for an IAM principal with `bedrock:InvokeModel` |
| `AWS_SECRET_ACCESS_KEY` | its secret access key |

`GITHUB_TOKEN` is provided by Actions automatically — do not add it.

The region and model are set in the workflow, not as secrets:
`us-east-1` and `us.anthropic.claude-sonnet-4-5-20250929-v1:0`. Anthropic
inference profiles are not offered in every region, and a region without one
fails with a bare "model identifier is invalid" that never mentions the region.

## What it costs

Bedrock bills the account those credentials belong to, on every pull request.
Depth scales with the risk engine's verdict; `HIGH` allows 400k tokens per agent
per change group. A small PR runs well under that — a measured single-agent run
was ~61k tokens (~$0.20) — but the ceiling on a large, high-risk PR with three
agents across several groups is a few dollars.

## Running it locally

```bash
pip install -e "./tools/reviewer[bedrock]"

export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1
export GITHUB_TOKEN=...            # contents:read + pull_requests:read
export REVIEWER_MODEL=bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0

# Dry run: prints the review, posts nothing, needs no write scope.
python -m reviewer.run_review krishranganathan7/code-review-agent 1 --clone .

python -m reviewer.run_review --scopes   # exactly which permissions it uses
```

Exit codes: `0` pass or warn, `1` blocking finding, `2` error.

## A limitation worth knowing

Only a Python `LanguageAdapter` ships, so every `.ts` file in this repository
degrades to whole-file review: the change analyzer cannot resolve changed
symbols and tells the agents to read the files themselves. Risk scoring, signal
detection and agent selection are unaffected — a dry run over the `feat/auth`
branch correctly returned `HIGH` with `auth, public_api, request_surface` and
selected all three agents. Attribution is simply coarser than it would be on a
Python codebase. A TypeScript adapter would close the gap.
