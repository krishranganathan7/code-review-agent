# AI code review

`.github/workflows/ai-review.yml` reviews every pull request against AWS Bedrock
and posts evidence-backed findings.

The reviewer itself lives in **[moder-mai/mai-code-review-agent](https://github.com/moder-mai/mai-code-review-agent)**
and is installed from there at run time. Nothing of it is vendored into this
repository, so this repository can be public while the reviewer stays private.

## Required repository secrets

Settings → Secrets and variables → Actions → *New repository secret*:

| Secret | What it is |
|---|---|
| `AWS_ACCESS_KEY_ID` | an IAM principal with `bedrock:InvokeModel` |
| `AWS_SECRET_ACCESS_KEY` | its secret access key |
| `REVIEWER_REPO_TOKEN` | fine-grained PAT, **Contents: Read** on `moder-mai/mai-code-review-agent` |

`GITHUB_TOKEN` is provided by Actions — do not add it. It is scoped to this
repository only, which is exactly why `REVIEWER_REPO_TOKEN` is needed to reach
the reviewer's repository.

Also required: Settings → Actions → General → **Workflow permissions →
"Read and write permissions"**. The job asks for `pull-requests: write` to post,
and a workflow cannot escalate past the repository default.

## How the verdict works

The bot never approves or requests changes. It posts a **COMMENT** review, and a
deterministic policy engine's verdict comes out as the job's exit code:

| verdict | exit | check |
|---|---|---|
| pass / warn | 0 | green |
| **block** | **1** | **red** |
| error | 2 | red |

A red check from a blocking finding is the system working. Add this job to
*Require status checks to pass* in branch protection to make it gate merges.

## Cost

Bedrock bills the AWS account behind those credentials on every pull request.
Depth scales with the risk engine, and prompt caching keeps the repeated prefix
cheap, but a large high-risk PR reviewed by three agents is still dollars rather
than cents. Draft PRs are skipped.

## Turning it down

- Drop `--publish` (and lower `pull-requests` to `read`) to run silently: the
  verdict still reaches the exit code and the trace artifact.
- `REVIEWER_REF` pins which version of the reviewer runs. It tracks `main`; set
  it to a tag when reviews need to be reproducible.
