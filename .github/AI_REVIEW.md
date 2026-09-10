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
| `REVIEWER_REPO_TOKEN` | **classic** PAT with the `repo` scope, able to read `moder-mai/mai-code-review-agent` |

`GITHUB_TOKEN` is provided by Actions — do not add it. It is scoped to this
repository only and cannot read a second one, which is exactly why the deploy
key is needed to reach the reviewer's repository.

A *classic* PAT specifically. The two narrower options are both unavailable:
`moder-mai` gates fine-grained tokens behind organisation approval and disables
deploy keys outright. A classic PAT belongs to the user rather than to a
resource owner, so it needs nobody's approval.

The cost is scope. `repo` is the narrowest classic scope that reads a private
repository, and it grants read **and write** to every repository its owner can
reach. Use a short expiry, revoke it when the test is finished, and replace it
with a fine-grained token or a deploy key as soon as the organisation permits
one.

Also required: Settings → Actions → General → **Workflow permissions →
"Read and write permissions"**. The job asks for `pull-requests: write` to post,
and a workflow cannot escalate past the repository default.

## How the verdict works

The bot never approves or requests changes. It posts a **COMMENT** review, and a
deterministic policy engine's verdict comes out as the reviewer's exit code. The
workflow then decides what that means for the job:

| reviewer exit | meaning | job, by default | job with `FAIL_ON_BLOCK=true` |
|---|---|---|---|
| 0 | pass / warn | green | green |
| 1 | **blocking finding** | green, with a warning annotation | **red** |
| 2 | the reviewer itself failed | **red** | **red** |

A blocking finding is a *verdict*, not a malfunction, so by default it does not
fail the job — a red X on every pull request teaches people to ignore the check.
The verdict is still reported: a warning annotation, a run summary, and the
review on the pull request itself.

A reviewer **error** always fails, whatever the setting. Nothing was reviewed,
and a green check would be claiming otherwise.

### Arming the merge gate

When you want this to actually block merges:

1. Settings → Secrets and variables → Actions → **Variables** → new repository
   variable `FAIL_ON_BLOCK` = `true`.
2. Add this job to **Require status checks to pass** in branch protection.

Then a blocking finding fails the check and branch protection refuses the merge
— while the bot still never casts a review vote of its own.

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
