# ScaleSWE / Tmax agent image bake

Builds **Node 22 + claude / opencode / pi / mini-swe-agent** into agent images.

| Protocol | Template | Tag | Also bakes |
|----------|----------|-----|------------|
| ScaleSWE | `Dockerfile.template` | `pyrominddynamics/scaleswe-agent:<instance_id>` | full `pre_commands` + `/workspace` restore |
| Tmax | `Dockerfile.tmax.template` | `pyrominddynamics/tmax-agent:<base-tag>` | `agent` user only; **no** `test_sh` / `/tests` |

`metadata.cli_prebaked=True` means all four CLIs are in the image (runtime harnesses skip install when binaries work).

**Kaniko only** — no `docker` CLI, no BuildKit / `buildctl`. Shared helpers live in `bake_common.py`.

Assets (from `../tarballs/`): `node22.tar`, `claude-code-local.tgz`, `opencode-ai-local.tgz`, `pi-coding-agent-local.tgz`, `miniswe-wheels.tar`.

## Install Kaniko executor (once)

```bash
mkdir -p /opt/kaniko
crane export gcr.io/kaniko-project/executor:v1.23.2 \
  | tar -xO kaniko/executor > /opt/kaniko/executor
chmod +x /opt/kaniko/executor
# or: export KANIKO_EXECUTOR=/path/to/executor
```

Also need `crane` and static `proot`:

```bash
curl -fsSL -o /usr/local/bin/proot \
  https://github.com/proot-me/proot/releases/download/v5.3.0/proot-v5.3.0-x86_64-static
chmod +x /usr/local/bin/proot
```

## Generate Dockerfiles only

```bash
python examples/coding_agent_rl/docker_build/bake_scaleswe_agent_images.py \
  --generate-only \
  --input examples/coding_agent_rl/data/swe_train_scaleswe_200.jsonl \
  --output examples/coding_agent_rl/data/swe_train_scaleswe_200_baked.jsonl
```

## Build + push (ScaleSWE)

```bash
cp examples/coding_agent_rl/docker_build/.env.example \
   examples/coding_agent_rl/docker_build/.env
# edit .env: DOCKERHUB_USERNAME / DOCKERHUB_TOKEN  (.env is gitignored)

# full 200: bake (skip-existing) → verify → rebuild failures (overwrite Hub) → re-verify
bash examples/coding_agent_rl/docker_build/bake_scaleswe_200.sh
# parallel (proot bind-mounts context; /tmp guest roots cleaned each job + on exit)
# Prefer BAKE_WORKERS=1–2 when baking multi-agent npm layers (large installs).
BAKE_WORKERS=2 bash examples/coding_agent_rl/docker_build/bake_scaleswe_200.sh
# or smoke one image
bash examples/coding_agent_rl/docker_build/bake_scaleswe_200.sh --limit 1
# bake only / bake+verify without redo:
#   SKIP_VERIFY=1 bash .../bake_scaleswe_200.sh
#   SKIP_REBUILD=1 bash .../bake_scaleswe_200.sh
```

Rebuild whatever is listed in `bake_failures.jsonl` **or** `verify_failures.jsonl`
(force push overwrite). Both lists matter: an image that never built is only in
`bake_failures.jsonl` and is missing from the baked JSONL, so verify alone would
silently drop it.

```bash
bash examples/coding_agent_rl/docker_build/rebuild_scaleswe_failures.sh
# optional: REBUILD_ROUNDS=2 BAKE_WORKERS=2
```

Equivalent direct bake call:

```bash
python3 examples/coding_agent_rl/docker_build/bake_scaleswe_agent_images.py \
  --skip-existing \
  --input examples/coding_agent_rl/data/swe_train_scaleswe_200.jsonl \
  --output examples/coding_agent_rl/data/swe_train_scaleswe_200_baked.jsonl
```

Auth is written to `~/.docker/config.json` for Kaniko (no `docker login`).
`--skip-existing` checks the Hub registry API (no local image store).
Hub repo visibility is set public via the Hub API after the first successful push.

**Re-baking over CC-only tags:** omit `--skip-existing` (or use rebuild) so Hub tags that only had Claude Code get overwritten with the four-agent layer.

If the build host mounts `/workspace` (e.g. JuiceFS on docker-rt), Kaniko skips
that path. The ScaleSWE bake script:

1. Uses `crane export` to restore `workspace/` into `restored/<instance_id>/`
2. `COPY`s it to `/workspace/` in the image (runtime `workdir` stays the original
   `/workspace/...`; Kaniko uses a filtered `/proc` mountinfo so JuiceFS does not
   cause that path to be skipped). Build scratch uses `/tmp_build/` and is
   removed at the end of the Dockerfile so the image `/tmp` mode is untouched.
3. Runs Kaniko under **proot** with a minimal guest root on `/tmp`
   (build context is **bind-mounted**, not copied, to respect `/tmp` quotas;
   leftovers under `/tmp/slime-kaniko*` are cleaned at start/end of each run)

Local tarball instead of push: `--no-push` → `<context>/kaniko_out/<instance_id>.tar`.

## Build + push (Tmax)

Tmax does **not** restore `/workspace` or run `pre_commands`. Verifier scripts stay out of the image (`test_sh` remains in jsonl only).

```bash
# multi-agent smoke (4 images from mixed_reward1)
INPUT=examples/coding_agent_rl/data/mixed_agents_bake_smoke_tmax.jsonl \
OUTPUT=examples/coding_agent_rl/data/mixed_agents_bake_smoke_tmax_baked.jsonl \
  bash examples/coding_agent_rl/docker_build/bake_tmax_smoke.sh

# or direct
python3 examples/coding_agent_rl/docker_build/bake_tmax_agent_images.py \
  --input examples/coding_agent_rl/data/mixed_agents_bake_smoke_tmax.jsonl \
  --output examples/coding_agent_rl/data/mixed_agents_bake_smoke_tmax_baked.jsonl \
  --workers 1
```

## Multi-agent bake smoke (both protocols)

```bash
# ScaleSWE 4 rows (raw aweaiteam/scaleswe) → baked scaleswe-agent
python3 examples/coding_agent_rl/docker_build/bake_scaleswe_agent_images.py \
  --input examples/coding_agent_rl/data/mixed_agents_bake_smoke_scaleswe.jsonl \
  --output examples/coding_agent_rl/data/mixed_agents_bake_smoke_scaleswe_baked.jsonl \
  --workers 1

python3 examples/coding_agent_rl/docker_build/verify_scaleswe_agent_images.py \
  --input examples/coding_agent_rl/data/mixed_agents_bake_smoke_scaleswe_baked.jsonl

# Tmax 4 rows → baked tmax-agent
bash examples/coding_agent_rl/docker_build/bake_tmax_smoke.sh
```

Then infer with the baked jsonl via `run_infer_cc_offload_traj.sh` / `run_infer_cc_tmax_traj.sh`.

## Verify baked images

### ScaleSWE

Uses `crane export` (no `docker run`). Checks `/etc/passwd`, rejects `deepswe/` /
`scaleswe/` / leftover `/tmp_build`, and if `/tmp` is present requires mode
`1777` with no bake tarballs under it (missing `/tmp` is allowed). Asserts
pre-baked CLIs under `/usr/local/bin` (`node`, `npm`, `claude`, `opencode`, `pi`,
`mini`). Asserts `metadata.workdir` is on branch `scaleswe`. Porcelain noise matching ScaleSWE
`pre_commands` keep-list (`*.egg-info`, `.tox`, `.venv`) and generated
`version.py` is ignored. Temp dirs under `/tmp/slime-verify-scaleswe` are
cleaned after each image and on exit.

```bash
# smoke one
python3 examples/coding_agent_rl/docker_build/verify_scaleswe_agent_images.py --limit 1
# all unique images from baked JSONL (default workers=4)
python3 examples/coding_agent_rl/docker_build/verify_scaleswe_agent_images.py
# failures → examples/coding_agent_rl/docker_build/verify_failures.jsonl
```

### Tmax

```bash
python3 examples/coding_agent_rl/docker_build/verify_tmax_agent_images.py \
  --input examples/coding_agent_rl/data/mixed_agents_bake_smoke_tmax_baked.jsonl
```

Checks `/home/user`, four agent CLIs, no `/tmp_build`, no `/tests`.
