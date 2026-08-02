"""Extract every system prompt found in coding-agent rollout ``.pt`` dumps.

The dumps store each trajectory as a flat token sequence (``sample["tokens"]``),
so the system prompt only exists as the ``<|im_start|>system ... <|im_end|>``
block(s) at the head of that sequence. This walks the dumps, slices those blocks
at the token level (cheap: no full-trajectory decode), decodes them, and writes
one JSONL line per distinct prompt with where it was seen.

slime's own ``<|llm_offload|>`` instructions, which the coding adapter appends to
every request, are cut out by default (``--keep-offload-append`` to retain them):
they are training-harness plumbing, not part of the agent's system prompt.

Two dedup levels (``--dedup``):

``exact``
    Group by verbatim text. Every per-sample detail (cwd, git status, kernel
    version) splits a group, so this stays close to the raw corpus.
``template``
    Mask the per-sample details first (see ``NORMALIZERS``) and group by the
    resulting template. This is what collapses "same prompt, different repo"
    down to the handful of prompt revisions actually in play.

Example::

    python tools/extract_system_prompts.py \\
        --dumps 'runs/agent_offload_pyrodash4b_docker_*/rollout_dumps' \\
        --tokenizer /workspace/models/pyromind/PyroDash-4B-SFT-0728_pad248320 \\
        --dedup template --out system_prompt.jsonl

    # Re-dedup an existing extract without re-reading 9 GB of dumps:
    python tools/extract_system_prompts.py --from-jsonl system_prompt.exact.jsonl \\
        --dedup template --out system_prompt.jsonl
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import multiprocessing as mp
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
from transformers import AutoTokenizer

logger = logging.getLogger("extract_system_prompts")

DEFAULT_TOKENIZER = "/workspace/models/pyromind/PyroDash-4B-SFT-0728_pad248320"
DEFAULT_DUMPS = "runs/agent_offload_pyrodash4b_docker_*/rollout_dumps"

# slime's own offload instructions, appended by the coding adapter (see
# examples/coding_agent_rl/offload.py). Not part of the agent's system prompt, so
# they are cut before anything else. One line each, three wordings in the wild:
# two appended at the very end, one injected ahead of gitStatus.
OFFLOAD_APPEND = re.compile(r"\n*For very difficult steps, you can output <\|llm_offload\|>N<\|/llm_offload\|>[^\n]*")


def strip_offload_append(text: str) -> str:
    """Remove the adapter-injected ``<|llm_offload|>`` instructions."""
    return OFFLOAD_APPEND.sub("", text)


# Per-sample noise that Claude Code bakes into the system prompt. Masking these
# leaves the prompt revision itself, which is what "distinct prompt" should mean.
# Deliberately NOT masked: model identity, knowledge cutoff, the tool list --
# those are real differences between prompt revisions.
NORMALIZERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gitStatus: This is the git status.*?(?=\n\nFor very difficult steps|\Z)", re.S), "<GIT_STATUS>"),
    (re.compile(r"^(\s*-?\s*)Working directory: .*$", re.M), r"\1Working directory: <CWD>"),
    (re.compile(r"/workspace/[A-Za-z0-9_.-]+"), "/workspace/<REPO>"),
    (re.compile(r"-workspace-[A-Za-z0-9_.-]+"), "-workspace-<REPO>"),
    (re.compile(r"\.claude/worktrees/agent-[0-9a-f]+"), ".claude/worktrees/agent-<ID>"),
    (re.compile(r"Linux \d+\.\d+\.\d+-\d+-generic"), "Linux <KERNEL>"),
    (re.compile(r"Today's date: [^\n]*"), "Today's date: <DATE>"),
]


def normalize_prompt(text: str) -> str:
    """Mask per-sample details so identical prompt revisions hash alike."""
    for pattern, replacement in NORMALIZERS:
        text = pattern.sub(replacement, text)
    return text


# Resolved once per worker from the tokenizer, not hardcoded: a future
# checkpoint could renumber the chat-template specials.
_TOK: Any = None
_IM_START: int = -1
_IM_END: int = -1
_SYSTEM: int = -1
_STRIP_OFFLOAD: bool = True


@dataclass
class PromptRecord:
    """One distinct system prompt plus the places it showed up."""

    text: str
    num_tokens: int
    count: int = 0
    runs: set[str] = field(default_factory=set)
    dumps: set[str] = field(default_factory=set)
    instance_ids: set[str] = field(default_factory=set)
    first_seen: dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "PromptRecord") -> None:
        self.count += other.count
        self.runs |= other.runs
        self.dumps |= other.dumps
        self.instance_ids |= other.instance_ids
        if not self.first_seen:
            self.first_seen = other.first_seen

    def to_json(self, sha: str) -> dict[str, Any]:
        return {
            "sha256": sha,
            "num_chars": len(self.text),
            "num_tokens": self.num_tokens,
            "count": self.count,
            "num_runs": len(self.runs),
            "num_dumps": len(self.dumps),
            "num_instances": len(self.instance_ids),
            "runs": sorted(self.runs),
            "instance_ids": sorted(self.instance_ids),
            "first_seen": self.first_seen,
            "text": self.text,
        }

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> "PromptRecord":
        return cls(
            text=doc["text"],
            num_tokens=int(doc.get("num_tokens", 0)),
            count=int(doc.get("count", 1)),
            runs=set(doc.get("runs") or []),
            dumps=set(doc.get("dumps") or []),
            instance_ids=set(doc.get("instance_ids") or []),
            first_seen=doc.get("first_seen") or {},
        )


def group_by_template(records: dict[str, PromptRecord]) -> list[dict[str, Any]]:
    """Collapse exact-dedup records into one entry per normalized template.

    The representative ``text`` is the most frequent variant, so the output stays
    a real prompt you can read, with ``normalized_text`` showing what was masked.
    """
    clusters: dict[str, list[tuple[str, PromptRecord]]] = {}
    for sha, rec in records.items():
        key = hashlib.sha256(normalize_prompt(rec.text).encode("utf-8")).hexdigest()
        clusters.setdefault(key, []).append((sha, rec))

    docs: list[dict[str, Any]] = []
    for key, members in clusters.items():
        members.sort(key=lambda kv: (-kv[1].count, kv[0]))
        rep_sha, rep = members[0]
        merged = PromptRecord(text=rep.text, num_tokens=rep.num_tokens, first_seen=rep.first_seen)
        for _, rec in members:
            merged.merge(rec)
        doc = merged.to_json(key)
        doc |= {
            "dedup": "template",
            "num_variants": len(members),
            "representative_sha256": rep_sha,
            "variant_sha256": [sha for sha, _ in members],
            "normalized_text": normalize_prompt(rep.text),
        }
        docs.append(doc)
    return sorted(docs, key=lambda d: (-d["count"], d["sha256"]))


def _init_worker(tokenizer_path: str, strip_offload: bool) -> None:
    global _TOK, _IM_START, _IM_END, _SYSTEM, _STRIP_OFFLOAD
    _STRIP_OFFLOAD = strip_offload
    _TOK = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    (_IM_START,) = _TOK.encode("<|im_start|>", add_special_tokens=False)
    (_IM_END,) = _TOK.encode("<|im_end|>", add_special_tokens=False)
    (_SYSTEM,) = _TOK.encode("system", add_special_tokens=False)


def iter_system_blocks(tokens: list[int]) -> Iterator[list[int]]:
    """Yield the token slice of every ``<|im_start|>system ... <|im_end|>`` block.

    The leading role marker (``system\\n``) and the closing ``<|im_end|>`` are
    excluded, so the slice decodes to the prompt body alone.
    """
    n = len(tokens)
    i = 0
    while i < n - 2:
        if tokens[i] == _IM_START and tokens[i + 1] == _SYSTEM:
            j = i + 2
            while j < n and tokens[j] != _IM_END:
                j += 1
            yield tokens[i + 2 : j]
            i = j
        i += 1


def _run_name(dump_path: Path) -> str:
    # .../<run>/rollout_dumps/rollout_3.pt -> <run>
    return dump_path.parent.parent.name


def scan_dump(path_str: str) -> tuple[str, dict[str, PromptRecord], int, str]:
    """Collect distinct system prompts from one dump file.

    Returns ``(path, {sha256: record}, num_samples, error)``; ``error`` is an
    empty string on success. Never raises, so one corrupt dump cannot kill the
    whole pass.
    """
    path = Path(path_str)
    out: dict[str, PromptRecord] = {}
    try:
        payload = torch.load(path, weights_only=False, map_location="cpu")
    except Exception as exc:
        return path_str, out, 0, f"{type(exc).__name__}: {exc}"

    samples = payload.get("samples") or []
    run = _run_name(path)
    # Identical token slices decode identically; cache so a 200-sample dump
    # decodes ~8 prompts instead of 200.
    decoded: dict[bytes, tuple[str, str, int]] = {}
    for sample in samples:
        tokens = sample.get("tokens")
        if tokens is None:
            continue
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        label = sample.get("label") or (sample.get("metadata") or {}).get("instance_id") or ""
        for block in iter_system_blocks(list(tokens)):
            if not block:
                continue
            key = repr(block).encode()
            hit = decoded.get(key)
            if hit is None:
                text = _TOK.decode(block, skip_special_tokens=False)
                num_tokens = len(block)
                if _STRIP_OFFLOAD:
                    stripped = strip_offload_append(text)
                    if stripped != text:
                        # Re-encode so num_tokens still describes the stored text.
                        text = stripped
                        num_tokens = len(_TOK.encode(text, add_special_tokens=False))
                hit = (hashlib.sha256(text.encode("utf-8")).hexdigest(), text, num_tokens)
                decoded[key] = hit
            sha, text, num_tokens = hit
            rec = out.get(sha)
            if rec is None:
                rec = out[sha] = PromptRecord(
                    text=text,
                    num_tokens=num_tokens,
                    first_seen={
                        "run": run,
                        "dump": path.name,
                        "rollout_id": payload.get("rollout_id"),
                        "index": sample.get("index"),
                        "instance_id": label,
                    },
                )
            rec.count += 1
            rec.runs.add(run)
            rec.dumps.add(f"{run}/{path.name}")
            if label:
                rec.instance_ids.add(label)
    return path_str, out, len(samples), ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dumps",
        action="append",
        dest="dump_globs",
        metavar="GLOB",
        help=f"glob of dump dirs or .pt files, repeatable (default: {DEFAULT_DUMPS})",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--out", type=Path, default=Path("system_prompt.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dedup",
        choices=("exact", "template"),
        default="exact",
        help="exact = verbatim text; template = mask per-sample details first",
    )
    parser.add_argument(
        "--from-jsonl",
        type=Path,
        help="re-dedup an existing extract instead of scanning dumps",
    )
    parser.add_argument(
        "--keep-offload-append",
        action="store_true",
        help="keep slime's injected <|llm_offload|> instructions (stripped by default)",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="omit the prompt body (index-only output, tiny file)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    merged: dict[str, PromptRecord] = {}
    total_samples = 0
    errors: list[tuple[str, str]] = []

    if args.from_jsonl:
        restripped = 0
        with args.from_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                doc = json.loads(line)
                rec = PromptRecord.from_json(doc)
                if not args.keep_offload_append:
                    text = strip_offload_append(rec.text)
                    if text != rec.text:
                        rec.text = text
                        restripped += 1
                sha = hashlib.sha256(rec.text.encode("utf-8")).hexdigest()
                if sha in merged:
                    merged[sha].merge(rec)
                else:
                    merged[sha] = rec
        logger.info("loaded %d records from %s", len(merged), args.from_jsonl)
        if restripped:
            logger.warning(
                "stripped the offload append from %d loaded records; their num_tokens still "
                "counts the pre-strip block (re-run from dumps for exact counts)",
                restripped,
            )
    else:
        globs = args.dump_globs or [DEFAULT_DUMPS]
        files: list[str] = []
        for pattern in globs:
            for hit in sorted(glob.glob(pattern)):
                p = Path(hit)
                files.extend(sorted(str(f) for f in p.glob("*.pt")) if p.is_dir() else [str(p)])
        files = sorted(set(files))
        if not files:
            raise SystemExit(f"no .pt dumps matched {globs}")
        logger.info("scanning %d dump files with %d workers", len(files), args.workers)

        ctx = mp.get_context("spawn")
        init_args = (args.tokenizer, not args.keep_offload_append)
        with ctx.Pool(args.workers, initializer=_init_worker, initargs=init_args) as pool:
            for done, (path, found, n_samples, err) in enumerate(pool.imap_unordered(scan_dump, files), start=1):
                if err:
                    errors.append((path, err))
                total_samples += n_samples
                for sha, rec in found.items():
                    if sha in merged:
                        merged[sha].merge(rec)
                    else:
                        merged[sha] = rec
                if done % 20 == 0 or done == len(files):
                    logger.info(
                        "  %d/%d files, %d samples, %d distinct prompts", done, len(files), total_samples, len(merged)
                    )

    if args.dedup == "template":
        docs = group_by_template(merged)
    else:
        docs = [rec.to_json(sha) for sha, rec in sorted(merged.items(), key=lambda kv: (-kv[1].count, kv[0]))]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for doc in docs:
            if args.no_text:
                doc.pop("text", None)
                doc.pop("normalized_text", None)
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    occurrences = sum(r.count for r in merged.values())
    logger.info(
        "wrote %d %s-deduped prompts (from %d exact texts, %d occurrences) -> %s (%.1f MB)",
        len(docs),
        args.dedup,
        len(merged),
        occurrences,
        args.out,
        args.out.stat().st_size / 1e6,
    )
    for path, err in errors:
        logger.warning("failed: %s (%s)", path, err)


if __name__ == "__main__":
    main()
