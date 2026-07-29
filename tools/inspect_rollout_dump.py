#!/usr/bin/env python3
"""Inspect slime rollout debug dump ``.pt`` files (esp. coding-agent / offload runs).

Examples::

    # Summary of one dump
    python tools/inspect_rollout_dump.py runs/.../rollout_0.pt

    # List trajectories (grouped fork segments)
    python tools/inspect_rollout_dump.py runs/.../rollout_0.pt list --offload --unsolved

    # Show one trajectory (all fork segments)
    python tools/inspect_rollout_dump.py runs/.../rollout_0.pt show \\
        --label asottile_add-trailing-comma_pr13 --group 9 --index 77

    # Decode a segment response (needs --tokenizer)
    python tools/inspect_rollout_dump.py runs/.../rollout_1.pt show \\
        --label asottile_add-trailing-comma_pr13 -g 9 -i 77 --seg 1 \\
        --tokenizer /path/to/hf --decode-response --max-chars 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

OFFLOAD_CLOSE_TOKEN_ID_DEFAULT = 248078


def _load_pt(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "samples" in obj:
        return obj
    if isinstance(obj, list):
        return {"rollout_id": None, "samples": obj}
    raise SystemExit(f"Unrecognized dump format: {type(obj)} keys={getattr(obj, 'keys', lambda: None)()}")


def _sample_dict(sample: Any) -> dict[str, Any]:
    if hasattr(sample, "to_dict"):
        sample = sample.to_dict()
    if not isinstance(sample, dict):
        raise TypeError(f"sample is not a dict: {type(sample)}")
    return sample


def _meta(s: dict[str, Any]) -> dict[str, Any]:
    m = s.get("metadata") or {}
    return m if isinstance(m, dict) else {}


def _offload_stats(s: dict[str, Any]) -> dict[str, Any]:
    st = _meta(s).get("offload_stats") or {}
    return st if isinstance(st, dict) else {}


def _traj_key(s: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (s.get("label"), s.get("group_index"), s.get("index"))


def _group_trajs(samples: list[dict[str, Any]]) -> dict[tuple[Any, Any, Any], list[dict[str, Any]]]:
    trajs: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        trajs[_traj_key(s)].append(s)
    for segs in trajs.values():
        segs.sort(key=lambda x: len(x.get("tokens") or []) - int(x.get("response_length") or 0))
    return trajs


def _reward(segs: list[dict[str, Any]]) -> Any:
    for s in reversed(segs):
        if s.get("reward") is not None:
            return s.get("reward")
    return None


def _is_solved(segs: list[dict[str, Any]]) -> bool | None:
    meta = _meta(segs[0])
    if "grading_solved" in meta:
        return bool(meta["grading_solved"])
    if "solved" in meta:
        v = meta["solved"]
        if isinstance(v, (int, float)):
            return bool(v > 0)
        return bool(v)
    r = _reward(segs)
    if isinstance(r, (int, float)):
        return r > 0
    return None


def _fmt_num(x: Any) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def cmd_summary(args: argparse.Namespace) -> None:
    data = _load_pt(args.pt_path)
    samples = [_sample_dict(s) for s in data["samples"]]
    trajs = _group_trajs(samples)

    n_trace = sum(1 for s in samples if isinstance(s.get("trace"), dict))
    rewards = [_reward(segs) for segs in trajs.values()]
    solved_flags = [_is_solved(segs) for segs in trajs.values()]
    n_solved = sum(1 for x in solved_flags if x is True)
    n_unsolved = sum(1 for x in solved_flags if x is False)
    n_unknown = sum(1 for x in solved_flags if x is None)

    offload_trajs = 0
    offload_counts: list[int] = []
    empty_patch = 0
    labels: Counter[str] = Counter()
    resp_lens: list[int] = []
    seg_counts: list[int] = []

    for key, segs in trajs.items():
        labels[str(key[0])] += 1
        seg_counts.append(len(segs))
        for s in segs:
            resp_lens.append(int(s.get("response_length") or 0))
        st = _offload_stats(segs[0])
        oc = int(st.get("offload_count") or 0)
        if oc > 0:
            offload_trajs += 1
            offload_counts.append(oc)
        if _meta(segs[0]).get("empty_patch"):
            empty_patch += 1

    print(f"pt:           {args.pt_path}")
    print(f"rollout_id:   {data.get('rollout_id')}")
    print(f"samples:      {len(samples)}  (fork segments)")
    print(f"trajectories: {len(trajs)}")
    print(f"with_trace:   {n_trace}/{len(samples)}  (timeline viewer needs this)")
    print(f"solved:       {n_solved}  unsolved: {n_unsolved}  unknown: {n_unknown}")
    print(f"empty_patch:  {empty_patch}/{len(trajs)}")
    print(f"offload traj: {offload_trajs}/{len(trajs)}")
    if offload_counts:
        print(
            f"offload_count: min={min(offload_counts)} median={sorted(offload_counts)[len(offload_counts)//2]} "
            f"max={max(offload_counts)} sum={sum(offload_counts)}"
        )
    if resp_lens:
        print(
            f"response_len: min={min(resp_lens)} median={sorted(resp_lens)[len(resp_lens)//2]} "
            f"max={max(resp_lens)} mean={sum(resp_lens)/len(resp_lens):.1f}"
        )
    if seg_counts:
        print(
            f"segs/traj:    min={min(seg_counts)} median={sorted(seg_counts)[len(seg_counts)//2]} "
            f"max={max(seg_counts)} mean={sum(seg_counts)/len(seg_counts):.1f}"
        )

    rew_vals = [r for r in rewards if isinstance(r, (int, float))]
    if rew_vals:
        print(f"reward:       min={min(rew_vals):.4g} mean={sum(rew_vals)/len(rew_vals):.4g} max={max(rew_vals):.4g}")

    print("\nTop labels:")
    for lab, c in labels.most_common(15):
        print(f"  {c:4d}  {lab}")

    # sample keys
    keys = sorted(samples[0].keys()) if samples else []
    print(f"\nsample keys ({len(keys)}): {', '.join(keys)}")
    meta_keys = sorted(_meta(samples[0]).keys()) if samples else []
    print(f"metadata keys: {', '.join(meta_keys) if meta_keys else '(none)'}")


def cmd_list(args: argparse.Namespace) -> None:
    data = _load_pt(args.pt_path)
    samples = [_sample_dict(s) for s in data["samples"]]
    trajs = _group_trajs(samples)

    rows = []
    for (label, g, i), segs in trajs.items():
        st = _offload_stats(segs[0])
        oc = int(st.get("offload_count") or 0)
        solved = _is_solved(segs)
        meta = _meta(segs[0])
        if args.offload and oc < 1:
            continue
        if args.unsolved and solved is not False and not (
            isinstance(_reward(segs), (int, float)) and _reward(segs) == 0
        ):
            # treat reward==0 as unsolved too
            if solved is True:
                continue
            if solved is None and not (isinstance(_reward(segs), (int, float)) and float(_reward(segs)) <= 0):
                continue
        if args.solved and solved is not True:
            continue
        if args.label and args.label not in str(label):
            continue
        if args.min_offload and oc < args.min_offload:
            continue
        rows.append(
            {
                "label": label,
                "group": g,
                "index": i,
                "n_seg": len(segs),
                "reward": _reward(segs),
                "solved": solved,
                "empty_patch": meta.get("empty_patch"),
                "offload_count": oc,
                "glm_out": st.get("glm_output_tokens"),
                "small_out": st.get("small_output_tokens"),
                "resp_sum": sum(int(s.get("response_length") or 0) for s in segs),
            }
        )

    rows.sort(key=lambda r: ( -(r["offload_count"] or 0), -(r["n_seg"] or 0), str(r["label"])))
    if args.limit:
        rows = rows[: args.limit]

    print(
        f"{'label':<48} {'g':>3} {'i':>4} {'segs':>4} {'rew':>7} {'sol':>4} "
        f"{'empty':>5} {'oc':>3} {'glm':>6} {'respΣ':>7}"
    )
    for r in rows:
        print(
            f"{str(r['label'])[:48]:<48} {r['group']!s:>3} {r['index']!s:>4} {r['n_seg']:>4} "
            f"{_fmt_num(r['reward']):>7} {str(r['solved']):>4} {str(r['empty_patch']):>5} "
            f"{r['offload_count']:>3} {_fmt_num(r['glm_out']):>6} {r['resp_sum']:>7}"
        )
    print(f"\nlisted {len(rows)} trajectories")


def _load_tokenizer(path: str | None):
    if not path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _decode(tok, ids: list[int], max_chars: int | None = None) -> str:
    text = tok.decode(ids, skip_special_tokens=False)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + f"\n...<truncated {len(text) - max_chars} chars>"
    return text


def _count_offload_close(resp_ids: list[int], close_id: int) -> int:
    return sum(1 for t in resp_ids if t == close_id)


def cmd_show(args: argparse.Namespace) -> None:
    data = _load_pt(args.pt_path)
    samples = [_sample_dict(s) for s in data["samples"]]
    trajs = _group_trajs(samples)

    # resolve key
    matches = []
    for key, segs in trajs.items():
        label, g, i = key
        if args.label and args.label not in str(label):
            continue
        if args.group is not None and g != args.group:
            continue
        if args.index is not None and i != args.index:
            continue
        matches.append((key, segs))

    if not matches:
        raise SystemExit("No trajectory matched filters (--label/--group/--index).")
    if len(matches) > 1 and (args.group is None or args.index is None):
        print(f"Multiple matches ({len(matches)}); refine with --group/--index:\n")
        for (label, g, i), segs in matches[:30]:
            st = _offload_stats(segs[0])
            print(f"  {label}  g={g} i={i} segs={len(segs)} oc={st.get('offload_count', 0)}")
        if len(matches) > 30:
            print(f"  ... +{len(matches) - 30} more")
        raise SystemExit(1)

    (label, g, i), segs = matches[0]
    meta = _meta(segs[0])
    st = _offload_stats(segs[0])
    tok = _load_tokenizer(args.tokenizer)
    close_id = args.offload_close_id

    print(f"label:        {label}")
    print(f"group/index:  {g} / {i}")
    print(f"n_segments:   {len(segs)}")
    print(f"reward:       {_reward(segs)}")
    print(f"solved:       {_is_solved(segs)}  grading_solved={meta.get('grading_solved')}  empty_patch={meta.get('empty_patch')}")
    print(f"offload_stats:{json.dumps(st, ensure_ascii=False)}")
    print(f"metadata:     {json.dumps({k: meta[k] for k in meta if k != 'offload_stats'}, ensure_ascii=False, default=str)[:800]}")
    print()

    for si, s in enumerate(segs):
        if args.seg is not None and si != args.seg:
            continue
        tokens = list(s.get("tokens") or [])
        resp_len = int(s.get("response_length") or 0)
        prompt_len = len(tokens) - resp_len
        resp_ids = tokens[-resp_len:] if resp_len else []
        mask = list(s.get("loss_mask") or [])
        n1 = sum(mask)
        n0 = len(mask) - n1
        n_close = _count_offload_close(resp_ids, close_id)
        after_close = 0
        mask0_after = 0
        if n_close and close_id in resp_ids:
            pos = len(resp_ids) - 1 - resp_ids[::-1].index(close_id)
            after_ids = resp_ids[pos + 1 :]
            after_close = len(after_ids)
            if mask and pos + 1 < len(mask):
                mask0_after = mask[pos + 1 :].count(0)

        print(
            f"--- seg {si:02d}  tokens={len(tokens)} prompt={prompt_len} resp={resp_len} "
            f"loss1={n1} loss0={n0} offload_close={n_close} after_close={after_close} "
            f"mask0_after={mask0_after} reward={s.get('reward')} status={s.get('status')}"
        )

        if args.decode_response and tok is not None:
            text = _decode(tok, resp_ids, args.max_chars)
            print(text)
            print()
        elif args.decode_response and tok is None:
            print("  (pass --tokenizer to decode)")

        if args.decode_full and tok is not None:
            text = _decode(tok, tokens, args.max_chars)
            print(text)
            print()

        if args.json_meta:
            print(json.dumps({"segment": si, "keys": sorted(s.keys()), "metadata": _meta(s)}, ensure_ascii=False, default=str)[:2000])


def cmd_keys(args: argparse.Namespace) -> None:
    data = _load_pt(args.pt_path)
    samples = [_sample_dict(s) for s in data["samples"]]
    if not samples:
        print("empty dump")
        return
    s = samples[args.i]
    print(f"sample[{args.i}] type fields:")
    for k, v in s.items():
        if isinstance(v, list):
            print(f"  {k}: list len={len(v)} elem={type(v[0]).__name__ if v else '?'}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())[:20]}")
        else:
            print(f"  {k}: {type(v).__name__} = {_fmt_num(v) if isinstance(v, (int, float)) else repr(v)[:80]}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pt_path", type=Path, help="Path to rollout_*.pt")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("summary", help="Dump-level summary (default)")
    sp.set_defaults(func=cmd_summary)

    lp = sub.add_parser("list", help="List trajectories")
    lp.add_argument("--offload", action="store_true", help="Only trajs with offload_count>=1")
    lp.add_argument("--min-offload", type=int, default=0)
    lp.add_argument("--unsolved", action="store_true")
    lp.add_argument("--solved", action="store_true")
    lp.add_argument("--label", type=str, default=None, help="Substring match on label")
    lp.add_argument("--limit", type=int, default=50)
    lp.set_defaults(func=cmd_list)

    shp = sub.add_parser("show", help="Show one trajectory / segment")
    shp.add_argument("--label", "-l", type=str, required=True, help="Exact or substring label")
    shp.add_argument("--group", "-g", type=int, default=None)
    shp.add_argument("--index", "-i", type=int, default=None)
    shp.add_argument("--seg", type=int, default=None, help="Only print this segment id")
    shp.add_argument("--tokenizer", "-t", type=str, default=None, help="HF tokenizer path for decode")
    shp.add_argument("--decode-response", action="store_true")
    shp.add_argument("--decode-full", action="store_true")
    shp.add_argument("--max-chars", type=int, default=4000)
    shp.add_argument("--offload-close-id", type=int, default=OFFLOAD_CLOSE_TOKEN_ID_DEFAULT)
    shp.add_argument("--json-meta", action="store_true")
    shp.set_defaults(func=cmd_show)

    kp = sub.add_parser("keys", help="Print field types for sample[i]")
    kp.add_argument("-i", type=int, default=0)
    kp.set_defaults(func=cmd_keys)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        args.func = cmd_summary
    args.func(args)


if __name__ == "__main__":
    main()
