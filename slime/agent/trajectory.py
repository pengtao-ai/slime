"""Build a per-session training trajectory from multi-turn conversation data.

The :class:`TrajectoryManager` builds one trajectory per session. ``record_turn``
feeds in each turn (prompt messages + the served model's sglang snapshot),
routing it into a per-sid message tree; ``get_trajectory`` then linearizes that
tree into a ``list[Sample]`` of loss-masked training rows, tolerating TITO
re-tokenization drift via fork/replace.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import os
from collections.abc import Iterator
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _gigpo_per_turn_enabled() -> bool:
    return os.environ.get("SLIME_GIGPO", "").strip().lower() in ("1", "true", "yes", "on")


# ===========================================================================
# TurnRecord
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class TurnRecord:
    """One sglang ``/generate`` snapshot: the contract between an adapter and the
    manager. Adapters build it from a turn's prompt/output token ids; ``record_turn``
    consumes it.

    ``output_loss_mask`` is optional and aligned 1:1 with ``output_ids`` when
    non-empty. Use it to keep non-trainable suffixes (e.g. mid-turn GLM offload
    text) inside the trajectory while excluding them from the loss. An empty
    list means "all output tokens are trainable" (legacy default).
    """

    prompt_ids: list[int]
    output_ids: list[int]
    finish_reason: str
    output_log_probs: list[float] = dataclasses.field(default_factory=list)
    ill_formed: bool = False
    output_loss_mask: list[int] = dataclasses.field(default_factory=list)


# ===========================================================================
# MessageNode
# ===========================================================================


class MessageNode:
    """One node in a session's routing tree, carrying a single chat message
    (``None`` for the dummy root and for an assistant leaf we generated but
    whose ``response_message`` was empty).

    The two kinds are distinguished by whether ``turn`` is set, which reflects
    WHERE the message came from:

    * **generated** (``turn is not None``): an assistant message the model
      actually generated this turn, fed in via ``record_turn``. ``turn`` holds
      its :class:`TurnRecord` -- the prompt/output ids, logprobs and finish
      reason that ``get_trajectory`` linearizes into training tokens.
    * **routing-only** (``turn is None``): the message came from the prompt, not
      from generation, so it only exists to route. This is every
      system/user/tool node, AND any assistant we did NOT generate: a foreign
      assistant the client replayed in a later prompt, or a prior generated turn
      demoted by the rewrite-merge in ``_try_merge_assistant_rewrite``.
    """

    def __init__(
        self,
        *,
        role: str | None = None,
        message: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: MessageNode | None = None,
    ) -> None:
        self.role = role
        self.message = message
        self.metadata = dict(metadata or {})
        self.parent: MessageNode | None = parent
        self.children: list[MessageNode] = []
        self.turn: TurnRecord | None = None  # the generated TurnRecord, else None (routing-only)
        self.turn_index: int | None = None
        # Shared by sibling leaf paths; the first to reach it trains on it, the rest
        # re-emit it as loss_mask=0 context -- so each response is trained exactly once.
        self.response_trained: bool = False

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, child: MessageNode) -> MessageNode:
        child.parent = self
        self.children.append(child)
        return child

    def path_from_root(self) -> list[MessageNode]:
        """Ordered list of nodes from the first non-root ancestor down to self."""
        chain: list[MessageNode] = []
        node: MessageNode | None = self
        while node is not None and not node.is_root:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def leaves(self) -> Iterator[MessageNode]:
        if not self.children:
            yield self
            return
        for c in self.children:
            yield from c.leaves()


# ===========================================================================
# drift classification — how an incoming turn's prompt relates to held tokens
# ===========================================================================


def _common_prefix_len(a: list[int], b: list[int], chunk: int = 4096) -> int:
    limit = min(len(a), len(b))
    matched = 0
    while matched < limit:
        chunk_end = min(matched + chunk, limit)
        if a[matched:chunk_end] == b[matched:chunk_end]:
            matched = chunk_end
        else:
            while matched < chunk_end and a[matched] == b[matched]:
                matched += 1
            return matched
    return matched


class DriftKind(enum.Enum):
    CLEAN = "clean"  # drift == 0: prompt_ids exactly extends held tokens; append the tail beyond them
    REALIGN = "realign"  # drift inside the most-recent response span and short incoming response; replace that span (loss_mask=0)
    FORK = "fork"  # everything else: close this builder, open a fresh one as a fork


# ===========================================================================
# SampleBuilder — accumulates turns into one trainable Sample (fork closes it)
# ===========================================================================


class _SampleBuilder:
    """Accumulates a chain's turns into the token sequence of one ``Sample``.

    A chain of turns is appended one at a time via :meth:`append_turn`. Ideally
    each turn's prompt exactly extends the tokens we already hold, but a replayed
    turn rarely re-tokenizes byte-for-byte: TITO round-trips and chat-template
    re-rendering both perturb the ids of content we've already seen. The builder
    handles this drift in a source-agnostic way, classified by where and how far
    the prompt diverges from the held tokens (see :meth:`classify_token_drift`):

    * **CLEAN** -- no drift; append the prompt tail beyond what we hold.
    * **REALIGN** -- a short divergence inside the most-recent response span;
      overwrite that span from the prompt as loss_mask=0 and keep accumulating.
    * **FORK** -- divergence too large or too early to absorb; this builder is
      rejected and the caller closes it and opens a fresh one. That boundary is
      the "fork".

    Each surviving builder yields one Sample.
    """

    def __init__(self, fork_threshold: int) -> None:
        self._fork_threshold = fork_threshold
        self.tokens: list[int] = []
        self.loss_mask: list[int] = []
        self.logprobs: list[float] = []
        self.last_response_start_idx: int | None = None
        self.leading_prompt_len: int = 0

    def classify_token_drift(self, turn: TurnRecord) -> DriftKind:
        """Decide how this builder should absorb ``turn``'s prompt.

        The incoming turn's prompt is expected to match the tokens this builder
        already holds as an exact prefix. When token drift has occurred -- the
        prompt diverges from the held tokens -- we decide whether to REALIGN
        (heal a short divergence inside the most-recent response span) or to FORK
        (``len(turn.output_ids) >= fork_threshold``, or the divergence sits too
        early to absorb). With no drift the turn is handled the CLEAN way -- a
        plain prefix extension.
        """
        realign_at = _common_prefix_len(self.tokens, turn.prompt_ids)
        drift = len(self.tokens) - realign_at

        if drift == 0:
            return DriftKind.CLEAN

        # REALIGN only heals drift that falls inside the most-recent response span
        # (and is short); divergence anywhere earlier, or an empty builder, forks.
        start = self.last_response_start_idx
        if start is not None and realign_at >= start and len(turn.output_ids) < self._fork_threshold:
            held_resp_len = len(self.tokens) - start
            new_tail_len = len(turn.prompt_ids) - start
            # Mid-turn LLM offload (and similar) returns a longer assistant echo than
            # the local model actually generated. REALIGN would wipe those generated
            # tokens to loss_mask=0; FORK instead so they stay trainable on their own
            # sample while the continuation opens a new builder.
            expand_slack = max(32, held_resp_len)
            if new_tail_len > held_resp_len + expand_slack:
                return DriftKind.FORK
            return DriftKind.REALIGN
        return DriftKind.FORK

    def append_turn(self, turn: TurnRecord, kind: DriftKind, *, trained: bool = True) -> None:
        """Append one turn into this SampleBuilder, branching on ``kind``: for REALIGN
        we overwrite the already-saved response span, for CLEAN we just append this
        turn's prompt tail."""
        assert kind is not DriftKind.FORK, "append_turn called on a builder that would fork"

        is_first_turn = self.last_response_start_idx is None

        # --- append this turn's prompt tail (loss_mask=0) ---
        if kind is DriftKind.REALIGN:
            self._align_to_prompt(turn.prompt_ids)  # drop the drifted tail, re-append from prompt
        else:  # CLEAN: held tokens are an exact prefix of prompt_ids; append the tail beyond them
            self._append_tokens(turn.prompt_ids[len(self.tokens) :], loss_mask=0)

        # --- append this turn's generated response (loss_mask=1 unless re-emitted as context) ---
        self.last_response_start_idx = len(self.tokens)
        self._append_output_tokens(turn, trained=trained)

        if is_first_turn:
            self.leading_prompt_len = len(turn.prompt_ids)

    def _align_to_prompt(self, prompt_ids: list[int]) -> None:
        """Heal REALIGN drift by overwriting the most-recent response span with
        ``prompt_ids`` as loss_mask=0: the drifted tokens carry no signal, and re-appending
        from the prompt keeps the builder contiguous. Earlier turns are untouched."""
        response_start = self.last_response_start_idx
        tail = prompt_ids[response_start:]
        self.tokens[response_start:] = tail
        self.loss_mask[response_start:] = [0] * len(tail)
        self.logprobs[response_start:] = [0.0] * len(tail)

    def _append_output_tokens(self, turn: TurnRecord, *, trained: bool) -> None:
        """Append ``turn.output_ids`` with optional per-token ``output_loss_mask``."""
        ids = turn.output_ids
        n = len(ids)
        if not n:
            return
        per_token = turn.output_loss_mask
        if per_token and len(per_token) != n:
            raise ValueError(
                f"turn.output_loss_mask length {len(per_token)} != output_ids length {n}"
            )
        if not trained:
            self._append_tokens(ids, loss_mask=0, logprobs=[0.0] * n)
            return
        if not per_token:
            self._append_tokens(
                ids, loss_mask=1, logprobs=turn.output_log_probs if turn.output_log_probs else None
            )
            return
        masks = list(per_token)
        raw_lps = list(turn.output_log_probs or [])
        if len(raw_lps) == n:
            logprobs = [float(lp) if m else 0.0 for lp, m in zip(raw_lps, masks)]
        else:
            logprobs = [
                (float(raw_lps[i]) if i < len(raw_lps) and masks[i] else 0.0) for i in range(n)
            ]
        self._append_tokens(ids, loss_masks=masks, logprobs=logprobs)

    def _append_tokens(
        self,
        ids: list[int],
        *,
        loss_mask: int | None = None,
        loss_masks: list[int] | None = None,
        logprobs: list[float] | None = None,
    ) -> None:
        if loss_masks is not None:
            if len(loss_masks) != len(ids):
                raise ValueError(f"loss_masks length {len(loss_masks)} != ids length {len(ids)}")
            self.tokens.extend(ids)
            self.loss_mask.extend(loss_masks)
            self.logprobs.extend(logprobs if logprobs is not None else [0.0] * len(ids))
            return
        assert loss_mask is not None
        self.tokens.extend(ids)
        self.loss_mask.extend([loss_mask] * len(ids))
        self.logprobs.extend(logprobs if logprobs else [0.0] * len(ids))

    def has_trained_response(self) -> bool:
        return any(self.loss_mask[self.leading_prompt_len :])

    def to_sample(
        self, base_sample: Sample, extra_metadata: dict[str, Any] | None, max_sample_tokens: int = 0
    ) -> Sample:
        """Emit the accumulated tokens as one ``Sample``, stripping the first-turn
        prompt so loss_mask / logprobs cover only the response region."""
        start = self.leading_prompt_len  # first-turn prompt stripped; response region starts here
        tokens = list(self.tokens)
        loss_mask = self.loss_mask
        logprobs = self.logprobs
        if max_sample_tokens and len(tokens) > max_sample_tokens:
            tokens = tokens[:max_sample_tokens]
            loss_mask = loss_mask[:max_sample_tokens]
            logprobs = logprobs[:max_sample_tokens]
        md = dict(extra_metadata or {})
        return Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=base_sample.prompt,
            label=base_sample.label,
            tokens=tokens,
            response_length=len(loss_mask) - start,
            loss_mask=loss_mask[start:],
            rollout_log_probs=logprobs[start:],
            reward=0.0,
            status=Sample.Status.COMPLETED,
            metadata=md,
        )


# ===========================================================================
# TrajectoryManager
# ===========================================================================


class TrajectoryManager:
    def __init__(self, *, fork_threshold_tokens: int | None = None) -> None:
        self._fork_threshold: int = 1024 if fork_threshold_tokens is None else fork_threshold_tokens
        self._trees: dict[str, MessageNode] = {}
        self._turn_count: dict[str, int] = {}

    # -------------------- public ------------------------------------------

    def has_session(self, sid: str) -> bool:
        return sid in self._trees

    def turn_count(self, sid: str) -> int:
        return self._turn_count.get(sid, 0)

    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not prompt_messages:
            logger.warning("record_turn(sid=%s): empty prompt_messages; skipping", sid)
            return
        assert not turn.output_log_probs or len(turn.output_log_probs) == len(turn.output_ids), (
            f"turn.output_log_probs length {len(turn.output_log_probs)} != "
            f"turn.output_ids length {len(turn.output_ids)}"
        )
        assert not turn.output_loss_mask or len(turn.output_loss_mask) == len(turn.output_ids), (
            f"turn.output_loss_mask length {len(turn.output_loss_mask)} != "
            f"turn.output_ids length {len(turn.output_ids)}"
        )

        root = self._trees.setdefault(sid, MessageNode())

        node, depth = self._find_mount_point(root, prompt_messages)
        node, depth = self._try_merge_assistant_rewrite(sid, node, prompt_messages, depth)
        node = self._mount_prompt_messages(node, prompt_messages[depth:])
        self._attach_assistant_leaf(sid, node, turn=turn, response_message=response_message, metadata=metadata)

    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample: Sample,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
        max_sample_tokens: int = 0,
    ) -> list[Sample]:
        """Linearize this sid's routing tree into slime ``Sample`` objects and
        consume the session.

        Each routing leaf yields one or more Samples; ``reward`` is assigned in
        full to every emitted Sample (not split across them), so each trained
        turn carries the trajectory's outcome reward. The sid is dropped
        afterwards, so a second call for the same sid returns ``[]``.
        """
        root = self._trees.get(sid)
        if root is None:
            return []

        samples: list[Sample] = []
        for routing_leaf in root.leaves():
            if routing_leaf.is_root:
                continue
            chain = routing_leaf.path_from_root()
            samples.extend(
                self._chain_to_samples(
                    chain, base_sample=base_sample, extra_metadata=extra_metadata, max_sample_tokens=max_sample_tokens
                )
            )

        for s in samples:
            s.reward = reward
            if s.metadata is None:
                s.metadata = {}
            # Ensure episode_reward is visible for GiGPO step discounting.
            s.metadata.setdefault("episode_reward", float(reward))
            if s.metadata.get("gigpo") and s.metadata.get("branch_key") == "main" and s.metadata.get(
                "is_terminal_step"
            ):
                s.metadata["step_immediate_reward"] = float(reward)

        # If GiGPO per-turn emitted multiple main leaves, only the globally last
        # main terminal step should carry immediate reward.
        if _gigpo_per_turn_enabled() and samples:
            main_terms = [
                s
                for s in samples
                if (s.metadata or {}).get("gigpo") and (s.metadata or {}).get("branch_key") == "main"
            ]
            for s in main_terms:
                s.metadata["step_immediate_reward"] = 0.0
                s.metadata["is_terminal_step"] = False
            if main_terms:
                # Highest turn_index among main steps.
                last = max(main_terms, key=lambda x: int((x.metadata or {}).get("turn_index") or 0))
                last.metadata["step_immediate_reward"] = float(reward)
                last.metadata["is_terminal_step"] = True

        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)
        return samples

    def drop_session(self, sid: str) -> None:
        self._trees.pop(sid, None)
        self._turn_count.pop(sid, None)

    # -------------------- internals ----------------------------------------

    def _find_mount_point(self, root: MessageNode, messages: list[dict[str, Any]]) -> tuple[MessageNode, int]:
        """Walk down the tree matching each message by role and dict equality (==),
        returning the deepest node that still matches and where to mount the rest."""
        node = root
        depth = 0
        while depth < len(messages):
            msg = messages[depth]
            next_child = None
            for child in node.children:
                if child.role == msg.get("role") and child.message == msg:
                    next_child = child
                    break
            if next_child is None:
                break
            node = next_child
            depth += 1
        return node, depth

    def _try_merge_assistant_rewrite(
        self,
        sid: str,
        node: MessageNode,
        prompt_messages: list[dict[str, Any]],
        depth: int,
    ) -> tuple[MessageNode, int]:
        """Merge a short assistant-rewrite onto its node instead of forking.

        A harness may replay a prior assistant message slightly re-rendered (e.g.
        whitespace) in a later prompt. It no longer matches the node we generated,
        so it would fork -- stranding the original generated turn as a dead-end
        leaf that still emits its own training Sample. Instead we overwrite that
        node's message in place and stop training its generated content (demote to
        routing-only), so only the live branch trains. This only applies below
        ``fork_threshold``: a long abandoned response carries enough real signal
        to fork and train standalone.

        Forking is always safe (a rewrite mounts as routing-only); this is purely
        a cleanup. So we merge only when the mount point has exactly one assistant
        child that is a leaf, generated (``turn`` set), and short (response <
        ``fork_threshold``), and fork otherwise, since absorbing destroys a
        generated TurnRecord irreversibly.
        """
        if self._fork_threshold <= 0:
            return node, depth  # feature off
        if depth >= len(prompt_messages) or prompt_messages[depth].get("role") != "assistant":
            return node, depth  # genuine non-assistant history fork -> leave it

        asst_children = [c for c in node.children if c.role == "assistant"]
        if len(asst_children) != 1:
            if len(asst_children) > 1:
                logger.warning(
                    "record_turn(sid=%s turn=%s): %d assistant children at mount "
                    "point; can't tell which the rewrite targets, so forking.",
                    sid,
                    self._turn_count.get(sid, 0) + 1,
                    len(asst_children),
                )
            return node, depth

        rewritten_node = asst_children[0]
        if (
            rewritten_node.children
            or rewritten_node.turn is None
            or len(rewritten_node.turn.output_ids) >= self._fork_threshold
        ):
            return node, depth

        rewritten_node.metadata["merged_rewrite"] = {
            "abandoned_turn_index": rewritten_node.turn_index,
            "abandoned_response_tokens": len(rewritten_node.turn.output_ids),
        }
        rewritten_node.turn = None
        rewritten_node.turn_index = None
        rewritten_node.message = prompt_messages[depth]
        return rewritten_node, depth + 1

    def _mount_prompt_messages(
        self,
        node: MessageNode,
        remaining_messages: list[dict[str, Any]],
    ) -> MessageNode:
        for m in remaining_messages:
            node = node.add_child(MessageNode(role=m.get("role"), message=m))
        return node

    def _attach_assistant_leaf(
        self,
        sid: str,
        node: MessageNode,
        *,
        turn: TurnRecord,
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        asst = MessageNode(
            role="assistant",
            message=response_message,
            metadata=dict(metadata or {}),
        )
        asst.turn = turn
        asst.turn_index = self._turn_count.get(sid, 0) + 1
        node.add_child(asst)
        self._turn_count[sid] = asst.turn_index

    def _split_chain_into_builders(self, chain: list[MessageNode]) -> list[_SampleBuilder]:
        """Pack the chain's generated turns into per-Sample token builders.

        Turns flow into the current builder until one can't extend it as an
        exact prefix (re-tokenization drift past what we can drop); that turn
        opens a new builder -- a fork. A generated turn shared by sibling leaves
        is trained only on the first leaf to claim it; later leaves re-emit it
        as loss_mask=0 context so the shared prefix isn't double-counted.
        """
        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]

        builders: list[_SampleBuilder] = []
        for asst_node in asst_nodes:
            trained = not asst_node.response_trained
            asst_node.response_trained = True

            if not builders or (kind := builders[-1].classify_token_drift(asst_node.turn)) is DriftKind.FORK:
                builders.append(_SampleBuilder(self._fork_threshold))
                builders[-1].append_turn(asst_node.turn, DriftKind.CLEAN, trained=trained)
            else:
                builders[-1].append_turn(asst_node.turn, kind, trained=trained)
        return builders

    def _chain_to_samples(
        self,
        chain: list[MessageNode],
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
        max_sample_tokens: int = 0,
    ) -> list[Sample]:

        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]
        truncated = bool(asst_nodes) and asst_nodes[-1].turn.finish_reason == "length"
        use_tool = any(bool((n.message or {}).get("tool_calls")) for n in asst_nodes)
        ill_formed = any(n.turn.ill_formed for n in asst_nodes)
        md = {
            **(extra_metadata or {}),
            "truncated": truncated,
            "use_tool": use_tool,
            "ill_formed": ill_formed,
        }
        if _gigpo_per_turn_enabled():
            return self._chain_to_per_turn_samples(
                chain,
                base_sample=base_sample,
                extra_metadata=md,
                max_sample_tokens=max_sample_tokens,
            )
        return [
            builder.to_sample(base_sample, md, max_sample_tokens)
            for builder in self._split_chain_into_builders(chain)
            if builder.has_trained_response()
        ]

    def _chain_to_per_turn_samples(
        self,
        chain: list[MessageNode],
        *,
        base_sample: Sample,
        extra_metadata: dict[str, Any] | None,
        max_sample_tokens: int = 0,
    ) -> list[Sample]:
        """Emit one Sample per trained SLM turn (GiGPO step rows)."""
        # Lazy import: coding-agent example helpers (optional dependency path).
        try:
            from examples.coding_agent_rl.gigpo_anchor import (
                branch_key_from_user_task,
                extract_text_content,
                extract_tool_name_and_input,
                make_anchor_obs,
            )
        except ImportError:
            logger.warning("GiGPO per-turn requested but gigpo_anchor import failed; falling back to packed samples")
            return [
                builder.to_sample(base_sample, extra_metadata or {}, max_sample_tokens)
                for builder in self._split_chain_into_builders(chain)
                if builder.has_trained_response()
            ]

        instance_id = str((extra_metadata or {}).get("instance_id") or base_sample.label or "")
        problem_statement = str((extra_metadata or {}).get("problem_statement") or "")
        workdir = str((extra_metadata or {}).get("workdir") or "")
        episode_user = ""
        if isinstance(base_sample.prompt, list) and base_sample.prompt:
            episode_user = extract_text_content(base_sample.prompt[0] if isinstance(base_sample.prompt[0], dict) else {})
        elif isinstance(base_sample.prompt, str):
            episode_user = base_sample.prompt

        # First user task on this chain → branch_key.
        first_user = ""
        for n in chain:
            if n.role == "user" and n.message:
                first_user = extract_text_content(n.message)
                if first_user.strip():
                    break
        bkey = branch_key_from_user_task(first_user, episode_user=episode_user or problem_statement)
        base_rid = base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index
        traj_uid = f"{base_rid}:{bkey}"

        asst_nodes = [n for n in chain if n.role == "assistant" and n.turn is not None]
        samples: list[Sample] = []
        for asst_node in asst_nodes:
            trained = not asst_node.response_trained
            asst_node.response_trained = True
            if not trained:
                continue
            turn = asst_node.turn
            assert turn is not None
            if not turn.output_ids:
                continue

            # Anchor = env observation before this action (prev tool result), or init.
            prev_tool_name: str | None = None
            prev_tool_input: dict[str, Any] | None = None
            prev_tool_result: str | None = None
            path = asst_node.path_from_root()
            # Walk backwards: find last user/tool message before this assistant; and the
            # assistant tool_use that produced it.
            for i in range(len(path) - 2, -1, -1):
                node = path[i]
                if node.role in ("user", "tool") and node.message:
                    prev_tool_result = extract_text_content(node.message)
                    # preceding assistant with tool call
                    for j in range(i - 1, -1, -1):
                        if path[j].role == "assistant" and path[j].message:
                            prev_tool_name, prev_tool_input = extract_tool_name_and_input(path[j].message)
                            break
                    break

            is_init = prev_tool_result is None and prev_tool_name is None
            anchor = make_anchor_obs(
                instance_id=instance_id,
                branch_key=bkey,
                tool_name=prev_tool_name,
                tool_input=prev_tool_input,
                tool_result_text=prev_tool_result,
                workdir=workdir,
                is_init=is_init,
                problem_statement=problem_statement or episode_user,
            )

            tokens = list(turn.prompt_ids) + list(turn.output_ids)
            resp_len = len(turn.output_ids)
            if turn.output_loss_mask:
                loss_mask = list(turn.output_loss_mask)
            else:
                loss_mask = [1] * resp_len
            logprobs = list(turn.output_log_probs) if turn.output_log_probs else [0.0] * resp_len
            if len(logprobs) != resp_len:
                logprobs = (logprobs + [0.0] * resp_len)[:resp_len]
            if max_sample_tokens and len(tokens) > max_sample_tokens:
                # Keep response tail within budget when possible.
                overflow = len(tokens) - max_sample_tokens
                if overflow >= len(turn.prompt_ids):
                    continue
                tokens = tokens[overflow:]
                # prompt truncated; response unchanged
            md = {
                **(extra_metadata or {}),
                "gigpo": True,
                "anchor_obs": anchor,
                "traj_uid": traj_uid,
                "branch_key": bkey,
                "turn_index": asst_node.turn_index,
                "step_immediate_reward": 0.0,
                "episode_reward": float((extra_metadata or {}).get("episode_reward", 0.0) or 0.0),
            }
            samples.append(
                Sample(
                    index=base_sample.index,
                    group_index=base_sample.group_index,
                    rollout_id=base_rid,
                    prompt=base_sample.prompt,
                    label=base_sample.label,
                    tokens=tokens,
                    response_length=resp_len,
                    loss_mask=loss_mask,
                    rollout_log_probs=logprobs,
                    reward=0.0,
                    status=Sample.Status.TRUNCATED if turn.finish_reason == "length" else Sample.Status.COMPLETED,
                    metadata=md,
                )
            )

        # Mark last main-branch turn as the terminal immediate reward carrier.
        if samples and bkey == "main":
            ep_r = float((extra_metadata or {}).get("episode_reward", 0.0) or 0.0)
            samples[-1].metadata["step_immediate_reward"] = ep_r
            samples[-1].metadata["is_terminal_step"] = True
        return samples


__all__ = [
    "TrajectoryManager",
    "TurnRecord",
]
