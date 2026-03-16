"""
Task 3 Curator: Memory Retrieval.
Scores whether the model used the selected memory based on its reason.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..shared import append_jsonl, ensure_dir, call_model_api, log_event, resolve_session_id, make_run_dir, normalize_text
from .generator import Task3Generator
from .evaluator import Task3Evaluator


def _task3_include_label() -> bool:
    """
    Controls whether `label` is included in *model-visible* payloads and dataset snapshots.

    Truthy by default. Set TASK3_INCLUDE_LABEL=0/false/no to disable.
    """
    env_flag = os.environ.get("TASK3_INCLUDE_LABEL")
    if env_flag is None or env_flag.strip() == "":
        return True
    return env_flag.strip() not in {"0", "false", "False", "no", "NO"}


def _sanitize_memory_item(m: Dict[str, Any], include_label: bool) -> Dict[str, Any]:
    """
    Produce a stable, JSON-serializable memory item for artifacts.

    When include_label is False, the returned dict will not contain `label`.
    """
    item = {"memory_id": m.get("memory_id"), "value": m.get("value")}
    if include_label:
        item["label"] = m.get("label")
    return item

# Task 3 Curator & Auditor Prompts Logic Outline:
# 1. Curator Role: Evaluates if the generated response successfully utilized the selected memory based on its reasoning.
#    - Format: Strictly returns JSON containing 'used_memory' (boolean), 'score' (float [0,1]), and 'reason' (string).
# 2. Auditor Role: Uses semantic reasoning to audit the generated query for quality and fairness.
#    - Hard Check (Leakage): Fails if the query contains distinctive phrases from the memory value/evidence that give away the answer.
#    - Soft Checks: Evaluates if the query has a concrete entity target and strictly depends on the selected memory.
#    - Format: Strictly returns JSON detailing 'leakage' dict, boolean flags, 'reasons' list, and 'feedback_for_generator'.

# Here are prompt examples

TASK3_CURATOR_PROMPT = "You are Task 3 Curator. Score whether the model used the selected memory. Output JSON only."

TASK3_CURATOR_AUDIT_PROMPT = "You are Task 3 Curator Auditor. Judge query leakage and memory dependency. Output JSON only."



class Task3Judger:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _probe_id(probe: Dict[str, Any]) -> str:
        return resolve_session_id(probe) or (probe.get("metadata") or {}).get("dialog_id", "") or ""

    @staticmethod
    def _structure_check(probe: Dict[str, Any]) -> tuple[str, List[str]]:
        if not probe.get("selected_memory") or not probe.get("selected_memory_id"):
            return "REWRITE", ["missing_selected_memory"]
        if not probe.get("query"):
            return "REWRITE", ["missing_query"]
        return "ACCEPT", []

    @staticmethod
    def _build_audit_payload(probe: Dict[str, Any]) -> Dict[str, Any]:
        selected = probe.get("selected_memory") or {}
        return {
            "query": probe.get("query", ""),
            "selected_memory": {
                "memory_id": selected.get("memory_id"),
                "label": selected.get("label"),
                "value": selected.get("value", ""),
                "evidence_text": selected.get("evidence_text", ""),
            },
        }

    def _audit_with_llm(
        self,
        probe: Dict[str, Any],
        curator_model_id: Optional[str],
        api_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if not curator_model_id:
            return None
        payload = self._build_audit_payload(probe)
        user_prompt = "INPUT_JSON:\n" + json.dumps(payload, ensure_ascii=True) + "\nEND_INPUT_JSON"
        raw = call_model_api(
            curator_model_id,
            "M_curate_audit",
            TASK3_CURATOR_AUDIT_PROMPT,
            user_prompt,
            api_config=api_config,
        )
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    def decide_generation_only(
        self,
        probe: Dict[str, Any],
        curator_model_id: Optional[str] = None,
        api_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        probe_id = self._probe_id(probe)
        verdict, reasons = self._structure_check(probe)
        if verdict != "ACCEPT":
            return {
                "probe_id": probe_id,
                "verdict": verdict,
                "reasons": reasons,
                "feedback_for_generator": "- Fill missing required fields and ensure a single selected_memory is set.\n- Make the query ask for one specific fact retrievable from that memory.",
                "rewritten_probe": None,
            }

        audit = self._audit_with_llm(probe, curator_model_id, api_config)
        if audit is None:
            # If audit is unavailable, allow generation to proceed without blocking.
            return {
                "probe_id": probe_id,
                "verdict": "ACCEPT",
                "reasons": ["audit_unavailable"],
                "feedback_for_generator": "",
                "rewritten_probe": None,
                "audit": None,
            }

        leakage = audit.get("leakage") if isinstance(audit.get("leakage"), dict) else {}
        is_leaky = bool(leakage.get("is_leaky"))
        evidence = str(leakage.get("evidence_from_query") or "").strip()
        query_text = str(probe.get("query") or "")

        if is_leaky:
            if not evidence or evidence not in query_text:
                is_leaky = False

        if is_leaky:
            fb = str(audit.get("feedback_for_generator") or "").strip() or (
                "- Remove leaked answer details from the query; keep only the slot/type description.\n"
                "- Ask for the specific missing fact so the model must retrieve it from memory."
            )
            return {
                "probe_id": probe_id,
                "verdict": "REWRITE",
                "reasons": ["query_leakage"],
                "feedback_for_generator": fb,
                "rewritten_probe": None,
                "audit": audit,
            }

        if bool(audit.get("is_virtual_world")):
            return {
                "probe_id": probe_id,
                "verdict": "REWRITE",
                "reasons": ["virtual_world_forbidden"],
                "feedback_for_generator": "- Select a PERSONALIZED memory about the user, NOT a virtual world or role-play memory.\n- Task 3 requires real-world persona retrieval.",
                "rewritten_probe": None,
                "audit": audit,
            }

        if not bool(audit.get("has_entity_target")) or not bool(audit.get("depends_on_selected_memory")):
            return {
                "probe_id": probe_id,
                "verdict": "REWRITE",
                "reasons": ["weak_memory_dependence"],
                "feedback_for_generator": str(audit.get("feedback_for_generator") or "").strip(),
                "rewritten_probe": None,
                "audit": audit,
            }

        return {
            "probe_id": probe_id,
            "verdict": "ACCEPT",
            "reasons": [],
            "feedback_for_generator": str(audit.get("feedback_for_generator") or "").strip(),
            "rewritten_probe": None,
            "audit": audit,
        }


class Task3Grader:
    def __init__(self, system_prompt: str = TASK3_CURATOR_PROMPT) -> None:
        self.system_prompt = system_prompt

    @staticmethod
    def _simple_overlap_score(reason: str, memory_text: str) -> float:
        reason_norm = normalize_text(reason or "")
        memory_norm = normalize_text(memory_text or "")
        if not reason_norm or not memory_norm:
            return 0.0
        tokens = set(memory_norm.split())
        if not tokens:
            return 0.0
        hit = sum(1 for t in tokens if t in reason_norm)
        return min(1.0, hit / max(len(tokens), 1))

    def score_report(
        self,
        probe: Dict[str, Any],
        report: Dict[str, Any],
        use_llm: bool = False,
        api_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        selected = probe.get("selected_memory") or {}
        memory_text = selected.get("value", "") or ""
        memory_text = memory_text + " " + (selected.get("evidence_text", "") or "")
        judge_model_id = (api_config or {}).get("judge_model_id")

        runs = []
        for run in report.get("runs", []):
            # If evaluator already produced a judge result (preferred), reuse it.
            existing_judge = run.get("judge")
            if isinstance(existing_judge, dict) and (
                "used_memory" in existing_judge or "score" in existing_judge
            ):
                runs.append({"model": run.get("model"), "judge": existing_judge})
                continue

            parsed = run.get("parsed_response") or {}
            # reason = parsed.get("reason", "")
            reason = parsed.get("value", "")
            if use_llm and judge_model_id:
                # print(judge_model_id)
                payload = {
                    "query": probe.get("query", ""),
                    "selected_memory": selected,
                    "model_response": parsed,
                }
                raw = call_model_api(
                    judge_model_id,
                    "M_judge",
                    self.system_prompt,
                    json.dumps(payload, ensure_ascii=True),
                    api_config=api_config,
                )
                try:
                    judge = json.loads(raw)
                except json.JSONDecodeError:
                    # judge = {"used_memory": False, "score": 0.0, "reason": "invalid_judge_output"}
                    score = self._simple_overlap_score(reason, memory_text)
                    judge = {
                        "used_memory": score >= 0.2,
                        "score": round(score, 4),
                        "reason": "token_overlap",
                    }
            else:
                # print(memory_text)
                score = self._simple_overlap_score(reason, memory_text)
                judge = {
                    "used_memory": score >= 0.2,
                    "score": round(score, 4),
                    "reason": "token_overlap",
                }
            runs.append({"model": run.get("model"), "judge": judge})
        return {"probe_id": Task3Judger._probe_id(probe), "runs": runs}


class Task3Curator:
    """Backward-compatible facade for Task3Judger/Grader."""

    def __init__(self, system_prompt: str = TASK3_CURATOR_PROMPT) -> None:
        self.judger = Task3Judger()
        self.grader = Task3Grader(system_prompt=system_prompt)

    def decide_generation_only(
        self,
        probe: Dict[str, Any],
        curator_model_id: Optional[str] = None,
        api_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.judger.decide_generation_only(
            probe,
            curator_model_id=curator_model_id,
            api_config=api_config,
        )

    def score_report(
        self,
        probe: Dict[str, Any],
        report: Dict[str, Any],
        use_llm: bool = True,
        api_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.grader.score_report(
            probe,
            report,
            use_llm=use_llm,
            api_config=api_config,
        )


def run_task3_pipeline(
    seeds: List[Dict[str, Any]],
    model_list: List[str],
    gen_call_model,
    eval_call_model,
    use_judge: bool = True,
    judge_call_model=None,
    max_attempts_per_seed: int = 5,
    output_dir: str | None = None,
    api_config: Dict[str, Any] | None = None,
    generator_model_id: str | None = None,
    curator_model_id: str | None = None,
    curator_use_llm: bool = False,
    curator_call_model=None,
    max_samples: int | None = None,
    skip_on_rewrite: bool = False,
    other_memories: List[Dict[str, Any]] | None = None,
    distract_n: int = 10,
) -> List[Dict[str, Any]]:
    import random
    
    generator = Task3Generator()
    judger = Task3Judger()
    grader = Task3Grader()
    evaluator = Task3Evaluator()

    results: List[Dict[str, Any]] = []
    if output_dir is None:
        output_dir = make_run_dir("runs", "task3_batch")
    ensure_dir(output_dir)

    events_path = f"{output_dir}/events.jsonl"
    log_fn = lambda event, payload: log_event(events_path, event, payload)

    local_api_config = dict(api_config or {})
    local_api_config.setdefault("debug_log_path", events_path)

    probes_path = f"{output_dir}/probes.jsonl"
    decisions_path = f"{output_dir}/decisions.jsonl"
    reports_path = f"{output_dir}/reports.jsonl"
    errors_path = f"{output_dir}/errors.jsonl"
    dataset_path = f"{output_dir}/dataset.jsonl"

    def log_msg(msg: str):
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

    log_msg(f"=== Starting Task 3 Pipeline ===")
    log_msg(f"Output directory: {output_dir}")

    for seed in seeds:
        if max_samples is not None and len(results) >= max_samples:
            break

        seed_id = resolve_session_id(seed) or seed.get("seed_id")
        log_msg(f"\n--- Processing Seed: {seed_id} ---")
        memories = seed.get("memories") or []
        selected_memory_id = seed.get("selected_memory_id")
        selected_memory = seed.get("selected_memory")
        if not selected_memory and selected_memory_id:
            for memory in memories:
                if memory.get("memory_id") == selected_memory_id:
                    selected_memory = memory
                    break
        if not selected_memory:
            log_fn("task3_seed_skipped", {"seed_id": seed_id, "reason": "missing_selected_memory"})
            continue
        seed = dict(seed)
        seed["selected_memory"] = selected_memory
        seed["selected_memory_id"] = selected_memory.get("memory_id")

        for attempt in range(1, max_attempts_per_seed + 1):
            try:
                log_msg(f"  [Attempt {attempt}] Generating probe...")
                probe = generator.generate_probe(
                    seed,
                    gen_call_model,
                    model_id=generator_model_id,
                    api_config=local_api_config,
                    log_fn=log_fn,
                )
                log_msg(f"    => Query: \"{probe.get('query')}\"")
                if seed.get("dialogue") is not None:
                    probe["dialogue"] = seed.get("dialogue")
                if probe.get("selected_memory") and not probe.get("selected_memory_id"):
                    probe["selected_memory_id"] = probe.get("selected_memory", {}).get("memory_id")
                append_jsonl(probes_path, {"seed_id": seed_id, "attempt": attempt, "probe": probe})
            except Exception as exc:  # noqa: BLE001
                error_text = str(exc)
                append_jsonl(errors_path, {"seed_id": seed_id, "attempt": attempt, "stage": "generate", "error": error_text})
                log_fn("task3_generate_error", {"seed_id": seed_id, "error": error_text})
                continue

            decision = judger.decide_generation_only(
                probe,
                curator_model_id=curator_model_id if curator_use_llm else None,
                api_config=local_api_config,
            )
            log_msg(f"    => Curator Verdict: {decision['verdict']} (Reasons: {decision.get('reasons')})")
            append_jsonl(decisions_path, {"seed_id": seed_id, "attempt": attempt, "decision": decision})
            log_fn("task3_curator_decision", {"seed_id": seed_id, "decision": decision})


            # === yx：强制接受逻辑 ===
            # 如果是最后一次尝试，强制接受，忽略 Judger 的 REWRITE 建议
            if attempt == max_attempts_per_seed:
                if decision["verdict"] != "ACCEPT":
                    log_msg(f"    [Max attempts reached] Forcing verdict to ACCEPT (was {decision['verdict']}).")
                    decision["verdict"] = "ACCEPT"
                    # 可选：记录一下这是强制通过的
                    decision["reasons"] = decision.get("reasons", []) + ["forced_accept_max_attempts"]
            # === 修改结束 ===
            
            if decision["verdict"] == "REWRITE":
                feedback = decision.get("feedback_for_generator") or ""
                log_msg(f"    [Rewrite Needed] Feedback: {feedback[:100]}...")
                if feedback:
                    seed = dict(seed)
                    seed["generator_feedback"] = feedback
                if skip_on_rewrite:
                    break
                continue
            if decision["verdict"] == "DISCARD":
                log_msg(f"    [Discarded] Skipping seed.")
                break

            # Distractor injection (benchmark default: always inject distractors when available)
            candidate_memories = [selected_memory]
            pool_source = other_memories if other_memories is not None else (seed.get("memories") or seed.get("memory_items") or [])
            
            # Exclude the selected memory from the distractor pool
            selected_id = selected_memory.get("memory_id")
            pool = [m for m in pool_source if m.get("memory_id") != selected_id]
            
            if pool:
                if distract_n > 0:
                    if len(pool) > distract_n:
                        distractors = random.sample(pool, distract_n)
                    else:
                        distractors = pool
                    candidate_memories.extend(distractors)
                    random.shuffle(candidate_memories)
                    log_msg(f"    Injected distractors: {len(distractors)}")
                else:
                    log_msg(f"    Injected distractors: 0 (distract_n=0)")
            
            probe["candidate_memories"] = candidate_memories

            report = None
            try:
                log_msg(f"    Evaluating models: {model_list}...")
                report = evaluator.evaluate_models(
                    probe,
                    model_list,
                    eval_call_model,
                    use_judge=use_judge,
                    judge_call_model=judge_call_model,
                    api_config=local_api_config,
                    log_fn=log_fn,
                )
                append_jsonl(reports_path, {"seed_id": seed_id, "attempt": attempt, "report": report})
            except Exception as exc:  # noqa: BLE001
                error_text = str(exc)
                log_msg(f"    [Error during eval] {error_text}")
                append_jsonl(errors_path, {"seed_id": seed_id, "attempt": attempt, "stage": "evaluate", "error": error_text})
                log_fn("task3_evaluate_error", {"seed_id": seed_id, "error": error_text})

            log_msg(f"    Scoring report with grader...")
            score = grader.score_report(
                probe,
                report,
                # Prefer evaluator judge output; curator will reuse it if present.
                use_llm=False,
                api_config=local_api_config,
            )

            log_msg(f"    => Final Score: {json.dumps(score.get('runs', [{}])[0].get('judge', {}).get('score'))}")
            log_fn("task3_grader_score", {"seed_id": seed_id, "score": score})
            append_jsonl(decisions_path, {"seed_id": seed_id, "attempt": attempt, "score": score})

            include_label = _task3_include_label()
            selected_mem = probe.get("selected_memory") or {}
            dataset_entry = {
                "seed_id": seed_id,
                "record": {
                    "dialogue": probe.get("dialogue", []),
                    "selected_memory_id": probe.get("selected_memory_id"),
                    "selected_memory": _sanitize_memory_item(selected_mem, include_label) if isinstance(selected_mem, dict) else {},
                    "candidate_memories": [
                        _sanitize_memory_item(m, include_label)
                        for m in (probe.get("candidate_memories") or [])
                        if isinstance(m, dict)
                    ],
                    "query": probe.get("query", ""),
                    "metadata": probe.get("metadata", {}),
                },
            }
            append_jsonl(dataset_path, dataset_entry)
            results.append({"probe": probe, "score": score, "report": report, "decision": decision})
            break

    return results
