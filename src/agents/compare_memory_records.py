# Helper module for repeated scoring use.
# Examples:
#   python scripts/compare_memory_records.py
#   python scripts/compare_memory_records.py --use-llm --config configs/api.json
#   python scripts/compare_memory_records.py --matcher hungarian --llm-weight 0.7
#   python scripts/compare_memory_records.py --use-llm --model-id judge
#
# When imported, call score_records(...) directly for fast reuse.
import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================
# 1) Similarity: label / value / type -> item_score
# ============================================================

_ws = re.compile(r"\s+")
_punct = re.compile(r"[^\w\s/.\-]+")


def normalize_text(s: Any) -> str:
    """Lowercase + strip + remove punctuation (keep / . -) + normalize whitespace.

    Robust to non-string inputs (e.g., model outputs lists/objects for value fields).
    """
    if s is None:
        s = ""
    elif not isinstance(s, str):
        try:
            # Prefer stable representation for lists/dicts.
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    s = s.lower().strip()
    s = _punct.sub(" ", s)
    s = _ws.sub(" ", s)
    return s


def tokenize(s: str) -> List[str]:
    """Tokenize by whitespace after normalization."""
    s = normalize_text(s)
    return [t for t in s.split(" ") if t]


def jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def label_similarity(label_a: Any, label_b: Any) -> float:
    """
    Treat labels as paths like "A/B/C".
    Similarity = 2 * LCP / (len(path_a) + len(path_b)),
    where LCP is the length of the longest common prefix.
    """
    if label_a is None:
        label_a = ""
    if label_b is None:
        label_b = ""
    if not isinstance(label_a, str):
        label_a = str(label_a)
    if not isinstance(label_b, str):
        label_b = str(label_b)
    pa = [p for p in (label_a or "").split("/") if p]
    pb = [p for p in (label_b or "").split("/") if p]
    if not pa or not pb:
        return 0.0
    lcp = 0
    for xa, xb in zip(pa, pb):
        if xa != xb:
            break
        lcp += 1
    return (2.0 * lcp) / (len(pa) + len(pb))


def value_similarity(value_a: str, value_b: str, token_weight: float = 0.6) -> float:
    """
    Value similarity = token Jaccard + character similarity (SequenceMatcher) blended.
    This is more robust to light paraphrases / small edits than pure char similarity.
    """
    if not value_a or not value_b:
        return 0.0
    ta = tokenize(value_a)
    tb = tokenize(value_b)
    token_score = jaccard(ta, tb)
    char_score = SequenceMatcher(None, normalize_text(value_a), normalize_text(value_b)).ratio()
    tw = max(0.0, min(1.0, token_weight))
    return tw * token_score + (1.0 - tw) * char_score


def _normalize_confidence(value: Any) -> float:
    """
    Normalize confidence into [0,1].
    Missing/invalid confidence defaults to 1.0 (do not penalize).
    """
    try:
        conf = float(value)
    except Exception:
        return 1.0
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def confidence_score(conf_a: float, conf_b: float) -> Tuple[float, float, float]:
    """
    Confidence penalty:
      - diff <= 0.2 => no penalty
      - diff > 0.2  => penalty grows linearly to max at diff=1.0
    Returns: (score in [0,1], diff, penalty)
    """
    diff = abs(conf_a - conf_b)
    penalty = max(0.0, diff - 0.2)
    score = 1.0 - (penalty / 0.8 if penalty > 0 else 0.0)
    if score < 0.0:
        score = 0.0
    return score, diff, penalty


def item_score(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    w_label: float,
    w_value: float,
    w_type: float,
    w_confidence: float,
    token_weight: float,
) -> Tuple[float, float, float, bool, float, float, float]:
    """
    Compute per-item similarity between one gold item and one pred item.

    Returns:
      score: weighted sum in [0,1]
      ls: label similarity in [0,1]
      vs: value similarity in [0,1]
      tm: whether type matches
      cs: confidence score in [0,1]
      cd: confidence absolute diff
      cp: confidence penalty beyond 0.2
    """
    ls = label_similarity(a.get("label", ""), b.get("label", ""))
    vs = value_similarity(a.get("value", ""), b.get("value", ""), token_weight=token_weight)
    tm = (a.get("type") == b.get("type"))
    ts = 1.0 if tm else 0.0
    conf_a = _normalize_confidence(a.get("confidence"))
    conf_b = _normalize_confidence(b.get("confidence"))
    cs, cd, cp = confidence_score(conf_a, conf_b)

    score = w_label * ls + w_value * vs + w_type * ts + w_confidence * cs
    score = 0.0 if score < 0.0 else (1.0 if score > 1.0 else score)
    return score, ls, vs, tm, cs, cd, cp


# ============================================================
# 2) Matching: compute all pair scores -> greedy 1-to-1 assignment
# ============================================================

@dataclass
class MatchPair:
    gold_id: str
    pred_id: str
    score: float
    label_sim: float
    value_sim: float
    type_match: bool
    confidence_score: float
    confidence_diff: float
    confidence_penalty: float
    label_gold: str
    label_pred: str
    value_gold: str
    value_pred: str


def greedy_match(
    gold_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
    *,
    min_pair_score: float,
    w_label: float,
    w_value: float,
    w_type: float,
    w_confidence: float,
    token_weight: float,
) -> Tuple[List[MatchPair], List[str], List[str]]:
    """
    Greedy 1-to-1 matching:

    1) Score all (gold_i, pred_j) pairs.
    2) Sort by score descending.
    3) Pick a pair if neither index is used and score >= threshold.
    4) Unused gold -> missing, unused pred -> extra.
    """
    scored: List[Tuple[float, int, int, float, float, bool, float, float, float]] = []
    for i, g in enumerate(gold_items):
        for j, p in enumerate(pred_items):
            s, ls, vs, tm, cs, cd, cp = item_score(
                g, p,
                w_label=w_label, w_value=w_value, w_type=w_type,
                w_confidence=w_confidence,
                token_weight=token_weight,
            )
            scored.append((s, i, j, ls, vs, tm, cs, cd, cp))

    scored.sort(reverse=True, key=lambda x: x[0])

    used_g = set()
    used_p = set()
    pairs: List[MatchPair] = []

    for s, i, j, ls, vs, tm, cs, cd, cp in scored:
        if s < min_pair_score:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)

        g = gold_items[i]
        p = pred_items[j]
        pairs.append(MatchPair(
            gold_id=g.get("memory_id", ""),
            pred_id=p.get("memory_id", ""),
            score=round(s, 4),
            label_sim=round(ls, 4),
            value_sim=round(vs, 4),
            type_match=tm,
            confidence_score=round(cs, 4),
            confidence_diff=round(cd, 4),
            confidence_penalty=round(cp, 4),
            label_gold=g.get("label", ""),
            label_pred=p.get("label", ""),
            value_gold=g.get("value", ""),
            value_pred=p.get("value", ""),
        ))

    missing = [g.get("memory_id", "") for i, g in enumerate(gold_items) if i not in used_g]
    extra = [p.get("memory_id", "") for j, p in enumerate(pred_items) if j not in used_p]
    return pairs, missing, extra


def _hungarian(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Hungarian algorithm for square cost matrix.
    Returns a list of (row, col) assignments.
    """
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = []
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment.append((p[j] - 1, j - 1))
    return assignment


def hungarian_match(
    gold_items: List[Dict[str, Any]],
    pred_items: List[Dict[str, Any]],
    *,
    min_pair_score: float,
    w_label: float,
    w_value: float,
    w_type: float,
    w_confidence: float,
    token_weight: float,
) -> Tuple[List[MatchPair], List[str], List[str]]:
    """
    Max-weight matching via Hungarian algorithm (global optimum).
    Converts similarity to cost and supports unmatched via dummy nodes.
    """
    ng = len(gold_items)
    np = len(pred_items)
    n = max(ng, np)
    if n == 0:
        return [], [], []

    # Precompute scores for real pairs.
    scores = [[0.0 for _ in range(np)] for _ in range(ng)]
    parts = [[(0.0, 0.0, False, 0.0, 0.0, 0.0) for _ in range(np)] for _ in range(ng)]
    for i, g in enumerate(gold_items):
        for j, p in enumerate(pred_items):
            s, ls, vs, tm, cs, cd, cp = item_score(
                g, p,
                w_label=w_label, w_value=w_value, w_type=w_type,
                w_confidence=w_confidence,
                token_weight=token_weight,
            )
            scores[i][j] = s
            parts[i][j] = (ls, vs, tm, cs, cd, cp)

    # Build square cost matrix. Dummy matches get score slightly below threshold.
    dummy_score = max(min_pair_score - 0.01, 0.0)
    cost = [[1.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i < ng and j < np:
                cost[i][j] = 1.0 - scores[i][j]
            else:
                cost[i][j] = 1.0 - dummy_score

    assignment = _hungarian(cost)
    used_g = set()
    used_p = set()
    pairs: List[MatchPair] = []

    for i, j in assignment:
        if i >= ng or j >= np:
            continue
        s = scores[i][j]
        if s < min_pair_score:
            continue
        used_g.add(i)
        used_p.add(j)
        ls, vs, tm, cs, cd, cp = parts[i][j]
        g = gold_items[i]
        p = pred_items[j]
        pairs.append(MatchPair(
            gold_id=g.get("memory_id", ""),
            pred_id=p.get("memory_id", ""),
            score=round(s, 4),
            label_sim=round(ls, 4),
            value_sim=round(vs, 4),
            type_match=tm,
            confidence_score=round(cs, 4),
            confidence_diff=round(cd, 4),
            confidence_penalty=round(cp, 4),
            label_gold=g.get("label", ""),
            label_pred=p.get("label", ""),
            value_gold=g.get("value", ""),
            value_pred=p.get("value", ""),
        ))

    missing = [g.get("memory_id", "") for i, g in enumerate(gold_items) if i not in used_g]
    extra = [p.get("memory_id", "") for j, p in enumerate(pred_items) if j not in used_p]
    return pairs, missing, extra


# ============================================================
# 3) Metrics: coverage(PR/F1) + quality(avg score) -> algo_score_100
# ============================================================

def prf1(tp: int, pred_n: int, gold_n: int) -> Dict[str, float]:
    """Compute precision/recall/F1 using TP=matched pairs, pred_n=len(pred), gold_n=len(gold)."""
    p = tp / pred_n if pred_n else 0.0
    r = tp / gold_n if gold_n else 0.0
    f1 = (2 * p * r) / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def clip100(x: float) -> float:
    """Clamp to [0,100]."""
    return 0.0 if x < 0.0 else (100.0 if x > 100.0 else x)


# ============================================================
# 4) LLM as judge: interface + runnable mock
# ============================================================

def build_judge_payload(
    record_gold: Dict[str, Any],
    record_pred: Dict[str, Any],
    matched_pairs: List[MatchPair],
    missing_ids: List[str],
    extra_ids: List[str],
) -> Dict[str, Any]:
    """Build payload for LLM judge using memory-to-memory comparison only."""
    gold_map = {x.get("memory_id", ""): x for x in (record_gold.get("memory_items") or [])}
    pred_map = {x.get("memory_id", ""): x for x in (record_pred.get("memory_items") or [])}

    return {
        "evidence": {
            "mode": "memory_only",
            "gold_memory_items": list(gold_map.values()),
            "pred_memory_items": list(pred_map.values()),
        },
        "pairs": [
            {
                "gold": gold_map.get(p.gold_id, {}),
                "pred": pred_map.get(p.pred_id, {}),
                "algo": asdict(p),
            }
            for p in matched_pairs
        ],
        "missing": [gold_map.get(mid, {}) for mid in missing_ids if mid in gold_map],
        "extra": [pred_map.get(mid, {}) for mid in extra_ids if mid in pred_map],
        "output_schema": {
            "pair_reviews": "list[{gold_id,pred_id,ok(bool),error_type,severity(0-3),rationale}]",
            "missing_reviews": "list[{id,should_have_extracted(bool),severity(0-3),rationale}]",
            "extra_reviews": "list[{id,hallucinated(bool),severity(0-3),rationale}]",
        }
    }


def compute_llm_score_100(llm_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert LLM's structured judgments into a 0-100 score (penalty-based).

    Penalties:
      - extra hallucination: base 15 * severity_multiplier
      - missing item that should have been extracted: base 10 * severity_multiplier
      - pair-level errors:
          VALUE_DRIFT 12, HALLUCINATION 15, LABEL_WRONG 6, TYPE_WRONG 6,
          OVERGENERAL 3, UNDERGENERAL 3, OTHER 4, EQUIVALENT 0

    severity_multiplier:
      0 -> 0.5, 1 -> 1.0, 2 -> 1.5, 3 -> 2.0
    """
    score = 100.0

    def mult(sev: Any) -> float:
        try:
            s = int(sev)
            s = 0 if s < 0 else (3 if s > 3 else s)
            return 0.5 + 0.5 * s
        except Exception:
            return 1.0

    pair_penalty_table = {
        "EQUIVALENT": 0.0,
        "VALUE_DRIFT": 12.0,
        "HALLUCINATION": 15.0,
        "LABEL_WRONG": 6.0,
        "TYPE_WRONG": 6.0,
        "OVERGENERAL": 3.0,
        "UNDERGENERAL": 3.0,
        "OTHER": 4.0,
    }

    breakdown = {"start": 100.0, "pair_penalty": 0.0, "missing_penalty": 0.0, "extra_penalty": 0.0}

    for pr in llm_out.get("pair_reviews", []) or []:
        et = str(pr.get("error_type", "OTHER")).upper()
        sev = mult(pr.get("severity", 1))
        pen = pair_penalty_table.get(et, 4.0) * sev
        score -= pen
        breakdown["pair_penalty"] += pen

    for mr in llm_out.get("missing_reviews", []) or []:
        if mr.get("should_have_extracted") is True:
            sev = mult(mr.get("severity", 1))
            pen = 10.0 * sev
            score -= pen
            breakdown["missing_penalty"] += pen

    for er in llm_out.get("extra_reviews", []) or []:
        if er.get("hallucinated") is True:
            sev = mult(er.get("severity", 1))
            pen = 15.0 * sev
            score -= pen
            breakdown["extra_penalty"] += pen

    score = clip100(score)
    for k in breakdown:
        breakdown[k] = round(breakdown[k], 4)
    breakdown["final"] = round(score, 4)
    return {"score_100": round(score, 4), "breakdown": breakdown}


def mock_llm_judge(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runnable mock judge (no real model calls).
    This lets you run end-to-end and see how LLM score affects the final score.

    Heuristic (memory-only):
      - Pairs: value_sim>=0.65 => EQUIVALENT else VALUE_DRIFT
      - Missing: always should_have_extracted
      - Extra: always hallucinated
    """
    pair_reviews = []
    for entry in payload.get("pairs", []) or []:
        gold = entry.get("gold", {}) or {}
        pred = entry.get("pred", {}) or {}
        algo = entry.get("algo", {}) or {}

        gid = gold.get("memory_id", "")
        pid = pred.get("memory_id", "")
        vs = float(algo.get("value_sim", 0.0) or 0.0)

        if vs >= 0.65:
            pair_reviews.append({
                "gold_id": gid,
                "pred_id": pid,
                "ok": True,
                "error_type": "EQUIVALENT",
                "severity": 0,
                "rationale": "Semantically close.",
            })
        else:
            pair_reviews.append({
                "gold_id": gid,
                "pred_id": pid,
                "ok": False,
                "error_type": "VALUE_DRIFT",
                "severity": 2,
                "rationale": "Meaning deviates from gold.",
            })

    missing_reviews = []
    for g in payload.get("missing", []) or []:
        mid = g.get("memory_id", "")
        missing_reviews.append({
            "id": mid,
            "should_have_extracted": True,
            "severity": 2,
            "rationale": "Missing gold memory item.",
        })

    extra_reviews = []
    for p in payload.get("extra", []) or []:
        mid = p.get("memory_id", "")
        extra_reviews.append({
            "id": mid,
            "hallucinated": True,
            "severity": 2,
            "rationale": "Extra prediction not in gold.",
        })

    return {"pair_reviews": pair_reviews, "missing_reviews": missing_reviews, "extra_reviews": extra_reviews}


# ============================================================
# 5) Real LLM judge (from configs/api.json)
# ============================================================

def load_api_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from model output."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model output is not valid JSON.")
    return json.loads(match.group(0))


def _call_chat_completion(
    endpoint: str,
    api_key: str,
    model_id: str,
    messages: List[Dict[str, str]],
    timeout: int,
) -> Dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 8000, # yx: more outputs
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP error {exc.code}: {error_body}") from exc


def build_llm_judge_from_config(
    config_path: str = "configs/api.json",
    model_id: Optional[str] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Build a judge callable that reads configs/api.json.
    It uses:
      - model_endpoints[model_id]
      - model_key_envs[model_id]
    """
    cfg = load_api_config(config_path)
    judge_model = model_id or cfg.get("judge_model_id")
    if not judge_model:
        raise ValueError("Missing judge_model_id in configs/api.json.")
    endpoint = cfg.get("model_endpoints", {}).get(judge_model)
    if not endpoint:
        raise ValueError(f"Missing endpoint for model {judge_model}.")
    env_key = cfg.get("model_keys", {}).get(judge_model)
    if not env_key:
        raise ValueError(f"Missing env key mapping for model {judge_model}.")
    api_key = env_key
    if not api_key:
        raise EnvironmentError(f"Missing API key in env var {env_key}.")
    timeout = int(cfg.get("timeout", 60))

    def judge(payload: Dict[str, Any]) -> Dict[str, Any]:
        # System Prompt Logic Outline:
        # 1. Role: Acts as a strict judge evaluating memory extraction quality.
        # 2. Constraint: Since raw dialogues are unavailable, the model conservatively assesses whether extra items 
        #    are obvious hallucinations and checks matched pairs for meaning deviations (e.g., VALUE_DRIFT).
        # 3. Format Strictness: Forces the model to output ONLY pure JSON. The output must strictly include 
        #    three arrays: 'pair_reviews', 'missing_reviews', and 'extra_reviews', with no markdown or extra text.

        # Here are examples
        system = ("You are a strict judge for memory extraction quality. Output ONLY valid JSON containing pair_reviews, missing_reviews, and extra_reviews without markdown.")
        user = (
            "Evaluate the following payload and output JSON only.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        response = _call_chat_completion(
            endpoint=endpoint,
            api_key=api_key,
            model_id=judge_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            timeout=timeout,
        )
        print(response)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return _extract_json(content)

    return judge


# ============================================================
# 6) Main entry: score_records
# ============================================================

def score_records(
    record_gold: Dict[str, Any],
    record_pred: Dict[str, Any],
    *,
    min_pair_score: float = 0.30,
    w_label: float = 0.40,
    w_value: float = 0.40,
    w_type: float = 0.10,
    w_confidence: float = 0.10,
    token_weight: float = 0.6,
    algo_quality_weight: float = 0.60,
    algo_coverage_weight: float = 0.40,
    llm_weight: float = 0.50,
    matcher: str = "greedy",
    llm_judge: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Score two records:
      - record_gold: human labeled (gold)
      - record_pred: model output (pred)

    Returns an explainable JSON including:
      - matching results (pairs/missing/extra)
      - algorithmic metrics and score (0-100)
      - optional LLM judge score (0-100)
      - final blended score (0-100)
    """
    gold_items = record_gold.get("memory_items", []) or []
    pred_items = record_pred.get("memory_items", []) or []

    # 1) Match memory items
    if matcher == "hungarian":
        pairs, missing_ids, extra_ids = hungarian_match(
            gold_items, pred_items,
            min_pair_score=min_pair_score,
            w_label=w_label, w_value=w_value, w_type=w_type, w_confidence=w_confidence,
            token_weight=token_weight,
        )
    else:
        pairs, missing_ids, extra_ids = greedy_match(
            gold_items, pred_items,
            min_pair_score=min_pair_score,
            w_label=w_label, w_value=w_value, w_type=w_type, w_confidence=w_confidence,
            token_weight=token_weight,
        )

    # 2) Coverage (soft PR/F1)
    tp = len(pairs)
    coverage = prf1(tp, len(pred_items), len(gold_items))

    # 3) Quality (only on matched pairs)
    denom = max(tp, 1)
    avg_pair_score = sum(p.score for p in pairs) / denom
    avg_label = sum(p.label_sim for p in pairs) / denom
    avg_value = sum(p.value_sim for p in pairs) / denom
    type_match_rate = sum(1 for p in pairs if p.type_match) / denom
    avg_conf_score = sum(p.confidence_score for p in pairs) / denom
    avg_conf_diff = sum(p.confidence_diff for p in pairs) / denom
    avg_conf_penalty = sum(p.confidence_penalty for p in pairs) / denom

    # 4) Algo score in [0,100] = quality + coverage weighted
    qw = max(0.0, min(1.0, algo_quality_weight))
    cw = max(0.0, min(1.0, algo_coverage_weight))
    if qw + cw == 0.0:
        qw, cw = 0.6, 0.4
    else:
        s = qw + cw
        qw, cw = qw / s, cw / s

    quality_component = 100.0 * qw * avg_pair_score
    coverage_component = 100.0 * cw * coverage["f1"]
    algo_score_100 = clip100(quality_component + coverage_component)

    result: Dict[str, Any] = {
        "matching": {
            "accepted_pairs": [asdict(p) for p in pairs],
            "missing_in_pred": missing_ids,
            "extra_in_pred": extra_ids,
        },
        "metrics": {
            "quality": {
                "avg_pair_score": round(avg_pair_score, 4),
                "avg_label": round(avg_label, 4),
                "avg_value": round(avg_value, 4),
                "type_match_rate": round(type_match_rate, 4),
                "avg_confidence_score": round(avg_conf_score, 4),
                "avg_confidence_diff": round(avg_conf_diff, 4),
                "avg_confidence_penalty": round(avg_conf_penalty, 4),
            },
            "coverage": coverage,
        },
        "algo_score": {
            "score_100": round(algo_score_100, 4),
            "breakdown": {
                "quality_weight": round(qw, 4),
                "coverage_weight": round(cw, 4),
                "quality_component": round(quality_component, 4),
                "coverage_component": round(coverage_component, 4),
            },
            "params": {
                "min_pair_score": min_pair_score,
                "w_label": w_label,
                "w_value": w_value,
                "w_type": w_type,
                "w_confidence": w_confidence,
                "token_weight": token_weight,
                "matcher": matcher,
            }
        },
        "llm_score": {
            "enabled": False,
            "score_100": None,
            "breakdown": None,
            "raw": None,
        },
        "final_score": {
            "llm_weight": 0.0,
            "algo_weight": 1.0,
            "score_100": round(algo_score_100, 4),
        },
        "explain": {
            "algo_score_100": "100*(quality_weight*avg_pair_score + coverage_weight*F1).",
            "quality": "Computed on matched pairs only; captures how similar matched items are.",
            "coverage": "Soft PR/F1 where TP = number of matched pairs.",
            "llm_score_100": "Optional; derived from structured LLM judgments on pairs/missing/extra.",
            "final_score_100": "Blend: (1-llm_weight)*algo + llm_weight*llm.",
        }
    }

    # 5) Optional LLM judge and blended final score
    if llm_judge is not None:
        payload = build_judge_payload(record_gold, record_pred, pairs, missing_ids, extra_ids)
        llm_out = llm_judge(payload)
        llm_scored = compute_llm_score_100(llm_out)

        lw = max(0.0, min(1.0, llm_weight))
        aw = 1.0 - lw
        final_100 = aw * algo_score_100 + lw * llm_scored["score_100"]

        result["llm_score"] = {
            "enabled": True,
            "score_100": llm_scored["score_100"],
            "breakdown": llm_scored["breakdown"],
            "raw": llm_out,
        }
        result["final_score"] = {
            "llm_weight": round(lw, 4),
            "algo_weight": round(aw, 4),
            "score_100": round(final_100, 4),
            "breakdown": {
                "algo_contribution": round(aw * algo_score_100, 4),
                "llm_contribution": round(lw * llm_scored["score_100"], 4),
            }
        }

    return result


# ============================================================
# 7) Coherent mock records (one-click runnable)
# ============================================================

def build_mock_records() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    A coherent example:
      Evidence says:
        - no pharmacy experience
        - has a tortoise named Bubbles
        - wants a professional onboarding email
        - prefers concise writing

    Gold has 4 items; Pred has:
      - a close match with label path slightly off
      - a good pet match
      - a paraphrased writing preference
      - an extra hallucinated preference (humorous tone)
      - missing the "Goals/Task" item
    """
    session = {
        "session_id": "sess_demo_001",
        "started_at": "2026-01-20T09:00:00+00:00",
        "ended_at": "2026-01-20T09:05:00+00:00",
        "turns": [
            {
                "utterance_index": 0,
                "timestamp": "2026-01-20T09:00:00+00:00",
                "role": "user",
                "text": (
                    "I start as a pharmacy intern tomorrow, but I have no pharmacy-related experience. "
                    "I also have a pet tortoise named Bubbles. "
                    "Can you help me draft a more professional onboarding email? "
                    "I prefer concise and well-structured wording."
                ),
            },
            {"utterance_index": 1, "timestamp": "2026-01-20T09:01:00+00:00", "role": "assistant", "text": "Sure."},
        ],
    }

    record_gold = {
        "sessions": [session],
        "intents_ranked": [
            {"intent_category": "Transactional Intent", "intent_subtype": "Service Utilization", "probability": 0.7}
        ],
        "memory_items": [
            {"memory_id": "g1", "type": "direct", "label": "Work/Experience", "value": "No prior pharmacy-related experience", "confidence": 0.95},
            {"memory_id": "g2", "type": "direct", "label": "Possessions/Pet", "value": "Has a tortoise named Bubbles", "confidence": 0.9},
            {"memory_id": "g3", "type": "indirect", "label": "Preferences/WritingStyle", "value": "Prefers concise, well-structured wording", "confidence": 0.8},
            {"memory_id": "g4", "type": "indirect", "label": "Goals/Task", "value": "Wants help drafting a professional onboarding email", "confidence": 0.9},
        ],
    }

    record_pred = {
        "sessions": [session],
        "intents_ranked": [
            {"intent_category": "Transactional Intent", "intent_subtype": "Service Utilization", "probability": 0.6}
        ],
        "memory_items": [
            {"memory_id": "p10", "type": "direct", "label": "Work/DomainExperience", "value": "No prior pharmacy work experience", "confidence": 0.3},
            {"memory_id": "p11", "type": "direct", "label": "Possessions/Pet", "value": "Has a pet tortoise named Bubbles", "confidence": 0.88},
            {"memory_id": "p12", "type": "indirect", "label": "Preferences/WritingStyle", "value": "Interested in concise and structured email phrasing", "confidence": 0.6},
            {"memory_id": "p13", "type": "indirect", "label": "Preferences/Tone", "value": "Prefers humorous tone", "confidence": 0.3},
        ],
    }

    return record_gold, record_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare memory records with optional LLM judge.")
    parser.add_argument("--config", default="configs/api.json", help="Path to api.json")
    parser.add_argument("--use-llm", action="store_true", help="Enable real LLM judge")
    parser.add_argument("--llm-weight", type=float, default=0.5, help="Blend weight for LLM score")
    parser.add_argument("--min-pair-score", type=float, default=0.30, help="Min score to accept a pair")
    parser.add_argument("--matcher", choices=["greedy", "hungarian"], default="greedy")
    parser.add_argument("--model-id", default=None, help="Override judge model id")
    args = parser.parse_args()

    gold, pred = build_mock_records()

    judge = None
    if args.use_llm:
        judge = build_llm_judge_from_config(config_path=args.config, model_id=args.model_id)

    out = score_records(
        gold,
        pred,
        llm_judge=judge,
        llm_weight=args.llm_weight,
        min_pair_score=args.min_pair_score,
        matcher=args.matcher,
    )

    print(json.dumps(out, ensure_ascii=False, indent=2))
