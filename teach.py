"""
Teaching store for the local builder (no API).

You teach it in two ways:
  save_taught(prompt, steps)   a build you liked becomes a recipe, matched by its prompt words
  rate(recipe_id, good)        +1 when a result is good, -1 when it is wrong

Matching: a prompt's words are compared with each taught recipe's keywords. A recipe with a
net rating below -2 is never used. Taught recipes are checked before the built-in ones.

Stored as JSON at TEACH_FILE (env LOCAL_TEACH_FILE, default ./taught_recipes.json).
"""
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TEACH_FILE = Path(os.getenv("LOCAL_TEACH_FILE", "taught_recipes.json"))
_lock = threading.RLock()

STOP = {"a", "an", "the", "with", "and", "of", "make", "me", "build", "create", "a", "to", "for",
        "in", "on", "my", "some", "small", "big", "tall", "little", "new"}
COLOURS = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black",
           "grey", "gray", "brown", "beige", "cream", "gold", "teal"}
MIN_NET_RATING = -2


def _words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z]+", (text or "").lower()) if w not in STOP]


def keywords_for(prompt: str) -> List[str]:
    """Keywords used for matching: prompt words without colours and stop words, keeping order, deduped."""
    seen, out = set(), []
    for w in _words(prompt):
        if w in COLOURS or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def _load() -> Dict[str, Any]:
    if not TEACH_FILE.exists():
        return {"recipes": []}
    try:
        return json.loads(TEACH_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"recipes": []}


def _save(data: Dict[str, Any]) -> None:
    tmp = TEACH_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(TEACH_FILE)


def save_taught(prompt: str, steps: List[Dict[str, Any]], source_job: Optional[str] = None) -> Dict[str, Any]:
    """Store a build as a recipe. steps = [{"name": str, "parts": [...]}] (same shape the builder uses)."""
    kw = keywords_for(prompt)
    if not kw:
        raise ValueError("Prompt needs at least one word that is not a colour or a stop word.")
    with _lock:
        data = _load()
        rec = {"id": uuid.uuid4().hex[:10], "prompt": prompt, "keywords": kw, "steps": steps,
               "rating": 0, "uses": 0, "source_job": source_job, "created": time.time()}
        data["recipes"].append(rec)
        _save(data)
        return rec


def rate(recipe_id: str, good: bool) -> Optional[Dict[str, Any]]:
    with _lock:
        data = _load()
        for rec in data["recipes"]:
            if rec["id"] == recipe_id:
                rec["rating"] += 1 if good else -1
                _save(data)
                return rec
    return None


def list_taught() -> List[Dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "steps"} | {"step_count": len(r["steps"])}
            for r in _load()["recipes"]]


def delete_taught(recipe_id: str) -> bool:
    with _lock:
        data = _load()
        before = len(data["recipes"])
        data["recipes"] = [r for r in data["recipes"] if r["id"] != recipe_id]
        _save(data)
        return len(data["recipes"]) < before


def match(prompt: str) -> Optional[Tuple[Dict[str, Any], int]]:
    """Best taught recipe for a prompt, as (recipe, overlap score). None if nothing matches."""
    words = set(_words(prompt))
    best, best_score = None, 0
    with _lock:
        recipes = _load()["recipes"]
    for rec in recipes:
        if rec["rating"] < MIN_NET_RATING:
            continue
        overlap = sum(1 for k in rec["keywords"] if k in words)
        # A recipe needs at least half its keywords present, and a rating can break ties.
        if overlap == 0 or overlap * 2 < len(rec["keywords"]):
            continue
        score = overlap * 10 + rec["rating"]
        if score > best_score:
            best, best_score = rec, score
    return (best, best_score) if best else None


def mark_used(recipe_id: str) -> None:
    with _lock:
        data = _load()
        for rec in data["recipes"]:
            if rec["id"] == recipe_id:
                rec["uses"] += 1
        _save(data)
