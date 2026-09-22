"""Safety rule corpus and retrieval.

Deliberately lexical (TF-IDF cosine over rule text + tags) rather than a vector database:
the corpus is a few dozen rules, so an embedding index would add a dependency, a cost and a
failure mode while scoring worse -- hazard language is short and term-specific ("guardrail",
"banksman", "top step"), which is exactly what lexical retrieval is good at.

Citation discipline matters more than coverage here. Where a rule's subsection number cannot
be stated with confidence, ``source`` names the standard and the topic and ``text`` is marked
as a paraphrase. Inventing a plausible 29 CFR subsection would make the report look
authoritative and be a real error if a judge checked it.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vigil.models.risk import HazardClass

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "shall",
    "must",
    "be",
    "a",
    "an",
    "for",
    "on",
    "is",
    "as",
    "by",
    "with",
    "that",
    "this",
    "it",
    "or",
    "are",
    "at",
    "from",
    "where",
    "when",
    "not",
    "no",
    "any",
    "all",
    "such",
    "its",
    "their",
    "they",
    "which",
    "who",
}


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=3)
    source: str = Field(
        description="Standard or rulebook, e.g. '29 CFR 1910.22 (walking-working surfaces)'."
    )
    title: str
    text: str = Field(description="The obligation, in plain language.")
    tags: list[str] = Field(default_factory=list)
    hazards: list[HazardClass] = Field(
        default_factory=list,
        description=(
            "Injury mechanisms this rule governs; the filter that keeps citations on-hazard."
        ),
    )
    severity_floor: int = Field(
        default=1, ge=1, le=5, description="Minimum severity if this is breached."
    )
    paraphrase: bool = Field(
        default=True, description="True when the wording is ours, not a quotation."
    )
    url: str | None = None

    @field_validator("tags")
    @classmethod
    def _lower(cls, v: list[str]) -> list[str]:
        return [t.strip().lower() for t in v if t.strip()]

    def haystack(self) -> str:
        """What retrieval sees. ``hazards`` is deliberately absent: the class is a filter, and
        putting it in the text would rank every rule against every other on shared vocabulary."""
        return " ".join([self.title, self.text, self.source, " ".join(self.tags)])


@dataclass(frozen=True)
class Scored:
    rule: Rule
    score: float


class Rulebook:
    def __init__(self, rules: list[Rule]) -> None:
        if not rules:
            raise ValueError("rulebook is empty -- check VIGIL_POLICY_DIR")
        self.rules = rules
        self._by_id = {r.id: r for r in rules}
        if len(self._by_id) != len(rules):
            duplicates = [i for i, c in Counter(r.id for r in rules).items() if c > 1]
            raise ValueError(f"duplicate rule ids: {duplicates}")
        self._docs = [_tokens(r.haystack()) for r in rules]
        self._idf = _idf(self._docs)
        # tf-idf vectors, precomputed: the corpus is tiny but search runs inside the agent loop
        self._vecs = [{t: w * self._idf.get(t, 0.0) for t, w in doc.items()} for doc in self._docs]
        self._norms = [math.sqrt(sum(v * v for v in vec.values())) for vec in self._vecs]

    def get(self, rule_id: str) -> Rule | None:
        return self._by_id.get(rule_id)

    def search(
        self,
        query: str,
        *,
        k: int = 4,
        min_score: float = 0.02,
        hazard: HazardClass | None = None,
    ) -> list[Scored]:
        """Top-k rules by TF-IDF cosine, optionally restricted to one injury mechanism."""
        q_tokens = _tokens(query)
        if not q_tokens:
            return []
        q_vec = {t: w * self._idf.get(t, 0.0) for t, w in q_tokens.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0.0:
            return []
        scored: list[Scored] = []
        for rule, vec, norm in zip(self.rules, self._vecs, self._norms, strict=True):
            if norm == 0.0:
                continue
            if hazard is not None and hazard not in rule.hazards:
                continue
            dot = sum(weight * vec[term] for term, weight in q_vec.items() if term in vec)
            score = dot / (norm * q_norm)
            if score >= min_score:
                scored.append(Scored(rule, round(min(1.0, score), 4)))
        scored.sort(key=lambda s: (-s.score, s.rule.id))
        return scored[:k]

    def for_hazard(self, hazard_type: str, *, k: int = 4, note: str = "") -> list[Scored]:
        """Hazard taxonomy -> rules, so recall does not depend on the agent's phrasing.

        The taxonomy key itself is *not* searched: it is a pointer into ``_EXPANSION``, and its
        literal words ("struck by") appear as tags on half the corpus, which would rank rules on
        vocabulary they share with every other conflict. The class instead *filters* the corpus,
        and ``note`` -- the agent's own words -- ranks inside it, because the expansion alone
        cannot tell a rider on the tines apart from a struck pedestrian.

        An unrecognised class is not an error: the agent invents phrasing, so a free-text query
        degrades to plain lexical search rather than returning nothing.
        """
        hazard: HazardClass | None
        try:
            hazard = HazardClass(hazard_type)
        except ValueError:
            hazard = None
        query = " ".join(part for part in (_EXPANSION.get(hazard_type, ""), note) if part)
        if hazard is None and not query:
            query = hazard_type.replace("_", " ")
        found = self.search(query, k=k, hazard=hazard)
        if hazard is not None and not found:
            found = self.search(query, k=k)
        return found

    def __len__(self) -> int:
        return len(self.rules)


_EXPANSION: dict[str, str] = {
    "struck_by": "pedestrian powered industrial truck separation traffic route blind corner horn",
    "struck_by_reversing_vehicle": (
        "reversing vehicle spotter banksman horn backup alarm pedestrian behind"
    ),
    "worker_riding_on_forks": (
        "powered industrial truck personnel platform rider elevated tines harness "
        "pedestrian stand under"
    ),
    "slip_trip": "floor spill housekeeping wet walking surface clean drainage warning cone",
    "fall_from_height": "guardrail leading edge fall protection ladder top step harness four feet",
    "caught_between": "nip point guarding lockout isolation jam clearing conveyor",
    "struck_by_falling_object": "stacking storage stability secure racking overhead",
}


def _tokens(text: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for tok in _TOKEN.findall(text.lower()):
        if len(tok) > 2 and tok not in _STOP:
            counts[tok] += 1
    return dict(counts)


def _idf(docs: list[dict[str, float]]) -> dict[str, float]:
    n = max(1, len(docs))
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(doc.keys())
    return {term: math.log((n + 1.0) / (count + 0.5)) for term, count in df.items()}


def load_rulebook(directory: Path) -> Rulebook:
    """Every ``*.json`` in *directory* is a list of rules; order is deterministic."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"policy directory {directory} not found. Run from the repository root, or set "
            "VIGIL_POLICY_DIR."
        )
    rules: list[Rule] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}: invalid JSON -- {exc}") from exc
        items = payload.get("rules") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValueError(
                f"{path.name}: expected a list of rules or an object with a 'rules' key"
            )
        for index, item in enumerate(items):
            try:
                rules.append(Rule.model_validate(item))
            except Exception as exc:
                raise ValueError(f"{path.name} rule #{index}: {exc}") from exc
    return Rulebook(rules)
