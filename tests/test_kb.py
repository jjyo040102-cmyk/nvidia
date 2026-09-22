"""Retrieval, and the citation gate that sits on top of it.

The product promise at stake here is not "we found a rule"; it is "we found the rule that
governs *this* mechanism of harm". A struck-by finding that cites a rider-on-the-tines rule is
worse than no citation, because it reads as authoritative, so these tests pin the filter rather
than the scores.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vigil.models.risk import HazardClass
from vigil.policy.kb import Rule, Rulebook, load_rulebook


def test_the_shipped_corpus_loads(rulebook: Rulebook) -> None:
    assert len(rulebook) == 19
    assert rulebook.get("osha-1910-178-pedestrians") is not None
    assert rulebook.get("not-a-rule") is None


def test_every_rule_is_indexed_by_a_mechanism(rulebook: Rulebook) -> None:
    """An untagged rule can never be filtered to, so it is invisible to a hazard query."""
    assert [rule.id for rule in rulebook.rules if not rule.hazards] == []


def test_tags_are_lowered_and_blanks_dropped() -> None:
    rule = Rule(
        id="site-x", source="Site rulebook", title="T", text="Body.", tags=[" Hi-Vis ", "", "FOSS"]
    )
    assert rule.tags == ["hi-vis", "foss"]
    assert rule.severity_floor == 1
    assert rule.paraphrase is True


def test_an_unknown_hazard_class_is_refused_at_the_boundary() -> None:
    with pytest.raises(ValidationError):
        Rule(
            id="site-y",
            source="s",
            title="t",
            text="x",
            hazards=["struck_by_drone"],  # not in the vocabulary
        )


def test_haystack_excludes_the_class_it_filters_on() -> None:
    """Putting the class into the text ranks every rule against every other on shared words."""
    rule = Rule(
        id="site-z",
        source="Site rulebook",
        title="Pedestrian separation",
        text="Keep people apart from trucks.",
        hazards=[HazardClass.WORKER_RIDING_ON_FORKS],
    )
    assert "worker_riding_on_forks" not in rule.haystack()
    assert "Pedestrian separation" in rule.haystack()


@pytest.mark.parametrize("hazard", [h.value for h in HazardClass])
def test_a_hazard_query_returns_only_rules_that_govern_it(rulebook: Rulebook, hazard: str) -> None:
    found = rulebook.for_hazard(hazard, k=6)
    assert found, f"no rule retrieved for {hazard}"
    tagged = {scored.rule.id for scored in found if HazardClass(hazard) in scored.rule.hazards}
    assert tagged == {scored.rule.id for scored in found}


def test_the_on_point_rule_wins_for_the_two_easily_confused_cases(rulebook: Rulebook) -> None:
    riders = rulebook.for_hazard("worker_riding_on_forks")
    reversing = rulebook.for_hazard("struck_by_reversing_vehicle")
    assert riders
    assert riders[0].rule.id == "site-no-riders-on-trucks"
    assert reversing
    assert reversing[0].rule.id == "site-reversing-banksman"


def test_a_struck_by_query_cannot_borrow_the_rider_rule(rulebook: Rulebook) -> None:
    """The regression this whole taxonomy exists for.

    Both rules mention forklifts, and the old text-only scorer ranked the rider rule first for
    a pedestrian struck at a blind corner purely on shared vocabulary.
    """
    note = "A worker is carried on the elevated tines while the truck travels toward the camera."
    found = rulebook.for_hazard("struck_by", note=note, k=8)
    assert found
    assert "site-no-riders-on-trucks" not in [s.rule.id for s in found]


def test_a_free_text_query_still_works(rulebook: Rulebook) -> None:
    found = rulebook.search("wet floor with no barrier put around it", k=3)
    assert found
    assert found[0].rule.id in {"site-spill-response", "osha-1910-22-surfaces"}


def test_nonsense_and_empty_queries_return_nothing_rather_than_noise(rulebook: Rulebook) -> None:
    assert rulebook.search("") == []
    assert rulebook.search("the and of to") == []
    assert rulebook.search("qqqq zzzz xxxx") == []


def test_scores_are_bounded_and_ordered(rulebook: Rulebook) -> None:
    found = rulebook.search("pedestrian near a moving forklift", k=6)
    assert len(found) <= 6
    scores = [s.score for s in found]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 for s in scores)


def test_an_unrecognised_class_degrades_to_lexical_search(rulebook: Rulebook) -> None:
    """Agents invent phrasing. Returning nothing would be a silent failure; filtering to a
    class the corpus has never heard of would be the same thing."""
    found = rulebook.for_hazard(
        "crushed_by_conveyor", note="jam cleared by hand on a moving belt", k=3
    )
    assert found
    assert found[0].rule.hazards  # free text still lands on a real, tagged rule


def test_an_empty_or_duplicated_corpus_refuses_to_load(tmp_path: Path) -> None:
    """Silence here would look like "no rule applies" in a report. It is a broken install."""
    with pytest.raises(ValueError, match="rulebook is empty"):
        Rulebook([])
    one = {"id": "site-dup", "source": "s", "title": "t", "text": "x", "hazards": ["slip_trip"]}
    _write(tmp_path, {"rules": [one, dict(one)]})
    with pytest.raises(ValueError, match="duplicate rule ids"):
        load_rulebook(tmp_path)


def _write(directory: Path, payload: object, name: str = "rules.json") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    return directory


def test_a_missing_policy_directory_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="VIGIL_POLICY_DIR"):
        load_rulebook(tmp_path / "nope")


def test_a_broken_corpus_fails_loudly_with_the_file_named(tmp_path: Path) -> None:
    _write(tmp_path, "not a list not a dict")
    with pytest.raises(ValueError, match="expected a list"):
        load_rulebook(tmp_path)

    broken = tmp_path / "broken"
    _write(broken, [{"id": "x"}])
    with pytest.raises(ValueError, match=r"rules\.json rule #0"):
        load_rulebook(broken)


def test_a_single_bad_rule_names_its_index(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "rules": [
                {"id": "site-good", "source": "s", "title": "t", "text": "x"},
                {"id": "ab", "source": "s", "title": "t", "text": "x"},
            ]
        },
    )
    with pytest.raises(ValueError, match=r"rules\.json rule #1"):
        load_rulebook(tmp_path)


def test_rule_files_are_merged_in_a_stable_order(tmp_path: Path) -> None:
    one = [
        {"id": "rule-one", "source": "s", "title": "one", "text": "body", "hazards": ["slip_trip"]},
    ]
    two = [
        {"id": "rule-two", "source": "s", "title": "two", "text": "body", "hazards": ["slip_trip"]},
    ]
    _write(tmp_path, one, "a_rules.json")
    _write(tmp_path, two, "b_rules.json")
    book = load_rulebook(tmp_path)
    assert [r.id for r in book.rules] == ["rule-one", "rule-two"]
