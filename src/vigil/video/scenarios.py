"""The eight authored clips: six hazards and two hazard-free controls.

Geometry is written forward from the moment of contact. Each hazard states when it became
anticipatable (``t_start``) and when contact would have landed (``t_impact``); paths are
then chosen so the two bodies actually meet at that instant. The controls matter as much
as the hazards -- a system that alarms on everything is not predictive, it is noise -- and
clip 5 deliberately repeats clip 1's layout with the worker staying inside the walkway so
the benchmark can tell "reasoned about sightlines" apart from "saw a forklift and screamed".
"""

from __future__ import annotations

from collections.abc import Callable

from vigil.video.synth import HazardLabel, Scenario, Track

Builder = Callable[[], Scenario]


def _t(*, ref: str, kind: str, path: list[tuple[float, float, float]], **kw: object) -> Track:
    return Track(ref=ref, kind=kind, path=path, **kw)  # type: ignore[arg-type]


# rack + pillar furniture reused across the bay clips
_BAY_FURNITURE = [
    _t(ref="R1", kind="rack", path=[(0.0, -4.2, 5.6)]),
    _t(ref="R2", kind="rack", path=[(0.0, -4.2, 3.1)]),
    _t(ref="R3", kind="rack", path=[(0.0, 5.6, 6.2)]),
]


def blind_corner_struck_by() -> Scenario:
    """Forklift rounds a pillar-controlled corner exactly as a worker crosses the lane."""
    return Scenario(
        id="blind_corner_struck_by",
        title="Struck-by at blind corner",
        duration_s=10.0,
        area_type="warehouse",
        camera_label="CAM-03 / BAY-3",
        tracks=[
            *_BAY_FURNITURE,
            _t(ref="S1", kind="pillar", path=[(0.0, 0.85, 4.6)]),
            _t(
                ref="FL1",
                kind="forklift",
                path=[(0.0, 1.90, 7.20), (10.0, 1.90, 1.60)],
                role="powered_machine",
                category="forklift",
            ),
            _t(
                ref="P1",
                kind="person",
                path=[(2.0, -1.20, 3.50), (9.8, 4.00, 3.50)],
                role="vulnerable_party",
                category="worker",
                ppe_missing=["hi_vis"],
                hold_from=9.8,
            ),
            _t(
                ref="P2",
                kind="person",
                path=[(0.0, -1.05, 6.40), (9.9, -1.05, 4.90)],
                category="worker",
            ),
        ],
        hazards=[
            HazardLabel(
                type="struck_by",
                t_start=3.3,
                t_impact=6.9,
                severity=5,
                entities=["P1", "FL1"],
                description=(
                    "Worker leaves the marked walkway and crosses the machine lane at y=3.5 while "
                    "an unloaded forklift travels toward the same point; pillar S1 blocks the "
                    "operator's sightline until the last ~1.5 m. Contact at approximately 6.9 s."
                ),
            )
        ],
        notes="High-visibility clothing absent, which compounds the occlusion.",
    )


def reversing_dock_no_spotter() -> Scenario:
    """Rigid vehicle reverses into a yard zone a pedestrian walks behind.

    The rear plane sweeps from y=4.45 down to y=3.20 across the first 7 s while the worker
    walks left along y=3.20, so the two arrive at the same ground line at the same instant.
    """
    return Scenario(
        id="reversing_dock_no_spotter",
        title="Reversing vehicle, no banksman",
        duration_s=11.0,
        area_type="loading_dock",
        lighting="low",
        camera_label="CAM-07 / DOCK-2",
        walkway_x=(-3.2, -2.0),
        machine_lane=False,
        tracks=[
            _t(ref="D1", kind="rack", path=[(0.0, -4.60, 6.40)]),
            _t(ref="C1", kind="crate", path=[(0.0, -2.40, 4.40)]),
            _t(
                ref="V1",
                kind="truck",
                path=[(0.0, 3.00, 6.20), (7.0, 3.00, 4.95)],
                role="reverse",
                category="rigid_vehicle",
                facing_override=0.0,
                hold_from=7.0,
            ),
            _t(
                ref="P1",
                kind="person",
                path=[(3.6, 5.60, 3.20), (7.0, 2.90, 3.20)],
                role="vulnerable_party",
                category="worker",
                hold_from=7.0,
            ),
        ],
        hazards=[
            HazardLabel(
                type="struck_by_reversing_vehicle",
                t_start=3.6,
                t_impact=7.0,
                severity=5,
                entities=["P1", "V1"],
                description=(
                    "Rigid vehicle is reversing toward the camera along x=3.0: its rear plane "
                    "moves from 4.45 m out to 3.20 m out over the first 7 s. The worker enters "
                    "that corridor at 3.6 m and walks into the swept line, so contact lands at "
                    "about 7.0 s. No banksman is in shot and the reversing path is not "
                    "barricaded, so nothing breaks the sequence."
                ),
            )
        ],
        notes="Rear-sight blind zone; audit-relevant because segregation is absent.",
    )


def unmarked_spill_slip() -> Scenario:
    """A spill has been on the floor for longer than anyone has walked over it."""
    return Scenario(
        id="unmarked_spill_slip",
        title="Unmarked spill in a walking route",
        duration_s=12.0,
        area_type="production_floor",
        surface="wet",
        camera_label="CAM-11 / LINE-4",
        walkway_x=(-1.9, -0.6),
        machine_lane=False,
        tracks=[
            _t(ref="SP1", kind="spill", path=[(0.0, -1.15, 4.30)]),
            _t(ref="M1", kind="machine", path=[(0.0, 3.10, 5.40)]),
            _t(
                ref="P1",
                kind="person",
                path=[(1.0, -1.15, 7.00), (11.0, -1.15, 1.90)],
                role="vulnerable_party",
                category="operator",
            ),
            _t(ref="K1", kind="cone", path=[(9.4, -0.35, 4.55)]),
            _t(ref="K2", kind="cone", path=[(9.4, -1.95, 4.05)]),
        ],
        hazards=[
            HazardLabel(
                type="slip_trip",
                t_start=4.2,
                t_impact=8.6,
                severity=3,
                entities=["P1", "SP1"],
                description=(
                    "Hydraulic fluid sheen spans the walkway at y=4.3 and no cone is in place "
                    "until 9.4 s, after the operator has already crossed it at 8.6 s."
                ),
            )
        ],
        notes=(
            "Tests whether the model reasons about an uncorrected condition over "
            "time, not just object presence."
        ),
    )


def ladder_overreach_fall() -> Scenario:
    """Worker stands on the top platform and reaches outside the stiles."""
    return Scenario(
        id="ladder_overreach_fall",
        title="Overreaching from a stepladder",
        duration_s=12.0,
        area_type="warehouse",
        lighting="backlit_or_glare",
        camera_label="CAM-02 / STAGE-1",
        walkway_x=(-3.6, -2.4),
        machine_lane=False,
        tracks=[
            _t(ref="L1", kind="ladder", path=[(0.0, 0.25, 4.70)]),
            _t(
                ref="P1",
                kind="person",
                path=[
                    (0.0, -0.90, 6.20),
                    (3.4, 0.25, 4.54),
                    (6.2, 0.25, 4.54),
                    (7.6, 0.53, 4.54),
                    (12.0, 0.53, 4.54),
                ],
                role="reach",
                category="worker",
                lift=1.51,
                lift_window=(3.4, 4.8),
                facing_override=1.5708,
            ),
            _t(ref="C1", kind="crate", path=[(0.0, 1.30, 4.35)]),
            _t(ref="C2", kind="crate", path=[(0.0, 1.30, 5.05)]),
        ],
        hazards=[
            HazardLabel(
                type="fall_from_height",
                t_start=6.2,
                t_impact=9.4,
                severity=4,
                entities=["P1", "L1"],
                description=(
                    "Occupied on the top platform at 1.51 m, which 29 CFR 1910.23 forbids as a "
                    "means of support; from 6.2 s he shifts to the edge of the deck and reaches "
                    "0.8 m clear of the stiles for a crate with no third point of contact, while "
                    "the ladder base is neither footed nor tied off. The overturning moment grows "
                    "until about 9.4 s."
                ),
            )
        ],
        notes=(
            "Fall direction is toward the stacked crates, which also creates "
            "a striking hazard below."
        ),
    )


def control_walkway_discipline() -> Scenario:
    """Clip 1's geometry, replayed with the worker staying inside the walkway."""
    return Scenario(
        id="control_walkway_discipline",
        title="Control -- same bay, walkway respected",
        duration_s=10.0,
        area_type="warehouse",
        camera_label="CAM-03 / BAY-3",
        tracks=[
            *_BAY_FURNITURE,
            _t(ref="S1", kind="pillar", path=[(0.0, 0.85, 4.6)]),
            _t(
                ref="FL1",
                kind="forklift",
                path=[(0.0, 1.90, 7.20), (10.0, 1.90, 1.60)],
                role="powered_machine",
                category="forklift",
            ),
            _t(
                ref="P1",
                kind="person",
                path=[(2.0, -1.20, 3.50), (9.8, -1.00, 3.50)],
                role="vulnerable_party",
                category="worker",
                hold_from=9.8,
            ),
            _t(
                ref="P2",
                kind="person",
                path=[(0.0, -1.05, 6.40), (9.9, -1.05, 4.90)],
                category="worker",
            ),
        ],
        hazards=[],
        notes=(
            "Matched control for blind_corner_struck_by. Everything a detector would key on is "
            "present -- forklift, pillar, workers, low-vis -- but nobody leaves the walkway, so a "
            "correct system stays quiet."
        ),
    )


def night_no_hivis_pedestrian() -> Scenario:
    """Same conflict class as clip 1, but the cue is lighting and clothing, not a corner."""
    return Scenario(
        id="night_no_hivis_pedestrian",
        title="Night shift, no hi-vis, shared route",
        duration_s=10.0,
        area_type="vehicle_yard",
        lighting="night_artificial",
        camera_label="CAM-14 / YARD-N",
        walkway_x=(-2.4, -1.2),
        machine_lane=False,
        tracks=[
            _t(ref="C1", kind="crate", path=[(0.0, -4.4, 5.2)]),
            _t(
                ref="FL1",
                kind="forklift",
                path=[(0.0, 0.60, 7.30), (10.0, 0.60, 2.10)],
                role="powered_machine",
                category="forklift",
                carries="load",
            ),
            _t(
                ref="P1",
                kind="person",
                path=[(1.4, 2.90, 4.10), (9.6, -0.90, 4.10)],
                role="vulnerable_party",
                category="worker",
                ppe_missing=["hi_vis"],
                carries="load",
            ),
        ],
        hazards=[
            HazardLabel(
                type="struck_by",
                t_start=3.9,
                t_impact=6.6,
                severity=4,
                entities=["P1", "FL1"],
                description=(
                    "Elevated load restricts the operator's forward view while a worker without "
                    "hi-vis crosses the travel route at 4.1 m under sodium-only lighting."
                ),
            )
        ],
        notes="Raised load plus absent hi-vis: the model must combine two weak cues.",
    )


def ride_on_forks() -> Scenario:
    """A worker rides the elevated tines while the truck travels, straight at a crossing route.

    Chosen over a leading-edge clip because it is unambiguous at 576p -- the person is
    standing on the forks, which is a rule violation on its own before anyone gets hurt --
    and it pairs two exposures: an elevated worker with no fall protection, and the
    pedestrian he is being carried towards.
    """
    return Scenario(
        id="ride_on_forks",
        title="Worker carried on the elevated tines",
        duration_s=11.0,
        area_type="warehouse",
        camera_label="CAM-06 / AISLE-B",
        walkway_x=(-1.6, -0.4),
        tracks=[
            _t(ref="R1", kind="rack", path=[(0.0, -4.30, 6.20)]),
            _t(
                ref="FL1",
                kind="forklift",
                path=[(0.0, 1.10, 7.00), (3.0, 1.10, 7.00), (8.0, 1.10, 4.85)],
                role="powered_machine",
                category="forklift",
                carries="rider",
                facing_override=3.14159,
                hold_from=8.0,
            ),
            _t(
                ref="P1",
                kind="person",
                path=[(0.0, 1.10, 5.45), (3.0, 1.10, 5.45), (8.0, 1.10, 3.30)],
                role="vulnerable_party",
                category="worker",
                ppe_missing=["harness"],
                lift=0.95,
                lift_window=(2.2, 3.0),
                hold_from=8.0,
            ),
            _t(
                ref="P2",
                kind="person",
                path=[(4.0, 4.60, 3.30), (8.0, 1.10, 3.30)],
                role="vulnerable_party",
                category="worker",
                hold_from=8.0,
            ),
        ],
        hazards=[
            HazardLabel(
                type="worker_riding_on_forks",
                t_start=4.0,
                t_impact=8.0,
                severity=5,
                entities=["P1", "P2", "FL1"],
                description=(
                    "A worker is carried on the tines at 0.95 m with no platform and no tether "
                    "while the truck travels toward the camera along x=1.10. From 4.0 s a second "
                    "worker crosses the same lane at 3.30 m; the elevated load path and the "
                    "pedestrian meet at about 8.0 s, so both a fall from height and a struck-by "
                    "are live on the same trajectory."
                ),
            )
        ],
        notes=(
            "Two exposures, one cause: stopping the travel ends both. Tests whether the agent "
            "reports the riding violation rather than only the converging paths."
        ),
    )


def control_housekeeping() -> Scenario:
    """Second control: a cable run and pallets present, but nothing is time-critical."""
    return Scenario(
        id="control_housekeeping",
        title="Control -- housekeeping, no live conflict",
        duration_s=11.0,
        area_type="production_floor",
        camera_label="CAM-09 / STORE-2",
        walkway_x=(-3.4, -2.2),
        machine_lane=False,
        tracks=[
            _t(ref="C1", kind="crate", path=[(0.0, 3.20, 5.60)]),
            _t(ref="C2", kind="crate", path=[(0.0, 3.20, 4.60)]),
            _t(ref="K1", kind="cone", path=[(0.0, 1.30, 5.10)]),
            _t(
                ref="P1",
                kind="person",
                path=[(0.5, -2.80, 6.60), (10.5, -2.80, 2.60)],
                role="bystander",
                category="operator",
            ),
            _t(
                ref="P2",
                kind="person",
                path=[(2.0, 2.10, 3.20), (9.0, 2.90, 3.20)],
                role="bystander",
                category="operator",
                carries="load",
            ),
        ],
        hazards=[],
        notes=(
            "Clutter, cones and a shared aisle invite a false positive. The powered traffic is "
            "parked and both workers stay on their own side of the store."
        ),
    )


HAZARD_SCENARIOS: list[Builder] = [
    blind_corner_struck_by,
    reversing_dock_no_spotter,
    unmarked_spill_slip,
    ladder_overreach_fall,
    night_no_hivis_pedestrian,
    ride_on_forks,
]

CONTROL_SCENARIOS: list[Builder] = [
    control_walkway_discipline,
    control_housekeeping,
]


def all_scenarios() -> list[Scenario]:
    return [build() for build in (*HAZARD_SCENARIOS, *CONTROL_SCENARIOS)]


def by_id(clip_id: str) -> Scenario:
    for scenario in all_scenarios():
        if scenario.id == clip_id:
            return scenario
    raise KeyError(f"unknown clip {clip_id!r}; have {[s.id for s in all_scenarios()]}")
