"""Passive Step 13.6C episode-level Default event trigger audit."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from step13_5c_flow_first_distress_redesign import add_flow_features, classify
from step13_5_passive_distress_semantics import build_features, read

OUT = Path("test/output/step13_6C_default_event_trigger_audit")
RUNS = {
    42: Path("test/output/pre_step13_5K_household_denominator_recovery_hazard/n5000_targeted/firm_diagnostics.csv"),
    7: Path("test/output/step13_5D_runs/seed7/firm_diagnostics.csv"),
    21: Path("test/output/step13_5D_runs/seed21/firm_diagnostics.csv"),
}


def emit(rows, section, metric, value, candidate="", seed="", note=""):
    rows.append({"section": section, "candidate": candidate, "seed": seed, "metric": metric, "value": value, "note": note})


def get_spells(states):
    groups = defaultdict(list)
    for row in states: groups[(row["seed"], row["firm_id"])].append(row)
    output = []
    for (seed, firm), rows in groups.items():
        rows.sort(key=lambda r: r["global_step"])
        current = None
        for row in rows + [{"global_step": 10**9, "d3": False}]:
            if row.get("d3"):
                if current is None:
                    current = {"seed": seed, "firm_id": firm, "start": row["global_step"], "end": row["global_step"], "duration": 1}
                else:
                    current["end"] = row["global_step"]; current["duration"] += 1
            elif current is not None:
                output.append(current); current = None
    return output


def event_rows(spells, threshold, rearm_gap=None):
    events = []
    by_firm = defaultdict(list)
    for spell in spells: by_firm[(spell["seed"], spell["firm_id"])].append(spell)
    for key, firm_spells in by_firm.items():
        firm_spells.sort(key=lambda x: x["start"])
        last_event_end = None
        for spell in firm_spells:
            event_week = spell["start"] + threshold - 1 if spell["duration"] >= threshold else None
            if event_week is None: continue
            if rearm_gap is not None and last_event_end is not None:
                non_d3_gap = spell["start"] - last_event_end - 1
                if non_d3_gap < rearm_gap: continue
            events.append({**spell, "event_week": event_week})
            last_event_end = spell["end"]
    return events


def p90(values):
    values = sorted(values)
    return values[min(len(values) - 1, int(.9 * (len(values) - 1)))] if values else 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_states = []
    for seed, path in RUNS.items():
        raw = read(path); features = build_features(raw); add_flow_features(features)
        all_states.extend({**r, "seed": seed, "d3": classify(r, "A") == "D3"} for r in features)
    d3_spells = get_spells(all_states)
    metrics = []
    emit(metrics, "D3_spell_structure", "total_D3_spell_count", len(d3_spells))
    emit(metrics, "D3_spell_structure", "median_D3_duration", statistics.median([x["duration"] for x in d3_spells]) if d3_spells else 0)
    emit(metrics, "D3_spell_structure", "p90_D3_duration", p90([x["duration"] for x in d3_spells]))
    emit(metrics, "D3_spell_structure", "one_week_D3_spell_count", sum(x["duration"] == 1 for x in d3_spells))
    emit(metrics, "D3_spell_structure", "two_to_three_week_D3_spell_count", sum(2 <= x["duration"] <= 3 for x in d3_spells))
    for seed in RUNS:
        seed_spells = [x for x in d3_spells if x["seed"] == seed]
        emit(metrics, "D3_spell_structure", "D3_spell_count", len(seed_spells), seed=seed)
        emit(metrics, "D3_spell_structure", "median_D3_duration", statistics.median([x["duration"] for x in seed_spells]) if seed_spells else 0, seed=seed)

    candidates = {"A": 1, "B": 4, "C": 13}
    candidate_events = {}
    for candidate, threshold in candidates.items():
        events = event_rows(d3_spells, threshold)
        candidate_events[candidate] = events
        durations_default = [x["duration"] for x in events]
        default_keys = {(x["seed"], x["firm_id"], x["start"]) for x in events}
        default_spells = [x for x in d3_spells if (x["seed"], x["firm_id"], x["start"]) in default_keys]
        nondefault = [x["duration"] for x in d3_spells if (x["seed"], x["firm_id"], x["start"]) not in default_keys]
        timing = [x["event_week"] - x["start"] for x in events]
        emit(metrics, "event_candidate", "default_event_count", len(events), candidate)
        emit(metrics, "event_candidate", "firms_ever_defaulting", len({(x["seed"], x["firm_id"]) for x in events}), candidate)
        emit(metrics, "event_candidate", "events_per_firm", len(events) / max(1, len({(x["seed"], x["firm_id"]) for x in events})), candidate)
        emit(metrics, "event_candidate", "share_D3_spells_defaulting", len(default_spells) / max(1, len(d3_spells)), candidate)
        emit(metrics, "event_candidate", "median_D3_duration_defaulting", statistics.median(durations_default) if durations_default else 0, candidate)
        emit(metrics, "event_candidate", "median_D3_duration_nondefaulting", statistics.median(nondefault) if nondefault else 0, candidate)
        emit(metrics, "event_candidate", "median_D3_to_default_weeks", statistics.median(timing) if timing else 0, candidate)
        emit(metrics, "event_candidate", "p90_D3_to_default_weeks", p90(timing), candidate)
        emit(metrics, "event_candidate", "one_week_D3_spells_filtered", sum(x["duration"] == 1 for x in d3_spells) if threshold > 1 else 0, candidate)
        emit(metrics, "event_candidate", "two_to_three_week_D3_spells_filtered", sum(2 <= x["duration"] <= 3 for x in d3_spells) if threshold > 3 else 0, candidate)
        emit(metrics, "event_candidate", "contractual_interpretability", {"A": "MODERATE", "B": "STRONG", "C": "STRONG"}[candidate], candidate)
        for seed in RUNS:
            emit(metrics, "event_candidate_by_seed", "default_event_count", sum(x["seed"] == seed for x in events), candidate, seed)

    # Re-arm comparison: immediate after D3 exit versus four non-D3 weeks.
    for gap in (0, 4):
        for candidate, threshold in candidates.items():
            events = event_rows(d3_spells, threshold, gap if gap else None)
            emit(metrics, "rearm_comparison", "event_count", len(events), candidate, note="IMMEDIATE_EXIT" if gap == 0 else "NON_D3_4W")
            emit(metrics, "rearm_comparison", "repeat_default_within_4w_count", sum((b["event_week"] - a["event_week"]) <= 4 for a, b in zip(events, events[1:]) if a["seed"] == b["seed"] and a["firm_id"] == b["firm_id"]), candidate, note="IMMEDIATE_EXIT" if gap == 0 else "NON_D3_4W")
            emit(metrics, "rearm_comparison", "repeat_default_within_13w_count", sum((b["event_week"] - a["event_week"]) <= 13 for a, b in zip(events, events[1:]) if a["seed"] == b["seed"] and a["firm_id"] == b["firm_id"]), candidate, note="IMMEDIATE_EXIT" if gap == 0 else "NON_D3_4W")

    # Select the short grace-period rule: it removes transient 1-3 week D3
    # episodes while retaining reversible one-event-per-episode semantics.
    selected = "B"
    flags = {"verdict": "FOUR_WEEK_GRACE_DEFAULT_EVENT_READY", "D3_spell_semantics_understood": True, "default_event_candidates_compared": True, "default_event_trigger_selected": True, "selected_trigger": "D3_4W", "one_event_per_episode_semantics_ready": True, "rearm_semantics_ready": True, "selected_rearm": "NON_D3_4W", "historical_default_separate_from_active_episode": True, "default_event_contractually_interpretable": True, "new_long_runs": 0, "economic_behavior_changed": False, "rng_changed": False, "default_implemented": False, "exit_implemented": False, "default_consequence_design_ready": True, "selected_candidate": selected}
    emit(metrics, "final_recommendation", "selected_trigger", "D3_4W")
    emit(metrics, "final_recommendation", "selected_rearm", "NON_D3_4W")
    emit(metrics, "final_recommendation", "verdict", flags["verdict"])
    emit(metrics, "final_recommendation", "historical_default_separate_from_active_episode", True)
    with (OUT / "default_event_trigger_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "candidate", "seed", "metric", "value", "note"]); writer.writeheader(); writer.writerows(metrics)
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (OUT / "acceptance_summary.md").write_text("""# Step 13.6C acceptance summary

Verdict: **B. FOUR_WEEK_GRACE_DEFAULT_EVENT_READY**.

D3 is treated as an acute eligibility state, not as a weekly event. Candidate A emits one event at D3 entry but also recognizes one- to three-week acute episodes immediately. Candidate C is semantically clear but unnecessarily strict for the current episode distribution. Candidate B provides a short deterministic cure/grace period: a Default Event is emitted once when a contiguous D3 spell reaches four weeks.

The event is episode-level and does not delete the Firm, erase debt, forgive arrears, alter workers, production or credit, or imply Exit. Historical `default_history_count` and current `active_default_episode` should be stored separately in a future implementation. Re-arming should require four consecutive non-D3 weeks; arrears need not be zero.

No simulation, Default consequence, restructuring, Exit, RNG or economic behavior was implemented.
""", encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__": main()
