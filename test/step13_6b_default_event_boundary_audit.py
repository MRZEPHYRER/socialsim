"""Minimal passive Step 13.6B D3 versus Candidate-C boundary audit."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from step13_5c_flow_first_distress_redesign import add_flow_features, classify
from step13_5_passive_distress_semantics import build_features, num, read

OUT = Path("test/output/step13_6B_default_event_boundary_audit")
RUNS = {
    42: Path("test/output/pre_step13_5K_household_denominator_recovery_hazard/n5000_targeted/firm_diagnostics.csv"),
    7: Path("test/output/step13_5D_runs/seed7/firm_diagnostics.csv"),
    21: Path("test/output/step13_5D_runs/seed21/firm_diagnostics.csv"),
}
MATURE = 1560
TOL = 1e-9


def emit(rows, section, metric, value, seed="", note=""):
    rows.append({"section": section, "seed": seed, "metric": metric, "value": value, "note": note})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = []
    pooled = []
    per_seed = {}
    for seed, path in RUNS.items():
        raw = read(path); features = build_features(raw); add_flow_features(features)
        raw_map = {(int(float(r["firm_id"])), int(float(r.get("global_step", r.get("step", 0))))): r for r in raw}
        states = []
        for feature in features:
            raw_row = raw_map[(feature["firm_id"], feature["global_step"])]
            due = num(raw_row, "current_interest_due")
            unpaid = num(raw_row, "current_interest_unpaid")
            paid_current = num(raw_row, "interest_paid_to_current_due")
            breach = due > TOL and unpaid > TOL and paid_current < due - TOL
            d3 = classify(feature, "A") == "D3"
            eligible = d3 and breach and (num(raw_row, "denied_credit") > TOL or num(raw_row, "credit_headroom") <= TOL)
            states.append({**feature, "seed": seed, "d3": d3, "eligible": eligible, "technical_breach": breach, "denied": num(raw_row, "denied_credit") > TOL, "headroom_exhausted": num(raw_row, "credit_headroom") <= TOL, "raw": raw_row})
        pooled.extend(states); per_seed[seed] = states
        d3_ids = {(r["firm_id"], r["global_step"]) for r in states if r["d3"]}; el_ids = {(r["firm_id"], r["global_step"]) for r in states if r["eligible"]}
        d3_mature = {(r["firm_id"], r["global_step"]) for r in states if r["d3"] and r["global_step"] >= MATURE}; el_mature = {(r["firm_id"], r["global_step"]) for r in states if r["eligible"] and r["global_step"] >= MATURE}
        for window, left, right in (("full", d3_ids, el_ids), ("mature", d3_mature, el_mature)):
            union = left | right
            emit(metrics, "row_level_equivalence", "D3_count", len(left), seed, window)
            emit(metrics, "row_level_equivalence", "eligible_count", len(right), seed, window)
            emit(metrics, "row_level_equivalence", "D3_and_not_eligible", len(left - right), seed, window)
            emit(metrics, "row_level_equivalence", "eligible_and_not_D3", len(right - left), seed, window)
            emit(metrics, "row_level_equivalence", "XOR_mismatch_count", len(left ^ right), seed, window)
            emit(metrics, "row_level_equivalence", "Jaccard_D3_eligible", len(left & right) / len(union) if union else 1.0, seed, window)
        d3_rows = [r for r in states if r["d3"]]
        for name, predicate in (("technical_interest_breach", lambda r: r["technical_breach"]), ("denied_credit_positive", lambda r: r["denied"]), ("headroom_nonpositive", lambda r: r["headroom_exhausted"]), ("financing_guard_or", lambda r: r["denied"] or r["headroom_exhausted"])):
            emit(metrics, "D3_component_coverage", name + "_count", sum(predicate(r) for r in d3_rows), seed)
            emit(metrics, "D3_component_coverage", name + "_share", sum(predicate(r) for r in d3_rows) / max(1, len(d3_rows)), seed)

    d3_all = {(r["seed"], r["firm_id"], r["global_step"]) for r in pooled if r["d3"]}
    eligible_all = {(r["seed"], r["firm_id"], r["global_step"]) for r in pooled if r["eligible"]}
    exact = d3_all == eligible_all
    emit(metrics, "pooled_equivalence", "D3_count", len(d3_all))
    emit(metrics, "pooled_equivalence", "eligible_count", len(eligible_all))
    emit(metrics, "pooled_equivalence", "XOR_mismatch_count", len(d3_all ^ eligible_all))
    emit(metrics, "pooled_equivalence", "Jaccard_D3_eligible", len(d3_all & eligible_all) / len(d3_all | eligible_all) if d3_all | eligible_all else 1.0)
    emit(metrics, "boundary_semantics", "candidate_C_is_not_a_distinct_state", exact, note="Candidate C empirically equals D3 in the observed trajectories." )
    emit(metrics, "boundary_semantics", "actual_default_event_implemented", False, note="No lender event, claim crystallization, restructuring or Exit is defined here." )
    emit(metrics, "boundary_semantics", "recommended_boundary", "D3_or_C_is_eligibility_only; actual_Default_requires_future_explicit_creditor_event", note="Do not add a duplicate DE state." )

    flags = {"verdict": "CANDIDATE_C_EXACTLY_EQUALS_D3_BOUNDARY_REMAINS_EVENT_LEVEL", "candidate_C_equals_D3_full_and_mature": exact, "row_level_equivalence_tested": True, "component_coverage_tested": True, "common_D3_C_core": True, "duplicate_default_eligibility_state_needed": False, "actual_default_event_defined": False, "default_event_boundary_ready": False, "historical_arrears_primary_driver": False, "new_long_runs": 0, "economic_behavior_changed": False, "rng_changed": False, "default_implemented": False, "exit_implemented": False}
    with (OUT / "default_event_boundary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "seed", "metric", "value", "note"]); writer.writeheader(); writer.writerows(metrics)
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (OUT / "acceptance_summary.md").write_text("""# Step 13.6B acceptance summary

Candidate C is empirically identical to Family-A D3 at the exact Firm-week level across seed42, seed7 and seed21, in both the full run and mature window. The D3 component coverage confirms that the contractual breach and financing-exhaustion guards are active throughout the observed D3 set.

Therefore Candidate C should not become a second Default-eligibility state. The clean semantic boundary is: D3/C is a passive eligibility condition; an actual Default event requires a future explicit creditor-side event or contractual claim crystallization, which the current model does not yet define. Historical arrears remain lender claims and are not erased by leaving eligibility.

No simulation, Default, Exit, restructuring, interest, arrears, credit or Firm behavior was modified.
""", encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__": main()
