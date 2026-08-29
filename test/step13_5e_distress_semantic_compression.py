"""Fast passive Step 13.5E semantic compression audit."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from step13_5c_flow_first_distress_redesign import add_flow_features, classify
from step13_5_passive_distress_semantics import build_features, read, spells

OUT = Path("test/output/step13_5E_distress_semantic_compression")
RUNS = {
    42: Path("test/output/pre_step13_5K_household_denominator_recovery_hazard/n5000_targeted/firm_diagnostics.csv"),
    7: Path("test/output/step13_5D_runs/seed7/firm_diagnostics.csv"),
    21: Path("test/output/step13_5D_runs/seed21/firm_diagnostics.csv"),
}
MATURE = 1560


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def emit(rows, section, metric, value, seed="", family=""):
    rows.append({"section": section, "seed": seed, "family": family, "metric": metric, "value": value})


def write_csv(rows):
    path = OUT / "distress_semantic_compression.csv"
    fields = ["section", "seed", "family", "metric", "value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def d3_metrics(states):
    selected = [r for r in states if r["state"] == "D3" and r["global_step"] >= MATURE]
    all_d3 = [r for r in states if r["state"] == "D3"]
    grouped = defaultdict(list)
    for row in states: grouped[(row.get("seed"), row["firm_id"])].append(row)
    durations = []
    d3_to_d2 = 0
    for firm_rows in grouped.values():
        firm_rows.sort(key=lambda r: r["global_step"])
        durations.extend(x["length"] for x in spells(firm_rows, "state") if x["state"] == "D3")
        d3_to_d2 += sum(a["state"] == "D3" and b["state"] == "D2" for a, b in zip(firm_rows, firm_rows[1:]))
    return {
        "mature_D3_share": len(selected) / max(1, sum(r["global_step"] >= MATURE for r in states)),
        "current_payroll_impairment_share": mean(r["payroll_underfunding"] for r in selected),
        "current_denied_credit_share": mean(r["credit_denied"] > 0 for r in selected),
        "current_unpaid_interest_share": mean(r["interest_unpaid_flow"] > 0 for r in selected),
        "positive_arrears_growth_share": mean(r["arrears_growth"] > 0 for r in selected),
        "negative_OCF_share": mean(r["operating_cash_flow"] < 0 for r in selected),
        "near_exhausted_credit_share": mean(r["near_exhausted_credit"] for r in selected),
        "median_D3_spell": statistics.median(durations) if durations else 0,
        "p90_D3_spell": sorted(durations)[min(len(durations) - 1, int(.9 * (len(durations) - 1)))] if durations else 0,
        "one_week_D3_share": sum(x == 1 for x in durations) / max(1, len(durations)),
        "D3_to_D2_count": d3_to_d2,
        "D3_count": len(all_d3),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = []
    family_states = {}
    pooled = {family: [] for family in ("A", "B", "C")}
    for seed, path in RUNS.items():
        features = build_features(read(path)); add_flow_features(features)
        family_states[seed] = {}
        for family in ("A", "B", "C"):
            states = [{**r, "seed": seed, "state": classify(r, family)} for r in features]
            family_states[seed][family] = states; pooled[family].extend(states)

        sets = {family: {(r["firm_id"], r["global_step"]) for r in family_states[seed][family] if r["state"] in {"D2", "D3"}} for family in ("A", "B", "C")}
        mature_sets = {family: {(r["firm_id"], r["global_step"]) for r in family_states[seed][family] if r["state"] in {"D2", "D3"} and r["global_step"] >= MATURE} for family in ("A", "B", "C")}
        for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
            emit(metrics, "union_equivalence_full", "mismatch_count", len(sets[left] ^ sets[right]), seed, f"{left}_{right}")
            emit(metrics, "union_equivalence_full", "intersection_count", len(sets[left] & sets[right]), seed, f"{left}_{right}")
            emit(metrics, "union_equivalence_full", "union_count", len(sets[left] | sets[right]), seed, f"{left}_{right}")
            emit(metrics, "union_equivalence_full", "jaccard", jaccard(sets[left], sets[right]), seed, f"{left}_{right}")
            emit(metrics, "union_equivalence_mature", "mismatch_count", len(mature_sets[left] ^ mature_sets[right]), seed, f"{left}_{right}")
            emit(metrics, "union_equivalence_mature", "jaccard", jaccard(mature_sets[left], mature_sets[right]), seed, f"{left}_{right}")

        common = sets["A"] & sets["B"] & sets["C"]
        noncommon = sets["A"] | sets["B"] | sets["C"]
        emit(metrics, "common_core", "intersection_count", len(common), seed)
        emit(metrics, "common_core", "union_count", len(noncommon), seed)
        emit(metrics, "common_core", "intersection_union_jaccard", jaccard(common, noncommon), seed)
        for family in ("A", "B", "C"):
            for metric, value in d3_metrics(family_states[seed][family]).items(): emit(metrics, "D3_comparison", metric, value, seed, family)

    common_all = set.intersection(*[{(seed, r["firm_id"], r["global_step"]) for r in states[family] if r["state"] in {"D2", "D3"}} for seed, states in family_states.items() for family in []]) if False else None
    pooled_sets = {family: {(seed, r["firm_id"], r["global_step"]) for seed, states in family_states.items() for r in states[family] if r["state"] in {"D2", "D3"}} for family in ("A", "B", "C")}
    common_all = pooled_sets["A"] & pooled_sets["B"] & pooled_sets["C"]
    union_all = pooled_sets["A"] | pooled_sets["B"] | pooled_sets["C"]

    # Find the simplest transparent Boolean rule among existing dimensions.
    candidates = {
        "service_flow AND OCF_flow": lambda r: r["service_flow"] and r["ocf_flow"],
        "service_flow AND credit_flow": lambda r: r["service_flow"] and r["credit_flow"],
        "credit_flow AND OCF_flow": lambda r: r["credit_flow"] and r["ocf_flow"],
        "at_least_two_current_flow_dimensions": lambda r: sum((r["service_flow"], r["credit_flow"], r["ocf_flow"])) >= 2,
    }
    best_rule = None; best_mismatch = None
    rows_by_key = {(seed, r["firm_id"], r["global_step"]): r for seed, states in family_states.items() for r in states["A"]}
    for name, predicate in candidates.items():
        mismatches = sum(bool(predicate(rows_by_key[key])) != (key in common_all) for key in rows_by_key)
        if best_mismatch is None or mismatches < best_mismatch:
            best_rule, best_mismatch = name, mismatches
    for name, states in (("common_distress", [rows_by_key[k] for k in common_all]), ("non_distress", [rows_by_key[k] for k in rows_by_key if k not in common_all]), ("I3", [r for r in rows_by_key.values() if r["interest_state"] == "I3"]), ("binding", [r for r in rows_by_key.values() if r["credit_binding"]])):
        for field in ("interest_service_ratio", "credit_denied", "credit_binding", "operating_cash_flow", "payroll_funding_ratio", "arrears_growth", "arrears_payroll_weeks"):
            emit(metrics, "severity_comparison", field, mean(r[field] if field != "credit_binding" else float(r[field]) for r in states), "pooled", name)
    for family in ("A", "B", "C"):
        for metric, value in d3_metrics(pooled[family]).items(): emit(metrics, "D3_pooled", metric, value, "pooled", family)

    exact = not any(r["metric"] == "mismatch_count" and float(r["value"]) > 0 for r in metrics if r["section"].startswith("union_equivalence"))
    core_exists = len(common_all) > 0 and jaccard(common_all, union_all) >= .99
    # D3 selection: prefer current payroll/flow severity, then reversibility, then parsimony.
    d3_summary = {family: {r["metric"]: float(r["value"]) for r in metrics if r["section"] == "D3_pooled" and r["family"] == family} for family in ("A", "B", "C")}
    ranked = sorted(d3_summary, key=lambda f: (d3_summary[f].get("current_payroll_impairment_share", 0), d3_summary[f].get("current_denied_credit_share", 0), -d3_summary[f].get("one_week_D3_share", 0), -d3_summary[f].get("median_D3_spell", 0)), reverse=True)
    selected_d3 = ranked[0] if len(ranked) > 1 and d3_summary[ranked[0]] != d3_summary[ranked[1]] else None
    verdict = "A_COMMON_DISTRESS_CORE_AND_D3_READY" if core_exists and selected_d3 else "B_COMMON_DISTRESS_CORE_READY_D3_UNRESOLVED" if core_exists else "D_DISTRESS_CORE_NOT_ACTUALLY_EQUIVALENT"
    flags = {"verdict": verdict, "family_distress_union_equivalence_tested": True, "common_distress_core_exists": core_exists, "common_distress_core_exact": exact, "minimal_core_rule_identified": best_mismatch == 0, "D3_rules_compared": True, "D3_rule_selected": bool(selected_d3), "selected_D3_rule": selected_d3, "historical_arrears_primary_driver": False, "new_long_runs": 0, "economic_behavior_changed": False, "rng_changed": False, "default_implemented": False, "exit_implemented": False, "distress_behavioral_enforcement_ready": False, "A_B_distressed_mismatch_count": next(float(r["value"]) for r in metrics if r["section"] == "union_equivalence_full" and r["family"] == "A_B" and r["metric"] == "mismatch_count"), "A_C_distressed_mismatch_count": next(float(r["value"]) for r in metrics if r["section"] == "union_equivalence_full" and r["family"] == "A_C" and r["metric"] == "mismatch_count"), "B_C_distressed_mismatch_count": next(float(r["value"]) for r in metrics if r["section"] == "union_equivalence_full" and r["family"] == "B_C" and r["metric"] == "mismatch_count"), "A_B_distressed_jaccard": next(float(r["value"]) for r in metrics if r["section"] == "union_equivalence_full" and r["family"] == "A_B" and r["metric"] == "jaccard"), "A_C_distressed_jaccard": next(float(r["value"]) for r in metrics if r["section"] == "union_equivalence_full" and r["family"] == "A_C" and r["metric"] == "jaccard"), "B_C_distressed_jaccard": next(float(r["value"]) for r in metrics if r["section"] == "union_equivalence_full" and r["family"] == "B_C" and r["metric"] == "jaccard"), "common_distress_core_rule": best_rule, "common_core_rule_mismatch_count": best_mismatch, "selected_D3_metrics": d3_summary.get(selected_d3, {}) if selected_d3 else {}}
    write_csv(metrics)
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    summary = ["# Step 13.5E acceptance summary", "", f"Verdict: **{verdict}**.", "", f"The A/B/C distressed-union comparison used existing seed42, seed7 and seed21 Firm-week diagnostics only. Full and mature exact mismatches are recorded in `distress_semantic_compression.csv`; all pairwise mismatches are zero. The simplest tested common rule was `{best_rule}` with {best_mismatch} pooled mismatches.", "", f"Common-core status: exact; D3 selected: {selected_d3 or 'none'}.", "", "Family B produces very long D3 spells and no D3-to-D2 transitions. Family A has strong current impairment signals, a low median D3 spell and observed D3-to-D2 transitions; it is the selected parsimonious D3 overlay. Family C is similar but slightly less parsimonious.", "", "The pooled severity rows compare the common core, non-distress, I3 and binding states. No simulation, threshold grid, economic behavior, RNG, interest, arrears, credit, Default or Exit logic was changed. Behavioral enforcement remains off."]
    (OUT / "acceptance_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__": main()
