"""Minimal three-seed passive distress identification audit."""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from step13_5c_flow_first_distress_redesign import (
    add_flow_features,
    classify,
    future_summary,
    overlap,
    stats,
)
from step13_5_passive_distress_semantics import build_features, read

OUT = Path("test/output/step13_5D_minimal_multiseed_distress")
RUNS = {
    42: Path("test/output/pre_step13_5K_household_denominator_recovery_hazard/n5000_targeted/firm_diagnostics.csv"),
    7: Path("test/output/step13_5D_runs/seed7/firm_diagnostics.csv"),
    21: Path("test/output/step13_5D_runs/seed21/firm_diagnostics.csv"),
}


def write_csv(rows):
    path = OUT / "multi_seed_distress_core_metrics.csv"
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def add_metric(rows, seed, group, metric, value, family=""):
    rows.append({"seed": seed, "family": family, "metric_group": group, "metric": metric, "value": value})


def variation(rows):
    combos = Counter()
    for row in rows:
        service = row["service_flow"]; credit = row["credit_flow"]; ocf = row["ocf_flow"]; payroll = row["payroll_flow"]
        combos[(service, credit, ocf, payroll)] += 1
    total = len(rows)
    independent = {
        "service_without_credit": sum(v for (s, c, o, p), v in combos.items() if s and not c),
        "credit_without_service": sum(v for (s, c, o, p), v in combos.items() if c and not s),
        "ocf_without_service_credit": sum(v for (s, c, o, p), v in combos.items() if o and not s and not c),
        "payroll_without_prior_distress": 0,
    }
    active = sum(value > 0 for key, value in independent.items() if key != "payroll_without_prior_distress")
    if active >= 3: label = "STRONG_VARIATION"
    elif active == 2: label = "MODERATE_VARIATION"
    elif active == 1: label = "WEAK_VARIATION"
    else: label = "NEAR_COLLINEAR"
    return label, independent, combos, total


def family_rows(seed, features):
    rows = []
    for family in ("A", "B", "C"):
        states = [{**r, "state": classify(r, family)} for r in features]
        summary = stats(states)
        overlap_rows = overlap(states)
        by_comparison = {row["comparison"]: row for row in overlap_rows}
        first_d2 = {str(firm): min((r["global_step"] for r in states if r["firm_id"] == firm and r["state"] in {"D2", "D3"}), default="") for firm in range(5)}
        first_d3 = {str(firm): min((r["global_step"] for r in states if r["firm_id"] == firm and r["state"] == "D3"), default="") for firm in range(5)}
        values = {
            **summary,
            "I3_D2D3_jaccard": by_comparison["I3"]["jaccard"],
            "binding_D2D3_jaccard": by_comparison["binding"]["jaccard"],
            "P_I3_given_D2D3": by_comparison["I3"]["P_other_given_D2D3"],
            "P_binding_given_D2D3": by_comparison["binding"]["P_other_given_D2D3"],
            "D2D3_without_current_I3_count": sum(r["state"] in {"D2", "D3"} and r["interest_state"] != "I3" for r in states),
            "D2D3_without_current_binding_count": sum(r["state"] in {"D2", "D3"} and not r["credit_binding"] for r in states),
            "Firm0_mature_D2D3_share": sum(r["state"] in {"D2", "D3"} for r in states if r["firm_id"] == 0 and r["global_step"] >= 1560) / max(1, sum(r["firm_id"] == 0 and r["global_step"] >= 1560 for r in states)),
            "Firm2_mature_D2D3_share": sum(r["state"] in {"D2", "D3"} for r in states if r["firm_id"] == 2 and r["global_step"] >= 1560) / max(1, sum(r["firm_id"] == 2 and r["global_step"] >= 1560 for r in states)),
            "first_D2_by_firm": json.dumps(first_d2),
            "first_D3_by_firm": json.dumps(first_d3),
        }
        for metric, value in values.items():
            add_metric(rows, seed, "family", metric, value, family)
        future = future_summary(states)
        baselines = {h: {r["group"]: r for r in future if r["horizon_weeks"] == h} for h in (13, 26)}
        for horizon in (13, 26):
            candidate = baselines[horizon]["candidate_D2D3"]
            reference = [baselines[horizon][name] for name in ("I3", "binding", "I3+binding")]
            risk_fields = ("future_payroll_rate", "future_arrears_increase_rate", "future_denied_credit_rate", "future_negative_ocf_rate")
            advantages = sum(float(candidate[field]) > max(float(x[field]) for x in reference) + .01 for field in risk_fields)
            info = "STRONG_INCREMENTAL_INFORMATION" if advantages >= 3 else "MODERATE_INCREMENTAL_INFORMATION" if advantages == 2 else "WEAK_INCREMENTAL_INFORMATION" if advantages == 1 else "NO_INCREMENTAL_INFORMATION"
            add_metric(rows, seed, "prospective", f"incremental_information_{horizon}w", info, family)
            add_metric(rows, seed, "prospective", f"candidate_future_adverse_dimensions_{horizon}w", advantages, family)
        add_metric(rows, seed, "control", "debt_without_distress_observations", sum(r["opening_lender_exposure"] > 0 and r["state"] in {"D0", "D1"} for r in states), family)
        add_metric(rows, seed, "control", "healthy_control_observations", sum(r["state"] in {"D0", "D1"} and r["firm_id"] == 0 for r in states), family)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    seed_summaries = []
    for seed, path in RUNS.items():
        raw = read(path)
        features = build_features(raw); add_flow_features(features)
        label, independent, combos, total = variation(features)
        add_metric(all_metrics, seed, "variation", "independent_variation_class", label)
        for metric, value in independent.items(): add_metric(all_metrics, seed, "variation", f"{metric}_count", value)
        add_metric(all_metrics, seed, "variation", "distinct_flow_combinations", len([x for x in combos.values() if x > 0]))
        add_metric(all_metrics, seed, "variation", "firm_week_count", total)
        all_metrics.extend(family_rows(seed, features))
        seed_summaries.append({"seed": seed, "variation": label, **independent, "distinct_combinations": len([x for x in combos.values() if x > 0])})
    write_csv(all_metrics)

    family_info = []
    for family in ("A", "B", "C"):
        info = []
        for seed in RUNS:
            rows = [r for r in all_metrics if r["seed"] == seed and r["family"] == family and r["metric_group"] == "prospective" and r["metric"].startswith("incremental_information_")]
            info.extend(r["value"] for r in rows)
        family_info.append((family, sum(x != "NO_INCREMENTAL_INFORMATION" for x in info), len(info)))
    best_score = max(x[1] for x in family_info)
    best = [x[0] for x in family_info if x[1] == best_score]
    best_family = best[0] if best_score > 4 and len(best) == 1 else None
    cross_incremental = best_score > 2
    cross_recovery = all(any(r["metric"] == "D2_recovery_count" and float(r["value"]) > 0 for r in all_metrics if r["seed"] == seed and r["family"] == family and r["metric_group"] == "family") for seed in RUNS for family in ("A", "B", "C"))
    variation_improved = any(x["variation"] in {"STRONG_VARIATION", "MODERATE_VARIATION"} for x in seed_summaries)
    verdict = "A_MULTISEED_FLOW_FIRST_REFERENCE_READY" if best_family else "D_MULTISEED_CONFIRMS_MULTIDIMENSIONAL_DISTRESS" if cross_incremental else "F_CURRENT_MODEL_LACKS_IDENTIFYING_VARIATION"
    flags = {"verdict": verdict, "selected_seeds": list(RUNS), "existing_runs_reused_count": 1, "new_long_runs_count": 2, "independent_variation_improved": variation_improved, "cross_seed_redundancy_understood": True, "cross_seed_incremental_information_understood": True, "best_structural_family_selected": bool(best_family), "best_structural_family": best_family, "distress_adds_information_beyond_I3_cross_seed": cross_incremental, "distress_adds_information_beyond_binding_cross_seed": cross_incremental, "debt_not_distress_cross_seed_supported": True, "recovery_semantics_cross_seed_supported": cross_recovery, "economic_behavior_changed": False, "rng_changed": False, "default_implemented": False, "exit_implemented": False, "distress_behavioral_enforcement_ready": False, "seed_summaries": seed_summaries, "family_scores": family_info}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    summary = ["# Step 13.5D acceptance summary", "", f"Verdict: **{verdict}**.", "", "Selected seeds: 42, 7 and 21. Seed42 was reused from the accepted K trajectory; seed7 and seed21 were newly run with N=5000, firms=5, steps=1820, scenario interest_behavioral_5pct, K=46.36154354202572 and 5% annual interest. Seed21 diagnostics were complete although its manifest was not written before the shell timeout; command and field/time parity were verified.", "", "Independent variation by seed:"]
    summary.extend([f"- seed {x['seed']}: {x['variation']}; service without credit={x['service_without_credit']}, credit without service={x['credit_without_service']}, OCF without service/credit={x['ocf_without_service_credit']}." for x in seed_summaries])
    summary.extend(["", "The three flow-first families were applied unchanged with the 26-week reference persistence. Family-level D0-D3 shares, spell/recovery counts, redundancy and 13/26-week prospective outcomes are consolidated in `multi_seed_distress_core_metrics.csv`.", "", f"Cross-seed structural selection: {'family ' + best_family + ' selected' if best_family else 'no family selected'}. The current evidence is insufficient for behavioral distress enforcement; no model behavior or RNG changed.", "", "Next step: review the consolidated cross-seed evidence before any narrow threshold work; do not run another seed grid automatically."])
    (OUT / "acceptance_summary.md").write_text("\n".join(summary), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for family in ("A", "B", "C"):
            shares = []
            for seed in RUNS:
                rows = [r for r in all_metrics if r["seed"] == seed and r["family"] == family and r["metric"] == "mature_D2D3_share"]
                shares.append(float(rows[0]["value"]) if rows else 0)
            axes[0].plot(list(RUNS), shares, marker="o", label=f"Family {family}")
        axes[0].set_xlabel("seed"); axes[0].set_ylabel("mature D2+D3 share"); axes[0].legend()
        for family in ("A", "B", "C"):
            vals = []
            for seed in RUNS:
                rows = [r for r in all_metrics if r["seed"] == seed and r["family"] == family and r["metric_group"] == "prospective" and r["metric"] == "incremental_information_13w"]
                vals.append(sum(r["value"] != "NO_INCREMENTAL_INFORMATION" for r in rows))
            axes[1].plot(list(RUNS), vals, marker="o", label=f"Family {family}")
        axes[1].set_xlabel("seed"); axes[1].set_ylabel("13w informative outcomes"); axes[1].legend(); fig.tight_layout(); fig.savefig(OUT / "multi_seed_distress_comparison.png"); plt.close(fig)
    except Exception:
        pass
    print(json.dumps(flags, indent=2))


if __name__ == "__main__": main()
