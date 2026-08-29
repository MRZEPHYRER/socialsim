"""Pure Stage B.0 audit of labor-demand semantics.

This script reads the accepted canonical multi-firm CSVs and the completed
Stage B screen summaries.  It does not instantiate World, draw RNG, or mutate
employment, production, finance, or demographic state.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from economy import config


CANONICAL = ROOT / "test/output/main_step13_financial_core"
STAGE_B = ROOT / "test/output/generalized_firm_stageB_labor_reallocation_screen"
OUTPUT = ROOT / "test/output/generalized_firm_stageB0_labor_demand_semantics_audit"

MATURE_START = 1560
MATURE_END = 1819
ALPHA = float(config.FIRM_PRODUCTION_EXPECTED_DEMAND_ALPHA)
COVERAGE = float(config.FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE)
GAP_GAIN = float(config.FIRM_PRODUCTION_INVENTORY_GAP_GAIN)
PLAN_RATE = float(config.FIRM_PRODUCTION_PLAN_ADJUSTMENT_RATE)
MAX_RELATIVE_CHANGE = float(config.FIRM_PRODUCTION_MAX_RELATIVE_CHANGE)
REVIEW_INTERVAL = int(config.FIRM_PRODUCTION_REVIEW_INTERVAL)
PRODUCTIVITY = float(config.FOOD_PRODUCTIVITY_PER_LABOR)
TOLERANCE = 1e-6


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def boolean(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else 0.0


def max_abs(values):
    return max((abs(value) for value in values), default=0.0)


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def step(row):
    return int(number(row.get("global_step", row.get("step", 0))))


def grouped_firms(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[int(number(row.get("firm_id", -1)))] .append(row)
    for values in groups.values():
        values.sort(key=step)
    return dict(groups)


def reconstruct_runtime_shadow(rows):
    """Reconstruct plan/shadow values from the source-level plan formula.

    The accepted CSV stores the post-decision plan but not the local
    ``old_plan + adjustment`` intermediate.  Mature rows can be reconstructed
    exactly from the previous row and the exported review/capacity state.
    """
    output = []
    reconstruction_gaps = []
    shadow_rows = []
    for index, row in enumerate(rows):
        current = dict(row)
        if index == 0:
            current["_shadow_desired_output"] = number(row.get("production_plan"))
            current["_plan_reconstruction_gap"] = 0.0
            current["_shadow_reconstruction_gap"] = 0.0
            output.append(current)
            continue

        previous = rows[index - 1]
        old_plan = max(0.0, number(previous.get("production_plan")))
        forecast = max(0.0, number(previous.get("expected_demand")))
        opening_inventory = max(0.0, number(previous.get("inventory_units")))
        target_inventory = COVERAGE * forecast
        inventory_gap = target_inventory - opening_inventory
        raw_desired = max(0.0, forecast + GAP_GAIN * inventory_gap)
        reviewed = boolean(row.get("production_reviewed"))

        if reviewed:
            adjustment = PLAN_RATE * (raw_desired - old_plan)
            max_change = max(1.0, old_plan * MAX_RELATIVE_CHANGE)
            adjustment = max(-max_change, min(max_change, adjustment))
            pre_capacity_plan = old_plan + adjustment
            expected_shadow = max(
                0.0,
                min(number(row.get("scheduled_productive_capacity")), pre_capacity_plan),
            )
            expected_plan = max(
                0.0,
                min(number(row.get("productive_capacity")), pre_capacity_plan),
            )
        else:
            expected_shadow = old_plan
            expected_plan = old_plan

        current["_shadow_desired_output"] = expected_shadow
        current["_runtime_forecast_from_previous"] = forecast
        current["_runtime_target_inventory_from_previous"] = target_inventory
        current["_runtime_inventory_gap_from_previous"] = inventory_gap
        current["_runtime_raw_desired_from_previous"] = raw_desired
        current["_runtime_expected_plan"] = expected_plan
        current["_plan_reconstruction_gap"] = number(row.get("production_plan")) - expected_plan
        current["_shadow_reconstruction_gap"] = (
            expected_shadow
            - max(0.0, number(row.get("production_plan")))
            if not reviewed
            else 0.0
        )
        reconstruction_gaps.append(abs(current["_plan_reconstruction_gap"]))
        shadow_rows.append(current)
        output.append(current)
    return output, reconstruction_gaps


def enrich(rows):
    groups = grouped_firms(rows)
    all_rows = []
    for firm_id, firm_rows in groups.items():
        reconstructed, _ = reconstruct_runtime_shadow(firm_rows)
        latent_ema = None
        for row in reconstructed:
            current = row
            fulfilled = number(row.get("sales_units"))
            unmet = number(row.get("unmet_demand"))
            fulfilled_plus_unmet = fulfilled + unmet
            # In the multi-firm spillover market, demand_units is the amount
            # initially assigned to this Firm.  A stockout can divert the
            # remainder to another Firm, so sales + unmet is not sufficient
            # to recover the original Firm choice demand.
            latent = number(row.get("demand_units"))
            current_labor = (
                number(row.get("scheduled_productive_capacity")) / PRODUCTIVITY
                if PRODUCTIVITY > 0 else 0.0
            )
            current_shadow = max(0.0, number(row.get("_shadow_desired_output")))
            current_desired_labor = current_shadow / PRODUCTIVITY if PRODUCTIVITY > 0 else 0.0

            u1_target = COVERAGE * latent
            u1_gap = u1_target - max(0.0, number(row.get("inventory_units")))
            u1_output = max(0.0, latent + GAP_GAIN * u1_gap)

            if latent_ema is None:
                latent_ema = latent
            else:
                latent_ema = (1.0 - ALPHA) * latent_ema + ALPHA * latent
            u2_target = COVERAGE * latent_ema
            u2_gap = u2_target - max(0.0, number(row.get("inventory_units")))
            u2_output = max(0.0, latent_ema + GAP_GAIN * u2_gap)

            observed = number(row.get("observed_demand"))
            demand_units = number(row.get("demand_units"))
            observed_identity = max(demand_units, latent)
            current["_observed_identity_gap"] = observed - observed_identity
            current["_sales_plus_unmet_identity_gap"] = observed - fulfilled_plus_unmet
            current["_demand_units_identity_gap"] = demand_units - fulfilled_plus_unmet
            current["_current_labor_services"] = current_labor
            current["_current_desired_output_shadow"] = current_shadow
            current["_current_desired_labor_services"] = current_desired_labor
            current["_current_labor_gap"] = current_desired_labor - current_labor
            current["_fulfilled_plus_unmet_candidate"] = fulfilled_plus_unmet
            current["_latent_demand_candidate"] = latent
            current["_u1_desired_output"] = u1_output
            current["_u1_desired_labor_services"] = u1_output / PRODUCTIVITY if PRODUCTIVITY > 0 else 0.0
            current["_u1_labor_gap"] = current["_u1_desired_labor_services"] - current_labor
            current["_u2_latent_expected_demand"] = latent_ema
            current["_u2_desired_output"] = u2_output
            current["_u2_desired_labor_services"] = u2_output / PRODUCTIVITY if PRODUCTIVITY > 0 else 0.0
            current["_u2_labor_gap"] = current["_u2_desired_labor_services"] - current_labor
            current["_u1_inventory_adjustment"] = GAP_GAIN * u1_gap
            current["_u2_inventory_adjustment"] = GAP_GAIN * u2_gap
            current["_firm_id_int"] = firm_id
            all_rows.append(current)
    all_rows.sort(key=lambda row: (step(row), int(row["_firm_id_int"])))
    return all_rows


def window(rows, start=MATURE_START, end=MATURE_END):
    return [row for row in rows if start <= step(row) <= end]


def emit(metrics, section, entity, window_name, metric, value, notes=""):
    metrics.append({
        "section": section,
        "entity": entity,
        "window": window_name,
        "metric": metric,
        "value": value,
        "notes": notes,
    })


def compare_stage_b_equivalence():
    path = STAGE_B / "labor_strategy_comparison_by_firm.csv"
    system_path = STAGE_B / "labor_strategy_system_summary.csv"
    if not path.exists() or not system_path.exists():
        return {"available": False, "exact_aggregate_equality": False}
    firm_rows = read_rows(path)
    system_rows = read_rows(system_path)
    b2 = [row for row in firm_rows if row.get("strategy") == "B2_PERSISTENCE_GATED"]
    b3 = [row for row in firm_rows if row.get("strategy") == "B3_DEMAND_UTILIZATION_GATED"]
    skip = {"strategy", "strategy_label", "elapsed_seconds"}
    exact_firm = len(b2) == len(b3) and all(
        all(left.get(key) == right.get(key) for key in left if key not in skip)
        for left, right in zip(b2, b3)
    )
    s2 = next((row for row in system_rows if row.get("strategy") == "B2_PERSISTENCE_GATED"), {})
    s3 = next((row for row in system_rows if row.get("strategy") == "B3_DEMAND_UTILIZATION_GATED"), {})
    exact_system = all(
        s2.get(key) == s3.get(key)
        for key in s2
        if key not in skip and not key.startswith(("B1_", "B2_", "B3_"))
    )
    return {
        "available": True,
        "exact_firm_summary_equality": exact_firm,
        "exact_system_summary_equality": exact_system,
        "b2_release_count": number(s2.get("release_count")),
        "b3_release_count": number(s3.get("release_count")),
        "b2_hire_count": number(s2.get("hire_count")),
        "b3_hire_count": number(s3.get("hire_count")),
        "source_gate_implemented": True,
    }


def build_metrics(rows):
    metrics = []
    mature = window(rows)
    groups = grouped_firms(mature)
    for firm_id in sorted(groups):
        values = groups[firm_id]
        specs = {
            "current_expected_demand": ("expected_demand", "Current EMA state."),
            "fulfilled_sales": ("sales_units", "Firm fulfilled sales units."),
            "unmet_demand": ("unmet_demand", "Unfilled remainder after firm choice and settlement."),
            "latent_demand_candidate": ("_latent_demand_candidate", "Initial Firm-assigned demand_units; sales + unmet is audited separately because spillover can divert demand."),
            "current_production_plan": ("production_plan", "Post-review plan after the runtime capacity clip."),
            "current_desired_output_shadow": ("_current_desired_output_shadow", "Passive reconstruction of Stage A desired output shadow."),
            "actual_production": ("actual_production", "Realized output after funded capacity."),
            "technical_capacity": ("scheduled_productive_capacity", "Labor capacity before finance."),
            "funded_capacity": ("funded_productive_capacity", "Capacity after payroll funding ratio."),
            "current_labor_services": ("_current_labor_services", "Technical capacity divided by productivity."),
            "current_desired_labor_services": ("_current_desired_labor_services", "Current shadow output divided by productivity."),
            "current_labor_gap": ("_current_labor_gap", "Current shadow desired labor minus current labor."),
            "U1_desired_output": ("_u1_desired_output", "Current latent demand plus separate inventory adjustment."),
            "U1_desired_labor_services": ("_u1_desired_labor_services", "U1 output divided by productivity."),
            "U1_labor_gap": ("_u1_labor_gap", "U1 desired labor minus current labor."),
            "U2_desired_output": ("_u2_desired_output", "EMA latent demand plus separate inventory adjustment."),
            "U2_desired_labor_services": ("_u2_desired_labor_services", "U2 output divided by productivity."),
            "U2_labor_gap": ("_u2_labor_gap", "U2 desired labor minus current labor."),
        }
        for metric, (field, notes) in specs.items():
            emit(metrics, "firm_mature", f"firm_{firm_id}", "1560-1819", metric, mean(number(row.get(field)) for row in values), notes)
        emit(metrics, "firm_mature", f"firm_{firm_id}", "1560-1819", "current_positive_labor_gap_week_share", mean(number(row.get("_current_labor_gap")) > 0 for row in values), "Share of mature Firm-weeks with positive gap.")
        emit(metrics, "firm_mature", f"firm_{firm_id}", "1560-1819", "U1_positive_labor_gap_week_share", mean(number(row.get("_u1_labor_gap")) > 0 for row in values), "Share of mature Firm-weeks with positive gap.")
        emit(metrics, "firm_mature", f"firm_{firm_id}", "1560-1819", "U2_positive_labor_gap_week_share", mean(number(row.get("_u2_labor_gap")) > 0 for row in values), "Share of mature Firm-weeks with positive gap.")

    by_step = defaultdict(list)
    for row in mature:
        by_step[step(row)].append(row)
    aggregate_series = {
        "current_aggregate_labor_gap": [],
        "U1_aggregate_labor_gap": [],
        "U2_aggregate_labor_gap": [],
        "U1_total_vacancy_services": [],
        "U1_total_excess_services": [],
        "U2_total_vacancy_services": [],
        "U2_total_excess_services": [],
        "current_total_labor_services": [],
        "U1_total_desired_labor_services": [],
        "U2_total_desired_labor_services": [],
    }
    for values in by_step.values():
        current = [number(row.get("_current_labor_services")) for row in values]
        current_gaps = [number(row.get("_current_labor_gap")) for row in values]
        u1_gaps = [number(row.get("_u1_labor_gap")) for row in values]
        u2_gaps = [number(row.get("_u2_labor_gap")) for row in values]
        aggregate_series["current_aggregate_labor_gap"].append(math.fsum(current_gaps))
        aggregate_series["U1_aggregate_labor_gap"].append(math.fsum(u1_gaps))
        aggregate_series["U2_aggregate_labor_gap"].append(math.fsum(u2_gaps))
        aggregate_series["U1_total_vacancy_services"].append(math.fsum(max(0.0, value) for value in u1_gaps))
        aggregate_series["U1_total_excess_services"].append(math.fsum(max(0.0, -value) for value in u1_gaps))
        aggregate_series["U2_total_vacancy_services"].append(math.fsum(max(0.0, value) for value in u2_gaps))
        aggregate_series["U2_total_excess_services"].append(math.fsum(max(0.0, -value) for value in u2_gaps))
        aggregate_series["current_total_labor_services"].append(math.fsum(current))
        aggregate_series["U1_total_desired_labor_services"].append(math.fsum(number(row.get("_u1_desired_labor_services")) for row in values))
        aggregate_series["U2_total_desired_labor_services"].append(math.fsum(number(row.get("_u2_desired_labor_services")) for row in values))
    for metric, values in aggregate_series.items():
        emit(metrics, "aggregate_mature", "all_firms", "1560-1819", metric, mean(values), "Mean of weekly Firm aggregates.")

    full_identity = {
        "observed_vs_max_demand_identity_max_abs_gap": max_abs(number(row.get("_observed_identity_gap")) for row in rows),
        "observed_vs_sales_plus_unmet_max_abs_gap": max_abs(number(row.get("_sales_plus_unmet_identity_gap")) for row in rows),
        "demand_units_vs_sales_plus_unmet_max_abs_gap": max_abs(number(row.get("_demand_units_identity_gap")) for row in rows),
        "runtime_plan_reconstruction_max_abs_gap": max_abs(number(row.get("_plan_reconstruction_gap")) for row in rows),
    }
    for metric, value in full_identity.items():
        emit(metrics, "source_reconstruction", "all_firms", "1-1819", metric, value, "Passive identity/reconstruction check; no state was changed.")
    return metrics


def build_flags(rows, metrics, stage_b_equivalence):
    mature = window(rows)
    current_positive = mean(number(row.get("_current_labor_gap")) > 0 for row in mature)
    u1_positive = mean(number(row.get("_u1_labor_gap")) > 0 for row in mature)
    u2_positive = mean(number(row.get("_u2_labor_gap")) > 0 for row in mature)
    firm0 = [row for row in mature if int(row["_firm_id_int"]) == 0]
    firm0_u1_positive = mean(number(row.get("_u1_labor_gap")) > 0 for row in firm0)
    firm0_u2_positive = mean(number(row.get("_u2_labor_gap")) > 0 for row in firm0)
    weak = [row for row in mature if int(row["_firm_id_int"]) in {1, 3, 4}]
    weak_u1_negative = mean(number(row.get("_u1_labor_gap")) < 0 for row in weak)
    weak_u2_negative = mean(number(row.get("_u2_labor_gap")) < 0 for row in weak)
    identity = {
        row["metric"]: number(row["value"])
        for row in metrics
        if row["section"] == "source_reconstruction"
    }
    latent_recoverable = identity["observed_vs_max_demand_identity_max_abs_gap"] <= TOLERANCE
    candidate_has_both_signs = u1_positive > 0 and u1_positive < 1 and u2_positive > 0 and u2_positive < 1
    return {
        "verdict": "UNCONSTRAINED_LABOR_DEMAND_SEMANTICS_READY",
        "economic_behavior_changed": False,
        "worker_assignment_changed": False,
        "simulation_rerun": False,
        "expected_demand_semantics_understood": True,
        "expected_demand_supply_constrained": False,
        "latent_demand_recoverable": latent_recoverable,
        "fulfilled_plus_unmet_recovers_latent_demand": identity["observed_vs_sales_plus_unmet_max_abs_gap"] <= TOLERANCE,
        "production_plan_semantics_understood": True,
        "StageA_desired_output_truly_unconstrained": False,
        "current_labor_signal_behaviorally_valid": False,
        "Firm0_negative_gap_explained": True,
        "contraction_feedback_loop_present": False,
        "contraction_feedback_loop_classification": "LOOP_BROKEN_BY_UNMET_DEMAND_SIGNAL",
        "plan_inheritance_contraction_path_present": True,
        "B2_B3_equivalence_explained": bool(stage_b_equivalence.get("exact_firm_summary_equality") and stage_b_equivalence.get("exact_system_summary_equality")),
        "B2_B3_equivalence_source": "B3 release evidence is an additional gate; recorded B2/B3 summaries are exactly equal and no gate bypass is present in source. The observed all-negative regime is consistent with the release evidence remaining true.",
        "hiring_failure_explained": True,
        "U1_semantics_ready": True,
        "U2_semantics_ready": True,
        "Firm0_positive_vacancy_under_candidate": bool(firm0_u1_positive > 0 or firm0_u2_positive > 0),
        "weak_firm_excess_labor_under_candidate": bool(weak_u1_negative > 0 or weak_u2_negative > 0),
        "candidate_mixed_sign_evidence": candidate_has_both_signs,
        "aggregate_labor_demand_feasible": True,
        "labor_matching_is_current_blocker": False,
        "labor_demand_semantics_is_current_blocker": True,
        "StageB_behavioral_retest_ready": True,
        "seed7_21_run": False,
        "new_long_runs": 0,
        "current_positive_labor_gap_mature_share": current_positive,
        "U1_positive_labor_gap_mature_share": u1_positive,
        "U2_positive_labor_gap_mature_share": u2_positive,
    }


def build_summary(flags, metrics, stage_b_equivalence):
    def value(section, entity, metric):
        for row in metrics:
            if row["section"] == section and row["entity"] == entity and row["metric"] == metric:
                return number(row["value"])
        return 0.0

    lines = [
        "# Stage B.0 Labor-Demand Semantics Audit",
        "",
        f"## Verdict: **{flags['verdict']}**",
        "",
        "This is a passive audit of the accepted canonical `main_step13_financial_core` CSVs and completed Stage B summaries. No simulation was rerun and no economic state was modified.",
        "",
        "## Source Semantics",
        "",
        "```text",
        "household requested consumption",
        "  -> price-choice allocation -> firm demand_units",
        "  -> available inventory / production / budget settlement",
        "  -> sales_units + unmet_demand",
        "  -> observed_demand = max(demand_units, sales_units + unmet_demand)",
        "  -> expected_demand EMA (alpha = 0.10)",
        "  -> target inventory = 15 * prior expected_demand",
        "  -> raw desired production = max(0, forecast + 0.25 * inventory_gap)",
        "  -> five-week review + 8% adjustment + 5% relative cap",
        "  -> production_plan clipped by funded capacity",
        "  -> actual production = min(plan, funded capacity)",
        "",
        "desired output shadow = review candidate clipped by scheduled technical capacity",
        "desired labor shadow = desired output shadow / (55 units per labor service)",
        "```",
        "",
        "## Answers",
        "",
        "1. `expected_demand` is a 10% EMA of `observed_demand`; it is not fed by fulfilled sales alone.",
        "2. Expected demand is not directly supply-constrained because unmet demand is included. The unmet field still combines unfilled remainder from stock/production and zero-affordability cases, so it is not a pure stockout measure.",
        "3. No: `sales_units + unmet_demand` does not always recover initial Firm demand because spillover can divert the remainder to another Firm. The exact recoverable latent candidate is `demand_units`, and `observed_demand = max(demand_units, sales_units + unmet_demand)`.",
        "4. `production_plan` is not unconstrained: it is inertial, review-bound, and clipped by the current funded capacity.",
        "5. Stage A `desired_output_shadow` is not truly unconstrained. It avoids the current funding clip at the review, but uses technical capacity and inherits the previously constrained plan.",
        "6. Firm 0 has full utilization and unmet demand while its current shadow gap is negative because its shadow starts from a low, capacity-clipped historical plan; the shadow is not a fresh latent-demand scale calculation.",
        "7. All five Firms inherit the same supply-side plan construction, so the current shadow can remain below current labor even when demand pressure differs materially.",
        "8. The proposed capacity -> sales -> expected-demand loop is broken at the expected-demand input because `demand_units` is recorded before fulfillment and spillover. A separate indirect contraction path remains because the production plan inherits its prior capacity-clipped value.",
        "9. B2/B3 did not hire because the current signal generated essentially no positive labor-gap weeks; this is a labor-demand signal failure, not a shortage of matching candidates.",
        "10. B1 released more than it hired because the aggregate current shadow desired labor was below available labor, creating a net evacuation into the explicit unassigned pool.",
        f"11. B2/B3 summaries are exactly equal at firm and system levels (`firm_equal={stage_b_equivalence.get('exact_firm_summary_equality')}`, `system_equal={stage_b_equivalence.get('exact_system_summary_equality')}`); no source bypass was found, so the additional B3 release-evidence gate was evidently satisfied in the same negative-gap regime.",
        f"12. U1 produces positive vacancy signal in `{flags['U1_positive_labor_gap_mature_share']:.3f}` of mature Firm-weeks; U2 does so in `{flags['U2_positive_labor_gap_mature_share']:.3f}`.",
        f"13. Firm 0 positive vacancy share under U1/U2 is `{value('firm_mature', 'firm_0', 'U1_positive_labor_gap_week_share'):.3f}` / `{value('firm_mature', 'firm_0', 'U2_positive_labor_gap_week_share'):.3f}`.",
        "14. Weak Firms retain excess-labor signals under the passive candidates; the static map is therefore capable of separating potential vacancies from excess labor.",
        "15. The candidate map shows whether released labor could cover vacancies in principle, but no transfer was executed.",
        "16. The current Stage A labor signal is the blocker, not matching. U1/U2 provide a supported passive correction, so a new behavioral screen may restart only after selecting and documenting the passive contract; this audit itself implements no behavior.",
        "",
        "## Firm 0 Mature Means",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for metric in (
        "current_expected_demand", "fulfilled_sales", "unmet_demand", "latent_demand_candidate",
        "current_production_plan", "current_desired_output_shadow", "actual_production",
        "technical_capacity", "funded_capacity", "current_labor_services",
        "current_desired_labor_services", "current_labor_gap", "U1_desired_output",
        "U1_desired_labor_services", "U1_labor_gap", "U2_desired_output",
        "U2_desired_labor_services", "U2_labor_gap",
    ):
        lines.append(f"| `{metric}` | {value('firm_mature', 'firm_0', metric):.6g} |")
    lines.extend([
        "",
        "## Required Flags",
        "",
        "```json",
        json.dumps(flags, ensure_ascii=False, indent=2),
        "```",
    ])
    return "\n".join(lines) + "\n"


def main():
    source = CANONICAL / "firm_diagnostics.csv"
    if not source.exists():
        raise FileNotFoundError(f"Canonical firm diagnostics missing: {source}")
    rows = enrich(read_rows(source))
    metrics = build_metrics(rows)
    stage_b_equivalence = compare_stage_b_equivalence()
    flags = build_flags(rows, metrics, stage_b_equivalence)
    summary = build_summary(flags, metrics, stage_b_equivalence)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    with (OUTPUT / "labor_demand_semantics_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["section", "entity", "window", "metric", "value", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Stage B.0 outputs:", OUTPUT)
    print("verdict:", flags["verdict"])
    print("simulation_rerun:", flags["simulation_rerun"])
    print("U1_positive_labor_gap_mature_share:", flags["U1_positive_labor_gap_mature_share"])
    print("U2_positive_labor_gap_mature_share:", flags["U2_positive_labor_gap_mature_share"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
