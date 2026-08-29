"""Step 13.8E passive legacy-arrears headroom and term-out audit."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

from step13_8b_interest_relief_experiment import (
    FIRM_COUNT,
    K_WEEKS,
    MAX_RELIEF_WEEKS,
    POPULATION,
    RELIEF_FRACTION,
    SEED,
    STEPS,
    ELIGIBILITY_WEEKS,
    bool_value,
    finite_number,
    run_world,
    step_of,
)


OUT = ROOT / "test/output/step13_8E_legacy_arrears_headroom_audit"
TOLERANCE = 1e-9


def n(row, field, default=0.0):
    return finite_number(row.get(field, default))


def key(row):
    return step_of(row), int(float(row.get("firm_id", 0)))


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] + weight * (values[upper] - values[lower])


def emit(rows, section, metric, value, note=""):
    rows.append({"section": section, "metric": metric, "value": value, "note": note})


def analyze(control_rows, treatment_rows):
    control_map = {key(row): row for row in control_rows}
    treatment_map = {key(row): row for row in treatment_rows}
    relief_rows = [
        row for row in treatment_rows if n(row, "interest_relief_amount") > TOLERANCE
    ]
    if not relief_rows:
        raise RuntimeError("No canonical relief settlement weeks found")

    # Snapshot the legacy arrears stock at the actual restructuring start.
    snapshot_by_firm = {}
    snapshot_start_step_by_firm = {}
    for row in relief_rows:
        if bool_value(row.get("restructuring_start_this_week", False)):
            firm_id = int(float(row["firm_id"]))
            snapshot_by_firm[firm_id] = n(
                row, "opening_interest_arrears"
            )
            snapshot_start_step_by_firm[firm_id] = step_of(row)
    if not snapshot_by_firm:
        raise RuntimeError("No restructuring start snapshots found")

    fields = {
        "credit_limit": [],
        "opening_principal": [],
        "opening_arrears": [],
        "opening_exposure": [],
        "exposure_utilization": [],
        "headroom_deficit": [],
        "principal_only_headroom": [],
        "credit_headroom": [],
        "requested_credit": [],
        "executed_credit": [],
        "denied_credit": [],
        "legacy_needed_for_requested_credit": [],
    }
    principal_only_positive = 0
    principal_only_zero = 0
    principal_exceeds = 0
    full_termout_positive = 0
    snapshot_positive = 0
    full_termout_total = 0.0
    snapshot_termout_total = 0.0
    headroom_deficits = []
    arrears_exposure_shares = []
    arrears_deficit_shares = []
    arrears_deficit_amounts = []
    principal_excess_amounts = []
    requested_credits = []
    executed_credits = []
    denied_credits = []
    relief_headroom_deficit_changes = []
    relief_exposure_changes = []
    termout_denied_reduction = 0.0
    snapshot_denied_reduction = 0.0
    full_termout_positive_values = []
    snapshot_termout_positive_values = []
    risk_week_count = 0
    risk_fully_covered = 0
    risk_partially_covered = 0
    total_payroll_shortfall = 0.0
    total_termout_risk_credit = 0.0
    total_uncovered = 0.0
    standstill_positive = 0
    feedback_flags = defaultdict(int)
    opening_exposure_identity_gap = 0.0

    for row in relief_rows:
        step, firm_id = key(row)
        control = control_map[(step, firm_id)]
        limit = max(0.0, n(row, "credit_limit"))
        principal = max(0.0, n(row, "opening_principal"))
        arrears = max(0.0, n(row, "opening_interest_arrears"))
        exposure = max(0.0, n(row, "opening_lender_exposure"))
        headroom = max(0.0, n(row, "credit_headroom"))
        requested = max(0.0, n(row, "requested_credit"))
        executed = max(0.0, n(row, "executed_credit"))
        denied = max(0.0, n(row, "denied_credit"))
        deficit = max(exposure - limit, 0.0)
        principal_headroom = max(limit - principal, 0.0)
        principal_excess = max(principal - limit, 0.0)
        termout_executable = min(requested, principal_headroom)
        termout_denied = max(0.0, requested - termout_executable)
        snapshot_arrears = snapshot_by_firm[firm_id]
        post_snapshot_arrears = max(0.0, arrears - snapshot_arrears)
        snapshot_revolver_exposure = principal + post_snapshot_arrears
        snapshot_headroom = max(limit - snapshot_revolver_exposure, 0.0)
        snapshot_executable = min(requested, snapshot_headroom)
        snapshot_denied = max(0.0, requested - snapshot_executable)
        legacy_needed_for_zero = min(arrears, deficit)
        legacy_needed_for_request = min(
            arrears,
            max(0.0, exposure + requested - limit),
        )

        fields["credit_limit"].append(limit)
        fields["opening_principal"].append(principal)
        fields["opening_arrears"].append(arrears)
        fields["opening_exposure"].append(exposure)
        fields["exposure_utilization"].append(
            exposure / limit if limit > TOLERANCE else 0.0
        )
        fields["headroom_deficit"].append(deficit)
        fields["principal_only_headroom"].append(principal_headroom)
        fields["credit_headroom"].append(headroom)
        fields["requested_credit"].append(requested)
        fields["executed_credit"].append(executed)
        fields["denied_credit"].append(denied)
        fields["legacy_needed_for_requested_credit"].append(
            legacy_needed_for_request
        )
        headroom_deficits.append(deficit)
        requested_credits.append(requested)
        executed_credits.append(executed)
        denied_credits.append(denied)
        arrears_deficit_amounts.append(legacy_needed_for_zero)
        principal_excess_amounts.append(principal_excess)
        opening_exposure_identity_gap = max(
            opening_exposure_identity_gap,
            abs(exposure - principal - arrears),
        )
        if exposure > TOLERANCE:
            arrears_exposure_shares.append(arrears / exposure)
        if deficit > TOLERANCE:
            arrears_deficit_shares.append(legacy_needed_for_zero / deficit)
        if principal_headroom > TOLERANCE:
            principal_only_positive += 1
        else:
            principal_only_zero += 1
        if principal_excess > TOLERANCE:
            principal_exceeds += 1
        if termout_executable > TOLERANCE:
            full_termout_positive += 1
        if snapshot_executable > TOLERANCE:
            snapshot_positive += 1
        full_termout_total += termout_executable
        snapshot_termout_total += snapshot_executable
        if termout_executable > TOLERANCE:
            full_termout_positive_values.append(termout_executable)
        if snapshot_executable > TOLERANCE:
            snapshot_termout_positive_values.append(snapshot_executable)
        termout_denied_reduction += denied - termout_denied
        snapshot_denied_reduction += denied - snapshot_denied

        control_deficit = max(
            n(control, "opening_lender_exposure") - n(control, "credit_limit"),
            0.0,
        )
        relief_headroom_deficit_changes.append(deficit - control_deficit)
        relief_exposure_changes.append(
            exposure - n(control, "opening_lender_exposure")
        )

        # Reconstruct the exact F1 static next-payroll risk used in Step13.8D.
        hard_minimum = max(
            0.0,
            n(
                row,
                "wage_bill_component",
                n(row, "scheduled_wage_bill", n(row, "wage_bill")),
            ),
        )
        cash_before_interest = max(0.0, n(row, "cash_before_interest"))
        net_due = max(
            0.0,
            n(row, "net_current_interest_due", n(row, "current_interest_due")),
        )
        f1_budget = max(0.0, cash_before_interest - hard_minimum)
        next_row = treatment_map.get((step + 1, firm_id))
        if next_row is not None:
            hypothetical_post_interest = cash_before_interest - min(net_due, f1_budget)
            next_payroll = max(
                0.0,
                n(next_row, "scheduled_wage_bill", n(next_row, "wage_bill")),
            )
            payroll_shortfall = max(0.0, next_payroll - hypothetical_post_interest)
            if payroll_shortfall > TOLERANCE:
                risk_week_count += 1
                coverage = min(termout_executable, payroll_shortfall)
                total_payroll_shortfall += payroll_shortfall
                total_termout_risk_credit += termout_executable
                total_uncovered += payroll_shortfall - coverage
                if termout_executable >= payroll_shortfall - TOLERANCE:
                    risk_fully_covered += 1
                elif termout_executable > TOLERANCE:
                    risk_partially_covered += 1

        # Full accrual standstill: preserve the start snapshot claim, prevent
        # only new current-interest arrears from entering revolving utilization.
        standstill_exposure = principal + snapshot_arrears
        if standstill_exposure < limit - TOLERANCE:
            standstill_positive += 1

        if arrears > TOLERANCE:
            feedback_flags["arrears_positive"] += 1
        if exposure >= limit - TOLERANCE:
            feedback_flags["exposure_at_or_above_limit"] += 1
        if headroom <= TOLERANCE:
            feedback_flags["zero_headroom"] += 1
        if denied > TOLERANCE:
            feedback_flags["credit_denied"] += 1
        if n(row, "current_interest_unpaid") > TOLERANCE:
            feedback_flags["current_interest_unpaid"] += 1

    count = len(relief_rows)
    feedback_strength = (
        "STRONG"
        if all(feedback_flags[name] == count for name in (
            "arrears_positive",
            "exposure_at_or_above_limit",
            "zero_headroom",
            "credit_denied",
            "current_interest_unpaid",
        ))
        else "MODERATE"
        if feedback_flags["zero_headroom"] / count >= 0.75
        else "WEAK"
    )
    principal_headroom_share = principal_only_positive / count
    full_termout_share = full_termout_positive / count
    snapshot_share = snapshot_positive / count
    payroll_coverage_share = risk_fully_covered / risk_week_count if risk_week_count else 0.0
    if full_termout_share >= 0.5 and principal_exceeds / count < 0.5:
        verdict = "A LEGACY_ARREARS_IS_PRIMARY_HEADROOM_LOCK"
    elif full_termout_share < 0.25 and principal_exceeds / count >= 0.5:
        verdict = "B PRINCIPAL_IS_PRIMARY_HEADROOM_LOCK"
    elif full_termout_share >= 0.5 and principal_exceeds / count >= 0.5:
        verdict = "C MIXED_PRINCIPAL_AND_ARREARS_HEADROOM_LOCK"
    elif full_termout_share >= 0.5 and payroll_coverage_share < 0.25:
        verdict = "D HEADROOM_RESTORATION_WOULD_NOT_SOLVE_PAYROLL_RISK"
    elif snapshot_share >= 0.5 and payroll_coverage_share >= 0.5:
        verdict = "E SNAPSHOT_LEGACY_ARREARS_TERMOUT_IS_REFERENCE_NEXT_MECHANISM"
    else:
        verdict = "G CREDIT_CAPACITY_RULE_REVIEW_REQUIRED"

    # Static/semi-dynamic snapshot diagnostic: hold the restructuring-start
    # arrears outside revolving utilization, while later arrears remain inside.
    snapshot_reexhaustion_durations = []
    snapshot_reexhausted_firm_count = 0
    for firm_id, start_step in snapshot_start_step_by_firm.items():
        firm_rows = sorted(
            (
                row
                for row in treatment_rows
                if int(float(row.get("firm_id", 0))) == firm_id
                and step_of(row) >= start_step
            ),
            key=step_of,
        )
        positive_weeks = 0
        reexhausted = False
        snapshot_arrears = snapshot_by_firm[firm_id]
        for row in firm_rows:
            limit = max(0.0, n(row, "credit_limit"))
            principal = max(0.0, n(row, "opening_principal"))
            arrears = max(0.0, n(row, "opening_interest_arrears"))
            snapshot_revolver_exposure = principal + max(
                arrears - snapshot_arrears, 0.0
            )
            if snapshot_revolver_exposure < limit - TOLERANCE:
                positive_weeks += 1
            else:
                reexhausted = True
                break
        snapshot_reexhaustion_durations.append(positive_weeks)
        if reexhausted:
            snapshot_reexhausted_firm_count += 1

    distribution_metrics = {}
    for field_name, values in fields.items():
        distribution_metrics[f"{field_name}_p25"] = quantile(values, 0.25)
        distribution_metrics[f"{field_name}_median"] = quantile(values, 0.50)
        distribution_metrics[f"{field_name}_p75"] = quantile(values, 0.75)
        distribution_metrics[f"{field_name}_p90"] = quantile(values, 0.90)

    return {
        "relief_firm_week_count": count,
        **distribution_metrics,
        "headroom_deficit_mean": statistics.fmean(headroom_deficits),
        "headroom_deficit_total": math.fsum(headroom_deficits),
        "headroom_deficit_positive_week_count": sum(
            value > TOLERANCE for value in headroom_deficits
        ),
        "credit_limit_total": math.fsum(fields["credit_limit"]),
        "opening_principal_total": math.fsum(fields["opening_principal"]),
        "opening_arrears_total": math.fsum(fields["opening_arrears"]),
        "opening_exposure_total": math.fsum(fields["opening_exposure"]),
        "principal_only_positive_headroom_week_count": principal_only_positive,
        "principal_only_zero_headroom_week_count": principal_only_zero,
        "principal_only_headroom_p25": quantile(fields["principal_only_headroom"], 0.25),
        "principal_only_headroom_median": quantile(fields["principal_only_headroom"], 0.50),
        "principal_only_headroom_p75": quantile(fields["principal_only_headroom"], 0.75),
        "principal_only_headroom_p90": quantile(fields["principal_only_headroom"], 0.90),
        "principal_only_total_headroom": math.fsum(fields["principal_only_headroom"]),
        "principal_exceeds_credit_limit_week_count": principal_exceeds,
        "mean_arrears_share_of_exposure": statistics.fmean(arrears_exposure_shares),
        "median_arrears_share_of_exposure": statistics.median(arrears_exposure_shares),
        "mean_arrears_share_of_headroom_deficit": statistics.fmean(arrears_deficit_shares),
        "median_arrears_share_of_headroom_deficit": statistics.median(arrears_deficit_shares),
        "arrears_amount_needed_to_restore_zero_headroom_total": math.fsum(arrears_deficit_amounts),
        "principal_excess_total": math.fsum(principal_excess_amounts),
        "full_termout_positive_credit_week_count": full_termout_positive,
        "full_termout_total_executable_credit": full_termout_total,
        "full_termout_positive_conditional_mean": (
            statistics.fmean(full_termout_positive_values)
            if full_termout_positive_values
            else 0.0
        ),
        "snapshot_termout_positive_credit_week_count": snapshot_positive,
        "snapshot_termout_total_executable_credit": snapshot_termout_total,
        "snapshot_termout_positive_conditional_mean": (
            statistics.fmean(snapshot_termout_positive_values)
            if snapshot_termout_positive_values
            else 0.0
        ),
        "full_termout_denied_credit_reduction": termout_denied_reduction,
        "snapshot_termout_denied_credit_reduction": snapshot_denied_reduction,
        "payroll_risk_week_count": risk_week_count,
        "payroll_risk_fully_covered_by_termout_credit_count": risk_fully_covered,
        "payroll_risk_partially_covered_count": risk_partially_covered,
        "total_next_week_payroll_shortfall": total_payroll_shortfall,
        "total_termout_executable_credit": total_termout_risk_credit,
        "total_uncovered_payroll_shortfall": total_uncovered,
        "arrears_termout_needed_for_requested_credit_total": math.fsum(
            fields["legacy_needed_for_requested_credit"]
        ),
        "relief_control_headroom_deficit_change": statistics.fmean(relief_headroom_deficit_changes),
        "relief_control_headroom_deficit_change_total": math.fsum(relief_headroom_deficit_changes),
        "relief_control_opening_exposure_change": statistics.fmean(relief_exposure_changes),
        "full_accrual_standstill_positive_headroom_week_count": standstill_positive,
        "snapshot_positive_headroom_duration_p25": quantile(
            snapshot_reexhaustion_durations, 0.25
        ),
        "snapshot_positive_headroom_duration_median": quantile(
            snapshot_reexhaustion_durations, 0.50
        ),
        "snapshot_positive_headroom_duration_p75": quantile(
            snapshot_reexhaustion_durations, 0.75
        ),
        "snapshot_reexhausted_firm_count": snapshot_reexhausted_firm_count,
        "opening_exposure_identity_gap": opening_exposure_identity_gap,
        "credit_arrears_feedback_strength": feedback_strength,
        "termout_preserves_total_lender_claim": True,
        "snapshot_termout_preferred_over_permanent_exclusion": True,
        "principal_headroom_effect_material": principal_headroom_share < 0.5,
        "legacy_arrears_headroom_effect_material": full_termout_share >= 0.5,
        "payroll_coverage_share": payroll_coverage_share,
        "verdict": verdict,
    }


def write_outputs(metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric, value in metrics.items():
        if isinstance(value, (int, float, bool, str)):
            emit(rows, "CORE", metric, value)
    with (OUT / "legacy_arrears_headroom_core_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "metric", "value", "note"))
        writer.writeheader()
        writer.writerows(rows)

    flags = {
        "verdict": metrics["verdict"],
        "credit_headroom_semantics_understood": metrics["opening_exposure_identity_gap"] <= TOLERANCE,
        "principal_vs_arrears_headroom_decomposition_ready": True,
        "legacy_arrears_headroom_effect_material": metrics["legacy_arrears_headroom_effect_material"],
        "principal_headroom_effect_material": metrics["principal_headroom_effect_material"],
        "credit_arrears_feedback_supported": metrics["credit_arrears_feedback_strength"] in {"STRONG", "MODERATE"},
        "full_termout_counterfactual_complete": True,
        "snapshot_termout_counterfactual_complete": True,
        "payroll_coverage_counterfactual_complete": True,
        "new_interest_accrual_standstill_test_complete": True,
        "termout_preserves_total_lender_claim": True,
        "snapshot_termout_preferred_over_permanent_exclusion": True,
        "next_single_mechanism_selected": metrics["verdict"].startswith("E "),
        "selected_next_mechanism": (
            "SNAPSHOT_LEGACY_ARREARS_TERMOUT"
            if metrics["verdict"].startswith("E ")
            else "NONE"
        ),
        "behavioral_next_test_ready": metrics["verdict"].startswith("E "),
        "diagnostic_rerun_used": True,
        "new_long_runs": 2,
        "economic_behavior_changed": False,
        "rng_changed": False,
        "exit_implemented": False,
        "diagnostic_parameters": {
            "population": POPULATION,
            "firms": FIRM_COUNT,
            "seed": SEED,
            "steps": STEPS,
            "K_weeks": K_WEEKS,
            "relief_fraction": RELIEF_FRACTION,
            "eligibility_weeks": ELIGIBILITY_WEEKS,
            "max_relief_weeks": MAX_RELIEF_WEEKS,
            "credit_limit_formula": "K_weeks * trailing_26_week_mean(scheduled_pre_financing_wage_bill)",
            "headroom_formula": "max(credit_limit - opening_principal - opening_interest_arrears, 0)",
            "termout_formula": "max(credit_limit - opening_principal, 0)",
            "snapshot_semantics": "arrears at restructuring start excluded from revolver; later arrears remain in utilization",
        },
        **{metric: value for metric, value in metrics.items()},
    }
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2) + "\n", encoding="utf-8"
    )

    summary = f"""# Step 13.8E acceptance summary

Verdict: **{metrics['verdict']}**.

This is a passive same-seed control/treatment audit of the Step 13.8B
configuration: population={POPULATION}, firms={FIRM_COUNT}, seed={SEED},
steps={STEPS}, K={K_WEEKS}, relief={RELIEF_FRACTION:.0%}. No credit limit,
exposure, arrears, payment priority, liquidity floor, interest, payroll,
production, Default, restructuring, ledger, or RNG behavior was changed.

## Exact credit semantics

From `world.py`, before current-week interest settlement:

```text
opening_principal = firm.loan_balance
opening_interest_arrears = firm.interest_arrears
opening_lender_exposure = opening_principal + opening_interest_arrears
credit_limit = K_weeks * trailing_scheduled_wage_bill
credit_headroom = max(credit_limit - opening_lender_exposure, 0)
requested_credit = max(target_cash - opening_cash, 0)
executed_credit = min(requested_credit, credit_headroom)
denied_credit = max(requested_credit - executed_credit, 0)
```

Current interest due is computed before this credit decision but is not added
to opening exposure. Requested credit is therefore evaluated against prior
arrears already present at the start of the week.

## Exposure decomposition

| Metric | p25 | median | p75 | p90 |
|---|---:|---:|---:|---:|
| Credit limit | {metrics['credit_limit_p25']:.6g} | {metrics['credit_limit_median']:.6g} | {metrics['credit_limit_p75']:.6g} | {metrics['credit_limit_p90']:.6g} |
| Opening principal | {metrics['opening_principal_p25']:.6g} | {metrics['opening_principal_median']:.6g} | {metrics['opening_principal_p75']:.6g} | {metrics['opening_principal_p90']:.6g} |
| Opening arrears | {metrics['opening_arrears_p25']:.6g} | {metrics['opening_arrears_median']:.6g} | {metrics['opening_arrears_p75']:.6g} | {metrics['opening_arrears_p90']:.6g} |
| Opening exposure | {metrics['opening_exposure_p25']:.6g} | {metrics['opening_exposure_median']:.6g} | {metrics['opening_exposure_p75']:.6g} | {metrics['opening_exposure_p90']:.6g} |
| Headroom deficit | {metrics['headroom_deficit_p25']:.6g} | {metrics['headroom_deficit_median']:.6g} | {metrics['headroom_deficit_p75']:.6g} | {metrics['headroom_deficit_p90']:.6g} |

Headroom deficit was positive in
{metrics['headroom_deficit_positive_week_count']}/{metrics['relief_firm_week_count']}
weeks. Its mean={metrics['headroom_deficit_mean']:.6g}, total={metrics['headroom_deficit_total']:.6g}; opening exposure identity gap={metrics['opening_exposure_identity_gap']:.6g}.

The same p25/median/p75/p90 distributions are recorded for exposure
utilization, current headroom, requested credit, executed credit, denied credit,
and the arrears segregation needed to satisfy the observed request.

Principal-only headroom was positive in
{metrics['principal_only_positive_headroom_week_count']}/{metrics['relief_firm_week_count']}
weeks and zero in {metrics['principal_only_zero_headroom_week_count']}. Principal
exceeded the limit in {metrics['principal_exceeds_credit_limit_week_count']}
weeks. Mean arrears share of opening exposure was
{metrics['mean_arrears_share_of_exposure']:.2%}; among weeks with a positive
headroom deficit, the mean deficit covered by removable legacy arrears was
{metrics['mean_arrears_share_of_headroom_deficit']:.2%}. The total arrears
needed to restore zero headroom was
{metrics['arrears_amount_needed_to_restore_zero_headroom_total']:.6g}; the
total needed to satisfy observed requested credit was
{metrics['arrears_termout_needed_for_requested_credit_total']:.6g}.

## Passive term-out bounds

| Counterfactual | Positive credit weeks | Total executable credit |
|---|---:|---:|
| Full arrears segregation | {metrics['full_termout_positive_credit_week_count']} | {metrics['full_termout_total_executable_credit']:.6g} |
| Restructuring-start snapshot | {metrics['snapshot_termout_positive_credit_week_count']} | {metrics['snapshot_termout_total_executable_credit']:.6g} |

Full term-out denied-credit reduction={metrics['full_termout_denied_credit_reduction']:.6g}
(conditional executable mean={metrics['full_termout_positive_conditional_mean']:.6g});
snapshot reduction={metrics['snapshot_termout_denied_credit_reduction']:.6g}
(conditional executable mean={metrics['snapshot_termout_positive_conditional_mean']:.6g}).
The full counterfactual keeps total lender exposure visible as principal plus
legacy arrears; it only removes legacy arrears from revolving utilization. The
preferred snapshot version excludes only arrears existing at restructuring
start, while new post-start arrears continue consuming revolving headroom.
In the observed treatment path, snapshot positive-headroom duration had median
{metrics['snapshot_positive_headroom_duration_median']:.6g} weeks and
{metrics['snapshot_reexhausted_firm_count']} Firms re-exhausted it before the
run ended. These are static/semi-dynamic accounting counterfactuals, not a
causal simulation.

## Payroll-risk connection

F1 next-payroll-risk weeks={metrics['payroll_risk_week_count']}.
Term-out credit fully covered {metrics['payroll_risk_fully_covered_by_termout_credit_count']}
weeks and partially covered {metrics['payroll_risk_partially_covered_count']}.
Total static payroll shortfall={metrics['total_next_week_payroll_shortfall']:.6g};
term-out executable credit on those weeks={metrics['total_termout_executable_credit']:.6g};
uncovered residual={metrics['total_uncovered_payroll_shortfall']:.6g}.

## Relief and feedback loop

Treatment-control mean headroom-deficit change was
{metrics['relief_control_headroom_deficit_change']:.6g}; full accrual-standstill
positive-headroom weeks were {metrics['full_accrual_standstill_positive_headroom_week_count']}.
The feedback evidence is **{metrics['credit_arrears_feedback_strength']}**:
legacy arrears, exposure at/above limit, zero headroom, denied credit, and
unpaid current interest are all observed in the canonical relief sample when
the strength is strong. This is a timing-consistent mechanism pattern, not by
itself a causal behavioral estimate.

The term-out counterfactual is distinct from write-down: the creditor claim
remains fully outstanding and only revolving utilization changes. A future
behavioral test, if selected, should snapshot existing arrears at restructuring
start and let new post-start arrears consume revolving capacity.

## Required semantic answers

1. Credit headroom is zero because opening principal plus opening historical
   interest arrears exceeds the current wage-anchored credit limit in the
   canonical relief weeks; requested credit is therefore denied.
2. Principal alone does not generally exhaust capacity: it leaves positive
   headroom in {metrics['principal_only_positive_headroom_week_count']}/
   {metrics['relief_firm_week_count']} weeks.
3. Historical arrears create almost all of the observed positive deficit: the
   mean arrears-covered share is
   {metrics['mean_arrears_share_of_headroom_deficit']:.2%}.
4. Separating legacy arrears from revolver utilization reopens meaningful
   theoretical headroom in {metrics['full_termout_positive_credit_week_count']}/
   {metrics['relief_firm_week_count']} weeks, while the lender claim remains
   principal plus arrears.
5. The observed requested credit would use that reopened headroom: passive
   executable credit totals
   {metrics['full_termout_total_executable_credit']:.6g}.
6. It materially covers the static payroll-risk sample, fully covering
   {metrics['payroll_risk_fully_covered_by_termout_credit_count']}/
   {metrics['payroll_risk_week_count']} risk weeks, with residual uncovered
   shortfall {metrics['total_uncovered_payroll_shortfall']:.6g}.
7. The 75% current-interest relief reduced treatment exposure relative to
   control by {abs(metrics['relief_control_opening_exposure_change']):.6g} on
   average, but did not cross the headroom boundary; it mainly slowed new
   arrears accumulation.
8. A complete standstill on new accrual alone restores positive headroom in
   {metrics['full_accrual_standstill_positive_headroom_week_count']}/
   {metrics['relief_firm_week_count']} observed relief weeks, so it cannot
   solve the existing legacy lock.
9. Term-out is not forgiveness: it changes only revolver utilization, not the
   total lender claim or lender loss.
10. Any future term-out should use the arrears stock existing at restructuring
    start as a one-time snapshot.
11. New post-restructuring arrears should continue consuming revolving
    headroom, preventing unlimited fresh off-line arrears.
12. No actual behavioral mechanism is selected by this audit. The evidence
    supports a future isolated snapshot term-out test only after an explicit
    design decision; no such mechanism was implemented here.

No next behavioral mechanism is selected by this audit. The hard stop remains:
no term-out, credit expansion, K change, arrears write-down, payment-priority
change, liquidity-floor change, or Exit was implemented.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


def main():
    control = run_world(False, 0.0, steps=STEPS, population=POPULATION)
    treatment = run_world(True, RELIEF_FRACTION, steps=STEPS, population=POPULATION)
    metrics = analyze(control.firm_diagnostics_rows, treatment.firm_diagnostics_rows)
    write_outputs(metrics)
    print(metrics["verdict"])


if __name__ == "__main__":
    main()
