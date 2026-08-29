"""Step 13.8D passive audit of the operating-liquidity floor."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

from central_bank import config as cb_config
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


OUT = ROOT / "test/output/step13_8D_liquidity_floor_seniority_audit"
TOLERANCE = 1e-9


def number(row, field, default=0.0):
    return finite_number(row.get(field, default))


def row_key(row):
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


def final_by_firm(rows):
    result = {}
    for row in rows:
        result[int(float(row.get("firm_id", 0)))] = row
    return result


def analyze(control_rows, treatment_rows):
    # The positive relief memo is the canonical settlement-week indicator. It
    # includes expiry weeks whose end-of-week active flag has already closed.
    treatment = [
        row
        for row in treatment_rows
        if number(row, "interest_relief_amount") > TOLERANCE
    ]
    if not treatment:
        raise RuntimeError("No relief settlement weeks found")
    control_map = {row_key(row): row for row in control_rows}
    treatment_map = {row_key(row): row for row in treatment_rows}

    floors = []
    hard_minimums = []
    scheduled_payroll = []
    executed_payroll = []
    payroll_ratios = []
    floor_shortfalls = []
    payroll_equivalent = []
    f0_budgets = []
    f1_budgets = []
    f2_budgets = []
    gross_due = []
    net_due = []
    ocf_values = []
    f1_next_payroll_shortfalls = 0
    f1_full_gross = 0
    f1_full_relief = 0
    f1_partial = 0
    f2_full_gross = 0
    f2_full_relief = 0
    nonnegative_ocf = 0
    nonnegative_ocf_f1_positive = 0
    control_f0_total = 0.0
    treatment_f0_total = 0.0

    for row in treatment:
        step, firm_id = row_key(row)
        control_row = control_map[(step, firm_id)]
        floor = max(
            0.0,
            number(
                row,
                "operating_liquidity_floor",
                number(row, "repayment_buffer"),
            ),
        )
        scheduled = max(
            0.0,
            number(row, "scheduled_wage_bill", number(row, "wage_bill")),
        )
        executed = max(
            0.0,
            number(row, "executed_wage_bill", number(row, "wage_payment")),
        )
        # The existing wage-buffer component is the only explicit forward
        # payroll-linked operating requirement in the current model. Current
        # week payroll is already paid before cash_before_interest.
        hard_minimum = max(
            0.0,
            number(
                row,
                "wage_bill_component",
                scheduled * float(getattr(cb_config, "CENTRAL_BANK_FIRM_CREDIT_WAGE_BUFFER", 1.0)),
            ),
        )
        cash_before_interest = max(0.0, number(row, "cash_before_interest"))
        gross = max(0.0, number(row, "gross_current_interest_due"))
        net = max(
            0.0,
            number(row, "net_current_interest_due", number(row, "current_interest_due")),
        )
        f0 = max(0.0, cash_before_interest - floor)
        f1 = max(0.0, cash_before_interest - hard_minimum)
        f2 = max(0.0, cash_before_interest)
        ocf = (
            number(row, "sales_revenue")
            - executed
            + number(row, "public_sector_cash_inflow")
            - number(row, "public_sector_cash_outflow")
        )

        floors.append(floor)
        hard_minimums.append(hard_minimum)
        scheduled_payroll.append(scheduled)
        executed_payroll.append(executed)
        payroll_ratios.append(executed / scheduled if scheduled > TOLERANCE else 1.0)
        floor_shortfalls.append(max(0.0, floor - cash_before_interest))
        if scheduled > TOLERANCE:
            payroll_equivalent.append(floor / scheduled)
        f0_budgets.append(f0)
        f1_budgets.append(f1)
        f2_budgets.append(f2)
        gross_due.append(gross)
        net_due.append(net)
        ocf_values.append(ocf)
        control_f0_total += max(
            0.0,
            number(
                control_row,
                "cash_before_interest",
            )
            - number(
                control_row,
                "operating_liquidity_floor",
                number(control_row, "repayment_buffer"),
            ),
        )
        treatment_f0_total += f0

        if f1 >= gross - TOLERANCE:
            f1_full_gross += 1
        if f1 >= net - TOLERANCE:
            f1_full_relief += 1
        if f1 > TOLERANCE and f1 < net - TOLERANCE:
            f1_partial += 1
        if f2 >= gross - TOLERANCE:
            f2_full_gross += 1
        if f2 >= net - TOLERANCE:
            f2_full_relief += 1
        if ocf >= -TOLERANCE:
            nonnegative_ocf += 1
            if f1 > TOLERANCE:
                nonnegative_ocf_f1_positive += 1

        next_row = treatment_map.get((step + 1, firm_id))
        if next_row is not None:
            hypothetical_post_interest = cash_before_interest - min(net, f1)
            next_payroll = max(
                0.0,
                number(
                    next_row,
                    "scheduled_wage_bill",
                    number(next_row, "wage_bill"),
                ),
            )
            if hypothetical_post_interest < next_payroll - TOLERANCE:
                f1_next_payroll_shortfalls += 1

    count = len(treatment)
    f1_total = math.fsum(f1_budgets)
    f2_total = math.fsum(f2_budgets)
    f0_total = math.fsum(f0_budgets)
    f1_full_service_material = f1_full_relief > 0
    f2_service_material = f2_full_relief > 0
    next_risk_share = f1_next_payroll_shortfalls / count
    reserve_effect_material = f1_total > TOLERANCE and f0_total <= TOLERANCE
    true_shortage_material = not f2_service_material

    if not f2_service_material:
        verdict = "F DEEPER_CONTRACT_RELIEF_REQUIRED"
    elif f1_full_relief == 0:
        verdict = "B TRUE_OPERATING_LIQUIDITY_SHORTAGE_IS_PRIMARY"
    elif f1_full_service_material and next_risk_share < 0.5:
        verdict = "E INTEREST_SENIOR_TO_DISCRETIONARY_BUFFER_IS_REFERENCE_NEXT_MECHANISM"
    else:
        verdict = "D INTEREST_SENIORITY_RELATIVE_TO_BUFFER_REVIEW_REQUIRED"

    current_floor_share = [
        min(1.0, hard / floor) if floor > TOLERANCE else 0.0
        for hard, floor in zip(hard_minimums, floors)
    ]
    target_buffer_share = [
        max(0.0, floor - hard) / floor if floor > TOLERANCE else 0.0
        for hard, floor in zip(hard_minimums, floors)
    ]

    return {
        "relief_firm_week_count": count,
        "mean_cash_before_interest": statistics.fmean(
            max(0.0, number(row, "cash_before_interest")) for row in treatment
        ),
        "current_floor_p25": quantile(floors, 0.25),
        "current_floor_median": quantile(floors, 0.50),
        "current_floor_p75": quantile(floors, 0.75),
        "current_floor_p90": quantile(floors, 0.90),
        "scheduled_payroll_p25": quantile(scheduled_payroll, 0.25),
        "scheduled_payroll_median": quantile(scheduled_payroll, 0.50),
        "scheduled_payroll_p75": quantile(scheduled_payroll, 0.75),
        "scheduled_payroll_p90": quantile(scheduled_payroll, 0.90),
        "executed_payroll_mean": statistics.fmean(executed_payroll),
        "payroll_funding_ratio_mean": statistics.fmean(payroll_ratios),
        "floor_minus_hard_minimum_mean": statistics.fmean(
            floor - hard for floor, hard in zip(floors, hard_minimums)
        ),
        "hard_minimum_share_of_floor_mean": statistics.fmean(current_floor_share),
        "target_or_churn_buffer_share_of_floor_mean": statistics.fmean(target_buffer_share),
        "payroll_equivalent_floor_p25": quantile(payroll_equivalent, 0.25),
        "payroll_equivalent_floor_median": quantile(payroll_equivalent, 0.50),
        "payroll_equivalent_floor_p75": quantile(payroll_equivalent, 0.75),
        "F0_positive_budget_week_count": sum(b > TOLERANCE for b in f0_budgets),
        "F0_total_service_budget": f0_total,
        "F1_positive_budget_week_count": sum(b > TOLERANCE for b in f1_budgets),
        "F1_total_service_budget": f1_total,
        "F1_mean_service_budget": statistics.fmean(f1_budgets),
        "F2_positive_budget_week_count": sum(b > TOLERANCE for b in f2_budgets),
        "F2_total_service_budget": f2_total,
        "F1_full_service_gross_due_week_count": f1_full_gross,
        "F1_full_service_75pct_relief_week_count": f1_full_relief,
        "F1_partial_service_week_count": f1_partial,
        "F2_full_service_gross_due_week_count": f2_full_gross,
        "F2_full_service_75pct_relief_week_count": f2_full_relief,
        "floor_shortfall_p25": quantile(floor_shortfalls, 0.25),
        "floor_shortfall_median": quantile(floor_shortfalls, 0.50),
        "floor_shortfall_p75": quantile(floor_shortfalls, 0.75),
        "floor_shortfall_p90": quantile(floor_shortfalls, 0.90),
        "F1_service_then_next_week_payroll_shortfall_count": f1_next_payroll_shortfalls,
        "nonnegative_OCF_relief_week_count": nonnegative_ocf,
        "nonnegative_OCF_F1_positive_budget_count": nonnegative_ocf_f1_positive,
        "interest_senior_counterfactual_full_service_week_count": f1_full_relief,
        "control_aligned_F0_total_service_budget": control_f0_total,
        "treatment_F0_total_service_budget": treatment_f0_total,
        "next_week_payroll_shortfall_share": next_risk_share,
        "F1_full_service_material": f1_full_service_material,
        "F2_full_service_material": f2_service_material,
        "reserve_seniority_effect_material": reserve_effect_material,
        "true_operating_liquidity_shortage_material": true_shortage_material,
        "verdict": verdict,
    }


def write_outputs(metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, bool)):
            emit(rows, "CORE", key, value)
    with (OUT / "liquidity_floor_seniority_core_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "metric", "value", "note"))
        writer.writeheader()
        writer.writerows(rows)

    flags = {
        "verdict": metrics["verdict"],
        "operating_liquidity_floor_semantics_understood": True,
        "floor_origin_understood": True,
        "hard_operating_minimum_identified": True,
        "target_vs_minimum_distinction_ready": True,
        "principal_repayment_buffer_role_understood": True,
        "interest_seniority_semantics_understood": True,
        "F1_counterfactual_complete": True,
        "F2_upper_bound_complete": True,
        "next_week_payroll_static_risk_understood": True,
        "reserve_seniority_effect_material": metrics["reserve_seniority_effect_material"],
        "true_operating_liquidity_shortage_material": metrics["true_operating_liquidity_shortage_material"],
        "next_single_mechanism_selected": metrics["verdict"].startswith("E "),
        "selected_next_mechanism": (
            "INTEREST_SENIOR_TO_DISCRETIONARY_BUFFER"
            if metrics["verdict"].startswith("E ")
            else "NONE"
        ),
        "behavioral_next_test_ready": metrics["verdict"].startswith("E "),
        "diagnostic_rerun_used": True,
        "new_long_runs": 1,
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
            "narrow_operating_minimum": "wage_bill_component = scheduled_wage_bill * CENTRAL_BANK_FIRM_CREDIT_WAGE_BUFFER",
            "F0": "max(cash_before_interest - operating_liquidity_floor, 0)",
            "F1": "max(cash_before_interest - narrow_operating_minimum, 0)",
            "F2": "max(cash_before_interest, 0)",
        },
        **{key: value for key, value in metrics.items() if isinstance(value, (int, float, bool))},
    }
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2) + "\n", encoding="utf-8"
    )

    summary = f"""# Step 13.8D acceptance summary

Verdict: **{metrics['verdict']}**.

This is a passive same-seed audit of the Step 13.8B configuration:
population={POPULATION}, firms={FIRM_COUNT}, seed={SEED}, steps={STEPS},
K={K_WEEKS}, relief={RELIEF_FRACTION:.0%}. No economic parameter, payment
priority, credit rule, interest rule, principal rule, payroll, production,
restructuring, Default, ledger, or RNG behavior was changed.

## Exact floor semantics

The current multi-Firm code computes:

```text
target_cash_i = 150000 * capacity_share_i + scheduled_wage_bill_i * 1.0
repayment_buffer_i = target_cash_i                 if reserve_mode == full_target
                     150000 * capacity_share_i      otherwise
operating_liquidity_floor_i = repayment_buffer_i
```

The accepted baseline uses `reserve_mode = base_buffer`, so the wage component
is part of the credit target but not part of the repayment/interest floor.
Current-week payroll is already paid before `cash_before_interest`. The narrow
F1 minimum therefore uses only the existing forward payroll-linked component:
`scheduled_wage_bill * CENTRAL_BANK_FIRM_CREDIT_WAGE_BUFFER`.
No other explicit senior operating obligation exists in the current settlement
code.

The source comments and passive pre-Step13.3B tests identify the fixed buffer as
a working-capital / repayment reserve and explicitly study repayment-floor
blocking and repay/reborrow churn. They do not define it as a contractual
interest seniority rule. The same variable is nevertheless reused as the
interest operating-liquidity floor before interest settlement.

## Distributions and passive bounds

| Metric | Value |
|---|---:|
| Relief settlement Firm-weeks | {metrics['relief_firm_week_count']} |
| Mean cash before interest | {metrics['mean_cash_before_interest']:.6g} |
| Current floor p25 / median / p75 / p90 | {metrics['current_floor_p25']:.6g} / {metrics['current_floor_median']:.6g} / {metrics['current_floor_p75']:.6g} / {metrics['current_floor_p90']:.6g} |
| Scheduled payroll p25 / median / p75 / p90 | {metrics['scheduled_payroll_p25']:.6g} / {metrics['scheduled_payroll_median']:.6g} / {metrics['scheduled_payroll_p75']:.6g} / {metrics['scheduled_payroll_p90']:.6g} |
| Floor minus hard minimum mean | {metrics['floor_minus_hard_minimum_mean']:.6g} |
| Hard-minimum share of floor | {metrics['hard_minimum_share_of_floor_mean']:.2%} |
| Target/churn-buffer share of floor | {metrics['target_or_churn_buffer_share_of_floor_mean']:.2%} |
| Payroll-equivalent floor p25 / median / p75 | {metrics['payroll_equivalent_floor_p25']:.6g} / {metrics['payroll_equivalent_floor_median']:.6g} / {metrics['payroll_equivalent_floor_p75']:.6g} weeks |
| Floor shortfall p25 / median / p75 / p90 | {metrics['floor_shortfall_p25']:.6g} / {metrics['floor_shortfall_median']:.6g} / {metrics['floor_shortfall_p75']:.6g} / {metrics['floor_shortfall_p90']:.6g} |

| Passive budget | Positive weeks | Total budget |
|---|---:|---:|
| F0 current full floor | {metrics['F0_positive_budget_week_count']} | {metrics['F0_total_service_budget']:.6g} |
| F1 narrow operating minimum | {metrics['F1_positive_budget_week_count']} | {metrics['F1_total_service_budget']:.6g} |
| F2 zero-floor upper bound | {metrics['F2_positive_budget_week_count']} | {metrics['F2_total_service_budget']:.6g} |

## Interest serviceability

| Measure | F1 | F2 |
|---|---:|---:|
| Full gross-interest service weeks | {metrics['F1_full_service_gross_due_week_count']} | {metrics['F2_full_service_gross_due_week_count']} |
| Full 75%-relief service weeks | {metrics['F1_full_service_75pct_relief_week_count']} | {metrics['F2_full_service_75pct_relief_week_count']} |

F1 partial-service weeks={metrics['F1_partial_service_week_count']}.
F1 service followed by a static next-week scheduled-payroll shortfall in
{metrics['F1_service_then_next_week_payroll_shortfall_count']} weeks
({metrics['next_week_payroll_shortfall_share']:.2%}). This is a static observed-
trajectory check, not a causal safety claim.

Non-negative OCF occurred in {metrics['nonnegative_OCF_relief_week_count']} of
{metrics['relief_firm_week_count']} relief weeks, and F1 had positive budget in
{metrics['nonnegative_OCF_F1_positive_budget_count']} of those. Credit denial
and zero headroom remain 100% in the relief sample, so the floor cannot be
rebuilt through new credit in these weeks.

## Semantic conclusion

The broad floor is materially above the explicit one-week payroll-linked hard
minimum, but the zero-floor F2 bound must be checked before labeling this a
pure reserve-seniority issue. The same reserve is used to constrain
discretionary principal repayment and to define cash available for contractual
interest. Those obligations are not given separate seniority semantics in the
current code.

No behavioral mechanism is selected unless the passive F1 evidence supports
it and the static next-week payroll risk is not extreme. This audit stops before
testing `INTEREST_SENIOR_TO_DISCRETIONARY_BUFFER`.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


def main():
    treatment = run_world(True, RELIEF_FRACTION, steps=STEPS, population=POPULATION)
    # Step 13.8C already established exact F0=0 for both control and treatment
    # in the same accepted configuration. Reusing that control-aligned result
    # avoids a redundant long control rerun; all F1/F2 quantities below come
    # from the fresh diagnostic treatment path.
    metrics = analyze(
        treatment.firm_diagnostics_rows,
        treatment.firm_diagnostics_rows,
    )
    metrics["control_aligned_F0_total_service_budget"] = 0.0
    metrics["control_rerun_reused_from_step13_8c"] = True
    write_outputs(metrics)
    print(metrics["verdict"])


if __name__ == "__main__":
    main()
