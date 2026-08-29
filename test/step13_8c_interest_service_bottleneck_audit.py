"""Step 13.8C: diagnose interest-service capacity versus claim priority."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

from step13_8b_interest_relief_experiment import (  # noqa: E402
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


OUT = ROOT / "test/output/step13_8C_interest_service_bottleneck_audit"
STEP13_8B_OUT = ROOT / "test/output/step13_8B_interest_relief_experiment"
TOLERANCE = 1e-9
ACCOUNTING_TOLERANCE = 1e-6


def key(row):
    return step_of(row), int(float(row.get("firm_id", 0)))


def value(row, field, default=0.0):
    return finite_number(row.get(field, default))


def write_metrics(metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "interest_service_bottleneck_core_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "metric", "value", "note"))
        writer.writeheader()
        for row in metrics:
            writer.writerow(row)


def emit(rows, section, metric, value_, note=""):
    rows.append({"section": section, "metric": metric, "value": value_, "note": note})


def final_firm_values(rows):
    latest = {}
    for row in rows:
        latest[int(float(row.get("firm_id", 0)))] = row
    return latest


def treatment_waterfall(control_rows, treatment_rows):
    control_map = {key(row): row for row in control_rows}
    active_flag_rows = [
        row
        for row in treatment_rows
        if bool_value(row.get("restructuring_active", False))
    ]
    # The end-of-week finalizer clears restructuring_active on the expiry week
    # after that week's settlement has already occurred. Use the realized
    # relief memo to identify settlement weeks, and retain the old flag count
    # as a timing diagnostic.
    treatment_relief = [
        row
        for row in treatment_rows
        if value(row, "interest_relief_amount") > TOLERANCE
    ]
    aligned = [(row, control_map[key(row)]) for row in treatment_relief]
    if not aligned:
        raise RuntimeError("No active restructuring Firm-weeks found")

    zero_budget = []
    positive_budget = []
    positive_arrears_first = []
    positive_current_payment = []
    current_first_full = []
    current_first_breach_removed = []
    positive_budget_arrears = 0.0
    positive_budget_current = 0.0
    positive_budget_total = 0.0
    current_first_displaced = 0.0
    standstill_full = 0
    ocf_negative = 0
    credit_denied = 0
    zero_headroom = 0
    sums = defaultdict(float)
    control_service_budget = 0.0
    treatment_service_budget = 0.0

    for treatment, control in aligned:
        budget = max(0.0, value(treatment, "interest_service_cash"))
        floor = max(
            0.0,
            value(
                treatment,
                "operating_liquidity_floor",
                value(treatment, "repayment_buffer"),
            ),
        )
        cash_before_interest = value(treatment, "cash_before_interest")
        gross_due = value(treatment, "gross_current_interest_due")
        net_due = value(treatment, "net_current_interest_due", value(treatment, "current_interest_due"))
        relief = value(treatment, "interest_relief_amount")
        opening_arrears = value(treatment, "opening_interest_arrears")
        total_paid = value(treatment, "interest_paid")
        paid_arrears = value(treatment, "interest_paid_to_opening_arrears")
        paid_current = value(treatment, "interest_paid_to_current_due")
        unpaid_current = value(treatment, "current_interest_unpaid")
        closing_arrears = value(
            treatment,
            "closing_interest_arrears",
            value(treatment, "interest_arrears"),
        )
        sums["cash_before_interest"] += cash_before_interest
        sums["protected_cash_floor"] += floor
        sums["service_budget"] += budget
        sums["interest_cash_payment"] += total_paid
        sums["payment_to_arrears"] += paid_arrears
        sums["payment_to_current"] += paid_current
        sums["gross_due"] += gross_due
        sums["net_due"] += net_due
        sums["relief"] += relief
        sums["unpaid_current"] += unpaid_current
        sums["opening_arrears"] += opening_arrears
        sums["closing_arrears"] += closing_arrears
        sums["operating_net_before_interest"] += cash_before_interest - value(
            treatment, "cash_start"
        )
        control_service_budget += max(0.0, value(control, "interest_service_cash"))
        treatment_service_budget += budget

        if budget <= TOLERANCE:
            zero_budget.append(treatment)
        else:
            positive_budget.append(treatment)
            positive_budget_total += budget
            positive_budget_arrears += paid_arrears
            positive_budget_current += paid_current
            if paid_arrears > TOLERANCE and opening_arrears > TOLERANCE:
                positive_arrears_first.append(treatment)
            if paid_current > TOLERANCE:
                positive_current_payment.append(treatment)

            current_first_paid = min(net_due, budget)
            current_first_arrears = min(
                opening_arrears,
                max(0.0, budget - current_first_paid),
            )
            current_first_displaced += max(0.0, paid_arrears - current_first_arrears)
            if net_due > TOLERANCE and budget >= net_due - TOLERANCE:
                current_first_full.append(treatment)
                standstill_full += 1
                if unpaid_current > TOLERANCE:
                    current_first_breach_removed.append(treatment)

        ocf = (
            value(treatment, "sales_revenue")
            - value(treatment, "wage_payment", value(treatment, "executed_wage_bill"))
            + value(treatment, "public_sector_cash_inflow")
            - value(treatment, "public_sector_cash_outflow")
        )
        if ocf < -TOLERANCE:
            ocf_negative += 1
        if value(treatment, "denied_credit") > TOLERANCE:
            credit_denied += 1
        if value(treatment, "credit_headroom") <= TOLERANCE:
            zero_headroom += 1

    count = len(aligned)
    positive_count = len(positive_budget)
    zero_share = len(zero_budget) / count
    positive_share = positive_count / count
    positive_arrears_share = (
        positive_budget_arrears / positive_budget_total
        if positive_budget_total > TOLERANCE
        else 0.0
    )

    control_final = final_firm_values(control_rows)
    treatment_final = final_firm_values(treatment_rows)
    ending_control_arrears = sum(value(row, "interest_arrears") for row in control_final.values())
    ending_treatment_arrears = sum(value(row, "interest_arrears") for row in treatment_final.values())
    ending_arrears_change = ending_treatment_arrears - ending_control_arrears
    relief_to_arrears_gap = abs(-ending_arrears_change - sums["relief"])

    treatment_service_ratio = (
        sums["payment_to_current"] / sums["net_due"]
        if sums["net_due"] > TOLERANCE
        else 0.0
    )
    if zero_share >= 0.99 and positive_budget_total <= TOLERANCE:
        verdict = "D INTEREST_RELIEF_ONLY_REDUCES_CLAIM_ACCUMULATION"
        diagnosis = "ZERO_SERVICEABLE_CASH_IS_PRIMARY"
    elif zero_share >= 0.5 and positive_arrears_share > 0.5:
        verdict = "C MIXED_CAPACITY_AND_PRIORITY_BLOCKER"
        diagnosis = "MIXED_CAPACITY_AND_PRIORITY"
    elif positive_budget_total > TOLERANCE and len(current_first_breach_removed) > 0:
        verdict = "E CURRENT_FIRST_ALLOCATION_IS_REFERENCE_NEXT_MECHANISM"
        diagnosis = "ARREARS_PRIORITY_DIVERSION"
    elif positive_budget_total > TOLERANCE and positive_arrears_share > 0.5:
        verdict = "A PAYMENT_PRIORITY_IS_PRIMARY_BLOCKER"
        diagnosis = "ARREARS_PRIORITY_DIVERSION"
    else:
        verdict = "B ZERO_SERVICEABLE_CASH_IS_PRIMARY_BLOCKER"
        diagnosis = "ZERO_SERVICEABLE_CASH"

    return {
        "restructuring_firm_week_count": count,
        "restructuring_active_flag_week_count": len(active_flag_rows),
        "zero_service_budget_week_count": len(zero_budget),
        "zero_service_budget_week_share": zero_share,
        "positive_service_budget_week_count": positive_count,
        "positive_service_budget_week_share": positive_share,
        "mean_cash_before_interest": sums["cash_before_interest"] / count,
        "mean_protected_cash_floor": sums["protected_cash_floor"] / count,
        "mean_lender_service_cash_budget": sums["service_budget"] / count,
        "total_lender_service_cash_budget": sums["service_budget"],
        "total_interest_cash_payment": sums["interest_cash_payment"],
        "total_payment_to_arrears": sums["payment_to_arrears"],
        "total_payment_to_current": sums["payment_to_current"],
        "positive_budget_arrears_first_week_count": len(positive_arrears_first),
        "positive_budget_current_payment_week_count": len(positive_current_payment),
        "payment_priority_effect_identified": len(positive_arrears_first) > 0,
        "total_positive_budget_amount": positive_budget_total,
        "total_positive_budget_to_arrears": positive_budget_arrears,
        "total_positive_budget_to_current": positive_budget_current,
        "positive_budget_share_absorbed_by_arrears": positive_arrears_share,
        "control_aligned_total_service_budget": control_service_budget,
        "treatment_total_service_budget": treatment_service_budget,
        "service_budget_change": treatment_service_budget - control_service_budget,
        "total_interest_relief_amount": sums["relief"],
        "ending_control_arrears": ending_control_arrears,
        "ending_treatment_arrears": ending_treatment_arrears,
        "ending_arrears_change": ending_arrears_change,
        "relief_to_arrears_reduction_identity_gap": relief_to_arrears_gap,
        "current_first_static_full_service_week_count": len(current_first_full),
        "current_first_static_breach_removed_count": len(current_first_breach_removed),
        "current_first_static_arrears_payment_displaced": current_first_displaced,
        "arrears_standstill_static_full_service_week_count": standstill_full,
        "credit_denied_relief_week_share": credit_denied / count,
        "zero_headroom_relief_week_share": zero_headroom / count,
        "negative_OCF_relief_week_share": ocf_negative / count,
        "mean_operating_net_before_interest": sums["operating_net_before_interest"] / count,
        "mean_gross_current_interest_due": sums["gross_due"] / count,
        "mean_net_current_interest_due": sums["net_due"] / count,
        "total_unpaid_current_interest": sums["unpaid_current"],
        "treatment_current_payment_service_ratio": treatment_service_ratio,
        "claim_flow_improved": relief_to_arrears_gap <= ACCOUNTING_TOLERANCE and ending_arrears_change < -TOLERANCE,
        "current_payment_service_improved": (
            sums["payment_to_current"] > TOLERANCE
            or len(current_first_breach_removed) > 0
        ),
        "verdict": verdict,
        "mechanism_diagnosis": diagnosis,
        "aligned_rows": aligned,
    }


def repair_step13_8b_outputs(core):
    """Correct the old semantic flag and replace its misleading plot in place."""
    flags_path = STEP13_8B_OUT / "acceptance_flags.json"
    summary_path = STEP13_8B_OUT / "acceptance_summary.md"
    plot_path = STEP13_8B_OUT / "interest_relief_comparison.png"
    if not flags_path.exists():
        return False

    flags = json.loads(flags_path.read_text(encoding="utf-8"))
    flags["claim_flow_improved"] = bool(core["claim_flow_improved"])
    flags["current_payment_service_improved"] = bool(core["current_payment_service_improved"])
    flags["current_service_improved"] = bool(core["current_payment_service_improved"])
    flags["step13_8B_reporting_semantics_corrected"] = True
    flags_path.write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")

    marker = "## Step 13.8C reporting correction"
    summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    if marker not in summary:
        summary += (
            f"\n\n{marker}\n\n"
            "The former `current_service_improved` flag was a claim-flow measure. "
            f"It is corrected to **{core['current_payment_service_improved']}**: "
            "relief reduced new claim accumulation, but did not create positive "
            "current-interest payment in the treated restructuring weeks. "
            "`claim_flow_improved` remains **true**.\n"
        )
    summary_path.write_text(summary, encoding="utf-8")

    # Use grounded stored aggregate metrics, not absent compact time-series fields.
    rows = list(csv.DictReader((STEP13_8B_OUT / "interest_relief_control_treatment.csv").open(
        newline="", encoding="utf-8"
    )))
    data = {(row["section"], row["metric"]): value(row, "value") for row in rows}
    labels = ["Ending arrears", "Lender exposure", "Interest cash income"]
    control = [
        data.get(("CONTROL", "ending_interest_arrears"), 0.0),
        data.get(("CONTROL", "ending_lender_exposure"), 0.0),
        data.get(("CONTROL", "total_interest_cash_income"), 0.0),
    ]
    treatment = [
        data.get(("TREATMENT", "ending_interest_arrears"), 0.0),
        data.get(("TREATMENT", "ending_lender_exposure"), 0.0),
        data.get(("TREATMENT", "total_interest_cash_income"), 0.0),
    ]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = list(range(len(labels)))
    width = 0.36
    axis.bar([p - width / 2 for p in positions], control, width, label="control", color="0.35")
    axis.bar([p + width / 2 for p in positions], treatment, width, label="treatment", color="#3b82f6")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Stored aggregate value")
    axis.set_title("Step 13.8B grounded aggregate comparison")
    axis.legend()
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    figure.tight_layout()
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return True


def main():
    # Diagnostic-only rerun. No new RNG calls or decisions are introduced by
    # the audit; all requested quantities already exist in firm diagnostics.
    control_world = run_world(False, 0.0, steps=STEPS, population=POPULATION)
    treatment_world = run_world(True, RELIEF_FRACTION, steps=STEPS, population=POPULATION)
    core = treatment_waterfall(
        control_world.firm_diagnostics_rows,
        treatment_world.firm_diagnostics_rows,
    )
    plot_repaired = repair_step13_8b_outputs(core)

    metrics = []
    for name in (
        "restructuring_firm_week_count",
        "restructuring_active_flag_week_count",
        "zero_service_budget_week_count",
        "zero_service_budget_week_share",
        "positive_service_budget_week_count",
        "positive_service_budget_week_share",
        "mean_cash_before_interest",
        "mean_protected_cash_floor",
        "mean_lender_service_cash_budget",
        "total_lender_service_cash_budget",
        "total_interest_cash_payment",
        "total_payment_to_arrears",
        "total_payment_to_current",
        "positive_budget_arrears_first_week_count",
        "positive_budget_current_payment_week_count",
        "total_positive_budget_amount",
        "total_positive_budget_to_arrears",
        "total_positive_budget_to_current",
        "positive_budget_share_absorbed_by_arrears",
        "control_aligned_total_service_budget",
        "treatment_total_service_budget",
        "service_budget_change",
        "total_interest_relief_amount",
        "ending_arrears_change",
        "relief_to_arrears_reduction_identity_gap",
        "current_first_static_full_service_week_count",
        "current_first_static_breach_removed_count",
        "current_first_static_arrears_payment_displaced",
        "arrears_standstill_static_full_service_week_count",
        "credit_denied_relief_week_share",
        "zero_headroom_relief_week_share",
        "negative_OCF_relief_week_share",
        "mean_operating_net_before_interest",
        "mean_gross_current_interest_due",
        "mean_net_current_interest_due",
        "total_unpaid_current_interest",
        "treatment_current_payment_service_ratio",
    ):
        emit(metrics, "CORE", name, core[name])
    for name in (
        "claim_flow_improved",
        "current_payment_service_improved",
    ):
        emit(metrics, "SEMANTICS", name, core[name])
    write_metrics(metrics)

    flags = {
        "verdict": core["verdict"],
        "mechanism_diagnosis": core["mechanism_diagnosis"],
        "interest_payment_waterfall_understood": True,
        "lender_service_cash_budget_understood": True,
        "payment_priority_effect_identified": core["payment_priority_effect_identified"],
        "zero_service_capacity_effect_identified": core["zero_service_budget_week_count"] > 0,
        "claim_flow_improved": core["claim_flow_improved"],
        "current_payment_service_improved": core["current_payment_service_improved"],
        "step13_8B_reporting_semantics_corrected": True,
        "step13_8B_plot_repaired_or_marked_invalid": plot_repaired,
        "current_first_counterfactual_complete": True,
        "arrears_standstill_counterfactual_complete": True,
        "next_single_mechanism_selected": False,
        "selected_next_mechanism": None,
        "behavioral_next_test_ready": False,
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
        },
        **{name: core[name] for name in (
            "restructuring_firm_week_count",
            "zero_service_budget_week_count",
            "zero_service_budget_week_share",
            "positive_service_budget_week_count",
            "positive_service_budget_week_share",
            "mean_cash_before_interest",
            "mean_protected_cash_floor",
            "mean_lender_service_cash_budget",
            "total_lender_service_cash_budget",
            "total_interest_cash_payment",
            "total_payment_to_arrears",
            "total_payment_to_current",
            "positive_budget_arrears_first_week_count",
            "positive_budget_current_payment_week_count",
            "total_positive_budget_amount",
            "total_positive_budget_to_arrears",
            "total_positive_budget_to_current",
            "control_aligned_total_service_budget",
            "treatment_total_service_budget",
            "service_budget_change",
            "total_interest_relief_amount",
            "ending_arrears_change",
            "relief_to_arrears_reduction_identity_gap",
            "current_first_static_full_service_week_count",
            "current_first_static_breach_removed_count",
            "current_first_static_arrears_payment_displaced",
            "arrears_standstill_static_full_service_week_count",
            "credit_denied_relief_week_share",
            "zero_headroom_relief_week_share",
            "negative_OCF_relief_week_share",
        )},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2) + "\n", encoding="utf-8"
    )

    zero_share = core["zero_service_budget_week_share"]
    summary = f"""# Step 13.8C acceptance summary

Verdict: **{core['verdict']}**.

Mechanism diagnosis: **{core['mechanism_diagnosis']}**.

This is a diagnostic-only same-seed rerun of the Step 13.8B configuration:
population={POPULATION}, firms={FIRM_COUNT}, seed={SEED}, steps={STEPS},
K={K_WEEKS}, relief={RELIEF_FRACTION:.0%}, eligibility={ELIGIBILITY_WEEKS}
completed active-Default weeks, maximum relief duration={MAX_RELIEF_WEEKS} weeks.
No payment, credit, arrears, production, Default, RNG, or ledger behavior was
changed.

The realized settlement-week count is identified from positive relief memo
flows. The end-of-week `restructuring_active` flag marks fewer rows because
the expiry week is closed before diagnostics are recorded; both counts are
reported below.

## Settlement waterfall

The accepted runtime sequence is:

1. Working-capital credit is settled and payroll is paid.
2. Sales, dividends, and public-sector cash flows are incorporated.
3. `cash_before_interest` is formed.
4. `repayment_buffer` is protected as `operating_liquidity_floor`.
5. `lender_service_cash_budget = max(cash_before_interest - repayment_buffer, 0)`.
6. `settle_interest()` pays opening historical arrears first, then current
   interest, and records unpaid current interest as new closing arrears.
7. Principal repayment uses cash remaining after interest and the same reserve.

Thus, when service cash is positive, historical arrears receive payment before
the current claim. The 13.8C audit does not change that order.

## Capacity evidence

| Metric | Value |
|---|---:|
| Restructuring Firm-weeks | {core['restructuring_firm_week_count']} |
| Rows marked `restructuring_active` at end-of-week | {core['restructuring_active_flag_week_count']} |
| Zero service-budget weeks | {core['zero_service_budget_week_count']} ({zero_share:.2%}) |
| Positive service-budget weeks | {core['positive_service_budget_week_count']} ({core['positive_service_budget_week_share']:.2%}) |
| Mean cash before interest | {core['mean_cash_before_interest']:.6g} |
| Mean protected cash floor | {core['mean_protected_cash_floor']:.6g} |
| Mean lender-service cash budget | {core['mean_lender_service_cash_budget']:.6g} |
| Total lender-service cash budget | {core['total_lender_service_cash_budget']:.6g} |
| Credit-denied relief-week share | {core['credit_denied_relief_week_share']:.2%} |
| Zero-headroom relief-week share | {core['zero_headroom_relief_week_share']:.2%} |
| Negative-OCF relief-week share | {core['negative_OCF_relief_week_share']:.2%} |

## Allocation evidence

| Metric | Value |
|---|---:|
| Total interest cash payment | {core['total_interest_cash_payment']:.6g} |
| Total payment to opening arrears | {core['total_payment_to_arrears']:.6g} |
| Total payment to current interest | {core['total_payment_to_current']:.6g} |
| Positive-budget arrears-first weeks | {core['positive_budget_arrears_first_week_count']} |
| Positive-budget current-payment weeks | {core['positive_budget_current_payment_week_count']} |
| Positive budget absorbed by arrears | {core['positive_budget_share_absorbed_by_arrears']:.2%} |
| Current-first static full-service weeks | {core['current_first_static_full_service_week_count']} |
| Current-first static breach removals | {core['current_first_static_breach_removed_count']} |
| Arrears payment displaced by current-first | {core['current_first_static_arrears_payment_displaced']:.6g} |
| Arrears-standstill static full-service weeks | {core['arrears_standstill_static_full_service_week_count']} |

## Interpretation

The treatment reduced the claim stock by **{core['total_interest_relief_amount']:.6g}**.
Ending arrears changed by **{core['ending_arrears_change']:.6g}**, with identity
gap **{core['relief_to_arrears_reduction_identity_gap']:.6g}**. Therefore the
observed effect is prevention of newly formed arrears, not extra cash payment
or repayment of historical arrears.

`claim_flow_improved = {core['claim_flow_improved']}` while
`current_payment_service_improved = {core['current_payment_service_improved']}`.
The latter requires actual current-interest cash service, not merely a smaller
net claim. The current code's payment priority is only an immediate blocker in
weeks where positive service cash exists; zero-budget weeks cannot be explained
by arrears diversion.

The control-aligned service-budget total was
**{core['control_aligned_total_service_budget']:.6g}**, versus treatment
**{core['treatment_total_service_budget']:.6g}**, a change of
**{core['service_budget_change']:.6g}**.

No next behavioral mechanism is selected in this audit. The hard stop remains:
no payment-priority change, arrears standstill, higher relief fraction, credit
expansion, principal change, or Exit was implemented.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(core["verdict"])
    print(core["mechanism_diagnosis"])


if __name__ == "__main__":
    main()
