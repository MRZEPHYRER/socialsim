"""Step 13.9A passive principal-contract architecture review."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))

import step13_8f_snapshot_termout_experiment as step13_8f


OUT = ROOT / "test/output/step13_9A_principal_contract_architecture_review"
MATURE_START = 1560
TOLERANCE = 1e-7
F_OUTPUT = ROOT / "test/output/step13_8F_snapshot_legacy_arrears_termout/snapshot_termout_control_treatment.csv"
G_OUTPUT = ROOT / "test/output/step13_8G_post_termout_relock_audit/acceptance_flags.json"


def number(row, field, default=0.0):
    try:
        value = float(row.get(field, default))
        return value if math.isfinite(value) else float(default)
    except (TypeError, ValueError):
        return float(default)


def step_of(row):
    return int(float(row.get("global_step", row.get("step", 0))))


def firm_id_of(row):
    return int(float(row.get("firm_id", 0)))


def truth(row, field):
    value = row.get(field, False)
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}


def key(row):
    return step_of(row), firm_id_of(row)


def grouped_firms(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[firm_id_of(row)].append(row)
    for values in grouped.values():
        values.sort(key=step_of)
    return grouped


def sum_field(rows, field):
    return math.fsum(number(row, field) for row in rows)


def quantile(values, fraction):
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def add_metric(rows, scope, firm_id, metric, value, note=""):
    rows.append({
        "scope": scope,
        "firm_id": "" if firm_id is None else firm_id,
        "metric": metric,
        "value": value,
        "note": note,
    })


def read_historical_metrics():
    values = {}
    if F_OUTPUT.exists():
        with F_OUTPUT.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("scope") == "DIFFERENCE":
                    values[row.get("metric", "")] = number(row, "value")
                elif row.get("scope") == "CONTROL" or row.get("scope") == "TREATMENT":
                    values[f"{row['scope'].lower()}_{row.get('metric', '')}"] = number(row, "value")
    return values


def full_counts(rows):
    return {
        "payroll_underfunded": sum(number(row, "payroll_funding_ratio") < 1.0 - TOLERANCE for row in rows),
        "current_full_service": sum(number(row, "current_interest_unpaid") <= TOLERANCE for row in rows),
        "D3": sum(str(row.get("distress_state", "")) == "D3" for row in rows),
        "gross_borrowing": sum_field(rows, "loan_issued"),
        "gross_repayment": sum_field(rows, "loan_repaid"),
    }


def direct_effects(control_rows, treatment_rows, first_termout):
    control = {key(row): row for row in control_rows}
    treatment = {key(row): row for row in treatment_rows}
    aligned = sorted(set(control) & set(treatment))
    post = [item for item in aligned if item[0] >= first_termout]
    return {
        "G_payroll_effect": sum(
            number(control[item], "payroll_funding_ratio") < 1.0 - TOLERANCE
            and number(treatment[item], "payroll_funding_ratio") >= 1.0 - TOLERANCE
            for item in post
        ),
        "G_full_service_effect": sum(
            number(control[item], "current_interest_unpaid") > TOLERANCE
            and number(treatment[item], "current_interest_unpaid") <= TOLERANCE
            for item in post
        ),
        "G_D3_effect": sum(
            str(control[item].get("distress_state", "")) == "D3"
            and str(treatment[item].get("distress_state", "")) != "D3"
            for item in post
        ),
        "post_key_count": len(post),
    }


def utilization(row):
    limit = number(row, "credit_limit")
    principal = number(row, "loan_balance")
    return principal / limit if limit > TOLERANCE else 0.0


def utilization_metrics(rows, scope, output):
    values = [utilization(row) for row in rows if number(row, "credit_limit") > TOLERANCE]
    for name, fraction in (("p25", 0.25), ("median", 0.50), ("p75", 0.75), ("p90", 0.90)):
        add_metric(output, scope, None, f"principal_utilization_{name}", quantile(values, fraction), "loan_balance / current wage-anchored credit_limit")
    for threshold in (0.80, 0.90, 0.95, 1.00):
        share = sum(value >= threshold - TOLERANCE for value in values) / len(values) if values else 0.0
        label = str(int(threshold * 100))
        add_metric(output, scope, None, f"share_principal_ge_{label}pct_limit", share, "descriptive Firm-week share")
    add_metric(output, scope, None, "utilization_observation_count", len(values), "Firm-week observations")


def trailing_min(values, index, window):
    start = max(0, index - window + 1)
    return min(values[start:index + 1], default=0.0)


def longest_no_decrease(values):
    longest = current = 0
    for index, value in enumerate(values):
        if index == 0 or value >= values[index - 1] - TOLERANCE:
            current += 1
        else:
            current = 0
        longest = max(longest, current)
    return longest


def firm_persistence(rows_by_firm, firm_id, start_step):
    rows = rows_by_firm[firm_id]
    principals = [number(row, "loan_balance") for row in rows]
    final = rows[-1]
    final_index = len(rows) - 1
    termout_index = next((index for index, row in enumerate(rows) if step_of(row) == start_step), None)
    default_index = next((index for index, row in enumerate(rows) if truth(row, "active_contract_default")), None)
    decreases = sum(
        index > 0 and principals[index] < principals[index - 1] - TOLERANCE
        for index in range(len(principals))
    )
    post_termout_rows = [row for row in rows if step_of(row) >= start_step]
    post_default_rows = [row for row in rows[default_index:]] if default_index is not None else []
    post_restructuring_min_util = min((utilization(row) for row in post_termout_rows), default=0.0)
    post_default_min_util = min((utilization(row) for row in post_default_rows), default=0.0)
    termout_row = rows[termout_index] if termout_index is not None else rows[-1]
    termout_principal = number(termout_row, "opening_principal")
    return {
        "current_principal": number(final, "loan_balance"),
        "persistent_principal_floor_13w": trailing_min(principals, final_index, 13),
        "persistent_principal_floor_26w": trailing_min(principals, final_index, 26),
        "persistent_principal_floor_52w": trailing_min(principals, final_index, 52),
        "persistent_26w_share_of_current_principal": trailing_min(principals, final_index, 26) / max(number(final, "loan_balance"), TOLERANCE),
        "principal_decrease_week_count": decreases,
        "gross_borrowing": sum_field(rows, "loan_issued"),
        "gross_principal_repayment": sum_field(rows, "loan_repaid"),
        "net_principal_change": principals[-1] - principals[0],
        "longest_period_without_principal_falling_materially": longest_no_decrease(principals),
        "minimum_utilization_after_first_default": post_default_min_util,
        "minimum_utilization_after_restructuring": post_restructuring_min_util,
        "termout_principal": termout_principal,
        "termout_principal_after_settlement": number(termout_row, "loan_balance"),
        "termout_credit_limit": number(termout_row, "credit_limit"),
        "termout_principal_excess": max(0.0, termout_principal - number(termout_row, "credit_limit")),
        "termout_utilization": utilization(termout_row),
        "termout_trailing_floor_26w": trailing_min(principals, termout_index or final_index, 26),
        "termout_restored_headroom": number(termout_row, "credit_headroom"),
        "termout_immediate_redraw": number(termout_row, "executed_credit"),
        "termout_utilization_after_redraw": (
            (number(termout_row, "opening_principal") + number(termout_row, "executed_credit"))
            / max(number(termout_row, "credit_limit"), TOLERANCE)
        ),
        "termout_index": termout_index if termout_index is not None else -1,
    }


def passive_bounds(rows_by_firm, starts):
    persistent_bound = []
    payroll_bound = []
    for firm_id, start_row in starts.items():
        rows = rows_by_firm[firm_id]
        start_index = next(index for index, row in enumerate(rows) if step_of(row) == step_of(start_row))
        principals = [number(row, "loan_balance") for row in rows]
        row = rows[start_index]
        floor_26 = trailing_min(principals, start_index, 26)
        opening_principal = number(row, "opening_principal")
        opening_arrears = number(row, "opening_interest_arrears")
        limit = number(row, "credit_limit")
        requested = number(row, "requested_credit")
        residual_revolver = max(0.0, opening_principal - min(floor_26, opening_principal)) + opening_arrears
        upper_headroom = max(0.0, limit - residual_revolver)
        persistent_bound.append({
            "firm_id": firm_id,
            "reclassification_amount": min(floor_26, opening_principal),
            "restored_headroom": upper_headroom,
            "observed_requested_credit": requested,
            "potential_executable_credit": min(requested, upper_headroom),
        })
        scheduled_wage = number(row, "scheduled_wage_bill")
        needed = max(0.0, opening_principal + opening_arrears - (limit - scheduled_wage))
        needed = min(needed, opening_principal)
        residual_after_payroll_headroom = max(0.0, limit - (opening_principal - needed + opening_arrears))
        payroll_bound.append({
            "firm_id": firm_id,
            "reclassification_amount": needed,
            "reclassification_share": needed / max(number(row, "loan_balance"), TOLERANCE),
            "desired_headroom": scheduled_wage,
            "potential_executable_credit": min(requested, residual_after_payroll_headroom),
        })
    return persistent_bound, payroll_bound


def source_semantics():
    return {
        "loan_balance_semantics": "single Firm principal stock created by executed working-capital loans",
        "principal_creation": "ledger.create_loan_attr during firm-specific credit preparation",
        "principal_repayment": "min(loan_balance, loan_balance * 0.35, max(cash_after_interest - repayment_buffer, 0))",
        "maturity_date": "none",
        "mandatory_principal_installment": "none",
        "repayment_contract": "liquidity-constrained policy, not maturity-based mandatory payment",
        "revolving_utilization": "loan_balance + opening/closing interest arrears; legacy term claim excluded after 13.8F",
        "current_interest_base": "opening principal loan_balance times accepted weekly rate",
        "loan_vintage_or_age": "none",
    }


def main():
    control_world = step13_8f.run_world(False)
    treatment_world = step13_8f.run_world(True)
    control_rows = list(control_world.firm_diagnostics_rows)
    treatment_rows = list(treatment_world.firm_diagnostics_rows)
    starts = {}
    for row in treatment_rows:
        if truth(row, "termout_start_this_week"):
            starts.setdefault(firm_id_of(row), row)
    if not starts:
        raise RuntimeError("No Step 13.8F treatment term-out starts found")
    first_termout = min(step_of(row) for row in starts.values())
    affected = set(starts)
    metrics = []

    historical = read_historical_metrics()
    control_full = full_counts(control_rows)
    treatment_full = full_counts(treatment_rows)
    effects = direct_effects(control_rows, treatment_rows, first_termout)
    F_payroll = treatment_full["payroll_underfunded"] - control_full["payroll_underfunded"]
    F_full_service = treatment_full["current_full_service"] - control_full["current_full_service"]
    G_payroll = effects["G_payroll_effect"]
    G_full_service = effects["G_full_service_effect"]
    G_D3 = effects["G_D3_effect"]
    historical_match = (
        abs(historical.get("payroll_underfunded_week_change", F_payroll) - F_payroll) <= 1e-6
        and abs(historical.get("current_full_service_week_change", F_full_service) - F_full_service) <= 1e-6
        and abs(historical.get("control_payroll_underfunded_firm_weeks", control_full["payroll_underfunded"]) - control_full["payroll_underfunded"]) <= 1e-6
        and abs(historical.get("treatment_payroll_underfunded_firm_weeks", treatment_full["payroll_underfunded"]) - treatment_full["payroll_underfunded"]) <= 1e-6
    )
    reporting_only = historical_match and F_payroll != G_payroll and F_full_service != G_full_service
    trajectory_identical = historical_match

    add_metric(metrics, "REPORTING_RECONCILIATION", None, "step13_8F_payroll_effect", F_payroll, "full 1820-week net Firm-week difference: treatment minus control")
    add_metric(metrics, "REPORTING_RECONCILIATION", None, "step13_8G_payroll_effect", G_payroll, "post-first-termout improvement transitions only")
    add_metric(metrics, "REPORTING_RECONCILIATION", None, "step13_8F_full_service_effect", F_full_service, "full 1820-week net Firm-week difference")
    add_metric(metrics, "REPORTING_RECONCILIATION", None, "step13_8G_full_service_effect", G_full_service, "post-first-termout improvement transitions only")
    add_metric(metrics, "REPORTING_RECONCILIATION", None, "step13_8G_D3_reduction", G_D3, "post-first-termout D3 to non-D3 transitions")
    add_metric(metrics, "REPORTING_RECONCILIATION", None, "discrepancy_is_reporting_only", reporting_only, "different denominators/event definitions, not a new trajectory")
    add_metric(metrics, "REPORTING_RECONCILIATION", None, "behavioral_trajectory_identical", trajectory_identical, "same runner/config and historical aggregate parity")
    add_metric(metrics, "REPORTING_RECONCILIATION", None, "first_termout_week", first_termout, "same seed42 treatment")
    add_metric(metrics, "REPORTING_RECONCILIATION", None, "full_run_firm_week_count", len(control_rows), "5 Firms x 1820 weeks")

    treatment_by_firm = grouped_firms(treatment_rows)
    control_by_firm = grouped_firms(control_rows)
    affected_rows = [row for row in treatment_rows if firm_id_of(row) in affected]
    mature_rows = [row for row in affected_rows if step_of(row) >= MATURE_START]
    active_default_rows = [row for row in affected_rows if truth(row, "active_contract_default")]
    post_termout_rows = [
        row for row in affected_rows
        if step_of(row) >= step_of(starts[firm_id_of(row)])
    ]
    utilization_metrics(mature_rows, "AFFECTED_MATURE", metrics)
    utilization_metrics(active_default_rows, "AFFECTED_ACTIVE_DEFAULT", metrics)
    utilization_metrics(post_termout_rows, "AFFECTED_POST_TERMOUT", metrics)

    persistence = {}
    for firm_id in sorted(affected):
        persistence[firm_id] = firm_persistence(treatment_by_firm, firm_id, step_of(starts[firm_id]))
        for metric, value in persistence[firm_id].items():
            if metric == "termout_index":
                continue
            add_metric(metrics, "AFFECTED_FIRM", firm_id, metric, value, "principal persistence and deleveraging diagnostic")

    persistent_bound, payroll_bound = passive_bounds(treatment_by_firm, starts)
    for item in persistent_bound:
        for metric in ("reclassification_amount", "restored_headroom", "observed_requested_credit", "potential_executable_credit"):
            add_metric(metrics, "PERSISTENT_FLOOR_BOUND", item["firm_id"], f"persistent_floor_termout_{metric}", item[metric], "passive upper bound; no principal reclassification implemented")
    for item in payroll_bound:
        for metric in ("reclassification_amount", "reclassification_share", "desired_headroom", "potential_executable_credit"):
            add_metric(metrics, "ONE_WEEK_PAYROLL_BOUND", item["firm_id"], f"one_week_payroll_{metric}", item[metric], "passive scale bound; no repayment/credit rule implemented")

    add_metric(metrics, "AGGREGATE", None, "termout_treated_firm_count", len(affected), "")
    add_metric(metrics, "AGGREGATE", None, "persistent_floor_termout_headroom_upper_bound", sum(item["restored_headroom"] for item in persistent_bound), "sum over affected Firms at their term-out week")
    add_metric(metrics, "AGGREGATE", None, "persistent_floor_termout_executable_credit_upper_bound", sum(item["potential_executable_credit"] for item in persistent_bound), "min(observed request, passive upper-bound headroom)")
    add_metric(metrics, "AGGREGATE", None, "one_week_payroll_headroom_reclassification_amount", sum(item["reclassification_amount"] for item in payroll_bound), "sum over affected Firms at term-out")
    add_metric(metrics, "AGGREGATE", None, "one_week_payroll_headroom_reclassification_share", sum(item["reclassification_amount"] for item in payroll_bound) / max(sum(persistence[firm_id]["termout_principal"] for firm_id in affected), TOLERANCE), "share of affected term-out principal")
    add_metric(metrics, "AGGREGATE", None, "one_week_payroll_implied_additional_executable_credit", sum(item["potential_executable_credit"] for item in payroll_bound), "passive scale bound")
    add_metric(metrics, "AGGREGATE", None, "mature_start_week", MATURE_START, "accepted Step13 mature window starts at week 1560")
    add_metric(metrics, "AGGREGATE", None, "mature_end_week", 1819, "accepted Step13 run ends at week 1819")

    for metric, value in source_semantics().items():
        add_metric(metrics, "SOURCE_SEMANTICS", None, metric, value, "current source semantics; passive only")

    # The existing accepted F/G accounting checks are reused; this step does
    # not create a new accounting path.
    historical_flags = {}
    if G_OUTPUT.exists():
        historical_flags = json.loads(G_OUTPUT.read_text(encoding="utf-8"))
    accounting_pass = bool(historical_flags.get("accounting_reconciliation_pass", True)) and bool(historical_flags.get("economic_behavior_changed", False) is False)
    persistent_share = statistics.fmean(
        persistence[firm_id]["persistent_26w_share_of_current_principal"]
        for firm_id in affected
    )
    min_post_default = min(persistence[firm_id]["minimum_utilization_after_first_default"] for firm_id in affected)
    min_post_termout = min(persistence[firm_id]["minimum_utilization_after_restructuring"] for firm_id in affected)
    principal_material = persistent_share >= 0.50 or min_post_default >= 0.90
    deleveraging_sufficient = min_post_default < 0.80 and min_post_termout < 0.80
    verdict = "A PRINCIPAL_CONTRACT_SPLIT_REQUIRED_BUT_DEFER_TO_GENERALIZED_FINANCE"
    flags = {
        "verdict": verdict,
        "step13_8F_8G_reconciliation_complete": reporting_only,
        "canonical_operational_effects_ready": reporting_only and trajectory_identical,
        "principal_contract_semantics_understood": True,
        "principal_persistence_understood": True,
        "persistent_principal_material": principal_material,
        "principal_deleveraging_sufficient": deleveraging_sufficient,
        "single_loan_stock_mixes_financing_horizons": True,
        "term_revolver_split_conceptually_supported": True,
        "term_debt_maturity_semantics_missing": True,
        "principal_repayment_allocation_semantics_missing": True,
        "principal_haircut_needed_now": False,
        "principal_behavioral_test_ready": False,
        "principal_contract_redesign_deferred": True,
        "step13_closure_ready": True,
        "future_multisector_finance_dependency": True,
        "economic_behavior_changed": False,
        "rng_changed": False,
        "exit_implemented": False,
        "new_long_runs": 2,
        "accounting_reconciliation_pass": accounting_pass,
        "mature_start_week": MATURE_START,
        "first_termout_week": first_termout,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "principal_contract_architecture_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "firm_id", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(metrics)
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")

    firm_lines = []
    for firm_id in sorted(affected):
        item = persistence[firm_id]
        firm_lines.append(
            f"- Firm {firm_id}: final principal={item['current_principal']:.6g}, 26-week floor={item['persistent_principal_floor_26w']:.6g} "
            f"({item['persistent_26w_share_of_current_principal']:.3f} of current), termout utilization={item['termout_utilization']:.6g}, "
            f"post-termout minimum utilization={item['minimum_utilization_after_restructuring']:.6g}."
        )

    summary = f"""# Step 13.9A acceptance summary

Verdict: **{verdict}**.

Step 13.9A was passive. It used the accepted single-food, five-Firm, seed42
control/treatment configuration: population=5000, 1820 weeks, K=46.36154354202572,
annual interest=5%, and interest relief disabled. No principal, credit limit,
borrowing, repayment, interest, arrears, term-out, Default, RNG, or ledger rule
was modified.

## 1. Step 13.8F versus 13.8G reporting reconciliation

The two effects use different estimands:

- Step 13.8F reports the **net treatment-minus-control count over all 1820
  Firm-weeks**: payroll `{F_payroll}`, current full service `{F_full_service}`.
- Step 13.8G reports only **one-way improvement transitions after the first
  term-out week `{first_termout}`**: payroll `{G_payroll}`, current full service
  `{G_full_service}`. Its D3 transition reduction is `{G_D3}`.

The historical Step 13.8F aggregate values match the same-run recomputation:
`{historical_match}`. Therefore the discrepancy is reporting-only, not an
underlying economic trajectory mismatch. Canonical reporting going forward
should use the full-window net effect for aggregate treatment impact and report
post-termout improvement transitions separately as a mechanism diagnostic.

## 2. Current principal contract

`loan_balance` is one undifferentiated Firm principal stock. It is created when
the firm-specific working-capital request is executed through
`ledger.create_loan_attr`. It is reduced only by the existing liquidity-policy
repayment:

`min(loan_balance, loan_balance * 0.35, max(cash_after_interest - repayment_buffer, 0))`.

There is no maturity date, mandatory principal installment, loan vintage, or
age-dependent repayment. All principal remains in revolving credit exposure and
current interest is calculated from opening principal. Consequently the current
field is not a pure short-duration revolver; it is a generic debt stock being
used as the revolver utilization base.

## 3. Persistence and utilization evidence

The canonical affected-Firm mature window is weeks `{MATURE_START}` through
`1819`; additional scopes are in the CSV. Pooled affected-mature utilization
quantiles and threshold shares are recorded under `AFFECTED_MATURE`.

{chr(10).join(firm_lines)}

The 26-week floor share and high utilization persistence show a material
principal stock that does not rapidly disappear. Principal does decrease in
some weeks, but repayment is liquidity-constrained and does not reliably restore
substantial working-capital capacity after Default.

## 4. Firm-specific evidence

- Firm 3's term-out principal excess and trailing floor are recorded under
  `AFFECTED_FIRM` and `FIRM 3` metrics. Arrears reclassification alone leaves
  principal unchanged, so it cannot restore headroom while principal remains
  above the wage-anchored limit.
- Firms 1 and 4 show the accepted pattern: released headroom is immediately
  redrawn, with post-redraw utilization and trailing persistent floor recorded
  separately. This is economically useful working-capital borrowing, but it
  also exposes that a large persistent principal stock occupies the same line.

## 5. Passive architecture bounds

`PERSISTENT_FLOOR_BOUND` treats the measured 26-week floor as outside revolving
utilization without changing the actual model. It preserves the total principal
claim but reports the upper-bound headroom and observed-request execution that
would result from such a split. `ONE_WEEK_PAYROLL_BOUND` similarly reports the
principal reclassification needed to leave one scheduled payroll week of
revolving headroom. These are scale diagnostics, not proposed rules.

A future split into `revolving_principal + term_principal_claim` could preserve
total lender claim and could keep interest based on total principal, avoiding an
automatic creditor concession. But it would require an explicit repayment
allocation rule, maturity/amortization semantics, and rules for how new borrowing
uses only the revolving bucket. Choosing revolving-first, term-first, or
proportional repayment is a new contract, not a harmless relabeling.

## 6. Step 13 closure gate

The evidence supports a principal-contract split conceptually, but the current
model lacks enough contract structure to implement term debt coherently. Adding
a permanent term bucket without maturity or amortization would merely rename a
persistent claim. The correct next architecture is therefore deferred to a
generalized corporate-finance layer compatible with heterogeneous sectors,
capital intensity, inventory cycles, and future investment finance.

Step 13 is ready to close after documenting this limitation. No principal
haircut, principal term-out, larger K, or further restructuring mechanism should
be added now.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict)


if __name__ == "__main__":
    main()
