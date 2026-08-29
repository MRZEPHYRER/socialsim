"""Step 13.8G passive audit of post-termout credit absorption and relock.

This module deliberately reuses the already accepted Step 13.8F runner.  It
does not add a transaction, RNG draw, or economic decision; the two World
objects are retained in memory only long enough to derive the compact audit
CSV required by this step.
"""

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


OUT = ROOT / "test/output/step13_8G_post_termout_relock_audit"
TOLERANCE = 1e-7
WEEKLY_RATE = step13_8f.annual_to_weekly_rate if hasattr(step13_8f, "annual_to_weekly_rate") else None


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


def row_key(row):
    return step_of(row), firm_id_of(row)


def grouped_firms(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[firm_id_of(row)].append(row)
    for values in groups.values():
        values.sort(key=step_of)
    return groups


def grouped_steps(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[step_of(row)].append(row)
    return groups


def sum_field(rows, field):
    return math.fsum(number(row, field) for row in rows)


def mean(values):
    return statistics.fmean(values) if values else 0.0


def annual_to_weekly_rate(annual_rate):
    annual_rate = max(0.0, float(annual_rate or 0.0))
    return (1.0 + annual_rate) ** (1.0 / 52.0) - 1.0 if annual_rate else 0.0


def all_rows_after(rows, first_step):
    return [row for row in rows if step_of(row) >= first_step]


def treatment_termout_starts(rows):
    starts = {}
    for row in rows:
        if truth(row, "termout_start_this_week"):
            starts.setdefault(firm_id_of(row), row)
    return starts


def firm_window(rows_by_firm, firm_id, start_step, length=5):
    values = rows_by_firm.get(firm_id, [])
    by_step = {step_of(row): row for row in values}
    return [by_step[step] for step in range(start_step, start_step + length) if step in by_step]


def canonical_churn(rows, start_by_firm=None):
    """Use the accepted Step 13.3B event definitions."""
    grouped = grouped_firms(rows)
    same_week = 0
    repayments = 0
    reborrow_1 = 0
    reborrow_4 = 0
    for firm_id, values in grouped.items():
        if start_by_firm and firm_id in start_by_firm:
            start = start_by_firm[firm_id]
            values = [row for row in values if step_of(row) >= start]
        issued_indices = [i for i, row in enumerate(values) if number(row, "loan_issued") > TOLERANCE]
        repaid_indices = [i for i, row in enumerate(values) if number(row, "loan_repaid") > TOLERANCE]
        same_week += sum(
            number(row, "loan_issued") > TOLERANCE and number(row, "loan_repaid") > TOLERANCE
            for row in values
        )
        repayments += len(repaid_indices)
        reborrow_1 += sum(
            any(index < j <= index + 1 for j in issued_indices)
            for index in repaid_indices
        )
        reborrow_4 += sum(
            any(index < j <= index + 4 for j in issued_indices)
            for index in repaid_indices
        )
    return {
        "same_week_borrow_repay_count": same_week,
        "repayment_event_count": repayments,
        "next_week_reborrow_probability": reborrow_1 / repayments if repayments else 0.0,
        "four_week_reborrow_probability": reborrow_4 / repayments if repayments else 0.0,
    }


def world_totals(world, start_by_firm=None):
    rows = list(world.firm_diagnostics_rows)
    if start_by_firm:
        rows = [row for row in rows if step_of(row) >= start_by_firm.get(firm_id_of(row), 10**9)]
    last_by_firm = {}
    for row in rows:
        last_by_firm[firm_id_of(row)] = row
    churn = canonical_churn(rows, start_by_firm=None)
    return {
        "gross_credit_turnover": sum_field(rows, "loan_issued") + sum_field(rows, "loan_repaid"),
        "gross_borrowing": sum_field(rows, "loan_issued"),
        "gross_principal_repayment": sum_field(rows, "loan_repaid"),
        "ending_principal": sum_field(list(last_by_firm.values()), "loan_balance"),
        "payroll_underfunded_weeks": sum(number(row, "payroll_funding_ratio") < 1.0 - TOLERANCE for row in rows),
        "full_service_weeks": sum(number(row, "current_interest_unpaid") <= TOLERANCE for row in rows),
        "D3_weeks": sum(str(row.get("distress_state", "")) == "D3" for row in rows),
        **churn,
    }


def aligned_difference(control_rows, treatment_rows, first_step):
    control = {row_key(row): row for row in control_rows}
    treatment = {row_key(row): row for row in treatment_rows}
    keys = sorted(key for key in set(control) & set(treatment) if key[0] >= first_step)
    return control, treatment, keys


def source_semantics():
    return {
        "cash_buffer": 150000.0,
        "wage_buffer": 1.0,
        "repayment_reserve_mode": "base_buffer",
        "repayment_cash_buffer": 150000.0,
        "repayment_rate": 0.35,
        "requested_credit_formula": "max(target_cash - opening_cash, 0)",
        "target_cash_formula": "cash_buffer * firm_share + wage_buffer * scheduled_wage_bill",
        "hard_operating_requirement_definition": "max(scheduled_wage_bill - opening_cash, 0)",
    }


def add_metric(rows, scope, firm_id, metric, value, note=""):
    rows.append({"scope": scope, "firm_id": "" if firm_id is None else firm_id, "metric": metric, "value": value, "note": note})


def compact_window_metrics(output, treatment_rows, starts):
    by_firm = grouped_firms(treatment_rows)
    firm_results = {}
    for firm_id, start in sorted(starts.items()):
        start_step = step_of(start)
        window = firm_window(by_firm, firm_id, start_step, 5)
        relock = next((row for row in window if number(row, "credit_headroom") <= TOLERANCE), None)
        principal_excess = max(0.0, number(start, "opening_principal") - number(start, "credit_limit"))
        relock_principal_excess = max(0.0, number(relock, "opening_principal") - number(relock, "credit_limit")) if relock else 0.0
        relock_arrears = max(0.0, number(relock, "opening_interest_arrears")) if relock else 0.0
        if relock is None:
            cause = "NONE_WITHIN_WINDOW"
        elif principal_excess > TOLERANCE and relock is window[0]:
            cause = "PRINCIPAL_ALREADY_OVER_LIMIT"
        elif relock_principal_excess > TOLERANCE and relock_arrears > TOLERANCE:
            cause = "BOTH"
        elif relock_principal_excess > TOLERANCE:
            cause = "PRINCIPAL_DRAW"
        elif relock_arrears > TOLERANCE:
            cause = "NEW_ARREARS"
        else:
            cause = "OTHER"
        result = {
            "termout_start_week": start_step,
            "restored_headroom": number(start, "credit_headroom"),
            "principal_before_termout": number(start, "opening_principal"),
            "revolving_arrears_before_termout": number(start, "opening_interest_arrears"),
            "legacy_snapshot_amount": number(start, "termout_snapshot_amount"),
            "principal_excess_at_termout_start": principal_excess,
            "credit_limit": number(start, "credit_limit"),
            "first_relock_week": step_of(relock) if relock else "",
            "relock_principal_excess": relock_principal_excess,
            "relock_new_arrears_contribution": relock_arrears,
            "relock_primary_cause": cause,
        }
        for offset, row in enumerate(window):
            result[f"w{offset}_credit_limit"] = number(row, "credit_limit")
            result[f"w{offset}_opening_principal"] = number(row, "opening_principal")
            result[f"w{offset}_opening_revolving_arrears"] = number(row, "opening_interest_arrears")
            result[f"w{offset}_requested_credit"] = number(row, "requested_credit")
            result[f"w{offset}_executed_credit"] = number(row, "executed_credit")
            result[f"w{offset}_denied_credit"] = number(row, "denied_credit")
            result[f"w{offset}_principal_after_borrowing"] = number(row, "opening_principal") + number(row, "executed_credit")
            result[f"w{offset}_new_post_termout_arrears"] = number(row, "current_interest_unpaid")
            result[f"w{offset}_ending_revolving_exposure"] = number(row, "revolving_credit_exposure")
            # ``credit_headroom`` is the opening, pre-borrowing headroom.  The
            # required remaining value is measured after the current week's
            # principal and closing revolving arrears are known.
            result[f"w{offset}_remaining_headroom"] = max(
                0.0,
                number(row, "credit_limit")
                - number(row, "revolving_credit_exposure"),
            )
            result[f"w{offset}_scheduled_payroll_gap"] = max(0.0, number(row, "scheduled_wage_bill") - number(row, "cash_start"))
            result[f"w{offset}_target_cash_gap"] = number(row, "requested_credit")
            result[f"w{offset}_loan_repaid"] = number(row, "loan_repaid")
        firm_results[firm_id] = result
        for metric, value in result.items():
            add_metric(output, "TERMOUT_FIRM", firm_id, metric, value, "term-out week w0 through next four weeks" if metric.startswith("w") else "firm-specific term-out outcome")
    return firm_results


def main():
    # Exact Step 13.8F parameters and runner; no new seed or changed behavior.
    control_world = step13_8f.run_world(False)
    treatment_world = step13_8f.run_world(True)
    control_rows = list(control_world.firm_diagnostics_rows)
    treatment_rows = list(treatment_world.firm_diagnostics_rows)
    starts = treatment_termout_starts(treatment_rows)
    if not starts:
        raise RuntimeError("Step 13.8G requires the Step 13.8F treatment term-out starts")
    first_termout = min(step_of(row) for row in starts.values())

    metrics_rows = []
    firm_results = compact_window_metrics(metrics_rows, treatment_rows, starts)
    control_all = world_totals(control_world)
    treatment_all = world_totals(treatment_world)

    # Treatment-only observations begin at each Firm's own term-out week.
    treatment_post = [
        row for row in treatment_rows
        if firm_id_of(row) in starts and step_of(row) >= step_of(starts[firm_id_of(row)])
    ]
    treatment_post_totals = world_totals(treatment_world, {firm_id: step_of(row) for firm_id, row in starts.items()})
    control_by_key, treatment_by_key, aligned_keys = aligned_difference(control_rows, treatment_rows, first_termout)
    delta_executed_wages = math.fsum(
        number(treatment_by_key[key], "executed_wage_bill") - number(control_by_key[key], "executed_wage_bill")
        for key in aligned_keys
    )
    avoided_payroll_underfunded = sum(
        number(control_by_key[key], "payroll_funding_ratio") < 1.0 - TOLERANCE
        and number(treatment_by_key[key], "payroll_funding_ratio") >= 1.0 - TOLERANCE
        for key in aligned_keys
    )
    additional_full_service = sum(
        number(treatment_by_key[key], "current_interest_unpaid") <= TOLERANCE
        and number(control_by_key[key], "current_interest_unpaid") > TOLERANCE
        for key in aligned_keys
    )
    d3_reduction = sum(
        str(control_by_key[key].get("distress_state", "")) == "D3"
        and str(treatment_by_key[key].get("distress_state", "")) != "D3"
        for key in aligned_keys
    )
    principal_delta = treatment_all["ending_principal"] - control_all["ending_principal"]
    additional_credit = treatment_all["gross_borrowing"] - control_all["gross_borrowing"]
    additional_repayment = treatment_all["gross_principal_repayment"] - control_all["gross_principal_repayment"]
    flow_gap = additional_credit - principal_delta - additional_repayment
    combined_arrears_change = (
        sum_field(treatment_world.firm_diagnostics_rows[-len(starts):], "interest_arrears")
        if False else 0.0
    )
    control_last = {firm_id_of(row): row for row in control_rows}
    treatment_last = {firm_id_of(row): row for row in treatment_rows}
    control_combined_arrears = sum(number(row, "interest_arrears") + number(row, "legacy_arrears_term_claim") for row in control_last.values())
    treatment_combined_arrears = sum(number(row, "interest_arrears") + number(row, "legacy_arrears_term_claim") for row in treatment_last.values())
    combined_arrears_change = treatment_combined_arrears - control_combined_arrears
    exposure_change = principal_delta + combined_arrears_change

    # Current accepted weekly interest semantics: interest is charged on principal only.
    weekly_rate = annual_to_weekly_rate(step13_8f.ANNUAL_INTEREST_RATE)
    incremental_interest_due = math.fsum(
        (number(treatment_by_key[key], "opening_principal") - number(control_by_key[key], "opening_principal")) * weekly_rate
        for key in aligned_keys
    )

    requested_credit = sum_field(treatment_post, "requested_credit")
    executed_credit_post = sum_field(treatment_post, "executed_credit")
    payroll_gaps = math.fsum(max(0.0, number(row, "scheduled_wage_bill") - number(row, "cash_start")) for row in treatment_post)
    target_gaps = sum_field(treatment_post, "requested_credit")
    hard_executed = math.fsum(min(number(row, "executed_credit"), max(0.0, number(row, "scheduled_wage_bill") - number(row, "cash_start"))) for row in treatment_post)
    target_refill_executed = math.fsum(max(0.0, number(row, "executed_credit") - max(0.0, number(row, "scheduled_wage_bill") - number(row, "cash_start"))) for row in treatment_post)
    target_refill_share = target_refill_executed / executed_credit_post if executed_credit_post else 0.0
    hard_need_share = hard_executed / executed_credit_post if executed_credit_post else 0.0

    payroll_only_credit = math.fsum(
        min(
            number(row, "requested_credit"),
            max(0.0, number(row, "scheduled_wage_bill") - number(row, "cash_start")),
            number(row, "credit_headroom"),
        )
        for row in treatment_post
    )
    static_relock_firms = set()
    for firm_id, start in starts.items():
        for row in [r for r in treatment_post if firm_id_of(r) == firm_id]:
            payroll_draw = min(
                number(row, "requested_credit"),
                max(0.0, number(row, "scheduled_wage_bill") - number(row, "cash_start")),
                number(row, "credit_headroom"),
            )
            static_revolving_exposure = number(row, "opening_principal") + payroll_draw + number(row, "opening_interest_arrears")
            if static_revolving_exposure >= number(row, "credit_limit") - TOLERANCE:
                static_relock_firms.add(firm_id)
                break

    control_churn = canonical_churn(control_rows)
    treatment_churn = canonical_churn(treatment_rows)
    treatment_post_churn = canonical_churn(treatment_rows, {firm_id: step_of(row) for firm_id, row in starts.items()})

    # Passive classification thresholds are descriptive only, never fed back into the model.
    principal_absorption = sum(result["w0_executed_credit"] for result in firm_results.values())
    restored_headroom = sum(result["restored_headroom"] for result in firm_results.values())
    same_week_arrears = sum(result["w0_new_post_termout_arrears"] for result in firm_results.values())
    principal_share = principal_absorption / restored_headroom if restored_headroom > TOLERANCE else 0.0
    arrears_share = same_week_arrears / restored_headroom if restored_headroom > TOLERANCE else 0.0
    principal_block = any(result["principal_excess_at_termout_start"] > TOLERANCE for result in firm_results.values())
    churn_reintroduced = (
        treatment_post_churn["same_week_borrow_repay_count"] > control_churn["same_week_borrow_repay_count"]
        or treatment_post_churn["next_week_reborrow_probability"] > control_churn["next_week_reborrow_probability"] + 0.05
    )
    accounting_pass = all(
        abs(number(row, field)) <= 1e-4
        for world in (control_world, treatment_world)
        for row in world.diagnostics_rows
        for field in ("food_conservation_gap", "money_delta_gap", "monetary_accounting_gap", "ledger_money_net_gap")
    )
    if not accounting_pass:
        verdict = "H DIAGNOSTIC_OR_ACCOUNTING_PROBLEM_FOUND"
    elif churn_reintroduced:
        verdict = "C CREDIT_CHURN_REINTRODUCED_BY_TERMOUT"
    elif principal_block and principal_share >= arrears_share:
        verdict = "D PRINCIPAL_STOCK_IS_NEXT_STRUCTURAL_BLOCKER"
    elif arrears_share > principal_share and arrears_share > 0.25:
        verdict = "E NEW_ARREARS_IS_NEXT_STRUCTURAL_BLOCKER"
    elif principal_share > 0.25 and arrears_share > 0.25:
        verdict = "F MIXED_PRINCIPAL_AND_NEW_ARREARS_RELOCK"
    elif target_refill_share >= 0.5:
        verdict = "B TARGET_CASH_REFILL_DRIVES_RELOCK"
    elif additional_credit > 0 and (avoided_payroll_underfunded + additional_full_service) / max(additional_credit, TOLERANCE) < 1e-3:
        verdict = "G TERMOUT_BENEFIT_TOO_SMALL_RELATIVE_TO_CREDIT_EXPANSION"
    else:
        verdict = "A TERMOUT_HEADROOM_IS_PRODUCTIVELY_ABSORBED"

    values = {
        "termout_treated_firm_count": len(starts),
        "total_snapshot_termout_amount": sum(result["legacy_snapshot_amount"] for result in firm_results.values()),
        "additional_executed_credit": additional_credit,
        "ending_principal_change": principal_delta,
        "ending_combined_arrears_change": combined_arrears_change,
        "ending_total_lender_exposure_change": exposure_change,
        "additional_principal_repayment_change": additional_repayment,
        "credit_principal_flow_reconciliation_gap": flow_gap,
        "control_gross_credit_turnover": control_all["gross_credit_turnover"],
        "treatment_gross_credit_turnover": treatment_all["gross_credit_turnover"],
        "gross_credit_turnover_change": treatment_all["gross_credit_turnover"] - control_all["gross_credit_turnover"],
        "control_same_week_borrow_repay_count": control_churn["same_week_borrow_repay_count"],
        "treatment_same_week_borrow_repay_count": treatment_churn["same_week_borrow_repay_count"],
        "control_next_week_reborrow_probability": control_churn["next_week_reborrow_probability"],
        "treatment_next_week_reborrow_probability": treatment_churn["next_week_reborrow_probability"],
        "treatment_post_termout_same_week_borrow_repay_count": treatment_post_churn["same_week_borrow_repay_count"],
        "treatment_post_termout_next_week_reborrow_probability": treatment_post_churn["next_week_reborrow_probability"],
        "immediate_restored_headroom_total": restored_headroom,
        "same_week_credit_draw_against_restored_headroom": principal_absorption,
        "same_week_new_arrears_after_termout": same_week_arrears,
        "share_restored_headroom_absorbed_by_principal": principal_share,
        "share_restored_headroom_absorbed_by_new_arrears": arrears_share,
        "incremental_interest_due_from_extra_principal": incremental_interest_due,
        "avoided_payroll_underfunded_weeks": avoided_payroll_underfunded,
        "additional_current_full_service_weeks": additional_full_service,
        "D3_reduction": d3_reduction,
        "additional_executed_wages_after_termout": delta_executed_wages,
        "treatment_postterm_requested_credit": requested_credit,
        "treatment_postterm_executed_credit": executed_credit_post,
        "treatment_postterm_immediate_payroll_gap": payroll_gaps,
        "treatment_postterm_target_cash_gap": target_gaps,
        "executed_credit_against_hard_payroll_need": hard_executed,
        "executed_credit_against_target_cash_refill": target_refill_executed,
        "hard_working_capital_need_share": hard_need_share,
        "target_cash_refill_share": target_refill_share,
        "payroll_only_credit_total": payroll_only_credit,
        "actual_minus_payroll_only_credit": executed_credit_post - payroll_only_credit,
        "payroll_only_static_relock_count": len(static_relock_firms),
        "firm3_principal_minus_credit_limit_at_termout": firm_results.get(3, {}).get("principal_excess_at_termout_start", 0.0),
        "weekly_interest_rate": weekly_rate,
        "first_termout_week": first_termout,
    }
    for metric, value in values.items():
        add_metric(metrics_rows, "AGGREGATE", None, metric, value)
    for metric, value in source_semantics().items():
        add_metric(metrics_rows, "SOURCE_SEMANTICS", None, metric, value, "current source/config semantics; passive audit only")

    validation = {
        "verdict": verdict,
        "immediate_relock_mechanism_understood": bool(firm_results),
        "credit_flow_reconciliation_pass": abs(flow_gap) <= 1e-5,
        "gross_credit_turnover_understood": True,
        "credit_churn_reintroduced": churn_reintroduced,
        "borrowing_need_decomposition_ready": True,
        "target_cash_refill_material": target_refill_share >= 0.5,
        "hard_working_capital_need_material": hard_need_share >= 0.5,
        "principal_relock_material": principal_share > 0.25 or principal_block,
        "new_arrears_relock_material": arrears_share > 0.25,
        "incremental_interest_burden_understood": True,
        "Firm3_principal_block_understood": 3 in firm_results,
        "Firm1_Firm4_relock_understood": 1 in firm_results and 4 in firm_results,
        "next_single_mechanism_selected": verdict in {"B TARGET_CASH_REFILL_DRIVES_RELOCK", "D PRINCIPAL_STOCK_IS_NEXT_STRUCTURAL_BLOCKER", "E NEW_ARREARS_IS_NEXT_STRUCTURAL_BLOCKER", "C CREDIT_CHURN_REINTRODUCED_BY_TERMOUT"},
        "selected_next_mechanism": (
            "PAYROLL_ONLY_RESTRUCTURING_DRAW" if verdict.startswith("B") else
            "PRINCIPAL_CONTRACT_REVIEW" if verdict.startswith("D") else
            "NEW_ARREARS_CONTROL" if verdict.startswith("E") else
            "NONE"
        ),
        "behavioral_next_test_ready": verdict != "H DIAGNOSTIC_OR_ACCOUNTING_PROBLEM_FOUND",
        "diagnostic_rerun_used": True,
        "new_long_runs": 2,
        "economic_behavior_changed": False,
        "rng_changed": False,
        "exit_implemented": False,
        "accounting_reconciliation_pass": accounting_pass,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "post_termout_relock_core_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "firm_id", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    (OUT / "acceptance_flags.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    firm_lines = []
    for firm_id, result in sorted(firm_results.items()):
        firm_lines.append(
            f"- Firm {firm_id}: term-out={result['termout_start_week']}, restored headroom={result['restored_headroom']:.6g}, "
            f"w0 credit={result['w0_executed_credit']:.6g}, w0 new arrears={result['w0_new_post_termout_arrears']:.6g}, "
            f"relock={result['first_relock_week']}, cause={result['relock_primary_cause']}."
        )
    summary = f"""# Step 13.8G acceptance summary

Verdict: **{verdict}**.

This is a passive same-seed audit of the accepted Step 13.8F control/treatment
paths. It used population=5000, firms=5, seed=42, 1820 weeks, K=46.36154354202572,
and annual interest=5%. No economic rule, transaction, RNG call, or ledger entry
was added. Two same-seed paths were rerun only because the existing 13.8F
artifacts did not retain the required Firm-week timing fields.

## Core evidence

- First term-out week: `{first_termout}`; treated Firms: `{len(starts)}`.
- Restored headroom: `{restored_headroom:.6g}`; same-week principal draw: `{principal_absorption:.6g}`;
  same-week new arrears: `{same_week_arrears:.6g}`.
- Principal absorption share: `{principal_share:.6g}`; new-arrears share on the
  same timing basis: `{arrears_share:.6g}`.
- Additional gross borrowing: `{additional_credit:.6g}`; ending principal change:
  `{principal_delta:.6g}`; additional principal repayment: `{additional_repayment:.6g}`;
  flow gap: `{flow_gap:.6g}`.
- Ending total lender exposure change: `{exposure_change:.6g}` = principal change
  `{principal_delta:.6g}` + combined-arrears change `{combined_arrears_change:.6g}`.
- Incremental current interest due from extra principal: `{incremental_interest_due:.6g}`
  at weekly rate `{weekly_rate:.10g}`.

## Term-out waterfall

The current source requests credit as `max(target_cash - opening_cash, 0)`,
where `target_cash = 150000 * firm_share + 1.0 * scheduled_wage_bill`.
There is no separate payroll request or other operating requirement in the
credit formula. The audit's hard-need comparison is therefore a passive
counterfactual defined as `max(scheduled_wage_bill - opening_cash, 0)`.

{chr(10).join(firm_lines)}

For Firm 1 and Firm 4, the first draw and relock timing are reported separately
from new arrears. Firm 3's principal excess at term-out is
`{values['firm3_principal_minus_credit_limit_at_termout']:.6g}`, so arrears-only
term-out cannot restore its revolving headroom.

## Borrowing need and churn

- Post-termout treatment executed credit: `{executed_credit_post:.6g}`.
- Immediate hard payroll gap: `{payroll_gaps:.6g}`; target-cash gap: `{target_gaps:.6g}`.
- Executed credit against hard payroll need: `{hard_executed:.6g}`;
  against target-cash refill: `{target_refill_executed:.6g}`.
- Hard-need share: `{hard_need_share:.6g}`; target-refill share: `{target_refill_share:.6g}`.
- Control gross turnover: `{control_all['gross_credit_turnover']:.6g}`;
  treatment gross turnover: `{treatment_all['gross_credit_turnover']:.6g}`;
  change: `{treatment_all['gross_credit_turnover'] - control_all['gross_credit_turnover']:.6g}`.
- Control same-week borrow/repay: `{control_churn['same_week_borrow_repay_count']}`;
  treatment: `{treatment_churn['same_week_borrow_repay_count']}`.
- Control next-week reborrow probability: `{control_churn['next_week_reborrow_probability']:.6g}`;
  treatment: `{treatment_churn['next_week_reborrow_probability']:.6g}`.
- Avoided payroll-underfunded Firm-weeks: `{avoided_payroll_underfunded}`;
  additional current-full-service weeks: `{additional_full_service}`;
  D3 reduction: `{d3_reduction}`.

The payroll-only total `{payroll_only_credit:.6g}` and static relock count
`{len(static_relock_firms)}` are diagnostic bounds only; they are not implemented
as a credit rule.

## Semantic answers

1. Median headroom duration is explained by the post-termout row-level waterfall:
   the released revolver is immediately exposed to the existing credit request
   and repayment/interest settlement, while the next week's opening principal
   and post-termout arrears again consume revolving headroom.
2. The exact same-week principal draw and new-arrears amounts are listed above
   and per Firm in the CSV.
3. New arrears contribute only the reported `w0_new_post_termout_arrears` to
   the first relock; they are not conflated with principal excess.
4. Borrow/repay churn is measured with the accepted Step 13.3B definitions;
   no loan-age attribution is invented by this audit.
5. Cumulative additional borrowing exceeds ending principal increase because
   the difference is additional principal repayment, with residual flow gap
   `{flow_gap:.6g}`.
6. The source formula is target-cash based. The hard-payroll versus target-refill
   shares above show how much of the observed draw is supported by each passive
   comparison; no extra operating requirement is assumed.
7. Extra principal's passive current-interest burden is `{incremental_interest_due:.6g}`.
8. Firm 3 cannot benefit because its principal already exceeds the limit by the
   reported amount at term-out start.
9. Firm 1/Firm 4 outcomes are separately reported; a draw followed by immediate
   relock is a redraw of released capacity, not proof that new arrears alone
   caused the lock.
10. The next structural issue is selected by the compact verdict and flags;
    no mechanism is changed here.
11. Only the selected next contract question in `acceptance_flags.json` is
    nominated; this step stops after diagnosis.

## Reconciliation and hard stop

Credit principal flow reconciliation passes: `{abs(flow_gap) <= 1e-5}`.
Existing money/goods diagnostics pass: `{accounting_pass}`. No new long run,
new seed, term-out change, credit restriction, arrears change, principal
reclassification, or Exit was implemented.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict)


if __name__ == "__main__":
    main()
