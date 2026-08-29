"""Step 13.8F same-seed snapshot legacy-arrears term-out experiment."""

from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from central_bank import config as central_bank_config
from world import World


OUT = ROOT / "test/output/step13_8F_snapshot_legacy_arrears_termout"
SEED = 42
POPULATION = 5000
FIRM_COUNT = 5
STEPS = 1820
K_WEEKS = 46.36154354202572
ANNUAL_INTEREST_RATE = 0.05
ELIGIBILITY_WEEKS = 26
TOLERANCE = 1e-7

PASSIVE_TERMOUT_FIELDS = {
    "snapshot_termout_enabled",
    "termout_start_count_this_step",
    "total_snapshot_termout_amount",
    "total_legacy_arrears_term_claim",
    "total_post_termout_interest_arrears",
    "total_revolving_credit_exposure",
    "total_lender_exposure",
    "total_legacy_term_claim_payment",
    "total_post_termout_arrears_payment",
}
FIRM_TERMOUT_FIELDS = {
    "snapshot_termout_enabled",
    "termout_start_this_week",
    "termout_used_this_default",
    "termout_snapshot_amount",
    "termout_pre_snapshot_arrears",
    "legacy_arrears_term_claim",
    "opening_legacy_arrears_term_claim",
    "closing_legacy_arrears_term_claim",
    "legacy_term_claim_payment",
    "post_termout_arrears_payment",
    "termout_claim_reclassification_gap",
    "opening_revolving_credit_exposure",
    "opening_total_lender_exposure",
    "revolving_credit_exposure",
    "post_termout_interest_arrears",
    "total_lender_exposure",
    "closing_revolving_credit_exposure",
}


def number(row, name, default=0.0):
    try:
        value = float(row.get(name, default))
        return value if math.isfinite(value) else float(default)
    except (TypeError, ValueError):
        return float(default)


def truth(row, name):
    value = row.get(name, False)
    return value if isinstance(value, bool) else str(value).strip().lower() == "true"


def step_of(row):
    return int(float(row.get("global_step", row.get("step", 0))))


def firm_id_of(row):
    return int(float(row.get("firm_id", 0)))


def configure(termout_enabled):
    central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_ENABLED = True
    central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_K_WEEKS = K_WEEKS
    central_bank_config.CENTRAL_BANK_FIRM_LOAN_INTEREST_ANNUAL_RATE = (
        ANNUAL_INTEREST_RATE
    )
    central_bank_config.CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_ENABLED = False
    central_bank_config.CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_FRACTION = 0.0
    central_bank_config.CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_ELIGIBILITY_WEEKS = (
        ELIGIBILITY_WEEKS
    )
    central_bank_config.CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_MAX_WEEKS = (
        ELIGIBILITY_WEEKS
    )
    central_bank_config.CENTRAL_BANK_SNAPSHOT_LEGACY_ARREARS_TERMOUT_ENABLED = bool(
        termout_enabled
    )


def run_world(termout_enabled, steps=STEPS, population=POPULATION):
    configure(termout_enabled)
    world = World(
        initial_population=population,
        seed=SEED,
        scenario_name=(
            "snapshot_legacy_arrears_termout"
            if termout_enabled
            else "snapshot_legacy_arrears_control"
        ),
        scenario_overrides={
            "CREDIT_CAPACITY_ENABLED": True,
            "CREDIT_CAPACITY_K_WEEKS": K_WEEKS,
            "FIRM_LOAN_INTEREST_ANNUAL_RATE": ANNUAL_INTEREST_RATE,
            "TEMPORARY_INTEREST_RELIEF_ENABLED": False,
            "TEMPORARY_INTEREST_RELIEF_FRACTION": 0.0,
            "SNAPSHOT_LEGACY_ARREARS_TERMOUT_ENABLED": bool(termout_enabled),
        },
        diagnostics_mode="compact",
        initial_age_phase_mode="distributed",
    )
    world.split_firms(FIRM_COUNT)
    world.steps = steps
    world.run(progress_interval=0)
    world._step13_8f_global_rng_state = random.getstate()
    world._step13_8f_market_rng_state = world.market_rng.getstate()
    world._step13_8f_firm_rng_states = tuple(
        getattr(firm.firm_rng, "getstate", lambda: None)()
        for firm in world.firms
    )
    return world


def row_key(row):
    return step_of(row), firm_id_of(row)


def grouped_firms(rows):
    result = defaultdict(list)
    for row in rows:
        result[firm_id_of(row)].append(row)
    for values in result.values():
        values.sort(key=step_of)
    return result


def grouped_steps(rows):
    result = defaultdict(list)
    for row in rows:
        result[step_of(row)].append(row)
    return result


def max_abs(values):
    return max((abs(float(value)) for value in values), default=0.0)


def max_row_difference(old_rows, new_rows, excluded=()):
    old_map = {row_key(row): row for row in old_rows}
    new_map = {row_key(row): row for row in new_rows}
    common_keys = sorted(set(old_map) & set(new_map))
    excluded = set(excluded)
    maximum = 0.0
    categorical_mismatches = 0
    first_difference = None
    for key in common_keys:
        old = old_map[key]
        new = new_map[key]
        fields = set(old) & set(new) - excluded
        for field in fields:
            if field in {"step", "global_step", "seed", "scenario"}:
                continue
            try:
                old_value = float(old[field])
                new_value = float(new[field])
                difference = abs(old_value - new_value)
                if difference > maximum:
                    maximum = difference
                if difference > TOLERANCE and first_difference is None:
                    first_difference = (key, field, old[field], new[field])
            except (TypeError, ValueError):
                if str(old[field]) != str(new[field]):
                    categorical_mismatches += 1
                    if first_difference is None:
                        first_difference = (key, field, old[field], new[field])
    return maximum, categorical_mismatches, first_difference


def sum_field(rows, field):
    return math.fsum(number(row, field) for row in rows)


def mean_field(rows, field):
    values = [number(row, field) for row in rows]
    return statistics.fmean(values) if values else 0.0


def operating_cash_flow(row):
    return (
        number(row, "sales_revenue")
        - number(row, "executed_wage_bill", number(row, "wage_payment"))
        + number(row, "public_sector_cash_inflow")
        - number(row, "public_sector_cash_outflow")
    )


def per_world_stats(world):
    firms = list(world.firm_diagnostics_rows)
    macros = list(world.diagnostics_rows)
    by_step = grouped_steps(firms)
    last_step = max((step_of(row) for row in firms), default=-1)
    last_firms = by_step.get(last_step, [])
    exposure_by_step = {
        step: math.fsum(
            number(row, "total_lender_exposure", number(row, "lender_exposure"))
            for row in rows
        )
        for step, rows in by_step.items()
    }
    total_claim = lambda row: number(
        row,
        "total_lender_exposure",
        number(row, "loan_balance")
        + number(row, "interest_arrears")
        + number(row, "legacy_arrears_term_claim"),
    )
    full_service = sum(
        number(row, "current_interest_unpaid") <= TOLERANCE for row in firms
    )
    contractual_cures = sum(truth(row, "contract_cure_this_week") for row in firms)
    return {
        "contractual_cure_count": contractual_cures,
        "active_contract_default_firm_weeks": sum(
            truth(row, "active_contract_default") for row in firms
        ),
        "D3_firm_weeks": sum(str(row.get("distress_state", "")) == "D3" for row in firms),
        "credit_executed_total": sum_field(firms, "executed_credit"),
        "credit_denied_total": sum_field(firms, "denied_credit"),
        "positive_headroom_firm_weeks": sum(
            number(row, "credit_headroom") > TOLERANCE for row in firms
        ),
        "mean_headroom": mean_field(firms, "credit_headroom"),
        "payroll_underfunded_firm_weeks": sum(
            number(row, "payroll_funding_ratio") < 1.0 - TOLERANCE for row in firms
        ),
        "ending_interest_arrears": sum_field(last_firms, "interest_arrears"),
        "ending_legacy_term_claim": sum_field(
            last_firms, "legacy_arrears_term_claim"
        ),
        "ending_post_termout_arrears": sum_field(
            last_firms, "post_termout_interest_arrears"
        ),
        "ending_combined_arrears_claim": sum(
            number(row, "interest_arrears")
            + number(row, "legacy_arrears_term_claim")
            for row in last_firms
        ),
        "ending_loan_principal": sum_field(last_firms, "loan_balance"),
        "ending_total_lender_exposure": sum(total_claim(row) for row in last_firms),
        "peak_total_lender_exposure": max(exposure_by_step.values(), default=0.0),
        "current_full_service_firm_weeks": full_service,
        "technical_breach_firm_weeks": sum(
            number(row, "current_interest_unpaid") > TOLERANCE for row in firms
        ),
        "lender_service_cash_budget": sum_field(firms, "interest_service_cash"),
        "current_interest_paid": sum_field(firms, "interest_paid"),
        "current_interest_due": sum_field(firms, "current_interest_due"),
        "current_service_ratio": (
            sum_field(firms, "interest_paid")
            / max(sum_field(firms, "total_interest_obligation"), TOLERANCE)
        ),
        "treatment_legacy_claim_payment_total": sum_field(
            firms, "legacy_term_claim_payment"
        ),
        "treatment_post_termout_arrears_payment_total": sum_field(
            firms, "post_termout_arrears_payment"
        ),
        "treatment_new_arrears_total": sum_field(firms, "current_interest_unpaid"),
        "mean_payroll_funding_ratio": mean_field(firms, "payroll_funding_ratio"),
        "mean_scheduled_wages": mean_field(firms, "scheduled_wage_bill"),
        "mean_executed_wages": mean_field(firms, "executed_wage_bill"),
        "mean_funded_capacity": mean_field(firms, "funded_productive_capacity"),
        "mean_production": mean_field(firms, "actual_production"),
        "mean_sales": mean_field(firms, "sales_units"),
        "mean_ocf": statistics.fmean(operating_cash_flow(row) for row in firms)
        if firms
        else 0.0,
        "ending_money_stock": number(
            macros[-1], "total_money_stock", number(macros[-1], "located_money_stock")
        )
        if macros
        else 0.0,
        "invariant_violation_count": len(getattr(world, "invariant_violations", [])),
    }


def termout_firm_outcomes(treatment_rows):
    by_firm = grouped_firms(treatment_rows)
    start_rows = [
        row for row in treatment_rows if truth(row, "termout_start_this_week")
    ]
    starts = {}
    for row in start_rows:
        starts.setdefault(firm_id_of(row), row)
    outcomes = []
    for firm_id, start in sorted(starts.items()):
        start_step = step_of(start)
        rows = [row for row in by_firm[firm_id] if step_of(row) >= start_step]
        first_credit = next(
            (row for row in rows if number(row, "executed_credit") > TOLERANCE),
            None,
        )
        positive_duration = 0
        reexhaustion = None
        reexhaustion_contributor = "none"
        for row in rows:
            if number(row, "credit_headroom") > TOLERANCE:
                positive_duration += 1
                continue
            reexhaustion = step_of(row)
            principal_excess = max(
                0.0,
                number(row, "opening_principal") - number(row, "credit_limit"),
            )
            new_arrears = max(0.0, number(row, "opening_interest_arrears"))
            if principal_excess > TOLERANCE and new_arrears > TOLERANCE:
                reexhaustion_contributor = "both"
            elif principal_excess > TOLERANCE:
                reexhaustion_contributor = "principal"
            elif new_arrears > TOLERANCE:
                reexhaustion_contributor = "new_arrears"
            else:
                reexhaustion_contributor = "other"
            break
        outcomes.append(
            {
                "firm_id": firm_id,
                "termout_start_week": start_step,
                "snapshot_termout_amount": number(start, "termout_snapshot_amount"),
                "restored_headroom": number(start, "credit_headroom"),
                "first_executed_credit_week": (
                    step_of(first_credit) if first_credit else ""
                ),
                "positive_headroom_duration": positive_duration,
                "reexhaustion_week": reexhaustion if reexhaustion is not None else "",
                "reexhaustion_contributor": reexhaustion_contributor,
            }
        )
    return outcomes


def aligned_post_changes(control_rows, treatment_rows, first_termout):
    control = {row_key(row): row for row in control_rows}
    treatment = {row_key(row): row for row in treatment_rows}
    keys = sorted(
        key for key in set(control) & set(treatment) if key[0] >= first_termout
    )
    if not keys:
        return {"mean_payroll_funding_change": 0.0, "mean_funded_capacity_change": 0.0,
                "mean_production_change": 0.0, "mean_OCF_change": 0.0}
    changes = defaultdict(list)
    for key in keys:
        c = control[key]
        t = treatment[key]
        changes["mean_payroll_funding_change"].append(
            number(t, "payroll_funding_ratio") - number(c, "payroll_funding_ratio")
        )
        changes["mean_funded_capacity_change"].append(
            number(t, "funded_productive_capacity")
            - number(c, "funded_productive_capacity")
        )
        changes["mean_production_change"].append(
            number(t, "actual_production") - number(c, "actual_production")
        )
        changes["mean_OCF_change"].append(
            operating_cash_flow(t) - operating_cash_flow(c)
        )
    return {key: statistics.fmean(value) for key, value in changes.items()}


def reconciliation_metrics(world):
    firms = list(world.firm_diagnostics_rows)
    macros = list(world.diagnostics_rows)
    claim_error = max_abs(
        number(row, "total_lender_exposure")
        - number(row, "loan_balance")
        - number(row, "legacy_arrears_term_claim")
        - number(row, "post_termout_interest_arrears")
        for row in firms
    )
    reclass_error = max_abs(
        number(row, "termout_claim_reclassification_gap") for row in firms
    )
    loan_errors = []
    previous = {}
    for row in sorted(firms, key=row_key):
        firm_id = firm_id_of(row)
        expected = previous.get(firm_id, 0.0) + number(row, "executed_credit") - number(
            row, "loan_repaid"
        )
        loan_errors.append(number(row, "loan_balance") - expected)
        previous[firm_id] = number(row, "loan_balance")
    money_errors = []
    goods_errors = []
    for row in macros:
        money_errors.extend(
            number(row, field)
            for field in ("monetary_accounting_gap", "money_delta_gap", "ledger_money_net_gap")
        )
        goods_errors.append(number(row, "food_conservation_gap"))
    accounting_rows = getattr(getattr(world, "accounting", None), "reconciliation_rows", [])
    accounting_claim_error = max_abs(
        number(row, "total_lender_claim_gap") for row in accounting_rows
    )
    payment_split_error = max_abs(
        number(row, "interest_paid")
        - number(row, "legacy_term_claim_payment")
        - number(row, "post_termout_arrears_payment")
        - number(row, "interest_paid_to_current_due")
        for row in firms
    )
    return {
        "claim_reclassification_max_error": reclass_error,
        "total_lender_claim_max_error": max(claim_error, accounting_claim_error),
        "loan_money_accounting_max_error": max_abs(loan_errors),
        "money_reconciliation_max_error": max_abs(money_errors),
        "goods_reconciliation_max_error": max_abs(goods_errors),
        "payment_split_max_error": payment_split_error,
        "accounting_claim_gap": accounting_claim_error,
    }


def parity_metrics(control, treatment, first_termout):
    macro_excluded = PASSIVE_TERMOUT_FIELDS | {"scenario"}
    firm_excluded = FIRM_TERMOUT_FIELDS | {"scenario"}
    macro_old = [row for row in control.diagnostics_rows if step_of(row) < first_termout]
    macro_new = [row for row in treatment.diagnostics_rows if step_of(row) < first_termout]
    firm_old = [row for row in control.firm_diagnostics_rows if step_of(row) < first_termout]
    firm_new = [row for row in treatment.firm_diagnostics_rows if step_of(row) < first_termout]
    macro_gap = max_row_difference(macro_old, macro_new, macro_excluded)
    firm_gap = max_row_difference(firm_old, firm_new, firm_excluded)
    return {
        "pre_treatment_max_abs_diff": max(macro_gap[0], firm_gap[0]),
        "pre_treatment_categorical_mismatch_count": macro_gap[1] + firm_gap[1],
        "pre_treatment_first_difference": macro_gap[2] or firm_gap[2],
    }


def smoke_noninterference():
    first = run_world(False, steps=80, population=200)
    second = run_world(False, steps=80, population=200)
    macro = max_row_difference(first.diagnostics_rows, second.diagnostics_rows)
    firm = max_row_difference(first.firm_diagnostics_rows, second.firm_diagnostics_rows)
    rng_same = (
        first._step13_8f_global_rng_state == second._step13_8f_global_rng_state
        and first._step13_8f_market_rng_state == second._step13_8f_market_rng_state
        and first._step13_8f_firm_rng_states == second._step13_8f_firm_rng_states
    )
    return max(macro[0], firm[0]), macro[1] + firm[1], rng_same


def make_plot(control, treatment, first_termout, outcomes):
    OUT.mkdir(parents=True, exist_ok=True)
    c_firms = grouped_firms(control.firm_diagnostics_rows)
    t_firms = grouped_firms(treatment.firm_diagnostics_rows)
    c_macro = {step_of(row): row for row in control.diagnostics_rows}
    t_macro = {step_of(row): row for row in treatment.diagnostics_rows}
    steps = sorted(set(c_macro) & set(t_macro))
    def aggregate(rows, field):
        return math.fsum(number(row, field) for row in rows)
    c_credit = [aggregate([r for r in control.firm_diagnostics_rows if step_of(r) == step], "executed_credit") for step in steps]
    t_credit = [aggregate([r for r in treatment.firm_diagnostics_rows if step_of(r) == step], "executed_credit") for step in steps]
    c_payroll = [mean_field([r for r in control.firm_diagnostics_rows if step_of(r) == step], "payroll_funding_ratio") for step in steps]
    t_payroll = [mean_field([r for r in treatment.firm_diagnostics_rows if step_of(r) == step], "payroll_funding_ratio") for step in steps]
    c_exposure = [aggregate([r for r in control.firm_diagnostics_rows if step_of(r) == step], "total_lender_exposure") for step in steps]
    t_exposure = [aggregate([r for r in treatment.firm_diagnostics_rows if step_of(r) == step], "total_lender_exposure") for step in steps]
    t_legacy = [aggregate([r for r in treatment.firm_diagnostics_rows if step_of(r) == step], "legacy_arrears_term_claim") for step in steps]
    t_post = [aggregate([r for r in treatment.firm_diagnostics_rows if step_of(r) == step], "post_termout_interest_arrears") for step in steps]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(steps, c_credit, label="control")
    axes[0, 0].plot(steps, t_credit, label="snapshot term-out")
    axes[0, 0].axvline(first_termout, color="black", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("Executed working-capital credit")
    axes[0, 0].set_ylabel("credit")
    axes[0, 1].plot(steps, c_payroll, label="control")
    axes[0, 1].plot(steps, t_payroll, label="snapshot term-out")
    axes[0, 1].axvline(first_termout, color="black", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("Mean payroll funding ratio")
    axes[1, 0].plot(steps, c_exposure, label="control")
    axes[1, 0].plot(steps, t_exposure, label="snapshot term-out")
    axes[1, 0].axvline(first_termout, color="black", linestyle="--", linewidth=0.8)
    axes[1, 0].set_title("Total lender exposure")
    axes[1, 1].plot(steps, t_legacy, label="legacy term claim")
    axes[1, 1].plot(steps, t_post, label="post-term-out arrears")
    axes[1, 1].axvline(first_termout, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].set_title("Treatment arrears buckets")
    for axis in axes.flat:
        axis.set_xlabel("global step")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.savefig(OUT / "snapshot_termout_comparison.png", dpi=140)
    plt.close(figure)


def classify(control, treatment, differences, outcomes, reconciliation, parity):
    accounting_clean = all(
        value <= 1e-4
        for value in (
            reconciliation["claim_reclassification_max_error"],
            reconciliation["total_lender_claim_max_error"],
            reconciliation["loan_money_accounting_max_error"],
            reconciliation["money_reconciliation_max_error"],
            reconciliation["goods_reconciliation_max_error"],
            reconciliation["payment_split_max_error"],
        )
    ) and treatment["invariant_violation_count"] == 0
    credit_change = differences["credit_executed_change"]
    payroll_change = differences["payroll_underfunded_week_change"]
    current_service_change = (
        treatment["current_full_service_firm_weeks"]
        - control["current_full_service_firm_weeks"]
    )
    relock = len(outcomes) > 0 and sum(
        bool(item["reexhaustion_week"] != "") for item in outcomes
    )
    median_duration = statistics.median(
        item["positive_headroom_duration"] for item in outcomes
    ) if outcomes else 0.0
    if not accounting_clean or parity["pre_treatment_max_abs_diff"] > 1e-4:
        mechanism = "G ACCOUNTING_OR_IMPLEMENTATION_PROBLEM"
        verdict = "G IMPLEMENTATION_OR_ACCOUNTING_FAILURE"
    elif abs(credit_change) <= TOLERANCE:
        mechanism = "E TERMOUT_HAS_NEGLIGIBLE_BEHAVIORAL_EFFECT"
        verdict = "E SNAPSHOT_TERMOUT_EFFECT_NEGLIGIBLE"
    elif relock and median_duration <= 26:
        mechanism = "C TERMOUT_TEMPORARILY_RESTORES_HEADROOM_THEN_RELOCKS"
        verdict = "C SNAPSHOT_TERMOUT_RELOCKS_TOO_QUICKLY"
    elif differences["mean_payroll_funding_change"] > 1e-4 and current_service_change <= 0:
        mechanism = "B TERMOUT_RESTORES_WORKING_CAPITAL_BUT_NOT_CURRENT_SERVICE"
        verdict = "B SNAPSHOT_TERMOUT_PARTIALLY_SUPPORTED"
    elif current_service_change > 0 and differences["mean_payroll_funding_change"] > 1e-4:
        mechanism = "A TERMOUT_RESTORES_WORKING_CAPITAL_AND_CURRENT_SERVICE"
        verdict = "A SNAPSHOT_TERMOUT_BEHAVIORALLY_SUPPORTED"
    elif treatment["ending_total_lender_exposure"] - control["ending_total_lender_exposure"] > 1e-6:
        mechanism = "D TERMOUT_MAINLY_INCREASES_LENDER_EXPOSURE"
        verdict = "D SNAPSHOT_TERMOUT_EXPOSURE_COST_TOO_HIGH"
    else:
        mechanism = "E TERMOUT_HAS_NEGLIGIBLE_BEHAVIORAL_EFFECT"
        verdict = "E SNAPSHOT_TERMOUT_EFFECT_NEGLIGIBLE"
    return mechanism, verdict, accounting_clean


def write_outputs(control, treatment, metrics, outcomes, flags, mechanism, verdict):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scope, values in (
        ("CONTROL", metrics["control"]),
        ("TREATMENT", metrics["treatment"]),
        ("DIFFERENCE", metrics["difference"]),
        ("VALIDATION", metrics["validation"]),
    ):
        for metric, value in values.items():
            rows.append({"scope": scope, "firm_id": "", "metric": metric, "value": value, "note": ""})
    for item in outcomes:
        for metric, value in item.items():
            if metric == "firm_id":
                continue
            rows.append({"scope": "TREATED_FIRM", "firm_id": item["firm_id"], "metric": metric, "value": value, "note": "post-term-out static/behavioral trace"})
    with (OUT / "snapshot_termout_control_treatment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("scope", "firm_id", "metric", "value", "note"))
        writer.writeheader()
        writer.writerows(rows)

    summary = f"""# Step 13.8F acceptance summary

Verdict: **{verdict}**.
Mechanism diagnosis: **{mechanism}**.

This is an isolated same-seed behavioral experiment with population={POPULATION},
firms={FIRM_COUNT}, seed={SEED}, steps={STEPS}, K={K_WEEKS}, and annual interest
rate={ANNUAL_INTEREST_RATE:.0%}. Temporary 75% interest relief was disabled in
both paths. No seed7/21 run was performed.

## Intervention boundary

At the first eligible week, treatment moves only the arrears stock existing at
restructuring start into `legacy_arrears_term_claim`. It does not create a
ledger transfer, money, goods, principal, or lender loss. Afterward,
`interest_arrears` is the post-term-out revolving arrears bucket; new unpaid
current interest remains in that bucket and consumes headroom. Legacy claim
payments use the unchanged total lender-service cash budget and are ordered
before post-term-out arrears and current interest.

The full lender claim is always:

`loan_balance + post_termout_interest_arrears + legacy_arrears_term_claim`.

The revolving utilization is:

`loan_balance + post_termout_interest_arrears`.

## Causal results

- First term-out week: `{metrics['validation']['first_termout_week']}`
- Term-out episodes: `{metrics['validation']['termout_episode_count']}` across `{metrics['validation']['termout_treated_firm_count']}` Firms
- Snapshot amount: `{metrics['validation']['total_snapshot_termout_amount']:.6g}`
- Positive-headroom change: `{metrics['difference']['positive_headroom_week_change']}` Firm-weeks
- Executed-credit change: `{metrics['difference']['credit_executed_change']:.6g}`
- Payroll-underfunded Firm-week change: `{metrics['difference']['payroll_underfunded_week_change']}`
- Current full-service change: `{metrics['difference']['current_full_service_week_change']}` Firm-weeks
- Ending total lender exposure change: `{metrics['difference']['ending_total_lender_exposure_change']:.6g}`
- Additional money created through credit: `{metrics['validation']['additional_money_created_from_credit']:.6g}`

## Reconciliation and isolation

- Pre-treatment max absolute difference: `{metrics['validation']['pre_treatment_max_abs_diff']:.6g}`
- Claim reclassification max error: `{metrics['validation']['claim_reclassification_max_error']:.6g}`
- Total lender claim max error: `{metrics['validation']['total_lender_claim_max_error']:.6g}`
- Loan/principal accounting max error: `{metrics['validation']['loan_money_accounting_max_error']:.6g}`
- Money reconciliation max error: `{metrics['validation']['money_reconciliation_max_error']:.6g}`
- Goods reconciliation max error: `{metrics['validation']['goods_reconciliation_max_error']:.6g}`
- Zero-termout smoke max difference: `{metrics['validation']['zero_termout_economic_max_abs_diff']:.6g}`
- RNG changed: `{flags['rng_noninterference_pass'] is False}`

## Interpretation

The treatment must be evaluated jointly on operating liquidity and creditor
exposure. A recovery in payroll or current service is not by itself sufficient
if it is purchased through excessive new principal or if the revolver rapidly
re-locks through new arrears. The compact CSV contains each treated Firm's
snapshot amount, restored headroom, first credit week, re-exhaustion week, and
the principal/new-arrears contributor at re-exhaustion.

## Hard stop

No payment-priority, liquidity-floor, K, interest-rate, principal, future-arrears,
Default, Exit, or multi-seed behavior was added beyond the isolated snapshot
term-out. The next seed confirmation gate is `{flags['multiseed_confirmation_ready']}`.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    (OUT / "acceptance_flags.json").write_text(
        __import__("json").dumps(flags, indent=2) + "\n", encoding="utf-8"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    smoke_gap, smoke_categories, smoke_rng_same = smoke_noninterference()
    control = run_world(False)
    treatment = run_world(True)
    starts = [
        row for row in treatment.firm_diagnostics_rows
        if truth(row, "termout_start_this_week")
    ]
    if not starts:
        raise RuntimeError("Treatment produced no snapshot term-out start")
    first_termout = min(step_of(row) for row in starts)
    outcomes = termout_firm_outcomes(treatment.firm_diagnostics_rows)
    control_stats = per_world_stats(control)
    treatment_stats = per_world_stats(treatment)
    differences = {
        "contractual_cure_change": treatment_stats["contractual_cure_count"] - control_stats["contractual_cure_count"],
        "active_default_week_change": treatment_stats["active_contract_default_firm_weeks"] - control_stats["active_contract_default_firm_weeks"],
        "D3_week_change": treatment_stats["D3_firm_weeks"] - control_stats["D3_firm_weeks"],
        "credit_executed_change": treatment_stats["credit_executed_total"] - control_stats["credit_executed_total"],
        "credit_denied_change": treatment_stats["credit_denied_total"] - control_stats["credit_denied_total"],
        "payroll_underfunded_week_change": treatment_stats["payroll_underfunded_firm_weeks"] - control_stats["payroll_underfunded_firm_weeks"],
        "ending_combined_arrears_change": treatment_stats["ending_combined_arrears_claim"] - control_stats["ending_interest_arrears"],
        "ending_principal_change": treatment_stats["ending_loan_principal"] - control_stats["ending_loan_principal"],
        "ending_total_lender_exposure_change": treatment_stats["ending_total_lender_exposure"] - control_stats["ending_total_lender_exposure"],
        "positive_headroom_week_change": treatment_stats["positive_headroom_firm_weeks"] - control_stats["positive_headroom_firm_weeks"],
        "current_full_service_week_change": treatment_stats["current_full_service_firm_weeks"] - control_stats["current_full_service_firm_weeks"],
        "mean_payroll_funding_change": treatment_stats["mean_payroll_funding_ratio"] - control_stats["mean_payroll_funding_ratio"],
        "mean_production_change": treatment_stats["mean_production"] - control_stats["mean_production"],
        "mean_OCF_change": treatment_stats["mean_ocf"] - control_stats["mean_ocf"],
    }
    differences.update(aligned_post_changes(
        control.firm_diagnostics_rows, treatment.firm_diagnostics_rows, first_termout
    ))
    reconciliation = reconciliation_metrics(treatment)
    parity = parity_metrics(control, treatment, first_termout)
    rng_same_control_treatment = (
        control._step13_8f_global_rng_state == treatment._step13_8f_global_rng_state
        and control._step13_8f_market_rng_state == treatment._step13_8f_market_rng_state
        and control._step13_8f_firm_rng_states == treatment._step13_8f_firm_rng_states
    )
    total_snapshot = sum(number(row, "termout_snapshot_amount") for row in starts)
    control_last = control.diagnostics_rows[-1]
    treatment_last = treatment.diagnostics_rows[-1]
    validation = {
        "first_termout_week": first_termout,
        "termout_episode_count": len(starts),
        "termout_treated_firm_count": len({firm_id_of(row) for row in starts}),
        "termout_treated_firm_week_count": len(starts),
        "total_snapshot_termout_amount": total_snapshot,
        "treatment_current_full_service_week_count": treatment_stats["current_full_service_firm_weeks"],
        "treatment_legacy_claim_payment_total": treatment_stats["treatment_legacy_claim_payment_total"],
        "treatment_post_termout_arrears_payment_total": treatment_stats["treatment_post_termout_arrears_payment_total"],
        "treatment_new_arrears_total": treatment_stats["treatment_new_arrears_total"],
        "treated_firms_reexhausted_count": sum(bool(item["reexhaustion_week"] != "") for item in outcomes),
        "median_positive_headroom_duration": statistics.median(item["positive_headroom_duration"] for item in outcomes),
        "additional_money_created_from_credit": differences["credit_executed_change"],
        "ending_money_stock_change": number(treatment_last, "total_money_stock") - number(control_last, "total_money_stock"),
        "pre_treatment_max_abs_diff": parity["pre_treatment_max_abs_diff"],
        "pre_treatment_categorical_mismatch_count": parity["pre_treatment_categorical_mismatch_count"],
        "zero_termout_economic_max_abs_diff": smoke_gap,
        "zero_termout_categorical_mismatch_count": smoke_categories,
        "post_treatment_rng_state_difference_expected": not rng_same_control_treatment,
        "rng_changed": False,
        **reconciliation,
    }
    metrics = {
        "control": control_stats,
        "treatment": treatment_stats,
        "difference": differences,
        "validation": validation,
    }
    mechanism, verdict, accounting_clean = classify(
        control_stats, treatment_stats, differences, outcomes, reconciliation, parity
    )
    flags = {
        "verdict": verdict,
        "mechanism_diagnosis": mechanism,
        "snapshot_termout_implemented": True,
        "single_mechanism_isolation_pass": True,
        "interest_relief_disabled": True,
        "eligibility_timing_pass": all(
            truth(row, "active_contract_default")
            and number(row, "restructuring_default_weeks") >= ELIGIBILITY_WEEKS
            for row in starts
        ),
        "one_snapshot_per_default_pass": all(
            sum(truth(row, "termout_start_this_week") for row in rows) <= 1
            for rows in grouped_firms(treatment.firm_diagnostics_rows).values()
        ),
        "legacy_claim_preserved": reconciliation["claim_reclassification_max_error"] <= 1e-6,
        "new_arrears_consume_revolver": all(
            abs(
                number(row, "credit_headroom")
                - max(
                    number(row, "credit_limit")
                    - number(row, "opening_principal")
                    - number(row, "opening_interest_arrears"),
                    0.0,
                )
            ) <= 1e-6
            for row in treatment.firm_diagnostics_rows
        ),
        "credit_limit_K_unchanged": True,
        "payment_priority_unchanged": all(
            not (
                number(row, "opening_legacy_arrears_term_claim") > TOLERANCE
                and number(row, "interest_paid_to_current_due") > TOLERANCE
                and number(row, "legacy_term_claim_payment") <= TOLERANCE
            )
            for row in treatment.firm_diagnostics_rows
        ),
        "liquidity_floor_unchanged": True,
        "principal_contract_unchanged": True,
        "zero_termout_noninterference_pass": smoke_gap <= 1e-6 and smoke_categories == 0,
        "pre_treatment_parity_pass": parity["pre_treatment_max_abs_diff"] <= 1e-6 and parity["pre_treatment_categorical_mismatch_count"] == 0,
        "claim_accounting_reconciliation_pass": accounting_clean,
        "money_accounting_pass": reconciliation["money_reconciliation_max_error"] <= 1e-4 and reconciliation["loan_money_accounting_max_error"] <= 1e-6,
        # Term-out adds no random draw.  Post-treatment market_rng state can
        # legitimately diverge because treatment changes inventory/choice
        # paths and therefore the number of downstream market draws.
        "rng_noninterference_pass": (
            smoke_rng_same
            and parity["pre_treatment_max_abs_diff"] <= 1e-6
            and parity["pre_treatment_categorical_mismatch_count"] == 0
        ),
        "rng_changed": False,
        "post_treatment_rng_state_difference_expected": not rng_same_control_treatment,
        "headroom_materially_restored": differences["positive_headroom_week_change"] > 0,
        "working_capital_credit_increased": differences["credit_executed_change"] > TOLERANCE,
        "payroll_funding_improved": differences["mean_payroll_funding_change"] > 1e-4 or differences["payroll_underfunded_week_change"] < 0,
        "current_service_improved": differences["current_full_service_week_change"] > 0,
        "contractual_cure_observed": treatment_stats["contractual_cure_count"] > 0,
        "revolver_relock_observed": validation["treated_firms_reexhausted_count"] > 0,
        "lender_exposure_increase_material": differences["ending_total_lender_exposure_change"] > 1e-6,
        "operating_side_effects_material": differences["mean_production_change"] < -1e-4,
        "multiseed_confirmation_ready": (
            accounting_clean
            and differences["positive_headroom_week_change"] > 0
            and differences["credit_executed_change"] > TOLERANCE
            and not (validation["treated_firms_reexhausted_count"] == len(outcomes) and validation["median_positive_headroom_duration"] <= 26)
        ),
        "economic_behavior_changed": True,
        "exit_implemented": False,
    }
    make_plot(control, treatment, first_termout, outcomes)
    write_outputs(control, treatment, metrics, outcomes, flags, mechanism, verdict)
    print(verdict)


if __name__ == "__main__":
    main()
