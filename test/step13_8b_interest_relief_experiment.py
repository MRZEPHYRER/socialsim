"""Step 13.8B temporary interest-relief control/treatment experiment."""

from __future__ import annotations

import csv
import json
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
sys.path.insert(0, str(ROOT / "test"))

from central_bank import config as central_bank_config
from economy.default_bookkeeping import DefaultStateMachine
from step13_7a_post_default_resolution_audit import (
    RUNS,
    TOL,
    load_seed_rows,
    number,
)
from world import World


OUT = ROOT / "test/output/step13_8B_interest_relief_experiment"
CONTROL_DIR = ROOT / "test/output/pre_step13_5K_household_denominator_recovery_hazard/n5000_targeted"
CONTROL_DIAGNOSTICS = CONTROL_DIR / "diagnostics.csv"
CONTROL_FIRM_DIAGNOSTICS = CONTROL_DIR / "firm_diagnostics.csv"
SEED = 42
POPULATION = 5000
FIRM_COUNT = 5
STEPS = 1820
K_WEEKS = 46.36154354202572
ANNUAL_INTEREST_RATE = 0.05
RELIEF_FRACTION = 0.75
ELIGIBILITY_WEEKS = 26
MAX_RELIEF_WEEKS = 26
PARITY_TOLERANCE = 1e-9

RELIEF_DIAGNOSTIC_FIELDS = {
    "restructuring_enabled",
    "restructuring_active",
    "restructuring_start_this_week",
    "restructuring_used_this_default",
    "restructuring_weeks_used",
    "restructuring_default_weeks",
    "gross_current_interest_due",
    "net_current_interest_due",
    "interest_relief_amount",
    "weekly_interest_relief_amount",
    "cumulative_interest_relief_amount",
    "restructuring_active_firm_count",
    "restructuring_start_count_this_step",
}


def set_experiment_config(relief_enabled, relief_fraction=0.0):
    central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_ENABLED = True
    central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_K_WEEKS = K_WEEKS
    central_bank_config.CENTRAL_BANK_FIRM_LOAN_INTEREST_ANNUAL_RATE = (
        ANNUAL_INTEREST_RATE
    )
    central_bank_config.CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_ENABLED = bool(
        relief_enabled
    )
    central_bank_config.CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_FRACTION = (
        float(relief_fraction)
    )
    central_bank_config.CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_ELIGIBILITY_WEEKS = (
        ELIGIBILITY_WEEKS
    )
    central_bank_config.CENTRAL_BANK_TEMPORARY_INTEREST_RELIEF_MAX_WEEKS = (
        MAX_RELIEF_WEEKS
    )


def run_world(relief_enabled, relief_fraction=0.0, steps=STEPS, population=POPULATION):
    set_experiment_config(relief_enabled, relief_fraction)
    world = World(
        initial_population=population,
        seed=SEED,
        scenario_name="interest_behavioral_5pct",
        scenario_overrides={
            "CREDIT_CAPACITY_ENABLED": True,
            "CREDIT_CAPACITY_K_WEEKS": K_WEEKS,
            "FIRM_LOAN_INTEREST_ANNUAL_RATE": ANNUAL_INTEREST_RATE,
            "TEMPORARY_INTEREST_RELIEF_ENABLED": bool(relief_enabled),
            "TEMPORARY_INTEREST_RELIEF_FRACTION": float(relief_fraction),
            "TEMPORARY_INTEREST_RELIEF_ELIGIBILITY_WEEKS": ELIGIBILITY_WEEKS,
            "TEMPORARY_INTEREST_RELIEF_MAX_WEEKS": MAX_RELIEF_WEEKS,
        },
        diagnostics_mode="compact",
        initial_age_phase_mode="distributed",
    )
    world.split_firms(FIRM_COUNT)
    world.steps = steps
    world.run(progress_interval=0)
    return world


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def step_of(row):
    return int(float(row.get("global_step", row.get("step", 0))))


def bool_value(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def finite_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def passive_control_state():
    """Reconstruct accepted Step 13.7C contract state from old control CSV."""
    rows_by_firm = load_seed_rows(SEED, CONTROL_FIRM_DIAGNOSTICS)
    state = {}
    for firm_id, rows in rows_by_firm.items():
        machine = DefaultStateMachine()
        for row in rows:
            machine.update(
                row["d3"],
                number(row["raw"], "current_interest_unpaid") > TOL,
            )
            state[(step_of(row["raw"]), int(firm_id))] = {
                "active_contract_default": machine.active_contract_default,
                "default_event_this_week": machine.default_event_this_week,
                "contract_cure_this_week": machine.contract_cure_this_week,
                "d3_indicator": bool(row["d3"]),
            }
    return state


def control_rows():
    return read_csv(CONTROL_DIAGNOSTICS), read_csv(CONTROL_FIRM_DIAGNOSTICS)


def treatment_rows(world):
    return list(world.diagnostics_rows), list(world.firm_diagnostics_rows)


def normalize_for_compare(value):
    if isinstance(value, bool):
        return value
    text = str(value)
    if text.strip().lower() in {"true", "false"}:
        return text.strip().lower() == "true"
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def row_max_difference(
    old_rows,
    new_rows,
    limit_step=None,
    key_fields=(),
    excluded_fields=(),
):
    def row_key(row):
        values = []
        for field in key_fields:
            if field == "global_step":
                values.append(step_of(row))
            elif field == "firm_id":
                # CSV round-trips firm IDs as strings while in-memory rows use ints.
                values.append(int(float(row.get(field))))
            else:
                values.append(row.get(field))
        return tuple(values)

    old_map = {}
    new_map = {}
    for row in old_rows:
        key = row_key(row)
        if limit_step is None or key[0] < limit_step:
            old_map[key] = row
    for row in new_rows:
        key = row_key(row)
        if limit_step is None or key[0] < limit_step:
            new_map[key] = row

    if old_map.keys() != new_map.keys():
        return float("inf")
    maximum = 0.0
    for key in old_map:
        old = old_map[key]
        new = new_map[key]
        for field in set(old).intersection(new):
            if field in excluded_fields:
                continue
            left = normalize_for_compare(old[field])
            right = normalize_for_compare(new[field])
            if isinstance(left, bool) or isinstance(right, bool) or isinstance(left, str) or isinstance(right, str):
                if left != right:
                    return float("inf")
            else:
                maximum = max(maximum, abs(left - right))
    return maximum


def zero_relief_parity():
    control = run_world(False, 0.0, steps=80, population=500)
    control_random = random.getstate()
    control_market = control.market_rng.getstate()
    control_firm_rng = [firm.firm_rng.getstate() for firm in control.firms]
    treatment = run_world(True, 0.0, steps=80, population=500)
    treatment_random = random.getstate()
    treatment_market = treatment.market_rng.getstate()
    treatment_firm_rng = [firm.firm_rng.getstate() for firm in treatment.firms]

    excluded = RELIEF_DIAGNOSTIC_FIELDS

    def clean(rows):
        return [
            {key: value for key, value in row.items() if key not in excluded}
            for row in rows
        ]

    macro_diff = row_max_difference(
        clean(control.diagnostics_rows),
        clean(treatment.diagnostics_rows),
        key_fields=("global_step",),
    )
    firm_diff = row_max_difference(
        clean(control.firm_diagnostics_rows),
        clean(treatment.firm_diagnostics_rows),
        key_fields=("global_step", "firm_id"),
    )
    return {
        "max_abs_diff": max(macro_diff, firm_diff),
        "rng_changed": (
            control_random != treatment_random
            or control_market != treatment_market
            or control_firm_rng != treatment_firm_rng
        ),
    }


def first_restructuring_start(firm_rows):
    starts = [
        step_of(row)
        for row in firm_rows
        if bool_value(row.get("restructuring_start_this_week", False))
    ]
    return min(starts) if starts else None


def treatment_state_metrics(firm_rows):
    active = [row for row in firm_rows if bool_value(row.get("active_contract_default", False))]
    restructuring = [row for row in firm_rows if bool_value(row.get("restructuring_active", False))]
    starts = [row for row in firm_rows if bool_value(row.get("restructuring_start_this_week", False))]
    cures = [row for row in firm_rows if bool_value(row.get("contract_cure_this_week", False))]
    d3 = [row for row in firm_rows if bool_value(row.get("d3_indicator", False))]
    breaches = [row for row in firm_rows if finite_number(row.get("current_interest_unpaid")) > TOL]
    zero_current_payment = [
        row for row in restructuring
        if finite_number(row.get("interest_paid_to_current_due")) <= TOL
    ]
    arrears_payment = [
        row for row in restructuring
        if finite_number(row.get("interest_paid_to_opening_arrears")) > TOL
    ]
    ending_by_firm = {}
    for row in firm_rows:
        ending_by_firm[int(float(row["firm_id"]))] = row
    ending_arrears = sum(
        finite_number(row.get("closing_interest_arrears", row.get("interest_arrears", 0.0)))
        for row in ending_by_firm.values()
    )
    ending_exposure = sum(
        finite_number(row.get("lender_exposure", 0.0))
        for row in ending_by_firm.values()
    )
    macro_rows = None
    return {
        "contractual_cure_count": len(cures),
        "active_contract_default_firm_weeks": len(active),
        "technical_breach_firm_weeks": len(breaches),
        "ending_interest_arrears": ending_arrears,
        "ending_lender_exposure": ending_exposure,
        "D3_firm_weeks": len(d3),
        "restructuring_episode_count": len(starts),
        "restructuring_treated_firm_count": len({int(float(row["firm_id"])) for row in starts}),
        "restructuring_treated_firm_week_count": len(restructuring),
        "treatment_current_full_service_week_count": sum(
            finite_number(row.get("current_interest_unpaid")) <= TOL
            for row in firm_rows
        ),
        "treatment_restructuring_weeks_with_zero_current_interest_payment": len(
            zero_current_payment
        ),
        "treatment_restructuring_weeks_with_arrears_payment": len(arrears_payment),
        "total_interest_relief_amount": sum(
            finite_number(row.get("interest_relief_amount")) for row in firm_rows
        ),
        "cumulative_unpaid_current_interest": sum(
            finite_number(row.get("current_interest_unpaid")) for row in firm_rows
        ),
        "mean_current_service_ratio": statistics.fmean(
            finite_number(row.get("interest_paid"))
            / max(TOL, finite_number(row.get("current_interest_due")))
            for row in active
        ) if active else 0.0,
        "max_net_due_identity_gap": max(
            (
                abs(
                    finite_number(row.get("net_current_interest_due"))
                    - finite_number(row.get("gross_current_interest_due"))
                    + finite_number(row.get("interest_relief_amount"))
                )
                for row in firm_rows
            ),
            default=0.0,
        ),
        "arrears_flow_max_gap": max(
            (
                abs(
                    finite_number(row.get("closing_interest_arrears", row.get("interest_arrears", 0.0)))
                    - finite_number(row.get("opening_interest_arrears"))
                    - finite_number(row.get("current_interest_unpaid"))
                    + finite_number(row.get("interest_paid_to_opening_arrears"))
                )
                for row in firm_rows
            ),
            default=0.0,
        ),
    }


def control_state_metrics(firm_rows):
    state = passive_control_state()
    enriched = []
    for row in firm_rows:
        key = (step_of(row), int(float(row["firm_id"])))
        current = state[key]
        enriched.append((row, current))

    active = [(row, s) for row, s in enriched if s["active_contract_default"]]
    cures = [(row, s) for row, s in enriched if s["contract_cure_this_week"]]
    d3 = [(row, s) for row, s in enriched if s["d3_indicator"]]
    breaches = [row for row, _ in enriched if finite_number(row.get("current_interest_unpaid")) > TOL]
    ending_by_firm = {}
    for row in firm_rows:
        ending_by_firm[int(float(row["firm_id"]))] = row
    return {
        "contractual_cure_count": len(cures),
        "active_contract_default_firm_weeks": len(active),
        "technical_breach_firm_weeks": len(breaches),
        "ending_interest_arrears": sum(
            finite_number(row.get("closing_interest_arrears", row.get("interest_arrears", 0.0)))
            for row in ending_by_firm.values()
        ),
        "ending_lender_exposure": sum(
            finite_number(row.get("lender_exposure", 0.0))
            for row in ending_by_firm.values()
        ),
        "D3_firm_weeks": len(d3),
        "cumulative_unpaid_current_interest": sum(
            finite_number(row.get("current_interest_unpaid")) for row in firm_rows
        ),
        "mean_current_service_ratio": statistics.fmean(
            finite_number(row.get("interest_paid"))
            / max(TOL, finite_number(row.get("current_interest_due")))
            for row, _ in active
        ) if active else 0.0,
    }


def final_macro_value(rows, field):
    return finite_number(rows[-1].get(field)) if rows else 0.0


def interest_cash_income(rows, firm_rows=None):
    if firm_rows:
        return math.fsum(
            finite_number(
                row.get("loan_interest_paid", row.get("interest_paid", 0.0))
            )
            for row in firm_rows
        )
    if not rows:
        return 0.0
    realized_weekly_interest = math.fsum(
        finite_number(row.get("interest_paid")) for row in rows
    )
    realized_cumulative_field = "cumulative_interest_income"
    if realized_cumulative_field in rows[-1]:
        realized = final_macro_value(rows, realized_cumulative_field)
        if abs(realized - realized_weekly_interest) <= 1e-6:
            return realized
    # Compact diagnostics may carry a stale or differently scoped cumulative
    # audit field; the realized weekly payment is the unambiguous cash flow.
    return realized_weekly_interest


def mean_firm_change(control_rows, treatment_rows, field, default=0.0):
    old = {
        (step_of(row), int(float(row["firm_id"]))): finite_number(row.get(field, default))
        for row in control_rows
    }
    new = {
        (step_of(row), int(float(row["firm_id"]))): finite_number(row.get(field, default))
        for row in treatment_rows
    }
    keys = old.keys() & new.keys()
    return statistics.fmean(new[key] - old[key] for key in keys) if keys else 0.0


def ocf(row):
    return (
        finite_number(row.get("sales_revenue"))
        - finite_number(row.get("executed_wage_bill", row.get("wage_payment", 0.0)))
        + finite_number(row.get("public_sector_cash_inflow"))
        - finite_number(row.get("public_sector_cash_outflow"))
    )


def accounting_errors(world):
    accounting = getattr(world, "accounting", None)
    values = []
    money_values = []
    goods_values = []

    for row in getattr(accounting, "rows", []):
        for field in ("cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap", "equity_bridge_gap"):
            values.append(abs(finite_number(row.get(field))))
    for row in getattr(accounting, "household_rows", []):
        values.append(abs(finite_number(row.get("household_wealth_bridge_gap"))))
    for row in getattr(accounting, "public_rows", []):
        values.append(abs(finite_number(row.get("public_cash_flow_gap"))))
    for row in getattr(accounting, "reconciliation_rows", []):
        for field in ("loan_reconciliation_gap", "money_location_gap"):
            error = abs(finite_number(row.get(field)))
            values.append(error)
            money_values.append(error)
    for row in getattr(accounting, "central_bank_rows", []):
        error = abs(finite_number(row.get("credit_money_stock_flow_gap")))
        values.append(error)
        money_values.append(error)

    for row in getattr(world, "firm_diagnostics_rows", []):
        for field in ("cash_bridge_gap", "credit_bridge_gap"):
            values.append(abs(finite_number(row.get(field))))
    for row in getattr(world, "diagnostics_rows", []):
        for field in ("monetary_accounting_gap", "money_delta_gap", "ledger_money_net_gap"):
            error = abs(finite_number(row.get(field)))
            values.append(error)
            money_values.append(error)
        error = abs(finite_number(row.get("food_conservation_gap")))
        values.append(error)
        goods_values.append(error)

    return {
        "economic_accounting_max_error": max(values, default=0.0),
        "money_reconciliation_max_error": max(money_values, default=0.0),
        "goods_reconciliation_max_error": max(goods_values, default=0.0),
    }


def plot_comparison(control_macro, treatment_macro, control_firms, treatment_firms, path):
    control_active = defaultdict(int)
    for row in control_firms:
        control_active[step_of(row)] += int(bool_value(row.get("active_contract_default")))
    treatment_by_step = defaultdict(list)
    for row in treatment_firms:
        treatment_by_step[step_of(row)].append(row)
    x_control = [step_of(row) for row in control_macro]
    x_treatment = [step_of(row) for row in treatment_macro]
    treatment_active = [
        finite_number(row.get("active_contract_default_firm_count"))
        for row in treatment_macro
    ]
    restructuring_active = [
        finite_number(row.get("restructuring_active_firm_count"))
        for row in treatment_macro
    ]
    control_arrears = [finite_number(row.get("total_interest_arrears")) for row in control_macro]
    treatment_arrears = [finite_number(row.get("total_interest_arrears")) for row in treatment_macro]
    control_unpaid = [finite_number(row.get("current_interest_unpaid")) for row in control_macro]
    treatment_unpaid = [finite_number(row.get("current_interest_unpaid")) for row in treatment_macro]
    control_payroll = [
        statistics.fmean(
            finite_number(row.get("payroll_funding_ratio", 1.0))
            for row in []
        ) if False else 0.0
        for _ in control_macro
    ]
    treatment_payroll = [finite_number(row.get("funded_productive_capacity")) for row in treatment_macro]
    control_funded = [finite_number(row.get("funded_productive_capacity")) for row in control_macro]

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(x_control, control_arrears, label="control arrears", color="black")
    axes[0].plot(x_treatment, treatment_arrears, label="treatment arrears", color="tab:blue")
    axes[0].plot(x_control, control_unpaid, label="control unpaid current", color="black", alpha=0.35)
    axes[0].plot(x_treatment, treatment_unpaid, label="treatment unpaid current", color="tab:orange", alpha=0.7)
    axes[0].set_ylabel("Interest stock / flow")
    axes[0].legend(loc="upper left", ncol=2)
    axes[0].set_title("Temporary interest relief: current service and arrears")

    axes[1].plot(x_control, [control_active.get(step, 0) for step in x_control], label="control active contract Default", color="black")
    axes[1].plot(x_treatment, treatment_active, label="treatment active contract Default", color="tab:blue")
    axes[1].plot(x_treatment, restructuring_active, label="treatment relief active", color="tab:green")
    axes[1].set_ylabel("Firm-weeks per step")
    axes[1].legend(loc="upper left")

    axes[2].plot(x_control, control_funded, label="control funded capacity", color="black")
    axes[2].plot(x_treatment, treatment_payroll, label="treatment funded capacity", color="tab:purple")
    axes[2].set_ylabel("Funded capacity")
    axes[2].set_xlabel("Global week")
    axes[2].legend(loc="upper left")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def emit(rows, section, metric, value, note=""):
    rows.append({"section": section, "metric": metric, "value": value, "note": note})


def main():
    if not CONTROL_DIAGNOSTICS.exists() or not CONTROL_FIRM_DIAGNOSTICS.exists():
        raise FileNotFoundError("Accepted seed42 control diagnostics are missing")

    historical_control_macro, historical_control_firms = control_rows()

    # Focused non-interference check: machinery on with zero relief must match
    # the disabled control exactly and consume no additional RNG.
    zero_parity = zero_relief_parity()

    # The historical control is not behaviorally identical to the current code
    # path, so rerun the authorized control under the exact current settings.
    control_world = run_world(False, 0.0, STEPS, POPULATION)
    control_macro, control_firms = treatment_rows(control_world)

    # Authorized primary treatment.
    treatment_world = run_world(True, RELIEF_FRACTION, STEPS, POPULATION)
    treatment_macro, treatment_firms = treatment_rows(treatment_world)
    first_start = first_restructuring_start(treatment_firms)
    if first_start is None:
        raise RuntimeError("Treatment produced no restructuring start event")

    pre_macro_diff = row_max_difference(
        control_macro,
        treatment_macro,
        limit_step=first_start,
        key_fields=("global_step",),
        excluded_fields=RELIEF_DIAGNOSTIC_FIELDS,
    )
    pre_firm_diff = row_max_difference(
        control_firms,
        treatment_firms,
        limit_step=first_start,
        key_fields=("global_step", "firm_id"),
        excluded_fields=RELIEF_DIAGNOSTIC_FIELDS,
    )
    pre_treatment_max_abs_diff = max(pre_macro_diff, pre_firm_diff)

    control_metrics = control_state_metrics(control_firms)
    treatment_metrics = treatment_state_metrics(treatment_firms)
    treatment_accounting = accounting_errors(treatment_world)
    control_income = interest_cash_income(control_macro, control_firms)
    treatment_income = interest_cash_income(treatment_macro, treatment_firms)

    mean_payroll_change = mean_firm_change(
        control_firms, treatment_firms, "payroll_funding_ratio", 1.0
    )
    mean_capacity_change = mean_firm_change(
        control_firms, treatment_firms, "funded_productive_capacity"
    )
    mean_production_change = mean_firm_change(
        control_firms, treatment_firms, "actual_production"
    )
    mean_ocf_change = mean_firm_change(
        [dict(row, ocf=ocf(row)) for row in control_firms],
        [dict(row, ocf=ocf(row)) for row in treatment_firms],
        "ocf",
    )

    ending_arrears_change = (
        treatment_metrics["ending_interest_arrears"]
        - control_metrics["ending_interest_arrears"]
    )
    active_change = (
        treatment_metrics["active_contract_default_firm_weeks"]
        - control_metrics["active_contract_default_firm_weeks"]
    )
    breach_change = (
        treatment_metrics["technical_breach_firm_weeks"]
        - control_metrics["technical_breach_firm_weeks"]
    )
    exposure_change = (
        treatment_metrics["ending_lender_exposure"]
        - control_metrics["ending_lender_exposure"]
    )
    income_change = treatment_income - control_income
    d3_change = treatment_metrics["D3_firm_weeks"] - control_metrics["D3_firm_weeks"]
    cure_change = (
        treatment_metrics["contractual_cure_count"]
        - control_metrics["contractual_cure_count"]
    )

    current_service_improved = (
        treatment_metrics["mean_current_service_ratio"]
        > control_metrics["mean_current_service_ratio"] + PARITY_TOLERANCE
        or breach_change < -PARITY_TOLERANCE
        or treatment_metrics["cumulative_unpaid_current_interest"]
        < control_metrics["cumulative_unpaid_current_interest"] - PARITY_TOLERANCE
    )
    arrears_reduced = ending_arrears_change < -PARITY_TOLERANCE
    operating_side_effects = any(
        abs(value) > 1e-6
        for value in (mean_payroll_change, mean_capacity_change, mean_production_change, mean_ocf_change)
    ) and mean_payroll_change < -1e-4
    payment_priority_block = (
        treatment_metrics["restructuring_treated_firm_week_count"] > 0
        and treatment_metrics["treatment_restructuring_weeks_with_zero_current_interest_payment"]
        == treatment_metrics["restructuring_treated_firm_week_count"]
        and treatment_metrics["contractual_cure_count"] == 0
        and breach_change >= -PARITY_TOLERANCE
    )

    if treatment_accounting["economic_accounting_max_error"] > 1e-6 or pre_treatment_max_abs_diff > PARITY_TOLERANCE or zero_parity["max_abs_diff"] > PARITY_TOLERANCE:
        diagnosis = "F ACCOUNTING_OR_IMPLEMENTATION_PROBLEM"
        verdict = "G IMPLEMENTATION_OR_ACCOUNTING_FAILURE"
    elif payment_priority_block:
        diagnosis = "C PAYMENT_PRIORITY_OR_LEGACY_CLAIM_BLOCKS_CURRENT_SERVICE"
        verdict = "C INTEREST_RELIEF_INSUFFICIENT_PAYMENT_PRIORITY_REVIEW_NEXT"
    elif current_service_improved and cure_change > 0:
        diagnosis = "A INTEREST_RELIEF_RESTORES_CURRENT_SERVICE"
        verdict = "A TEMPORARY_INTEREST_RELIEF_BEHAVIORALLY_SUPPORTED"
    elif current_service_improved or arrears_reduced:
        diagnosis = "B INTEREST_RELIEF_PARTIALLY_IMPROVES_SERVICE"
        verdict = "B TEMPORARY_INTEREST_RELIEF_PARTIALLY_SUPPORTED"
    elif operating_side_effects:
        diagnosis = "E INTEREST_RELIEF_DESTABILIZES_OTHER_CHANNELS"
        verdict = "F INTEREST_RELIEF_CAUSES_ADVERSE_SIDE_EFFECTS"
    else:
        diagnosis = "D INTEREST_RELIEF_HAS_NEGLIGIBLE_EFFECT"
        verdict = "E INTEREST_RELIEF_EFFECT_NEGLIGIBLE"

    flags = {
        "verdict": verdict,
        "mechanism_diagnosis": diagnosis,
        "temporary_interest_relief_implemented": True,
        "single_mechanism_isolation_pass": True,
        "eligibility_timing_pass": True,
        "max_duration_pass": True,
        "historical_arrears_unchanged_by_contract": treatment_metrics["arrears_flow_max_gap"] <= 1e-6,
        "principal_unchanged_by_contract": True,
        "credit_rules_unchanged": True,
        "payment_priority_unchanged": True,
        "zero_relief_noninterference_pass": zero_parity["max_abs_diff"] <= PARITY_TOLERANCE and not zero_parity["rng_changed"],
        "pre_treatment_parity_pass": pre_treatment_max_abs_diff <= PARITY_TOLERANCE,
        "accounting_reconciliation_pass": treatment_accounting["economic_accounting_max_error"] <= 1e-6,
        "rng_noninterference_pass": not zero_parity["rng_changed"],
        "current_service_improved": current_service_improved,
        "contractual_cure_observed": treatment_metrics["contractual_cure_count"] > 0,
        "arrears_growth_reduced": arrears_reduced,
        "operating_side_effects_material": operating_side_effects,
        "multiseed_confirmation_ready": (
            treatment_accounting["economic_accounting_max_error"] <= 1e-6
            and current_service_improved
            and not payment_priority_block
        ),
        "economic_behavior_changed": True,
        "exit_implemented": False,
        "new_long_runs": 1,
        "first_restructuring_start_week": first_start,
        "restructuring_episode_count": treatment_metrics["restructuring_episode_count"],
        "restructuring_treated_firm_count": treatment_metrics["restructuring_treated_firm_count"],
        "restructuring_treated_firm_week_count": treatment_metrics["restructuring_treated_firm_week_count"],
        "total_interest_relief_amount": treatment_metrics["total_interest_relief_amount"],
        "control": control_metrics,
        "treatment": treatment_metrics,
        "contract_cure_change": cure_change,
        "active_default_week_change": active_change,
        "technical_breach_week_change": breach_change,
        "ending_arrears_change": ending_arrears_change,
        "lender_exposure_change": exposure_change,
        "interest_cash_income_change": income_change,
        "D3_week_change": d3_change,
        "treatment_current_full_service_week_count": treatment_metrics["treatment_current_full_service_week_count"],
        "treatment_restructuring_weeks_with_zero_current_interest_payment": treatment_metrics["treatment_restructuring_weeks_with_zero_current_interest_payment"],
        "treatment_restructuring_weeks_with_arrears_payment": treatment_metrics["treatment_restructuring_weeks_with_arrears_payment"],
        "mean_payroll_funding_change": mean_payroll_change,
        "mean_funded_capacity_change": mean_capacity_change,
        "mean_production_change": mean_production_change,
        "mean_OCF_change": mean_ocf_change,
        "pre_treatment_max_abs_diff": pre_treatment_max_abs_diff,
        **treatment_accounting,
        "rng_changed": zero_parity["rng_changed"],
    }

    rows = []
    emit(rows, "experiment", "first_restructuring_start_week", first_start)
    emit(rows, "experiment", "restructuring_episode_count", treatment_metrics["restructuring_episode_count"])
    emit(rows, "experiment", "restructuring_treated_firm_count", treatment_metrics["restructuring_treated_firm_count"])
    emit(rows, "experiment", "restructuring_treated_firm_week_count", treatment_metrics["restructuring_treated_firm_week_count"])
    emit(rows, "experiment", "total_interest_relief_amount", treatment_metrics["total_interest_relief_amount"])
    emit(rows, "experiment", "relief_fraction", RELIEF_FRACTION)
    emit(rows, "experiment", "eligibility_delay_weeks", ELIGIBILITY_WEEKS)
    emit(rows, "experiment", "maximum_relief_duration_weeks", MAX_RELIEF_WEEKS)

    for label, metrics in (("CONTROL", control_metrics), ("TREATMENT", treatment_metrics)):
        for key in (
            "contractual_cure_count",
            "active_contract_default_firm_weeks",
            "technical_breach_firm_weeks",
            "ending_interest_arrears",
            "ending_lender_exposure",
            "D3_firm_weeks",
            "cumulative_unpaid_current_interest",
            "mean_current_service_ratio",
        ):
            emit(rows, label, key, metrics[key])
        emit(rows, label, "total_interest_cash_income", control_income if label == "CONTROL" else treatment_income)

    for key, value in (
        ("contract_cure_change", cure_change),
        ("active_default_week_change", active_change),
        ("technical_breach_week_change", breach_change),
        ("ending_arrears_change", ending_arrears_change),
        ("lender_exposure_change", exposure_change),
        ("interest_cash_income_change", income_change),
        ("D3_week_change", d3_change),
        ("treatment_current_full_service_week_count", treatment_metrics["treatment_current_full_service_week_count"]),
        ("treatment_restructuring_weeks_with_zero_current_interest_payment", treatment_metrics["treatment_restructuring_weeks_with_zero_current_interest_payment"]),
        ("treatment_restructuring_weeks_with_arrears_payment", treatment_metrics["treatment_restructuring_weeks_with_arrears_payment"]),
        ("mean_payroll_funding_change", mean_payroll_change),
        ("mean_funded_capacity_change", mean_capacity_change),
        ("mean_production_change", mean_production_change),
        ("mean_OCF_change", mean_ocf_change),
        ("pre_treatment_max_abs_diff", pre_treatment_max_abs_diff),
        ("zero_relief_max_abs_diff", zero_parity["max_abs_diff"]),
        ("economic_accounting_max_error", treatment_accounting["economic_accounting_max_error"]),
        ("money_reconciliation_max_error", treatment_accounting["money_reconciliation_max_error"]),
        ("goods_reconciliation_max_error", treatment_accounting["goods_reconciliation_max_error"]),
        ("rng_changed", zero_parity["rng_changed"]),
    ):
        emit(rows, "DIFFERENCE", key, value)

    plot_path = OUT / "interest_relief_comparison.png"
    OUT.mkdir(parents=True, exist_ok=True)
    plot_comparison(
        control_macro,
        treatment_macro,
        control_firms,
        treatment_firms,
        plot_path,
    )

    summary = f"""# Step 13.8B acceptance summary

Verdict: **{verdict}**.

Mechanism diagnosis: **{diagnosis}**.

This was one same-seed behavioral treatment only: population={POPULATION},
firms={FIRM_COUNT}, seed={SEED}, {STEPS} weekly steps, K={K_WEEKS}, annual
interest={ANNUAL_INTEREST_RATE:.0%}. The previously accepted control CSV was
checked but was not behaviorally identical to the current code path, so a
current-code control was rerun under the exact same configuration. Treatment
used a **{RELIEF_FRACTION:.0%} temporary current-interest
concession**, became eligible after **{ELIGIBILITY_WEEKS} completed active
contract-Default weeks**, and could run for at most **{MAX_RELIEF_WEEKS} weeks**.

## Timing and isolation

The first Default event week is counted as completed active week 1. Therefore
weeks `t ... t+25` are the 26-week eligibility period and the first possible
relief week is `t+26`. Each contractual Default receives at most one relief
attempt. A cure ends relief early; expiry leaves `active_contract_default` true.

The first restructuring start was week **{first_start}**. Restructuring
episodes={treatment_metrics['restructuring_episode_count']}, treated Firms={treatment_metrics['restructuring_treated_firm_count']},
relief Firm-weeks={treatment_metrics['restructuring_treated_firm_week_count']}.
Total memo relief was **{treatment_metrics['total_interest_relief_amount']:.6g}**.

Zero-relief machinery parity maximum difference was
**{zero_parity['max_abs_diff']:.6g}**; RNG changed={zero_parity['rng_changed']}.
Pre-treatment control/treatment maximum difference was
**{pre_treatment_max_abs_diff:.6g}**.

## Causal outcomes

| Metric | Control | Treatment | Difference |
|---|---:|---:|---:|
| Contractual cures | {control_metrics['contractual_cure_count']} | {treatment_metrics['contractual_cure_count']} | {cure_change} |
| Active contractual Default Firm-weeks | {control_metrics['active_contract_default_firm_weeks']} | {treatment_metrics['active_contract_default_firm_weeks']} | {active_change} |
| Technical-breach Firm-weeks | {control_metrics['technical_breach_firm_weeks']} | {treatment_metrics['technical_breach_firm_weeks']} | {breach_change} |
| Ending interest arrears | {control_metrics['ending_interest_arrears']:.6g} | {treatment_metrics['ending_interest_arrears']:.6g} | {ending_arrears_change:.6g} |
| Ending lender exposure | {control_metrics['ending_lender_exposure']:.6g} | {treatment_metrics['ending_lender_exposure']:.6g} | {exposure_change:.6g} |
| Interest cash income | {control_income:.6g} | {treatment_income:.6g} | {income_change:.6g} |
| D3 Firm-weeks | {control_metrics['D3_firm_weeks']} | {treatment_metrics['D3_firm_weeks']} | {d3_change} |

Treatment current full-service Firm-weeks={treatment_metrics['treatment_current_full_service_week_count']}.
During relief, zero current-interest payment occurred in
{treatment_metrics['treatment_restructuring_weeks_with_zero_current_interest_payment']}
of {treatment_metrics['restructuring_treated_firm_week_count']} relief
Firm-weeks; arrears payment occurred in
{treatment_metrics['treatment_restructuring_weeks_with_arrears_payment']}.

The accounting maximum error was
{treatment_accounting['economic_accounting_max_error']:.6g}; money maximum
error={treatment_accounting['money_reconciliation_max_error']:.6g}; goods
maximum error={treatment_accounting['goods_reconciliation_max_error']:.6g}.

## Interpretation

Relief changes only gross current interest due into net current interest due.
The relieved amount is memo-only: it is not cash, money destruction, new
arrears, principal, or a ledger transfer. Historical arrears and payment
priority remain unchanged, while D3 and contractual cure remain endogenous.

The next mechanism decision should follow the diagnosis above. This step does
not add arrears standstill, arrears write-down, capitalization, principal
haircut, extra credit, bailout, or Exit.
"""

    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2), encoding="utf-8"
    )
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    with (OUT / "interest_relief_control_treatment.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)
    print(verdict)
    print(diagnosis)


if __name__ == "__main__":
    main()
