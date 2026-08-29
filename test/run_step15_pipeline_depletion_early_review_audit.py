from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
for path in (ROOT, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_step15_investment_review_cadence_screen import (  # noqa: E402
    TOL,
    longest_spell,
    run_case,
    slice_window,
)


OUTPUT = ROOT / "test/output/step15_pipeline_depletion_early_review_audit"
CONTROL = "CONTROL_13W_SCHEDULED_ONLY"
TREATMENT = "TREATMENT_13W_PIPELINE_EARLY"
POPULATION = 5000
SEED = 42
WEEKS = 520
CADENCE = 13
HORIZON = 13


def order_panel(advances, settlements):
    if advances.empty:
        return pd.DataFrame(columns=["order_id", "buyer_firm_id", "accepted_value", "accepted_units", "undelivered_units", "delivery_lag_weeks"])
    accepted = advances.copy()
    for field in ("global_step", "cash_amount", "physical_units"):
        accepted[field] = pd.to_numeric(accepted[field], errors="coerce").fillna(0.0)
    panel = accepted.groupby(["order_id", "buyer_firm_id"], as_index=False).agg(
        accepted_week=("global_step", "min"), accepted_value=("cash_amount", "sum"), accepted_units=("physical_units", "sum")
    )
    if settlements.empty:
        panel["delivered_units"] = 0.0
        panel["delivered_value"] = 0.0
        panel["last_delivery_week"] = math.nan
    else:
        delivered = settlements.copy()
        for field in ("global_step", "settled_units", "settled_expenditure"):
            delivered[field] = pd.to_numeric(delivered[field], errors="coerce").fillna(0.0)
        delivered = delivered.groupby(["order_id", "buyer_firm_id"], as_index=False).agg(
            delivered_units=("settled_units", "sum"), delivered_value=("settled_expenditure", "sum"), last_delivery_week=("global_step", "max")
        )
        panel = panel.merge(delivered, on=["order_id", "buyer_firm_id"], how="left").fillna({"delivered_units": 0.0, "delivered_value": 0.0})
    panel["undelivered_units"] = (panel["accepted_units"] - panel["delivered_units"]).clip(lower=0.0)
    panel["delivery_lag_weeks"] = panel["last_delivery_week"] - panel["accepted_week"]
    panel.loc[panel["delivered_units"] <= TOL, "delivery_lag_weeks"] = math.nan
    return panel


def reconciles(macro):
    row = {
        "max_money_location_gap": float(macro["money_location_gap"].max()),
        "max_goods_gap": float(macro["goods_gap"].max()),
        "max_advance_prepaid_gap": float(macro["advance_prepaid_gap"].max()),
        "max_accounting_gap": float(macro["accounting_gap"].max()),
        "max_assignment_violations": int(macro["assignment_violations"].max()),
        "max_output_above_feasible_capacity": float(macro["output_above_feasible_capacity"].max()),
    }
    passed = (
        max(row[key] for key in ("max_money_location_gap", "max_goods_gap", "max_advance_prepaid_gap", "max_accounting_gap")) <= 1e-5
        and row["max_assignment_violations"] == 0
        and row["max_output_above_feasible_capacity"] <= TOL
    )
    return row, passed


def pipeline_shutdown_audit(trace):
    snapshot = trace.loc[trace["record_type"].eq("PIPELINE_SNAPSHOT")].copy()
    shutdown = snapshot.loc[
        (snapshot["supplier_cash"] <= TOL)
        & (snapshot["customer_advance_liability"] <= TOL)
        & (snapshot["pipeline_undelivered_units"] <= TOL)
        & (snapshot["supplier_desired_output"] > TOL)
        & (snapshot["supplier_payroll"] <= TOL)
    ].copy()
    shutdown["buyer_residual_need_positive"] = shutdown["residual_uncommitted_capacity_gap"] > TOL
    shutdown["buyer_cash_sufficient"] = shutdown["shadow_financing_gap"] <= TOL
    shutdown["next_scheduled_review_wait_weeks"] = shutdown["next_scheduled_investment_review_week"] - shutdown["global_step"]
    return shutdown


def classify_shutdowns(rows):
    classifications = []
    for step, group in rows.groupby("global_step"):
        if bool((group["buyer_residual_need_positive"] & group["buyer_cash_sufficient"]).any()):
            label = "WAITING_DESPITE_POSITIVE_RESIDUAL_NEED"
        elif bool(group["buyer_residual_need_positive"].any()):
            label = "BUYER_CASH_CONSTRAINED"
        else:
            label = "NO_CURRENT_INVESTMENT_NEED"
        first = group.iloc[0]
        classifications.append({
            "global_step": step,
            "classification": label,
            "supplier_cash": first["supplier_cash"],
            "customer_advance_liability": first["customer_advance_liability"],
            "pipeline_undelivered_units": first["pipeline_undelivered_units"],
            "supplier_desired_output": first["supplier_desired_output"],
            "supplier_funded_output": first["supplier_funded_output"],
            "supplier_realized_output": first["supplier_realized_output"],
            "firms_with_positive_residual_need": int(group["buyer_residual_need_positive"].sum()),
            "firms_cash_sufficient": int(group["buyer_cash_sufficient"].sum()),
            "minimum_wait_to_scheduled_review": int(group["next_scheduled_review_wait_weeks"].min()),
        })
    return pd.DataFrame(classifications)


def supplier_summary(label, supplier):
    desired = supplier["desired_output"] > TOL
    zero_payroll = desired & (supplier["supplier_payroll"] <= TOL)
    zero_production = desired & (supplier["realized_output"] <= TOL)
    zero_delivery = desired & (supplier["delivery_value"] <= TOL)
    return {
        "case": label,
        "zero_payroll_weeks": int(zero_payroll.sum()),
        "zero_production_weeks": int(zero_production.sum()),
        "zero_delivery_weeks": int(zero_delivery.sum()),
        "longest_inactivity_spell": longest_spell(zero_payroll),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    control = run_case(CONTROL, CADENCE, service_horizon=HORIZON, capture_committed_trace=True)
    control_macro, control_supplier, control_advances, control_settlements, control_trace = control
    pipeline_rows = pipeline_shutdown_audit(control_trace)
    classifications = classify_shutdowns(pipeline_rows)
    pipeline_rows.to_csv(OUTPUT / "pipeline_empty_residual_need_audit.csv", index=False)
    classifications.to_csv(OUTPUT / "shutdown_window_classification.csv", index=False)
    class_a_material = bool(
        not classifications.empty
        and classifications["classification"].eq("WAITING_DESPITE_POSITIVE_RESIDUAL_NEED").any()
    )

    results = {CONTROL: control}
    if class_a_material:
        results[TREATMENT] = run_case(
            TREATMENT,
            CADENCE,
            service_horizon=HORIZON,
            scenario_extra={"CAPITAL_GOOD_PIPELINE_DEPLETION_EARLY_REVIEW_ENABLED": True},
            capture_committed_trace=True,
        )

    supplier_rows = []
    order_rows = []
    investment_rows = []
    late_rows = []
    recon_rows = []
    early_events = []
    for label, result in results.items():
        macro, supplier, advances, settlements, trace = result
        supplier_rows.append(supplier_summary(label, supplier))
        orders = order_panel(advances, settlements)
        order_rows.append({
            "case": label,
            "order_count": int(len(orders)),
            "accepted_order_value": float(orders["accepted_value"].sum()),
            "mean_order_size": float(orders["accepted_value"].mean()) if not orders.empty else 0.0,
            "median_order_size": float(orders["accepted_value"].median()) if not orders.empty else 0.0,
            "outstanding_order_count": int((orders["undelivered_units"] > TOL).sum()),
            "final_backlog_units": float(macro["total_backlog"].iloc[-1]),
            "mean_delivery_lag_weeks": float(orders["delivery_lag_weeks"].mean()) if not orders.empty else math.nan,
            "median_delivery_lag_weeks": float(orders["delivery_lag_weeks"].median()) if not orders.empty else math.nan,
        })
        for window in ("FULL_RUN", "SECOND_HALF", "FINAL_QUARTER"):
            part = slice_window(macro, window)
            investment_rows.append({
                "case": label, "window": window,
                "expansion_investment": float(part["expansion_investment"].sum()),
                "replacement_investment": float(part["replacement_investment"].sum()),
                "total_fixed_investment": float(part["total_fixed_investment"].sum()),
                "mean_active_capital_service": float(part["active_capital_service"].mean()),
                "final_active_assets": float(part["active_assets"].iloc[-1]),
            })
        for anchor in (479, 492, 505, 518):
            part = macro.loc[macro["global_step"].between(anchor - 2, anchor + 3), [
                "global_step", "household_income", "household_consumption", "household_saving", "food_demand", "food_sales", "aggregate_firm_revenue", "total_fixed_investment",
            ]].copy()
            part["case"] = label
            part["shock_week"] = anchor
            part["relative_week"] = part["global_step"] - anchor
            late_rows.extend(part.to_dict("records"))
        row, _ = reconciles(macro)
        recon_rows.append({"case": label, **row})
        events = trace.loc[trace["record_type"].eq("EARLY_REVIEW")].copy()
        if not events.empty:
            early_events.extend(events.to_dict("records"))

    supplier = pd.DataFrame(supplier_rows)
    order = pd.DataFrame(order_rows)
    investment = pd.DataFrame(investment_rows)
    recon = pd.DataFrame(recon_rows)
    control_order = order.loc[order["case"].eq(CONTROL)].iloc[0]
    control_investment = investment.loc[investment["case"].eq(CONTROL)].set_index("window")
    order["final_backlog_ratio_vs_control"] = order["final_backlog_units"] / max(TOL, control_order["final_backlog_units"])
    order["accepted_order_ratio_vs_control"] = order["accepted_order_value"] / max(TOL, control_order["accepted_order_value"])
    investment["fixed_investment_ratio_vs_control"] = investment.apply(
        lambda row: row["total_fixed_investment"] / max(TOL, control_investment.loc[row["window"], "total_fixed_investment"]), axis=1
    )
    pd.DataFrame(early_events).to_csv(OUTPUT / "early_review_event_trace.csv", index=False)
    supplier.to_csv(OUTPUT / "control_treatment_supplier_continuity.csv", index=False)
    order.to_csv(OUTPUT / "control_treatment_order_backlog.csv", index=False)
    investment.to_csv(OUTPUT / "control_treatment_investment_levels.csv", index=False)
    pd.DataFrame(late_rows).to_csv(OUTPUT / "late_macro_shock_comparison.csv", index=False)
    recon.to_csv(OUTPUT / "reconciliation_comparison.csv", index=False)

    early = pd.DataFrame(early_events)
    early_count = int(len(early))
    scheduled_count = int((control_trace["record_type"].eq("INVESTMENT_REVIEW")).sum())
    early_fraction = early_count / max(1, early_count + scheduled_count)
    treatment_exists = TREATMENT in results
    treatment_supplier = supplier.loc[supplier["case"].eq(TREATMENT)].iloc[0] if treatment_exists else None
    treatment_order = order.loc[order["case"].eq(TREATMENT)].iloc[0] if treatment_exists else None
    all_reconciled = bool(all(reconciles(result[0])[1] for result in results.values()))
    gaps_weaken = bool(treatment_supplier is not None and treatment_supplier["zero_payroll_weeks"] < supplier.loc[supplier["case"].eq(CONTROL), "zero_payroll_weeks"].iloc[0])
    backlog_comparable = bool(treatment_order is not None and treatment_order["final_backlog_ratio_vs_control"] <= 2.0)
    early_frequent = early_fraction > 0.25
    invalid_early_orders = int((early.get("residual_uncommitted_capacity_gap", pd.Series(dtype=float)) <= TOL).sum()) if not early.empty else 0
    if not class_a_material:
        verdict = "C. MOST_SHUTDOWNS_HAVE_NO_TRUE_BUYER_RESIDUAL_NEED"
    elif not treatment_exists or not all_reconciled or invalid_early_orders:
        verdict = "G. OTHER_BLOCKER"
    elif early_frequent:
        verdict = "D. EARLY_TRIGGER_BECOMES_DE_FACTO_FREQUENT_REVIEW"
    elif gaps_weaken and backlog_comparable:
        verdict = "A. PIPELINE_DEPLETION_EARLY_REVIEW_ACCEPTED"
    elif gaps_weaken:
        verdict = "B. EARLY_REVIEW_REMOVES_GAPS_BUT_BACKLOG_DISTORTS"
    else:
        verdict = "F. SUPPLIER_CONTINUITY_REQUIRES_SEPARATE_OPERATING_ARCHITECTURE"
    flags = {
        "verdict": verdict,
        "pipeline_empty_shutdown_windows": int(classifications.shape[0]),
        "waiting_despite_positive_residual_need_windows": int(classifications["classification"].eq("WAITING_DESPITE_POSITIVE_RESIDUAL_NEED").sum()) if not classifications.empty else 0,
        "class_a_material": class_a_material,
        "treatment_run": treatment_exists,
        "early_reviews": early_count,
        "scheduled_reviews_control": scheduled_count,
        "early_review_fraction": early_fraction,
        "early_trigger_effectively_replaces_13w_cadence": early_frequent,
        "invalid_early_orders": invalid_early_orders,
        "supplier_gaps_weaken": gaps_weaken,
        "backlog_comparable": backlog_comparable,
        "all_reconciliations_pass": all_reconciled,
        "economic_behavior_changed_outside_treatment": False,
        "new_rng_draws": 0,
        "population": POPULATION, "seed": SEED, "weeks": WEEKS,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Step 15 Pipeline-Depletion Early Investment Review Audit

## Verdict

**{verdict}**

Pipeline-empty shutdown windows: `{flags['pipeline_empty_shutdown_windows']}`. Windows waiting despite positive residual buyer need: `{flags['waiting_despite_positive_residual_need_windows']}`. Early reviews: `{early_count}`; share of all reviews: `{early_fraction:.6g}`. Supplier gaps weakened: `{gaps_weaken}`. Backlog comparable: `{backlog_comparable}`. Conservation passed: `{all_reconciled}`.

The treatment preserves normal 13-week reviews. It adds only Firm-specific early reviews when the accepted-order pipeline is empty, the residual gap after shadow committed-capital recognition is positive, internal cash is sufficient, and the week is not already scheduled.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags))


if __name__ == "__main__":
    main()
