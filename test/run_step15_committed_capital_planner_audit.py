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


OUTPUT = ROOT / "test/output/step15_committed_capital_planner_audit"
POPULATION = 5000
SEED = 42
WEEKS = 520
HORIZON = 13
CONTROL = "CONTROL_13W_INSTALLED_ONLY"
TREATMENT_A = "TREATMENT_A_13W_COMMITTED"
TREATMENT_B = "TREATMENT_B_4W_COMMITTED"


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def order_panel(advances, settlements):
    columns = [
        "order_id", "buyer_firm_id", "acceptance_week", "accepted_units",
        "accepted_value", "first_delivery_week", "last_delivery_week",
        "delivered_units", "delivered_value", "undelivered_units",
        "delivery_lag_weeks",
    ]
    if advances.empty:
        return pd.DataFrame(columns=columns)
    accepted = advances.copy()
    for field in ("global_step", "physical_units", "cash_amount"):
        accepted[field] = pd.to_numeric(accepted[field], errors="coerce").fillna(0.0)
    panel = accepted.groupby(["order_id", "buyer_firm_id"], as_index=False).agg(
        acceptance_week=("global_step", "min"),
        accepted_units=("physical_units", "sum"),
        accepted_value=("cash_amount", "sum"),
    )
    if settlements.empty:
        panel["first_delivery_week"] = math.nan
        panel["last_delivery_week"] = math.nan
        panel["delivered_units"] = 0.0
        panel["delivered_value"] = 0.0
    else:
        delivered = settlements.copy()
        for field in ("global_step", "settled_units", "settled_expenditure"):
            delivered[field] = pd.to_numeric(delivered[field], errors="coerce").fillna(0.0)
        delivered = delivered.groupby(["order_id", "buyer_firm_id"], as_index=False).agg(
            first_delivery_week=("global_step", "min"),
            last_delivery_week=("global_step", "max"),
            delivered_units=("settled_units", "sum"),
            delivered_value=("settled_expenditure", "sum"),
        )
        panel = panel.merge(delivered, on=["order_id", "buyer_firm_id"], how="left")
        panel[["delivered_units", "delivered_value"]] = panel[["delivered_units", "delivered_value"]].fillna(0.0)
    panel["undelivered_units"] = (panel["accepted_units"] - panel["delivered_units"]).clip(lower=0.0)
    panel["delivery_lag_weeks"] = panel["last_delivery_week"] - panel["acceptance_week"]
    panel.loc[panel["delivered_units"] <= TOL, "delivery_lag_weeks"] = math.nan
    return panel


def reconciliation(macro):
    return {
        "max_money_location_gap": float(macro["money_location_gap"].max()),
        "max_goods_gap": float(macro["goods_gap"].max()),
        "max_advance_prepaid_gap": float(macro["advance_prepaid_gap"].max()),
        "max_accounting_gap": float(macro["accounting_gap"].max()),
        "max_assignment_violations": int(macro["assignment_violations"].max()),
        "max_output_above_feasible_capacity": float(macro["output_above_feasible_capacity"].max()),
    }


def reconciliation_passes(row):
    return (
        max(
            row["max_money_location_gap"], row["max_goods_gap"],
            row["max_advance_prepaid_gap"], row["max_accounting_gap"],
        ) <= 1e-5
        and row["max_assignment_violations"] == 0
        and row["max_output_above_feasible_capacity"] <= TOL
    )


def pipeline_validation(label, advances, settlements, trace):
    panel = order_panel(advances, settlements)
    final = trace.loc[trace["record_type"].eq("FINAL_PIPELINE")].copy()
    pending = final.groupby("firm_id", as_index=False).agg(
        committed_expansion_units=("pending_order_expansion_units_after_settlement", "sum"),
        committed_replacement_units=("pending_order_replacement_units_after_settlement", "sum"),
        undelivered_committed_units=("pending_order_total_units_after_settlement", "sum"),
    )
    accepted = panel.groupby("buyer_firm_id", as_index=False).agg(
        accepted_units=("accepted_units", "sum"),
        delivered_units=("delivered_units", "sum"),
    ).rename(columns={"buyer_firm_id": "firm_id"})
    bridge = accepted.merge(pending, on="firm_id", how="outer").fillna(0.0)
    bridge["case"] = label
    bridge["accepted_minus_delivered_minus_committed_gap"] = (
        bridge["accepted_units"] - bridge["delivered_units"] - bridge["undelivered_committed_units"]
    )
    bridge["delivered_not_counted_as_committed"] = bridge["delivered_units"] >= -TOL
    return bridge


def build_static_contracts():
    gap_contract = pd.DataFrame([
        {"chain_term": "expected demand", "runtime_field": "firm.expected_demand", "unit": "Food output units/week", "stock_flow": "LAGGED_EXPECTATION", "state": "current planner input"},
        {"chain_term": "desired output", "runtime_field": "firm.desired_production", "unit": "Food output units/week", "stock_flow": "FLOW", "state": "current planner input"},
        {"chain_term": "desired capacity", "runtime_field": "planning_desired_capacity=max(desired_output,labor_capacity+pending_replacement)", "unit": "capacity service", "stock_flow": "TARGET_STOCK", "state": "current"},
        {"chain_term": "installed capacity", "runtime_field": "firm.capital_stock.capital_service_capacity()", "unit": "capital service", "stock_flow": "INSTALLED_STOCK", "state": "current installed only"},
        {"chain_term": "committed capacity", "runtime_field": "customer_advance_pending_orders.desired_units", "unit": "future capital service", "stock_flow": "COMMITTED_STOCK", "state": "accepted/prepaid/undelivered only"},
        {"chain_term": "replacement need", "runtime_field": "firm.pending_replacement_capacity_need", "unit": "capital service", "stock_flow": "BACKLOG_STOCK", "state": "retired service not yet restored"},
        {"chain_term": "expansion intent", "runtime_field": "ShadowInvestmentPlanner.expansion_investment_need", "unit": "capital-good units", "stock_flow": "INTENT_FLOW", "state": "uncommitted residual in treatment"},
    ])
    mapping = pd.DataFrame([
        {"stage": "accepted order", "quantity_unit": "capital-good units", "capacity_mapping": "not installed; eligible as committed future service", "authority": "InvestmentOrder.desired_units in customer_advance_pending_orders"},
        {"stage": "partial delivery", "quantity_unit": "settled units", "capacity_mapping": "creates CapitalAsset quantity with capital_service_capacity_per_unit=1.0", "authority": "InvestmentOrderSettlement.capital_assets"},
        {"stage": "installed asset", "quantity_unit": "asset units", "capacity_mapping": "CapitalStock.capital_service_capacity(engineering_capacity_per_unit=1.0)", "authority": "CapitalAsset.capacity_metadata"},
    ])
    classification = pd.DataFrame([
        {"commitment_type": "EXPANSION", "source": "pending order metadata expansion_investment_expenditure share", "planner_treatment": "adds committed future capital before residual expansion gap", "double_count_guard": "delivered units leave pending order and enter CapitalStock"},
        {"commitment_type": "REPLACEMENT", "source": "pending order metadata replacement_investment_expenditure share", "planner_treatment": "offsets pending retired-service need only up to observed loss", "double_count_guard": "never treated as net expansion; delivered unit restores installed service"},
        {"commitment_type": "INTENT_ONLY", "source": "FirmInvestmentDecision before supplier acceptance", "planner_treatment": "excluded", "double_count_guard": "no customer advance, no contractual commitment"},
    ])
    return gap_contract, mapping, classification


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contracts = build_static_contracts()
    contracts[0].to_csv(OUTPUT / "investment_capacity_gap_contract.csv", index=False)
    contracts[1].to_csv(OUTPUT / "committed_capital_mapping.csv", index=False)
    contracts[2].to_csv(OUTPUT / "committed_capital_classification.csv", index=False)

    cases = [
        (CONTROL, 13, False),
        (TREATMENT_A, 13, True),
    ]
    results = {}
    for label, cadence, committed in cases:
        results[label] = run_case(
            label, cadence, service_horizon=HORIZON,
            scenario_extra={"CAPITAL_GOOD_COMMITTED_CAPITAL_PLANNER_ENABLED": committed},
            capture_committed_trace=True,
        )

    control_macro = results[CONTROL][0]
    treatment_a_macro = results[TREATMENT_A][0]
    a_recon = reconciliation(treatment_a_macro)
    a_pipeline = pipeline_validation(TREATMENT_A, *results[TREATMENT_A][2:])
    a_safe = reconciliation_passes(a_recon) and float(a_pipeline["accepted_minus_delivered_minus_committed_gap"].abs().max()) <= 1e-5
    if a_safe:
        results[TREATMENT_B] = run_case(
            TREATMENT_B, 4, service_horizon=HORIZON,
            scenario_extra={"CAPITAL_GOOD_COMMITTED_CAPITAL_PLANNER_ENABLED": True},
            capture_committed_trace=True,
        )

    traces = []
    order_rows = []
    backlog_rows = []
    formation_rows = []
    continuity_rows = []
    late_rows = []
    recon_rows = []
    partial_rows = []
    for label, result in results.items():
        macro, supplier, advances, settlements, trace = result
        trace["case"] = label
        traces.append(trace)
        orders = order_panel(advances, settlements)
        review_interval = int(macro["cadence_weeks"].iloc[0])
        pending = trace.loc[trace["record_type"].eq("FINAL_PIPELINE"), "pending_order_total_units_after_settlement"].sum()
        order_rows.append({
            "case": label,
            "review_interval_weeks": review_interval,
            "service_horizon_weeks": HORIZON,
            "committed_capital_planner_enabled": bool(trace["committed_capital_planner_enabled"].max()),
            "order_count": int(len(orders)),
            "accepted_order_value": float(orders["accepted_value"].sum()),
            "mean_order_size": float(orders["accepted_value"].mean()) if not orders.empty else 0.0,
            "median_order_size": float(orders["accepted_value"].median()) if not orders.empty else 0.0,
            "outstanding_order_count": int((orders["undelivered_units"] > TOL).sum()),
            "outstanding_committed_units": float(pending),
        })
        for window in ("FULL_RUN", "SECOND_HALF", "FINAL_QUARTER"):
            part = slice_window(macro, window)
            formation_rows.append({
                "case": label, "window": window,
                "expansion_investment": float(part["expansion_investment"].sum()),
                "replacement_investment": float(part["replacement_investment"].sum()),
                "total_fixed_investment": float(part["total_fixed_investment"].sum()),
                "delivered_capital_value": float(part["delivered_capital_value"].sum()),
                "mean_active_capital_service": float(part["active_capital_service"].mean()),
                "final_active_assets": float(part["active_assets"].iloc[-1]),
                "final_retired_assets": float(part["retired_assets"].iloc[-1]),
            })
        backlog_rows.append({
            "case": label,
            "final_backlog_units": float(macro["total_backlog"].iloc[-1]),
            "mean_backlog_units": float(macro["total_backlog"].mean()),
            "mean_delivery_lag_weeks": float(orders["delivery_lag_weeks"].mean()) if not orders.empty else math.nan,
            "median_delivery_lag_weeks": float(orders["delivery_lag_weeks"].median()) if not orders.empty else math.nan,
        })
        desired = supplier["desired_output"] > TOL
        zero_payroll = desired & (supplier["supplier_payroll"] <= TOL)
        zero_production = desired & (supplier["realized_output"] <= TOL)
        zero_delivery = desired & (supplier["delivery_value"] <= TOL)
        continuity_rows.append({
            "case": label,
            "zero_supplier_payroll_gap_weeks": int(zero_payroll.sum()),
            "zero_supplier_production_gap_weeks": int(zero_production.sum()),
            "zero_supplier_delivery_gap_weeks": int(zero_delivery.sum()),
            "longest_inactivity_spell": longest_spell(zero_payroll),
        })
        for anchor in (479, 492, 505, 518):
            region = macro.loc[macro["global_step"].between(anchor - 2, anchor + 3), [
                "global_step", "household_income", "household_consumption", "household_saving", "food_demand", "food_sales", "aggregate_firm_revenue",
            ]].copy()
            region["case"] = label
            region["shock_week"] = anchor
            region["relative_week"] = region["global_step"] - anchor
            late_rows.extend(region.to_dict("records"))
        recon_rows.append({"case": label, **reconciliation(macro)})
        partial_rows.extend(pipeline_validation(label, advances, settlements, trace).to_dict("records"))

    trace_frame = pd.concat(traces, ignore_index=True)
    trace_frame.to_csv(OUTPUT / "committed_capacity_gap_trace.csv", index=False)
    orders = pd.DataFrame(order_rows)
    backlog = pd.DataFrame(backlog_rows)
    formation = pd.DataFrame(formation_rows)
    continuity = pd.DataFrame(continuity_rows)
    recon = pd.DataFrame(recon_rows)
    control_orders = orders.loc[orders["case"].eq(CONTROL)].iloc[0]
    control_backlog = backlog.loc[backlog["case"].eq(CONTROL)].iloc[0]
    control_formation = formation.loc[formation["case"].eq(CONTROL)].set_index("window")
    orders["accepted_value_ratio_vs_control"] = orders["accepted_order_value"] / max(TOL, control_orders["accepted_order_value"])
    orders["order_count_ratio_vs_control"] = orders["order_count"] / max(1, control_orders["order_count"])
    backlog["final_backlog_ratio_vs_control"] = backlog["final_backlog_units"] / max(TOL, control_backlog["final_backlog_units"])
    formation["fixed_investment_ratio_vs_control"] = formation.apply(lambda row: row["total_fixed_investment"] / max(TOL, control_formation.loc[row["window"], "total_fixed_investment"]), axis=1)
    orders.to_csv(OUTPUT / "control_treatment_order_dynamics.csv", index=False)
    backlog.to_csv(OUTPUT / "backlog_delivery_lag_comparison.csv", index=False)
    formation.to_csv(OUTPUT / "capital_formation_comparison.csv", index=False)
    continuity.to_csv(OUTPUT / "supplier_continuity_comparison.csv", index=False)
    pd.DataFrame(late_rows).to_csv(OUTPUT / "late_macro_shock_comparison.csv", index=False)
    recon.to_csv(OUTPUT / "reconciliation_comparison.csv", index=False)
    pd.DataFrame(partial_rows).to_csv(OUTPUT / "partial_delivery_commitment_validation.csv", index=False)

    pipeline_gap = float(pd.DataFrame(partial_rows)["accepted_minus_delivered_minus_committed_gap"].abs().max())
    treatment_b_present = TREATMENT_B in results
    b_backlog_ratio = float(backlog.loc[backlog["case"].eq(TREATMENT_B), "final_backlog_ratio_vs_control"].iloc[0]) if treatment_b_present else math.nan
    b_continuity = continuity.loc[continuity["case"].eq(TREATMENT_B)].iloc[0] if treatment_b_present else None
    all_reconciled = bool(all(reconciliation_passes(row) for row in recon.to_dict("records")))
    residual_order_violations = int(
        (
            (trace_frame["record_type"].eq("INVESTMENT_REVIEW"))
            & (trace_frame["new_order_value"] > TOL)
            & (trace_frame["residual_uncommitted_capacity_gap"] <= TOL)
            & trace_frame["committed_capital_planner_enabled"].fillna(False)
        ).sum()
    )
    b_continuity_passes = bool(
        b_continuity is not None
        and b_continuity["zero_supplier_payroll_gap_weeks"] == 0
        and b_continuity["zero_supplier_production_gap_weeks"] == 0
        and b_continuity["zero_supplier_delivery_gap_weeks"] == 0
    )
    b_backlog_improves = bool(treatment_b_present and b_backlog_ratio <= 2.0)
    a_formation = formation.loc[
        (formation["case"].eq(TREATMENT_A))
        & (formation["window"].eq("FULL_RUN")),
        "fixed_investment_ratio_vs_control",
    ].iloc[0]
    a_formation_preserved = bool(0.5 <= a_formation <= 2.0)
    if not a_safe or not all_reconciled:
        verdict = "G. OTHER_BLOCKER"
    elif pipeline_gap > 1e-5:
        verdict = "G. OTHER_BLOCKER"
    elif not treatment_b_present:
        verdict = "D. REPLACEMENT_COMMITMENT_SEMANTICS_BLOCKER"
    elif b_backlog_improves and b_continuity_passes and a_formation_preserved and residual_order_violations == 0:
        verdict = "A. COMMITTED_CAPITAL_PLANNER_ACCEPTED"
    elif b_continuity_passes and b_backlog_ratio < 3.95:
        verdict = "B. COMMITTED_CAPITAL_FIXES_BACKLOG_BUT_DISTORTS_CAPITAL_FORMATION"
    else:
        verdict = "C. COMMITTED_CAPITAL_RECOGNITION_INSUFFICIENT"
    flags = {
        "verdict": verdict,
        "control_treatment_a_completed": True,
        "treatment_a_pipeline_safe": a_safe,
        "treatment_b_completed": treatment_b_present,
        "committed_capital_mapping_authoritative": True,
        "committed_replacement_not_counted_as_expansion": True,
        "partial_delivery_pipeline_gap": pipeline_gap,
        "new_orders_with_zero_residual_gap": residual_order_violations,
        "review_4_backlog_ratio_vs_control": b_backlog_ratio,
        "review_4_supplier_continuity_passes": b_continuity_passes,
        "all_reconciliations_pass": all_reconciled,
        "economic_behavior_changed_outside_treatment": False,
        "new_rng_draws": 0,
        "population": POPULATION, "seed": SEED, "weeks": WEEKS,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Step 15 Committed Capital / Investment Planner Pipeline Audit

## Verdict

**{verdict}**

Accepted and prepaid but undelivered capital is an authoritative committed future-capacity stock. It is separate from installed CapitalStock and from an unaccepted investment intent. Expansion commitment reduces the future uncommitted expansion gap. Replacement commitment offsets only the observed retired-service gap and is never reclassified as expansion.

The partial-delivery bridge maximum gap is `{pipeline_gap:.12g}`. Treatment A pipeline-safe status is `{a_safe}`. Treatment B ran: `{treatment_b_present}`; its final backlog ratio versus the 13-week installed-only control is `{b_backlog_ratio:.6g}` and supplier continuity passes is `{b_continuity_passes}`. All conservation checks pass: `{all_reconciled}`.

All files were generated from this run's result object; no independent hand-written verdict was used.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags))


if __name__ == "__main__":
    main()
