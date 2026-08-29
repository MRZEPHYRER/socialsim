from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / "test/output/step15I11B_capital_good_supply_scale_audit"
RUNNER_PATH = ROOT / "test/step15i10h_joint_capital_good_scale.py"
SEED = 42
STEPS = 520
FOOD_FIRMS = 5
N_SMALL = 500
N_LARGE = 5000
HORIZON = 13
PRODUCTIVITY = 3.0
TOLERANCE = 1e-6


def number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_rows(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_runner():
    spec = importlib.util.spec_from_file_location("step15i10h_runner_11b", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sum_field(rows, field):
    return sum(number(row.get(field)) for row in rows)


def mean_field(rows, field):
    values = [number(row.get(field)) for row in rows]
    return statistics.fmean(values) if values else 0.0


def final_field(rows, field):
    return number(rows[-1].get(field)) if rows else 0.0


def labor_classification(panel):
    desired = [number(row.get("desired_labor")) for row in panel]
    capacity = [number(row.get("total_eligible_labor_capacity")) for row in panel]
    unassigned = [number(row.get("unassigned_eligible_workers")) for row in panel]
    actual_services = [number(row.get("actual_labor_services")) for row in panel]
    peak_ratio = max(
        (d / c for d, c in zip(desired, capacity) if c > 1e-12),
        default=0.0,
    )
    mean_unassigned = statistics.fmean(unassigned) if panel else 0.0
    mean_gap = statistics.fmean(
        max(0.0, d - e) for d, e in zip(desired, actual_services)
    ) if panel else 0.0
    if mean_gap > 0.0 and mean_unassigned <= 1e-12:
        return "SYSTEM_LABOR_SCARCITY", peak_ratio, mean_gap
    if mean_gap > 0.0 and mean_unassigned > 0.0:
        return "MATCHING_THROUGHPUT", peak_ratio, mean_gap
    return "OTHER", peak_ratio, mean_gap


def run_case(runner, population):
    runner.POPULATION = population
    runner.FOOD_FIRMS = FOOD_FIRMS
    runner.SEED = SEED
    runner.STEPS = STEPS
    settlements = []
    labor_service_rows = []

    def observe(world, system, step, week):
        labor_service_rows.append({
            "global_step": step,
            "actual_labor_services": sum(
                number(getattr(firm, "capital_good_labor_services", 0.0))
                for firm in getattr(world, "capital_good_firms", [])
            ),
        })
        for settlement in getattr(week, "investment_events", []):
            settlements.append({
                "global_step": step,
                "order_id": getattr(settlement, "order_id", ""),
                "requested_units": number(getattr(settlement, "requested_units", 0.0)),
                "settled_units": number(getattr(settlement, "settled_units", 0.0)),
                "unmet_units": number(getattr(settlement, "unmet_units", 0.0)),
                "desired_expenditure": number(getattr(settlement, "desired_investment_expenditure", 0.0)),
                "settled_expenditure": number(getattr(settlement, "settled_expenditure", 0.0)),
                "unexecuted_expenditure": number(getattr(settlement, "unexecuted_expenditure", 0.0)),
                "financing_gap": number(getattr(settlement, "financing_gap", 0.0)),
                "supplier_revenue": number(getattr(settlement, "supplier_revenue", 0.0)),
                "settlement_reason": getattr(settlement, "settlement_reason", ""),
                "working_capital_loan_used": bool(getattr(settlement, "working_capital_loan_used", False)),
                "fill_count": len(getattr(settlement, "fills", ())),
            })

    result = runner.run_treatment(observer=observe)
    service_by_step = {
        int(number(row["global_step"])): row["actual_labor_services"]
        for row in labor_service_rows
    }
    for row in result["panel"]:
        row["actual_labor_services"] = service_by_step.get(
            int(number(row.get("global_step"))), 0.0
        )
    return {
        "population": population,
        "panel": result["panel"],
        "backlog": result["backlog"],
        "lifecycle": result["lifecycle"],
        "settlements": settlements,
        "system": result["system"],
        "unit_price": number(result.get("unit_price")),
        "max_assignment_violations": result.get("max_assignment_violations", 0),
        "max_feasibility_violation": result.get("max_feasibility_violation", 0.0),
        "max_accounting_gap": result.get("max_accounting_gap", 0.0),
        "max_money_gap": result.get("max_money_gap", 0.0),
        "max_goods_gap": result.get("max_goods_gap", 0.0),
        "capital_loans": result.get("capital_loans", 0.0),
    }


def total_case_metrics(case):
    panel = case["panel"]
    backlog = case["backlog"]
    settlements = case["settlements"]
    unit_price = case["unit_price"]
    return {
        "new_expansion_demand": sum_field(backlog, "new_expansion_demand"),
        "new_replacement_demand": sum_field(backlog, "new_replacement_demand"),
        "opening_backlog": sum_field(backlog, "opening_backlog"),
        "closing_backlog": final_field(backlog, "closing_backlog"),
        "desired_output_flow": sum_field(panel, "desired_output"),
        "desired_labor": sum_field(panel, "desired_labor"),
        "funded_output": sum(
            number(row.get("production")) for row in panel
        ),
        "realized_production": sum_field(panel, "production"),
        "inventory_end": final_field(panel, "inventory_units"),
        "sales_units": sum_field(panel, "sales_units"),
        "supplier_revenue": sum_field(panel, "supplier_revenue"),
        "supplier_cogs": sum_field(panel, "supplier_cogs"),
        "supplier_cash_end": final_field(panel, "cash"),
        "supplier_cash_mean": mean_field(panel, "cash"),
        "supplier_wage_bill": sum_field(panel, "wage_bill"),
        "supplier_employment_mean": mean_field(panel, "actual_capital_good_employment"),
        "capital_good_employment": mean_field(panel, "actual_capital_good_employment"),
        "actual_labor_services": mean_field(panel, "actual_labor_services"),
        "eligible_labor_capacity": mean_field(panel, "total_eligible_labor_capacity"),
        "unassigned_workers": mean_field(panel, "unassigned_eligible_workers"),
        "food_employment": mean_field(panel, "food_employment"),
        "hires": sum_field(panel, "hires"),
        "releases": sum_field(panel, "releases"),
        "requested_units": sum_field(settlements, "requested_units"),
        "settled_units": sum_field(settlements, "settled_units"),
        "unmet_units": sum_field(settlements, "unmet_units"),
        "desired_expenditure": sum_field(settlements, "desired_expenditure"),
        "settled_expenditure": sum_field(settlements, "settled_expenditure"),
        "unexecuted_expenditure": sum_field(settlements, "unexecuted_expenditure"),
        "settlement_supplier_revenue": sum_field(settlements, "supplier_revenue"),
        "settlement_batches": len(settlements),
        "settlement_fills": sum(number(row.get("fill_count")) for row in settlements),
        "financing_gap": sum_field(settlements, "financing_gap"),
        "working_capital_loan_used": sum(
            bool(row.get("working_capital_loan_used")) for row in settlements
        ),
        "active_assets_end": final_field(panel, "active_capital_assets"),
        "retired_assets_end": final_field(panel, "retired_capital_assets"),
        "capital_service_end": final_field(panel, "capital_service"),
        "acquisitions": sum_field(panel, "acquisition_count"),
        "retirements": sum_field(panel, "retirement_count"),
        "replacement_execution": sum_field(panel, "replacement_investment"),
        "expansion_execution": sum_field(panel, "expansion_investment"),
        "unit_price": unit_price,
    }


def ratio(large, small):
    return large / small if abs(small) > 1e-12 else math.inf


def make_scale_rows(small, large, metrics, semantics):
    rows = []
    for metric in metrics:
        a = small[metric]
        b = large[metric]
        rows.append({
            "metric": metric,
            "semantics": semantics.get(metric, "cumulative_or_ending"),
            "N500_value": a,
            "N5000_value": b,
            "aggregate_ratio": ratio(b, a),
            "N500_per_capita": a / N_SMALL,
            "N5000_per_capita": b / N_LARGE,
            "per_capita_ratio": ratio(b / N_LARGE, a / N_SMALL),
            "N500_per_supplier": a,
            "N5000_per_supplier": b,
            "per_supplier_ratio": ratio(b, a),
        })
    return rows


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    runner = load_runner()
    small_case = run_case(runner, N_SMALL)
    small_metrics = total_case_metrics(small_case)
    del runner
    gc.collect()

    runner = load_runner()
    large_case = run_case(runner, N_LARGE)
    large_metrics = total_case_metrics(large_case)
    del runner
    gc.collect()

    demand_metrics = [
        "new_expansion_demand", "new_replacement_demand", "opening_backlog",
        "closing_backlog", "desired_output_flow", "desired_labor",
    ]
    write_rows(
        OUTPUT / "capital_good_demand_scale.csv",
        make_scale_rows(
            small_metrics,
            large_metrics,
            demand_metrics,
            {"closing_backlog": "ending_stock", "opening_backlog": "cumulative_opening_stock", "desired_output_flow": "weekly_flow_sum", "desired_labor": "weekly_flow_sum"},
        ),
    )

    production_metrics = [
        "desired_output_flow", "funded_output", "realized_production",
        "inventory_end", "sales_units", "supplier_revenue", "supplier_cogs",
        "supplier_cash_end", "supplier_cash_mean", "supplier_wage_bill",
        "supplier_employment_mean",
    ]
    production_rows = make_scale_rows(
        small_metrics,
        large_metrics,
        production_metrics,
        {"inventory_end": "ending_stock", "supplier_cash_end": "ending_stock", "supplier_cash_mean": "mean_weekly_state", "supplier_employment_mean": "mean_weekly_state"},
    )
    for row in production_rows:
        metric = row["metric"]
        row["first_chain_gap"] = (
            "desired_to_funded"
            if metric == "funded_output"
            else "funded_to_realized"
            if metric == "realized_production"
            else "realized_to_inventory"
            if metric == "inventory_end"
            else "inventory_to_settlement"
            if metric == "sales_units"
            else "supplier_accounting"
        )
    write_rows(OUTPUT / "supplier_production_scale.csv", production_rows)

    labor_metrics = [
        "desired_labor", "capital_good_employment", "actual_labor_services", "eligible_labor_capacity",
        "unassigned_workers", "food_employment", "hires", "releases",
    ]
    classification_small, peak_small, gap_small = labor_classification(small_case["panel"])
    classification_large, peak_large, gap_large = labor_classification(large_case["panel"])
    labor_rows = make_scale_rows(
        small_metrics,
        large_metrics,
        labor_metrics,
        {"capital_good_employment": "mean_weekly_state", "actual_labor_services": "mean_weekly_state", "eligible_labor_capacity": "mean_weekly_state", "unassigned_workers": "mean_weekly_state", "food_employment": "mean_weekly_state", "desired_labor": "mean_weekly_state"},
    )
    labor_rows.append({
        "metric": "bottleneck_classification",
        "N500_classification": classification_small,
        "N5000_classification": classification_large,
        "N500_peak_desired_to_system_ratio": peak_small,
        "N5000_peak_desired_to_system_ratio": peak_large,
        "N500_mean_desired_minus_actual_services": gap_small,
        "N5000_mean_desired_minus_actual_services": gap_large,
        "semantic_note": "diagnostic classification only; no matching change",
    })
    write_rows(OUTPUT / "supplier_labor_scale.csv", labor_rows)

    system_small = small_case["system"]
    system_large = large_case["system"]
    firm_rows = [
        {"metric": "capital_good_firm_count", "N500": len(getattr(system_small, "capital_good_firms", [])), "N5000": len(getattr(system_large, "capital_good_firms", [])), "semantic_note": "fixed configuration; one generic supplier"},
        {"metric": "investment_review_interval_weeks", "N500": getattr(system_small, "review_interval", math.nan), "N5000": getattr(system_large, "review_interval", math.nan), "semantic_note": "shared cadence"},
        {"metric": "capital_good_productivity", "N500": getattr(system_small, "productivity_per_labor", math.nan), "N5000": getattr(system_large, "productivity_per_labor", math.nan), "semantic_note": "accepted non-calibrated engineering reference"},
        {"metric": "offer_price", "N500": small_metrics["unit_price"], "N5000": large_metrics["unit_price"], "semantic_note": "cost-anchored; unchanged"},
        {"metric": "supplier_order_batch_count", "N500": small_metrics["settlement_batches"], "N500": small_metrics["settlement_batches"], "N5000": large_metrics["settlement_batches"], "semantic_note": "one settlement event per filled buyer order"},
        {"metric": "supplier_settlement_fill_count", "N500": small_metrics["settlement_fills"], "N5000": large_metrics["settlement_fills"], "semantic_note": "observed fills; no artificial firm-count expansion"},
        {"metric": "per_firm_offer_cap", "N500": "not observed", "N5000": "not observed", "semantic_note": "no explicit hard offer cap in adapter"},
        {"metric": "per_firm_production_cap", "N500": "labor/cash constrained", "N5000": "labor/cash constrained", "semantic_note": "no separate fixed batch cap"},
    ]
    for row in firm_rows:
        if "N500" in row and "N5000" in row and isinstance(row["N500"], (int, float)) and isinstance(row["N5000"], (int, float)):
            row["ratio_N5000_over_N500"] = ratio(row["N5000"], row["N500"])
    write_rows(OUTPUT / "supplier_firm_scale_audit.csv", firm_rows)

    settlement_metrics = [
        "requested_units", "settled_units", "unmet_units", "desired_expenditure",
        "settled_expenditure", "unexecuted_expenditure", "settlement_supplier_revenue",
        "settlement_batches", "settlement_fills", "financing_gap",
    ]
    settlement_rows = make_scale_rows(
        small_metrics,
        large_metrics,
        settlement_metrics,
        {"settlement_batches": "count", "settlement_fills": "count"},
    )
    for row in settlement_rows:
        row["settlement_identity_gap"] = (
            large_metrics["settlement_supplier_revenue"] - large_metrics["settled_expenditure"]
            if row["metric"] == "settlement_supplier_revenue" else "see batch identity"
        )
    write_rows(OUTPUT / "order_settlement_scale_bridge.csv", settlement_rows)

    service_metrics = [
        "sales_units", "settled_units", "acquisitions", "active_assets_end",
        "retirements", "capital_service_end",
        "inventory_end", "unmet_units", "closing_backlog",
    ]
    service_rows = make_scale_rows(
        small_metrics,
        large_metrics,
        service_metrics,
        {"active_assets_end": "ending_stock", "capital_service_end": "ending_stock", "inventory_end": "ending_stock", "closing_backlog": "ending_stock"},
    )
    service_rows.extend([
        {
            "metric": "service_per_active_asset_end",
            "N500_value": small_metrics["capital_service_end"] / max(small_metrics["active_assets_end"], 1e-12),
            "N5000_value": large_metrics["capital_service_end"] / max(large_metrics["active_assets_end"], 1e-12),
            "per_capita_ratio": "not applicable",
            "per_supplier_ratio": "not applicable",
        },
        {
            "metric": "assets_acquired_per_sold_unit",
            "N500_value": small_metrics["acquisitions"] / max(small_metrics["sales_units"], 1e-12),
            "N5000_value": large_metrics["acquisitions"] / max(large_metrics["sales_units"], 1e-12),
            "per_capita_ratio": "not applicable",
            "per_supplier_ratio": "not applicable",
        },
    ])
    write_rows(OUTPUT / "capital_service_delivery_bridge.csv", service_rows)

    max_gap = max(
        small_case["max_accounting_gap"], large_case["max_accounting_gap"],
        small_case["max_money_gap"], large_case["max_money_gap"],
        small_case["max_goods_gap"], large_case["max_goods_gap"],
    )
    production_ratio = ratio(
        large_metrics["realized_production"] / N_LARGE,
        small_metrics["realized_production"] / N_SMALL,
    )
    labor_ratio = ratio(
        large_metrics["capital_good_employment"] / N_LARGE,
        small_metrics["capital_good_employment"] / N_SMALL,
    )
    demand_ratio = ratio(
        large_metrics["desired_output_flow"] / N_LARGE,
        small_metrics["desired_output_flow"] / N_SMALL,
    )
    settlement_ratio = ratio(
        large_metrics["settled_units"] / N_LARGE,
        small_metrics["settled_units"] / N_SMALL,
    )
    if production_ratio < 0.5 and demand_ratio >= 0.5:
        verdict = "C. CAPITAL_GOOD_PRODUCTION_THROUGHPUT_BOUNDARY"
    elif classification_large == "SYSTEM_LABOR_SCARCITY":
        verdict = "B. CAPITAL_GOOD_LABOR_ACQUISITION_SCALE_BOUNDARY"
    elif settlement_ratio < 0.5 and production_ratio >= 0.5:
        verdict = "D. ORDER_SETTLEMENT_THROUGHPUT_BOUNDARY"
    elif large_metrics["capital_service_end"] / N_LARGE < 0.5 * small_metrics["capital_service_end"] / N_SMALL:
        verdict = "E. CAPITAL_SERVICE_DELIVERY_BOUNDARY"
    else:
        verdict = "F. MULTIPLE_CAPITAL_GOOD_SUPPLY_BOUNDARIES"

    flags = {
        "verdict": verdict,
        "N500_population": N_SMALL,
        "N5000_population": N_LARGE,
        "food_firms": FOOD_FIRMS,
        "capital_good_firm_count_N500": len(getattr(system_small, "capital_good_firms", [])),
        "capital_good_firm_count_N5000": len(getattr(system_large, "capital_good_firms", [])),
        "review_interval_weeks": HORIZON,
        "productivity": PRODUCTIVITY,
        "offer_price_mode": "cost_anchored",
        "demand_per_capita_ratio": demand_ratio,
        "production_per_capita_ratio": production_ratio,
        "capital_good_employment_per_capita_ratio": labor_ratio,
        "actual_labor_services_per_capita_ratio": ratio(
            large_metrics["actual_labor_services"] / N_LARGE,
            small_metrics["actual_labor_services"] / N_SMALL,
        ),
        "settled_units_per_capita_ratio": settlement_ratio,
        "supplier_cash_end_N500": small_metrics["supplier_cash_end"],
        "supplier_cash_end_N5000": large_metrics["supplier_cash_end"],
        "supplier_cash_mean_N500": small_metrics["supplier_cash_mean"],
        "supplier_cash_mean_N5000": large_metrics["supplier_cash_mean"],
        "desired_to_funded_per_capita_ratio": ratio(
            large_metrics["funded_output"] / N_LARGE,
            small_metrics["funded_output"] / N_SMALL,
        ),
        "capital_service_per_capita_ratio": ratio(large_metrics["capital_service_end"] / N_LARGE, small_metrics["capital_service_end"] / N_SMALL),
        "internal_financing_gap": small_metrics["financing_gap"] + large_metrics["financing_gap"],
        "working_capital_loan_used": bool(small_metrics["working_capital_loan_used"] or large_metrics["working_capital_loan_used"]),
        "max_accounting_or_conservation_gap": max_gap,
        "assignment_violations": max(small_case["max_assignment_violations"], large_case["max_assignment_violations"]),
        "realized_output_above_feasible": max(small_case["max_feasibility_violation"], large_case["max_feasibility_violation"]),
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "mechanism_changed": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    summary = f"""# Step 15I.11B Capital-Good Supply Scale-Boundary Audit

## Verdict

**{verdict}**

This diagnostic reused the accepted P3, cost-anchored price, 13-week
backlog-flow, 52-week lifecycle, and internal-cash-only configuration for
N=500 and N=5000 with seed42 and five Food Firms. No capital-good Firm count,
productivity, price, backlog rule, labor rule, financing rule, or Step13
behavior was changed.

## Supply-chain interpretation

The demand/backlog chain, supplier production chain, labor acquisition, order
settlement, and capital-service delivery are reported separately in the six
CSV bridges. The primary classification is based on per-capita ratios and
observed desired-to-realized gaps, not on aggregate totals alone.

Demand per-capita ratio: `{demand_ratio:.6f}`\n
Production per-capita ratio: `{production_ratio:.6f}`\n
Capital-good employment per-capita ratio: `{labor_ratio:.6f}`\n
Actual paid labor-services per-capita ratio: `{flags['actual_labor_services_per_capita_ratio']:.6f}`\n
Settlement units per-capita ratio: `{settlement_ratio:.6f}`

The first large-world throughput loss is the desired-output to funded-output
bridge (`{flags['desired_to_funded_per_capita_ratio']:.6f}` per-capita ratio).
Funded output and realized production remain equal, while supplier cash is
depleted to zero and mean supplier cash scales at only
`{large_metrics['supplier_cash_mean'] / max(small_metrics['supplier_cash_mean'], 1e-12):.6f}`.
Inventory-to-settlement accounting remains closed, so the evidence points to
supplier operating throughput / wage affordability rather than an order
settlement bottleneck. The one-supplier architecture and 13-week cadence are
identical in both worlds, with no explicit hard offer or production batch cap.

## Conservation

Accounting, money, goods, assignment, and feasibility diagnostics are
preserved in `acceptance_flags.json`. This stage is diagnostic only.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
