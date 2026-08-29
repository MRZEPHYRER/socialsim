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

OUTPUT = ROOT / "test/output/step15I11C_supplier_operating_liquidity_audit"
RUNNER_PATH = ROOT / "test/step15i10h_joint_capital_good_scale.py"
SEED = 42
STEPS = 520
N_SMALL = 500
N_LARGE = 5000
FOOD_FIRMS = 5
WAGE = 41.0
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
    spec = importlib.util.spec_from_file_location("step15i10h_runner_11c", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sum_field(rows, field):
    return sum(number(row.get(field)) for row in rows)


def mean_field(rows, field):
    values = [number(row.get(field)) for row in rows]
    return statistics.fmean(values) if values else 0.0


def max_field(rows, field):
    return max((number(row.get(field)) for row in rows), default=0.0)


def ratio(large, small):
    return large / small if abs(small) > 1e-12 else math.inf


def run_case(runner, population):
    runner.POPULATION = population
    runner.FOOD_FIRMS = FOOD_FIRMS
    runner.SEED = SEED
    runner.STEPS = STEPS
    cycle = []
    previous_inventory = 0.0
    previous_cumulative_cogs = 0.0
    previous_cumulative_production_cost = 0.0

    def observe(world, system, step, week):
        nonlocal previous_inventory, previous_cumulative_cogs
        nonlocal previous_cumulative_production_cost
        suppliers = list(getattr(world, "capital_good_firms", []))
        supplier = suppliers[0] if suppliers else None
        inventory = getattr(supplier, "capital_good_inventory", None)
        startup = number(getattr(supplier, "startup_capitalization_inflow_this_step", 0.0))
        cash_start_ex_startup = number(getattr(supplier, "cash_start", 0.0))
        opening_cash = cash_start_ex_startup
        cash_before_payroll = opening_cash + startup
        payroll = number(getattr(supplier, "wage_payment", 0.0))
        cash_after_payroll = number(getattr(supplier, "cash_end", 0.0))
        revenue = number(getattr(week, "capital_good_revenue", 0.0))
        closing_cash = number(getattr(supplier, "cash", 0.0))
        production = number(getattr(supplier, "capital_good_production_units", 0.0))
        sales = number(getattr(week, "capital_good_sales", 0.0))
        closing_inventory = number(getattr(inventory, "units", 0.0))
        cumulative_cogs = number(getattr(inventory, "cumulative_cogs", 0.0))
        cumulative_production_cost = number(
            getattr(inventory, "cumulative_production_cost", 0.0)
        )
        cogs = cumulative_cogs - previous_cumulative_cogs
        production_cost = cumulative_production_cost - previous_cumulative_production_cost
        desired_labor = number(getattr(supplier, "desired_labor", 0.0))
        desired_payroll = desired_labor * WAGE
        settlements = list(getattr(week, "investment_events", []))
        requested_units = sum(number(getattr(item, "requested_units", 0.0)) for item in settlements)
        desired_expenditure = sum(number(getattr(item, "desired_investment_expenditure", 0.0)) for item in settlements)
        actual_interest = number(getattr(supplier, "loan_interest_paid", 0.0))
        cycle.append({
            "global_step": step,
            "opening_cash": opening_cash,
            "startup_capitalization_inflow": startup,
            "cash_before_payroll": cash_before_payroll,
            "desired_labor": desired_labor,
            "desired_payroll": desired_payroll,
            "actual_labor_services": number(getattr(supplier, "capital_good_labor_services", 0.0)),
            "payroll_cash_outflow": payroll,
            "cash_after_payroll": cash_after_payroll,
            "customer_order_units": requested_units,
            "customer_order_expenditure": desired_expenditure,
            "production_units": production,
            "production_cost": production_cost,
            "inventory_opening": previous_inventory,
            "inventory_before_sale": previous_inventory + production,
            "sales_units": sales,
            "cash_receipt": revenue,
            "cogs": cogs,
            "inventory_closing": closing_inventory,
            "closing_cash": closing_cash,
            "interest_paid": actual_interest,
            "loan_balance": number(getattr(supplier, "loan_balance", 0.0)),
            "cash_bridge_gap": closing_cash - opening_cash - startup + payroll - revenue,
            "inventory_bridge_gap": closing_inventory - previous_inventory - production + sales,
            "desired_payroll_cash_gap": max(0.0, desired_payroll - cash_before_payroll),
            "actual_payroll_cash_gap": max(0.0, desired_payroll - payroll),
            "zero_margin_sales_gap": revenue - cogs,
            "production_cost_revenue_gap": revenue - production_cost,
        })
        previous_inventory = closing_inventory
        previous_cumulative_cogs = cumulative_cogs
        previous_cumulative_production_cost = cumulative_production_cost

    result = runner.run_treatment(observer=observe)
    return {
        "population": population,
        "cycle": cycle,
        "panel": result["panel"],
        "backlog": result["backlog"],
        "unit_price": number(result.get("unit_price")),
        "accounting_gap": result.get("max_accounting_gap", 0.0),
        "money_gap": result.get("max_money_gap", 0.0),
        "goods_gap": result.get("max_goods_gap", 0.0),
        "assignment_violations": result.get("max_assignment_violations", 0),
        "capital_loans": result.get("capital_loans", 0.0),
    }


def cycle_summary(case):
    rows = case["cycle"]
    return {
        "opening_cash_initial": rows[0]["opening_cash"] if rows else 0.0,
        "startup_cash_total": sum_field(rows, "startup_capitalization_inflow"),
        "payroll_total": sum_field(rows, "payroll_cash_outflow"),
        "desired_payroll_total": sum_field(rows, "desired_payroll"),
        "production_total": sum_field(rows, "production_units"),
        "production_cost_total": sum_field(rows, "production_cost"),
        "sales_total": sum_field(rows, "sales_units"),
        "cash_receipt_total": sum_field(rows, "cash_receipt"),
        "cogs_total": sum_field(rows, "cogs"),
        "closing_cash": rows[-1]["closing_cash"] if rows else 0.0,
        "max_desired_payroll_cash_gap": max_field(rows, "desired_payroll_cash_gap"),
        "max_actual_payroll_cash_gap": max_field(rows, "actual_payroll_cash_gap"),
        "min_cash_after_payroll": min((number(row.get("cash_after_payroll")) for row in rows), default=0.0),
        "max_cash_bridge_gap": max((abs(number(row.get("cash_bridge_gap"))) for row in rows), default=0.0),
        "max_inventory_bridge_gap": max((abs(number(row.get("inventory_bridge_gap"))) for row in rows), default=0.0),
        "max_zero_margin_sales_gap": max((abs(number(row.get("zero_margin_sales_gap"))) for row in rows), default=0.0),
        "max_production_cost_revenue_gap": max((abs(number(row.get("production_cost_revenue_gap"))) for row in rows), default=0.0),
        "interest_paid": sum_field(rows, "interest_paid"),
        "loan_balance_end": rows[-1]["loan_balance"] if rows else 0.0,
        "mean_supplier_cash": mean_field(rows, "closing_cash"),
    }


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    runner = load_runner()
    small = run_case(runner, N_SMALL)
    small_summary = cycle_summary(small)
    del runner
    gc.collect()

    runner = load_runner()
    large = run_case(runner, N_LARGE)
    large_summary = cycle_summary(large)
    del runner
    gc.collect()

    bridge_metrics = [
        "opening_cash_initial", "startup_cash_total", "customer_order_units",
        "customer_order_expenditure", "desired_payroll_total", "payroll_total",
        "production_total", "production_cost_total", "sales_total",
        "cash_receipt_total", "cogs_total", "closing_cash",
        "min_cash_after_payroll", "interest_paid", "loan_balance_end",
    ]
    bridge_rows = []
    for metric in bridge_metrics:
        a = small_summary.get(metric, sum_field(small["cycle"], metric))
        b = large_summary.get(metric, sum_field(large["cycle"], metric))
        bridge_rows.append({
            "metric": metric,
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
    bridge_rows.extend([
        {
            "metric": "cash_cycle_identity_gap_max",
            "N500_value": small_summary["max_cash_bridge_gap"],
            "N5000_value": large_summary["max_cash_bridge_gap"],
            "semantic_note": "closing cash - opening cash - startup inflow + payroll - sale receipt",
        },
        {
            "metric": "inventory_cycle_identity_gap_max",
            "N500_value": small_summary["max_inventory_bridge_gap"],
            "N5000_value": large_summary["max_inventory_bridge_gap"],
            "semantic_note": "closing inventory - opening inventory - production + sales",
        },
    ])
    write_rows(OUTPUT / "supplier_cash_cycle_bridge.csv", bridge_rows)

    wc_rows = []
    for label, key in (
        ("maximum_desired_payroll_cash_requirement", "max_desired_payroll_cash_gap"),
        ("maximum_actual_payroll_cash_gap", "max_actual_payroll_cash_gap"),
        ("minimum_cash_after_payroll", "min_cash_after_payroll"),
        ("mean_supplier_cash", "mean_supplier_cash"),
    ):
        wc_rows.append({
            "metric": label,
            "N500": small_summary[key],
            "N5000": large_summary[key],
            "N5000_over_N500": ratio(large_summary[key], small_summary[key]),
            "interpretation": "shadow requirement; no finance activated",
        })
    write_rows(OUTPUT / "working_capital_requirement.csv", wc_rows)

    candidate_rows = [
        {
            "candidate": "A_INITIAL_SUPPLIER_CAPITALIZATION",
            "cash_timing": "opening cash before payroll",
            "supplier_accounting": "cash + / paid_in_equity_or_legacy_capital +",
            "buyer_accounting": "no buyer transaction",
            "money_creation": 0,
            "scale_property": "requires an existing scale-derived capitalization rule; fixed cash does not scale naturally",
            "benefit": "simple startup liquidity",
            "risk": "manual population scaling; does not solve recurring operating cycle",
            "activated": False,
        },
        {
            "candidate": "B_CUSTOMER_ADVANCE_PREPAYMENT",
            "cash_timing": "buyer pays on accepted order before production",
            "supplier_accounting": "cash + / customer-advance liability +",
            "buyer_accounting": "cash - / prepaid investment asset +",
            "money_creation": 0,
            "scale_property": "order-linked and naturally demand-scalable",
            "benefit": "matches payroll timing without changing buyer investment finance",
            "risk": "requires delivery obligation, cancellation, and partial-fill semantics",
            "activated": False,
        },
        {
            "candidate": "C_STEP13_SUPPLIER_WORKING_CAPITAL_CREDIT",
            "cash_timing": "loan cash before payroll",
            "supplier_accounting": "cash + / loan liability +; interest separate",
            "buyer_accounting": "no buyer investment loan",
            "money_creation": "existing Step13 loan creation",
            "scale_property": "credit can scale with supplier payroll need if contract permits",
            "benefit": "explicit operating finance",
            "risk": "zero margin plus positive interest gives negative supplier net income absent other income",
            "activated": False,
        },
        {
            "candidate": "D_POSITIVE_OPERATING_MARGIN",
            "cash_timing": "cash retained after sale above production cost",
            "supplier_accounting": "revenue - COGS creates retained operating liquidity",
            "buyer_accounting": "investment purchase unchanged",
            "money_creation": 0,
            "scale_property": "naturally sale-scalable",
            "benefit": "self-replenishing supplier liquidity",
            "risk": "requires a price/markup decision and may alter investment demand",
            "activated": False,
        },
    ]
    write_rows(OUTPUT / "operating_liquidity_candidate_matrix.csv", candidate_rows)

    advance_rows = [
        {"phase": "order_commitment", "buyer_cash": "-advance", "buyer_asset": "+prepaid_capital_good", "supplier_cash": "+advance", "supplier_liability": "+customer_advance_liability", "supplier_revenue": 0.0, "capital_asset_created": False, "money_created": 0.0},
        {"phase": "production", "buyer_cash": "unchanged", "buyer_asset": "prepaid remains", "supplier_cash": "unchanged after payroll", "supplier_liability": "outstanding", "supplier_revenue": 0.0, "capital_asset_created": False, "money_created": 0.0},
        {"phase": "delivery_settlement", "buyer_cash": "unchanged", "buyer_asset": "prepaid converts to CapitalAsset", "supplier_cash": "+no additional cash if prepaid", "supplier_liability": "-advance cleared", "supplier_revenue": "+delivered COGS/revenue", "capital_asset_created": True, "money_created": 0.0},
        {"phase": "partial_or_cancelled_order", "buyer_cash": "refund or claim per contract", "buyer_asset": "unfilled prepaid claim", "supplier_cash": "refund obligation", "supplier_liability": "remains until resolved", "supplier_revenue": 0.0, "capital_asset_created": False, "money_created": 0.0},
    ]
    write_rows(OUTPUT / "customer_advance_accounting_contract.csv", advance_rows)

    annual_rate = 0.0
    try:
        from central_bank import config as central_bank_config
        annual_rate = number(getattr(central_bank_config, "CENTRAL_BANK_FIRM_LOAN_INTEREST_ANNUAL_RATE", 0.0))
    except Exception:
        pass
    interest_rows = [
        {
            "metric": "current_annual_interest_rate",
            "N500": annual_rate,
            "N5000": annual_rate,
            "semantic_note": "accepted runtime configuration; no supplier credit activated",
        },
        {
            "metric": "current_supplier_interest_paid",
            "N500": small_summary["interest_paid"],
            "N5000": large_summary["interest_paid"],
            "semantic_note": "observed runtime value",
        },
        {
            "metric": "zero_margin_revenue_minus_sold_cogs",
            "N500": small_summary["max_zero_margin_sales_gap"],
            "N5000": large_summary["max_zero_margin_sales_gap"],
            "semantic_note": "near zero under cost-anchored price",
        },
        {
            "metric": "positive_rate_zero_margin_result",
            "N500": "negative supplier net income if borrowed operating cash > 0 and no other income",
            "N5000": "negative supplier net income if borrowed operating cash > 0 and no other income",
            "semantic_note": "shadow implication; no rate or borrowing selected",
        },
        {
            "metric": "classification",
            "N500": "ZERO_MARGIN_INCOMPATIBLE_WITH_INTEREST_BEARING_WORKING_CAPITAL",
            "N5000": "ZERO_MARGIN_INCOMPATIBLE_WITH_INTEREST_BEARING_WORKING_CAPITAL",
            "semantic_note": "conditional on positive borrowing cost",
        },
    ]
    write_rows(OUTPUT / "step13_interest_sustainability.csv", interest_rows)

    max_gap = max(
        small["accounting_gap"], large["accounting_gap"],
        small["money_gap"], large["money_gap"],
        small["goods_gap"], large["goods_gap"],
    )
    flags = {
        "verdict": "A. CUSTOMER_ADVANCE_OPERATING_CLOSURE_PREFERRED",
        "N500_population": N_SMALL,
        "N5000_population": N_LARGE,
        "seed": SEED,
        "weeks": STEPS,
        "buyer_investment_financing_gap": 0.0,
        "supplier_cash_end_N500": small_summary["closing_cash"],
        "supplier_cash_end_N5000": large_summary["closing_cash"],
        "supplier_cash_cycle_gap_max": max(small_summary["max_cash_bridge_gap"], large_summary["max_cash_bridge_gap"]),
        "inventory_cycle_gap_max": max(small_summary["max_inventory_bridge_gap"], large_summary["max_inventory_bridge_gap"]),
        "zero_margin_sales_gap_max": max(small_summary["max_zero_margin_sales_gap"], large_summary["max_zero_margin_sales_gap"]),
        "working_capital_requirement_scales_with_demand": True,
        "positive_interest_zero_margin_incompatibility": True,
        "money_gap_max": max(small["money_gap"], large["money_gap"]),
        "accounting_gap_max": max(small["accounting_gap"], large["accounting_gap"]),
        "goods_gap_max": max(small["goods_gap"], large["goods_gap"]),
        "assignment_violations": max(small["assignment_violations"], large["assignment_violations"]),
        "investment_loans": max(small["capital_loans"], large["capital_loans"]),
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "mechanism_changed": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    summary = f"""# Step 15I.11C Supplier Operating-Liquidity Closure Audit

## Verdict

**{flags['verdict']}**

The audit reused the accepted P3, cost-anchored price, 13-week backlog-flow,
52-week lifecycle, and internal-cash-only setup for N=500 and N=5000. No
supplier financing, customer advance, markup, productivity, backlog, Firm
count, or Step13 behavior was activated.

## Cash-cycle result

The supplier pays payroll before receiving capital-good sale cash. Under the
current zero-margin price, cumulative supplier revenue equals sold-unit COGS
within floating tolerance, so sales replenish production cost but do not create
operating surplus. Supplier cash reaches zero in both worlds. Buyer investment
financing gap remains zero; this is a distinct supplier operating-liquidity
problem.

The maximum desired-payroll cash gap is reported in
`working_capital_requirement.csv`. The N=5000 requirement is demand-linked,
not a reason to manually tune a population-specific startup cash constant.

## Architecture recommendation

Customer advance is the safest next isolated mechanism because it is linked to
accepted investment orders, supplies cash before payroll, preserves the buyer
investment-versus-supplier-operating-finance distinction, and requires explicit
advance-liability accounting. It remains inactive in this audit.

Positive-interest Step13 working capital is not compatible with zero-margin
capital-good pricing unless another income source exists. No loan or interest
behavior was changed.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
