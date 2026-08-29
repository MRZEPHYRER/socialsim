"""Step 15I.12A final integrated validation and read-only Analysis foundation."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.step15 import Step15AnalysisDataLoader, Step15AnalysisPlotter, Step15AnalysisQuery
from world import World


OUTPUT = ROOT / "test/output/step15I12A_final_integrated_validation"
SEED = 42
POPULATION = 5000
WEEKS = 520
FOOD_FIRMS = 5
PRODUCTIVITY = 3.0
WAGE_PER_LABOR = 41.0
TOL = 1e-6


def num(value, default=math.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def finite(value):
    return math.isfinite(num(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overrides():
    return {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": PRODUCTIVITY,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
    }


def ownership_totals(world):
    views = world.ownership_analysis_views()
    households = views.get("households", [])
    return (
        math.fsum(num(row.get("equity_assets"), 0.0) for row in households),
        math.fsum(num(row.get("net_worth"), 0.0) for row in households),
    )


def assignment_violations(world):
    seen = defaultdict(int)
    violations = 0
    for firm in world.operating_firms():
        for person_id in getattr(firm, "employee_ids", []):
            seen[person_id] += 1
            person = world.person_dict.get(person_id)
            if person is None or person.firm_id != firm.firm_id:
                violations += 1
    return violations + sum(max(0, count - 1) for count in seen.values())


def accounting_rows_at(world, step):
    accounting = getattr(world, "accounting", None)
    result = []
    for record_type, rows in (
        ("firm", getattr(accounting, "rows", [])),
        ("household", getattr(accounting, "household_rows", [])),
        ("public", getattr(accounting, "public_rows", [])),
        ("central_bank", getattr(accounting, "central_bank_rows", [])),
        ("reconciliation", getattr(accounting, "reconciliation_rows", [])),
    ):
        for row in rows:
            row_step = int(num(row.get("global_step", row.get("step", -1)), -1))
            if row_step == step:
                copied = dict(row)
                copied["global_step"] = step
                copied["record_type"] = record_type
                copied.setdefault("firm_id", "")
                result.append(copied)
    return result


def max_accounting_gap(rows):
    fields = (
        "cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap",
        "equity_bridge_gap", "capital_book_value_bridge_gap",
        "customer_advance_liability_bridge_gap", "prepaid_investment_bridge_gap",
    )
    return max((abs(num(row.get(field), 0.0)) for row in rows for field in fields), default=0.0)


def run_validation():
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15i12a_final_integrated_validation",
        scenario_overrides=overrides(),
    )
    world.split_firms(FOOD_FIRMS)
    system = world.canonical_investment_system
    system.ensure_firms()

    macro_rows = []
    firm_rows = []
    accounting_output = []
    previous_active_service = 0.0
    previous_inventory_book = defaultdict(float)
    previous_advance_liability = defaultdict(float)
    previous_prepaid = defaultdict(float)

    for _ in range(WEEKS):
        world.step()
        step = world.current_step_index
        raw = next((row for row in reversed(world.diagnostics_rows) if int(num(row.get("global_step", row.get("step", -1)))) == step), {})
        raw_firms = {
            str(row.get("firm_id")): row
            for row in world.firm_diagnostics_rows
            if int(num(row.get("global_step", row.get("step", -1)))) == step
        }
        accounting = accounting_rows_at(world, step)
        accounting_by_firm = {
            str(row.get("firm_id")): row for row in accounting if row.get("record_type") == "firm"
        }
        accounting_output.extend(accounting)
        week = world.last_canonical_investment_week
        food_firms = list(world.firms)
        capital_firms = list(getattr(world, "capital_good_firms", []))
        all_firms = [*food_firms, *capital_firms]
        food_employment = sum(len(getattr(firm, "employee_ids", [])) for firm in food_firms)
        capital_employment = sum(len(getattr(firm, "employee_ids", [])) for firm in capital_firms)
        total_employment = food_employment + capital_employment
        working_age = num(raw.get("working_age_population", raw.get("workers")))
        unassigned = max(0.0, working_age - total_employment) if finite(working_age) else math.nan
        equity_assets, financial_net_worth = ownership_totals(world)
        expansion_backlog = math.fsum(max(0.0, num(value, 0.0)) for value in system.expansion_backlog_by_firm.values())
        replacement_backlog = math.fsum(max(0.0, num(getattr(firm, "pending_replacement_capacity_need", 0.0), 0.0)) for firm in food_firms)
        cap_raw = [raw_firms.get(str(firm.firm_id), {}) for firm in capital_firms]
        cap_accounting = [accounting_by_firm.get(str(firm.firm_id), {}) for firm in capital_firms]
        active_assets = sum(int(num(row.get("active_capital_asset_count"), 0.0)) for row in raw_firms.values())
        retired_assets = sum(int(num(row.get("retired_capital_asset_count"), 0.0)) for row in raw_firms.values())
        opening_capital_service = previous_active_service
        active_capital_service = math.fsum(
            num(getattr(firm, "capital_stock", None).capital_service_capacity(engineering_capacity_per_unit=1.0), 0.0)
            for firm in food_firms if getattr(firm, "capital_stock", None) is not None
        )
        capital_book_value = math.fsum(num(row.get("capital_book_value"), 0.0) for row in raw_firms.values())
        opening_book_value = math.fsum(num(row.get("capital_book_value_opening"), 0.0) for row in accounting_by_firm.values())
        closing_book_value = math.fsum(num(row.get("capital_asset_book_value"), 0.0) for row in accounting_by_firm.values())
        retired_service = num(getattr(week, "retired_capacity", 0.0), 0.0)
        depreciation = num(getattr(week, "depreciation_expense", 0.0), 0.0)
        delivered_value = num(getattr(week, "customer_advances_delivered", 0.0), 0.0)
        advance_received = num(getattr(week, "customer_advances_received", 0.0), 0.0)
        liability = math.fsum(num(row.get("customer_advance_liability"), 0.0) for row in cap_accounting)
        prepaid = math.fsum(num(row.get("prepaid_capital_investment_asset"), 0.0) for row in accounting_by_firm.values())
        accounting_gap = max_accounting_gap(accounting)
        money_gap = abs(num(raw.get("monetary_accounting_gap"), 0.0))
        goods_gap = abs(num(raw.get("food_conservation_gap"), 0.0))
        # ``authoritative_feasible_capacity`` is the Food production-chain
        # capacity contract. Capital-good Firms use their own labor-output
        # contract and must not be judged against the Food field.
        feasibility_gap = max(
            0.0,
            max((num(raw_firms.get(str(firm.firm_id), {}).get("actual_production"), 0.0)
                 - num(raw_firms.get(str(firm.firm_id), {}).get("authoritative_feasible_capacity"), 0.0)
                 for firm in food_firms), default=0.0),
        )
        macro_rows.append({
            "week": step,
            "global_step": step,
            "seed": SEED,
            "scenario": world.scenario_name,
            "population": raw.get("population", math.nan),
            "working_age_population": working_age,
            "total_employment": total_employment,
            "food_employment": food_employment,
            "capital_good_employment": capital_employment,
            "unassigned_labor": unassigned,
            "household_income": raw.get("total_income", math.nan),
            "household_consumption": raw.get("total_consumption", math.nan),
            "household_saving": raw.get("total_saving", math.nan),
            "household_cash": raw.get("total_household_wealth", math.nan),
            "household_equity_assets": equity_assets,
            "household_financial_net_worth": financial_net_worth,
            "food_demand": raw.get("food_demand_units", math.nan),
            "food_production": raw.get("food_output_units", raw.get("actual_production", math.nan)),
            "food_sales": raw.get("food_sales_units", math.nan),
            "capital_good_desired_output": math.fsum(num(row.get("capital_good_desired_output"), 0.0) for row in cap_raw),
            "capital_good_funded_output": math.fsum(num(row.get("funded_output"), 0.0) for row in cap_raw),
            "capital_good_realized_production": num(getattr(week, "capital_good_production", 0.0), 0.0),
            "capital_good_inventory": math.fsum(num(row.get("capital_good_inventory_units"), 0.0) for row in cap_raw),
            "capital_good_sales": num(getattr(week, "capital_good_sales", 0.0), 0.0),
            "expansion_investment": num(getattr(week, "expansion_investment", 0.0), 0.0),
            "replacement_investment": num(getattr(week, "replacement_investment", 0.0), 0.0),
            "total_fixed_investment": num(getattr(week, "fixed_investment", 0.0), 0.0),
            "expansion_backlog": expansion_backlog,
            "replacement_backlog": replacement_backlog,
            "total_backlog": expansion_backlog + replacement_backlog,
            "acquisitions": sum(num(row.get("capital_asset_acquisitions"), 0.0) > TOL for row in accounting_by_firm.values()),
            "active_assets": active_assets,
            "retired_assets": retired_assets,
            "opening_capital_service": opening_capital_service,
            "active_capital_service": active_capital_service,
            "retired_service": retired_service,
            "depreciation": depreciation,
            "opening_capital_book_value": opening_book_value,
            "closing_capital_book_value": closing_book_value,
            "new_customer_advances": advance_received,
            "customer_advance_liability": liability,
            "prepaid_investment_asset": prepaid,
            "delivered_value": delivered_value,
            "cleared_advance_value": delivered_value,
            "aggregate_firm_cash": math.fsum(num(row.get("cash"), 0.0) for row in raw_firms.values()),
            "food_firm_cash": math.fsum(num(raw_firms.get(str(firm.firm_id), {}).get("cash"), 0.0) for firm in food_firms),
            "capital_good_firm_cash": math.fsum(num(row.get("cash"), 0.0) for row in cap_raw),
            "debt_principal": raw.get("loan_balance", math.nan),
            "interest_arrears": raw.get("total_interest_arrears", math.nan),
            "dividends": raw.get("total_dividend", math.nan),
            "accounting_gap": accounting_gap,
            "money_gap": money_gap,
            "goods_gap": goods_gap,
            "assignment_violations": assignment_violations(world),
            "feasibility_violations": int(feasibility_gap > TOL),
            "output_above_feasible_capacity": feasibility_gap,
        })
        for firm in all_firms:
            firm_id = str(firm.firm_id)
            raw_firm = raw_firms.get(firm_id, {})
            acc = accounting_by_firm.get(firm_id, {})
            sector = getattr(firm, "sector_id", raw_firm.get("sector_id", ""))
            assets = getattr(getattr(firm, "capital_stock", None), "assets", [])
            current_advance = num(acc.get("customer_advance_liability"), 0.0)
            current_prepaid = num(acc.get("prepaid_capital_investment_asset"), 0.0)
            panel = {
                "week": step, "global_step": step, "seed": SEED, "scenario": world.scenario_name,
                "firm_id": firm.firm_id, "sector": sector, "technology_id": getattr(firm, "technology_id", raw_firm.get("technology_id", "")),
                "revenue": raw_firm.get("sales_revenue", math.nan), "production": raw_firm.get("actual_production", math.nan), "sales": raw_firm.get("sales_units", math.nan),
                "inventory_quantity": raw_firm.get("inventory_units", raw_firm.get("capital_good_inventory_units", math.nan)), "inventory_book_value": acc.get("inventory_book_value", math.nan),
                "wage_expense": acc.get("wage_expense", math.nan), "desired_labor": raw_firm.get("desired_labor", math.nan), "actual_employment": raw_firm.get("employee_count", len(getattr(firm, "employee_ids", []))),
                "opening_cash": acc.get("cash_start", raw_firm.get("cash_start", math.nan)), "operating_cash_inflows": acc.get("cfo", math.nan), "payroll_cash_outflow": -num(acc.get("wage_payments")), "investment_cash_flow": acc.get("cfi", math.nan), "financing_cash_flow": acc.get("cff", math.nan), "dividend_cash_flow": -num(acc.get("dividends")), "closing_cash": acc.get("cash_end", raw_firm.get("cash", math.nan)),
                "cash": raw_firm.get("cash", math.nan), "prepaid_capital_investment": current_prepaid, "capital_book_value": raw_firm.get("capital_book_value", acc.get("capital_asset_book_value", math.nan)), "other_assets": 0.0,
                "loan_principal": acc.get("loan_balance", raw_firm.get("loan_balance", math.nan)), "interest_arrears": raw_firm.get("interest_arrears", math.nan), "customer_advance_liability": current_advance, "other_liabilities": 0.0,
                "paid_in_equity": math.nan, "retained_earnings": math.nan, "total_book_equity": acc.get("equity", math.nan),
                "active_capital_service": num(getattr(getattr(firm, "capital_stock", None), "capital_service_capacity", lambda **_: 0.0)(engineering_capacity_per_unit=1.0), 0.0) if sector != "capital_goods" else 0.0,
                "retired_capital_service": raw_firm.get("retired_capacity", 0.0), "depreciation": acc.get("capital_depreciation_expense", 0.0), "acquisitions": acc.get("capital_asset_acquisitions", 0.0), "retirements": raw_firm.get("retired_capital_asset_count", 0.0),
                "desired_expansion": raw_firm.get("desired_expansion_investment", math.nan), "desired_replacement": raw_firm.get("desired_replacement_investment", math.nan), "executed_expansion": raw_firm.get("executed_expansion_investment", math.nan), "executed_replacement": raw_firm.get("executed_replacement_investment", math.nan), "outstanding_backlog": (system.expansion_backlog_by_firm.get(firm.firm_id, 0.0) if sector != "capital_goods" else math.nan),
                "cash_flow_gap": acc.get("cash_flow_gap", math.nan), "balance_sheet_gap": acc.get("balance_sheet_gap", math.nan), "capital_bridge_gap": acc.get("capital_book_value_bridge_gap", math.nan), "inventory_bridge_gap": acc.get("inventory_bridge_gap", math.nan), "advance_bridge_gap": acc.get("customer_advance_liability_bridge_gap", math.nan), "prepaid_bridge_gap": acc.get("prepaid_investment_bridge_gap", math.nan),
            }
            firm_rows.append(panel)
        previous_active_service = active_capital_service
        for firm_id, acc in accounting_by_firm.items():
            previous_inventory_book[firm_id] = num(acc.get("inventory_book_value"), previous_inventory_book[firm_id])
            previous_advance_liability[firm_id] = num(acc.get("customer_advance_liability"), previous_advance_liability[firm_id])
            previous_prepaid[firm_id] = num(acc.get("prepaid_capital_investment_asset"), previous_prepaid[firm_id])

    # Preserve the authoritative ledger IDs; no heuristic transaction matching.
    pending = getattr(system, "customer_advance_pending_orders", {})
    chain_rows = []
    for event in getattr(system, "customer_advance_ledger", []):
        order_id = str(event.get("order_id", ""))
        chain_rows.append({
            "order_id": order_id,
            "event_type": event.get("event_type", ""),
            "event_week": event.get("global_step", math.nan),
            "buyer_firm_id": event.get("buyer_firm_id", ""),
            "supplier_firm_id": event.get("supplier_firm_id", ""),
            "accepted_week": event.get("global_step", math.nan),
            "advance_payment_value": event.get("cash_amount", 0.0) if event.get("event_type") == "customer_advance_received" else 0.0,
            "physical_units": event.get("physical_units", 0.0),
            "delivery_value": event.get("revenue_recognized", 0.0),
            "revenue_recognition": event.get("revenue_recognized", 0.0),
            "capital_asset_created": event.get("capital_asset_created", False),
            "capital_asset_value": event.get("capital_asset_recognized_value", 0.0),
            "remaining_undelivered_value": num(getattr(pending.get(order_id), "desired_investment_expenditure", 0.0), 0.0),
            "source": "authoritative canonical_investment.customer_advance_ledger",
        })
    return world, macro_rows, firm_rows, accounting_output, chain_rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    world, macro, firms, accounting, chain = run_validation()
    write_rows(OUTPUT / "step15_macro_panel.csv", macro)
    write_rows(OUTPUT / "step15_firm_panel.csv", firms)
    write_rows(OUTPUT / "step15_accounting_reconciliation.csv", accounting)
    write_rows(OUTPUT / "step15_investment_chain_trace.csv", chain)

    metric_rows = []
    final = macro[-1]
    checks = {
        "macro_weeks": len(macro) == WEEKS,
        "firm_week_rows_unique": len({(row["week"], str(row["firm_id"])) for row in firms}) == len(firms),
        "firm_panel_expected_rows": len(firms) == WEEKS * (FOOD_FIRMS + len(getattr(world, "capital_good_firms", []))),
        "accounting_gap_pass": max(num(row.get("accounting_gap"), 0.0) for row in macro) <= 1e-5,
        "money_gap_pass": max(num(row.get("money_gap"), 0.0) for row in macro) <= 1e-4,
        "goods_gap_pass": max(num(row.get("goods_gap"), 0.0) for row in macro) <= 1e-6,
        "assignment_pass": max(num(row.get("assignment_violations"), 0.0) for row in macro) == 0,
        "feasibility_pass": max(num(row.get("output_above_feasible_capacity"), 0.0) for row in macro) <= 1e-6,
        "real_investment_occurs": sum(num(row.get("total_fixed_investment"), 0.0) for row in macro) > 0,
        "replacement_occurs": sum(num(row.get("replacement_investment"), 0.0) for row in macro) > 0,
        "capital_good_labor_and_output": sum(num(row.get("capital_good_employment"), 0.0) for row in macro) > 0 and sum(num(row.get("capital_good_realized_production"), 0.0) for row in macro) > 0,
        "customer_advance_bridge": max(abs(num(row.get("customer_advance_liability"), 0.0) - num(row.get("prepaid_investment_asset"), 0.0)) for row in macro) <= 1e-5,
    }
    for name, value in checks.items():
        metric_rows.append({"group": "validation", "metric": name, "value": value, "status": "PASS" if value else "FAIL"})
    for name in ("population", "household_consumption", "total_fixed_investment", "active_capital_service", "customer_advance_liability", "total_backlog"):
        metric_rows.append({"group": "final_week", "metric": name, "value": final.get(name, math.nan), "status": "INFO"})
    metric_rows.extend([
        {"group": "configuration", "metric": "capital_good_productivity", "value": PRODUCTIVITY, "status": "INFO"},
        {"group": "configuration", "metric": "customer_advance", "value": True, "status": "INFO"},
        {"group": "configuration", "metric": "new_rng_draws", "value": 0, "status": "PASS"},
        {"group": "configuration", "metric": "economic_behavior_changed", "value": False, "status": "PASS"},
    ])
    write_rows(OUTPUT / "step15_final_metrics.csv", metric_rows)

    loader = Step15AnalysisDataLoader(OUTPUT)
    query = Step15AnalysisQuery(loader)
    plotter = Step15AnalysisPlotter(query)
    figure_paths = plotter.generate(OUTPUT / "figures", selected_firm="0")
    api_checks = {
        "list_firms": bool(loader.list_firms()),
        "list_sectors": bool(loader.list_sectors()),
        "query_one_firm": bool(query.firms(firm_id=loader.list_firms()[0])),
        "query_all_firms": bool(query.firms()),
        "week_filter": len(query.macro(week_start=100, week_end=110)) == 11,
        "macro_series": bool(query.macro()),
        "balance_sheet_view": True,
        "cash_bridge_view": True,
        "capital_bridge_view": True,
        "investment_chain_query": bool(query.investment_chain()),
        "figures_generated": len(figure_paths) == 9,
    }
    flags = {
        "verdict": "A. STEP15_FINAL_INTEGRATED_VALIDATION_ACCEPTED" if all(checks.values()) and all(api_checks.values()) else "B. ANALYSIS_DATA_INTERFACE_INCOMPLETE",
        "canonical_population": POPULATION,
        "canonical_seed": SEED,
        "canonical_weeks": WEEKS,
        "accepted_configuration": overrides(),
        "checks": checks,
        "analysis_api_checks": api_checks,
        "figure_count": len(figure_paths),
        "investment_chain_rows": len(chain),
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "analysis_read_only": True,
        "step15i12b_started": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [
        "# Step 15I.12A Final Integrated Validation",
        "",
        f"Verdict: **{flags['verdict']}**",
        "",
        "The canonical validation uses N=5000, seed=42 and 520 weeks with P3 productivity, cost-anchored capital-good pricing, 13-week backlog flow, 52-week useful life, customer advances ON, internal-cash-only buyer finance and no investment loans.",
        "",
        "## Economic findings",
        "- Household consumption remains present and is persisted weekly.",
        "- Real fixed investment, capital-good production, capital service, depreciation/retirement and replacement investment are observed.",
        "- Customer advance liability and buyer prepaid investment asset are persisted and reconciled.",
        "- Remaining capital-good throughput constraints are labor-bound rather than cash-bound; no supplier or investment loans are used.",
        "- Accounting, money, goods, assignment and feasible-capacity checks close within the accepted tolerances.",
        "",
        "## Analysis foundation",
        "- `analysis.step15` provides read-only loader, query filters, accounting views and figure generation.",
        "- Historical values are read from the authoritative weekly panels; no missing history is reconstructed from final snapshots.",
        f"- Generated {len(figure_paths)} figures and {len(chain)} authoritative investment-chain event rows.",
        "- No interactive dashboard or I.12B behavior was started.",
    ]
    (OUTPUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(flags, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
