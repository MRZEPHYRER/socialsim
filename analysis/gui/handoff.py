"""Read-only adapter from Analysis v2 tables to the windowed workbench."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def _write(path, rows, fields=None):
    inferred = []
    for row in rows:
        for field in row:
            if field not in inferred:
                inferred.append(field)
    fields = list(fields or inferred)
    for field in inferred:
        if field not in fields:
            fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["week"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_gui_compatibility_panels(context, output_dir):
    """Persist a presentation adapter; never mutate simulation state.

    Main runs historically produced Analysis v2 tables while the Step 15 GUI
    consumed Step 15 panel filenames.  These lightweight views bridge only
    those schemas and keep the actual run directory authoritative.
    """
    output_dir = Path(output_dir)
    def read_csv(path):
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    macro_source = list(context.tables.get("weekly_macro", []))
    firm_source = list(context.tables.get("weekly_firm", []))
    raw_firms = list(getattr(context.world, "firm_diagnostics_rows", []))
    if not raw_firms:
        raw_firms = list(getattr(context, "firm_diagnostics", []))
    raw_index = {
        (str(row.get("step", row.get("global_step", 0))), str(row.get("firm_id"))): row
        for row in raw_firms
    }
    by_step_sector = defaultdict(lambda: defaultdict(float))
    firm_rows = []
    for row in firm_source:
        step = row.get("step", 0)
        fid = str(row.get("firm_id"))
        raw = raw_index.get((str(step), fid), {})
        sector = raw.get("sector", raw.get("sector_id", "food"))
        by_step_sector[str(step)][sector] += float(row.get("employee_count") or 0.0)
        firm_rows.append({
            "week": step,
            "global_step": step,
            "firm_id": fid,
            "sector": sector,
            "technology_id": raw.get("technology_id", "UNAVAILABLE"),
            "revenue": raw.get("sales_revenue", row.get("revenue")),
            "production": raw.get("production", row.get("production")),
            "sales": raw.get("sales_units", row.get("sales")),
            "inventory_quantity": raw.get("inventory_units", row.get("inventory")),
            "cash": raw.get("cash", row.get("cash")),
            "profit": raw.get("profit", row.get("profit")),
            "actual_employment": raw.get("employee_count", row.get("employee_count")),
            "desired_labor": raw.get("desired_labor", "UNAVAILABLE"),
            "loan_principal": raw.get("closing_principal", row.get("closing_principal")),
            "customer_advance_liability": raw.get("customer_advance_liability", "UNAVAILABLE"),
            "investment_expenditure": raw.get("investment_expenditure", "UNAVAILABLE"),
            "expansion_investment": raw.get("executed_expansion_investment", "UNAVAILABLE"),
            "replacement_investment": raw.get("executed_replacement_investment", "UNAVAILABLE"),
            "active_capital_service": raw.get("active_capital_service", "UNAVAILABLE"),
            "capital_book_value": raw.get("capital_book_value", "UNAVAILABLE"),
            "depreciation": raw.get("capital_depreciation_expense", "UNAVAILABLE"),
            "retirements": raw.get("retired_capital_asset_count", "UNAVAILABLE"),
            "acquisitions": raw.get("capital_asset_count", "UNAVAILABLE"),
            "outstanding_backlog": raw.get("outstanding_backlog", "UNAVAILABLE"),
        })

    canonical_macro = {
        str(row.get("global_step", row.get("step", 0))): row
        for row in getattr(context.world, "diagnostics_rows", [])
    }
    canonical_firms_by_step = defaultdict(list)
    for row in raw_firms:
        canonical_firms_by_step[str(row.get("global_step", row.get("step", 0)))].append(row)

    def number(row, field, default=0.0):
        try:
            value = row.get(field, default)
            if value in (None, "", "NA", "UNAVAILABLE"):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    macro_rows = []
    for row in macro_source:
        step = row.get("step", row.get("simulation_week", 0))
        raw_macro = canonical_macro.get(str(step), {})
        step_firms = canonical_firms_by_step.get(str(step), [])
        capital_firms = [item for item in step_firms if item.get("sector_id") == "capital_goods"]
        food_firms = [item for item in step_firms if item.get("sector_id") == "food"]
        income = row.get("executed_payroll", row.get("household_wealth", "UNAVAILABLE"))
        consumption = row.get("consumption", "UNAVAILABLE")
        try:
            saving = float(income) - float(consumption)
        except (TypeError, ValueError):
            saving = "UNAVAILABLE"
        macro_rows.append({
            "week": step,
            "global_step": step,
            "scenario": getattr(context.world, "scenario_name", "baseline"),
            "seed": getattr(context.world, "seed", "UNAVAILABLE"),
            "population": row.get("population"),
            "working_age_population": row.get("working_age_population"),
            "total_employment": row.get("employment"),
            "food_employment": by_step_sector[str(step)].get("food", 0.0),
            "capital_good_employment": by_step_sector[str(step)].get("capital_goods", 0.0),
            "unassigned_labor": row.get("unassigned_labor", "UNAVAILABLE"),
            "household_income": raw_macro.get("wage_bill", income),
            "household_consumption": raw_macro.get("total_consumption", consumption),
            "household_saving": raw_macro.get("total_saving", saving),
            "household_cash": row.get("household_money", row.get("household_wealth")),
            "household_financial_net_worth": row.get("household_wealth"),
            "food_production": row.get("production"),
            "food_demand": row.get("consumption"),
            "capital_good_desired_output": sum(number(item, "desired_production") for item in capital_firms),
            "capital_good_realized_production": sum(number(item, "capital_good_production_units") for item in capital_firms),
            "total_fixed_investment": sum(number(item, "investment_expenditure") for item in food_firms),
            "aggregate_firm_cash": row.get("aggregate_firm_cash"),
            "debt_principal": row.get("principal"),
            "active_capital_service": sum(number(item, "active_capital_service") for item in food_firms),
            "total_backlog": sum(number(item, "outstanding_backlog") for item in food_firms),
            "expansion_investment": sum(number(item, "executed_expansion_investment") for item in food_firms),
            "replacement_investment": sum(number(item, "executed_replacement_investment") for item in food_firms),
            "active_assets": sum(number(item, "active_capital_asset_count") for item in food_firms),
            "retired_assets": sum(number(item, "retired_capital_asset_count") for item in food_firms),
            "depreciation": sum(number(item, "capital_depreciation_expense") for item in food_firms),
            "capital_good_employment": sum(number(item, "employee_count") for item in capital_firms),
            "customer_advance_liability": sum(number(item, "customer_advance_liability") for item in capital_firms),
            "prepaid_investment_asset": sum(number(item, "prepaid_capital_investment_asset") for item in food_firms),
            "accounting_gap": row.get("monetary_accounting_gap", "UNAVAILABLE"),
            "money_gap": row.get("money_delta_gap", "UNAVAILABLE"),
            "goods_gap": row.get("goods_conservation_gap", "UNAVAILABLE"),
        })
    _write(output_dir / "step15_macro_panel.csv", macro_rows)
    _write(output_dir / "step15_firm_panel.csv", firm_rows)
    _write(output_dir / "step15_accounting_reconciliation.csv", context.tables.get("accounting_weekly", []))

    # The raw diagnostics stream already contains authoritative demographic
    # state. Keep this handoff separate from the economic macro panel so the
    # social GUI never reconstructs age history from a final snapshot.
    raw_demography = read_csv(output_dir / "diagnostics.csv")
    demographic_fields = (
        "global_step", "simulation_week", "simulation_year", "population",
        "births", "deaths", "average_age_years", "age_0_19_count",
        "age_20_39_count", "age_40_64_count", "age_65_plus_count",
        "age_0_19_share", "age_20_39_share", "age_40_64_share",
        "age_65_plus_share", "working_age_population", "elderly_population",
        "households", "active_households", "average_household_size",
        "married_households", "single_parent_households", "empty_households",
        "marriage_market_executed", "matches_formed",
    )
    _write(
        output_dir / "step15_demographic_panel.csv",
        raw_demography,
        fields=demographic_fields,
    )
    for source, target in (
        ("demographic_events.csv", "step15_demographic_events.csv"),
        ("marriage_market_diagnostics.csv", "step15_marriage_panel.csv"),
    ):
        source_path = output_dir / source
        if source_path.exists():
            (output_dir / target).write_bytes(source_path.read_bytes())
    provenance = list(getattr(context.world, "capital_asset_event_rows", []))
    investment_system = getattr(context.world, "canonical_investment_system", None)
    advance_events = list(getattr(investment_system, "customer_advance_ledger", []))
    provenance = [*advance_events, *provenance]
    chain_rows = []
    for row in provenance:
        chain_rows.append({
            "event_week": row.get("global_step", "UNAVAILABLE"),
            "event_type": row.get("event_type", "UNAVAILABLE"),
            "order_id": row.get("order_id", "UNAVAILABLE"),
            "buyer_firm_id": row.get("buyer_firm_id", row.get("owner_firm_id", "UNAVAILABLE")),
            "supplier_firm_id": row.get("supplier_firm_id", "UNAVAILABLE"),
            "investment_source": row.get("investment_source", "UNAVAILABLE"),
            "replacement_trigger_id": row.get("replacement_trigger_id", "UNAVAILABLE"),
            "asset_id": row.get("asset_id", "UNAVAILABLE"),
            "capital_asset_created": row.get("capital_asset_created", False),
            "physical_units": row.get("physical_asset_units", row.get("settled_units", "UNAVAILABLE")),
            "delivery_value": row.get("settled_expenditure", row.get("acquisition_cost", "UNAVAILABLE")),
            "advance_payment_value": row.get("cash_amount", "UNAVAILABLE"),
            "revenue_recognition": row.get("revenue_recognized", "UNAVAILABLE"),
        })
    _write(
        output_dir / "step15_investment_chain_trace.csv",
        chain_rows,
        fields=("event_week", "event_type", "order_id", "buyer_firm_id", "supplier_firm_id", "investment_source", "replacement_trigger_id", "asset_id", "capital_asset_created", "physical_units", "delivery_value", "advance_payment_value", "revenue_recognition"),
    )
    asset_ledger = output_dir / "capital_provenance" / "capital_asset_ledger.csv"
    if asset_ledger.exists():
        target = output_dir / "step15_capital_asset_ledger.csv"
        target.write_bytes(asset_ledger.read_bytes())
    provenance_path = output_dir / "capital_provenance" / "capital_provenance_events.csv"
    if provenance_path.exists():
        target = output_dir / "step15_capital_provenance_events.csv"
        target.write_bytes(provenance_path.read_bytes())
    _write(output_dir / "step15_final_metrics.csv", macro_rows[-1:] if macro_rows else [])
    return output_dir
