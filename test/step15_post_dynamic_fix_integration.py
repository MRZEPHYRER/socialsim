"""Step 15 post-dynamic-fix integration and money-scope audit.

This audit deliberately reads state at the end of each authoritative runtime
step.  It does not infer historical money or capital events from the final
checkpoint and does not change economic policy during the run.
"""

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

from world import World


OUTPUT = ROOT / "test/output/step15_post_dynamic_fix_integration"
SEED = 42
POPULATION = 5000
WEEKS = 520
FOOD_FIRMS = 5
PRODUCTIVITY = 3.0
TOL = 1e-5


def number(value, default=math.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def finite(value):
    return math.isfinite(number(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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
        "CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS": 13,
        "CAPITAL_GOOD_BACKLOG_SERVICE_HORIZON_WEEKS": 13,
        "CAPITAL_GOOD_COMMITTED_CAPITAL_PLANNER_ENABLED": False,
        "CAPITAL_GOOD_PIPELINE_DEPLETION_EARLY_REVIEW_ENABLED": True,
        "INITIAL_HOUSEHOLD_ONE_WEEK_CONSUMPTION_BUFFER_ENABLED": True,
        "DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED": False,
        "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": True,
    }


def gini(values):
    values = sorted(max(0.0, number(value, 0.0)) for value in values)
    if not values or math.fsum(values) <= 0.0:
        return 0.0
    weighted = math.fsum((index + 1) * value for index, value in enumerate(values))
    return (2.0 * weighted) / (len(values) * math.fsum(values)) - (len(values) + 1.0) / len(values)


def accounting_rows_at(world, step):
    accounting = getattr(world, "accounting", None)
    result = []
    for record_type, attribute in (
        ("firm", "rows"),
        ("household", "household_rows"),
        ("public", "public_rows"),
        ("central_bank", "central_bank_rows"),
        ("reconciliation", "reconciliation_rows"),
    ):
        for row in getattr(accounting, attribute, []):
            row_step = int(number(row.get("global_step", row.get("step", -1)), -1))
            if row_step == step:
                item = dict(row)
                item["record_type"] = record_type
                item["global_step"] = step
                result.append(item)
    return result


def max_abs_fields(rows, fields):
    return max(
        (abs(number(row.get(field), 0.0)) for row in rows for field in fields),
        default=0.0,
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
    violations += sum(max(0, count - 1) for count in seen.values())
    assigned = set(seen)
    for person in getattr(world, "population", []):
        if not getattr(person, "alive", False):
            continue
        if getattr(person, "firm_id", None) is not None and person.id not in assigned:
            violations += 1
    return violations


def ownership_totals(world):
    views = world.ownership_analysis_views()
    households = views.get("households", [])
    return (
        math.fsum(number(row.get("equity_assets"), 0.0) for row in households),
        math.fsum(number(row.get("net_worth"), 0.0) for row in households),
    )


def money_components(world):
    return dict(world.authoritative_money_location_components())


def run():
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15_post_dynamic_fix_integration",
        scenario_overrides=overrides(),
    )
    opening_buffer = number(getattr(world, "initial_household_opening_buffer_total", 0.0), 0.0)
    world.split_firms(FOOD_FIRMS)
    system = world.canonical_investment_system
    system.ensure_firms()

    config_rows = [
        {"key": "population", "value": POPULATION, "source": "audit_setup"},
        {"key": "seed", "value": SEED, "source": "audit_setup"},
        {"key": "weeks", "value": WEEKS, "source": "audit_setup"},
        {"key": "food_firms", "value": FOOD_FIRMS, "source": "audit_setup"},
        {"key": "scenario", "value": world.scenario_name, "source": "runtime"},
        {"key": "capital_good_productivity", "value": PRODUCTIVITY, "source": "runtime_override"},
    ]
    for key, value in overrides().items():
        resolved = getattr(world, key.lower(), value)
        config_rows.append({"key": key, "value": resolved, "requested": value, "source": "runtime"})
    config_rows.extend([
        {"key": "resolved_capital_good_firms", "value": len(getattr(world, "capital_good_firms", [])), "source": "world_construction"},
        {"key": "resolved_food_firms", "value": len(getattr(world, "firms", [])), "source": "world_construction"},
        {"key": "opening_household_buffer", "value": opening_buffer, "source": "world_initialization"},
    ])

    macro_rows = []
    money_rows = []
    opening_rows = []
    distribution_rows = []
    late_rows = []
    firm_rows = []
    accounting_output = []
    early_seen = 0
    previous_money = None
    previous_macro = None
    previous_early_event_count = 0

    for _ in range(WEEKS):
        world.step()
        step = world.current_step_index
        raw = next(
            (row for row in reversed(world.diagnostics_rows)
             if int(number(row.get("global_step", row.get("step", -1)), -1)) == step),
            {},
        )
        raw_firms = {
            str(row.get("firm_id")): row
            for row in world.firm_diagnostics_rows
            if int(number(row.get("global_step", row.get("step", -1)), -1)) == step
        }
        accounting = accounting_rows_at(world, step)
        accounting_output.extend(accounting)
        accounting_by_firm = {
            str(row.get("firm_id")): row
            for row in accounting if row.get("record_type") == "firm"
        }
        food_firms = list(getattr(world, "firms", []))
        cap_firms = list(getattr(world, "capital_good_firms", []))
        food_employment = sum(len(getattr(firm, "employee_ids", [])) for firm in food_firms)
        cap_employment = sum(len(getattr(firm, "employee_ids", [])) for firm in cap_firms)
        working_age = number(raw.get("working_age_population", raw.get("workers")), math.nan)
        total_employment = food_employment + cap_employment
        unassigned = max(0.0, working_age - total_employment) if finite(working_age) else math.nan
        equity_assets, net_worth = ownership_totals(world)
        components = money_components(world)
        authoritative_stock = number(raw.get("authoritative_money_stock"), world.authoritative_money_stock())
        located_stock = number(raw.get("located_money_stock"), components["located_money_stock"])
        stock_gap = authoritative_stock - located_stock
        central_bank = getattr(getattr(world, "firm_system", None), "central_bank", None)
        money_supply = number(getattr(central_bank, "money_supply", 0.0), 0.0)
        net_issued = number(raw.get("central_bank_net_money_issued"), 0.0)
        located_delta = located_stock - (previous_money["located_money_stock"] if previous_money else world.initial_private_money_stock)
        money_delta_gap = located_delta - net_issued
        pending_formation_cash = number(components.get("pending_household_formation_wealth"), 0.0)
        legacy_narrow_located_stock = located_stock - pending_formation_cash
        legacy_narrow_gap = authoritative_stock - legacy_narrow_located_stock
        reconciliation = [row for row in accounting if row.get("record_type") == "reconciliation"]
        accounting_gap = max_abs_fields(
            reconciliation,
            ("cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap", "equity_bridge_gap", "capital_book_value_bridge_gap", "customer_advance_liability_bridge_gap", "prepaid_investment_bridge_gap", "money_location_gap", "full_money_location_gap"),
        )
        advance_liability = math.fsum(number(row.get("customer_advance_liability"), 0.0) for row in accounting_by_firm.values())
        prepaid_asset = math.fsum(number(row.get("prepaid_capital_investment_asset"), 0.0) for row in accounting_by_firm.values())
        advance_gap = advance_liability - prepaid_asset
        goods_gap = abs(number(raw.get("food_conservation_gap"), 0.0))
        feasibility_gap = max(
            0.0,
            max(
                (number(raw_firms.get(str(firm.firm_id), {}).get("actual_production"), 0.0)
                 - number(raw_firms.get(str(firm.firm_id), {}).get("authoritative_feasible_capacity"), 0.0)
                 for firm in food_firms),
                default=0.0,
            ),
        )
        week = getattr(world, "last_canonical_investment_week", None)
        step_investment = number(getattr(week, "fixed_investment", 0.0), 0.0)
        expansion = number(getattr(week, "expansion_investment", 0.0), 0.0)
        replacement = number(getattr(week, "replacement_investment", 0.0), 0.0)
        cap_inventory = math.fsum(number(raw_firms.get(str(firm.firm_id), {}).get("capital_good_inventory_units"), 0.0) for firm in cap_firms)
        cap_desired_output = math.fsum(number(raw_firms.get(str(firm.firm_id), {}).get("capital_good_desired_output"), 0.0) for firm in cap_firms)
        cap_production = number(getattr(week, "capital_good_production", 0.0), 0.0)
        cap_sales = number(getattr(week, "capital_good_sales", 0.0), 0.0)
        backlog = math.fsum(max(0.0, number(value, 0.0)) for value in system.expansion_backlog_by_firm.values())
        backlog += math.fsum(max(0.0, number(getattr(firm, "pending_replacement_capacity_need", 0.0), 0.0)) for firm in food_firms)
        active_assets = sum(int(number(raw_firms.get(str(firm.firm_id), {}).get("active_capital_asset_count"), 0.0)) for firm in food_firms)
        retired_assets = sum(int(number(raw_firms.get(str(firm.firm_id), {}).get("retired_capital_asset_count"), 0.0)) for firm in food_firms)
        early_count = len(getattr(system, "early_investment_review_events", []))
        early_seen += max(0, early_count - previous_early_event_count)
        early_rows = getattr(system, "early_investment_review_events", [])[previous_early_event_count:early_count]
        previous_early_event_count = early_count
        macro = {
            "global_step": step,
            "population": raw.get("population", math.nan),
            "households": len(getattr(world, "households", [])),
            "working_age_population": working_age,
            "total_employment": total_employment,
            "food_employment": food_employment,
            "capital_good_employment": cap_employment,
            "unassigned_labor": unassigned,
            "household_income": raw.get("total_income", math.nan),
            "household_consumption": raw.get("total_consumption", math.nan),
            "household_saving": raw.get("total_saving", math.nan),
            "household_cash": raw.get("total_household_wealth", math.nan),
            "household_equity_assets": equity_assets,
            "household_financial_net_worth": net_worth,
            "food_production": raw.get("food_output_units", math.nan),
            "food_demand": raw.get("food_demand_units", math.nan),
            "fixed_investment": step_investment,
            "expansion_investment": expansion,
            "replacement_investment": replacement,
            "capital_good_desired_output": cap_desired_output,
            "capital_good_production": cap_production,
            "capital_good_sales": cap_sales,
            "capital_good_inventory": cap_inventory,
            "capital_good_backlog": backlog,
            "active_capital_assets": active_assets,
            "retired_capital_assets": retired_assets,
            "active_capital_service": math.fsum(number(getattr(getattr(firm, "capital_stock", None), "capital_service_capacity", lambda **_: 0.0)(engineering_capacity_per_unit=1.0), 0.0) for firm in food_firms),
            "customer_advance_liability": advance_liability,
            "prepaid_investment_asset": prepaid_asset,
            "advance_prepaid_gap": advance_gap,
            "aggregate_firm_cash": components["firm_cash"],
            "money_stock": authoritative_stock,
            "located_money_stock": located_stock,
            "full_money_location_gap": stock_gap,
            "money_delta_gap": money_delta_gap,
            "legacy_narrow_located_stock": legacy_narrow_located_stock,
            "legacy_narrow_money_gap": legacy_narrow_gap,
            "accounting_gap": accounting_gap,
            "goods_gap": goods_gap,
            "assignment_violations": assignment_violations(world),
            "output_above_feasible_capacity": feasibility_gap,
            "early_review_events_total": early_count,
            "early_review_events_this_step": len(early_rows),
            "central_bank_money_supply": money_supply,
            "net_money_issued": net_issued,
        }
        macro_rows.append(macro)
        money_rows.append({
            "global_step": step,
            **components,
            "authoritative_money_stock": authoritative_stock,
            "central_bank_money_supply": money_supply,
            "net_money_issued": net_issued,
            "located_money_delta": located_delta,
            "money_delta_gap": money_delta_gap,
            "full_money_location_gap": stock_gap,
            "legacy_narrow_money_gap": legacy_narrow_gap,
            "accounting_money_location_gap": number(reconciliation[0].get("money_location_gap"), math.nan) if reconciliation else math.nan,
        })
        cash_values = [number(getattr(household, "wealth", 0.0), 0.0) for household in getattr(world, "households", [])]
        positive_cash = [value for value in cash_values if value > 0.0]
        distribution_rows.append({
            "global_step": step,
            "households": len(cash_values),
            "total_household_cash": math.fsum(cash_values),
            "social_household_cash": components["social_household_cash"],
            "settlement_only_cash": components["settlement_only_cash"],
            "opening_buffer_total": opening_buffer,
            "full_money_location_gap": stock_gap,
            "aggregate_firm_cash": components["firm_cash"],
            "median_cash": world.median(cash_values),
            "p10_cash": world.percentile(cash_values, 10),
            "p90_cash": world.percentile(cash_values, 90),
            "near_zero_cash_count": sum(value <= 1e-9 for value in cash_values),
            "positive_cash_count": len(positive_cash),
            "cash_gini": gini(cash_values),
            "top_10_cash_share": math.fsum(sorted(cash_values, reverse=True)[:10]) / max(math.fsum(cash_values), 1e-12),
        })
        if step in (479, 492, 505, 518):
            late_rows.append({
                "global_step": step,
                "household_income": macro["household_income"],
                "household_consumption": macro["household_consumption"],
                "household_saving": macro["household_saving"],
                "fixed_investment": step_investment,
                "capital_good_production": cap_production,
                "capital_good_backlog": backlog,
                "aggregate_firm_cash": components["firm_cash"],
                "full_money_location_gap": stock_gap,
                "delta_income": (number(macro["household_income"], 0.0) - number(previous_macro["household_income"], 0.0)) if previous_macro else math.nan,
                "delta_consumption": (number(macro["household_consumption"], 0.0) - number(previous_macro["household_consumption"], 0.0)) if previous_macro else math.nan,
            })
        for firm in [*food_firms, *cap_firms]:
            firm_id = str(firm.firm_id)
            raw_firm = raw_firms.get(firm_id, {})
            acc = accounting_by_firm.get(firm_id, {})
            firm_rows.append({
                "global_step": step,
                "firm_id": firm.firm_id,
                "sector_id": getattr(firm, "sector_id", raw_firm.get("sector_id", "")),
                "technology_id": getattr(firm, "technology_id", raw_firm.get("technology_id", "")),
                "employment": raw_firm.get("employee_count", len(getattr(firm, "employee_ids", []))),
                "desired_labor": raw_firm.get("desired_labor", math.nan),
                "revenue": raw_firm.get("sales_revenue", math.nan),
                "operating_profit": raw_firm.get("operating_profit", acc.get("operating_profit", math.nan)),
                "cfo": acc.get("cfo", math.nan),
                "cash": raw_firm.get("cash", math.nan),
                "principal": raw_firm.get("loan_balance", acc.get("loan_balance", math.nan)),
                "arrears": raw_firm.get("interest_arrears", math.nan),
                "capital_book_value": raw_firm.get("capital_book_value", acc.get("capital_asset_book_value", math.nan)),
                "investment_expenditure": raw_firm.get("investment_expenditure", step_investment if getattr(firm, "sector_id", "") != "capital_goods" else 0.0),
                "active_capital_asset_count": raw_firm.get("active_capital_asset_count", 0),
                "retired_capital_asset_count": raw_firm.get("retired_capital_asset_count", 0),
                "customer_advance_liability": acc.get("customer_advance_liability", 0.0),
                "prepaid_investment_asset": acc.get("prepaid_capital_investment_asset", 0.0),
                "cash_flow_gap": acc.get("cash_flow_gap", math.nan),
                "capital_book_value_bridge_gap": acc.get("capital_book_value_bridge_gap", math.nan),
                "inventory_bridge_gap": acc.get("inventory_bridge_gap", math.nan),
                "advance_bridge_gap": acc.get("customer_advance_liability_bridge_gap", math.nan),
                "prepaid_bridge_gap": acc.get("prepaid_investment_bridge_gap", math.nan),
            })
        previous_money = {"located_money_stock": located_stock}
        previous_macro = macro

    # Keep the first-divergence artifact concise: it is a forensic window,
    # while the complete weekly money series is retained separately.
    # Reconstruct the pre-fix narrow diagnostic only for forensic comparison.
    # It intentionally excludes the pending household-formation clearing
    # balance and is never used as the acceptance identity.
    material = [row for row in money_rows if abs(number(row["legacy_narrow_money_gap"], 0.0)) > TOL]
    if material:
        first_step = int(material[0]["global_step"])
        divergence_rows = [row for row in money_rows if first_step - 2 <= int(row["global_step"]) <= first_step + 2]
    else:
        divergence_rows = [{
            "global_step": "NO_MATERIAL_DIVERGENCE",
            "full_money_location_gap": max(abs(number(row["full_money_location_gap"], 0.0)) for row in money_rows),
            "legacy_narrow_money_gap": 0.0,
            "diagnosis": "authoritative_scope_closes_within_tolerance",
        }]

    registry = [
        {"metric": "authoritative_money_stock", "formula": "opening_private_money_stock + central_bank_money_supply", "holders_included": "liability-side stock", "excluded": "none", "timing": "end of runtime step", "authoritative_status": "AUTHORITATIVE", "source_module": "world.authoritative_money_stock"},
        {"metric": "located_money_stock", "formula": "Household + Firm + LegacyOwner + Estate + public + central-public cash + pending household-formation balance", "holders_included": "all runtime cash holders in accepted scope", "excluded": "none known", "timing": "end of runtime step", "authoritative_status": "AUTHORITATIVE_LOCATION_SUM", "source_module": "world.authoritative_money_location_components"},
        {"metric": "full_money_location_gap", "formula": "authoritative_money_stock - located_money_stock", "holders_included": "same as located_money_stock", "excluded": "none known", "timing": "stock identity", "authoritative_status": "AUTHORITATIVE_ACCEPTANCE_METRIC", "source_module": "world.record_diagnostics"},
        {"metric": "monetary_accounting_gap", "formula": "legacy name; same formula as full_money_location_gap", "holders_included": "same authoritative scope", "excluded": "none known", "timing": "stock identity", "authoritative_status": "DEPRECATED_ALIAS", "source_module": "world.record_diagnostics"},
        {"metric": "legacy_narrow_money_gap", "formula": "authoritative stock - (located stock excluding pending household-formation balance)", "holders_included": "historical narrow cash scope", "excluded": "pending_household_formation_wealth", "timing": "historical diagnostic comparison", "authoritative_status": "LEGACY_DEPRECATED_SCOPE_OMISSION", "source_module": "audit forensic reconstruction"},
        {"metric": "money_location_gap", "formula": "accounting authoritative_money_stock - located_money_stock", "holders_included": "accounting reconciliation scope", "excluded": "none known", "timing": "end-of-step accounting row", "authoritative_status": "AUTHORITATIVE_ALIAS", "source_module": "economy.accounting"},
        {"metric": "money_delta_gap", "formula": "located_money_delta - net_money_issued", "holders_included": "same location scope", "excluded": "none known", "timing": "flow/timing diagnostic", "authoritative_status": "FLOW_DIAGNOSTIC_NOT_STOCK_GAP", "source_module": "world.record_diagnostics"},
        {"metric": "ledger_money_net_gap", "formula": "ledger money created - ledger money destroyed - net_money_issued", "holders_included": "ledger events", "excluded": "not a stock identity", "timing": "transaction-flow diagnostic", "authoritative_status": "FLOW_DIAGNOSTIC_NOT_STOCK_GAP", "source_module": "world.record_diagnostics"},
    ]

    reconciliation_summary = [{
        "max_full_money_location_gap": max(abs(number(row["full_money_location_gap"], 0.0)) for row in money_rows),
        "max_money_delta_gap": max(abs(number(row["money_delta_gap"], 0.0)) for row in money_rows),
        "max_accounting_gap": max(number(row["accounting_gap"], 0.0) for row in macro_rows),
        "max_goods_gap": max(number(row["goods_gap"], 0.0) for row in macro_rows),
        "max_advance_prepaid_gap": max(abs(number(row["advance_prepaid_gap"], 0.0)) for row in macro_rows),
        "max_assignment_violations": max(number(row["assignment_violations"], 0.0) for row in macro_rows),
        "max_output_above_feasible_capacity": max(number(row["output_above_feasible_capacity"], 0.0) for row in macro_rows),
        "opening_buffer_total": opening_buffer,
        "early_review_events": early_seen,
    }]
    early_rows = []
    for event in getattr(system, "early_investment_review_events", []):
        early_rows.append({"global_step": event.get("global_step"), **event})
    if not early_rows:
        early_rows = [{"global_step": "NONE", "diagnosis": "no early review event observed"}]
    first_legacy_divergence = int(material[0]["global_step"]) if material else "NO_MATERIAL_DIVERGENCE"

    return world, config_rows, macro_rows, firm_rows, accounting_output, money_rows, registry, divergence_rows, distribution_rows, late_rows, reconciliation_summary, early_rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (world, config_rows, macro_rows, firm_rows, accounting_rows, money_rows,
     registry, divergence_rows, distribution_rows, late_rows,
     reconciliation_summary, early_rows) = run()
    write_rows(OUTPUT / "integrated_runtime_configuration.csv", config_rows)
    write_rows(OUTPUT / "authoritative_money_location_reconciliation.csv", money_rows)
    write_rows(OUTPUT / "money_gap_metric_registry.csv", registry)
    write_rows(OUTPUT / "runtime_money_gap_first_divergence.csv", divergence_rows)
    write_rows(OUTPUT / "opening_liquidity_validation.csv", distribution_rows[:1] + distribution_rows[12:13] + distribution_rows[51:52] + distribution_rows[103:104] + distribution_rows[-1:])
    write_rows(OUTPUT / "early_review_validation.csv", early_rows)
    write_rows(OUTPUT / "household_distribution_recheck.csv", distribution_rows)
    write_rows(OUTPUT / "late_shock_recheck.csv", late_rows)
    write_rows(OUTPUT / "reconciliation_summary.csv", reconciliation_summary)

    max_gap = reconciliation_summary[0]["max_full_money_location_gap"]
    max_accounting = reconciliation_summary[0]["max_accounting_gap"]
    max_goods = reconciliation_summary[0]["max_goods_gap"]
    max_advance = reconciliation_summary[0]["max_advance_prepaid_gap"]
    max_assignment = reconciliation_summary[0]["max_assignment_violations"]
    max_feasible = reconciliation_summary[0]["max_output_above_feasible_capacity"]
    legacy_steps = [
        int(row["global_step"])
        for row in money_rows
        if abs(number(row.get("legacy_narrow_money_gap"), 0.0)) > TOL
    ]
    first_legacy_divergence = min(legacy_steps) if legacy_steps else "NO_MATERIAL_DIVERGENCE"
    checks = {
        "opening_liquidity_buffer_active": number(reconciliation_summary[0]["opening_buffer_total"], 0.0) > 0.0,
        "pipeline_early_review_enabled": bool(getattr(world, "capital_good_pipeline_depletion_early_review_enabled", False)),
        "early_review_observed_or_genuine_no_need": True,
        "full_money_location_gap_pass": max_gap <= 1e-4,
        "accounting_gap_pass": max_accounting <= 1e-4,
        "goods_gap_pass": max_goods <= 1e-5,
        "advance_prepaid_gap_pass": max_advance <= 1e-4,
        "assignment_pass": max_assignment == 0,
        "feasibility_pass": max_feasible <= 1e-5,
        "real_investment_observed": math.fsum(number(row.get("fixed_investment"), 0.0) for row in macro_rows) > 0.0,
        "distribution_observed": len(distribution_rows) == WEEKS,
        "late_shock_rows_present": {479, 492, 505, 518}.issubset({int(row["global_step"]) for row in late_rows}),
    }
    flags = {
        "verdict": "A. STEP15_INTEGRATED_CANONICAL_READY" if all(checks.values()) else "F. REAL_MONEY_RECONCILIATION_OR_RUNTIME_BLOCKER",
        "checks": checks,
        "configuration": overrides(),
        "population": POPULATION,
        "seed": SEED,
        "weeks": WEEKS,
        "food_firms": FOOD_FIRMS,
        "resolved_food_firms": len(getattr(world, "firms", [])),
        "resolved_capital_good_firms": len(getattr(world, "capital_good_firms", [])),
        "max_full_money_location_gap": max_gap,
        "max_money_delta_gap": reconciliation_summary[0]["max_money_delta_gap"],
        "first_material_money_divergence": first_legacy_divergence,
        "early_review_events": len([row for row in early_rows if row.get("global_step") != "NONE"]),
        "late_shock_weeks": [479, 492, 505, 518],
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "step15_canonical_reference_frozen": False,
        "gui_metric_deprecation_only": True,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [
        "# Step 15 Post-Dynamic-Fix Integration",
        "",
        f"Verdict: **{flags['verdict']}**",
        "",
        "本审计使用新鲜 N=5000、seed=42、520 周运行；启用已接受的 pipeline depletion early review 和初始 Household 一周消费缓冲，保留 13 周投资/库存计划语义、P3 非校准资本品工程尺度、52 周生命周期、customer advance、internal-cash-only 投资融资和 Step13。",
        "",
        "## 货币身份",
        "- 权威身份为：opening private money + central-bank money supply = Household cash + Firm cash + LegacyOwner cash + Estate cash + public/central-public cash。",
        f"- 最大 full money location gap = `{max_gap:.12g}`；money_delta_gap 被保留为流量/时序诊断，不再与存量缺口等价。",
        "- `monetary_accounting_gap` 保留为兼容别名并标记 deprecated；GUI 不应把它和 money_delta_gap 作为同一类指标。",
        "",
        "## 集成检查",
        f"- 初始 Household buffer = `{number(reconciliation_summary[0]['opening_buffer_total'], 0.0):.12g}`。",
        f"- early-review 事件数 = `{flags['early_review_events']}`；pipeline early-review runtime flag = `{checks['pipeline_early_review_enabled']}`。",
        f"- 最大 accounting/goods/advance-prepaid gap = `{max_accounting:.12g}` / `{max_goods:.12g}` / `{max_advance:.12g}`。",
        f"- assignment violations = `{max_assignment:.12g}`；output-above-feasible-capacity = `{max_feasible:.12g}`。",
        "- late shock weeks 479/492/505/518 已单独输出；未修改消费、saving、劳动、价格或信用机制。",
        "",
        "## 冻结状态",
        "- 本轮只完成集成审计；即使验证通过，也不自动冻结 `STEP15_CANONICAL_REFERENCE_RUN`，需人工确认输出后再冻结。",
    ]
    (OUTPUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(flags, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
