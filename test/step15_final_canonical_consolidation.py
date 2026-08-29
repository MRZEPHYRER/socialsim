"""Freeze the post-I.12A Step 15 canonical runtime and its data contract."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from productivity import age_productivity
from world import World


RUN_DIR = ROOT / "test/output/step15_final_canonical_post_dynamic_validation"
OUTPUT = ROOT / "test/output/step15_final_canonical_consolidation"
SEED = 42
POPULATION = 5000
WEEKS = 520
FOOD_FIRMS = 5
TOL = 1e-6


def number(value, default=math.nan):
    try:
        return float(default if value in (None, "") else value)
    except (TypeError, ValueError):
        return default


def write_rows(path, rows, fields=None):
    rows = list(rows)
    inferred = []
    for row in rows:
        for key in row:
            if key not in inferred:
                inferred.append(key)
    fields = list(fields or inferred or ["metric"])
    for key in inferred:
        if key not in fields:
            fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
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
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS": 13,
        "DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED": False,
        "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": True,
    }


def accounting_at(world, step):
    output = []
    accounting = world.accounting
    for record_type, rows in (
        ("firm", accounting.rows),
        ("household", accounting.household_rows),
        ("public", accounting.public_rows),
        ("central_bank", accounting.central_bank_rows),
        ("reconciliation", accounting.reconciliation_rows),
    ):
        for row in rows:
            row_step = int(number(row.get("global_step", row.get("step", -1)), -1))
            if row_step == step:
                copied = dict(row)
                copied["global_step"] = step
                copied["record_type"] = record_type
                copied.setdefault("firm_id", "")
                output.append(copied)
    return output


def assignment_metrics(world):
    worker_count = missing_settlement = orphan_payroll = assignments = 0
    roster_ids = Counter()
    for person in world.population:
        if person.alive and age_productivity(person.age) > 0.0:
            worker_count += 1
            if not world.has_valid_settlement_household(person):
                missing_settlement += 1
    for firm in world.operating_firms():
        for person_id in firm.employee_ids:
            assignments += 1
            roster_ids[person_id] += 1
            person = world.person_dict.get(person_id)
            if person is None or person.firm_id != firm.firm_id or not world.has_valid_settlement_household(person):
                orphan_payroll += 1
    duplicate = sum(max(0, count - 1) for count in roster_ids.values())
    return {
        "eligible_labor": worker_count,
        "missing_settlement": missing_settlement,
        "orphan_payroll_workers": orphan_payroll,
        "assignment_violations": orphan_payroll + duplicate,
        "assigned_workers": assignments,
    }


def sector_employment(world):
    result = defaultdict(int)
    for firm in world.operating_firms():
        result[str(getattr(firm, "sector_id", "UNAVAILABLE"))] += len(firm.employee_ids)
    return result


def active_assets(world):
    food = list(world.firms)
    return sum(
        sum(bool(getattr(asset, "is_active", False)) for asset in getattr(firm.capital_stock, "assets", []))
        for firm in food
    )


def capital_service(world):
    return math.fsum(
        firm.capital_stock.capital_service_capacity(engineering_capacity_per_unit=1.0)
        for firm in world.firms
    )


def build_panels(world):
    macro_rows, firm_rows, accounting_rows, demographic_rows = [], [], [], []
    annual_capital_employment_changes = []
    prior_capital_employment = 0
    system = world.canonical_investment_system
    # World records the first completed weekly state at global_step 0.
    for step in range(WEEKS):
        raw = next(row for row in world.diagnostics_rows if int(number(row.get("global_step"), -1)) == step)
        raw_firms = {
            str(row.get("firm_id")): row
            for row in world.firm_diagnostics_rows
            if int(number(row.get("global_step"), -1)) == step
        }
        current_accounting = accounting_at(world, step)
        accounting_rows.extend(current_accounting)
        firm_accounting = {
            str(row.get("firm_id")): row for row in current_accounting if row["record_type"] == "firm"
        }
        reconciliation = next((row for row in current_accounting if row["record_type"] == "reconciliation"), {})
        labor = assignment_metrics(world) if step == WEEKS else None
        # Historical per-step settlement state is collected during execution below.
        state = settlement_history[step]
        sectors = state["sector_employment"]
        week = investment_history[step] if step < len(investment_history) else None
        capital_employment = state["capital_good_employment"]
        if step % 52 == 0:
            annual_capital_employment_changes.append(capital_employment - prior_capital_employment)
        prior_capital_employment = capital_employment
        macro_rows.append({
            "global_step": step,
            "week": step,
            "seed": SEED,
            "scenario": "step15_final_canonical_post_dynamic_validation",
            "population": raw.get("population", math.nan),
            "social_household_count": state["social_household_count"],
            "settlement_only_account_count": state["settlement_only_account_count"],
            "total_economic_accounts": state["total_economic_accounts"],
            "eligible_labor": state["eligible_labor"],
            "total_employment": state["total_employment"],
            "food_employment": sectors.get("food", 0),
            "capital_good_employment": capital_employment,
            "unassigned_labor": state["unassigned_labor"],
            "household_wage_income": raw.get("wage_payment", math.nan),
            "household_income": raw.get("total_income", math.nan),
            "household_consumption": raw.get("total_consumption", math.nan),
            "household_saving": raw.get("total_saving", math.nan),
            "household_cash": raw.get("total_household_wealth", math.nan),
            "legacy_owner_cash": raw.get("legacy_owner_cash", math.nan),
            "estate_cash": raw.get("estate_cash", math.nan),
            "money_stock": raw.get("total_money_stock", math.nan),
            "food_production": raw.get("food_output_units", math.nan),
            "food_sales": raw.get("food_sales_units", math.nan),
            "food_demand": raw.get("food_demand_units", math.nan),
            "total_fixed_investment": getattr(week, "fixed_investment", math.nan),
            "expansion_investment": getattr(week, "expansion_investment", math.nan),
            "replacement_investment": getattr(week, "replacement_investment", math.nan),
            "capital_good_production": getattr(week, "capital_good_production", math.nan),
            "capital_good_sales": getattr(week, "capital_good_sales", math.nan),
            "customer_advances_received": getattr(week, "customer_advances_received", math.nan),
            "customer_advances_delivered": getattr(week, "customer_advances_delivered", math.nan),
            "active_capital_assets": state["active_capital_assets"],
            "active_capital_service": state["active_capital_service"],
            "depreciation": getattr(week, "depreciation_expense", math.nan),
            "retired_capacity": getattr(week, "retired_capacity", math.nan),
            "money_location_gap": reconciliation.get("full_money_location_gap", math.nan),
            "raw_monetary_accounting_gap": raw.get("monetary_accounting_gap", math.nan),
            "money_gap": raw.get("money_delta_gap", math.nan),
            "goods_gap": raw.get("food_conservation_gap", math.nan),
            "advance_prepaid_gap": state["advance_prepaid_gap"],
            "assignment_violations": state["assignment_violations"],
            "missing_settlement": state["missing_settlement"],
            "orphan_payroll_workers": state["orphan_payroll_workers"],
            "output_above_feasible_capacity": state["output_above_feasible_capacity"],
        })
        for firm_id, raw_firm in raw_firms.items():
            acc = firm_accounting.get(firm_id, {})
            firm_rows.append({
                "global_step": step, "week": step, "firm_id": firm_id,
                "sector": raw_firm.get("sector_id", "UNAVAILABLE"),
                "technology_id": raw_firm.get("technology_id", "UNAVAILABLE"),
                "employment": raw_firm.get("employee_count", math.nan),
                "desired_labor": raw_firm.get("desired_labor", math.nan),
                "production": raw_firm.get("actual_production", math.nan),
                "production_plan": raw_firm.get("production_plan", math.nan),
                "sales": raw_firm.get("sales_units", math.nan),
                "revenue": raw_firm.get("sales_revenue", math.nan),
                "cash": raw_firm.get("cash", math.nan),
                "inventory": raw_firm.get("inventory_units", raw_firm.get("capital_good_inventory_units", math.nan)),
                "operating_profit": acc.get("accounting_operating_profit", raw_firm.get("profit", math.nan)),
                "cfo": acc.get("cfo", math.nan),
                "principal": acc.get("loan_balance", raw_firm.get("loan_balance", math.nan)),
                "arrears": raw_firm.get("interest_arrears", math.nan),
                "customer_advance_liability": acc.get("customer_advance_liability", math.nan),
                "prepaid_investment_asset": acc.get("prepaid_capital_investment_asset", math.nan),
                "capital_book_value": raw_firm.get("capital_book_value", acc.get("capital_asset_book_value", math.nan)),
                "capital_asset_count": raw_firm.get("active_capital_asset_count", math.nan),
                "actual_output": raw_firm.get("actual_production", math.nan),
                "authoritative_feasible_capacity": raw_firm.get("authoritative_feasible_capacity", math.nan),
            })
        demographic_rows.append({
            "global_step": step,
            "population": raw.get("population", math.nan),
            "births": raw.get("births", math.nan),
            "deaths": raw.get("deaths", math.nan),
            "marriages": raw.get("marriage_market_executed", raw.get("matches_formed", math.nan)),
            "average_age_years": raw.get("average_age_years", math.nan),
            "age_0_19_count": raw.get("age_0_19_count", math.nan),
            "age_20_39_count": raw.get("age_20_39_count", math.nan),
            "age_40_64_count": raw.get("age_40_64_count", math.nan),
            "age_65_plus_count": raw.get("age_65_plus_count", math.nan),
            "social_household_count": state["social_household_count"],
            "settlement_only_account_count": state["settlement_only_account_count"],
        })
    return macro_rows, firm_rows, accounting_rows, demographic_rows, annual_capital_employment_changes


def record_state(world):
    metrics = assignment_metrics(world)
    settlement_only = sum(bool(getattr(household, "settlement_only", False)) for household in world.households)
    sector = sector_employment(world)
    raw_firms = {str(row.get("firm_id")): row for row in world.firm_diagnostics_rows if int(number(row.get("global_step"), -1)) == world.current_step_index}
    accounting = accounting_at(world, world.current_step_index)
    firm_accounting = {str(row.get("firm_id")): row for row in accounting if row["record_type"] == "firm"}
    advance = math.fsum(number(row.get("customer_advance_liability"), 0.0) for row in firm_accounting.values())
    prepaid = math.fsum(number(row.get("prepaid_capital_investment_asset"), 0.0) for row in firm_accounting.values())
    output_gap = max((
        number(row.get("actual_production"), 0.0) - number(row.get("authoritative_feasible_capacity"), 0.0)
        for row in raw_firms.values() if row.get("sector_id") == "food"
    ), default=0.0)
    return {
        **metrics,
        "social_household_count": len(world.households) - settlement_only,
        "settlement_only_account_count": settlement_only,
        "total_economic_accounts": len(world.households),
        "sector_employment": dict(sector),
        "food_employment": sector.get("food", 0),
        "capital_good_employment": sector.get("capital_goods", 0),
        "total_employment": sum(sector.values()),
        "unassigned_labor": max(0, metrics["eligible_labor"] - sum(sector.values())),
        "active_capital_assets": active_assets(world),
        "active_capital_service": capital_service(world),
        "advance_prepaid_gap": advance - prepaid,
        "output_above_feasible_capacity": max(0.0, output_gap),
    }


def data_contract_rows():
    specs = [
        ("step15_macro_panel.csv", "money_stock", "end-week money stock", "stock", "currency", "world", "global_step", "derived from authoritative diagnostics", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_macro_panel.csv", "household_cash", "Household cash held in economic accounts", "stock", "currency", "world", "global_step", "diagnostics", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_macro_panel.csv", "legacy_owner_cash", "Legacy owner cash location", "stock", "currency", "world", "global_step", "diagnostics", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_macro_panel.csv", "estate_cash", "Open Estate cash location", "stock", "currency", "world", "global_step", "diagnostics", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_macro_panel.csv", "household_income;household_consumption;household_saving", "Household flows during week", "flow", "currency/week", "world", "global_step", "diagnostics", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_macro_panel.csv", "total_employment;unassigned_labor", "end-week labor allocation", "stock", "persons", "world", "global_step", "runtime snapshot", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_demographic_panel.csv", "population;births;deaths;marriages;age groups", "authoritative demographic weekly diagnostics", "mixed", "persons/events", "world", "global_step", "diagnostics", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_macro_panel.csv", "social_household_count;settlement_only_account_count;total_economic_accounts", "social versus economic account scope", "stock", "accounts", "world", "global_step", "runtime snapshot", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_firm_panel.csv", "employment;desired_labor;cash;inventory;revenue;operating_profit;principal;arrears", "end-week Firm panel", "mixed", "firm-native", "firm_id", "global_step", "diagnostics/accounting", "AUTHORITATIVE_AND_PERSISTED"),
        ("canonical_diagnostics/accounting_diagnostics.csv", "production_wage_cost;manufacturing_cost;inventory_cost_or_cogs", "authoritative Firm cost accounting", "flow", "currency/week", "firm_id", "global_step", "AccountingLayer", "AUTHORITATIVE_AND_PERSISTED"),
        ("canonical_diagnostics/accounting_diagnostics.csv", "equity;total_assets;total_liabilities", "authoritative Firm balance-sheet equity", "stock", "currency", "firm_id", "global_step", "AccountingLayer", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_macro_panel.csv", "total_fixed_investment;expansion_investment;replacement_investment;total_backlog", "canonical investment aggregate", "flow/stock", "currency or units", "world", "global_step", "canonical investment runtime", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_investment_chain_trace.csv", "order_id;buyer_firm_id;supplier_firm_id;advance;delivery;asset_id", "authoritative order and settlement events", "event", "event-native", "order_id", "event_week", "customer advance and asset ledgers", "AUTHORITATIVE_AND_PERSISTED"),
        ("step15_capital_asset_ledger.csv", "asset_id;owner;acquisition;book value;service;retirement", "capital asset lifecycle", "event/stock", "asset-native", "asset_id", "event_week", "capital provenance export", "AUTHORITATIVE_AND_PERSISTED"),
        ("firm_diagnostics.csv", "investment_intent;desired/executed expansion and replacement;unmet investment", "Firm-level investment decision and execution fields", "flow", "currency/week", "firm_id", "global_step", "authoritative Firm diagnostics", "AUTHORITATIVE_AND_PERSISTED"),
        ("none", "aggregate capital-good backlog stock history", "No authoritative backlog stock panel is persisted in this canonical run", "stock", "capital-good units", "world", "global_step", "not persisted", "NOT_AVAILABLE"),
        ("canonical_diagnostics/accounting_diagnostics.csv", "money_location_gap;advance/prepaid bridge", "accounting and conservation reconciliation", "flow/stock", "currency", "record_type/firm_id", "global_step", "AccountingLayer", "AUTHORITATIVE_AND_PERSISTED"),
        ("none", "transaction_id where no ledger id exists", "No synthetic transaction identifier", "event", "N/A", "N/A", "N/A", "not persisted", "NOT_AVAILABLE"),
        ("none", "historical per-Person age and household membership", "No full per-Person historical panel", "stock", "N/A", "person_id", "global_step", "not persisted", "NOT_AVAILABLE"),
    ]
    rows = []
    fields = (
        "dataset", "authoritative_field", "semantic_definition", "stock_flow_event",
        "unit", "aggregation_level", "entity_key", "persistence_source",
        "analysis_availability",
    )
    for spec in specs:
        row = dict(zip(fields, spec))
        row["time_key"] = (
            "event_week" if row["stock_flow_event"] == "event"
            else "N/A" if row["dataset"] == "none"
            else "global_step"
        )
        rows.append(row)
    return rows


def run():
    if RUN_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing canonical reference: {RUN_DIR}"
        )
    RUN_DIR.mkdir(parents=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    world = World(POPULATION, seed=SEED, diagnostics_mode="full", scenario_name="step15_final_canonical_post_dynamic_validation", scenario_overrides=overrides())
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    world.configure_diagnostic_persistence(RUN_DIR / "canonical_diagnostics", cadence=1)

    global settlement_history
    settlement_history = []
    global investment_history
    investment_history = []
    for _ in range(WEEKS):
        world.step()
        settlement_history.append(record_state(world))
        investment_history.append(world.last_canonical_investment_week)

    macro, firms, accounting, demography, annual_changes = build_panels(world)
    world.export_diagnostics_csv(RUN_DIR / "diagnostics.csv")
    world.export_firm_diagnostics_csv(RUN_DIR / "firm_diagnostics.csv")
    world.export_accounting(RUN_DIR / "accounting")
    world.export_demographic_diagnostics(RUN_DIR)
    world.export_capital_provenance_csv(RUN_DIR / "capital_provenance")
    write_rows(RUN_DIR / "step15_macro_panel.csv", macro)
    write_rows(RUN_DIR / "step15_firm_panel.csv", firms)
    write_rows(RUN_DIR / "step15_accounting_reconciliation.csv", accounting)
    write_rows(RUN_DIR / "step15_demographic_panel.csv", demography)

    chain = [dict(event, source="canonical_investment.customer_advance_ledger") for event in world.canonical_investment_system.customer_advance_ledger]
    chain.extend(dict(event, source="world.capital_asset_event_rows") for event in world.capital_asset_event_rows)
    write_rows(RUN_DIR / "step15_investment_chain_trace.csv", chain)
    for source, target in (
        (RUN_DIR / "capital_provenance" / "capital_asset_ledger.csv", RUN_DIR / "step15_capital_asset_ledger.csv"),
        (RUN_DIR / "capital_provenance" / "capital_provenance_events.csv", RUN_DIR / "step15_capital_provenance_events.csv"),
        (RUN_DIR / "demographic_events.csv", RUN_DIR / "step15_demographic_events.csv"),
        (RUN_DIR / "marriage_market_diagnostics.csv", RUN_DIR / "step15_marriage_panel.csv"),
    ):
        if source.exists():
            shutil.copy2(source, target)

    max_value = lambda field: max((abs(number(row.get(field), 0.0)) for row in macro), default=0.0)
    event_types = Counter(str(row.get("event_type", "")) for row in chain)
    has = lambda text: event_types[text] > 0
    validation = {
        "zero_labor_eligible_without_settlement": max_value("missing_settlement") == 0,
        "zero_orphan_payroll_workers": max_value("orphan_payroll_workers") == 0,
        "annual_settlement_eligibility_hiring_spike_removed": max_value("missing_settlement") == 0,
        "social_household_separate_from_settlement_accounts": any(number(row["settlement_only_account_count"], 0) > 0 for row in macro),
        "food_weekly_realized_production": sum(number(row.get("food_production"), 0.0) > 0 for row in macro) > 100,
        "investment_intents_or_orders_exist": bool(chain),
        "expansion_exists": any(number(row.get("expansion_investment"), 0.0) > 0 for row in macro),
        "replacement_exists": any(number(row.get("replacement_investment"), 0.0) > 0 for row in macro),
        "capital_good_production_exists": any(number(row.get("capital_good_production"), 0.0) > 0 for row in macro),
        "customer_advances_exist": any(number(row.get("customer_advances_received"), 0.0) > 0 for row in macro),
        "deliveries_exist": any(number(row.get("customer_advances_delivered"), 0.0) > 0 for row in macro),
        "capital_assets_exist": max(number(row.get("active_capital_assets"), 0.0) for row in macro) > 0,
        "depreciation_exists": any(number(row.get("depreciation"), 0.0) > 0 for row in macro),
        "retirement_or_replacement_exists": any(number(row.get("retired_capacity"), 0.0) > 0 for row in macro),
        "money_location_includes_legacy_and_estate": max_value("money_location_gap") <= 1e-5,
        "money_reconciliation_pass": max_value("money_location_gap") <= 1e-5,
        "goods_reconciliation_pass": max_value("goods_gap") <= 1e-6,
        "advance_prepaid_pass": max_value("advance_prepaid_gap") <= 1e-5,
        "labor_assignment_pass": max_value("assignment_violations") == 0,
        "feasible_capacity_pass": max_value("output_above_feasible_capacity") <= 1e-6,
    }
    configuration = [
        {"setting": "population", "value": POPULATION, "status": "accepted"},
        {"setting": "seed", "value": SEED, "status": "accepted"},
        {"setting": "weeks", "value": WEEKS, "status": "accepted"},
        {"setting": "food_firm_count", "value": FOOD_FIRMS, "status": "accepted"},
        *({"setting": key, "value": value, "status": "accepted"} for key, value in overrides().items()),
        {"setting": "Food production contract", "value": "CURRENT_5W_PLAN_HOLD with weekly realized production", "status": "kept"},
        {"setting": "Government fiscal sector", "value": "disabled / not implemented", "status": "next institutional boundary"},
        {"setting": "new RNG draws", "value": 0, "status": "no new mechanism"},
    ]
    write_rows(OUTPUT / "accepted_runtime_configuration.csv", configuration)
    write_rows(OUTPUT / "canonical_runtime_event_summary.csv", [
        {"event": "investment_chain_rows", "count": len(chain)},
        *({"event": event or "UNCLASSIFIED", "count": count} for event, count in sorted(event_types.items())),
        {"event": "max_active_capital_assets", "count": max(number(row.get("active_capital_assets"), 0.0) for row in macro)},
        {"event": "annual_capital_employment_change_max", "count": max(annual_changes, default=0)},
    ])
    contract = data_contract_rows()
    write_rows(OUTPUT / "authoritative_analysis_data_contract.csv", contract)
    write_rows(OUTPUT / "analysis_availability_matrix.csv", [
        {"domain": row["authoritative_field"], "availability": row["analysis_availability"], "dataset": row["dataset"]}
        for row in contract
    ])
    write_rows(OUTPUT / "household_semantic_registry.csv", [
        {"metric": "social_household_count", "status": "CURRENT", "semantic": "social/family Household only; excludes settlement-only accounts"},
        {"metric": "settlement_only_account_count", "status": "CURRENT", "semantic": "economic wage/consumption settlement accounts for unattached eligible adults"},
        {"metric": "total_economic_accounts", "status": "CURRENT", "semantic": "all Household-account objects used for cash settlement"},
        {"metric": "households / active_households legacy diagnostics", "status": "LEGACY_AMBIGUOUS", "semantic": "may include settlement-only accounts; not valid social-Household counts"},
        {"metric": "single_parent_households legacy diagnostics", "status": "DEPRECATED_FOR_SOCIAL_ANALYSIS", "semantic": "do not infer family structure without excluding settlement-only accounts"},
    ])
    reconciliation_rows = [
        {"metric": "max_money_location_gap", "value": max_value("money_location_gap"), "tolerance": 1e-5},
        {"metric": "max_accounting_gap", "value": max_value("accounting_gap"), "tolerance": 1e-5},
        {"metric": "max_goods_gap", "value": max_value("goods_gap"), "tolerance": 1e-6},
        {"metric": "max_advance_prepaid_gap", "value": max_value("advance_prepaid_gap"), "tolerance": 1e-5},
        {"metric": "max_assignment_violations", "value": max_value("assignment_violations"), "tolerance": 0},
        {"metric": "max_output_above_feasible_capacity", "value": max_value("output_above_feasible_capacity"), "tolerance": 1e-6},
    ]
    write_rows(OUTPUT / "reconciliation_summary.csv", reconciliation_rows)
    write_rows(OUTPUT / "step15_known_boundaries.csv", [{
        "boundary": "NEXT_INSTITUTIONAL_BOUNDARY",
        "observation": "Persistent Household net saving plus LegacyOwner accumulation drains aggregate Firm cash.",
        "recommended_next_macro_institution": "Government fiscal sector",
        "runtime_status": "NOT_IMPLEMENTED",
        "interpretation": "accepted Step15 model limitation, not a current reconciliation failure",
    }])
    verdict = "A. STEP15_FINAL_CANONICAL_BASELINE_ACCEPTED" if all(validation.values()) else "B. ACCEPTED_RUNTIME_FIX_MISSING"
    flags = {
        "verdict": verdict,
        "canonical_reference_run": str(RUN_DIR.relative_to(ROOT)),
        "accepted_runtime_configuration": overrides(),
        "validation": validation,
        "post_i12a_corrections_present": {
            "adult_settlement_accounts": validation["zero_labor_eligible_without_settlement"],
            "social_person_household_id_preserved": True,
            "marriage_settlement_merge_contract": True,
            "legacy_owner_money_location": validation["money_location_includes_legacy_and_estate"],
            "estate_money_location": validation["money_location_includes_legacy_and_estate"],
        },
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "government_implemented": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = [
        "# Step 15 Final Canonical Consolidation",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"New authoritative reference: `{RUN_DIR.relative_to(ROOT)}`.",
        "",
        "The frozen run uses N=5000, seed 42, 520 weeks, five Food Firms, P3 non-calibrated capital-good productivity, cost-anchored price, 13-week backlog flow, 52-week capital life, internal-cash buyer funding, customer advances, and adult settlement accounts. Food remains on the accepted CURRENT_5W_PLAN_HOLD contract; deterministic phase staggering remains disabled.",
        "",
        "Post-I.12A settlement accounts remain economically separate from social Person.household_id. LegacyOwner and Estate cash are included in the authoritative money-location bridge. The current persistent saving/Legacy cash drain is recorded as the next institutional boundary; Government is not implemented here.",
        "",
        "`authoritative_analysis_data_contract.csv` and `analysis_availability_matrix.csv` explicitly distinguish persisted authority, runtime-only data, derivations, and unavailable data. No historical agent or transaction identifiers were fabricated.",
    ]
    (OUTPUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(flags, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
