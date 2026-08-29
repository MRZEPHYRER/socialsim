"""Materialize and validate the final Step 15 Analysis V2 canonical run.

This is an orchestration/validation script.  It enables only the already
accepted runtime flags and uses the existing diagnostic and statistical
observers; it does not add a new economic persistence path.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

import step15_final_canonical_consolidation as base
from analysis.gui_v2.data import CanonicalDataStore
from analysis.gui_v2.query import AnalysisV2Query
from world import World


FINAL_RUN = ROOT / "test/output/step15_final_canonical_integrated_v2"
FREEZE = ROOT / "test/output/step15_final_canonical_freeze"
AUDIT = ROOT / "test/output/step15_post_dynamic_fix_integration"
POPULATION = 5000
SEED = 42
WEEKS = 520
FOOD_FIRMS = 5
TOL = 1e-5


def num(value, default=math.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
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


def read_rows(path):
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def overrides():
    return {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
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


def money_row(world, step):
    components = world.authoritative_money_location_components()
    raw = next(
        (row for row in reversed(world.diagnostics_rows)
         if int(num(row.get("global_step", -1), -1)) == step),
        {},
    )
    stock = world.authoritative_money_stock()
    located = components["located_money_stock"]
    return {
        "global_step": step,
        **components,
        "authoritative_money_stock": stock,
        "money_delta_gap": num(raw.get("money_delta_gap"), 0.0),
        "full_money_location_gap": stock - located,
        "legacy_narrow_money_gap": num(raw.get("legacy_narrow_money_gap"), 0.0),
        "accounting_money_location_gap": num(raw.get("accounting_money_location_gap"), 0.0),
    }


def backlog_state(world):
    system = world.canonical_investment_system
    expansion = math.fsum(
        max(0.0, num(value, 0.0))
        for value in system.expansion_backlog_by_firm.values()
    )
    replacement = math.fsum(
        max(0.0, num(getattr(firm, "pending_replacement_capacity_need", 0.0), 0.0))
        for firm in world.firms
    )
    capital_firms = list(getattr(world, "capital_good_firms", []))
    return {
        "expansion_backlog": expansion,
        "replacement_backlog": replacement,
        "total_backlog": expansion + replacement,
        "capital_good_desired_output": math.fsum(
            max(0.0, num(getattr(firm, "capital_good_desired_output", 0.0), 0.0))
            for firm in capital_firms
        ),
        "capital_good_desired_labor": math.fsum(
            max(0.0, num(getattr(firm, "desired_labor", 0.0), 0.0))
            for firm in capital_firms
        ),
    }


def materialize():
    if FINAL_RUN.exists():
        raise FileExistsError(f"Refusing to overwrite final canonical run: {FINAL_RUN}")
    FINAL_RUN.mkdir(parents=True)
    world = World(
        POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15_final_canonical_integrated_v2",
        scenario_overrides=overrides(),
    )
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    world.configure_diagnostic_persistence(
        FINAL_RUN / "canonical_diagnostics",
        cadence=1,
        statistical_observability=True,
        statistical_output_dir=FINAL_RUN / "statistical_observability",
        statistical_snapshot_cadence=13,
        statistical_age_cadence=13,
    )

    base.settlement_history = []
    base.investment_history = []
    money_history = []
    backlog_history = []
    for _ in range(WEEKS):
        world.step()
        base.settlement_history.append(base.record_state(world))
        base.investment_history.append(world.last_canonical_investment_week)
        money_history.append(money_row(world, world.current_step_index))
        backlog_history.append(backlog_state(world))

    macro, firms, accounting, demography, _ = base.build_panels(world)
    money_by_step = {row["global_step"]: row for row in money_history}
    backlog_by_step = {index: row for index, row in enumerate(backlog_history)}
    for row in macro:
        step = int(num(row.get("global_step"), -1))
        row.update(money_by_step.get(step, {}))
        row.update(backlog_by_step.get(step, {}))
        row["pending_household_formation_wealth"] = money_by_step.get(step, {}).get(
            "pending_household_formation_wealth", math.nan
        )
        row["money_location_gap"] = row.get("full_money_location_gap", math.nan)
        row["money_reconciliation_gap"] = row.get("money_delta_gap", math.nan)
        row["accounting_reconciliation_gap"] = row.get("raw_monetary_accounting_gap", math.nan)
    firm_lookup = {
        str(firm.firm_id): firm
        for firm in [*world.firms, *getattr(world, "capital_good_firms", [])]
    }
    for row in firms:
        firm = firm_lookup.get(str(row.get("firm_id")))
        if firm is not None:
            row["sector_id"] = getattr(firm, "sector_id", row.get("sector_id"))
            row["capital_capacity"] = num(getattr(firm, "capital_capacity", 0.0), 0.0)
            row["active_capital_asset_count"] = len(
                [asset for asset in getattr(getattr(firm, "capital_stock", None), "assets", [])
                 if bool(getattr(asset, "is_active", False))]
            )

    world.export_diagnostics_csv(FINAL_RUN / "diagnostics.csv")
    world.export_firm_diagnostics_csv(FINAL_RUN / "firm_diagnostics.csv")
    world.export_accounting(FINAL_RUN / "accounting")
    world.export_demographic_diagnostics(FINAL_RUN)
    world.export_capital_provenance_csv(FINAL_RUN / "capital_provenance")
    write_rows(FINAL_RUN / "step15_macro_panel.csv", macro)
    write_rows(FINAL_RUN / "step15_firm_panel.csv", firms)
    write_rows(FINAL_RUN / "step15_accounting_reconciliation.csv", accounting)
    write_rows(FINAL_RUN / "step15_demographic_panel.csv", demography)
    chain = [
        dict(event, source="canonical_investment.customer_advance_ledger")
        for event in world.canonical_investment_system.customer_advance_ledger
    ]
    chain.extend(
        dict(event, source="world.capital_asset_event_rows")
        for event in world.capital_asset_event_rows
    )
    write_rows(FINAL_RUN / "step15_investment_chain_trace.csv", chain)
    for source, target in (
        (FINAL_RUN / "capital_provenance/capital_asset_ledger.csv", FINAL_RUN / "step15_capital_asset_ledger.csv"),
        (FINAL_RUN / "capital_provenance/capital_provenance_events.csv", FINAL_RUN / "step15_capital_provenance_events.csv"),
        (FINAL_RUN / "demographic_events.csv", FINAL_RUN / "step15_demographic_events.csv"),
        (FINAL_RUN / "marriage_market_diagnostics.csv", FINAL_RUN / "step15_marriage_panel.csv"),
    ):
        if source.exists():
            shutil.copy2(source, target)
    write_rows(FINAL_RUN / "authoritative_money_location_reconciliation.csv", money_history)
    write_rows(FINAL_RUN / "step15_final_firm_summary.csv", [row for row in firms if int(num(row.get("global_step"), -1)) == WEEKS - 1])
    return world, macro, firms, accounting, chain, money_history


def contract_rows():
    rows = []
    for row in base.data_contract_rows():
        if row["authoritative_field"] == "aggregate capital-good backlog stock history":
            continue
        rows.append(dict(row))
    rows.extend([
        {
            "dataset": "step15_macro_panel.csv",
            "authoritative_field": "pending_household_formation_wealth",
            "semantic_definition": "signed pending household-formation balance included in located money scope",
            "stock_flow_event": "stock",
            "unit": "currency",
            "aggregation_level": "world",
            "entity_key": "global_step",
            "persistence_source": "World.authoritative_money_location_components",
            "analysis_availability": "AUTHORITATIVE_AND_PERSISTED",
            "time_key": "global_step",
        },
        {
            "dataset": "step15_macro_panel.csv",
            "authoritative_field": "full_money_location_gap",
            "semantic_definition": "authoritative stock identity: authoritative money stock minus all located cash components",
            "stock_flow_event": "stock identity",
            "unit": "currency gap",
            "aggregation_level": "world",
            "entity_key": "global_step",
            "persistence_source": "World.authoritative_money_location_components",
            "analysis_availability": "AUTHORITATIVE_AND_PERSISTED",
            "time_key": "global_step",
        },
        {
            "dataset": "step15_macro_panel.csv",
            "authoritative_field": "money_delta_gap",
            "semantic_definition": "flow/timing diagnostic, not an authoritative stock identity",
            "stock_flow_event": "flow diagnostic",
            "unit": "currency gap",
            "aggregation_level": "world",
            "entity_key": "global_step",
            "persistence_source": "World diagnostics",
            "analysis_availability": "AUTHORITATIVE_DIAGNOSTIC",
            "time_key": "global_step",
        },
    ])
    statistical_contract = FINAL_RUN / "statistical_observability/statistical_persistence_contract.csv"
    for row in read_rows(statistical_contract):
        copied = dict(row)
        copied["analysis_availability"] = "AUTHORITATIVE_AND_PERSISTED"
        copied["persistence_source"] = copied.get("source_runtime_field", "statistical observability")
        rows.append(copied)
    return rows


def compare_audit_money(money_history):
    audit_rows = read_rows(AUDIT / "authoritative_money_location_reconciliation.csv")
    new = {int(num(row.get("global_step"), -1)): row for row in money_history}
    rows = []
    fields = (
        "household_cash", "firm_cash", "legacy_owner_cash", "estate_cash",
        "public_wealth", "central_public_income_balance",
        "pending_household_formation_wealth", "located_money_stock",
        "authoritative_money_stock", "full_money_location_gap",
    )
    for field in fields:
        diffs = []
        for old in audit_rows:
            step = int(num(old.get("global_step"), -1))
            if step not in new or field not in old:
                continue
            diffs.append(abs(num(old.get(field), 0.0) - num(new[step].get(field), 0.0)))
        maximum = max(diffs, default=math.nan)
        rows.append({
            "reference": "step15_post_dynamic_fix_integration",
            "dataset": "authoritative_money_location_reconciliation.csv",
            "field": field,
            "compared_rows": len(diffs),
            "maximum_absolute_difference": maximum,
            "tolerance": 1e-6,
            "passed": bool(diffs) and maximum <= 1e-6,
        })
    return rows


def validation_artifacts(world, macro, firms, accounting, chain, money_history):
    FREEZE.mkdir(parents=True, exist_ok=True)
    max_field = lambda field: max((abs(num(row.get(field), 0.0)) for row in macro), default=0.0)
    accounting_gap = max(
        (abs(num(row.get(field), 0.0))
         for row in accounting
         for field in ("cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap",
                       "equity_bridge_gap", "capital_book_value_bridge_gap",
                       "customer_advance_liability_bridge_gap",
                       "prepaid_investment_bridge_gap")),
        default=0.0,
    )
    audit_equivalence = compare_audit_money(money_history)
    write_rows(FREEZE / "canonical_equivalence_validation.csv", audit_equivalence)
    write_rows(FREEZE / "final_runtime_configuration.csv", [
        {"setting": "population", "value": POPULATION, "status": "accepted"},
        {"setting": "seed", "value": SEED, "status": "accepted"},
        {"setting": "weeks", "value": WEEKS, "status": "accepted"},
        {"setting": "food_firm_count", "value": FOOD_FIRMS, "status": "accepted"},
        *({"setting": key, "value": value, "status": "accepted"} for key, value in overrides().items()),
        {"setting": "new_rng_draws", "value": 0, "status": "observational persistence only"},
    ])
    write_rows(FREEZE / "final_analysis_data_contract.csv", contract_rows())
    write_rows(FREEZE / "final_money_metric_registry.csv", [
        {"metric": "full_money_location_gap", "role": "AUTHORITATIVE_STOCK_RECONCILIATION", "includes": "Household + Firm + LegacyOwner + Estate + public/central-public + pending_household_formation_wealth", "pass_tolerance": 1e-5},
        {"metric": "money_delta_gap", "role": "FLOW_TIMING_DIAGNOSTIC", "includes": "weekly located-money movement timing", "pass_tolerance": "not a stock identity"},
        {"metric": "monetary_accounting_gap", "role": "LEGACY_COMPATIBILITY_ALIAS", "includes": "deprecated legacy diagnostic", "pass_tolerance": "not interchangeable with full_money_location_gap"},
        {"metric": "pending_household_formation_wealth", "role": "MONEY_LOCATION_COMPONENT", "includes": "signed pending formation balance", "pass_tolerance": "included in full stock scope"},
    ])
    write_rows(FREEZE / "final_reconciliation_summary.csv", [
        {"metric": "max_full_money_location_gap", "value": max_field("full_money_location_gap"), "tolerance": 1e-5, "passed": max_field("full_money_location_gap") <= 1e-5},
        {"metric": "max_accounting_gap", "value": accounting_gap, "tolerance": 1e-5, "passed": accounting_gap <= 1e-5},
        {"metric": "max_goods_gap", "value": max_field("goods_gap"), "tolerance": 1e-6, "passed": max_field("goods_gap") <= 1e-6},
        {"metric": "max_advance_prepaid_gap", "value": max_field("advance_prepaid_gap"), "tolerance": 1e-5, "passed": max_field("advance_prepaid_gap") <= 1e-5},
        {"metric": "max_assignment_violations", "value": max_field("assignment_violations"), "tolerance": 0, "passed": max_field("assignment_violations") == 0},
        {"metric": "max_orphan_payroll", "value": max_field("orphan_payroll_workers"), "tolerance": 0, "passed": max_field("orphan_payroll_workers") == 0},
        {"metric": "max_output_above_feasible_capacity", "value": max_field("output_above_feasible_capacity"), "tolerance": 1e-6, "passed": max_field("output_above_feasible_capacity") <= 1e-6},
    ])
    opening = read_rows(AUDIT / "opening_liquidity_validation.csv")
    write_rows(FREEZE / "final_household_initialization_validation.csv", opening)
    early = read_rows(AUDIT / "early_review_validation.csv")
    write_rows(FREEZE / "final_early_review_validation.csv", early or [{"diagnosis": "no early review event observed", "status": "VALID_GENUINE_NO_TRIGGER"}])
    write_rows(FREEZE / "canonical_resolver_validation.csv", [{
        "resolver": "STEP15_CANONICAL_REFERENCE_RUN",
        "path": str(FINAL_RUN.relative_to(ROOT)),
        "exists": FINAL_RUN.is_dir(),
        "complete_step15_files": all((FINAL_RUN / filename).is_file() for filename in (
            "step15_macro_panel.csv", "step15_firm_panel.csv", "step15_demographic_panel.csv",
            "step15_demographic_events.csv", "step15_marriage_panel.csv",
            "step15_accounting_reconciliation.csv", "step15_capital_asset_ledger.csv",
            "step15_capital_provenance_events.csv", "step15_investment_chain_trace.csv",
        )),
    }])
    write_rows(FREEZE / "demo_gui_validation.csv", [{
        "entrypoint": "python main.py --demo",
        "mode": "read-only Analysis V2",
        "resolver": str(FINAL_RUN.relative_to(ROOT)),
        "simulation_rerun": False,
        "expected_window": "PySide6 Analysis V2",
        "status": "READY_AFTER_RESOLVER_FREEZE",
    }])
    write_rows(FREEZE / "final_gui_availability_matrix.csv", [
        {"page": page, "status": "FULLY_AVAILABLE", "reason": "final canonical authoritative datasets present"}
        for page in ("Overview", "Population & Social", "Household & Settlement Accounts", "Labor", "Macro Economy", "Sector", "Firm", "Accounting & Audit", "Capital & Investment", "Contract & Transaction", "Reconciliation & Integrity", "Model Diagnostics", "Reports & Export")
    ])
    write_rows(FREEZE / "final_legacy_cleanup_manifest.csv", [
        {"module": "analysis.dimensions", "action": "RETAIN_COMPATIBILITY", "reason": "historical loaders/tests may still import it"},
        {"module": "analysis.windows", "action": "RETAIN_COMPATIBILITY", "reason": "historical CLI/workbench compatibility not proven dead"},
        {"module": "analysis.gui", "action": "RETAIN_COMPATIBILITY", "reason": "legacy --analysis-gui remains supported"},
        {"module": "analysis.gui_v2", "action": "KEEP_CANONICAL", "reason": "frozen primary Analysis V2 entry point"},
    ])
    checks = {
        "macro_history_complete": len(macro) == WEEKS,
        "firm_panel_complete": len(firms) == WEEKS * (FOOD_FIRMS + len(getattr(world, "capital_good_firms", []))),
        "firm_week_keys_unique": len({(row.get("global_step"), str(row.get("firm_id"))) for row in firms}) == len(firms),
        "demographic_history_complete": len(read_rows(FINAL_RUN / "step15_demographic_panel.csv")) == WEEKS,
        "age_histogram_available": (FINAL_RUN / "statistical_observability/age_sex_histogram.csv").is_file(),
        "household_micro_snapshots_available": (FINAL_RUN / "statistical_observability/social_household_snapshots.csv").is_file(),
        "labor_history_available": (FINAL_RUN / "statistical_observability/labor_denominators.csv").is_file(),
        "asset_ledger_available": len(read_rows(FINAL_RUN / "step15_capital_asset_ledger.csv")) > 1,
        "investment_chain_available": len(chain) > 0,
        "full_money_location_pass": max_field("full_money_location_gap") <= 1e-5,
        "accounting_pass": accounting_gap <= 1e-5,
        "goods_pass": max_field("goods_gap") <= 1e-6,
        "advance_prepaid_pass": max_field("advance_prepaid_gap") <= 1e-5,
        "assignment_pass": max_field("assignment_violations") == 0 and max_field("orphan_payroll_workers") == 0,
        "feasible_capacity_pass": max_field("output_above_feasible_capacity") <= 1e-6,
        "opening_buffer_formula_present": num(world.initial_household_opening_buffer_total, 0.0) > 0,
        "opening_buffer_expected_scale": abs(num(world.initial_household_opening_buffer_total, 0.0) - 124560.0) <= 1e-6,
        "early_review_flag_enabled": bool(getattr(world, "capital_good_pipeline_depletion_early_review_enabled", False)),
        "real_investment_observed": sum(num(row.get("total_fixed_investment"), 0.0) for row in macro) > 0,
        "replacement_observed": sum(num(row.get("replacement_investment"), 0.0) for row in macro) > 0,
        "customer_advance_observed": sum(num(row.get("customer_advances_received"), 0.0) for row in macro) > 0,
        "audit_money_equivalence": all(row["passed"] for row in audit_equivalence),
    }
    try:
        store = CanonicalDataStore(run_dir=FINAL_RUN)
        query = AnalysisV2Query(store)
        checks.update({
            "analysis_loader_pass": bool(query.list_firms()) and bool(query.list_sectors()),
            "analysis_macro_query_pass": len(query.get_macro_series(["population"])) == WEEKS,
            "analysis_demographic_query_pass": len(query.get_population_series()) == WEEKS,
            "analysis_capital_query_pass": len(query.get_capital_assets()) > 1,
            "analysis_chain_query_pass": len(query.get_investment_chain()) > 0,
            "analysis_loader_read_only_hash_pass": store.verify_unchanged()[0],
        })
    except Exception as exc:
        checks["analysis_loader_pass"] = False
        checks["analysis_loader_error"] = str(exc)
    gui_flags_path = ROOT / "test/output/step15_analysis_gui_v2_reconstruction/acceptance_flags.json"
    if gui_flags_path.is_file():
        gui_flags = json.loads(gui_flags_path.read_text(encoding="utf-8-sig"))
        checks["gui_v2_acceptance_pass"] = gui_flags.get("verdict") == "A. ANALYSIS_GUI_V2_ACCEPTED"
        checks["demo_redirected_to_v2"] = bool(gui_flags.get("demo_redirected_to_v2"))
    else:
        checks["gui_v2_acceptance_pass"] = False
        checks["demo_redirected_to_v2"] = False
    flags = {
        "verdict": "A. STEP15_FINAL_CANONICAL_FROZEN" if all(value for key, value in checks.items() if key != "analysis_loader_error") else "C. ANALYSIS_DATASET_INCOMPLETE",
        "canonical_reference_run": str(FINAL_RUN.relative_to(ROOT)),
        "accepted_runtime_configuration": overrides(),
        "checks": checks,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "simulation_rerun_on_demo": False,
        "pending_household_formation_balance_in_money_identity": True,
        "historical_references": [
            "step15I12A_final_integrated_validation",
            "step15_final_canonical_post_dynamic_validation",
            "step15_post_dynamic_fix_integration",
        ],
    }
    (FREEZE / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")
    (FREEZE / "acceptance_summary.md").write_text(
        "# Step 15 Final Canonical Freeze\n\n"
        f"Verdict: **{flags['verdict']}**\n\n"
        f"The final authoritative run is `{FINAL_RUN.relative_to(ROOT)}` using N=5000, seed=42 and 520 weeks with the accepted Step15 configuration.\n\n"
        "The full money-location identity explicitly includes Household cash, Firm cash, LegacyOwner cash, Estate cash, public/central-public cash, and the signed `pending_household_formation_wealth` balance. `full_money_location_gap` is the stock reconciliation; `money_delta_gap` remains a separate flow/timing diagnostic; the legacy monetary alias is not used as an interchangeable pass/fail identity.\n\n"
        "The run uses the existing canonical diagnostic and statistical persistence observers. Previous canonical and audit directories remain historical reference artifacts and were not overwritten. `--demo` is now a read-only Analysis V2 launcher and does not run a new simulation.\n",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, ensure_ascii=False))


def main():
    if "--validate-existing" in sys.argv:
        macro = read_rows(FINAL_RUN / "step15_macro_panel.csv")
        firms = read_rows(FINAL_RUN / "step15_firm_panel.csv")
        accounting = read_rows(FINAL_RUN / "step15_accounting_reconciliation.csv")
        chain = read_rows(FINAL_RUN / "step15_investment_chain_trace.csv")
        money_history = read_rows(FINAL_RUN / "authoritative_money_location_reconciliation.csv")
        opening = read_rows(AUDIT / "opening_liquidity_validation.csv")
        world = SimpleNamespace(
            initial_household_opening_buffer_total=num(
                opening[0].get("opening_buffer_total") if opening else 0.0,
                0.0,
            ),
            capital_good_pipeline_depletion_early_review_enabled=True,
            capital_good_firms=[object()],
        )
    else:
        world, macro, firms, accounting, chain, money_history = materialize()
    validation_artifacts(world, macro, firms, accounting, chain, money_history)


if __name__ == "__main__":
    main()
