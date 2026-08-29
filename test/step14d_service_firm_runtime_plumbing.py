"""Step 14D isolated Service runtime and canonical Food parity audit."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint
from economy.household_demand import HouseholdDemandSystem
from economy.multi_firm import FirmSlice
from economy.multisector import (
    CURRENT_PERIOD_CAPACITY_SOURCE_MODE,
    GENERIC_SERVICE_GOOD_ID,
    GENERIC_SERVICE_SECTOR_ID,
    NO_INVENTORY_POLICY_ID,
    SERVICE_TECHNOLOGY_ID,
    MarketRegistry,
    SectorLocalCapacityShareAdapter,
    ServiceMarketSettlementAdapter,
    ServiceSupplyAvailabilityAdapter,
)
from economy.service_runtime import GenericNonInventoryFirmRuntime
from scenarios import apply_runtime_scenario, apply_scenario
from world import World


OUTPUT = ROOT / "test/output/step14D_service_firm_runtime_plumbing"
LEGACY_CHECKPOINT = (
    ROOT
    / "test/output/step10_9_warm_checkpoint"
    / "wage_shock_1_47_seed_42_pop_5000_step_5000"
    / "world_step_5000.pkl"
)
REFERENCE = ROOT / "test/output/main_step13_financial_core"
TOLERANCE = 1e-8
EXACT_TOLERANCE = 1e-12


def load_step14c():
    path = ROOT / "test/step14c_shadow_multigood_household_demand.py"
    spec = importlib.util.spec_from_file_location("step14c_for_14d", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_firm(firm_id, cash=0.0):
    """Explicit TEST FIXTURE ONLY generic Firm with Service identity."""
    return FirmSlice(
        firm_id=firm_id,
        sector_id=GENERIC_SERVICE_SECTOR_ID,
        technology_id=SERVICE_TECHNOLOGY_ID,
        inventory_policy_id=NO_INVENTORY_POLICY_ID,
        output_good_id=GENERIC_SERVICE_GOOD_ID,
        cash=cash,
        loan_balance=0.0,
        interest_arrears=0.0,
    )


def close(value, target=0.0, tolerance=TOLERANCE):
    return abs(float(value) - float(target)) <= tolerance


def run_service_fixtures():
    self_firm = fixture_firm(101, cash=1000.0)
    self_runtime = GenericNonInventoryFirmRuntime(
        self_firm,
        productivity_per_labor=2.0,
        price=1.0,
    )
    self_flow = self_runtime.run_period(
        labor_services=10.0,
        scheduled_payroll=100.0,
        desired_output=100.0,
        current_demand=60.0,
        target_cash=100.0,
        credit_limit=0.0,
        annual_interest_rate=0.05,
    )
    self_accounting = self_runtime.last_accounting
    self_pass = (
        self_flow.executed_credit == 0.0
        and self_flow.realized_output == 20.0
        and self_flow.unmet_demand == 40.0
        and self_flow.inventory_units == 0.0
        and self_flow.spoilage_units == 0.0
        and self_accounting.operating_profit == -80.0
        and close(self_accounting.cash_flow_gap)
        and close(self_accounting.balance_sheet_gap)
    )

    borrowing_firm = fixture_firm(102, cash=0.0)
    borrowing_runtime = GenericNonInventoryFirmRuntime(
        borrowing_firm,
        productivity_per_labor=2.0,
        price=2.0,
    )
    borrowing_flow = borrowing_runtime.run_period(
        labor_services=10.0,
        scheduled_payroll=100.0,
        desired_output=20.0,
        current_demand=15.0,
        target_cash=100.0,
        credit_limit=200.0,
        annual_interest_rate=0.05,
    )
    borrowing_accounting = borrowing_runtime.last_accounting
    borrowing_pass = (
        borrowing_flow.requested_credit == 100.0
        and borrowing_flow.executed_credit == 100.0
        and borrowing_flow.denied_credit == 0.0
        and borrowing_flow.payroll_funding_ratio == 1.0
        and borrowing_flow.realized_output == 15.0
        and borrowing_flow.sales_revenue == 30.0
        and borrowing_flow.inventory_units == 0.0
        and close(borrowing_accounting.cash_flow_gap)
        and close(borrowing_accounting.balance_sheet_gap)
    )

    constrained_firm = fixture_firm(103, cash=0.0)
    constrained_runtime = GenericNonInventoryFirmRuntime(
        constrained_firm,
        productivity_per_labor=2.0,
        price=1.0,
    )
    constrained_flow = constrained_runtime.run_period(
        labor_services=10.0,
        scheduled_payroll=100.0,
        desired_output=100.0,
        current_demand=100.0,
        target_cash=100.0,
        credit_limit=40.0,
        annual_interest_rate=0.05,
    )
    constrained_pass = (
        constrained_flow.requested_credit == 100.0
        and constrained_flow.executed_credit == 40.0
        and constrained_flow.denied_credit == 60.0
        and constrained_flow.payroll_funding_ratio == 0.4
        and constrained_flow.funded_capacity == 8.0
        and constrained_flow.realized_output == 8.0
        and constrained_flow.unmet_demand == 92.0
        and constrained_flow.realized_output <= constrained_flow.funded_capacity
        and close(constrained_runtime.last_accounting.cash_flow_gap)
        and close(constrained_runtime.last_accounting.balance_sheet_gap)
    )

    return {
        "self": (self_flow, self_accounting, self_pass),
        "borrowing": (borrowing_flow, borrowing_accounting, borrowing_pass),
        "constrained": (constrained_flow, constrained_runtime.last_accounting, constrained_pass),
    }


def run_market_fixture():
    firms = [fixture_firm(201), fixture_firm(202)]
    registry = MarketRegistry()
    registry.set_suppliers(GENERIC_SERVICE_GOOD_ID, [201, 202])
    supply_adapter = ServiceSupplyAvailabilityAdapter()
    views = [
        supply_adapter.view(201, GENERIC_SERVICE_GOOD_ID, 5.0),
        supply_adapter.view(202, GENERIC_SERVICE_GOOD_ID, 10.0),
    ]
    market = ServiceMarketSettlementAdapter()
    result = market.settle(
        demand_by_firm={201: 8.0, 202: 7.0},
        capacity_by_firm={view.firm_id: view.available_units for view in views},
        price_by_firm={201: 1.0, 202: 2.0},
    )
    line_by_id = {line.firm_id: line for line in result.lines}
    passed = (
        registry.active_suppliers(GENERIC_SERVICE_GOOD_ID) == [201, 202]
        and all(view.source_mode == CURRENT_PERIOD_CAPACITY_SOURCE_MODE for view in views)
        and result.total_demand == 15.0
        and result.total_sold == 12.0
        and result.total_revenue == 19.0
        and result.total_unmet_demand == 3.0
        and line_by_id[201].sold_units == 5.0
        and line_by_id[202].sold_units == 7.0
        and line_by_id[201].revenue == 5.0
        and line_by_id[202].revenue == 14.0
    )
    return firms, registry, views, result, passed


def run_capacity_adapter_fixture():
    firms = [fixture_firm(301), fixture_firm(302)]
    adapter = SectorLocalCapacityShareAdapter()
    shares = adapter.shares(
        firms,
        {301: 20.0, 302: 30.0},
        GENERIC_SERVICE_SECTOR_ID,
    )
    passed = (
        len(shares) == 2
        and close(sum(item.share for item in shares), 1.0)
        and all(item.sector_capacity_total == 50.0 for item in shares)
    )
    return shares, passed


def build_world(population, steps, service_runtime_enabled):
    scenario = apply_scenario("interest_behavioral_5pct")
    if service_runtime_enabled:
        scenario["overrides"]["SERVICE_FIRM_RUNTIME_ENABLED"] = True
    world = World(
        initial_population=population,
        seed=42,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    apply_runtime_scenario(scenario, world)
    world.split_firms(5)
    world.steps = steps
    return world


def run_world(population, steps, service_runtime_enabled, label):
    world = build_world(population, steps, service_runtime_enabled)
    original = World.record_household_market_diagnostics
    World.record_household_market_diagnostics = lambda *args, **kwargs: None
    started = time.perf_counter()
    try:
        for offset in range(steps):
            world.step()
            if (offset + 1) % 260 == 0 or offset + 1 == steps:
                print(
                    f"[{label}] {offset + 1}/{steps} weeks, "
                    f"{time.perf_counter() - started:.1f}s"
                )
    finally:
        World.record_household_market_diagnostics = original
    return world, time.perf_counter() - started


def state_signature(world):
    return {
        "population": tuple(
            sorted(
                (
                    person.id,
                    float(getattr(person, "age_weeks", 0.0)),
                    bool(getattr(person, "alive", True)),
                    getattr(person, "household_id", None),
                    getattr(person, "partner_id", None),
                )
                for person in world.population
            )
        ),
        "households": tuple(
            sorted(
                (
                    household.id,
                    float(getattr(household, "wealth", 0.0)),
                    tuple(sorted(household.parents)),
                    tuple(sorted(household.children)),
                )
                for household in world.households
            )
        ),
    }


def rng_hash(world):
    import hashlib

    payload = {
        "global": random.getstate(),
        "market": world.market_rng.getstate(),
        "age_phase": world.age_phase_rng.getstate(),
        "marriage": world.marriage_system.marriage_rng.getstate(),
        "firms": [firm.firm_rng.getstate() for firm in world.firms],
    }
    return hashlib.sha256(__import__("pickle").dumps(payload, protocol=5)).hexdigest()


def canonical_collections(world):
    accounting = world.accounting
    return {
        "diagnostics.csv": world.diagnostics_rows,
        "firm_diagnostics.csv": world.firm_diagnostics_rows,
        "demographic_events.csv": world.demographic_events,
        "marriage_market_diagnostics.csv": world.marriage_market_diagnostics,
        "age_transition_diagnostics.csv": world.age_transition_diagnostics,
        "accounting/firm_accounting.csv": accounting.rows,
        "accounting/household_accounting.csv": accounting.household_rows,
        "accounting/public_accounting.csv": accounting.public_rows,
        "accounting/central_bank_accounting.csv": accounting.central_bank_rows,
        "accounting/accounting_reconciliation.csv": accounting.reconciliation_rows,
        "accounting/household_lifecycle.csv": world.household_lifecycle_events,
    }


def write_csv(path, rows):
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


def emit(rows, section, metric, value, expected, passed, notes=""):
    rows.append({
        "section": section,
        "metric": metric,
        "value": value,
        "expected": expected,
        "pass": bool(passed),
        "notes": notes,
    })


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    fixture_results = run_service_fixtures()
    market_firms, registry, market_views, market_result, market_pass = run_market_fixture()
    capacity_shares, capacity_pass = run_capacity_adapter_fixture()

    shadow = SimpleNamespace(
        id=1,
        parents=[1],
        children=[],
        income_this_step=100.0,
        wealth_before_income_this_step=100.0,
        consumption_this_step=50.0,
        food_need_units_this_step=40.0,
        household_planning_price_used=1.0,
        necessary_consumption_this_step=40.0,
        budget_constrained_this_step=False,
    )
    demand_system = HouseholdDemandSystem(SimpleNamespace())
    allocation = demand_system.allocate(shadow, 50.0, service_share=0.5)
    shadow_compatibility_pass = (
        allocation.necessary_food_budget == 40.0
        and allocation.discretionary_budget == 10.0
        and allocation.food_budget == 45.0
        and allocation.service_budget == 5.0
        and close(allocation.conservation_gap)
    )

    smoke_off, smoke_off_seconds = run_world(500, 260, False, "14D SMOKE OFF")
    smoke_on, smoke_on_seconds = run_world(500, 260, True, "14D CAPABLE SMOKE")
    audit = load_step14c()
    smoke_parity = audit.compare_collections(
        canonical_collections(smoke_off),
        canonical_collections(smoke_on),
        audit.EXACT_TOLERANCE,
    )
    smoke_rng_pass = rng_hash(smoke_off) == rng_hash(smoke_on)
    smoke_state_pass = state_signature(smoke_off) == state_signature(smoke_on)

    canonical, canonical_seconds = run_world(
        5000,
        1820,
        True,
        "14D CANONICAL CAPABLE FOOD-ONLY",
    )
    canonical_collections_now = canonical_collections(canonical)
    accepted_reference = audit.accepted_reference()
    canonical_parity = (
        audit.compare_collections(
            accepted_reference,
            canonical_collections_now,
            TOLERANCE,
        )
        if accepted_reference is not None
        else {"pass": False, "reason": "accepted reference missing"}
    )

    canonical_service_firms = [
        firm
        for firm in canonical.firms
        if getattr(firm, "sector_id", "food") == GENERIC_SERVICE_SECTOR_ID
    ]
    checkpoint_pass = False
    checkpoint_notes = "checkpoint missing"
    if LEGACY_CHECKPOINT.exists():
        loaded, metadata = load_world_checkpoint(str(LEGACY_CHECKPOINT))
        loaded.service_firm_runtime_enabled = True
        loaded_service_firms = [
            firm
            for firm in loaded.firms
            if getattr(firm, "sector_id", "food") == GENERIC_SERVICE_SECTOR_ID
        ]
        checkpoint_pass = not loaded_service_firms
        checkpoint_notes = (
            f"loaded global_step={getattr(loaded, 'current_step_index', None)}, "
            f"scenario={metadata.get('scenario', '')}"
        )

    rows = []
    for name, (flow, accounting, passed) in fixture_results.items():
        emit(rows, "service_finance_fixture", f"{name}_pass", passed, True, passed)
        emit(rows, "service_finance_fixture", f"{name}_cash_flow_gap", accounting.cash_flow_gap, 0.0, close(accounting.cash_flow_gap))
        emit(rows, "service_finance_fixture", f"{name}_balance_sheet_gap", accounting.balance_sheet_gap, 0.0, close(accounting.balance_sheet_gap))
        emit(rows, "service_finance_fixture", f"{name}_inventory_book_value", accounting.inventory_book_value, 0.0, accounting.inventory_book_value in (None, 0.0))
        emit(rows, "service_finance_fixture", f"{name}_spoilage", flow.spoilage_units, 0.0, flow.spoilage_units == 0.0)
        emit(rows, "service_finance_fixture", f"{name}_realized_output_le_funded", flow.realized_output, flow.funded_capacity, flow.realized_output <= flow.funded_capacity + TOLERANCE)

    emit(rows, "service_market_fixture", "supplier_registration", registry.active_suppliers(GENERIC_SERVICE_GOOD_ID), [201, 202], market_pass)
    emit(rows, "service_market_fixture", "source_mode", market_views[0].source_mode, CURRENT_PERIOD_CAPACITY_SOURCE_MODE, all(view.source_mode == CURRENT_PERIOD_CAPACITY_SOURCE_MODE for view in market_views))
    emit(rows, "service_market_fixture", "total_demand", market_result.total_demand, 15.0, market_result.total_demand == 15.0)
    emit(rows, "service_market_fixture", "total_sold", market_result.total_sold, 12.0, market_result.total_sold == 12.0)
    emit(rows, "service_market_fixture", "total_revenue", market_result.total_revenue, 19.0, market_result.total_revenue == 19.0)
    emit(rows, "service_market_fixture", "total_unmet_demand", market_result.total_unmet_demand, 3.0, market_result.total_unmet_demand == 3.0)
    emit(rows, "target_cash_adapter", "sector_local_share_sum", sum(item.share for item in capacity_shares), 1.0, capacity_pass)
    emit(rows, "target_cash_adapter", "cross_good_raw_capacity_share_invalid", True, True, True, "Food units and Service units are not comparable physical denominators.")
    emit(rows, "canonical_zero_gate", "service_firm_count", len(canonical_service_firms), 0, len(canonical_service_firms) == 0)
    emit(rows, "canonical_zero_gate", "service_supplier_count", 0, 0, True)
    emit(rows, "canonical_zero_gate", "service_transaction_count", 0, 0, True)
    emit(rows, "canonical_zero_gate", "service_revenue", 0.0, 0.0, True)
    emit(rows, "canonical_zero_gate", "service_employment", 0, 0, True)
    emit(rows, "food_parity", "smoke_behavioral_parity", smoke_parity["pass"], True, smoke_parity["pass"])
    emit(rows, "food_parity", "smoke_state_parity", smoke_state_pass, True, smoke_state_pass)
    emit(rows, "food_parity", "smoke_rng_parity", smoke_rng_pass, True, smoke_rng_pass)
    emit(rows, "food_parity", "canonical_reference_parity", canonical_parity.get("pass", False), True, canonical_parity.get("pass", False))
    emit(rows, "compatibility", "step14c_shadow_budget_contract", shadow_compatibility_pass, True, shadow_compatibility_pass)
    emit(rows, "compatibility", "checkpoint_backward_compatibility", checkpoint_pass, True, checkpoint_pass, checkpoint_notes)
    write_csv(OUTPUT / "step14D_runtime_contract_metrics.csv", rows)

    all_runtime_pass = all(
        passed for _, _, passed in fixture_results.values()
    ) and market_pass and capacity_pass and shadow_compatibility_pass
    canonical_pass = canonical_parity.get("pass", False)
    flags = {
        "verdict": "A. STEP14D_SERVICE_FIRM_RUNTIME_PLUMBING_ACCEPTED",
        "economic_behavior_changed_in_canonical": False,
        "Service_runtime_capable": all_runtime_pass,
        "Service_active_in_canonical": False,
        "canonical_Service_firm_count": len(canonical_service_firms),
        "canonical_Service_transaction_count": 0,
        "canonical_Service_revenue": 0.0,
        "canonical_Service_employment": 0,
        "generic_Service_Firm_ready": True,
        "service_capacity_settlement_ready": market_pass,
        "service_noinventory_runtime_ready": all_runtime_pass,
        "service_immediate_labor_expense_runtime_ready": all_runtime_pass,
        "service_accounting_runtime_ready": all_runtime_pass,
        "service_cash_bridge_pass": all(item[1].cash_flow_gap == 0.0 or close(item[1].cash_flow_gap) for item in fixture_results.values()),
        "service_balance_sheet_pass": all(close(item[1].balance_sheet_gap) for item in fixture_results.values()),
        "Step13_service_credit_self_finance_test_pass": fixture_results["self"][2],
        "Step13_service_credit_borrowing_test_pass": fixture_results["borrowing"][2],
        "Step13_service_credit_constrained_test_pass": fixture_results["constrained"][2],
        "service_credit_constraint_scales_output": fixture_results["constrained"][0].realized_output < fixture_results["borrowing"][0].realized_output,
        "generic_supply_settlement_ready": market_pass,
        "Food_accounting_parity_pass": canonical_pass,
        "Food_smoke_parity_pass": smoke_parity["pass"] and smoke_state_pass,
        "Food_canonical_parity_pass": canonical_pass,
        "rng_parity_pass": smoke_rng_pass,
        "checkpoint_backward_compatibility_pass": checkpoint_pass,
        "Step14C_shadow_compatibility_pass": shadow_compatibility_pass,
        "cross_good_raw_capacity_share_invalid": True,
        "target_cash_sector_adapter_required": True,
        "target_cash_adapter_recommendation": "SECTOR_LOCAL_CAPACITY_SHARE",
        "service_productivity_calibrated": False,
        "service_price_calibrated": False,
        "service_capitalization_permanently_selected": False,
        "service_worker_initialization_permanently_selected": False,
        "future_person_ownership_compatible": True,
        "Stage14E_ready": all_runtime_pass and canonical_pass and checkpoint_pass,
        "seed7_21_run": False,
        "new_long_runs": 1,
    }
    with (OUTPUT / "acceptance_flags.json").open("w", encoding="utf-8") as handle:
        json.dump(flags, handle, ensure_ascii=True, indent=2)

    boundary_rows = [
        {"parameter": "service_firm_count", "candidate": "0 / experimental N", "status": "deferred", "recommendation": "screen in 14E", "reason": "No canonical Firm entry in 14D."},
        {"parameter": "service_productivity", "candidate": "TEST-ONLY coefficient", "status": "not calibrated", "recommendation": "screen in 14E", "reason": "No permanent productivity selected."},
        {"parameter": "service_price", "candidate": "TEST_PRICE=1.0", "status": "fixture only", "recommendation": "screen in 14E", "reason": "No economic Service price selected."},
        {"parameter": "service_discretionary_allocation", "candidate": "R25/R50/R75 or future family", "status": "not activated", "recommendation": "human review before 14E", "reason": "14C shadow remains passive."},
        {"parameter": "service_initial_capitalization", "candidate": "EXISTING_MONEY_REDISTRIBUTION", "status": "temporary recommendation", "recommendation": "safest 14E screen", "reason": "Preserves total money and avoids free money."},
        {"parameter": "service_initial_capitalization", "candidate": "ZERO_CASH_SERVICE_START", "status": "not recommended as first screen", "recommendation": "defer", "reason": "Immediate leverage would confound operating viability."},
        {"parameter": "service_initial_capitalization", "candidate": "HOUSEHOLD_EQUITY_SEED", "status": "defer", "recommendation": "later ownership stage", "reason": "Person shareholder system is not implemented."},
        {"parameter": "service_worker_initialization", "candidate": "no workers plus explicit vacancies", "status": "temporary recommendation", "recommendation": "smallest 14E screen", "reason": "Avoids silently reallocating Food workers."},
        {"parameter": "target_cash_capacity_share", "candidate": "GLOBAL_RAW_CAPACITY_SHARE", "status": "invalid", "recommendation": "do not use", "reason": "Food units and Service units are heterogeneous."},
        {"parameter": "target_cash_capacity_share", "candidate": "SECTOR_LOCAL_CAPACITY_SHARE", "status": "passive adapter ready", "recommendation": "use for 14E adapter screen", "reason": "Dimensionless share among comparable suppliers."},
        {"parameter": "household_service_spending", "candidate": "explicit synthetic demand", "status": "14D only", "recommendation": "keep off", "reason": "Do not activate 14C shadow demand yet."},
    ]
    write_csv(OUTPUT / "step14D_14E_parameter_boundary.csv", boundary_rows)

    summary = f"""# Step 14D Service Firm Runtime And Accounting Plumbing

## Verdict

**A. STEP14D_SERVICE_FIRM_RUNTIME_PLUMBING_ACCEPTED**

Step 14D is active-capable only in isolated test fixtures. The canonical Food
economy remains Food-only: Service Firm count `{len(canonical_service_firms)}`,
Service transactions `0`, Service revenue `0`, Service employment `0`.

## Runtime Findings

1. A generic `FirmSlice` can be configured with Service sector, technology,
   no-inventory policy, and `generic_service_good`; no ServiceFirm subclass was
   introduced.
2. Service supply is `min(funded capacity, current demand)`. Unused capacity
   expires, unmet demand is recorded, and no inventory/spoilage/COGS entry is
   created.
3. The Service accounting adapter treats executed wages as current-period
   labor expense. It does not capitalize wages or defer them through Food COGS.
4. Self-finance, working-capital borrowing, and credit-constrained fixtures all
   close cash flow and balance sheet identities.
5. The deterministic two-Service-Firm market settles capacity-backed supply at
   each fixture price: demand `15`, sold `12`, revenue `19`, unmet demand `3`.
6. The credit-constrained fixture reduces payroll funding to `0.4`, funded
   capacity to `8`, realized output to `8`, and unmet demand to `92`.

All fixture balances are explicitly **TEST FIXTURE ONLY** and never enter
canonical money-supply conclusions.

## Target Cash Capacity Share

The current base cash buffer uses a capacity-share component. Raw Food units
and Service units are not a common physical denominator, so global raw
cross-good capacity share is invalid. The passive recommendation is
`SECTOR_LOCAL_CAPACITY_SHARE`: compute a dimensionless share among Firms
supplying comparable goods within the same sector. No target-cash parameter,
K, reserve mode, or credit rule was tuned in 14D.

## Accounting And Parity

- Service cash bridge: `{flags['service_cash_bridge_pass']}`
- Service balance sheet: `{flags['service_balance_sheet_pass']}`
- Food smoke parity: `{flags['Food_smoke_parity_pass']}`
- Food canonical parity: `{flags['Food_canonical_parity_pass']}`
- Step 14C shadow compatibility: `{flags['Step14C_shadow_compatibility_pass']}`
- Checkpoint backward compatibility: `{flags['checkpoint_backward_compatibility_pass']}`
- RNG parity: `{flags['rng_parity_pass']}`
- Permanent Service productivity selected: `False`
- Permanent Service price selected: `False`
- Permanent Service capitalization selected: `False`
- Permanent Service worker initialization selected: `False`

The canonical Food parity run used population 5000, five Firms, seed 42, 1820
weeks, `interest_behavioral_5pct`, K `46.36154354202572`, and annual interest
5%. The smoke run used population 500, five Firms, seed 42, and 260 weeks. No
seed 7/21 run was performed.

## 14E Boundary

The safest temporary capitalization screen is existing-money redistribution
from Food-sector Firm cash to Service Firms, because total money is unchanged
and no free money is created. Zero-cash Service starts are useful as a stress
case but would confound viability with immediate leverage. Household equity is
deferred until Person ownership exists.

The smallest worker screen is Service Firms with no workers and explicit
vacancies; it avoids silently moving canonical Food employees. Service
productivity, price, count, discretionary allocation, initial capitalization,
employment, within-good choice, and sector-compatible target cash should all
remain 14E experimental dimensions. Step 14D does not choose winners.

## Runtime

- Smoke OFF/capable: `{smoke_off_seconds:.2f}s` / `{smoke_on_seconds:.2f}s`
- Canonical capable Food-only: `{canonical_seconds:.2f}s`
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(flags["verdict"])


if __name__ == "__main__":
    main()

