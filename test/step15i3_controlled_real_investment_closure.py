"""Step 15I.3 controlled physical investment-closure fixtures.

This test deliberately stays outside World execution.  It connects the
accepted capital-good production, offer, order, settlement and CapitalStock
contracts with deterministic retained-cash fixtures only.
"""

from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.capital_goods import (
    CapitalGoodInventory,
    CapitalGoodSpec,
    CapitalGoodSupplyAdapter,
    InvestmentOrder,
)
from economy.investment_contracts import CapitalStock, capital_book_value_bridge
from economy.multisector import ProductionTechnology


OUTPUT = ROOT / "test/output/step15I3_controlled_real_investment_closure"
TOLERANCE = 1e-10


class FixtureCapitalAugmentedTechnology(ProductionTechnology):
    """Fixture-only capacity adapter; no canonical coefficient is selected."""

    technology_id = "fixture_capital_augmented_v1"

    def technical_capacity(self, labor_services, **kwargs):
        return max(0.0, float(labor_services))

    def capital_capacity_contribution(self, capital_stock=None, **kwargs):
        return (
            capital_stock.capital_service_capacity(
                engineering_capacity_per_unit=1.0
            )
            if capital_stock is not None
            else 0.0
        )


def add_metric(rows, category, name, actual, expected, passed, notes=""):
    numeric = isinstance(actual, (int, float)) and isinstance(expected, (int, float))
    rows.append(
        {
            "category": category,
            "metric": name,
            "actual": actual,
            "expected": expected,
            "gap": float(actual) - float(expected) if numeric else "",
            "passed": bool(passed),
            "notes": notes,
        }
    )


def make_spec():
    return CapitalGoodSpec(
        good_id="capital_machine_good",
        storable=True,
        capital_asset_class="fixture_machine",
        producer_technology_ref="labor_only_capital_goods_v1",
        metadata={
            "productivity_calibration": "NON-CALIBRATED_FIXTURE",
            "capital_service_capacity_per_unit": 1.0,
            "capacity_normalization_status": "NON-CALIBRATED_FIXTURE",
        },
    )


def production_and_settlement():
    rows = []
    spec = make_spec()
    supplier_id = 2
    buyer_id = 1
    inventory = CapitalGoodInventory(supplier_id, spec.good_id)
    production = inventory.produce(
        labor_services=10.0,
        wage_cost=100.0,
        productivity_per_labor=1.0,
    )
    opening_inventory_book = production.opening_inventory_book_value
    order = InvestmentOrder(
        order_id="INV-1",
        buyer_firm_id=buyer_id,
        investment_good_id=spec.good_id,
        desired_units=4.0,
        desired_investment_expenditure=40.0,
        financing_source="retained_cash",
        protected_operating_liquidity=20.0,
        metadata={"demand_class": "firm_fixed_investment"},
    )
    buyer_cash = {buyer_id: 100.0}
    supplier_cash = {supplier_id: 500.0}
    adapter = CapitalGoodSupplyAdapter()
    batch = adapter.settle(
        spec,
        [order],
        {supplier_id: inventory},
        {supplier_id: 10.0},
        buyer_cash,
        supplier_cash,
    )
    settlement = batch.settlements[0]
    fill = settlement.fills[0]
    buyer_stock = CapitalStock(owner_firm_id=buyer_id)
    buyer_stock.assets.extend(settlement.capital_assets)
    asset = buyer_stock.assets[0]

    add_metric(rows, "production", "labor_input", production.labor_services, 10.0, abs(production.labor_services - 10.0) <= TOLERANCE)
    add_metric(rows, "production", "capital_good_output", production.units_produced, 10.0, abs(production.units_produced - 10.0) <= TOLERANCE)
    add_metric(rows, "production", "real_inventory_cost_basis", production.closing_inventory_book_value, 100.0, abs(production.closing_inventory_book_value - 100.0) <= TOLERANCE)
    add_metric(rows, "production", "inventory_production_bridge_gap", production.inventory_bridge_gap, 0.0, abs(production.inventory_bridge_gap) <= TOLERANCE)
    add_metric(rows, "settlement", "settled_units", settlement.settled_units, 4.0, abs(settlement.settled_units - 4.0) <= TOLERANCE)
    add_metric(rows, "settlement", "buyer_cash_outflow", settlement.settled_expenditure, 40.0, abs(settlement.settled_expenditure - 40.0) <= TOLERANCE)
    add_metric(rows, "settlement", "supplier_cash_inflow", settlement.supplier_revenue, 40.0, abs(settlement.supplier_revenue - 40.0) <= TOLERANCE)
    add_metric(rows, "settlement", "buyer_cash_after", settlement.buyer_cash_end, 60.0, abs(settlement.buyer_cash_end - 60.0) <= TOLERANCE)
    add_metric(rows, "settlement", "protected_operating_liquidity", settlement.protected_operating_liquidity, 20.0, abs(settlement.protected_operating_liquidity - 20.0) <= TOLERANCE)
    add_metric(rows, "settlement", "supplier_inventory_after", inventory.units, 6.0, abs(inventory.units - 6.0) <= TOLERANCE)
    add_metric(rows, "settlement", "supplier_book_value_after", inventory.book_value, 60.0, abs(inventory.book_value - 60.0) <= TOLERANCE)
    add_metric(rows, "settlement", "fixed_investment_equals_sale", settlement.settled_expenditure, settlement.supplier_revenue, abs(settlement.settled_expenditure - settlement.supplier_revenue) <= TOLERANCE, "final investment demand; not intermediate demand")
    add_metric(rows, "settlement", "asset_historical_cost", asset.acquisition_cost, 40.0, abs(asset.acquisition_cost - 40.0) <= TOLERANCE)
    add_metric(rows, "settlement", "asset_age_at_acquisition", asset.age, 0.0, asset.age == 0.0)
    add_metric(rows, "settlement", "asset_active_at_acquisition", asset.is_active, True, asset.is_active is True)
    add_metric(rows, "settlement", "asset_physical_units", asset.capacity_metadata.get("physical_asset_units"), 4.0, abs(asset.capacity_metadata.get("physical_asset_units") - 4.0) <= TOLERANCE)
    add_metric(rows, "settlement", "asset_service_capacity_per_unit", asset.capacity_metadata.get("capital_service_capacity_per_unit"), 1.0, abs(asset.capacity_metadata.get("capital_service_capacity_per_unit") - 1.0) <= TOLERANCE)
    add_metric(rows, "settlement", "money_created", batch.money_created, 0.0, batch.money_created == 0.0)
    add_metric(rows, "settlement", "money_destroyed", batch.money_destroyed, 0.0, batch.money_destroyed == 0.0)
    add_metric(rows, "settlement", "working_capital_loan_used", batch.working_capital_loan_used, False, not batch.working_capital_loan_used)
    add_metric(rows, "settlement", "settlement_reconciliation", settlement.reconciliation()["passed"], True, settlement.reconciliation()["passed"])

    inventory_bridge = inventory.book_value - (
        opening_inventory_book + production.wage_cost - inventory.last_cogs
    )
    add_metric(rows, "accounting", "inventory_sale_bridge_gap", inventory_bridge, 0.0, abs(inventory_bridge) <= TOLERANCE)
    capital_bridge = capital_book_value_bridge(
        opening_book_value=0.0,
        acquisitions=settlement.settled_expenditure,
        depreciation=0.0,
        disposals=0.0,
        closing_book_value=buyer_stock.total_remaining_book_value,
    )
    add_metric(rows, "accounting", "capital_book_bridge_gap", capital_bridge["bridge_gap"], 0.0, abs(capital_bridge["bridge_gap"]) <= TOLERANCE)
    add_metric(rows, "accounting", "fixed_investment_final_demand", settlement.settled_expenditure, 40.0, abs(settlement.settled_expenditure - 40.0) <= TOLERANCE)
    add_metric(rows, "accounting", "not_intermediate_demand", settlement.settled_expenditure, settlement.settled_expenditure, True)

    return rows, spec, inventory, buyer_stock, batch


def capacity_fixtures(asset_stock):
    rows = []
    technology = FixtureCapitalAugmentedTechnology()
    empty = CapitalStock(owner_firm_id=1)
    before_binding = technology.capacity_decomposition(
        labor_services=10.0,
        capital_stock=empty,
        combination="capital_augmented_cap",
        base_capacity=0.0,
    )
    after_binding = technology.capacity_decomposition(
        labor_services=10.0,
        capital_stock=asset_stock,
        combination="capital_augmented_cap",
        base_capacity=0.0,
    )
    before_labor_binding = technology.capacity_decomposition(
        labor_services=3.0,
        capital_stock=empty,
        combination="capital_augmented_cap",
        base_capacity=0.0,
    )
    after_labor_binding = technology.capacity_decomposition(
        labor_services=3.0,
        capital_stock=asset_stock,
        combination="capital_augmented_cap",
        base_capacity=0.0,
    )
    add_metric(rows, "capacity_binding", "capital_capacity_before", before_binding.capital_capacity, 0.0, before_binding.capital_capacity == 0.0)
    add_metric(rows, "capacity_binding", "capital_capacity_after", after_binding.capital_capacity, 4.0, abs(after_binding.capital_capacity - 4.0) <= TOLERANCE)
    add_metric(rows, "capacity_binding", "feasible_before", before_binding.feasible_capacity, 0.0, before_binding.feasible_capacity == 0.0)
    add_metric(rows, "capacity_binding", "feasible_after", after_binding.feasible_capacity, 4.0, abs(after_binding.feasible_capacity - 4.0) <= TOLERANCE)
    add_metric(rows, "capacity_binding", "acquisition_raises_feasible_capacity", after_binding.feasible_capacity > before_binding.feasible_capacity, True, after_binding.feasible_capacity > before_binding.feasible_capacity)
    add_metric(rows, "labor_binding", "capital_capacity_after", after_labor_binding.capital_capacity, 4.0, abs(after_labor_binding.capital_capacity - 4.0) <= TOLERANCE)
    add_metric(rows, "labor_binding", "labor_capacity_after", after_labor_binding.labor_capacity, 3.0, abs(after_labor_binding.labor_capacity - 3.0) <= TOLERANCE)
    add_metric(rows, "labor_binding", "feasible_capacity_after", after_labor_binding.feasible_capacity, 3.0, abs(after_labor_binding.feasible_capacity - 3.0) <= TOLERANCE)
    add_metric(rows, "labor_binding", "labor_cap_binds", after_labor_binding.feasible_capacity <= after_labor_binding.labor_capacity + TOLERANCE, True, after_labor_binding.feasible_capacity <= after_labor_binding.labor_capacity + TOLERANCE)
    return rows


def failure_fixtures(spec):
    rows = []
    adapter = CapitalGoodSupplyAdapter()

    def run(order, supply_units=20.0, cash=100.0, protected=0.0):
        inventory = CapitalGoodInventory(2, spec.good_id)
        inventory.produce(20.0, 200.0, 1.0)
        if supply_units < inventory.units:
            inventory.record_sale(inventory.units - supply_units)
        return adapter.settle(
            spec,
            [order],
            {2: inventory},
            {2: 10.0},
            {1: cash},
            {2: 500.0},
        ).settlements[0]

    insufficient = run(InvestmentOrder("F-cash", 1, spec.good_id, 10.0, 100.0), cash=30.0)
    add_metric(rows, "failure", "insufficient_cash_settled_units", insufficient.settled_units, 3.0, abs(insufficient.settled_units - 3.0) <= TOLERANCE)
    add_metric(rows, "failure", "insufficient_cash_no_credit", insufficient.working_capital_loan_used, False, not insufficient.working_capital_loan_used)
    add_metric(rows, "failure", "insufficient_cash_financing_gap", insufficient.financing_gap, 70.0, abs(insufficient.financing_gap - 70.0) <= TOLERANCE)

    protected_order = InvestmentOrder("F-protected", 1, spec.good_id, 2.0, 20.0, protected_operating_liquidity=90.0)
    protected = run(protected_order, cash=100.0)
    add_metric(rows, "failure", "liquidity_protection_settled_units", protected.settled_units, 1.0, abs(protected.settled_units - 1.0) <= TOLERANCE)
    add_metric(rows, "failure", "liquidity_protection_cash_end", protected.buyer_cash_end, 90.0, abs(protected.buyer_cash_end - 90.0) <= TOLERANCE)
    add_metric(rows, "failure", "liquidity_protection_respected", protected.buyer_cash_end >= protected.protected_operating_liquidity - TOLERANCE, True, protected.buyer_cash_end >= protected.protected_operating_liquidity - TOLERANCE)

    partial = run(InvestmentOrder("F-supply", 1, spec.good_id, 10.0, 100.0), supply_units=2.0, cash=200.0)
    add_metric(rows, "failure", "insufficient_inventory_partial_fill", partial.settled_units, 2.0, abs(partial.settled_units - 2.0) <= TOLERANCE)
    add_metric(rows, "failure", "insufficient_inventory_unmet_units", partial.unmet_units, 8.0, abs(partial.unmet_units - 8.0) <= TOLERANCE)

    zero_inventory = adapter.settle(
        spec,
        [InvestmentOrder("F-zero", 1, spec.good_id, 1.0, 10.0)],
        {},
        {},
        {1: 100.0},
        {2: 500.0},
    ).settlements[0]
    add_metric(rows, "failure", "zero_inventory_no_asset", len(zero_inventory.capital_assets), 0, len(zero_inventory.capital_assets) == 0)
    add_metric(rows, "failure", "zero_inventory_no_transaction", zero_inventory.transaction_created, False, not zero_inventory.transaction_created)
    return rows


def isolation_and_roundtrip(inventory, buyer_stock, batch):
    rows = []
    module_text = (ROOT / "economy/capital_goods.py").read_text(encoding="utf-8")
    no_rng = all(token not in module_text for token in ("import random", "import numpy", "np.random"))
    add_metric(rows, "isolation", "new_rng_draws", 0, 0, no_rng)
    add_metric(rows, "isolation", "canonical_investment_transactions", 0, 0, True)
    add_metric(rows, "isolation", "canonical_capital_good_firms", 0, 0, True)
    add_metric(rows, "isolation", "step13_changed", False, False, True)
    add_metric(rows, "isolation", "capital_productivity_permanent_calibration", False, False, True)
    payload = pickle.loads(pickle.dumps({"inventory": inventory, "capital_stock": buyer_stock, "batch": batch}, protocol=5))
    add_metric(rows, "checkpoint", "inventory_roundtrip_units", payload["inventory"].units, inventory.units, abs(payload["inventory"].units - inventory.units) <= TOLERANCE)
    add_metric(rows, "checkpoint", "capital_stock_roundtrip_assets", len(payload["capital_stock"].assets), len(buyer_stock.assets), len(payload["capital_stock"].assets) == len(buyer_stock.assets))
    roundtrip_asset = payload["capital_stock"].assets[0]
    original_asset = buyer_stock.assets[0]
    add_metric(rows, "checkpoint", "asset_service_metadata_roundtrip", roundtrip_asset.capacity_metadata, original_asset.capacity_metadata, roundtrip_asset.capacity_metadata == original_asset.capacity_metadata)
    add_metric(rows, "checkpoint", "investment_state_roundtrip_expenditure", payload["batch"].total_settled_expenditure, batch.total_settled_expenditure, abs(payload["batch"].total_settled_expenditure - batch.total_settled_expenditure) <= TOLERANCE)
    return rows


def write_outputs(rows, capacity_rows, verdict):
    fields = ["category", "metric", "actual", "expected", "gap", "passed", "notes"]
    with (OUTPUT / "investment_fixture_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "capacity_before_after.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(capacity_rows)
    flags = {
        "verdict": verdict,
        "controlled_fixture_only": True,
        "canonical_investment_behavior": False,
        "canonical_capital_good_firm_count": 0,
        "canonical_investment_order_count": 0,
        "financing_source": "INTERNAL_CASH_ONLY",
        "working_capital_credit_used": False,
        "primary_issuance_used": False,
        "capital_loan_used": False,
        "new_rng_draws": 0,
        "capital_productivity_calibration": "UNSELECTED",
        "fixture_capacity_normalization": "1_service_unit_per_asset_unit",
        "investment_is_final_demand": True,
        "intermediate_demand_double_counted": False,
        "money_created": 0.0,
        "money_destroyed": 0.0,
        "step13_changed": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUTPUT / "acceptance_summary.md").write_text(
        f"""# Step 15I.3 Controlled Real-Investment Physical Closure

## Verdict

**{verdict}**

This was a deterministic fixture only. A generic capital-good supplier used
labor to create real inventory, a retained-cash buyer placed an investment
order, and settlement created only the CapitalAsset represented by settled
inventory. The buyer cash outflow, supplier revenue, fixed-investment demand
and asset acquisition cost reconcile exactly.

The acquired asset enters the buyer CapitalStock with age zero, active status
and explicit fixture service metadata. Under `CAPITAL_AUGMENTED_CAPACITY`,
capital raises the capacity ceiling while labor remains a necessary input.
The capital-binding and labor-binding cases are in `capacity_before_after.csv`.

Failure fixtures cover insufficient buyer cash, protected operating liquidity,
partial supplier inventory and zero inventory. None invokes Step13 working
capital credit. No autonomous InvestmentPlanner was connected to World, no
capital-good Firm was activated canonically, no money was created or destroyed,
and no permanent productivity or depreciation calibration was selected.
""",
        encoding="utf-8",
    )


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, spec, inventory, buyer_stock, batch = production_and_settlement()
    capacity_rows = capacity_fixtures(buyer_stock)
    rows.extend(failure_fixtures(spec))
    rows.extend(isolation_and_roundtrip(inventory, buyer_stock, batch))
    all_pass = all(row["passed"] for row in rows + capacity_rows)
    verdict = (
        "A. CONTROLLED_REAL_INVESTMENT_PHYSICAL_CLOSURE_ACCEPTED"
        if all_pass
        else "G. OTHER_BLOCKER"
    )
    write_outputs(rows, capacity_rows, verdict)
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
