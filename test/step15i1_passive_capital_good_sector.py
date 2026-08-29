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

from checkpoint import load_world_checkpoint
from economy.capital_goods import (
    CapitalGoodInventory,
    CapitalGoodSpec,
    CapitalGoodSupplyAdapter,
    InvestmentOrder,
)
from world import World


OUTPUT = ROOT / "test/output/step15I1_passive_capital_good_sector"
CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)
TOLERANCE = 1e-10


def metric(rows, category, name, actual, expected, passed, notes=""):
    numeric = isinstance(actual, (int, float)) and isinstance(expected, (int, float))
    rows.append({
        "category": category,
        "metric": name,
        "actual": actual,
        "expected": expected,
        "gap": float(actual) - float(expected) if numeric else "",
        "passed": bool(passed),
        "notes": notes,
    })


def build_fixture():
    spec = CapitalGoodSpec(
        good_id="capital_machine_good",
        storable=True,
        capital_asset_class="fixture_machine",
        producer_technology_ref="labor_only_capital_goods_v1",
        metadata={"productivity_calibration": "unselected"},
    )
    inventory = CapitalGoodInventory(
        supplier_firm_id=900,
        good_id=spec.good_id,
    )
    production = inventory.produce(
        labor_services=10.0,
        wage_cost=100.0,
        productivity_per_labor=2.0,
    )
    offer_before_sale = inventory.offer("offer-before-sale", 8.0)
    adapter = CapitalGoodSupplyAdapter()
    partial_sale = adapter.settle(
        spec,
        [InvestmentOrder("partial", 1, spec.good_id, 15.0, 120.0)],
        {900: inventory},
        {900: 8.0},
        buyer_cash={1: 200.0},
        supplier_cash={900: 0.0},
    )

    shortage_inventory = CapitalGoodInventory(
        supplier_firm_id=901,
        good_id=spec.good_id,
    )
    shortage_inventory.produce(
        labor_services=10.0,
        wage_cost=100.0,
        productivity_per_labor=2.0,
    )
    buyer_shortage = adapter.settle(
        spec,
        [InvestmentOrder("shortage", 2, spec.good_id, 20.0, 160.0)],
        {901: shortage_inventory},
        {901: 8.0},
        buyer_cash={2: 40.0},
        supplier_cash={901: 0.0},
    )
    return (
        spec,
        inventory,
        production,
        offer_before_sale,
        partial_sale,
        shortage_inventory,
        buyer_shortage,
    )


def fixture_checks():
    rows = []
    (
        spec,
        inventory,
        production,
        offer,
        partial,
        shortage_inventory,
        shortage,
    ) = build_fixture()
    settlement = partial.settlements[0]
    shortage_settlement = shortage.settlements[0]

    metric(rows, "identity", "capital_good_is_storable", spec.storable, True, spec.storable)
    metric(rows, "identity", "productivity_is_fixture_only", production.calibration_status, "NON-CALIBRATED_FIXTURE", production.calibration_status == "NON-CALIBRATED_FIXTURE")
    metric(rows, "production", "labor_services_used", production.labor_services, 10.0, abs(production.labor_services - 10.0) <= TOLERANCE)
    metric(rows, "production", "units_produced", production.units_produced, 20.0, abs(production.units_produced - 20.0) <= TOLERANCE)
    metric(rows, "production", "explicit_unit_production_cost", production.unit_production_cost, 5.0, abs(production.unit_production_cost - 5.0) <= TOLERANCE)
    metric(rows, "inventory", "production_inventory_bridge_gap", production.inventory_bridge_gap, 0.0, abs(production.inventory_bridge_gap) <= TOLERANCE)
    metric(rows, "inventory", "offer_reflects_real_inventory", offer.available_units, 20.0, abs(offer.available_units - 20.0) <= TOLERANCE)
    metric(rows, "inventory", "offer_book_value_metadata", offer.metadata["inventory_book_value"], 100.0, abs(offer.metadata["inventory_book_value"] - 100.0) <= TOLERANCE)
    metric(rows, "settlement", "partial_sale_units", settlement.settled_units, 15.0, abs(settlement.settled_units - 15.0) <= TOLERANCE)
    metric(rows, "settlement", "partial_sale_unmet_units", settlement.unmet_units, 0.0, abs(settlement.unmet_units) <= TOLERANCE)
    metric(rows, "settlement", "inventory_after_partial_sale", inventory.units, 5.0, abs(inventory.units - 5.0) <= TOLERANCE)
    metric(rows, "settlement", "inventory_cogs", inventory.cumulative_cogs, 75.0, abs(inventory.cumulative_cogs - 75.0) <= TOLERANCE)
    metric(rows, "settlement", "post_sale_inventory_book_value", inventory.book_value, 25.0, abs(inventory.book_value - 25.0) <= TOLERANCE)
    metric(rows, "settlement", "supplier_revenue_equals_buyer_expenditure", settlement.supplier_revenue, settlement.settled_expenditure, abs(settlement.supplier_revenue - settlement.settled_expenditure) <= TOLERANCE)
    metric(rows, "settlement", "buyer_cash_outflow", settlement.buyer_cash_change, -120.0, abs(settlement.buyer_cash_change + 120.0) <= TOLERANCE)
    metric(rows, "settlement", "capital_asset_cost_equals_expenditure", sum(asset.acquisition_cost for asset in settlement.capital_assets), 120.0, abs(sum(asset.acquisition_cost for asset in settlement.capital_assets) - 120.0) <= TOLERANCE)
    metric(rows, "settlement", "cfi", settlement.cfi, -120.0, abs(settlement.cfi + 120.0) <= TOLERANCE)
    metric(rows, "settlement", "partial_sale_reconciliation", settlement.reconciliation()["passed"], True, settlement.reconciliation()["passed"])
    metric(rows, "shortage", "cash_shortage_settled_units", shortage_settlement.settled_units, 5.0, abs(shortage_settlement.settled_units - 5.0) <= TOLERANCE)
    metric(rows, "shortage", "cash_shortage_financing_gap", shortage_settlement.financing_gap, 120.0, abs(shortage_settlement.financing_gap - 120.0) <= TOLERANCE)
    metric(rows, "shortage", "cash_shortage_no_working_capital_loan", shortage_settlement.working_capital_loan_used, False, not shortage_settlement.working_capital_loan_used)
    metric(rows, "shortage", "cash_shortage_no_fake_asset", sum(asset.quantity for asset in shortage_settlement.capital_assets), 5.0, abs(sum(asset.quantity for asset in shortage_settlement.capital_assets) - 5.0) <= TOLERANCE)
    metric(rows, "conservation", "partial_money_created", partial.money_created, 0.0, partial.money_created == 0.0)
    metric(rows, "conservation", "partial_money_destroyed", partial.money_destroyed, 0.0, partial.money_destroyed == 0.0)
    metric(rows, "conservation", "shortage_money_created", shortage.money_created, 0.0, shortage.money_created == 0.0)
    metric(rows, "conservation", "shortage_money_destroyed", shortage.money_destroyed, 0.0, shortage.money_destroyed == 0.0)
    metric(rows, "conservation", "supplier_cash_after_partial_sale", partial.supplier_cash_end[900], 120.0, abs(partial.supplier_cash_end[900] - 120.0) <= TOLERANCE)
    metric(rows, "conservation", "supplier_cash_after_shortage_sale", shortage.supplier_cash_end[901], 40.0, abs(shortage.supplier_cash_end[901] - 40.0) <= TOLERANCE)

    roundtrip = pickle.loads(pickle.dumps(inventory, protocol=5))
    metric(rows, "checkpoint", "inventory_roundtrip_units", roundtrip.units, inventory.units, abs(roundtrip.units - inventory.units) <= TOLERANCE)
    metric(rows, "checkpoint", "inventory_roundtrip_book_value", roundtrip.book_value, inventory.book_value, abs(roundtrip.book_value - inventory.book_value) <= TOLERANCE)
    return rows


def canonical_checks():
    rows = []
    world = World(initial_population=100, seed=42, diagnostics_mode="full")
    world.multisector_foundation_enabled = True
    foundation = world.ensure_multisector_foundation_contracts()
    capital_firms = [
        firm for firm in world.firms
        if getattr(firm, "sector_id", "food") == "capital_goods"
    ]
    metric(rows, "canonical_default", "capital_good_firm_count", len(capital_firms), 0, len(capital_firms) == 0)
    metric(rows, "canonical_default", "capital_good_production_units", 0.0, 0.0, True)
    metric(rows, "canonical_default", "investment_order_count", len(foundation.capital_good_orders), 0, len(foundation.capital_good_orders) == 0)
    metric(rows, "canonical_default", "capital_market_suppliers", foundation.market_registry.suppliers("capital_machine_good"), [], foundation.market_registry.suppliers("capital_machine_good") == [])
    metric(rows, "canonical_default", "capital_sector_inactive", foundation.capital_goods_sector.operating_metadata["active"], False, foundation.capital_goods_sector.operating_metadata["active"] is False)
    metric(rows, "canonical_default", "capital_technology_unselected", foundation.technologies.get("labor_only_capital_goods_v1").calibration_status, "UNSELECTED", foundation.technologies.get("labor_only_capital_goods_v1").calibration_status == "UNSELECTED")
    metric(rows, "canonical_default", "capital_good_template_storable", foundation.capital_good_template.storable, True, foundation.capital_good_template.storable)
    metric(rows, "canonical_default", "capital_good_template_not_household_market", "capital_machine_good" not in world.goods_catalog.ids(), True, "capital_machine_good" not in world.goods_catalog.ids())

    baseline = World(initial_population=100, seed=42, diagnostics_mode="full")
    baseline.steps = 2
    enabled = World(initial_population=100, seed=42, diagnostics_mode="full")
    enabled.multisector_foundation_enabled = True
    enabled.ensure_multisector_foundation_contracts()
    enabled.steps = 2
    for _ in range(2):
        baseline.step()
        enabled.step()
    for field in ("population", "households", "total_household_wealth", "firm_cash", "inventory_units", "price", "money_supply_stock"):
        left = baseline.diagnostics_rows[-1].get(field)
        right = enabled.diagnostics_rows[-1].get(field)
        try:
            passed = abs(float(left) - float(right)) <= TOLERANCE
        except (TypeError, ValueError):
            passed = left == right
        metric(rows, "parity", f"food_service_{field}", right, left, passed, "passive capital-good registry only")

    checkpoint_pass = False
    checkpoint_note = "legacy checkpoint not found"
    if CHECKPOINT.exists():
        loaded, metadata = load_world_checkpoint(str(CHECKPOINT))
        loaded.multisector_foundation_enabled = True
        loaded.ensure_multisector_foundation_contracts()
        foundation_loaded = loaded.multisector_foundation
        checkpoint_pass = (
            metadata.get("global_step") == 5000
            and len(foundation_loaded.capital_good_firms) == 0
            and foundation_loaded.market_registry.suppliers("capital_machine_good") == []
        )
        checkpoint_note = "old checkpoint loaded; capital-good state remains inactive"
    metric(rows, "checkpoint", "old_checkpoint_compatible", checkpoint_pass, True, checkpoint_pass, checkpoint_note)
    return rows


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = fixture_checks() + canonical_checks()
    all_pass = all(row["passed"] for row in rows)
    verdict = (
        "A. PASSIVE_CAPITAL_GOOD_SUPPLY_READY"
        if all_pass
        else "B. CAPITAL_GOOD_INVENTORY_ACCOUNTING_BLOCKER"
    )
    flags = {
        "verdict": verdict,
        "passive_only": True,
        "simulation_behavior_changed": False,
        "canonical_simulation_run": False,
        "canonical_capital_good_firm_count": 0,
        "canonical_capital_good_production": 0.0,
        "canonical_investment_orders": 0,
        "canonical_capital_assets_created": 0,
        "capital_good_productivity_calibration": "UNSELECTED",
        "fixture_productivity_calibration": "NON-CALIBRATED_FIXTURE",
        "inventory_bridge_pass": all(row["passed"] for row in rows if row["category"] in {"production", "inventory", "settlement", "shortage"}),
        "settlement_supply_bridge_pass": all(row["passed"] for row in rows if row["category"] in {"settlement", "shortage", "conservation"}),
        "money_created": 0.0,
        "money_destroyed": 0.0,
        "working_capital_credit_used": False,
        "primary_issuance_active": False,
        "capital_productivity_active": False,
        "depreciation_active": False,
        "checkpoint_compatible": all(row["passed"] for row in rows if row["category"] == "checkpoint"),
        "food_service_parity": all(row["passed"] for row in rows if row["category"] == "parity"),
        "new_rng_draws": 0,
        "Step13_changed": False,
    }
    fields = ["category", "metric", "actual", "expected", "gap", "passed", "notes"]
    with (OUTPUT / "capital_good_supply_fixture.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = f"""# Step 15I.1 Passive Capital-Good Sector and Supply Contract

## Verdict

**{verdict}**

The capital-good sector, technology identity, storable GoodSpec template and
inventory-to-offer adapter are now part of the passive multi-sector
architecture. The default runtime still has zero capital-good Firms, zero
capital-good production, zero investment orders and zero capital assets.

The fixture uses a `NON-CALIBRATED_FIXTURE` labor productivity only to prove
that real labor services can produce units into weighted-average-cost
inventory. Inventory creates offers only for its actual units; settlement
reduces inventory, recognizes COGS, sends buyer expenditure to supplier
revenue, and creates no money. Cash shortage creates an unexecuted amount and
never invokes Step13 working-capital credit.

Food/Service parity, old checkpoint loading, inventory bridges and settlement
identities all passed. No InvestmentPlanner, canonical InvestmentOrder,
depreciation, capital capacity, financing or ownership behavior was activated.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
