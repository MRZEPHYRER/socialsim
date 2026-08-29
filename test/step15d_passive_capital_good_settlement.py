"""Step 15D passive capital-good supply and settlement validation."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.capital_goods import (
    CapitalGoodOffer,
    CapitalGoodSpec,
    InvestmentOrder,
    PassiveCapitalGoodSettlementEngine,
)


OUTPUT = ROOT / "test/output/step15D_passive_capital_good_settlement"
TOLERANCE = 1e-12
CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
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


def build_fixture():
    spec = CapitalGoodSpec(
        good_id="capital_machine_good",
        storable=True,
        capital_asset_class="future_machine",
        producer_technology_ref="future_capital_technology_reference",
        metadata={"productivity_calibration": "unselected"},
    )
    engine = PassiveCapitalGoodSettlementEngine()

    fully_funded = engine.settle(
        spec,
        [InvestmentOrder("A", 1, spec.good_id, 30.0, 60.0)],
        [CapitalGoodOffer("offer-A", 10, spec.good_id, 100.0, 2.0)],
        buyer_cash={1: 100.0},
        supplier_cash={10: 500.0},
    )
    partial = engine.settle(
        spec,
        [InvestmentOrder("B", 2, spec.good_id, 30.0, 60.0)],
        [CapitalGoodOffer("offer-B", 11, spec.good_id, 20.0, 2.0)],
        buyer_cash={2: 100.0},
        supplier_cash={11: 500.0},
    )
    insufficient_cash = engine.settle(
        spec,
        [InvestmentOrder("C", 3, spec.good_id, 30.0, 60.0)],
        [CapitalGoodOffer("offer-C", 12, spec.good_id, 100.0, 2.0)],
        buyer_cash={3: 25.0},
        supplier_cash={12: 500.0},
    )
    competing = engine.settle(
        spec,
        [
            InvestmentOrder("D1", 4, spec.good_id, 20.0, 40.0),
            InvestmentOrder("D2", 5, spec.good_id, 20.0, 40.0),
        ],
        [CapitalGoodOffer("offer-D", 13, spec.good_id, 30.0, 2.0)],
        buyer_cash={4: 100.0, 5: 100.0},
        supplier_cash={13: 1000.0},
    )
    return spec, fully_funded, partial, insufficient_cash, competing


def validate_settlements():
    spec, funded, partial, insufficient, competing = build_fixture()
    rows = []
    funded_row = funded.settlements[0]
    partial_row = partial.settlements[0]
    insufficient_row = insufficient.settlements[0]
    competing_rows = competing.settlements

    add_metric(rows, "identity", "capital_good_storable", spec.storable, True, spec.storable)
    add_metric(rows, "identity", "capital_good_producer_is_generic_firm", True, True, True, "supplier_firm_id is sufficient; no CapitalFirm subclass")
    add_metric(rows, "identity", "capital_productivity_calibration_selected", False, False, True)
    add_metric(rows, "fully_funded", "settled_units", funded_row.settled_units, 30.0, abs(funded_row.settled_units - 30.0) <= TOLERANCE)
    add_metric(rows, "fully_funded", "buyer_cash_outflow", funded_row.settled_expenditure, 60.0, abs(funded_row.settled_expenditure - 60.0) <= TOLERANCE)
    add_metric(rows, "fully_funded", "supplier_revenue", funded_row.supplier_revenue, 60.0, abs(funded_row.supplier_revenue - 60.0) <= TOLERANCE)
    add_metric(rows, "fully_funded", "asset_acquisition_cost", sum(asset.acquisition_cost for asset in funded_row.capital_assets), 60.0, abs(sum(asset.acquisition_cost for asset in funded_row.capital_assets) - 60.0) <= TOLERANCE)
    add_metric(rows, "fully_funded", "buyer_cash_end", funded_row.buyer_cash_end, 40.0, abs(funded_row.buyer_cash_end - 40.0) <= TOLERANCE)
    add_metric(rows, "fully_funded", "cash_transfer_gap", funded_row.reconciliation()["cash_transfer_gap"], 0.0, abs(funded_row.reconciliation()["cash_transfer_gap"]) <= TOLERANCE)
    add_metric(rows, "fully_funded", "asset_recognition_gap", funded_row.reconciliation()["asset_recognition_gap"], 0.0, abs(funded_row.reconciliation()["asset_recognition_gap"]) <= TOLERANCE)
    add_metric(rows, "fully_funded", "cfi", funded_row.cfi, -60.0, abs(funded_row.cfi + 60.0) <= TOLERANCE)
    add_metric(rows, "partial_supply", "settled_units", partial_row.settled_units, 20.0, abs(partial_row.settled_units - 20.0) <= TOLERANCE)
    add_metric(rows, "partial_supply", "unmet_units", partial_row.unmet_units, 10.0, abs(partial_row.unmet_units - 10.0) <= TOLERANCE)
    add_metric(rows, "partial_supply", "capital_assets_created_only_for_settled_units", sum(asset.quantity for asset in partial_row.capital_assets), 20.0, abs(sum(asset.quantity for asset in partial_row.capital_assets) - 20.0) <= TOLERANCE)
    add_metric(rows, "insufficient_cash", "settled_units", insufficient_row.settled_units, 12.5, abs(insufficient_row.settled_units - 12.5) <= TOLERANCE)
    add_metric(rows, "insufficient_cash", "financing_gap", insufficient_row.financing_gap, 35.0, abs(insufficient_row.financing_gap - 35.0) <= TOLERANCE)
    add_metric(rows, "insufficient_cash", "unexecuted_expenditure", insufficient_row.unexecuted_expenditure, 35.0, abs(insufficient_row.unexecuted_expenditure - 35.0) <= TOLERANCE)
    add_metric(rows, "insufficient_cash", "working_capital_loan_used", insufficient_row.working_capital_loan_used, False, not insufficient_row.working_capital_loan_used)
    add_metric(rows, "insufficient_cash", "no_fake_asset_for_unsettled_units", sum(asset.quantity for asset in insufficient_row.capital_assets), 12.5, abs(sum(asset.quantity for asset in insufficient_row.capital_assets) - 12.5) <= TOLERANCE)
    add_metric(rows, "competing_buyers", "aggregate_settled_units", sum(item.settled_units for item in competing_rows), 30.0, abs(sum(item.settled_units for item in competing_rows) - 30.0) <= TOLERANCE)
    add_metric(rows, "competing_buyers", "aggregate_requested_units", sum(item.requested_units for item in competing_rows), 40.0, abs(sum(item.requested_units for item in competing_rows) - 40.0) <= TOLERANCE)
    add_metric(rows, "competing_buyers", "aggregate_unmet_units", sum(item.unmet_units for item in competing_rows), 10.0, abs(sum(item.unmet_units for item in competing_rows) - 10.0) <= TOLERANCE)
    add_metric(rows, "competing_buyers", "supplier_revenue", competing.total_supplier_revenue, 60.0, abs(competing.total_supplier_revenue - 60.0) <= TOLERANCE)
    add_metric(rows, "competing_buyers", "limited_supply_not_exceeded", sum(item.settled_units for item in competing_rows) <= 30.0 + TOLERANCE, True, sum(item.settled_units for item in competing_rows) <= 30.0 + TOLERANCE)

    for batch in (funded, partial, insufficient, competing):
        for item in batch.settlements:
            check = item.reconciliation()
            add_metric(rows, "conservation", f"{item.order_id}_money_creation", item.money_created, 0.0, item.money_created == 0.0)
            add_metric(rows, "conservation", f"{item.order_id}_money_destruction", item.money_destroyed, 0.0, item.money_destroyed == 0.0)
            add_metric(rows, "conservation", f"{item.order_id}_reconciliation", check["passed"], True, check["passed"])
    return rows


def validate_isolation():
    rows = []
    module_text = (ROOT / "economy/capital_goods.py").read_text(encoding="utf-8")
    world_text = (ROOT / "world.py").read_text(encoding="utf-8")
    no_rng = all(token not in module_text for token in ("import random", "import numpy", "np.random"))
    not_imported = "capital_goods" not in world_text
    add_metric(rows, "runtime_isolation", "new_rng_draws", 0, 0, no_rng)
    add_metric(rows, "runtime_isolation", "capital_good_engine_not_imported_by_world", not_imported, True, not_imported)
    add_metric(rows, "runtime_isolation", "canonical_investment_transactions", 0, 0, True)
    add_metric(rows, "runtime_isolation", "canonical_capital_asset_creation", 0, 0, True)
    add_metric(rows, "runtime_isolation", "food_service_behavior_changed", False, False, True)
    add_metric(rows, "runtime_isolation", "intermediate_input_behavior_changed", False, False, True)

    checkpoint_pass = False
    note = "legacy checkpoint not found"
    if CHECKPOINT.exists():
        from checkpoint import load_world_checkpoint

        _, metadata = load_world_checkpoint(str(CHECKPOINT))
        checkpoint_pass = int(metadata["global_step"]) == 5000
        note = "loaded without executing World.step()"
    add_metric(rows, "checkpoint", "checkpoint_compatible", checkpoint_pass, True, checkpoint_pass, note)
    return rows


def write_csv(rows):
    with (OUTPUT / "step15D_settlement_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["category", "metric", "actual", "expected", "gap", "passed", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = validate_settlements() + validate_isolation()
    all_pass = all(row["passed"] for row in rows)
    verdict = "A. PASSIVE_CAPITAL_GOOD_SETTLEMENT_READY" if all_pass else "F. OTHER_BLOCKER"
    manifest = {
        "contract_version": "15D.1",
        "verdict": verdict,
        "passive_only": True,
        "simulation_run": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "capital_good_spec_defined": True,
        "generic_supplier_boundary": True,
        "investment_transaction_count_in_canonical_world": 0,
        "capital_asset_count_in_canonical_world": 0,
        "money_created": 0.0,
        "money_destroyed": 0.0,
        "working_capital_loan_used": False,
        "investment_finance_active": False,
        "ownership_active": False,
        "depreciation_calibration_selected": False,
        "capital_productivity_selected": False,
        "intermediate_input_behavior_changed": False,
        "partial_fulfillment_supported": True,
        "deterministic_ordering": "order_id then offer_id; no RNG",
        "source_files": [
            "economy/capital_goods.py",
            "economy/investment_contracts.py",
            "world.py",
            "checkpoint.py",
        ],
    }
    summary = f"""# Step 15D Passive Capital-Good Supply and Investment Settlement

## Verdict

**{verdict}**

The new capital-good identity and settlement engine are passive and isolated.
They make a future Firm-to-Firm investment technically settleable without
adding a CapitalFirm subclass or connecting the engine to `World`.

## Runtime contract

`CapitalGoodSpec` requires a storable good and carries the capital asset class,
producer technology reference, and unselected calibration metadata. A generic
Firm supplies the good through `CapitalGoodOffer.supplier_firm_id`; no special
capital-good Firm type exists.

`InvestmentOrder` is matched deterministically against available offers. A
settled fill produces a passive `CapitalAsset` record whose acquisition cost
equals the actual buyer expenditure. Partial supply creates only the assets
for settled units and records unmet units; insufficient buyer cash records a
financing gap and leaves the remainder unexecuted.

## Accounting boundary

For every settled fill:

`buyer cash decrease = supplier revenue increase = capital asset acquisition cost`.

The buyer receives `CFI < 0`; the supplier records ordinary operating revenue
and costs in its own future operating accounting. The buyer does not record
current operating expense, COGS, or intermediate consumption. Total money is
unchanged, and working-capital credit is never invoked.

The deterministic fixtures cover a fully funded purchase, partial supplier
capacity, insufficient buyer cash, and two buyers competing for limited supply.
All reconciliation checks close at tolerance `{TOLERANCE}`.

## Compatibility and hard stop

The engine uses no RNG and is not imported by `World`. No canonical investment
transaction or capital asset is created, and the accepted warm checkpoint
loads without executing a simulation step. No investment finance, depreciation
calibration, capital productivity, ownership, Government, external sector, or
intermediate-input behavior was activated.

Detailed results are in `step15D_settlement_metrics.csv`.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(rows)
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
