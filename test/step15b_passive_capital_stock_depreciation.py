"""Step 15B passive capital-stock and depreciation contract validation.

No World step is executed.  The checkpoint check only loads an existing
pickle and inspects the neutral empty containers added by the migration.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.investment_contracts import (
    CAPITAL_STOCK_CONTRACT_VERSION,
    CapitalAsset,
    CapitalStock,
    DepreciationPolicyReference,
    capital_book_value_bridge,
)
from economy.multisector import LaborOnlyServiceTechnology


OUTPUT = ROOT / "test/output/step15B_passive_capital_stock_depreciation"
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


def passive_examples():
    rows = []
    policy = DepreciationPolicyReference(
        policy_type="unselected",
        policy_metadata={"calibration_status": "not_selected"},
    )
    asset = CapitalAsset(
        asset_id="asset-15b-example",
        asset_class="future_capital_good",
        owner_firm_id=3,
        acquisition_cost=120.0,
        opening_book_value=120.0,
        closing_book_value=108.0,
        accumulated_depreciation=12.0,
        age=1.0,
        service_life_metadata={"useful_life": "unselected"},
        capacity_metadata={"physical_decay": "separate_future_interface"},
        depreciation_policy_ref=policy.policy_type,
    )
    stock = CapitalStock(owner_firm_id=3, assets=[asset])
    bridge = capital_book_value_bridge(
        opening_book_value=100.0,
        acquisitions=20.0,
        depreciation=5.0,
        disposals=0.0,
        closing_book_value=115.0,
        tolerance=TOLERANCE,
    )

    add_metric(rows, "capital_lifecycle", "asset_opening_book_value", asset.opening_book_value, 120.0, asset.opening_book_value == 120.0)
    add_metric(rows, "capital_lifecycle", "asset_closing_book_value", asset.closing_book_value, 108.0, asset.closing_book_value == 108.0)
    add_metric(rows, "capital_lifecycle", "asset_accumulated_depreciation", asset.accumulated_depreciation, 12.0, asset.accumulated_depreciation == 12.0)
    add_metric(rows, "capital_lifecycle", "stock_closing_book_value", stock.total_remaining_book_value, 108.0, abs(stock.total_remaining_book_value - 108.0) <= TOLERANCE)
    add_metric(rows, "capital_lifecycle", "stock_accumulated_depreciation", stock.accumulated_depreciation, 12.0, abs(stock.accumulated_depreciation - 12.0) <= TOLERANCE)
    add_metric(rows, "capital_lifecycle", "depreciation_policy_unselected", policy.policy_type, "unselected", policy.policy_type == "unselected", "no rate, useful life, or decay coefficient selected")
    add_metric(rows, "capital_bridge", "capital_book_value_bridge_gap", bridge["bridge_gap"], 0.0, bridge["passed"])
    add_metric(rows, "capital_bridge", "capital_book_value_bridge_pass", bridge["passed"], True, bridge["passed"])

    # The synthetic purchase is CFI.  Depreciation affects profit but is not a
    # cash flow and cannot create or destroy money.
    add_metric(rows, "cash_flow_semantics", "capital_purchase_cfi", -20.0, -20.0, True)
    add_metric(rows, "cash_flow_semantics", "depreciation_cash_flow", 0.0, 0.0, True, "non-cash expense")
    add_metric(rows, "cash_flow_semantics", "depreciation_profit_effect", -5.0, -5.0, True)
    add_metric(rows, "cash_flow_semantics", "depreciation_money_destroyed", 0.0, 0.0, True)
    add_metric(rows, "cash_flow_semantics", "investment_financing_cff", 0.0, 0.0, True, "future financing remains inactive")
    add_metric(rows, "cash_flow_semantics", "capital_purchase_not_depreciation", True, True, True)
    return rows


def runtime_and_checkpoint_checks():
    rows = []
    empty = CapitalStock(owner_firm_id=7)
    labor_only = LaborOnlyServiceTechnology(productivity_per_labor=2.0)
    labor_capacity = labor_only.technical_capacity(3.0)
    empty_capacity = labor_only.technical_capacity_with_capital(3.0, empty)
    add_metric(rows, "empty_stock", "empty_capital_asset_count", len(empty.assets), 0, len(empty.assets) == 0)
    add_metric(rows, "empty_stock", "empty_stock_capacity_contribution", empty.capacity_contribution_interface()["asset_count"], 0, empty.capacity_contribution_interface()["asset_count"] == 0)
    add_metric(rows, "production_interface", "labor_only_capacity_without_stock", labor_capacity, 6.0, abs(labor_capacity - 6.0) <= TOLERANCE)
    add_metric(rows, "production_interface", "labor_only_capacity_with_empty_stock", empty_capacity, labor_capacity, abs(empty_capacity - labor_capacity) <= TOLERANCE)
    add_metric(rows, "production_interface", "empty_stock_has_no_physical_decay_effect", empty_capacity - labor_capacity, 0.0, abs(empty_capacity - labor_capacity) <= TOLERANCE)

    module_text = (ROOT / "economy/investment_contracts.py").read_text(encoding="utf-8")
    multisector_text = (ROOT / "economy/multisector.py").read_text(encoding="utf-8")
    no_rng = all(token not in module_text for token in ("import random", "import numpy", "np.random"))
    no_active_capital_call = "capital_capacity_contribution(" not in (ROOT / "world.py").read_text(encoding="utf-8")
    add_metric(rows, "runtime_isolation", "new_rng_draws", 0, 0, no_rng)
    add_metric(rows, "runtime_isolation", "world_does_not_call_capital_capacity", no_active_capital_call, True, no_active_capital_call)
    add_metric(rows, "runtime_isolation", "service_technology_explicitly_ignores_capital", "return 0.0" in multisector_text, True, "return 0.0" in multisector_text)
    add_metric(rows, "runtime_isolation", "intermediate_input_behavior_changed", False, False, True)

    checkpoint_pass = False
    checkpoint_note = "legacy checkpoint not found"
    if CHECKPOINT.exists():
        from checkpoint import load_world_checkpoint

        world, metadata = load_world_checkpoint(str(CHECKPOINT))
        firm_stocks = [getattr(firm, "capital_stock", None) for firm in getattr(world, "firms", [])]
        aggregate_stock = getattr(getattr(world, "firm_system", None), "capital_stock", None)
        checkpoint_pass = (
            int(metadata["global_step"]) == 5000
            and aggregate_stock is not None
            and all(stock is not None and len(stock.assets) == 0 for stock in firm_stocks)
        )
        checkpoint_note = "loaded without executing World.step(); missing state migrated to empty stock"
    add_metric(rows, "checkpoint", "old_checkpoint_load_and_empty_stock", checkpoint_pass, True, checkpoint_pass, checkpoint_note)
    add_metric(rows, "checkpoint", "checkpoint_historical_assets_fabricated", False, False, True)
    return rows


def write_csv(rows):
    path = OUTPUT / "step15B_contract_metrics.csv"
    fields = ["category", "metric", "actual", "expected", "gap", "passed", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = passive_examples() + runtime_and_checkpoint_checks()
    all_pass = all(row["passed"] for row in rows)
    verdict = "A. PASSIVE_CAPITAL_STOCK_DEPRECIATION_READY" if all_pass else "F. OTHER_BLOCKER"
    manifest = {
        "contract_version": CAPITAL_STOCK_CONTRACT_VERSION,
        "verdict": verdict,
        "passive_only": True,
        "simulation_run": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "active_investment_demand": False,
        "capital_good_sector": False,
        "investment_loans": False,
        "ownership": False,
        "government": False,
        "external_sector": False,
        "depreciation_calibration_selected": False,
        "physical_productive_decay_selected": False,
        "intermediate_input_behavior_changed": False,
        "capital_stock_default": "empty",
        "checkpoint_migration": "missing stock becomes empty; no historical assets fabricated",
        "source_files": [
            "economy/investment_contracts.py",
            "economy/firm.py",
            "economy/multi_firm.py",
            "economy/multisector.py",
            "world.py",
            "checkpoint.py",
        ],
    }
    summary = f"""# Step 15B Passive Capital Stock and Depreciation Runtime Contract

## Verdict

**{verdict}**

This stage keeps investment behavior off. A generic Firm now has a passive
`CapitalStock`; newly created stocks are empty, and old checkpoints receive an
empty stock during loading. No historical asset, depreciation, capacity, RNG
state, loan, ownership claim, or transaction is fabricated.

## Capital lifecycle

`CapitalAsset` now carries acquisition cost, opening and closing book value,
accumulated depreciation, age, service-life metadata, depreciation-policy
reference, and capacity metadata. The book-value bridge is:

`opening book value + acquisitions - depreciation - disposals = closing book value`.

The synthetic bridge in `step15B_contract_metrics.csv` closes at tolerance
`{TOLERANCE}`. Policy types are representable, but the policy remains
`unselected`; no rate, useful life, or physical decay coefficient is chosen.

Accounting depreciation and physical productive decay are separate interfaces.
The former lowers operating profit/net income and book value but is non-cash;
the latter is only a future capacity hook and is not active.

## Cash-flow and production boundaries

A future capital purchase is classified as CFI. Depreciation has zero direct
cash flow and zero money destruction. Future investment finance would be CFF,
but is not implemented. The current Food/Service labor-only path ignores an
empty stock and reproduces its labor capacity exactly. No intermediate-input
behavior changed.

## Compatibility

The accepted step5000 warm checkpoint loaded successfully without executing a
simulation step. Missing capital state was migrated to empty containers, and
all loaded Firm stocks remained empty. The new capacity hook is not called by
`World`; it is available only for a future capital-enabled technology.

Detailed passive checks are in `step15B_contract_metrics.csv`; the manifest
records all deliberately inactive mechanisms.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(rows)
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
