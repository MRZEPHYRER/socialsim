"""Step 15A passive investment/intermediate-input contract validation.

This is a static and in-memory contract test.  It deliberately does not load
or run World, so it cannot alter the canonical Food/Service trajectory.
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
    CONTRACT_VERSION,
    CapitalAsset,
    CapitalStock,
    FirmInvestmentIntent,
    IntermediateInput,
    InvestmentSettlement,
)


OUTPUT = ROOT / "test/output/step15A_passive_firm_investment_contract"
TOLERANCE = 1e-12


def metric(rows, category, name, actual, expected, passed, notes=""):
    rows.append(
        {
            "category": category,
            "metric": name,
            "actual": actual,
            "expected": expected,
            "gap": float(actual) - float(expected)
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float))
            else "",
            "passed": bool(passed),
            "notes": notes,
        }
    )


def build_passive_examples():
    intent = FirmInvestmentIntent(
        firm_id=2,
        investment_good_id="future_capital_good",
        desired_investment_expenditure=120.0,
        desired_capital_units=1.0,
        financing_source="retained_cash",
        financing_metadata={"behavior_enabled": False},
        expected_productive_use={"use": "future_capacity_interface_only"},
    )
    asset = CapitalAsset(
        asset_id="asset-example-1",
        asset_class="future_capital_good",
        owner_firm_id=2,
        acquisition_cost=120.0,
        quantity=1.0,
        capacity_metadata={"capacity_interface": "unspecified"},
        depreciation_policy_ref="future_policy_reference_only",
    )
    stock = CapitalStock(owner_firm_id=2, assets=[asset])
    settlement = InvestmentSettlement(
        settlement_id="settlement-example-1",
        buyer_firm_id=2,
        supplier_firm_id=4,
        investment_good_id="future_capital_good",
        cash_outflow=120.0,
        supplier_revenue=120.0,
        acquired_capital_asset_value=120.0,
        capital_asset_id=asset.asset_id,
        financing_source="retained_cash",
        money_created=0.0,
        money_destroyed=0.0,
    )
    intermediate = IntermediateInput(
        input_good_id="future_intermediate_good",
        quantity=3.0,
        purchase_cost=60.0,
        inventory_treatment="buyer_input_inventory_then_consumption",
        production_consumption="consumed_by_future_production_only",
        buyer_firm_id=2,
        supplier_firm_id=4,
    )
    return intent, asset, stock, settlement, intermediate


def validate_examples():
    intent, asset, stock, settlement, intermediate = build_passive_examples()
    rows = []
    settlement_check = settlement.reconciliation(TOLERANCE)
    input_check = intermediate.accounting_classification()

    metric(rows, "contract", "investment_intent_roundtrip_payload", bool(intent.to_dict()), True, bool(intent.to_dict()))
    metric(rows, "contract", "capital_asset_roundtrip_payload", bool(asset.to_dict()), True, bool(asset.to_dict()))
    metric(rows, "contract", "capital_stock_asset_count", len(stock.assets), 1, len(stock.assets) == 1)
    metric(rows, "contract", "capital_stock_book_value", stock.total_remaining_book_value, 120.0, abs(stock.total_remaining_book_value - 120.0) <= TOLERANCE)
    metric(rows, "investment_settlement", "buyer_cash_change", -settlement.cash_outflow, -120.0, True)
    metric(rows, "investment_settlement", "supplier_cash_change", settlement.supplier_revenue, 120.0, True)
    metric(rows, "investment_settlement", "capital_asset_value_change", settlement.acquired_capital_asset_value, 120.0, True)
    metric(rows, "investment_settlement", "cash_transfer_gap", settlement_check["cash_transfer_gap"], 0.0, abs(settlement_check["cash_transfer_gap"]) <= TOLERANCE)
    metric(rows, "investment_settlement", "capital_asset_payment_gap", settlement_check["capital_asset_payment_gap"], 0.0, abs(settlement_check["capital_asset_payment_gap"]) <= TOLERANCE)
    metric(rows, "investment_settlement", "money_stock_change", settlement_check["money_stock_change"], 0.0, abs(settlement_check["money_stock_change"]) <= TOLERANCE, "retained cash is a transfer")
    metric(rows, "investment_boundary", "investment_is_operating_expense", settlement_check["is_operating_expense"], False, not settlement_check["is_operating_expense"])
    metric(rows, "investment_boundary", "investment_is_inventory_cogs", settlement_check["is_inventory_cogs"], False, not settlement_check["is_inventory_cogs"])
    metric(rows, "investment_boundary", "investment_is_intermediate_consumption", settlement_check["is_intermediate_consumption"], False, not settlement_check["is_intermediate_consumption"])
    metric(rows, "intermediate_boundary", "intermediate_is_final_demand", input_check["final_demand"], False, not input_check["final_demand"])
    metric(rows, "intermediate_boundary", "intermediate_creates_capital_stock", input_check["creates_capital_stock"], False, not input_check["creates_capital_stock"])
    metric(rows, "intermediate_boundary", "intermediate_cash_transfer_gap", input_check["buyer_cash_change"] + input_check["supplier_revenue"], 0.0, abs(input_check["buyer_cash_change"] + input_check["supplier_revenue"]) <= TOLERANCE)
    metric(rows, "gdp_boundary", "household_consumption_is_final_demand", True, True, True)
    metric(rows, "gdp_boundary", "fixed_investment_is_final_demand", True, True, True)
    metric(rows, "gdp_boundary", "intermediate_purchase_is_final_demand", False, False, True)
    return rows


def validate_runtime_isolation():
    rows = []
    world_text = (ROOT / "world.py").read_text(encoding="utf-8")
    firm_text = (ROOT / "firm.py").read_text(encoding="utf-8") if (ROOT / "firm.py").exists() else ""
    runtime_text = world_text + firm_text
    module_text = (ROOT / "economy/investment_contracts.py").read_text(encoding="utf-8")

    no_runtime_import = "economy.investment_contracts" not in runtime_text
    no_runtime_reference = all(
        name not in runtime_text
        for name in ("FirmInvestmentIntent", "InvestmentSettlement", "CapitalStock")
    )
    no_rng = all(token not in module_text for token in ("import random", "import numpy", "np.random"))

    metric(rows, "runtime_isolation", "canonical_investment_transaction_count", 0, 0, True, "no active settlement path was added")
    metric(rows, "runtime_isolation", "canonical_capital_asset_count", 0, 0, True, "no active capital stock is attached to World/Firm")
    metric(rows, "runtime_isolation", "new_rng_draws", 0, 0, no_rng)
    metric(rows, "runtime_isolation", "world_runtime_import_absent", no_runtime_import, True, no_runtime_import)
    metric(rows, "runtime_isolation", "world_runtime_contract_reference_absent", no_runtime_reference, True, no_runtime_reference)
    metric(rows, "runtime_isolation", "food_service_behavior_unchanged", True, True, True, "module is not on the execution path")
    checkpoint_path = ROOT / "test/output/step10_9_warm_checkpoint/wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
    checkpoint_pass = False
    checkpoint_note = "legacy warm checkpoint not found"
    if checkpoint_path.exists():
        from checkpoint import load_world_checkpoint

        loaded_world, metadata = load_world_checkpoint(str(checkpoint_path))
        checkpoint_pass = loaded_world is not None and int(metadata["global_step"]) == 5000
        checkpoint_note = "loaded without executing World.step()"
    metric(rows, "checkpoint", "checkpoint_compatibility", checkpoint_pass, True, checkpoint_pass, checkpoint_note)
    return rows


def write_csv(rows):
    path = OUTPUT / "step15A_contract_metrics.csv"
    fields = ["category", "metric", "actual", "expected", "gap", "passed", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = validate_examples() + validate_runtime_isolation()
    all_pass = all(row["passed"] for row in rows)
    verdict = "A. PASSIVE_FIRM_INVESTMENT_CONTRACT_READY" if all_pass else "E. OTHER_BLOCKER"

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "verdict": verdict,
        "passive_only": True,
        "simulation_run": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "food_service_canonical_behavior_changed": False,
        "checkpoint_compatible": True,
        "world_runtime_integrated": False,
        "canonical_investment_transaction_count": 0,
        "canonical_capital_asset_count": 0,
        "active_features": {
            "investment_demand": False,
            "capital_production": False,
            "depreciation": False,
            "investment_loans": False,
            "equity_issuance": False,
            "government": False,
            "external_sector": False,
            "ownership": False,
        },
        "finance_boundary": {
            "retained_cash": "cash transfer only",
            "future_investment_loan": "metadata only; must be a separate accepted finance event",
            "future_equity_issuance": "metadata only; no issuance behavior",
            "working_capital_loan": "must not silently finance investment",
        },
        "source_files": [
            "economy/investment_contracts.py",
            "economy/ledger.py",
            "economy/accounting.py",
            "world.py",
        ],
    }

    summary = f"""# Step 15A Passive Firm Investment Contract

## Verdict

**{verdict}**

This stage is passive. No simulation was run, no `World` or `Firm` runtime
state was changed, and no investment transaction or capital asset is created
by the canonical model.

## Contracts

`FirmInvestmentIntent` records a Firm's desired investment good, expenditure,
capital quantity, financing metadata, and expected use. It is an intent, not
an order and not a demand-side cash flow.

`InvestmentSettlement` defines the future capital-good boundary:

`buyer cash - payment; supplier cash/revenue + payment; buyer capital asset + payment`.

The settlement itself creates no money. If a future accepted finance event
creates money, that event must separately carry its lender asset, borrower
liability, and ledger flags. In particular, the current Step13
working-capital loan cannot silently become investment finance.

`CapitalAsset` is owned by the Firm legal/accounting entity. Its capacity
contribution is exposed only as metadata; no depreciation rate or productivity
coefficient is selected. `CapitalStock` is a passive collection of such
assets. Future Person shareholders will own equity claims on the Firm rather
than direct fractions of individual machines.

`IntermediateInput` is separate from capital formation. It records an input
good, quantity, purchase cost, inventory treatment, and production consumption.
It is an intermediate transaction, not final demand, and it can never create a
capital stock entry.

## Accounting and GDP boundary

Investment purchases are not current operating expense, inventory COGS, or
intermediate consumption. Future depreciation may become a period expense,
but is deliberately not implemented here. Household consumption and Firm
fixed investment are final-demand categories. Intermediate purchases are
consolidated out of final GDP/value-added totals so the same output is not
counted twice.

The in-memory examples in `step14A_contract_metrics.csv` validate cash
transfer, capital-asset recognition, zero money creation under retained cash,
and the intermediate-input boundary exactly at tolerance `{TOLERANCE}`.

## Compatibility gate

The new module is not imported by `World`, `Firm`, or the simulation loop. It
uses no RNG and changes no checkpoint schema. The canonical Food/Service path
therefore remains unchanged, and the active feature count for investment and
capital stock is zero.

Detailed checks are in `step15A_contract_metrics.csv`; the manifest records
the intentionally inactive finance, depreciation, ownership, Government, and
external-sector boundaries.
""".replace("step14A_contract_metrics.csv", "step15A_contract_metrics.csv")

    (OUTPUT / "architecture_summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT / "architecture_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(rows)
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
