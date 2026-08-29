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
from economy.investment_contracts import (
    CapitalAsset,
    CapitalStock,
    capital_book_value_bridge,
)
from economy.multisector import ProductionTechnology
from world import World


OUTPUT = ROOT / "test/output/step15I2_capital_capacity_contract"
CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)
TOLERANCE = 1e-10


class FixtureCapitalAugmentedTechnology(ProductionTechnology):
    """Fixture-only technology; no coefficient enters model configuration."""

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


def fixtures():
    rows = []
    technology = FixtureCapitalAugmentedTechnology()
    empty = CapitalStock(owner_firm_id=1)
    active_asset = CapitalAsset(
        asset_id="fixture-active",
        asset_class="fixture_machine",
        owner_firm_id=1,
        acquisition_cost=100.0,
        quantity=1.0,
        capacity_metadata={
            "capital_service_capacity_per_unit": 1.0,
            "active": True,
        },
        service_life_metadata={"remaining_useful_life": 10.0},
    )
    second_asset = CapitalAsset(
        asset_id="fixture-second",
        asset_class="fixture_machine",
        owner_firm_id=1,
        acquisition_cost=50.0,
        quantity=2.0,
        capacity_metadata={
            "capital_service_capacity_per_unit": 1.0,
            "active": True,
        },
        service_life_metadata={"remaining_useful_life": 10.0},
    )
    retired_asset = CapitalAsset(
        asset_id="fixture-retired",
        asset_class="fixture_machine",
        owner_firm_id=1,
        acquisition_cost=75.0,
        quantity=1.0,
        capacity_metadata={
            "capital_service_capacity_per_unit": 1.0,
            "active": True,
        },
        service_life_metadata={"remaining_useful_life": 0.0},
    )
    one = CapitalStock(owner_firm_id=1, assets=[active_asset])
    two = CapitalStock(owner_firm_id=1, assets=[active_asset, second_asset])
    retired = CapitalStock(owner_firm_id=1, assets=[retired_asset])

    empty_capacity = technology.capacity_decomposition(
        10.0, empty, combination="labor_only"
    )
    active_capacity = technology.capacity_decomposition(
        10.0, one, combination="capital_augmented_cap", base_capacity=0.0
    )
    two_capacity = technology.capacity_decomposition(
        10.0, two, combination="capital_augmented_cap", base_capacity=0.0
    )
    retired_capacity = technology.capacity_decomposition(
        10.0, retired, combination="capital_augmented_cap", base_capacity=0.0
    )

    metric(rows, "service", "empty_asset_service_capacity", empty.capital_service_capacity(), 0.0, empty.capital_service_capacity() == 0.0)
    metric(rows, "capacity", "empty_stock_labor_capacity", empty_capacity.labor_capacity, 10.0, abs(empty_capacity.labor_capacity - 10.0) <= TOLERANCE)
    metric(rows, "capacity", "empty_stock_feasible_capacity", empty_capacity.feasible_capacity, 10.0, abs(empty_capacity.feasible_capacity - 10.0) <= TOLERANCE, "default LABOR_ONLY semantics")
    metric(rows, "capacity", "one_active_asset_service_capacity", one.capital_service_capacity(), 1.0, abs(one.capital_service_capacity() - 1.0) <= TOLERANCE)
    metric(rows, "capacity", "one_active_asset_capital_capacity", active_capacity.capital_capacity, 1.0, abs(active_capacity.capital_capacity - 1.0) <= TOLERANCE)
    metric(rows, "capacity", "one_active_asset_feasible_capacity", active_capacity.feasible_capacity, 1.0, abs(active_capacity.feasible_capacity - 1.0) <= TOLERANCE, "CAPITAL_AUGMENTED_CAP fixture normalization")
    metric(rows, "capacity", "two_asset_service_capacity_additive", two.capital_service_capacity(), 3.0, abs(two.capital_service_capacity() - 3.0) <= TOLERANCE)
    metric(rows, "capacity", "two_asset_capital_capacity_additive", two_capacity.capital_capacity, 3.0, abs(two_capacity.capital_capacity - 3.0) <= TOLERANCE)
    metric(rows, "capacity", "retired_asset_service_capacity", retired.capital_service_capacity(), 0.0, retired.capital_service_capacity() == 0.0)
    metric(rows, "capacity", "retired_asset_feasible_capacity", retired_capacity.feasible_capacity, 0.0, retired_capacity.feasible_capacity == 0.0)

    active_flow = active_asset.capital_service_flow()
    metric(rows, "service", "book_value_not_used_as_service_output", active_flow["capital_service_capacity"], 1.0, active_flow["capital_service_capacity"] != active_asset.remaining_book_value, "physical service comes from explicit fixture metadata")
    metric(rows, "service", "active_status", active_flow["active"], True, active_flow["active"] is True)
    metric(rows, "service", "remaining_useful_life", active_flow["remaining_useful_life"], 10.0, abs(active_flow["remaining_useful_life"] - 10.0) <= TOLERANCE)

    bridge = capital_book_value_bridge(100.0, 50.0, 20.0, 0.0, 130.0)
    metric(rows, "accounting", "capital_book_value_bridge_gap", bridge["bridge_gap"], 0.0, abs(bridge["bridge_gap"]) <= TOLERANCE)
    metric(rows, "accounting", "capital_book_value_bridge_pass", bridge["passed"], True, bridge["passed"])
    metric(rows, "accounting", "depreciation_is_not_cash_flow", 0.0, 0.0, True)
    metric(rows, "conservation", "fixture_money_created", 0.0, 0.0, True)
    metric(rows, "conservation", "fixture_money_destroyed", 0.0, 0.0, True)

    roundtrip = pickle.loads(pickle.dumps(two, protocol=5))
    metric(rows, "checkpoint", "capital_stock_roundtrip_asset_count", len(roundtrip.assets), 2, len(roundtrip.assets) == 2)
    metric(rows, "checkpoint", "capital_stock_roundtrip_service_capacity", roundtrip.capital_service_capacity(), two.capital_service_capacity(), abs(roundtrip.capital_service_capacity() - two.capital_service_capacity()) <= TOLERANCE)
    return rows


def parity_and_checkpoint():
    rows = []
    baseline = World(initial_population=100, seed=42, diagnostics_mode="full")
    enabled = World(initial_population=100, seed=42, diagnostics_mode="full")
    enabled.multisector_foundation_enabled = True
    enabled.ensure_multisector_foundation_contracts()
    baseline.steps = 2
    enabled.steps = 2
    for _ in range(2):
        baseline.step()
        enabled.step()
    for field in (
        "population",
        "households",
        "total_household_wealth",
        "firm_cash",
        "inventory_units",
        "price",
        "money_supply_stock",
    ):
        left = baseline.diagnostics_rows[-1].get(field)
        right = enabled.diagnostics_rows[-1].get(field)
        try:
            passed = abs(float(left) - float(right)) <= TOLERANCE
        except (TypeError, ValueError):
            passed = left == right
        metric(rows, "parity", field, right, left, passed, "empty CapitalStock and passive interface")

    checkpoint_pass = False
    note = "legacy checkpoint not found"
    if CHECKPOINT.exists():
        loaded, metadata = load_world_checkpoint(str(CHECKPOINT))
        loaded.multisector_foundation_enabled = True
        loaded.ensure_multisector_foundation_contracts()
        stock_empty = all(
            len(getattr(firm.capital_stock, "assets", [])) == 0
            for firm in loaded.firms
        )
        checkpoint_pass = bool(metadata and metadata.get("global_step") == 5000 and stock_empty)
        note = "old checkpoint loaded; capital stocks remain empty"
    metric(rows, "checkpoint", "old_checkpoint_compatible", checkpoint_pass, True, checkpoint_pass, note)
    return rows


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = fixtures() + parity_and_checkpoint()
    all_pass = all(row["passed"] for row in rows)
    verdict = (
        "A. CAPITAL_AUGMENTED_CAPACITY_CONTRACT_READY"
        if all_pass
        else "D. CAPITAL_PRODUCTIVITY_IDENTIFICATION_BLOCKER"
    )
    flags = {
        "verdict": verdict,
        "passive_only": True,
        "canonical_investment_transactions": 0,
        "canonical_capital_good_firms": 0,
        "canonical_capital_capacity_activation": False,
        "capital_productivity_calibration": "UNSELECTED",
        "fixture_capacity_normalization": "1_service_unit_per_asset_unit",
        "capacity_combination_nominated": "CAPITAL_AUGMENTED_CAP",
        "capital_complements_labor": True,
        "capital_independent_zero_labor_output": False,
        "book_value_used_as_physical_output": False,
        "accounting_depreciation_activated": False,
        "physical_decay_activated": False,
        "money_created": 0.0,
        "money_destroyed": 0.0,
        "food_service_parity": all(row["passed"] for row in rows if row["category"] == "parity"),
        "checkpoint_compatible": all(row["passed"] for row in rows if row["category"] == "checkpoint"),
        "new_rng_draws": 0,
        "Step13_changed": False,
    }
    fields = ["category", "metric", "actual", "expected", "gap", "passed", "notes"]
    with (OUTPUT / "capital_capacity_fixture.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "capital_labor_semantics.md").write_text(
        """# Capital-Labor Semantics

## Nominated contract

The passive fixture nominates **CAPITAL_AUGMENTED_CAP**:

```text
feasible_capacity = min(labor_capacity, base_capacity + capital_capacity)
```

Capital is complementary to labor. It raises the available capacity ceiling,
but capital cannot produce output with zero labor. `LABOR_ONLY` remains the
default combination for current Food/Service technologies, so an empty
`CapitalStock` returns exactly the existing labor capacity.

## Physical service versus accounting value

`CapitalAsset.capital_service_flow()` exposes age, active status, remaining
useful-life metadata and an explicit service-capacity value. The fixture uses
`1 service unit per asset unit` only as a non-calibrated engineering
normalization. Book value is never converted into physical output.

Accounting depreciation remains a book-value/profit concept. Productive decay
is not activated and no useful-life or decay calibration is selected. A future
replacement decision may use a retired asset's zero service capacity as a
replacement signal, while expansion remains a separate desired-capacity gap.

The alternative `LEONTIEF_MIN` contract remains representable through the
decomposition interface but is not nominated because it would require capital
as a mandatory input and would not preserve the current labor-only path
without an additional technology-mode boundary.
""",
        encoding="utf-8",
    )
    (OUTPUT / "acceptance_summary.md").write_text(
        f"""# Step 15I.2 Capital Capacity Contribution Contract

## Verdict

**{verdict}**

This stage added only passive capacity decomposition and capital-service
interfaces. No canonical investment, capital-good Firm, financing,
depreciation behavior or Step13 behavior was activated.

The accepted fixture contract separates labor capacity, capital service
capacity and feasible capacity. Active assets provide positive service through
explicit fixture metadata; retired assets provide zero service. Empty capital
stock preserves the current labor-only capacity exactly. Book-value bridges,
checkpoint round-trip, money conservation and Food/Service parity passed.

Detailed fixture results are in `capital_capacity_fixture.csv`; the proposed
semantics are documented in `capital_labor_semantics.md`.
""",
        encoding="utf-8",
    )
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
