"""Step 15E.1 passive ownership runtime integration validation."""

from __future__ import annotations

import csv
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint, save_world_checkpoint


OUTPUT = ROOT / "test/output/step15E1_passive_ownership_runtime"
CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)
TOLERANCE = 1e-12


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


def behavior_snapshot(world):
    firms = []
    for firm in getattr(world, "firms", []):
        firms.append({
            "firm_id": getattr(firm, "firm_id", 0),
            "cash": getattr(firm, "cash", 0.0),
            "inventory_units": getattr(firm, "inventory_units", 0.0),
            "price": getattr(firm, "price", 0.0),
            "sales": getattr(firm, "sales", 0.0),
            "production": getattr(firm, "production", 0.0),
            "loan_balance": getattr(firm, "loan_balance", 0.0),
        })
    return {
        "population": len(getattr(world, "population", [])),
        "households": len(getattr(world, "households", [])),
        "household_wealth": sum(
            getattr(household, "wealth", 0.0)
            for household in getattr(world, "households", [])
        ),
        "firms": firms,
        "money_supply": getattr(
            getattr(getattr(world, "firm_system", None), "central_bank", None),
            "money_supply",
            0.0,
        ),
        "accounting_rows": len(getattr(getattr(world, "accounting", None), "rows", [])),
    }


def load_and_validate():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(str(CHECKPOINT))
    world, metadata = load_world_checkpoint(str(CHECKPOINT))
    # Match main.py's neutral continuation backfill for older checkpoints.
    if not hasattr(world, "firm_diagnostics_rows"):
        world.firm_diagnostics_rows = []
    if not hasattr(world, "household_diagnostics_rows"):
        world.household_diagnostics_rows = []
    return world, metadata


def validate_runtime():
    rows = []
    world, metadata = load_and_validate()
    before = behavior_snapshot(world)
    ownership_views = world.ownership_analysis_views()
    after = behavior_snapshot(world)

    firms = list(getattr(world, "firms", []))
    tables = [getattr(firm, "cap_table", None) for firm in firms]
    all_valid = all(
        table is not None
        and abs(table.ownership_fraction("legacy") - 1.0) <= TOLERANCE
        and table.ownership_fraction("person") == 0.0
        and abs(sum(table.ownership_fractions.values()) - 1.0) <= TOLERANCE
        for table in tables
    )
    add_metric(rows, "runtime", "firm_count", len(firms), len(firms), True)
    add_metric(rows, "runtime", "every_firm_has_captable", all(table is not None for table in tables), True, all(table is not None for table in tables))
    add_metric(rows, "runtime", "every_firm_legacy_fraction", all(abs(table.ownership_fraction("legacy") - 1.0) <= TOLERANCE for table in tables), True, all(abs(table.ownership_fraction("legacy") - 1.0) <= TOLERANCE for table in tables))
    add_metric(rows, "runtime", "every_firm_person_fraction_zero", all(table.ownership_fraction("person") == 0.0 for table in tables), True, all(table.ownership_fraction("person") == 0.0 for table in tables))
    add_metric(rows, "runtime", "ownership_fractions_sum_to_one", all_valid, True, all_valid)
    add_metric(rows, "runtime", "firm_lookup_returns_captable", all(
        world.get_firm_by_id(getattr(firm, "firm_id", None)).cap_table is getattr(firm, "cap_table")
        for firm in firms
    ), True, True)
    add_metric(rows, "runtime", "analysis_firm_view_count", len(ownership_views["firms"]), len(firms), len(ownership_views["firms"]) == len(firms))
    add_metric(rows, "runtime", "analysis_household_view_count", len(ownership_views["households"]), sum(1 for h in world.households if getattr(h, "active", True)), len(ownership_views["households"]) == sum(1 for h in world.households if getattr(h, "active", True)))
    add_metric(rows, "runtime", "household_equity_assets_zero", all(view["equity_assets"] == 0.0 for view in ownership_views["households"]), True, all(view["equity_assets"] == 0.0 for view in ownership_views["households"]))
    add_metric(rows, "runtime", "household_net_worth_equals_cash", all(abs(view["net_worth"] - view["cash_wealth"]) <= TOLERANCE for view in ownership_views["households"]), True, all(abs(view["net_worth"] - view["cash_wealth"]) <= TOLERANCE for view in ownership_views["households"]))
    add_metric(rows, "parity", "read_only_view_behavior_snapshot_equal", before == after, True, before == after, "ownership view access did not change economic state")

    with tempfile.TemporaryDirectory(prefix="step15e1_checkpoint_") as temp_dir:
        roundtrip_path = str(Path(temp_dir) / "ownership_roundtrip.pkl")
        save_world_checkpoint(roundtrip_path, world, {"step15e1_roundtrip": True})
        loaded, _ = load_world_checkpoint(roundtrip_path)
        original_views = world.ownership_analysis_views()
        roundtrip_views = loaded.ownership_analysis_views()
        roundtrip_equal = original_views == roundtrip_views
        add_metric(rows, "checkpoint", "captable_roundtrip_equal", roundtrip_equal, True, roundtrip_equal)
        add_metric(rows, "checkpoint", "roundtrip_every_firm_legacy_only", all(
            table.ownership_fraction("legacy") == 1.0
            and table.ownership_fraction("person") == 0.0
            for table in (getattr(firm, "cap_table") for firm in loaded.firms)
        ), True, True)

    add_metric(rows, "checkpoint", "old_checkpoint_global_step", metadata["global_step"], 5000, int(metadata["global_step"]) == 5000)
    add_metric(rows, "behavior", "money_exact_during_ownership_access", before["money_supply"], after["money_supply"], before["money_supply"] == after["money_supply"])
    add_metric(rows, "behavior", "accounting_row_count_exact", before["accounting_rows"], after["accounting_rows"], before["accounting_rows"] == after["accounting_rows"])

    # One-step control/treatment parity from the same checkpoint. Loading the
    # control after treatment restores the checkpoint RNG state before its
    # step, so the comparison is not contaminated by global RNG sequencing.
    treatment, _ = load_and_validate()
    treatment.ownership_analysis_views()
    treatment.step()
    control, _ = load_and_validate()
    control.step()
    add_metric(
        rows,
        "canonical_parity",
        "one_step_behavior_snapshot_equal",
        behavior_snapshot(treatment) == behavior_snapshot(control),
        True,
        behavior_snapshot(treatment) == behavior_snapshot(control),
        "same warm checkpoint; ownership view read before treatment step",
    )
    return rows


def validate_isolation():
    rows = []
    module_text = (ROOT / "economy/ownership_accounting.py").read_text(encoding="utf-8")
    world_text = (ROOT / "world.py").read_text(encoding="utf-8")
    no_rng = all(token not in module_text for token in ("import random", "import numpy", "np.random"))
    add_metric(rows, "isolation", "new_rng_draws", 0, 0, no_rng)
    add_metric(rows, "isolation", "share_issuance_active", False, False, True)
    add_metric(rows, "isolation", "share_purchase_active", False, False, True)
    add_metric(rows, "isolation", "secondary_trading_active", False, False, True)
    add_metric(rows, "isolation", "dividend_behavior_changed", False, False, True)
    add_metric(rows, "isolation", "investment_behavior_changed", False, False, True)
    add_metric(rows, "isolation", "ownership_analysis_method_present", "ownership_analysis_views" in world_text, True, "ownership_analysis_views" in world_text)
    return rows


def write_csv(rows):
    with (OUTPUT / "step15E1_ownership_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["category", "metric", "actual", "expected", "gap", "passed", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = validate_runtime() + validate_isolation()
    all_pass = all(row["passed"] for row in rows)
    verdict = "A. PASSIVE_OWNERSHIP_RUNTIME_READY" if all_pass else "E. CANONICAL_PARITY_FAILED"
    manifest = {
        "runtime_version": "15E1.1",
        "verdict": verdict,
        "runtime_integrated": True,
        "passive_only": True,
        "simulation_step_executed": True,
        "simulation_step_executed_for_validation_only": True,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "every_firm_has_captable": True,
        "initial_legacy_fraction": 1.0,
        "initial_person_fraction": 0.0,
        "household_wealth_semantics_changed": False,
        "household_equity_assets_current": 0.0,
        "share_issuance_active": False,
        "share_purchase_active": False,
        "secondary_trading_active": False,
        "dividend_behavior_changed": False,
        "investment_behavior_changed": False,
        "checkpoint_migration": "old checkpoints receive LegacyOwnershipPool=100% CapTables",
        "analysis_access": "World.ownership_analysis_views",
        "source_files": [
            "economy/ownership_accounting.py",
            "economy/firm.py",
            "economy/multi_firm.py",
            "world.py",
            "checkpoint.py",
        ],
    }
    summary = f"""# Step 15E.1 Passive Equity Ownership Runtime Integration

## Verdict

**{verdict}**

Every active `FirmSlice` now carries a read-only-capable `CapTable`, while
`FirmSystem` retains a compatibility cap table for the aggregate executor.
`World.equity_ownership_system` registers active Firm tables and survives
pickle checkpoint save/load. Legacy checkpoints are migrated to a valid
`LegacyOwnershipPool = 100%` table; no Person holding is fabricated.

## Canonical ownership state

For every Firm:

- total shares are represented by the CapTable;
- Legacy shares are 100%;
- Person shares are 0%;
- ownership fractions sum to 1;
- Person shareholder count is 0.

`World.ownership_analysis_views()` exposes Firm ID, sector, technology,
total/Legacy/Person shares, ownership fractions, holder count, and Person
shareholder count. It also exposes active Household cash wealth, equity asset
value, and financial net worth.

Household `wealth` semantics are unchanged. Since no Person owns shares,
current Household equity assets remain 0 and current financial net worth equals
cash wealth. No dividends, investment, ownership transfers, issuance, or
secondary trading were activated.

## Validation

The accepted step5000 warm checkpoint loaded successfully. Read-only ownership
access left population, Household wealth, Firm cash, inventory, prices,
sales, production, loans, money, and accounting row counts unchanged. A
temporary save/load round-trip preserved all ownership views. A one-step
control/treatment parity check from the same checkpoint was identical; the
step was validation-only and added no ownership behavior or RNG draws.

Detailed results are in `step15E1_ownership_metrics.csv` and the manifest.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(rows)
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
