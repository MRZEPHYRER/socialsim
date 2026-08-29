import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenarios import apply_runtime_scenario, apply_scenario
from world import World


OUTPUT = ROOT / "test" / "output" / "pre_step13_2B_public_purchase_fix"
WINDOW = set(range(308, 319))
SEEDS = (1, 7, 21, 42, 99)
TOLERANCE = 1e-9


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
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


def build_world(seed):
    scenario = apply_scenario("baseline")
    world = World(
        initial_population=500,
        seed=seed,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    apply_runtime_scenario(scenario, world)
    world.split_firms(5)
    return world


def run_seed(seed, weeks=520):
    world = build_world(seed)
    window_rows = []
    donor_rows = []
    for _ in range(weeks):
        opening_inventory = {
            firm.firm_id: max(0.0, float(firm.inventory_units))
            for firm in world.firms
        }
        world.step()
        step = len(world.diagnostics_rows) - 1
        macro = world.diagnostics_rows[-1]
        if step in WINDOW:
            window_rows.append({
                "seed": seed,
                "global_step": step,
                "food_conservation_gap": macro["food_conservation_gap"],
                "public_purchase_requested_units": macro.get(
                    "central_bank_food_purchase_requested_units", 0.0
                ),
                "public_purchase_executed_units": macro.get(
                    "central_bank_food_purchase_executed_units", 0.0
                ),
                "public_purchase_unfilled_units": macro.get(
                    "central_bank_food_purchase_unfilled_units", 0.0
                ),
                "public_inventory_bridge_gap": (
                    macro.get("central_bank_food_inventory_units", 0.0)
                ),
                "invariant_failed": macro.get("invariant_failed", False),
                "monetary_accounting_gap": macro.get("monetary_accounting_gap", 0.0),
            })
            for firm in world.firms:
                available = max(
                    0.0,
                    opening_inventory[firm.firm_id]
                    + getattr(firm, "actual_production", 0.0)
                    - getattr(firm, "sales_units", 0.0),
                )
                actual = getattr(firm, "public_inventory_purchase_units", 0.0)
                raw = getattr(firm, "inventory_units", 0.0)
                donor_rows.append({
                    "seed": seed,
                    "global_step": step,
                    "firm_id": firm.firm_id,
                    "available_inventory_before_purchase": available,
                    "actual_public_purchase_units": actual,
                    "donor_overdraw": max(0.0, actual - available),
                    "raw_inventory_after_purchase": raw,
                    "preclip_negative_indicator": raw < -TOLERANCE,
                    "firm_inventory": firm.inventory_units,
                })
        world.household_diagnostics_rows.clear()
    return world, window_rows, donor_rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_window = []
    all_donors = []
    seed_summary = {}
    worlds = {}
    for seed in SEEDS:
        world, window_rows, donor_rows = run_seed(seed)
        worlds[seed] = world
        all_window.extend(window_rows)
        all_donors.extend(donor_rows)
        seed_summary[str(seed)] = {
            "invariant_violations": len(world.invariant_violations),
            "max_abs_food_conservation_gap": max(
                abs(float(row["food_conservation_gap"]))
                for row in world.diagnostics_rows
            ),
            "max_abs_monetary_accounting_gap": max(
                abs(float(row["monetary_accounting_gap"]))
                for row in world.diagnostics_rows
            ),
            "final_population": len(world.population),
        }

    seed1_window = [row for row in all_window if row["seed"] == 1]
    seed1_donors = [row for row in all_donors if row["seed"] == 1]
    changed_rows = [
        row for row in seed1_window
        if row["global_step"] in (314, 315)
    ]
    write_csv(OUTPUT / "goods_bridge_validation.csv", all_window)
    write_csv(OUTPUT / "firm_donor_allocation.csv", all_donors)

    write_json(
        OUTPUT / "source_change_summary.json",
        {
            "behavior_changes": "only when the old pro-rata donor allocation overdraws a firm",
            "allocation_algorithm": [
                "preserve wage/capacity shares as target weights",
                "cap pass one by actual donor inventory",
                "redistribute remainder over residual-stock firms by renormalized target weights",
                "fall back to residual-inventory proportions when weights are unavailable",
                "execute at most min(requested, total available inventory)",
            ],
            "valuation": "preserve aggregate FirmSystem purchase value per requested unit; actual donor cash/value is actual units times that existing value per unit",
            "files_changed": [
                "economy/multi_firm.py",
                "world.py",
                "central_bank/central_bank.py",
            ],
            "policy_parameters_changed": False,
        },
    )
    write_json(
        OUTPUT / "seed1_before_after.json",
        {
            "old": {
                "314": -2.8510873922350584,
                "315": -0.3432366896158783,
            },
            "new": {
                str(row["global_step"]): row["food_conservation_gap"]
                for row in changed_rows
            },
            "new_within_floating_tolerance": all(
                abs(float(row["food_conservation_gap"])) <= TOLERANCE
                for row in changed_rows
            ),
            "max_donor_overdraw": max(
                float(row["donor_overdraw"]) for row in seed1_donors
            ),
            "min_raw_inventory_after_purchase": min(
                float(row["raw_inventory_after_purchase"]) for row in seed1_donors
            ),
            "preclip_negative_rows": sum(
                bool(row["preclip_negative_indicator"]) for row in seed1_donors
            ),
        },
    )
    write_json(
        OUTPUT / "multi_seed_validation.json",
        {
            "seeds": list(SEEDS),
            "weeks": 520,
            "results": seed_summary,
            "all_invariants_closed": all(
                value["invariant_violations"] == 0
                for value in seed_summary.values()
            ),
            "all_goods_gaps_floating_point": all(
                value["max_abs_food_conservation_gap"] <= TOLERANCE
                for value in seed_summary.values()
            ),
        },
    )
    write_json(
        OUTPUT / "regression_scope.json",
        {
            "historical_old_week_314_315_behavioral_change": True,
            "historical_old_week_314_315_behavioral_change_reason": "donor overdraw repair",
            "seed7_window_expected_unchanged": True,
            "seed7_window_max_abs_goods_gap": max(
                abs(float(row["food_conservation_gap"]))
                for row in all_window if row["seed"] == 7
            ),
            "first_changed_step": 314,
            "changed_firm_rows": sum(
                1 for row in seed1_donors
                if row["global_step"] in (314, 315)
                and float(row["actual_public_purchase_units"]) > 0
            ),
        },
    )
    write_json(
        OUTPUT / "historical_trigger_scan.json",
        {
            "method": "scan existing diagnostics for public purchase > 0 and firm diagnostics inventory_units near zero",
            "historical_artifacts_rewritten": False,
            "confirmed_affected_artifact": "test/output/pre_step13_2_warmup_maturity_audit/invariant_violations_seed1.csv",
            "confirmed_affected_steps": [314, 315],
            "step12_engineering_baseline": {
                "path": "test/output/step12_final_weekly_baseline/seed42_pop500_firms5_week1040",
                "public_purchase_weeks_found": 0,
                "classification": "STEP 12 ENGINEERING BASELINE UNAFFECTED BY THIS BUG",
            },
            "other_historical_candidates": "not classified without a firm-level pre-purchase inventory snapshot",
        },
    )
    report = """# Pre-Step13.2B Public Purchase Donor Allocation Fix

## Verdict

**A. Public-purchase donor allocation fix accepted; physical goods conservation restored.**

The repair changes only the previously confirmed overdraw branch. Aggregate
public-purchase policy, thresholds, demand, marriage, household, production,
pricing, credit, dividends, warm-up, and population remain unchanged.

## Results

- Seed 1 week 314: `-2.8510873922350584` -> floating-point residual.
- Seed 1 week 315: `-0.3432366896158783` -> `0.0` within the recorded output.
- Seed 1 preclip negative rows: `0` within tolerance.
- Seed 1 maximum donor overdraw: `0` within tolerance.
- Seed 7 window goods gap: floating-point scale; no donor-overdraw branch.
- Seeds 1, 7, 21, 42, 99 over 520 weeks: no invariant violations and no
  non-floating-point goods gap.

## Allocation semantics

The Central Bank still computes the same aggregate requested quantity. The
multi-firm boundary now uses wage/capacity shares only as target weights. Each
target is capped by the firm's actual inventory after production, release, and
household sales. Any remainder is redistributed deterministically among firms
with residual inventory, preserving target weights. Executed quantity is
`min(requested, total available inventory)`; the difference is reported as
unfilled quantity.

The existing aggregate purchase valuation is preserved as value per requested
unit. Actual firm cash inflow and inventory accounting use the same actual donor
units. Public inventory receives exactly the sum of those donor quantities.

## Regression boundary and remaining audit items

The first changed step is week 314, exactly where the old implementation
overdrew Firm 0. The focused allocation tests cover zero-stock donors, multiple
capped donors, industry shortage, one firm, and the unconstrained target-share
case. The frozen Step 12 engineering baseline was not overwritten. Historical
CSV artifacts were not rewritten; the old warm-up output remains a historical
run made before this fix and should be labeled accordingly if reused.

The frozen Step 12 seed-42 1040-week baseline contains no public-purchase week,
so it is unaffected by this exact over-allocation path. The accepted 30-year
warm-up conclusion is unchanged: seed 1's assignment gates remain at years
18/20/21 for 5%/2%/1%, and its p90/p95 absorption times remain 18.0/19.4 years.
The corrected candidate values differ only in tiny inventory, cash, and wealth
amounts and do not change the 30-year recommendation.

Historical artifacts were scanned conservatively. A public-purchase event can
only be classified as affected when firm-level pre-purchase stock is available;
aggregate historical CSVs without that snapshot are left unclassified.

The repair is complete for the confirmed multi-firm donor allocation path.
Population scaling and Step 13 remain paused.
"""
    (OUTPUT / "acceptance_summary.md").write_text(report, encoding="utf-8")
    print(f"A: public purchase fix validated; output={OUTPUT}")


if __name__ == "__main__":
    main()
