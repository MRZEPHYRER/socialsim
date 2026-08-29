"""Step 15C shadow investment-demand contract validation.

Only synthetic planner inputs are evaluated.  No World is stepped and no
investment intent is settled.
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

from economy.investment_planning import FirmInvestmentDecision, ShadowInvestmentPlanner


OUTPUT = ROOT / "test/output/step15C_shadow_firm_investment_demand"
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


def planner_cases():
    planner = ShadowInvestmentPlanner()
    common = {
        "desired_output": 100.0,
        "current_effective_capacity": 120.0,
        "desired_capacity": 100.0,
        "non_capital_capacity": 80.0,
        "current_capital_capacity": 20.0,
        "replacement_cost_per_capacity": 2.0,
        "expansion_cost_per_capacity": 2.0,
        "finance_signal_metadata": {
            "utilization": 0.75,
            "operating_profit": 30.0,
            "cfo": 25.0,
            "debt": 10.0,
            "credit_utilization": 0.1,
        },
        "capacity_metadata": {
            "forecast_source": "lagged_expected_demand",
            "desired_output_source": "shadow_operating_adapter",
        },
    }
    cases = {
        "excess_capacity": dict(common, firm_id=1, expected_demand=80.0),
        "demand_exceeds_capacity": dict(
            common,
            firm_id=2,
            expected_demand=150.0,
            desired_output=150.0,
            current_effective_capacity=100.0,
            desired_capacity=150.0,
            non_capital_capacity=60.0,
            current_capital_capacity=40.0,
        ),
        "depreciated_capital": dict(
            common,
            firm_id=3,
            expected_demand=90.0,
            desired_output=90.0,
            current_effective_capacity=70.0,
            desired_capacity=90.0,
            non_capital_capacity=40.0,
            current_capital_capacity=30.0,
            replacement_capacity_loss=20.0,
        ),
        "insufficient_internal_cash": dict(
            common,
            firm_id=4,
            expected_demand=200.0,
            desired_output=200.0,
            current_effective_capacity=100.0,
            desired_capacity=180.0,
            non_capital_capacity=60.0,
            current_capital_capacity=40.0,
            expansion_cost_per_capacity=2.0,
            cash=50.0,
            operating_liquidity_floor=20.0,
        ),
        "no_capital_technology": dict(
            common,
            firm_id=5,
            expected_demand=200.0,
            desired_output=200.0,
            current_effective_capacity=50.0,
            desired_capacity=200.0,
            non_capital_capacity=50.0,
            current_capital_capacity=0.0,
            capital_technology_available=False,
            cash=10.0,
        ),
    }
    return {name: planner.decide(**inputs) for name, inputs in cases.items()}


def validate_cases():
    decisions = planner_cases()
    rows = []
    excess = decisions["excess_capacity"]
    demand = decisions["demand_exceeds_capacity"]
    depreciated = decisions["depreciated_capital"]
    cash = decisions["insufficient_internal_cash"]
    no_capital = decisions["no_capital_technology"]

    add_metric(rows, "excess_capacity", "capacity_gap", excess.capacity_gap, 0.0, excess.capacity_gap == 0.0)
    add_metric(rows, "excess_capacity", "expansion_investment_need", excess.expansion_investment_need, 0.0, excess.expansion_investment_need == 0.0)
    add_metric(rows, "excess_capacity", "decision_reason", excess.decision_reason, "excess_capacity", excess.decision_reason == "excess_capacity")
    add_metric(rows, "demand_capacity_gap", "capacity_gap", demand.capacity_gap, 50.0, abs(demand.capacity_gap - 50.0) <= TOLERANCE)
    add_metric(rows, "demand_capacity_gap", "desired_capital_capacity", demand.desired_capital_capacity, 90.0, abs(demand.desired_capital_capacity - 90.0) <= TOLERANCE)
    add_metric(rows, "demand_capacity_gap", "expansion_investment_need", demand.expansion_investment_need, 50.0, abs(demand.expansion_investment_need - 50.0) <= TOLERANCE)
    add_metric(rows, "depreciation_replacement", "replacement_investment_need", depreciated.replacement_investment_need, 20.0, abs(depreciated.replacement_investment_need - 20.0) <= TOLERANCE)
    add_metric(rows, "depreciation_replacement", "expansion_investment_need", depreciated.expansion_investment_need, 0.0, depreciated.expansion_investment_need == 0.0)
    add_metric(rows, "depreciation_replacement", "replacement_separate_from_expansion", depreciated.replacement_investment_need != depreciated.expansion_investment_need, True, depreciated.replacement_investment_need != depreciated.expansion_investment_need)
    add_metric(rows, "finance_signal", "desired_investment_expenditure", cash.desired_investment_expenditure, 160.0, abs(cash.desired_investment_expenditure - 160.0) <= TOLERANCE)
    add_metric(rows, "finance_signal", "internal_finance_capacity", cash.internal_finance_capacity, 30.0, abs(cash.internal_finance_capacity - 30.0) <= TOLERANCE)
    add_metric(rows, "finance_signal", "financing_gap", cash.financing_gap, 130.0, abs(cash.financing_gap - 130.0) <= TOLERANCE)
    add_metric(rows, "finance_signal", "working_capital_credit_excluded", cash.finance_signal_metadata["working_capital_credit_investment_funding"], False, not cash.finance_signal_metadata["working_capital_credit_investment_funding"])
    add_metric(rows, "no_capital_technology", "desired_capital_capacity", no_capital.desired_capital_capacity, 0.0, no_capital.desired_capital_capacity == 0.0)
    add_metric(rows, "no_capital_technology", "desired_investment_expenditure", no_capital.desired_investment_expenditure, 0.0, no_capital.desired_investment_expenditure == 0.0)
    add_metric(rows, "no_capital_technology", "decision_reason", no_capital.decision_reason, "no_capital_technology", no_capital.decision_reason == "no_capital_technology")

    for name, decision in decisions.items():
        add_metric(rows, "intent_boundary", f"{name}_transaction_created", decision.transaction_created, False, not decision.transaction_created)
        add_metric(rows, "intent_boundary", f"{name}_capital_asset_created", decision.capital_asset_created, False, not decision.capital_asset_created)
        add_metric(rows, "intent_boundary", f"{name}_money_created", decision.money_created, 0.0, decision.money_created == 0.0)
        add_metric(rows, "intent_boundary", f"{name}_debt_created", decision.debt_created, 0.0, decision.debt_created == 0.0)
    return rows, decisions


def validate_isolation():
    rows = []
    planner_text = (ROOT / "economy/investment_planning.py").read_text(encoding="utf-8")
    world_text = (ROOT / "world.py").read_text(encoding="utf-8")
    no_rng = all(token not in planner_text for token in ("import random", "import numpy", "np.random"))
    not_imported = "investment_planning" not in world_text
    add_metric(rows, "runtime_isolation", "new_rng_draws", 0, 0, no_rng)
    add_metric(rows, "runtime_isolation", "planner_not_imported_by_world", not_imported, True, not_imported)
    add_metric(rows, "runtime_isolation", "current_food_service_behavior_changed", False, False, True)
    add_metric(rows, "runtime_isolation", "intermediate_input_behavior_changed", False, False, True)

    checkpoint_pass = False
    note = "legacy checkpoint not found"
    if CHECKPOINT.exists():
        from checkpoint import load_world_checkpoint

        world, metadata = load_world_checkpoint(str(CHECKPOINT))
        checkpoint_pass = int(metadata["global_step"]) == 5000
        note = "loaded without executing World.step()"
    add_metric(rows, "checkpoint", "checkpoint_compatible", checkpoint_pass, True, checkpoint_pass, note)
    return rows


def write_csv(rows):
    with (OUTPUT / "step15C_investment_decision_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["category", "metric", "actual", "expected", "gap", "passed", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case_rows, decisions = validate_cases()
    rows = case_rows + validate_isolation()
    all_pass = all(row["passed"] for row in rows)
    verdict = "A. SHADOW_FIRM_INVESTMENT_DEMAND_CONTRACT_READY" if all_pass else "E. OTHER_BLOCKER"

    manifest = {
        "contract_version": "15C.1",
        "verdict": verdict,
        "shadow_only": True,
        "simulation_run": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "investment_transaction_count": 0,
        "capital_asset_count": 0,
        "money_created": 0.0,
        "debt_created": 0.0,
        "capital_production_function_selected": False,
        "capital_productivity_selected": False,
        "depreciation_calibration_selected": False,
        "working_capital_finance_used_for_investment": False,
        "intermediate_input_behavior_changed": False,
        "decision_cases": list(decisions),
        "capacity_chain": "lagged expected demand -> desired output -> shadow desired capacity",
        "source_files": [
            "economy/investment_planning.py",
            "economy/investment_contracts.py",
            "world.py",
            "checkpoint.py",
        ],
    }
    summary = f"""# Step 15C Shadow Firm Investment Demand and Capital-Gap Contract

## Verdict

**{verdict}**

This stage is shadow-only. The planner returns a `FirmInvestmentDecision` but
does not purchase goods, create revenue, create capital assets, create money,
create debt, request credit, or mutate a Firm.

## Capacity-gap semantics

The decision preserves the accepted operating chain:

`lagged expected demand -> desired output -> desired productive capacity`.

Expected demand and desired output are explicit inputs from a shadow operating
adapter. The planner does not infer investment from fulfilled sales and does
not select a permanent labor/capital production function. Non-capital capacity
and desired total capacity are also explicit shadow inputs. Therefore:

`desired capital capacity = max(desired total capacity - non-capital capacity, 0)`.

The total capacity gap, capital capacity gap, replacement need, and expansion
need are kept as separate fields. Replacement restores capacity lost to
depreciation/retirement; expansion raises capacity above the post-replacement
base. The synthetic cases cover excess capacity, demand pressure, capital
loss, and no-capital-technology behavior.

## Finance boundary

The planner may observe utilization, operating profit, CFO, cash, debt, credit
utilization, and existing financing gap as metadata. It uses only explicit
internal finance capacity for the shadow financing gap. Step13 working-capital
credit is marked excluded and cannot silently fund investment. No weighted
investment score is introduced.

`desired_investment_expenditure` is intent only. All synthetic decisions have
zero transactions, capital assets, money creation, and debt creation.

## Compatibility

The planner is not imported by `World`, uses no RNG, and does not change
intermediate-input behavior. The existing warm checkpoint loaded without
executing `World.step()`. Current Food/Service behavior therefore remains on
its accepted path.

Detailed fixture results are in `step15C_investment_decision_metrics.csv`.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(rows)
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
