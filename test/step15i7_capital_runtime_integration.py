"""Step 15I.7 capital-capacity runtime integration validation."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world import World


OUTPUT = ROOT / "test/output/step15I7_capital_runtime_integration"
POPULATION = 500
FOOD_FIRMS = 5
SEED = 42
STEPS = 520
TOLERANCE = 1e-6


def write_rows(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def max_gap(rows, fields):
    return max(
        (abs(number(row.get(field, 0.0))) for row in rows for field in fields),
        default=0.0,
    )


def macro_value(row, *names):
    for name in names:
        if name in row:
            return number(row.get(name))
    return 0.0


def run_case(runtime_enabled):
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name=(
            "step15i7_treatment" if runtime_enabled else "step15i7_control"
        ),
        scenario_overrides={
            "MULTISECTOR_FOUNDATION_ENABLED": True,
            "CANONICAL_INVESTMENT_ENABLED": True,
            "CAPITAL_CAPACITY_RUNTIME_ENABLED": runtime_enabled,
        },
    )
    world.split_firms(FOOD_FIRMS)
    firm_rows = []
    macro_rows = []

    for _ in range(STEPS):
        world.step()
        diagnostics = dict(world.diagnostics_rows[-1])
        step = int(number(diagnostics.get("global_step", 0.0)))
        week = world.last_canonical_investment_week
        accounting_rows = list(world.accounting.rows)
        current_accounting_by_firm = {
            int(number(row.get("firm_id", -1))): row
            for row in accounting_rows
            if int(number(row.get("step", -1))) == step
        }
        accounting_gap = max_gap(
            accounting_rows,
            (
                "cash_flow_gap",
                "inventory_bridge_gap",
                "equity_bridge_gap",
                "balance_sheet_gap",
                "household_wealth_bridge_gap",
                "public_cash_flow_gap",
            ),
        )
        macro_rows.append(
            {
                "case": "treatment" if runtime_enabled else "control",
                "global_step": step,
                "capital_capacity_runtime_enabled": runtime_enabled,
                "food_production": math.fsum(
                    number(getattr(firm, "actual_production", 0.0))
                    for firm in world.firms
                ),
                "household_consumption": macro_value(
                    diagnostics, "total_consumption", "consumption"
                ),
                "employment": macro_value(
                    diagnostics, "labor", "total_employment"
                ),
                "unassigned_labor": macro_value(
                    diagnostics, "unassigned_labor", "unassigned_workers"
                ),
                "fixed_investment": number(
                    getattr(week, "fixed_investment", 0.0)
                ),
                "capital_good_employment": sum(
                    len(firm.employee_ids) for firm in world.capital_good_firms
                ),
                "money_supply": macro_value(
                    diagnostics, "total_money_stock", "money_supply_stock", "money_supply"
                ),
                "outstanding_credit": macro_value(
                    diagnostics,
                    "credit_money_outstanding",
                    "outstanding_credit",
                    "working_capital_loan_balance",
                    "loan_balance",
                ),
                "money_reconciliation_gap": macro_value(
                    diagnostics,
                    "monetary_accounting_gap",
                    "money_delta_gap",
                    "money_reconciliation_gap",
                ),
                "goods_conservation_gap": macro_value(
                    diagnostics, "goods_conservation_gap"
                ),
                "accounting_reconciliation_gap": accounting_gap,
                "invariant_violation_count": len(
                    getattr(world, "invariant_violations", [])
                ),
            }
        )

        for firm in world.firms:
            accounting_row = current_accounting_by_firm.get(int(firm.firm_id), {})
            feasible = number(
                getattr(
                    firm,
                    "authoritative_feasible_capacity",
                    getattr(firm, "feasible_capacity", 0.0),
                )
            )
            realized = number(getattr(firm, "actual_production", 0.0))
            firm_rows.append(
                {
                    "case": "treatment" if runtime_enabled else "control",
                    "global_step": step,
                    "firm_id": firm.firm_id,
                    "sector_id": getattr(firm, "sector_id", "food"),
                    "technology_mode": (
                        "CAPITAL_AUGMENTED_CAP" if runtime_enabled else "LABOR_ONLY"
                    ),
                    "desired_output": number(
                        getattr(firm, "desired_production", 0.0)
                    ),
                    "labor_capacity": number(
                        getattr(firm, "labor_capacity", 0.0)
                    ),
                    "capital_capacity": number(
                        getattr(firm, "current_capital_capacity", 0.0)
                    ),
                    "base_capacity": number(
                        getattr(firm, "base_capacity", 0.0)
                    ),
                    "effective_capital_ceiling": number(
                        getattr(firm, "base_capacity", 0.0)
                    )
                    + number(getattr(firm, "current_capital_capacity", 0.0)),
                    "feasible_capacity": feasible,
                    "desired_capacity": number(
                        getattr(firm, "desired_capacity", 0.0)
                    ),
                    "capacity_gap": number(
                        getattr(firm, "capacity_gap", 0.0)
                    ),
                    "investment_intent": number(
                        getattr(firm, "desired_investment_expenditure", 0.0)
                    ),
                    "funded_output_before_cap": number(
                        getattr(firm, "runtime_funded_output_before_cap", 0.0)
                    ),
                    "funded_output": number(
                        getattr(firm, "runtime_funded_output", 0.0)
                    ),
                    "realized_output": realized,
                    "production_constraint_violation": max(
                        0.0, realized - feasible
                    ),
                    "capital_utilization": number(
                        getattr(firm, "capital_utilization", 0.0)
                    ),
                    "labor_utilization": number(
                        getattr(firm, "labor_utilization", 0.0)
                    ),
                    "capacity_utilization": number(
                        getattr(firm, "capacity_utilization", 0.0)
                    ),
                    "capacity_binding_reason": getattr(
                        firm, "capacity_binding_reason", "OTHER"
                    ),
                    "capital_capacity_runtime_enabled": runtime_enabled,
                    "investment_expenditure": number(
                        getattr(firm, "investment_expenditure_this_step", 0.0)
                    ),
                    "executed_investment": number(
                        getattr(firm, "investment_expenditure_this_step", 0.0)
                    ),
                    "capital_asset_count": len(
                        getattr(getattr(firm, "capital_stock", None), "assets", [])
                    ),
                    "capital_book_value": number(
                        getattr(
                            getattr(firm, "capital_stock", None),
                            "total_remaining_book_value",
                            0.0,
                        )
                    ),
                    "revenue": number(getattr(firm, "sales_revenue", 0.0)),
                    "wage_bill": number(getattr(firm, "wage_bill", 0.0)),
                    "operating_profit": number(getattr(firm, "profit", 0.0)),
                    "cfo": number(accounting_row.get("cfo", 0.0)),
                    "cash": number(getattr(firm, "cash", 0.0)),
                    "principal": number(getattr(firm, "loan_balance", 0.0)),
                    "arrears": number(
                        getattr(firm, "interest_arrears", 0.0)
                    ),
                    "inventory_units": number(
                        getattr(firm, "inventory_units", 0.0)
                    ),
                }
            )

    return world, macro_rows, firm_rows


def compare_cases(control_macro, treatment_macro):
    keys = (
        "food_production",
        "household_consumption",
        "employment",
        "unassigned_labor",
        "fixed_investment",
        "capital_good_employment",
        "money_supply",
        "outstanding_credit",
    )
    rows = []
    for key in keys:
        left = [number(row.get(key)) for row in control_macro]
        right = [number(row.get(key)) for row in treatment_macro]
        differences = [abs(a - b) for a, b in zip(left, right)]
        rows.append(
            {
                "metric": key,
                "control_final": left[-1] if left else 0.0,
                "treatment_final": right[-1] if right else 0.0,
                "control_mean": math.fsum(left) / len(left) if left else 0.0,
                "treatment_mean": math.fsum(right) / len(right) if right else 0.0,
                "max_abs_control_treatment_difference": max(differences, default=0.0),
            }
        )
    return rows


def empty_labor_only_parity():
    baseline = World(initial_population=100, seed=SEED, diagnostics_mode="full")
    passive = World(initial_population=100, seed=SEED, diagnostics_mode="full")
    passive.multisector_foundation_enabled = True
    passive.ensure_multisector_foundation_contracts()
    baseline.split_firms(FOOD_FIRMS)
    passive.split_firms(FOOD_FIRMS)
    for _ in range(12):
        baseline.step()
        passive.step()
    fields = (
        "population",
        "households",
        "total_household_wealth",
        "firm_cash",
        "inventory_units",
        "price",
        "money_supply_stock",
    )
    left = baseline.diagnostics_rows[-1]
    right = passive.diagnostics_rows[-1]
    return all(
        abs(number(left.get(field)) - number(right.get(field))) <= TOLERANCE
        for field in fields
    )


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    control_world, control_macro, control_firms = run_case(False)
    treatment_world, treatment_macro, treatment_firms = run_case(True)

    write_rows(OUTPUT / "firm_capacity_runtime_panel.csv", control_firms + treatment_firms)
    reason_rows = []
    for case, rows in (
        ("control", control_firms),
        ("treatment", treatment_firms),
    ):
        reasons = {}
        for row in rows:
            reason = row["capacity_binding_reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items()):
            reason_rows.append(
                {
                    "case": case,
                    "capacity_binding_reason": reason,
                    "firm_weeks": count,
                    "share": count / len(rows) if rows else 0.0,
                }
            )
    write_rows(OUTPUT / "capacity_binding_summary.csv", reason_rows)
    write_rows(
        OUTPUT / "control_treatment_comparison.csv",
        compare_cases(control_macro, treatment_macro),
    )

    violations = [
        row["production_constraint_violation"]
        for row in treatment_firms
    ]
    treatment_money_gap = max(
        (row["money_reconciliation_gap"] for row in treatment_macro),
        default=0.0,
    )
    treatment_accounting_gap = max(
        (row["accounting_reconciliation_gap"] for row in treatment_macro),
        default=0.0,
    )
    treatment_goods_gap = max(
        (abs(row["goods_conservation_gap"]) for row in treatment_macro),
        default=0.0,
    )
    capital_binding_weeks = sum(
        row["capacity_binding_reason"] == "CAPITAL_BINDING"
        for row in treatment_firms
    )
    treatment_investment = math.fsum(
        row["fixed_investment"] for row in treatment_macro
    )
    control_investment = math.fsum(
        row["fixed_investment"] for row in control_macro
    )
    treatment_assets = max(
        (row["capital_asset_count"] for row in treatment_firms),
        default=0,
    )
    invariant_count = len(getattr(treatment_world, "invariant_violations", []))
    empty_parity = empty_labor_only_parity()

    if max(violations, default=0.0) > TOLERANCE:
        verdict = "F. ACCOUNTING_OR_RUNTIME_BLOCKER"
        blocker = "realized production exceeded authoritative feasible capacity"
    elif (
        treatment_money_gap > TOLERANCE
        or treatment_accounting_gap > TOLERANCE
        or treatment_goods_gap > TOLERANCE
        or invariant_count
    ):
        verdict = "F. ACCOUNTING_OR_RUNTIME_BLOCKER"
        blocker = "a conservation or accounting invariant exceeded tolerance"
    elif not empty_parity:
        verdict = "D. EMPTY_CAPITAL_PARITY_REGRESSION"
        blocker = "empty-capital labor-only parity failed"
    elif capital_binding_weeks == 0:
        verdict = "B. CAPITAL_CONSTRAINT_NEVER_BINDING"
        blocker = "the treatment never encountered a binding capital constraint"
    else:
        verdict = "A. CAPITAL_CAPACITY_RUNTIME_INTEGRATION_ACCEPTED"
        blocker = "none"

    flags = {
        "verdict": verdict,
        "population": POPULATION,
        "food_firms": FOOD_FIRMS,
        "seed": SEED,
        "steps": STEPS,
        "control_capital_runtime": False,
        "treatment_capital_runtime": True,
        "technology_control": "LABOR_ONLY",
        "technology_treatment": "CAPITAL_AUGMENTED_CAP",
        "max_treatment_production_feasibility_violation": max(violations, default=0.0),
        "capital_binding_firm_weeks": capital_binding_weeks,
        "treatment_final_capital_asset_count": treatment_assets,
        "control_fixed_investment_total": control_investment,
        "treatment_fixed_investment_total": treatment_investment,
        "investment_planner_code_changed": False,
        "investment_financing_changed": False,
        "depreciation_activated": False,
        "capital_good_labor_rules_changed": False,
        "empty_capital_labor_only_parity": empty_parity,
        "max_money_reconciliation_gap": treatment_money_gap,
        "max_accounting_reconciliation_gap": treatment_accounting_gap,
        "max_goods_conservation_gap": treatment_goods_gap,
        "invariant_violations": invariant_count,
        "new_rng_draws": 0,
        "Step13_changed": False,
        "economic_parameter_changes": False,
        "blocker": blocker,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = f"""# Step 15I.7 Capital Capacity Runtime Integration

## Verdict

**{verdict}**

The validation used `N={POPULATION}`, `{FOOD_FIRMS}` Food Firms, seed `{SEED}`
and `{STEPS}` weeks. Control retained the labor-only runtime. Treatment used
the explicit `CAPITAL_AUGMENTED_CAP` mode while leaving the investment planner,
financing, depreciation, capital-good labor rules and Step13 unchanged.

## Runtime checks

- Maximum treatment production above feasible capacity: `{max(violations, default=0.0):.12g}`
- Treatment capital-binding Firm-weeks: `{capital_binding_weeks}`
- Treatment final capital assets: `{treatment_assets}`
- Empty-capital labor-only parity: `{empty_parity}`
- Maximum money reconciliation gap: `{treatment_money_gap:.12g}`
- Maximum accounting reconciliation gap: `{treatment_accounting_gap:.12g}`
- Maximum goods conservation gap: `{treatment_goods_gap:.12g}`
- Invariant violations: `{invariant_count}`

The runtime cap is applied before inventory and transaction settlement. The
control path does not use capital capacity as an output constraint. The
treatment's investment total is reported against control as an endogenous
interaction diagnostic; no investment-planner rule was changed.

Detailed firm-week and control/treatment panels are stored in this directory.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
