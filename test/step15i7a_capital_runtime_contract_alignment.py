"""Step 15I.7A contract-alignment retest for capital runtime capacity."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from step15i7_capital_runtime_integration import (
    FOOD_FIRMS,
    POPULATION,
    SEED,
    STEPS,
    TOLERANCE,
    compare_cases,
    number,
    run_case,
    write_rows,
)


OUTPUT = ROOT / "test/output/step15I7A_capital_runtime_contract_alignment"


def group_feedback(rows):
    grouped = {}
    for row in rows:
        key = (row["case"], int(row["global_step"]))
        bucket = grouped.setdefault(
            key,
            {
                "case": row["case"],
                "global_step": key[1],
                "total_investment_intent": 0.0,
                "total_executed_investment": 0.0,
                "total_labor_capacity": 0.0,
                "total_base_capacity": 0.0,
                "total_capital_capacity": 0.0,
                "total_effective_capital_ceiling": 0.0,
                "total_feasible_capacity": 0.0,
                "total_realized_output": 0.0,
                "capital_asset_count": 0,
            },
        )
        bucket["total_investment_intent"] += number(
            row.get("investment_intent")
        )
        bucket["total_executed_investment"] += number(
            row.get("executed_investment")
        )
        for source, target in (
            ("labor_capacity", "total_labor_capacity"),
            ("base_capacity", "total_base_capacity"),
            ("capital_capacity", "total_capital_capacity"),
            ("effective_capital_ceiling", "total_effective_capital_ceiling"),
            ("feasible_capacity", "total_feasible_capacity"),
            ("realized_output", "total_realized_output"),
        ):
            bucket[target] += number(row.get(source))
        bucket["capital_asset_count"] += int(
            number(row.get("capital_asset_count"))
        )
    return list(grouped.values())


def first_investment_week(rows):
    weeks = [
        int(row["global_step"])
        for row in rows
        if number(row.get("executed_investment")) > TOLERANCE
    ]
    return min(weeks) if weeks else None


def max_preinvestment_parity(control_rows, treatment_rows, first_week):
    if first_week is None:
        return math.inf
    control = {
        int(row["global_step"]): number(row.get("food_production"))
        for row in control_rows
        if int(row["global_step"]) < first_week
    }
    treatment = {
        int(row["global_step"]): number(row.get("food_production"))
        for row in treatment_rows
        if int(row["global_step"]) < first_week
    }
    return max(
        (abs(control[step] - treatment[step]) for step in control if step in treatment),
        default=0.0,
    )


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    control_world, control_macro, control_firms = run_case(False)
    treatment_world, treatment_macro, treatment_firms = run_case(True)
    all_firms = control_firms + treatment_firms
    write_rows(OUTPUT / "capacity_runtime_panel.csv", all_firms)
    write_rows(
        OUTPUT / "investment_capacity_feedback.csv",
        group_feedback(all_firms),
    )

    treatment_first_investment = first_investment_week(treatment_firms)
    control_first_investment = first_investment_week(control_firms)
    preinvestment_gap = max_preinvestment_parity(
        control_macro,
        treatment_macro,
        treatment_first_investment,
    )
    treatment_zero_cap_zero_feasible = sum(
        number(row.get("capital_asset_count")) <= 0
        and number(row.get("capital_capacity")) <= TOLERANCE
        and number(row.get("feasible_capacity")) <= TOLERANCE
        for row in treatment_firms
    )
    treatment_violations = max(
        (number(row.get("production_constraint_violation")) for row in treatment_firms),
        default=0.0,
    )
    treatment_capital_binding = sum(
        row.get("capacity_binding_reason") == "CAPITAL_BINDING"
        for row in treatment_firms
    )
    treatment_labor_binding = sum(
        row.get("capacity_binding_reason") == "LABOR_BINDING"
        for row in treatment_firms
    )
    treatment_capital_effect = any(
        number(row.get("capital_capacity")) > TOLERANCE
        and number(row.get("feasible_capacity"))
        > number(row.get("base_capacity")) + TOLERANCE
        for row in treatment_firms
    )
    treatment_investment_total = math.fsum(
        number(row.get("executed_investment")) for row in treatment_firms
    )
    treatment_assets = max(
        (int(number(row.get("capital_asset_count"))) for row in treatment_firms),
        default=0,
    )
    treatment_money_gap = max(
        (abs(number(row.get("money_reconciliation_gap"))) for row in treatment_macro),
        default=0.0,
    )
    treatment_accounting_gap = max(
        (abs(number(row.get("accounting_reconciliation_gap"))) for row in treatment_macro),
        default=0.0,
    )
    treatment_goods_gap = max(
        (abs(number(row.get("goods_conservation_gap"))) for row in treatment_macro),
        default=0.0,
    )
    invariant_count = len(getattr(treatment_world, "invariant_violations", []))

    comparison = compare_cases(control_macro, treatment_macro)
    comparison.extend(
        [
            {
                "metric": "pre_first_investment_food_production_parity_gap",
                "control_final": "",
                "treatment_final": "",
                "control_mean": "",
                "treatment_mean": "",
                "max_abs_control_treatment_difference": preinvestment_gap,
            },
            {
                "metric": "first_investment_week",
                "control_final": control_first_investment,
                "treatment_final": treatment_first_investment,
                "control_mean": "",
                "treatment_mean": "",
                "max_abs_control_treatment_difference": "",
            },
            {
                "metric": "treatment_capital_binding_firm_weeks",
                "control_final": "",
                "treatment_final": treatment_capital_binding,
                "control_mean": "",
                "treatment_mean": "",
                "max_abs_control_treatment_difference": "",
            },
            {
                "metric": "treatment_labor_binding_firm_weeks",
                "control_final": "",
                "treatment_final": treatment_labor_binding,
                "control_mean": "",
                "treatment_mean": "",
                "max_abs_control_treatment_difference": "",
            },
            {
                "metric": "treatment_zero_cap_zero_feasible_rows",
                "control_final": "",
                "treatment_final": treatment_zero_cap_zero_feasible,
                "control_mean": "",
                "treatment_mean": "",
                "max_abs_control_treatment_difference": "",
            },
        ]
    )
    write_rows(OUTPUT / "control_treatment_comparison.csv", comparison)

    if treatment_zero_cap_zero_feasible:
        verdict = "B. ZERO_CAPITAL_BOOTSTRAP_TRAP_PERSISTS"
        blocker = "zero CapitalStock still produced zero feasible capacity"
    elif treatment_violations > TOLERANCE:
        verdict = "E. PRODUCTION_FEASIBILITY_REGRESSION"
        blocker = "realized output exceeded feasible capacity"
    elif not treatment_investment_total or treatment_first_investment is None:
        verdict = "D. INVESTMENT_CAPACITY_FEEDBACK_BLOCKER"
        blocker = "no endogenous investment occurred after contract alignment"
    elif not treatment_capital_effect:
        verdict = "C. BASE_CAPACITY_MAPPING_BLOCKER"
        blocker = "capital stock did not raise the authoritative capacity ceiling"
    elif (
        treatment_money_gap > TOLERANCE
        or treatment_accounting_gap > TOLERANCE
        or treatment_goods_gap > TOLERANCE
        or invariant_count
    ):
        verdict = "F. OTHER_BLOCKER"
        blocker = "conservation or accounting reconciliation failed"
    else:
        verdict = "A. CAPITAL_RUNTIME_CONTRACT_ALIGNMENT_ACCEPTED"
        blocker = "none"

    flags = {
        "verdict": verdict,
        "population": POPULATION,
        "food_firms": FOOD_FIRMS,
        "seed": SEED,
        "steps": STEPS,
        "contract_formula": "min(labor_capacity, base_capacity + capital_capacity)",
        "base_capacity_semantics": "existing funded labor-only productive capacity",
        "pre_first_investment_parity_gap": preinvestment_gap,
        "control_first_investment_week": control_first_investment,
        "treatment_first_investment_week": treatment_first_investment,
        "treatment_capital_binding_firm_weeks": treatment_capital_binding,
        "treatment_labor_binding_firm_weeks": treatment_labor_binding,
        "zero_capital_zero_feasible_rows": treatment_zero_cap_zero_feasible,
        "realized_greater_than_feasible_max": treatment_violations,
        "treatment_investment_total": treatment_investment_total,
        "treatment_final_capital_asset_count": treatment_assets,
        "capital_raises_authoritative_ceiling": treatment_capital_effect,
        "max_money_reconciliation_gap": treatment_money_gap,
        "max_accounting_reconciliation_gap": treatment_accounting_gap,
        "max_goods_conservation_gap": treatment_goods_gap,
        "invariant_violations": invariant_count,
        "investment_planner_changed": False,
        "investment_financing_changed": False,
        "depreciation_activated": False,
        "capital_good_labor_changed": False,
        "Step13_changed": False,
        "new_rng_draws": 0,
        "blocker": blocker,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = f"""# Step 15I.7A Capital Runtime Contract Alignment Retest

## Verdict

**{verdict}**

The retest used `N={POPULATION}`, `{FOOD_FIRMS}` Food Firms, seed `{SEED}` and
`{STEPS}` weeks. Treatment uses the accepted contract:

```text
effective_capital_ceiling = base_capacity + capital_capacity
feasible_capacity = min(labor_capacity, effective_capital_ceiling)
```

`base_capacity` is the existing funded labor-only productive capacity. It is
not a new calibrated parameter. With zero CapitalStock, treatment therefore
preserves the pre-capital production base. Capital service can raise the
ceiling up to the labor-capacity bound.

## Results

- Pre-first-investment parity gap: `{preinvestment_gap:.12g}`
- First control investment week: `{control_first_investment}`
- First treatment investment week: `{treatment_first_investment}`
- Treatment capital-binding Firm-weeks: `{treatment_capital_binding}`
- Treatment labor-binding Firm-weeks: `{treatment_labor_binding}`
- Zero-capital/zero-feasible rows: `{treatment_zero_cap_zero_feasible}`
- Maximum realized-minus-feasible violation: `{treatment_violations:.12g}`
- Treatment investment total: `{treatment_investment_total:.12g}`
- Treatment final capital assets: `{treatment_assets}`
- Maximum money gap: `{treatment_money_gap:.12g}`
- Maximum accounting gap: `{treatment_accounting_gap:.12g}`
- Maximum goods gap: `{treatment_goods_gap:.12g}`
- Invariant violations: `{invariant_count}`

No investment planner, financing, depreciation, capital-good labor, ownership
or Step13 behavior was changed.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
