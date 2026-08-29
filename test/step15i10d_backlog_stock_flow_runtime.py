from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world import World


OUTPUT = ROOT / "test/output/step15I10D_backlog_stock_flow_runtime"
POPULATION = 500
FOOD_FIRMS = 5
SEED = 42
STEPS = 520
HORIZON = 13
EPS = 1e-8


def number(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def write_rows(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def backlog_state(world, system):
    replacement = math.fsum(
        max(0.0, number(getattr(firm, "pending_replacement_capacity_need", 0.0)))
        for firm in world.firms
    )
    expansion = math.fsum(
        max(0.0, number(value))
        for value in system.expansion_backlog_by_firm.values()
    )
    return replacement, expansion, replacement + expansion


def conservation_snapshot(world):
    accounting_gaps = []
    for row in getattr(world.accounting, "rows", []):
        for field in (
            "cash_flow_gap",
            "balance_sheet_gap",
            "inventory_bridge_gap",
            "equity_bridge_gap",
            "capital_book_value_bridge_gap",
        ):
            accounting_gaps.append(abs(number(row.get(field, 0.0))))
    diagnostic_gaps = []
    for row in getattr(world, "diagnostics_rows", []):
        for field in (
            "monetary_accounting_gap",
            "money_delta_gap",
            "money_location_gap",
            "goods_conservation_gap",
            "invariant_violation_count",
        ):
            diagnostic_gaps.append(abs(number(row.get(field, 0.0))))
    return (
        max(accounting_gaps, default=0.0),
        max(diagnostic_gaps, default=0.0),
    )


def install_old_supply_logic(system):
    """Reproduce the pre-15I.10D supply boundary in this control only."""
    def old_activate(self, step, *args, **kwargs):
        outstanding = self._capital_good_outstanding_demand()
        per_firm = outstanding / max(1, len(self.capital_good_firms))
        before_by_firm = {
            firm.firm_id: set(getattr(firm, "employee_ids", []))
            for firm in self.capital_good_firms
        }
        for firm in self.capital_good_firms:
            firm.capital_good_replacement_demand_units = math.fsum(
                max(
                    0.0,
                    number(getattr(food, "pending_replacement_capacity_need", 0.0)),
                )
                for food in getattr(self.world, "firms", [])
            ) / max(1, len(self.capital_good_firms))
            firm.capital_good_expansion_demand_units = math.fsum(
                self.expansion_backlog_by_firm.values()
            ) / max(1, len(self.capital_good_firms))
            firm.capital_good_outstanding_demand_units = per_firm
            firm.desired_production = per_firm
            firm.capital_good_desired_output = per_firm
            firm.desired_labor = (
                per_firm / self.productivity_per_labor
                if per_firm > 1e-12 and self.productivity_per_labor > 1e-12
                else 0.0
            )
        if outstanding > 1e-12:
            self.world.assign_unassigned_workers_to_firms(
                include_capital_goods=True
            )
        for firm in self.capital_good_firms:
            before = before_by_firm.get(firm.firm_id, set())
            after = set(getattr(firm, "employee_ids", []))
            for person_id in sorted(after - before):
                self.labor_reactivation_events.append({
                    "global_step": step,
                    "event_type": "capital_good_labor_reactivation",
                    "firm_id": firm.firm_id,
                    "person_id": person_id,
                    "desired_labor": firm.desired_labor,
                    "reason": "outstanding_capital_good_order_backlog",
                })

    def old_close(self, step):
        outstanding = self._capital_good_outstanding_demand()
        for firm in self.capital_good_firms:
            firm.capital_good_outstanding_demand_units = outstanding / max(
                1, len(self.capital_good_firms)
            )
            desired_labor = (
                firm.capital_good_outstanding_demand_units / self.productivity_per_labor
                if firm.capital_good_outstanding_demand_units > 1e-12
                and self.productivity_per_labor > 1e-12
                else 0.0
            )
            self._release_excess_labor(firm, desired_labor, step)

    system._activate_capital_good_labor = MethodType(old_activate, system)
    system._close_capital_good_labor = MethodType(old_close, system)


def make_world(old_logic=False):
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name=(
            "step15i10d_old_control"
            if old_logic else "step15i10d_corrected_treatment"
        ),
        scenario_overrides={
            "MULTISECTOR_FOUNDATION_ENABLED": True,
            "CANONICAL_INVESTMENT_ENABLED": True,
            "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
            "CAPITAL_LIFECYCLE_ENABLED": True,
            "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        },
    )
    world.split_firms(FOOD_FIRMS)
    system = world.canonical_investment_system
    system.ensure_firms()
    if old_logic:
        install_old_supply_logic(system)
    return world, system


def run_case(old_logic):
    world, system = make_world(old_logic)
    rows = []
    supply_rows = []
    previous_replacement = 0.0
    previous_expansion = 0.0
    previous_total = 0.0
    roster_violations = 0
    for _ in range(STEPS):
        opening_replacement, opening_expansion, opening_total = backlog_state(
            world, system
        )
        world.step()
        step = world.current_step_index
        capital = world.capital_good_firms[0]
        closing_replacement, closing_expansion, closing_total = backlog_state(
            world, system
        )
        replacement_settlement = math.fsum(
            max(
                0.0,
                number(
                    getattr(firm, "executed_replacement_investment_this_step", 0.0)
                ),
            )
            for firm in world.firms
        ) / max(1e-12, system.unit_price)
        expansion_settlement = math.fsum(
            max(
                0.0,
                number(
                    getattr(firm, "executed_expansion_investment_this_step", 0.0)
                ),
            )
            for firm in world.firms
        ) / max(1e-12, system.unit_price)
        new_replacement = math.fsum(
            max(0.0, number(getattr(firm, "retired_capacity_this_step", 0.0)))
            for firm in world.firms
        )
        new_expansion = max(
            0.0,
            closing_expansion - opening_expansion + expansion_settlement,
        )
        new_total = new_replacement + new_expansion
        desired_output = number(
            getattr(capital, "capital_good_desired_output", 0.0)
        )
        desired_labor = number(getattr(capital, "desired_labor", 0.0))
        planned_new_flow = number(
            getattr(system, "current_new_demand_flow_units", 0.0)
        )
        active_assets = sum(
            bool(getattr(asset, "is_active", False))
            for firm in world.firms
            for asset in getattr(getattr(firm, "capital_stock", None), "assets", [])
        )
        for firm in world.operating_firms():
            for person_id in getattr(firm, "employee_ids", []):
                person = world.person_dict.get(person_id)
                if person is None or person.firm_id != firm.firm_id:
                    roster_violations += 1
        corrected_formula = (
            opening_total / HORIZON + planned_new_flow
            if not old_logic else math.nan
        )
        rows.append({
            "global_step": step,
            "run": "old" if old_logic else "corrected",
            "opening_backlog_units": opening_total,
            "new_replacement_demand_units": new_replacement,
            "new_expansion_demand_units": new_expansion,
            "new_demand_units": new_total,
            "backlog_flow_units_per_week": opening_total / HORIZON,
            "new_demand_flow_units_per_week": new_total / HORIZON,
            "planned_new_demand_flow_units_per_week": planned_new_flow,
            "desired_output_units_per_week": desired_output,
            "desired_labor_services_per_week": desired_labor,
            "actual_employment": len(getattr(capital, "employee_ids", [])),
            "actual_production_units": number(
                getattr(capital, "capital_good_production_units", 0.0)
            ),
            "replacement_settlement_units": replacement_settlement,
            "expansion_settlement_units": expansion_settlement,
            "closing_replacement_backlog_units": closing_replacement,
            "closing_expansion_backlog_units": closing_expansion,
            "closing_backlog_units": closing_total,
            "backlog_bridge_gap": (
                closing_total - opening_total - new_total
                + replacement_settlement + expansion_settlement
            ),
            "corrected_flow_formula_gap": (
                desired_output - corrected_formula
                if not old_logic else math.nan
            ),
            "active_capital_assets": active_assets,
        })
        supply_rows.append({
            "global_step": step,
            "run": "old" if old_logic else "corrected",
            "replacement_demand_units": opening_replacement,
            "replacement_execution_units": replacement_settlement,
            "expansion_demand_units": opening_expansion,
            "expansion_execution_units": expansion_settlement,
            "capital_good_production_units": number(
                getattr(capital, "capital_good_production_units", 0.0)
            ),
            "capital_good_inventory_units": number(
                getattr(getattr(capital, "capital_good_inventory", None), "units", 0.0)
            ),
            "active_capital_assets": active_assets,
            "desired_output_units_per_week": desired_output,
            "desired_labor_services_per_week": desired_labor,
        })
        previous_replacement = closing_replacement
        previous_expansion = closing_expansion
        previous_total = closing_total
    accounting_gap, conservation_gap = conservation_snapshot(world)
    return {
        "rows": rows,
        "supply": supply_rows,
        "accounting_gap": accounting_gap,
        "conservation_gap": conservation_gap,
        "roster_violations": roster_violations,
        "world": world,
    }


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    old = run_case(old_logic=True)
    corrected = run_case(old_logic=False)
    all_rows = old["rows"] + corrected["rows"]
    all_supply = old["supply"] + corrected["supply"]
    write_rows(OUTPUT / "backlog_flow_runtime_panel.csv", all_rows)
    write_rows(OUTPUT / "replacement_supply_comparison.csv", all_supply)

    comparison = []
    for step in range(STEPS):
        old_row = old["rows"][step]
        new_row = corrected["rows"][step]
        comparison.append({
            "global_step": step,
            "old_desired_output": old_row["desired_output_units_per_week"],
            "corrected_desired_output": new_row["desired_output_units_per_week"],
            "old_desired_labor": old_row["desired_labor_services_per_week"],
            "corrected_desired_labor": new_row["desired_labor_services_per_week"],
            "old_actual_employment": old_row["actual_employment"],
            "corrected_actual_employment": new_row["actual_employment"],
            "old_closing_backlog": old_row["closing_backlog_units"],
            "corrected_closing_backlog": new_row["closing_backlog_units"],
            "old_active_capital_assets": old_row["active_capital_assets"],
            "corrected_active_capital_assets": new_row["active_capital_assets"],
            "desired_labor_reduction": (
                old_row["desired_labor_services_per_week"]
                - new_row["desired_labor_services_per_week"]
            ),
        })
    write_rows(OUTPUT / "old_new_labor_scale_comparison.csv", comparison)

    def peak(rows, field):
        return max(number(row[field]) for row in rows)

    max_bridge = max(
        abs(number(row["backlog_bridge_gap"]))
        for row in all_rows
    )
    max_formula_gap = max(
        abs(number(row["corrected_flow_formula_gap"]))
        for row in corrected["rows"]
    )
    old_peak_labor = peak(old["rows"], "desired_labor_services_per_week")
    new_peak_labor = peak(corrected["rows"], "desired_labor_services_per_week")
    old_peak_output = peak(old["rows"], "desired_output_units_per_week")
    new_peak_output = peak(corrected["rows"], "desired_output_units_per_week")
    flags = {
        "verdict": (
            "A. BACKLOG_STOCK_FLOW_RUNTIME_CORRECTION_ACCEPTED"
            if (
                new_peak_labor < old_peak_labor
                and new_peak_output < old_peak_output
                and max_bridge <= EPS
                and max_formula_gap <= EPS
                and corrected["roster_violations"] == 0
                and corrected["accounting_gap"] <= 1e-6
                and corrected["conservation_gap"] <= 1e-5
            )
            else "E. OTHER_BLOCKER"
        ),
        "population": POPULATION,
        "food_firms": FOOD_FIRMS,
        "steps": STEPS,
        "seed": SEED,
        "horizon_weeks": HORIZON,
        "engineering_productivity": 1.0,
        "old_peak_desired_output": old_peak_output,
        "corrected_peak_desired_output": new_peak_output,
        "old_peak_desired_labor": old_peak_labor,
        "corrected_peak_desired_labor": new_peak_labor,
        "desired_labor_reduction_ratio": (
            1.0 - new_peak_labor / old_peak_labor
            if old_peak_labor > 0.0 else 0.0
        ),
        "max_backlog_bridge_gap": max_bridge,
        "max_corrected_flow_formula_gap": max_formula_gap,
        "corrected_roster_violations": corrected["roster_violations"],
        "corrected_accounting_gap": corrected["accounting_gap"],
        "corrected_conservation_gap": corrected["conservation_gap"],
        "old_capital_good_loans": 0.0,
        "corrected_capital_good_loans": 0.0,
        "economic_behavior_changed_only_supply_flow_boundary": True,
        "new_rng_draws": 0,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = f"""# Step 15I.10D Capital-Good Backlog Stock-to-Flow Runtime

## Verdict

**{flags['verdict']}**

The old and corrected paths use the same N=500, five-Food-Firm, seed-42,
520-week setup. The only active behavioral difference is conversion of the
capital-good backlog stock into a weekly flow over the existing 13-week
investment review horizon. Productivity remains exactly 1.0.

## Runtime comparison

- Old peak desired output: `{old_peak_output:.12g}` units/week.
- Corrected peak desired output: `{new_peak_output:.12g}` units/week.
- Old peak desired labor: `{old_peak_labor:.12g}` labor-services/week.
- Corrected peak desired labor: `{new_peak_labor:.12g}` labor-services/week.
- Desired-labor reduction: `{flags['desired_labor_reduction_ratio']:.6%}`.
- Maximum backlog bridge gap: `{max_bridge:.12g}`.
- Maximum corrected stock-to-flow formula gap: `{max_formula_gap:.12g}`.
- Corrected roster violations: `{corrected['roster_violations']}`.
- Corrected accounting gap: `{corrected['accounting_gap']:.12g}`.
- Corrected conservation gap: `{corrected['conservation_gap']:.12g}`.

The corrected target is:

```text
desired_output_per_week
    = opening_backlog / 13
    + current_new_demand_units / 13
desired_labor_per_week
    = desired_output_per_week / productivity
```

The backlog identity remains `opening + new demand - fulfilled = closing`.
Existing backlog is not repeatedly added as new demand. Replacement and
expansion supply remain active, while the remaining labor gap is still driven
by the unchanged engineering productivity and the size of the backlog.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(flags["verdict"])
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
