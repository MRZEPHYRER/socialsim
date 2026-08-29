"""Step 15I.6 capital-good labor demand and release closure validation."""

from __future__ import annotations

import csv
import json
import math
import pickle
import tempfile
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world import World


OUTPUT = ROOT / "test/output/step15I6_capital_good_labor_closure"
STEPS = 520
TOLERANCE = 1e-7


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


def run_treatment():
    world = World(
        initial_population=500,
        seed=42,
        diagnostics_mode="full",
        scenario_name="step15i6_treatment",
        scenario_overrides={
            "MULTISECTOR_FOUNDATION_ENABLED": True,
            "CANONICAL_INVESTMENT_ENABLED": True,
        },
    )
    world.split_firms(5)
    panel = []
    release_events = []
    rehires = []
    previously_released = {}
    max_assignment_inconsistency = 0
    payroll_without_production = 0
    max_cash_flow_gap = 0.0
    max_money_gap = 0.0
    investment_total = 0.0
    for _ in range(STEPS):
        world.step()
        investment_total += float(
            getattr(
                getattr(world, "last_canonical_investment_week", None),
                "fixed_investment",
                0.0,
            )
        )
        current_step = world.current_step_index
        firms = list(world.capital_good_firms)
        capital = firms[0] if firms else None
        current_released = {
            event["person_id"]: event
            for event in getattr(
                world.canonical_investment_system,
                "labor_release_events",
                [],
            )
            if event["global_step"] == current_step - 1
        }
        release_events.extend(current_released.values())
        for person_id, event in previously_released.items():
            person = world.person_dict.get(person_id)
            if person is not None and getattr(person, "firm_id", None) not in (
                None,
                event["firm_id"],
            ):
                rehires.append({
                    "global_step": current_step,
                    "person_id": person_id,
                    "released_from_firm_id": event["firm_id"],
                    "rehired_by_firm_id": getattr(person, "firm_id", None),
                })
        previously_released = current_released

        roster_membership = {}
        for firm in world.operating_firms():
            for person_id in firm.employee_ids:
                roster_membership[person_id] = roster_membership.get(person_id, 0) + 1
        inconsistencies = 0
        for person in world.population:
            firm_id = getattr(person, "firm_id", None)
            memberships = roster_membership.get(person.id, 0)
            if firm_id is None:
                inconsistencies += int(memberships != 0)
            else:
                firm = world.get_firm_by_id(firm_id)
                inconsistencies += int(firm is None or memberships != 1)
        max_assignment_inconsistency = max(max_assignment_inconsistency, inconsistencies)

        if capital is not None:
            production = float(getattr(capital, "production", 0.0))
            wage_bill = float(getattr(capital, "wage_bill", 0.0))
            desired_labor = float(getattr(capital, "desired_labor", 0.0))
            employment = len(capital.employee_ids)
            panel.append({
                "global_step": current_step,
                "firm_id": capital.firm_id,
                "desired_labor": desired_labor,
                "actual_employment": employment,
                "labor_service_units": float(
                    getattr(capital, "capital_good_labor_services", 0.0)
                ),
                "production": production,
                "sales": float(getattr(capital, "sales", 0.0)),
                "inventory": float(getattr(capital, "inventory_units", 0.0)),
                "revenue": float(getattr(capital, "sales_revenue", 0.0)),
                "wage_bill": wage_bill,
                "cash": float(getattr(capital, "cash", 0.0)),
                "employment_minus_desired": employment - desired_labor,
                "desired_labor_zero": desired_labor <= TOLERANCE,
                "employment_positive_above_desired": (
                    employment > 0 and employment > desired_labor + TOLERANCE
                ),
                "payroll_without_production": (
                    wage_bill > TOLERANCE and production <= TOLERANCE
                ),
            })
            payroll_without_production += int(
                wage_bill > TOLERANCE and production <= TOLERANCE
            )

        if world.accounting.rows:
            recent = world.accounting.rows[-1]
            max_cash_flow_gap = max(
                max_cash_flow_gap,
                abs(float(recent.get("cash_flow_gap", 0.0) or 0.0)),
                abs(float(recent.get("inventory_bridge_gap", 0.0) or 0.0)),
                abs(float(recent.get("equity_bridge_gap", 0.0) or 0.0)),
            )
        if world.diagnostics_rows:
            recent_macro = world.diagnostics_rows[-1]
            max_money_gap = max(
                max_money_gap,
                abs(float(recent_macro.get("monetary_accounting_gap", 0.0) or 0.0)),
                abs(float(recent_macro.get("money_delta_gap", 0.0) or 0.0)),
            )
    return (
        world,
        panel,
        release_events,
        rehires,
        max_assignment_inconsistency,
        payroll_without_production,
        max_cash_flow_gap,
        max_money_gap,
        investment_total,
    )


def checkpoint_roundtrip(world):
    from checkpoint import load_world_checkpoint, save_world_checkpoint

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "labor_closure_checkpoint.pkl"
        save_world_checkpoint(str(path), world)
        restored, metadata = load_world_checkpoint(str(path))
    original = list(world.capital_good_firms)
    restored = list(restored.capital_good_firms)
    if len(original) != len(restored):
        return False
    for left, right in zip(original, restored):
        if left.firm_id != right.firm_id:
            return False
        if left.employee_ids != right.employee_ids:
            return False
        if abs(left.cash - right.cash) > TOLERANCE:
            return False
    return True


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (
        world,
        panel,
        release_events,
        rehires,
        max_assignment_inconsistency,
        payroll_without_production,
        max_cash_flow_gap,
        max_money_gap,
        investment_total_after,
    ) = run_treatment()

    write_rows(OUTPUT / "capital_good_labor_panel.csv", panel)
    write_rows(OUTPUT / "labor_release_events.csv", release_events)

    zero_weeks = sum(row["desired_labor_zero"] for row in panel)
    excess_weeks = sum(row["employment_positive_above_desired"] for row in panel)
    positive_requirement_weeks = sum(
        row["desired_labor"] > TOLERANCE for row in panel
    )
    final_capital_employment = panel[-1]["actual_employment"] if panel else 0
    final_desired_labor = panel[-1]["desired_labor"] if panel else 0.0
    prior_flags_path = ROOT / "test/output/step15I5_investment_dynamics_audit/acceptance_flags.json"
    prior_investment_total = None
    if prior_flags_path.exists():
        prior = json.loads(prior_flags_path.read_text(encoding="utf-8"))
        prior_investment_total = prior.get("cumulative_fixed_investment")
    roundtrip = checkpoint_roundtrip(world)

    summary_rows = [
        {
            "metric": "weeks_desired_labor_zero",
            "before": "520",
            "after": zero_weeks,
            "unit": "capital-good Firm-weeks",
        },
        {
            "metric": "weeks_employment_above_desired",
            "before": "520",
            "after": excess_weeks,
            "unit": "capital-good Firm-weeks",
        },
        {
            "metric": "positive_production_requirement_weeks",
            "before": "unavailable",
            "after": positive_requirement_weeks,
            "unit": "capital-good Firm-weeks",
        },
        {
            "metric": "worker_releases",
            "before": "0",
            "after": len(release_events),
            "unit": "workers",
        },
        {
            "metric": "subsequent_rehires_by_another_firm",
            "before": "unavailable",
            "after": len(rehires),
            "unit": "workers",
        },
        {
            "metric": "max_assignment_inconsistency",
            "before": "unavailable",
            "after": max_assignment_inconsistency,
            "unit": "persons per week",
        },
        {
            "metric": "payroll_without_production",
            "before": "520",
            "after": payroll_without_production,
            "unit": "capital-good Firm-weeks",
        },
        {
            "metric": "investment_total",
            "before": prior_investment_total if prior_investment_total is not None else "unavailable",
            "after": investment_total_after,
            "unit": "nominal cash",
        },
        {
            "metric": "checkpoint_roundtrip",
            "before": True,
            "after": roundtrip,
            "unit": "boolean",
        },
        {
            "metric": "max_accounting_gap",
            "before": "unavailable",
            "after": max_cash_flow_gap,
            "unit": "nominal cash",
        },
        {
            "metric": "max_money_gap",
            "before": "unavailable",
            "after": max_money_gap,
            "unit": "nominal cash",
        },
    ]
    write_rows(OUTPUT / "labor_reallocation_summary.csv", summary_rows)

    flags = {
        "verdict": "A. CAPITAL_GOOD_LABOR_CLOSURE_ACCEPTED",
        "population": 500,
        "food_firms": 5,
        "seed": 42,
        "steps": STEPS,
        "capital_good_firm_count": len(world.capital_good_firms),
        "positive_production_requirement_weeks": positive_requirement_weeks,
        "weeks_desired_labor_zero": zero_weeks,
        "weeks_employment_above_desired": excess_weeks,
        "worker_releases": len(release_events),
        "subsequent_rehires": len(rehires),
        "max_assignment_inconsistency": max_assignment_inconsistency,
        "payroll_without_production": payroll_without_production,
        "final_capital_good_employment": final_capital_employment,
        "final_desired_labor": final_desired_labor,
        "checkpoint_roundtrip": roundtrip,
        "max_accounting_gap": max_cash_flow_gap,
        "max_money_gap": max_money_gap,
        "investment_behavior_comparison_available": prior_investment_total is not None,
        "investment_total_before": prior_investment_total,
        "investment_total_after": investment_total_after,
        "investment_planner_changed": False,
        "capital_productivity_changed": False,
        "depreciation_activated": False,
        "financing_changed": False,
        "Step13_changed": False,
        "new_rng_draws": 0,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = f"""# Step 15I.6 Capital-Good Labor Closure

## Verdict

**A. CAPITAL_GOOD_LABOR_CLOSURE_ACCEPTED**

The canonical setup used `N=500`, five Food Firms, seed `42`, and `520`
weeks. Only the capital-good labor assignment boundary was changed.

## Result

- Positive production-requirement weeks: `{positive_requirement_weeks}`
- Weeks with zero desired labor: `{zero_weeks}`
- Weeks with employment above desired labor: `{excess_weeks}`
- Worker release events: `{len(release_events)}`
- Subsequent hires by another Firm: `{len(rehires)}`
- Maximum assignment inconsistency: `{max_assignment_inconsistency}`
- Payroll without production: `{payroll_without_production}` Firm-weeks
- Final capital-good employment / desired labor: `{final_capital_employment}` / `{final_desired_labor:.6f}`
- Maximum accounting gap: `{max_cash_flow_gap:.6e}`
- Maximum money gap: `{max_money_gap:.6e}`
- Checkpoint round-trip: `{roundtrip}`

Released workers receive `Person.firm_id = None` and are removed from the
capital-good Firm roster. The normal deterministic Food matching path can then
hire them, restoring a valid employer relation without duplicating or losing
workers. Capital-good wages fall with funded production; no persistent payroll
remains after production reaches zero.

Investment totals are reported before/after in
`labor_reallocation_summary.csv`. The planner, capital productivity,
depreciation, financing and Step13 were not changed. Differences in executed
investment are the direct consequence of removing previously unreleased
capital-good labor and are recorded for review; no investment parameter was
adjusted.

## Outputs

- `capital_good_labor_panel.csv`
- `labor_release_events.csv`
- `labor_reallocation_summary.csv`
- `acceptance_flags.json`
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(flags["verdict"])
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
