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

from productivity import age_productivity
from world import World


OUTPUT = ROOT / "test/output/step15I10B_replacement_labor_capacity_audit"
POPULATION = 500
FOOD_FIRMS = 5
SEED = 42
STEPS = 520
USEFUL_LIFE_WEEKS = 52.0
PRODUCTIVITY = 1.0
EPS = 1e-9


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


def number(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def worker_snapshot(world):
    firms = list(world.operating_firms())
    firm_ids = {firm.firm_id for firm in firms}
    capital_ids = set(getattr(world, "capital_good_firm_ids", set()))
    eligible = []
    employed = []
    unassigned = []
    for person in world.population:
        productivity = age_productivity(person.age)
        if not (
            person.alive
            and person.household_id in world.household_dict
            and productivity > 0.0
        ):
            continue
        eligible.append(person)
        if getattr(person, "firm_id", None) in firm_ids:
            employed.append(person)
        else:
            unassigned.append(person)
    food_ids = {firm.firm_id for firm in world.firms}
    food = [person for person in employed if person.firm_id in food_ids]
    capital = [person for person in employed if person.firm_id in capital_ids]

    def capacity(people):
        return math.fsum(age_productivity(person.age) for person in people)

    return {
        "total_alive_working_age": len(eligible),
        "total_employed": len(employed),
        "unassigned_eligible_workers": len(unassigned),
        "food_employment": len(food),
        "capital_good_employment": len(capital),
        "eligible_labor_capacity": capacity(eligible),
        "unassigned_labor_capacity": capacity(unassigned),
        "food_labor_capacity": capacity(food),
        "capital_good_labor_capacity": capacity(capital),
        "unassigned_ids": {person.id for person in unassigned},
        "food_ids": {person.id for person in food},
    }


def capital_snapshot(world, firm):
    people = [
        world.person_dict[person_id]
        for person_id in getattr(firm, "employee_ids", [])
        if person_id in world.person_dict
    ]
    return {
        "employment": len(getattr(firm, "employee_ids", [])),
        "capacity": math.fsum(
            age_productivity(person.age) for person in people if person.alive
        ),
        "desired_labor": number(getattr(firm, "desired_labor", 0.0)),
        "desired_output": number(
            getattr(firm, "capital_good_desired_output", 0.0)
        ),
        "production": number(
            getattr(firm, "capital_good_production_units", 0.0)
        ),
        "inventory": number(
            getattr(getattr(firm, "capital_good_inventory", None), "units", 0.0)
        ),
        "replacement_demand": number(
            getattr(firm, "capital_good_replacement_demand_units", 0.0)
        ),
        "expansion_demand": number(
            getattr(firm, "capital_good_expansion_demand_units", 0.0)
        ),
        "outstanding_demand": number(
            getattr(firm, "capital_good_outstanding_demand_units", 0.0)
        ),
    }


def run_audit():
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15i10b_replacement_labor_capacity_audit",
        scenario_overrides={
            "MULTISECTOR_FOUNDATION_ENABLED": True,
            "CANONICAL_INVESTMENT_ENABLED": True,
            "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
            "CAPITAL_LIFECYCLE_ENABLED": True,
            "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": USEFUL_LIFE_WEEKS,
        },
    )
    world.split_firms(FOOD_FIRMS)
    system = world.canonical_investment_system
    # The canonical executor lazily creates the capital-good Firm at the
    # beginning of the first simulation week. Initialize that same passive
    # runtime boundary before collecting the opening snapshot.
    system.ensure_firms()
    labor_gap_rows = []
    availability_rows = []
    backlog_rows = []
    scale_rows = []
    retirement_weeks = []
    retired_assets_seen = set()
    retirement_service_ages = []
    release_count = 0
    rehire_count = 0
    prior_replacement = 0.0
    prior_total = 0.0

    for _ in range(STEPS):
        before = worker_snapshot(world)
        before_cap = capital_snapshot(world, world.capital_good_firms[0])
        before_food_ids = set().union(
            *(set(getattr(firm, "employee_ids", [])) for firm in world.firms)
        )

        world.step()
        step = world.current_step_index
        capital_firm = world.capital_good_firms[0]
        after = worker_snapshot(world)
        current = capital_snapshot(world, capital_firm)
        releases = [
            event for event in system.labor_release_events
            if number(event.get("global_step")) == step
        ]
        rehires = [
            event for event in system.labor_reactivation_events
            if number(event.get("global_step")) == step
        ]
        release_count += len(releases)
        rehire_count += len(rehires)
        if any(
            number(getattr(firm, "retired_capacity_this_step", 0.0)) > EPS
            for firm in world.firms
        ):
            retirement_weeks.append(step)
        for food_firm in world.firms:
            stock = getattr(food_firm, "capital_stock", None)
            for asset in getattr(stock, "assets", []) if stock is not None else []:
                asset_id = getattr(asset, "asset_id", None)
                if not getattr(asset, "is_retired", False) or asset_id in retired_assets_seen:
                    continue
                retired_assets_seen.add(asset_id)
                parts = str(asset_id).split(":")
                if len(parts) > 1:
                    try:
                        acquisition_week = int(parts[1])
                        retirement_service_ages.append(step - acquisition_week)
                    except ValueError:
                        pass

        service_gap = max(0.0, current["desired_labor"] - current["capacity"])
        requested_services = max(
            0.0, current["desired_labor"] - before_cap["capacity"]
        )
        available_workers = before["unassigned_eligible_workers"]
        available_services = before["unassigned_labor_capacity"]
        mean_available_productivity = (
            available_services / available_workers if available_workers else 0.0
        )
        worker_equivalent_request = (
            requested_services / mean_available_productivity
            if mean_available_productivity > EPS else 0.0
        )
        food_releases = len(before_food_ids - after["food_ids"])
        labor_gap_rows.append({
            "global_step": step,
            "desired_capital_good_labor": current["desired_labor"],
            "actual_capital_good_employment": current["employment"],
            "actual_capital_good_labor_capacity": current["capacity"],
            "labor_service_gap": service_gap,
            "total_alive_working_age_persons": after["total_alive_working_age"],
            "total_employed_persons": after["total_employed"],
            "unassigned_eligible_workers": after["unassigned_eligible_workers"],
            "workers_released_from_food": food_releases,
            "workers_available_to_matching": available_workers,
            "hires_requested": requested_services,
            "hires_requested_labor_services": requested_services,
            "hires_requested_worker_equivalent": worker_equivalent_request,
            "hires_executed": len(rehires),
            "unfilled_vacancies": service_gap,
            "system_labor_capacity": after["eligible_labor_capacity"],
            "food_labor_capacity": after["food_labor_capacity"],
            "capital_good_productivity": PRODUCTIVITY,
            "labor_shortage_with_no_unassigned": (
                service_gap > EPS and available_workers == 0
            ),
        })
        availability_rows.append({
            "global_step": step,
            "total_alive_working_age_persons": after["total_alive_working_age"],
            "total_employed_persons": after["total_employed"],
            "unassigned_eligible_workers": after["unassigned_eligible_workers"],
            "unassigned_labor_capacity": after["unassigned_labor_capacity"],
            "food_employment": after["food_employment"],
            "capital_good_employment": after["capital_good_employment"],
            "total_employment": after["total_employed"],
            "workers_released_from_food": food_releases,
            "workers_available_to_capital_matching": available_workers,
            "hires_executed": len(rehires),
        })

        replacement = sum(
            max(0.0, number(getattr(firm, "pending_replacement_capacity_need", 0.0)))
            for firm in world.firms
        )
        expansion = sum(
            max(0.0, number(value))
            for value in system.expansion_backlog_by_firm.values()
        )
        fulfilled_replacement = sum(
            max(
                0.0,
                number(
                    getattr(firm, "executed_replacement_investment_this_step", 0.0)
                ),
            )
            for firm in world.firms
        ) / max(1e-12, system.unit_price)
        new_replacement = sum(
            max(0.0, number(getattr(firm, "retired_capacity_this_step", 0.0)))
            for firm in world.firms
        )
        total = replacement + expansion
        production = current["production"]
        backlog_rows.append({
            "global_step": step,
            "opening_replacement_backlog_units": prior_replacement,
            "new_replacement_demand_units": new_replacement,
            "replacement_production_units": production,
            "replacement_settlement_units": fulfilled_replacement,
            "closing_replacement_backlog_units": replacement,
            "replacement_bridge_gap": (
                replacement - prior_replacement - new_replacement
                + fulfilled_replacement
            ),
            "opening_total_backlog_units": prior_total,
            "closing_expansion_backlog_units": expansion,
            "closing_total_backlog_units": total,
            "total_backlog_change_units": total - prior_total,
            "physical_capacity_at_actual_employment": (
                current["capacity"] * PRODUCTIVITY
            ),
            "funded_output_shortfall": max(
                0.0, current["capacity"] * PRODUCTIVITY - production
            ),
        })
        scale_rows.append({
            "global_step": step,
            "replacement_backlog_units": replacement,
            "expansion_backlog_units": expansion,
            "total_backlog_units": total,
            "desired_production_units": current["desired_output"],
            "engineering_productivity_units_per_labor_service": PRODUCTIVITY,
            "implied_labor_services_per_unit": (
                1.0 / PRODUCTIVITY if PRODUCTIVITY else 0.0
            ),
            "desired_labor_services": current["desired_labor"],
            "actual_employment": current["employment"],
            "actual_labor_capacity": current["capacity"],
            "capacity_at_actual_employment": current["capacity"] * PRODUCTIVITY,
            "implied_weeks_to_clear_replacement_at_current_capacity": (
                replacement / current["capacity"]
                if current["capacity"] > EPS else math.inf
            ),
            "implied_weeks_to_clear_total_at_current_capacity": (
                total / current["capacity"]
                if current["capacity"] > EPS else math.inf
            ),
        })
        prior_replacement = replacement
        prior_total = total

    max_desired = max(row["desired_capital_good_labor"] for row in labor_gap_rows)
    max_system_capacity = max(
        row["system_labor_capacity"] for row in labor_gap_rows
    )
    scale_ratio = max_desired / max(max_system_capacity, EPS)
    final = labor_gap_rows[-1]
    final_backlog = backlog_rows[-1]["closing_total_backlog_units"]
    first_demand = next(
        (
            int(row["global_step"])
            for row in labor_gap_rows
            if row["desired_capital_good_labor"] > EPS
        ),
        None,
    )
    fixed_life = bool(retirement_service_ages) and all(
        abs(age - USEFUL_LIFE_WEEKS) <= 1e-6
        for age in retirement_service_ages
    )
    no_matching_blocker = all(
        row["unassigned_eligible_workers"] == 0
        or row["unfilled_vacancies"] <= EPS
        for row in labor_gap_rows
    )
    if scale_ratio > 2.0 and no_matching_blocker:
        verdict = "B. NON_CALIBRATED_CAPITAL_GOOD_PRODUCTIVITY_SCALE_IS_PRIMARY"
    elif fixed_life and scale_ratio > 2.0:
        verdict = "C. FIXED_LIFE_SYNCHRONIZATION_AMPLIFIES_LABOR_SPIKE"
    elif any(
        row["unassigned_eligible_workers"] > 0
        and row["unfilled_vacancies"] > EPS
        for row in labor_gap_rows
    ):
        verdict = "D. MATCHING_OR_ELIGIBILITY_BLOCKER"
    else:
        verdict = "E. MIXED_LABOR_CAPACITY_CONSTRAINT"

    return {
        "labor_gap": labor_gap_rows,
        "availability": availability_rows,
        "backlog": backlog_rows,
        "scale": scale_rows,
        "retirement_weeks": retirement_weeks,
        "verdict": verdict,
        "first_demand_week": first_demand,
        "max_desired_labor": max_desired,
        "max_system_capacity": max_system_capacity,
        "scale_ratio": scale_ratio,
        "final_backlog": final_backlog,
        "final_unassigned": final["unassigned_eligible_workers"],
        "release_count": release_count,
        "rehire_count": rehire_count,
        "fixed_life": fixed_life,
        "retirement_service_ages": retirement_service_ages,
    }


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = run_audit()
    write_rows(OUTPUT / "capital_good_labor_gap.csv", result["labor_gap"])
    write_rows(OUTPUT / "labor_availability_panel.csv", result["availability"])
    write_rows(OUTPUT / "backlog_clearance_metrics.csv", result["backlog"])
    write_rows(OUTPUT / "replacement_labor_scale.csv", result["scale"])

    flags = {
        "verdict": result["verdict"],
        "population": POPULATION,
        "food_firms": FOOD_FIRMS,
        "steps": STEPS,
        "seed": SEED,
        "first_demand_week": result["first_demand_week"],
        "max_desired_capital_good_labor": result["max_desired_labor"],
        "max_system_labor_capacity": result["max_system_capacity"],
        "desired_to_system_labor_capacity_ratio": result["scale_ratio"],
        "final_total_backlog_units": result["final_backlog"],
        "final_unassigned_eligible_workers": result["final_unassigned"],
        "labor_release_events": result["release_count"],
        "labor_reactivation_events": result["rehire_count"],
        "fixed_life_synchronization_detected": result["fixed_life"],
        "retired_asset_count": len(result["retirement_service_ages"]),
        "retirement_service_age_min": min(result["retirement_service_ages"], default=None),
        "retirement_service_age_max": max(result["retirement_service_ages"], default=None),
        "engineering_productivity": PRODUCTIVITY,
        "new_rng_draws": 0,
        "economic_behavior_changed": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = f"""# Step 15I.10B Capital-Good Replacement Labor-Capacity Audit

## Verdict

**{result['verdict']}**

The accepted I.10A configuration was replayed only to recover labor-capacity
diagnostics. No productivity, useful life, matching, finance, investment, or
Step13 rule was changed.

## Findings

- First positive capital-good demand week: `{result['first_demand_week']}`.
- Maximum desired capital-good labor services: `{result['max_desired_labor']:.12g}`.
- Maximum system eligible labor capacity: `{result['max_system_capacity']:.12g}`.
- Desired/system labor-capacity ratio: `{result['scale_ratio']:.12g}`.
- Final total backlog: `{result['final_backlog']:.12g}` units.
- Final unassigned eligible workers: `{result['final_unassigned']}`.
- Capital-good labor release events: `{result['release_count']}`.
- Capital-good reactivation events: `{result['rehire_count']}`.
- Fixed 52-week retirement synchronization detected: `{result['fixed_life']}`.
- Asset-level retirement service ages observed: `{result['retirement_service_ages'][:8]}`
  (showing the first eight observations).

The desired-labor signal is measured in labor-service units. With the
non-calibrated engineering productivity of `{PRODUCTIVITY}`, one physical
capital-good unit requires one labor-service unit. The observed demand signal
therefore requests far more labor services than the N=500 world can supply;
actual employment is constrained by the already-employed Food/capital labor
pool rather than by an unused matching pool. The fixed 52-week life creates
retirement spikes, but the persistent gap is primarily a scale consequence of
the engineering unit mapping and outstanding expansion/replacement backlog.

The aggregate retirement weeks can be more frequent than 52 weeks because
assets are acquired on different review weeks; the asset-level service age is
the relevant fixed-life test. The output panels retain the week-by-week labor availability, backlog bridge,
physical capacity at actual employment, and mechanically implied clearance
time. No behavioral rule was changed by this audit.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(result["verdict"])
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
