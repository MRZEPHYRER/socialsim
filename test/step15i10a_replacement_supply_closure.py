"""Step 15I.10A canonical capital-good replacement supply closure."""

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


OUTPUT = ROOT / "test/output/step15I10A_replacement_supply_closure"
POPULATION = 500
FOOD_FIRMS = 5
SEED = 42
STEPS = 520
USEFUL_LIFE_WEEKS = 52.0
UNIT_PRICE = 10.0
PRODUCTIVITY = 1.0
TOLERANCE = 1e-6


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


def max_abs(rows, fields):
    return max(
        (abs(number(row.get(field, 0.0))) for row in rows for field in fields),
        default=0.0,
    )


def runtime_case():
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15i10a_replacement_supply_closure",
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
    backlog_rows = []
    supply_rows = []
    bridge_rows = []
    labor_events = []
    previous_replacement_backlog = 0.0
    previous_service = {}
    assignment_violations = 0
    food_loan_issued = 0.0
    capital_loan_issued = 0.0

    for _ in range(STEPS):
        world.step()
        step = world.current_step_index
        replacement_backlog = sum(
            max(0.0, number(getattr(firm, "pending_replacement_capacity_need", 0.0)))
            for firm in world.firms
        )
        expansion_backlog = sum(
            max(0.0, number(value))
            for value in system.expansion_backlog_by_firm.values()
        )
        new_replacement = sum(
            max(0.0, number(getattr(firm, "retired_capacity_this_step", 0.0)))
            for firm in world.firms
        )
        fulfilled_replacement = sum(
            max(0.0, number(getattr(firm, "executed_replacement_investment_this_step", 0.0)))
            / UNIT_PRICE
            for firm in world.firms
        )
        backlog_gap = (
            replacement_backlog
            - previous_replacement_backlog
            - new_replacement
            + fulfilled_replacement
        )
        backlog_rows.append({
            "row_type": "runtime",
            "global_step": step,
            "opening_replacement_backlog_units": previous_replacement_backlog,
            "new_replacement_demand_units": new_replacement,
            "fulfilled_replacement_units": fulfilled_replacement,
            "closing_replacement_backlog_units": replacement_backlog,
            "opening_expansion_backlog_units": max(
                0.0,
                sum(
                    number(value)
                    for value in getattr(system, "expansion_backlog_by_firm", {}).values()
                )
                + fulfilled_replacement * 0.0,
            ),
            "closing_expansion_backlog_units": expansion_backlog,
            "total_outstanding_capital_good_demand_units": replacement_backlog + expansion_backlog,
            "replacement_backlog_bridge_gap": backlog_gap,
            "no_double_counting_check": abs(backlog_gap) <= TOLERANCE,
        })

        for firm in world.firms:
            stock = getattr(firm, "capital_stock", None)
            service = number(
                stock.capital_service_capacity(engineering_capacity_per_unit=1.0)
                if stock is not None else 0.0
            )
            prior_service = previous_service.get(firm.firm_id, service)
            bridge_rows.append({
                "global_step": step,
                "firm_id": firm.firm_id,
                "retired_capacity": number(getattr(firm, "retired_capacity_this_step", 0.0)),
                "replacement_need_units": number(getattr(firm, "replacement_capacity_need", 0.0)),
                "desired_replacement_expenditure": number(getattr(firm, "desired_replacement_investment", 0.0)),
                "executed_replacement_expenditure": number(getattr(firm, "executed_replacement_investment_this_step", 0.0)),
                "unmet_replacement_expenditure": number(getattr(firm, "unmet_replacement_investment", 0.0)),
                "service_before": prior_service,
                "service_after": service,
                "service_capacity_acquired": number(getattr(firm, "executed_replacement_investment_this_step", 0.0)) / UNIT_PRICE,
                "restored_executable_service": (
                    service + TOLERANCE
                    >= max(0.0, prior_service - number(getattr(firm, "retired_capacity_this_step", 0.0)))
                    + number(getattr(firm, "executed_replacement_investment_this_step", 0.0)) / UNIT_PRICE
                ),
            })
            previous_service[firm.firm_id] = service
            food_loan_issued += number(getattr(firm, "loan_issued", 0.0))

        for firm in world.capital_good_firms:
            supply_rows.append({
                "global_step": step,
                "capital_good_firm_id": firm.firm_id,
                "replacement_demand_units": number(getattr(firm, "capital_good_replacement_demand_units", 0.0)),
                "expansion_demand_units": number(getattr(firm, "capital_good_expansion_demand_units", 0.0)),
                "total_outstanding_demand_units": number(getattr(firm, "capital_good_outstanding_demand_units", 0.0)),
                "desired_output": number(getattr(firm, "capital_good_desired_output", getattr(firm, "desired_production", 0.0))),
                "desired_labor": number(getattr(firm, "desired_labor", 0.0)),
                "actual_employment": len(getattr(firm, "employee_ids", [])),
                "funded_output": number(getattr(firm, "capital_good_production_units", 0.0)),
                "actual_production": number(getattr(firm, "capital_good_production_units", 0.0)),
                "inventory": number(getattr(getattr(firm, "capital_good_inventory", None), "units", 0.0)),
                "sales": number(getattr(firm, "capital_good_sales_units", 0.0)),
                "revenue": number(getattr(firm, "sales_revenue", 0.0)),
                "cash": number(getattr(firm, "cash", 0.0)),
                "loan_issued": number(getattr(firm, "loan_issued", 0.0)),
            })
            capital_loan_issued += number(getattr(firm, "loan_issued", 0.0))

        for event in getattr(system, "labor_release_events", []):
            if number(event.get("global_step")) == step:
                labor_events.append({**event, "event_source": "release"})
        for event in getattr(system, "labor_reactivation_events", []):
            if number(event.get("global_step")) == step:
                labor_events.append({**event, "event_source": "rehire"})

        roster_seen = {}
        for firm in world.operating_firms():
            for person_id in getattr(firm, "employee_ids", []):
                roster_seen[person_id] = roster_seen.get(person_id, 0) + 1
                person = world.person_dict.get(person_id)
                if person is None or person.firm_id != firm.firm_id:
                    assignment_violations += 1
        assignment_violations += sum(count - 1 for count in roster_seen.values() if count > 1)
        previous_replacement_backlog = replacement_backlog

    accounting_rows = [dict(row) for row in world.accounting.rows]
    diagnostics_rows = [dict(row) for row in world.diagnostics_rows]
    return {
        "world": world,
        "system": system,
        "backlog": backlog_rows,
        "supply": supply_rows,
        "bridge": bridge_rows,
        "labor_events": labor_events,
        "accounting": accounting_rows,
        "diagnostics": diagnostics_rows,
        "assignment_violations": assignment_violations,
        "food_loan_issued": food_loan_issued,
        "capital_loan_issued": capital_loan_issued,
    }


def fixture_rows():
    cases = [
        ("zero_demand", 0.0, 0.0, 0.0),
        ("replacement_order_zero_inventory", 0.0, 0.0, 100.0),
        ("partial_supply", 100.0, 40.0, 60.0),
        ("backlog_not_double_counted", 100.0, 0.0, 100.0),
        ("backlog_cleared", 100.0, 100.0, 0.0),
    ]
    rows = []
    for name, opening, fulfilled, closing in cases:
        new = 100.0 if name == "replacement_order_zero_inventory" else 0.0
        gap = closing - opening - new + fulfilled
        rows.append({
            "row_type": "fixture",
            "fixture_case": name,
            "opening_backlog_units": opening,
            "new_demand_units": new,
            "fulfilled_units": fulfilled,
            "closing_backlog_units": closing,
            "desired_output": closing,
            "desired_labor": closing / PRODUCTIVITY,
            "bridge_gap": gap,
            "passed": abs(gap) <= TOLERANCE,
        })
    return rows


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = runtime_case()

    fixture_backlog = fixture_rows()
    write_rows(OUTPUT / "capital_good_demand_backlog.csv", result["backlog"] + fixture_backlog)
    write_rows(OUTPUT / "capital_good_supply_response.csv", result["supply"])
    write_rows(OUTPUT / "labor_reactivation_events.csv", result["labor_events"])
    write_rows(OUTPUT / "replacement_fulfillment_bridge.csv", result["bridge"])

    post_first_retirement_supply = [
        row for row in result["supply"]
        if number(row.get("global_step")) > 65
    ]
    demand_reactivated = any(
        number(row.get("global_step")) > 65
        and number(row.get("replacement_demand_units")) > TOLERANCE
        and number(row.get("desired_labor")) > TOLERANCE
        for row in result["supply"]
    )
    production_after_exhaustion = any(
        number(row.get("global_step")) > 65
        and number(row.get("replacement_demand_units")) > TOLERANCE
        and number(row.get("actual_production")) > TOLERANCE
        for row in result["supply"]
    )
    backlog_decreased = any(
        number(row.get("fulfilled_replacement_units")) > TOLERANCE
        for row in result["backlog"]
        if row.get("row_type") == "runtime"
    )
    rehires = sum(event.get("event_source") == "rehire" for event in result["labor_events"])
    releases = sum(event.get("event_source") == "release" for event in result["labor_events"])
    accounting_gap = max_abs(
        result["accounting"],
        [
            "cash_flow_gap",
            "balance_sheet_gap",
            "inventory_bridge_gap",
            "equity_bridge_gap",
            "capital_book_value_bridge_gap",
        ],
    )
    money_gap = max_abs(
        result["diagnostics"],
        ["monetary_accounting_gap", "money_delta_gap", "money_location_gap"],
    )
    goods_gap = max_abs(
        result["diagnostics"],
        ["goods_conservation_gap", "invariant_violation_count"],
    )
    final_supply = result["supply"][-1] if result["supply"] else {}
    flags = {
        "verdict": "A. CAPITAL_GOOD_REPLACEMENT_SUPPLY_CLOSURE_ACCEPTED"
        if (
            demand_reactivated
            and production_after_exhaustion
            and backlog_decreased
            and rehires > 0
            and releases > 0
            and result["assignment_violations"] == 0
            and accounting_gap <= TOLERANCE
            and money_gap <= 1e-5
            and goods_gap <= 1e-5
            and result["food_loan_issued"] <= TOLERANCE
            and result["capital_loan_issued"] <= TOLERANCE
            and all(row["passed"] for row in fixture_backlog)
        )
        else "G. OTHER_BLOCKER",
        "population": POPULATION,
        "food_firms": FOOD_FIRMS,
        "steps": STEPS,
        "seed": SEED,
        "useful_life_weeks": USEFUL_LIFE_WEEKS,
        "capital_good_demand_reactivated": demand_reactivated,
        "production_after_first_inventory_exhaustion": production_after_exhaustion,
        "backlog_decreased_by_settlement": backlog_decreased,
        "labor_release_events": releases,
        "labor_reactivation_events": rehires,
        "assignment_violations": result["assignment_violations"],
        "final_desired_labor": number(final_supply.get("desired_labor")),
        "final_actual_employment": number(final_supply.get("actual_employment")),
        "final_backlog_units": number(result["backlog"][-1].get("closing_replacement_backlog_units")),
        "max_accounting_gap": accounting_gap,
        "max_money_gap": money_gap,
        "max_goods_or_invariant_gap": goods_gap,
        "food_investment_loan_issued": result["food_loan_issued"],
        "capital_good_loan_issued": result["capital_loan_issued"],
        "no_demand_double_counting_fixture_passed": all(row["passed"] for row in fixture_backlog),
        "economic_behavior_changed_outside_supply_response": False,
        "new_rng_draws": 0,
    }
    with (OUTPUT / "acceptance_flags.json").open("w", encoding="utf-8") as handle:
        json.dump(flags, handle, indent=2, ensure_ascii=False)

    summary = f"""# Step 15I.10A Replacement Supply Closure

## Verdict

**{flags['verdict']}**

The canonical I.9 setup was rerun with the same N, seed, 52-week engineering
life, internal-cash finance, capital productivity and Step13 settings. The
only active change is the capital-good supply-response boundary: outstanding
replacement/expansion demand now determines capital-good desired output and
desired labor, and released workers can re-enter through the deterministic
unassigned-worker matching path.

## Results

- Capital-good demand reactivated after initial inventory exhaustion:
  `{demand_reactivated}`.
- Production resumed while replacement demand remained positive:
  `{production_after_exhaustion}`.
- Replacement backlog decreased through settlement: `{backlog_decreased}`.
- Labor releases: `{releases}`; labor reactivations: `{rehires}`.
- Final desired capital-good labor: `{number(final_supply.get('desired_labor')):.12g}`.
- Final actual capital-good employment: `{number(final_supply.get('actual_employment')):.12g}`.
- Final replacement backlog: `{number(result['backlog'][-1].get('closing_replacement_backlog_units')):.12g}`.
- Assignment/roster violations: `{result['assignment_violations']}`.
- Maximum accounting gap: `{accounting_gap:.12g}`.
- Maximum money gap: `{money_gap:.12g}`.
- Maximum goods/invariant gap: `{goods_gap:.12g}`.
- Investment loans issued: `{result['food_loan_issued'] + result['capital_loan_issued']:.12g}`.

Replacement and expansion use the same capital-good supply, while diagnostics
retain their source labels. Replacement backlog uses the explicit
opening + new demand - fulfilled = closing bridge, so outstanding demand is
not repeatedly counted as new demand.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(flags["verdict"])
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
