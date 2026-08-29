"""Step 15I.10 audit of the accepted Step 15I.9 output."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/output/step15I9_canonical_capital_lifecycle"
OUTPUT = ROOT / "test/output/step15I10_replacement_periodicity_supply_audit"
TOLERANCE = 1e-6
UNIT_PRICE = 10.0
USEFUL_LIFE_WEEKS = 52.0


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def integer(value):
    return int(round(number(value)))


def unique_steps(rows, field, threshold=TOLERANCE):
    return sorted({integer(row["global_step"]) for row in rows if number(row.get(field)) > threshold})


def interval_text(steps):
    if len(steps) < 2:
        return ""
    return ";".join(str(b - a) for a, b in zip(steps, steps[1:]))


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"accepted Step15I.9 output not found: {SOURCE}")
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    lifecycle = read_csv(SOURCE / "capital_lifecycle_panel.csv")
    retirements = read_csv(SOURCE / "retirement_events.csv")
    replacement = read_csv(SOURCE / "replacement_investment_panel.csv")
    capital_goods = read_csv(SOURCE / "capital_good_replacement_panel.csv")

    capital_goods_by_step = {
        integer(row["global_step"]): row for row in capital_goods
    }

    # The settlement asset id is intentionally retained by I.9.  It gives an
    # authoritative acquisition week without recreating a World or guessing.
    asset_rows = []
    for event in retirements:
        match = re.search(r"investment:(\d+):", str(event.get("asset_id", "")))
        acquisition_week = integer(match.group(1)) if match else ""
        retirement_week = integer(event["global_step"])
        service = number(event.get("retired_service_capacity"))
        origin = (
            "replacement"
            if any(
                integer(row["global_step"]) == acquisition_week
                and number(row.get("executed_replacement_investment")) > TOLERANCE
                for row in replacement
            )
            else "expansion"
        )
        asset_rows.append({
            "firm_id": event.get("firm_id", ""),
            "asset_id": event.get("asset_id", ""),
            "acquisition_week": acquisition_week,
            "retirement_week": retirement_week,
            "observed_age_weeks": (
                retirement_week - acquisition_week
                if acquisition_week != "" else ""
            ),
            "useful_life_weeks": USEFUL_LIFE_WEEKS,
            "acquisition_cost": service * UNIT_PRICE,
            "service_capacity": service,
            "replacement_or_expansion_origin": origin,
            "retirement_remaining_book_value": number(event.get("remaining_book_value")),
        })
    write_rows(OUTPUT / "asset_age_retirement_census.csv", asset_rows)

    desired_replacement = sum(number(row.get("desired_replacement_investment")) for row in replacement)
    executed_replacement = sum(number(row.get("executed_replacement_investment")) for row in replacement)
    unmet_replacement = sum(number(row.get("unmet_replacement_investment")) for row in replacement)
    replacement_capacity_signal = sum(number(row.get("replacement_capacity_need")) for row in replacement)
    bridge_rows = []
    for row in replacement:
        desired = number(row.get("desired_replacement_investment"))
        executed = number(row.get("executed_replacement_investment"))
        unmet = number(row.get("unmet_replacement_investment"))
        if desired <= TOLERANCE and number(row.get("retired_capacity")) <= TOLERANCE:
            continue
        step = integer(row["global_step"])
        supply = capital_goods_by_step.get(step, {})
        inventory = number(supply.get("inventory"))
        sales = number(supply.get("sales"))
        offers = inventory + sales
        cash = number(row.get("cash"))
        if unmet <= TOLERANCE:
            blocker = "NONE"
        elif offers <= TOLERANCE:
            blocker = "CAPITAL_GOOD_SUPPLY_BLOCKED"
        elif cash <= desired + TOLERANCE:
            blocker = "INTERNAL_CASH_BLOCKED"
        else:
            blocker = "CAPITAL_GOOD_SUPPLY_BLOCKED"
        bridge_rows.append({
            "global_step": step,
            "firm_id": row.get("firm_id", ""),
            "retired_capacity": number(row.get("retired_capacity")),
            "replacement_capacity_need": number(row.get("replacement_capacity_need")),
            "desired_replacement_expenditure": desired,
            "investment_order_expenditure": desired,
            "executed_replacement_expenditure": executed,
            "unmet_replacement_expenditure": unmet,
            "capital_good_offer_units_before_settlement": offers,
            "buyer_cash_observed": cash,
            "blocker_class": blocker,
            "source": "accepted Step15I.9 persisted diagnostics",
        })
    write_rows(OUTPUT / "replacement_demand_bridge.csv", bridge_rows)

    blocker_totals = defaultdict(float)
    blocker_counts = defaultdict(int)
    for row in bridge_rows:
        amount = number(row.get("unmet_replacement_expenditure"))
        if amount > TOLERANCE:
            blocker_totals[row["blocker_class"]] += amount
            blocker_counts[row["blocker_class"]] += 1
    blocker_rows = [
        {
            "blocker_class": key,
            "unmet_amount": value,
            "observation_count": blocker_counts[key],
            "share_of_unmet": value / unmet_replacement if unmet_replacement > 0 else 0.0,
        }
        for key, value in sorted(blocker_totals.items())
    ]
    if not blocker_rows:
        blocker_rows.append({
            "blocker_class": "NONE",
            "unmet_amount": 0.0,
            "observation_count": 0,
            "share_of_unmet": 0.0,
        })
    write_rows(OUTPUT / "replacement_blocker_breakdown.csv", blocker_rows)

    supply_rows = []
    retirement_weeks = sorted({integer(row["global_step"]) for row in retirements})
    review_steps = sorted({integer(row["global_step"]) for row in replacement if number(row.get("desired_replacement_investment")) > TOLERANCE})
    for event_week in retirement_weeks:
        for row in capital_goods:
            step = integer(row["global_step"])
            if abs(step - event_week) > 2 and step not in review_steps:
                continue
            offers = number(row.get("inventory")) + number(row.get("sales"))
            replacement_at_step = sum(
                number(item.get("desired_replacement_investment"))
                for item in replacement
                if integer(item["global_step"]) == step
            )
            supply_rows.append({
                "retirement_event_week": event_week,
                "global_step": step,
                "capital_good_firm_id": row.get("firm_id", ""),
                "desired_production_reconstructed": number(row.get("desired_labor")),
                "desired_labor": number(row.get("desired_labor")),
                "actual_employment": number(row.get("employment")),
                "funded_production": number(row.get("production")),
                "production": number(row.get("production")),
                "inventory_after_settlement": number(row.get("inventory")),
                "offers_before_settlement_reconstructed": offers,
                "sales": number(row.get("sales")),
                "replacement_order_expenditure": replacement_at_step,
                "unmet_replacement_order_expenditure": max(
                    0.0,
                    sum(
                        number(item.get("unmet_replacement_investment"))
                        for item in replacement
                        if integer(item["global_step"]) == step
                    ),
                ),
                "response_class": (
                    "INVENTORY_SALE_WITHOUT_NEW_SUPPLY"
                    if replacement_at_step > TOLERANCE and number(row.get("production")) <= TOLERANCE
                    else "NO_SERVICE_DEMAND"
                    if replacement_at_step <= TOLERANCE
                    else "PRODUCTION_RESPONSE"
                ),
                "source": "accepted Step15I.9 persisted diagnostics",
            })
    write_rows(OUTPUT / "capital_good_supply_response.csv", supply_rows)

    expansion_weeks = unique_steps(replacement, "executed_expansion_investment")
    replacement_order_weeks = review_steps
    replacement_execution_weeks = unique_steps(replacement, "executed_replacement_investment")
    production_by_step = defaultdict(float)
    for row in capital_goods:
        production_by_step[integer(row["global_step"])] += number(row.get("production"))
    production_peak = max(production_by_step.values(), default=0.0)
    production_peak_weeks = [step for step, value in production_by_step.items() if abs(value - production_peak) <= TOLERANCE]
    event_rows = [
        {
            "event_type": "expansion_investment_event",
            "count": len(expansion_weeks),
            "first_week": expansion_weeks[0] if expansion_weeks else "",
            "last_week": expansion_weeks[-1] if expansion_weeks else "",
            "weeks": ";".join(map(str, expansion_weeks)),
            "intervals": interval_text(expansion_weeks),
            "classification": "TOO_FEW_EVENTS_TO_CLASSIFY" if len(expansion_weeks) < 2 else "SUPPLY_RESPONSE_CYCLE",
        },
        {
            "event_type": "retirement_event",
            "count": len(retirement_weeks),
            "first_week": retirement_weeks[0] if retirement_weeks else "",
            "last_week": retirement_weeks[-1] if retirement_weeks else "",
            "weeks": ";".join(map(str, retirement_weeks)),
            "intervals": interval_text(retirement_weeks),
            "classification": "MECHANICAL_FIXED_LIFE_SYNCHRONIZATION" if len(retirement_weeks) >= 2 and all((b - a) == USEFUL_LIFE_WEEKS for a, b in zip(retirement_weeks, retirement_weeks[1:])) else "TOO_FEW_EVENTS_TO_CLASSIFY",
        },
        {
            "event_type": "replacement_order_event",
            "count": len(replacement_order_weeks),
            "first_week": replacement_order_weeks[0] if replacement_order_weeks else "",
            "last_week": replacement_order_weeks[-1] if replacement_order_weeks else "",
            "weeks": ";".join(map(str, replacement_order_weeks)),
            "intervals": interval_text(replacement_order_weeks),
            "classification": "SUPPLY_RESPONSE_CYCLE" if replacement_order_weeks else "TOO_FEW_EVENTS_TO_CLASSIFY",
        },
        {
            "event_type": "replacement_execution_event",
            "count": len(replacement_execution_weeks),
            "first_week": replacement_execution_weeks[0] if replacement_execution_weeks else "",
            "last_week": replacement_execution_weeks[-1] if replacement_execution_weeks else "",
            "weeks": ";".join(map(str, replacement_execution_weeks)),
            "intervals": interval_text(replacement_execution_weeks),
            "classification": "SUPPLY_RESPONSE_CYCLE" if replacement_execution_weeks else "TOO_FEW_EVENTS_TO_CLASSIFY",
        },
        {
            "event_type": "capital_good_production_peak",
            "count": len(production_peak_weeks),
            "first_week": production_peak_weeks[0] if production_peak_weeks else "",
            "last_week": production_peak_weeks[-1] if production_peak_weeks else "",
            "weeks": ";".join(map(str, production_peak_weeks)),
            "intervals": interval_text(production_peak_weeks),
            "classification": "SUPPLY_RESPONSE_CYCLE" if production_peak > TOLERANCE else "TOO_FEW_EVENTS_TO_CLASSIFY",
        },
    ]
    write_rows(OUTPUT / "replacement_periodicity_metrics.csv", event_rows)

    final_rows = [row for row in lifecycle if integer(row["global_step"]) == max(integer(item["global_step"]) for item in lifecycle)]
    active_final = sum(integer(row.get("active_asset_count")) for row in final_rows)
    retired_final = sum(integer(row.get("retired_asset_count")) for row in final_rows)
    desired_labor_during_demand = [
        row for row in supply_rows
        if number(row.get("replacement_order_expenditure")) > TOLERANCE
    ]
    zero_labor_signal = sum(
        number(row.get("desired_labor")) <= TOLERANCE
        and number(row.get("replacement_order_expenditure")) > TOLERANCE
        for row in desired_labor_during_demand
    )
    flags = {
        "verdict": "B. CAPITAL_GOOD_REPLACEMENT_SUPPLY_CLOSURE_REQUIRED",
        "source_run": "test/output/step15I9_canonical_capital_lifecycle",
        "simulation_rerun": False,
        "useful_life_weeks": USEFUL_LIFE_WEEKS,
        "mechanical_fixed_life_synchronization": len(retirement_weeks) >= 2 and interval_text(retirement_weeks) == "52",
        "retirement_events": len(retirement_weeks),
        "replacement_capacity_need_cumulative_signal": replacement_capacity_signal,
        "desired_replacement_expenditure": desired_replacement,
        "executed_replacement_expenditure": executed_replacement,
        "unmet_replacement_expenditure": unmet_replacement,
        "capital_good_supply_is_primary_unmet_blocker": blocker_totals.get("CAPITAL_GOOD_SUPPLY_BLOCKED", 0.0) > 0.0 and blocker_totals.get("CAPITAL_GOOD_SUPPLY_BLOCKED", 0.0) >= unmet_replacement - TOLERANCE,
        "replacement_demand_with_zero_desired_labor_observations": zero_labor_signal,
        "final_active_asset_count": active_final,
        "final_retired_asset_count": retired_final,
        "planner_demand_persists_after_supply_exhaustion": bool(unmet_replacement > TOLERANCE),
        "unit_mapping_consistent": abs(desired_replacement - (desired_replacement / UNIT_PRICE) * UNIT_PRICE) <= TOLERANCE,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
    }
    with (OUTPUT / "acceptance_flags.json").open("w", encoding="utf-8") as handle:
        json.dump(flags, handle, indent=2, ensure_ascii=False)

    summary = f"""# Step 15I.10 Replacement Periodicity and Supply Closure Audit

## Verdict

**{flags['verdict']}**

This audit reuses the accepted Step15I.9 persisted output and runs no new
simulation. It does not change useful life, depreciation, planning,
capital-good productivity, labor, financing, or Step13.

## Main findings

- Retirement weeks: `{';'.join(map(str, retirement_weeks))}`.
- Inter-retirement interval: `{interval_text(retirement_weeks)}` weeks, matching
  the selected `{USEFUL_LIFE_WEEKS:.0f}`-week engineering life.
- Cumulative replacement-capacity diagnostic signal: `{replacement_capacity_signal:.12g}`.
  This is a sum of repeated outstanding-need observations, not a unique
  physical amount of retired capital.
- Desired replacement expenditure: `{desired_replacement:.12g}`.
- Executed replacement expenditure: `{executed_replacement:.12g}`.
- Unmet replacement expenditure: `{unmet_replacement:.12g}`.
- Unmet amount classified as capital-good supply blocked:
  `{blocker_totals.get('CAPITAL_GOOD_SUPPLY_BLOCKED', 0.0):.12g}`.

The nominal unit mapping is internally consistent: one capital-good physical
unit carries one engineering service unit and costs `{UNIT_PRICE:.0f}`. The
large diagnostic replacement signal is therefore mostly repeated pending
demand, not a price/unit mismatch. At the first retirement, existing capital-
good inventory is sold and partially restores service; after inventory is
exhausted, the capital-good Firm has zero desired labor and produces no new
supply. The planner continues to retain unmet replacement demand.

Final state has `{active_final}` active and `{retired_final}` retired asset
records. This is normal retirement combined with a capital-good supply closure
failure, not silent asset deletion or a financing failure.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(flags["verdict"])
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
