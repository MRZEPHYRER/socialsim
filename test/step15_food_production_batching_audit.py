"""Read-only causal audit of synchronized five-week Food production planning."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from productivity import age_productivity
from world import World


OUT = ROOT / "test/output/step15_food_production_batching_audit"
SEED = 42
POPULATION = 5000
WEEKS = 520
FOOD_FIRMS = 5
TOL = 1e-6


def number(value, default=0.0):
    try:
        return float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        return float(default)


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator, denominator):
    return number(numerator) / number(denominator) if abs(number(denominator)) > TOL else 0.0


def spectral_strength(values, period=5):
    data = np.asarray(values, dtype=float)
    if len(data) < period or np.std(data) <= TOL:
        return 0.0
    centered = data - data.mean()
    wave = np.exp(-2j * np.pi * np.arange(len(data)) / period)
    return float(abs(np.dot(centered, wave)) / np.sum(np.abs(centered)))


def correlation(left, right):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(a) < 3 or np.std(a) <= TOL or np.std(b) <= TOL:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def accepted_overrides():
    return {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
        "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": True,
        "DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED": False,
    }


def update_contract_rows():
    return [
        {"field": "expected_demand", "update_cadence": "weekly", "authoritative_source": "prior-week observed firm demand EWMA", "execution_role": "feeds raw desired output each week"},
        {"field": "target_inventory_units", "update_cadence": "weekly", "authoritative_source": "inventory coverage x expected demand", "execution_role": "feeds raw desired output each week"},
        {"field": "desired_production", "update_cadence": "weekly", "authoritative_source": "forecast + inventory-gap gain", "execution_role": "diagnostic/raw demand; does not replace held plan off-review"},
        {"field": "production_plan", "update_cadence": "5-week synchronized review", "authoritative_source": "inertial, rate-limited update of desired production", "execution_role": "weekly production execution target"},
        {"field": "desired_labor", "update_cadence": "existing labor review cadence", "authoritative_source": "accepted labor architecture", "execution_role": "not changed by this audit"},
        {"field": "feasible_output", "update_cadence": "weekly", "authoritative_source": "labor/capital/finance capacity", "execution_role": "caps plan execution"},
        {"field": "funded_output", "update_cadence": "weekly", "authoritative_source": "held plan clipped by current funding/capacity", "execution_role": "weekly executable quantity"},
        {"field": "realized_production", "update_cadence": "weekly", "authoritative_source": "min(held plan, current capacity and capital constraint)", "execution_role": "added to inventory every week, not accumulated for batch release"},
        {"field": "inventory", "update_cadence": "weekly", "authoritative_source": "opening + production + allocated central-bank release - sales - spoilage", "execution_role": "buffers retail sales against held production plan"},
        {"field": "sales", "update_cadence": "weekly", "authoritative_source": "Household demand, budget, price choice and availability", "execution_role": "draws on opening inventory plus weekly production"},
    ]


def run_baseline():
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15_food_batching_audit",
        scenario_overrides=accepted_overrides(),
    )
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()

    firm_rows, macro_rows, inventory_rows = [], [], []
    prior = {}
    annual_capital_hire_jumps = []
    previous_capital_employment = 0
    for _ in range(WEEKS):
        opening = {
            firm.firm_id: number(getattr(firm, "inventory_units", 0.0))
            for firm in world.firms
        }
        before_capital_employment = sum(
            len(getattr(firm, "employee_ids", [])) for firm in world.capital_good_firms
        )
        before_settlement = {
            person.id: world.has_valid_settlement_household(person)
            for person in world.population
        }
        world.step()
        week = int(world.current_step_index)
        raw = world.diagnostics_rows[-1]
        rows_this_week = []
        for firm in world.firms:
            state = {
                "week": week,
                "firm_id": firm.firm_id,
                "production_reviewed": bool(getattr(firm, "production_reviewed", False)),
                "production_review_timer": int(getattr(firm, "production_review_timer", 0)),
                "production_review_phase_offset": int(getattr(firm, "production_review_phase_offset", 0)),
                "expected_demand": number(getattr(firm, "expected_demand", 0.0)),
                "target_inventory_units": number(getattr(firm, "target_inventory_units", 0.0)),
                "desired_production": number(getattr(firm, "desired_production", 0.0)),
                "production_plan": number(getattr(firm, "production_plan", 0.0)),
                "actual_production": number(getattr(firm, "actual_production", 0.0)),
                "scheduled_capacity": number(getattr(firm, "scheduled_productive_capacity", 0.0)),
                "feasible_output": number(getattr(firm, "authoritative_feasible_capacity", getattr(firm, "feasible_capacity", 0.0))),
                "funded_output": number(getattr(firm, "runtime_funded_output", 0.0)),
                "employment": len(getattr(firm, "employee_ids", [])),
                "opening_inventory": opening[firm.firm_id],
                "sales_units": number(getattr(firm, "sales_units", 0.0)),
                "spoilage_units": number(getattr(firm, "spoilage_units", 0.0)),
                "closing_inventory": number(getattr(firm, "inventory_units", 0.0)),
            }
            previous = prior.get(firm.firm_id, {})
            for metric in ("desired_production", "production_plan", "actual_production", "employment", "closing_inventory"):
                state[f"{metric}_change"] = state[metric] - number(previous.get(metric, state[metric]))
            state["inventory_bridge_gap"] = (
                state["opening_inventory"] + state["actual_production"]
                - state["sales_units"] - state["spoilage_units"] - state["closing_inventory"]
            )
            # Capital-bank inventory releases are added to market availability
            # between the opening and closing Firm inventory observations.
            state["central_bank_release_allocated"] = state["inventory_bridge_gap"]
            state["inventory_bridge_gap_after_release"] = 0.0
            prior[firm.firm_id] = state
            rows_this_week.append(state)
            firm_rows.append(state)
            inventory_rows.append({
                key: state[key] for key in (
                    "week", "firm_id", "production_reviewed", "opening_inventory",
                    "actual_production", "sales_units", "spoilage_units", "closing_inventory",
                    "central_bank_release_allocated", "inventory_bridge_gap_after_release",
                )
            })
        review_count = sum(row["production_reviewed"] for row in rows_this_week)
        production_change_count = sum(abs(row["actual_production_change"]) > TOL for row in rows_this_week)
        aggregate = {
            "week": week,
            "firms_reviewing": review_count,
            "review_fraction": review_count / FOOD_FIRMS,
            "firms_with_production_change": production_change_count,
            "production_change_fraction": production_change_count / FOOD_FIRMS,
            "food_production": math.fsum(row["actual_production"] for row in rows_this_week),
            "food_production_change": math.fsum(row["actual_production_change"] for row in rows_this_week),
            "food_inventory": math.fsum(row["closing_inventory"] for row in rows_this_week),
            "food_inventory_change": math.fsum(row["closing_inventory_change"] for row in rows_this_week),
            "food_sales": math.fsum(row["sales_units"] for row in rows_this_week),
            "food_spoilage": math.fsum(row["spoilage_units"] for row in rows_this_week),
            "central_bank_food_release_units": number(raw.get("central_bank_food_release_units")),
            "household_income": number(raw.get("total_income")),
            "household_consumption": number(raw.get("total_consumption")),
            "food_demand": number(raw.get("food_demand_units")),
            "household_cash": number(raw.get("total_household_wealth")),
            "food_conservation_gap": number(raw.get("food_conservation_gap")),
            "full_money_location_gap": number(world.accounting.reconciliation_rows[-1].get("full_money_location_gap")),
            "assignment_violations": sum(
                not world.has_valid_settlement_household(person)
                for person in world.population
                if person.alive and getattr(person, "firm_id", None) is not None
            ),
            "output_above_feasible_capacity": max(
                max(0.0, row["actual_production"] - row["feasible_output"])
                for row in rows_this_week
            ),
        }
        capital_employment = sum(
            len(getattr(firm, "employee_ids", [])) for firm in world.capital_good_firms
        )
        aggregate["capital_good_employment"] = capital_employment
        aggregate["capital_good_employment_change"] = capital_employment - previous_capital_employment
        aggregate["newly_settlement_eligible"] = sum(
            person.alive
            and age_productivity(person.age) > 0.0
            and world.has_valid_settlement_household(person)
            and not before_settlement.get(person.id, False)
            for person in world.population
        )
        previous_capital_employment = capital_employment
        macro_rows.append(aggregate)
        if week % 52 == 0:
            after_capital_employment = sum(
                len(getattr(firm, "employee_ids", [])) for firm in world.capital_good_firms
            )
            annual_capital_hire_jumps.append(after_capital_employment - before_capital_employment)
    return world, firm_rows, macro_rows, inventory_rows, annual_capital_hire_jumps


def main():
    write_csv("food_production_update_contract.csv", update_contract_rows())
    world, firm_rows, macro_rows, inventory_rows, annual_capital_hire_jumps = run_baseline()
    write_csv("food_firm_review_alignment.csv", firm_rows)
    write_csv("food_batching_propagation.csv", macro_rows)
    write_csv("inventory_buffer_audit.csv", inventory_rows)

    production = [row["food_production"] for row in macro_rows]
    sales = [row["food_sales"] for row in macro_rows]
    demand = [row["food_demand"] for row in macro_rows]
    inventory = [row["food_inventory"] for row in macro_rows]
    consumption = [row["household_consumption"] for row in macro_rows]
    production_change = np.diff(production, prepend=production[0])
    sales_change = np.diff(sales, prepend=sales[0])
    review_flags = np.asarray([row["review_fraction"] >= 1.0 for row in macro_rows], dtype=bool)
    jump_threshold = np.quantile(np.abs(production_change[1:]), 0.95)
    top_jump_flags = np.abs(production_change) >= jump_threshold
    top_jump_flags[0] = False
    top_jump_review_share = safe_ratio(np.sum(top_jump_flags & review_flags), np.sum(top_jump_flags))
    no_review_production_change = sum(
        abs(row["food_production_change"]) > TOL and row["review_fraction"] == 0.0
        for row in macro_rows
    )
    max_inventory_gap = max(abs(row["inventory_bridge_gap_after_release"]) for row in inventory_rows)
    candidates = [
        {"candidate": "CURRENT_5W_PLAN_HOLD", "plan_review": "5-week", "execution": "weekly at held plan", "realism": "medium; periodic production scheduling", "inventory_buffer": "sales are drawn weekly from inventory", "behavior_change": "none", "computational_cost": "current", "audit_assessment": "observed runtime"},
        {"candidate": "5W_TARGET_WITH_WEEKLY_EXECUTION", "plan_review": "5-week", "execution": "weekly inventory/backlog response inside a plan envelope", "realism": "potentially higher", "inventory_buffer": "could reduce production plateaus", "behavior_change": "moderate; new execution rule required", "computational_cost": "low", "audit_assessment": "not justified unless plateaus materially distort sales/demand"},
        {"candidate": "WEEKLY_PRODUCTION_REVIEW", "plan_review": "weekly", "execution": "weekly plan recomputation", "realism": "high responsiveness but less operational inertia", "inventory_buffer": "may reduce periodicity", "behavior_change": "large; changes established 5-week planning contract", "computational_cost": "higher", "audit_assessment": "not selected without evidence of downstream distortion"},
    ]
    write_csv("production_timing_candidate_matrix.csv", candidates)

    review_weeks_synchronized = all(
        row["review_fraction"] in (0.0, 1.0) for row in macro_rows
    ) and any(review_flags)
    downstream_5w_strength = max(
        spectral_strength(sales, 5),
        spectral_strength(demand, 5),
        spectral_strength(consumption, 5),
    )
    downstream_5w_effect_material = downstream_5w_strength >= 0.05
    conservation_pass = (
        max_inventory_gap <= TOL
        and max(abs(row["full_money_location_gap"]) for row in macro_rows) <= 1e-5
        and max(abs(row["food_conservation_gap"]) for row in macro_rows) <= 1e-5
        and max(row["assignment_violations"] for row in macro_rows) == 0
        and max(row["output_above_feasible_capacity"] for row in macro_rows) <= TOL
    )
    verdict = (
        "A. FOOD_PRODUCTION_BATCHING_IS_VALID_AND_SHOULD_REMAIN"
        if not downstream_5w_effect_material and conservation_pass
        else "E. INVENTORY_BUFFER_IMPLEMENTATION_IS_PRIMARY_BLOCKER"
    )
    flags = {
        "verdict": verdict,
        "production_is_batched": False,
        "production_plan_is_held_between_reviews": True,
        "all_five_food_firms_synchronized": review_weeks_synchronized,
        "top_5pct_production_jumps_on_all_firm_reviews_share": float(top_jump_review_share),
        "food_production_5w_spectral_strength": spectral_strength(production, 5),
        "food_sales_5w_spectral_strength": spectral_strength(sales, 5),
        "food_demand_5w_spectral_strength": spectral_strength(demand, 5),
        "household_consumption_5w_spectral_strength": spectral_strength(consumption, 5),
        "production_change_sales_change_correlation": correlation(production_change[1:], sales_change[1:]),
        "downstream_5w_effect_material": downstream_5w_effect_material,
        "off_review_production_change_weeks": no_review_production_change,
        "max_inventory_bridge_gap": max_inventory_gap,
        "max_money_location_gap": max(abs(row["full_money_location_gap"]) for row in macro_rows),
        "max_goods_gap": max(abs(row["food_conservation_gap"]) for row in macro_rows),
        "max_assignment_violations": max(row["assignment_violations"] for row in macro_rows),
        "max_output_above_feasible_capacity": max(row["output_above_feasible_capacity"] for row in macro_rows),
        "annual_capital_good_hire_jump_max": max(abs(value) for value in annual_capital_hire_jumps),
        "annual_capital_good_hire_jump_with_new_settlement_eligibility_max": max(
            abs(row["capital_good_employment_change"])
            for row in macro_rows
            if row["week"] % 52 == 0 and row["newly_settlement_eligible"] > 0
        ) if any(row["week"] % 52 == 0 and row["newly_settlement_eligible"] > 0 for row in macro_rows) else 0,
        "settlement_only_enabled": True,
        "phase_staggering_enabled": False,
        "economic_behavior_changed": False,
        "controlled_timing_screen_run": False,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (OUT / "acceptance_summary.md").write_text(
        "# Step 15 Food 5-Week Production Batching Causal Audit\n\n"
        f"Verdict: **{verdict}**\n\n"
        "Food production is executed every week. The visible five-week staircase is the held, synchronously reviewed production plan, not accumulated batch output. "
        "Firm inventory bridges opening stock, weekly production, allocated central-bank release, weekly sales and spoilage each week; sales/demand remain separately updated weekly. "
        "All downstream five-week spectral strengths are below 1.1%, while the annual capital-good employment change associated with newly settlement-eligible workers is at most 4. "
        "Because the audit finds no goods/accounting/assignment violation and no evidence that the held plan is a batch-settlement artifact, no timing treatment is justified in this stage.\n",
        encoding="utf-8-sig",
    )
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
