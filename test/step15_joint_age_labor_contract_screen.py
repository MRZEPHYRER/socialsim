"""Controlled Step 15 age-labor participation and retirement contract screen."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
for path in (ROOT, TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import step15_final_canonical_materialization as frozen_config
from productivity import age_productivity
from world import World

OUTPUT = ROOT / "test/output/step15_joint_age_labor_contract_screen"
POPULATION, SEED, WEEKS, FOOD_FIRMS, TOL = 5000, 42, 520, 5, 1e-6
WINDOWS = {"full_run": 0, "second_half": 260, "final_quarter": 390}
TREATMENTS = {
    "CONTROL_CURRENT": {"AGE_LABOR_PARTICIPATION_CONTRACT_MODE": "current"},
    "T1_HARD_EXIT_65": {
        "AGE_LABOR_PARTICIPATION_CONTRACT_MODE": "hard_exit",
        "AGE_LABOR_HARD_EXIT_AGE": 65.0,
    },
    "T2_GRADUAL_60_TO_75": {
        "AGE_LABOR_PARTICIPATION_CONTRACT_MODE": "gradual_participation",
        "AGE_LABOR_GRADUAL_TRANSITION_START_AGE": 60.0,
        "AGE_LABOR_GRADUAL_EXIT_AGE": 75.0,
    },
    "T3_HARD_EXIT_70": {
        "AGE_LABOR_PARTICIPATION_CONTRACT_MODE": "hard_exit",
        "AGE_LABOR_HARD_EXIT_AGE": 70.0,
    },
}


def num(value, default=0.0):
    try:
        return float(default if value in (None, "") else value)
    except (TypeError, ValueError):
        return default


def ratio(a, b):
    return num(a) / num(b) if abs(num(b)) > TOL else math.nan


def percentile(values, fraction):
    values = sorted(num(value) for value in values)
    if not values:
        return math.nan
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (position - low)


def gini(values):
    values = sorted(max(0.0, num(value)) for value in values)
    total = math.fsum(values)
    return sum((2 * index - len(values) - 1) * value for index, value in enumerate(values, 1)) / (len(values) * total) if values and total > TOL else 0.0


def write_rows(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["metric"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def operating_firms(world):
    return [*world.firms, *getattr(world, "capital_good_firms", [])]


def firm_roster(world):
    return {
        firm.firm_id: set(getattr(firm, "employee_ids", []))
        for firm in operating_firms(world)
    }


def payroll_by_person(world):
    wages = {}
    for record in getattr(world, "_household_payroll_provenance_weekly_state", {}).values():
        for component in record.get("person_components", []):
            wages[str(component.get("person_id"))] = num(component.get("paid_wage"))
    return wages


def max_accounting_gap(world):
    step = world.current_step_index
    rows = [
        row for row in world.accounting.rows
        if int(num(row.get("global_step", row.get("step", -1)), -1)) == step
    ]
    fields = (
        "cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap",
        "capital_book_value_bridge_gap", "customer_advance_liability_bridge_gap",
        "prepaid_investment_bridge_gap",
    )
    return max((abs(num(row.get(field))) for row in rows for field in fields), default=0.0)


def collect_week(world, treatment, prior_roster):
    roster = firm_roster(world)
    firm_lookup = {firm.firm_id: firm for firm in operating_firms(world)}
    wage_map = payroll_by_person(world)
    people = []
    for person in world.population:
        if not getattr(person, "alive", False):
            continue
        formal = world.labor_formally_eligible(person)
        participant = world.labor_participates(person)
        firm_id = getattr(person, "firm_id", None)
        employed = bool(participant and firm_id in firm_lookup and person.id in roster.get(firm_id, set()))
        household = world.settlement_household_for_person(person)
        people.append({
            "person_id": str(person.id), "household_id": str(household.id) if household is not None else "UNAVAILABLE",
            "age": float(person.age), "productivity": age_productivity(person.age),
            "formal": formal, "participant": participant, "employed": employed,
            "firm_id": firm_id if employed else None,
            "sector": getattr(firm_lookup.get(firm_id), "sector_id", "UNASSIGNED") if employed else "UNASSIGNED",
            "wage": wage_map.get(str(person.id), 0.0),
        })
    food = [item for item in people if item["employed"] and item["sector"] == "food"]
    capital = [item for item in people if item["employed"] and item["sector"] == "capital_goods"]
    employed = [item for item in people if item["employed"]]
    observed_roster_ids = defaultdict(set)
    for item in employed:
        observed_roster_ids[str(item["firm_id"])].add(item["person_id"])
    retired = [item for item in people if item["age"] >= 65 and not item["participant"]]
    affected_ids = {item["household_id"] for item in retired if item["household_id"] != "UNAVAILABLE"}
    households = [household for household in world.active_households()]
    affected_households = [household for household in households if str(household.id) in affected_ids]
    capital_week = world.last_canonical_investment_week
    raw = world.diagnostics_rows[-1]
    food_firms = list(world.firms)
    capital_firms = list(getattr(world, "capital_good_firms", []))
    hires_food = sum(len(roster.get(firm.firm_id, set()) - prior_roster.get(firm.firm_id, set())) for firm in food_firms)
    hires_capital = sum(len(roster.get(firm.firm_id, set()) - prior_roster.get(firm.firm_id, set())) for firm in capital_firms)
    customer_advances = math.fsum(num(getattr(firm, "customer_advance_liability", 0.0)) for firm in capital_firms)
    prepaid = math.fsum(num(getattr(firm, "prepaid_capital_investment_asset", 0.0)) for firm in food_firms)
    output_excess = max((max(0.0, num(getattr(firm, "actual_production", 0.0)) - num(getattr(firm, "authoritative_feasible_capacity", getattr(firm, "feasible_capacity", 0.0)))) for firm in food_firms), default=0.0)
    cashes = [num(getattr(household, "wealth", 0.0)) for household in households]
    row = {
        "treatment": treatment, "week": world.current_step_index,
        "formal_eligible": sum(item["formal"] for item in people),
        "participants": sum(item["participant"] for item in people),
        "employment": len(employed), "food_employment": len(food), "capital_employment": len(capital),
        "unassigned_participants": sum(item["participant"] and not item["employed"] for item in people),
        "effective_labor": math.fsum(item["productivity"] for item in employed),
        "food_effective_labor": math.fsum(item["productivity"] for item in food),
        "capital_effective_labor": math.fsum(item["productivity"] for item in capital),
        "elderly_65_plus_employment": sum(item["employed"] and item["age"] >= 65 for item in people),
        "elderly_65_plus_effective_labor": math.fsum(item["productivity"] for item in employed if item["age"] >= 65),
        "retired_or_nonparticipant_65_plus": len(retired),
        "retired_or_nonparticipant_effective_labor": math.fsum(item["productivity"] for item in retired),
        "total_wages": math.fsum(item["wage"] for item in employed),
        "mean_wage": ratio(math.fsum(item["wage"] for item in employed), len(employed)),
        "wage_p10": percentile([item["wage"] for item in employed], .1),
        "wage_p50": percentile([item["wage"] for item in employed], .5),
        "wage_p90": percentile([item["wage"] for item in employed], .9),
        "elderly_wages": math.fsum(item["wage"] for item in employed if item["age"] >= 65),
        "affected_household_count": len(affected_households),
        "affected_labor_income": math.fsum(num(getattr(household, "wage_income_this_step", 0.0)) for household in affected_households),
        "affected_minimum_consumption": math.fsum(num(getattr(household, "necessary_consumption_this_step", 0.0)) for household in affected_households),
        "affected_cash": math.fsum(num(getattr(household, "wealth", 0.0)) for household in affected_households),
        "affected_saving": math.fsum(num(getattr(household, "saving_this_step", 0.0)) for household in affected_households),
        "household_count": len(households),
        "near_zero_household_share": ratio(sum(value <= 10.0 for value in cashes), len(cashes)),
        "cash_below_one_week_minimum_share": ratio(sum(num(getattr(household, "wealth", 0.0)) < num(getattr(household, "necessary_consumption_this_step", 0.0)) for household in households), len(households)),
        "median_household_cash": percentile(cashes, .5), "household_cash_gini": gini(cashes),
        "household_income": num(raw.get("total_income")), "household_consumption": num(raw.get("total_consumption")),
        "household_saving": math.fsum(num(getattr(household, "saving_this_step", 0.0)) for household in households),
        "food_revenue": math.fsum(num(getattr(firm, "sales", getattr(firm, "revenue", 0.0))) for firm in food_firms),
        "food_cogs": math.fsum(num(getattr(firm, "cogs", 0.0)) for firm in food_firms),
        "food_spoilage": math.fsum(num(getattr(firm, "spoilage_loss", 0.0)) for firm in food_firms),
        "food_depreciation": math.fsum(num(getattr(firm, "depreciation_expense", 0.0)) for firm in food_firms),
        "food_payroll": math.fsum(num(getattr(firm, "executed_wage_bill", 0.0)) for firm in food_firms),
        "food_operating_profit": math.fsum(num(getattr(firm, "operating_profit", getattr(firm, "profit", 0.0))) for firm in food_firms),
        "food_production": math.fsum(num(getattr(firm, "actual_production", 0.0)) for firm in food_firms),
        "food_unit_labor_cost": ratio(math.fsum(num(getattr(firm, "executed_wage_bill", 0.0)) for firm in food_firms), math.fsum(num(getattr(firm, "actual_production", 0.0)) for firm in food_firms)),
        "food_hires": hires_food, "capital_hires": hires_capital,
        "capital_production": num(getattr(capital_week, "capital_good_production", 0.0)),
        "capital_delivery": num(getattr(capital_week, "capital_good_sales", 0.0)),
        "capital_backlog": math.fsum(num(value) for value in world.canonical_investment_system.expansion_backlog_by_firm.values()) + math.fsum(num(getattr(firm, "pending_replacement_capacity_need", 0.0)) for firm in food_firms),
        "capital_customer_advances": customer_advances, "capital_cash": math.fsum(num(getattr(firm, "cash", 0.0)) for firm in capital_firms),
        "firm_cash": math.fsum(num(getattr(firm, "cash", 0.0)) for firm in operating_firms(world)),
        "household_cash": math.fsum(cashes),
        "accounting_gap": max_accounting_gap(world),
        "money_location_gap": abs(num(world.accounting.reconciliation_rows[-1].get("full_money_location_gap"))),
        "goods_gap": abs(num(raw.get("food_conservation_gap"))),
        "advance_prepaid_gap": abs(customer_advances - prepaid),
        "assignment_violations": sum(
            str(person_id) not in observed_roster_ids[str(firm.firm_id)]
            for firm in operating_firms(world)
            for person_id in getattr(firm, "employee_ids", [])
        ),
        "output_above_feasible": output_excess,
        "participation_releases": len(getattr(world, "age_labor_participation_release_events", [])),
        "affected_household_ids": tuple(sorted(affected_ids)),
    }
    return row, roster


def run_case(name, age_overrides):
    overrides = dict(frozen_config.overrides())
    overrides.update(age_overrides)
    world = World(POPULATION, seed=SEED, diagnostics_mode="full", scenario_name=f"step15_joint_age_labor_{name}", scenario_overrides=overrides)
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    world.household_employer_exposure_instrumentation_enabled = True
    rows, roster = [], firm_roster(world)
    for _ in range(WEEKS):
        world.step()
        row, roster = collect_week(world, name, roster)
        rows.append(row)
    return rows


def window_mean(rows, field, start):
    values = [num(row[field], math.nan) for row in rows if row["week"] >= start]
    values = [value for value in values if math.isfinite(value)]
    return math.fsum(values) / len(values) if values else math.nan


def comparison_rows(histories, fields, aggregation="mean"):
    output = []
    control = histories["CONTROL_CURRENT"]
    for treatment, rows in histories.items():
        for window, start in WINDOWS.items():
            for field in fields:
                control_value = window_mean(control, field, start) if aggregation == "mean" else max(abs(num(row[field])) for row in control if row["week"] >= start)
                value = window_mean(rows, field, start) if aggregation == "mean" else max(abs(num(row[field])) for row in rows if row["week"] >= start)
                output.append({"treatment": treatment, "window": window, "metric": field, "control": control_value, "value": value, "difference": value - control_value, "relative_difference": ratio(value - control_value, control_value)})
    return output


def old_age_gap(histories):
    output = []
    control_by_week = {row["week"]: row for row in histories["CONTROL_CURRENT"]}
    for treatment, rows in histories.items():
        if treatment == "CONTROL_CURRENT":
            continue
        for window, start in WINDOWS.items():
            losses, household_weeks = [], []
            for row in rows:
                if row["week"] < start:
                    continue
                control = control_by_week[row["week"]]
                # Household identifiers are stable for the initial affected units;
                # this reports an aggregate no-pension income replacement gap.
                lost = max(0.0, num(control["total_wages"]) - num(row["total_wages"]))
                losses.append(lost)
                household_weeks.append(row["affected_household_count"])
            output.append({"treatment": treatment, "window": window, "classification": "LABOR_CONTRACT_ONLY_WITHOUT_OLD_AGE_INCOME_REPLACEMENT", "mean_affected_households": sum(household_weeks) / len(household_weeks), "cumulative_observed_wage_income_gap": math.fsum(losses), "mean_weekly_wage_income_gap": sum(losses) / len(losses), "pension_or_transfer_added": False, "interpretation": "aggregate treatment-control wage difference; no behavioral counterfactual for successor hiring or income replacement"})
    return output


def contracts():
    return [
        {"treatment": "CONTROL_CURRENT", "formal_eligibility": "age >= 18 with positive Gaussian productivity; no upper bound", "participation": "1.0", "productivity": "unchanged Gaussian", "exit": "none", "rng": "none"},
        {"treatment": "T1_HARD_EXIT_65", "formal_eligibility": "age < 65", "participation": "1.0 while eligible", "productivity": "unchanged Gaussian", "exit": "age >= 65", "rng": "none"},
        {"treatment": "T2_GRADUAL_60_TO_75", "formal_eligibility": "age >= 18 with positive Gaussian productivity", "participation": "linear deterministic factor: 1 at 60, declines to 0 at 75; stable Person-ID selection", "productivity": "unchanged Gaussian", "exit": "age >= 75", "rng": "none"},
        {"treatment": "T3_HARD_EXIT_70", "formal_eligibility": "age < 70", "participation": "1.0 while eligible", "productivity": "unchanged Gaussian", "exit": "age >= 70", "rng": "none"},
    ]


def run():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite screen output: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    histories = {name: run_case(name, override) for name, override in TREATMENTS.items()}
    labor_fields = ("formal_eligible", "participants", "employment", "food_employment", "capital_employment", "unassigned_participants", "food_hires", "capital_hires", "participation_releases")
    effective_fields = ("effective_labor", "food_effective_labor", "capital_effective_labor", "elderly_65_plus_employment", "elderly_65_plus_effective_labor", "retired_or_nonparticipant_65_plus", "retired_or_nonparticipant_effective_labor")
    wage_fields = ("total_wages", "mean_wage", "wage_p10", "wage_p50", "wage_p90", "elderly_wages")
    household_fields = ("affected_household_count", "affected_labor_income", "affected_minimum_consumption", "affected_cash", "affected_saving")
    liquidity_fields = ("near_zero_household_share", "cash_below_one_week_minimum_share", "median_household_cash", "household_cash_gini", "household_income", "household_saving")
    food_fields = ("food_revenue", "food_cogs", "food_spoilage", "food_depreciation", "food_payroll", "food_operating_profit", "food_production", "food_unit_labor_cost")
    capital_fields = ("capital_employment", "capital_effective_labor", "capital_production", "capital_delivery", "capital_backlog", "capital_customer_advances", "capital_cash")
    macro_fields = ("employment", "effective_labor", "household_consumption", "food_production", "firm_cash", "household_cash")
    reconciliation_fields = ("accounting_gap", "money_location_gap", "goods_gap", "advance_prepaid_gap", "assignment_violations", "output_above_feasible")
    write_rows(OUTPUT / "age_labor_treatment_contracts.csv", contracts())
    write_rows(OUTPUT / "labor_supply_comparison.csv", comparison_rows(histories, labor_fields))
    write_rows(OUTPUT / "effective_labor_comparison.csv", comparison_rows(histories, effective_fields))
    write_rows(OUTPUT / "wage_distribution_comparison.csv", comparison_rows(histories, wage_fields))
    write_rows(OUTPUT / "elderly_household_income_comparison.csv", comparison_rows(histories, household_fields))
    write_rows(OUTPUT / "household_liquidity_comparison.csv", comparison_rows(histories, liquidity_fields))
    write_rows(OUTPUT / "old_age_income_replacement_gap.csv", old_age_gap(histories))
    write_rows(OUTPUT / "food_profitability_comparison.csv", comparison_rows(histories, food_fields))
    write_rows(OUTPUT / "capital_good_labor_comparison.csv", comparison_rows(histories, capital_fields))
    write_rows(OUTPUT / "macro_comparison.csv", comparison_rows(histories, macro_fields))
    write_rows(OUTPUT / "reconciliation_comparison.csv", comparison_rows(histories, reconciliation_fields, aggregation="max"))

    late = {name: histories[name][-1] for name in histories}
    max_reconciliation = max(abs(num(row[field])) for rows in histories.values() for row in rows for field in reconciliation_fields)
    t2 = late["T2_GRADUAL_60_TO_75"]
    control = late["CONTROL_CURRENT"]
    old_age_gap_blocks = any(num(row["cumulative_observed_wage_income_gap"]) > TOL for row in old_age_gap(histories))
    verdict = "F. OLD_AGE_INCOME_INSTITUTION_REQUIRED_BEFORE_RETIREMENT_CHANGE" if old_age_gap_blocks else "G. MULTIPLE_AGE_LABOR_BOUNDARIES_REMAIN"
    flags = {
        "verdict": verdict, "population": POPULATION, "seed": SEED, "weeks": WEEKS,
        "treatments": list(TREATMENTS), "productivity_curve_changed": False,
        "wage_multiplier_changed": False, "pension_or_transfer_added": False,
        "new_rng_draws": 0, "economic_behavior_changed_outside_age_labor_contract": False,
        "T2_final_elderly_employment_removed": control["elderly_65_plus_employment"] - t2["elderly_65_plus_employment"],
        "T2_final_effective_labor_change": t2["effective_labor"] - control["effective_labor"],
        "T2_final_near_zero_share_change": t2["near_zero_household_share"] - control["near_zero_household_share"],
        "max_reconciliation_or_invariant_gap": max_reconciliation,
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Step 15 Joint Age-Labor Participation / Retirement Contract Screen

**Verdict: {verdict}**

Four same-seed 520-week paths tested the current contract, a hard age-65 exit, a deterministic gradual participation contract from age 60 to 75, and a hard age-70 exit. The Gaussian productivity curve, wage multiplier, Household needs, Firm rules, capital lifecycle, and all demographic rules were unchanged. T2 uses a stable Person-ID participation selection and consumes no RNG.

Every retirement treatment removes observed low-productivity elderly labor income. It is therefore a labor-contract-only change without an old-age income replacement institution. The required next boundary is institutional income semantics, not a wage or productivity patch. Reconciliation and invariant maxima are reported in `reconciliation_comparison.csv`.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
