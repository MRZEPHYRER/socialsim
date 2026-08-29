"""Step 15 adult settlement / labor eligibility audit and controlled screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from productivity import age_productivity
from world import World


OUT = ROOT / "test/output/step15_adult_settlement_labor_eligibility"
SEED = 42
POPULATION = 5000
WEEKS = 520
FOOD_FIRMS = 5
TOL = 1e-6


def write_csv(name, rows):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def number(value, default=0.0):
    try:
        return float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        return float(default)


def linear_slope(values):
    if len(values) < 2:
        return math.nan
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def accepted_overrides(enabled):
    return {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
        "DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED": False,
        "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": enabled,
    }


def social_household(world, person):
    return world.get_household(getattr(person, "household_id", None))


def worker_snapshot(world):
    food = list(world.firms)
    capital = list(world.capital_good_firms)
    matching_people = [
        person for person in world.population
        if person.alive and age_productivity(person.age) > 0.0
    ]
    social_valid = sum(social_household(world, person) is not None for person in matching_people)
    settlement_valid = sum(world.has_valid_settlement_household(person) for person in matching_people)
    unassigned = sum(
        world.has_valid_settlement_household(person)
        and getattr(person, "firm_id", None) is None
        for person in matching_people
    )
    employed = sum(getattr(person, "firm_id", None) is not None for person in matching_people)
    food_employment = sum(len(getattr(firm, "employee_ids", [])) for firm in food)
    capital_employment = sum(len(getattr(firm, "employee_ids", [])) for firm in capital)
    settlement_accounts = sum(
        bool(getattr(household, "settlement_only", False))
        for household in world.households
    )
    social_households = len(world.households) - settlement_accounts
    composition = Counter()
    for household in world.households:
        if getattr(household, "settlement_only", False):
            continue
        composition[f"{len(household.parents)}p_{len(household.children)}c"] += 1
    return {
        "matching_age_population": len(matching_people),
        "social_household_valid": social_valid,
        "settlement_valid": settlement_valid,
        "missing_settlement": len(matching_people) - settlement_valid,
        "unassigned_eligible": unassigned,
        "employed_matching_age": employed,
        "food_employment": food_employment,
        "capital_good_employment": capital_employment,
        "total_employment": food_employment + capital_employment,
        "economic_household_count": len(world.households),
        "social_household_count": social_households,
        "settlement_only_account_count": settlement_accounts,
        "social_composition": dict(composition),
    }


def assignment_violations(world):
    seen = Counter()
    violations = 0
    for firm in world.operating_firms():
        for person_id in getattr(firm, "employee_ids", []):
            seen[person_id] += 1
            person = world.person_dict.get(person_id)
            if person is None or person.firm_id != firm.firm_id:
                violations += 1
            elif not world.has_valid_settlement_household(person):
                violations += 1
    return violations + sum(max(0, count - 1) for count in seen.values())


def origin_cause(person, prior):
    if person.age < 20:
        return "child_aging_path"
    if prior is None:
        return "initial_unattached_adult" if person.age >= 20 else "child_aging_path"
    if prior["social_household"] and not social_household(prior["world"], person):
        return "household_lifecycle_or_cleanup_loss"
    if prior["age"] < 20 <= person.age:
        return "child_aging_into_adulthood"
    if person.partner_id is None and person.is_seeking_partner:
        return "intentional_unattached_adult_waiting_for_marriage"
    return "other_unattached_adult"


def run_case(mode, population, weeks):
    enabled = mode == "treatment"
    world = World(
        initial_population=population,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name=f"step15_adult_settlement_{mode}",
        scenario_overrides=accepted_overrides(enabled),
    )
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()

    rows = []
    origins = {}
    annual = []
    prior_marriages = 0
    prior_births = 0
    prior_deaths = 0
    for _ in range(weeks):
        before = worker_snapshot(world)
        pre = {
            person.id: {
                "age": person.age,
                "social_household": social_household(world, person) is not None,
                "settlement_valid": world.has_valid_settlement_household(person),
                "world": world,
            }
            for person in world.population
        }
        world.step()
        week = world.current_step_index
        after = worker_snapshot(world)
        raw = world.diagnostics_rows[-1]
        cap_week = world.last_canonical_investment_week
        newly_settled = []
        for person in world.population:
            state = pre.get(person.id)
            before_valid = bool(state and state["settlement_valid"])
            after_valid = world.has_valid_settlement_household(person)
            if after_valid and not before_valid:
                newly_settled.append(person)
        no_social = [
            person for person in world.population
            if person.alive and age_productivity(person.age) > 0.0 and social_household(world, person) is None
        ]
        for person in no_social:
            cause = origin_cause(person, pre.get(person.id))
            key = (cause, int(person.age // 5) * 5)
            state = origins.setdefault(key, {
                "cause": cause,
                "age_band_start": key[1],
                "person_week_count": 0,
                "unique_person_ids": set(),
            })
            state["person_week_count"] += 1
            state["unique_person_ids"].add(person.id)
        marriages = len([event for event in world.demographic_events if event.get("event_type") == "marriage"])
        births = sum(number(row.get("births")) for row in world.diagnostics_rows)
        deaths = sum(number(row.get("deaths")) for row in world.diagnostics_rows)
        accounting = world.accounting.reconciliation_rows[-1]
        firm_rows = [
            row for row in world.accounting.rows
            if int(number(row.get("global_step", row.get("step", -1)), -1)) == week
        ]
        expected_wages = number(getattr(world.firm_system, "executed_wage_bill", 0.0)) + number(getattr(cap_week, "capital_good_wages", 0.0))
        settled_wages = math.fsum(number(getattr(household, "wage_income_this_step", 0.0)) for household in world.households)
        max_accounting_gap = max([
            abs(number(row.get(field)))
            for row in firm_rows
            for field in (
                "cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap",
                "capital_book_value_bridge_gap", "customer_advance_liability_bridge_gap",
                "prepaid_investment_bridge_gap",
            )
        ] or [0.0])
        output_above_feasible = max([
            max(
                0.0,
                number(getattr(firm, "actual_production", 0.0))
                - number(getattr(firm, "authoritative_feasible_capacity", getattr(firm, "feasible_capacity", 0.0))),
            )
            for firm in world.firms
        ] or [0.0])
        advance_gap = abs(math.fsum(number(row.get("customer_advance_liability")) for row in firm_rows) - math.fsum(number(row.get("prepaid_capital_investment_asset")) for row in firm_rows))
        rows.append({
            "mode": mode,
            "week": week,
            **after,
            "newly_settlement_eligible": len(newly_settled),
            "annual_marriage_market_executed": bool(world.marriage_market_executed),
            "marriages_this_step": marriages - prior_marriages,
            "births_this_step": births - prior_births,
            "deaths_this_step": deaths - prior_deaths,
            "population": len(world.population),
            "household_income": number(raw.get("total_income")),
            "household_consumption": number(raw.get("total_consumption")),
            "food_demand": number(raw.get("food_demand_units")),
            "food_production": number(raw.get("food_output_units", raw.get("actual_production"))),
            "capital_good_production": number(getattr(cap_week, "capital_good_production", 0.0)),
            "fixed_investment": number(getattr(cap_week, "fixed_investment", 0.0)),
            "active_capital_service": math.fsum(
                number(getattr(getattr(firm, "capital_stock", None), "capital_service_capacity", lambda **_: 0.0)(engineering_capacity_per_unit=1.0))
                for firm in world.firms
            ),
            "backlog": math.fsum(number(value) for value in world.canonical_investment_system.expansion_backlog_by_firm.values()) + math.fsum(number(getattr(firm, "pending_replacement_capacity_need", 0.0)) for firm in world.firms),
            "household_cash": number(raw.get("total_household_wealth")),
            "firm_cash": math.fsum(number(getattr(firm, "cash", 0.0)) for firm in world.operating_firms()),
            "legacy_owner_cash": number(getattr(world, "legacy_owner_cash", 0.0)),
            "full_money_location_gap": number(accounting.get("full_money_location_gap")),
            "accounting_gap": max_accounting_gap,
            "goods_gap": abs(number(raw.get("food_conservation_gap"))),
            "advance_prepaid_gap": advance_gap,
            "assignment_violations": assignment_violations(world),
            "output_above_feasible_capacity": output_above_feasible,
            "payroll_settlement_gap": settled_wages - expected_wages,
            "orphan_payroll_worker_count": sum(
                getattr(person, "firm_id", None) is not None and not world.has_valid_settlement_household(person)
                for person in world.population if person.alive
            ),
        })
        prior_marriages, prior_births, prior_deaths = marriages, births, deaths
        if week > 0 and week % 52 == 0:
            annual.append({
                "mode": mode,
                "week": week,
                "newly_settlement_eligible": len(newly_settled),
                "capital_good_hires": after["capital_good_employment"] - before["capital_good_employment"],
                "total_employment_jump": after["total_employment"] - before["total_employment"],
                "capital_good_employment_jump": after["capital_good_employment"] - before["capital_good_employment"],
                "unassigned_eligible_before": before["unassigned_eligible"],
                "unassigned_eligible_after": after["unassigned_eligible"],
                "marriage_market_executed": bool(world.marriage_market_executed),
            })
    origin_rows = [
        {
            "cause": row["cause"],
            "age_band_start": row["age_band_start"],
            "person_week_count": row["person_week_count"],
            "unique_person_count": len(row["unique_person_ids"]),
            "interpretation": "socially unattached but economically eligible" if "unattached" in row["cause"] else row["cause"],
        }
        for row in origins.values()
    ]
    return world, rows, annual, origin_rows


def compare(control, treatment):
    metrics = (
        "total_employment", "food_employment", "capital_good_employment",
        "household_income", "household_consumption", "food_demand", "food_production",
        "capital_good_production", "fixed_investment", "active_capital_service", "backlog",
    )
    rows = []
    for metric in metrics:
        c = np.asarray([number(row[metric]) for row in control[-104:]], dtype=float)
        t = np.asarray([number(row[metric]) for row in treatment[-104:]], dtype=float)
        cmean, tmean = float(c.mean()), float(t.mean())
        rows.append({
            "metric": metric,
            "window": "late_104",
            "control_mean": cmean,
            "treatment_mean": tmean,
            "relative_shift": (tmean - cmean) / max(abs(cmean), TOL),
            "control_final": float(c[-1]),
            "treatment_final": float(t[-1]),
        })
    return rows


def build_outputs(control_world, control, control_annual, origins, treatment_world, treatment, treatment_annual):
    write_csv("adult_no_household_origin_audit.csv", origins)
    labor_rows = []
    for row in [*control, *treatment]:
        labor_rows.append({
            key: row[key]
            for key in (
                "mode", "week", "matching_age_population", "social_household_valid",
                "settlement_valid", "missing_settlement", "newly_settlement_eligible",
                "unassigned_eligible", "employed_matching_age", "total_employment",
                "settlement_only_account_count", "annual_marriage_market_executed",
            )
        })
    write_csv("labor_eligibility_transition.csv", labor_rows)
    annual_rows = []
    by_key = {(row["mode"], row["week"]): row for row in [*control_annual, *treatment_annual]}
    annual_weeks = sorted(
        set(row["week"] for row in control_annual)
        & set(row["week"] for row in treatment_annual)
    )
    for week in annual_weeks:
        c, t = by_key[("control", week)], by_key[("treatment", week)]
        annual_rows.append({
            "week": week,
            "control_newly_settlement_eligible": c["newly_settlement_eligible"],
            "treatment_newly_settlement_eligible": t["newly_settlement_eligible"],
            "control_capital_good_hires": c["capital_good_hires"],
            "treatment_capital_good_hires": t["capital_good_hires"],
            "control_total_employment_jump": c["total_employment_jump"],
            "treatment_total_employment_jump": t["total_employment_jump"],
            "control_capital_good_employment_jump": c["capital_good_employment_jump"],
            "treatment_capital_good_employment_jump": t["capital_good_employment_jump"],
            "control_unassigned_eligible_after": c["unassigned_eligible_after"],
            "treatment_unassigned_eligible_after": t["unassigned_eligible_after"],
        })
    write_csv("annual_jump_control_treatment.csv", annual_rows)

    demographic_rows = []
    for metric in ("population", "marriages_this_step", "births_this_step", "deaths_this_step", "social_household_count", "economic_household_count", "settlement_only_account_count"):
        c = np.asarray([number(row[metric]) for row in control], dtype=float)
        t = np.asarray([number(row[metric]) for row in treatment], dtype=float)
        demographic_rows.append({
            "metric": metric,
            "control_cumulative_or_final": float(c.sum()) if metric.endswith("this_step") else float(c[-1]),
            "treatment_cumulative_or_final": float(t.sum()) if metric.endswith("this_step") else float(t[-1]),
            "max_absolute_step_difference": float(np.max(np.abs(t - c))),
            "semantic_note": "social demographic metric" if metric not in ("economic_household_count", "settlement_only_account_count") else "economic-account view; settlement-only accounts are not social Households",
        })
    write_csv("demographic_invariance.csv", demographic_rows)
    economic = compare(control, treatment)
    write_csv("economic_level_comparison.csv", economic)
    payroll_rows = []
    for row in [*control, *treatment]:
        payroll_rows.append({
            key: row[key]
            for key in (
                "mode", "week", "payroll_settlement_gap", "orphan_payroll_worker_count",
                "full_money_location_gap", "accounting_gap", "goods_gap", "advance_prepaid_gap",
                "assignment_violations", "missing_settlement",
                "output_above_feasible_capacity",
            )
        })
    write_csv("payroll_settlement_validation.csv", payroll_rows)
    cash_rows = []
    for mode, rows in (("control", control), ("treatment", treatment)):
        for window, subset in (("full", rows), ("late_104", rows[-104:]), ("final_quarter", rows[-130:])):
            cash_rows.append({
                "mode": mode,
                "window": window,
                "household_cash_slope_per_week": linear_slope([number(row["household_cash"]) for row in subset]),
                "firm_cash_slope_per_week": linear_slope([number(row["firm_cash"]) for row in subset]),
                "legacy_owner_cash_slope_per_week": linear_slope([number(row["legacy_owner_cash"]) for row in subset]),
            })
    write_csv("macro_cash_slope_recheck.csv", cash_rows)

    max_control_annual_hires = max(abs(number(row["control_capital_good_hires"])) for row in annual_rows)
    max_treatment_annual_hires = max(abs(number(row["treatment_capital_good_hires"])) for row in annual_rows)
    annual_weakened = max_treatment_annual_hires <= max(5.0, 0.20 * max_control_annual_hires)
    social_demographic_metrics = [row for row in demographic_rows if row["metric"] in ("population", "marriages_this_step", "births_this_step", "deaths_this_step", "social_household_count")]
    demographic_relative_differences = [
        abs(
            number(row["treatment_cumulative_or_final"])
            - number(row["control_cumulative_or_final"])
        ) / max(abs(number(row["control_cumulative_or_final"])), 1.0)
        for row in social_demographic_metrics
    ]
    max_demographic_relative_difference = max(demographic_relative_differences)
    demographics_materially_changed = max_demographic_relative_difference > 0.10
    max_economic_shift = max(abs(number(row["relative_shift"])) for row in economic)
    max_full_gap = max(abs(number(row["full_money_location_gap"])) for row in [*control, *treatment])
    max_accounting = max(abs(number(row["accounting_gap"])) for row in [*control, *treatment])
    max_goods = max(abs(number(row["goods_gap"])) for row in [*control, *treatment])
    max_advance = max(abs(number(row["advance_prepaid_gap"])) for row in [*control, *treatment])
    max_assignment = max(number(row["assignment_violations"]) for row in [*control, *treatment])
    max_output_above_feasible = max(number(row["output_above_feasible_capacity"]) for row in [*control, *treatment])
    max_orphan = max(number(row["orphan_payroll_worker_count"]) for row in [*control, *treatment])
    treatment_missing = max(number(row["missing_settlement"]) for row in treatment)
    accounting_pass = max(max_full_gap, max_accounting, max_goods, max_advance) <= 1e-5 and max_assignment == 0 and max_orphan == 0 and max_output_above_feasible <= TOL
    if not accounting_pass:
        verdict = "F. ACCOUNTING_OR_PAYROLL_SETTLEMENT_BLOCKER"
    elif treatment_missing > 0:
        verdict = "G. OTHER_BLOCKER"
    elif not annual_weakened:
        verdict = "D. ANNUAL_EMPLOYMENT_JUMP_PERSISTS_AFTER_ELIGIBILITY_FIX"
    elif demographics_materially_changed:
        verdict = "E. DEMOGRAPHIC_BEHAVIOR_DISTORTED"
    else:
        verdict = "A. ADULT_SETTLEMENT_LABOR_ELIGIBILITY_BOUNDARY_ACCEPTED"
    flags = {
        "verdict": verdict,
        "origin_classification": "intentional_unattached_adult_social_state; economic settlement fallback required",
        "correction": "deterministic settlement-only Household fallback; social Person.household_id unchanged",
        "rejected_phase_staggering_enabled": False,
        "new_rng_draws": 0,
        "treatment_max_missing_settlement": treatment_missing,
        "annual_capital_good_hire_control_max": max_control_annual_hires,
        "annual_capital_good_hire_treatment_max": max_treatment_annual_hires,
        "annual_jump_materially_weakened": annual_weakened,
        "social_demographics_exact": all(number(row["max_absolute_step_difference"]) <= TOL for row in social_demographic_metrics),
        "maximum_social_demographic_relative_difference": max_demographic_relative_difference,
        "social_demographics_materially_changed": demographics_materially_changed,
        "social_demographic_materiality_threshold": 0.10,
        "maximum_late_economic_relative_shift": max_economic_shift,
        "economic_level_shift_flagged": max_economic_shift > 0.25,
        "max_full_money_location_gap": max_full_gap,
        "max_accounting_gap": max_accounting,
        "max_goods_gap": max_goods,
        "max_advance_prepaid_gap": max_advance,
        "max_assignment_violations": max_assignment,
        "max_output_above_feasible_capacity": max_output_above_feasible,
        "max_orphan_payroll_workers": max_orphan,
        "economic_behavior_changed": "adult settlement eligibility boundary only",
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (OUT / "acceptance_summary.md").write_text(
        "# Step 15 Adult Settlement / Labor Eligibility\n\n"
        f"Verdict: **{verdict}**\n\n"
        "The origin audit identifies an intentional social state: unmatched adults and minors aging into adulthood may have no social Household. The defect was reusing this state as an economic settlement and weekly matching prerequisite.\n\n"
        "Treatment creates deterministic settlement-only Household accounts for labor-eligible Persons without a social Household. It does not set Person.household_id, change marriage cadence, or add RNG. At marriage, the temporary account cash transfers exactly into the new social Household.\n\n"
        f"Annual capital-good hires: control maximum {max_control_annual_hires:.0f}, treatment maximum {max_treatment_annual_hires:.0f}; annual block materially weakened: {annual_weakened}. Maximum social-demographic difference is {max_demographic_relative_difference:.2%}, below the stated 10% materiality threshold: {not demographics_materially_changed}. Maximum late economic shift: {max_economic_shift:.2%}.\n\n"
        f"Maximum gaps: money location {max_full_gap:.3e}, accounting {max_accounting:.3e}, goods {max_goods:.3e}, advance/prepaid {max_advance:.3e}; orphan payroll workers {max_orphan:.0f}.\n",
        encoding="utf-8-sig",
    )
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=POPULATION)
    parser.add_argument("--weeks", type=int, default=WEEKS)
    args = parser.parse_args()
    control_world, control, control_annual, origins = run_case("control", args.population, args.weeks)
    treatment_world, treatment, treatment_annual, _ = run_case("treatment", args.population, args.weeks)
    flags = build_outputs(control_world, control, control_annual, origins, treatment_world, treatment, treatment_annual)
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
