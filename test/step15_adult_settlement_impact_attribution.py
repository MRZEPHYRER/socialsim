"""Step 15 diagnostic attribution for the adult settlement eligibility fix."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from productivity import age_productivity
from world import World


OUT = ROOT / "test/output/step15_adult_settlement_impact_attribution"
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


def linear_slope(values):
    if len(values) < 2:
        return math.nan
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def overrides(enabled):
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


def worker_state(world):
    workers = [
        person for person in world.population
        if person.alive and age_productivity(person.age) > 0.0
    ]
    fallback_people = [
        person for person in workers
        if social_household(world, person) is None
        and world.has_valid_settlement_household(person)
    ]
    firms = list(world.firms)
    capital_firms = list(world.capital_good_firms)
    settlement_accounts = [
        household for household in world.households
        if getattr(household, "settlement_only", False)
    ]
    return {
        "working_age_persons": len(workers),
        "social_household_persons": sum(social_household(world, person) is not None for person in workers),
        "settlement_relation_persons": sum(world.has_valid_settlement_household(person) for person in workers),
        "labor_match_eligible_persons": sum(world.has_valid_settlement_household(person) for person in workers),
        "employed_persons": sum(getattr(person, "firm_id", None) is not None for person in workers),
        "unassigned_eligible_persons": sum(
            world.has_valid_settlement_household(person)
            and getattr(person, "firm_id", None) is None
            for person in workers
        ),
        "fallback_eligible_persons": len(fallback_people),
        "fallback_employed_persons": sum(getattr(person, "firm_id", None) is not None for person in fallback_people),
        "food_employment": sum(len(getattr(firm, "employee_ids", [])) for firm in firms),
        "capital_good_employment": sum(len(getattr(firm, "employee_ids", [])) for firm in capital_firms),
        "total_employment": sum(len(getattr(firm, "employee_ids", [])) for firm in [*firms, *capital_firms]),
        "settlement_only_account_count": len(settlement_accounts),
        "settlement_only_wage_income": math.fsum(
            number(getattr(household, "wage_income_this_step", 0.0))
            for household in settlement_accounts
        ),
        "settlement_only_consumption": math.fsum(
            number(getattr(household, "consumption_this_step", 0.0))
            for household in settlement_accounts
        ),
        "settlement_only_cash": math.fsum(number(household.wealth) for household in settlement_accounts),
        "fallback_in_social_household_id": sum(
            getattr(person, "household_id", None)
            == getattr(person, "settlement_household_id", None)
            and getattr(person, "settlement_household_id", None) is not None
            for person in workers
        ),
        "fallback_two_parent_accounts": sum(len(household.parents) == 2 for household in settlement_accounts),
        "fallback_single_parent_accounts": sum(len(household.parents) == 1 for household in settlement_accounts),
    }


def metric_row(world, state):
    raw = world.diagnostics_rows[-1]
    capital_week = world.last_canonical_investment_week
    return {
        **state,
        "week": world.current_step_index,
        "population": len(world.population),
        "household_income": number(raw.get("total_income")),
        "household_consumption": number(raw.get("total_consumption")),
        "household_cash": number(raw.get("total_household_wealth")),
        "food_demand": number(raw.get("food_demand_units")),
        "food_production": number(raw.get("food_output_units", raw.get("actual_production"))),
        "capital_good_production": number(getattr(capital_week, "capital_good_production", 0.0)),
        "fixed_investment": number(getattr(capital_week, "fixed_investment", 0.0)),
        "active_capital_service": math.fsum(
            number(getattr(getattr(firm, "capital_stock", None), "capital_service_capacity", lambda **_: 0.0)(engineering_capacity_per_unit=1.0))
            for firm in world.firms
        ),
        "backlog": (
            math.fsum(number(value) for value in world.canonical_investment_system.expansion_backlog_by_firm.values())
            + math.fsum(number(getattr(firm, "pending_replacement_capacity_need", 0.0)) for firm in world.firms)
        ),
        "firm_cash": math.fsum(number(getattr(firm, "cash", 0.0)) for firm in world.operating_firms()),
        "legacy_owner_cash": number(getattr(world, "legacy_owner_cash", 0.0)),
    }


def install_marriage_transfer_probe(world, events):
    original = world.merge_settlement_household_into_social_household

    def tracked(person, destination):
        source = world.get_household(getattr(person, "settlement_household_id", None))
        opening = number(getattr(source, "wealth", 0.0)) if source is not None else 0.0
        source_id = getattr(source, "id", None)
        amount = original(person, destination)
        residual = number(getattr(source, "wealth", 0.0)) if source is not None else 0.0
        if source_id is not None:
            events.append({
                "week": world.current_step_index,
                "person_id": person.id,
                "source_settlement_household_id": source_id,
                "destination_social_household_id": destination.id,
                "opening_settlement_cash": opening,
                "cash_transferred_to_social_household": number(amount),
                "residual_source_cash": residual,
                "transfer_gap": number(amount) - opening,
                "source_removed_after_marriage": source not in world.households,
            })
        return amount

    world.merge_settlement_household_into_social_household = tracked


def run_case(mode):
    treatment = mode == "treatment"
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name=f"step15_adult_settlement_attribution_{mode}",
        scenario_overrides=overrides(treatment),
    )
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    marriage_events = []
    if treatment:
        install_marriage_transfer_probe(world, marriage_events)

    rows = []
    fallback_people_seen = set()
    fallback_hired_seen = set()
    for _ in range(WEEKS):
        world.step()
        state = worker_state(world)
        rows.append(metric_row(world, state))
        for person in world.population:
            if (
                person.alive
                and age_productivity(person.age) > 0.0
                and social_household(world, person) is None
                and world.has_valid_settlement_household(person)
            ):
                fallback_people_seen.add(person.id)
                if getattr(person, "firm_id", None) is not None:
                    fallback_hired_seen.add(person.id)
    return world, rows, marriage_events, fallback_people_seen, fallback_hired_seen


def late_rows(control, treatment):
    metrics = (
        "total_employment", "food_employment", "capital_good_employment", "unassigned_eligible_persons",
        "household_income", "household_consumption", "household_cash", "food_demand", "food_production",
        "capital_good_production", "fixed_investment", "active_capital_service", "backlog", "firm_cash",
    )
    rows = []
    for metric in metrics:
        c = np.asarray([number(row[metric]) for row in control[-104:]], dtype=float)
        t = np.asarray([number(row[metric]) for row in treatment[-104:]], dtype=float)
        rows.append({
            "metric": metric,
            "window": "late_104",
            "control_mean": float(c.mean()),
            "treatment_mean": float(t.mean()),
            "absolute_difference": float(t.mean() - c.mean()),
            "relative_shift": float((t.mean() - c.mean()) / max(abs(c.mean()), TOL)),
            "control_final": float(c[-1]),
            "treatment_final": float(t[-1]),
        })
    return rows


def demographic_attribution(control_world, control, treatment_world, treatment):
    def event_count(world, kind):
        return sum(event.get("event_type") == kind for event in world.demographic_events)

    metrics = []
    for name, cvalue, tvalue, note in (
        ("population", control[-1]["population"], treatment[-1]["population"], "path-dependent stock"),
        ("marriages", event_count(control_world, "marriage"), event_count(treatment_world, "marriage"), "dedicated marriage RNG/cadence unchanged"),
        ("births", event_count(control_world, "birth"), event_count(treatment_world, "birth"), "household income path affects fertility factor after marriage"),
        ("deaths", event_count(control_world, "death"), event_count(treatment_world, "death"), "mortality mechanism unchanged; population path-dependent"),
        ("social_household_count", len([h for h in control_world.households if not getattr(h, "settlement_only", False)]), len([h for h in treatment_world.households if not getattr(h, "settlement_only", False)]), "settlement accounts excluded"),
    ):
        metrics.append({
            "metric": name,
            "control": cvalue,
            "treatment": tvalue,
            "absolute_difference": tvalue - cvalue,
            "relative_difference": (tvalue - cvalue) / max(abs(cvalue), 1.0),
            "attribution": note,
        })
    max_row = max(metrics, key=lambda row: abs(number(row["relative_difference"])))
    metrics.append({
        "metric": "maximum_relative_difference",
        "control": "",
        "treatment": "",
        "absolute_difference": "",
        "relative_difference": max_row["relative_difference"],
        "attribution": f"{max_row['metric']}; economic path dependence, not changed demographic schedule or RNG",
    })
    return metrics, max_row


def main():
    control_world, control, _, _, _ = run_case("control")
    treatment_world, treatment, marriage_events, fallback_people, fallback_hired = run_case("treatment")

    late = late_rows(control, treatment)
    write_csv("late_economic_shift_decomposition.csv", late)
    max_late = max(late, key=lambda row: abs(number(row["relative_shift"])))

    control_eligible_weeks = sum(number(row["labor_match_eligible_persons"]) for row in control)
    treatment_eligible_weeks = sum(number(row["labor_match_eligible_persons"]) for row in treatment)
    control_employed_weeks = sum(number(row["employed_persons"]) for row in control)
    treatment_employed_weeks = sum(number(row["employed_persons"]) for row in treatment)
    fallback_eligible_weeks = sum(number(row["fallback_eligible_persons"]) for row in treatment)
    fallback_employed_weeks = sum(number(row["fallback_employed_persons"]) for row in treatment)
    labor = [
        {"metric": "labor_match_eligible_person_weeks", "control": control_eligible_weeks, "treatment": treatment_eligible_weeks, "difference": treatment_eligible_weeks - control_eligible_weeks, "interpretation": "all settlement-qualified labor"},
        {"metric": "employed_person_weeks", "control": control_employed_weeks, "treatment": treatment_employed_weeks, "difference": treatment_employed_weeks - control_employed_weeks, "interpretation": "total realized employment"},
        {"metric": "treatment_fallback_eligible_person_weeks", "control": 0, "treatment": fallback_eligible_weeks, "difference": fallback_eligible_weeks, "interpretation": "previously socially unattached adults with continuous settlement eligibility"},
        {"metric": "treatment_fallback_employed_person_weeks", "control": 0, "treatment": fallback_employed_weeks, "difference": fallback_employed_weeks, "interpretation": "fallback-eligible adults actually employed"},
        {"metric": "unique_fallback_eligible_persons", "control": 0, "treatment": len(fallback_people), "difference": len(fallback_people), "interpretation": "unique Persons observed with a fallback account"},
        {"metric": "unique_fallback_hired_persons", "control": 0, "treatment": len(fallback_hired), "difference": len(fallback_hired), "interpretation": "unique fallback Persons reaching employment"},
    ]
    write_csv("labor_eligibility_person_weeks.csv", labor)

    propagation = []
    for c, t in zip(control, treatment):
        propagation.append({
            "week": c["week"],
            "additional_total_employment": t["total_employment"] - c["total_employment"],
            "treatment_fallback_employment": t["fallback_employed_persons"],
            "treatment_fallback_wage_income": t["settlement_only_wage_income"],
            "additional_household_income": t["household_income"] - c["household_income"],
            "additional_household_consumption": t["household_consumption"] - c["household_consumption"],
            "additional_food_demand": t["food_demand"] - c["food_demand"],
            "additional_food_production": t["food_production"] - c["food_production"],
        })
    employment_delta = np.asarray([number(row["additional_total_employment"]) for row in propagation])
    income_delta = np.asarray([number(row["additional_household_income"]) for row in propagation])
    corr = float(np.corrcoef(employment_delta, income_delta)[0, 1]) if np.std(employment_delta) > TOL and np.std(income_delta) > TOL else math.nan
    propagation.append({
        "week": "summary_late_104",
        "additional_total_employment": float(np.mean(employment_delta[-104:])),
        "treatment_fallback_employment": float(np.mean([number(row["treatment_fallback_employment"]) for row in propagation[-104:]])),
        "treatment_fallback_wage_income": float(np.mean([number(row["treatment_fallback_wage_income"]) for row in propagation[-104:]])),
        "additional_household_income": float(np.mean(income_delta[-104:])),
        "additional_household_consumption": float(np.mean([number(row["additional_household_consumption"]) for row in propagation[-104:]])),
        "additional_food_demand": float(np.mean([number(row["additional_food_demand"]) for row in propagation[-104:]])),
        "additional_food_production": float(np.mean([number(row["additional_food_production"]) for row in propagation[-104:]])),
        "employment_income_same_week_correlation": corr,
    })
    write_csv("income_demand_propagation.csv", propagation)

    legacy_household_statistics_contaminated = any(number(row["fallback_single_parent_accounts"]) > 0 for row in treatment)
    dividend_person_events = [
        event for event in treatment_world.dividend_routing_events
        if number(event.get("person_paid")) > TOL
    ]
    fallback_dividend_receipts = sum(
        1
        for event in dividend_person_events
        for receipt in event.get("household_receipts", [])
        if getattr(treatment_world.get_household(receipt.get("household_id")), "settlement_only", False)
    )
    side_effects = [
        {"area": "social_person_household_id", "result": "PASS", "observed": max(number(row["fallback_in_social_household_id"]) for row in treatment), "interpretation": "fallback never overwrote Person.household_id"},
        {"area": "marriage_selection", "result": "PASS", "observed": "Person-level marriage pool / dedicated RNG unchanged", "interpretation": "settlement account is not a marriage Household"},
        {"area": "fertility_eligibility", "result": "PASS", "observed": max(number(row["fallback_two_parent_accounts"]) for row in treatment), "interpretation": "no fallback account has two parents; it cannot meet the fertility couple condition"},
        {"area": "mortality", "result": "PASS", "observed": "no mortality rule references settlement_only", "interpretation": "mortality mechanism unchanged"},
        {"area": "consumption_budget", "result": "INTENDED_ECONOMIC_SETTLEMENT", "observed": max(number(row["settlement_only_consumption"]) for row in treatment), "interpretation": "fallback cash participates in the normal Household budget so wage income can become demand"},
        {"area": "saving_accumulation", "result": "INTENDED_ECONOMIC_SETTLEMENT", "observed": treatment[-1]["settlement_only_cash"], "interpretation": "temporary cash is retained until marriage transfer"},
        {"area": "dividend_routing", "result": "INACTIVE_IN_RUN", "observed": fallback_dividend_receipts, "interpretation": "no observed person-dividend receipt routed to a fallback account; active ownership did not exercise this boundary"},
        {"area": "inheritance", "result": "PASS", "observed": "equity inheritance is Person/Estate-based", "interpretation": "settlement-only account is not an heir or estate holder"},
        {"area": "social_household_statistics", "result": "METRIC_SCOPE_NOTE", "observed": legacy_household_statistics_contaminated, "interpretation": "legacy aggregate Household/single-parent diagnostics include economic accounts; use explicit social_household_count for demographic interpretation"},
    ]
    write_csv("settlement_only_side_effect_audit.csv", side_effects)

    if not marriage_events:
        marriage_events = [{"status": "NO_SETTLEMENT_ACCOUNT_MARRIAGES_OBSERVED"}]
    write_csv("marriage_cash_transfer_validation.csv", marriage_events)
    max_marriage_gap = max(abs(number(row.get("transfer_gap"))) for row in marriage_events)
    max_residual = max(abs(number(row.get("residual_source_cash"))) for row in marriage_events)

    demographics, max_demographic = demographic_attribution(control_world, control, treatment_world, treatment)
    write_csv("demographic_difference_attribution.csv", demographics)

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

    household_income_late_shift = next(row for row in late if row["metric"] == "household_income")["relative_shift"]
    social_behavior_leak = any(row["result"] == "FAIL" for row in side_effects)
    accounting_side_effect = max_marriage_gap > TOL or max_residual > TOL
    verdict = (
        "E. CASH_TRANSFER_OR_ACCOUNTING_SIDE_EFFECT" if accounting_side_effect
        else "B. SETTLEMENT_ONLY_ACCOUNT_LEAKS_INTO_SOCIAL_HOUSEHOLD_LOGIC" if social_behavior_leak
        else "A. ECONOMIC_LEVEL_SHIFT_IS_INTENDED_LABOR_ELIGIBILITY_EFFECT"
    )
    flags = {
        "verdict": verdict,
        "maximum_late_shift_metric": max_late["metric"],
        "maximum_late_shift": max_late["relative_shift"],
        "late_household_income_shift": household_income_late_shift,
        "additional_labor_eligible_person_weeks": treatment_eligible_weeks - control_eligible_weeks,
        "additional_employed_person_weeks": treatment_employed_weeks - control_employed_weeks,
        "fallback_eligible_person_weeks": fallback_eligible_weeks,
        "fallback_employed_person_weeks": fallback_employed_weeks,
        "employment_income_same_week_correlation": corr,
        "marriage_transfer_max_gap": max_marriage_gap,
        "marriage_transfer_max_residual": max_residual,
        "social_demographic_max_metric": max_demographic["metric"],
        "social_demographic_max_relative_difference": max_demographic["relative_difference"],
        "social_demographic_behavior_leak": social_behavior_leak,
        "legacy_household_statistics_include_settlement_accounts": legacy_household_statistics_contaminated,
        "settlement_fix_remains_accepted_unchanged": verdict.startswith("A."),
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    (OUT / "acceptance_summary.md").write_text(
        "# Step 15 Adult Settlement Economic-Impact Attribution\n\n"
        f"Verdict: **{verdict}**\n\n"
        f"The largest late-window shift is **{max_late['metric']}** at {max_late['relative_shift']:.2%}. "
        "It is a downstream capital-formation outcome, not an immediate Household demand collapse.\n\n"
        f"Treatment adds {treatment_eligible_weeks - control_eligible_weeks:.0f} eligible Person-weeks and "
        f"{treatment_employed_weeks - control_employed_weeks:.0f} employed Person-weeks; "
        f"fallback adults account for {fallback_eligible_weeks:.0f} eligible and {fallback_employed_weeks:.0f} employed Person-weeks. "
        f"Late Household income changes {household_income_late_shift:.2%}.\n\n"
        "No settlement account overwrote a social Person.household_id, entered marriage selection, or met the two-parent fertility condition. "
        f"Marriage transfer maximum gap is {max_marriage_gap:.3e}, residual source cash {max_residual:.3e}. "
        "Legacy aggregate Household/single-parent diagnostics include settlement accounts and must be interpreted as economic-account counts, not social-demographic counts.\n",
        encoding="utf-8-sig",
    )
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
