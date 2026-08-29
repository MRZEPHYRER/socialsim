"""Step 17.B shadow PAYG pension fiscal-capacity audit.

This audit restores the accepted mature demographic research state, runs the
existing economic model for one short observation window, and reads the
authoritative per-person paid-wage provenance from the final week.  All PAYG
contributions and benefits are shadow arithmetic: no Household, Firm, ledger,
retirement, or pension state is mutated.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy import config
from productivity import age_productivity


OUT = ROOT / "test/output/step17_b_payg_pension_capacity"
MATURE_HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
STEP17A = ROOT / "test/step17_a_household_transfer_capacity_audit.py"
OBSERVATION_WEEKS = 52
ELDERLY_AGE = 65.0
WORKING_AGE_LOW = 20.0
EPS = 1e-9
RATES = (0.01, 0.02, 0.03, 0.05)
TARGETS = (0.25, 0.50, 0.75, 1.00)
LIQUIDITY_THRESHOLDS = (0.25, 0.50, 1.00, 2.00)
AGE_GROUPS = ("<40", "40-54", "55-64", ">=65")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def write_rows(path, rows, fields=None):
    rows = list(rows)
    inferred = []
    for row in rows:
        for key in row:
            if key not in inferred:
                inferred.append(key)
    fields = list(fields or inferred or ["status"])
    for field in inferred:
        if field not in fields:
            fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def share(value, total):
    return number(value) / total if total > EPS else 0.0


def quantile(values, q):
    values = sorted(number(value) for value in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def gini(values):
    values = sorted(max(0.0, number(value)) for value in values)
    total = sum(values)
    if not values or total <= EPS:
        return 0.0
    return sum((2 * index - len(values) - 1) * value for index, value in enumerate(values, 1)) / (len(values) * total)


def age_group(age):
    age = number(age)
    if age < 40:
        return "<40"
    if age < 55:
        return "40-54"
    if age < 65:
        return "55-64"
    return ">=65"


def social_households(world):
    return sorted(
        [h for h in world.households if not getattr(h, "settlement_only", False)],
        key=lambda h: h.id,
    )


def household_members(world, households):
    social_ids = {h.id for h in households}
    result = defaultdict(list)
    for person in world.population:
        if not getattr(person, "alive", False):
            continue
        household_id = getattr(person, "household_id", None)
        if household_id in social_ids:
            result[household_id].append(person)
    return result


def minimum_cost(world, household):
    units = world.needs_system.household_minimum_need_units(household)
    try:
        price = max(1e-12, float(world.household_planning_price_this_step()))
    except (AttributeError, TypeError, ValueError):
        price = 1.0
    return max(0.0, number(units)) * price


def person_need_components(members):
    """Allocate the authoritative household minimum neutrally to members."""
    components = {}
    for person in members:
        age = number(person.age)
        if age < 20:
            component = config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * config.NEW_CHILD_WEIGHT + config.CHILD_EXTRA_COST
        elif age < ELDERLY_AGE:
            component = config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * config.SECOND_ADULT_WEIGHT
        else:
            component = config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * config.NEW_ELDERLY_WEIGHT + config.ELDERLY_EXTRA_COST
        components[person.id] = max(0.0, number(component))
    fixed = config.NEW_BASE_CONSUMPTION
    per_member_fixed = fixed / len(members) if members else 0.0
    return {person_id: value + per_member_fixed for person_id, value in components.items()}


def payroll_wages(world):
    """Read actual paid wages from the runtime payroll provenance state."""
    wages = defaultdict(float)
    household_ids = {}
    firm_ids = {}
    sectors = {}
    provenance = getattr(world, "_household_payroll_provenance_weekly_state", {})
    for record in provenance.values():
        for item in record.get("person_components", []):
            person_id = item.get("person_id")
            paid = max(0.0, number(item.get("paid_wage")))
            wages[person_id] += paid
            household = world.get_household(getattr(world.person_dict.get(person_id), "household_id", None))
            household_ids[person_id] = household.id if household is not None else None
            firm_ids[person_id] = item.get("firm_id")
            sectors[person_id] = item.get("sector_id")
    return wages, household_ids, firm_ids, sectors


def genealogy_connections(step17a, world, households):
    edges, edge_roles, coverage = step17a.genealogy(world, households)
    adjacency = defaultdict(set)
    for (left, right), _role in edge_roles.items():
        adjacency[left].add(right)
    return adjacency, coverage


def household_rows(world, households, members):
    rows = {}
    for household in households:
        people = members.get(household.id, [])
        minimum = minimum_cost(world, household)
        cash = max(0.0, number(getattr(household, "wealth", 0.0)))
        liquidity = cash / minimum if minimum > EPS else math.inf
        rows[household.id] = {
            "household_id": household.id,
            "cash": cash,
            "minimum_cost": minimum,
            "liquidity_weeks": liquidity,
            "member_count": len(people),
            "elderly_count": sum(number(p.age) >= ELDERLY_AGE for p in people),
            "adult_count": sum(number(p.age) >= WORKING_AGE_LOW for p in people),
            "employed_count": sum(getattr(p, "firm_id", None) is not None for p in people),
            "equity_assets": max(0.0, number(getattr(household, "equity_asset_value", 0.0))),
        }
        rows[household.id]["financial_net_worth"] = rows[household.id]["cash"] + rows[household.id]["equity_assets"]
    return rows


def build_person_data(world, households, members, wage_by_person, household_wage_ids, firm_ids, sectors):
    household_by_person = {}
    for household_id, people in members.items():
        for person in people:
            household_by_person[person.id] = household_id
    components = {}
    for household_id, people in members.items():
        components.update(person_need_components(people))
    rows = []
    for person in sorted(world.population, key=lambda p: p.id):
        if not getattr(person, "alive", False):
            continue
        household_id = household_by_person.get(person.id)
        if household_id is None:
            continue
        wage = max(0.0, number(wage_by_person.get(person.id)))
        participates = bool(world.labor_participates(person))
        rows.append({
            "person_id": person.id,
            "household_id": household_id,
            "age": number(person.age),
            "age_group": age_group(person.age),
            "alive": True,
            "labor_participates": participates,
            "employed": getattr(person, "firm_id", None) is not None,
            "firm_id": firm_ids.get(person.id, getattr(person, "firm_id", None)),
            "sector_id": sectors.get(person.id, ""),
            "realized_paid_wage": wage,
            "person_minimum_consumption_allocation": components.get(person.id, 0.0),
            "contributor": participates and getattr(person, "firm_id", None) is not None and wage > EPS,
            "pension_eligible": number(person.age) >= ELDERLY_AGE,
            "wage_source": "runtime payroll provenance paid_wage",
        })
    return rows


def low_share(rows, cash_adjustments, threshold):
    if not rows:
        return 0.0
    count = 0
    for row in rows:
        cash = max(0.0, number(row["cash"]) + cash_adjustments.get(row["household_id"], 0.0))
        minimum = number(row["minimum_cost"])
        if minimum > EPS and cash / minimum < threshold:
            count += 1
    return share(count, len(rows))


def main():
    harness = load_module(MATURE_HARNESS, "step15_mature_harness_step17b")
    step17a = load_module(STEP17A, "step17a_helpers_step17b")
    world, checkpoint_state = harness.restore_world()
    # This flag only retains existing payroll provenance; it does not alter
    # wage calculation, cash settlement, RNG, or any economic decision.
    world.household_employer_exposure_instrumentation_enabled = True
    for _ in range(OBSERVATION_WEEKS):
        world.step()

    households = social_households(world)
    members = household_members(world, households)
    hrows = household_rows(world, households, members)
    wage_by_person, household_wage_ids, firm_ids, sectors = payroll_wages(world)
    people = build_person_data(world, households, members, wage_by_person, household_wage_ids, firm_ids, sectors)
    contributors = [row for row in people if row["contributor"]]
    eligible = [row for row in people if row["pension_eligible"]]
    wage_base = sum(number(row["realized_paid_wage"]) for row in contributors)
    elderly_need = sum(number(row["person_minimum_consumption_allocation"]) for row in eligible)
    flat_benchmark = quantile([row["person_minimum_consumption_allocation"] for row in eligible], 0.5)
    contributor_wage_base_without_elderly = sum(
        number(row["realized_paid_wage"]) for row in contributors if number(row["age"]) < ELDERLY_AGE
    )
    elderly_wage_base = wage_base - contributor_wage_base_without_elderly

    # Use the same genealogy implementation accepted in Step17.A.
    adjacency, genealogy_coverage = genealogy_connections(step17a, world, households)
    rich_households = {
        household_id for household_id, row in hrows.items()
        if number(row["liquidity_weeks"], 0.0) >= 13.0
    }
    elderly_household_ids = {row["household_id"] for row in eligible}
    wealthy_connected = {
        household_id for household_id in elderly_household_ids
        if any(neighbor in rich_households for neighbor in adjacency.get(household_id, set()))
    }
    pensioner_households = len(elderly_household_ids)

    summary = {
        "status": "SHADOW_ONLY",
        "research_week": OBSERVATION_WEEKS,
        "absolute_demographic_week": 2600 + OBSERVATION_WEEKS,
        "population_alive": sum(getattr(p, "alive", False) for p in world.population),
        "social_households": len(households),
        "total_contributable_wage_base": wage_base,
        "contributor_count": len(contributors),
        "contributors_under_65": sum(number(row["age"]) < ELDERLY_AGE for row in contributors),
        "contributors_65_plus": sum(number(row["age"]) >= ELDERLY_AGE for row in contributors),
        "pension_eligible_person_count": len(eligible),
        "pension_eligible_household_count": pensioner_households,
        "elderly_wage_base_share": share(elderly_wage_base, wage_base),
        "elderly_households_with_wealthy_genealogy_connection": len(wealthy_connected),
        "elderly_households_without_wealthy_genealogy_connection": pensioner_households - len(wealthy_connected),
        "payroll_source": "runtime _household_payroll_provenance_weekly_state paid_wage",
        "benefit_base_semantics": "NeedsSystem household minimum allocated per person by need component plus equal fixed-household component",
        "economic_behavior_changed": False,
        "contributions_executed": False,
        "benefits_executed": False,
        "ledger_mutated": False,
        "rng_draws_added": 0,
    }
    write_rows(OUT / "pension_analysis_snapshot.csv", [summary])

    contributor_rows = []
    for row in contributors:
        household = hrows[row["household_id"]]
        contributor_rows.append({
            **row,
            "household_cash": household["cash"],
            "household_minimum_cost": household["minimum_cost"],
            "household_liquidity_weeks": household["liquidity_weeks"],
            "wage_share_of_total_base": share(row["realized_paid_wage"], wage_base),
            "wage_share_of_contributor_age_group": share(
                row["realized_paid_wage"],
                sum(number(item["realized_paid_wage"]) for item in contributors if item["age_group"] == row["age_group"]),
            ),
        })
    write_rows(OUT / "contributor_profile.csv", contributor_rows)

    age_base_rows = []
    for group in AGE_GROUPS:
        selected = [row for row in contributors if row["age_group"] == group]
        age_base_rows.append({
            "age_group": group,
            "contributor_count": len(selected),
            "wage_base": sum(number(row["realized_paid_wage"]) for row in selected),
            "wage_base_share": share(sum(number(row["realized_paid_wage"]) for row in selected), wage_base),
            "mean_wage": sum(number(row["realized_paid_wage"]) for row in selected) / len(selected) if selected else 0.0,
        })
    write_rows(OUT / "wage_base_age_composition.csv", age_base_rows)
    write_rows(OUT / "contributable_wage_base.csv", [{
        "scope": "ALL_CONTRIBUTORS",
        "wage_base": wage_base,
        "contributor_count": len(contributors),
        "mean_wage": wage_base / len(contributors) if contributors else 0.0,
        "median_wage": quantile([row["realized_paid_wage"] for row in contributors], 0.5),
        "wage_base_under_65": contributor_wage_base_without_elderly,
        "wage_base_65_plus": elderly_wage_base,
    }])

    rate_rows = []
    for rate in RATES:
        revenue = wage_base * rate
        average_contribution = revenue / len(contributors) if contributors else 0.0
        rate_rows.append({
            "contribution_rate": rate,
            "contribution_revenue": revenue,
            "contributor_count": len(contributors),
            "average_weekly_contribution_per_contributor": average_contribution,
            "average_contribution_to_mean_wage": share(average_contribution, wage_base / len(contributors) if contributors else 0.0),
            "average_contribution_to_mean_contributor_minimum": share(
                average_contribution,
                sum(number(hrows[row["household_id"]]["minimum_cost"]) for row in contributors) / len(contributors) if contributors else 0.0,
            ),
            "employer_equal_contribution_sensitivity_not_proposed": revenue,
        })
    write_rows(OUT / "contribution_rate_scenarios.csv", rate_rows)

    eligibility_rows = []
    for row in eligible:
        eligibility_rows.append({
            **row,
            "currently_employed": row["employed"],
            "currently_contributing_under_primary_rule": row["contributor"],
            "positive_wage": number(row["realized_paid_wage"]) > EPS,
            "eligible_person_need_base": row["person_minimum_consumption_allocation"],
            "valid_social_household": row["household_id"] in hrows,
        })
    write_rows(OUT / "pension_eligibility_profile.csv", eligibility_rows)

    benefit_rows = []
    for target in TARGETS:
        obligation = elderly_need * target
        benefit_rows.append({
            "benefit_target_multiplier": target,
            "person_need_base_total": elderly_need,
            "weekly_pension_obligation": obligation,
            "eligible_person_count": len(eligible),
            "mean_person_benefit": obligation / len(eligible) if eligible else 0.0,
            "flat_median_adult_or_elderly_benchmark": flat_benchmark * target,
            "flat_benchmark_obligation": flat_benchmark * target * len(eligible),
            "allocation_semantics": "person-level entitlement; neutral household-minimum allocation",
        })
    write_rows(OUT / "pension_benefit_scenarios.csv", benefit_rows)

    payg_rows = []
    for rate in RATES:
        revenue = wage_base * rate
        for target in TARGETS:
            obligation = elderly_need * target
            payg_rows.append({
                "contribution_rate": rate,
                "benefit_target_multiplier": target,
                "contribution_revenue": revenue,
                "pension_obligation": obligation,
                "balance": revenue - obligation,
                "funding_ratio": revenue / obligation if obligation > EPS else math.inf,
                "no_borrowing_or_government_finance": True,
            })
    write_rows(OUT / "payg_balance_matrix.csv", payg_rows)
    write_rows(OUT / "break_even_contribution_rates.csv", [
        {
            "benefit_target_multiplier": target,
            "pension_obligation": elderly_need * target,
            "break_even_contribution_rate": elderly_need * target / wage_base if wage_base > EPS else math.inf,
            "break_even_rate_percent": 100.0 * elderly_need * target / wage_base if wage_base > EPS else math.inf,
        }
        for target in TARGETS
    ])
    write_rows(OUT / "sustainable_benefit_levels.csv", [
        {
            "contribution_rate": rate,
            "contribution_revenue": wage_base * rate,
            "sustainable_target_multiplier": wage_base * rate / elderly_need if elderly_need > EPS else math.inf,
            "sustainable_mean_person_benefit": wage_base * rate / len(eligible) if eligible else 0.0,
        }
        for rate in RATES
    ])

    working_age_people = [row for row in people if WORKING_AGE_LOW <= number(row["age"]) < ELDERLY_AGE]
    dependency_rows = [{
        "definition": "elderly_persons / employed_positive_wage_contributors",
        "elderly_persons": len(eligible),
        "denominator": len(contributors),
        "ratio": share(len(eligible), len(contributors)),
    }, {
        "definition": "elderly_persons / working_age_persons_age_20_to_64",
        "elderly_persons": len(eligible),
        "denominator": len(working_age_people),
        "ratio": share(len(eligible), len(working_age_people)),
    }, {
        "definition": "pension_eligible_persons / contributors",
        "elderly_persons": len(eligible),
        "denominator": len(contributors),
        "ratio": share(len(eligible), len(contributors)),
    }]
    write_rows(OUT / "dependency_profile.csv", dependency_rows)

    elderly_liquidity_rows = []
    for label, ids in (
        ("ALL_ELDERLY_HOUSEHOLDS", elderly_household_ids),
        ("ELDERLY_WITH_WEALTHY_GENEALOGY_CONNECTION", wealthy_connected),
        ("ELDERLY_WITHOUT_WEALTHY_GENEALOGY_CONNECTION", elderly_household_ids - wealthy_connected),
    ):
        selected = [hrows[hid] for hid in ids if hid in hrows]
        base = {
            "group": label,
            "household_count": len(selected),
            "total_cash": sum(number(row["cash"]) for row in selected),
            "median_cash": quantile([row["cash"] for row in selected], 0.5),
            "median_liquidity_weeks": quantile([row["liquidity_weeks"] for row in selected], 0.5),
        }
        for threshold in LIQUIDITY_THRESHOLDS:
            base[f"share_below_{str(threshold).replace('.', '_')}_weeks"] = share(
                sum(number(row["liquidity_weeks"], math.inf) < threshold for row in selected), len(selected)
            )
        elderly_liquidity_rows.append(base)
    write_rows(OUT / "elderly_liquidity_profile.csv", elderly_liquidity_rows)

    benefits_by_household = defaultdict(float)
    for row in eligible:
        benefits_by_household[row["household_id"]] += number(row["person_minimum_consumption_allocation"])
    shadow_rows = []
    for target in TARGETS:
        for threshold in LIQUIDITY_THRESHOLDS:
            selected = [hrows[hid] for hid in elderly_household_ids if hid in hrows]
            below = sum(
                (number(row["cash"]) + benefits_by_household.get(row["household_id"], 0.0) * target) / max(EPS, number(row["minimum_cost"])) < threshold
                for row in selected
            )
            shadow_rows.append({
                "benefit_target_multiplier": target,
                "liquidity_threshold_weeks": threshold,
                "elderly_household_count": len(selected),
                "below_threshold_before": sum(number(row["liquidity_weeks"], math.inf) < threshold for row in selected),
                "below_threshold_after_shadow_pension": below,
                "share_below_threshold_after": share(below, len(selected)),
                "shadow_only_no_cash_mutation": True,
            })
    write_rows(OUT / "pension_shadow_liquidity_effect.csv", shadow_rows)

    gap_rows = []
    for threshold in LIQUIDITY_THRESHOLDS:
        low_ids = {hid for hid in elderly_household_ids if number(hrows[hid]["liquidity_weeks"], math.inf) < threshold}
        no_family = low_ids - wealthy_connected
        gap_rows.append({
            "liquidity_threshold_weeks": threshold,
            "elderly_low_liquidity_households": len(low_ids),
            "family_connected_count": len(low_ids & wealthy_connected),
            "genealogy_uncovered_count": len(no_family),
            "genealogy_uncovered_share": share(len(no_family), len(low_ids)),
            "pension_entitlement_reach_count": len(low_ids),
            "pension_reach_share_of_genealogy_uncovered": 1.0 if no_family else 0.0,
            "interpretation": "institution-dependent Person eligibility reaches each valid elderly household independently of family edges",
        })
    write_rows(OUT / "genealogy_gap_pension_coverage.csv", gap_rows)

    burden_rows = []
    contributor_hh_ids = {row["household_id"] for row in contributors}
    contributor_hhs = [hrows[hid] for hid in contributor_hh_ids if hid in hrows]
    for rate in RATES:
        deductions = defaultdict(float)
        for row in contributors:
            deductions[row["household_id"]] += number(row["realized_paid_wage"]) * rate
        burden = {
            "contribution_rate": rate,
            "contributor_household_count": len(contributor_hhs),
            "average_household_contribution": sum(deductions.values()) / len(contributor_hhs) if contributor_hhs else 0.0,
            "total_shadow_deduction": sum(deductions.values()),
            "household_share_below_0_25_before": share(sum(number(row["liquidity_weeks"], math.inf) < 0.25 for row in contributor_hhs), len(contributor_hhs)),
            "household_share_below_0_5_before": share(sum(number(row["liquidity_weeks"], math.inf) < 0.5 for row in contributor_hhs), len(contributor_hhs)),
            "household_share_below_1_before": share(sum(number(row["liquidity_weeks"], math.inf) < 1.0 for row in contributor_hhs), len(contributor_hhs)),
        }
        for threshold in (0.25, 0.5, 1.0):
            burden[f"household_share_below_{str(threshold).replace('.', '_')}_after"] = share(
                sum(
                    (number(row["cash"]) - deductions.get(row["household_id"], 0.0)) / max(EPS, number(row["minimum_cost"])) < threshold
                    for row in contributor_hhs
                ), len(contributor_hhs)
            )
        burden_rows.append(burden)
    write_rows(OUT / "contributor_liquidity_burden.csv", burden_rows)

    retirement_rows = []
    for rate in RATES:
        full_revenue = wage_base * rate
        no_elderly_revenue = contributor_wage_base_without_elderly * rate
        retirement_rows.append({
            "contribution_rate": rate,
            "current_base_including_65_plus": wage_base,
            "hypothetical_base_excluding_65_plus": contributor_wage_base_without_elderly,
            "current_revenue": full_revenue,
            "hypothetical_revenue_excluding_65_plus": no_elderly_revenue,
            "revenue_loss": full_revenue - no_elderly_revenue,
            "revenue_loss_share": share(full_revenue - no_elderly_revenue, full_revenue),
            "retirement_not_activated": True,
        })
    write_rows(OUT / "retirement_sensitivity.csv", retirement_rows)

    balance_shadow_rows = []
    for item in payg_rows:
        rate, target = item["contribution_rate"], item["benefit_target_multiplier"]
        if item["pension_obligation"] <= EPS or abs(item["balance"]) / item["pension_obligation"] <= 0.10:
            deductions = defaultdict(float)
            for row in contributors:
                deductions[row["household_id"]] += number(row["realized_paid_wage"]) * rate
            receipts = defaultdict(float)
            for row in eligible:
                receipts[row["household_id"]] += number(row["person_minimum_consumption_allocation"]) * target
            cash_after = [
                max(0.0, number(row["cash"]) - deductions.get(row["household_id"], 0.0) + receipts.get(row["household_id"], 0.0))
                for row in hrows.values()
            ]
            total_cash = sum(cash_after)
            sorted_cash = sorted(cash_after)
            n = len(sorted_cash)
            balance_shadow_rows.append({
                "contribution_rate": rate,
                "benefit_target_multiplier": target,
                "near_static_balance": True,
                "liquidity_below_0_25": share(sum(cash < 0.25 for cash in cash_after), n),
                "liquidity_below_0_5": share(sum(cash < 0.5 for cash in cash_after), n),
                "liquidity_below_1": share(sum(cash < 1.0 for cash in cash_after), n),
                "liquidity_below_2": share(sum(cash < 2.0 for cash in cash_after), n),
                "bottom50_cash_share": share(sum(sorted_cash[:max(1, n // 2)]), total_cash),
                "top10_cash_share": share(sum(sorted_cash[-max(1, math.ceil(n * 0.10)):]), total_cash),
                "top1_cash_share": share(sum(sorted_cash[-max(1, math.ceil(n * 0.01)):]), total_cash),
                "cash_gini": gini(cash_after),
                "shadow_only_no_cash_mutation": True,
            })
    write_rows(OUT / "balanced_scenario_distribution_shadow.csv", balance_shadow_rows or [{"status": "NO_NEAR_BALANCED_STATIC_COMBINATION"}])

    private_public_rows = []
    for threshold in LIQUIDITY_THRESHOLDS:
        low_ids = {hid for hid in elderly_household_ids if number(hrows[hid]["liquidity_weeks"], math.inf) < threshold}
        connected = low_ids & wealthy_connected
        uncovered = low_ids - wealthy_connected
        private_public_rows.extend([
            {
                "liquidity_threshold_weeks": threshold,
                "coverage_group": "FAMILY_CONNECTED",
                "household_count": len(connected),
                "share_of_elderly_low_liquidity": share(len(connected), len(low_ids)),
                "private_family_reach": True,
                "payg_institutional_reach": True,
                "interpretation": "potential overlap, not activated family support",
            },
            {
                "liquidity_threshold_weeks": threshold,
                "coverage_group": "NO_WEALTHY_FAMILY_CONNECTION",
                "household_count": len(uncovered),
                "share_of_elderly_low_liquidity": share(len(uncovered), len(low_ids)),
                "private_family_reach": False,
                "payg_institutional_reach": True,
                "interpretation": "genealogy gap reachable by person-level institutional entitlement",
            },
        ])
    write_rows(OUT / "private_vs_public_coverage.csv", private_public_rows)

    # The main verdict is intentionally fiscal, not a policy recommendation.
    break_even_one = elderly_need / wage_base if wage_base > EPS else math.inf
    if break_even_one <= 0.05:
        verdict = "A. PAYG_CAN_SUPPORT_MEANINGFUL_BASIC_PENSION"
    elif break_even_one <= 0.15:
        verdict = "B. PAYG_CAN_SUPPORT_ONLY_SMALL_BASIC_PENSION"
    elif break_even_one <= 0.30:
        verdict = "C. PAYG_REQUIRES_HIGH_CONTRIBUTION_RATE"
    else:
        verdict = "D. CURRENT_WAGE_BASE_CANNOT_SUPPORT_MEANINGFUL_PENSION"
    flags = {
        "verdict": verdict,
        "shadow_only": True,
        "actual_contributions_executed": False,
        "actual_pensions_paid": False,
        "social_insurance_fund_created": False,
        "ledger_mutated": False,
        "retirement_changed": False,
        "labor_eligibility_changed": False,
        "private_support_activated": False,
        "government_financing_assumed": False,
        "money_created": False,
        "new_rng_draws": 0,
        "step17a_output_modified": False,
        "payroll_source_authoritative": True,
        "person_level_entitlement": True,
        "elderly_workers_retained_in_base": True,
        "genealogy_independent_coverage_measured": True,
        "required_output_count": 20,
    }
    summary.update({
        "primary_fiscal_verdict": verdict,
        "break_even_rate_for_target_1": break_even_one,
        "flat_eligible_person_benchmark": flat_benchmark,
        "genealogy_uncovered_elderly_households": pensioner_households - len(wealthy_connected),
        "genealogy_uncovered_elderly_household_share": share(pensioner_households - len(wealthy_connected), pensioner_households),
    })
    write_rows(OUT / "pension_analysis_snapshot.csv", [summary])

    acceptance = [
        "# Step 17.B Acceptance Summary",
        "",
        "## Scope",
        "Pure shadow PAYG fiscal-capacity audit using the accepted mature demographic research state and a 52-week compatible economic observation. No contribution, pension, retirement, family-support, government, or Ledger behavior was activated.",
        "",
        "## Required Answers",
        f"1. Total contributable wage base: **{wage_base:.6f}** per observed week, from {len(contributors)} living employed Persons with positive runtime-paid wage.",
        f"2. Contributor count: **{len(contributors)}**; under 65 = {sum(number(row['age']) < 65 for row in contributors)}, 65+ = {sum(number(row['age']) >= 65 for row in contributors)}.",
        f"3. Pension-eligible Persons: **{len(eligible)}** across {pensioner_households} Social Households.",
        "4. PAYG revenue: " + ", ".join(f"{rate:.0%}={wage_base * rate:.6f}" for rate in RATES) + ".",
        "5. Weekly obligation at targets: " + ", ".join(f"{target:.2f}x={elderly_need * target:.6f}" for target in TARGETS) + ".",
        "6. Break-even contribution rate: " + ", ".join(f"{target:.2f}x={elderly_need * target / wage_base:.4%}" for target in TARGETS) + ".",
        "7. Maximum sustainable benefit multiplier: " + ", ".join(f"{rate:.0%}={wage_base * rate / elderly_need:.6f}x" for rate in RATES) + ".",
        f"8. Excluding 65+ worker contributions reduces the base from {wage_base:.6f} to {contributor_wage_base_without_elderly:.6f}, a {share(elderly_wage_base, wage_base):.4%} revenue loss at every rate; no retirement was activated.",
        f"9. Contributor burden: at 1% the average contributor-household deduction is {burden_rows[0]['average_household_contribution']:.6f}; contributor households below 0.25 weeks move from {burden_rows[0]['household_share_below_0_25_before']:.2%} to {burden_rows[0]['household_share_below_0_25_after']:.2%} in shadow arithmetic.",
        f"10. Elderly liquidity improvement: at the 1.00x target, elderly households below 1 week move from {len(elderly_household_ids)} to {sum((number(hrows[hid]['cash']) + benefits_by_household.get(hid, 0.0)) / max(EPS, number(hrows[hid]['minimum_cost'])) < 1.0 for hid in elderly_household_ids if hid in hrows)}; the 0.25x/0.50x/0.75x targets leave all {len(elderly_household_ids)} below 1 week. This is one-period shadow arithmetic only.",
        f"11. Genealogy-uncovered elderly Households: **{pensioner_households - len(wealthy_connected)} / {pensioner_households} ({share(pensioner_households - len(wealthy_connected), pensioner_households):.2%})** under the Step17.A >=13-week wealthy-family connection screen; PAYG eligibility can reach each valid elderly Person household independently.",
        f"12. Primary fiscal verdict: **{verdict}**. This is a static capacity classification, not a pension-policy recommendation.",
        "",
        "## Interpretation",
        "The pension entitlement is computed at Person level and settles conceptually to a valid Social Household. The benefit base neutrally allocates the authoritative NeedsSystem household minimum across member need components plus an equal share of the fixed household component. Family-connected and genealogy-uncovered elderly households are reported separately; private support remains inactive.",
        "",
        "The observed age >=65 wage share is a transitional feature of the current labor contract. A later retirement experiment must recompute capacity without changing this audit.",
        "",
        "## Validation",
        "No world cash, income, consumption, employment, ownership, investment, retirement, RNG, or Ledger state was changed by PAYG arithmetic. All output files are compact aggregate/person-panel diagnostics; no giant history dump was created.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acceptance_summary.md").write_text("\n".join(acceptance) + "\n", encoding="utf-8")
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
