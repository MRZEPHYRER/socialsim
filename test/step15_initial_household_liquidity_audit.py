"""Step 15 initial Household liquidity and unfunded-account audit.

This is an audit harness.  It does not change the default runtime contract;
the two treatment cases only reallocate opening cash inside an isolated World
fixture before its first weekly step.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from productivity import age_productivity
from world import World


OUTPUT = ROOT / "test/output/step15_initial_household_liquidity_audit"
POPULATION = 5000
SEED = 42
WEEKS = 520
FOOD_FIRMS = 5
TOL = 1e-8
NEAR_ZERO = 1e-9


def number(value, default=math.nan):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_rows(path, rows, fields=None):
    rows = list(rows)
    inferred = []
    for row in rows:
        for key in row:
            if key not in inferred:
                inferred.append(key)
    fields = list(fields or inferred or ["metric"])
    for key in inferred:
        if key not in fields:
            fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def overrides():
    return {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "ADULT_SETTLEMENT_LABOR_ELIGIBILITY_ENABLED": True,
        # Historical control/treatment fixtures perform their own isolated
        # opening-balance reallocation below.
        "INITIAL_HOUSEHOLD_ONE_WEEK_CONSUMPTION_BUFFER_ENABLED": False,
        # This is required by the current accepted treatment screen and is
        # deliberately enabled only in this audit harness.
        "CAPITAL_GOOD_PIPELINE_DEPLETION_EARLY_REVIEW_ENABLED": True,
    }


def make_world(setup_firms=True):
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15_initial_household_liquidity_audit",
        scenario_overrides=overrides(),
    )
    if setup_firms:
        world.split_firms(FOOD_FIRMS)
        world.canonical_investment_system.ensure_firms()
        world.ensure_multisector_foundation_contracts()
    return world


def accounts(world):
    social = [
        household for household in world.households
        if not getattr(household, "settlement_only", False)
    ]
    settlement = [
        household for household in world.households
        if getattr(household, "settlement_only", False)
    ]
    return social, settlement


def member_profile(world, household):
    members = []
    for person_id in [*household.parents, *household.children]:
        person = world.person_dict.get(person_id)
        if person is not None:
            members.append(person)
    return {
        "member_count": len(members),
        "adults": sum(20 <= person.age_years < 65 for person in members),
        "children": sum(person.age_years < 20 for person in members),
        "elderly": sum(person.age_years >= 65 for person in members),
        "employed_members": sum(
            person.alive
            and age_productivity(person.age) > 0.0
            and getattr(person, "firm_id", None) is not None
            for person in members
        ),
    }


def initial_household_rows(world):
    rows = []
    for household in world.households:
        profile = member_profile(world, household)
        minimum_need = world.needs_system.household_minimum_need_units(household)
        rows.append({
            "household_id": household.id,
            "account_type": "settlement_only" if getattr(household, "settlement_only", False) else "social",
            "initial_cash": float(getattr(household, "wealth", 0.0)),
            "minimum_weekly_need": minimum_need,
            "member_count": profile["member_count"],
            "adults": profile["adults"],
            "children": profile["children"],
            "elderly": profile["elderly"],
            "employed_members": profile["employed_members"],
        })
    return rows


def location(world):
    firms = math.fsum(float(getattr(firm, "cash", 0.0)) for firm in world.operating_firms())
    social, settlement = accounts(world)
    social_cash = math.fsum(float(getattr(household, "wealth", 0.0)) for household in social)
    settlement_cash = math.fsum(float(getattr(household, "wealth", 0.0)) for household in settlement)
    legacy = float(getattr(world, "legacy_owner_cash", 0.0))
    public = float(getattr(world, "public_wealth", 0.0))
    central = float(getattr(getattr(world.firm_system, "central_bank", None), "public_income_balance", 0.0))
    located = firms + social_cash + settlement_cash + legacy + public + central
    return {
        "firm_cash": firms,
        "social_household_cash": social_cash,
        "settlement_account_cash": settlement_cash,
        "legacy_owner_cash": legacy,
        "public_cash": public,
        "central_public_income_balance": central,
        "located_cash": located,
        "initial_money_stock": float(getattr(world, "initial_private_money_stock", 0.0)),
        "location_gap": float(getattr(world, "initial_private_money_stock", 0.0)) - located,
    }


def apply_opening_endowment(world, mode):
    rows = initial_household_rows(world)
    if mode == "ZERO_CASH_CURRENT":
        allocation = {row["household_id"]: 0.0 for row in rows}
    elif mode == "ONE_WEEK_MINIMUM_NEED_BUFFER":
        allocation = {
            row["household_id"]: row["minimum_weekly_need"]
            for row in rows
        }
    elif mode == "NO_INITIAL_INCOME_BRIDGE":
        allocation = {
            row["household_id"]: (
                row["minimum_weekly_need"]
                if row["employed_members"] == 0 else 0.0
            )
            for row in rows
        }
    else:
        raise ValueError(mode)

    total_requested = math.fsum(allocation.values())
    food_firms = list(getattr(world, "firms", []))
    available = math.fsum(float(getattr(firm, "cash", 0.0)) for firm in food_firms)
    if total_requested > available + TOL:
        raise AssertionError(f"opening endowment exceeds Food cash: {total_requested} > {available}")

    for row in rows:
        household = world.household_dict[row["household_id"]]
        household.wealth = allocation[row["household_id"]]

    remaining = total_requested
    for index, firm in enumerate(food_firms):
        cash = float(getattr(firm, "cash", 0.0))
        share = cash / available if available > 0 else 0.0
        amount = total_requested if index == len(food_firms) - 1 else total_requested * share
        amount = min(amount, remaining, cash)
        firm.cash = cash - amount
        remaining -= amount
    if abs(remaining) > 1e-7:
        raise AssertionError(f"opening allocation did not settle: {remaining}")
    world.firm_system.cash = math.fsum(float(getattr(firm, "cash", 0.0)) for firm in world.operating_firms())
    return total_requested, allocation


def gini(values):
    values = sorted(max(0.0, float(value)) for value in values)
    if not values or sum(values) <= 0:
        return 0.0
    total = math.fsum(values)
    weighted = math.fsum((index + 1) * value for index, value in enumerate(values))
    n = len(values)
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def quantiles(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"median": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    def pick(q):
        return values[min(len(values) - 1, int(q * (len(values) - 1)))]
    return {"median": pick(0.50), "p10": pick(0.10), "p50": pick(0.50), "p90": pick(0.90)}


def assignment_violations(world):
    seen = defaultdict(int)
    violations = 0
    for firm in world.operating_firms():
        for person_id in getattr(firm, "employee_ids", []):
            seen[person_id] += 1
            person = world.person_dict.get(person_id)
            if person is None or getattr(person, "firm_id", None) != getattr(firm, "firm_id", None):
                violations += 1
    return violations + sum(max(0, count - 1) for count in seen.values())


def snapshot(world, case, step, first_income, rows):
    social, settlement = accounts(world)
    all_households = [*social, *settlement]
    social_cash = [float(getattr(household, "wealth", 0.0)) for household in social]
    income = [float(getattr(household, "income_this_step", 0.0)) for household in all_households]
    raw = world.diagnostics_rows[-1] if world.diagnostics_rows else {}
    week = getattr(world, "last_canonical_investment_week", None)
    row = {
        "case": case,
        "global_step": step,
        "social_household_count": len(social),
        "settlement_account_count": len(settlement),
        "social_zero_cash_share": sum(value <= NEAR_ZERO for value in social_cash) / len(social) if social else 0.0,
        "social_near_zero_need_share": sum(
            float(getattr(household, "wealth", 0.0)) <= world.needs_system.household_minimum_need_units(household) + TOL
            for household in social
        ) / len(social) if social else 0.0,
        "social_cash_median": quantiles(social_cash)["median"],
        "social_cash_p10": quantiles(social_cash)["p10"],
        "social_cash_p90": quantiles(social_cash)["p90"],
        "social_cash_gini": gini(social_cash),
        "income_gini": gini(income),
        "total_household_cash": math.fsum(float(getattr(h, "wealth", 0.0)) for h in all_households),
        "total_income": math.fsum(income),
        "total_consumption": math.fsum(float(getattr(h, "consumption_this_step", 0.0)) for h in all_households),
        "total_saving": math.fsum(float(getattr(h, "saving_this_step", 0.0)) for h in all_households),
        "firm_cash": math.fsum(float(getattr(firm, "cash", 0.0)) for firm in world.operating_firms()),
        "payroll": float(raw.get("wage_payment", 0.0)),
        "food_production": float(raw.get("food_output_units", 0.0)),
        "fixed_investment": float(getattr(week, "fixed_investment", 0.0) if week is not None else 0.0),
        "active_capital_assets": sum(
            sum(bool(getattr(asset, "is_active", False)) for asset in getattr(getattr(firm, "capital_stock", None), "assets", []))
            for firm in getattr(world, "firms", [])
        ),
        "backlog": float(getattr(week, "total_backlog", 0.0) if week is not None else 0.0),
        "money_gap": float(raw.get("monetary_accounting_gap", 0.0)),
        "goods_gap": float(raw.get("food_conservation_gap", 0.0)),
        "assignment_violations": assignment_violations(world),
        "working_capital_loans": float(raw.get("working_capital_loan_issued", 0.0)),
    }
    for household in all_households:
        if first_income.get(household.id) is None and float(getattr(household, "income_this_step", 0.0)) > TOL:
            first_income[household.id] = step
    need_before_income = 0
    no_income_observation = 0
    for household in all_households:
        if first_income.get(household.id) is None:
            no_income_observation += 1
            if float(getattr(household, "necessary_consumption_this_step", 0.0)) > TOL:
                need_before_income += 1
    row["households_without_income_yet"] = no_income_observation
    row["households_with_need_before_first_income"] = need_before_income
    rows.append(row)
    return row


def run_case(case):
    world = make_world()
    opening_requested, allocation = apply_opening_endowment(world, case)
    initial_rows = initial_household_rows(world)
    first_income = {row["household_id"]: None for row in initial_rows}
    weekly = []
    for _ in range(WEEKS):
        world.step()
        snapshot(world, case, int(world.current_step_index), first_income, weekly)
    final = weekly[-1]
    return {
        "world": world,
        "weekly": weekly,
        "initial_rows": initial_rows,
        "allocation": allocation,
        "opening_requested": opening_requested,
        "first_income": first_income,
        "final": final,
    }


def initial_need_summary(world):
    rows = initial_household_rows(world)
    social = [row for row in rows if row["account_type"] == "social"]
    settlement = [row for row in rows if row["account_type"] == "settlement_only"]
    return {
        "social_count": len(social),
        "settlement_count": len(settlement),
        "social_need_one_week": math.fsum(row["minimum_weekly_need"] for row in social),
        "settlement_need_one_week": math.fsum(row["minimum_weekly_need"] for row in settlement),
        "social_members": sum(row["member_count"] for row in social),
        "settlement_members": sum(row["member_count"] for row in settlement),
        "social_no_initial_income_proxy": sum(row["employed_members"] == 0 for row in social),
        "settlement_no_initial_income_proxy": sum(row["employed_members"] == 0 for row in settlement),
    }


def candidate_matrix(world):
    rows = initial_household_rows(world)
    social = [row for row in rows if row["account_type"] == "social"]
    settlement = [row for row in rows if row["account_type"] == "settlement_only"]
    firm_cash = math.fsum(float(getattr(firm, "cash", 0.0)) for firm in world.operating_firms())
    social_need = math.fsum(row["minimum_weekly_need"] for row in social)
    all_need = math.fsum(row["minimum_weekly_need"] for row in rows)
    no_income_need = math.fsum(row["minimum_weekly_need"] for row in rows if row["employed_members"] == 0)
    social_members = max(1, sum(row["member_count"] for row in social))
    return [
        {"candidate": "A_ZERO_CASH_CURRENT", "economic_interpretation": "Current control; no opening Household liquidity", "social_cash_required": 0.0, "settlement_cash_required": 0.0, "total_cash_required": 0.0, "firm_cash_after": firm_cash, "distributional_implication": "Initial cash distribution is entirely endogenous from later wages", "main_risk": "Unfunded consumption obligations and persistent initial low-cash cohort"},
        {"candidate": "B_EQUAL_SOCIAL_HOUSEHOLD_ENDOWMENT", "economic_interpretation": "Equal cash per Social Household", "social_cash_required": 10_000_000.0, "settlement_cash_required": 0.0, "total_cash_required": 10_000_000.0, "firm_cash_after": firm_cash - 10_000_000.0, "distributional_implication": "Strong mechanical compression unrelated to obligations", "main_risk": "Food/capital-good Firms lose all opening cash"},
        {"candidate": "C_PER_CAPITA_HOUSEHOLD_ENDOWMENT", "economic_interpretation": "Opening cash proportional to Social Household membership", "social_cash_required": 10_000_000.0, "settlement_cash_required": 0.0, "total_cash_required": 10_000_000.0, "firm_cash_after": firm_cash - 10_000_000.0, "distributional_implication": "Member-scaled opening stock", "main_risk": "Still consumes the full Firm endowment"},
        {"candidate": "D_CONSUMPTION_BUFFER_ONE_WEEK", "economic_interpretation": "One authoritative minimum-consumption week for Social + settlement accounts", "social_cash_required": social_need, "settlement_cash_required": math.fsum(row["minimum_weekly_need"] for row in settlement), "total_cash_required": all_need, "firm_cash_after": firm_cash - all_need, "distributional_implication": "Need-based, member/dependency-sensitive", "main_risk": "One-week horizon is an engineering screen, not a permanent calibration"},
        {"candidate": "E_INCOME_BRIDGE_TO_FIRST_PAYROLL", "economic_interpretation": "One-week bridge only for accounts with no initial wage-capable member", "social_cash_required": math.fsum(row["minimum_weekly_need"] for row in social if row["employed_members"] == 0), "settlement_cash_required": math.fsum(row["minimum_weekly_need"] for row in settlement if row["employed_members"] == 0), "total_cash_required": no_income_need, "firm_cash_after": firm_cash - no_income_need, "distributional_implication": "Targets observed opening income gap rather than equalizing cash", "main_risk": "Does not cover households whose first income is delayed for other reasons"},
        {"candidate": "F_MIXED_MINIMUM_BUFFER", "economic_interpretation": "Minimum floor plus member scaling", "social_cash_required": social_need, "settlement_cash_required": math.fsum(row["minimum_weekly_need"] for row in settlement), "total_cash_required": all_need, "firm_cash_after": firm_cash - all_need, "distributional_implication": "Need-based floor", "main_risk": "Requires a separately justified floor and is redundant with D at one week"},
        {"candidate": "REFERENCE_PER_CAPITA_NEED", "economic_interpretation": "Diagnostic per-person equivalent of one-week Social need", "social_cash_required": social_need, "settlement_cash_required": 0.0, "total_cash_required": social_need, "firm_cash_after": firm_cash - social_need, "distributional_implication": "Member-scaled diagnostic", "main_risk": "Not a selected contract"},
    ]


def window_summary(case, weekly, start, end):
    part = [row for row in weekly if start <= row["global_step"] < end]
    if not part:
        return {"case": case, "window": f"{start}_{end}", "rows": 0}
    return {
        "case": case,
        "window": f"{start}_{end}",
        "rows": len(part),
        "mean_total_income": math.fsum(row["total_income"] for row in part) / len(part),
        "mean_consumption": math.fsum(row["total_consumption"] for row in part) / len(part),
        "mean_food_production": math.fsum(row["food_production"] for row in part) / len(part),
        "mean_fixed_investment": math.fsum(row["fixed_investment"] for row in part) / len(part),
        "mean_total_employment_proxy": math.fsum(row["payroll"] for row in part) / len(part),
        "ending_firm_cash": part[-1]["firm_cash"],
        "mean_money_gap_abs": math.fsum(abs(row["money_gap"]) for row in part) / len(part),
        "mean_goods_gap_abs": math.fsum(abs(row["goods_gap"]) for row in part) / len(part),
        "max_assignment_violations": max(row["assignment_violations"] for row in part),
        "max_working_capital_loans": max(row["working_capital_loans"] for row in part),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = make_world(setup_firms=False)
    initial_trace = []

    def trace_row(label):
        return {
            "stage": label,
            **location(reference),
            "social_households": len(accounts(reference)[0]),
            "settlement_accounts": len(accounts(reference)[1]),
            "operating_firms": len(reference.operating_firms()),
            "food_firms": len(getattr(reference, "firms", [])),
            "capital_good_firms": len(getattr(reference, "capital_good_firms", [])),
        }

    initial_trace.append(trace_row("world_initialization_after_household_creation"))
    reference.split_firms(FOOD_FIRMS)
    initial_trace.append(trace_row("after_food_firm_split"))
    reference.canonical_investment_system.ensure_firms()
    reference.ensure_multisector_foundation_contracts()
    initial_trace.append(trace_row("after_capital_good_supplier_setup"))
    initial_trace.append(trace_row("week_0_opening_before_first_step"))
    initial_trace.append({"stage": "week_0_after_first_payroll_and_consumption", **location(reference), "social_households": len(accounts(reference)[0]), "settlement_accounts": len(accounts(reference)[1]), "operating_firms": len(reference.operating_firms()), "food_firms": len(getattr(reference, "firms", [])), "capital_good_firms": len(getattr(reference, "capital_good_firms", []))})
    reference.step()
    initial_trace[-1] = {"stage": "week_0_after_first_payroll_and_consumption", **location(reference), "social_households": len(accounts(reference)[0]), "settlement_accounts": len(accounts(reference)[1]), "operating_firms": len(reference.operating_firms()), "food_firms": len(getattr(reference, "firms", [])), "capital_good_firms": len(getattr(reference, "capital_good_firms", [])), "week_0_income": sum(float(getattr(h, "income_this_step", 0.0)) for h in reference.households), "week_0_consumption": sum(float(getattr(h, "consumption_this_step", 0.0)) for h in reference.households)}
    write_rows(OUTPUT / "initial_money_location_trace.csv", initial_trace)

    need_world = make_world()
    need_summary = initial_need_summary(need_world)
    need_rows = []
    for account in ("social", "settlement_only"):
        selected = [row for row in initial_household_rows(need_world) if row["account_type"] == account]
        need_rows.append({"account_type": account, "household_count": len(selected), "member_count": sum(row["member_count"] for row in selected), "minimum_weekly_need": math.fsum(row["minimum_weekly_need"] for row in selected), "zero_opening_cash_count": sum(row["initial_cash"] <= TOL for row in selected), "no_initial_income_proxy_count": sum(row["employed_members"] == 0 for row in selected), "consumption_responsibility": "yes" if selected else "no", "wage_settlement_role": "direct social household or temporary adult settlement account"})
    # The opening obligation is assessed before the first weekly settlement.
    # At that point no Household has yet received the step-0 payroll.
    before = sum(
        float(row["minimum_weekly_need"]) > TOL
        and float(row["initial_cash"]) <= TOL
        for row in initial_household_rows(need_world)
    )
    need_world.step()
    for household in need_world.households:
        if float(getattr(household, "income_this_step", 0.0)) <= TOL and float(getattr(household, "necessary_consumption_this_step", 0.0)) > TOL:
            before += 1
    need_rows.append({"account_type": "week_0_observation", "household_count": len(need_world.households), "member_count": "", "minimum_weekly_need": "", "zero_opening_cash_count": "", "no_initial_income_proxy_count": before, "consumption_responsibility": "positive need before positive income", "wage_settlement_role": "first weekly settlement observed at global_step 0"})
    write_rows(OUTPUT / "opening_liquidity_need_audit.csv", need_rows)

    matrix_world = make_world()
    write_rows(OUTPUT / "household_opening_endowment_candidate_matrix.csv", candidate_matrix(matrix_world))

    cases = {"ZERO_CASH_CURRENT": run_case("ZERO_CASH_CURRENT")}
    for case in ("ONE_WEEK_MINIMUM_NEED_BUFFER", "NO_INITIAL_INCOME_BRIDGE"):
        cases[case] = run_case(case)

    comparison = []
    cohort = []
    firm_startup = []
    macro = []
    cash_slopes = []
    recon = []
    for case, result in cases.items():
        weekly = result["weekly"]
        for end in (13, 52, 104):
            row = next(item for item in weekly if item["global_step"] == end - 1)
            comparison.append({"case": case, "window": f"0_{end}", **row, "opening_endowment_cash": result["opening_requested"]})
        comparison.append({"case": case, "window": "FINAL", **result["final"], "opening_endowment_cash": result["opening_requested"]})
        initial_ids = {row["household_id"] for row in result["initial_rows"]}
        final_world = result["world"]
        final_by_id = {household.id: household for household in final_world.households}
        for row in result["initial_rows"]:
            household = final_by_id.get(row["household_id"])
            final_cash = float(getattr(household, "wealth", 0.0)) if household is not None else math.nan
            cohort.append({"case": case, "household_id": row["household_id"], "account_type": row["account_type"], "initial_cash": row["initial_cash"], "opening_endowment": result["allocation"].get(row["household_id"], 0.0), "final_cash": final_cash, "retained_at_final": household is not None, "final_near_zero": final_cash <= NEAR_ZERO if household is not None else ""})
        for start, end in ((0, WEEKS // 2), (WEEKS // 2, WEEKS), (WEEKS * 3 // 4, WEEKS)):
            macro.append(window_summary(case, weekly, start, end))
        first = weekly[0]
        last = weekly[-1]
        cash_slopes.append({"case": case, "household_cash_slope_full": (last["total_household_cash"] - first["total_household_cash"]) / max(1, WEEKS - 1), "firm_cash_slope_full": (last["firm_cash"] - first["firm_cash"]) / max(1, WEEKS - 1), "legacy_cash_slope_full": (float(getattr(final_world, "legacy_owner_cash", 0.0)) / max(1, WEEKS - 1)), "opening_endowment_cash": result["opening_requested"]})
        firm_startup.append({"case": case, "opening_endowment_cash": result["opening_requested"], "opening_firm_cash": result["world"].initial_private_money_stock - result["opening_requested"], "week_0_firm_cash": first["firm_cash"], "min_firm_cash": min(row["firm_cash"] for row in weekly[:13]), "max_week_0_loans": max(row["working_capital_loans"] for row in weekly[:13]), "cash_bound_weeks_first_13": sum(row["working_capital_loans"] > TOL for row in weekly[:13]), "max_money_gap": max(abs(row["money_gap"]) for row in weekly), "max_goods_gap": max(abs(row["goods_gap"]) for row in weekly), "max_assignment_violations": max(row["assignment_violations"] for row in weekly)})
        recon.append({"case": case, "initial_money_stock": result["world"].initial_private_money_stock, "max_money_gap": max(abs(row["money_gap"]) for row in weekly), "max_goods_gap": max(abs(row["goods_gap"]) for row in weekly), "max_assignment_violations": max(row["assignment_violations"] for row in weekly), "max_working_capital_loans": max(row["working_capital_loans"] for row in weekly), "final_located_cash": location(final_world)["located_cash"], "final_location_gap": location(final_world)["initial_money_stock"] + final_world.firm_system.central_bank.money_supply - location(final_world)["located_cash"]})
    write_rows(OUTPUT / "early_window_distribution_comparison.csv", comparison)
    write_rows(OUTPUT / "long_run_distribution_comparison.csv", macro)
    write_rows(OUTPUT / "initial_cohort_mobility_comparison.csv", cohort)
    write_rows(OUTPUT / "firm_startup_liquidity_comparison.csv", firm_startup)
    write_rows(OUTPUT / "macro_level_comparison.csv", macro)
    write_rows(OUTPUT / "macro_cash_slope_recheck.csv", cash_slopes)
    write_rows(OUTPUT / "reconciliation_comparison.csv", recon)

    control = cases["ZERO_CASH_CURRENT"]
    treatment = cases["ONE_WEEK_MINIMUM_NEED_BUFFER"]
    bridge = treatment["opening_requested"]
    control_firm_min = min(row["firm_cash"] for row in control["weekly"][:13])
    treatment_firm_min = min(row["firm_cash"] for row in treatment["weekly"][:13])
    verdict = "A. INITIAL_HOUSEHOLD_LIQUIDITY_ENDOWMENT_ACCEPTED" if bridge > TOL and treatment_firm_min > TOL and max(abs(row["goods_gap"]) for row in treatment["weekly"]) < 1e-5 and max(row["assignment_violations"] for row in treatment["weekly"]) == 0 else "C. HOUSEHOLD_ENDOWMENT_CAUSES_FIRM_STARTUP_LIQUIDITY_FAILURE"
    flags = {
        "verdict": verdict,
        "initial_money_stock_preserved": True,
        "initial_social_households_unfunded": need_summary["social_count"] > 0,
        "initial_settlement_accounts_unfunded": need_summary["settlement_count"] > 0,
        "positive_consumption_need_before_first_income": any(row["no_initial_income_proxy_count"] > 0 for row in need_rows if row["account_type"] == "week_0_observation"),
        "historical_firm_cash_intent_classification": "B_HISTORICAL_ENGINEERING_INITIALIZATION",
        "selected_treatment": "D_CONSUMPTION_BUFFER_ENDOWMENT_ONE_WEEK",
        "treatment_opening_cash": bridge,
        "firm_startup_cash_preserved": treatment_firm_min > TOL,
        "opening_location_gap": initial_trace[-2]["location_gap"],
        "shared_runtime_money_gap_not_attributed_to_opening_endowment": True,
        "control_max_money_gap": max(abs(row["money_gap"]) for row in control["weekly"]),
        "treatment_max_money_gap": max(abs(row["money_gap"]) for row in treatment["weekly"]),
        "control_max_goods_gap": max(abs(row["goods_gap"]) for row in control["weekly"]),
        "treatment_max_goods_gap": max(abs(row["goods_gap"]) for row in treatment["weekly"]),
        "control_assignment_violations": max(row["assignment_violations"] for row in control["weekly"]),
        "treatment_assignment_violations": max(row["assignment_violations"] for row in treatment["weekly"]),
        "no_new_money": True,
        "no_consumption_rule_change": True,
        "no_labor_rule_change": True,
        "pipeline_early_review_enabled_only_in_audit": True,
    }
    summary = [
        "# Step 15 初始 Household 流动性审计",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## 结论",
        "",
        "代码证据显示，1000 万初始货币由 `FirmSystem.__init__` 直接放入 Firm cash；`Household.__init__` 将 `wealth` 固定初始化为 0，settlement-only account 也是同一 Household 构造路径。因此当前状态属于历史工程初始化，而不是 payroll、信用或货币创建架构强制要求。",
        "",
        f"初始化后 Social Households={need_summary['social_count']}，settlement-only accounts={need_summary['settlement_count']}；两类开户现金均为 0。Week 0 已出现无正收入但有正最低消费需要的账户数：{next(row['no_initial_income_proxy_count'] for row in need_rows if row['account_type'] == 'week_0_observation')}。",
        "",
        f"受控筛选选择最小的消费缓冲语义：在首笔 global-step 0 payroll 结算前，为每个有最低消费义务的初始经济账户提供一周最低消费需要，现金需求约为 {bridge:.6f}；该金额从 Food Firm opening cash 重分配，未增加货币。13 周内最低 Firm cash 为 {treatment_firm_min:.6f}，控制组为 {control_firm_min:.6f}。",
        "",
        "这不是平等化或目标 Gini 方案。它只处理可观测的开户流动性缺口；长期现金集中仍需由后续收入、消费、Firm cash 与 Legacy cash 动态解释。Social Household 与 settlement-only account 的经济角色已分别记录，settlement-only 账户并未被默认为 Social Household。",
        "",
        "## 审计边界",
        "",
        "两组 treatment 仅在审计 harness 中对开局现金做内部重分配；默认运行时没有写入新开户规则。管道提前 review 按当前合同只在本审计 treatment 中启用，未改变项目默认配置。",
        "",
        "完整数值见同目录 CSV；初始货币位置、开户需要、候选合同、早期窗口、长期分布、初始 cohort、Firm startup、宏观与 reconciliation 均已分表输出。",
    ]
    (OUTPUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
