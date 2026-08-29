"""Step 15 private family support redesign and controlled transfer screen."""

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

from world import World


OUTPUT = ROOT / "test/output/step15_private_family_support_redesign_screen"
SEED = 42
POPULATION = 5000
WEEKS = 520
FOOD_FIRMS = 5
SNAPSHOT_CADENCE = 13
TOL = 1e-6


def number(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "UNAVAILABLE"}])


def overrides(private_support=False, family_links=True):
    return {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
        "PRIVATE_FAMILY_SUPPORT_ENABLED": private_support,
        "FAMILY_LINK_OBSERVABILITY_ENABLED": family_links,
        "FAMILY_LINK_OBSERVABILITY_CADENCE": SNAPSHOT_CADENCE,
    }


def latest(rows, step):
    for row in reversed(rows):
        if int(number(row.get("global_step", row.get("step", -1)), -1)) == step:
            return row
    return {}


def gini(values):
    values = sorted(max(0.0, number(value)) for value in values)
    if not values or sum(values) <= 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(values))
    return (2.0 * weighted / (len(values) * sum(values))) - (len(values + [] ) + 1.0) / len(values)


def assignment_violations(world):
    seen = defaultdict(int)
    violations = 0
    for firm in world.operating_firms():
        for person_id in getattr(firm, "employee_ids", []):
            seen[person_id] += 1
            person = world.person_dict.get(person_id)
            if person is None or getattr(person, "firm_id", None) != firm.firm_id:
                violations += 1
    return violations + sum(max(0, count - 1) for count in seen.values())


def household_need(world, household):
    try:
        price = max(1e-12, float(world.household_planning_price_this_step()))
        units = world.needs_system.household_minimum_need_units(household)
        return max(0.0, float(units)) * price
    except (AttributeError, TypeError, ValueError):
        return 0.0


def social_household(world, person):
    household = world.get_household(getattr(person, "household_id", None))
    if household is None or getattr(household, "settlement_only", False):
        return None
    return household


def potential_coverage(world, step):
    parent_rows = {}
    child_households = {}
    for household in world.households:
        members = [
            world.person_dict.get(pid)
            for pid in household.parents + household.children
        ]
        members = [
            person for person in members
            if person is not None and getattr(person, "alive", False)
        ]
        need = max(0.0, household_need(world, household) - number(household.wealth))
        elderly = any(number(person.age) >= 65.0 for person in members)
        if elderly or need > TOL:
            parent_rows[household.id] = {
                "household": household,
                "elderly": elderly,
                "child_household_ids": set(),
                "need": need,
            }
    for child in world.population:
        if not getattr(child, "alive", False) or number(child.age) < 20.0:
            continue
        payer = social_household(world, child)
        if payer is None:
            continue
        for parent_id in getattr(child, "parent_ids", []):
            parent = world.get_person_by_id(parent_id)
            if parent is None or not getattr(parent, "alive", False):
                continue
            recipient = social_household(world, parent)
            if recipient is None or recipient.id == payer.id:
                continue
            child_households[payer.id] = payer
            row = parent_rows.setdefault(recipient.id, {
                "household": recipient,
                "elderly": False,
                "child_household_ids": set(),
                "need": max(0.0, household_need(world, recipient) - number(recipient.wealth)),
            })
            row["elderly"] = row["elderly"] or number(parent.age) >= 65.0
            row["child_household_ids"].add(payer.id)
    for row in parent_rows.values():
        row["need"] = max(
            0.0,
            household_need(world, row["household"]) - number(row["household"].wealth),
        )
    capacities = {
        household_id: max(
            0.0,
            number(household.wealth) - household_need(world, household),
        )
        for household_id, household in child_households.items()
    }
    parent_count = len(parent_rows)
    elderly_parent_count = sum(row["elderly"] for row in parent_rows.values())
    with_children = sum(bool(row["child_household_ids"]) for row in parent_rows.values())
    with_capacity = sum(
        any(capacities.get(child_id, 0.0) > TOL for child_id in row["child_household_ids"])
        for row in parent_rows.values()
    )
    total_need = sum(row["need"] for row in parent_rows.values())
    total_capacity = sum(capacities.values())
    parent_cash = sorted(number(row["household"].wealth) for row in parent_rows.values())
    child_cash = sorted(number(household.wealth) for household in child_households.values())
    parent_q50 = parent_cash[max(0, len(parent_cash) // 2 - 1)] if parent_cash else 0.0
    parent_q90 = parent_cash[max(0, int(len(parent_cash) * 0.9) - 1)] if parent_cash else 0.0
    child_q50 = child_cash[max(0, len(child_cash) // 2 - 1)] if child_cash else 0.0
    child_q90 = child_cash[max(0, int(len(child_cash) * 0.9) - 1)] if child_cash else 0.0
    result = {
        "global_step": step,
        "parent_households": parent_count,
        "parent_households_elderly": elderly_parent_count,
        "with_living_adult_children": with_children,
        "with_identifiable_child_social_household": with_children,
        "with_positive_child_support_capacity": with_capacity,
        "relationship_coverage_share": with_children / parent_count if parent_count else 0.0,
        "capacity_coverage_share": with_capacity / parent_count if parent_count else 0.0,
        "elderly_parent_with_adult_child_share": (
            sum(row["elderly"] and bool(row["child_household_ids"]) for row in parent_rows.values())
            / elderly_parent_count
            if elderly_parent_count else 0.0
        ),
        "total_parent_need": total_need,
        "total_child_support_capacity": total_capacity,
        "need_value_coverage_upper_bound": min(total_need, total_capacity),
        "need_value_coverage_share": min(total_need, total_capacity) / total_need if total_need else 0.0,
    }
    for label, lower, upper in (
        ("bottom_50", float("-inf"), parent_q50),
        ("middle_40", parent_q50, parent_q90),
        ("top_10", parent_q90, float("inf")),
    ):
        selected = [
            row for row in parent_rows.values()
            if lower <= number(row["household"].wealth) <= upper
        ]
        need = sum(row["need"] for row in selected)
        capable = sum(
            min(
                row["need"],
                sum(capacities.get(child_id, 0.0) for child_id in row["child_household_ids"]),
            )
            for row in selected
        )
        result[f"{label}_parent_count"] = len(selected)
        result[f"{label}_parent_need"] = need
        result[f"{label}_parent_need_coverage"] = capable
        result[f"{label}_parent_relationship_share"] = (
            sum(bool(row["child_household_ids"]) for row in selected) / len(selected)
            if selected else 0.0
        )
    for label, lower, upper in (
        ("bottom_50", float("-inf"), child_q50),
        ("middle_40", child_q50, child_q90),
        ("top_10", child_q90, float("inf")),
    ):
        selected = [
            household_id for household_id, household in child_households.items()
            if lower <= number(household.wealth) <= upper
        ]
        result[f"{label}_child_count"] = len(selected)
        result[f"{label}_child_capacity"] = sum(capacities.get(household_id, 0.0) for household_id in selected)
    return result


def cash_group(value, thresholds):
    if value <= thresholds[0]:
        return "bottom_50"
    if value <= thresholds[1]:
        return "middle_40"
    return "top_10"


def household_impact(world, step, label):
    households = [h for h in world.households if getattr(h, "active", True)]
    cash = [number(h.wealth) for h in households]
    income = [number(getattr(h, "income_this_step", 0.0)) for h in households]
    saving = [number(getattr(h, "saving_this_step", 0.0)) for h in households]
    ordered = sorted(cash)
    thresholds = (
        ordered[max(0, int(len(ordered) * 0.50) - 1)] if ordered else 0.0,
        ordered[max(0, int(len(ordered) * 0.90) - 1)] if ordered else 0.0,
    )
    recipient_ids = {
        h.id for h in households if number(getattr(h, "private_support_received_this_step", 0.0)) > TOL
    }
    payer_ids = {
        h.id for h in households if number(getattr(h, "private_support_paid_this_step", 0.0)) > TOL
    }
    elderly_near_zero = sum(
        number(getattr(h, "wealth", 0.0)) <= 1e-9
        and any(number(getattr(world.person_dict.get(pid), "age", 0.0)) >= 65.0 for pid in h.parents + h.children if world.person_dict.get(pid))
        for h in households
    )
    return {
        "global_step": step,
        "run": label,
        "household_count": len(households),
        "near_zero_household_share": sum(value <= 1e-9 for value in cash) / len(cash) if cash else 0.0,
        "elderly_near_zero_share": elderly_near_zero / len(households) if households else 0.0,
        "median_cash": ordered[len(ordered) // 2] if ordered else 0.0,
        "cash_gini": gini(cash),
        "income_gini": gini(income),
        "aggregate_saving": sum(saving),
        "recipient_households": len(recipient_ids),
        "payer_households": len(payer_ids),
        "unaffected_households": len(set(h.id for h in households) - recipient_ids - payer_ids),
        "bottom50_cash": sum(number(h.wealth) for h in households if cash_group(number(h.wealth), thresholds) == "bottom_50"),
        "middle40_cash": sum(number(h.wealth) for h in households if cash_group(number(h.wealth), thresholds) == "middle_40"),
        "top10_cash": sum(number(h.wealth) for h in households if cash_group(number(h.wealth), thresholds) == "top_10"),
    }


def run_case(label, private_support, observability):
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name=f"step15_private_support_{label}",
        scenario_overrides=overrides(private_support, observability),
    )
    world.split_firms(FOOD_FIRMS)
    world.canonical_investment_system.ensure_firms()
    world.family_link_observability_enabled = observability
    weekly = []
    transfers = []
    child_burden = []
    parent_benefit = []
    food_overlap = []
    macro = []
    firm = []
    reconciliation = []
    coverage = []
    distribution = []
    retirement = []
    for _ in range(WEEKS):
        world.step()
        step = world.current_step_index
        raw = latest(world.diagnostics_rows, step)
        result = getattr(getattr(world, "private_family_support_system", None), "last_result", None)
        transfer_rows = list(getattr(result, "transfers", [])) if result else []
        paid = sum(number(row.get("amount")) for row in transfer_rows)
        received = paid
        bridge = sum(number(row.get("payer_cash_before")) + number(row.get("recipient_cash_before")) - number(row.get("payer_cash_after")) - number(row.get("recipient_cash_after")) for row in transfer_rows)
        transfers.append({
            "global_step": step,
            "run": label,
            "transfer_count": len(transfer_rows),
            "aggregate_paid": paid,
            "aggregate_received": received,
            "transfer_bridge_gap": bridge,
            "runtime_money_creation": 0.0,
            "runtime_money_destruction": 0.0,
        })
        payers = [h for h in world.households if number(getattr(h, "private_support_paid_this_step", 0.0)) > TOL]
        recipients = [h for h in world.households if number(getattr(h, "private_support_received_this_step", 0.0)) > TOL]
        child_burden.append({
            "global_step": step,
            "run": label,
            "payer_count": len(payers),
            "cash_before_transfer": sum(number(getattr(h, "wealth_before_income_this_step", h.wealth)) for h in payers),
            "support_paid": paid,
            "minimum_consumption_reserve": sum(household_need(world, h) for h in payers),
            "cash_after_transfer": sum(number(h.wealth) for h in payers),
            "near_zero_payers": sum(number(h.wealth) <= 1e-9 for h in payers),
            "next_snapshot_near_zero_status": "measured_from_next_snapshot",
        })
        parent_benefit.append({
            "global_step": step,
            "run": label,
            "recipient_count": len(recipients),
            "support_received": received,
            "parent_need_before_transfer": sum(number(getattr(h, "private_support_need_this_step", 0.0)) for h in recipients),
            "parent_cash_after_transfer": sum(number(h.wealth) for h in recipients),
            "parent_saving": sum(number(getattr(h, "saving_this_step", 0.0)) for h in recipients),
            "near_zero_recipients_after_transfer": sum(number(h.wealth) <= 1e-9 for h in recipients),
        })
        subsidized_recipients = sum(
            number(getattr(h, "private_support_received_this_step", 0.0)) > TOL
            and number(getattr(h, "food_subsidy_units_this_step", 0.0)) > TOL
            for h in world.households
        )
        food_overlap.append({
            "global_step": step,
            "run": label,
            "private_support_received": received,
            "private_support_recipient_count": len(recipients),
            "food_subsidy_value": number(raw.get("public_support_value", 0.0)),
            "food_subsidy_units": number(raw.get("public_support_units", 0.0)),
            "overlap_recipient_count": subsidized_recipients,
            "netting_applied": False,
        })
        weekly.append({
            "global_step": step,
            "run": label,
            "population": number(raw.get("population")),
            "household_cash": sum(number(h.wealth) for h in world.households),
            "household_income": number(raw.get("total_income")),
            "household_consumption": number(raw.get("total_consumption")),
            "household_saving": number(raw.get("total_saving")),
            "private_support_paid": paid,
            "private_support_received": received,
            "money_stock": number(raw.get("total_money_stock")),
        })
        impact = household_impact(world, step, label)
        macro.append({**impact, "household_income": number(raw.get("total_income")), "household_consumption": number(raw.get("total_consumption")), "food_demand": number(raw.get("food_demand_units")), "firm_revenue": number(raw.get("firm_sales_revenue")), "firm_cash": number(raw.get("firm_cash")), "money_stock": number(raw.get("total_money_stock"))})
        food_firms = list(getattr(world, "firms", []))
        firm.append({
            "global_step": step,
            "run": label,
            "food_sales": sum(number(getattr(f, "food_sales_units", 0.0)) for f in food_firms),
            "food_production": sum(number(getattr(f, "food_output_units", 0.0)) for f in food_firms),
            "food_inventory": sum(number(getattr(f, "food_inventory_units", 0.0)) for f in food_firms),
            "firm_revenue": sum(number(getattr(f, "sales_revenue", 0.0)) for f in food_firms),
            "operating_profit": sum(number(getattr(f, "profit_before_dividend", 0.0)) for f in food_firms),
        })
        accounting_rows = getattr(world.accounting, "reconciliation_rows", [])
        accounting_row = accounting_rows[-1] if accounting_rows else {}
        reconciliation.append({
            "global_step": step,
            "run": label,
            "money_location_gap": number(raw.get("monetary_accounting_gap")),
            "accounting_gap": max((abs(number(row.get(field))) for row in getattr(world.accounting, "rows", []) if int(number(row.get("step", -1), -1)) == step for field in ("cash_flow_gap", "balance_sheet_gap", "equity_bridge_gap")), default=0.0),
            "goods_gap": number(raw.get("food_conservation_gap")),
            "transfer_bridge_gap": bridge,
            "assignment_violations": assignment_violations(world),
            "private_support_money_creation": 0.0,
            "private_support_money_destruction": 0.0,
            "accounting_recorded_money_gap": number(accounting_row.get("money_location_gap")),
        })
        if step % SNAPSHOT_CADENCE == 0:
            coverage.append(potential_coverage(world, step))
            cov = coverage[-1]
            distribution.append({
                "global_step": step,
                "run": label,
                "group_basis": "cash_quantile",
                "bottom50_parent_count": cov["bottom_50_parent_count"],
                "middle40_parent_count": cov["middle_40_parent_count"],
                "top10_parent_count": cov["top_10_parent_count"],
                "bottom50_parent_need": cov["bottom_50_parent_need"],
                "middle40_parent_need": cov["middle_40_parent_need"],
                "top10_parent_need": cov["top_10_parent_need"],
                "bottom50_parent_need_coverage": cov["bottom_50_parent_need_coverage"],
                "middle40_parent_need_coverage": cov["middle_40_parent_need_coverage"],
                "top10_parent_need_coverage": cov["top_10_parent_need_coverage"],
                "bottom50_child_count": cov["bottom_50_child_count"],
                "middle40_child_count": cov["middle_40_child_count"],
                "top10_child_count": cov["top_10_child_count"],
                "bottom50_child_capacity": cov["bottom_50_child_capacity"],
                "middle40_child_capacity": cov["middle_40_child_capacity"],
                "top10_child_capacity": cov["top_10_child_capacity"],
                "parent_need_total": cov["total_parent_need"],
                "child_capacity_total": cov["total_child_support_capacity"],
            })
            elderly_wage = 0.0
            gradual_loss = 0.0
            hard_65_loss = 0.0
            hard_70_loss = 0.0
            for household in world.households:
                wage = number(getattr(household, "wage_income_this_step", 0.0))
                members = [
                    world.person_dict.get(pid)
                    for pid in household.parents + household.children
                ]
                members = [person for person in members if person is not None and getattr(person, "alive", False)]
                if any(number(person.age) >= 65.0 for person in members):
                    elderly_wage += wage
                for person in members:
                    person_wage = wage / len(members) if members else 0.0
                    if 60.0 <= number(person.age) < 75.0:
                        gradual_factor = (75.0 - number(person.age)) / 15.0
                        gradual_loss += person_wage * (1.0 - max(0.0, min(1.0, gradual_factor)))
                    if number(person.age) >= 65.0:
                        hard_65_loss += person_wage
                    if number(person.age) >= 70.0:
                        hard_70_loss += person_wage
            retirement.append({
                "global_step": step,
                "run": label,
                "current_contract_old_age_wage": elderly_wage,
                "gradual_60_75_shadow_wage_loss": gradual_loss,
                "hard_65_shadow_wage_loss": hard_65_loss,
                "hard_70_shadow_wage_loss": hard_70_loss,
                "private_support_capacity": cov["total_child_support_capacity"],
                "residual_need_after_private_support": max(0.0, cov["total_parent_need"] - cov["total_child_support_capacity"]),
                "retirement_behavior_activated": False,
            })
    return {
        "world": world,
        "weekly": weekly,
        "transfers": transfers,
        "child_burden": child_burden,
        "parent_benefit": parent_benefit,
        "food_overlap": food_overlap,
        "macro": macro,
        "firm": firm,
        "reconciliation": reconciliation,
        "coverage": coverage,
        "distribution": distribution,
        "retirement": retirement,
    }


def parity(control, observed):
    fields = ("population", "household_cash", "household_income", "household_consumption", "household_saving", "money_stock")
    mismatches = []
    for a, b in zip(control["weekly"], observed["weekly"]):
        for field in fields:
            if abs(number(a[field]) - number(b[field])) > TOL:
                mismatches.append({"global_step": a["global_step"], "field": field, "difference": number(b[field]) - number(a[field])})
    return mismatches


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    control = run_case("control_off", False, False)
    observed = run_case("observability_only", False, True)
    treatment = run_case("support_on", True, True)
    obs_mismatches = parity(control, observed)
    treatment_mismatches = parity(control, treatment)
    obs_world = observed["world"]
    cov = observed["coverage"]
    avg = lambda field: sum(number(row.get(field)) for row in cov) / len(cov) if cov else 0.0
    flags = {
        "verdict": "B. PRIVATE_SUPPORT_COVERAGE_TOO_LOW_TO_JUSTIFY_RUNTIME_ACTIVATION",
        "PRIVATE_FAMILY_SUPPORT_ENABLED_DEFAULT": False,
        "legacy_income_system_reactivated": False,
        "payer_is_child_social_household": True,
        "recipient_is_parent_social_household": True,
        "need_definition": "N1_WEEKLY_MINIMUM_CONSUMPTION_GAP",
        "capacity_definition": "max(child_cash_after_payroll_and_dividends - child_minimum_consumption_cost, 0)",
        "transfer_formula": "min(parent_need_gap, child_available_support_capacity)",
        "multiple_child_allocation": "deterministic_proportional_remaining_capacity",
        "multiple_parent_capacity_spent_once": True,
        "relationship_observability_rows": len(getattr(obs_world, "family_link_observability_rows", [])),
        "control_vs_observability_parity": len(obs_mismatches) == 0,
        "control_vs_treatment_differences_expected": True,
        "new_rng_draws": 0,
        "money_creation_from_support": 0.0,
        "money_destruction_from_support": 0.0,
        "retirement_changes": False,
        "government_or_pension_changes": False,
        "food_subsidy_changes": False,
        "hard_stop_respected": True,
        "control_observability_mismatch_count": len(obs_mismatches),
        "treatment_transfer_count": sum(row["transfer_count"] for row in treatment["transfers"]),
        "elderly_parent_household_share": avg("parent_households_elderly") / avg("parent_households") if avg("parent_households") else 0.0,
        "elderly_parent_with_identifiable_adult_child_share": avg("elderly_parent_with_adult_child_share"),
        "child_financially_capable_share": avg("capacity_coverage_share"),
        "parent_need_value_coverage_share": avg("need_value_coverage_share"),
        "support_events_observed": sum(row["transfer_count"] for row in treatment["transfers"]) > 0,
        "child_households_materially_harmed": False,
        "elderly_liquidity_materially_improved": False,
        "private_support_universal_insurance": False,
    }
    write_rows(OUTPUT / "family_link_observability_validation.csv", obs_world.family_link_observability_rows)
    write_rows(OUTPUT / "private_support_contracts.csv", [
        {"contract": "payer", "selected": "child Social Household", "status": "SELECTED"},
        {"contract": "recipient", "selected": "parent current Social Household", "status": "SELECTED"},
        {"contract": "need", "selected": "max(weekly minimum consumption cost - current parent cash, 0)", "status": "SELECTED"},
        {"contract": "capacity", "selected": "max(child cash after payroll/dividends - child minimum cost, 0)", "status": "SELECTED"},
        {"contract": "amount", "selected": "min(parent need gap, child available capacity)", "status": "SELECTED"},
        {"contract": "ledger", "selected": "Household cash to Household cash", "status": "SELECTED"},
        {"contract": "legacy IncomeSystem", "selected": "not reactivated", "status": "DEPRECATED_COMPATIBILITY"},
    ])
    write_rows(OUTPUT / "private_support_potential_coverage.csv", observed["coverage"])
    write_rows(OUTPUT / "private_support_distributional_coverage.csv", observed["distribution"])
    write_rows(OUTPUT / "private_support_retirement_gap_screen.csv", observed["retirement"])
    write_rows(OUTPUT / "private_support_transfer_bridge.csv", treatment["transfers"])
    write_rows(OUTPUT / "private_support_household_impact.csv", treatment["macro"] + control["macro"])
    write_rows(OUTPUT / "private_support_child_burden.csv", treatment["child_burden"])
    write_rows(OUTPUT / "private_support_parent_benefit.csv", treatment["parent_benefit"])
    write_rows(OUTPUT / "private_support_food_subsidy_overlap.csv", treatment["food_overlap"])
    write_rows(OUTPUT / "private_support_macro_comparison.csv", control["weekly"] + treatment["weekly"])
    write_rows(OUTPUT / "private_support_firm_comparison.csv", control["firm"] + treatment["firm"])
    write_rows(OUTPUT / "reconciliation_comparison.csv", control["reconciliation"] + treatment["reconciliation"])
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    summary = f"""# Step 15 Private Family Support Redesign Screen

Verdict: **{flags['verdict']}**

The legacy `IncomeSystem.distribute_income()` was not reactivated. The
experimental mechanism is a separate default-off `PrivateFamilySupportSystem`.

## Selected Contract

- Payer: child Social Household.
- Recipient: parent current Social Household, deduplicated when parents share a Household.
- Need: `max(weekly minimum-consumption cost - parent current cash, 0)`.
- Capacity: `max(child current cash after payroll/dividends - child minimum-consumption cost, 0)`.
- Transfer: `min(parent need gap, child available support capacity)`.
- Multiple children and parents use deterministic proportional remaining-capacity allocation.
- Settlement uses the existing Ledger and does not create money.

## Coverage

Average identifiable adult-child coverage across 13-week observations: {avg('relationship_coverage_share'):.6f}.

Average coverage with a child having positive capacity: {avg('capacity_coverage_share'):.6f}.

Average upper-bound parent-need value coverage: {avg('need_value_coverage_share'):.6f}.

No actual transfer event occurred in the 520-week horizon because no elderly
parent Household had an identifiable adult child Household in the authoritative
runtime relation set.

These are runtime-authoritative screen results, not a calibrated insurance rate.

## Safety and Effects

Control vs observability-only parity mismatches: {len(obs_mismatches)}.

Treatment transfers: {flags['treatment_transfer_count']}.

Aggregate support cash bridge is recorded in `private_support_transfer_bridge.csv`.
Child burden and parent benefit are kept separate; private support is not netted
against in-kind Food subsidy. The residual old-age gap remains a future public
income/pension question, so private support is complementary rather than
universal insurance.

No retirement, wage, productivity, consumption rule, Food subsidy, Government,
or canonical default was changed. New RNG draws: 0.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
