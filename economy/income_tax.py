"""Active Step17.S personal wage-tax withholding.

The module is intentionally narrow: only Person wage income is taxable in
this stage.  Pension, dividends, private support, inheritance, and other
transfers remain outside the active base.
"""
from __future__ import annotations

import math
from collections import defaultdict


EPSILON = 1e-12
TAX_YEAR_WEEKS = 52


def _number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def progressive_tax_liability(annual_income, brackets):
    """Return tax from true marginal segments, never a threshold cliff."""
    income = max(0.0, _number(annual_income))
    total = 0.0
    ordered = sorted((max(0.0, _number(lower)), max(0.0, min(1.0, _number(rate)))) for lower, rate in brackets)
    for index, (lower, rate) in enumerate(ordered):
        upper = ordered[index + 1][0] if index + 1 < len(ordered) else income
        if income > lower:
            total += max(0.0, min(income, upper) - lower) * rate
    return max(0.0, total)


def _ensure_person_tax_state(person, step):
    year = int(step) // TAX_YEAR_WEEKS
    if not hasattr(person, "tax_year_index"):
        person.tax_year_index = year
    if not hasattr(person, "cumulative_taxable_wage"):
        person.cumulative_taxable_wage = 0.0
    if not hasattr(person, "cumulative_tax_withheld"):
        person.cumulative_tax_withheld = 0.0
    if not hasattr(person, "estimated_or_accrued_annual_liability"):
        person.estimated_or_accrued_annual_liability = 0.0
    if not hasattr(person, "last_personal_tax_this_step"):
        person.last_personal_tax_this_step = 0.0
    if person.tax_year_index != year:
        person.tax_year_index = year
        person.cumulative_taxable_wage = 0.0
        person.cumulative_tax_withheld = 0.0
        person.estimated_or_accrued_annual_liability = 0.0
    person.last_personal_tax_this_step = 0.0


def _weekly_person_wages(world):
    wages = defaultdict(float)
    provenance = getattr(world, "_household_payroll_provenance_weekly_state", {})
    for record in provenance.values():
        for item in record.get("person_components", []):
            wages[item.get("person_id")] += max(0.0, _number(item.get("paid_wage")))
    return wages


def settle_personal_income_tax(world, step, brackets):
    """Withhold progressive wage tax through each Person's Social Household."""
    budget = world.public_budget
    wages = _weekly_person_wages(world)
    elapsed = int(step) % TAX_YEAR_WEEKS + 1
    scheduled_by_person = {}
    household_scheduled = defaultdict(float)
    records = []
    for person in sorted(getattr(world, "population", []), key=lambda item: item.id):
        if not getattr(person, "alive", False):
            continue
        _ensure_person_tax_state(person, step)
        person._personal_tax_brackets = list(brackets)
        weekly_wage = max(0.0, wages.get(person.id, 0.0))
        person.cumulative_taxable_wage += weekly_wage
        # Use year-to-date cumulative wages against the annual brackets.
        # This avoids charging a low-income Person on a temporary run-rate.
        estimated_annual = progressive_tax_liability(person.cumulative_taxable_wage, brackets)
        required_cumulative = estimated_annual
        scheduled = max(0.0, required_cumulative - person.cumulative_tax_withheld)
        person.estimated_or_accrued_annual_liability = estimated_annual
        household_id = getattr(person, "household_id", None)
        household = world.settlement_household_for_person(person)
        if household is None or weekly_wage <= EPSILON:
            scheduled = 0.0
        else:
            household_scheduled[household.id] += scheduled
        scheduled_by_person[person.id] = scheduled
        records.append({
            "global_step": int(step),
            "tax_type": "PERSONAL_INCOME_TAX",
            "person_id": person.id,
            "household_id": household_id,
            "tax_year_index": person.tax_year_index,
            "taxable_wage_this_step": weekly_wage,
            "cumulative_taxable_wage": person.cumulative_taxable_wage,
            "estimated_annual_liability": estimated_annual,
            "scheduled_tax": scheduled,
            "actual_tax": 0.0,
            "tax_shortfall": 0.0,
            "taxable_component": "wage_income",
        })

    actual_by_household = {}
    for household_id in sorted(household_scheduled):
        household = world.get_household(household_id)
        scheduled = household_scheduled[household_id]
        actual = min(scheduled, max(0.0, _number(getattr(household, "wealth", 0.0))))
        if actual > EPSILON:
            world.ledger.transfer_attrs(
                payer_obj=household,
                payer_attr="wealth",
                payer_name=f"household.{household.id}.wealth",
                receiver_obj=budget,
                receiver_attr="cash",
                receiver_name=budget.account_id,
                amount=actual,
                reason="personal_income_tax",
            )
        actual_by_household[household_id] = actual
        household.personal_income_tax_this_step = actual
        household.disposable_income_this_step = max(
            0.0, _number(getattr(household, "income_this_step", 0.0)) - actual
        )
        budget.scheduled_tax_this_step += scheduled
        budget.actual_tax_this_step += actual
        for person_record in records:
            if person_record["household_id"] == household_id:
                share = (person_record["scheduled_tax"] / scheduled) if scheduled > EPSILON else 0.0
                person_actual = actual * share
                person_record["actual_tax"] = person_actual
                person_record["tax_shortfall"] = max(0.0, person_record["scheduled_tax"] - person_actual)
                person = world.get_person_by_id(person_record["person_id"])
                if person is not None:
                    person.cumulative_tax_withheld += person_actual
                    person.last_personal_tax_this_step = person_actual
    for person_record in records:
        budget.tax_records.append(person_record)
    budget.personal_tax_records = records
    budget.personal_tax_scheduled_this_step = sum(x["scheduled_tax"] for x in records)
    budget.personal_tax_actual_this_step = sum(x["actual_tax"] for x in records)
    budget.personal_tax_shortfall_this_step = sum(x["tax_shortfall"] for x in records)
    return {
        "scheduled": budget.personal_tax_scheduled_this_step,
        "actual": budget.personal_tax_actual_this_step,
        "shortfall": budget.personal_tax_shortfall_this_step,
        "records": records,
        "household_actual": actual_by_household,
    }


def year_end_reconcile(world, step):
    """Close the just-completed tax year diagnostically, without refunds/debt."""
    rows = getattr(world, "personal_tax_year_reconciliation", None)
    if rows is None:
        world.personal_tax_year_reconciliation = []
        rows = world.personal_tax_year_reconciliation
    year = int(step) // TAX_YEAR_WEEKS
    for person in getattr(world, "population", []):
        if int(getattr(person, "tax_year_index", year)) != year:
            continue
        if (not getattr(person, "alive", False)
                and _number(getattr(person, "cumulative_taxable_wage", 0.0)) <= EPSILON
                and _number(getattr(person, "cumulative_tax_withheld", 0.0)) <= EPSILON):
            continue
            continue
        brackets = list(getattr(person, "_personal_tax_brackets", []))
        liability = progressive_tax_liability(
            _number(getattr(person, "cumulative_taxable_wage", 0.0)),
            brackets,
        ) if brackets else _number(getattr(person, "estimated_or_accrued_annual_liability", 0.0))
        withheld = _number(getattr(person, "cumulative_tax_withheld", 0.0))
        rows.append({
            "global_step": int(step),
            "tax_year_index": year,
            "person_id": person.id,
            "annual_taxable_wage": _number(getattr(person, "cumulative_taxable_wage", 0.0)),
            "final_annual_liability": liability,
            "cumulative_withholding": withheld,
            "diagnostic_balance": withheld - liability,
            "refund_or_additional_tax_not_settled": True,
        })
