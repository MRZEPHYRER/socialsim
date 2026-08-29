"""Default-off PAYG pension research runtime.

The system settles only real cash transfers through the World ledger. It never
creates a pension backstop or changes labor eligibility.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import math
from economy import config

EPSILON = 1e-12
ELDERLY_AGE = 65.0

@dataclass
class SocialInsuranceFund:
    fund_id: str = "social_insurance_fund"
    cash: float = 0.0
    contribution_inflow_this_step: float = 0.0
    employee_contribution_inflow_this_step: float = 0.0
    employer_contribution_inflow_this_step: float = 0.0
    public_pension_transfer_inflow_this_step: float = 0.0
    pension_outflow_this_step: float = 0.0
    contribution_history: list[float] = field(default_factory=list)
    pension_history: list[float] = field(default_factory=list)

    def reset_step(self):
        self.contribution_inflow_this_step = 0.0
        self.employee_contribution_inflow_this_step = 0.0
        self.employer_contribution_inflow_this_step = 0.0
        self.public_pension_transfer_inflow_this_step = 0.0
        self.pension_outflow_this_step = 0.0

@dataclass
class PaygPensionResult:
    step: int
    execution_status: str = "INACTIVE"
    scheduled_contribution: float = 0.0
    actual_contribution: float = 0.0
    contribution_shortfall: float = 0.0
    scheduled_employer_contribution: float = 0.0
    actual_employer_contribution: float = 0.0
    employer_contribution_shortfall: float = 0.0
    scheduled_public_pension_transfer: float = 0.0
    actual_public_pension_transfer: float = 0.0
    public_pension_transfer_shortfall: float = 0.0
    scheduled_pension: float = 0.0
    actual_pension: float = 0.0
    pension_funding_ratio: float = 1.0
    contributor_person_count: int = 0
    contributor_household_count: int = 0
    pensioner_person_count: int = 0
    pension_recipient_household_count: int = 0
    contribution_records: list[dict] = field(default_factory=list)
    employer_contribution_records: list[dict] = field(default_factory=list)
    pension_records: list[dict] = field(default_factory=list)

def _number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default

class PaygPensionSystem:
    """Employee-funded, cash-constrained PAYG research policy."""

    def __init__(self, world, fund=None):
        self.world = world
        self.fund = fund or SocialInsuranceFund()
        self.last_result = PaygPensionResult(step=0)
        self.history = []
    def pension_eligibility_age(self):
        """Single research-configurable eligibility boundary for PAYG."""
        return max(0.0, _number(getattr(self.world, "pension_eligibility_age", ELDERLY_AGE), ELDERLY_AGE))

    def is_pension_eligible(self, person):
        """Eligibility is distinct from retirement and labor participation."""
        return bool(
            getattr(person, "alive", False)
            and _number(getattr(person, "age", 0.0)) >= self.pension_eligibility_age()
        )

    def _social_household(self, person):
        household = self.world.get_household(getattr(person, "household_id", None))
        if household is not None and not getattr(household, "settlement_only", False):
            return household
        # Settlement-only accounts are valid cash recipients when the
        # accepted adult-settlement contract has created one. They are not
        # social households and must not be promoted or silently discarded.
        settlement = getattr(self.world, "settlement_household_for_person", lambda _person: None)(person)
        return settlement if settlement is not None else None

    def _person_need_components(self, household):
        members = []
        for person_id in [*getattr(household, "parents", []), *getattr(household, "children", [])]:
            person = self.world.get_person_by_id(person_id)
            if person is not None and getattr(person, "alive", False):
                members.append(person)
        if not members:
            return {}
        values = {}
        for person in members:
            age = _number(getattr(person, "age", 0.0))
            if age < 20:
                value = config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * config.NEW_CHILD_WEIGHT + config.CHILD_EXTRA_COST
            elif age < ELDERLY_AGE:
                value = config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * config.SECOND_ADULT_WEIGHT
            else:
                value = config.NEW_BASIC_CONSUMPTION_PER_EQUIVALENT_MEMBER * config.NEW_ELDERLY_WEIGHT + config.ELDERLY_EXTRA_COST
            values[person.id] = max(0.0, _number(value))
        fixed = config.NEW_BASE_CONSUMPTION / len(members)
        return {person_id: value + fixed for person_id, value in values.items()}

    def _payroll_records(self):
        records = {}
        provenance = getattr(self.world, "_household_payroll_provenance_weekly_state", {})
        for household_record in provenance.values():
            for item in household_record.get("person_components", []):
                person_id = item.get("person_id")
                paid = max(0.0, _number(item.get("paid_wage")))
                if person_id is not None:
                    records[person_id] = paid
        return records

    def _firm_payroll_records(self):
        """Return current actual payroll by Firm after the weekly wage run."""
        firms = getattr(self.world, "operating_firms", lambda: getattr(self.world, "firms", []))()
        records = []
        for firm in sorted(firms, key=lambda item: str(getattr(item, "firm_id", ""))):
            payroll = 0.0
            for attr in ("executed_wage_bill", "wage_payment", "wage_bill"):
                if hasattr(firm, attr):
                    payroll = max(0.0, _number(getattr(firm, attr)))
                    if payroll > EPSILON:
                        break
            records.append({
                "firm": firm,
                "firm_id": getattr(firm, "firm_id", getattr(firm, "id", "")),
                "sector_id": getattr(firm, "sector_id", "unknown"),
                "payroll": payroll,
            })
        return records

    def settle(self, step, defer_pension=False):
        self.fund.reset_step()
        result = PaygPensionResult(step=step)
        for household in getattr(self.world, "households", []):
            household.payg_contribution_this_step = 0.0
            household.pension_income_this_step = 0.0
        if not getattr(self.world, "payg_pension_enabled", False):
            self.last_result = result
            self.history.append(self.weekly_row(result))
            return result

        rate = max(0.0, min(1.0, _number(getattr(self.world, "payg_contribution_rate", config.PAYG_CONTRIBUTION_RATE))))
        target = max(0.0, _number(getattr(self.world, "payg_pension_target_multiplier", config.PAYG_PENSION_TARGET_MULTIPLIER)))
        wages = self._payroll_records()
        contribution_by_household = defaultdict(float)
        contributor_people = set()
        for person in sorted(getattr(self.world, "population", []), key=lambda item: item.id):
            wage = max(0.0, wages.get(person.id, 0.0))
            household = self._social_household(person)
            if wage <= EPSILON or household is None:
                continue
            if getattr(self.world, "payg_pre_retirement_contributor_only", False) and self.is_pension_eligible(person):
                continue
            amount = wage * rate
            contribution_by_household[household.id] += amount
            contributor_people.add(person.id)
            result.contribution_records.append({
                "person_id": person.id,
                "household_id": household.id,
                "realized_paid_wage": wage,
                "scheduled_contribution": amount,
            })

        result.scheduled_contribution = sum(contribution_by_household.values())
        actual_by_household = {}
        for household_id in sorted(contribution_by_household):
            household = self.world.get_household(household_id)
            scheduled = contribution_by_household[household_id]
            actual = min(scheduled, max(0.0, _number(getattr(household, "wealth", 0.0))))
            if actual > EPSILON:
                self.world.ledger.transfer_attrs(
                    payer_obj=household, payer_attr="wealth",
                    payer_name=f"household.{household.id}.wealth",
                    receiver_obj=self.fund, receiver_attr="cash",
                    receiver_name=self.fund.fund_id, amount=actual,
                    reason="payg_employee_contribution",
                )
            household.payg_contribution_this_step = actual
            actual_by_household[household_id] = actual
            result.actual_contribution += actual
        result.contribution_shortfall = max(0.0, result.scheduled_contribution - result.actual_contribution)

        # Employer contributions are explicit Firm-to-Fund transfers. The
        # default rate is zero, so existing PAYG behavior remains unchanged.
        employer_rate = max(0.0, min(1.0, _number(
            getattr(self.world, "payg_employer_contribution_rate", 0.0)
        )))
        for record in self._firm_payroll_records():
            firm = record["firm"]
            payroll = record["payroll"]
            scheduled = payroll * employer_rate
            available_cash = max(0.0, _number(getattr(firm, "cash", 0.0)))
            actual = min(scheduled, available_cash)
            if actual > EPSILON:
                self.world.ledger.transfer_attrs(
                    payer_obj=firm,
                    payer_attr="cash",
                    payer_name=f"firm.{record['firm_id']}.cash",
                    receiver_obj=self.fund,
                    receiver_attr="cash",
                    receiver_name=self.fund.fund_id,
                    amount=actual,
                    reason="payg_employer_contribution",
                )
            result.scheduled_employer_contribution += scheduled
            result.actual_employer_contribution += actual
            result.employer_contribution_records.append({
                "firm_id": record["firm_id"],
                "sector_id": record["sector_id"],
                "payroll": payroll,
                "scheduled_employer_contribution": scheduled,
                "actual_employer_contribution": actual,
                "employer_contribution_shortfall": max(0.0, scheduled - actual),
                "opening_cash_at_settlement": available_cash,
                "closing_cash_after_contribution": available_cash - actual,
            })
        result.employer_contribution_shortfall = max(
            0.0,
            result.scheduled_employer_contribution - result.actual_employer_contribution,
        )
        self.fund.employee_contribution_inflow_this_step = result.actual_contribution
        self.fund.employer_contribution_inflow_this_step = result.actual_employer_contribution
        self.fund.contribution_inflow_this_step = (
            result.actual_contribution + result.actual_employer_contribution
        )
        if hasattr(self.world, "record_payg_cash_safety_phase"):
            self.world.record_payg_cash_safety_phase("after_payg_contribution")

        if defer_pension:
            deferred_eligible = []
            for person in sorted(getattr(self.world, "population", []), key=lambda item: item.id):
                if not self.is_pension_eligible(person):
                    continue
                household = self._social_household(person)
                if household is None:
                    continue
                components = self._person_need_components(household)
                scheduled = max(0.0, _number(components.get(person.id, 0.0))) * target
                deferred_eligible.append((person, household, scheduled))
            result.scheduled_pension = sum(item[2] for item in deferred_eligible)
            result.pensioner_person_count = len(deferred_eligible)
            result.contributor_person_count = len(contributor_people)
            result.contributor_household_count = len(actual_by_household)
            result.execution_status = "ACTIVE"
            self._deferred_pension_result = result
            self.last_result = result
            return result

        eligible = []
        for person in sorted(getattr(self.world, "population", []), key=lambda item: item.id):
            if not self.is_pension_eligible(person):
                continue
            household = self._social_household(person)
            if household is None:
                continue
            components = self._person_need_components(household)
            scheduled = max(0.0, _number(components.get(person.id, 0.0))) * target
            eligible.append((person, household, scheduled))
        result.scheduled_pension = sum(item[2] for item in eligible)
        result.pensioner_person_count = len(eligible)
        result.pension_funding_ratio = (
            min(1.0, self.fund.cash / result.scheduled_pension)
            if result.scheduled_pension > EPSILON else 1.0
        )
        pension_by_household = defaultdict(float)
        for person, household, scheduled in eligible:
            actual = scheduled * result.pension_funding_ratio
            pension_by_household[household.id] += actual
            result.pension_records.append({
                "person_id": person.id, "household_id": household.id,
                "scheduled_pension": scheduled, "actual_pension": actual,
            })
        for household_id in sorted(pension_by_household):
            household = self.world.get_household(household_id)
            amount = pension_by_household[household_id]
            if amount > EPSILON:
                self.world.ledger.transfer_attrs(
                    payer_obj=self.fund, payer_attr="cash",
                    payer_name=self.fund.fund_id,
                    receiver_obj=household, receiver_attr="wealth",
                    receiver_name=f"household.{household.id}.wealth",
                    amount=amount, reason="payg_pension_payment",
                )
            household.pension_income_this_step += amount
            household.income_this_step += amount
            result.actual_pension += amount
        if hasattr(self.world, "record_payg_cash_safety_phase"):
            self.world.record_payg_cash_safety_phase("after_pension")
        result.contributor_person_count = len(contributor_people)
        result.contributor_household_count = len(actual_by_household)
        result.pension_recipient_household_count = len(pension_by_household)
        result.execution_status = "ACTIVE"
        self.fund.pension_outflow_this_step = result.actual_pension
        self.fund.contribution_history.append(result.actual_contribution)
        self.fund.pension_history.append(result.actual_pension)
        self.last_result = result
        self.history.append(self.weekly_row(result))
        return result

    def complete_deferred_public_pension(self, step):
        """Complete a Q-branch pension after tax has entered Government."""
        result = getattr(self, "_deferred_pension_result", None)
        if result is None:
            return None
        scheduled_residual = max(
            0.0,
            result.scheduled_pension
            - result.actual_contribution
            - result.actual_employer_contribution,
        )
        result.scheduled_public_pension_transfer = scheduled_residual
        transfer = self.world.transfer_public_pension(step, scheduled_residual)
        result.actual_public_pension_transfer = transfer["actual"]
        result.public_pension_transfer_shortfall = transfer["shortfall"]
        self.fund.public_pension_transfer_inflow_this_step = transfer["actual"]
        self.fund.contribution_inflow_this_step += transfer["actual"]
        result.pension_funding_ratio = (
            min(1.0, self.fund.cash / result.scheduled_pension)
            if result.scheduled_pension > EPSILON else 1.0
        )
        pension_by_household = defaultdict(float)
        for person in sorted(getattr(self.world, "population", []), key=lambda item: item.id):
            if not self.is_pension_eligible(person):
                continue
            household = self._social_household(person)
            if household is None:
                continue
            components = self._person_need_components(household)
            scheduled = max(0.0, _number(components.get(person.id, 0.0))) * max(
                0.0, _number(getattr(self.world, "payg_pension_target_multiplier", 0.0))
            )
            pension_by_household[household.id] += scheduled * result.pension_funding_ratio
            result.pension_records.append({
                "person_id": person.id, "household_id": household.id,
                "scheduled_pension": scheduled,
                "actual_pension": scheduled * result.pension_funding_ratio,
            })
        for household_id in sorted(pension_by_household):
            household = self.world.get_household(household_id)
            amount = pension_by_household[household_id]
            if amount > EPSILON:
                self.world.ledger.transfer_attrs(
                    payer_obj=self.fund, payer_attr="cash",
                    payer_name=self.fund.fund_id, receiver_obj=household,
                    receiver_attr="wealth", receiver_name=f"household.{household.id}.wealth",
                    amount=amount, reason="payg_pension_payment",
                )
            household.pension_income_this_step += amount
            household.income_this_step += amount
            result.actual_pension += amount
        result.pensioner_person_count = len(result.pension_records)
        result.pension_recipient_household_count = len(pension_by_household)
        result.execution_status = "ACTIVE"
        self.fund.pension_outflow_this_step = result.actual_pension
        self.fund.contribution_history.append(
            result.actual_contribution + result.actual_employer_contribution + result.actual_public_pension_transfer
        )
        self.fund.pension_history.append(result.actual_pension)
        self.last_result = result
        weekly = self.weekly_row(result)
        self.history.append(weekly)
        world_history = getattr(self.world, "payg_weekly_history", [])
        if world_history and world_history[-1].get("global_step") == step:
            world_history[-1].update(weekly)
        elif hasattr(self.world, "payg_weekly_history"):
            self.world.payg_weekly_history.append(weekly)
        self._deferred_pension_result = None
        return result
    def weekly_row(self, result):
        return {
            "global_step": result.step,
            "execution_status": result.execution_status,
            "scheduled_contribution": result.scheduled_contribution,
            "actual_contribution": result.actual_contribution,
            "contribution_shortfall": result.contribution_shortfall,
            "scheduled_employer_contribution": result.scheduled_employer_contribution,
            "actual_employer_contribution": result.actual_employer_contribution,
            "employer_contribution_shortfall": result.employer_contribution_shortfall,
            "scheduled_public_pension_transfer": result.scheduled_public_pension_transfer,
            "actual_public_pension_transfer": result.actual_public_pension_transfer,
            "public_pension_transfer_shortfall": result.public_pension_transfer_shortfall,
            "scheduled_pension": result.scheduled_pension,
            "actual_pension": result.actual_pension,
            "pension_funding_ratio": result.pension_funding_ratio,
            "contributor_person_count": result.contributor_person_count,
            "contributor_household_count": result.contributor_household_count,
            "pensioner_person_count": result.pensioner_person_count,
            "pension_recipient_household_count": result.pension_recipient_household_count,
            "fund_opening_cash": math.nan,
            "fund_closing_cash": self.fund.cash,
        }
    
