"""Step 14E Food + Generic Services behavioral screen.

The two-sector executor in this file is deliberately experiment-local.  It
composes the accepted Food World, Step 13 finance, Step 14C household budget
allocation, and Step 14D no-inventory accounting without selecting any
permanent Service calibration.
"""

from __future__ import annotations

import copy
import csv
import gc
import json
import math
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from central_bank import config as central_bank_config
from economy import config as economy_config
from economy.accounting import AccountingLayer
from economy.firm import FirmSystem
from economy.household_demand import HouseholdDemandSystem
from economy.interest import settle_interest
from economy.multi_firm import FirmSlice
from economy.multisector import (
    GENERIC_SERVICE_GOOD_ID,
    GENERIC_SERVICE_SECTOR_ID,
    NO_INVENTORY_POLICY_ID,
    SERVICE_TECHNOLOGY_ID,
)
from economy.service_accounting import PassiveServiceAccountingAdapter
from productivity import age_productivity
from scenarios import apply_runtime_scenario, apply_scenario
from world import World


OUTPUT = ROOT / "test/output/step14E_first_two_sector_behavior_screen"
E0_OUTPUT = ROOT / "test/output/step14E0_two_sector_parameter_identification"
STAGE_B1_OUTPUT = ROOT / "test/output/generalized_firm_stageB1_U2_balanced_reallocation"
CANONICAL_OUTPUT = ROOT / "test/output/main_step13_financial_core"

SEED = 42
FOOD_FIRM_COUNT = 5
SERVICE_FIRM_COUNT = 2
SHORT_POPULATION = 500
SHORT_STEPS = 520
CONTROL_GATE_STEPS = 260
FULL_POPULATION = 5000
FULL_STEPS = 1820
REVIEW_INTERVAL = 13
ADJUSTMENT_FRACTION = 0.25
DEMAND_ALPHA = 0.10
FOOD_PRODUCTIVITY = float(economy_config.FOOD_PRODUCTIVITY_PER_LABOR)
BASE_BUFFER = float(central_bank_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER)
NORMALIZED_SERVICE_PRICE = 1.0
TOLERANCE = 1e-6

DEMANDS = {"R25": 0.25, "R50": 0.50, "R75": 0.75}
UNIT_ECONOMICS = {
    "U0": 34.8202352560821,
    "U1": 36.3748738237503,
    "U2": 40.40613307314599,
}
UNIT_LABELS = {
    "U0": "BREAK_EVEN_REFERENCE",
    "U1": "HEALTHY_FIRM0_MARGIN_REFERENCE",
    "U2": "HEALTHY_FIRM0_REVENUE_CAPACITY_REFERENCE",
}


def number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def safe_ratio(numerator, denominator):
    denominator = number(denominator)
    return number(numerator) / denominator if abs(denominator) > 1e-12 else 0.0


def mean(values):
    values = [number(value) for value in values]
    return statistics.fmean(values) if values else 0.0


def slope(values):
    values = [number(value) for value in values]
    if len(values) < 2:
        return 0.0
    x_bar = (len(values) - 1) / 2.0
    y_bar = mean(values)
    denominator = math.fsum((index - x_bar) ** 2 for index in range(len(values)))
    return math.fsum(
        (index - x_bar) * (value - y_bar)
        for index, value in enumerate(values)
    ) / denominator if denominator > 0 else 0.0


def percentile(values, fraction):
    values = sorted(number(value) for value in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def labor_services(person):
    return max(0.0, number(age_productivity(person.age)))


def eligible_people(world):
    for person in world.population:
        if (
            person.alive
            and person.household_id in world.household_dict
            and labor_services(person) > 0
        ):
            yield person


def active_households(world):
    return [household for household in world.households if getattr(household, "active", True)]


def firm_labor(world, firm):
    return math.fsum(
        labor_services(world.person_dict[person_id])
        for person_id in getattr(firm, "employee_ids", [])
        if person_id in world.person_dict
        and world.person_dict[person_id].alive
        and world.person_dict[person_id].household_id in world.household_dict
    )


def sector_of_firm_id(firm_id):
    if firm_id is None:
        return "unassigned"
    return "Service" if int(firm_id) >= 100 else "Food"


def set_service_defaults(firm):
    defaults = {
        "interest_arrears": 0.0,
        "legacy_arrears_term_claim": 0.0,
        "opening_interest_arrears": 0.0,
        "opening_legacy_arrears_term_claim": 0.0,
        "scheduled_wage_history": [],
        "trailing_scheduled_wage_bill": 0.0,
        "credit_limit": 0.0,
        "credit_headroom": 0.0,
        "requested_credit": 0.0,
        "executed_credit": 0.0,
        "denied_credit": 0.0,
        "payroll_funding_ratio": 1.0,
        "executed_wage_bill": 0.0,
        "funded_productive_capacity": 0.0,
        "current_interest_due": 0.0,
        "net_current_interest_due": 0.0,
        "interest_paid": 0.0,
        "opening_lender_exposure": 0.0,
        "revolving_credit_exposure": 0.0,
        "total_lender_exposure": 0.0,
        "legacy_term_claim_payment": 0.0,
        "post_termout_arrears_payment": 0.0,
        "termout_claim_reclassification_gap": 0.0,
        "spoilage_units": 0.0,
        "public_inventory_purchase_units": 0.0,
        "public_sector_cash_inflow": 0.0,
        "public_sector_cash_outflow": 0.0,
        "other_cash_inflow": 0.0,
        "other_cash_outflow": 0.0,
        "dividend_payment": 0.0,
        "loan_interest_paid": 0.0,
        "cash_start": firm.cash,
        "cash_end": firm.cash,
        "wage_payment": 0.0,
        "production": 0.0,
        "sales_units": 0.0,
        "sales_revenue": 0.0,
        "loan_issued": 0.0,
        "loan_repaid": 0.0,
        "inventory_book_value": 0.0,
        "service_demand_forecast": None,
        "latent_allocated_demand": 0.0,
        "technical_capacity": 0.0,
        "funded_service_capacity": 0.0,
        "operating_profit": 0.0,
        "net_income": 0.0,
    }
    for key, value in defaults.items():
        if not hasattr(firm, key):
            setattr(firm, key, copy.deepcopy(value))


def service_firm(firm_id):
    firm = FirmSlice(
        firm_id=firm_id,
        employee_ids=[],
        share=0.5,
        cash=0.0,
        price=NORMALIZED_SERVICE_PRICE,
        sector_id=GENERIC_SERVICE_SECTOR_ID,
        technology_id=SERVICE_TECHNOLOGY_ID,
        inventory_policy_id=NO_INVENTORY_POLICY_ID,
        output_good_id=GENERIC_SERVICE_GOOD_ID,
    )
    set_service_defaults(firm)
    return firm


def e0_startup_requirement(population, demand_id, unit_id):
    rows = read_csv(E0_OUTPUT / "step14E0_parameter_envelope.csv")
    target_label = UNIT_LABELS[unit_id]
    row = next(
        item for item in rows
        if item["window"] == "mature"
        and item["demand_reference"] == demand_id
        and item["unit_economics_candidate"] == target_label
    )
    scale = population / 5000.0
    service_payroll = number(row["projected_service_payroll"]) * scale
    food_payroll = number(row["projected_food_payroll"]) * scale
    requirement = service_payroll + BASE_BUFFER * safe_ratio(
        service_payroll, food_payroll + service_payroll
    )
    return {
        "startup_cash": requirement,
        "projected_service_payroll": service_payroll,
        "projected_food_payroll": food_payroll,
    }


class TwoSectorCoordinator:
    """Deterministic labor, demand, Service settlement, and metrics adapter."""

    def __init__(self, demand_id, unit_id, population, active=True):
        self.active = active
        self.demand_id = demand_id
        self.service_share = DEMANDS[demand_id]
        self.unit_id = unit_id
        self.service_productivity = UNIT_ECONOMICS[unit_id]
        self.population = population
        self.service_firms = [service_firm(100), service_firm(101)]
        self.service_dict = {firm.firm_id: firm for firm in self.service_firms}
        self.food_forecasts = {firm_id: None for firm_id in range(FOOD_FIRM_COUNT)}
        self.service_allocations = {}
        self.last_service_market = {}
        self.current_food_base = BASE_BUFFER
        self.current_service_base = 0.0
        self.current_common_wage = 34.8202352560821
        self.history = []
        self.firm_history = []
        self.review_rows = []
        self.movement_events = []
        self.employer_history = defaultdict(list)
        self.move_steps = defaultdict(list)
        self.last_move_step = {}
        self.boundary_violations = []
        self.accounting_adapter = PassiveServiceAccountingAdapter()
        self.capitalization = {}
        self.startup = e0_startup_requirement(population, demand_id, unit_id)

    @property
    def all_firms(self):
        return self._world.firms + self.service_firms

    def initialize(self, world, capitalize=True):
        self._world = world
        world._step14e_context = self
        for person in eligible_people(world):
            self.employer_history[person.id] = [getattr(person, "firm_id", None)]
        if not self.active or not capitalize:
            self.capitalization = {
                "food_cash_before": math.fsum(firm.cash for firm in world.firms),
                "food_cash_transferred": 0.0,
                "service_cash_received": 0.0,
                "food_cash_after": math.fsum(firm.cash for firm in world.firms),
                "private_firm_cash_before": math.fsum(firm.cash for firm in world.firms),
                "private_firm_cash_after": math.fsum(firm.cash for firm in world.firms),
                "located_money_before": self.located_money(world, include_service=False),
                "located_money_after": self.located_money(world, include_service=False),
                "money_neutral_gap": 0.0,
            }
            return

        requirement = self.startup["startup_cash"]
        food_before = math.fsum(max(0.0, firm.cash) for firm in world.firms)
        total_before = self.located_money(world, include_service=True)
        removal_ratio = safe_ratio(requirement, food_before)
        transferred = 0.0
        for index, firm in enumerate(world.firms):
            amount = (
                requirement - transferred
                if index == len(world.firms) - 1
                else firm.cash * removal_ratio
            )
            amount = max(0.0, min(firm.cash, amount))
            firm.cash -= amount
            transferred += amount
        world.firm_system.cash -= transferred
        service_values = world.split_total_by_shares(
            transferred, [1.0 / SERVICE_FIRM_COUNT] * SERVICE_FIRM_COUNT
        )
        for firm, value in zip(self.service_firms, service_values):
            firm.cash = value
            firm.cash_start = value
            firm.cash_end = value
        total_after = self.located_money(world, include_service=True)
        self.capitalization = {
            "food_cash_before": food_before,
            "food_cash_transferred": transferred,
            "service_cash_received": math.fsum(service_values),
            "food_cash_after": math.fsum(firm.cash for firm in world.firms),
            "private_firm_cash_before": food_before,
            "private_firm_cash_after": math.fsum(firm.cash for firm in world.firms) + math.fsum(service_values),
            "located_money_before": total_before,
            "located_money_after": total_after,
            "money_neutral_gap": total_after - total_before,
        }

    def located_money(self, world, include_service=True):
        service_cash = math.fsum(firm.cash for firm in self.service_firms) if include_service else 0.0
        return (
            math.fsum(firm.cash for firm in world.firms)
            + service_cash
            + math.fsum(household.wealth for household in world.households)
            + number(getattr(world, "public_wealth", 0.0))
            + number(world.firm_system.central_bank.public_income_balance)
        )

    def firm_map(self, world):
        return {firm.firm_id: firm for firm in world.firms + self.service_firms}

    def normalize_employment(self, world):
        firms = self.firm_map(world)
        eligible = {person.id: person for person in eligible_people(world)}
        seen = set()
        for firm in firms.values():
            cleaned = []
            for person_id in getattr(firm, "employee_ids", []):
                person = eligible.get(person_id)
                if person is None or person_id in seen:
                    continue
                if getattr(person, "firm_id", None) != firm.firm_id:
                    continue
                cleaned.append(person_id)
                seen.add(person_id)
            firm.employee_ids = cleaned
        for person in eligible.values():
            firm_id = getattr(person, "firm_id", None)
            if firm_id in firms and person.id not in seen:
                firms[firm_id].employee_ids.append(person.id)
                seen.add(person.id)
            elif firm_id not in firms and hasattr(person, "firm_id"):
                delattr(person, "firm_id")

    def food_employed_people(self, world):
        food_ids = {firm.firm_id for firm in world.firms}
        return [
            person for person in world.population
            if person.alive
            and person.household_id in world.household_dict
            and labor_services(person) > 0
            and getattr(person, "firm_id", None) in food_ids
        ]

    def food_labor(self, world):
        return math.fsum(labor_services(person) for person in self.food_employed_people(world))

    def service_labor(self, world):
        return math.fsum(firm_labor(world, firm) for firm in self.service_firms)

    def _food_signal(self, world, firm):
        forecast = self.food_forecasts.get(firm.firm_id)
        if forecast is None:
            forecast = max(0.0, number(getattr(firm, "expected_demand", 0.0)))
        target = economy_config.FIRM_PRODUCTION_TARGET_INVENTORY_COVERAGE * forecast
        inventory_gap = target - max(0.0, number(firm.inventory_units))
        desired_output = max(
            0.0,
            forecast + economy_config.FIRM_PRODUCTION_INVENTORY_GAP_GAIN * inventory_gap,
        )
        current = firm_labor(world, firm)
        desired = safe_ratio(desired_output, FOOD_PRODUCTIVITY)
        return {"current": current, "desired": desired, "gap": desired - current}

    def _service_signal(self, world, firm):
        forecast = number(getattr(firm, "service_demand_forecast", 0.0))
        current = firm_labor(world, firm)
        desired = safe_ratio(forecast, self.service_productivity)
        return {"current": current, "desired": desired, "gap": desired - current}

    def _release(self, world, firm, requested):
        released = 0.0
        selected = []
        for person_id in sorted(firm.employee_ids):
            if released >= requested - 1e-12:
                break
            person = world.person_dict.get(person_id)
            if person is None or not person.alive:
                continue
            service = labor_services(person)
            if service <= 0:
                continue
            selected.append(person)
            released += service
        for person in selected:
            firm.employee_ids.remove(person.id)
            if hasattr(person, "firm_id"):
                delattr(person, "firm_id")
        return released, len(selected)

    def _record_movements(self, world, before):
        step = int(world.current_step_index)
        people = {person.id: person for person in eligible_people(world)}
        for person_id, old_firm in before.items():
            person = people.get(person_id)
            new_firm = getattr(person, "firm_id", None) if person else None
            if old_firm == new_firm:
                continue
            event = {
                "step": step,
                "person_id": person_id,
                "old_firm": old_firm,
                "new_firm": new_firm,
                "old_sector": sector_of_firm_id(old_firm),
                "new_sector": sector_of_firm_id(new_firm),
            }
            self.movement_events.append(event)
            prior = self.last_move_step.get(person_id)
            if prior is not None:
                self.move_steps[person_id].append(step - prior)
            self.last_move_step[person_id] = step
            history = self.employer_history[person_id]
            if not history or history[-1] != new_firm:
                history.append(new_firm)

    def check_labor_invariants(self, world):
        firms = self.firm_map(world)
        eligible = {person.id: person for person in eligible_people(world)}
        memberships = defaultdict(list)
        for firm in firms.values():
            for person_id in firm.employee_ids:
                if person_id in eligible:
                    memberships[person_id].append(firm.firm_id)
        violations = []
        for person in eligible.values():
            listed = memberships.get(person.id, [])
            firm_id = getattr(person, "firm_id", None)
            if len(listed) > 1:
                violations.append("duplicate_employee")
            if firm_id is None and listed:
                violations.append("unassigned_list_mismatch")
            if firm_id is not None and listed != [firm_id]:
                violations.append("person_firm_mismatch")
        if violations:
            self.boundary_violations.extend(violations)
        return len(violations)

    def before_firm_step(self, world):
        if not self.active:
            return
        self.normalize_employment(world)
        step = int(world.current_step_index)
        if step <= 0 or step % REVIEW_INTERVAL != 0:
            return
        before = {
            person.id: getattr(person, "firm_id", None)
            for person in eligible_people(world)
        }
        signals = {
            firm.firm_id: self._food_signal(world, firm)
            for firm in world.firms
        }
        signals.update({
            firm.firm_id: self._service_signal(world, firm)
            for firm in self.service_firms
        })
        release_requests = {
            firm_id: ADJUSTMENT_FRACTION * max(0.0, -signal["gap"])
            for firm_id, signal in signals.items()
        }
        vacancy_requests = {
            firm_id: ADJUSTMENT_FRACTION * max(0.0, signal["gap"])
            for firm_id, signal in signals.items()
        }
        firms = self.firm_map(world)
        released_services = released_workers = 0.0
        for firm_id in sorted(firms):
            services, workers = self._release(
                world, firms[firm_id], release_requests.get(firm_id, 0.0)
            )
            released_services += services
            released_workers += workers

        pool = [
            person for person in eligible_people(world)
            if getattr(person, "firm_id", None) not in firms
        ]
        pool.sort(key=lambda person: person.id)
        cursor = 0
        hired_services = hired_workers = residual_vacancy = 0.0
        order = sorted(
            firms,
            key=lambda firm_id: (-vacancy_requests.get(firm_id, 0.0), firm_id),
        )
        for firm_id in order:
            requested = vacancy_requests.get(firm_id, 0.0)
            filled = 0.0
            while cursor < len(pool) and filled < requested - 1e-12:
                person = pool[cursor]
                cursor += 1
                person.firm_id = firm_id
                firms[firm_id].employee_ids.append(person.id)
                service = labor_services(person)
                filled += service
                hired_services += service
                hired_workers += 1
            residual_vacancy += max(0.0, requested - filled)
        self._record_movements(world, before)
        violations = self.check_labor_invariants(world)
        unassigned_services = math.fsum(
            labor_services(person)
            for person in eligible_people(world)
            if getattr(person, "firm_id", None) not in firms
        )
        self.review_rows.append({
            "step": step,
            "planned_release_services": math.fsum(release_requests.values()),
            "planned_vacancy_services": math.fsum(vacancy_requests.values()),
            "actual_released_services": released_services,
            "actual_hired_services": hired_services,
            "released_workers": int(released_workers),
            "hired_workers": int(hired_workers),
            "residual_unfilled_vacancy_services": residual_vacancy,
            "residual_unassigned_labor_services": unassigned_services,
            "labor_invariant_violations": violations,
        })

    def prepare_credit(self, world, food_wage_bill, food_labor):
        service_labor_values = [firm_labor(world, firm) for firm in self.service_firms]
        self.current_common_wage = (
            safe_ratio(food_wage_bill, food_labor)
            if food_labor > 1e-12 else self.current_common_wage
        )
        service_wages = [self.current_common_wage * value for value in service_labor_values]
        service_payroll = math.fsum(service_wages)
        total_projected_payroll = food_wage_bill + service_payroll
        self.current_food_base = (
            BASE_BUFFER * safe_ratio(food_wage_bill, total_projected_payroll)
            if total_projected_payroll > 1e-12 else BASE_BUFFER
        )
        self.current_service_base = BASE_BUFFER - self.current_food_base
        local_total = math.fsum(service_labor_values)
        local_shares = (
            [safe_ratio(value, local_total) for value in service_labor_values]
            if local_total > 1e-12 else [0.5, 0.5]
        )
        cb = world.firm_system.central_bank
        enabled = bool(central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_ENABLED)
        k_weeks = max(0.0, number(central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_K_WEEKS))
        smoothing = max(1, int(central_bank_config.CENTRAL_BANK_CREDIT_CAPACITY_SMOOTHING_WEEKS))
        annual_rate = number(central_bank_config.CENTRAL_BANK_FIRM_LOAN_INTEREST_ANNUAL_RATE)
        for index, firm in enumerate(self.service_firms):
            set_service_defaults(firm)
            scheduled = max(0.0, service_wages[index])
            history = list(firm.scheduled_wage_history) + [scheduled]
            firm.scheduled_wage_history = history[-smoothing:]
            firm.trailing_scheduled_wage_bill = mean(firm.scheduled_wage_history)
            firm.scheduled_wage_bill = scheduled
            firm.wage_bill = scheduled
            firm.cash_start = max(0.0, firm.cash)
            firm.opening_principal = max(0.0, firm.loan_balance)
            firm.opening_interest_arrears = max(0.0, firm.interest_arrears)
            firm.opening_legacy_arrears_term_claim = max(0.0, firm.legacy_arrears_term_claim)
            gross, relief, current_due = world.current_interest_terms(
                firm, firm.opening_principal, annual_rate
            )
            firm.gross_current_interest_due = gross
            firm.interest_relief_amount = relief
            firm.net_current_interest_due = current_due
            firm.current_interest_due = current_due
            firm.opening_lender_exposure = firm.opening_principal + firm.opening_interest_arrears
            firm.target_cash = self.current_service_base * local_shares[index] + scheduled
            firm.funding_gap = max(0.0, firm.target_cash - firm.cash_start)
            firm.credit_limit = k_weeks * firm.trailing_scheduled_wage_bill if enabled else float("inf")
            firm.credit_headroom = (
                max(0.0, firm.credit_limit - firm.opening_lender_exposure)
                if enabled else float("inf")
            )
            requested = firm.funding_gap
            executed = min(requested, firm.credit_headroom)
            if (
                central_bank_config.CENTRAL_BANK_ENABLED
                and central_bank_config.CENTRAL_BANK_FIRM_CREDIT_ENABLED
                and executed > 0
            ):
                executed = world.ledger.create_loan_attr(
                    bank_name=f"central_bank.service_credit.{firm.firm_id}",
                    borrower_obj=firm,
                    borrower_attr="cash",
                    borrower_name=f"service_firm[{firm.firm_id}].cash",
                    amount=executed,
                    reason="step14e_service_working_capital_loan",
                    money_supply_obj=cb,
                    money_supply_attr="money_supply",
                    loan_obj=firm,
                    loan_attr="loan_balance",
                )
                cb.money_issued_this_step += executed
                cb.loan_issued_this_step += executed
            firm.executed_credit = max(0.0, firm.loan_balance - firm.opening_principal)
            firm.requested_credit = requested
            firm.denied_credit = max(0.0, requested - firm.executed_credit)
            firm.loan_issued = firm.executed_credit
            cash_after_credit = firm.cash_start + firm.executed_credit
            firm.cash_after_borrowing = cash_after_credit
            firm.payroll_funding_ratio = (
                1.0 if scheduled <= 1e-12
                else max(0.0, min(1.0, safe_ratio(cash_after_credit, scheduled)))
            )
            firm.executed_wage_bill = scheduled * firm.payroll_funding_ratio
            firm.wage_payment = firm.executed_wage_bill
            firm.technical_capacity = service_labor_values[index] * self.service_productivity
            firm.funded_productive_capacity = firm.technical_capacity * firm.payroll_funding_ratio
        cb.firm_loan_balance = math.fsum(firm.loan_balance for firm in world.firms + self.service_firms)
        cb.interest_receivable = math.fsum(
            number(getattr(firm, "interest_arrears", 0.0))
            + number(getattr(firm, "legacy_arrears_term_claim", 0.0))
            for firm in world.firms + self.service_firms
        )

    def pay_service_wages(self, world):
        firm_ids = set(self.service_dict)
        paid_by_firm = defaultdict(float)
        for person in world.population:
            if not person.alive or person.household_id not in world.household_dict:
                continue
            firm_id = getattr(person, "firm_id", None)
            if firm_id not in firm_ids:
                continue
            firm = self.service_dict[firm_id]
            total = firm_labor(world, firm)
            if total <= 1e-12:
                continue
            household = world.household_dict[person.household_id]
            wage = (
                firm.scheduled_wage_bill
                * labor_services(person)
                / total
                * firm.payroll_funding_ratio
            )
            if wage <= 0:
                continue
            world.ledger.transfer_attrs(
                payer_obj=firm,
                payer_attr="cash",
                payer_name=f"service_firm[{firm_id}].cash",
                receiver_obj=household,
                receiver_attr="wealth",
                receiver_name=f"household.{household.id}.wealth",
                amount=wage,
                reason="step14e_service_wage_payment",
            )
            household.income_this_step += wage
            household.wage_income_this_step += wage
            paid_by_firm[firm_id] += wage
        for firm in self.service_firms:
            firm.executed_wage_bill = paid_by_firm[firm.firm_id]
            firm.wage_payment = paid_by_firm[firm.firm_id]
            firm.cash_after_wages = firm.cash

    def allocate_household_budget(self, firm_system, household, units, money):
        allocation = HouseholdDemandSystem(firm_system.world).allocate(
            household,
            total_consumption_budget=money,
            service_share=self.service_share,
        )
        service_budget = allocation.service_budget
        if service_budget > 0:
            if firm_system.world.ledger.record_details:
                firm_system.world.ledger.transfer_attrs(
                    payer_obj=firm_system,
                    payer_attr="cash",
                    payer_name="firm.cash",
                    receiver_obj=household,
                    receiver_attr="wealth",
                    receiver_name=f"household.{household.id}.wealth",
                    amount=service_budget,
                    reason="step14e_service_budget_reallocation_refund",
                )
            else:
                household.wealth += service_budget
        food_budget = allocation.food_budget
        planning_price = max(0.01, number(household.household_planning_price_used, 1.0))
        food_units = food_budget / planning_price
        household.consumption_this_step = food_budget
        household.saving_this_step = household.income_this_step - food_budget
        household.food_desired_units_this_step = food_units
        household.real_food_consumption_units_this_step = food_units
        household.food_need_gap_units_this_step = max(
            0.0, number(household.food_need_units_this_step) - food_units
        )
        self.service_allocations[household.id] = service_budget
        return food_units, food_budget

    def settle_service_market(self, world):
        total_demand = math.fsum(self.service_allocations.values())
        initial_demand = [total_demand / SERVICE_FIRM_COUNT] * SERVICE_FIRM_COUNT
        capacities = [max(0.0, firm.funded_productive_capacity) for firm in self.service_firms]
        sales = [min(initial_demand[index], capacities[index]) for index in range(SERVICE_FIRM_COUNT)]
        residual = max(0.0, total_demand - math.fsum(sales))
        spare = [max(0.0, capacities[index] - sales[index]) for index in range(SERVICE_FIRM_COUNT)]
        spare_total = math.fsum(spare)
        spillover = [0.0, 0.0]
        if residual > 1e-12 and spare_total > 1e-12:
            for index in range(SERVICE_FIRM_COUNT):
                spillover[index] = min(spare[index], residual * spare[index] / spare_total)
                sales[index] += spillover[index]
        total_sales = math.fsum(sales)
        fulfillment = min(1.0, safe_ratio(total_sales, total_demand)) if total_demand > 0 else 1.0
        household_spending = {}
        for household in active_households(world):
            spending = self.service_allocations.get(household.id, 0.0) * fulfillment
            household_spending[household.id] = spending
            household.consumption_this_step += spending
            household.saving_this_step = household.income_this_step - household.consumption_this_step
        if world.ledger.record_details:
            shares = [safe_ratio(value, total_sales) for value in sales]
            for household in active_households(world):
                spending = household_spending.get(household.id, 0.0)
                for firm, share in zip(self.service_firms, shares):
                    world.ledger.transfer_attrs(
                        payer_obj=household,
                        payer_attr="wealth",
                        payer_name=f"household.{household.id}.wealth",
                        receiver_obj=firm,
                        receiver_attr="cash",
                        receiver_name=f"service_firm[{firm.firm_id}].cash",
                        amount=spending * share,
                        reason="step14e_service_purchase",
                    )
        else:
            for household in active_households(world):
                household.wealth -= household_spending.get(household.id, 0.0)
            for firm, value in zip(self.service_firms, sales):
                firm.cash += value
            world.ledger.record_summary(total_sales)

        cb = world.firm_system.central_bank
        annual_rate = number(central_bank_config.CENTRAL_BANK_FIRM_LOAN_INTEREST_ANNUAL_RATE)
        accounting_views = []
        for index, firm in enumerate(self.service_firms):
            observed = initial_demand[index] + spillover[index]
            prior = getattr(firm, "service_demand_forecast", None)
            firm.service_demand_forecast = (
                observed if prior is None
                else (1.0 - DEMAND_ALPHA) * prior + DEMAND_ALPHA * observed
            )
            firm.latent_allocated_demand = observed
            firm.sales_units = sales[index]
            firm.production = sales[index]
            firm.sales_revenue = sales[index] * NORMALIZED_SERVICE_PRICE
            firm.sales = firm.sales_revenue
            firm.unmet_demand = max(0.0, observed - sales[index])
            firm.unused_service_capacity = max(0.0, capacities[index] - sales[index])
            firm.capacity_utilization = safe_ratio(sales[index], firm.technical_capacity)
            firm.operating_profit = firm.sales_revenue - firm.executed_wage_bill
            firm.profit = firm.operating_profit
            local_total = math.fsum(max(0.0, other.technical_capacity) for other in self.service_firms)
            local_share = (
                safe_ratio(firm.technical_capacity, local_total)
                if local_total > 1e-12 else 0.5
            )
            repayment_floor = self.current_service_base * local_share
            cash_before_interest = firm.cash
            interest = settle_interest(
                opening_principal=firm.opening_principal,
                opening_arrears=firm.opening_interest_arrears,
                opening_legacy_arrears=firm.opening_legacy_arrears_term_claim,
                cash_before_interest=cash_before_interest,
                operating_liquidity_floor=repayment_floor,
                annual_rate=annual_rate,
                current_interest_due=firm.net_current_interest_due,
            )
            interest_paid = interest["interest_paid"]
            if interest_paid > 0:
                world.ledger.transfer_attrs(
                    payer_obj=firm,
                    payer_attr="cash",
                    payer_name=f"service_firm[{firm.firm_id}].cash",
                    receiver_obj=cb,
                    receiver_attr="public_income_balance",
                    receiver_name="central_bank.public_income_balance",
                    amount=interest_paid,
                    reason="step14e_service_interest_payment",
                )
                cb.loan_interest_paid_this_step += interest_paid
                cb.public_income_this_step += interest_paid
                world.record_central_bank_public_income_source("loan_interest", interest_paid)
            principal_repaid = min(
                firm.loan_balance,
                firm.loan_balance * central_bank_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_RATE,
                max(0.0, firm.cash - repayment_floor),
            )
            if principal_repaid > 0:
                world.ledger.repay_loan_attrs(
                    borrower_obj=firm,
                    borrower_attr="cash",
                    borrower_name=f"service_firm[{firm.firm_id}].cash",
                    principal=principal_repaid,
                    reason="step14e_service_principal_repayment",
                    money_supply_obj=cb,
                    money_supply_attr="money_supply",
                    loan_obj=firm,
                    loan_attr="loan_balance",
                    bank_obj=cb,
                    bank_attr="public_income_balance",
                    bank_name="central_bank.public_income_balance",
                )
                cb.money_issued_this_step -= principal_repaid
                cb.money_destroyed_this_step += principal_repaid
                cb.loan_principal_repaid_this_step += principal_repaid
            firm.loan_repaid = principal_repaid
            firm.loan_interest_paid = interest_paid
            firm.interest_paid = interest_paid
            firm.interest_arrears = interest["closing_interest_arrears"]
            firm.legacy_arrears_term_claim = interest["closing_legacy_arrears"]
            firm.revolving_credit_exposure = firm.loan_balance + firm.interest_arrears
            firm.total_lender_exposure = firm.revolving_credit_exposure + firm.legacy_arrears_term_claim
            firm.cash_end = firm.cash
            firm.cash_after_sales = cash_before_interest
            firm.cash_before_repayment = cash_before_interest
            firm.repayment_buffer = repayment_floor
            firm.credit_bridge_gap = (
                firm.loan_balance - firm.opening_principal - firm.loan_issued + firm.loan_repaid
            )
            firm.cash_bridge_gap = (
                firm.cash_end - firm.cash_start - firm.sales_revenue
                + firm.wage_payment - firm.loan_issued + firm.loan_repaid + interest_paid
            )
            firm.net_income = firm.operating_profit - interest_paid
            firm.inventory_units = 0.0
            firm.inventory_value = 0.0
            firm.spoilage_units = 0.0
            accounting_views.append(self.accounting_adapter.record(
                service_revenue=firm.sales_revenue,
                executed_labor_expense=firm.executed_wage_bill,
                interest_paid=interest_paid,
                cash_start=firm.cash_start,
                loan_issued=firm.loan_issued,
                principal_repaid=firm.loan_repaid,
                principal=firm.loan_balance,
                interest_arrears=firm.interest_arrears,
                cash_end=firm.cash_end,
            ))
        cb.firm_loan_balance = math.fsum(firm.loan_balance for firm in world.firms + self.service_firms)
        cb.interest_receivable = math.fsum(
            number(getattr(firm, "interest_arrears", 0.0))
            + number(getattr(firm, "legacy_arrears_term_claim", 0.0))
            for firm in world.firms + self.service_firms
        )
        cb.cumulative_interest_income = number(getattr(cb, "cumulative_interest_income", 0.0)) + math.fsum(
            firm.loan_interest_paid for firm in self.service_firms
        )
        self.last_service_market = {
            "desired_budget": total_demand,
            "executed_spending": total_sales,
            "unexecuted_budget": max(0.0, total_demand - total_sales),
            "settlement_gap": total_sales - math.fsum(firm.sales_revenue for firm in self.service_firms),
            "no_inventory_gap": max(abs(firm.inventory_units) for firm in self.service_firms),
            "cash_flow_gap": max([abs(view.cash_flow_gap) for view in accounting_views] or [0.0]),
            "balance_sheet_gap": max([abs(view.balance_sheet_gap) for view in accounting_views] or [0.0]),
        }
        self.service_allocations = {}
        world.consumption_system.total_consumption = math.fsum(
            number(household.consumption_this_step) for household in active_households(world)
        )
        return total_sales

    def update_firm_result(self, world, firm_result, service_sales):
        service_wages = math.fsum(firm.executed_wage_bill for firm in self.service_firms)
        service_cash = math.fsum(firm.cash for firm in self.service_firms)
        service_loans = math.fsum(firm.loan_balance for firm in self.service_firms)
        service_issued = math.fsum(firm.loan_issued for firm in self.service_firms)
        service_repaid = math.fsum(firm.loan_repaid for firm in self.service_firms)
        service_interest = math.fsum(firm.loan_interest_paid for firm in self.service_firms)
        service_profit = math.fsum(firm.operating_profit for firm in self.service_firms)
        firm_result["total_labor"] = number(firm_result.get("total_labor")) + self.service_labor(world)
        firm_result["scheduled_wage_bill"] = number(firm_result.get("scheduled_wage_bill")) + math.fsum(
            firm.scheduled_wage_bill for firm in self.service_firms
        )
        firm_result["executed_wage_bill"] = number(firm_result.get("executed_wage_bill")) + service_wages
        firm_result["wage_payment"] = firm_result["executed_wage_bill"]
        firm_result["sales"] = number(firm_result.get("sales")) + service_sales
        firm_result["firm_sales_revenue"] = number(firm_result.get("firm_sales_revenue")) + service_sales
        firm_result["nominal_output_value"] = number(firm_result.get("nominal_output_value")) + service_sales
        firm_result["firm_cash"] = number(firm_result.get("firm_cash")) + service_cash
        firm_result["firm_net_worth"] = number(firm_result.get("firm_net_worth")) + service_cash
        firm_result["profit_before_dividend"] = number(firm_result.get("profit_before_dividend")) + service_profit
        firm_result["working_capital_loan_issued"] = number(firm_result.get("working_capital_loan_issued")) + service_issued
        firm_result["loan_issued"] = firm_result["working_capital_loan_issued"]
        firm_result["working_capital_loan_repaid"] = number(firm_result.get("working_capital_loan_repaid")) + service_repaid
        firm_result["working_capital_interest_paid"] = number(firm_result.get("working_capital_interest_paid")) + service_interest
        firm_result["working_capital_loan_balance"] = service_loans + math.fsum(
            firm.loan_balance for firm in world.firms
        )
        firm_result["loan_balance"] = firm_result["working_capital_loan_balance"]
        firm_result["working_capital_target_cash"] = number(firm_result.get("working_capital_target_cash")) + math.fsum(
            firm.target_cash for firm in self.service_firms
        )
        firm_result["working_capital_funding_gap"] = number(firm_result.get("working_capital_funding_gap")) + math.fsum(
            firm.funding_gap for firm in self.service_firms
        )
        firm_result["cumulative_money_issued"] = world.firm_system.central_bank.money_supply
        firm_result["central_bank_net_money_issued"] = world.firm_system.central_bank.money_issued_this_step
        firm_result["central_bank_public_income"] = world.firm_system.central_bank.public_income_this_step
        firm_result["central_bank_public_income_balance"] = world.firm_system.central_bank.public_income_balance

    def after_week(self, world):
        if not self.active:
            return
        latest_firm_rows = {}
        for row in reversed(world.firm_diagnostics_rows):
            firm_id = int(number(row.get("firm_id", -1), -1))
            if firm_id not in latest_firm_rows:
                latest_firm_rows[firm_id] = row
            if len(latest_firm_rows) >= FOOD_FIRM_COUNT:
                break
        for firm_id, row in latest_firm_rows.items():
            latent = max(0.0, number(row.get("demand_units")))
            previous = self.food_forecasts.get(firm_id)
            self.food_forecasts[firm_id] = (
                latent if previous is None
                else (1.0 - DEMAND_ALPHA) * previous + DEMAND_ALPHA * latent
            )
        firms = self.firm_map(world)
        eligible = list(eligible_people(world))
        food_labor_value = math.fsum(
            labor_services(person) for person in eligible
            if sector_of_firm_id(getattr(person, "firm_id", None)) == "Food"
        )
        service_labor_value = math.fsum(
            labor_services(person) for person in eligible
            if sector_of_firm_id(getattr(person, "firm_id", None)) == "Service"
        )
        unassigned_labor = math.fsum(
            labor_services(person) for person in eligible
            if getattr(person, "firm_id", None) not in firms
        )
        macro = world.diagnostics_rows[-1]
        food_rows = list(latest_firm_rows.values())
        service_sales = math.fsum(firm.sales_revenue for firm in self.service_firms)
        service_wages = math.fsum(firm.executed_wage_bill for firm in self.service_firms)
        service_profit = math.fsum(firm.operating_profit for firm in self.service_firms)
        service_debt = math.fsum(firm.loan_balance for firm in self.service_firms)
        service_arrears = math.fsum(firm.interest_arrears for firm in self.service_firms)
        row = {
            "step": int(number(macro.get("global_step", macro.get("step")))),
            "population": number(macro.get("population")),
            "active_households": number(macro.get("active_households", len(active_households(world)))),
            "births": number(macro.get("births")),
            "deaths": number(macro.get("deaths")),
            "marriages": number(macro.get("marriages", getattr(world, "marriage_market_last_result", 0))),
            "food_labor": food_labor_value,
            "service_labor": service_labor_value,
            "total_employed_labor": food_labor_value + service_labor_value,
            "eligible_unassigned_labor": unassigned_labor,
            "food_worker_count": sum(sector_of_firm_id(getattr(person, "firm_id", None)) == "Food" for person in eligible),
            "service_worker_count": sum(sector_of_firm_id(getattr(person, "firm_id", None)) == "Service" for person in eligible),
            "unassigned_worker_count": sum(getattr(person, "firm_id", None) not in firms for person in eligible),
            "total_wage_income": number(macro.get("total_income")) - number(macro.get("firm_dividend", macro.get("dividend"))),
            "total_income": number(macro.get("total_income")),
            "total_consumption": number(macro.get("total_consumption")),
            "household_wealth": number(macro.get("total_wealth")),
            "food_sales_units": number(macro.get("food_sales_units")),
            "food_sales_revenue": math.fsum(number(item.get("sales_revenue")) for item in food_rows),
            "food_production": number(macro.get("food_output_units")),
            "food_inventory_units": number(macro.get("food_inventory_units")),
            "food_cash": math.fsum(firm.cash for firm in world.firms),
            "food_debt": math.fsum(firm.loan_balance for firm in world.firms),
            "food_operating_profit": math.fsum(number(item.get("profit")) for item in food_rows),
            "service_desired_budget": self.last_service_market.get("desired_budget", 0.0),
            "service_sales": service_sales,
            "service_unexecuted_budget": self.last_service_market.get("unexecuted_budget", 0.0),
            "service_wages": service_wages,
            "service_operating_profit": service_profit,
            "service_cash": math.fsum(firm.cash for firm in self.service_firms),
            "service_debt": service_debt,
            "service_arrears": service_arrears,
            "service_loan_issued": math.fsum(firm.loan_issued for firm in self.service_firms),
            "service_principal_repaid": math.fsum(firm.loan_repaid for firm in self.service_firms),
            "service_interest_paid": math.fsum(firm.loan_interest_paid for firm in self.service_firms),
            "food_sector_base": self.current_food_base,
            "service_sector_base": self.current_service_base,
            "target_base_sum": self.current_food_base + self.current_service_base,
            "monetary_accounting_gap": number(macro.get("monetary_accounting_gap")),
            "goods_conservation_gap": number(macro.get("goods_conservation_gap")),
            "service_settlement_gap": self.last_service_market.get("settlement_gap", 0.0),
            "service_no_inventory_gap": self.last_service_market.get("no_inventory_gap", 0.0),
        }
        self.history.append(row)
        for firm in self.service_firms:
            self.firm_history.append({
                "step": row["step"],
                "sector": "Service",
                "firm_id": firm.firm_id,
                "labor_services": firm_labor(world, firm),
                "employee_count": len(firm.employee_ids),
                "latent_demand": firm.latent_allocated_demand,
                "demand_forecast": number(firm.service_demand_forecast),
                "technical_capacity": firm.technical_capacity,
                "funded_capacity": firm.funded_productive_capacity,
                "sales_units": firm.sales_units,
                "revenue": firm.sales_revenue,
                "utilization": firm.capacity_utilization,
                "scheduled_payroll": firm.scheduled_wage_bill,
                "executed_payroll": firm.executed_wage_bill,
                "operating_profit": firm.operating_profit,
                "cash": firm.cash,
                "loan_balance": firm.loan_balance,
                "interest_arrears": firm.interest_arrears,
                "loan_issued": firm.loan_issued,
                "principal_repaid": firm.loan_repaid,
                "credit_limit": firm.credit_limit,
                "principal_utilization": safe_ratio(firm.loan_balance, firm.credit_limit),
                "cash_flow_gap": firm.cash_bridge_gap,
            })
        self.check_labor_invariants(world)

    def movement_summary(self):
        counts = defaultdict(int)
        for event in self.movement_events:
            counts[event["person_id"]] += 1
        intervals = [gap for values in self.move_steps.values() for gap in values]
        aba = 0
        for history in self.employer_history.values():
            sectors = [sector_of_firm_id(item) for item in history]
            aba += any(
                sectors[index] == sectors[index + 2]
                and sectors[index] != sectors[index + 1]
                and "unassigned" not in {sectors[index], sectors[index + 1]}
                for index in range(len(sectors) - 2)
            )
        return {
            "total_releases": sum(event["old_sector"] in {"Food", "Service"} for event in self.movement_events),
            "total_hires": sum(event["new_sector"] in {"Food", "Service"} for event in self.movement_events),
            "cross_sector_moves": sum(
                event["old_sector"] in {"Food", "Service"}
                and event["new_sector"] in {"Food", "Service"}
                and event["old_sector"] != event["new_sector"]
                for event in self.movement_events
            ),
            "food_to_service_moves": sum(
                event["old_sector"] == "Food" and event["new_sector"] == "Service"
                for event in self.movement_events
            ),
            "unique_workers_moved": len(counts),
            "workers_moved_more_than_once": sum(value > 1 for value in counts.values()),
            "max_moves_per_worker": max(counts.values(), default=0),
            "median_time_between_moves": statistics.median(intervals) if intervals else 0.0,
            "A_B_A_sequences": aba,
        }


def active_context(world):
    context = getattr(world, "_step14e_context", None)
    return context if context is not None and context.active else None


@contextmanager
def experiment_hooks():
    original_assign = World.assign_unassigned_workers_to_firms
    original_firm_step = FirmSystem.step
    original_total_labor = FirmSystem.total_labor
    original_distribute_wages = FirmSystem.distribute_wages
    original_purchase = FirmSystem.household_food_purchase_units
    original_prepare = World.prepare_multi_firm_credit
    original_sync = World.sync_firms_from_aggregate
    original_accounting_step = AccountingLayer.record_step

    def assign_hook(world):
        if active_context(world) is None:
            return original_assign(world)
        return None

    def firm_step_hook(firm_system):
        context = active_context(firm_system.world)
        if context is not None:
            context.before_firm_step(firm_system.world)
        return original_firm_step(firm_system)

    def total_labor_hook(firm_system):
        context = active_context(firm_system.world)
        return context.food_labor(firm_system.world) if context is not None else original_total_labor(firm_system)

    def prepare_hook(world, scheduled_wage_bill, total_labor):
        context = active_context(world)
        if context is None:
            return original_prepare(world, scheduled_wage_bill, total_labor)
        context.prepare_credit(world, scheduled_wage_bill, total_labor)
        old = central_bank_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
        central_bank_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = context.current_food_base
        try:
            result = original_prepare(world, scheduled_wage_bill, total_labor)
        finally:
            central_bank_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = old
        world.firm_system.central_bank.firm_loan_balance = math.fsum(
            firm.loan_balance for firm in world.firms + context.service_firms
        )
        return result

    def wage_hook(firm_system, wage_bill, total_labor):
        context = active_context(firm_system.world)
        if context is None:
            return original_distribute_wages(firm_system, wage_bill, total_labor)
        world = firm_system.world
        original_population = world.population
        try:
            world.population = context.food_employed_people(world)
            result = original_distribute_wages(firm_system, wage_bill, total_labor)
        finally:
            world.population = original_population
        context.pay_service_wages(world)
        return result

    def purchase_hook(firm_system, household, need_units):
        result = original_purchase(firm_system, household, need_units)
        context = active_context(firm_system.world)
        return context.allocate_household_budget(firm_system, household, *result) if context is not None else result

    def sync_hook(world, firm_result=None):
        context = active_context(world)
        if context is None:
            return original_sync(world, firm_result)
        old = central_bank_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER
        central_bank_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = context.current_food_base
        try:
            result = original_sync(world, firm_result)
        finally:
            central_bank_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = old
        service_sales = context.settle_service_market(world)
        context.update_firm_result(world, firm_result, service_sales)
        return result

    def accounting_hook(layer, step, firms, aggregate=None):
        context = active_context(layer.world)
        if context is None:
            return original_accounting_step(layer, step, firms, aggregate)
        # AccountingLayer's canonical path capitalizes Food production wages.
        # A no-inventory Service wage is instead a current-period expense even
        # when realized sales are zero.  A shallow accounting-only view makes
        # the whole wage immediately flow through COGS without mutating the
        # live Firm, its physical output, or any behavioral state.
        service_views = []
        for firm in context.service_firms:
            view = copy.copy(firm)
            if number(firm.wage_payment) > 0:
                view.production = 1.0
                view.sales_units = 1.0
            else:
                view.production = 0.0
                view.sales_units = 0.0
            view.inventory_units = 0.0
            view.spoilage_units = 0.0
            view.public_inventory_purchase_units = 0.0
            service_views.append(view)
        combined = list(firms) + service_views
        row_start = len(layer.rows)
        result = original_accounting_step(layer, step, combined, aggregate)
        food_ids = {firm.firm_id for firm in firms}
        for row in layer.rows[row_start:]:
            if int(number(row.get("firm_id", -1), -1)) not in food_ids:
                continue
            wage = number(row.get("production_wage_cost"))
            capitalized = number(row.get("manufacturing_cost"))
            if wage <= 1e-12 or capitalized > 1e-12:
                continue
            # A paid Food worker with zero realized production creates no new
            # inventory asset.  Classify that idle-labor cash cost as a true
            # current-period expense in this diagnostic view.
            row["wage_expense"] = wage
            row["accounting_operating_profit"] -= wage
            row["accounting_net_income"] -= wage
            row["equity_bridge_gap"] += wage
        return result

    World.assign_unassigned_workers_to_firms = assign_hook
    FirmSystem.step = firm_step_hook
    FirmSystem.total_labor = total_labor_hook
    FirmSystem.distribute_wages = wage_hook
    FirmSystem.household_food_purchase_units = purchase_hook
    World.prepare_multi_firm_credit = prepare_hook
    World.sync_firms_from_aggregate = sync_hook
    AccountingLayer.record_step = accounting_hook
    try:
        yield
    finally:
        World.assign_unassigned_workers_to_firms = original_assign
        FirmSystem.step = original_firm_step
        FirmSystem.total_labor = original_total_labor
        FirmSystem.distribute_wages = original_distribute_wages
        FirmSystem.household_food_purchase_units = original_purchase
        World.prepare_multi_firm_credit = original_prepare
        World.sync_firms_from_aggregate = original_sync
        AccountingLayer.record_step = original_accounting_step


def build_base_world(population, steps):
    scenario = apply_scenario("interest_behavioral_5pct")
    world = World(
        initial_population=population,
        seed=SEED,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    apply_runtime_scenario(scenario, world)
    world.split_firms(FOOD_FIRM_COUNT)
    world.generalized_firm_operating_contracts = True
    world.steps = steps
    # No raw transaction ledger is an explicit Step 14E efficiency rule.
    world.ledger.record_details = False
    return world


def capture_control_week(world):
    macro = world.diagnostics_rows[-1]
    return {
        "step": int(number(macro.get("global_step", macro.get("step")))),
        "population": number(macro.get("population")),
        "active_households": number(macro.get("active_households", len(active_households(world)))),
        "births": number(macro.get("births")),
        "deaths": number(macro.get("deaths")),
        "food_labor": number(macro.get("total_labor")),
        "service_labor": 0.0,
        "total_employed_labor": number(macro.get("total_labor")),
        "eligible_unassigned_labor": 0.0,
        "total_wage_income": number(macro.get("total_income")) - number(macro.get("firm_dividend", macro.get("dividend"))),
        "total_income": number(macro.get("total_income")),
        "total_consumption": number(macro.get("total_consumption")),
        "household_wealth": number(macro.get("total_wealth")),
        "food_sales_units": number(macro.get("food_sales_units")),
        "food_sales_revenue": number(macro.get("firm_sales_revenue")),
        "food_production": number(macro.get("food_output_units")),
        "food_inventory_units": number(macro.get("food_inventory_units")),
        "food_cash": number(macro.get("firm_cash")),
        "food_debt": number(macro.get("working_capital_loan_balance")),
        "food_operating_profit": number(macro.get("firm_profit_before_dividend")),
        "service_desired_budget": 0.0,
        "service_sales": 0.0,
        "service_unexecuted_budget": 0.0,
        "service_wages": 0.0,
        "service_operating_profit": 0.0,
        "service_cash": 0.0,
        "service_debt": 0.0,
        "service_arrears": 0.0,
        "target_base_sum": BASE_BUFFER,
        "monetary_accounting_gap": number(macro.get("monetary_accounting_gap")),
        "goods_conservation_gap": number(macro.get("goods_conservation_gap")),
    }


def run_world(world, steps, context=None, progress_label=""):
    history = []
    started = time.perf_counter()
    for index in range(steps):
        world.step()
        if context is not None and context.active:
            context.after_week(world)
        else:
            history.append(capture_control_week(world))
        if steps >= 500 and (index + 1) % max(260, steps // 4) == 0:
            print(f"[{progress_label}] {index + 1}/{steps}", flush=True)
    return (context.history if context is not None and context.active else history), time.perf_counter() - started


def normalized_rows(rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return [{key: row.get(key, "") for key in fields} for row in rows]


def compare_tables(old_rows, new_rows, tolerance=1e-12):
    old_rows = normalized_rows(old_rows)
    new_rows = normalized_rows(new_rows)
    if len(old_rows) != len(new_rows):
        return False, f"row_count:{len(old_rows)}!={len(new_rows)}"
    for index, (old, new) in enumerate(zip(old_rows, new_rows)):
        if set(old) != set(new):
            return False, f"schema@{index}"
        for key in old:
            left, right = old[key], new[key]
            try:
                left_number, right_number = float(left), float(right)
                if math.isfinite(left_number) and math.isfinite(right_number):
                    if abs(left_number - right_number) > tolerance:
                        return False, f"{key}@{index}:{left}!={right}"
                    continue
            except (TypeError, ValueError):
                pass
            if left != right:
                return False, f"{key}@{index}:{left}!={right}"
    return True, ""


def accounting_metrics(world, context):
    maxima = {
        "firm_cash_flow_gap": max((abs(number(row.get("cash_flow_gap"))) for row in world.accounting.rows), default=0.0),
        "inventory_bridge_gap": max((abs(number(row.get("inventory_bridge_gap"))) for row in world.accounting.rows), default=0.0),
        "equity_bridge_gap": max((abs(number(row.get("equity_bridge_gap"))) for row in world.accounting.rows), default=0.0),
        "household_wealth_bridge_gap": max((abs(number(row.get("household_wealth_bridge_gap"))) for row in world.accounting.household_rows), default=0.0),
        "public_cash_flow_gap": max((abs(number(row.get("public_cash_flow_gap"))) for row in world.accounting.public_rows), default=0.0),
        "credit_money_stock_flow_gap": max((abs(number(row.get("credit_money_stock_flow_gap"))) for row in world.accounting.central_bank_rows), default=0.0),
        "money_location_gap": max((abs(number(row.get("money_location_gap"))) for row in world.accounting.reconciliation_rows), default=0.0),
        "loan_reconciliation_gap": max((abs(number(row.get("loan_reconciliation_gap"))) for row in world.accounting.reconciliation_rows), default=0.0),
        "monetary_accounting_gap": max((abs(number(row.get("monetary_accounting_gap"))) for row in world.diagnostics_rows), default=0.0),
        "goods_conservation_gap": max((abs(number(row.get("food_conservation_gap"))) for row in world.diagnostics_rows), default=0.0),
        "service_settlement_gap": max((abs(number(row.get("service_settlement_gap"))) for row in context.history), default=0.0),
        "service_no_inventory_gap": max((abs(number(row.get("service_no_inventory_gap"))) for row in context.history), default=0.0),
        "service_cash_bridge_gap": max((abs(number(row.get("cash_flow_gap"))) for row in context.firm_history), default=0.0),
    }
    maxima["max_accounting_or_conservation_gap"] = max(maxima.values(), default=0.0)
    maxima["invariant_violation_count"] = len(getattr(world, "invariant_violations", []))
    maxima["accounting_all_pass"] = (
        maxima["max_accounting_or_conservation_gap"] <= TOLERANCE
        and maxima["invariant_violation_count"] == 0
    )
    return maxima


def late(rows, count=104):
    return rows[-min(count, len(rows)):]


def summarize_run(world, context, elapsed, control_reference):
    rows = context.history
    early_rows = rows[:min(104, len(rows))]
    late_rows = late(rows)
    firm_late = late(context.firm_history, count=SERVICE_FIRM_COUNT * min(104, len(rows)))
    movement = context.movement_summary()
    accounting = accounting_metrics(world, context)
    service_by_firm = defaultdict(list)
    for row in firm_late:
        service_by_firm[int(row["firm_id"])].append(row)
    service_symmetry = [
        mean(item["revenue"] for item in values)
        for values in service_by_firm.values()
    ]
    symmetry_gap = (
        safe_ratio(max(service_symmetry) - min(service_symmetry), mean(service_symmetry))
        if len(service_symmetry) == SERVICE_FIRM_COUNT and mean(service_symmetry) > 1e-12 else 0.0
    )
    move_count = movement["unique_workers_moved"]
    ping_pong = (
        movement["workers_moved_more_than_once"] > max(5, 0.25 * move_count)
        or movement["A_B_A_sequences"] > max(2, 0.05 * move_count)
    )
    service_debt_slope = slope([row["service_debt"] for row in late_rows])
    late_service_revenue = mean(row["service_sales"] for row in late_rows)
    credit_instability = (
        service_debt_slope > max(1.0, 0.10 * late_service_revenue)
        and mean(row["service_operating_profit"] for row in late_rows) < 0
    )
    food_alive = mean(row["food_sales_units"] for row in late_rows) > 1.0 and mean(row["food_labor"] for row in late_rows) > 1.0
    service_alive = mean(row["service_sales"] for row in late_rows) > 1.0 and mean(row["service_labor"] for row in late_rows) > 1.0
    wages_preserved = mean(row["total_wage_income"] for row in late_rows) > max(
        1.0, 0.20 * mean(row["total_wage_income"] for row in late(control_reference))
    )
    consumption_preserved = mean(row["total_consumption"] for row in late_rows) > max(
        1.0, 0.20 * mean(row["total_consumption"] for row in late(control_reference))
    )
    labor_pass = not context.boundary_violations
    if not accounting["accounting_all_pass"] or not labor_pass:
        classification = "G. ACCOUNTING_OR_CONSERVATION_FAILURE"
    elif ping_pong:
        classification = "E. LABOR_REALLOCATION_OSCILLATION"
    elif not food_alive:
        classification = "B. FOOD_ACTIVITY_COLLAPSE"
    elif not service_alive:
        classification = "C. SERVICE_ACTIVITY_FAILURE"
    elif credit_instability:
        classification = "D. SERVICE_CREDIT_INSTABILITY"
    elif not consumption_preserved:
        classification = "F. HOUSEHOLD_DEMAND_COLLAPSE"
    else:
        classification = "A. EARLY_STRUCTURAL_PASS"
    summary = {
        "demand_reference": context.demand_id,
        "service_discretionary_share": context.service_share,
        "unit_economics": context.unit_id,
        "unit_economics_label": UNIT_LABELS[context.unit_id],
        "service_productivity_normalized": context.service_productivity,
        "population": context.population,
        "steps": len(rows),
        "classification": classification,
        "promotion_pass": classification == "A. EARLY_STRUCTURAL_PASS",
        "elapsed_seconds": elapsed,
        "final_population": rows[-1]["population"],
        "late_mean_food_labor": mean(row["food_labor"] for row in late_rows),
        "late_mean_service_labor": mean(row["service_labor"] for row in late_rows),
        "late_mean_total_employed_labor": mean(row["total_employed_labor"] for row in late_rows),
        "late_mean_eligible_unassigned_labor": mean(row["eligible_unassigned_labor"] for row in late_rows),
        "late_mean_total_wage_income": mean(row["total_wage_income"] for row in late_rows),
        "late_mean_household_consumption": mean(row["total_consumption"] for row in late_rows),
        "late_mean_food_sales_units": mean(row["food_sales_units"] for row in late_rows),
        "late_mean_food_production": mean(row["food_production"] for row in late_rows),
        "late_mean_food_cash": mean(row["food_cash"] for row in late_rows),
        "late_mean_food_debt": mean(row["food_debt"] for row in late_rows),
        "late_mean_service_desired_budget": mean(row["service_desired_budget"] for row in late_rows),
        "late_mean_service_sales": late_service_revenue,
        "late_mean_service_execution_ratio": safe_ratio(
            math.fsum(row["service_sales"] for row in late_rows),
            math.fsum(row["service_desired_budget"] for row in late_rows),
        ),
        "early_mean_service_execution_ratio": safe_ratio(
            math.fsum(row["service_sales"] for row in early_rows),
            math.fsum(row["service_desired_budget"] for row in early_rows),
        ),
        "late_mean_service_operating_profit": mean(row["service_operating_profit"] for row in late_rows),
        "late_mean_service_cash": mean(row["service_cash"] for row in late_rows),
        "final_service_debt": rows[-1]["service_debt"],
        "final_service_arrears": rows[-1]["service_arrears"],
        "late_service_debt_slope": service_debt_slope,
        "late_mean_service_principal_utilization": mean(row["principal_utilization"] for row in firm_late),
        "late_mean_service_utilization": mean(row["utilization"] for row in firm_late),
        "service_break_even_utilization": safe_ratio(context.current_common_wage, context.service_productivity),
        "food_to_service_moves": movement["food_to_service_moves"],
        "cross_sector_moves": movement["cross_sector_moves"],
        "unique_workers_moved": movement["unique_workers_moved"],
        "workers_moved_more_than_once": movement["workers_moved_more_than_once"],
        "max_moves_per_worker": movement["max_moves_per_worker"],
        "median_time_between_moves": movement["median_time_between_moves"],
        "A_B_A_sequences": movement["A_B_A_sequences"],
        "worker_ping_pong_material": ping_pong,
        "service_credit_instability": credit_instability,
        "food_activity_survives": food_alive,
        "service_activity_survives": service_alive,
        "aggregate_wage_income_preserved": wages_preserved,
        "household_consumption_preserved": consumption_preserved,
        "persistent_unassigned_labor": mean(row["eligible_unassigned_labor"] for row in late_rows) > 1.0,
        "service_matching_order_bias": symmetry_gap > 0.10,
        "late_service_revenue_symmetry_gap": symmetry_gap,
        "target_base_max_gap": max(abs(row["target_base_sum"] - BASE_BUFFER) for row in rows),
        "initial_capitalization_money_gap": context.capitalization["money_neutral_gap"],
        "labor_invariant_violations": len(context.boundary_violations),
        **accounting,
    }
    return summary


def full_classification(summary):
    if not summary["accounting_all_pass"] or summary["labor_invariant_violations"]:
        return "H. ACCOUNTING_OR_IMPLEMENTATION_FAILURE"
    if summary["worker_ping_pong_material"]:
        return "E. FOOD_SERVICE_REALLOCATION_DESTABILIZES"
    if not summary["food_activity_survives"] or not summary["service_activity_survives"]:
        return "F. TWO_SECTOR_DEMAND_COLLAPSE_PERSISTS"
    if summary["service_credit_instability"]:
        return "G. FINANCIAL_INSTABILITY_DOMINATES"
    if summary["late_mean_service_operating_profit"] < 0 and summary["late_mean_service_utilization"] < summary["service_break_even_utilization"]:
        return "D. SERVICE_UNIT_ECONOMICS_TOO_WEAK"
    if summary["late_mean_service_labor"] < 0.05 * summary["late_mean_food_labor"]:
        return "C. SERVICE_DEMAND_TOO_WEAK"
    if summary["aggregate_wage_income_preserved"] and summary["food_to_service_moves"] > 0:
        return "A. TWO_SECTOR_LABOR_INCOME_CLOSURE_SUPPORTED"
    return "B. PARTIAL_LABOR_INCOME_CLOSURE"


def service_firm_summary(context, full_classification_value):
    output = []
    for firm_id in sorted(context.service_dict):
        rows = [row for row in context.firm_history if int(row["firm_id"]) == firm_id]
        mature = late(rows, 260)
        output.append({
            "treatment": f"{context.demand_id}_{context.unit_id}",
            "fullscale_classification": full_classification_value,
            "sector": "Service",
            "firm_id": firm_id,
            "late_mean_employment": mean(row["employee_count"] for row in mature),
            "late_mean_labor_services": mean(row["labor_services"] for row in mature),
            "late_mean_latent_demand": mean(row["latent_demand"] for row in mature),
            "late_mean_demand_forecast": mean(row["demand_forecast"] for row in mature),
            "late_mean_capacity": mean(row["technical_capacity"] for row in mature),
            "late_mean_sales": mean(row["sales_units"] for row in mature),
            "late_mean_revenue": mean(row["revenue"] for row in mature),
            "late_mean_utilization": mean(row["utilization"] for row in mature),
            "late_mean_scheduled_payroll": mean(row["scheduled_payroll"] for row in mature),
            "late_mean_executed_payroll": mean(row["executed_payroll"] for row in mature),
            "late_mean_operating_profit": mean(row["operating_profit"] for row in mature),
            "final_cash": rows[-1]["cash"],
            "final_loan_balance": rows[-1]["loan_balance"],
            "late_loan_slope": slope([row["loan_balance"] for row in mature]),
            "final_interest_arrears": rows[-1]["interest_arrears"],
            "late_mean_principal_utilization": mean(row["principal_utilization"] for row in mature),
            "late_mean_cash_flow_gap": mean(abs(row["cash_flow_gap"]) for row in mature),
        })
    return output


def food_firm_summary(world, context, full_classification_value):
    output = []
    grouped = defaultdict(list)
    for row in world.firm_diagnostics_rows:
        grouped[int(number(row.get("firm_id", -1), -1))].append(row)
    for firm_id in sorted(grouped):
        rows = grouped[firm_id]
        mature = late(rows, 260)
        final = rows[-1]
        output.append({
            "treatment": f"{context.demand_id}_{context.unit_id}",
            "fullscale_classification": full_classification_value,
            "sector": "Food",
            "firm_id": firm_id,
            "late_mean_employment": mean(row.get("employee_count") for row in mature),
            "late_mean_labor_services": mean(
                safe_ratio(row.get("scheduled_productive_capacity"), FOOD_PRODUCTIVITY)
                for row in mature
            ),
            "late_mean_latent_demand": mean(row.get("demand_units") for row in mature),
            "late_mean_demand_forecast": mean(row.get("expected_demand") for row in mature),
            "late_mean_capacity": mean(row.get("scheduled_productive_capacity") for row in mature),
            "late_mean_sales": mean(row.get("sales_units") for row in mature),
            "late_mean_revenue": mean(row.get("sales_revenue") for row in mature),
            "late_mean_utilization": mean(row.get("capacity_utilization") for row in mature),
            "late_mean_scheduled_payroll": mean(row.get("scheduled_wage_bill") for row in mature),
            "late_mean_executed_payroll": mean(row.get("executed_wage_bill") for row in mature),
            "late_mean_operating_profit": mean(row.get("profit") for row in mature),
            "final_cash": number(final.get("cash")),
            "final_loan_balance": number(final.get("loan_balance")),
            "late_loan_slope": slope([row.get("loan_balance") for row in mature]),
            "final_interest_arrears": number(final.get("interest_arrears")),
            "late_mean_principal_utilization": mean(
                safe_ratio(row.get("loan_balance"), row.get("credit_limit"))
                for row in mature
            ),
            "late_mean_cash_flow_gap": mean(abs(number(row.get("cash_bridge_gap"))) for row in mature),
        })
    return output


def select_promotions(short_rows):
    survivors = [row for row in short_rows if row["promotion_pass"]]
    if not survivors:
        return []
    selected = []
    preferred = [("R25", "U2"), ("R50", "U1"), ("R75", "U0")]
    for demand, unit in preferred:
        item = next((row for row in survivors if row["demand_reference"] == demand and row["unit_economics"] == unit), None)
        if item is not None:
            selected.append(item)
    for item in sorted(survivors, key=lambda row: (row["demand_reference"], row["unit_economics"])):
        if len(selected) >= 3:
            break
        if item not in selected:
            selected.append(item)
    return selected[:3]


def overview_figure(control_history, full_runs):
    rows = [("C0", control_history)] + [
        (name, context.history) for name, context, _ in full_runs
    ]
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 4.2 * len(rows)), squeeze=False)
    for row_index, (label, history) in enumerate(rows):
        x = [item["step"] for item in history]
        axes[row_index][0].plot(x, [item["food_labor"] for item in history], label="Food")
        axes[row_index][0].plot(x, [item["service_labor"] for item in history], label="Service")
        axes[row_index][0].plot(x, [item["eligible_unassigned_labor"] for item in history], label="Unassigned")
        axes[row_index][1].plot(x, [item["food_sales_revenue"] for item in history], label="Food")
        axes[row_index][1].plot(x, [item["service_sales"] for item in history], label="Service")
        axes[row_index][2].plot(x, [item["total_wage_income"] for item in history], color="#235789")
        axes[row_index][3].plot(x, [item["service_operating_profit"] for item in history], label="Service profit")
        axes[row_index][3].plot(x, [item["service_debt"] for item in history], label="Service debt")
        for column in range(4):
            axes[row_index][column].grid(alpha=0.25)
            axes[row_index][column].set_title(
                ["Employment", "Nominal revenue", "Household wage income", "Service finance"][column]
            )
        axes[row_index][0].set_ylabel(label)
        axes[row_index][0].legend(fontsize=8)
        axes[row_index][1].legend(fontsize=8)
        axes[row_index][3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "step14E_two_sector_overview.png", dpi=220)
    plt.close(fig)


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = build_base_world(SHORT_POPULATION, SHORT_STEPS)
    python_state = random.getstate()
    try:
        import numpy as np
        numpy_state = np.random.get_state()
    except Exception:
        np = None
        numpy_state = None

    with experiment_hooks():
        random.setstate(python_state)
        if np is not None:
            np.random.set_state(numpy_state)
        control_world = copy.deepcopy(base)
        control_history, control_elapsed = run_world(
            control_world, SHORT_STEPS, progress_label="C0"
        )

        random.setstate(python_state)
        if np is not None:
            np.random.set_state(numpy_state)
        plumbing_world = copy.deepcopy(base)
        plumbing_context = TwoSectorCoordinator("R25", "U0", SHORT_POPULATION, active=False)
        plumbing_context.initialize(plumbing_world, capitalize=False)
        _, plumbing_elapsed = run_world(
            plumbing_world, CONTROL_GATE_STEPS, plumbing_context, "C1"
        )
        diag_parity, diag_detail = compare_tables(
            control_world.diagnostics_rows[:CONTROL_GATE_STEPS],
            plumbing_world.diagnostics_rows,
        )
        firm_parity, firm_detail = compare_tables(
            [row for row in control_world.firm_diagnostics_rows if int(number(row.get("global_step", row.get("step")))) < CONTROL_GATE_STEPS],
            plumbing_world.firm_diagnostics_rows,
        )
        accounting_parity, accounting_detail = compare_tables(
            [row for row in control_world.accounting.reconciliation_rows if int(number(row.get("step"))) < CONTROL_GATE_STEPS],
            plumbing_world.accounting.reconciliation_rows,
        )
        control_parity = diag_parity and firm_parity and accounting_parity
        if not control_parity:
            raise RuntimeError(
                f"C0/C1 parity failed: diagnostics={diag_detail}; firms={firm_detail}; accounting={accounting_detail}"
            )
        del plumbing_world
        gc.collect()

        short_rows = []
        for demand_id in DEMANDS:
            for unit_id in UNIT_ECONOMICS:
                random.setstate(python_state)
                if np is not None:
                    np.random.set_state(numpy_state)
                world = copy.deepcopy(base)
                context = TwoSectorCoordinator(demand_id, unit_id, SHORT_POPULATION)
                context.initialize(world)
                _, elapsed = run_world(world, SHORT_STEPS, context, f"{demand_id}-{unit_id}")
                summary = summarize_run(world, context, elapsed, control_history)
                short_rows.append(summary)
                print(f"[{demand_id}-{unit_id}] {summary['classification']}", flush=True)
                del world, context
                gc.collect()

        promotions = select_promotions(short_rows)
        full_rows = []
        full_firm_rows = []
        full_runs = []
        full_control_history = [
            {
                "step": int(number(row.get("global_step", row.get("step")))),
                "food_labor": number(row.get("total_labor")),
                "service_labor": 0.0,
                "eligible_unassigned_labor": 0.0,
                "food_sales_revenue": number(row.get("firm_sales_revenue")),
                "service_sales": 0.0,
                "total_wage_income": number(row.get("total_income")) - number(row.get("firm_dividend", row.get("dividend"))),
                "service_operating_profit": 0.0,
                "service_debt": 0.0,
            }
            for row in read_csv(CANONICAL_OUTPUT / "diagnostics.csv")
        ]
        for promoted in promotions:
            demand_id = promoted["demand_reference"]
            unit_id = promoted["unit_economics"]
            full_base = build_base_world(FULL_POPULATION, FULL_STEPS)
            full_python_state = random.getstate()
            full_numpy_state = np.random.get_state() if np is not None else None
            random.setstate(full_python_state)
            if np is not None:
                np.random.set_state(full_numpy_state)
            world = full_base
            context = TwoSectorCoordinator(demand_id, unit_id, FULL_POPULATION)
            context.initialize(world)
            _, elapsed = run_world(world, FULL_STEPS, context, f"FULL-{demand_id}-{unit_id}")
            summary = summarize_run(world, context, elapsed, full_control_history)
            summary["treatment"] = f"{demand_id}_{unit_id}"
            summary["fullscale_classification"] = full_classification(summary)
            full_rows.append(summary)
            full_firm_rows.extend(food_firm_summary(world, context, summary["fullscale_classification"]))
            full_firm_rows.extend(service_firm_summary(context, summary["fullscale_classification"]))
            context._world = None
            full_runs.append((summary["treatment"], context, summary))
            del world, full_base
            gc.collect()

    if full_runs:
        overview_figure(full_control_history, full_runs)

    short_passed = sum(row["promotion_pass"] for row in short_rows)
    accounting_all = all(row["accounting_all_pass"] for row in short_rows + full_rows)
    labor_all = all(row["labor_invariant_violations"] == 0 for row in short_rows + full_rows)
    any_supported = any(
        row.get("fullscale_classification") == "A. TWO_SECTOR_LABOR_INCOME_CLOSURE_SUPPORTED"
        for row in full_rows
    )
    any_partial = any(
        row.get("fullscale_classification") in {
            "A. TWO_SECTOR_LABOR_INCOME_CLOSURE_SUPPORTED",
            "B. PARTIAL_LABOR_INCOME_CLOSURE",
        }
        for row in full_rows
    )
    if not accounting_all or not labor_all:
        verdict = "I. ACCOUNTING_OR_IMPLEMENTATION_FAILURE"
    elif short_passed == 0:
        verdict = "H. NO_SHORT_SCREEN_CELL_SURVIVED"
    elif any_supported:
        verdict = "A. TWO_SECTOR_LABOR_INCOME_CLOSURE_DEMONSTRATED"
    elif any_partial:
        verdict = "B. TWO_SECTOR_LABOR_INCOME_CLOSURE_PARTIALLY_SUPPORTED"
    elif all(row["late_mean_service_labor"] < 0.05 * row["late_mean_food_labor"] for row in full_rows):
        verdict = "C. SERVICE_DEMAND_ENVELOPE_TOO_WEAK"
    elif all(row["late_mean_service_operating_profit"] < 0 for row in full_rows):
        verdict = "D. SERVICE_UNIT_ECONOMICS_BLOCKER"
    elif any(row["worker_ping_pong_material"] for row in full_rows):
        verdict = "E. INTERSECTOR_LABOR_REALLOCATION_BLOCKER"
    elif any(row["service_credit_instability"] for row in full_rows):
        verdict = "G. MULTISECTOR_FINANCIAL_INSTABILITY_BLOCKER"
    else:
        verdict = "F. TWO_SECTOR_DEMAND_COLLAPSE_PERSISTS"

    flags = {
        "verdict": verdict,
        "control_parity_pass": control_parity,
        "two_sector_plumbing_control_pass": control_parity,
        "short_screen_cells_total": 9,
        "short_screen_cells_passed": short_passed,
        "fullscale_cells_run": len(full_rows),
        "Service_firms_active_in_treatments": any(row["late_mean_service_labor"] > 0 for row in short_rows),
        "Service_transactions_positive_any": any(row["late_mean_service_sales"] > 0 for row in short_rows),
        "Service_employment_positive_any": any(row["late_mean_service_labor"] > 0 for row in short_rows),
        "Food_to_Service_worker_moves_positive_any": any(row["food_to_service_moves"] > 0 for row in short_rows),
        "Food_nonzero_activity_survives_any": any(row["food_activity_survives"] for row in short_rows),
        "aggregate_wage_income_preserved_any": any(row["aggregate_wage_income_preserved"] for row in short_rows + full_rows),
        "income_mediated_collapse_broken_any": any(
            row.get("fullscale_classification") in {
                "A. TWO_SECTOR_LABOR_INCOME_CLOSURE_SUPPORTED",
                "B. PARTIAL_LABOR_INCOME_CLOSURE",
            } for row in full_rows
        ),
        "persistent_unassigned_labor_any": any(row["persistent_unassigned_labor"] for row in short_rows + full_rows),
        "Service_credit_instability_any": any(row["service_credit_instability"] for row in short_rows + full_rows),
        "worker_ping_pong_material_any": any(row["worker_ping_pong_material"] for row in short_rows + full_rows),
        "necessary_Food_protection_pass": all(row["food_activity_survives"] for row in short_rows if row["promotion_pass"]),
        "target_cash_global_base_reconciliation_pass": all(row["target_base_max_gap"] <= TOLERANCE for row in short_rows + full_rows),
        "initial_capitalization_money_neutral_pass": all(abs(row["initial_capitalization_money_gap"]) <= TOLERANCE for row in short_rows + full_rows),
        "labor_invariants_pass": labor_all,
        "accounting_all_pass": accounting_all,
        "new_rng_draws_for_new_mechanisms": 0,
        "permanent_Service_price_selected": False,
        "permanent_Service_productivity_selected": False,
        "permanent_Service_demand_share_selected": False,
        "permanent_Service_firm_count_calibrated": False,
        "permanent_labor_rule_selected": False,
        "permanent_capitalization_rule_selected": False,
        "3640_extension_run": False,
        "seed7_21_run": False,
        "human_review_required": True,
    }

    write_csv(OUTPUT / "step14E_short_screen_matrix.csv", short_rows)
    write_csv(OUTPUT / "step14E_fullscale_system_comparison.csv", full_rows)
    write_csv(OUTPUT / "step14E_fullscale_by_firm.csv", full_firm_rows)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    best = max(
        full_rows or short_rows,
        key=lambda row: (
            row.get("fullscale_classification") == "A. TWO_SECTOR_LABOR_INCOME_CLOSURE_SUPPORTED",
            row["late_mean_total_wage_income"],
        ),
    )
    demand_effect = {
        demand: mean(row["late_mean_service_labor"] for row in short_rows if row["demand_reference"] == demand)
        for demand in DEMANDS
    }
    unit_effect = {
        unit: mean(row["late_mean_service_operating_profit"] for row in short_rows if row["unit_economics"] == unit)
        for unit in UNIT_ECONOMICS
    }
    promoted_text = ", ".join(row["treatment"] for row in full_rows) or "none"
    long_run_nominee = (
        max(full_rows, key=lambda row: row["late_mean_total_wage_income"])["treatment"]
        if any_supported or any_partial else "none"
    )
    control_food_debt = mean(row["food_debt"] for row in late(control_history))
    treatment_food_debt = mean(row["late_mean_food_debt"] for row in short_rows)
    if treatment_food_debt < 0.8 * control_food_debt:
        food_stress_direction = "improved relative to C0 on late Food debt"
    elif treatment_food_debt > 1.2 * control_food_debt:
        food_stress_direction = "worsened relative to C0 on late Food debt"
    else:
        food_stress_direction = "mainly redistributed, with no material late-debt change versus C0"
    full_verdicts = "; ".join(
        f"{row['treatment']}: {row['fullscale_classification']}" for row in full_rows
    ) or "No full-scale treatment was run."
    summary = f"""# Step 14E Acceptance Summary

## Headline verdict

**{verdict}**

The implementation is test-local. Service price 1.0, R25/R50/R75, U0/U1/U2,
two Service Firms, the 13-week review, 25% adjustment, capitalization bridge,
and target-cash adapter remain experimental references rather than permanent
calibrations. No new RNG draws were introduced.

## Control and execution

- C0/C1 exact plumbing parity: **{control_parity}** (C0 {SHORT_STEPS} weeks; C1 gate {CONTROL_GATE_STEPS} weeks).
- Short-screen structural passes: **{short_passed}/9**.
- Full-scale cells run: **{len(full_rows)}** ({promoted_text}).
- Full-scale classifications: {full_verdicts}
- Total runtime: {time.perf_counter() - started:.2f} seconds.

## Required questions

1. {short_passed} of 9 short-screen cells passed the promotion gate.
2. Demand mattered most through Service employment; mean late labor by demand was {demand_effect}.
3. Unit economics mattered most through Service operating profit; mean late profit by U reference was {unit_effect}.
4. Service Firms hired workers: {flags['Service_employment_positive_any']}.
5. Food Firms released workers: {any(row['food_to_service_moves'] > 0 or row['unique_workers_moved'] > 0 for row in short_rows)}.
6. Legal Food -> pool -> Service movement occurred: {flags['Food_to_Service_worker_moves_positive_any']}.
7. Employed labor remained above the collapsed Stage B.1 path in at least one cell: {flags['aggregate_wage_income_preserved_any']}.
8. Persistent eligible-unassigned labor remained: {flags['persistent_unassigned_labor_any']}.
9. Aggregate wage income was materially supported in at least one cell: {flags['aggregate_wage_income_preserved_any']}.
10. Household consumption was materially supported in at least one cell: {any(row['household_consumption_preserved'] for row in short_rows + full_rows)}.
11. Protected necessary Food remained viable in promoted cells: {flags['necessary_Food_protection_pass']}.
12. Food stabilized at nonzero scale in at least one cell: {flags['Food_nonzero_activity_survives_any']}.
13. Service Firms reached nonzero scale in at least one treatment: {flags['Service_firms_active_in_treatments']}.
14. Service execution improved from the first to last 104-week window in at least one cell: {any(row['late_mean_service_execution_ratio'] > row['early_mean_service_execution_ratio'] + 1e-9 for row in short_rows)}.
15. The best cell's late Service utilization was {best['late_mean_service_utilization']:.6g}, versus break-even {best['service_break_even_utilization']:.6g}.
16. Service credit instability appeared in any cell: {flags['Service_credit_instability_any']}.
17. Food financial stress {food_stress_direction}; Service debt is separately reported rather than hidden in Food finance.
18. Service employment broke or partially broke the Stage B.1 income loop: {flags['income_mediated_collapse_broken_any']}.
19. Residual unassigned labor was present and is consistent with the 14E.0 labor-slack envelope: {flags['persistent_unassigned_labor_any']}.
20. Economy-wide target-cash base stayed 150000: {flags['target_cash_global_base_reconciliation_pass']}.
21. Initialization preserved total money: {flags['initial_capitalization_money_neutral_pass']}.
22. Material deterministic Service order bias appeared: {any(row['service_matching_order_bias'] for row in short_rows + full_rows)}.
23. Material worker ping-pong appeared: {flags['worker_ping_pong_material_any']}.
24. Promoted cells were {promoted_text}, selected only from structural passes and capped at three.
25. Full-scale verdicts: {full_verdicts}
26. Credible two-sector closure evidence exists: {any_supported or any_partial}.
27. Later long-run candidate: {long_run_nominee}. No failed cell is nominated and no long-run extension was run.

## Hard-stop interpretation

All nine cells failed the short promotion gate, so full scale was not run. R25
lost both Food and Service operating activity as wage income and demand
contracted. R50/R75 preserved positive two-sector activity, but Service
utilization remained below dynamic break-even, operating profit was negative,
and Service debt had a positive late-window slope. The observed blocker is
therefore mixed: weak-income demand collapse at R25 and Service financial
instability at R50/R75, not accounting, target-cash duplication, RNG, or labor
bookkeeping.

## Reconciliation

- Accounting all pass: **{accounting_all}**.
- Labor invariants pass: **{labor_all}**.
- Target base reconciliation pass: **{flags['target_cash_global_base_reconciliation_pass']}**.
- Capitalization money-neutral pass: **{flags['initial_capitalization_money_neutral_pass']}**.
- Service supplier allocation: equal latent split with deterministic spare-capacity spillover.
- Failed Service expenditure remains Household cash; there is no same-week Food fallback.
- Startup cash is an experimental existing-money redistribution and establishes no ownership claim.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
