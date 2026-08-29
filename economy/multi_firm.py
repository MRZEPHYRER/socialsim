from dataclasses import dataclass, field
import math
import random

from economy import config
from economy.investment_contracts import CapitalStock
from economy.ownership_accounting import CapTable
from economy.multisector import default_firm_identity
from economy.scheduling import deterministic_review_phase
from productivity import age_productivity


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


@dataclass
class FirmSlice:
    firm_id: int
    employee_ids: list = field(default_factory=list)
    share: float = 1.0
    productive_capacity: float = 0.0
    cash: float = 0.0
    inventory_units: float = 0.0
    inventory_value: float = 0.0
    expected_demand: float = 0.0
    expected_sales: float = 0.0
    loan_balance: float = 0.0
    credit_principal_state: object = None
    price: float = 0.0
    wage_bill: float = 0.0
    sales: float = 0.0
    production: float = 0.0
    observed_demand: float = 0.0
    target_inventory_units: float = 0.0
    inventory_gap_units: float = 0.0
    production_plan: float = 0.0
    desired_production: float = 0.0
    desired_labor: float = 0.0
    actual_production: float = 0.0
    capacity_utilization: float = 0.0
    production_reviewed: bool = False
    production_change: float = 0.0
    production_review_timer: int = 0
    production_review_phase_offset: int = 0
    production_review_count: int = 0
    investment_review_phase_offset: int = 0
    investment_reviewed_this_step: bool = False
    investment_review_count: int = 0
    previous_actual_production: float = 0.0
    bootstrap_production_units: object = None
    profit: float = 0.0
    loan_issued: float = 0.0
    loan_repaid: float = 0.0
    cash_start: float = 0.0
    opening_principal: float = 0.0
    base_buffer_component: float = 0.0
    wage_bill_component: float = 0.0
    cash_after_borrowing: float = 0.0
    cash_after_wages: float = 0.0
    cash_after_sales: float = 0.0
    cash_before_repayment: float = 0.0
    repayment_buffer: float = 0.0
    closing_principal: float = 0.0
    credit_bridge_gap: float = 0.0
    sales_revenue: float = 0.0
    wage_payment: float = 0.0
    target_cash: float = 0.0
    funding_gap: float = 0.0
    dividend_payment: float = 0.0
    person_dividend_paid: float = 0.0
    legacy_dividend_entitlement: float = 0.0
    dividend_routing_gap: float = 0.0
    dividend_recipient_count: int = 0
    dividend_eligible_profit: float = 0.0
    dividend_payout_ratio: float = 0.0
    retained_profit_after_dividend: float = 0.0
    dividend_funding_source: str = "none"
    cash_before_dividend: float = 0.0
    cash_after_dividend: float = 0.0
    public_sector_cash_inflow: float = 0.0
    public_sector_cash_outflow: float = 0.0
    public_inventory_purchase_units: float = 0.0
    loan_interest_paid: float = 0.0
    other_cash_inflow: float = 0.0
    other_cash_outflow: float = 0.0
    # Step 15I.11D: customer-advance contract state.  These are neutral when
    # the customer-advance screen is disabled.
    customer_advance_liability: float = 0.0
    customer_advance_received_this_step: float = 0.0
    customer_advance_delivered_this_step: float = 0.0
    prepaid_capital_investment_asset: float = 0.0
    prepaid_capital_investment_paid_this_step: float = 0.0
    prepaid_capital_investment_capitalized_this_step: float = 0.0
    sales_collections_this_step: float = 0.0
    investment_cash_outflow_this_step: float = 0.0
    cash_end: float = 0.0
    cash_bridge_gap: float = 0.0
    cash_depletion_stage: str = "not_depleted"
    credit_allocation_mode: str = "not_recorded"
    relative_price: float = 1.0
    choice_probability: float = 1.0
    actual_market_share: float = 1.0
    unit_market_share: float = 1.0
    revenue_market_share: float = 1.0
    unmet_demand: float = 0.0
    demand_units: float = 0.0
    sales_units: float = 0.0
    firm_sales_units: float = 0.0
    public_inventory_sales_units: float = 0.0
    transaction_price: float = 0.0
    transaction_revenue: float = 0.0
    unit_market_share: float = 1.0
    revenue_market_share: float = 1.0
    unit_labor_cost: float = 0.0
    realized_unit_labor_cost: float = 0.0
    normal_unit_labor_cost: float = 0.0
    learner_proposed_price: float = 0.0
    price_floor: float = 0.0
    price_floor_binding: bool = False
    price_before_floor: float = 0.0
    final_executed_price: float = 0.0
    margin: float = 0.0
    inventory_coverage: float = 0.0
    inventory_gap: float = 0.0
    expected_profit: float = 0.0
    expected_share: float = 0.0
    smoothed_profit: float = 0.0
    baseline_profit: float = 0.0
    evaluated_profit: float = 0.0
    estimated_gradient: float = 0.0
    evaluation_timer: int = 0
    last_test_price: float = 0.0
    last_direction: int = 0
    last_profit_delta: float = 0.0
    last_price_direction: int = 0
    last_review_price: float = 0.0
    last_review_profit: float = 0.0
    last_review_share: float = 0.0
    last_review_inventory_gap: float = 0.0
    price_reviewed: bool = False
    # Step 14A passive identity; never read by the current Food executor.
    sector_id: str = "food"
    technology_id: str = "labor_only_food_v1"
    inventory_policy_id: str = "food_inventory_v1"
    output_good_id: str = "basic_consumption_good"
    active: bool = True
    # Step 15I.1 passive capital-good supply state.  A generic FirmSlice can
    # later carry capital-good inventory without requiring a special subclass.
    capital_good_inventory: object = None
    capital_good_production_units: float = 0.0
    capital_good_sales_units: float = 0.0
    capital_good_unit_production_cost: float = 0.0
    capital_good_labor_services: float = 0.0
    price_decision: str = "none"
    price_change: float = 0.0
    firm_rng: object = None
    scheduled_wage_bill: float = 0.0
    scheduled_wage_history: list = field(default_factory=list)
    trailing_scheduled_wage_bill: float = 0.0
    credit_limit: float = float("inf")
    credit_headroom: float = float("inf")
    requested_credit: float = 0.0
    executed_credit: float = 0.0
    denied_credit: float = 0.0
    credit_limit_binding: bool = False
    target_liquidity_shortfall: float = 0.0
    payroll_cash_shortfall: float = 0.0
    payroll_funding_ratio: float = 1.0
    executed_wage_bill: float = 0.0
    scheduled_productive_capacity: float = 0.0
    funded_productive_capacity: float = 0.0
    production_finance_constrained: bool = False
    financing_state: int = 0
    dividend_paid_while_credit_binding: float = 0.0
    dividend_paid_while_payroll_constrained: float = 0.0
    interest_arrears: float = 0.0
    opening_interest_arrears: float = 0.0
    current_interest_due: float = 0.0
    total_interest_obligation: float = 0.0
    interest_paid_to_opening_arrears: float = 0.0
    interest_paid_to_current_due: float = 0.0
    current_interest_unpaid: float = 0.0
    closing_interest_arrears: float = 0.0
    opening_lender_exposure: float = 0.0
    cash_before_interest: float = 0.0
    operating_liquidity_floor: float = 0.0
    interest_service_cash: float = 0.0
    cash_after_interest: float = 0.0
    cash_before_principal_repayment: float = 0.0
    cash_after_principal_repayment: float = 0.0
    dividend_before_interest_reservation_candidate: float = 0.0
    dividend_after_interest_reservation: float = 0.0
    dividend_reduced_for_interest: float = 0.0
    # Step 13.6E/13.7C: passive Default bookkeeping only.  These fields never
    # enter an economic decision.  D3 is the acute distress phase; contractual
    # Default remains active until current-interest service cures it.
    default_history_count: int = 0
    active_contract_default: bool = False
    active_default_episode: bool = False
    default_event_this_week: bool = False
    consecutive_d3_weeks: int = 0
    consecutive_non_d3_weeks: int = 0
    consecutive_no_breach_weeks: int = 0
    contract_cure_this_week: bool = False
    acute_default_phase_exit_this_week: bool = False
    default_distress_history: list = field(default_factory=list)
    default_distress_state: str = "D0"
    default_distress_features: dict = field(default_factory=dict)
    default_last_event_step: object = None
    # Step 13.8B: one temporary current-interest relief attempt per
    # contractual Default.  These fields are inert when the experiment is off.
    restructuring_used_this_default: bool = False
    restructuring_active: bool = False
    restructuring_start_this_week: bool = False
    restructuring_weeks_used: int = 0
    restructuring_default_weeks: int = 0
    weekly_interest_relief_amount: float = 0.0
    cumulative_interest_relief_amount: float = 0.0
    gross_current_interest_due: float = 0.0
    net_current_interest_due: float = 0.0
    interest_relief_amount: float = 0.0
    # Step 13.8F: historical arrears separated from revolving utilization at
    # one restructuring start, while remaining a fully payable lender claim.
    termout_used_this_default: bool = False
    termout_start_this_week: bool = False
    termout_snapshot_amount: float = 0.0
    legacy_arrears_term_claim: float = 0.0
    opening_legacy_arrears_term_claim: float = 0.0
    closing_legacy_arrears_term_claim: float = 0.0
    legacy_term_claim_payment: float = 0.0
    post_termout_arrears_payment: float = 0.0
    termout_pre_snapshot_arrears: float = 0.0
    termout_claim_reclassification_gap: float = 0.0
    # Step 15B passive runtime container.  Empty stock is ignored by current
    # Food/Service production and therefore contributes no capacity.
    capital_stock: object = None
    # Step 15E.1 passive ownership state.  Initial ownership is legacy-only.
    cap_table: object = None
    # Step 15E.2 controlled issuance accounting; inert in canonical runs.
    paid_in_equity: float = 0.0
    equity_issuance_cash_this_step: float = 0.0
    equity_issuance_shares_this_step: float = 0.0
    equity_issuance_buyer_count: int = 0

    def __post_init__(self):
        if self.capital_stock is None:
            self.capital_stock = CapitalStock(owner_firm_id=self.firm_id)
        elif self.capital_stock.owner_firm_id != self.firm_id:
            raise ValueError("FirmSlice capital stock owner must match firm_id")
        if self.cap_table is None:
            self.cap_table = CapTable.initial_legacy_owned(self.firm_id)
        elif self.cap_table.firm_id != self.firm_id:
            raise ValueError("FirmSlice cap table owner must match firm_id")

    def cash_to_wage_bill(self):
        return safe_ratio(self.cash, self.wage_bill)


def relative_price_factors(firm_count, spread):
    if firm_count <= 1 or spread <= 0:
        return [1.0 for _ in range(firm_count)]

    midpoint = (firm_count - 1) / 2

    return [
        1.0 + spread * (index - midpoint) / max(1.0, midpoint)
        for index in range(firm_count)
    ]


def balanced_employee_groups(world, firm_count):
    groups = [[] for _ in range(firm_count)]
    capacities = [0.0 for _ in range(firm_count)]
    workers = []
    household_dict = world.household_dict

    for person in world.population:
        if not person.alive or person.household_id not in household_dict:
            continue

        productivity = age_productivity(person.age)

        if productivity <= 0 or not world.labor_participates(person):
            continue

        workers.append((productivity, person.id))

    workers.sort(reverse=True)

    for productivity, person_id in workers:
        index = min(range(firm_count), key=lambda item: capacities[item])
        groups[index].append(person_id)
        capacities[index] += productivity

    return groups, capacities


def split_values(total, shares):
    values = []
    assigned = 0.0

    for index, share in enumerate(shares):
        if index == len(shares) - 1:
            value = total - assigned
        else:
            value = total * share
            assigned += value

        values.append(value)

    return values


def allocate_inventory_constrained_purchase(
    requested_units,
    target_shares,
    available_inventory,
    tolerance=1e-12,
):
    """Allocate a purchase against actual donor inventory deterministically."""
    requested = max(0.0, float(requested_units))
    available = [max(0.0, float(value)) for value in available_inventory]
    if len(target_shares) != len(available):
        raise ValueError("target_shares and available_inventory must have equal length")
    if not available or requested <= tolerance:
        return [0.0 for _ in available]
    weights = [max(0.0, float(value)) for value in target_shares]
    weight_total = math.fsum(weights)
    if weight_total <= tolerance:
        weights = [1.0 for _ in available]
        weight_total = float(len(available))
    weights = [value / weight_total for value in weights]
    executed_limit = min(requested, math.fsum(available))
    actual = [
        min(available[index], executed_limit * weights[index])
        for index in range(len(available))
    ]
    remainder = max(0.0, executed_limit - math.fsum(actual))
    for _ in range(len(available) + 1):
        if remainder <= tolerance:
            break
        eligible = [
            index for index in range(len(available))
            if available[index] - actual[index] > tolerance
        ]
        if not eligible:
            break
        eligible_weight = math.fsum(weights[index] for index in eligible)
        residual_total = math.fsum(
            available[index] - actual[index] for index in eligible
        )
        additions = (
            {
                index: remainder * weights[index] / eligible_weight
                for index in eligible
            }
            if eligible_weight > tolerance
            else {
                index: remainder
                * (available[index] - actual[index])
                / residual_total
                for index in eligible
            }
        )
        added = 0.0
        for index in eligible:
            quantity = min(
                available[index] - actual[index],
                max(0.0, additions[index]),
            )
            actual[index] += quantity
            added += quantity
        if added <= tolerance:
            break
        remainder -= added
    if remainder > tolerance:
        for index in range(len(available)):
            quantity = min(available[index] - actual[index], remainder)
            actual[index] += max(0.0, quantity)
            remainder -= max(0.0, quantity)
            if remainder <= tolerance:
                break
    return [
        min(available[index], max(0.0, actual[index]))
        for index in range(len(available))
    ]


def split_single_firm(world, firm_count):
    if firm_count < 1:
        raise ValueError("firm_count must be at least 1")

    aggregate = world.firm_system
    groups, capacities = balanced_employee_groups(world, firm_count)
    total_capacity = sum(capacities)

    # ``Person.firm_id`` is the authoritative employer relation.  Older
    # single-firm initialization could leave that field pointing at the
    # compatibility Firm even when the person was not in its employee roster
    # (for example, a non-eligible adult).  Clear only those stale references
    # before assigning the new deterministic Firm groups; this does not alter
    # the labor rules, but prevents orphan payroll/employer identities after a
    # multi-Firm split.
    assigned_ids = {
        person_id
        for group in groups
        for person_id in group
    }
    target_firm_ids = set(range(firm_count))
    for person in getattr(world, "population", []):
        if (
            getattr(person, "firm_id", None) in target_firm_ids
            and person.id not in assigned_ids
        ):
            person.firm_id = None

    if total_capacity > 0:
        shares = [capacity / total_capacity for capacity in capacities]
    else:
        shares = [1.0 / firm_count for _ in range(firm_count)]

    cash_values = split_values(aggregate.cash, shares)
    inventory_units_values = split_values(aggregate.food_inventory_units, shares)
    inventory_value_values = split_values(aggregate.food_inventory_value, shares)
    loan_values = split_values(
        aggregate.working_capital_loan_balance,
        shares,
    )
    expected_demand_total = aggregate.food_demand_units
    expected_sales_total = aggregate.expected_sales or 0.0
    expected_demand_values = split_values(expected_demand_total, shares)
    expected_sales_values = split_values(expected_sales_total, shares)
    relative_prices = relative_price_factors(
        firm_count,
        getattr(world, "initial_firm_relative_price_spread", 0.0),
    )

    firms = []

    for firm_id in range(firm_count):
        for person_id in groups[firm_id]:
            person = world.person_dict.get(person_id)

            if person is not None:
                person.firm_id = firm_id

        firm = FirmSlice(
            firm_id=firm_id,
            employee_ids=groups[firm_id],
            share=shares[firm_id],
            productive_capacity=capacities[firm_id],
            cash=cash_values[firm_id],
            inventory_units=inventory_units_values[firm_id],
            inventory_value=inventory_value_values[firm_id],
            expected_demand=expected_demand_values[firm_id],
            expected_sales=expected_sales_values[firm_id],
            production_plan=aggregate.food_output_units * shares[firm_id],
            desired_production=aggregate.food_output_units * shares[firm_id],
            actual_production=aggregate.food_output_units * shares[firm_id],
            previous_actual_production=aggregate.food_output_units * shares[firm_id],
            bootstrap_production_units=None,
            loan_balance=loan_values[firm_id],
            price=aggregate.price * relative_prices[firm_id],
            relative_price=relative_prices[firm_id],
            expected_profit=aggregate.profit_before_dividend * shares[firm_id],
            smoothed_profit=aggregate.profit_before_dividend * shares[firm_id],
            baseline_profit=aggregate.profit_before_dividend * shares[firm_id],
            evaluated_profit=aggregate.profit_before_dividend * shares[firm_id],
            last_test_price=aggregate.price * relative_prices[firm_id],
            expected_share=shares[firm_id],
            firm_rng=random.Random(
                f"{world.seed}:firm:{firm_id}"
                if world.seed is not None
                else None
            ),
            **default_firm_identity(),
        )
        if getattr(world, "deterministic_review_phase_staggering_enabled", False):
            production_cadence = max(
                1,
                int(config.FIRM_PRODUCTION_REVIEW_INTERVAL_WEEKS),
            )
            investment_cadence = max(
                1,
                int(config.CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS),
            )
            firm.production_review_phase_offset = deterministic_review_phase(
                firm_id,
                production_cadence,
                "production",
            )
            firm.production_review_timer = firm.production_review_phase_offset
            firm.investment_review_phase_offset = deterministic_review_phase(
                firm_id,
                investment_cadence,
                "investment",
            )
        firms.append(firm)

    world.firms = firms
    world.firm_dict = {firm.firm_id: firm for firm in firms}
    world.firm_by_id = world.firm_dict
    world.firm_count = firm_count
    world.sync_firms_from_aggregate()


def ensure_single_firm_view(world):
    if getattr(world, "firms", None):
        return

    world.firms = []
    world.firm_count = 1
    split_single_firm(world, 1)
