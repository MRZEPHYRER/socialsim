"""Small, explicitly gated canonical investment screen for Step 15I.4.

This module is deliberately narrow.  It owns the first active capital-good
sector and retained-cash investment transaction, but it does not provide
credit, equity finance, depreciation, or autonomous portfolio behavior.
"""

from dataclasses import dataclass, field, replace
import math

from economy import config
from economy.capital_goods import (
    CapitalGoodInventory,
    CapitalGoodSpec,
    CapitalGoodSupplyAdapter,
    InvestmentOrder,
)
from economy.investment_planning import ShadowInvestmentPlanner
from economy.multi_firm import FirmSlice
from economy.multisector import (
    CAPITAL_GOODS_SECTOR_ID,
    CAPITAL_GOODS_TECHNOLOGY_ID,
    CAPITAL_GOOD_INVENTORY_POLICY_ID,
    CAPITAL_MACHINE_GOOD_ID,
    default_firm_identity,
)
from economy.ownership_accounting import CapTable
from economy.scheduling import review_due
from productivity import age_productivity


PIPELINE_QUANTITY_TOLERANCE = 1e-9


@dataclass
class CanonicalInvestmentWeek:
    step: int
    capital_good_production: float = 0.0
    capital_good_sales: float = 0.0
    capital_good_revenue: float = 0.0
    capital_good_wages: float = 0.0
    fixed_investment: float = 0.0
    replacement_investment: float = 0.0
    expansion_investment: float = 0.0
    depreciation_expense: float = 0.0
    retired_capacity: float = 0.0
    investment_orders: int = 0
    customer_advances_received: float = 0.0
    customer_advances_delivered: float = 0.0
    prepaid_investment_cash_paid: float = 0.0
    investment_events: list = field(default_factory=list)


class CanonicalInvestmentSystem:
    """Deterministic capital supplier and internal-cash settlement layer."""

    VERSION = 1

    def __init__(self, world):
        self.world = world
        self.enabled = bool(
            getattr(world, "canonical_investment_enabled", False)
        )
        self.capital_good_firm_count = max(
            1,
            int(getattr(world, "capital_good_firm_count", config.CAPITAL_GOOD_FIRM_COUNT)),
        )
        self.startup_cash_fraction = max(
            0.0,
            float(
                getattr(
                    world,
                    "capital_good_startup_cash_fraction",
                    config.CAPITAL_GOOD_STARTUP_CASH_FRACTION,
                )
            ),
        )
        self.labor_share = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        world,
                        "capital_good_labor_share",
                        config.CAPITAL_GOOD_LABOR_SHARE,
                    )
                ),
            ),
        )
        self.productivity_per_labor = max(
            0.0,
            float(
                getattr(
                    world,
                    "capital_good_fixture_productivity_per_labor",
                    config.CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR,
                )
            ),
        )
        configured_unit_price = max(
            1e-12,
            float(
                getattr(
                    world,
                    "capital_good_unit_price",
                    config.CAPITAL_GOOD_UNIT_PRICE,
                )
            ),
        )
        self.offer_price_mode = str(
            getattr(world, "capital_good_offer_price_mode", "fixed_engineering")
        ).strip().lower()
        if self.offer_price_mode == "cost_anchored":
            # Screen-only price semantics: quote equals labor-based historical
            # unit cost, with no markup.
            self.unit_price = max(
                1e-12,
                float(config.FIRM_WAGE_PER_LABOR)
                / max(self.productivity_per_labor, 1e-12),
            )
        else:
            self.unit_price = configured_unit_price
        self.review_interval = max(
            1,
            int(
                getattr(
                    world,
                    "capital_good_investment_review_interval_weeks",
                    config.CAPITAL_GOOD_INVESTMENT_REVIEW_INTERVAL_WEEKS,
                )
            ),
        )
        # The service horizon converts a supplier backlog stock (and the
        # review-period additions to it) into a weekly production flow.  It
        # intentionally defaults to the historical 13-week value but is not
        # a buyer decision-frequency parameter.
        self.backlog_service_horizon = max(
            1,
            int(
                getattr(
                    world,
                    "capital_good_backlog_service_horizon_weeks",
                    config.CAPITAL_GOOD_BACKLOG_SERVICE_HORIZON_WEEKS,
                )
            ),
        )
        self.committed_capital_planner_enabled = bool(
            getattr(world, "capital_good_committed_capital_planner_enabled", False)
        )
        self.pipeline_depletion_early_review_enabled = bool(
            getattr(
                world,
                "capital_good_pipeline_depletion_early_review_enabled",
                False,
            )
        )
        self.early_investment_review_events = []
        self.pipeline_depleted_last_step = False
        self.pipeline_has_accepted_orders = False
        self.planner = ShadowInvestmentPlanner()
        self.supply_adapter = CapitalGoodSupplyAdapter()
        self.capital_good_firms = []
        self.capital_good_firm_ids = set()
        self.initialized = False
        self.pending_household_wages = {}
        self.labor_release_events = []
        self.labor_reactivation_events = []
        self.expansion_backlog_by_firm = {}
        # Existing backlog is a stock; new review demand is carried separately
        # as a one-period flow for the next production plan.
        self.pending_expansion_demand_flow = 0.0
        self.opening_replacement_backlog_units = 0.0
        self.opening_expansion_backlog_units = 0.0
        self.opening_total_backlog_units = 0.0
        self.lifecycle_enabled = bool(
            getattr(world, "capital_lifecycle_enabled", False)
        )
        self.customer_advance_enabled = bool(
            getattr(world, "capital_good_customer_advance_enabled", False)
        )
        # This is deliberately an explicit engineering screen.  It is not a
        # negotiated trade-credit term and does not invoke Step 13 finance.
        self.customer_advance_fraction = 1.0
        self.customer_advance_pending_orders = {}
        self.customer_advance_ledger = []
        self.useful_life_weeks = max(
            1.0,
            float(getattr(world, "capital_lifecycle_useful_life_weeks", 52.0)),
        )
        self.last_week = CanonicalInvestmentWeek(step=0)

    def _ensure_customer_advance_state(self):
        """Backfill neutral state on checkpoints created before Step 15I.11D."""
        if not hasattr(self, "backlog_service_horizon"):
            self.backlog_service_horizon = max(
                1,
                int(
                    getattr(
                        self.world,
                        "capital_good_backlog_service_horizon_weeks",
                        config.CAPITAL_GOOD_BACKLOG_SERVICE_HORIZON_WEEKS,
                    )
                ),
            )
        if not hasattr(self, "committed_capital_planner_enabled"):
            self.committed_capital_planner_enabled = bool(
                getattr(
                    self.world,
                    "capital_good_committed_capital_planner_enabled",
                    False,
                )
            )
        if not hasattr(self, "pipeline_depletion_early_review_enabled"):
            self.pipeline_depletion_early_review_enabled = bool(
                getattr(
                    self.world,
                    "capital_good_pipeline_depletion_early_review_enabled",
                    False,
                )
            )
        if not hasattr(self, "early_investment_review_events"):
            self.early_investment_review_events = []
        if not hasattr(self, "pipeline_depleted_last_step"):
            self.pipeline_depleted_last_step = False
        if not hasattr(self, "pipeline_has_accepted_orders"):
            self.pipeline_has_accepted_orders = False
        if not hasattr(self, "customer_advance_enabled"):
            self.customer_advance_enabled = bool(
                getattr(self.world, "capital_good_customer_advance_enabled", False)
            )
        if not hasattr(self, "customer_advance_fraction"):
            self.customer_advance_fraction = 1.0
        if not hasattr(self, "customer_advance_pending_orders"):
            self.customer_advance_pending_orders = {}
        if not hasattr(self, "customer_advance_ledger"):
            self.customer_advance_ledger = []

    def operating_firms(self):
        return [*getattr(self.world, "firms", []), *self.capital_good_firms]

    def _rebuild_lookup(self):
        self.world.firm_dict = {
            firm.firm_id: firm for firm in self.operating_firms()
        }

        self.world.firm_by_id = self.world.firm_dict
        self.world.capital_good_firms = self.capital_good_firms
        self.world.capital_good_firm_ids = set(self.capital_good_firm_ids)

    def ensure_firms(self):
        if self.initialized:
            return
        if not self.enabled:
            self._rebuild_lookup()
            self.initialized = True
            return

        food_firms = list(getattr(self.world, "firms", []))
        if not food_firms:
            self.initialized = True
            return

        total_workers = [
            person
            for person in getattr(self.world, "population", [])
            if getattr(person, "alive", False)
            and self.world.has_valid_settlement_household(person)
            and self.world.labor_participates(person)
        ]
        total_workers.sort(key=lambda person: (person.id,))
        requested_workers = int(len(total_workers) * self.labor_share)
        requested_workers = max(1, requested_workers) if total_workers else 0
        selected = total_workers[:requested_workers]

        groups = [[] for _ in range(self.capital_good_firm_count)]
        for index, person in enumerate(selected):
            groups[index % self.capital_good_firm_count].append(person.id)
        selected_ids = {person.id for person in selected}
        # A worker transferred into the capital-good sector must be removed
        # from the source Food roster as well as receiving the new employer id.
        # Otherwise the same Person is present in two Firm rosters while only
        # one employer is authoritative.
        for food_firm in food_firms:
            food_firm.employee_ids = [
                person_id
                for person_id in food_firm.employee_ids
                if person_id not in selected_ids
            ]

        startup_total = math.fsum(
            max(0.0, float(getattr(firm, "cash", 0.0)))
            for firm in food_firms
        ) * self.startup_cash_fraction
        startup_by_firm = startup_total / self.capital_good_firm_count

        for firm_index in range(self.capital_good_firm_count):
            firm_id = 100000 + firm_index
            firm = FirmSlice(
                firm_id=firm_id,
                employee_ids=groups[firm_index],
                share=0.0,
                cash=startup_by_firm,
                price=self.unit_price,
                relative_price=1.0,
                capital_good_inventory=CapitalGoodInventory(
                    supplier_firm_id=firm_id,
                    good_id=CAPITAL_MACHINE_GOOD_ID,
                ),
                **default_firm_identity(),
            )
            firm.sector_id = CAPITAL_GOODS_SECTOR_ID
            firm.technology_id = CAPITAL_GOODS_TECHNOLOGY_ID
            firm.inventory_policy_id = CAPITAL_GOOD_INVENTORY_POLICY_ID
            firm.output_good_id = CAPITAL_MACHINE_GOOD_ID
            firm.capital_good_unit_production_cost = 0.0
            firm.capital_good_labor_services = 0.0
            firm.customer_advance_liability = 0.0
            firm.prepaid_capital_investment_asset = 0.0
            firm.desired_labor = sum(
                age_productivity(
                    self.world.person_dict[person_id].age
                )
                for person_id in groups[firm_index]
                if person_id in self.world.person_dict
            )
            firm.startup_capitalization_inflow_this_step = startup_by_firm
            firm.startup_capitalization_outflow_this_step = 0.0
            firm.paid_in_equity = 0.0
            firm.cap_table = CapTable.initial_legacy_owned(firm_id)
            self.capital_good_firms.append(firm)
            self.capital_good_firm_ids.add(firm_id)

        # Startup cash is an internal capitalization transfer from existing
        # Food Firm cash, never new money.
        remaining = startup_total
        for firm in food_firms:
            transfer = min(max(0.0, float(getattr(firm, "cash", 0.0))), remaining)
            firm.cash -= transfer
            remaining -= transfer
            if remaining <= 1e-12:
                break
        self.world.firm_system.cash -= startup_total
        selected_firm_by_person = {
            person_id: 100000 + group_index
            for group_index, group in enumerate(groups)
            for person_id in group
        }
        for person in selected:
            person.firm_id = selected_firm_by_person[person.id]
        self._rebuild_lookup()
        # The World registry is current after all capital-good Firms exist.
        if hasattr(self.world, "assign_bootstrap_founders"):
            self.world.assign_bootstrap_founders()
        self.initialized = True

    def _release_excess_labor(self, firm, desired_labor, step):
        """Release whole workers until assigned capacity fits the requirement."""
        desired = max(0.0, float(desired_labor))
        employee_ids = list(getattr(firm, "employee_ids", []))
        contributions = {
            person_id: max(
                0.0,
                age_productivity(
                    self.world.person_dict[person_id].age
                ),
            )
            for person_id in employee_ids
            if person_id in self.world.person_dict
            and getattr(self.world.person_dict[person_id], "alive", False)
            and self.world.labor_participates(self.world.person_dict[person_id])
        }
        assigned_capacity = math.fsum(contributions.values())
        employment_before = len(employee_ids)
        released = []
        # Keep earlier roster members deterministically and release from the
        # end.  A worker is released only when removing the whole worker still
        # leaves capacity at or above the desired requirement.
        for person_id in reversed(employee_ids):
            if assigned_capacity <= desired + 1e-12:
                break
            contribution = contributions.get(person_id, 0.0)
            if assigned_capacity - contribution + 1e-12 < desired:
                continue
            person = self.world.person_dict.get(person_id)
            if person is None:
                continue
            assigned_capacity -= contribution
            released.append(person_id)
            person.firm_id = None
        if released:
            released_set = set(released)
            firm.employee_ids = [
                person_id
                for person_id in employee_ids
                if person_id not in released_set
            ]
            employment_after = len(firm.employee_ids)
            for person_id in released:
                self.labor_release_events.append({
                    "global_step": step,
                    "event_type": "capital_good_labor_release",
                    "firm_id": firm.firm_id,
                    "person_id": person_id,
                    "desired_labor": desired,
                    "assigned_capacity_before": assigned_capacity
                    + math.fsum(contributions.get(item, 0.0) for item in released),
                    "assigned_capacity_after": assigned_capacity,
                    "employment_before": employment_before,
                    "employment_after": employment_after,
                    "person_firm_id_after": None,
                    "reason": "funded_production_requirement_fell",
                })
        firm.desired_labor = desired
        return released

    def _close_capital_good_labor(self, step):
        """Keep capital-good labor tied to the next weekly flow plan."""
        outstanding = self._capital_good_outstanding_demand()
        divisor = max(1, len(self.capital_good_firms))
        next_flow = max(
            0.0,
            float(self.pending_expansion_demand_flow)
            / max(1, self.backlog_service_horizon),
        )
        desired_output = (
            outstanding / max(1, self.backlog_service_horizon) + next_flow
        )
        for firm in self.capital_good_firms:
            firm.capital_good_outstanding_demand_units = outstanding / divisor
            firm.capital_good_backlog_flow_units = (
                outstanding / max(1, self.backlog_service_horizon) / divisor
            )
            firm.capital_good_new_demand_flow_units = next_flow / divisor
            desired_labor = (
                desired_output / divisor / self.productivity_per_labor
                if desired_output > 1e-12
                and self.productivity_per_labor > 1e-12
                else 0.0
            )
            self._release_excess_labor(firm, desired_labor, step)

    def _capital_good_outstanding_demand(self):
        replacement = math.fsum(
            max(
                0.0,
                float(getattr(firm, "pending_replacement_capacity_need", 0.0)),
            )
            for firm in getattr(self.world, "firms", [])
        )
        expansion = math.fsum(self.expansion_backlog_by_firm.values())
        return replacement + expansion

    def _activate_capital_good_labor(
        self,
        step,
        new_replacement_demand_flow=0.0,
        new_expansion_demand_flow=0.0,
    ):
        """Convert opening backlog stock into a weekly production flow."""
        opening_backlog = max(0.0, self.opening_total_backlog_units)
        backlog_flow = opening_backlog / max(1, self.backlog_service_horizon)
        new_demand_units = max(
            0.0,
            float(new_replacement_demand_flow)
            + float(new_expansion_demand_flow),
        )
        # Review-period additions are converted to weekly flow units before
        # they enter the production target.
        new_demand_flow = new_demand_units / max(
            1, self.backlog_service_horizon
        )
        desired_output = backlog_flow + new_demand_flow
        self.current_backlog_flow_units = backlog_flow
        self.current_new_demand_flow_units = new_demand_flow
        self.current_desired_output_units = desired_output
        divisor = max(1, len(self.capital_good_firms))
        per_stock = opening_backlog / divisor
        per_backlog_flow = backlog_flow / divisor
        per_new_flow = new_demand_flow / divisor
        per_desired_output = desired_output / divisor
        before_by_firm = {
            firm.firm_id: set(getattr(firm, "employee_ids", []))
            for firm in self.capital_good_firms
        }
        for firm in self.capital_good_firms:
            firm.capital_good_replacement_demand_units = math.fsum(
                max(
                    0.0,
                    float(getattr(food, "pending_replacement_capacity_need", 0.0)),
                )
                for food in getattr(self.world, "firms", [])
            ) / max(1, len(self.capital_good_firms))
            firm.capital_good_expansion_demand_units = math.fsum(
                self.expansion_backlog_by_firm.values()
            ) / max(1, len(self.capital_good_firms))
            firm.capital_good_outstanding_demand_units = per_stock
            firm.capital_good_backlog_flow_units = per_backlog_flow
            firm.capital_good_new_demand_flow_units = per_new_flow
            firm.desired_production = per_desired_output
            firm.capital_good_desired_output = per_desired_output
            firm.desired_labor = (
                per_desired_output / self.productivity_per_labor
                if per_desired_output > 1e-12
                and self.productivity_per_labor > 1e-12
                else 0.0
            )
        if desired_output > 1e-12:
            self.world.assign_unassigned_workers_to_firms(
                include_capital_goods=True
            )
        for firm in self.capital_good_firms:
            before = before_by_firm.get(firm.firm_id, set())
            after = set(getattr(firm, "employee_ids", []))
            for person_id in sorted(after - before):
                self.labor_reactivation_events.append({
                    "global_step": step,
                    "event_type": "capital_good_labor_reactivation",
                    "firm_id": firm.firm_id,
                    "person_id": person_id,
                    "desired_labor": firm.desired_labor,
                    "reason": "outstanding_capital_good_order_backlog",
                })

    def _wage_per_labor(self):
        multiplier = self.world.scenario_multiplier(
            "WAGE_MULTIPLIER_STEP",
            "WAGE_MULTIPLIER",
        )
        return max(0.0, config.FIRM_WAGE_PER_LABOR * multiplier)

    def _reset_lifecycle_fields(self, firm):
        stock = getattr(firm, "capital_stock", None)
        opening_book = (
            float(getattr(stock, "total_remaining_book_value", 0.0))
            if stock is not None
            else 0.0
        )
        firm.capital_book_value_opening_this_step = opening_book
        firm.capital_asset_acquisitions_this_step = 0.0
        firm.capital_asset_disposals_this_step = 0.0
        firm.capital_depreciation_expense_this_step = 0.0
        firm.retired_capacity_this_step = 0.0
        firm.active_capital_asset_count = sum(
            bool(getattr(asset, "is_active", False))
            for asset in getattr(stock, "assets", [])
        ) if stock is not None else 0
        firm.retired_capital_asset_count = (
            len(getattr(stock, "assets", [])) - firm.active_capital_asset_count
            if stock is not None else 0
        )
        firm.executed_replacement_investment_this_step = 0.0
        firm.executed_expansion_investment_this_step = 0.0
        firm.unmet_replacement_investment = 0.0
        firm.unmet_expansion_investment = 0.0
        firm.replacement_investment_need = max(
            0.0, float(getattr(firm, "pending_replacement_capacity_need", 0.0))
        )
        firm.desired_replacement_investment = 0.0
        firm.desired_expansion_investment = 0.0

    def _advance_capital_lifecycle(self, step):
        """Advance accounting age and emit retirement signals only when gated."""
        week = CanonicalInvestmentWeek(step=step)
        for firm in getattr(self.world, "firms", []):
            self._reset_lifecycle_fields(firm)
            stock = getattr(firm, "capital_stock", None)
            if stock is None or not self.lifecycle_enabled:
                continue

            previous_service = stock.capital_service_capacity(
                engineering_capacity_per_unit=1.0
            )
            updated_stock, depreciation = stock.advance_accounting_period(1.0)
            current_service = updated_stock.capital_service_capacity(
                engineering_capacity_per_unit=1.0
            )
            retired_capacity = max(0.0, previous_service - current_service)
            firm.capital_stock = updated_stock
            firm.capital_depreciation_expense_this_step = depreciation
            firm.retired_capacity_this_step = retired_capacity
            firm.pending_replacement_capacity_need = max(
                0.0,
                float(getattr(firm, "pending_replacement_capacity_need", 0.0))
                + retired_capacity,
            )
            firm.replacement_capacity_need = firm.pending_replacement_capacity_need
            firm.active_capital_asset_count = sum(
                bool(asset.is_active) for asset in updated_stock.assets
            )
            firm.retired_capital_asset_count = (
                len(updated_stock.assets) - firm.active_capital_asset_count
            )
            firm.capital_lifecycle_events = [
                {
                    "global_step": step,
                    "firm_id": firm.firm_id,
                    "asset_id": asset.asset_id,
                    "event_type": "asset_retired",
                    "retired_service_capacity": max(
                        0.0,
                        float(getattr(asset, "quantity", 0.0))
                        * float(
                            getattr(
                                asset,
                                "capacity_metadata",
                                {},
                            ).get("capital_service_capacity_per_unit", 0.0)
                            or 0.0
                        ),
                    ),
                    "remaining_book_value": asset.remaining_book_value,
                    "active": asset.is_active,
                }
                for asset in updated_stock.assets
                if asset.is_retired
                and not any(
                    prior.asset_id == asset.asset_id and prior.is_retired
                    for prior in stock.assets
                )
            ]
            for event in firm.capital_lifecycle_events:
                asset = next(
                    (item for item in updated_stock.assets if item.asset_id == event["asset_id"]),
                    None,
                )
                if asset is not None:
                    asset.capacity_metadata["retirement_week"] = step
                self.world.capital_asset_event_rows.append(dict(event))
            week.depreciation_expense += depreciation
            week.retired_capacity += retired_capacity
        return week

    def _pay_and_produce(self, step):
        self.pending_household_wages = {}
        # Read-only payroll provenance consumed by the statistical observer.
        # It does not participate in matching, settlement, or any decision.
        self.pending_person_wages = {}
        week = CanonicalInvestmentWeek(step=step)
        wage_per_labor = self._wage_per_labor()
        for firm in self.capital_good_firms:
            startup_inflow = max(
                0.0,
                float(getattr(firm, "startup_capitalization_inflow_this_step", 0.0)),
            )
            firm.cash_start = max(0.0, firm.cash - startup_inflow)
            if not self.customer_advance_enabled:
                firm.accounting_cash_start_this_step = firm.cash_start
            labor = self.world.firm_employee_capacity(firm)
            affordable_labor = (
                labor
                if wage_per_labor <= 1e-12
                else min(labor, max(0.0, firm.cash) / wage_per_labor)
            )
            # A cash-depleted supplier can leave a sub-tolerance residual
            # labor allocation after division by the wage rate. Treat that
            # numerical residue as zero so it cannot create a wage payment
            # without measurable capital-good output.
            if affordable_labor <= 1e-12:
                affordable_labor = 0.0
            wage = affordable_labor * wage_per_labor
            firm.capital_good_labor_services = affordable_labor
            firm.wage_bill = wage
            firm.wage_payment = wage
            firm.executed_wage_bill = wage
            firm.cash -= wage
            if labor > 1e-12:
                for person_id in firm.employee_ids:
                    person = self.world.person_dict.get(person_id)
                    if (
                        person is None
                        or not person.alive
                        or not self.world.labor_participates(person)
                    ):
                        continue
                    settlement_household = self.world.settlement_household_for_person(person)
                    if settlement_household is None:
                        continue
                    household_id = settlement_household.id
                    share = age_productivity(person.age) / labor
                    scheduled_wage = age_productivity(person.age) * wage_per_labor
                    paid_wage = wage * share
                    self.pending_household_wages[household_id] = (
                        self.pending_household_wages.get(household_id, 0.0)
                        + paid_wage
                    )
                    self.pending_person_wages[person_id] = {
                        "household_id": household_id,
                        "firm_id": getattr(firm, "firm_id", None),
                        "sector_id": getattr(firm, "sector_id", "capital_goods"),
                        "age": float(person.age),
                        "productivity": age_productivity(person.age),
                        "scheduled_wage": scheduled_wage,
                        "paid_wage": paid_wage,
                    }
            production = firm.capital_good_inventory.produce(
                affordable_labor,
                wage,
                self.productivity_per_labor,
            )
            firm.capital_good_production_units = production.units_produced
            firm.production = production.units_produced
            firm.actual_production = production.units_produced
            firm.inventory_units = firm.capital_good_inventory.units
            firm.inventory_value = (
                firm.capital_good_inventory.units * self.unit_price
            )
            firm.capital_good_unit_production_cost = production.unit_production_cost
            firm.cash_end = firm.cash
            firm.sales_revenue = 0.0
            firm.sales = 0.0
            firm.sales_units = 0.0
            firm.loan_issued = 0.0
            firm.loan_repaid = 0.0
            firm.dividend_payment = 0.0
            firm.other_cash_inflow = 0.0
            firm.other_cash_outflow = 0.0
            firm.sales_collections_this_step = 0.0
            firm.startup_capitalization_inflow_this_step = startup_inflow
            firm.startup_capitalization_outflow_this_step = 0.0
            week.capital_good_production += production.units_produced
            week.capital_good_wages += wage
        self.last_week = week
        return week

    def _protected_liquidity(self, firm):
        existing = max(0.0, float(getattr(firm, "target_cash", 0.0)))
        if existing > 0.0:
            return existing
        wage_bill = max(0.0, float(getattr(firm, "scheduled_wage_bill", 0.0)))
        return max(
            0.0,
            float(getattr(self.world.firm_system.central_bank, "firm_cash_buffer", 0.0))
            + wage_bill,
        )

    def _committed_capacity_by_firm(self):
        """Return accepted, undelivered service capacity by investment class."""
        committed = {}
        if not self.customer_advance_enabled:
            return committed
        for order in self.customer_advance_pending_orders.values():
            buyer_id = order.buyer_firm_id
            units = max(0.0, float(getattr(order, "desired_units", 0.0)))
            if units <= PIPELINE_QUANTITY_TOLERANCE:
                continue
            metadata = dict(getattr(order, "metadata", {}) or {})
            replacement_value = max(
                0.0,
                float(metadata.get("replacement_investment_expenditure", 0.0)),
            )
            expansion_value = max(
                0.0,
                float(metadata.get("expansion_investment_expenditure", 0.0)),
            )
            classified_value = replacement_value + expansion_value
            replacement_share = (
                replacement_value / classified_value
                if classified_value > 1e-12 else 0.0
            )
            entry = committed.setdefault(
                buyer_id,
                {"expansion": 0.0, "replacement": 0.0, "total": 0.0},
            )
            replacement_units = units * replacement_share
            entry["replacement"] += replacement_units
            entry["expansion"] += units - replacement_units
            entry["total"] += units
        return committed

    def _build_orders(self, step):
        orders = []
        firms = list(getattr(self.world, "firms", []))
        committed_by_firm = self._committed_capacity_by_firm()
        pipeline_empty = not any(
            float(values.get("total", 0.0)) > PIPELINE_QUANTITY_TOLERANCE
            for values in committed_by_firm.values()
        )
        for firm in firms:
            firm.investment_reviewed_this_step = False
        staggered = bool(
            getattr(
                self.world,
                "deterministic_review_phase_staggering_enabled",
                False,
            )
        )
        for firm in firms:
            phase = (
                int(getattr(firm, "investment_review_phase_offset", 0))
                if staggered
                else 0
            )
            scheduled_review = review_due(step, self.review_interval, phase)
            early_candidate = (
                self.pipeline_depletion_early_review_enabled
                and pipeline_empty
                and self.pipeline_has_accepted_orders
                and not scheduled_review
            )
            if not scheduled_review and not early_candidate:
                continue
            labor_capacity = self.world.firm_employee_capacity(firm)
            labor_output_capacity = labor_capacity * self.world.firm_system.food_productivity
            desired_output = max(
                0.0,
                float(getattr(firm, "desired_production", 0.0)),
            )
            pending_replacement = max(
                0.0,
                float(getattr(firm, "pending_replacement_capacity_need", 0.0)),
            )
            planning_desired_capacity = max(
                desired_output,
                labor_output_capacity + pending_replacement,
            )
            capital_stock = getattr(firm, "capital_stock", None)
            current_capital = (
                capital_stock.capital_service_capacity(
                    engineering_capacity_per_unit=1.0
                )
                if capital_stock is not None
                else 0.0
            )
            committed = committed_by_firm.get(
                firm.firm_id,
                {"expansion": 0.0, "replacement": 0.0, "total": 0.0},
            )
            use_committed_capacity = (
                self.committed_capital_planner_enabled or early_candidate
            )
            committed_expansion = (
                float(committed["expansion"])
                if use_committed_capacity else 0.0
            )
            committed_replacement = (
                float(committed["replacement"])
                if use_committed_capacity else 0.0
            )
            decision = self.planner.decide(
                firm_id=firm.firm_id,
                expected_demand=max(0.0, float(getattr(firm, "expected_demand", 0.0))),
                desired_output=desired_output,
                current_effective_capacity=labor_output_capacity,
                desired_capacity=planning_desired_capacity,
                non_capital_capacity=labor_output_capacity,
                current_capital_capacity=current_capital,
                committed_expansion_capacity=committed_expansion,
                committed_replacement_capacity=committed_replacement,
                replacement_capacity_loss=pending_replacement,
                capital_technology_available=True,
                expansion_cost_per_capacity=self.unit_price,
                replacement_cost_per_capacity=self.unit_price,
                cash=max(0.0, float(getattr(firm, "cash", 0.0))),
                operating_liquidity_floor=self._protected_liquidity(firm),
                capacity_metadata={
                    "capital_capacity_combination": "CAPITAL_AUGMENTED_CAP",
                    "technology_mode": "screen_only",
                },
            )
            firm.expected_demand = decision.expected_demand
            firm.labor_capacity = labor_output_capacity
            firm.feasible_capacity = min(
                labor_output_capacity,
                decision.current_capital_capacity,
            )
            firm.desired_capacity = decision.desired_capacity
            firm.capacity_gap = decision.capacity_gap
            firm.current_capital_capacity = decision.current_capital_capacity
            firm.committed_expansion_capacity = decision.committed_expansion_capacity
            firm.committed_replacement_capacity = decision.committed_replacement_capacity
            firm.committed_future_capital_capacity = (
                decision.committed_expansion_capacity
                + decision.committed_replacement_capacity
            )
            firm.effective_future_capital_capacity = (
                decision.effective_future_capital_capacity
            )
            firm.residual_uncommitted_capacity_gap = (
                decision.residual_uncommitted_capacity_gap
            )
            firm.desired_capital_capacity = decision.desired_capital_capacity
            firm.expansion_investment_need = decision.expansion_investment_need
            firm.replacement_investment_need = decision.replacement_investment_need
            firm.desired_investment_expenditure = decision.desired_investment_expenditure
            firm.investment_financing_gap = decision.financing_gap
            firm.investment_decision_reason = decision.decision_reason
            firm.desired_replacement_investment = decision.replacement_investment_expenditure
            current_expansion_units = (
                decision.expansion_investment_expenditure / self.unit_price
            )
            opening_expansion_units = self.expansion_backlog_by_firm.get(
                firm.firm_id,
                0.0,
            )
            # Only genuinely new expansion demand becomes a next-period flow;
            # the pre-existing expansion stock is never re-added each week.
            new_expansion_units = max(
                0.0,
                current_expansion_units - opening_expansion_units,
            )
            self.pending_expansion_demand_flow += new_expansion_units
            outstanding_expansion_units = max(
                opening_expansion_units,
                current_expansion_units,
            )
            self.expansion_backlog_by_firm[firm.firm_id] = (
                outstanding_expansion_units
            )
            expansion_expenditure = outstanding_expansion_units * self.unit_price
            firm.desired_expansion_investment = expansion_expenditure
            firm.replacement_capacity_need = decision.replacement_investment_need
            desired_total_expenditure = (
                decision.replacement_investment_expenditure
                + expansion_expenditure
            )
            firm.desired_investment_expenditure = desired_total_expenditure
            firm.investable_cash = max(
                0.0,
                firm.cash - self._protected_liquidity(firm),
            )
            # Recompute after the cash view is attached; this remains an
            # internal-cash-only diagnostic and never invokes credit.
            firm.investment_financing_gap = max(
                0.0,
                desired_total_expenditure - firm.investable_cash,
            )
            if early_candidate and (
                decision.residual_uncommitted_capacity_gap
                <= PIPELINE_QUANTITY_TOLERANCE
                or firm.investment_financing_gap > 1e-12
            ):
                continue
            firm.investment_reviewed_this_step = True
            firm.investment_review_count = int(
                getattr(firm, "investment_review_count", 0)
            ) + 1
            if early_candidate:
                self.early_investment_review_events.append({
                    "global_step": step,
                    "firm_id": firm.firm_id,
                    "trigger_reason": "depleted_historical_pipeline_positive_residual_need",
                    "gross_capacity_gap": max(
                        0.0,
                        decision.desired_capital_capacity
                        - decision.current_capital_capacity,
                    ),
                    "committed_expansion_capacity": (
                        decision.committed_expansion_capacity
                    ),
                    "committed_replacement_capacity": (
                        decision.committed_replacement_capacity
                    ),
                    "residual_uncommitted_capacity_gap": (
                        decision.residual_uncommitted_capacity_gap
                    ),
                    "desired_investment_expenditure": desired_total_expenditure,
                    "available_internal_cash": firm.investable_cash,
                    "scheduled_review": False,
                })
            if desired_total_expenditure <= 1e-12:
                continue
            units = desired_total_expenditure / self.unit_price
            if self.customer_advance_enabled:
                # An accepted but not-yet-delivered order already represents
                # part of this backlog.  Do not ask the buyer to prepay it a
                # second time at the next review.
                pending_units = math.fsum(
                    max(0.0, float(order.desired_units))
                    for order in self.customer_advance_pending_orders.values()
                    if order.buyer_firm_id == firm.firm_id
                )
                units = max(0.0, units - pending_units)
                desired_total_expenditure = units * self.unit_price
                if units <= 1e-12:
                    continue
                scale = (
                    desired_total_expenditure
                    / max(
                        1e-12,
                        decision.replacement_investment_expenditure
                        + expansion_expenditure,
                    )
                )
            else:
                scale = 1.0
            orders.append(
                InvestmentOrder(
                    order_id=f"investment:{step}:{firm.firm_id}",
                    buyer_firm_id=firm.firm_id,
                    investment_good_id=CAPITAL_MACHINE_GOOD_ID,
                    desired_units=units,
                    desired_investment_expenditure=desired_total_expenditure,
                    financing_source="retained_cash",
                    protected_operating_liquidity=self._protected_liquidity(firm),
                    metadata={
                        "demand_class": "firm_fixed_investment",
                        "investment_source": (
                            "REPLACEMENT" if decision.replacement_investment_expenditure > expansion_expenditure
                            else "EXPANSION" if expansion_expenditure > decision.replacement_investment_expenditure
                            else "MIXED"
                        ),
                        "replacement_trigger_id": (
                            f"replacement:{step}:{firm.firm_id}"
                            if decision.replacement_investment_expenditure > 1e-12 else ""
                        ),
                        "replacement_capacity_need": decision.replacement_investment_need,
                        "expansion_capacity_need": outstanding_expansion_units,
                        "replacement_investment_expenditure": decision.replacement_investment_expenditure * scale,
                        "expansion_investment_expenditure": expansion_expenditure * scale,
                    },
                )
            )
        # A historical accepted order distinguishes a genuinely depleted
        # pipeline from an empty startup state.  Once an early review creates
        # an order, the non-empty pipeline blocks further early reviews until
        # delivery clears it again.
        self.pipeline_depleted_last_step = False
        return orders

    def _collect_customer_advances(self, step, orders):
        """Collect full advances for the newly accepted order portion.

        The buyer's cash and prepaid asset move immediately; the supplier's
        cash and contract liability move immediately.  No revenue or capital
        asset is recognized until the ordinary inventory settlement delivers
        units.
        """
        self._ensure_customer_advance_state()
        if not self.customer_advance_enabled:
            return []
        accepted = []
        suppliers = sorted(self.capital_good_firms, key=lambda item: str(item.firm_id))
        supplier = suppliers[0] if suppliers else None
        if supplier is None:
            return accepted
        for order in orders:
            buyer = self.world.get_firm_by_id(order.buyer_firm_id)
            if buyer is None:
                continue
            price = self.unit_price
            protected = min(
                max(0.0, float(getattr(buyer, "cash", 0.0))),
                max(0.0, float(order.protected_operating_liquidity)),
            )
            spendable = max(0.0, float(getattr(buyer, "cash", 0.0)) - protected)
            accepted_units = min(
                max(0.0, float(order.desired_units)),
                spendable / price if price > 1e-12 else 0.0,
            ) * self.customer_advance_fraction
            accepted_amount = accepted_units * price
            if accepted_amount <= 1e-12:
                continue
            accepted_order = replace(
                order,
                desired_units=accepted_units,
                desired_investment_expenditure=accepted_amount,
                metadata={
                    **dict(order.metadata),
                    "customer_advance_enabled": True,
                    "customer_advance_paid": accepted_amount,
                    "customer_advance_fraction": self.customer_advance_fraction,
                },
            )
            accepted.append(accepted_order)
            self.pipeline_has_accepted_orders = True
            buyer.cash -= accepted_amount
            buyer.prepaid_capital_investment_asset = (
                max(0.0, float(getattr(buyer, "prepaid_capital_investment_asset", 0.0)))
                + accepted_amount
            )
            buyer.prepaid_capital_investment_paid_this_step = (
                max(0.0, float(getattr(buyer, "prepaid_capital_investment_paid_this_step", 0.0)))
                + accepted_amount
            )
            buyer.investment_cash_outflow_this_step = (
                max(0.0, float(getattr(buyer, "investment_cash_outflow_this_step", 0.0)))
                + accepted_amount
            )
            supplier.cash += accepted_amount
            supplier.customer_advance_liability = (
                max(0.0, float(getattr(supplier, "customer_advance_liability", 0.0)))
                + accepted_amount
            )
            supplier.customer_advance_received_this_step = (
                max(0.0, float(getattr(supplier, "customer_advance_received_this_step", 0.0)))
                + accepted_amount
            )
            self.customer_advance_pending_orders[accepted_order.order_id] = accepted_order
            self.customer_advance_ledger.append({
                "global_step": step,
                "event_type": "customer_advance_received",
                "order_id": accepted_order.order_id,
                "buyer_firm_id": buyer.firm_id,
                "supplier_firm_id": supplier.firm_id,
                "physical_units": accepted_units,
                "cash_amount": accepted_amount,
                "buyer_cash_change": -accepted_amount,
                "supplier_cash_change": accepted_amount,
                "buyer_prepaid_asset_change": accepted_amount,
                "supplier_advance_liability_change": accepted_amount,
                "revenue_recognized": 0.0,
                "capital_asset_created": False,
                "capital_asset_recognized_value": 0.0,
            })
        return accepted

    def _settle_orders(self, step, orders):
        if not orders:
            return
        self._ensure_customer_advance_state()
        opening_pending_units = math.fsum(
            max(0.0, float(order.desired_units))
            for order in self.customer_advance_pending_orders.values()
        )
        if self.customer_advance_enabled:
            protected_by_buyer = {}
            for order in orders:
                protected_by_buyer[order.buyer_firm_id] = max(
                    protected_by_buyer.get(order.buyer_firm_id, 0.0),
                    float(order.protected_operating_liquidity),
                )
            buyer_cash = {
                firm.firm_id: max(
                    0.0,
                    float(getattr(firm, "prepaid_capital_investment_asset", 0.0))
                    + protected_by_buyer.get(firm.firm_id, 0.0),
                )
                for firm in getattr(self.world, "firms", [])
            }
        else:
            buyer_cash = {
                firm.firm_id: max(0.0, float(firm.cash))
                for firm in getattr(self.world, "firms", [])
            }
        supplier_cash = {
            firm.firm_id: max(0.0, float(firm.cash))
            for firm in self.capital_good_firms
        }
        inventories = {
            firm.firm_id: firm.capital_good_inventory
            for firm in self.capital_good_firms
        }
        unit_prices = {firm.firm_id: self.unit_price for firm in self.capital_good_firms}
        base_spec = self.world.ensure_multisector_foundation_contracts().capital_good_contract
        spec = CapitalGoodSpec(
            good_id=base_spec.good_id,
            storable=base_spec.storable,
            capital_asset_class=base_spec.capital_asset_class,
            producer_technology_ref=base_spec.producer_technology_ref,
            metadata={
                **dict(base_spec.metadata),
                "capital_service_capacity_per_unit": 1.0,
                "capacity_normalization_status": "NON-CALIBRATED_FIXTURE",
                **(
                    {
                        "service_life_metadata": {
                            "useful_life_weeks": self.useful_life_weeks,
                            "calibration_status": "NON_CALIBRATED_ENGINEERING_SCREEN",
                        },
                        "depreciation_policy_ref": "straight_line_engineering_screen",
                    }
                    if self.lifecycle_enabled
                    else {}
                ),
            },
        )
        batch = self.supply_adapter.settle(
            spec,
            orders,
            inventories,
            unit_prices,
            buyer_cash,
            supplier_cash,
        )
        orders_by_id = {order.order_id: order for order in orders}
        prepaid_before_by_buyer = {
            firm.firm_id: max(
                0.0,
                float(getattr(firm, "prepaid_capital_investment_asset", 0.0)),
            )
            for firm in getattr(self.world, "firms", [])
        }
        delivered_by_buyer = {}
        delivered_by_supplier = {}
        for settlement in batch.settlements:
            delivered_by_buyer[settlement.buyer_firm_id] = (
                delivered_by_buyer.get(settlement.buyer_firm_id, 0.0)
                + settlement.settled_expenditure
            )
            for fill in settlement.fills:
                delivered_by_supplier[fill.supplier_firm_id] = (
                    delivered_by_supplier.get(fill.supplier_firm_id, 0.0)
                    + fill.supplier_revenue
                )
        for settlement in batch.settlements:
            buyer = self.world.get_firm_by_id(settlement.buyer_firm_id)
            if buyer is None:
                continue
            if self.customer_advance_enabled:
                prepaid_before = prepaid_before_by_buyer.get(buyer.firm_id, 0.0)
                delivered_total = delivered_by_buyer.get(buyer.firm_id, 0.0)
                remaining_prepaid = max(0.0, prepaid_before - delivered_total)
                # The settlement adapter sees the prepaid asset as synthetic
                # spendable cash.  Remove that synthetic balance again so the
                # delivery itself cannot charge the buyer a second time.
                # Delivery consumes the already-paid prepaid asset only; the
                # buyer's actual cash was reduced at acceptance and is not
                # touched again here.
                buyer.cash = float(getattr(buyer, "cash", 0.0))
                buyer.prepaid_capital_investment_asset = remaining_prepaid
                buyer.prepaid_capital_investment_capitalized_this_step = delivered_total
                buyer.investment_cash_outflow_this_step = 0.0
            else:
                buyer.cash = batch.buyer_cash_end[buyer.firm_id]
            order_metadata = dict(
                getattr(orders_by_id.get(settlement.order_id), "metadata", {})
            )
            self.world.capital_asset_event_rows.append({
                "global_step": step,
                "event_type": "investment_order_settled",
                "order_id": settlement.order_id,
                "buyer_firm_id": settlement.buyer_firm_id,
                "supplier_firm_id": ";".join(str(fill.supplier_firm_id) for fill in settlement.fills),
                "investment_source": order_metadata.get("investment_source", "UNAVAILABLE"),
                "replacement_trigger_id": order_metadata.get("replacement_trigger_id", "UNAVAILABLE"),
                "requested_units": settlement.requested_units,
                "settled_units": settlement.settled_units,
                "unmet_units": settlement.unmet_units,
                "settled_expenditure": settlement.settled_expenditure,
                "capital_asset_created": bool(settlement.capital_assets),
            })
            buyer.capital_stock.assets.extend(settlement.capital_assets)
            for asset in settlement.capital_assets:
                metadata = dict(getattr(asset, "capacity_metadata", {}) or {})
                # CapitalAsset is a frozen record, but its metadata mapping is
                # intentionally the passive provenance extension point.
                asset.capacity_metadata["acquisition_week"] = step
                source = order_metadata.get("investment_source", "UNAVAILABLE")
                event = {
                    "global_step": step,
                    "event_type": "asset_acquired",
                    "asset_id": asset.asset_id,
                    "owner_firm_id": asset.owner_firm_id,
                    "order_id": settlement.order_id,
                    "investment_source": source,
                    "replaced_asset_id": order_metadata.get("replaced_asset_id", "UNAVAILABLE"),
                    "replacement_trigger_id": order_metadata.get("replacement_trigger_id", "UNAVAILABLE"),
                    "replacement_order_id": settlement.order_id if source in ("REPLACEMENT", "MIXED") else "UNAVAILABLE",
                    "acquisition_cost": asset.acquisition_cost,
                    "physical_asset_units": asset.quantity,
                }
                getattr(self.world, "capital_asset_event_rows", []).append(event)
            buyer.investment_expenditure_this_step = settlement.settled_expenditure
            acquired_value = sum(
                asset.acquisition_cost for asset in settlement.capital_assets
            )
            buyer.investment_assets_acquired_this_step = acquired_value
            buyer.capital_asset_acquisitions_this_step = (
                float(getattr(buyer, "capital_asset_acquisitions_this_step", 0.0))
                + acquired_value
            )
            buyer.investment_order_expenditure = settlement.desired_investment_expenditure
            buyer.investment_financing_gap = settlement.financing_gap
            desired_total = max(
                0.0,
                float(order_metadata.get("replacement_investment_expenditure", 0.0))
                + float(order_metadata.get("expansion_investment_expenditure", 0.0)),
            )
            replacement_share = (
                float(order_metadata.get("replacement_investment_expenditure", 0.0))
                / desired_total
                if desired_total > 1e-12 else 0.0
            )
            replacement_spend = settlement.settled_expenditure * replacement_share
            expansion_spend = settlement.settled_expenditure - replacement_spend
            buyer.executed_replacement_investment_this_step = replacement_spend
            buyer.executed_expansion_investment_this_step = expansion_spend
            buyer.unmet_replacement_investment = max(
                0.0,
                float(order_metadata.get("replacement_investment_expenditure", 0.0))
                - replacement_spend,
            )
            buyer.unmet_expansion_investment = max(
                0.0,
                float(order_metadata.get("expansion_investment_expenditure", 0.0))
                - expansion_spend,
            )
            buyer.pending_replacement_capacity_need = max(
                0.0,
                float(getattr(buyer, "pending_replacement_capacity_need", 0.0))
                - replacement_spend / self.unit_price,
            )
            buyer.replacement_capacity_need = buyer.pending_replacement_capacity_need
            self.expansion_backlog_by_firm[buyer.firm_id] = max(
                0.0,
                self.expansion_backlog_by_firm.get(buyer.firm_id, 0.0)
                - expansion_spend / self.unit_price,
            )
            buyer.active_capital_asset_count = sum(
                bool(asset.is_active) for asset in buyer.capital_stock.assets
            )
            buyer.retired_capital_asset_count = (
                len(buyer.capital_stock.assets) - buyer.active_capital_asset_count
            )
            self.last_week.fixed_investment += settlement.settled_expenditure
            self.last_week.replacement_investment += replacement_spend
            self.last_week.expansion_investment += expansion_spend
            self.last_week.investment_orders += 1
            self.last_week.investment_events.append(settlement)
            if self.customer_advance_enabled:
                pending_order = orders_by_id.get(settlement.order_id)
                if pending_order is not None:
                    if settlement.unmet_units > 1e-12:
                        self.customer_advance_pending_orders[settlement.order_id] = replace(
                            pending_order,
                            desired_units=settlement.unmet_units,
                            desired_investment_expenditure=(
                                settlement.unmet_units * self.unit_price
                            ),
                        )
                    else:
                        self.customer_advance_pending_orders.pop(
                            settlement.order_id,
                            None,
                        )
        for firm in self.capital_good_firms:
            delivered_revenue = delivered_by_supplier.get(firm.firm_id, 0.0)
            if self.customer_advance_enabled:
                # The adapter models a delivery cash receipt.  In the advance
                # path that receipt already arrived when the order was
                # accepted, so remove the synthetic duplicate here.
                firm.cash = max(
                    0.0,
                    batch.supplier_cash_end.get(firm.firm_id, firm.cash)
                    - delivered_revenue,
                )
                firm.customer_advance_delivered_this_step = delivered_revenue
                firm.customer_advance_liability = max(
                    0.0,
                    float(getattr(firm, "customer_advance_liability", 0.0))
                    - delivered_revenue,
                )
                firm.sales_collections_this_step = 0.0
                self.last_week.customer_advances_received += float(
                    getattr(firm, "customer_advance_received_this_step", 0.0)
                )
                self.last_week.customer_advances_delivered += delivered_revenue
            else:
                firm.cash = batch.supplier_cash_end.get(firm.firm_id, firm.cash)
            sold = math.fsum(
                fill.settled_units
                for settlement in batch.settlements
                for fill in settlement.fills
                if fill.supplier_firm_id == firm.firm_id
            )
            revenue = sold * self.unit_price
            firm.capital_good_sales_units = sold
            firm.sales_units = sold
            firm.sales = revenue
            firm.sales_revenue = revenue
            firm.transaction_price = self.unit_price
            firm.transaction_revenue = revenue
            firm.inventory_units = firm.capital_good_inventory.units
            firm.inventory_value = firm.capital_good_inventory.units * self.unit_price
            firm.profit = revenue - firm.wage_payment
            firm.profit_before_dividend = firm.profit
            firm.cash_end = firm.cash
            if not self.customer_advance_enabled:
                firm.sales_collections_this_step = revenue
            if self.customer_advance_enabled:
                self.customer_advance_ledger.append({
                    "global_step": step,
                    "event_type": "customer_advance_delivery_recognition",
                    "order_id": ";".join(
                        str(item.order_id)
                        for item in batch.settlements
                        if any(
                            fill.supplier_firm_id == firm.firm_id
                            for fill in item.fills
                        )
                    ),
                    "buyer_firm_id": "",
                    "supplier_firm_id": firm.firm_id,
                    "physical_units": sold,
                    "cash_amount": 0.0,
                    "buyer_cash_change": 0.0,
                    "supplier_cash_change": 0.0,
                    "buyer_prepaid_asset_change": -delivered_revenue,
                    "supplier_advance_liability_change": -delivered_revenue,
                    "revenue_recognized": revenue,
                    "capital_asset_created": bool(delivered_revenue > 1e-12),
                    "capital_asset_recognized_value": delivered_revenue,
                })
            self.last_week.capital_good_sales += sold
            self.last_week.capital_good_revenue += revenue
        if self.customer_advance_enabled:
            closing_pending_units = math.fsum(
                max(0.0, float(order.desired_units))
                for order in self.customer_advance_pending_orders.values()
            )
            self.pipeline_depleted_last_step = bool(
                opening_pending_units > PIPELINE_QUANTITY_TOLERANCE
                and closing_pending_units <= PIPELINE_QUANTITY_TOLERANCE
            )

    def prepare_week(self, step):
        self.ensure_firms()
        self._ensure_customer_advance_state()
        if not self.enabled:
            self.pending_household_wages = {}
            return CanonicalInvestmentWeek(step=step)
        # The startup transfer belongs only to the first recorded week.  The
        # prior week's accounting record is complete before the next prepare
        # call, so clear the one-shot marker here rather than carrying it into
        # every later cash-flow bridge.
        if step > 0:
            for firm in self.capital_good_firms:
                firm.startup_capitalization_inflow_this_step = 0.0
                firm.startup_capitalization_outflow_this_step = 0.0
        for firm in getattr(self.world, "firms", []):
            firm.accounting_cash_start_this_step = firm.cash
            firm.startup_capitalization_inflow_this_step = 0.0
            firm.startup_capitalization_outflow_this_step = 0.0
            firm.investment_expenditure_this_step = 0.0
            firm.prepaid_capital_investment_paid_this_step = 0.0
            firm.prepaid_capital_investment_capitalized_this_step = 0.0
            firm.customer_advance_delivered_this_step = 0.0
            firm.investment_cash_outflow_this_step = 0.0
            firm.investment_assets_acquired_this_step = 0.0
            firm.investment_order_expenditure = 0.0
            firm.investment_financing_gap = 0.0
            firm.pending_replacement_capacity_need = max(
                0.0,
                float(getattr(firm, "pending_replacement_capacity_need", 0.0)),
            )
        for firm in self.capital_good_firms:
            startup_inflow = max(
                0.0,
                float(getattr(firm, "startup_capitalization_inflow_this_step", 0.0)),
            )
            # Keep the supplier's accounting opening cash before the new
            # customer advance, while allowing the advance to fund payroll.
            firm.accounting_cash_start_this_step = max(0.0, firm.cash - startup_inflow)
            firm.customer_advance_received_this_step = 0.0
            firm.customer_advance_delivered_this_step = 0.0
            firm.sales_collections_this_step = 0.0
        self.opening_replacement_backlog_units = math.fsum(
            max(
                0.0,
                float(getattr(firm, "pending_replacement_capacity_need", 0.0)),
            )
            for firm in getattr(self.world, "firms", [])
        )
        self.opening_expansion_backlog_units = math.fsum(
            max(0.0, float(value))
            for value in self.expansion_backlog_by_firm.values()
        )
        self.opening_total_backlog_units = (
            self.opening_replacement_backlog_units
            + self.opening_expansion_backlog_units
        )
        new_expansion_demand_flow = self.pending_expansion_demand_flow
        self.pending_expansion_demand_flow = 0.0
        lifecycle_week = self._advance_capital_lifecycle(step)
        self._activate_capital_good_labor(
            step,
            new_replacement_demand_flow=lifecycle_week.retired_capacity,
            new_expansion_demand_flow=new_expansion_demand_flow,
        )
        if self.customer_advance_enabled:
            new_orders = self._build_orders(step)
            self._collect_customer_advances(step, new_orders)
            orders = list(self.customer_advance_pending_orders.values())
            week = self._pay_and_produce(step)
            week.prepaid_investment_cash_paid = math.fsum(
                max(
                    0.0,
                    float(getattr(firm, "prepaid_capital_investment_paid_this_step", 0.0)),
                )
                for firm in getattr(self.world, "firms", [])
            )
        else:
            week = self._pay_and_produce(step)
        week.depreciation_expense = lifecycle_week.depreciation_expense
        week.retired_capacity = lifecycle_week.retired_capacity
        if not self.customer_advance_enabled:
            orders = self._build_orders(step)
        self._settle_orders(step, orders)
        self._close_capital_good_labor(step)
        return self.last_week

    def add_pending_wages(self):
        total = 0.0
        for household_id, amount in self.pending_household_wages.items():
            household = self.world.household_dict.get(household_id)
            if household is None:
                continue
            household.income_this_step += amount
            household.wage_income_this_step += amount
            total += amount
        return total


__all__ = ["CanonicalInvestmentSystem", "CanonicalInvestmentWeek"]

