"""Deterministic capital-good Firm entry for the Step17.Y smoke.

The system is deliberately narrow and opt-in.  It only observes the accepted
capital-good backlog/profit signal, transfers real founder cash, and registers
a generic FirmSlice with the existing capital-good supplier.  It does not
create demand, draw RNG, grant credit, or enter Food.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from economy.capital_goods import CapitalGoodInventory
from economy import config
from economy.founder_ownership import eligible_founders
from economy.multi_firm import FirmSlice
from economy.multisector import (
    CAPITAL_GOODS_SECTOR_ID,
    CAPITAL_GOODS_TECHNOLOGY_ID,
    CAPITAL_GOOD_INVENTORY_POLICY_ID,
    CAPITAL_MACHINE_GOOD_ID,
    default_firm_identity,
)
from economy.ownership_accounting import CapTable
from productivity import age_productivity


EPS = 1e-9


@dataclass
class FirmFormationProposal:
    founder_person_id: object
    proposed_sector: str
    formation_week: int
    entry_reason: str
    required_startup_cash: float
    founder_cash_contribution: float
    external_financing_requested: float
    required_initial_capacity: float
    required_initial_labor: float
    required_initial_inventory: float
    required_working_capital: float
    formation_status: str = "PROPOSED"
    rejection_reason: str = ""

    def to_dict(self):
        return asdict(self)


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


class EndogenousCapitalGoodsEntrySystem:
    """One deterministic, capital-goods-only entry contract."""

    VERSION = 1

    def __init__(self, world):
        self.world = world
        self.enabled = bool(
            getattr(world, "endogenous_capital_goods_entry_enabled", False)
        )
        self.review_interval = 13
        self.cooldown_weeks = 13
        self.required_persistence_weeks = 13
        self.history = []
        self.formation_events = []
        self.formation_failures = []
        self.proposals = []
        self.last_formation_week = None
        self.next_firm_id = 100001

    def _capital_firms(self):
        return list(getattr(self.world, "capital_good_firms", []))

    def _signal_snapshot(self, step):
        firms = self._capital_firms()
        outstanding = math.fsum(
            max(0.0, _number(getattr(firm, "capital_good_outstanding_demand_units", 0.0)))
            for firm in firms
        )
        sales = math.fsum(
            max(0.0, _number(getattr(firm, "capital_good_sales_units", 0.0)))
            for firm in firms
        )
        unmet = max(0.0, outstanding - sales)
        profit = math.fsum(
            _number(getattr(firm, "profit", getattr(firm, "profit_before_dividend", 0.0)))
            for firm in firms
        )
        eligible_labor = math.fsum(
            max(0.0, age_productivity(getattr(person, "age", 0.0)))
            for person in getattr(self.world, "population", [])
            if getattr(person, "alive", False)
            and self.world.labor_participates(person)
            and self.world.has_valid_settlement_household(person)
        )
        return {
            "global_step": int(step),
            "capital_good_firm_count": len(firms),
            "backlog_units": outstanding,
            "sales_units": sales,
            "unmet_demand_units": unmet,
            "unmet_demand_ratio": unmet / max(outstanding, EPS),
            "operating_profit": profit,
            "positive_operating_profit": int(profit > EPS),
            "eligible_labor_capacity": eligible_labor,
            "entry_signal_current": int(unmet / max(outstanding, EPS) >= 0.05 and profit > EPS),
        }

    def record_week(self, step):
        if not self.enabled:
            return
        self.history.append(self._signal_snapshot(step))

    def _startup_labor(self):
        # Accepted Step17.X fixture: N=100 -> 1.0, N=500 -> 1.8 labor
        # services.  This is an engineering smoke contract, not calibration.
        return max(1.0, 0.0036 * float(len(getattr(self.world, "population", []))))

    def _protected_household_cash(self, household):
        return max(0.0, _number(getattr(household, "target_wealth_this_step", 0.0)))

    def _required_cash(self):
        labor = self._startup_labor()
        wage = max(0.0, _number(getattr(config, "FIRM_WAGE_PER_LABOR", 41.0)))
        return 2.0 * labor * wage

    def _founder_rows(self):
        rows = []
        for person in eligible_founders(self.world):
            household = self.world.get_household(getattr(person, "household_id", None))
            if household is None:
                continue
            cash = max(0.0, _number(getattr(household, "wealth", 0.0)))
            protected = self._protected_household_cash(household)
            rows.append({
                "person": person,
                "household": household,
                "cash": cash,
                "protected": protected,
                "capacity": max(0.0, cash - protected),
            })
        return sorted(rows, key=lambda row: (-row["capacity"], row["person"].id))

    def _persistent_signal(self, step):
        if len(self.history) < self.required_persistence_weeks:
            return False
        recent = self.history[-self.required_persistence_weeks:]
        expected = list(range(step - self.required_persistence_weeks, step))
        if [row["global_step"] for row in recent] != expected:
            return False
        return all(row["entry_signal_current"] for row in recent)

    def _failure(self, step, reason, proposal=None, metadata=None):
        row = {
            "global_step": int(step),
            "sector_id": CAPITAL_GOODS_SECTOR_ID,
            "failure_reason": reason,
            "firm_created": 0,
            "founder_person_id": getattr(proposal, "founder_person_id", "") if proposal else "",
            "required_startup_cash": getattr(proposal, "required_startup_cash", 0.0) if proposal else 0.0,
            **dict(metadata or {}),
        }
        self.formation_failures.append(row)
        return row

    def _new_firm_id(self):
        existing = {
            getattr(firm, "firm_id", None)
            for firm in self.world.operating_firms()
        }
        candidate = self.next_firm_id
        while candidate in existing:
            candidate += 1
        self.next_firm_id = candidate + 1
        return candidate

    def _create_firm(self, step, founder_row, required_cash):
        person = founder_row["person"]
        household = founder_row["household"]
        firm_id = self._new_firm_id()
        system = getattr(self.world, "canonical_investment_system", None)
        if system is None:
            return self._failure(step, "CAPITAL_GOOD_SYSTEM_UNAVAILABLE")
        price = max(1e-12, _number(getattr(system, "unit_price", 10.0)))
        firm = FirmSlice(
            firm_id=firm_id,
            employee_ids=[],
            share=0.0,
            cash=0.0,
            price=price,
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
        firm.paid_in_equity = required_cash
        firm.cap_table = CapTable.initial_person_owned(
            firm_id=firm_id,
            person_id=person.id,
            issued_shares=100.0,
        )
        assignment = {
            "schema": "person_founder_bootstrap_v1",
            "firm_id": firm_id,
            "founder_person_id": person.id,
            "ownership_fraction": 1.0,
            "assignment_source": "endogenous_capital_goods_entry",
            "formation_context": "ENDOGENOUS_CAPITAL_GOODS_ENTRY",
            "equity_cash_contribution": required_cash,
            "equity_non_cash_contribution": 0.0,
            "bootstrap_compatibility_amount": 0.0,
            "provenance_status": "ENDOGENOUS_FOUNDER_EQUITY",
            "ownership_fallback_reason": "",
        }
        firm.founder_assignment = assignment
        firm.ownership_fallback_reason = ""
        firm.startup_capitalization_inflow_this_step = required_cash
        firm.startup_capitalization_outflow_this_step = 0.0
        before_cash = _number(getattr(household, "wealth", 0.0))
        paid = self.world.ledger.transfer_attrs(
            payer_obj=household,
            payer_attr="wealth",
            payer_name=f"household.{household.id}.wealth",
            receiver_obj=firm,
            receiver_attr="cash",
            receiver_name=f"firm.{firm_id}.cash",
            amount=required_cash,
            reason="endogenous_firm_founder_equity",
        )
        if paid + EPS < required_cash:
            return self._failure(step, "FOUNDER_EQUITY_TRANSFER_FAILED", metadata={"paid": paid})
        system.capital_good_firms.append(firm)
        system.capital_good_firm_ids.add(firm_id)
        system._rebuild_lookup()
        self.world.founder_assignment_rows.append(dict(assignment))
        self.world.founder_assigned_firm_ids.add(firm_id)
        ownership = getattr(self.world, "equity_ownership_system", None)
        if ownership is not None:
            ownership.cap_tables[firm_id] = firm.cap_table
        after_cash = _number(getattr(household, "wealth", 0.0))
        event = {
            "global_step": int(step),
            "firm_id": firm_id,
            "founder_person_id": person.id,
            "founder_household_id": household.id,
            "sector_id": CAPITAL_GOODS_SECTOR_ID,
            "entry_signal": "13-week persistent backlog/unmet demand plus positive operating profit",
            "founder_equity": paid,
            "loan_financing": 0.0,
            "starting_cash": _number(getattr(firm, "cash", 0.0)),
            "starting_capacity": 0.0,
            "starting_labor": self._startup_labor(),
            "formation_status": "ACCEPTED",
            "household_cash_before": before_cash,
            "household_cash_after": after_cash,
            "protected_reserve": founder_row["protected"],
            "minimum_post_contribution_liquidity": after_cash - founder_row["protected"],
            "primary_issuance": 0,
            "new_rng_draws": 0,
        }
        self.formation_events.append(event)
        self.last_formation_week = int(step)
        return event

    def before_production(self, step):
        if not self.enabled or step <= 0 or step % self.review_interval:
            return None
        if self.last_formation_week is not None and step - self.last_formation_week < self.cooldown_weeks:
            self._failure(step, "ENTRY_COOLDOWN_ACTIVE")
            return None
        signal = self._persistent_signal(step)
        if not signal:
            self._failure(step, "MARKET_SIGNAL_NOT_PERSISTENT")
            return None
        founder_rows = self._founder_rows()
        if not founder_rows:
            self._failure(step, "NO_ELIGIBLE_FOUNDER")
            return None
        required_cash = self._required_cash()
        selected = founder_rows[0]
        if selected["capacity"] + EPS < required_cash:
            self._failure(
                step,
                "INSUFFICIENT_FOUNDER_EQUITY",
                metadata={
                    "selected_founder_person_id": selected["person"].id,
                    "founder_equity_capacity": selected["capacity"],
                    "required_startup_cash": required_cash,
                },
            )
            return None
        proposal = FirmFormationProposal(
            founder_person_id=selected["person"].id,
            proposed_sector=CAPITAL_GOODS_SECTOR_ID,
            formation_week=int(step),
            entry_reason="13-week persistent backlog/unmet demand plus positive operating profit",
            required_startup_cash=required_cash,
            founder_cash_contribution=required_cash,
            external_financing_requested=0.0,
            required_initial_capacity=0.0,
            required_initial_labor=self._startup_labor(),
            required_initial_inventory=0.0,
            required_working_capital=required_cash,
        )
        self.proposals.append(proposal.to_dict())
        active_founder_firms = sum(
            getattr(firm, "founder_assignment", {}).get("founder_person_id") == selected["person"].id
            for firm in self.world.operating_firms()
        )
        event = self._create_firm(step, selected, required_cash)
        if event is not None and event.get("formation_status") == "ACCEPTED":
            event.update({
                "already_founder": int(active_founder_firms > 0),
                "active_founder_firm_count_before": active_founder_firms,
                "next_best_distinct_founder": (
                    founder_rows[1]["person"].id if len(founder_rows) > 1 else ""
                ),
                "next_best_distinct_capacity": (
                    founder_rows[1]["capacity"] if len(founder_rows) > 1 else 0.0
                ),
            })
        return event

    def state_rows(self):
        return list(self.history)


__all__ = ["FirmFormationProposal", "EndogenousCapitalGoodsEntrySystem"]
