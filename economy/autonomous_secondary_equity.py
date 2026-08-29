"""Minimal deterministic autonomous secondary-equity purchase screen."""

import math
from dataclasses import dataclass, field

from economy.household_equity_demand import ShadowHouseholdEquityDemandSystem
from economy.secondary_legacy_transfer import ControlledSecondaryLegacyTransfer


AUTONOMOUS_SECONDARY_SCHEMA = "autonomous_secondary_equity_screen_v2"


@dataclass
class AutonomousSecondaryPurchaseResult:
    step: int
    reviewed: bool = False
    valuation_step: object = None
    book_equity_used: float = 0.0
    total_shares_used: float = 0.0
    reference_price_used: float = 0.0
    valuation_status: str = "NOT_REVIEWED"
    decisions: list = field(default_factory=list)
    fills: list = field(default_factory=list)
    transfer_results: list = field(default_factory=list)
    total_payment: float = 0.0
    total_shares_sold: float = 0.0
    buyer_count: int = 0
    firm_cash_change: float = 0.0
    paid_in_equity_change: float = 0.0
    money_created: float = 0.0
    money_destroyed: float = 0.0


class AutonomousSecondaryEquityPurchaseSystem:
    """Small deterministic buyer screen; disabled unless explicitly enabled."""

    def __init__(
        self,
        world,
        price_per_share=10.0,
        review_interval_weeks=52,
        max_available_cash_fraction=0.01,
        max_buyers_per_firm=10,
        max_household_equity_share_of_assets=0.10,
        max_person_ownership_fraction=0.05,
        reference_price_mode="engineering_fixed",
    ):
        self.world = world
        self.price_per_share = float(price_per_share)
        if reference_price_mode not in {
            "engineering_fixed",
            "lagged_book_equity",
        }:
            raise ValueError("unsupported secondary equity reference_price_mode")
        self.reference_price_mode = reference_price_mode
        self.review_interval_weeks = max(1, int(review_interval_weeks))
        self.max_available_cash_fraction = max(
            0.0, float(max_available_cash_fraction)
        )
        self.max_buyers_per_firm = (
            None
            if max_buyers_per_firm is None
            else max(1, int(max_buyers_per_firm))
        )
        self.max_household_equity_share_of_assets = float(
            max_household_equity_share_of_assets
        )
        self.max_person_ownership_fraction = float(
            max_person_ownership_fraction
        )
        self.demand_system = ShadowHouseholdEquityDemandSystem()
        self.previous_total_shares_by_firm = {}

    @staticmethod
    def _liquidity_reserve(household):
        target = max(
            0.0,
            float(getattr(household, "target_wealth_this_step", 0.0)),
        )
        if target <= 0.0:
            target = max(
                0.0,
                float(getattr(household, "necessary_consumption_this_step", 0.0))
                * 26.0,
            )
        return target

    def _firms(self):
        firms = list(getattr(self.world, "firms", []))
        if firms:
            return firms
        return [self.world.firm_system]

    def _snapshot_total_shares(self):
        for firm in self._firms():
            table = getattr(firm, "cap_table", None)
            if table is not None:
                self.previous_total_shares_by_firm[
                    getattr(firm, "firm_id", 0)
                ] = float(table.total_shares)

    def _locked_valuation(self, firm, step):
        table = getattr(firm, "cap_table", None)
        current_total_shares = (
            float(table.total_shares) if table is not None else 0.0
        )
        if self.reference_price_mode == "engineering_fixed":
            return {
                "valuation_step": step,
                "book_equity_used": float("nan"),
                "total_shares_used": current_total_shares,
                "reference_price_used": self.price_per_share,
                "status": "ENGINEERING_FIXED_PRICE",
            }

        accounting = getattr(self.world, "accounting", None)
        state = None
        if accounting is not None:
            state = getattr(accounting, "firms", {}).get(
                getattr(firm, "firm_id", 0)
            )
        previous_row = None
        if state is not None and getattr(state, "rows", None):
            previous_row = state.rows[-1]
        if previous_row is None:
            return {
                "valuation_step": None,
                "book_equity_used": float("nan"),
                "total_shares_used": float("nan"),
                "reference_price_used": float("nan"),
                "status": "NO_LAGGED_BOOK_EQUITY",
            }

        book_equity = float(previous_row.get("equity", 0.0))
        valuation_step = previous_row.get(
            "global_step", previous_row.get("step")
        )
        total_shares = self.previous_total_shares_by_firm.get(
            getattr(firm, "firm_id", 0), current_total_shares
        )
        if book_equity <= 0.0:
            return {
                "valuation_step": valuation_step,
                "book_equity_used": book_equity,
                "total_shares_used": total_shares,
                "reference_price_used": float("nan"),
                "status": "NON_POSITIVE_EQUITY",
            }
        if total_shares <= 0.0:
            return {
                "valuation_step": valuation_step,
                "book_equity_used": book_equity,
                "total_shares_used": total_shares,
                "reference_price_used": float("nan"),
                "status": "NO_SHARES",
            }
        return {
            "valuation_step": valuation_step,
            "book_equity_used": book_equity,
            "total_shares_used": total_shares,
            "reference_price_used": book_equity / total_shares,
            "status": "LAGGED_BOOK_EQUITY",
        }

    def review(self, step):
        result = AutonomousSecondaryPurchaseResult(step=int(step))
        if not hasattr(self.world, "autonomous_secondary_purchase_history"):
            self.world.autonomous_secondary_purchase_history = []
        if not hasattr(self.world, "autonomous_secondary_demand_history"):
            self.world.autonomous_secondary_demand_history = []
        for household in getattr(self.world, "households", []):
            household.equity_purchase_cash_outflow_this_step = 0.0
        if step % self.review_interval_weeks != 0:
            self._snapshot_total_shares()
            self.world.last_autonomous_secondary_purchase_result = result
            return result

        result.reviewed = True
        used_households = set()
        for firm in self._firms():
            table = getattr(firm, "cap_table", None)
            if table is None or table.legacy_shares <= 1e-12:
                continue
            valuation = self._locked_valuation(firm, step)
            if result.valuation_status == "NOT_REVIEWED":
                result.valuation_step = valuation["valuation_step"]
                result.book_equity_used = valuation["book_equity_used"]
                result.total_shares_used = valuation["total_shares_used"]
                result.reference_price_used = valuation["reference_price_used"]
                result.valuation_status = valuation["status"]
            locked_price = valuation["reference_price_used"]
            if not math.isfinite(float(locked_price)) or locked_price <= 0.0:
                continue
            candidates = []
            decision_records = {}
            for household in sorted(
                getattr(self.world, "households", []),
                key=lambda item: getattr(item, "id", 0),
            ):
                if household.id in used_households:
                    continue
                person = self.demand_system.select_buyer_person(
                    household,
                    getattr(self.world, "population", []),
                )
                if person is None:
                    continue
                available = max(
                    0.0,
                    float(getattr(household, "wealth", 0.0))
                    - self._liquidity_reserve(household)
                    - float(getattr(household, "necessary_consumption_this_step", 0.0)),
                )
                requested_budget = available * self.max_available_cash_fraction
                decision = self.demand_system.decide(
                    household_id=household.id,
                    buyer_person_id=person.id,
                    firm_id=getattr(firm, "firm_id", 0),
                    household_cash=float(getattr(household, "wealth", 0.0)),
                    liquidity_reserve=self._liquidity_reserve(household),
                    necessary_consumption_buffer=float(
                        getattr(household, "necessary_consumption_this_step", 0.0)
                    ),
                    offered_transfer_price=locked_price,
                    current_equity_assets=float(
                        getattr(household, "equity_asset_value", 0.0)
                    ),
                    current_firm_equity_assets=float(
                        getattr(household, "equity_asset_value", 0.0)
                    ),
                    current_person_firm_shares=sum(
                        holding.shares
                        for holding in table.holdings
                        if holding.holder_type == "person"
                        and holding.holder_id == person.id
                    ),
                    firm_total_shares=table.total_shares,
                    available_legacy_shares=table.legacy_shares,
                    desired_equity_budget=requested_budget,
                    max_household_equity_share_of_assets=(
                        self.max_household_equity_share_of_assets
                    ),
                    max_person_ownership_fraction=(
                        self.max_person_ownership_fraction
                    ),
                )
                decision_record = decision.to_dict()
                decision_record.update({
                    "decision_step": int(step),
                    "valuation_step": valuation["valuation_step"],
                    "book_equity_used": valuation["book_equity_used"],
                    "total_shares_used": valuation["total_shares_used"],
                    "reference_price": locked_price,
                    "fillable_shares": decision.shadow_fillable_shares,
                    "executed_shares": 0.0,
                    "executed_cash": 0.0,
                    "unmet_shares": decision.unmet_shadow_shares,
                    "block_rejection_reason": decision.decision_reason,
                    "reference_equity_value": float(
                        getattr(household, "equity_asset_value", 0.0)
                    ),
                    "unrealized_gain_loss": 0.0,
                })
                result.decisions.append(decision_record)
                decision_records[(household.id, person.id)] = decision_record
                if decision.desired_shares <= 1e-12:
                    continue
                candidates.append((person, household, decision.desired_shares))
                used_households.add(household.id)
                if (
                    self.max_buyers_per_firm is not None
                    and len(candidates) >= self.max_buyers_per_firm
                ):
                    break

            if not candidates:
                continue
            executor = ControlledSecondaryLegacyTransfer(
                self.world,
                price_per_share=locked_price,
            )
            transfer = executor.execute(firm, candidates)
            result.transfer_results.append(transfer)
            result.fills.extend(transfer.fills)
            result.total_payment += transfer.total_payment
            result.total_shares_sold += transfer.total_shares_sold
            result.buyer_count += sum(
                fill["shares_filled"] > 1e-12 for fill in transfer.fills
            )
            result.firm_cash_change += (
                transfer.firm_cash_after - transfer.firm_cash_before
            )
            result.paid_in_equity_change += (
                transfer.paid_in_equity_after
                - transfer.paid_in_equity_before
            )
            for fill in transfer.fills:
                record = decision_records.get(
                    (fill["buyer_household_id"], fill["buyer_person_id"])
                )
                if record is None:
                    continue
                record["executed_shares"] = fill["shares_filled"]
                record["executed_cash"] = fill["payment"]
                record["unmet_shares"] = max(
                    0.0,
                    record["desired_shares"] - fill["shares_filled"],
                )
                record["block_rejection_reason"] = (
                    "executed"
                    if fill["shares_filled"] > 1e-12
                    else record["decision_reason"]
                )
                person_shares_after = sum(
                    holding.shares
                    for holding in firm.cap_table.holdings
                    if holding.holder_type == "person"
                    and holding.holder_id == record["buyer_person_id"]
                )
                record["reference_equity_value"] = (
                    float(person_shares_after) * locked_price
                )
                record["unrealized_gain_loss"] = (
                    record["reference_equity_value"]
                    - float(
                        fill.get(
                            "household_equity_after",
                            record["current_equity_assets"]
                            + record["executed_cash"],
                        )
                    )
                )

        self.world.autonomous_secondary_demand_history.extend(
            result.decisions
        )

        self.world.autonomous_secondary_purchase_history.append({
            "schema": AUTONOMOUS_SECONDARY_SCHEMA,
            "step": int(step),
            "reviewed": result.reviewed,
            "total_payment": result.total_payment,
            "total_shares_sold": result.total_shares_sold,
            "buyer_count": result.buyer_count,
            "valuation_step": result.valuation_step,
            "book_equity_used": result.book_equity_used,
            "total_shares_used": result.total_shares_used,
            "reference_price_used": result.reference_price_used,
            "valuation_status": result.valuation_status,
            "decision_count": len(result.decisions),
            "firm_cash_change": result.firm_cash_change,
            "paid_in_equity_change": result.paid_in_equity_change,
        })
        self._snapshot_total_shares()
        self.world.last_autonomous_secondary_purchase_result = result
        return result


__all__ = [
    "AUTONOMOUS_SECONDARY_SCHEMA",
    "AutonomousSecondaryPurchaseResult",
    "AutonomousSecondaryEquityPurchaseSystem",
]
