"""Deterministic shareholder-death and equity-estate settlement layer."""

from dataclasses import dataclass, field

from economy.ledger import object_account


SHAREHOLDER_ESTATE_SCHEMA = "shareholder_estate_inheritance_v1"


@dataclass
class EstateAccount:
    estate_id: str
    deceased_person_id: int
    opened_step: int
    status: str = "open"
    cash: float = 0.0
    estate_dividend_income: float = 0.0
    shares_by_firm: dict = field(default_factory=dict)
    cost_basis_by_firm: dict = field(default_factory=dict)
    heir_person_ids: list = field(default_factory=list)
    closed_step: object = None

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.status = getattr(self, "status", "open")
        self.cash = float(getattr(self, "cash", 0.0))
        self.estate_dividend_income = float(
            getattr(self, "estate_dividend_income", 0.0)
        )
        self.shares_by_firm = dict(getattr(self, "shares_by_firm", {}))
        self.cost_basis_by_firm = dict(
            getattr(self, "cost_basis_by_firm", {})
        )
        self.heir_person_ids = list(getattr(self, "heir_person_ids", []))
        self.closed_step = getattr(self, "closed_step", None)


class ShareholderEstateSystem:
    """Move deceased Person holdings to an explicit Estate holder.

    The system is deterministic and uses no random draws.  Heirs are ordered
    as alive direct children, alive partner, then other alive members of the
    deceased person's Household.  Shares and carried historical cost basis are
    split equally among the selected heirs.
    """

    def __init__(self, world):
        self.world = world

    def ensure_state(self):
        world = self.world
        if not hasattr(world, "estate_accounts"):
            world.estate_accounts = {}
        if not hasattr(world, "shareholder_estate_by_deceased"):
            world.shareholder_estate_by_deceased = {}
        if not hasattr(world, "shareholder_estate_events"):
            world.shareholder_estate_events = []
        return world

    def _firms(self):
        firms = list(getattr(self.world, "firms", []))
        return firms or [self.world.firm_system]

    def _valid_heirs(self, person):
        candidates = []
        seen = set()

        def add(person_id):
            if person_id is None or person_id in seen:
                return
            candidate = getattr(self.world, "person_dict", {}).get(person_id)
            if candidate is not None and getattr(candidate, "alive", False):
                seen.add(person_id)
                candidates.append(candidate)

        for person_id in sorted(getattr(person, "children_ids", [])):
            add(person_id)
        add(getattr(person, "partner_id", None))

        household = self.world.get_household(
            getattr(person, "household_id", None)
        )
        if household is not None:
            for person_id in sorted(
                list(getattr(household, "parents", []))
                + list(getattr(household, "children", []))
            ):
                add(person_id)
        return candidates

    @staticmethod
    def _person_basis(person, firm_id, shares):
        basis = dict(getattr(person, "equity_cost_basis", {}))
        total_basis = float(basis.get(firm_id, 0.0))
        holdings = dict(getattr(person, "equity_holdings", {}))
        held_shares = float(holdings.get(firm_id, shares))
        if held_shares <= 0.0:
            return 0.0
        return total_basis * min(1.0, shares / held_shares)

    def _set_person_holding(self, person, firm_id, shares):
        person.equity_holdings = dict(getattr(person, "equity_holdings", {}))
        if shares <= 1e-12:
            person.equity_holdings.pop(firm_id, None)
        else:
            person.equity_holdings[firm_id] = shares

    def _set_person_basis(self, person, firm_id, basis):
        person.equity_cost_basis = dict(
            getattr(person, "equity_cost_basis", {})
        )
        if basis <= 1e-12:
            person.equity_cost_basis.pop(firm_id, None)
        else:
            person.equity_cost_basis[firm_id] = basis

    def _record_event(self, event):
        event = {"schema": SHAREHOLDER_ESTATE_SCHEMA, **event}
        self.world.shareholder_estate_events.append(event)
        return event

    def open_for_death(self, person, step=None):
        self.ensure_state()
        person_id = getattr(person, "id", None)
        if person_id in self.world.shareholder_estate_by_deceased:
            estate_id = self.world.shareholder_estate_by_deceased[person_id]
            return self.world.estate_accounts[estate_id]

        step = int(
            getattr(self.world, "current_step_index", 0)
            if step is None
            else step
        )
        holdings = []
        for firm in self._firms():
            table = getattr(firm, "cap_table", None)
            if table is None:
                continue
            shares = sum(
                holding.shares
                for holding in table.holdings
                if holding.holder_type == "person"
                and holding.holder_id == person_id
            )
            if shares > 1e-12:
                holdings.append((firm, shares))

        if not holdings:
            return None

        estate_id = f"estate:{person_id}:{len(self.world.estate_accounts)}"
        account = EstateAccount(
            estate_id=estate_id,
            deceased_person_id=person_id,
            opened_step=step,
        )
        self.world.estate_accounts[estate_id] = account
        self.world.shareholder_estate_by_deceased[person_id] = estate_id

        for firm, shares in holdings:
            firm_id = getattr(firm, "firm_id", 0)
            table = firm.cap_table
            basis = self._person_basis(person, firm_id, shares)
            firm.cap_table = table.with_person_to_estate(
                person_id, estate_id, shares
            )
            if hasattr(self.world, "equity_ownership_system"):
                self.world.equity_ownership_system.cap_tables[firm_id] = (
                    firm.cap_table
                )
            account.shares_by_firm[firm_id] = shares
            account.cost_basis_by_firm[firm_id] = basis
            self._set_person_holding(person, firm_id, 0.0)
            self._set_person_basis(person, firm_id, 0.0)

            source_household = self.world.get_household(
                getattr(person, "household_id", None)
            )
            if source_household is not None and basis > 0.0:
                source_household.equity_asset_value = max(
                    0.0,
                    float(source_household.equity_asset_value) - basis,
                )

            self._record_event({
                "event_type": "person_shares_to_estate",
                "global_step": step,
                "estate_id": estate_id,
                "deceased_person_id": person_id,
                "firm_id": firm_id,
                "shares_transferred": shares,
                "shares_before": table.total_shares,
                "shares_after": firm.cap_table.total_shares,
                "cost_basis_transferred": basis,
            })

        heirs = self._valid_heirs(person)
        account.heir_person_ids = [heir.id for heir in heirs]
        if heirs:
            self._distribute(account, heirs, step)
        return account

    def _distribute(self, account, heirs, step):
        if account.status != "open":
            return
        share_count = len(heirs)
        for firm in self._firms():
            firm_id = getattr(firm, "firm_id", 0)
            shares = float(account.shares_by_firm.get(firm_id, 0.0))
            if shares <= 1e-12:
                continue
            for heir in heirs:
                amount = shares / share_count
                firm.cap_table = firm.cap_table.with_estate_to_person(
                    account.estate_id, heir.id, amount
                )
                if hasattr(self.world, "equity_ownership_system"):
                    self.world.equity_ownership_system.cap_tables[firm_id] = (
                        firm.cap_table
                    )
                self._set_person_holding(
                    heir,
                    firm_id,
                    float(getattr(heir, "equity_holdings", {}).get(firm_id, 0.0))
                    + amount,
                )
                basis = float(
                    account.cost_basis_by_firm.get(firm_id, 0.0)
                ) / share_count
                self._set_person_basis(
                    heir,
                    firm_id,
                    float(
                        getattr(heir, "equity_cost_basis", {}).get(
                            firm_id, 0.0
                        )
                    )
                    + basis,
                )
                heir_household = self.world.get_household(heir.household_id)
                if heir_household is not None and basis > 0.0:
                    heir_household.equity_asset_value += basis
                self._record_event({
                    "event_type": "estate_shares_to_heir",
                    "global_step": step,
                    "estate_id": account.estate_id,
                    "deceased_person_id": account.deceased_person_id,
                    "heir_person_id": heir.id,
                    "heir_household_id": getattr(heir, "household_id", None),
                    "firm_id": firm_id,
                    "shares_transferred": amount,
                    "cost_basis_transferred": basis,
                })
            account.shares_by_firm[firm_id] = 0.0

        if account.cash > 0.0:
            amount = account.cash / share_count
            for heir in heirs:
                household = self.world.get_household(heir.household_id)
                if household is None:
                    continue
                self.world.ledger.transfer(
                    payer=object_account(
                        account,
                        "cash",
                        f"{account.estate_id}.cash",
                    ),
                    receiver=object_account(
                        household,
                        "wealth",
                        f"household.{household.id}.wealth",
                    ),
                    amount=amount,
                    reason="estate_cash_to_heir",
                )

        account.status = "distributed"
        account.closed_step = step
        self._record_event({
            "event_type": "estate_closed_to_heirs",
            "global_step": step,
            "estate_id": account.estate_id,
            "deceased_person_id": account.deceased_person_id,
            "heir_count": share_count,
        })

    def resolve_open_estate(self, estate_id, heir_person_ids, step=None):
        self.ensure_state()
        account = self.world.estate_accounts.get(estate_id)
        if account is None:
            raise ValueError(f"unknown EstateAccount: {estate_id}")
        heirs = []
        for person_id in sorted(set(heir_person_ids)):
            person = self.world.person_dict.get(person_id)
            if person is not None and getattr(person, "alive", False):
                heirs.append(person)
        if not heirs:
            return False
        self._distribute(
            account,
            heirs,
            int(getattr(self.world, "current_step_index", 0) if step is None else step),
        )
        return True


__all__ = ["EstateAccount", "ShareholderEstateSystem", "SHAREHOLDER_ESTATE_SCHEMA"]
