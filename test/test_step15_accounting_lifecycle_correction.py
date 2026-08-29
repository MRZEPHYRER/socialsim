from types import SimpleNamespace

from economy.accounting import AccountingLayer
from marriage import MarriageSystem


class _Person:
    def __init__(self, person_id, sex, household_id):
        self.id = person_id
        self.sex = sex
        self.age = 28
        self.age_years = 28
        self.household_id = household_id
        self.partner_id = None
        self.is_seeking_partner = True
        self.alive = True


class _Household:
    def __init__(self, household_id):
        self.id = household_id
        self.wealth = 0.0
        self.parents = []


class _MarriageWorld:
    seed = 42

    def __init__(self):
        self.male = _Person(1, "M", 10)
        self.female = _Person(2, "F", 11)
        self.population = [self.male, self.female]
        self.pending_household_formation_wealth = 12.0
        self.transfer_calls = []
        self.lifecycle_events = []
        self.demographic_events = []

    def leave_household(self, person):
        person.household_id = None
        return 5.0 if person.sex == "M" else 7.0

    def create_household(self):
        self.created_household = _Household(100)
        return self.created_household

    def merge_settlement_household_into_social_household(self, person, destination):
        amount = 3.0 if person.sex == "M" else 4.0
        destination.wealth += amount
        return amount

    def transfer_signed_balance(
        self,
        source_obj,
        source_attr,
        destination_obj,
        destination_attr,
        amount,
        **_kwargs,
    ):
        setattr(source_obj, source_attr, getattr(source_obj, source_attr) - amount)
        setattr(destination_obj, destination_attr, getattr(destination_obj, destination_attr) + amount)
        self.transfer_calls.append(amount)

    def record_lifecycle_transfer_event(self, **event):
        self.lifecycle_events.append(event)

    def record_household_lifecycle_event(self, **event):
        self.lifecycle_events.append(event)

    def add_person_to_household(self, person, household, _role):
        person.household_id = household.id
        household.parents.append(person.id)

    def record_demographic_event(self, *_args, **event):
        self.demographic_events.append(event)


def _accounting_world():
    return SimpleNamespace(
        time_metadata=lambda _step: {},
        capital_good_customer_advance_enabled=True,
        firm_system=SimpleNamespace(
            central_bank=SimpleNamespace(
                firm_loan_balance=0.0,
                interest_receivable=0.0,
                money_supply=0.0,
                public_income_balance=0.0,
            )
        ),
        households=[],
        public_wealth=0.0,
        legacy_owner_cash=0.0,
        estate_accounts={},
        pending_household_formation_wealth=0.0,
        initial_private_money_stock=0.0,
    )


def _firm(**overrides):
    values = {
        "firm_id": 100000,
        "sector_id": "capital_goods",
        "inventory_units": 0.0,
        "price": 10.0,
        "production": 0.0,
        "sales_units": 0.0,
        "spoilage_units": 0.0,
        "public_inventory_purchase_units": 0.0,
        "wage_payment": 0.0,
        "sales_revenue": 0.0,
        "public_sector_cash_inflow": 0.0,
        "other_cash_inflow": 0.0,
        "other_cash_outflow": 0.0,
        "loan_issued": 0.0,
        "loan_repaid": 0.0,
        "loan_interest_paid": 0.0,
        "interest_arrears": 3.0,
        "legacy_arrears_term_claim": 6.0,
        "loan_balance": 11.0,
        "dividend_payment": 0.0,
        "equity_issuance_cash_this_step": 0.0,
        "startup_capitalization_inflow_this_step": 0.0,
        "startup_capitalization_outflow_this_step": 0.0,
        "investment_expenditure_this_step": 0.0,
        "customer_advance_received_this_step": 0.0,
        "customer_advance_delivered_this_step": 0.0,
        "prepaid_capital_investment_paid_this_step": 0.0,
        "prepaid_capital_investment_capitalized_this_step": 0.0,
        "prepaid_capital_investment_asset": 0.0,
        "sales_collections_this_step": 0.0,
        "capital_stock": SimpleNamespace(total_remaining_book_value=0.0),
        "capital_depreciation_expense_this_step": 0.0,
        "capital_book_value_opening_this_step": 0.0,
        "capital_asset_acquisitions_this_step": 0.0,
        "capital_asset_disposals_this_step": 0.0,
        "capital_good_inventory": SimpleNamespace(last_cogs=0.0, book_value=0.0),
        "cash": 100.0,
        "cash_start": 100.0,
        "accounting_cash_start_this_step": 100.0,
        "customer_advance_liability": 17.0,
        "inventory_units": 0.0,
        "profit": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_marriage_keeps_settlement_cash_out_of_pending_transfer():
    world = _MarriageWorld()

    assert MarriageSystem(world).process_marriage() == 1

    # Five plus seven came from prior social households and was the only
    # amount held by the pending formation account. Three plus four moved
    # directly from temporary settlement accounts.
    assert world.transfer_calls == [12.0]
    assert world.pending_household_formation_wealth == 0.0
    assert world.created_household.wealth == 19.0
    pending_event = next(
        event
        for event in world.lifecycle_events
        if event.get("event_type") == "pending_formation_to_new_household"
    )
    assert pending_event["amount"] == 12.0
    assert world.demographic_events[0]["wealth_transferred"] == 19.0


def test_accounting_exports_true_loan_and_advance_components_separately():
    accounting = AccountingLayer(_accounting_world())
    accounting.record_firms(0, [_firm()])
    row = accounting.rows[-1]

    assert row["loan_principal"] == 11.0
    assert row["loan_balance"] == 11.0
    assert row["customer_advance_liability"] == 17.0
    assert row["revolving_principal_claim"] == 11.0
    assert row["total_lender_claim"] == 20.0
    assert row["total_liabilities"] == 37.0
    assert accounting.reconciliation_rows[-1]["firm_loan_liabilities"] == 11.0


def test_startup_capitalization_is_financing_not_operating_revenue():
    accounting = AccountingLayer(_accounting_world())
    accounting.record_firms(
        0,
        [_firm(
            cash=10.0,
            cash_start=0.0,
            accounting_cash_start_this_step=0.0,
            loan_balance=0.0,
            interest_arrears=0.0,
            legacy_arrears_term_claim=0.0,
            customer_advance_liability=0.0,
            startup_capitalization_inflow_this_step=10.0,
        )],
    )
    row = accounting.rows[-1]

    assert row["other_existing_operating_revenue"] == 0.0
    assert row["accounting_revenue"] == 0.0
    assert row["accounting_operating_profit"] == 0.0
    assert row["startup_capitalization_financing_cashflow"] == 10.0
    assert row["cfo"] == 0.0
    assert row["cff"] == 10.0
    assert row["cash_flow_gap"] == 0.0
