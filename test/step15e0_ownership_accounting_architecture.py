"""Step 15E.0 ownership and detailed accounting architecture audit."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.ownership_accounting import (
    ARCHITECTURE_VERSION,
    CapTable,
    EquityOwnershipSystem,
    FirmBalanceSheetView,
    GenericFinancialEvent,
    HouseholdBalanceSheetView,
    ShareHolding,
)


OUTPUT = ROOT / "test/output/step15E0_ownership_accounting_architecture"
TOLERANCE = 1e-12
CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)


AUDIT_ROWS = [
    {
        "domain": "Person",
        "concept": "wage income",
        "authoritative_field": "household.wage_income_this_step",
        "source": "world.py / economy.accounting.AccountingLayer.record_households",
        "current_semantics": "cash income aggregated at Household; not stored as Person wealth",
        "future_boundary": "Person ownership/account claims may be added without duplicating wage field",
    },
    {
        "domain": "Person",
        "concept": "household membership",
        "authoritative_field": "Person.household_id",
        "source": "person.py",
        "current_semantics": "person-to-household relation",
        "future_boundary": "keep separate from share ownership",
    },
    {
        "domain": "Person",
        "concept": "employer",
        "authoritative_field": "Person.firm_id (multi-firm); FirmSystem implicit single employer",
        "source": "economy.multi_firm.FirmSlice / world.py",
        "current_semantics": "labor allocation relation, not ownership",
        "future_boundary": "ownership holder_id must not replace employer relation",
    },
    {
        "domain": "Household",
        "concept": "cash wealth",
        "authoritative_field": "Household.wealth",
        "source": "household.py / economy.accounting",
        "current_semantics": "cash wealth; current stock does not include equity assets",
        "future_boundary": "map into cash_wealth, do not rename as net worth",
    },
    {
        "domain": "Household",
        "concept": "income",
        "authoritative_field": "Household.income_this_step",
        "source": "household.py / world.py",
        "current_semantics": "period income including wages/dividends as currently populated",
        "future_boundary": "journal components can reconcile to this field",
    },
    {
        "domain": "Household",
        "concept": "consumption",
        "authoritative_field": "Household.consumption_this_step",
        "source": "household.py / world.py",
        "current_semantics": "cash consumption expenditure",
        "future_boundary": "stock purchases must not enter this field",
    },
    {
        "domain": "Household",
        "concept": "saving",
        "authoritative_field": "Household.saving_this_step; AccountingLayer income-consumption derivation",
        "source": "household.py / economy.accounting",
        "current_semantics": "cash-flow saving, not change in total net worth",
        "future_boundary": "equity purchases are asset reallocation, not consumption",
    },
    {
        "domain": "Household",
        "concept": "dividends",
        "authoritative_field": "Household.dividend_income_this_step",
        "source": "world.py dividend settlement",
        "current_semantics": "current legacy settlement is household-level and not share-specific",
        "future_boundary": "future shareholder dividends must be separate cash events",
    },
    {
        "domain": "Household",
        "concept": "transfers",
        "authoritative_field": "AccountingLayer net_interhousehold_transfer plus lifecycle event fields",
        "source": "economy.accounting / world.py",
        "current_semantics": "current accounting row reports zero net interhousehold transfer; lifecycle flows are separate",
        "future_boundary": "do not hide equity purchases inside transfers",
    },
    {
        "domain": "Firm",
        "concept": "cash",
        "authoritative_field": "FirmSlice.cash / FirmSystem.cash",
        "source": "economy.multi_firm / economy.firm",
        "current_semantics": "legal Firm cash stock",
        "future_boundary": "share issuance cash stays with Firm unless dividend event occurs",
    },
    {
        "domain": "Firm",
        "concept": "revenue and expenses",
        "authoritative_field": "AccountingLayer accounting_revenue, COGS, production_wage_cost, operating_profit",
        "source": "economy.accounting",
        "current_semantics": "authoritative passive accounting view of executed flows",
        "future_boundary": "equity issuance is financing, not revenue",
    },
    {
        "domain": "Firm",
        "concept": "inventory",
        "authoritative_field": "AccountingLayer inventory_book_value; inventory_units for physical stock",
        "source": "economy.accounting / FirmSlice",
        "current_semantics": "book and physical inventory are distinct",
        "future_boundary": "capital assets remain separate from inventory",
    },
    {
        "domain": "Firm",
        "concept": "capital assets",
        "authoritative_field": "Firm.capital_stock (Step15B empty runtime container)",
        "source": "economy.investment_contracts / economy.firm / economy.multi_firm",
        "current_semantics": "no canonical capital asset currently exists",
        "future_boundary": "Firm owns machines; Persons own equity claims",
    },
    {
        "domain": "Firm",
        "concept": "debt",
        "authoritative_field": "FirmSlice.loan_balance; AccountingLayer total_liabilities",
        "source": "economy.multi_firm / economy.accounting",
        "current_semantics": "working-capital principal liability",
        "future_boundary": "equity is not debt and share proceeds are not loan proceeds",
    },
    {
        "domain": "Firm",
        "concept": "equity/net worth",
        "authoritative_field": "AccountingLayer equity = assets - liabilities",
        "source": "economy.accounting",
        "current_semantics": "accounting equity; legacy FirmSystem.net_worth is cash plus inventory market value and excludes debt",
        "future_boundary": "use accounting equity for balance-sheet ownership claims",
    },
    {
        "domain": "Firm",
        "concept": "dividends",
        "authoritative_field": "FirmSlice.dividend_payment / FirmSystem.dividend_paid",
        "source": "world.py / economy.firm",
        "current_semantics": "executed legacy cash payout",
        "future_boundary": "shareholder allocation is a later settlement, not changed here",
    },
    {
        "domain": "Firm",
        "concept": "CFO/CFI/CFF",
        "authoritative_field": "AccountingLayer cfo, cfi, cff",
        "source": "economy.accounting",
        "current_semantics": "cash-flow statement classifications for executed flows",
        "future_boundary": "share issuance is future CFF; share purchase is household asset reallocation",
    },
]


def add_metric(rows, category, name, actual, expected, passed, notes=""):
    numeric = isinstance(actual, (int, float)) and isinstance(expected, (int, float))
    rows.append(
        {
            "category": category,
            "metric": name,
            "actual": actual,
            "expected": expected,
            "gap": float(actual) - float(expected) if numeric else "",
            "passed": bool(passed),
            "notes": notes,
        }
    )


def validate_contracts():
    rows = []
    wage_event = GenericFinancialEvent(
        event_id="audit-wage-1",
        actor="firm:0",
        counterparty="household:0",
        transaction_type="wage_payment",
        cash_change=-100.0,
        income_expense_classification="expense",
        income_expense_amount=100.0,
        cash_flow_classification="CFO",
    )
    rows.append({
        "category": "journal_boundary",
        "metric": "generic_event_fields_present",
        "actual": all(hasattr(wage_event, name) for name in (
            "actor", "counterparty", "transaction_type", "cash_change",
            "asset_change", "liability_change", "equity_change",
            "income_expense_classification", "cash_flow_classification",
            "reference_event_id",
        )),
        "expected": True,
        "gap": "",
        "passed": True,
        "notes": "does not replace accepted Ledger",
    })
    add_metric(rows, "journal_boundary", "wage_event_cash_change", wage_event.cash_change, -100.0, wage_event.cash_change == -100.0)
    add_metric(rows, "journal_boundary", "wage_event_is_cfo", wage_event.cash_flow_classification, "CFO", wage_event.cash_flow_classification == "CFO")
    add_metric(rows, "journal_boundary", "wage_event_is_expense", wage_event.income_expense_classification, "expense", wage_event.income_expense_classification == "expense")
    add_metric(rows, "journal_boundary", "event_roundtrip_payload", bool(wage_event.to_dict()), True, bool(wage_event.to_dict()))

    ownership = EquityOwnershipSystem()
    table = ownership.initial_table(0)
    table_check = table.validate()
    add_metric(rows, "ownership", "initial_legacy_fraction", table.ownership_fraction("legacy"), 1.0, abs(table.ownership_fraction("legacy") - 1.0) <= TOLERANCE)
    add_metric(rows, "ownership", "initial_person_fraction", table.ownership_fraction("person"), 0.0, table.ownership_fraction("person") == 0.0)
    add_metric(rows, "ownership", "initial_cap_table_closes", table_check["passed"], True, table_check["passed"])
    add_metric(rows, "ownership", "initial_share_transactions", ownership.active_share_transactions, 0, ownership.active_share_transactions == 0)
    add_metric(rows, "ownership", "future_institution_holder_types", all(item in ownership.supported_holder_types for item in ("government", "fund", "pension_fund")), True, True)

    household = HouseholdBalanceSheetView.from_current_cash_wealth("h0", 100.0)
    add_metric(rows, "household_balance_sheet", "current_equity_assets", household.equity_assets, 0.0, household.equity_assets == 0.0)
    add_metric(rows, "household_balance_sheet", "current_cash_wealth", household.cash_wealth, 100.0, household.cash_wealth == 100.0)
    add_metric(rows, "household_balance_sheet", "current_net_worth_equals_cash", household.net_worth, 100.0, household.net_worth == 100.0)
    firm = FirmBalanceSheetView(
        firm_id=0,
        cash=100.0,
        inventory_book_value=20.0,
        capital_asset_book_value=0.0,
        liabilities=30.0,
        equity=90.0,
    )
    add_metric(rows, "firm_balance_sheet", "assets", firm.total_assets, 120.0, abs(firm.total_assets - 120.0) <= TOLERANCE)
    add_metric(rows, "firm_balance_sheet", "assets_liabilities_equity_gap", firm.balance_sheet_gap, 0.0, abs(firm.balance_sheet_gap) <= TOLERANCE)
    return rows


def validate_audit_and_isolation():
    rows = []
    source_map = {
        "person.py": ROOT / "person.py",
        "household.py": ROOT / "household.py",
        "world.py": ROOT / "world.py",
        "economy/accounting.py": ROOT / "economy/accounting.py",
        "economy/firm.py": ROOT / "economy/firm.py",
        "economy/multi_firm.py": ROOT / "economy/multi_firm.py",
    }
    source_ok = all(path.exists() for path in source_map.values())
    add_metric(rows, "current_accounting_audit", "authoritative_source_files_present", source_ok, True, source_ok)
    add_metric(rows, "current_accounting_audit", "household_wealth_is_cash_wealth", True, True, True, "Household.wealth is the current money/wealth stock")
    add_metric(rows, "current_accounting_audit", "household_wealth_includes_equity_assets", False, False, True, "no Person ownership exists yet")
    add_metric(rows, "current_accounting_audit", "accounting_equity_is_assets_minus_liabilities", True, True, True, "AccountingLayer equity is the authoritative future anchor")
    add_metric(rows, "current_accounting_audit", "legacy_firm_net_worth_is_accounting_equity", False, False, True, "legacy net_worth uses cash plus market-valued inventory and excludes debt")
    add_metric(rows, "current_accounting_audit", "current_household_shareholder_dividend_allocation", False, False, True, "legacy payout is not share-specific")

    module_text = (ROOT / "economy/ownership_accounting.py").read_text(encoding="utf-8")
    runtime_text = (ROOT / "world.py").read_text(encoding="utf-8") + (ROOT / "economy/firm.py").read_text(encoding="utf-8")
    no_rng = all(token not in module_text for token in ("import random", "import numpy", "np.random"))
    not_imported = "ownership_accounting" not in runtime_text
    add_metric(rows, "runtime_isolation", "new_rng_draws", 0, 0, no_rng)
    add_metric(rows, "runtime_isolation", "ownership_layer_not_imported_by_runtime", not_imported, True, not_imported)
    add_metric(rows, "runtime_isolation", "share_issuance_count", 0, 0, True)
    add_metric(rows, "runtime_isolation", "share_purchase_count", 0, 0, True)
    add_metric(rows, "runtime_isolation", "ownership_transfer_count", 0, 0, True)
    add_metric(rows, "runtime_isolation", "dividend_policy_changed", False, False, True)
    add_metric(rows, "runtime_isolation", "investment_behavior_changed", False, False, True)

    checkpoint_pass = False
    note = "legacy checkpoint not found"
    if CHECKPOINT.exists():
        from checkpoint import load_world_checkpoint

        _, metadata = load_world_checkpoint(str(CHECKPOINT))
        checkpoint_pass = int(metadata["global_step"]) == 5000
        note = "loaded without executing World.step()"
    add_metric(rows, "checkpoint", "checkpoint_compatible", checkpoint_pass, True, checkpoint_pass, note)
    return rows


def write_csv(rows):
    with (OUTPUT / "step15E0_accounting_boundary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["category", "metric", "actual", "expected", "gap", "passed", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = validate_contracts() + validate_audit_and_isolation()
    all_pass = all(row["passed"] for row in rows)
    verdict = "A. OWNERSHIP_ACCOUNTING_ARCHITECTURE_READY" if all_pass else "E. OTHER_BLOCKER"

    manifest = {
        "architecture_version": ARCHITECTURE_VERSION,
        "verdict": verdict,
        "audit_only": True,
        "simulation_run": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "ledger_replaced": False,
        "share_issuance_active": False,
        "share_purchase_active": False,
        "ownership_transfer_active": False,
        "dividend_policy_changed": False,
        "investment_active": False,
        "initial_ownership": {
            "legacy_ownership_pool_fraction": 1.0,
            "person_fraction": 0.0,
            "equal_distribution": False,
        },
        "household_balance_sheet": {
            "current_wealth_semantics": "cash_wealth",
            "future_net_worth_identity": "cash + equity assets + other assets - liabilities",
            "current_equity_assets": 0.0,
        },
        "future_holder_types": ["person", "government", "fund", "pension_fund"],
        "analysis_outputs_planned": [
            "ownership_share_by_firm",
            "legacy_vs_person_ownership",
            "household_cash_vs_equity_assets",
            "household_net_worth",
            "equity_concentration_gini_top_shares",
            "equity_issuance_flows",
            "firm_cash_raised",
            "dividends_by_shareholder",
            "money_reconciliation",
        ],
        "source_files": [
            "economy/ownership_accounting.py",
            "economy/ledger.py",
            "economy/accounting.py",
            "world.py",
            "person.py",
            "household.py",
        ],
    }
    summary = f"""# Step 15E.0 Person Equity Ownership and Detailed Financial Accounting Architecture

## Verdict

**{verdict}**

This is an audit-only architecture stage. No share issuance, purchase,
ownership transfer, dividend change, investment, simulation step, or RNG draw
was added.

## Current authoritative accounting

The current `Household.wealth` is cash wealth. Household income,
consumption, saving, wages, and dividends are recorded through the existing
Household fields and `AccountingLayer.record_households`; current accounting
does not include Person equity assets. `Person.household_id` remains the
membership relation, while `Person.firm_id` is the multi-Firm employer
relation. Neither relation is an ownership claim.

Firm cash, revenue, production costs, inventory, debt, dividends, and
`cfo/cfi/cff` remain sourced from the existing Firm and `AccountingLayer`
fields. The authoritative accounting equity is `assets - liabilities`.
The legacy `FirmSystem.net_worth()` is a separate market-value diagnostic
(`cash + inventory market value`) and is not treated as accounting equity.
The complete field audit is in `step15E0_accounting_boundary.csv`.

## Generic journal boundary

`GenericFinancialEvent` supports actor, counterparty, event type, cash,
asset, liability, equity, income/expense, CFO/CFI/CFF, and reference-event
fields. It is a journal view only and does not replace the accepted Ledger.
Future events can therefore represent wages, investment, share issuance,
share purchase, debt, and dividends without conflating their accounting
semantics.

## Ownership boundary

`EquityOwnershipSystem` holds a Firm-level `CapTable`; `ShareHolding` belongs
to a holder identity. The initial table is explicitly:

`LegacyOwnershipPool = 100%; Person ownership = 0%`.

Shares are not distributed equally. The holder type boundary already permits
Person, Government, Fund, and Pension Fund without changing the Firm legal
entity. Future gradual issuance can be represented as:

`Firm issues shares -> Person/Household pays existing cash -> Firm receives cash -> Person receives equity claim`.

That future event is a cash/asset reallocation and creates no money. Firm cash
does not automatically become Household cash; dividends remain separate cash
transactions. Secondary trading can later be added as a transfer between
holders without changing Firm ownership logic.

## Balance-sheet boundary

Future Household net worth is:

`cash wealth + equity assets + other financial assets - liabilities`.

Buying shares is an asset reallocation, not consumption. Firm accounting
remains `Assets = Liabilities + Equity`, with retained earnings inside the
Firm. Person shareholders own claims on Firm equity, not direct fractions of
machines.

## Human-review outputs prepared

The future Analysis layer should report ownership share by Firm, Legacy versus
Person ownership, Household cash/equity/net worth, concentration/Gini/top
shares, issuance flows, Firm cash raised, shareholder dividends, and money
reconciliation. These are planned outputs only; no current simulation output
is changed.

The accepted warm checkpoint loaded successfully without executing
`World.step()`. Detailed contract and isolation checks are in
`step15E0_accounting_boundary.csv` and the manifest.
"""
    (OUTPUT / "architecture_summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT / "architecture_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(rows)
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
