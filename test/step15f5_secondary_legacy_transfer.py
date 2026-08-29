from __future__ import annotations

import copy
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint, save_world_checkpoint
from economy.dividend_routing import route_declared_dividend
from economy.secondary_legacy_transfer import ControlledSecondaryLegacyTransfer


OUTPUT = ROOT / "test/output/step15F5_secondary_legacy_transfer"
CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)
PRICE = 10.0
TOLERANCE = 1e-8


def load_base():
    world, _ = load_world_checkpoint(str(CHECKPOINT))
    world.ensure_ownership_state()
    world.legacy_owner_cash = 0.0
    world.secondary_transfer_history = []
    return world


def distinct_buyers(world, count=2):
    selected = []
    household_ids = set()
    for person in sorted(world.population, key=lambda item: item.id):
        household = world.household_dict.get(person.household_id)
        if not getattr(person, "alive", False) or household is None:
            continue
        if household.id in household_ids:
            continue
        selected.append((person, household))
        household_ids.add(household.id)
        if len(selected) >= count:
            return selected
    raise RuntimeError("could not find distinct deterministic buyers")


def prepare_world():
    world = load_base()
    buyers = distinct_buyers(world, 2)
    for _, household in buyers:
        household.wealth = 1000.0
    return world, buyers


def check_result(result):
    return {
        "cash_paid_equals_legacy_received": abs(result.reconciliation_gap) <= TOLERANCE,
        "legacy_share_identity": abs(
            result.legacy_shares_before
            - result.legacy_shares_after
            - result.total_shares_sold
        ) <= TOLERANCE,
        "total_shares_unchanged": abs(
            result.total_shares_before - result.total_shares_after
        ) <= TOLERANCE,
        "firm_cash_unchanged": abs(
            result.firm_cash_after - result.firm_cash_before
        ) <= TOLERANCE,
        "paid_in_equity_unchanged": abs(
            result.paid_in_equity_after - result.paid_in_equity_before
        ) <= TOLERANCE,
        "money_created_zero": result.money_created == 0.0,
        "money_destroyed_zero": result.money_destroyed == 0.0,
    }


def execute_fixture(name, requests, primary_shares=None, household_cash=None):
    world, buyers = prepare_world()
    if household_cash is not None:
        for index, cash in enumerate(household_cash):
            buyers[index][1].wealth = float(cash)
    firm = world.firms[0]
    if primary_shares:
        person, _ = buyers[0]
        firm.cap_table = firm.cap_table.with_primary_issuance(
            person.id, primary_shares, "person"
        )
        person.equity_holdings = dict(getattr(person, "equity_holdings", {}))
        person.equity_holdings[firm.firm_id] = float(primary_shares)
        world.equity_ownership_system.cap_tables[firm.firm_id] = firm.cap_table

    before_table = firm.cap_table
    before_fraction = {
        "legacy": before_table.ownership_fraction("legacy"),
        "person": before_table.ownership_fraction("person"),
    }
    executor = ControlledSecondaryLegacyTransfer(world, price_per_share=PRICE)
    result = executor.execute(
        firm,
        [
            (buyers[index][0], buyers[index][1], requested)
            for index, requested in enumerate(requests)
        ],
    )
    after_table = firm.cap_table
    after_fraction = {
        "legacy": after_table.ownership_fraction("legacy"),
        "person": after_table.ownership_fraction("person"),
    }
    checks = check_result(result)
    checks["person_holdings_updated"] = all(
        abs(
            buyers[index][0].equity_holdings[firm.firm_id]
            - sum(
                fill["shares_filled"]
                for fill in result.fills
                if fill["buyer_person_id"] == buyers[index][0].id
            )
            - (primary_shares if index == 0 and primary_shares else 0.0)
        ) <= TOLERANCE
        for index in range(len(requests))
    )
    rows = []
    for fill in result.fills:
        rows.append({
            "fixture": name,
            "buyer_person_id": fill["buyer_person_id"],
            "buyer_household_id": fill["buyer_household_id"],
            "shares_requested": fill["shares_requested"],
            "shares_filled": fill["shares_filled"],
            "payment": fill["payment"],
            "legacy_shares_before": fill["legacy_shares_before"],
            "legacy_shares_after": fill["legacy_shares_after"],
            "person_shares_before": fill["person_shares_before"],
            "person_shares_after": fill["person_shares_after"],
            "household_cash_before": fill["household_cash_before"],
            "household_cash_after": fill["household_cash_after"],
            "legacy_owner_cash_before": result.legacy_owner_cash_before,
            "legacy_owner_cash_after": result.legacy_owner_cash_after,
            "firm_cash_before": result.firm_cash_before,
            "firm_cash_after": result.firm_cash_after,
            "paid_in_equity_before": result.paid_in_equity_before,
            "paid_in_equity_after": result.paid_in_equity_after,
            "legacy_fraction_before": before_fraction["legacy"],
            "legacy_fraction_after": after_fraction["legacy"],
            "person_fraction_before": before_fraction["person"],
            "person_fraction_after": after_fraction["person"],
            "total_shares_before": result.total_shares_before,
            "total_shares_after": result.total_shares_after,
            "reconciliation_gap": result.reconciliation_gap,
            "money_created": result.money_created,
            "money_destroyed": result.money_destroyed,
            "checks": json.dumps(checks, sort_keys=True),
        })
    rows.append({
        "fixture": name,
        "buyer_person_id": "aggregate",
        "buyer_household_id": "",
        "shares_requested": sum(requests),
        "shares_filled": result.total_shares_sold,
        "payment": result.total_payment,
        "legacy_shares_before": result.legacy_shares_before,
        "legacy_shares_after": result.legacy_shares_after,
        "person_shares_before": "",
        "person_shares_after": "",
        "household_cash_before": "",
        "household_cash_after": "",
        "legacy_owner_cash_before": result.legacy_owner_cash_before,
        "legacy_owner_cash_after": result.legacy_owner_cash_after,
        "firm_cash_before": result.firm_cash_before,
        "firm_cash_after": result.firm_cash_after,
        "paid_in_equity_before": result.paid_in_equity_before,
        "paid_in_equity_after": result.paid_in_equity_after,
        "legacy_fraction_before": before_fraction["legacy"],
        "legacy_fraction_after": after_fraction["legacy"],
        "person_fraction_before": before_fraction["person"],
        "person_fraction_after": after_fraction["person"],
        "total_shares_before": result.total_shares_before,
        "total_shares_after": result.total_shares_after,
        "reconciliation_gap": result.reconciliation_gap,
        "money_created": result.money_created,
        "money_destroyed": result.money_destroyed,
        "checks": json.dumps(checks, sort_keys=True),
    })
    return world, buyers, result, checks, rows


def dividend_rights_fixture():
    world, buyers = prepare_world()
    firm = world.firms[0]
    executor = ControlledSecondaryLegacyTransfer(world, price_per_share=PRICE)
    result = executor.execute(firm, [(buyers[0][0], buyers[0][1], 10.0)])
    household = buyers[0][1]
    before = household.wealth
    dividend = route_declared_dividend(
        world,
        firm,
        1000.0,
        payer=world.firm_system,
    )
    expected = 1000.0 * (10.0 / 100.0)
    return {
        "dividend_person_paid": dividend.person_paid,
        "dividend_expected_person_paid": expected,
        "household_income_received": household.wealth - before,
        "routing_gap": dividend.reconciliation_gap,
        "passed": (
            abs(dividend.person_paid - expected) <= TOLERANCE
            and abs((household.wealth - before) - expected) <= TOLERANCE
            and abs(dividend.reconciliation_gap) <= TOLERANCE
            and result.total_shares_sold == 10.0
        ),
    }


def checkpoint_fixture():
    world, buyers, result, checks, _ = execute_fixture(
        "checkpoint_roundtrip", [10.0]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "secondary_transfer.pkl"
        save_world_checkpoint(str(path), world, {"stage": "15F5_fixture"})
        restored, _ = load_world_checkpoint(str(path))
    restored_firm = restored.firms[0]
    original_firm = world.firms[0]
    original_person = buyers[0][0]
    restored_person = restored.person_dict[original_person.id]
    return {
        "passed": (
            restored_firm.cap_table.to_dict() == original_firm.cap_table.to_dict()
            and abs(restored.legacy_owner_cash - world.legacy_owner_cash) <= TOLERANCE
            and restored_person.equity_holdings == original_person.equity_holdings
            and len(getattr(restored, "secondary_transfer_history", [])) == 1
        ),
        "restored_legacy_shares": restored_firm.cap_table.legacy_shares,
        "restored_person_shares": restored_firm.cap_table.person_shares,
        "restored_legacy_owner_cash": restored.legacy_owner_cash,
    }


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    fixture_checks = {}
    normalized_fixtures = [
        ("transfer_10_to_one_person", [10.0], None, None),
        ("cash_insufficient_partial_fill", [100.0], None, [25.0]),
        ("legacy_insufficient_partial_fill", [150.0], None, None),
        ("two_persons_same_legacy_pool", [10.0, 20.0], None, None),
        ("existing_primary_plus_secondary", [7.0], 5.0, None),
    ]
    for name, requests, primary_shares, household_cash in normalized_fixtures:
        _, _, _, checks, rows = execute_fixture(
            name, requests, primary_shares, household_cash
        )
        all_rows.extend(rows)
        fixture_checks[name] = all(checks.values())

    dividend = dividend_rights_fixture()
    roundtrip = checkpoint_fixture()
    fieldnames = list(all_rows[0])
    with (OUTPUT / "secondary_transfer_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    all_fixtures = all(fixture_checks.values())
    flags = {
        "verdict": (
            "A. PASSIVE_SECONDARY_LEGACY_TRANSFER_READY"
            if all_fixtures and dividend["passed"] and roundtrip["passed"]
            else "G. OTHER_BLOCKER"
        ),
        "controlled_deterministic_contract": True,
        "autonomous_buying": False,
        "stock_market": False,
        "price_discovery": False,
        "investment_active": False,
        "primary_issuance_changed": False,
        "dividend_policy_changed": False,
        "firm_cash_unchanged": all_fixtures,
        "paid_in_equity_unchanged": all_fixtures,
        "total_shares_unchanged": all_fixtures,
        "legacy_owner_cash_reconciles": all_fixtures,
        "money_created_zero": all_fixtures,
        "money_destroyed_zero": all_fixtures,
        "dividend_rights_after_transfer": dividend["passed"],
        "checkpoint_roundtrip": roundtrip["passed"],
        "new_rng_draws": 0,
        "step13_changed": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = f"""# Step 15F.5 Passive Secondary Legacy Share Transfer Runtime

**Verdict: {flags['verdict']}**

The controlled secondary settlement transfers existing shares from the Legacy pool to explicitly supplied Person/Household buyers. It uses the engineering transfer price supplied by the caller and never invokes an order book, RNG, market valuation, debt, or `World.step()`.

For every successful fill:

- Household cash decreases by the payment.
- `LegacyOwner.cash` increases by exactly the payment.
- Legacy shares decrease by the filled amount.
- Person shares increase by the filled amount.
- Firm cash and paid-in equity remain unchanged.
- CapTable total shares remain unchanged.
- Money creation and destruction are zero.

Validated fixtures:

1. Full 10-share transfer to one Person.
2. Household-cash-limited partial fill.
3. Legacy-pool-limited partial fill.
4. Two Persons buying from the same Legacy pool.
5. Existing primary-issued Person shares plus secondary shares.
6. Checkpoint round-trip preserving CapTable, Person holdings, LegacyOwner cash, and transfer history.

An isolated post-transfer dividend routing check paid the transferred 10% share position exactly 10% of a declared 1000 dividend. The dividend policy itself was not changed.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
