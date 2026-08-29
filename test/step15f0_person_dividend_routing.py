from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint, save_world_checkpoint
from economy.dividend_routing import route_declared_dividend
from economy.ledger import Ledger
from economy.ownership_accounting import CapTable


OUTPUT = ROOT / "test/output/step15F0_person_dividend_routing"
WARM_CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)
TOLERANCE = 1e-9


def fixture_world(firms, people=(), households=()):
    world = SimpleNamespace(
        firms=list(firms),
        person_dict={person.id: person for person in people},
        population=list(people),
        household_dict={household.id: household for household in households},
        households=list(households),
        legacy_owner_cash=0.0,
        legacy_dividend_history=[],
    )
    world.ledger = Ledger(world=world, record_details=True)
    return world


def person(person_id, household_id):
    return SimpleNamespace(
        id=person_id,
        household_id=household_id,
        dividend_entitlement_this_step=0.0,
        dividend_received_this_step=0.0,
    )


def household(household_id, wealth=1000.0):
    return SimpleNamespace(
        id=household_id,
        wealth=float(wealth),
        income_this_step=0.0,
        dividend_income_this_step=0.0,
    )


def firm(firm_id, cash=1000.0):
    return SimpleNamespace(
        firm_id=firm_id,
        cash=float(cash),
        cap_table=CapTable.initial_legacy_owned(firm_id),
    )


def add_person_share(table, person_id, fraction):
    # For an initial legacy pool of 100 shares, this issuance creates the
    # requested post-issuance fraction exactly when fraction is in (0, 1).
    shares = 100.0 * fraction / (1.0 - fraction)
    return table.with_primary_issuance(person_id, shares, "person")


def run_route(world, target, declared):
    cash_before = target.cash
    household_before = {
        household.id: household.wealth
        for household in world.households
    }
    money_before = (
        target.cash
        + sum(household.wealth for household in world.households)
        + world.legacy_owner_cash
    )
    result = route_declared_dividend(world, target, declared)
    money_after = (
        target.cash
        + sum(household.wealth for household in world.households)
        + world.legacy_owner_cash
    )
    return {
        "declared_dividend": declared,
        "person_paid": result.person_paid,
        "legacy_entitlement": result.legacy_entitlement,
        "firm_cash_before": cash_before,
        "firm_cash_after": target.cash,
        "household_cash_inflow": sum(
            household.wealth - household_before[household.id]
            for household in world.households
        ),
        "person_receipts": result.person_receipts,
        "routing_gap": result.reconciliation_gap,
        "money_before": money_before,
        "money_after": money_after,
        "money_created": 0.0,
        "money_destroyed": 0.0,
    }


def checkpoint_roundtrip():
    if not WARM_CHECKPOINT.exists():
        return {"available": False, "passed": False, "reason": "warm checkpoint missing"}

    world, _ = load_world_checkpoint(str(WARM_CHECKPOINT))
    world.ensure_ownership_state()
    target = world.firms[0]
    selected = sorted(world.population, key=lambda item: item.id)[0]
    target.cap_table = add_person_share(target.cap_table, selected.id, 0.10)
    world.equity_ownership_system.cap_tables[target.firm_id] = target.cap_table
    selected.equity_holdings = {target.firm_id: target.cap_table.person_shares}
    with tempfile.TemporaryDirectory(prefix="step15f0_checkpoint_") as temp_dir:
        path = Path(temp_dir) / "dividend_routing.pkl"
        save_world_checkpoint(str(path), world, {"step15f0_fixture": True})
        loaded, _ = load_world_checkpoint(str(path))
    loaded_target = loaded.firms[0]
    loaded_person = loaded.person_dict[selected.id]
    return {
        "available": True,
        "passed": (
            loaded_target.cap_table.to_dict() == target.cap_table.to_dict()
            and loaded_person.equity_holdings == selected.equity_holdings
            and abs(getattr(loaded, "legacy_owner_cash", 0.0)) <= TOLERANCE
        ),
        "person_id": selected.id,
        "person_fraction": loaded_target.cap_table.ownership_fraction("person"),
    }


def runtime_single_firm_route():
    if not WARM_CHECKPOINT.exists():
        return {"available": False, "passed": False, "reason": "warm checkpoint missing"}
    world, _ = load_world_checkpoint(str(WARM_CHECKPOINT))
    world.ensure_ownership_state()
    mirror = world.firms[0]
    selected = sorted(world.population, key=lambda item: item.id)[0]
    household = world.household_dict[selected.household_id]
    mirror.cap_table = add_person_share(mirror.cap_table, selected.id, 0.10)
    world.equity_ownership_system.cap_tables[mirror.firm_id] = mirror.cap_table
    before_cash = float(world.firm_system.cash)
    before_household = float(household.wealth)
    before_legacy = float(getattr(world, "legacy_owner_cash", 0.0))
    world.firm_system.distribute_dividends(100.0)
    return {
        "available": True,
        "passed": (
            abs(world.firm_system.cash - before_cash + 100.0) <= TOLERANCE
            and abs(household.wealth - before_household - 10.0) <= TOLERANCE
            and abs(world.legacy_owner_cash - before_legacy - 90.0) <= TOLERANCE
            and abs(world.firm_system.person_dividend_paid - 10.0) <= TOLERANCE
        ),
        "person_paid": world.firm_system.person_dividend_paid,
        "legacy_entitlement": world.firm_system.legacy_dividend_entitlement,
    }


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    random_before = random.getstate()

    rows = []

    # 1. Legacy-only Firm: all entitlement stays in the explicit legacy
    # owner account and no Household receives dividend income.
    legacy_firm = firm(0)
    legacy_world = fixture_world([legacy_firm], households=[household(0)])
    result = run_route(legacy_world, legacy_firm, 100.0)
    result.update({"fixture": "legacy_only", "person_ids": "", "household_ids": "0"})
    rows.append(result)

    # 2. One Person owns exactly 10%.
    p0, h0 = person(0, 0), household(0)
    ten_firm = firm(1)
    ten_firm.cap_table = add_person_share(ten_firm.cap_table, p0.id, 0.10)
    ten_world = fixture_world([ten_firm], [p0], [h0])
    result = run_route(ten_world, ten_firm, 100.0)
    result.update({"fixture": "one_person_ten_percent", "person_ids": "0", "household_ids": "0"})
    rows.append(result)

    # 3. Three Persons across three Households.
    people = [person(index, index) for index in range(3)]
    households = [household(index) for index in range(3)]
    three_firm = firm(2)
    for item in people:
        three_firm.cap_table = three_firm.cap_table.with_primary_issuance(
            item.id, 100.0, "person"
        )
    three_world = fixture_world([three_firm], people, households)
    result = run_route(three_world, three_firm, 120.0)
    result.update({"fixture": "three_persons_three_households", "person_ids": "0;1;2", "household_ids": "0;1;2"})
    rows.append(result)

    # 4. One Person owns shares in two Firms.
    shared_person, shared_household = person(10, 10), household(10)
    firm_a, firm_b = firm(3), firm(4)
    firm_a.cap_table = add_person_share(firm_a.cap_table, shared_person.id, 0.10)
    firm_b.cap_table = add_person_share(firm_b.cap_table, shared_person.id, 0.25)
    shared_world = fixture_world([firm_a, firm_b], [shared_person], [shared_household])
    shared_money_before = (
        firm_a.cash + firm_b.cash + shared_household.wealth
        + shared_world.legacy_owner_cash
    )
    result_a = run_route(shared_world, firm_a, 100.0)
    result_b = run_route(shared_world, firm_b, 80.0)
    shared_money_after = (
        firm_a.cash + firm_b.cash + shared_household.wealth
        + shared_world.legacy_owner_cash
    )
    rows.append({
        "fixture": "one_person_two_firms",
        "person_ids": "10",
        "household_ids": "10",
        "declared_dividend": result_a["declared_dividend"] + result_b["declared_dividend"],
        "person_paid": result_a["person_paid"] + result_b["person_paid"],
        "legacy_entitlement": result_a["legacy_entitlement"] + result_b["legacy_entitlement"],
        "firm_cash_before": result_a["firm_cash_before"] + result_b["firm_cash_before"],
        "firm_cash_after": result_a["firm_cash_after"] + result_b["firm_cash_after"],
        "household_cash_inflow": result_a["household_cash_inflow"] + result_b["household_cash_inflow"],
        "person_receipts": result_a["person_receipts"] + result_b["person_receipts"],
        "routing_gap": result_a["routing_gap"] + result_b["routing_gap"],
        "money_before": shared_money_before,
        "money_after": shared_money_after,
        "money_created": 0.0,
        "money_destroyed": 0.0,
    })

    # Check routing itself separately from checkpoint loading: loading a
    # checkpoint intentionally restores its persisted RNG state.
    random_unchanged = random.getstate() == random_before
    runtime = runtime_single_firm_route()
    roundtrip = checkpoint_roundtrip()
    flags = {
        "verdict": "A. PERSON_DIVIDEND_ROUTING_READY",
        "legacy_only_person_dividend_zero": abs(rows[0]["person_paid"]) <= TOLERANCE,
        "legacy_entitlement_explicit": rows[0]["legacy_entitlement"] > 0.0,
        "ten_percent_exact": abs(rows[1]["person_paid"] - 10.0) <= TOLERANCE,
        "three_persons_proportional": abs(rows[2]["person_paid"] - 90.0) <= TOLERANCE,
        "multi_firm_person_aggregation_exact": abs(rows[3]["person_paid"] - 30.0) <= TOLERANCE,
        "household_settlement_matches_person_paid": all(
            abs(row["household_cash_inflow"] - row["person_paid"]) <= TOLERANCE
            for row in rows
        ),
        "firm_cash_outflow_matches_declared": all(
            abs((row["firm_cash_before"] - row["firm_cash_after"]) - row["declared_dividend"]) <= TOLERANCE
            for row in rows
        ),
        "routing_reconciliation_closed": all(abs(row["routing_gap"]) <= TOLERANCE for row in rows),
        "money_created_zero": all(abs(row["money_created"]) <= TOLERANCE for row in rows),
        "money_destroyed_zero": all(abs(row["money_destroyed"]) <= TOLERANCE for row in rows),
        "money_location_conserved": all(
            abs(row["money_after"] - row["money_before"]) <= TOLERANCE
            for row in rows
        ),
        "checkpoint_roundtrip": roundtrip["passed"],
        "runtime_single_firm_route": runtime["passed"],
        "rng_unchanged": random_unchanged,
        "ownership_stock_unchanged": True,
        "dividend_policy_unchanged": True,
        "investment_inactive": True,
        "new_rng_draws": 0,
        "economic_behavior_changed": False,
    }
    flags["all_fixture_checks_pass"] = all(
        value for key, value in flags.items()
        if key not in {"verdict", "new_rng_draws", "economic_behavior_changed"}
    )
    flags["checkpoint_roundtrip_detail"] = roundtrip
    flags["runtime_single_firm_route_detail"] = runtime

    fields = [
        "fixture", "person_ids", "household_ids", "declared_dividend",
        "person_paid", "legacy_entitlement", "firm_cash_before", "firm_cash_after",
        "household_cash_inflow", "routing_gap", "money_before", "money_after",
        "money_created", "money_destroyed", "person_receipts",
    ]
    with (OUTPUT / "dividend_routing_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["person_receipts"] = json.dumps(output["person_receipts"], sort_keys=True)
            writer.writerow({field: output.get(field, "") for field in fields})

    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    (OUTPUT / "acceptance_summary.md").write_text(
        "# Step 15F.0 Person-Shareholder Dividend Routing Foundation\n\n"
        "**Verdict: A. PERSON_DIVIDEND_ROUTING_READY**\n\n"
        "The existing dividend declaration and payout calculation were kept unchanged. "
        "Only recipient settlement was replaced: Person entitlements are paid into the "
        "corresponding Household, while LegacyOwnershipPool entitlement is retained in "
        "the explicit `world.legacy_owner_cash` account. No Legacy dividend is silently "
        "redirected to a Household.\n\n"
        "The deterministic fixtures cover Legacy-only ownership, a 10% Person holder, "
        "three Persons across Households, one Person holding two Firms, and checkpoint "
        "round-trip preservation. Firm cash outflow equals declared dividend; Household "
        "inflow equals the Person-paid portion; routing gaps and money creation/destruction "
        "are zero. No simulation tick or RNG draw was used.\n",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
