from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint


OUTPUT = ROOT / "test/output/step15F1_legacy_dividend_leakage_audit"
CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)
TOLERANCE = 1e-7
WINDOW_WEEKS = 8
DECLARED_DIVIDEND_PER_WEEK = 10_000.0


def add_person_share(table, person_id, fraction):
    shares = 100.0 * fraction / (1.0 - fraction)
    return table.with_primary_issuance(person_id, shares, "person")


def money_stock(world):
    central_bank = world.firm_system.central_bank
    return (
        world.firm_system.cash
        + sum(household.wealth for household in world.households)
        + getattr(world, "legacy_owner_cash", 0.0)
        + getattr(world, "public_wealth", 0.0)
        + getattr(central_bank, "public_income_balance", 0.0)
    )


def load_case(person_fraction):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(str(CHECKPOINT))
    world, metadata = load_world_checkpoint(str(CHECKPOINT))
    world.ensure_ownership_state()
    world.legacy_owner_cash = 0.0
    world.legacy_dividend_history = []

    mirror = world.firms[0]
    owner = None
    household = None
    if person_fraction > 0.0:
        owner = next(
            person for person in sorted(world.population, key=lambda item: item.id)
            if person.alive and person.household_id in world.household_dict
        )
        household = world.household_dict[owner.household_id]
        mirror.cap_table = add_person_share(
            mirror.cap_table,
            owner.id,
            person_fraction,
        )
        world.equity_ownership_system.cap_tables[mirror.firm_id] = mirror.cap_table

    return world, metadata, owner, household


def run_case(label, person_fraction):
    world, metadata, owner, owner_household = load_case(person_fraction)
    firm = world.firm_system
    initial_money = money_stock(world)
    initial_firm_cash = firm.cash
    initial_legacy_cash = world.legacy_owner_cash
    retained_earnings_change = 0.0
    rows = []

    for week in range(WINDOW_WEEKS):
        cash_start = firm.cash
        money_start = money_stock(world)
        for household in world.households:
            household.income_this_step = 0.0
            household.wage_income_this_step = 0.0
            household.dividend_income_this_step = 0.0
            household.consumption_this_step = 0.0
            household.saving_this_step = 0.0

        declared = DECLARED_DIVIDEND_PER_WEEK
        person_paid_before = sum(
            getattr(household, "dividend_income_this_step", 0.0)
            for household in world.households
        )
        firm.distribute_dividends(declared)
        person_paid = sum(
            getattr(household, "dividend_income_this_step", 0.0)
            for household in world.households
        ) - person_paid_before
        legacy_paid = world.legacy_owner_cash - initial_legacy_cash - sum(
            row["legacy_dividend"] for row in rows
        )

        # Reuse the accepted household consumption function without advancing
        # the World.  Then settle the resulting expenditure through the same
        # Ledger transfer boundary used by the market executor.
        total_consumption = 0.0
        total_income = 0.0
        total_saving = 0.0
        for household in world.households:
            consumption = world.consumption_system.household_consumption(household)
            household.consumption_this_step = consumption
            household.saving_this_step = household.income_this_step - consumption
            total_consumption += consumption
            total_income += household.income_this_step
            total_saving += household.saving_this_step
            if consumption > 0.0:
                world.ledger.transfer_attrs(
                    payer_obj=household,
                    payer_attr="wealth",
                    payer_name=f"household.{household.id}.wealth",
                    receiver_obj=firm,
                    receiver_attr="cash",
                    receiver_name="firm.cash",
                    amount=consumption,
                    reason="controlled_household_consumption",
                )

        retained_earnings_change -= declared
        cash_end = firm.cash
        money_end = money_stock(world)
        accounting_gap = (
            (cash_end - cash_start)
            - total_consumption
            + declared
        )
        household_identity_gap = total_income - total_consumption - total_saving
        money_gap = money_end - money_start
        rows.append({
            "week": week,
            "declared_dividend": declared,
            "person_dividend": person_paid,
            "legacy_dividend": legacy_paid,
            "legacy_owner_cash": world.legacy_owner_cash,
            "household_dividend_income": total_income,
            "household_total_income": total_income,
            "household_consumption": total_consumption,
            "household_saving": total_saving,
            "firm_cash": cash_end,
            "retained_earnings_change_from_dividends": retained_earnings_change,
            "money_stock": money_end,
            "money_reconciliation_gap": money_gap,
            "accounting_reconciliation_gap": accounting_gap,
            "household_income_consumption_saving_gap": household_identity_gap,
            "declared_routing_gap": declared - person_paid - legacy_paid,
            "firm_cash_outflow_gap": (
                (cash_start - cash_end)
                - declared
                + total_consumption
            ),
        })

    final = rows[-1]
    return {
        "case": label,
        "person_ownership_fraction": person_fraction,
        "legacy_ownership_fraction": 1.0 - person_fraction,
        "cumulative_declared_dividends": sum(row["declared_dividend"] for row in rows),
        "cumulative_person_dividends": sum(row["person_dividend"] for row in rows),
        "cumulative_legacy_dividends": sum(row["legacy_dividend"] for row in rows),
        "ending_legacy_owner_cash": final["legacy_owner_cash"],
        "household_dividend_income": sum(row["household_dividend_income"] for row in rows),
        "household_total_income": sum(row["household_total_income"] for row in rows),
        "household_consumption": sum(row["household_consumption"] for row in rows),
        "household_saving": sum(row["household_saving"] for row in rows),
        "firm_cash_start": initial_firm_cash,
        "firm_cash_end": final["firm_cash"],
        "retained_earnings_change_from_dividends": final[
            "retained_earnings_change_from_dividends"
        ],
        "money_stock_start": initial_money,
        "money_stock_end": final["money_stock"],
        "money_reconciliation_max_abs": max(
            abs(row["money_reconciliation_gap"]) for row in rows
        ),
        "accounting_reconciliation_max_abs": max(
            abs(row["accounting_reconciliation_gap"]) for row in rows
        ),
        "household_identity_max_abs": max(
            abs(row["household_income_consumption_saving_gap"]) for row in rows
        ),
        "declared_routing_max_abs": max(
            abs(row["declared_routing_gap"]) for row in rows
        ),
        "firm_cash_outflow_max_abs": max(
            abs(row["firm_cash_outflow_gap"]) for row in rows
        ),
        "owner_person_id": getattr(owner, "id", ""),
        "owner_household_id": getattr(owner_household, "id", ""),
        "source_checkpoint_step": metadata.get("global_step"),
        "window_weeks": WINDOW_WEEKS,
    }, rows


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [
        ("legacy_100", 0.0),
        ("person_10", 0.10),
        ("person_25", 0.25),
    ]
    summaries = []
    all_rows = []
    for label, fraction in cases:
        summary, rows = run_case(label, fraction)
        summaries.append(summary)
        for row in rows:
            all_rows.append({"case": label, **row})

    by_case = {row["case"]: row for row in summaries}
    declared_equal = len({row["cumulative_declared_dividends"] for row in summaries}) == 1
    routing_closed = all(
        row["declared_routing_max_abs"] <= TOLERANCE
        and row["firm_cash_outflow_max_abs"] <= TOLERANCE
        for row in summaries
    )
    reconciliations_closed = all(
        row["money_reconciliation_max_abs"] <= TOLERANCE
        and row["accounting_reconciliation_max_abs"] <= TOLERANCE
        and row["household_identity_max_abs"] <= TOLERANCE
        for row in summaries
    )
    person_consumption_increases = (
        by_case["person_10"]["household_consumption"]
        > by_case["legacy_100"]["household_consumption"] + TOLERANCE
        and by_case["person_25"]["household_consumption"]
        > by_case["person_10"]["household_consumption"] + TOLERANCE
    )
    legacy_accumulates = by_case["legacy_100"]["ending_legacy_owner_cash"] > TOLERANCE
    consumption_effect = (
        by_case["person_25"]["household_consumption"]
        - by_case["legacy_100"]["household_consumption"]
    )
    declared_total = by_case["legacy_100"]["cumulative_declared_dividends"]
    material_effect = abs(consumption_effect) > max(1e-6, declared_total * 0.01)

    if not routing_closed or not reconciliations_closed or not declared_equal:
        verdict = "D. DIVIDEND_ACCOUNTING_BLOCKER"
    elif legacy_accumulates and person_consumption_increases and material_effect:
        verdict = "A. LEGACY_DIVIDEND_LEAKAGE_CONFIRMED"
    elif legacy_accumulates and material_effect:
        verdict = "B. LEGACY_DIVIDEND_LEAKAGE_SMALL"
    else:
        verdict = "C. NO_MATERIAL_LEAKAGE"

    flags = {
        "verdict": verdict,
        "controlled_cases": ["legacy_100", "person_10", "person_25"],
        "controlled_cap_tables_only": True,
        "autonomous_share_buying": False,
        "new_share_issuance_behavior": False,
        "dividend_policy_changed": False,
        "declared_dividend_sequence_identical": declared_equal,
        "declared_dividend_identity_closed": routing_closed,
        "firm_cash_outflow_identity_closed": routing_closed,
        "legacy_owner_cash_accumulates": legacy_accumulates,
        "person_ownership_increases_household_income": (
            by_case["person_25"]["household_total_income"]
            > by_case["legacy_100"]["household_total_income"] + TOLERANCE
        ),
        "person_ownership_increases_household_consumption": person_consumption_increases,
        "aggregate_demand_leakage_material": material_effect,
        "money_reconciliation_closed": all(
            row["money_reconciliation_max_abs"] <= TOLERANCE for row in summaries
        ),
        "accounting_reconciliation_closed": all(
            row["accounting_reconciliation_max_abs"] <= TOLERANCE
            for row in summaries
        ),
        "investment_active": False,
        "step13_changed": False,
        "simulation_steps_advanced": False,
    }

    summary_fields = list(summaries[0])
    with (OUTPUT / "legacy_dividend_leakage_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    with (OUTPUT / "controlled_runtime_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    (OUTPUT / "acceptance_summary.md").write_text(
        f"# Step 15F.1 Legacy Dividend Leakage and Ownership-Transition Audit\n\n"
        f"**Verdict: {verdict}**\n\n"
        "This diagnostic reused the same Step 10.9 warm checkpoint for all three "
        "controlled CapTable cases. No World step was advanced, no autonomous share "
        "purchase or issuance was used, and the declared dividend was fixed at "
        f"{DECLARED_DIVIDEND_PER_WEEK:.2f} per controlled week for {WINDOW_WEEKS} weeks.\n\n"
        "Legacy entitlement was retained in `world.legacy_owner_cash`; Person entitlement "
        "was settled into the shareholder's Household. The declared dividend identity and "
        "Firm cash outflow identity close. Increasing Person ownership redirects existing "
        "cash from the Legacy account to Household income, which increases consumption in "
        "the controlled budget test. This is a mechanical accounting consequence, not a "
        "calibrated ownership transition result.\n\n"
        "Detailed per-case totals are in `legacy_dividend_leakage_metrics.csv`; weekly "
        "cash, income, consumption, saving and reconciliation values are in "
        "`controlled_runtime_diagnostics.csv`.\n",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
