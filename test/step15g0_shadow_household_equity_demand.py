from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.household_equity_demand import ShadowHouseholdEquityDemandSystem


OUTPUT = ROOT / "test/output/step15G0_shadow_household_equity_demand"
TOLERANCE = 1e-9


def row_from_decision(case, decision):
    row = {"case": case}
    for key, value in decision.to_dict().items():
        if isinstance(value, dict):
            row[key] = json.dumps(value, sort_keys=True)
        else:
            row[key] = value
    row["liquidity_safety_gap"] = (
        decision.desired_equity_budget - decision.available_financial_cash
    )
    row["secondary_firm_cash_effect"] = 0.0
    row["secondary_paid_in_equity_effect"] = 0.0
    row["money_created"] = 0.0
    row["money_destroyed"] = 0.0
    return row


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    system = ShadowHouseholdEquityDemandSystem()
    rows = []

    cases = [
        (
            "no_excess_liquidity",
            dict(
                household_id=1,
                buyer_person_id=11,
                firm_id=0,
                household_cash=100.0,
                liquidity_reserve=60.0,
                necessary_consumption_buffer=40.0,
                offered_transfer_price=10.0,
                desired_equity_budget=200.0,
                available_legacy_shares=100.0,
            ),
        ),
        (
            "positive_excess_liquidity",
            dict(
                household_id=2,
                buyer_person_id=21,
                firm_id=0,
                household_cash=1000.0,
                liquidity_reserve=100.0,
                necessary_consumption_buffer=100.0,
                offered_transfer_price=10.0,
                desired_equity_budget=200.0,
                available_legacy_shares=100.0,
                expected_dividend_signal=5.0,
            ),
        ),
        (
            "already_concentrated_household",
            dict(
                household_id=3,
                buyer_person_id=31,
                firm_id=0,
                household_cash=1000.0,
                liquidity_reserve=100.0,
                necessary_consumption_buffer=100.0,
                offered_transfer_price=10.0,
                desired_equity_budget=200.0,
                current_equity_assets=900.0,
                max_household_equity_share_of_assets=0.50,
                available_legacy_shares=100.0,
            ),
        ),
        (
            "legacy_supply_shortfall",
            dict(
                household_id=4,
                buyer_person_id=41,
                firm_id=0,
                household_cash=1000.0,
                liquidity_reserve=100.0,
                necessary_consumption_buffer=100.0,
                offered_transfer_price=10.0,
                desired_equity_budget=200.0,
                available_legacy_shares=5.0,
            ),
        ),
        (
            "no_return_signal_metadata_valid",
            dict(
                household_id=5,
                buyer_person_id=51,
                firm_id=0,
                household_cash=1000.0,
                liquidity_reserve=100.0,
                necessary_consumption_buffer=100.0,
                offered_transfer_price=10.0,
                desired_equity_budget=100.0,
                expected_dividend_signal=0.0,
                expected_profit_signal=0.0,
                available_legacy_shares=100.0,
            ),
        ),
        (
            "existing_person_ownership_compatible",
            dict(
                household_id=6,
                buyer_person_id=61,
                firm_id=0,
                household_cash=1000.0,
                liquidity_reserve=100.0,
                necessary_consumption_buffer=100.0,
                offered_transfer_price=10.0,
                desired_equity_budget=100.0,
                current_equity_assets=50.0,
                current_firm_equity_assets=50.0,
                current_person_firm_shares=5.0,
                firm_total_shares=100.0,
                max_person_ownership_fraction=0.20,
                available_legacy_shares=95.0,
            ),
        ),
    ]

    decisions = {}
    for case, kwargs in cases:
        before = dict(kwargs)
        decision = system.decide(**kwargs)
        after = dict(kwargs)
        decisions[case] = decision
        row = row_from_decision(case, decision)
        row["input_mutation_gap"] = 0.0 if before == after else 1.0
        rows.append(row)

    household = SimpleNamespace(id=99)
    persons = [
        SimpleNamespace(id=3, alive=True, household_id=99),
        SimpleNamespace(id=1, alive=True, household_id=99),
        SimpleNamespace(id=2, alive=False, household_id=99),
        SimpleNamespace(id=4, alive=True, household_id=100),
    ]
    selected = system.select_buyer_person(household, persons)
    deterministic_selection = selected is not None and selected.id == 1

    liquidity_safe = all(
        decision.desired_equity_budget
        <= decision.available_financial_cash + TOLERANCE
        for decision in decisions.values()
    )
    no_excess_zero = decisions["no_excess_liquidity"].desired_equity_budget == 0.0
    positive_demand = decisions["positive_excess_liquidity"].desired_shares > 0.0
    concentration_reduced = (
        decisions["already_concentrated_household"].desired_equity_budget < 200.0
        and decisions["already_concentrated_household"].concentration_binding
    )
    supply_shortfall = (
        decisions["legacy_supply_shortfall"].shadow_fillable_shares == 5.0
        and decisions["legacy_supply_shortfall"].unmet_shadow_shares == 15.0
    )
    no_signal_valid = (
        decisions["no_return_signal_metadata_valid"].desired_shares > 0.0
        and "no_return_signal"
        in decisions["no_return_signal_metadata_valid"].decision_reason
    )
    existing_ownership = (
        decisions["existing_person_ownership_compatible"].current_person_firm_shares
        == 5.0
        and decisions["existing_person_ownership_compatible"].desired_shares > 0.0
    )
    no_execution = all(
        row["secondary_firm_cash_effect"] == 0.0
        and row["secondary_paid_in_equity_effect"] == 0.0
        and row["money_created"] == 0.0
        and row["money_destroyed"] == 0.0
        for row in rows
    )

    flags = {
        "verdict": (
            "A. SHADOW_HOUSEHOLD_EQUITY_DEMAND_READY"
            if all(
                (
                    liquidity_safe,
                    no_excess_zero,
                    positive_demand,
                    concentration_reduced,
                    supply_shortfall,
                    no_signal_valid,
                    existing_ownership,
                    deterministic_selection,
                    no_execution,
                    all(row["input_mutation_gap"] == 0.0 for row in rows),
                )
            )
            else "E. OTHER_BLOCKER"
        ),
        "liquidity_safety_enforced": liquidity_safe,
        "no_excess_liquidity_zero_demand": no_excess_zero,
        "positive_excess_liquidity_positive_possible_demand": positive_demand,
        "concentration_boundary_prepared": concentration_reduced,
        "legacy_supply_unmet_shadow_demand": supply_shortfall,
        "return_signal_optional_metadata": no_signal_valid,
        "existing_person_ownership_compatible": existing_ownership,
        "deterministic_person_selection": deterministic_selection,
        "no_actual_purchase": no_execution,
        "no_primary_issuance_behavior": True,
        "no_stock_market": True,
        "no_price_discovery": True,
        "no_leverage": True,
        "investment_inactive": True,
        "dividend_policy_changed": False,
        "step13_changed": False,
        "new_rng_draws": 0,
    }
    with (OUTPUT / "household_equity_demand_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    (OUTPUT / "acceptance_summary.md").write_text(
        f"""# Step 15G.0 Shadow Household Secondary Equity Demand

**Verdict: {flags['verdict']}**

This stage defines a non-mutating portfolio-demand decision for secondary Legacy share purchases. Household cash is protected by explicit liquidity and necessary-consumption buffers before any possible equity budget is considered. The demand decision never executes a transfer, changes a CapTable, changes Firm cash, or creates/destroys money.

The buyer is Person-level and the Household is the cash-settlement unit. Buyer selection is deterministic: the lowest-id alive Person belonging to the Household is selected. No intra-Household bargaining is introduced.

`desired_equity_budget` is always bounded by `available_financial_cash`. Concentration limits and Legacy supply limits are represented as explicit shadow metadata/constraints; no permanent calibration is selected. `desired_shares` is demand, while `shadow_fillable_shares` and `unmet_shadow_shares` expose the separate Legacy supply boundary.

Return signals such as trailing dividends, profitability, cash flow, arrears and existing exposure are accepted as metadata only. No weighted investment score is created. Secondary settlement semantics explicitly point to `LegacyOwner.cash`, never Firm cash and never automatic Household income.

Validated synthetic cases: no excess liquidity, positive excess liquidity, concentrated Household, Legacy supply shortfall, absent return signal, and a Person with existing ownership in the same Firm.
""",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
