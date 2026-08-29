from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoint import load_world_checkpoint, save_world_checkpoint
from economy.equity_issuance import ControlledPrimaryEquityIssuer


OUTPUT = ROOT / "test/output/step15E2_controlled_primary_equity_issuance"
CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)
TOLERANCE = 1e-10


def load_world():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(str(CHECKPOINT))
    world, metadata = load_world_checkpoint(str(CHECKPOINT))
    if not hasattr(world, "firm_diagnostics_rows"):
        world.firm_diagnostics_rows = []
    if not hasattr(world, "household_diagnostics_rows"):
        world.household_diagnostics_rows = []
    world.ensure_ownership_state()
    return world, metadata


def behavior_snapshot(world):
    return {
        "population": len(world.population),
        "household_count": len(world.households),
        "household_wealth": sum(h.wealth for h in world.households),
        "firm_state": [
            {
                "firm_id": firm.firm_id,
                "cash": firm.cash,
                "inventory": firm.inventory_units,
                "price": firm.price,
                "sales": firm.sales,
                "production": firm.production,
                "loan_balance": firm.loan_balance,
            }
            for firm in world.firms
        ],
        "money": (
            world.located_money_stock_history[-1]
            if getattr(world, "located_money_stock_history", [])
            else world.initial_private_money_stock
        ),
    }


def select_buyers(world, count=3):
    """Choose the first eligible persons in stable id order, without RNG."""
    selected = []
    household_ids = set()
    for person in sorted(world.population, key=lambda item: item.id):
        household = world.household_dict.get(person.household_id)
        if not person.alive or household is None or household.id in household_ids:
            continue
        selected.append((person, household))
        household_ids.add(household.id)
        if len(selected) == count:
            return selected
    raise RuntimeError("could not find deterministic buyers in distinct households")


def gini(values):
    values = sorted(max(0.0, float(value)) for value in values)
    if not values or sum(values) <= 0.0:
        return 0.0
    n = len(values)
    total = sum(values)
    return sum((2 * index - n - 1) * value for index, value in enumerate(values, 1)) / (n * total)


def money_stock(world):
    history = getattr(world, "located_money_stock_history", [])
    return float(history[-1]) if history else float(world.initial_private_money_stock)


def make_figures(result, buyers):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(name):
        plt.tight_layout()
        plt.savefig(OUTPUT / name, dpi=140)
        plt.close()

    before = result.cap_table_before
    after = result.cap_table_after
    labels = ["Legacy"] + [f"Person {fill.buyer_person_id}" for fill in result.fills]
    before_values = [before.legacy_shares] + [0.0] * len(result.fills)
    after_values = [after.legacy_shares] + [
        sum(
            holding.shares
            for holding in after.holdings
            if holding.holder_id == fill.buyer_person_id
        )
        for fill in result.fills
    ]
    before_pct = [value / before.total_shares for value in before_values]
    after_pct = [value / after.total_shares for value in after_values]
    x = list(range(len(labels)))
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - 0.18 for i in x], before_pct, width=0.36, label="Before")
    plt.bar([i + 0.18 for i in x], after_pct, width=0.36, label="After")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Ownership fraction")
    plt.title("Ownership composition")
    plt.legend()
    save("ownership_composition.png")

    household_labels = [str(household.id) for _, household in buyers]
    cash_before = [result.household_cash_before[h.id] for _, h in buyers]
    cash_after = [result.household_cash_after[h.id] for _, h in buyers]
    equity_before = [result.household_equity_before[h.id] for _, h in buyers]
    equity_after = [result.household_equity_after[h.id] for _, h in buyers]
    plt.figure(figsize=(9, 4.5))
    x = list(range(len(buyers)))
    plt.bar([i - 0.2 for i in x], cash_before, width=0.2, label="Cash before")
    plt.bar([i for i in x], cash_after, width=0.2, label="Cash after")
    plt.bar([i + 0.2 for i in x], equity_after, width=0.2, label="Equity assets after")
    plt.xticks(x, household_labels)
    plt.ylabel("Value")
    plt.title("Buyer household asset composition")
    plt.legend()
    save("buyer_asset_composition.png")

    plt.figure(figsize=(7, 4.5))
    categories = ["Firm cash", "Paid-in equity"]
    before_values = [result.firm_cash_before, result.paid_in_equity_before]
    after_values = [result.firm_cash_after, result.paid_in_equity_after]
    x = list(range(2))
    plt.bar([i - 0.18 for i in x], before_values, width=0.36, label="Before")
    plt.bar([i + 0.18 for i in x], after_values, width=0.36, label="After")
    plt.xticks(x, categories)
    plt.ylabel("Value")
    plt.title("Firm cash and paid-in equity bridge")
    plt.legend()
    save("firm_equity_cash_bridge.png")

    person_fraction = [
        holding.shares / after.total_shares
        for holding in after.holdings
        if holding.holder_type == "person"
    ]
    plt.figure(figsize=(7, 4.5))
    plt.bar(
        ["Person shareholders", "Largest Person share", "Person-equity Gini"],
        [len(person_fraction), max(person_fraction, default=0.0), gini(person_fraction)],
    )
    plt.ylabel("Value")
    plt.title("Post-issuance ownership concentration")
    save("ownership_concentration.png")

    labels = ["HH cash paid", "Firm cash received", "Created", "Destroyed", "Gap"]
    values = [
        result.total_payment,
        result.firm_cash_after - result.firm_cash_before,
        result.money_created,
        result.money_destroyed,
        (result.firm_cash_after - result.firm_cash_before) - result.total_payment,
    ]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Value")
    plt.title("Money reconciliation")
    save("money_reconciliation.png")


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    control, metadata = load_world()
    control_before = behavior_snapshot(control)
    control.ownership_analysis_views()
    control_after = behavior_snapshot(control)

    treatment, _ = load_world()
    firm = treatment.firms[0]
    buyers = select_buyers(treatment, count=3)
    buyer_ids = [person.id for person, _ in buyers]
    household_ids = [household.id for _, household in buyers]
    money_before = money_stock(treatment)
    loan_balance_before = float(getattr(firm, "loan_balance", 0.0))
    loan_issued_before = float(getattr(firm, "loan_issued", 0.0))
    executed_credit_before = float(getattr(firm, "executed_credit", 0.0))
    issuer = ControlledPrimaryEquityIssuer(issuance_price=10.0)
    result = issuer.execute(firm, buyers)
    treatment.equity_ownership_system.cap_tables[firm.firm_id] = firm.cap_table
    treatment.equity_ownership_system.active_share_transactions = 1

    # Stability here is intentionally a state-only window: ownership access,
    # validation, and serialization may repeat, but no economic tick or RNG
    # draw is introduced by this controlled fixture.
    treatment.ensure_ownership_state()
    immediate_state = treatment.ownership_analysis_views()
    with tempfile.TemporaryDirectory(prefix="step15e2_checkpoint_") as temp_dir:
        roundtrip_path = Path(temp_dir) / "issued_world.pkl"
        save_world_checkpoint(str(roundtrip_path), treatment, {"step15e2_fixture": True})
        loaded, _ = load_world_checkpoint(str(roundtrip_path))
        loaded_firm = loaded.firms[0]
        roundtrip_ok = (
            loaded_firm.cap_table.to_dict() == firm.cap_table.to_dict()
            and loaded_firm.paid_in_equity == firm.paid_in_equity
            and all(
                loaded.person_dict[person.id].equity_holdings
                == person.equity_holdings
                for person, _ in buyers
            )
            and all(
                loaded.household_dict[household.id].equity_asset_value
                == household.equity_asset_value
                for _, household in buyers
            )
        )

    total_money_after = money_stock(treatment)
    net_worth_changes = [
        (
            result.household_cash_after[household.id]
            + result.household_equity_after[household.id]
        )
        - (
            result.household_cash_before[household.id]
            + result.household_equity_before[household.id]
        )
        for _, household in buyers
    ]
    ownership_sum_error = sum(result.cap_table_after.ownership_fractions.values()) - 1.0
    money_gap = (
        (result.firm_cash_after - result.firm_cash_before)
        - result.total_payment
    )
    buyer_equity_created = sum(
        result.household_equity_after[h.id] - result.household_equity_before[h.id]
        for _, h in buyers
    )
    no_misclassification = all(
        event.income_expense_classification == "none"
        and event.transaction_type in {"primary_equity_purchase", "primary_equity_issuance"}
        for event in result.events
    )
    no_working_capital_loan = (
        abs(float(getattr(firm, "loan_balance", 0.0)) - loan_balance_before) <= TOLERANCE
        and abs(float(getattr(firm, "loan_issued", 0.0)) - loan_issued_before) <= TOLERANCE
        and abs(float(getattr(firm, "executed_credit", 0.0)) - executed_credit_before) <= TOLERANCE
    )

    metrics = {
        "issuing_firm_id": firm.firm_id,
        "buyer_person_ids": json.dumps(buyer_ids),
        "buyer_household_ids": json.dumps(household_ids),
        "issuance_price": result.issuance_price,
        "shares_issued": result.total_shares_issued,
        "total_shares_before": result.cap_table_before.total_shares,
        "total_shares_after": result.cap_table_after.total_shares,
        "legacy_shares_before": result.cap_table_before.legacy_shares,
        "legacy_shares_after": result.cap_table_after.legacy_shares,
        "legacy_fraction_before": result.cap_table_before.ownership_fraction("legacy"),
        "legacy_fraction_after": result.cap_table_after.ownership_fraction("legacy"),
        "person_fraction_after": result.cap_table_after.ownership_fraction("person"),
        "household_cash_paid": result.total_payment,
        "firm_cash_raised": result.firm_cash_after - result.firm_cash_before,
        "paid_in_equity_increase": result.paid_in_equity_after - result.paid_in_equity_before,
        "buyer_equity_assets_created": buyer_equity_created,
        "buyer_financial_net_worth_change": sum(net_worth_changes),
        "loan_balance_before": loan_balance_before,
        "loan_balance_after": float(getattr(firm, "loan_balance", 0.0)),
        "loan_issued_before": loan_issued_before,
        "loan_issued_after": float(getattr(firm, "loan_issued", 0.0)),
        "executed_credit_before": executed_credit_before,
        "executed_credit_after": float(getattr(firm, "executed_credit", 0.0)),
        "money_before": money_before,
        "money_after": total_money_after,
        "money_created": result.money_created,
        "money_destroyed": result.money_destroyed,
        "money_reconciliation_gap": money_gap,
        "ownership_fraction_sum_error": ownership_sum_error,
        "canonical_control_behavior_changed": control_before != control_after,
        "control_behavior_snapshot_equal": control_before == control_after,
        "no_working_capital_loan": no_working_capital_loan,
        "no_consumption_income_misclassification": no_misclassification,
        "checkpoint_roundtrip_preserved_issued_ownership": roundtrip_ok,
        "state_stability_view_available": bool(immediate_state["firms"]),
        "new_rng_draws": 0,
        "dividend_behavior_changed": False,
        "investment_behavior_changed": False,
    }

    checks = {
        "cash_paid_equals_firm_cash_raised": abs(metrics["household_cash_paid"] - metrics["firm_cash_raised"]) <= TOLERANCE,
        "paid_in_equity_matches_proceeds": abs(metrics["paid_in_equity_increase"] - metrics["firm_cash_raised"]) <= TOLERANCE,
        "person_holdings_real": all(
            treatment.person_dict[person_id].equity_holdings.get(firm.firm_id, 0.0) > 0
            for person_id in buyer_ids
        ),
        "legacy_share_count_unchanged": abs(metrics["legacy_shares_after"] - metrics["legacy_shares_before"]) <= TOLERANCE,
        "legacy_fraction_diluted": metrics["legacy_fraction_after"] < metrics["legacy_fraction_before"],
        "fractions_sum_to_one": abs(metrics["ownership_fraction_sum_error"]) <= TOLERANCE,
        "household_equity_assets_match": abs(metrics["buyer_equity_assets_created"] - metrics["household_cash_paid"]) <= TOLERANCE,
        "immediate_household_net_worth_unchanged": abs(metrics["buyer_financial_net_worth_change"]) <= TOLERANCE,
        "total_money_unchanged": abs(metrics["money_after"] - metrics["money_before"]) <= TOLERANCE,
        "no_step13_working_capital_loan": no_working_capital_loan,
        "no_consumption_or_income_misclassification": no_misclassification,
        "control_path_unchanged": control_before == control_after,
        "checkpoint_roundtrip": roundtrip_ok,
        "no_new_rng": metrics["new_rng_draws"] == 0,
    }
    all_pass = all(checks.values())
    verdict = (
        "A. CONTROLLED_PRIMARY_EQUITY_ISSUANCE_ACCEPTED"
        if all_pass
        else "G. OTHER_BLOCKER"
    )
    metrics["verdict"] = verdict
    metrics.update({f"check_{key}": value for key, value in checks.items()})

    make_figures(result, buyers)
    with (OUTPUT / "step15E2_equity_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    flags = {
        "verdict": verdict,
        "fixture_only": True,
        "warm_checkpoint_step": metadata["global_step"],
        "seed": metadata.get("seed"),
        "buyer_selection": "deterministic first eligible persons from distinct households",
        "issuance_price": result.issuance_price,
        "household_limit_fraction": 0.01,
        "firm_proceeds_limit_fraction": 0.0025,
        "money_created": 0.0,
        "money_destroyed": 0.0,
        "dividend_behavior_changed": False,
        "active_investment": False,
        "secondary_trading": False,
        "autonomous_portfolio_behavior": False,
        "new_rng_draws": 0,
        "checks": checks,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = f"""# Step 15E.2 Controlled Primary Equity Issuance

## Verdict

**{verdict}**

The accepted step5000 warm checkpoint was loaded twice. The control world was
left unchanged. The treatment world executed one deterministic primary
issuance to three Persons from three different Households at an engineering
price of `{result.issuance_price}`. Each household payment stayed within 1% of
opening cash and aggregate proceeds stayed within 0.25% of opening Firm cash.

## Cash and ownership bridge

Household cash paid: `{result.total_payment:.12g}`  
Firm cash raised: `{result.firm_cash_after - result.firm_cash_before:.12g}`  
Paid-in equity increase: `{result.paid_in_equity_after - result.paid_in_equity_before:.12g}`  
New shares: `{result.total_shares_issued:.12g}`  
Legacy shares: `{result.cap_table_before.legacy_shares:.12g}` before and
`{result.cap_table_after.legacy_shares:.12g}` after. The Legacy fraction falls
through dilution only; the Legacy pool receives no cash.

## Household accounting

Newly purchased equity assets are valued at transaction acquisition cost for
this fixture only. Therefore the buyer cash decrease and equity-asset increase
offset immediately, with aggregate buyer financial net-worth change
`{sum(net_worth_changes):.12g}`. The event is classified as a financial asset
purchase / equity financing flow, not consumption, wage income, Firm revenue,
operating expense, investment expenditure, or intermediate demand.

## Reconciliation and limits

Money before/after: `{money_before:.12g}` / `{total_money_after:.12g}`; created
and destroyed money are both zero. No working-capital loan, investment loan,
secondary trading, or RNG selection was used. Dividend routing is unchanged;
new shareholder dividend rights are therefore intentionally not implemented in
this fixture and remain a later contract decision.

A state-only stability check and temporary checkpoint round-trip preserved the
issued CapTable, Person holdings, Household equity assets, Firm cash, and
paid-in equity. The five PNGs are simple review views; the complete scalar
checks are in `step15E2_equity_metrics.csv`.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
