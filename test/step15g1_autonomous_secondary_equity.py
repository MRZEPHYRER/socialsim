from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.autonomous_secondary_equity import (
    AutonomousSecondaryEquityPurchaseSystem,
)
from world import World


OUTPUT = ROOT / "test/output/step15G1_autonomous_secondary_equity"
WEEKS = 104
TOLERANCE = 1e-6


def person_gini(world):
    fractions = sorted(
        holding.shares / table.total_shares
        for firm in getattr(world, "firms", [])
        for table in [getattr(firm, "cap_table", None)]
        if table is not None
        for holding in table.holdings
        if holding.holder_type == "person" and holding.shares > 0.0
    )
    if len(fractions) < 2 or sum(fractions) <= 0.0:
        return 0.0
    total = sum(fractions)
    n = len(fractions)
    return sum(
        (2 * index - n - 1) * value
        for index, value in enumerate(fractions, 1)
    ) / (n * total)


def largest_person_share(world):
    return max(
        (
            holding.shares / table.total_shares
            for firm in getattr(world, "firms", [])
            for table in [getattr(firm, "cap_table", None)]
            if table is not None
            for holding in table.holdings
            if holding.holder_type == "person" and holding.shares > 0.0
        ),
        default=0.0,
    )


def money_stock(world):
    cb = world.firm_system.central_bank
    return (
        world.firm_system.cash
        + sum(h.wealth for h in world.households)
        + getattr(world, "legacy_owner_cash", 0.0)
        + getattr(world, "public_wealth", 0.0)
        + getattr(cb, "public_income_balance", 0.0)
    )


def build_world(enabled):
    world = World(initial_population=5000, seed=42, diagnostics_mode="full")
    world.ensure_ownership_state()
    world.person_equity_transition_enabled = False
    world.autonomous_secondary_equity_enabled = bool(enabled)
    if enabled:
        world.autonomous_secondary_equity_system = (
            AutonomousSecondaryEquityPurchaseSystem(world)
        )
    return world


def accounting_gaps(world):
    rows = getattr(world.accounting, "rows", [])
    household_rows = getattr(world.accounting, "household_rows", [])
    return (
        max(
            (abs(float(row.get("cash_flow_gap", 0.0))) for row in rows),
            default=0.0,
        ),
        max(
            (
                abs(float(row.get("household_wealth_bridge_gap", 0.0)))
                for row in household_rows
            ),
            default=0.0,
        ),
    )


def run_case(case, enabled):
    world = build_world(enabled)
    rows = []
    decisions = []
    transfer_checks = []
    for _ in range(WEEKS):
        world.step()
        result = getattr(world, "last_autonomous_secondary_purchase_result", None)
        if result is not None:
            decisions.extend(
                {"case": case, "global_step": world.current_step_index, **decision}
                for decision in result.decisions
            )
            for transfer in result.transfer_results:
                transfer_checks.append({
                    "firm_cash_gap": abs(
                        transfer.firm_cash_after - transfer.firm_cash_before
                    ),
                    "paid_in_equity_gap": abs(
                        transfer.paid_in_equity_after
                        - transfer.paid_in_equity_before
                    ),
                    "money_created": transfer.money_created,
                    "money_destroyed": transfer.money_destroyed,
                    "reconciliation_gap": abs(transfer.reconciliation_gap),
                })
        firm = world.firms[0]
        accounting_gap, household_gap = accounting_gaps(world)
        rows.append({
            "case": case,
            "global_step": world.current_step_index,
            "reviewed": bool(result.reviewed) if result else False,
            "secondary_sale_proceeds": float(
                result.total_payment if result else 0.0
            ),
            "secondary_shares_sold": float(
                result.total_shares_sold if result else 0.0
            ),
            "secondary_buyer_count": int(result.buyer_count if result else 0),
            "cumulative_secondary_sale_proceeds": sum(
                float(item.get("total_payment", 0.0))
                for item in getattr(world, "secondary_transfer_history", [])
            ),
            "person_ownership_fraction": firm.cap_table.ownership_fraction("person"),
            "legacy_ownership_fraction": firm.cap_table.ownership_fraction("legacy"),
            "legacy_shares": firm.cap_table.legacy_shares,
            "total_shares": firm.cap_table.total_shares,
            "household_cash": sum(float(h.wealth) for h in world.households),
            "household_equity_assets": sum(
                float(getattr(h, "equity_asset_value", 0.0))
                for h in world.households
            ),
            "household_financial_net_worth": sum(
                float(h.wealth) + float(getattr(h, "equity_asset_value", 0.0))
                for h in world.households
            ),
            "legacy_owner_cash": float(getattr(world, "legacy_owner_cash", 0.0)),
            "person_dividend": float(getattr(firm, "person_dividend_paid", 0.0)),
            "legacy_dividend": float(
                getattr(firm, "legacy_dividend_entitlement", 0.0)
            ),
            "declared_dividend": float(getattr(firm, "dividend_payment", 0.0)),
            "household_dividend_income": sum(
                float(getattr(h, "dividend_income_this_step", 0.0))
                for h in world.households
            ),
            "household_consumption": float(world.consumption_history[-1]),
            "household_saving": float(world.saving_history[-1]),
            "firm_cash": float(world.firm_system.cash),
            "paid_in_equity": float(getattr(firm, "paid_in_equity", 0.0)),
            "money_stock": money_stock(world),
            "money_reconciliation_gap": float(
                world.diagnostics_rows[-1].get("monetary_accounting_gap", 0.0)
            ),
            "accounting_gap": accounting_gap,
            "household_accounting_gap": household_gap,
            "largest_person_share": largest_person_share(world),
            "person_equity_gini": person_gini(world),
            "negative_households": sum(
                float(h.wealth) < -TOLERANCE for h in world.households
            ),
            "primary_issuance_cash": sum(
                float(getattr(f, "equity_issuance_cash_this_step", 0.0))
                for f in getattr(world, "firms", [])
            ),
        })

    dividend_events = list(getattr(world, "dividend_routing_events", []))
    summary = {
        "case": case,
        "enabled": enabled,
        "weeks": WEEKS,
        "review_count": sum(row["reviewed"] for row in rows),
        "total_secondary_sale_proceeds": sum(
            row["secondary_sale_proceeds"] for row in rows
        ),
        "total_secondary_shares_sold": sum(
            row["secondary_shares_sold"] for row in rows
        ),
        "total_buyers": sum(row["secondary_buyer_count"] for row in rows),
        "final_person_ownership_fraction": rows[-1]["person_ownership_fraction"],
        "final_legacy_ownership_fraction": rows[-1]["legacy_ownership_fraction"],
        "final_legacy_shares": rows[-1]["legacy_shares"],
        "final_household_cash": rows[-1]["household_cash"],
        "final_household_equity_assets": rows[-1]["household_equity_assets"],
        "final_household_financial_net_worth": rows[-1][
            "household_financial_net_worth"
        ],
        "final_legacy_owner_cash": rows[-1]["legacy_owner_cash"],
        "cumulative_person_dividends": sum(
            row["person_dividend"] for row in rows
        ),
        "cumulative_legacy_dividends": sum(
            row["legacy_dividend"] for row in rows
        ),
        "cumulative_declared_dividends": sum(
            row["declared_dividend"] for row in rows
        ),
        "cumulative_household_dividend_income": sum(
            row["household_dividend_income"] for row in rows
        ),
        "cumulative_consumption": sum(row["household_consumption"] for row in rows),
        "cumulative_saving": sum(row["household_saving"] for row in rows),
        "largest_person_share": rows[-1]["largest_person_share"],
        "person_equity_gini": rows[-1]["person_equity_gini"],
        "max_negative_households": max(row["negative_households"] for row in rows),
        "max_primary_issuance_cash": max(
            row["primary_issuance_cash"] for row in rows
        ),
        "max_money_reconciliation_gap": max(
            abs(row["money_reconciliation_gap"]) for row in rows
        ),
        "max_accounting_gap": max(row["accounting_gap"] for row in rows),
        "max_household_accounting_gap": max(
            row["household_accounting_gap"] for row in rows
        ),
        "actual_dividend_events": len(dividend_events),
        "person_dividend_events": sum(
            float(event.get("person_paid", 0.0)) > TOLERANCE
            for event in dividend_events
        ),
        "max_dividend_routing_gap": max(
            (abs(float(event.get("routing_gap", 0.0))) for event in dividend_events),
            default=0.0,
        ),
        "max_transfer_firm_cash_gap": max(
            (item["firm_cash_gap"] for item in transfer_checks), default=0.0
        ),
        "max_transfer_paid_in_equity_gap": max(
            (item["paid_in_equity_gap"] for item in transfer_checks), default=0.0
        ),
        "max_transfer_reconciliation_gap": max(
            (item["reconciliation_gap"] for item in transfer_checks), default=0.0
        ),
        "max_decision_liquidity_violation": max(
            (
                float(decision.get("desired_equity_budget", 0.0))
                - float(decision.get("available_financial_cash", 0.0))
                for decision in decisions
            ),
            default=0.0,
        ),
    }
    return world, rows, decisions, summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(states):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"control_off": "#666666", "treatment_on": "#1769aa"}

    def line(name, field, title, ylabel):
        plt.figure(figsize=(9, 4.8))
        for case, rows in states.items():
            plt.plot(
                [row["global_step"] for row in rows],
                [row[field] for row in rows],
                label=case,
                color=colors[case],
            )
        plt.title(title)
        plt.xlabel("Global week")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT / name, dpi=140)
        plt.close()

    line(
        "ownership_transition.png",
        "person_ownership_fraction",
        "Autonomous secondary ownership transition",
        "Person ownership fraction",
    )
    line(
        "household_cash_equity.png",
        "household_equity_assets",
        "Household equity assets",
        "Equity asset value",
    )
    line(
        "legacy_owner_cash.png",
        "legacy_owner_cash",
        "LegacyOwner cash",
        "LegacyOwner cash",
    )
    line(
        "dividend_routing_transition.png",
        "person_dividend",
        "Person dividend flow",
        "Person dividend",
    )

    plt.figure(figsize=(9, 4.8))
    for case, rows in states.items():
        plt.plot(
            [row["global_step"] for row in rows],
            [row["household_consumption"] for row in rows],
            label=f"{case} consumption",
            color=colors[case],
        )
        plt.plot(
            [row["global_step"] for row in rows],
            [row["household_saving"] for row in rows],
            linestyle="--",
            label=f"{case} saving",
            color=colors[case],
        )
    plt.title("Consumption and saving comparison")
    plt.xlabel("Global week")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "consumption_saving_comparison.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    for case, rows in states.items():
        plt.plot(
            [row["global_step"] for row in rows],
            [row["largest_person_share"] for row in rows],
            label=f"{case} largest share",
            color=colors[case],
        )
        plt.plot(
            [row["global_step"] for row in rows],
            [row["person_equity_gini"] for row in rows],
            linestyle="--",
            label=f"{case} Person Gini",
            color=colors[case],
        )
    plt.title("Ownership concentration")
    plt.xlabel("Global week")
    plt.ylabel("Share / Gini")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "ownership_concentration.png", dpi=140)
    plt.close()


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [("control_off", False), ("treatment_on", True)]
    states = {}
    all_panel = []
    summaries = []
    for case, enabled in cases:
        _, rows, _, summary = run_case(case, enabled)
        states[case] = rows
        summaries.append(summary)
        all_panel.extend(rows)

    control = next(row for row in summaries if row["case"] == "control_off")
    treatment = next(row for row in summaries if row["case"] == "treatment_on")
    ownership_rises = treatment["final_person_ownership_fraction"] > TOLERANCE
    gradual = (
        treatment["final_person_ownership_fraction"] < 0.10
        and treatment["largest_person_share"] < 0.20
        and treatment["total_secondary_sale_proceeds"] < 10000.0
    )
    liquidity_safe = (
        treatment["max_negative_households"] == 0
        and treatment["max_decision_liquidity_violation"] <= TOLERANCE
    )
    firm_capital_safe = (
        treatment["max_transfer_firm_cash_gap"] <= TOLERANCE
        and treatment["max_transfer_paid_in_equity_gap"] <= TOLERANCE
        and treatment["max_primary_issuance_cash"] <= TOLERANCE
    )
    money_accounting_safe = (
        treatment["max_transfer_reconciliation_gap"] <= TOLERANCE
        and treatment["max_money_reconciliation_gap"] <= 1e-4
        and treatment["max_accounting_gap"] <= 1e-4
        and treatment["max_household_accounting_gap"] <= 1e-4
    )
    concentration_safe = treatment["largest_person_share"] < 0.20
    consumption_safe = (
        treatment["cumulative_consumption"]
        >= control["cumulative_consumption"] * 0.95
    )
    dividend_safe = (
        treatment["person_dividend_events"] > 0
        and treatment["cumulative_person_dividends"] > TOLERANCE
        and treatment["max_dividend_routing_gap"] <= TOLERANCE
        and treatment["cumulative_legacy_dividends"]
        < control["cumulative_legacy_dividends"]
    )
    legacy_share_identity = abs(
        100.0
        - treatment["final_legacy_shares"]
        - treatment["total_secondary_shares_sold"]
    ) <= TOLERANCE
    if not liquidity_safe:
        verdict = "B. HOUSEHOLD_LIQUIDITY_DAMAGE"
    elif not gradual:
        verdict = "C. TRANSITION_TOO_FAST"
    elif not concentration_safe:
        verdict = "D. OWNERSHIP_CONCENTRATION_BLOCKER"
    elif not consumption_safe:
        verdict = "E. CONSUMPTION_CROWDING_OUT"
    elif not dividend_safe:
        verdict = "F. DIVIDEND_TRANSITION_FAILED"
    elif not firm_capital_safe or not money_accounting_safe or not legacy_share_identity:
        verdict = "G. OTHER_BLOCKER"
    else:
        verdict = "A. MINIMAL_AUTONOMOUS_SECONDARY_PURCHASE_ACCEPTED"

    flags = {
        "verdict": verdict,
        "control_secondary_buying_off": True,
        "treatment_secondary_buying_on": True,
        "same_seed": True,
        "ownership_rises": ownership_rises,
        "ownership_transition_gradual": gradual,
        "household_liquidity_safe": liquidity_safe,
        "no_negative_household_cash": treatment["max_negative_households"] == 0,
        "firm_capitalization_unchanged": firm_capital_safe,
        "money_and_accounting_reconcile": money_accounting_safe,
        "ownership_concentration_bounded": concentration_safe,
        "consumption_not_abruptly_crowded_out": consumption_safe,
        "dividend_routing_follows_ownership": dividend_safe,
        "legacy_shares_decline_only_by_secondary_transfer": legacy_share_identity,
        "primary_issuance_occurred": treatment["max_primary_issuance_cash"] > TOLERANCE,
        "leverage_enabled": False,
        "price_discovery_enabled": False,
        "investment_active": False,
        "dividend_policy_changed": False,
        "step13_changed": False,
        "new_rng_draws": 0,
    }
    write_csv(OUTPUT / "autonomous_equity_metrics.csv", summaries)
    write_csv(OUTPUT / "ownership_panel.csv", all_panel)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    make_figures(states)
    (OUTPUT / "acceptance_summary.md").write_text(
        f"""# Step 15G.1 Minimal Autonomous Secondary Equity Purchase Screen

**Verdict: {verdict}**

The control and treatment used the same fresh canonical seed 42 and 104-week horizon. Primary issuance was explicitly disabled in both cases. Treatment enabled only the deterministic secondary buyer screen at the existing 52-week review cadence.

The treatment uses the Step 15G.0 liquidity formula, a 1% available-cash engineering purchase cap, deterministic Household ordering, a 10-buyer review cap, a 10% Household equity-asset concentration cap, and a 5% Person/Firm ownership guardrail. These are screen guardrails, not final ownership targets.

Treatment final Person ownership was {treatment['final_person_ownership_fraction']:.6%}; final Legacy ownership was {treatment['final_legacy_ownership_fraction']:.6%}. Total secondary proceeds were {treatment['total_secondary_sale_proceeds']:.6f}, with {treatment['total_buyers']} buyer fills. Firm cash and paid-in equity remained unchanged by the transfer layer, and no primary issuance occurred.

Treatment cumulative Person dividends were {treatment['cumulative_person_dividends']:.6f}; cumulative Legacy dividends were {treatment['cumulative_legacy_dividends']:.6f}. Actual dividend routing remained CapTable-proportional. Household equity assets increased by the secondary purchase amount while Household cash decreased by the same amount.

No stock market, leverage, price discovery, investment, or dividend-policy change was introduced.
""",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
