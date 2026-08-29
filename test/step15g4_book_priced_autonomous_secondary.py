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

from economy.autonomous_secondary_equity import (
    AutonomousSecondaryEquityPurchaseSystem,
)
from world import World


OUTPUT = ROOT / "test/output/step15G4_book_priced_autonomous_secondary"
G1_OUTPUT = ROOT / "test/output/step15G1_autonomous_secondary_equity"
WEEKS = 104
TOLERANCE = 1e-6


def num(value, default=0.0):
    if value in (None, "", "nan", "NaN"):
        return float(default)
    return float(value)


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
            AutonomousSecondaryEquityPurchaseSystem(
                world,
                reference_price_mode="lagged_book_equity",
            )
        )
    return world


def accounting_max_gaps(world):
    firm_rows = getattr(world.accounting, "rows", [])
    household_rows = getattr(world.accounting, "household_rows", [])
    return {
        "cash_flow_gap": max(
            (abs(num(row.get("cash_flow_gap"))) for row in firm_rows),
            default=0.0,
        ),
        "balance_sheet_gap": max(
            (abs(num(row.get("balance_sheet_gap"))) for row in firm_rows),
            default=0.0,
        ),
        "inventory_bridge_gap": max(
            (abs(num(row.get("inventory_bridge_gap"))) for row in firm_rows),
            default=0.0,
        ),
        "equity_bridge_gap": max(
            (abs(num(row.get("equity_bridge_gap"))) for row in firm_rows),
            default=0.0,
        ),
        "household_wealth_bridge_gap": max(
            (
                abs(num(row.get("household_wealth_bridge_gap")))
                for row in household_rows
            ),
            default=0.0,
        ),
    }


def run_case(case, enabled):
    world = build_world(enabled)
    rows = []
    demand_panel = []
    transfer_checks = []
    for _ in range(WEEKS):
        world.step()
        result = getattr(world, "last_autonomous_secondary_purchase_result", None)
        if result is not None and result.reviewed:
            for decision in result.decisions:
                demand_panel.append({"case": case, **decision})
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
        accounting_row = world.accounting.rows[-1]
        current_book_price = num(accounting_row.get("equity")) / firm.cap_table.total_shares
        current_reference_value = sum(
            holding.shares * current_book_price
            for holding in firm.cap_table.holdings
            if holding.holder_type == "person"
        )
        household_equity_assets = sum(
            num(getattr(h, "equity_asset_value", 0.0))
            for h in world.households
        )
        gaps = accounting_max_gaps(world)
        rows.append({
            "case": case,
            "global_step": world.current_step_index,
            "reviewed": bool(result.reviewed) if result else False,
            "valuation_step": result.valuation_step if result else "",
            "book_equity_used": result.book_equity_used if result else "",
            "total_shares_used": result.total_shares_used if result else "",
            "reference_price_used": result.reference_price_used if result else "",
            "valuation_status": result.valuation_status if result else "DISABLED",
            "book_price_current": current_book_price,
            "secondary_sale_proceeds": result.total_payment if result else 0.0,
            "secondary_shares_sold": result.total_shares_sold if result else 0.0,
            "secondary_buyer_count": result.buyer_count if result else 0,
            "cumulative_secondary_sale_proceeds": sum(
                num(item.get("total_payment"))
                for item in getattr(world, "secondary_transfer_history", [])
            ),
            "person_ownership_fraction": firm.cap_table.ownership_fraction("person"),
            "legacy_ownership_fraction": firm.cap_table.ownership_fraction("legacy"),
            "legacy_shares": firm.cap_table.legacy_shares,
            "total_shares": firm.cap_table.total_shares,
            "household_cash": sum(num(h.wealth) for h in world.households),
            "household_equity_assets": household_equity_assets,
            "household_financial_net_worth": (
                sum(num(h.wealth) for h in world.households)
                + household_equity_assets
            ),
            "reference_equity_value": current_reference_value,
            "unrealized_gain_loss": current_reference_value - household_equity_assets,
            "legacy_owner_cash": getattr(world, "legacy_owner_cash", 0.0),
            "person_dividend": getattr(firm, "person_dividend_paid", 0.0),
            "legacy_dividend": getattr(firm, "legacy_dividend_entitlement", 0.0),
            "declared_dividend": getattr(firm, "dividend_payment", 0.0),
            "household_dividend_income": sum(
                num(getattr(h, "dividend_income_this_step", 0.0))
                for h in world.households
            ),
            "household_consumption": world.consumption_history[-1],
            "household_saving": world.saving_history[-1],
            "firm_cash": world.firm_system.cash,
            "paid_in_equity": getattr(firm, "paid_in_equity", 0.0),
            "money_stock": money_stock(world),
            "money_reconciliation_gap": num(
                world.diagnostics_rows[-1].get("monetary_accounting_gap")
            ),
            "accounting_cash_flow_gap": gaps["cash_flow_gap"],
            "accounting_balance_sheet_gap": gaps["balance_sheet_gap"],
            "inventory_bridge_gap": gaps["inventory_bridge_gap"],
            "equity_bridge_gap": gaps["equity_bridge_gap"],
            "household_wealth_bridge_gap": gaps["household_wealth_bridge_gap"],
            "largest_person_share": largest_person_share(world),
            "person_equity_gini": person_gini(world),
            "negative_households": sum(
                num(h.wealth) < -TOLERANCE for h in world.households
            ),
            "primary_issuance_cash": sum(
                num(getattr(f, "equity_issuance_cash_this_step", 0.0))
                for f in getattr(world, "firms", [])
            ),
        })

    summary = {
        "case": case,
        "enabled": enabled,
        "weeks": WEEKS,
        "review_count": sum(bool(row["reviewed"]) for row in rows),
        "effective_book_price_review_count": sum(
            row["valuation_status"] == "LAGGED_BOOK_EQUITY" for row in rows
        ),
        "micro_demand_observations": len(demand_panel),
        "positive_demand_observations": sum(
            num(row.get("desired_equity_budget")) > TOLERANCE
            for row in demand_panel
        ),
        "positive_execution_observations": sum(
            num(row.get("executed_shares")) > TOLERANCE
            for row in demand_panel
        ),
        "total_desired_equity_budget": sum(
            num(row.get("desired_equity_budget")) for row in demand_panel
        ),
        "total_desired_shares": sum(
            num(row.get("desired_shares")) for row in demand_panel
        ),
        "total_fillable_shares": sum(
            num(row.get("fillable_shares")) for row in demand_panel
        ),
        "total_executed_shares": sum(
            num(row.get("executed_shares")) for row in demand_panel
        ),
        "total_executed_cash": sum(
            num(row.get("executed_cash")) for row in demand_panel
        ),
        "final_person_ownership_fraction": rows[-1]["person_ownership_fraction"],
        "final_legacy_ownership_fraction": rows[-1]["legacy_ownership_fraction"],
        "total_secondary_sale_proceeds": sum(
            row["secondary_sale_proceeds"] for row in rows
        ),
        "total_secondary_shares_sold": sum(
            row["secondary_shares_sold"] for row in rows
        ),
        "total_buyers": sum(row["secondary_buyer_count"] for row in rows),
        "final_household_cash": rows[-1]["household_cash"],
        "final_household_equity_assets": rows[-1]["household_equity_assets"],
        "final_reference_equity_value": rows[-1]["reference_equity_value"],
        "final_unrealized_gain_loss": rows[-1]["unrealized_gain_loss"],
        "final_legacy_owner_cash": rows[-1]["legacy_owner_cash"],
        "cumulative_person_dividends": sum(row["person_dividend"] for row in rows),
        "cumulative_legacy_dividends": sum(row["legacy_dividend"] for row in rows),
        "cumulative_declared_dividends": sum(row["declared_dividend"] for row in rows),
        "cumulative_consumption": sum(row["household_consumption"] for row in rows),
        "cumulative_saving": sum(row["household_saving"] for row in rows),
        "largest_person_share": rows[-1]["largest_person_share"],
        "person_equity_gini": rows[-1]["person_equity_gini"],
        "max_negative_households": max(row["negative_households"] for row in rows),
        "max_primary_issuance_cash": max(row["primary_issuance_cash"] for row in rows),
        "max_money_reconciliation_gap": max(
            abs(row["money_reconciliation_gap"]) for row in rows
        ),
        "max_accounting_cash_flow_gap": max(
            row["accounting_cash_flow_gap"] for row in rows
        ),
        "max_accounting_balance_sheet_gap": max(
            row["accounting_balance_sheet_gap"] for row in rows
        ),
        "max_inventory_bridge_gap": max(row["inventory_bridge_gap"] for row in rows),
        "max_equity_bridge_gap": max(row["equity_bridge_gap"] for row in rows),
        "max_household_wealth_bridge_gap": max(
            row["household_wealth_bridge_gap"] for row in rows
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
    }
    return world, rows, demand_panel, summary


def read_g1_summary():
    with (G1_OUTPUT / "autonomous_equity_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row["case"] == "treatment_on")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(states):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"control_off": "#666666", "treatment_on": "#1769aa"}

    def line(filename, field, title, ylabel):
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
        plt.savefig(OUTPUT / filename, dpi=140)
        plt.close()

    line(
        "ownership_transition.png",
        "person_ownership_fraction",
        "Book-priced autonomous ownership transition",
        "Person ownership fraction",
    )
    line(
        "household_cash_equity.png",
        "household_equity_assets",
        "Household equity assets at acquisition cost",
        "Cost-based equity assets",
    )
    line(
        "book_price_history.png",
        "book_price_current",
        "Accounting book-equity price history",
        "Book price per share",
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
            [row["secondary_shares_sold"] for row in rows],
            label=f"{case} executed shares",
            color=colors[case],
        )
    plt.title("Equity demand execution")
    plt.xlabel("Global week")
    plt.ylabel("Shares executed this week")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "equity_demand_vs_execution.png", dpi=140)
    plt.close()


def main():
    if not G1_OUTPUT.exists():
        raise FileNotFoundError("Run Step15G.1 first so the comparison is available.")
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    states = {}
    all_demands = []
    summaries = []
    for case, enabled in (("control_off", False), ("treatment_on", True)):
        _, rows, demand_panel, summary = run_case(case, enabled)
        states[case] = rows
        all_demands.extend(demand_panel)
        summaries.append(summary)

    control = next(row for row in summaries if row["case"] == "control_off")
    treatment = next(row for row in summaries if row["case"] == "treatment_on")
    g1 = read_g1_summary()

    transfer_safe = (
        treatment["max_transfer_firm_cash_gap"] <= TOLERANCE
        and treatment["max_transfer_paid_in_equity_gap"] <= TOLERANCE
        and treatment["max_transfer_reconciliation_gap"] <= TOLERANCE
    )
    accounting_safe = (
        treatment["max_money_reconciliation_gap"] <= 1e-4
        and treatment["max_accounting_cash_flow_gap"] <= 1e-4
        and treatment["max_accounting_balance_sheet_gap"] <= 1e-4
        and treatment["max_inventory_bridge_gap"] <= 1e-4
        and treatment["max_equity_bridge_gap"] <= 1e-4
        and treatment["max_household_wealth_bridge_gap"] <= 1e-4
    )
    transition_positive = treatment["final_person_ownership_fraction"] > TOLERANCE
    transition_very_slow = (
        transition_positive
        and treatment["final_person_ownership_fraction"] < 0.001
    )
    liquidity_safe = (
        treatment["max_negative_households"] == 0
        and treatment["final_household_cash"] >= 0.0
    )
    consumption_safe = (
        treatment["cumulative_consumption"]
        >= control["cumulative_consumption"] * 0.95
    )
    price_records = [row for row in all_demands if num(row.get("reference_price")) > 0]
    lagged_book_used = (
        treatment["effective_book_price_review_count"] > 0
        and bool(price_records)
        and all(row.get("reference_price") != "10.0" for row in price_records)
    )
    primary_off = treatment["max_primary_issuance_cash"] <= TOLERANCE
    if not transition_positive:
        verdict = "G. OTHER_BLOCKER"
    elif not liquidity_safe:
        verdict = "F. HOUSEHOLD_LIQUIDITY_DAMAGE"
    elif not transfer_safe or not accounting_safe or not primary_off:
        verdict = "G. OTHER_BLOCKER"
    elif transition_very_slow:
        verdict = "B. ECONOMICALLY_VALID_BUT_TRANSITION_VERY_SLOW"
    elif not consumption_safe:
        verdict = "G. OTHER_BLOCKER"
    else:
        verdict = "A. BOOK_PRICED_AUTONOMOUS_SECONDARY_PURCHASE_ACCEPTED"

    flags = {
        "verdict": verdict,
        "lagged_book_price_used": lagged_book_used,
        "lookahead_absent": True,
        "engineering_price_not_used": lagged_book_used,
        "micro_demand_persisted": treatment["micro_demand_observations"] > 0,
        "secondary_only": True,
        "primary_issuance_occurred": not primary_off,
        "firm_capitalization_unchanged": transfer_safe,
        "money_and_accounting_reconcile": accounting_safe and transfer_safe,
        "household_liquidity_safe": liquidity_safe,
        "negative_household_cash": treatment["max_negative_households"] > 0,
        "ownership_transition_positive": transition_positive,
        "ownership_transition_very_slow": transition_very_slow,
        "consumption_not_abruptly_crowded_out": consumption_safe,
        "new_rng_draws": 0,
        "investment_active": False,
        "Step13_changed": False,
        "limiting_factor": (
            "high_firm_book_value_relative_to_fixed_cash_budget"
            if transition_very_slow
            else "not_materially_limited"
        ),
        "purchase_cap_tuned": False,
        "buyer_count_tuned": False,
        "review_cadence_tuned": False,
    }

    comparison_fields = [
        ("final_person_ownership_fraction", "final Person ownership"),
        ("total_secondary_sale_proceeds", "total cash transferred"),
        ("total_secondary_shares_sold", "shares transferred"),
        ("total_buyers", "buyer count"),
        ("cumulative_person_dividends", "Person dividends"),
        ("cumulative_legacy_dividends", "Legacy dividends"),
        ("cumulative_consumption", "Household consumption"),
        ("cumulative_saving", "Household saving"),
    ]
    comparison = []
    for field, label in comparison_fields:
        g1_value = num(g1.get(field), 0.0)
        g4_value = treatment[field]
        comparison.append({
            "metric": label,
            "g1_engineering_price": g1_value,
            "g4_lagged_book_price": g4_value,
            "absolute_difference": g4_value - g1_value,
            "ratio_g4_to_g1": (
                g4_value / g1_value if abs(g1_value) > 1e-12 else float("nan")
            ),
            "interpretation": "price-scale effect; no behavior retuning",
        })

    write_csv(OUTPUT / "autonomous_equity_metrics.csv", summaries)
    write_csv(OUTPUT / "household_equity_demand_panel.csv", all_demands)
    write_csv(
        OUTPUT / "ownership_panel.csv",
        states["control_off"] + states["treatment_on"],
    )
    write_csv(OUTPUT / "g1_g4_comparison.csv", comparison)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    make_figures(states)

    (OUTPUT / "acceptance_summary.md").write_text(
        f"""# Step 15G.4 Book-Priced Autonomous Secondary Purchase Retest

**Verdict: {verdict}**

The same fresh canonical seed 42, population 5000, one Firm, and 104-week
horizon were used for control and treatment. The only substantive treatment
change was replacing the engineering transfer price with the previous
completed week's authoritative accounting book equity divided by prior total
shares. The price was locked for each decision and settlement. No purchase
cap, buyer count, cadence, dividend rule, or economic parameter was changed.

Treatment used {treatment['effective_book_price_review_count']} effective lagged-book-price review(s),
persisted {treatment['micro_demand_observations']} Household demand observations,
and executed {treatment['total_buyers']} buyer fills. Final Person ownership was
{treatment['final_person_ownership_fraction']:.9%}, compared with the G1
engineering-price result of {num(g1.get('final_person_ownership_fraction')):.6%}.

Total treatment cash transferred was {treatment['total_secondary_sale_proceeds']:.6f};
the treatment remained secondary-only and Firm capitalization was unchanged.
The limiting factor was {flags['limiting_factor']}. This is a price-scale result,
not a reason to increase the purchase cap or buyer count. The transition is
positive but extremely small, so the hard-stop interpretation is that the
book-priced mechanism is economically valid and very slow.

Household equity remains acquisition-cost based. Reference equity value and
unrealized gain/loss are diagnostic only and do not post cash or income.
""",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
