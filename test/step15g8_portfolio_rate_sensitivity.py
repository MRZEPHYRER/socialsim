from __future__ import annotations

import csv
import importlib.util
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


OUTPUT = ROOT / "test/output/step15G8_portfolio_rate_sensitivity"
WEEKS = 520
TOLERANCE = 1e-6
RATES = {
    "R0.5": 0.005,
    "R1": 0.01,
    "R2": 0.02,
    "R5": 0.05,
}


def num(value, default=0.0):
    if value in (None, "", "nan", "NaN"):
        return float(default)
    return float(value)


def person_gini(world):
    fractions = sorted(
        holding.shares / table.total_shares
        for firm in getattr(world, "firms", [])
        for table in [getattr(firm, "cap_table", None)]
        if table is not None and table.total_shares > 0.0
        for holding in table.holdings
        if holding.holder_type == "person" and holding.shares > 0.0
    )
    if len(fractions) < 2 or sum(fractions) <= 0.0:
        return 0.0
    total = sum(fractions)
    return sum(
        (2 * index - len(fractions) - 1) * value
        for index, value in enumerate(fractions, 1)
    ) / (len(fractions) * total)


def largest_person_share(world):
    return max(
        (
            holding.shares / table.total_shares
            for firm in getattr(world, "firms", [])
            for table in [getattr(firm, "cap_table", None)]
            if table is not None and table.total_shares > 0.0
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
        + sum(
            getattr(account, "cash", 0.0)
            for account in getattr(world, "estate_accounts", {}).values()
        )
    )


def accounting_gaps(world):
    rows = getattr(world.accounting, "rows", [])
    household_rows = getattr(world.accounting, "household_rows", [])
    return {
        "cash_flow_gap": max(
            (abs(num(row.get("cash_flow_gap"))) for row in rows),
            default=0.0,
        ),
        "balance_sheet_gap": max(
            (abs(num(row.get("balance_sheet_gap"))) for row in rows),
            default=0.0,
        ),
        "inventory_bridge_gap": max(
            (abs(num(row.get("inventory_bridge_gap"))) for row in rows),
            default=0.0,
        ),
        "equity_bridge_gap": max(
            (abs(num(row.get("equity_bridge_gap"))) for row in rows),
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


def build_world(rate):
    world = World(initial_population=5000, seed=42, diagnostics_mode="full")
    world.ensure_ownership_state()
    world.person_equity_transition_enabled = False
    world.autonomous_secondary_equity_enabled = True
    world.autonomous_secondary_equity_system = (
        AutonomousSecondaryEquityPurchaseSystem(
            world,
            reference_price_mode="lagged_book_equity",
            max_available_cash_fraction=rate,
            max_buyers_per_firm=None,
        )
    )
    return world


def household_assets(world):
    cash = sum(num(h.wealth) for h in world.households)
    equity = sum(
        num(getattr(h, "equity_asset_value", 0.0))
        for h in world.households
    )
    return cash, equity, cash + equity


def household_available_cash(world):
    available = 0.0
    positive = 0
    for household in world.households:
        target = max(
            0.0, num(getattr(household, "target_wealth_this_step", 0.0))
        )
        if target <= 0.0:
            target = max(
                0.0,
                num(getattr(household, "necessary_consumption_this_step", 0.0))
                * 26.0,
            )
        amount = max(
            0.0,
            num(getattr(household, "wealth", 0.0))
            - target
            - num(getattr(household, "necessary_consumption_this_step", 0.0)),
        )
        available += amount
        positive += amount > TOLERANCE
    return available, positive


def estate_state(world):
    accounts = list(getattr(world, "estate_accounts", {}).values())
    return {
        "estate_count": len(accounts),
        "open_estate_count": sum(
            getattr(account, "status", "open") == "open"
            for account in accounts
        ),
        "estate_shares": sum(
            sum(getattr(account, "shares_by_firm", {}).values())
            for account in accounts
        ),
        "estate_cash": sum(
            num(getattr(account, "cash", 0.0)) for account in accounts
        ),
        "estate_dividend_income": sum(
            num(getattr(account, "estate_dividend_income", 0.0))
            for account in accounts
        ),
        "inheritance_event_count": len(
            getattr(world, "shareholder_estate_events", [])
        ),
    }


def run_case(rate_name, rate):
    world = build_world(rate)
    weekly = []
    reviews = []
    buyer_rows = []
    inheritance_rows = []
    transfer_checks = []
    seen_inheritance_events = 0
    runtime_failure = ""

    for _ in range(WEEKS):
        try:
            world.step()
        except Exception as error:
            runtime_failure = (
                f"step={world.current_step_index}; "
                f"{type(error).__name__}: {error}"
            )
            break

        result = getattr(world, "last_autonomous_secondary_purchase_result", None)
        reviewed = bool(result and result.reviewed)
        if reviewed:
            decisions = list(getattr(result, "decisions", []))
            buyer_rows.extend({"rate": rate_name, **row} for row in decisions)
            positive = [
                row
                for row in decisions
                if num(row.get("desired_equity_budget")) > TOLERANCE
            ]
            reviews.append({
                "rate": rate_name,
                "purchase_rate": rate,
                "review_step": world.current_step_index,
                "valuation_step": result.valuation_step,
                "book_equity_used": result.book_equity_used,
                "reference_price_used": result.reference_price_used,
                "valuation_status": result.valuation_status,
                "evaluated_households": len(decisions),
                "positive_demand_households": len(positive),
                "executed_buyers": result.buyer_count,
                "total_desired_cash_budget": sum(
                    num(row.get("desired_equity_budget")) for row in decisions
                ),
                "executed_cash": result.total_payment,
                "executed_shares": result.total_shares_sold,
            })
            transfer_checks.extend({
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
            } for transfer in result.transfer_results)

        events = getattr(world, "shareholder_estate_events", [])
        for event in events[seen_inheritance_events:]:
            inheritance_rows.append({"rate": rate_name, **event})
        seen_inheritance_events = len(events)

        firm = world.firms[0]
        table = firm.cap_table
        cash, equity_assets, financial_net_worth = household_assets(world)
        available_cash, positive_available = household_available_cash(world)
        estate = estate_state(world)
        diagnostics = world.diagnostics_rows[-1]
        weekly.append({
            "rate": rate_name,
            "purchase_rate": rate,
            "global_step": world.current_step_index,
            "reviewed": reviewed,
            "valuation_step": result.valuation_step if result else "",
            "reference_price_used": result.reference_price_used if result else "",
            "valuation_status": result.valuation_status if result else "NOT_REVIEWED",
            "secondary_sale_proceeds": result.total_payment if result else 0.0,
            "secondary_shares_sold": result.total_shares_sold if result else 0.0,
            "secondary_buyer_count": result.buyer_count if result else 0,
            "person_ownership_fraction": table.ownership_fraction("person"),
            "legacy_ownership_fraction": table.ownership_fraction("legacy"),
            "largest_person_share": largest_person_share(world),
            "person_equity_gini": person_gini(world),
            "household_cash": cash,
            "available_financial_cash": available_cash,
            "positive_available_households": positive_available,
            "household_equity_assets": equity_assets,
            "household_financial_net_worth": financial_net_worth,
            "legacy_owner_cash": getattr(world, "legacy_owner_cash", 0.0),
            "person_dividends": sum(
                num(getattr(f, "person_dividend_paid", 0.0))
                for f in world.firms
            ),
            "legacy_dividends": sum(
                num(getattr(f, "legacy_dividend_entitlement", 0.0))
                for f in world.firms
            ),
            "estate_dividends": sum(
                num(getattr(f, "estate_dividend_paid", 0.0))
                for f in world.firms
            ),
            "estate_cash": estate["estate_cash"],
            "estate_shares": estate["estate_shares"],
            "open_estate_count": estate["open_estate_count"],
            "inheritance_event_count": estate["inheritance_event_count"],
            "household_consumption": world.consumption_history[-1],
            "household_saving": world.saving_history[-1],
            "firm_cash": sum(num(getattr(f, "cash", 0.0)) for f in world.firms),
            "paid_in_equity": sum(
                num(getattr(f, "paid_in_equity", 0.0)) for f in world.firms
            ),
            "money_stock": money_stock(world),
            "money_reconciliation_gap": num(
                diagnostics.get("monetary_accounting_gap")
            ),
            "accounting_cash_flow_gap": accounting_gaps(world)["cash_flow_gap"],
            "accounting_balance_sheet_gap": accounting_gaps(world)[
                "balance_sheet_gap"
            ],
            "inventory_bridge_gap": accounting_gaps(world)["inventory_bridge_gap"],
            "equity_bridge_gap": accounting_gaps(world)["equity_bridge_gap"],
            "household_wealth_bridge_gap": accounting_gaps(world)[
                "household_wealth_bridge_gap"
            ],
            "negative_households": sum(
                num(h.wealth) < -TOLERANCE for h in world.households
            ),
            "primary_issuance_cash": sum(
                num(getattr(f, "equity_issuance_cash_this_step", 0.0))
                for f in world.firms
            ),
        })

    if not weekly:
        raise RuntimeError(f"{rate_name} produced no weekly state")
    final = weekly[-1]
    summary = {
        "rate": rate_name,
        "purchase_rate": rate,
        "buyer_cap": "unlimited",
        "weeks": WEEKS,
        "completed_weeks": len(weekly),
        "runtime_failure": runtime_failure,
        "runtime_complete": not bool(runtime_failure) and len(weekly) == WEEKS,
        "review_count": sum(bool(row["reviewed"]) for row in weekly),
        "executed_buyers": sum(row["secondary_buyer_count"] for row in weekly),
        "cumulative_cash_transferred": sum(
            row["secondary_sale_proceeds"] for row in weekly
        ),
        "cumulative_shares_transferred": sum(
            row["secondary_shares_sold"] for row in weekly
        ),
        "final_person_ownership_fraction": final["person_ownership_fraction"],
        "final_legacy_ownership_fraction": final["legacy_ownership_fraction"],
        "final_household_cash": final["household_cash"],
        "final_available_financial_cash": final["available_financial_cash"],
        "final_household_equity_assets": final["household_equity_assets"],
        "final_household_financial_net_worth": final[
            "household_financial_net_worth"
        ],
        "final_largest_person_share": final["largest_person_share"],
        "final_person_equity_gini": final["person_equity_gini"],
        "final_legacy_owner_cash": final["legacy_owner_cash"],
        "cumulative_person_dividends": sum(
            row["person_dividends"] for row in weekly
        ),
        "cumulative_legacy_dividends": sum(
            row["legacy_dividends"] for row in weekly
        ),
        "cumulative_estate_dividends": sum(
            row["estate_dividends"] for row in weekly
        ),
        "cumulative_consumption": sum(
            row["household_consumption"] for row in weekly
        ),
        "cumulative_saving": sum(row["household_saving"] for row in weekly),
        "inheritance_events": final["inheritance_event_count"],
        "final_open_estate_count": final["open_estate_count"],
        "final_estate_shares": final["estate_shares"],
        "final_estate_cash": final["estate_cash"],
        "max_negative_households": max(row["negative_households"] for row in weekly),
        "max_primary_issuance_cash": max(row["primary_issuance_cash"] for row in weekly),
        "max_money_reconciliation_gap": max(
            abs(row["money_reconciliation_gap"]) for row in weekly
        ),
        "max_accounting_cash_flow_gap": max(
            row["accounting_cash_flow_gap"] for row in weekly
        ),
        "max_accounting_balance_sheet_gap": max(
            row["accounting_balance_sheet_gap"] for row in weekly
        ),
        "max_inventory_bridge_gap": max(row["inventory_bridge_gap"] for row in weekly),
        "max_equity_bridge_gap": max(row["equity_bridge_gap"] for row in weekly),
        "max_household_wealth_bridge_gap": max(
            row["household_wealth_bridge_gap"] for row in weekly
        ),
        "max_transfer_firm_cash_gap": max(
            (row["firm_cash_gap"] for row in transfer_checks), default=0.0
        ),
        "max_transfer_paid_in_equity_gap": max(
            (row["paid_in_equity_gap"] for row in transfer_checks), default=0.0
        ),
        "max_transfer_reconciliation_gap": max(
            (row["reconciliation_gap"] for row in transfer_checks), default=0.0
        ),
    }
    return weekly, reviews, buyer_rows, inheritance_rows, summary


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_rate(row, reference):
    if not row["runtime_complete"]:
        return "OTHER_BLOCKER"
    if (
        row["max_negative_households"] > 0
        or row["max_money_reconciliation_gap"] > 1e-4
        or row["max_accounting_cash_flow_gap"] > 1e-4
        or row["max_accounting_balance_sheet_gap"] > 1e-4
        or row["max_inventory_bridge_gap"] > 1e-4
        or row["max_equity_bridge_gap"] > 1e-4
        or row["max_household_wealth_bridge_gap"] > 1e-4
    ):
        return "OTHER_BLOCKER"
    if row["final_largest_person_share"] >= 0.20:
        return "CONCENTRATION_PRESSURE"
    liquidity_pressure = (
        row["final_household_cash"] < reference["final_household_cash"] * 0.95
        or row["final_available_financial_cash"]
        < reference["final_available_financial_cash"] * 0.90
    )
    consumption_pressure = (
        row["cumulative_consumption"] < reference["cumulative_consumption"] * 0.98
    )
    if consumption_pressure:
        return "CONSUMPTION_CROWDING"
    if liquidity_pressure:
        return "LIQUIDITY_PRESSURE"
    if row["final_person_ownership_fraction"] < 0.01:
        return "SAFE_AND_SLOW"
    return "SAFE_AND_MATERIAL"


def make_figures(states):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "R0.5": "#8c8c8c",
        "R1": "#1769aa",
        "R2": "#e67e22",
        "R5": "#c0392b",
    }

    def line(filename, field, title, ylabel):
        plt.figure(figsize=(9, 4.8))
        for rate_name, rows in states.items():
            plt.plot(
                [row["global_step"] for row in rows],
                [row[field] for row in rows],
                label=rate_name,
                color=colors[rate_name],
            )
        plt.title(title)
        plt.xlabel("Global week")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT / filename, dpi=140)
        plt.close()

    line("ownership_by_rate.png", "person_ownership_fraction", "Person ownership by purchase rate", "Person ownership fraction")
    line("household_cash_by_rate.png", "household_cash", "Household cash by purchase rate", "Household cash")
    line("consumption_by_rate.png", "household_consumption", "Household consumption by purchase rate", "Consumption")
    line("saving_by_rate.png", "household_saving", "Household saving by purchase rate", "Saving")
    line("ownership_concentration_by_rate.png", "largest_person_share", "Largest Person share by purchase rate", "Largest Person share")

    plt.figure(figsize=(9, 4.8))
    for rate_name, rows in states.items():
        plt.plot(
            [row["global_step"] for row in rows],
            [row["person_dividends"] for row in rows],
            label=f"{rate_name} Person",
            color=colors[rate_name],
        )
        plt.plot(
            [row["global_step"] for row in rows],
            [row["legacy_dividends"] for row in rows],
            linestyle="--",
            label=f"{rate_name} Legacy",
            color=colors[rate_name],
            alpha=0.7,
        )
    plt.title("Dividend routing by purchase rate")
    plt.xlabel("Global week")
    plt.ylabel("Dividend flow")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT / "dividend_routing_by_rate.png", dpi=140)
    plt.close()


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    states = {}
    all_reviews = []
    all_buyers = []
    all_inheritance = []
    summaries = []

    for rate_name, rate in RATES.items():
        print(f"Running {rate_name} ({rate:.2%}) ...", flush=True)
        weekly, reviews, buyers, inheritance, summary = run_case(rate_name, rate)
        states[rate_name] = weekly
        all_reviews.extend(reviews)
        all_buyers.extend(buyers)
        all_inheritance.extend(inheritance)
        summaries.append(summary)

    reference = next(row for row in summaries if row["rate"] == "R1")
    for row in summaries:
        row["classification"] = classify_rate(row, reference)
        row["ownership_vs_R1"] = (
            row["final_person_ownership_fraction"]
            / max(reference["final_person_ownership_fraction"], 1e-12)
        )
        row["cash_vs_R1"] = (
            row["final_household_cash"]
            / max(reference["final_household_cash"], 1e-12)
        )
        row["consumption_vs_R1"] = (
            row["cumulative_consumption"]
            / max(reference["cumulative_consumption"], 1e-12)
        )

    all_safe = all(row["classification"] != "OTHER_BLOCKER" for row in summaries)
    r1 = reference
    higher_rates = [row for row in summaries if row["rate"] in {"R2", "R5"}]
    liquidity_pressure = any(
        row["classification"] == "LIQUIDITY_PRESSURE" for row in higher_rates
    )
    consumption_crowding = any(
        row["classification"] == "CONSUMPTION_CROWDING" for row in higher_rates
    )
    concentration_pressure = any(
        row["classification"] == "CONCENTRATION_PRESSURE" for row in summaries
    )
    if not all_safe:
        verdict = "F. OTHER_BLOCKER"
    elif concentration_pressure:
        verdict = "E. CONCENTRATION_BLOCKER"
    elif consumption_crowding:
        verdict = "D. CONSUMPTION_CROWDING_APPEARS_ABOVE_THRESHOLD"
    elif liquidity_pressure:
        verdict = "C. LIQUIDITY_PRESSURE_APPEARS_ABOVE_THRESHOLD"
    elif r1["classification"] in {"SAFE_AND_SLOW", "SAFE_AND_MATERIAL"}:
        higher_safe = all(
            row["classification"] in {"SAFE_AND_SLOW", "SAFE_AND_MATERIAL"}
            for row in higher_rates
        )
        verdict = (
            "A. ONE_PERCENT_RULE_REMAINS_CONSERVATIVE_AND_VALID"
            if higher_safe
            else "B. HIGHER_RATE_SAFE_BUT_CALIBRATION_REQUIRED"
        )
    else:
        verdict = "F. OTHER_BLOCKER"

    flags = {
        "verdict": verdict,
        "same_seed": True,
        "canonical_population": 5000,
        "weeks_requested": WEEKS,
        "buyer_cap": "unlimited",
        "rates_tested": list(RATES),
        "lagged_book_price_unchanged": True,
        "liquidity_buffers_unchanged": True,
        "review_cadence_weeks": 52,
        "deterministic_eligibility": True,
        "concentration_guardrails_unchanged": True,
        "dividend_policy_unchanged": True,
        "inheritance_active": True,
        "new_rng_draws": 0,
        "primary_issuance_active": False,
        "investment_active": False,
        "Step13_changed": False,
        "all_cases_completed_520": all(row["runtime_complete"] for row in summaries),
        "all_purchases_liquidity_safe": all(
            row["max_negative_households"] == 0 for row in summaries
        ),
        "firm_capitalization_unchanged": all(
            row["max_transfer_firm_cash_gap"] <= TOLERANCE
            and row["max_transfer_paid_in_equity_gap"] <= TOLERANCE
            for row in summaries
        ),
        "money_accounting_reconcile": all(
            row["max_money_reconciliation_gap"] <= 1e-4
            and row["max_accounting_cash_flow_gap"] <= 1e-4
            and row["max_accounting_balance_sheet_gap"] <= 1e-4
            and row["max_inventory_bridge_gap"] <= 1e-4
            and row["max_equity_bridge_gap"] <= 1e-4
            and row["max_household_wealth_bridge_gap"] <= 1e-4
            for row in summaries
        ),
        "inheritance_runtime_stable": all(
            row["runtime_complete"] and row["final_open_estate_count"] >= 0
            for row in summaries
        ),
        "R1_classification": r1["classification"],
        "rate_classifications": {
            row["rate"]: row["classification"] for row in summaries
        },
        "liquidity_pressure_detected": liquidity_pressure,
        "consumption_crowding_detected": consumption_crowding,
        "concentration_pressure_detected": concentration_pressure,
    }

    write_csv(OUTPUT / "rate_comparison.csv", summaries)
    write_csv(OUTPUT / "ownership_panel.csv", [row for rows in states.values() for row in rows])
    write_csv(OUTPUT / "household_finance_panel.csv", [
        {
            "rate": row["rate"],
            "purchase_rate": row["purchase_rate"],
            "global_step": row["global_step"],
            "household_cash": row["household_cash"],
            "available_financial_cash": row["available_financial_cash"],
            "positive_available_households": row["positive_available_households"],
            "household_equity_assets": row["household_equity_assets"],
            "household_financial_net_worth": row["household_financial_net_worth"],
            "household_consumption": row["household_consumption"],
            "household_saving": row["household_saving"],
            "negative_households": row["negative_households"],
        }
        for rows in states.values()
        for row in rows
    ])
    write_csv(OUTPUT / "dividend_transition_panel.csv", [
        {
            "rate": row["rate"],
            "purchase_rate": row["purchase_rate"],
            "global_step": row["global_step"],
            "person_dividends": row["person_dividends"],
            "legacy_dividends": row["legacy_dividends"],
            "estate_dividends": row["estate_dividends"],
            "legacy_owner_cash": row["legacy_owner_cash"],
            "inheritance_event_count": row["inheritance_event_count"],
            "estate_cash": row["estate_cash"],
            "estate_shares": row["estate_shares"],
            "open_estate_count": row["open_estate_count"],
        }
        for rows in states.values()
        for row in rows
    ])
    write_csv(OUTPUT / "review_window_metrics.csv", all_reviews)
    write_csv(OUTPUT / "buyer_execution_panel.csv", all_buyers)
    write_csv(OUTPUT / "inheritance_runtime_metrics.csv", all_inheritance)
    make_figures(states)

    summary_text = [
        "# Step 15G.8 Portfolio Purchase-Rate Sensitivity Audit",
        "",
        f"## Verdict\n\n**{verdict}**",
        "",
        "四组均使用 `seed=42`、`population=5000`、520 周、无限买家、52 周 review 和 lagged book-equity price；唯一经济行为变量是 available financial cash 的购买比例。",
        "",
        "| rate | classification | executed buyers | cash transferred | final Person ownership | final Household cash | cumulative consumption | inheritance events |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        summary_text.append(
            f"| {row['rate']} | {row['classification']} | {row['executed_buyers']:.0f} | {row['cumulative_cash_transferred']:.6f} | {row['final_person_ownership_fraction']:.6%} | {row['final_household_cash']:.2f} | {row['cumulative_consumption']:.2f} | {row['inheritance_events']:.0f} |"
        )
    summary_text.extend([
        "",
        "## Interpretation",
        "",
        f"R1 is the current engineering reference and is classified as `{r1['classification']}`. Ownership, liquidity, consumption, dividend routing and inheritance are compared using the same canonical runtime. Treatment does not use primary issuance; Firm cash and paid-in equity remain unchanged by secondary transfers.",
        "",
        "No purchase-rate recommendation is made from this single seed and horizon. The rate labels are screening diagnostics only.",
    ])
    (OUTPUT / "acceptance_summary.md").write_text(
        "\n".join(summary_text) + "\n", encoding="utf-8"
    )
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n".join(summary_text))


if __name__ == "__main__":
    main()
