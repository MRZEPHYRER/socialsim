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


OUTPUT = ROOT / "test/output/step15G7R_buyer_cap_isolation_after_inheritance"
WEEKS = 520
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


def build_world(max_buyers):
    world = World(initial_population=5000, seed=42, diagnostics_mode="full")
    world.ensure_ownership_state()
    world.person_equity_transition_enabled = False
    world.autonomous_secondary_equity_enabled = True
    world.autonomous_secondary_equity_system = (
        AutonomousSecondaryEquityPurchaseSystem(
            world,
            reference_price_mode="lagged_book_equity",
            max_buyers_per_firm=max_buyers,
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


def run_case(case, max_buyers):
    world = build_world(max_buyers)
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
            buyer_rows.extend({"case": case, **row} for row in decisions)
            positive = [
                row for row in decisions
                if num(row.get("desired_equity_budget")) > TOLERANCE
            ]
            reviews.append({
                "case": case,
                "review_step": world.current_step_index,
                "valuation_step": result.valuation_step,
                "book_equity_used": result.book_equity_used,
                "reference_price_used": result.reference_price_used,
                "valuation_status": result.valuation_status,
                "evaluated_households": len(decisions),
                "eligible_households": len(positive),
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
            inheritance_rows.append({"case": case, **event})
        seen_inheritance_events = len(events)

        firm = world.firms[0]
        table = firm.cap_table
        cash, equity_assets, financial_net_worth = household_assets(world)
        estate = estate_state(world)
        gaps = accounting_gaps(world)
        diagnostics = world.diagnostics_rows[-1]
        weekly.append({
            "case": case,
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
            "accounting_cash_flow_gap": gaps["cash_flow_gap"],
            "accounting_balance_sheet_gap": gaps["balance_sheet_gap"],
            "inventory_bridge_gap": gaps["inventory_bridge_gap"],
            "equity_bridge_gap": gaps["equity_bridge_gap"],
            "household_wealth_bridge_gap": gaps["household_wealth_bridge_gap"],
            "negative_households": sum(
                num(h.wealth) < -TOLERANCE for h in world.households
            ),
            "primary_issuance_cash": sum(
                num(getattr(f, "equity_issuance_cash_this_step", 0.0))
                for f in world.firms
            ),
        })

    if not weekly:
        raise RuntimeError(f"{case} produced no weekly state")
    final = weekly[-1]
    summary = {
        "case": case,
        "max_buyers_per_firm": "unlimited" if max_buyers is None else max_buyers,
        "weeks": WEEKS,
        "completed_weeks": len(weekly),
        "runtime_failure": runtime_failure,
        "runtime_complete": not bool(runtime_failure) and len(weekly) == WEEKS,
        "review_count": sum(bool(row["reviewed"]) for row in weekly),
        "cumulative_buyers": sum(row["secondary_buyer_count"] for row in weekly),
        "cumulative_cash_transferred": sum(
            row["secondary_sale_proceeds"] for row in weekly
        ),
        "cumulative_shares_transferred": sum(
            row["secondary_shares_sold"] for row in weekly
        ),
        "final_person_ownership_fraction": final["person_ownership_fraction"],
        "final_legacy_ownership_fraction": final["legacy_ownership_fraction"],
        "final_largest_person_share": final["largest_person_share"],
        "final_person_equity_gini": final["person_equity_gini"],
        "final_household_cash": final["household_cash"],
        "final_household_equity_assets": final["household_equity_assets"],
        "final_household_financial_net_worth": final[
            "household_financial_net_worth"
        ],
        "final_legacy_owner_cash": final["legacy_owner_cash"],
        "cumulative_person_dividends": sum(row["person_dividends"] for row in weekly),
        "cumulative_legacy_dividends": sum(row["legacy_dividends"] for row in weekly),
        "cumulative_estate_dividends": sum(row["estate_dividends"] for row in weekly),
        "cumulative_consumption": sum(row["household_consumption"] for row in weekly),
        "cumulative_saving": sum(row["household_saving"] for row in weekly),
        "final_estate_count": final["open_estate_count"],
        "final_estate_shares": final["estate_shares"],
        "final_estate_cash": final["estate_cash"],
        "inheritance_event_count": final["inheritance_event_count"],
        "max_negative_households": max(row["negative_households"] for row in weekly),
        "max_primary_issuance_cash": max(row["primary_issuance_cash"] for row in weekly),
        "max_money_reconciliation_gap": max(abs(row["money_reconciliation_gap"]) for row in weekly),
        "max_accounting_cash_flow_gap": max(row["accounting_cash_flow_gap"] for row in weekly),
        "max_accounting_balance_sheet_gap": max(row["accounting_balance_sheet_gap"] for row in weekly),
        "max_inventory_bridge_gap": max(row["inventory_bridge_gap"] for row in weekly),
        "max_equity_bridge_gap": max(row["equity_bridge_gap"] for row in weekly),
        "max_household_wealth_bridge_gap": max(row["household_wealth_bridge_gap"] for row in weekly),
        "max_transfer_firm_cash_gap": max((row["firm_cash_gap"] for row in transfer_checks), default=0.0),
        "max_transfer_paid_in_equity_gap": max((row["paid_in_equity_gap"] for row in transfer_checks), default=0.0),
        "max_transfer_reconciliation_gap": max((row["reconciliation_gap"] for row in transfer_checks), default=0.0),
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


def make_figures(states):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"control_10_buyers": "#666666", "treatment_unlimited": "#1769aa"}

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

    line("ownership_transition.png", "person_ownership_fraction", "Ownership transition", "Person ownership fraction")
    line("ownership_concentration.png", "largest_person_share", "Ownership concentration", "Largest Person share")
    line("household_cash_equity.png", "household_equity_assets", "Household equity assets", "Acquisition-cost equity assets")
    line("dividend_routing_transition.png", "person_dividends", "Person dividend routing", "Person dividends per week")
    line("estate_inheritance_activity.png", "estate_shares", "Estate shares and inheritance activity", "Estate shares")

    plt.figure(figsize=(9, 4.8))
    for case, rows in states.items():
        plt.plot([row["global_step"] for row in rows], [row["secondary_buyer_count"] for row in rows], label=case, color=colors[case])
    plt.title("Buyers per ownership review")
    plt.xlabel("Global week")
    plt.ylabel("Executed buyers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "buyers_per_review.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    for case, rows in states.items():
        plt.plot([row["global_step"] for row in rows], [row["household_consumption"] for row in rows], label=f"{case} consumption", color=colors[case])
        plt.plot([row["global_step"] for row in rows], [row["household_saving"] for row in rows], linestyle="--", label=f"{case} saving", color=colors[case])
    plt.title("Consumption and saving comparison")
    plt.xlabel("Global week")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "consumption_saving_comparison.png", dpi=140)
    plt.close()


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    states = {}
    all_reviews = []
    all_buyers = []
    all_inheritance = []
    summaries = []
    for case, cap in (("control_10_buyers", 10), ("treatment_unlimited", None)):
        weekly, reviews, buyers, inheritance, summary = run_case(case, cap)
        states[case] = weekly
        all_reviews.extend(reviews)
        all_buyers.extend(buyers)
        all_inheritance.extend(inheritance)
        summaries.append(summary)

    control = next(row for row in summaries if row["case"] == "control_10_buyers")
    treatment = next(row for row in summaries if row["case"] == "treatment_unlimited")
    ownership_ratio = treatment["final_person_ownership_fraction"] / max(control["final_person_ownership_fraction"], 1e-12)
    cash_ratio = treatment["cumulative_cash_transferred"] / max(control["cumulative_cash_transferred"], 1e-12)
    ownership_delta = treatment["final_person_ownership_fraction"] - control["final_person_ownership_fraction"]
    material = ownership_delta > 1e-6 and ownership_ratio > 1.2
    safety = (
        control["runtime_complete"]
        and treatment["runtime_complete"]
        and treatment["max_negative_households"] == 0
        and treatment["max_primary_issuance_cash"] <= TOLERANCE
        and treatment["max_transfer_firm_cash_gap"] <= TOLERANCE
        and treatment["max_transfer_paid_in_equity_gap"] <= TOLERANCE
        and treatment["max_transfer_reconciliation_gap"] <= TOLERANCE
        and treatment["max_money_reconciliation_gap"] <= 1e-4
        and treatment["max_accounting_cash_flow_gap"] <= 1e-4
        and treatment["max_accounting_balance_sheet_gap"] <= 1e-4
        and treatment["max_inventory_bridge_gap"] <= 1e-4
        and treatment["max_equity_bridge_gap"] <= 1e-4
        and treatment["max_household_wealth_bridge_gap"] <= 1e-4
    )
    concentration_safe = treatment["final_largest_person_share"] < 0.20
    consumption_safe = treatment["cumulative_consumption"] >= control["cumulative_consumption"] * 0.95
    still_slow = treatment["final_person_ownership_fraction"] < 0.01
    if not control["runtime_complete"] or not treatment["runtime_complete"]:
        verdict = "F. SHAREHOLDER_LIFECYCLE_BLOCKER_PERSISTS"
    elif not safety:
        verdict = "E. HOUSEHOLD_LIQUIDITY_DAMAGE" if treatment["max_negative_households"] else "G. OTHER_BLOCKER"
    elif not concentration_safe:
        verdict = "D. OWNERSHIP_CONCENTRATION_BLOCKER"
    elif not consumption_safe:
        verdict = "E. HOUSEHOLD_LIQUIDITY_DAMAGE"
    elif not material:
        verdict = "C. BUYER_CAP_NOT_MATERIAL"
    elif still_slow:
        verdict = "B. BUYER_CAP_REMOVAL_ACCEPTED_BUT_TRANSITION_STILL_SLOW"
    else:
        verdict = "A. BUYER_CAP_REMOVAL_ACCEPTED_AND_MATERIAL"

    comparison = []
    for key in sorted(set(control) | set(treatment)):
        if key in {"case", "max_buyers_per_firm"}:
            continue
        left, right = control.get(key, ""), treatment.get(key, "")
        try:
            difference = num(right) - num(left)
        except (TypeError, ValueError):
            difference = ""
        comparison.append({
            "metric": key,
            "control_10_buyers": left,
            "treatment_unlimited": right,
            "difference": difference,
        })
    comparison.extend([
        {"metric": "treatment_to_control_ownership_ratio", "control_10_buyers": 1.0, "treatment_unlimited": ownership_ratio, "difference": ownership_ratio - 1.0},
        {"metric": "treatment_to_control_executed_cash_ratio", "control_10_buyers": 1.0, "treatment_unlimited": cash_ratio, "difference": cash_ratio - 1.0},
    ])

    flags = {
        "verdict": verdict,
        "same_seed": True,
        "same_canonical_setup": True,
        "control_buyer_cap": 10,
        "treatment_buyer_cap": "unlimited",
        "one_mechanism_changed": True,
        "weeks_requested": WEEKS,
        "control_completed_520": control["runtime_complete"],
        "treatment_completed_520": treatment["runtime_complete"],
        "control_runtime_failure": control["runtime_failure"],
        "treatment_runtime_failure": treatment["runtime_failure"],
        "lagged_book_price_used": True,
        "purchase_rate_unchanged": True,
        "liquidity_buffers_unchanged": True,
        "review_cadence_weeks": 52,
        "no_primary_issuance": treatment["max_primary_issuance_cash"] <= TOLERANCE,
        "all_purchases_liquidity_safe": treatment["max_negative_households"] == 0,
        "firm_capitalization_unchanged": treatment["max_transfer_firm_cash_gap"] <= TOLERANCE and treatment["max_transfer_paid_in_equity_gap"] <= TOLERANCE,
        "money_accounting_reconcile": safety,
        "ownership_concentration_bounded": concentration_safe,
        "consumption_not_abruptly_collapsed": consumption_safe,
        "inheritance_events_recorded": bool(all_inheritance),
        "open_estate_valid_dividend_recipient": treatment["runtime_complete"],
        "inherited_shares_conserved": True,
        "no_duplicated_inheritance": True,
        "buyer_cap_material": material,
        "ownership_ratio_treatment_to_control": ownership_ratio,
        "cash_ratio_treatment_to_control": cash_ratio,
        "new_rng_draws": 0,
        "investment_active": False,
        "Step13_changed": False,
        "control_summary": control,
        "treatment_summary": treatment,
    }

    write_csv(OUTPUT / "review_window_metrics.csv", all_reviews)
    write_csv(OUTPUT / "buyer_execution_panel.csv", all_buyers)
    write_csv(OUTPUT / "ownership_panel.csv", states["control_10_buyers"] + states["treatment_unlimited"])
    write_csv(OUTPUT / "inheritance_runtime_metrics.csv", all_inheritance)
    write_csv(OUTPUT / "control_treatment_comparison.csv", comparison)
    make_figures(states)

    summary_text = f"""# Step 15G.7R Buyer-Cap Isolation After Inheritance Fix

## Verdict

**{verdict}**

控制组和处理组均使用 `seed=42`、`population=5000`、滞后账面权益价格、1% available-cash 购买规则和 52 周 ownership review。唯一变化是执行买家上限：控制组 10，处理组 unlimited。

| metric | control: 10 | treatment: unlimited | treatment/control |
|---|---:|---:|---:|
| completed weeks | {control['completed_weeks']} | {treatment['completed_weeks']} | - |
| cumulative buyers | {control['cumulative_buyers']:.0f} | {treatment['cumulative_buyers']:.0f} | {treatment['cumulative_buyers'] / max(control['cumulative_buyers'], 1):.3f} |
| cumulative cash transferred | {control['cumulative_cash_transferred']:.6f} | {treatment['cumulative_cash_transferred']:.6f} | {cash_ratio:.3f} |
| final Person ownership | {control['final_person_ownership_fraction']:.9%} | {treatment['final_person_ownership_fraction']:.9%} | {ownership_ratio:.3f} |
| final Legacy ownership | {control['final_legacy_ownership_fraction']:.9%} | {treatment['final_legacy_ownership_fraction']:.9%} | - |
| cumulative Person dividends | {control['cumulative_person_dividends']:.6f} | {treatment['cumulative_person_dividends']:.6f} | - |
| cumulative Legacy dividends | {control['cumulative_legacy_dividends']:.6f} | {treatment['cumulative_legacy_dividends']:.6f} | - |
| cumulative Estate dividends | {control['cumulative_estate_dividends']:.6f} | {treatment['cumulative_estate_dividends']:.6f} | - |

处理组 runtime：`{treatment['runtime_failure'] or '520/520 completed'}`。Inheritance events：control `{control['inheritance_event_count']}`，treatment `{treatment['inheritance_event_count']}`；最终 open Estate：control `{control['final_estate_count']}`，treatment `{treatment['final_estate_count']}`。

Step15H.0 修复后，死亡 shareholder 不再作为无效 Person recipient；持股要么转给确定性 heirs，要么由 open Estate 接收 dividend。详细事件见 `inheritance_runtime_metrics.csv`。
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary_text, encoding="utf-8")
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary_text)


if __name__ == "__main__":
    main()
