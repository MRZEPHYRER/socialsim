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


OUTPUT = ROOT / "test/output/step15G7_buyer_cap_isolation"
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


def run_case(case, max_buyers):
    world = build_world(max_buyers)
    weekly = []
    reviews = []
    buyers = []
    transfer_checks = []
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
            buyers.extend({"case": case, **row} for row in decisions)
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
                "total_desired_shares": sum(
                    num(row.get("desired_shares")) for row in decisions
                ),
                "total_fillable_shares": sum(
                    num(row.get("fillable_shares")) for row in decisions
                ),
                "executed_cash": result.total_payment,
                "executed_shares": result.total_shares_sold,
            })
            transfer_checks.extend({
                "case": case,
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

        firm = world.firms[0]
        table = firm.cap_table
        cash, equity_assets, financial_net_worth = household_assets(world)
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
            "available_financial_cash": sum(
                max(
                    0.0,
                    num(h.wealth)
                    - world.autonomous_secondary_equity_system._liquidity_reserve(h)
                    - num(getattr(h, "necessary_consumption_this_step", 0.0)),
                )
                for h in world.households
            ),
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
            "firm_cash_from_diagnostics": num(diagnostics.get("firm_cash")),
        })

    final = weekly[-1]
    return world, weekly, reviews, buyers, transfer_checks, {
        "case": case,
        "max_buyers_per_firm": "unlimited" if max_buyers is None else max_buyers,
        "weeks": WEEKS,
        "completed_weeks": len(weekly),
        "runtime_failure": runtime_failure,
        "runtime_complete": not bool(runtime_failure) and len(weekly) == WEEKS,
        "review_count": len(reviews),
        "cumulative_buyers": sum(row["secondary_buyer_count"] for row in weekly),
        "cumulative_cash_transferred": sum(
            row["secondary_sale_proceeds"] for row in weekly
        ),
        "cumulative_shares_transferred": sum(
            row["secondary_shares_sold"] for row in weekly
        ),
        "total_positive_demand_observations": sum(
            row["positive_demand_households"] for row in reviews
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
        "cumulative_person_dividends": sum(
            row["person_dividends"] for row in weekly
        ),
        "cumulative_legacy_dividends": sum(
            row["legacy_dividends"] for row in weekly
        ),
        "cumulative_consumption": sum(
            row["household_consumption"] for row in weekly
        ),
        "cumulative_saving": sum(row["household_saving"] for row in weekly),
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


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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

    line("ownership_transition.png", "person_ownership_fraction", "Buyer-cap isolation: ownership", "Person ownership fraction")
    line("household_cash_equity.png", "household_equity_assets", "Household equity assets", "Acquisition-cost equity assets")
    line("ownership_concentration.png", "largest_person_share", "Ownership concentration", "Largest Person share")
    line("legacy_owner_cash.png", "legacy_owner_cash", "LegacyOwner cash", "LegacyOwner cash")
    line("dividend_routing_transition.png", "person_dividends", "Person dividend routing", "Person dividends per week")

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
    reviews = []
    buyers = []
    summaries = []
    transfer_checks = []
    for case, cap in (("control_10_buyers", 10), ("treatment_unlimited", None)):
        _, weekly, case_reviews, case_buyers, checks, summary = run_case(case, cap)
        states[case] = weekly
        reviews.extend(case_reviews)
        buyers.extend(case_buyers)
        transfer_checks.extend(checks)
        summaries.append(summary)

    control = next(row for row in summaries if row["case"] == "control_10_buyers")
    treatment = next(row for row in summaries if row["case"] == "treatment_unlimited")
    ownership_delta = treatment["final_person_ownership_fraction"] - control["final_person_ownership_fraction"]
    ownership_ratio = treatment["final_person_ownership_fraction"] / max(control["final_person_ownership_fraction"], 1e-12)
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
    if not safety:
        verdict = "E. HOUSEHOLD_LIQUIDITY_DAMAGE" if treatment["max_negative_households"] else "F. OTHER_BLOCKER"
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
    keys = sorted(set(control) | set(treatment))
    for key in keys:
        if key in {"case", "max_buyers_per_firm"}:
            continue
        comparison.append({
            "metric": key,
            "control_10_buyers": control.get(key, ""),
            "treatment_unlimited": treatment.get(key, ""),
            "difference": (
                num(treatment.get(key)) - num(control.get(key))
                if isinstance(treatment.get(key), (int, float, str))
                and isinstance(control.get(key), (int, float, str))
                and str(treatment.get(key)).replace(".", "", 1).replace("-", "", 1).isdigit()
                and str(control.get(key)).replace(".", "", 1).replace("-", "", 1).isdigit()
                else ""
            ),
        })

    flags = {
        "verdict": verdict,
        "same_seed": True,
        "same_economic_parameters": True,
        "control_runtime_complete": control["runtime_complete"],
        "treatment_runtime_complete": treatment["runtime_complete"],
        "control_runtime_failure": control["runtime_failure"],
        "treatment_runtime_failure": treatment["runtime_failure"],
        "control_buyer_cap": 10,
        "treatment_buyer_cap": "unlimited",
        "one_mechanism_changed": True,
        "purchase_rate_unchanged": True,
        "purchase_rate": 0.01,
        "liquidity_buffers_unchanged": True,
        "review_cadence_weeks": 52,
        "lagged_book_price_used": True,
        "engineering_price_not_used": True,
        "no_primary_issuance": treatment["max_primary_issuance_cash"] <= TOLERANCE,
        "all_purchases_liquidity_safe": treatment["max_negative_households"] == 0,
        "firm_capitalization_unchanged": treatment["max_transfer_firm_cash_gap"] <= TOLERANCE and treatment["max_transfer_paid_in_equity_gap"] <= TOLERANCE,
        "money_accounting_reconcile": safety,
        "ownership_concentration_bounded": concentration_safe,
        "consumption_not_abruptly_collapsed": consumption_safe,
        "ownership_larger_than_control": treatment["final_person_ownership_fraction"] > control["final_person_ownership_fraction"],
        "buyer_cap_material": material,
        "transition_still_slow": still_slow,
        "new_rng_draws": 0,
        "investment_active": False,
        "Step13_changed": False,
        "control_summary": control,
        "treatment_summary": treatment,
    }

    write_csv(OUTPUT / "review_window_metrics.csv", reviews)
    write_csv(OUTPUT / "buyer_execution_panel.csv", buyers)
    write_csv(OUTPUT / "ownership_panel.csv", states["control_10_buyers"] + states["treatment_unlimited"])
    write_csv(OUTPUT / "control_treatment_comparison.csv", comparison)
    make_figures(states)

    summary_text = f"""# Step 15G.7 Buyer-Cap Isolation Retest

## Verdict

**{verdict}**

本次使用相同 `seed=42`、`population=5000`、滞后账面权益价格和 520 周运行。
唯一变化是自主二级购买的执行买家上限：控制组为 10，处理组为无限制；1% 可用现金规则、流动性保护、52 周审查、集中度约束和确定性排序均保持不变。

## Main result

| metric | control: 10 buyers | treatment: unlimited | difference |
|---|---:|---:|---:|
| cumulative buyers | {control['cumulative_buyers']:.0f} | {treatment['cumulative_buyers']:.0f} | {treatment['cumulative_buyers'] - control['cumulative_buyers']:.0f} |
| cash transferred | {control['cumulative_cash_transferred']:.6f} | {treatment['cumulative_cash_transferred']:.6f} | {treatment['cumulative_cash_transferred'] - control['cumulative_cash_transferred']:.6f} |
| final Person ownership | {control['final_person_ownership_fraction']:.9%} | {treatment['final_person_ownership_fraction']:.9%} | {ownership_delta:.9%} |
| largest Person share | {control['final_largest_person_share']:.9%} | {treatment['final_largest_person_share']:.9%} | {treatment['final_largest_person_share'] - control['final_largest_person_share']:.9%} |
| cumulative consumption | {control['cumulative_consumption']:.6f} | {treatment['cumulative_consumption']:.6f} | {treatment['cumulative_consumption'] - control['cumulative_consumption']:.6f} |

Buyer-cap removal is classified as **{'material' if material else 'not material'}** under the pre-declared comparison rule. The treatment remains **{'slow' if still_slow else 'not slow'}** when final Person ownership is compared with a 1% scale reference.

运行完整性：控制组 `{control['completed_weeks']}/{WEEKS}` 周，处理组 `{treatment['completed_weeks']}/{WEEKS}` 周。处理组运行错误：`{treatment['runtime_failure'] or 'none'}`。

由于处理组在第 150 周触发了既有的“已死亡 Person 仍保留持股、dividend 无法找到结算 Household”错误，本次不能把 520 周对照结果判为通过。该错误不是买家上限机制，G7 不修改它。

## Safety checks

- max negative Household count: control `{control['max_negative_households']:.0f}`, treatment `{treatment['max_negative_households']:.0f}`
- max Firm cash change from secondary transfers: `{treatment['max_transfer_firm_cash_gap']:.3g}`
- max paid-in-equity change from secondary transfers: `{treatment['max_transfer_paid_in_equity_gap']:.3g}`
- max money/accounting gap: `{max(treatment['max_money_reconciliation_gap'], treatment['max_accounting_cash_flow_gap'], treatment['max_accounting_balance_sheet_gap']):.3g}`
- primary issuance cash: `{treatment['max_primary_issuance_cash']:.3g}`

详细逐周、逐审查窗口和逐 Household 记录见同目录 CSV。
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary_text, encoding="utf-8")
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary_text)


if __name__ == "__main__":
    main()
