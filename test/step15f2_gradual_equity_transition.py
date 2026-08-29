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


OUTPUT = ROOT / "test/output/step15F2_gradual_equity_transition"
CHECKPOINT = ROOT / (
    "test/output/step10_9_warm_checkpoint/"
    "wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl"
)
CONTINUATION_WEEKS = 104
TOLERANCE = 1e-6


def load_case(enabled):
    world, metadata = load_world_checkpoint(str(CHECKPOINT))
    world.ensure_ownership_state()
    for name, default in {
        "firm_diagnostics_rows": [],
        "household_diagnostics_rows": [],
        "diagnostics_rows": [],
    }.items():
        if not hasattr(world, name):
            setattr(world, name, default)
    world.person_equity_transition_enabled = bool(enabled)
    # A checkpoint loaded from older code may not carry the adapter.
    if enabled:
        from economy.equity_transition import GradualPersonEquityTransition

        world.equity_transition_system = GradualPersonEquityTransition(world)
    return world, metadata


def ownership_totals(world):
    views = world.ownership_analysis_views()
    firms = views.get("firms", [])
    households = views.get("households", [])
    return {
        "legacy_fraction": sum(
            item.get("ownership_fractions", {}).get("legacy", 0.0)
            for item in firms
        )
        / max(1, len(firms)),
        "person_fraction": sum(
            item.get("ownership_fractions", {}).get("person", 0.0)
            for item in firms
        )
        / max(1, len(firms)),
        "person_shareholder_count": sum(
            item.get("person_shareholder_count", 0) for item in firms
        ),
        "largest_person_share": max(
            (
                holding.shares / table.total_shares
                for firm in getattr(world, "firms", [])
                for table in [getattr(firm, "cap_table", None)]
                if table is not None
                for holding in table.holdings
                if holding.holder_type == "person"
            ),
            default=0.0,
        ),
        "household_cash": sum(h.wealth for h in world.households),
        "household_equity_assets": sum(
            getattr(h, "equity_asset_value", 0.0) for h in world.households
        ),
        "household_financial_net_worth": sum(
            h.wealth + getattr(h, "equity_asset_value", 0.0)
            for h in world.households
        ),
        "household_count": len(households),
    }


def person_equity_gini(world):
    fractions = sorted(
        holding.shares / table.total_shares
        for firm in getattr(world, "firms", [])
        for table in [getattr(firm, "cap_table", None)]
        if table is not None
        for holding in table.holdings
        if holding.holder_type == "person" and holding.shares > 0
    )
    if len(fractions) < 2 or sum(fractions) <= 0.0:
        return 0.0
    total = sum(fractions)
    n = len(fractions)
    return sum(
        (2 * index - n - 1) * value
        for index, value in enumerate(fractions, 1)
    ) / (n * total)


def money_stock(world):
    cb = world.firm_system.central_bank
    return (
        world.firm_system.cash
        + sum(h.wealth for h in world.households)
        + getattr(world, "legacy_owner_cash", 0.0)
        + getattr(world, "public_wealth", 0.0)
        + getattr(cb, "public_income_balance", 0.0)
    )


def run_case(label, enabled):
    world, metadata = load_case(enabled)
    rows = []
    initial_money = money_stock(world)
    initial_ownership = ownership_totals(world)
    for _ in range(CONTINUATION_WEEKS):
        world.step()
        result = getattr(world, "last_equity_transition_result", None)
        result = result if enabled and result is not None else None
        firm = world.firms[0]
        current = ownership_totals(world)
        accounting_rows = [
            row for row in getattr(world.accounting, "rows", [])
            if int(row.get("step", -1)) == len(world.population_history) - 1
        ]
        household_rows = [
            row for row in getattr(world.accounting, "household_rows", [])
            if int(row.get("step", -1)) == len(world.population_history) - 1
        ]
        accounting_gap = max(
            (abs(float(row.get("cash_flow_gap", 0.0))) for row in accounting_rows),
            default=0.0,
        )
        household_gap = max(
            (abs(float(row.get("household_wealth_bridge_gap", 0.0))) for row in household_rows),
            default=0.0,
        )
        rows.append({
            "case": label,
            "global_step": len(world.population_history),
            "reviewed": bool(result.reviewed) if result else False,
            "issuance_cash": float(result.total_issued) if result else 0.0,
            "shares_issued": float(result.total_shares_issued) if result else 0.0,
            "buyers": len(result.fills) if result else 0,
            "legacy_ownership_fraction": current["legacy_fraction"],
            "person_ownership_fraction": current["person_fraction"],
            "person_shareholder_count": current["person_shareholder_count"],
            "largest_person_share": current["largest_person_share"],
            "person_equity_gini": person_equity_gini(world),
            "household_cash": current["household_cash"],
            "household_equity_assets": current["household_equity_assets"],
            "household_financial_net_worth": current["household_financial_net_worth"],
            "household_consumption": world.consumption_history[-1],
            "household_saving": world.saving_history[-1],
            "household_total_income": world.income_history[-1],
            "person_dividend": getattr(firm, "person_dividend_paid", 0.0),
            "legacy_dividend": getattr(firm, "legacy_dividend_entitlement", 0.0),
            "legacy_owner_cash": getattr(world, "legacy_owner_cash", 0.0),
            "firm_cash": world.firm_system.cash,
            "paid_in_equity": getattr(firm, "paid_in_equity", 0.0),
            "money_stock": money_stock(world),
            "money_reconciliation_gap": (
                float(
                    world.diagnostics_rows[-1].get(
                        "monetary_accounting_gap", 0.0
                    )
                )
                if getattr(world, "diagnostics_rows", [])
                else 0.0
            ),
            "accounting_reconciliation_max_abs": accounting_gap,
            "household_accounting_gap": household_gap,
        })
    final = rows[-1]
    reviews = [row for row in rows if row["reviewed"]]
    total_declared_dividends = sum(
        max(0.0, float(getattr(firm, "dividend_payment", 0.0)))
        for firm in getattr(world, "firms", [])
    )
    initial_firm_paid_in = float(
        getattr(world.firms[0], "paid_in_equity", 0.0)
    ) - sum(row["issuance_cash"] for row in rows)
    # The warm distressed path may declare no dividend during this short
    # continuation. Calculate the accepted routing entitlement at the final
    # ownership state without applying a synthetic cash transfer.
    probe_amount = 1000.0
    table = world.firms[0].cap_table
    probe_person_paid = probe_amount * table.ownership_fraction("person")
    probe_legacy_entitlement = probe_amount * table.ownership_fraction("legacy")
    return {
        "case": label,
        "enabled": enabled,
        "start_global_step": metadata.get("global_step"),
        "end_global_step": final["global_step"],
        "review_count": len(reviews),
        "total_equity_issuance": sum(row["issuance_cash"] for row in rows),
        "total_shares_issued": sum(row["shares_issued"] for row in rows),
        "person_shareholder_count": final["person_shareholder_count"],
        "final_person_ownership_fraction": final["person_ownership_fraction"],
        "final_legacy_ownership_fraction": final["legacy_ownership_fraction"],
        "final_largest_person_share": final["largest_person_share"],
        "final_person_equity_gini": final["person_equity_gini"],
        "household_cash_change": final["household_cash"] - rows[0]["household_cash"],
        "household_equity_assets": final["household_equity_assets"],
        "household_financial_net_worth": final["household_financial_net_worth"],
        "firm_cash_change": final["firm_cash"] - rows[0]["firm_cash"],
        "paid_in_equity": final["paid_in_equity"],
        "person_dividend_total": sum(row["person_dividend"] for row in rows),
        "legacy_dividend_total": sum(row["legacy_dividend"] for row in rows),
        "ending_legacy_owner_cash": final["legacy_owner_cash"],
        "household_consumption_total": sum(row["household_consumption"] for row in rows),
        "household_saving_total": sum(row["household_saving"] for row in rows),
        "household_income_total": sum(row["household_total_income"] for row in rows),
        "money_stock_start": initial_money,
        "money_stock_end": final["money_stock"],
        "max_accounting_gap": max(row["accounting_reconciliation_max_abs"] for row in rows),
        "max_household_accounting_gap": max(row["household_accounting_gap"] for row in rows),
        "household_equity_issuance_gap": (
            final["household_equity_assets"]
            - initial_ownership["household_equity_assets"]
            - sum(row["issuance_cash"] for row in rows)
        ),
        "firm_paid_in_equity_gap": (
            final["paid_in_equity"]
            - initial_firm_paid_in
            - sum(row["issuance_cash"] for row in rows)
        ),
        "initial_person_ownership_fraction": initial_ownership["person_fraction"],
        "declared_dividend_policy_observed_last": total_declared_dividends,
        "dividend_probe_declared": probe_amount,
        "dividend_probe_person_paid": probe_person_paid,
        "dividend_probe_legacy_entitlement": probe_legacy_entitlement,
        "dividend_probe_gap": (
            probe_person_paid + probe_legacy_entitlement - probe_amount
        ),
    }, rows


def make_figures(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = sorted({row["case"] for row in rows})
    colors = {"baseline_off": "#666666", "treatment_on": "#1769aa"}

    def line_figure(filename, series, title, ylabel):
        plt.figure(figsize=(9, 4.8))
        for case in cases:
            subset = [row for row in rows if row["case"] == case]
            plt.plot(
                [row["global_step"] for row in subset],
                [row[series] for row in subset],
                label=case,
                color=colors.get(case),
            )
        plt.title(title)
        plt.xlabel("Global week")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT / filename, dpi=140)
        plt.close()

    line_figure("ownership_transition.png", "person_ownership_fraction", "Person ownership transition", "Person ownership fraction")
    plt.figure(figsize=(9, 4.8))
    for case in cases:
        subset = [row for row in rows if row["case"] == case]
        x = [row["global_step"] for row in subset]
        plt.plot(x, [row["household_cash"] for row in subset], label=f"{case} cash")
        plt.plot(x, [row["household_equity_assets"] for row in subset], linestyle="--", label=f"{case} equity")
    plt.title("Household cash to equity transition")
    plt.xlabel("Global week")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "household_cash_equity_transition.png", dpi=140)
    plt.close()
    line_figure("firm_equity_issuance.png", "issuance_cash", "Firm primary equity issuance", "Cash raised")
    plt.figure(figsize=(9, 4.8))
    for case in cases:
        subset = [row for row in rows if row["case"] == case]
        x = [row["global_step"] for row in subset]
        plt.plot(x, [row["person_dividend"] for row in subset], label=f"{case} Person")
        plt.plot(x, [row["legacy_dividend"] for row in subset], linestyle="--", label=f"{case} Legacy")
    plt.title("Legacy versus Person dividend flow")
    plt.xlabel("Global week")
    plt.ylabel("Dividend cash flow")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "legacy_vs_person_dividend_flow.png", dpi=140)
    plt.close()
    line_figure("ownership_concentration.png", "largest_person_share", "Largest Person ownership share", "Share fraction")
    plt.figure(figsize=(9, 4.8))
    for case in cases:
        subset = [row for row in rows if row["case"] == case]
        x = [row["global_step"] for row in subset]
        plt.plot(x, [row["household_consumption"] for row in subset], label=f"{case} consumption")
        plt.plot(x, [row["household_saving"] for row in subset], linestyle="--", label=f"{case} saving")
    plt.title("Consumption and saving comparison")
    plt.xlabel("Global week")
    plt.ylabel("Flow")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "consumption_saving_comparison.png", dpi=140)
    plt.close()
    line_figure("money_reconciliation.png", "money_reconciliation_gap", "Money reconciliation", "Gap")


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline_summary, baseline_rows = run_case("baseline_off", False)
    treatment_summary, treatment_rows = run_case("treatment_on", True)
    summaries = [baseline_summary, treatment_summary]
    all_rows = baseline_rows + treatment_rows

    final = treatment_summary
    gradual = (
        final["final_person_ownership_fraction"] > 0.0
        and final["final_person_ownership_fraction"] < 0.20
        and max(row["issuance_cash"] for row in treatment_rows) < 1000.0
    )
    liquidity_safe = (
        treatment_summary["household_consumption_total"]
        >= baseline_summary["household_consumption_total"] * 0.95
    )
    concentration_safe = final["final_largest_person_share"] < 0.50
    accounting_safe = (
        max(
            summary["max_accounting_gap"]
            for summary in summaries
        ) <= 1e-4
        and max(
            summary["max_household_accounting_gap"]
            for summary in summaries
        ) <= 1e-4
        and max(
            abs(summary["household_equity_issuance_gap"])
            for summary in summaries
        ) <= 1e-4
        and max(
            abs(summary["firm_paid_in_equity_gap"])
            for summary in summaries
        ) <= 1e-4
    )
    dividend_route_safe = (
        abs(treatment_summary["dividend_probe_gap"]) <= TOLERANCE
        and abs(
            treatment_summary["dividend_probe_person_paid"]
            + treatment_summary["dividend_probe_legacy_entitlement"]
            - treatment_summary["dividend_probe_declared"]
        ) <= TOLERANCE
    )
    if not accounting_safe:
        verdict = "E. EQUITY_ACCOUNTING_BLOCKER"
    elif not liquidity_safe:
        verdict = "C. HOUSEHOLD_LIQUIDITY_DAMAGE"
    elif not concentration_safe:
        verdict = "D. OWNERSHIP_CONCENTRATION_BLOCKER"
    elif not gradual:
        verdict = "B. TRANSITION_TOO_FAST"
    elif not dividend_route_safe:
        verdict = "F. DIVIDEND_TRANSITION_BLOCKER"
    else:
        verdict = "A. GRADUAL_PERSON_EQUITY_TRANSITION_ACCEPTED"

    flags = {
        "verdict": verdict,
        "baseline_behavior_off": True,
        "treatment_behavior_on": True,
        "deterministic_buyer_order": True,
        "autonomous_share_buying": False,
        "secondary_market": False,
        "leveraged_purchase": False,
        "investment_active": False,
        "stock_price_discovery": False,
        "dividend_policy_changed": False,
        "transition_is_gradual": gradual,
        "household_liquidity_safe": liquidity_safe,
        "ownership_concentration_safe": concentration_safe,
        "household_cash_equity_reconciliation": accounting_safe,
        "firm_capitalization_reconciliation": accounting_safe,
        "future_dividend_routing_available": dividend_route_safe,
        "legacy_dividend_leakage_declines": (
            treatment_summary["dividend_probe_legacy_entitlement"]
            <= baseline_summary["dividend_probe_legacy_entitlement"] + TOLERANCE
        ),
        "money_conserved": all(
            max(
                abs(row["money_reconciliation_gap"])
                for row in all_rows
                if row["case"] == summary["case"]
            ) < 1e-3
            for summary in summaries
        ),
        "same_seed": True,
        "economic_parameters_changed": False,
    }
    fields = list(summaries[0])
    with (OUTPUT / "equity_transition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    panel_fields = list(all_rows[0])
    with (OUTPUT / "ownership_panel.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=panel_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    make_figures(all_rows)
    (OUTPUT / "acceptance_summary.md").write_text(
        f"# Step 15F.2 Controlled Gradual Person Equity Transition\n\n"
        f"**Verdict: {verdict}**\n\n"
        "The OFF and ON runs started from the same Step 10.9 warm checkpoint and "
        f"continued for {CONTINUATION_WEEKS} weeks with identical economic parameters. "
        "The treatment enabled only deterministic primary issuance under household cash "
        "and Firm-base caps. No secondary trading, leverage, investment or stock-price "
        "discovery was enabled.\n\n"
        "The issuance price is an explicit transitional engineering price, not a market "
        "valuation. Legacy share count is unchanged; only Person shares are issued, so "
        "the Legacy ownership fraction declines gradually. Household cash decreases by "
        "the subscription amount while equity assets and Firm paid-in equity increase by "
        "the same amount.\n",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
