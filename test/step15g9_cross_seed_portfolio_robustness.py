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


OUTPUT = ROOT / "test/output/step15G9_cross_seed_portfolio_robustness"
SEEDS = (42, 7, 21, 84, 123)
RATES = {"R1": 0.01, "R2": 0.02, "R5": 0.05}
WEEKS = 520
TOLERANCE = 1e-6
GAP_TOLERANCE = 1e-4


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


def build_world(seed, rate):
    world = World(initial_population=5000, seed=seed, diagnostics_mode="full")
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


def run_case(seed, rate_name, rate):
    world = build_world(seed, rate)
    cumulative = {
        "executed_buyers": 0,
        "cash_transferred": 0.0,
        "person_dividends": 0.0,
        "legacy_dividends": 0.0,
    }
    max_negative_households = 0
    max_gaps = {
        "money_reconciliation_gap": 0.0,
        "accounting_cash_flow_gap": 0.0,
        "accounting_balance_sheet_gap": 0.0,
        "inventory_bridge_gap": 0.0,
        "equity_bridge_gap": 0.0,
        "household_wealth_bridge_gap": 0.0,
    }
    cumulative_consumption = 0.0
    cumulative_saving = 0.0
    transfer_checks = []
    runtime_failure = ""
    steps_completed = 0

    for _ in range(WEEKS):
        try:
            world.step()
        except Exception as error:
            runtime_failure = (
                f"step={world.current_step_index}; "
                f"{type(error).__name__}: {error}"
            )
            break
        steps_completed += 1

        result = getattr(world, "last_autonomous_secondary_purchase_result", None)
        if result is not None:
            cumulative["executed_buyers"] += result.buyer_count
            cumulative["cash_transferred"] += result.total_payment
            for transfer in result.transfer_results:
                transfer_checks.append({
                    "firm_cash_gap": abs(
                        transfer.firm_cash_after - transfer.firm_cash_before
                    ),
                    "paid_in_equity_gap": abs(
                        transfer.paid_in_equity_after
                        - transfer.paid_in_equity_before
                    ),
                    "reconciliation_gap": abs(transfer.reconciliation_gap),
                })

        cumulative["person_dividends"] += sum(
            num(getattr(firm, "person_dividend_paid", 0.0))
            for firm in world.firms
        )
        cumulative["legacy_dividends"] += sum(
            num(getattr(firm, "legacy_dividend_entitlement", 0.0))
            for firm in world.firms
        )
        cumulative_consumption += num(world.consumption_history[-1])
        cumulative_saving += num(world.saving_history[-1])

        max_negative_households = max(
            max_negative_households,
            sum(num(h.wealth) < -TOLERANCE for h in world.households),
        )
        diagnostics = world.diagnostics_rows[-1]
        gaps = accounting_gaps(world)
        max_gaps["money_reconciliation_gap"] = max(
            max_gaps["money_reconciliation_gap"],
            abs(num(diagnostics.get("monetary_accounting_gap"))),
        )
        for key in (
            "accounting_cash_flow_gap",
            "accounting_balance_sheet_gap",
            "inventory_bridge_gap",
            "equity_bridge_gap",
            "household_wealth_bridge_gap",
        ):
            source = {
                "accounting_cash_flow_gap": "cash_flow_gap",
                "accounting_balance_sheet_gap": "balance_sheet_gap",
                "inventory_bridge_gap": "inventory_bridge_gap",
                "equity_bridge_gap": "equity_bridge_gap",
                "household_wealth_bridge_gap": "household_wealth_bridge_gap",
            }[key]
            max_gaps[key] = max(max_gaps[key], gaps[source])

    if not world.diagnostics_rows:
        raise RuntimeError(f"{seed}/{rate_name} produced no diagnostics")

    firm = world.firms[0]
    table = firm.cap_table
    final_household_cash = sum(num(h.wealth) for h in world.households)
    final_equity_assets = sum(
        num(getattr(h, "equity_asset_value", 0.0)) for h in world.households
    )
    estate_accounts = list(getattr(world, "estate_accounts", {}).values())
    estate_dividends = sum(
        num(getattr(account, "estate_dividend_income", 0.0))
        for account in estate_accounts
    )
    events = len(getattr(world, "shareholder_estate_events", []))
    final = {
        "seed": seed,
        "rate": rate_name,
        "purchase_rate": rate,
        "buyer_cap": "unlimited",
        "weeks": WEEKS,
        "completed_weeks": steps_completed,
        "runtime_complete": not bool(runtime_failure)
        and steps_completed == WEEKS,
        "runtime_failure": runtime_failure,
        "final_person_ownership": table.ownership_fraction("person"),
        "final_legacy_ownership": table.ownership_fraction("legacy"),
        "executed_buyers": cumulative["executed_buyers"],
        "cumulative_cash_transferred": cumulative["cash_transferred"],
        "final_household_cash": final_household_cash,
        "final_household_equity_assets": final_equity_assets,
        "final_household_financial_net_worth": (
            final_household_cash + final_equity_assets
        ),
        "cumulative_consumption": cumulative_consumption,
        "cumulative_saving": cumulative_saving,
        "person_dividends": cumulative["person_dividends"],
        "legacy_dividends": cumulative["legacy_dividends"],
        "estate_dividends": estate_dividends,
        "largest_person_share": largest_person_share(world),
        "person_equity_gini": person_gini(world),
        "inheritance_events": events,
        "open_estates": sum(
            getattr(account, "status", "open") == "open"
            for account in estate_accounts
        ),
        "estate_shares": sum(
            sum(getattr(account, "shares_by_firm", {}).values())
            for account in estate_accounts
        ),
        "estate_cash": sum(
            num(getattr(account, "cash", 0.0)) for account in estate_accounts
        ),
        "negative_household_count_max": max_negative_households,
        "money_stock_final": money_stock(world),
        "firm_cash_final": sum(num(firm.cash) for firm in world.firms),
        "paid_in_equity_final": sum(
            num(getattr(firm, "paid_in_equity", 0.0)) for firm in world.firms
        ),
        "max_money_reconciliation_gap": max_gaps["money_reconciliation_gap"],
        "max_accounting_cash_flow_gap": max_gaps["accounting_cash_flow_gap"],
        "max_accounting_balance_sheet_gap": max_gaps[
            "accounting_balance_sheet_gap"
        ],
        "max_inventory_bridge_gap": max_gaps["inventory_bridge_gap"],
        "max_equity_bridge_gap": max_gaps["equity_bridge_gap"],
        "max_household_wealth_bridge_gap": max_gaps[
            "household_wealth_bridge_gap"
        ],
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
    return final


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def sd(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for rate_name, rate in RATES.items():
            print(f"Running seed={seed}, {rate_name} ({rate:.2%}) ...", flush=True)
            rows.append(run_case(seed, rate_name, rate))

    metrics = [
        "final_person_ownership",
        "cumulative_cash_transferred",
        "final_household_cash",
        "cumulative_consumption",
        "cumulative_saving",
        "person_dividends",
        "legacy_dividends",
        "largest_person_share",
        "person_equity_gini",
        "inheritance_events",
        "negative_household_count_max",
        "max_money_reconciliation_gap",
        "max_accounting_cash_flow_gap",
        "max_accounting_balance_sheet_gap",
        "max_inventory_bridge_gap",
        "max_equity_bridge_gap",
        "max_household_wealth_bridge_gap",
    ]
    summary = []
    for rate_name in RATES:
        subset = [row for row in rows if row["rate"] == rate_name]
        for metric in metrics:
            values = [float(row[metric]) for row in subset]
            summary.append({
                "rate": rate_name,
                "purchase_rate": RATES[rate_name],
                "metric": metric,
                "seed_count": len(values),
                "mean": mean(values),
                "sd": sd(values),
                "min": min(values),
                "max": max(values),
            })

    by_seed = {seed: {row["rate"]: row for row in rows if row["seed"] == seed} for seed in SEEDS}
    r1_rows = [row for row in rows if row["rate"] == "R1"]
    all_runtime = all(row["runtime_complete"] for row in rows)
    all_r1_safe = all(
        row["negative_household_count_max"] == 0
        and row["max_money_reconciliation_gap"] <= GAP_TOLERANCE
        and row["max_accounting_cash_flow_gap"] <= GAP_TOLERANCE
        and row["max_accounting_balance_sheet_gap"] <= GAP_TOLERANCE
        and row["max_inventory_bridge_gap"] <= GAP_TOLERANCE
        and row["max_equity_bridge_gap"] <= GAP_TOLERANCE
        and row["max_household_wealth_bridge_gap"] <= GAP_TOLERANCE
        for row in r1_rows
    )
    all_rates_safe = all(
        row["negative_household_count_max"] == 0
        and row["max_money_reconciliation_gap"] <= GAP_TOLERANCE
        and row["max_accounting_cash_flow_gap"] <= GAP_TOLERANCE
        and row["max_accounting_balance_sheet_gap"] <= GAP_TOLERANCE
        and row["max_inventory_bridge_gap"] <= GAP_TOLERANCE
        and row["max_equity_bridge_gap"] <= GAP_TOLERANCE
        and row["max_household_wealth_bridge_gap"] <= GAP_TOLERANCE
        for row in rows
    )
    r1_material = all(row["final_person_ownership"] >= 0.01 for row in r1_rows)
    no_concentration_failure = all(
        row["largest_person_share"] < 0.20 for row in rows
    )
    no_consumption_collapse = all(
        by_seed[seed][rate]["cumulative_consumption"]
        >= by_seed[seed]["R1"]["cumulative_consumption"] * 0.95
        for seed in SEEDS
        for rate in RATES
    )
    monotonic_count = sum(
        by_seed[seed]["R1"]["final_person_ownership"]
        <= by_seed[seed]["R2"]["final_person_ownership"] + 1e-12
        and by_seed[seed]["R2"]["final_person_ownership"]
        <= by_seed[seed]["R5"]["final_person_ownership"] + 1e-12
        for seed in SEEDS
    )
    monotonic_share = monotonic_count / len(SEEDS)
    r1_ownership_values = [row["final_person_ownership"] for row in r1_rows]
    r1_cv = sd(r1_ownership_values) / max(mean(r1_ownership_values), 1e-12)
    material_seed_sensitivity = r1_cv > 0.50
    higher_rate_saturation = (
        mean([row["final_person_ownership"] for row in rows if row["rate"] == "R5"])
        / max(
            mean([row["final_person_ownership"] for row in rows if row["rate"] == "R2"]),
            1e-12,
        )
        < 1.50
    )
    inheritance_stable = all(
        row["inheritance_events"] > 0 and row["open_estates"] >= 0
        for row in rows
    )

    if not all_runtime or not all_rates_safe:
        verdict = "D. LIQUIDITY_OR_CONSUMPTION_FAILURE_IN_SOME_SEEDS"
    elif not no_concentration_failure:
        verdict = "E. OWNERSHIP_CONCENTRATION_FAILURE_IN_SOME_SEEDS"
    elif material_seed_sensitivity:
        verdict = "C. MATERIAL_SEED_SENSITIVITY_FOUND"
    elif not all_r1_safe or not r1_material or not inheritance_stable:
        verdict = "F. OTHER_BLOCKER"
    elif no_consumption_collapse and monotonic_share >= 0.80:
        verdict = "A. ONE_PERCENT_RULE_ROBUST_ACROSS_SEEDS"
    else:
        verdict = "B. HIGHER_RATES_ROBUST_BUT_CALIBRATION_STILL_REQUIRED"

    flags = {
        "verdict": verdict,
        "seeds": list(SEEDS),
        "rates": list(RATES),
        "weeks": WEEKS,
        "population": 5000,
        "buyer_cap": "unlimited",
        "lagged_book_price": True,
        "review_cadence_weeks": 52,
        "liquidity_buffers_unchanged": True,
        "concentration_guards_unchanged": True,
        "inheritance_active": True,
        "primary_issuance": False,
        "leverage": False,
        "investment_active": False,
        "dividend_policy_changed": False,
        "Step13_changed": False,
        "new_rng_draws": 0,
        "all_runs_completed": all_runtime,
        "R1_all_seeds_liquidity_safe": all_r1_safe,
        "R1_all_seeds_materially_active": r1_material,
        "all_rates_no_consumption_collapse": no_consumption_collapse,
        "ownership_monotonic_seed_share": monotonic_share,
        "higher_rate_saturation_present": higher_rate_saturation,
        "inheritance_stable": inheritance_stable,
        "concentration_stable": no_concentration_failure,
        "R1_ownership_cv": r1_cv,
        "material_seed_sensitivity_threshold": 0.50,
        "runs": len(rows),
    }

    write_csv(OUTPUT / "seed_rate_results.csv", rows)
    write_csv(OUTPUT / "cross_seed_summary.csv", summary)

    lines = [
        "# Step 15G.9 Cross-Seed Portfolio-Rate Robustness",
        "",
        f"## Verdict\n\n**{verdict}**",
        "",
        f"使用 seeds `{', '.join(map(str, SEEDS))}`，每个 seed 运行 R1/R2/R5 各 520 周。所有运行均采用 unlimited buyers、lagged book-equity price、52 周 review 和已接受的继承架构。",
        "",
        "| rate | ownership mean | ownership sd | ownership min | ownership max | consumption mean | cash mean | inheritance mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rate_name in RATES:
        subset = [row for row in rows if row["rate"] == rate_name]
        ownership = [row["final_person_ownership"] for row in subset]
        consumption = [row["cumulative_consumption"] for row in subset]
        cash = [row["final_household_cash"] for row in subset]
        inheritance = [row["inheritance_events"] for row in subset]
        lines.append(
            f"| {rate_name} | {mean(ownership):.6%} | {sd(ownership):.6%} | {min(ownership):.6%} | {max(ownership):.6%} | {mean(consumption):.2f} | {mean(cash):.2f} | {mean(inheritance):.1f} |"
        )
    lines.extend([
        "",
        f"R1 ownership coefficient of variation = `{r1_cv:.4f}`; monotonic ownership ordering R1 ≤ R2 ≤ R5 holds for `{monotonic_count}/{len(SEEDS)}` seeds. Higher-rate saturation flag = `{higher_rate_saturation}`.",
        "",
        "本实验只判断跨 seed 的稳健性，不根据平均所有权比例选择永久购买率。",
    ])
    (OUTPUT / "acceptance_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
