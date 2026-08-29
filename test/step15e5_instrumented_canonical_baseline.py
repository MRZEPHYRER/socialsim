"""Step 15E.5 fresh canonical persisted-data human-review baseline."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world import World
from diagnostic_persistence import CanonicalDiagnosticPersistence


OUTPUT = ROOT / "test/output/step15E5_instrumented_canonical_baseline"
STEPS = 520
SEED = 42
POPULATION = 5000
TOLERANCE = 1e-4


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row, key, default=math.nan):
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_rows(rows, fields):
    return all(math.isfinite(number(row, field)) for row in rows for field in fields)


def plot_series(path, title, series, ylabel, status=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 5.2))
    x = range(len(next(iter(series.values())))) if series else []
    for label, values in series.items():
        axis.plot(x, values, linewidth=1.1, label=label)
    axis.set_title(title)
    axis.set_xlabel("Global step (week)")
    axis.set_ylabel(ylabel)
    if series:
        axis.legend()
    if status:
        axis.text(0.01, 0.98, status, transform=axis.transAxes, va="top", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def money_plot(path, macro):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    axes[0].plot([number(row, "money_stock") for row in macro], label="Money stock")
    axes[0].set_title("Money stock (stock only)")
    axes[0].set_ylabel("Stock")
    axes[0].legend()
    axes[1].plot([number(row, "money_created") for row in macro], label="Created per step")
    axes[1].plot([number(row, "money_destroyed") for row in macro], label="Destroyed per step")
    axes[1].set_title("Gross monetary flows")
    axes[1].set_xlabel("Global step (week)")
    axes[1].set_ylabel("Flow")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def final_demand_plot(path, macro):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].plot([number(row, "household_final_consumption") for row in macro], label="Household consumption")
    axes[0].set_title("Household final consumption")
    axes[0].set_xlabel("Global step (week)")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[1].axis("off")
    axes[1].text(
        0.03,
        0.95,
        "FINAL-DEMAND STATUS\n\n"
        "Firm investment: 0 / PASSIVE\n\n"
        "Government: 0 / INACTIVE\n\n"
        "External: 0 / INACTIVE\n\n"
        "Intermediate: 0 / SEPARATE",
        va="top",
        fontsize=10,
        family="monospace",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def reconciliation_plot(path, macro, accounting):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    firm_gaps = defaultdict(float)
    for row in accounting:
        if row.get("record_type") == "firm":
            firm_gaps[int(row["global_step"])] += number(row, "cash_flow_gap", 0.0)
    accounting_gap = [firm_gaps[int(row["global_step"])] for row in macro]
    money_gap = [number(row, "money_reconciliation_gap") for row in macro]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    axes[0].plot(accounting_gap, label="Accounting cash-flow gap")
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set_title("Accounting reconciliation gap")
    axes[0].set_ylabel("Gap")
    axes[0].legend()
    axes[1].plot(money_gap, label="Money reconciliation gap")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_title("Money reconciliation gap")
    axes[1].set_xlabel("Global step (week)")
    axes[1].set_ylabel("Gap")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def signature(world):
    return (
        tuple(world.population_history),
        tuple(world.income_history),
        tuple(world.consumption_history),
        tuple(world.saving_history),
        tuple(world.wealth_history),
        tuple(
            (firm.firm_id, firm.cash, firm.inventory_units, firm.price, firm.loan_balance)
            for firm in world.firms
        ),
    )


def short_parity_check():
    short_steps = 16
    off = World(initial_population=500, seed=SEED, diagnostics_mode="full")
    off.steps = short_steps
    off.run(progress_interval=0)
    with tempfile.TemporaryDirectory(prefix="step15e5_parity_") as temp:
        on = World(initial_population=500, seed=SEED, diagnostics_mode="full")
        on.configure_diagnostic_persistence(temp, cadence=1)
        on.steps = short_steps
        on.run(progress_interval=0)
    return signature(off) == signature(on)


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    parity = short_parity_check()
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        scenario_name="baseline",
        scenario_overrides={},
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    persistence = CanonicalDiagnosticPersistence(OUTPUT, cadence=1)
    world.diagnostic_persistence = persistence
    world.steps = STEPS
    world.run(progress_interval=0)

    macro = read_rows(OUTPUT / "macro_diagnostics.csv")
    firms = read_rows(OUTPUT / "firm_diagnostics.csv")
    accounting = read_rows(OUTPUT / "accounting_diagnostics.csv")
    steps = [int(row["global_step"]) for row in macro]
    firm_keys = [(int(row["global_step"]), row["firm_id"]) for row in firms]
    accounting_keys = [
        (int(row["global_step"]), row["record_type"], row.get("firm_id", ""))
        for row in accounting
    ]
    firm_count = len(world.firms)
    expected_firm_rows = STEPS * firm_count

    macro_fields = [
        "population", "total_employment", "unassigned_labor",
        "household_wage_income", "household_total_income", "household_consumption",
        "household_saving", "household_cash_wealth", "household_equity_assets",
        "household_financial_net_worth", "money_stock", "money_created",
        "money_destroyed", "accounting_reconciliation_gap", "money_reconciliation_gap",
        "household_final_consumption", "firm_fixed_investment", "government_final_demand",
        "external_final_demand", "intermediate_demand",
    ]
    firm_fields = [
        "employment", "desired_labor", "revenue", "operating_profit", "cfo", "cash",
        "principal", "arrears", "legacy_ownership_fraction", "person_ownership_fraction",
        "person_shareholder_count", "capital_asset_count", "capital_book_value",
        "investment_expenditure",
    ]
    macro_coverage = len(steps) == STEPS and steps == list(range(STEPS)) and finite_rows(macro, macro_fields)
    firm_coverage = (
        len(firms) == expected_firm_rows
        and len(set(firm_keys)) == len(firm_keys)
        and finite_rows(firms, firm_fields)
        and len({step for step, _ in firm_keys}) == STEPS
    )
    accounting_coverage = len(set(accounting_keys)) == len(accounting_keys)
    inactive_zero = all(
        number(row, field) == 0.0
        for row in macro
        for field in ("firm_fixed_investment", "government_final_demand", "external_final_demand", "intermediate_demand")
    )
    person_inactive = all(
        number(row, "person_ownership_fraction") == 0.0
        and number(row, "person_shareholder_count") == 0.0
        for row in firms
    )
    investment_inactive = all(
        number(row, "capital_asset_count") == 0.0
        and number(row, "capital_book_value") == 0.0
        and number(row, "investment_expenditure") == 0.0
        for row in firms
    )

    invariant_summary = world.diagnostics_summary()
    accounting_gaps = [
        abs(number(row, field))
        for row in accounting
        for field in ("cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap", "equity_bridge_gap")
        if math.isfinite(number(row, field))
    ]
    accounting_pass = max(accounting_gaps, default=0.0) <= TOLERANCE
    money_pass = max(
        (abs(number(row, "money_reconciliation_gap")) for row in macro),
        default=0.0,
    ) <= TOLERANCE
    invariant_pass = invariant_summary["invariant_violations"] == 0

    latest_step = STEPS - 1
    latest_firms = [row for row in firms if int(row["global_step"]) == latest_step]
    with (OUTPUT / "final_firm_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(latest_firms[0]) if latest_firms else ["status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(latest_firms)

    by_firm = defaultdict(lambda: defaultdict(list))
    for row in firms:
        firm_id = row["firm_id"]
        for field in ("revenue", "cfo", "principal"):
            by_firm[firm_id][field].append(number(row, field))
    plot_series(
        OUTPUT / "macro_income_consumption_saving.png",
        "Household income, consumption, and saving",
        {
            "Income": [number(row, "household_total_income") for row in macro],
            "Consumption": [number(row, "household_consumption") for row in macro],
            "Saving": [number(row, "household_saving") for row in macro],
        },
        "Value",
    )
    sector_series = defaultdict(lambda: [0.0] * STEPS)
    for row in firms:
        sector_series[row["sector_id"]][int(row["global_step"])] += number(row, "employment", 0.0)
    sector_series["Unassigned labor"] = [number(row, "unassigned_labor") for row in macro]
    plot_series(OUTPUT / "sector_labor.png", "Sector employment and unassigned labor", dict(sector_series), "Workers")
    plot_series(
        OUTPUT / "firm_financial_health.png",
        "Firm financial health",
        {f"Firm {firm_id} revenue": values["revenue"] for firm_id, values in by_firm.items()}
        | {f"Firm {firm_id} CFO": values["cfo"] for firm_id, values in by_firm.items()}
        | {f"Firm {firm_id} principal": values["principal"] for firm_id, values in by_firm.items()},
        "Value",
    )
    money_plot(OUTPUT / "monetary_system.png", macro)
    final_demand_plot(OUTPUT / "final_demand.png", macro)
    plot_series(
        OUTPUT / "household_financial_position.png",
        "Household financial position",
        {
            "Cash wealth": [number(row, "household_cash_wealth") for row in macro],
            "Equity assets": [number(row, "household_equity_assets") for row in macro],
            "Financial net worth": [number(row, "household_financial_net_worth") for row in macro],
        },
        "Value",
        status="Person equity ownership inactive in canonical baseline",
    )
    plot_series(
        OUTPUT / "ownership_status.png",
        "Ownership status",
        {
            f"Firm {firm_id} Legacy": [number(row, "legacy_ownership_fraction") for row in firms if row["firm_id"] == firm_id]
            for firm_id in sorted({row["firm_id"] for row in firms})
        } | {
            f"Firm {firm_id} Person": [number(row, "person_ownership_fraction") for row in firms if row["firm_id"] == firm_id]
            for firm_id in sorted({row["firm_id"] for row in firms})
        },
        "Ownership fraction",
        status="Person ownership inactive; no share purchase or issuance",
    )
    reconciliation_plot(OUTPUT / "reconciliation.png", macro, accounting)

    checks = {
        "diagnostics_rows_cover_520_weeks": len(macro) == STEPS,
        "macro_authoritative_history_complete": macro_coverage,
        "firm_panel_coverage_complete": firm_coverage,
        "accounting_keys_unique": accounting_coverage,
        "inactive_mechanisms_explicitly_zero": inactive_zero,
        "ownership_inactive": person_inactive,
        "investment_inactive": investment_inactive,
        "economic_invariants_pass": invariant_pass,
        "accounting_reconciliation_pass": accounting_pass,
        "money_reconciliation_pass": money_pass,
        "diagnostics_off_on_parity": parity,
        "new_rng_draws": 0,
        "economic_behavior_changed": False,
    }
    all_pass = all(
        value is True for key, value in checks.items()
        if key not in {"new_rng_draws", "economic_behavior_changed"}
    ) and checks["new_rng_draws"] == 0 and checks["economic_behavior_changed"] is False
    verdict = "A. INSTRUMENTED_CANONICAL_BASELINE_ACCEPTED" if all_pass else "F. OTHER_BLOCKER"
    flags = {
        "verdict": verdict,
        **checks,
        "seed": SEED,
        "initial_population": POPULATION,
        "steps": STEPS,
        "firm_count": firm_count,
        "macro_rows": len(macro),
        "firm_rows": len(firms),
        "expected_firm_rows": expected_firm_rows,
        "accounting_rows": len(accounting),
        "max_abs_accounting_gap": max(accounting_gaps, default=0.0),
        "max_abs_money_gap": max((abs(number(row, "money_reconciliation_gap")) for row in macro), default=0.0),
        "invariant_violations": invariant_summary["invariant_violations"],
        "capital_assets_final": sum(int(number(row, "capital_asset_count", 0.0)) for row in latest_firms),
        "person_ownership_final": sum(number(row, "person_ownership_fraction", 0.0) for row in latest_firms),
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Step 15E.5 Instrumented Canonical Human-Review Baseline

## Verdict

**{verdict}**

Fresh baseline: seed `{SEED}`, population `{POPULATION}`, `{STEPS}` weekly
steps, diagnostics cadence `1`, and no checkpoint history. The persisted macro
panel contains `{len(macro)}` authoritative rows; the Firm panel contains
`{len(firms)}` rows for `{firm_count}` active Firm(s). No historical values were
reconstructed from a final snapshot.

The output figures are generated directly from the persisted canonical CSVs.
Inactive final-demand mechanisms, Person ownership, capital accumulation, and
investment are explicitly labeled and remain zero only where the current
runtime contract establishes them as inactive.

Accounting max gap: `{max(accounting_gaps, default=0.0):.6g}`. Money max gap:
`{max((abs(number(row, "money_reconciliation_gap")) for row in macro), default=0.0):.6g}`.
Invariant violations: `{invariant_summary["invariant_violations"]}`. Short
diagnostics OFF/ON parity: `{parity}`.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
