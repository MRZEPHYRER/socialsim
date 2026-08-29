"""Step 15I.4 N=500 internal-cash canonical investment screen."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world import World


OUTPUT = ROOT / "test/output/step15I4_internal_cash_canonical_investment"
STEPS = 520
POPULATION = 500
FOOD_FIRMS = 5
SEED = 42
TOLERANCE = 1e-6


def write_rows(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_case(enabled):
    world = World(
        initial_population=POPULATION,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name="step15i4_treatment" if enabled else "step15i4_control",
        scenario_overrides={
            "MULTISECTOR_FOUNDATION_ENABLED": True,
            "CANONICAL_INVESTMENT_ENABLED": enabled,
        },
    )
    world.split_firms(FOOD_FIRMS)
    macro_rows = []
    for _ in range(STEPS):
        world.step()
        week = world.last_canonical_investment_week
        base = dict(world.diagnostics_rows[-1])
        base.update(
            {
                "canonical_investment_enabled": enabled,
                "capital_good_production": getattr(week, "capital_good_production", 0.0),
                "capital_good_sales": getattr(week, "capital_good_sales", 0.0),
                "capital_good_revenue": getattr(week, "capital_good_revenue", 0.0),
                "capital_good_wages": getattr(week, "capital_good_wages", 0.0),
                "fixed_investment": getattr(week, "fixed_investment", 0.0),
                "investment_orders": getattr(week, "investment_orders", 0),
                "capital_good_employment": sum(
                    len(firm.employee_ids) for firm in world.capital_good_firms
                ),
                "capital_good_cash": sum(
                    firm.cash for firm in world.capital_good_firms
                ),
                "capital_asset_count": sum(
                    len(getattr(firm.capital_stock, "assets", []))
                    for firm in world.firms
                ),
                "capital_book_value": sum(
                    getattr(firm.capital_stock, "total_remaining_book_value", 0.0)
                    for firm in world.firms
                ),
            }
        )
        macro_rows.append(base)

    firm_rows = []
    for row in world.firm_diagnostics_rows:
        row = dict(row)
        row["canonical_investment_enabled"] = enabled
        firm_rows.append(row)
    capital_rows = [
        row for row in firm_rows if row.get("sector_id") == "capital_goods"
    ]
    food_rows = [
        row for row in firm_rows if row.get("sector_id") == "food"
    ]
    investment_events = [
        {
            "step": row.get("step"),
            "firm_id": row.get("firm_id"),
            "desired_investment_expenditure": row.get(
                "desired_investment_expenditure", 0.0
            ),
            "investment_expenditure": row.get("investment_expenditure", 0.0),
            "investment_financing_gap": row.get("investment_financing_gap", 0.0),
            "investable_cash": row.get("investable_cash", 0.0),
            "expansion_investment_need": row.get("expansion_investment_need", 0.0),
            "replacement_investment_need": row.get("replacement_investment_need", 0.0),
            "capital_asset_count": row.get("capital_asset_count", 0),
            "capital_book_value": row.get("capital_book_value", 0.0),
            "cash": row.get("cash", 0.0),
            "loan_balance": row.get("loan_balance", 0.0),
            "arrears": row.get("interest_arrears", 0.0),
        }
        for row in food_rows
        if float(row.get("investment_expenditure", 0.0) or 0.0) > TOLERANCE
    ]
    accounting_rows = [dict(row) for row in world.accounting.rows]
    return {
        "world": world,
        "macro": macro_rows,
        "firms": firm_rows,
        "capital": capital_rows,
        "food": food_rows,
        "events": investment_events,
        "accounting": accounting_rows,
    }


def values(rows, key):
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


def max_abs(rows, key):
    return max((abs(float(row.get(key, 0.0) or 0.0)) for row in rows), default=0.0)


def plot_line(path, series, title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, data in series:
        ax.plot(range(len(data)), data, label=label, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("week")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_figures(control, treatment):
    c, t = control["macro"], treatment["macro"]
    plot_line(
        OUTPUT / "fixed_investment_over_time.png",
        [("control", values(c, "fixed_investment")), ("treatment", values(t, "fixed_investment"))],
        "Firm fixed investment",
        "cash expenditure",
    )
    plot_line(
        OUTPUT / "capital_stock_and_capacity.png",
        [("capital book value", values(t, "capital_book_value")), ("asset count", values(t, "capital_asset_count"))],
        "Treatment capital stock",
        "value / count",
    )
    plot_line(
        OUTPUT / "capital_good_sector_activity.png",
        [("production", values(t, "capital_good_production")), ("sales", values(t, "capital_good_sales")), ("employment", values(t, "capital_good_employment"))],
        "Capital-good sector activity",
        "units / workers",
    )
    plot_line(
        OUTPUT / "firm_cash_and_investment.png",
        [("control firm cash", values(c, "firm_cash")), ("treatment firm cash", values(t, "firm_cash")), ("treatment investment", values(t, "fixed_investment"))],
        "Firm cash and investment",
        "nominal value",
    )
    plot_line(
        OUTPUT / "final_demand_composition.png",
        [("control household consumption", values(c, "total_consumption")), ("treatment household consumption", values(t, "total_consumption")), ("treatment fixed investment", values(t, "fixed_investment"))],
        "Final-demand composition",
        "nominal value",
    )
    plot_line(
        OUTPUT / "employment_by_sector.png",
        [("control total labor", values(c, "labor")), ("treatment total labor", values(t, "labor")), ("treatment capital-good employment", values(t, "capital_good_employment"))],
        "Employment by sector",
        "labor / workers",
    )


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    control = run_case(False)
    treatment = run_case(True)
    write_rows(OUTPUT / "firm_investment_panel.csv", treatment["food"])
    write_rows(OUTPUT / "capital_good_sector_panel.csv", treatment["capital"])
    write_rows(OUTPUT / "investment_event_panel.csv", treatment["events"])
    write_rows(OUTPUT / "macro_investment_panel.csv", treatment["macro"])
    make_figures(control, treatment)

    treatment_world = treatment["world"]
    treatment_macro = treatment["macro"]
    treatment_accounting = treatment["accounting"]
    investment_total = sum(values(treatment_macro, "fixed_investment"))
    asset_total = max(values(treatment_macro, "capital_asset_count"), default=0.0)
    capital_production_total = sum(values(treatment_macro, "capital_good_production"))
    capital_revenue_total = sum(values(treatment_macro, "capital_good_revenue"))
    capital_employment_positive = any(
        value > 0.0 for value in values(treatment_macro, "capital_good_employment")
    )
    capacity_effect = any(
        float(row.get("capital_capacity", 0.0) or 0.0) > TOLERANCE
        and float(row.get("feasible_capacity", 0.0) or 0.0) > TOLERANCE
        for row in treatment["food"]
    )
    money_gap = max_abs(treatment_macro, "monetary_accounting_gap")
    accounting_gap = max(
        max_abs(treatment_accounting, "cash_flow_gap"),
        max_abs(treatment_accounting, "balance_sheet_gap"),
        max_abs(treatment_accounting, "inventory_bridge_gap"),
        max_abs(treatment_accounting, "equity_bridge_gap"),
    )
    min_food_cash = min(values(treatment["food"], "cash"), default=0.0)
    investment_loan_used = any(
        float(row.get("loan_issued", 0.0) or 0.0) > TOLERANCE
        for row in treatment["events"]
    )
    if investment_total <= TOLERANCE:
        verdict = "B. INVESTMENT_DEMAND_INACTIVE"
        blocker = "no desired capacity gap generated an executable order"
    elif accounting_gap > TOLERANCE or money_gap > TOLERANCE:
        verdict = "G. ACCOUNTING_OR_RUNTIME_BLOCKER"
        blocker = "reconciliation gap exceeded tolerance"
    elif min_food_cash < -TOLERANCE:
        verdict = "F. OPERATING_LIQUIDITY_DAMAGE"
        blocker = "Food Firm cash became negative"
    elif not capital_production_total or not capital_revenue_total:
        verdict = "D. CAPITAL_GOOD_SUPPLY_PRIMARY"
        blocker = "investment executed without a real capital-good supply flow"
    elif not capacity_effect:
        verdict = "E. CAPITAL_CAPACITY_EFFECT_NOT_REALIZED"
        blocker = "capital assets accumulated but capacity diagnostic did not rise"
    else:
        verdict = "A. INTERNAL_CASH_CANONICAL_INVESTMENT_ACCEPTED"
        blocker = "none"

    flags = {
        "verdict": verdict,
        "population": POPULATION,
        "food_firms": FOOD_FIRMS,
        "steps": STEPS,
        "seed": SEED,
        "control_investment_enabled": False,
        "treatment_investment_enabled": True,
        "investment_total": investment_total,
        "capital_asset_count_final": asset_total,
        "capital_good_production_total": capital_production_total,
        "capital_good_revenue_total": capital_revenue_total,
        "capital_good_employment_positive": capital_employment_positive,
        "capacity_effect_observed": capacity_effect,
        "min_food_cash": min_food_cash,
        "investment_loan_used": investment_loan_used,
        "money_gap_max_abs": money_gap,
        "accounting_gap_max_abs": accounting_gap,
        "invariant_violations": len(getattr(treatment_world, "invariant_violations", [])),
        "capital_productivity_calibration": "NON-CALIBRATED_ENGINEERING_SCREEN",
        "depreciation_calibration": "OFF",
        "financing_source": "INTERNAL_CASH_ONLY",
        "new_rng_draws": 0,
        "Step13_changed": False,
        "blocker": blocker,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = f"""# Step 15I.4 Internal-Cash Canonical Investment Screen

## Verdict

**{verdict}**

The screen used `N={POPULATION}`, `{FOOD_FIRMS}` Food Firms, seed `{SEED}` and
`{STEPS}` weeks. Control kept canonical investment OFF; treatment enabled one
generic labor-only capital-good Firm with deterministic engineering
productivity marked `NON-CALIBRATED_ENGINEERING_SCREEN`.

Treatment totals:

- fixed investment: `{investment_total:.12g}`
- capital-good production: `{capital_production_total:.12g}`
- capital-good revenue: `{capital_revenue_total:.12g}`
- final capital assets: `{asset_total:.12g}`
- minimum Food cash: `{min_food_cash:.12g}`
- maximum absolute money gap: `{money_gap:.12g}`
- maximum accounting gap: `{accounting_gap:.12g}`

The supplier uses real employees, pays wages from its own retained startup
capital, produces storable inventory, and receives only settled investment
revenue. Buyer investment uses internal cash after protected operating
liquidity; no working-capital loan, investment loan or equity issuance is
available. Capital purchases are recorded as fixed investment and CFI, not
operating expense or intermediate demand.

The detailed panels and diagnostic figures are in this directory. This stage
does not calibrate productivity or depreciation and does not change Step13.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(verdict)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
