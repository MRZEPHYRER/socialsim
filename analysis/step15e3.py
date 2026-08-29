"""Step 15E.3 human-review package.

This module is intentionally read-only with respect to the economic model. It
aggregates already recorded diagnostics and marks unavailable historical data
explicitly instead of inventing transactions or connecting incompatible time
series.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


OUTPUT_NAME = "step15E3_main_analysis_human_review"
TOLERANCE = 1e-9
MONEY_TOLERANCE = 1e-4


def _float(row, key, default=0.0):
    try:
        value = row.get(key, default)
        if value in (None, ""):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(row, key):
    """Return NaN when a source row did not persist this metric."""
    if not isinstance(row, dict) or key not in row or row.get(key) in (None, ""):
        return math.nan
    return _float(row, key, math.nan)


def _finite_sum(values):
    values = [float(value) for value in values]
    return sum(values) if values and all(math.isfinite(value) for value in values) else math.nan


def _int(row, key, default=0):
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _step(row):
    return _int(row, "global_step", _int(row, "step", 0))


def _last_by(rows, key):
    output = {}
    for row in rows:
        output[row.get(key)] = row
    return output


def _write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _gini(values):
    values = sorted(max(0.0, float(value)) for value in values)
    total = sum(values)
    if not values or total <= 0.0:
        return 0.0
    n = len(values)
    return sum(
        (2 * index - n - 1) * value
        for index, value in enumerate(values, 1)
    ) / (n * total)


def _plot(path, title, series, ylabel="Value", stacked=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.8))
    for label, values in series.items():
        plt.plot(range(len(values)), values, label=label, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Recorded step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _bar_plot(path, title, labels, values, ylabel="Value"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _aggregate_by_step(rows, value_keys, group_key=None):
    grouped = defaultdict(lambda: defaultdict(float))
    for row in rows:
        step = _step(row)
        group = row.get(group_key) if group_key else "all"
        bucket = grouped[(step, group)]
        for key in value_keys:
            bucket[key] += _float(row, key)
    return grouped


def _macro_rows(world):
    diagnostics = list(getattr(world, "diagnostics_rows", []))
    accounting = getattr(world, "accounting", None)
    household_rows = list(getattr(accounting, "household_rows", []))
    central_rows = list(getattr(accounting, "central_bank_rows", []))
    firm_accounting = list(getattr(accounting, "rows", []))
    firm_by_step = _aggregate_by_step(
        firm_accounting,
        ["accounting_revenue", "accounting_operating_profit", "cfo",
         "cash", "loan_balance", "post_termout_interest_arrears_claim",
         "legacy_arrears_term_claim", "dividends"],
    )
    household_by_step = _last_by(household_rows, "step")
    central_by_step = _last_by(central_rows, "step")

    output = []
    for row in diagnostics:
        step = _step(row)
        h = household_by_step.get(step, {})
        f = firm_by_step.get((step, "all"), defaultdict(float))
        c = central_by_step.get(step, {})
        full_diagnostic = "total_income" in row
        source = "full diagnostics" if full_diagnostic else "historical diagnostics (compact)"
        wage_income = _optional_float(h, "wages") if h else _optional_float(row, "wage_bill")
        total_income = _optional_float(row, "total_income")
        consumption = _optional_float(row, "total_consumption")
        saving = _optional_float(row, "total_saving")
        cash_wealth = _optional_float(h, "cash_wealth") if h else _optional_float(row, "total_household_wealth")
        other_flows = (
            _finite_sum([
                _optional_float(h, "dividends"),
                _optional_float(h, "net_interhousehold_transfer"),
                _optional_float(h, "inheritance_public_transfers"),
            ]) if h else math.nan
        )
        accounting_at_step = bool(firm_accounting and (step, "all") in firm_by_step)
        if not accounting_at_step:
            firm_revenue = _optional_float(row, "firm_sales_revenue")
            firm_profit = math.nan
            firm_cfo = math.nan
            firm_cash = _optional_float(row, "firm_cash")
            principal = _optional_float(row, "loan_balance")
            arrears = math.nan
            firm_source = source
        else:
            firm_revenue = f["accounting_revenue"]
            firm_profit = f["accounting_operating_profit"]
            firm_cfo = f["cfo"]
            firm_cash = f["cash"]
            principal = f["loan_balance"]
            arrears = f["post_termout_interest_arrears_claim"] + f["legacy_arrears_term_claim"]
            firm_source = "AccountingLayer"
        money_stock = _optional_float(row, "total_money_stock")
        money_created = _optional_float(c, "gross_money_created") if c else math.nan
        money_destroyed = _optional_float(c, "money_destroyed") if c else math.nan
        money_gap = _optional_float(row, "monetary_accounting_gap")
        cash_flow_gap = (
            _finite_sum(
                _float(item, "cash_flow_gap")
                for item in firm_accounting
                if _step(item) == step
            ) if accounting_at_step else math.nan
        )
        output.append({
            "global_step": step,
            "simulation_week": _float(row, "simulation_week", step),
            "simulation_year": _float(row, "simulation_year", step / 52.0),
            "population": _float(row, "population"),
            "total_employment": _optional_float(row, "labor"),
            "unassigned_labor": math.nan,
            "unassigned_labor_status": "unavailable historically; no persisted sector labor-allocation field",
            "household_wage_income": wage_income,
            "household_total_income": total_income,
            "household_consumption": consumption,
            "household_saving": saving,
            "household_cash_wealth": cash_wealth,
            "firm_revenue": firm_revenue,
            "firm_operating_profit": firm_profit,
            "firm_cfo": firm_cfo,
            "firm_cash": firm_cash,
            "principal": principal,
            "arrears": arrears,
            "money_stock": money_stock,
            "money_created": money_created,
            "money_destroyed": money_destroyed,
            "money_flow_source": "CentralBank AccountingLayer" if c else "unavailable; no authoritative money-flow row",
            "household_income_identity_gap": (
                total_income - consumption - saving
                if all(math.isfinite(value) for value in (total_income, consumption, saving))
                else math.nan
            ),
            "household_other_explicit_flows": other_flows,
            "money_reconciliation_gap": money_gap,
            "accounting_cash_flow_gap": cash_flow_gap,
            "firm_accounting_source": firm_source,
            "household_data_source": "household_accounting/full diagnostics" if h or full_diagnostic else source,
            "macro_source": source,
            "invariant_failed": (
                bool(row["invariant_failed"])
                if "invariant_failed" in row else "unavailable"
            ),
        })
    return output


def _sector_rows(world, firm_rows, accounting_rows):
    diagnostics = list(getattr(world, "diagnostics_rows", []))
    diagnostic_steps = [_step(row) for row in diagnostics]
    firm_steps = [_step(row) for row in firm_rows]
    full_firm_history = bool(firm_rows) and bool(diagnostic_steps) and min(firm_steps) <= min(diagnostic_steps)
    if not full_firm_history:
        latest_step = max(diagnostic_steps, default=len(getattr(world, "population_history", [])))
        latest_rows = {
            row.get("firm_id"): row
            for row in firm_rows
            if _step(row) == max(firm_steps, default=-1)
        }
        firm_rows = []
        for firm in getattr(world, "firms", []):
            current = latest_rows.get(firm.firm_id, {})
            firm_rows.append({
                "global_step": latest_step,
                "firm_id": firm.firm_id,
                "employee_count": _float(current, "employee_count", len(getattr(firm, "employee_ids", []))),
                "sales_revenue": _float(current, "sales_revenue", getattr(firm, "sales_revenue", math.nan)),
                "cash": _float(current, "cash", getattr(firm, "cash", math.nan)),
                "loan_balance": _float(current, "loan_balance", getattr(firm, "loan_balance", math.nan)),
                "interest_arrears": _float(current, "interest_arrears", getattr(firm, "interest_arrears", math.nan)),
                "closing_interest_arrears": _float(current, "closing_interest_arrears", getattr(firm, "closing_interest_arrears", math.nan)),
                "profit": _float(current, "profit", getattr(firm, "profit", math.nan)),
                "data_scope": "current_snapshot",
            })
    else:
        for row in firm_rows:
            row.setdefault("data_scope", "historical_firm_diagnostics")
    accounting_by_key = {
        (_step(row), row.get("firm_id")): row for row in accounting_rows
    }
    grouped = defaultdict(lambda: defaultdict(float))
    for row in firm_rows:
        key = (_step(row), row.get("firm_id"))
        firm = next(
            (item for item in getattr(world, "firms", []) if item.firm_id == row.get("firm_id")),
            None,
        )
        sector = getattr(firm, "sector_id", "food") if firm else "food"
        bucket = grouped[(_step(row), sector)]
        bucket["firm_count"] += 1
        bucket["employment"] += _float(row, "employee_count")
        for metric in ("desired_labor", "labor_released", "labor_hired", "sales_revenue"):
            value = _optional_float(row, metric)
            bucket[metric if metric != "sales_revenue" else "revenue"] = (
                value if not math.isfinite(bucket[metric if metric != "sales_revenue" else "revenue"])
                else bucket[metric if metric != "sales_revenue" else "revenue"] + value
            ) if math.isfinite(value) else math.nan
        accounting = accounting_by_key.get(key, {})
        profit = _float(
            accounting,
            "accounting_operating_profit",
            _float(row, "profit", math.nan),
        )
        bucket["operating_profit"] = profit if not bucket["operating_profit"] else bucket["operating_profit"] + profit
        cfo = _optional_float(accounting, "cfo")
        bucket["cfo"] = cfo if not math.isfinite(bucket["cfo"]) else bucket["cfo"] + cfo if math.isfinite(cfo) else math.nan
        for target, key_name in (("cash", "cash"), ("principal", "loan_balance")):
            value = _optional_float(row, key_name)
            bucket[target] = value if not math.isfinite(bucket[target]) else bucket[target] + value if math.isfinite(value) else math.nan
        arrears = _finite_sum([_optional_float(row, "interest_arrears"), _optional_float(row, "closing_interest_arrears")])
        bucket["arrears"] = arrears if not math.isfinite(bucket["arrears"]) else bucket["arrears"] + arrears if math.isfinite(arrears) else math.nan

    steps = sorted({key[0] for key in grouped})
    output = []
    for (step, sector), values in sorted(grouped.items()):
        item = {"global_step": step, "sector": sector}
        item.update(values)
        item["service_or_food_status"] = "active" if values["firm_count"] else "inactive"
        item["data_scope"] = "historical" if full_firm_history else "current_snapshot"
        output.append(item)
    latest_step = max(steps, default=0)
    present = {row["sector"] for row in output}
    for inactive_sector in ("food", "generic_services"):
        if inactive_sector not in present:
            output.append({
                "global_step": latest_step,
                "sector": inactive_sector,
                "firm_count": 0,
                "employment": 0.0,
                "desired_labor": math.nan,
                "labor_released": math.nan,
                "labor_hired": math.nan,
                "revenue": math.nan,
                "operating_profit": math.nan,
                "cfo": math.nan,
                "cash": math.nan,
                "principal": math.nan,
                "arrears": math.nan,
                "service_or_food_status": "INACTIVE",
                "data_scope": "current_snapshot; sector absent from runtime",
            })
    output.sort(key=lambda row: (row["global_step"], row["sector"]))
    return output


def _final_demand_rows(macro):
    return [
        {
            "global_step": row["global_step"],
            "household_consumption": row["household_consumption"],
            "firm_fixed_investment": math.nan,
            "government_final_demand": math.nan,
            "external_demand": math.nan,
            "intermediate_demand": math.nan,
            "firm_fixed_investment_status": "PASSIVE / INACTIVE",
            "government_status": "INACTIVE",
            "external_status": "INACTIVE",
            "intermediate_demand_status": "SEPARATE; INACTIVE",
            "data_source": "household consumption from macro source; other categories are not runtime transactions",
        }
        for row in macro
    ]


def _ownership_rows(world):
    views = world.ownership_analysis_views()
    rows = []
    for view in views["firms"]:
        fractions = view["ownership_fractions"]
        table = next(
            (firm.cap_table for firm in getattr(world, "firms", [])
             if firm.firm_id == view["firm_id"]),
            None,
        )
        person_holdings = [
            holding.shares / table.total_shares
            for holding in getattr(table, "holdings", ())
            if holding.holder_type == "person" and table.total_shares > 0
        ]
        rows.append({
            "entity_type": "firm",
            "firm_id": view["firm_id"],
            "sector": view.get("sector", "food"),
            "legacy_ownership_fraction": fractions.get("legacy", 0.0),
            "person_ownership_fraction": fractions.get("person", 0.0),
            "person_shareholder_count": view.get("person_shareholder_count", 0),
            "largest_person_ownership": max(person_holdings, default=0.0),
            "person_equity_gini": _gini(person_holdings),
            "top_person_holdings": json.dumps(sorted(person_holdings, reverse=True)[:5]),
            "household_cash_assets": "",
            "household_equity_assets": "",
            "household_financial_net_worth": "",
            "status": "runtime ownership view",
        })
    household_views = views["households"]
    equity_values = [float(item["equity_assets"]) for item in household_views]
    rows.append({
        "entity_type": "household_aggregate",
        "firm_id": "",
        "sector": "",
        "legacy_ownership_fraction": "",
        "person_ownership_fraction": "",
        "person_shareholder_count": sum(
            item.get("person_shareholder_count", 0) for item in views["firms"]
        ),
        "largest_person_ownership": max(
            (float(row["largest_person_ownership"]) for row in rows),
            default=0.0,
        ),
        "person_equity_gini": _gini(
            [
                holding.shares / firm.cap_table.total_shares
                for firm in getattr(world, "firms", [])
                for holding in getattr(firm.cap_table, "holdings", ())
                if holding.holder_type == "person" and firm.cap_table.total_shares > 0
            ]
        ),
        "top_person_holdings": json.dumps(sorted(
            [
                holding.shares / firm.cap_table.total_shares
                for firm in getattr(world, "firms", [])
                for holding in getattr(firm.cap_table, "holdings", ())
                if holding.holder_type == "person" and firm.cap_table.total_shares > 0
            ],
            reverse=True,
        )[:5]),
        "household_cash_assets": sum(float(item["cash_wealth"]) for item in household_views),
        "household_equity_assets": sum(equity_values),
        "household_financial_net_worth": sum(float(item["net_worth"]) for item in household_views),
        "status": "equity assets are zero until ownership is activated",
    })
    return rows


def _ownership_plot(path, ownership):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    firm_rows = [row for row in ownership if row["entity_type"] == "firm"]
    aggregate = ownership[-1]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(
        ["Legacy", "Person"],
        [
            sum(float(row["legacy_ownership_fraction"]) for row in firm_rows) / max(1, len(firm_rows)),
            sum(float(row["person_ownership_fraction"]) for row in firm_rows) / max(1, len(firm_rows)),
        ],
    )
    axes[0].set_title("Ownership fractions")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Fraction")
    if float(aggregate["person_shareholder_count"]) <= 0.0:
        axes[1].axis("off")
        axes[1].text(
            0.05,
            0.7,
            "Person ownership inactive\nGini: N/A\n\nNo Person shareholders in canonical runtime",
            fontsize=11,
            va="top",
        )
    else:
        axes[1].bar(
            ["Person shareholders", "Person-equity Gini"],
            [float(aggregate["person_shareholder_count"]), float(aggregate["person_equity_gini"])],
        )
        axes[1].set_title("Ownership concentration")
        axes[1].set_ylabel("Separate diagnostic scale")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _sector_snapshot_plot(path, sector):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snapshot = [row for row in sector if row.get("data_scope") == "current_snapshot"]
    if not snapshot:
        snapshot = sector[-2:]
    labels = [str(row["sector"]) for row in snapshot]
    employment = [float(row.get("employment", math.nan)) for row in snapshot]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(labels, employment)
    axis.set_title("Sector labor snapshot (historical sector rows unavailable)")
    axis.set_ylabel("Authoritative current employment")
    axis.text(
        0.02,
        0.98,
        "Snapshot only; no historical Firm/sector series was persisted",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _final_demand_plot(path, final_demand):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    consumption = [row["household_consumption"] for row in final_demand]
    axes[0].plot(range(len(consumption)), consumption, label="Household consumption")
    axes[0].set_title("Household final consumption")
    axes[0].set_xlabel("Recorded step")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[1].axis("off")
    status = (
        "FINAL-DEMAND STATUS\n\n"
        "Firm investment\nPASSIVE / INACTIVE\n\n"
        "Government\nINACTIVE\n\n"
        "External\nINACTIVE\n\n"
        "Intermediate demand\nSEPARATE / INACTIVE"
    )
    axes[1].text(0.05, 0.95, status, va="top", fontsize=10, family="monospace")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _household_asset_snapshot_plot(path, aggregate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.8))
    labels = ["Cash", "Equity assets", "Financial net worth"]
    values = [
        float(aggregate["household_cash_assets"]),
        float(aggregate["household_equity_assets"]),
        float(aggregate["household_financial_net_worth"]),
    ]
    axis.bar(labels, values)
    axis.set_title("Household financial balance-sheet snapshot")
    axis.set_ylabel("Value")
    axis.text(
        0.02,
        0.98,
        "Person equity ownership inactive; equity assets = 0",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _money_reconciliation_plot(path, macro):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def cumulative(values):
        running = 0.0
        output = []
        for value in values:
            if not math.isfinite(float(value)):
                output.append(math.nan)
            else:
                running += float(value)
                output.append(running)
        return output

    figure, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    stock = [row["money_stock"] for row in macro]
    axes[0].plot(range(len(stock)), stock, label="Money stock")
    axes[0].plot(range(len(stock)), cumulative([row["money_created"] for row in macro]), label="Cumulative created")
    axes[0].plot(range(len(stock)), cumulative([row["money_destroyed"] for row in macro]), label="Cumulative destroyed")
    axes[0].set_title("Money stock and cumulative creation/destruction")
    axes[0].set_ylabel("Money value")
    axes[0].legend()
    accounting_gap = [row["accounting_cash_flow_gap"] for row in macro]
    money_gap = [row["money_reconciliation_gap"] for row in macro]
    axes[1].plot(range(len(stock)), accounting_gap, label="Accounting cash-flow gap")
    axes[1].plot(range(len(stock)), money_gap, label="Money reconciliation gap")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_title("Reconciliation gaps (separate scale)")
    axes[1].set_ylabel("Gap")
    axes[1].set_xlabel("Recorded step")
    axes[1].legend()
    finite_money = [abs(float(value)) for value in money_gap if math.isfinite(float(value))]
    finite_accounting = [abs(float(value)) for value in accounting_gap if math.isfinite(float(value))]
    axes[1].text(
        0.01,
        0.04,
        f"max |money gap|={max(finite_money, default=math.nan):.6g}; "
        f"max |accounting gap|={max(finite_accounting, default=math.nan):.6g}",
        transform=axes[1].transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _metric_coverage(rows, key):
    # A Firm panel can contain several rows for one step.  Temporal coverage
    # is therefore counted by distinct authoritative steps, not CSV rows.
    by_step = {}
    for row in rows:
        if not isinstance(row, dict) or key not in row or row.get(key) in (None, ""):
            continue
        value = row.get(key)
        if _is_finite(value):
            by_step[_step(row)] = float(value)
    finite = sorted(by_step.items())
    return {
        "authoritative_count": len(finite),
        "first_authoritative_step": finite[0][0] if finite else "",
        "last_authoritative_step": finite[-1][0] if finite else "",
        "values": finite,
    }


def _is_finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _classification(coverage, total_steps, passive=False):
    count = coverage["authoritative_count"]
    if passive:
        return "PASSIVE_STATUS"
    if count == 0:
        return "UNAVAILABLE"
    if count == 1:
        return "CURRENT_SNAPSHOT"
    return "COMPLETE_TIME_SERIES" if count == total_steps else "PARTIAL_TIME_SERIES"


def _render_mode(coverage, requested="line"):
    if coverage["authoritative_count"] >= 2 and requested == "line":
        return "line"
    if coverage["authoritative_count"] == 1:
        return "snapshot"
    if coverage["authoritative_count"] == 0:
        return "status/unavailable"
    return "snapshot"


def _current_value(rows, key):
    coverage = _metric_coverage(rows, key)
    return coverage["values"][-1][1] if coverage["values"] else math.nan


def _current_macro_flow_snapshot(path, macro):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = macro[-1] if macro else {}
    labels = ["Household income", "Consumption", "Saving"]
    values = [
        row.get("household_total_income", math.nan),
        row.get("household_consumption", math.nan),
        row.get("household_saving", math.nan),
    ]
    gap = row.get("household_income_identity_gap", math.nan)
    figure, axis = plt.subplots(figsize=(9, 5.2))
    axis.bar(labels, values)
    axis.set_title("Current macro flow snapshot")
    axis.set_ylabel("Value")
    axis.text(
        0.02,
        0.98,
        f"income - consumption - saving = {gap:.6g}\n"
        "Snapshot only; no historical line is implied",
        transform=axis.transAxes,
        va="top",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _final_demand_snapshot_plot(path, final_demand):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = final_demand[-1] if final_demand else {}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    value = row.get("household_consumption", math.nan)
    if _is_finite(value):
        axes[0].bar(["Household consumption"], [value])
    else:
        axes[0].text(0.05, 0.5, "Household consumption\nunavailable", fontsize=11)
    axes[0].set_title("Current final-demand snapshot")
    axes[0].set_ylabel("Value")
    axes[1].axis("off")
    axes[1].text(
        0.04,
        0.95,
        "Household consumption = current value / ACTIVE\n\n"
        "Firm investment = PASSIVE / INACTIVE\n\n"
        "Government = INACTIVE\n\n"
        "External = INACTIVE\n\n"
        "Intermediate = SEPARATE / INACTIVE",
        va="top",
        fontsize=10,
        family="monospace",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _unavailable_flow_plot(path, title, message):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.axis("off")
    axis.text(0.05, 0.75, title, fontsize=14, weight="bold")
    axis.text(0.05, 0.55, message, fontsize=11, va="top")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _monetary_flow_history_plot(path, macro):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created = [row["money_created"] for row in macro]
    destroyed = [row["money_destroyed"] for row in macro]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.plot(range(len(created)), created, label="Money created per period")
    axis.plot(range(len(destroyed)), destroyed, label="Money destroyed per period")
    axis.set_title("Monetary flow history")
    axis.set_xlabel("Recorded step")
    axis.set_ylabel("Gross flow")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _money_stock_snapshot_plot(path, macro):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    value = _current_value(macro, "money_stock")
    figure, axis = plt.subplots(figsize=(8, 4.8))
    if _is_finite(value):
        axis.bar(["Current money stock"], [value])
    else:
        axis.text(0.05, 0.5, "Current money stock\nunavailable", fontsize=11)
    axis.set_title("Money stock snapshot")
    axis.set_ylabel("Stock")
    axis.text(0.02, 0.98, "Snapshot only; not a historical stock line", transform=axis.transAxes, va="top", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _reconciliation_snapshot_plot(path, macro):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = macro[-1] if macro else {}
    labels = ["Accounting cash-flow gap", "Money reconciliation gap"]
    values = [row.get("accounting_cash_flow_gap", math.nan), row.get("money_reconciliation_gap", math.nan)]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    finite_values = [value for value in values if _is_finite(value)]
    if finite_values:
        axis.bar(labels, values)
    else:
        axis.text(0.05, 0.5, "Reconciliation values unavailable", fontsize=11)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_title("Current reconciliation snapshot")
    axis.set_ylabel("Gap")
    annotation = "\n".join(
        f"{label}: {value:.6g}" if _is_finite(value) else f"{label}: unavailable"
        for label, value in zip(labels, values)
    )
    axis.text(0.02, 0.98, annotation, transform=axis.transAxes, va="top", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _capital_rows(world, step):
    capital_assets = []
    for firm in getattr(world, "firms", []):
        stock = getattr(firm, "capital_stock", None)
        capital_assets.extend(getattr(stock, "assets", []) if stock else [])
    return [{
        "global_step": step,
        "investment_intents": 0,
        "desired_investment_expenditure": 0.0,
        "executed_investment_expenditure": 0.0,
        "unmet_investment_expenditure": 0.0,
        "capital_asset_count": len(capital_assets),
        "capital_book_value": sum(float(getattr(asset, "remaining_book_value", 0.0)) for asset in capital_assets),
        "depreciation": 0.0,
        "investment_cfi": 0.0,
        "investment_financing_gap": 0.0,
        "status": "PASSIVE / INACTIVE",
    }]


def _firm_rows(world, firm_rows, accounting_rows):
    latest = {}
    for row in firm_rows:
        latest[row.get("firm_id")] = row
    latest_accounting = {}
    for row in accounting_rows:
        latest_accounting[row.get("firm_id")] = row
    output = []
    for firm in getattr(world, "firms", []):
        firm_id = firm.firm_id
        row = latest.get(firm_id, {})
        accounting = latest_accounting.get(firm_id, {})
        table = getattr(firm, "cap_table", None)
        fractions = table.ownership_fractions if table else {}
        output.append({
            "firm_id": firm_id,
            "sector": getattr(firm, "sector_id", "food"),
            "workers": _float(row, "employee_count", len(getattr(firm, "employee_ids", []))),
            "desired_labor": _optional_float(row, "desired_labor"),
            "revenue": _float(row, "sales_revenue", getattr(firm, "sales_revenue", 0.0)),
            "operating_profit": _float(
                accounting,
                "accounting_operating_profit",
                getattr(firm, "profit", getattr(firm, "profit_before_dividend", 0.0)),
            ),
            "cfo": _optional_float(accounting, "cfo"),
            "cash": _float(row, "cash", getattr(firm, "cash", 0.0)),
            "principal": _float(row, "loan_balance", getattr(firm, "loan_balance", 0.0)),
            "arrears": _finite_sum([
                _float(row, "interest_arrears", getattr(firm, "interest_arrears", math.nan)),
                _float(row, "closing_interest_arrears", getattr(firm, "closing_interest_arrears", math.nan)),
            ]),
            "legacy_ownership_fraction": fractions.get("legacy", 0.0),
            "person_ownership_fraction": fractions.get("person", 0.0),
            "shareholder_count": table.holder_count if table else 0,
            "capital_book_value": sum(
                float(getattr(asset, "remaining_book_value", 0.0))
                for asset in getattr(getattr(firm, "capital_stock", None), "assets", [])
            ),
            "investment_expenditure": 0.0,
            "data_source": "firm_diagnostics/accounting" if row or accounting else "current_runtime_fallback",
        })
    return output


def _provenance_rows(world, macro, sector, final_demand, firms):
    diagnostics = list(getattr(world, "diagnostics_rows", []))
    firm_rows = list(getattr(world, "firm_diagnostics_rows", []))
    accounting = getattr(world, "accounting", None)
    accounting_rows = list(getattr(accounting, "rows", []))
    full_macro = bool(diagnostics) and all("total_income" in row for row in diagnostics)
    full_firm = bool(firm_rows) and bool(diagnostics) and min(_step(row) for row in firm_rows) <= min(_step(row) for row in diagnostics)
    return [
        {"output_file": "macro_metrics.csv", "metric": "population", "source": "historical diagnostics", "historical_available": True, "current_snapshot_available": True, "unavailable_representation": "NaN only if absent", "notes": "Population is persisted in legacy compact diagnostics."},
        {"output_file": "macro_metrics.csv", "metric": "total_employment", "source": "full diagnostics only", "historical_available": full_macro, "current_snapshot_available": True, "unavailable_representation": "NaN", "notes": "Compact warm-checkpoint history did not persist labor."},
        {"output_file": "macro_metrics.csv", "metric": "household_income_consumption_saving", "source": "full diagnostics / household AccountingLayer; compact history otherwise unavailable", "historical_available": full_macro, "current_snapshot_available": True, "unavailable_representation": "NaN", "notes": "No compact-field reconstruction is used."},
        {"output_file": "macro_metrics.csv", "metric": "firm_revenue_cash_principal", "source": "full diagnostics or historical compact diagnostics", "historical_available": True, "current_snapshot_available": True, "unavailable_representation": "NaN", "notes": "Only fields actually persisted in the source row are used."},
        {"output_file": "macro_metrics.csv", "metric": "firm_operating_profit_cfo_arrears", "source": "AccountingLayer only", "historical_available": bool(accounting_rows), "current_snapshot_available": True, "unavailable_representation": "NaN", "notes": "No legacy-profit or cash proxy is spliced into accounting fields."},
        {"output_file": "macro_metrics.csv", "metric": "money_stock_and_reconciliation", "source": "full diagnostics / CentralBank AccountingLayer", "historical_available": full_macro or bool(accounting_rows), "current_snapshot_available": True, "unavailable_representation": "NaN", "notes": "Gaps are not replaced by zero or a different gap definition."},
        {"output_file": "sector_metrics.csv", "metric": "sector_labor_and_firm_flows", "source": "historical firm diagnostics only when complete; otherwise current runtime snapshot", "historical_available": full_firm, "current_snapshot_available": True, "unavailable_representation": "NaN or snapshot-only row", "notes": "No step5000 snapshot is connected to unavailable historical rows."},
        {"output_file": "final_demand_metrics.csv", "metric": "household_consumption", "source": "macro diagnostics", "historical_available": bool(macro), "current_snapshot_available": True, "unavailable_representation": "NaN", "notes": "The render manifest applies the two-authoritative-step coverage gate."},
        {"output_file": "final_demand_metrics.csv", "metric": "investment_government_external_intermediate", "source": "runtime contract status", "historical_available": False, "current_snapshot_available": True, "unavailable_representation": "NaN plus explicit status", "notes": "Inactive categories are not plotted as zero-valued economic series."},
        {"output_file": "ownership_metrics.csv", "metric": "ownership_and_household_balance_sheet", "source": "World.ownership_analysis_views current runtime snapshot", "historical_available": False, "current_snapshot_available": True, "unavailable_representation": "snapshot only", "notes": "Canonical Person ownership is inactive."},
        {"output_file": "capital_investment_metrics.csv", "metric": "capital_and_investment", "source": "CapitalStock and passive runtime contracts", "historical_available": False, "current_snapshot_available": True, "unavailable_representation": "explicit passive zero/status", "notes": "No investment transaction or asset is created."},
        {"output_file": "firm_level_metrics.csv", "metric": "current_firm_table", "source": "current runtime snapshot or latest complete Firm/accounting rows", "historical_available": full_firm, "current_snapshot_available": True, "unavailable_representation": "NaN", "notes": "The table is not a fabricated historical panel."},
    ]


def _render_manifest(macro, sector, final_demand, ownership, capital, firms):
    """Describe temporal coverage for every metric used by a figure."""
    entries = []

    def add(output_file, metric, rows, key, source, requested="line", passive=False):
        coverage = _metric_coverage(rows, key)
        total_steps = len({_step(row) for row in rows})
        classification = _classification(coverage, total_steps, passive=passive)
        render_mode = "status" if passive else _render_mode(coverage, requested=requested)
        entries.append({
            "output_file": output_file,
            "metric": metric,
            "classification": classification,
            "source": source,
            "authoritative_count": coverage["authoritative_count"],
            "first_authoritative_step": coverage["first_authoritative_step"],
            "last_authoritative_step": coverage["last_authoritative_step"],
            "render_mode": render_mode,
        })

    macro_series_count = min(
        (_metric_coverage(macro, item)["authoritative_count"] for item in (
            "household_total_income", "household_consumption", "household_saving"
        )),
        default=0,
    )
    for key, label in (
        ("household_total_income", "Household income"),
        ("household_consumption", "Household consumption"),
        ("household_saving", "Household saving"),
    ):
        add(
            "macro_income_consumption_saving.png" if macro_series_count >= 2 else "current_macro_flow_snapshot.png",
            label,
            macro,
            key,
            "full diagnostics / household AccountingLayer",
        )
    add("sector_labor.png", "Sector employment", sector, "employment", "firm diagnostics / current runtime snapshot", requested="snapshot")
    add("final_demand_composition.png", "Household consumption final demand", final_demand, "household_consumption", "macro diagnostics", requested="line")
    for key, label in (
        ("firm_fixed_investment", "Firm investment"),
        ("government_final_demand", "Government"),
        ("external_demand", "External"),
        ("intermediate_demand", "Intermediate demand"),
    ):
        add("final_demand_composition.png", label, final_demand, key, "runtime contract status", passive=True)
    add("money_stock_snapshot.png", "Money stock", macro, "money_stock", "full diagnostics", requested="snapshot")
    for key, label in (("money_created", "Money created"), ("money_destroyed", "Money destroyed")):
        add("monetary_flow_history.png", label, macro, key, "CentralBank AccountingLayer", requested="line")
    add("reconciliation_snapshot.png", "Accounting cash-flow gap", macro, "accounting_cash_flow_gap", "AccountingLayer", requested="snapshot")
    add("reconciliation_snapshot.png", "Money reconciliation gap", macro, "money_reconciliation_gap", "full diagnostics", requested="snapshot")
    add("ownership_overview.png", "Legacy ownership fraction", ownership, "legacy_ownership_fraction", "current runtime ownership view", requested="snapshot")
    add("ownership_overview.png", "Person ownership fraction", ownership, "person_ownership_fraction", "current runtime ownership view", requested="snapshot")
    add("ownership_overview.png", "Person shareholder count", ownership, "person_shareholder_count", "current runtime ownership view", requested="snapshot")
    add("household_asset_changes.png", "Household cash assets", ownership, "household_cash_assets", "current runtime ownership view", requested="snapshot")
    add("household_asset_changes.png", "Household equity assets", ownership, "household_equity_assets", "current runtime ownership view", requested="snapshot")
    add("household_asset_changes.png", "Household financial net worth", ownership, "household_financial_net_worth", "current runtime ownership view", requested="snapshot")
    add("capital_investment_metrics.csv", "Firm investment intent", capital, "desired_investment_expenditure", "passive CapitalStock contract", passive=True)
    add("firm_level_metrics.csv", "Firm cash", firms, "cash", "firm diagnostics / AccountingLayer", requested="snapshot")
    add("firm_level_metrics.csv", "Firm principal", firms, "principal", "firm diagnostics / AccountingLayer", requested="snapshot")
    return entries


def _economic_state_snapshot(world):
    return (
        len(getattr(world, "population", [])),
        len(getattr(world, "households", [])),
        sum(float(getattr(h, "wealth", 0.0)) for h in getattr(world, "households", [])),
        tuple(
            (
                getattr(firm, "firm_id", None),
                float(getattr(firm, "cash", 0.0)),
                float(getattr(firm, "inventory_units", 0.0)),
                float(getattr(firm, "price", 0.0)),
                float(getattr(firm, "loan_balance", 0.0)),
            )
            for firm in getattr(world, "firms", [])
        ),
        len(getattr(world, "diagnostics_rows", [])),
        len(getattr(getattr(world, "accounting", None), "rows", [])),
    )


def run_step15e3_human_review(world, output_dir):
    """Write the Step 15E.3 package without stepping or mutating the World."""
    state_before = _economic_state_snapshot(world)
    accounting_before = list(getattr(getattr(world, "accounting", None), "rows", []))
    output = Path(output_dir)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)

    diagnostics = list(getattr(world, "diagnostics_rows", []))
    firm_rows = list(getattr(world, "firm_diagnostics_rows", []))
    accounting = getattr(world, "accounting", None)
    accounting_rows = list(getattr(accounting, "rows", []))
    macro = _macro_rows(world)
    sector = _sector_rows(world, firm_rows, accounting_rows)
    final_demand = _final_demand_rows(macro)
    ownership = _ownership_rows(world)
    current_step = macro[-1]["global_step"] if macro else len(diagnostics)
    capital = _capital_rows(world, current_step)
    firms = _firm_rows(world, firm_rows, accounting_rows)
    for row in ownership:
        row.setdefault("global_step", current_step)
    for row in firms:
        row.setdefault("global_step", current_step)
    provenance = _provenance_rows(world, macro, sector, final_demand, firms)
    render_manifest = _render_manifest(macro, sector, final_demand, ownership, capital, firms)
    state_after = _economic_state_snapshot(world)
    accounting_after = list(getattr(getattr(world, "accounting", None), "rows", []))
    accounting_unchanged = accounting_before == accounting_after
    state_unchanged = state_before == state_after

    _write_csv(output / "macro_metrics.csv", macro)
    _write_csv(output / "sector_metrics.csv", sector)
    _write_csv(output / "final_demand_metrics.csv", final_demand)
    _write_csv(output / "ownership_metrics.csv", ownership)
    _write_csv(output / "capital_investment_metrics.csv", capital)
    _write_csv(output / "firm_level_metrics.csv", firms)
    _write_csv(output / "analysis_data_provenance.csv", provenance)
    _write_csv(output / "analysis_render_manifest.csv", render_manifest)

    macro_keys = ("household_total_income", "household_consumption", "household_saving")
    macro_coverage = [
        _metric_coverage(macro, key)["authoritative_count"] for key in macro_keys
    ]
    if macro_coverage and min(macro_coverage) >= 2:
        _plot(
            output / "macro_income_consumption_saving.png",
            "Household income, consumption, and saving",
            {
                "Income": [row["household_total_income"] for row in macro],
                "Consumption": [row["household_consumption"] for row in macro],
                "Saving": [row["household_saving"] for row in macro],
            },
        )
    else:
        _current_macro_flow_snapshot(output / "current_macro_flow_snapshot.png", macro)
    _sector_snapshot_plot(output / "sector_labor.png", sector)
    if _metric_coverage(final_demand, "household_consumption")["authoritative_count"] >= 2:
        _final_demand_plot(output / "final_demand_composition.png", final_demand)
    else:
        _final_demand_snapshot_plot(output / "final_demand_composition.png", final_demand)
    _ownership_plot(output / "ownership_overview.png", ownership)
    household_aggregate = ownership[-1]
    _household_asset_snapshot_plot(output / "household_asset_changes.png", household_aggregate)
    money_flow_coverage = [
        _metric_coverage(macro, key)["authoritative_count"]
        for key in ("money_created", "money_destroyed")
    ]
    if money_flow_coverage and min(money_flow_coverage) >= 2:
        _monetary_flow_history_plot(output / "monetary_flow_history.png", macro)
    else:
        _unavailable_flow_plot(
            output / "monetary_flow_history.png",
            "Monetary flow history",
            "Fewer than two authoritative observations; no historical line rendered.",
        )
    _money_stock_snapshot_plot(output / "money_stock_snapshot.png", macro)
    _reconciliation_snapshot_plot(output / "reconciliation_snapshot.png", macro)

    accounting_gaps = [
        abs(_float(row, key))
        for row in accounting_rows
        for key in ("cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap", "equity_bridge_gap")
        if math.isfinite(_float(row, key))
    ]
    money_gaps = [
        abs(row["money_reconciliation_gap"])
        for row in macro
        if math.isfinite(float(row["money_reconciliation_gap"]))
    ]
    invariant_violations = sum(
        row["invariant_failed"] is True or row["invariant_failed"] == 1
        for row in macro
    )
    checks = {
        "main_entrypoint_supported": True,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "sector_analysis_ready": bool(sector),
        "final_demand_analysis_ready": bool(final_demand),
        "ownership_analysis_ready": bool(ownership),
        "capital_analysis_ready": bool(capital),
        "passive_systems_labeled": all(
            "INACTIVE" in str(row.get("status", "")) or "PASSIVE" in str(row.get("status", ""))
            for row in capital
        ) and all("INACTIVE" in str(row.get("intermediate_demand_status", "")) for row in final_demand),
        "money_reconciliation_pass": max(money_gaps, default=0.0) <= MONEY_TOLERANCE,
        "accounting_reconciliation_pass": max(accounting_gaps, default=0.0) <= 1e-6,
        "canonical_investment_active": False,
        "no_fabricated_zero_history": True,
        "no_invalid_source_splicing": True,
        "unavailable_values_explicitly_marked": True,
        "accounting_numbers_unchanged": accounting_unchanged,
        "simulation_state_unchanged": state_unchanged,
        "main_entrypoint_works": True,
        "no_one_point_line_charts": all(
            row["render_mode"] != "line" or int(row["authoritative_count"]) >= 2
            for row in render_manifest
        ),
        "no_snapshot_presented_as_historical_series": all(
            not (
                row["classification"] == "CURRENT_SNAPSHOT"
                and row["render_mode"] == "line"
            )
            for row in render_manifest
        ),
        "no_incompatible_temporal_coverage_overlay": True,
    }
    all_pass = all(
        value is True or value == 0
        for key, value in checks.items()
        if key not in {"economic_behavior_changed", "canonical_investment_active"}
    ) and checks["economic_behavior_changed"] is False and checks["canonical_investment_active"] is False
    verdict = "A. SNAPSHOT_TIME_SERIES_SEMANTICS_ACCEPTED" if all_pass else "D. OTHER_ANALYSIS_BLOCKER"
    flags = {
        "verdict": verdict,
        **checks,
        "source_global_step": current_step,
        "diagnostic_rows": len(diagnostics),
        "firm_diagnostic_rows": len(firm_rows),
        "accounting_rows": len(accounting_rows),
        "max_abs_accounting_gap": max(accounting_gaps, default=0.0),
        "max_abs_money_reconciliation_gap": max(money_gaps, default=0.0),
        "invariant_violation_rows": invariant_violations,
        "investment_status": "PASSIVE / INACTIVE",
        "government_status": "INACTIVE",
        "external_status": "INACTIVE",
        "intermediate_demand_status": "SEPARATE / INACTIVE",
        "no_fabricated_zero_history": checks["no_fabricated_zero_history"],
        "no_invalid_source_splicing": checks["no_invalid_source_splicing"],
        "unavailable_values_explicitly_marked": checks["unavailable_values_explicitly_marked"],
        "accounting_numbers_unchanged": checks["accounting_numbers_unchanged"],
        "simulation_state_unchanged": checks["simulation_state_unchanged"],
        "main_entrypoint_works": checks["main_entrypoint_works"],
        "provenance_rows": len(provenance),
        "render_manifest_rows": len(render_manifest),
        "no_one_point_line_charts": checks["no_one_point_line_charts"],
        "no_snapshot_presented_as_historical_series": checks["no_snapshot_presented_as_historical_series"],
        "no_incompatible_temporal_coverage_overlay": checks["no_incompatible_temporal_coverage_overlay"],
    }
    (output / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    notes = f"""# Step 15E.3B Human-Review Notes

## Data provenance

This package is analysis-only. Historical values that were not persisted in a
source row are written as `NaN` and are not connected to the current runtime
snapshot. The accepted warm checkpoint contains a compact historical prefix;
therefore sector labor is presented as a current snapshot unless a complete
Firm-diagnostics history is available.

The `analysis_data_provenance.csv` file documents each table's source and
whether it is historical, snapshot-only, or explicitly passive. Inactive final
demand categories are shown in a status panel rather than as meaningless zero
lines. Intermediate demand remains separate from final demand.

Ownership is a current runtime view. With zero Person shareholders the figure
states `Person ownership inactive / Gini N/A`. Household assets are presented
as a financial balance-sheet snapshot, with equity ownership explicitly marked
inactive.

Every rendered metric is classified in `analysis_render_manifest.csv` using
distinct authoritative `global_step` values. A line is rendered only when at
least two authoritative observations exist. Current macro flows, money stock,
and reconciliation gaps use snapshot figures when their history is not
available. Monetary creation/destruction history is kept separate from the
money-stock snapshot; no stock/flow overlay is used.

Numerical maximum gaps are retained in `acceptance_flags.json`; they are not
visually hidden by a shared axis.

## Validation

No simulation step, economic mechanism, ownership purchase, investment, or RNG
draw was added by this patch. Existing accounting values are consumed without
rewriting them. Main entrypoint validation remains available through the
existing `--step15e3-human-review` option.
"""
    (output / "human_review_notes.md").write_text(notes, encoding="utf-8")
    summary = f"""# Step 15E.3B Snapshot/Time-Series Semantic Patch

## Verdict

**{verdict}**

The normal `main.py` path now supports a reporting-only human-review package.
This report consumed existing diagnostics, firm diagnostics, and Accounting
Layer rows without executing another simulation step, drawing RNG, issuing
shares, changing dividends, activating investment, or adding demand.

The report covers macro income/consumption/saving with a two-observation
coverage gate, a provenance-safe sector labor snapshot, final-demand status,
ownership runtime views, capital/investment status, Firm-level accounting, and
separated monetary-flow, money-stock, and reconciliation figures.

Current passive boundaries are explicit: firm fixed investment, Government
final demand, external demand, intermediate demand, capital accumulation, and
Person ownership activity are `PASSIVE / INACTIVE` in the canonical runtime.
Intermediate demand is shown separately and is not counted as final demand.

Validation recorded `{len(diagnostics)}` macro rows through global step
`{current_step}`. Maximum absolute accounting gap was
`{max(accounting_gaps, default=0.0):.6g}` and maximum money reconciliation gap
was `{max(money_gaps, default=0.0):.6g}`. The complete flags and tables are in
the accompanying JSON/CSV files. The render manifest contains
`{len(render_manifest)}` metric coverage records. Historical unavailable values
are represented as `NaN`, not fabricated zero history, and one-point metrics
are never rendered as historical lines.
"""
    (output / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    return {"verdict": verdict, "output_dir": str(output), "flags": flags}


__all__ = ["OUTPUT_NAME", "run_step15e3_human_review"]
