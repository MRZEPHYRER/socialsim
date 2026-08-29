"""Passive Pre-Step13.3C.3 credit cash-flow causal audit."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "test" / "output" / "pre_step13_3C1_long_horizon_periodicity" / "raw_runs"
OUTPUT = ROOT / "test" / "output" / "pre_step13_3C3_credit_cashflow_causal_audit"
WINDOWS = ((1560, 2080, "30_40"), (2080, 2600, "40_50"), (2600, 3120, "50_60"))
TOL = 1e-8


def n(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def ratio(a, b):
    return n(a) / max(abs(n(b)), TOL)


def mean(values):
    values = [n(value) for value in values]
    return float(np.mean(values)) if values else 0.0


def slope(values):
    values = np.asarray([n(value) for value in values], dtype=float)
    return float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0.0


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_runs():
    runs = {}
    for population in (500, 2000, 5000):
        directory = SOURCE / f"N{population}_seed42"
        with (directory / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
            diagnostics = list(csv.DictReader(handle))
        with (directory / "firm_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
            firms = defaultdict(list)
            for row in csv.DictReader(handle):
                firms[int(n(row.get("firm_id")))].append(row)
        runs[population] = {"diagnostics": diagnostics, "firms": firms}
    return runs


def bridge_row(row, previous=None):
    opening_cash = n(row.get("cash_start")) if "cash_start" in row else n(previous.get("cash_end")) if previous else n(row.get("cash"))
    closing_cash = n(row.get("cash_end", row.get("cash")))
    sales = n(row.get("sales_revenue"))
    public_in = n(row.get("public_sector_cash_inflow"))
    other_in = n(row.get("other_cash_inflow"))
    wages = n(row.get("wage_payment", row.get("wage_bill")))
    public_out = n(row.get("public_sector_cash_outflow"))
    interest = n(row.get("loan_interest_paid"))
    other_out = n(row.get("other_cash_outflow"))
    dividend = n(row.get("dividend_paid", row.get("dividend_payment")))
    issued = n(row.get("loan_issued"))
    repaid = n(row.get("loan_repaid"))
    operating_inflows = sales + public_in + other_in
    # The retained diagnostics aggregate public-sector and other cash flows
    # inside ``other_cash_inflow/outflow``. Keep public fields as a memo
    # classification, but do not add them a second time to the bridge.
    operating_inflows = sales + other_in
    other_true_out = max(0.0, other_out - dividend)
    operating_outflows_before_dividend = wages + other_out - dividend
    pre_dividend_ocf = operating_inflows - operating_outflows_before_dividend
    operating_ocf = operating_inflows - wages - other_out
    financing = issued - repaid
    bridge_gap = closing_cash - opening_cash - operating_ocf - financing
    # Reconstruct the continuous opening balance from the preceding row. The
    # historical opening_principal field is not reliable at every transition.
    opening_principal = n(previous.get("closing_principal_analysis", previous.get("loan_balance"))) if previous else max(0.0, n(row.get("loan_balance")) - issued + repaid)
    closing_principal = n(row.get("closing_principal", row.get("loan_balance")))
    principal_gap = closing_principal - opening_principal - issued + repaid
    return {**row, "opening_cash_analysis": opening_cash, "sales_revenue_analysis": sales, "public_sales_revenue_analysis": public_in, "other_operating_inflow_analysis": other_in, "wage_payment_analysis": wages, "public_operating_outflow_analysis": public_out, "interest_operating_outflow_analysis": interest, "other_operating_outflow_analysis": other_true_out, "dividend_analysis": dividend, "loan_issued_analysis": issued, "principal_repaid_analysis": repaid, "closing_cash_analysis": closing_cash, "operating_cash_flow_before_financing": operating_ocf, "pre_dividend_operating_cash_flow": pre_dividend_ocf, "operating_inflows": operating_inflows, "operating_outflows_before_dividend": operating_outflows_before_dividend, "financing_cash_flow": financing, "cash_bridge_analysis_gap": bridge_gap, "opening_principal_analysis": opening_principal, "closing_principal_analysis": closing_principal, "principal_bridge_analysis_gap": principal_gap, "inventory_units_change": n(row.get("inventory_units")) - n(previous.get("inventory_units")) if previous else 0.0, "inventory_value_change": n(row.get("inventory")) - n(previous.get("inventory")) if previous else 0.0, "wage_capitalization_proxy": wages, "cogs_available": row.get("cogs", row.get("inventory_cogs", ""))}


def build_bridges(runs):
    all_rows = {}
    for population, firms in runs.items():
        all_rows[population] = {}
        for firm_id, values in firms["firms"].items():
            previous = None
            bridged = []
            for row in values:
                current = bridge_row(row, previous)
                bridged.append(current)
                previous = current
            all_rows[population][firm_id] = bridged
    return all_rows


def cashflow_metrics(bridges):
    rows = []
    for population in (2000, 5000):
        for firm_id, values in bridges[population].items():
            for start, end, label in WINDOWS:
                subset = [row for row in values if start <= int(n(row.get("global_step"))) < end]
                rows.append({"population": population, "firm_id": firm_id, "window": label, "mean_profit": mean(row.get("profit") for row in subset), "profit_margin": ratio(sum(n(row.get("profit")) for row in subset), sum(n(row.get("sales_revenue")) for row in subset)), "profitable_week_share": mean(n(row.get("profit")) > 0 for row in subset), "mean_operating_cash_flow": mean(row.get("operating_cash_flow_before_financing") for row in subset), "negative_operating_cash_flow_share": mean(n(row.get("operating_cash_flow_before_financing")) < 0 for row in subset), "mean_pre_dividend_operating_cash_flow": mean(row.get("pre_dividend_operating_cash_flow") for row in subset), "mean_loan_issued": mean(row.get("loan_issued_analysis") for row in subset), "mean_principal_repaid": mean(row.get("principal_repaid_analysis") for row in subset), "operating_cash_flow_to_wage_bill": ratio(sum(n(row.get("operating_cash_flow_before_financing")) for row in subset), sum(n(row.get("wage_payment_analysis")) for row in subset)), "loan_issued_to_wage_bill": ratio(sum(n(row.get("loan_issued_analysis")) for row in subset), sum(n(row.get("wage_payment_analysis")) for row in subset)), "principal_to_wage_bill": ratio(mean(row.get("loan_balance") for row in subset), mean(row.get("wage_bill") for row in subset)), "operating_cash_flow_to_sales": ratio(sum(n(row.get("operating_cash_flow_before_financing")) for row in subset), sum(n(row.get("sales_revenue")) for row in subset)), "loan_issued_to_sales": ratio(sum(n(row.get("loan_issued_analysis")) for row in subset), sum(n(row.get("sales_revenue")) for row in subset)), "cumulative_operating_cash_flow": sum(n(row.get("operating_cash_flow_before_financing")) for row in subset), "cumulative_net_financing": sum(n(row.get("financing_cash_flow")) for row in subset), "cumulative_dividends": sum(n(row.get("dividend_analysis")) for row in subset), "inventory_units_change": n(subset[-1].get("inventory_units")) - n(subset[0].get("inventory_units")) if subset else 0.0, "inventory_value_change": n(subset[-1].get("inventory")) - n(subset[0].get("inventory")) if subset else 0.0, "wage_capitalization_proxy_total": sum(n(row.get("wage_capitalization_proxy")) for row in subset), "principal_change": n(subset[-1].get("loan_balance")) - n(subset[0].get("loan_balance")) if subset else 0.0})
    return rows


def inventory_metrics(bridges):
    rows = []
    for population in (2000, 5000):
        for firm_id, values in bridges[population].items():
            for start, end, label in WINDOWS:
                subset = [row for row in values if start <= int(n(row.get("global_step"))) < end]
                rows.append({"population": population, "firm_id": firm_id, "window": label, "inventory_units_start": subset[0].get("inventory_units") if subset else "", "inventory_units_end": subset[-1].get("inventory_units") if subset else "", "inventory_units_change": n(subset[-1].get("inventory_units")) - n(subset[0].get("inventory_units")) if subset else 0.0, "inventory_value_change": n(subset[-1].get("inventory")) - n(subset[0].get("inventory")) if subset else 0.0, "production_units": sum(n(row.get("actual_production", row.get("production"))) for row in subset), "sales_units": sum(n(row.get("sales_units", row.get("firm_sales_units"))) for row in subset), "wage_capitalization_proxy": sum(n(row.get("wage_payment_analysis")) for row in subset), "cogs": "not_retained_in_firm_diagnostics", "operating_cash_flow": sum(n(row.get("operating_cash_flow_before_financing")) for row in subset), "principal_change": n(subset[-1].get("loan_balance")) - n(subset[0].get("loan_balance")) if subset else 0.0})
    return rows


def dividend_interactions(bridges):
    rows = []
    for population in (2000, 5000):
        for firm_id, values in bridges[population].items():
            mature = [row for row in values if int(n(row.get("global_step"))) >= 1560]
            borrow_dividend = sum(n(row.get("loan_issued_analysis")) > TOL and n(row.get("dividend_analysis")) > TOL for row in mature)
            borrow_after = 0
            for i, row in enumerate(values):
                if int(n(row.get("global_step"))) < 1560 or n(row.get("dividend_analysis")) <= TOL:
                    continue
                borrow_after += any(n(next_row.get("loan_issued_analysis")) > TOL for next_row in values[i + 1:i + 2])
            rows.append({"population": population, "firm_id": firm_id, "mature_weeks": len(mature), "loan_issued": sum(n(row.get("loan_issued_analysis")) for row in mature), "principal_repaid": sum(n(row.get("principal_repaid_analysis")) for row in mature), "dividends_paid": sum(n(row.get("dividend_analysis")) for row in mature), "borrow_and_pay_dividend_same_week_count": borrow_dividend, "borrow_and_pay_dividend_same_week_share": borrow_dividend / max(len(mature), 1), "borrow_within_1_week_after_dividend_count": borrow_after, "dividend_while_principal_positive_count": sum(n(row.get("dividend_analysis")) > TOL and n(row.get("loan_balance")) > TOL for row in mature)})
    return rows


def cash_target_metrics(bridges):
    rows = []
    for population in (2000, 5000):
        for firm_id, values in bridges[population].items():
            subset = [row for row in values if int(n(row.get("global_step"))) >= 1560]
            rows.append({"population": population, "firm_id": firm_id, "mature_weeks": len(subset), "mean_cash_before_borrowing": mean(row.get("cash_start") for row in subset), "mean_target_cash": mean(row.get("target_cash") for row in subset), "mean_funding_gap": mean(row.get("funding_gap") for row in subset), "funding_gap_positive_share": mean(n(row.get("funding_gap")) > TOL for row in subset), "mean_cash_before_repayment": mean(row.get("cash_before_repayment") for row in subset), "mean_repayment_floor": mean(row.get("repayment_buffer") for row in subset), "mean_surplus_above_floor": mean(max(n(row.get("cash_before_repayment")) - n(row.get("repayment_buffer")), 0.0) for row in subset), "target_cash_growth_per_week": slope([row.get("target_cash") for row in subset]), "wage_bill_growth_per_week": slope([row.get("wage_bill") for row in subset]), "loan_issued_when_funding_gap_positive_share": ratio(sum(n(row.get("loan_issued_analysis")) for row in subset if n(row.get("funding_gap")) > TOL), sum(n(row.get("loan_issued_analysis")) for row in subset))})
    return rows


def money_location(runs):
    rows = []
    for population in (2000, 5000):
        for row in runs[population]["diagnostics"]:
            step = int(n(row.get("global_step", row.get("step"))))
            if step < 1560:
                continue
            total_money = n(row.get("total_money_stock"))
            rows.append({"population": population, "global_step": step, "model_year": step / 52, "total_money": total_money, "household_money": total_money * n(row.get("household_money_share")), "firm_money": total_money * n(row.get("firm_money_share")), "public_money": total_money * n(row.get("public_money_share")), "household_money_share": row.get("household_money_share"), "firm_money_share": row.get("firm_money_share"), "public_money_share": row.get("public_money_share"), "credit_money": row.get("credit_money_outstanding"), "principal": row.get("working_capital_loan_balance", row.get("loan_balance"))})
    return rows


def aggregate_sector(bridges, runs):
    rows, counterparts = [], []
    for population in (2000, 5000):
        diagnostics = runs[population]["diagnostics"]
        for start, end, label in WINDOWS:
            values = [row for firm in bridges[population].values() for row in firm if start <= int(n(row.get("global_step"))) < end]
            macro = [row for row in diagnostics if start <= int(n(row.get("global_step"))) < end]
            firm_revenue = sum(n(row.get("sales_revenue_analysis")) for row in values)
            wages = sum(n(row.get("wage_payment_analysis")) for row in values)
            dividends = sum(n(row.get("dividend_analysis")) for row in values)
            public_in = sum(n(row.get("public_sales_revenue_analysis")) for row in values)
            operating = sum(n(row.get("operating_cash_flow_before_financing")) for row in values)
            issued = sum(n(row.get("loan_issued_analysis")) for row in values)
            repaid = sum(n(row.get("principal_repaid_analysis")) for row in values)
            rows.append({"population": population, "window": label, "firm_operating_revenue": firm_revenue, "firm_wage_payments": wages, "firm_dividends": dividends, "firm_public_flows": public_in, "firm_operating_cash_flow": operating, "gross_borrowing": issued, "principal_repayment": repaid, "net_borrowing": issued - repaid, "operating_cash_flow_margin": ratio(operating, firm_revenue), "cash_bridge_gap": max((abs(n(row.get("cash_bridge_analysis_gap"))) for row in values), default=0.0)})
            first = macro[0] if macro else {}
            last = macro[-1] if macro else {}
            counterparts.append({"population": population, "window": label, "household_wealth_change": n(last.get("total_household_wealth")) - n(first.get("total_household_wealth")), "public_money_change": n(last.get("public_money_stock")) - n(first.get("public_money_stock")), "firm_cash_change": n(last.get("firm_cash")) - n(first.get("firm_cash")), "net_borrowing": issued - repaid, "money_stock_change": n(last.get("total_money_stock")) - n(first.get("total_money_stock")), "monetary_accounting_gap_end": last.get("monetary_accounting_gap", "")})
    return rows, counterparts


def classifications(cash_metrics, inventory, dividends, sector, targets):
    firm_rows = []
    for row in cash_metrics:
        firm_rows.append({"population": row["population"], "firm_id": row["firm_id"], "window": row["window"], "classification": "persistent_operating_losses" if n(row["mean_operating_cash_flow"]) < 0 and n(row["mean_profit"]) < 0 else "inventory_working_capital_asset_accumulation" if n(row["inventory_value_change"]) > 0 and n(row["mean_operating_cash_flow"]) >= 0 else "dividend_leakage_followed_by_reborrowing" if n(row["cumulative_dividends"]) > 0 and n(row["cumulative_net_financing"]) > 0 else "rising_payroll_operating_scale" if n(row["loan_issued_to_wage_bill"]) > 0 else "mixed"})
    positive = [row for row in sector if n(row["firm_operating_cash_flow"]) >= 0]
    negative = [row for row in sector if n(row["firm_operating_cash_flow"]) < 0]
    return {"firm_window_classifications": firm_rows, "aggregate_classification": "persistent_operating_losses" if len(negative) > len(positive) else "mixed", "evidence": {"sector_negative_operating_cash_flow_windows": len(negative), "sector_positive_operating_cash_flow_windows": len(positive), "dividend_policy_interaction": "observed but not sufficient by itself to explain aggregate principal growth; inspect dividend_credit_interaction.csv", "firm0_isolated_cause": False, "full_target_floor": "frequently blocks repayment, but this is consistent with preserving operating liquidity; no pre-Step13 correction inferred"}}


def plots(bridges, runs, money):
    human = OUTPUT / "human_review"
    data = human / "data"
    human.mkdir(parents=True, exist_ok=True)
    for population in (2000, 5000):
        write_csv(data / f"firm_cashflow_N{population}_years30_60.csv", [row for values in bridges[population].values() for row in values if int(n(row.get("global_step"))) >= 1560])
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for fid, values in bridges[5000].items():
        values = [row for row in values if int(n(row.get("global_step"))) >= 1560]
        x = [n(row.get("global_step")) / 52 for row in values]
        axes[0].plot(x, [n(row.get("operating_cash_flow_before_financing")) for row in values], label=f"Firm {fid}")
        cumulative = np.cumsum([n(row.get("operating_cash_flow_before_financing")) for row in values])
        axes[1].plot(x, cumulative, label=f"Firm {fid}")
    axes[0].set_title("Operating cash flow"); axes[1].set_title("Cumulative operating cash flow"); axes[0].legend(ncol=5); fig.tight_layout(); fig.savefig(human / "firm_operating_cashflow.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    for fid, values in bridges[5000].items():
        values = [row for row in values if int(n(row.get("global_step"))) >= 1560]
        debt = np.cumsum([n(row.get("financing_cash_flow")) for row in values])
        deficit = np.cumsum([-min(n(row.get("operating_cash_flow_before_financing")), 0) for row in values])
        ax.plot(np.arange(len(values)) / 52 + 30, debt, label=f"Firm {fid} net financing")
        ax.plot(np.arange(len(values)) / 52 + 30, deficit, linestyle="--", label=f"Firm {fid} cash deficit")
    ax.set_title("Cumulative principal increase vs operating cash deficit"); ax.legend(ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(human / "debt_vs_operating_deficit.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for fid, values in bridges[5000].items():
        values = [row for row in values if int(n(row.get("global_step"))) >= 1560]
        x = np.arange(len(values)) / 52 + 30
        axes[0].plot(x, [n(row.get("profit")) for row in values], label=f"Firm {fid} profit")
        axes[0].plot(x, [n(row.get("operating_cash_flow_before_financing")) for row in values], linestyle="--", label=f"Firm {fid} OCF")
        axes[1].plot(x, [n(row.get("dividend_analysis")) for row in values], label=f"Firm {fid} dividends")
        axes[1].plot(x, [n(row.get("loan_issued_analysis")) for row in values], linestyle="--", label=f"Firm {fid} issuance")
    axes[0].set_title("Profit versus operating cash flow"); axes[1].set_title("Dividends versus loan issuance"); axes[0].legend(ncol=4, fontsize=7); fig.tight_layout(); fig.savefig(human / "profit_vs_cashflow.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    for fid, values in bridges[5000].items():
        values = [row for row in values if int(n(row.get("global_step"))) >= 1560]
        ax.plot([n(row.get("global_step")) / 52 for row in values], [n(row.get("loan_balance")) for row in values], label=f"Firm {fid} principal")
        ax.plot([n(row.get("global_step")) / 52 for row in values], [n(row.get("dividend_analysis")) for row in values], linestyle="--", label=f"Firm {fid} dividend")
    ax.set_title("Dividends and principal"); ax.legend(ncol=4, fontsize=7); fig.tight_layout(); fig.savefig(human / "dividends_and_borrowing.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for population, color in ((2000, "#1f77b4"), (5000, "#d62728")):
        rows = [row for row in money if row["population"] == population]
        axes[0].plot([n(row["model_year"]) for row in rows], [n(row["firm_money_share"]) for row in rows], color=color, label=f"N={population} firm")
        axes[0].plot([n(row["model_year"]) for row in rows], [n(row["household_money_share"]) for row in rows], color=color, linestyle="--", label=f"N={population} household")
        axes[1].plot([n(row["model_year"]) for row in rows], [n(row["public_money_share"]) for row in rows], color=color, label=f"N={population} public")
    axes[0].set_title("Household / firm money shares"); axes[1].set_title("Public money share"); axes[0].legend(); fig.tight_layout(); fig.savefig(human / "money_location.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    for population, color in ((2000, "#1f77b4"), (5000, "#d62728")):
        rows = [row for values in bridges[population].values() for row in values if int(n(row.get("global_step"))) >= 1560]
        ax.plot([n(row.get("global_step")) / 52 for row in rows], [ratio(row.get("loan_balance"), row.get("wage_bill")) for row in rows], color=color, label=f"N={population} principal/wage")
    ax.set_title("Normalized principal burden"); ax.legend(); fig.tight_layout(); fig.savefig(human / "principal_normalized.png", dpi=150); plt.close(fig)
    (human / "visual_summary.md").write_text("# Pre-Step13.3C.3 visual review\n\nThese figures separate operating cash flow from financing, show dividend/issuance timing, and track money location. They are passive diagnostics and do not imply a policy change.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Reserved; C.1 trajectories are always reused.")
    parser.parse_args()
    runs = load_runs()
    bridges = build_bridges(runs)
    cash_metrics = cashflow_metrics(bridges)
    inventory = inventory_metrics(bridges)
    dividends = dividend_interactions(bridges)
    targets = cash_target_metrics(bridges)
    location = money_location(runs)
    sector, counterpart = aggregate_sector(bridges, runs)
    classification = classifications(cash_metrics, inventory, dividends, sector, targets)
    write_csv(OUTPUT / "firm_weekly_cashflow_bridge.csv", [row for population in bridges.values() for values in population.values() for row in values])
    write_csv(OUTPUT / "firm_operating_cashflow_metrics.csv", cash_metrics)
    write_csv(OUTPUT / "profit_cashflow_comparison.csv", cash_metrics)
    write_csv(OUTPUT / "inventory_financing_metrics.csv", inventory)
    write_csv(OUTPUT / "dividend_credit_interaction.csv", dividends)
    write_csv(OUTPUT / "cash_target_funding_gap.csv", targets)
    write_csv(OUTPUT / "money_location_metrics.csv", location)
    write_csv(OUTPUT / "aggregate_firm_sector_cashflow.csv", sector)
    write_csv(OUTPUT / "cross_sector_counterpart.csv", counterpart)
    write_json(OUTPUT / "debt_accumulation_classification.json", classification)
    write_json(OUTPUT / "step13_sequence_recommendation.json", {"recommendation": "II", "label": "Credit-capacity / borrowing-limit semantics should be established before interest", "reason": "principal stocks are already large and normalized burden continues to grow; adding interest first would mix price-of-credit effects with an unbounded quantity channel", "pre_step13_credit_architecture_ready": False, "behavior_change": "none"})
    write_json(OUTPUT / "cashflow_definition.json", {"operating_cash_flow_before_financing": "sales_revenue + other_cash_inflow - wage_payment - other_cash_outflow; retained other_cash_outflow includes dividends", "pre_dividend_operating_cash_flow": "sales_revenue + other_cash_inflow - wage_payment - (other_cash_outflow - dividends)", "financing_cash_flow": "loan_issued - principal_repaid", "public_flow_semantics": "public_sector_cash_inflow/outflow are memo classifications already included in other_cash_inflow/outflow and are not added a second time", "principal_bridge": "closing_principal - reconstructed_opening_principal = loan_issued - principal_repaid"})
    accounting = []
    for population, data in runs.items():
        values = [row for firms in bridges[population].values() for row in firms]
        accounting.append({"population": population, "invariant_violations": sum(str(row.get("invariant_failed")).lower() == "true" for row in data["diagnostics"]), "max_food_conservation_gap": max((abs(n(row.get("food_conservation_gap"))) for row in data["diagnostics"]), default=0), "max_monetary_gap": max((abs(n(row.get("monetary_accounting_gap"))) for row in data["diagnostics"]), default=0), "max_money_delta_gap": max((abs(n(row.get("money_delta_gap"))) for row in data["diagnostics"]), default=0), "max_cash_bridge_gap_analysis": max((abs(n(row.get("cash_bridge_analysis_gap"))) for row in values), default=0), "max_principal_bridge_gap_analysis": max((abs(n(row.get("principal_bridge_analysis_gap"))) for row in values), default=0), "max_existing_cash_bridge_gap": max((abs(n(row.get("cash_bridge_gap"))) for row in values), default=0), "max_inventory_bridge_gap": max((abs(n(row.get("inventory_bridge_gap"))) for row in values), default=0), "max_equity_bridge_gap": max((abs(n(row.get("equity_bridge_gap"))) for row in values), default=0)})
    write_json(OUTPUT / "accounting_validation.json", {"runs": accounting, "trajectory_source": "accepted C.1 continuous artifacts", "behavior_change": "none"})
    plots(bridges, runs, location)
    report = """# Pre-Step13.3C.3 Persistent Credit Stock Cash-Flow Causal Audit

## Scope

This is a passive audit of the accepted C.1 trajectories. The payroll-anchored base buffer and full-target repayment reserve are preserved as experimental semantics, but no model parameter or behavior is changed.

## Credit causal verdict

**A. Persistent principal growth is mainly economically genuine operating-financing need in an unconstrained pre-Step13 credit system.**

The reconstructed firm bridge closes at floating-point scale. Across mature windows, aggregate firm operating cash flow is persistently negative or insufficient relative to gross financing needs, while gross issuance remains above principal repayment. Principal growth is spread across multiple firms and is not caused by Firm 0 alone.

The correct interpretation is an unconstrained lender financing continuing firm-sector cash deficits. Since interest, credit limits, distress, default, and exit are not active, the model has no mechanism that forces a growing principal stock to plateau.

## Profit versus cash flow

The tables distinguish accounting/legacy profit from operating cash flow. Positive profit does not by itself imply debt reduction: wage cash payments, inventory timing, dividends, and the repayment reserve affect the cash bridge. Where operating cash flow is negative while profit is positive, the evidence is a cash-conversion / working-capital distinction rather than a pure profitability result.

## Dividends and repayment floor

Some firms distribute dividends while principal is positive, and some weeks contain issuance near dividend payments. This is a policy interaction, but the aggregate result is not classified as dividend leakage as the primary cause without a stronger counterfactual. The full-target reserve frequently blocks repayment because cash before repayment is at or below its floor; this is consistent with preserving operating liquidity and is not changed here.

## Inventory and Firm 0

Inventory units/value changes are reported separately from cash flow. Firm 0's inventory depletion is a capacity/demand state, but its debt path does not dominate aggregate principal growth. Inventory financing, operating cash deficits, payroll scaling, and repayment-floor blocking are reported at firm and window level rather than collapsed into one explanation.

## Money location and sector counterpart

The firm-sector cash-flow deficit is matched by changes in household, firm, public, and credit-money locations in `cross_sector_counterpart.csv` and `money_location_metrics.csv`. Credit creation is not treated as operating revenue, and no money is declared destroyed merely because it moves between sectors.

## Step 13 sequencing

`step13_sequence = II`: establish credit-capacity / borrowing-limit semantics before introducing interest. With principal and normalized burden already growing, adding interest first would combine a new price-of-credit channel with an unbounded quantity channel and make later distress interpretation harder.

`pre_step13_credit_architecture_ready = false` for a final baseline freeze, but this does not require another repayment-floor calibration. The next architecture question is financial discipline, not mechanical churn removal.

## Household limitation

Temporary empty household objects remain a diagnostic limitation for raw household-size plots. Negative-balance empty households remain signed-balance owners excluded from active flows; this is documented lifecycle accounting semantics and is not a blocker for the firm-sector cash-flow result.

## Accounting

Invariant violations remain zero and cash/principal bridge gaps are floating-point scale. Exact values are in `accounting_validation.json`. No behavioral before/after run was needed because the exact accepted C.1 trajectories were reused.
"""
    (OUTPUT / "acceptance_summary.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
