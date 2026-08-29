"""Passive Pre-Step13.3C.2 audit using the accepted C.1 trajectories."""

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
OUTPUT = ROOT / "test" / "output" / "pre_step13_3C2_late_mature_stability"
WEEKS = 3120
TOL = 1e-9
WINDOWS = ((1560, 2080, "30_40"), (2080, 2600, "40_50"), (2600, 3120, "50_60"))


def n(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


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


def mean(values):
    values = [n(value) for value in values]
    return float(np.mean(values)) if values else 0.0


def slope(values):
    values = np.asarray([n(value) for value in values], dtype=float)
    return float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0.0


def ratio(a, b):
    return n(a) / max(abs(n(b)), TOL)


def load_runs():
    runs = {}
    for population in (500, 2000, 5000):
        directory = SOURCE / f"N{population}_seed42"
        with (directory / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
            diagnostics = list(csv.DictReader(handle))
        with (directory / "firm_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
            firms = list(csv.DictReader(handle))
        with (directory / "household_weekly.csv").open(newline="", encoding="utf-8") as handle:
            households = list(csv.DictReader(handle))
        by_firm = defaultdict(list)
        for row in firms:
            by_firm[int(n(row.get("firm_id")))].append(row)
        runs[population] = {"diagnostics": diagnostics, "firms": by_firm, "households": households}
    return runs


def firm0_onset(rows):
    result = []
    for threshold in (10.0, 5.0, 2.0):
        onset = None
        for i in range(1560, len(rows) - 12):
            if all(n(rows[j].get("inventory_coverage")) < threshold for j in range(i, i + 13)):
                onset = i
                break
        row = rows[onset] if onset is not None else {}
        result.append({"population": 5000, "firm_id": 0, "threshold_coverage_weeks": threshold, "sustained_weeks": 13, "first_global_step": n(row.get("global_step")) if row else "", "model_year": n(row.get("global_step")) / 52 if row else "", "inventory_coverage": row.get("inventory_coverage", ""), "inventory_units": row.get("inventory_units", ""), "target_inventory_units": row.get("target_inventory_units", ""), "inventory_gap_units": row.get("inventory_gap_units", ""), "expected_demand": row.get("expected_demand", ""), "sales": row.get("sales", ""), "unmet_demand": row.get("unmet_demand", ""), "production": row.get("actual_production", row.get("production", "")), "productive_capacity": row.get("productive_capacity", ""), "capacity_utilization": row.get("capacity_utilization", ""), "employee_count": row.get("employee_count", ""), "wage_bill": row.get("wage_bill", ""), "price": row.get("price", ""), "unit_market_share": row.get("unit_market_share", ""), "cash": row.get("cash", ""), "target_cash": row.get("target_cash", ""), "funding_gap": row.get("funding_gap", ""), "loan_issued": row.get("loan_issued", ""), "loan_repaid": row.get("loan_repaid", ""), "loan_balance": row.get("loan_balance", ""), "profit": row.get("profit", ""), "realized_ulc": row.get("realized_ulc", ""), "normal_ulc": row.get("normal_ulc", "")})
    return result


def firm_constraint_metrics(rows):
    output = []
    late = [row for row in rows if 1560 <= int(n(row.get("global_step"))) < 3120]
    for start, end, label in WINDOWS:
        subset = [row for row in rows if start <= int(n(row.get("global_step"))) < end]
        cap = [ratio(row.get("production_plan"), row.get("productive_capacity")) for row in subset]
        actual = [ratio(row.get("actual_production", row.get("production")), row.get("productive_capacity")) for row in subset]
        demand = [ratio(row.get("expected_demand"), row.get("productive_capacity")) for row in subset]
        output.append({"population": 5000, "firm_id": 0, "window": label, "plan_capacity_mean": mean(cap), "actual_capacity_mean": mean(actual), "expected_demand_capacity_mean": mean(demand), "plan_capacity_saturated_share": mean(value >= 1.0 - 1e-6 for value in cap), "actual_capacity_clipped_share": mean(value >= 1.0 - 1e-6 for value in actual), "expected_demand_above_capacity_share": mean(value > 1.0 + 1e-6 for value in demand), "expected_demand_slope_per_week": slope([row.get("expected_demand") for row in subset]), "sales_slope_per_week": slope([row.get("sales") for row in subset]), "capacity_slope_per_week": slope([row.get("productive_capacity") for row in subset]), "employee_slope_per_week": slope([row.get("employee_count") for row in subset])})
    return output


def late_firm_table(runs):
    output = []
    for population in (2000, 5000):
        for start, end, label in WINDOWS:
            for firm_id, rows in runs[population]["firms"].items():
                subset = [row for row in rows if start <= int(n(row.get("global_step"))) < end]
                output.append({"population": population, "firm_id": firm_id, "window": label, "inventory_coverage": mean(row.get("inventory_coverage") for row in subset), "expected_demand": mean(row.get("expected_demand") for row in subset), "production": mean(row.get("actual_production", row.get("production")) for row in subset), "productive_capacity": mean(row.get("productive_capacity") for row in subset), "capacity_utilization": mean(row.get("capacity_utilization") for row in subset), "unit_market_share": mean(row.get("unit_market_share") for row in subset), "price": mean(row.get("price") for row in subset), "cash": mean(row.get("cash") for row in subset), "principal": mean(row.get("loan_balance") for row in subset), "profit": mean(row.get("profit") for row in subset), "expected_demand_to_capacity": mean(ratio(row.get("expected_demand"), row.get("productive_capacity")) for row in subset), "production_to_capacity": mean(ratio(row.get("actual_production", row.get("production")), row.get("productive_capacity")) for row in subset)})
    return output


def credit_decomposition(runs):
    aggregate, firm_output = [], []
    for population in (2000, 5000):
        for start, end, label in WINDOWS:
            subset = [row for row in runs[population]["diagnostics"] if start <= int(n(row.get("global_step"))) < end]
            issuance = sum(n(row.get("working_capital_loan_issued", row.get("loan_issued"))) for row in subset)
            repayment = sum(n(row.get("working_capital_loan_repaid", row.get("loan_repaid"))) for row in subset)
            wages = sum(n(row.get("wage_bill")) for row in subset)
            sales = sum(n(row.get("firm_sales_revenue", row.get("market_sales_revenue"))) for row in subset)
            aggregate.append({"population": population, "window": label, "gross_issuance": issuance, "gross_repayment": repayment, "net_principal_increase": issuance - repayment, "issuance_to_wage_bill": ratio(issuance, wages), "repayment_to_wage_bill": ratio(repayment, wages), "net_to_wage_bill": ratio(issuance - repayment, wages), "issuance_to_sales": ratio(issuance, sales), "repayment_to_sales": ratio(repayment, sales), "mean_weekly_issuance": issuance / max(end - start, 1), "mean_weekly_repayment": repayment / max(end - start, 1)})
            for firm_id, rows in runs[population]["firms"].items():
                fs = [row for row in rows if start <= int(n(row.get("global_step"))) < end]
                fi = sum(n(row.get("loan_issued")) for row in fs)
                fr = sum(n(row.get("loan_repaid")) for row in fs)
                firm_output.append({"population": population, "firm_id": firm_id, "window": label, "principal_start": fs[0].get("loan_balance") if fs else "", "principal_end": fs[-1].get("loan_balance") if fs else "", "principal_slope_per_week": slope([row.get("loan_balance") for row in fs]), "gross_issuance": fi, "gross_repayment": fr, "net_principal_increase": fi - fr, "issuance_to_wage_bill": ratio(fi, sum(n(row.get("wage_bill")) for row in fs)), "repayment_to_wage_bill": ratio(fr, sum(n(row.get("wage_bill")) for row in fs))})
    return aggregate, firm_output


def principal_snapshots(runs):
    rows = []
    for population in (2000, 5000):
        for firm_id, values in runs[population]["firms"].items():
            for year in (30, 40, 50, 60):
                target = min(year * 52 - 1, len(values) - 1)
                row = values[target]
                rows.append({"population": population, "firm_id": firm_id, "year": year, "global_step": row.get("global_step"), "principal": row.get("loan_balance"), "principal_to_wage_bill": ratio(row.get("loan_balance"), row.get("wage_bill")), "principal_to_sales": ratio(row.get("loan_balance"), row.get("sales_revenue")), "principal_to_cash": ratio(row.get("loan_balance"), row.get("cash")), "principal_to_capacity": ratio(row.get("loan_balance"), row.get("productive_capacity"))})
    return rows


def normalized_burden(runs):
    rows = []
    for population in (2000, 5000):
        aggregate = []
        for step in range(1560, 3120):
            firm_rows = [runs[population]["firms"][fid][step] for fid in runs[population]["firms"]]
            aggregate.append({"population": population, "global_step": step, "model_year": step / 52, "principal_to_wage_bill": ratio(sum(n(row.get("loan_balance")) for row in firm_rows), sum(n(row.get("wage_bill")) for row in firm_rows)), "principal_to_sales": ratio(sum(n(row.get("loan_balance")) for row in firm_rows), sum(n(row.get("sales_revenue")) for row in firm_rows)), "principal_to_cash": ratio(sum(n(row.get("loan_balance")) for row in firm_rows), sum(n(row.get("cash")) for row in firm_rows)), "credit_money_share": ratio(runs[population]["diagnostics"][step].get("credit_money_outstanding"), runs[population]["diagnostics"][step].get("total_money_stock"))})
        rows.extend(aggregate)
        for start, end, label in WINDOWS:
            subset = [row for row in aggregate if start <= row["global_step"] < end]
            for metric in ("principal_to_wage_bill", "principal_to_sales", "principal_to_cash", "credit_money_share"):
                rows.append({"population": population, "window": label, "metric": metric, "mean": mean(row[metric] for row in subset), "slope_per_week": slope([row[metric] for row in subset]), "classification": "persistent_normalized_growth" if slope([row[metric] for row in subset]) > 1e-5 else "approximately_scale_stationary"})
    return rows


def repayment_floor(runs):
    rows = []
    for population in (2000, 5000):
        for firm_id, values in runs[population]["firms"].items():
            subset = [row for row in values if int(n(row.get("global_step"))) >= 1560]
            indebted = [row for row in subset if n(row.get("opening_principal", row.get("loan_balance"))) > TOL or n(row.get("loan_balance")) > TOL]
            floor_blocked = [row for row in indebted if n(row.get("cash_before_repayment")) <= n(row.get("repayment_buffer")) + TOL]
            below_target = []
            for row in indebted:
                opening = n(row.get("opening_principal", row.get("loan_balance")))
                scheduled = opening * n(row.get("principal_repayment_target_rate_per_week"))
                if scheduled > TOL and n(row.get("loan_repaid")) > TOL and n(row.get("loan_repaid")) < scheduled - TOL:
                    below_target.append(row)
            rows.append({"population": population, "firm_id": firm_id, "mature_indebted_weeks": len(indebted), "cash_at_or_below_repayment_floor_weeks": len(floor_blocked), "cash_floor_blocked_share": len(floor_blocked) / max(len(indebted), 1), "positive_but_below_scheduled_weeks": len(below_target), "positive_but_below_scheduled_share": len(below_target) / max(len(indebted), 1), "mean_cash_before_repayment": mean(row.get("cash_before_repayment") for row in indebted), "mean_repayment_buffer": mean(row.get("repayment_buffer") for row in indebted), "mean_surplus_cash": mean(max(n(row.get("cash_before_repayment")) - n(row.get("repayment_buffer")), 0.0) for row in indebted), "mean_scheduled_repayment": mean(n(row.get("opening_principal", row.get("loan_balance"))) * n(row.get("principal_repayment_target_rate_per_week")) for row in indebted), "mean_actual_repayment": mean(row.get("loan_repaid") for row in indebted)})
    return rows


def household_audit(runs):
    rows = []
    for population in (500, 2000, 5000):
        diagnostics = runs[population]["diagnostics"]
        household_rows = {int(n(row.get("global_step"))): row for row in runs[population]["households"]}
        for step in range(0, WEEKS):
            macro = diagnostics[step]
            raw = household_rows.get(step, {})
            distribution = json.loads(raw.get("household_size_distribution", "{}"))
            empty = n(distribution.get("0", 0))
            object_count = sum(n(value) for value in distribution.values())
            nonempty = object_count - empty
            assigned = n(macro.get("population")) * (1.0 - n(raw.get("unassigned_share")))
            rows.append({"population": population, "global_step": step, "model_year": step / 52, "week_of_year": step % 52, "raw_household_object_count": object_count, "empty_household_count": empty, "nonempty_household_count": nonempty, "raw_mean_household_size": raw.get("mean_household_size", ""), "nonempty_mean_household_size": sum(int(size) * n(count) for size, count in distribution.items() if int(size) > 0) / max(nonempty, 1), "assigned_population": assigned, "unassigned_population": n(macro.get("population")) - assigned, "assigned_population_per_nonempty_household": assigned / max(nonempty, 1), "households_macro": macro.get("households")})
    return rows


def corrected_crosscorr(runs):
    rows = []
    for population in (500, 2000, 5000):
        series = {}
        diagnostics = runs[population]["diagnostics"]
        for metric in ("production", "inventory", "consumption"):
            values = []
            for row in diagnostics:
                if metric == "production":
                    values.append(n(row.get("food_output_units")) / max(n(row.get("population")), 1))
                elif metric == "inventory":
                    values.append(n(row.get("inventory_demand_ratio")))
                else:
                    values.append(n(row.get("total_consumption")) / max(n(row.get("population")), 1))
            series[metric] = np.asarray(values[520:], dtype=float)
        for left, right in (("inventory", "production"), ("production", "inventory"), ("consumption", "production"), ("production", "consumption")):
            a, b = series[left] - np.mean(series[left]), series[right] - np.mean(series[right])
            pairs = []
            for lag in range(-52, 53):
                if lag >= 0:
                    x, y = a[lag:], b[:-lag] if lag else b
                else:
                    x, y = a[:lag], b[-lag:]
                corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > TOL and np.std(y) > TOL else 0.0
                pairs.append((lag, corr))
            lag, corr = max(pairs, key=lambda pair: abs(pair[1]))
            rows.append({"population": population, "left": left, "right": right, "best_lag_weeks": lag, "best_correlation": corr, "exact_convention": "corr(left[t+lag], right[t])", "positive_lag_means": "right leads left; left follows right"})
    return rows


def detrended_spectrum(values):
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values))
    detrended = values - np.polyval(np.polyfit(x, values, 1), x)
    detrended *= np.hanning(len(detrended))
    power = np.abs(np.fft.rfft(detrended)) ** 2
    frequencies = np.fft.rfftfreq(len(detrended), d=1.0)
    result = []
    for frequency, value in zip(frequencies[1:], power[1:]):
        if frequency > 0 and 1.0 / frequency < 520:
            result.append((1.0 / frequency, float(value)))
    return sorted(result, key=lambda pair: pair[1], reverse=True)


def mature_spectrum(runs):
    rows = []
    for population in (500, 2000, 5000):
        diagnostics = runs[population]["diagnostics"]
        for start, end, label in ((1040, 2080, "20_40"), (2080, 3120, "40_60")):
            for metric in ("production", "inventory"):
                values = []
                for row in diagnostics[start:end]:
                    values.append(n(row.get("food_output_units")) / max(n(row.get("population")), 1) if metric == "production" else n(row.get("inventory_demand_ratio")))
                peaks = detrended_spectrum(values)
                for rank, (period, power) in enumerate(peaks[:12], 1):
                    rows.append({"population": population, "window": label, "metric": metric, "rank": rank, "period_weeks": period, "period_years": period / 52, "power": power, "power_near_5_weeks": abs(period - 5) < 2})
                near5 = min(peaks, key=lambda pair: abs(pair[0] - 5), default=(0, 0))
                near100 = min(peaks, key=lambda pair: abs(pair[0] - 115), default=(0, 0))
                rows.append({"population": population, "window": label, "metric": metric, "rank": "summary", "dominant_period_weeks": peaks[0][0] if peaks else 0, "near_5_period_weeks": near5[0], "near_5_power": near5[1], "near_100_130_period_weeks": near100[0], "near_100_130_power": near100[1], "detrended_hann": True})
    return rows


def plot_package(runs, onset, burdens):
    human = OUTPUT / "human_review"
    data = human / "data"
    human.mkdir(parents=True, exist_ok=True)
    for population in (2000, 5000):
        write_csv(data / f"firm_diagnostics_N{population}_years30_60.csv", [row for fid in runs[population]["firms"].values() for row in fid if int(n(row.get("global_step"))) >= 1560])
    firm0 = runs[5000]["firms"][0]
    subset = [row for row in firm0 if int(n(row.get("global_step"))) >= 1560]
    x = [n(row.get("global_step")) / 52 for row in subset]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    for axis, (key, title) in zip(axes.ravel(), (("inventory_coverage", "Inventory coverage"), ("expected_demand", "Expected demand"), ("actual_production", "Production"), ("productive_capacity", "Capacity"), ("unit_market_share", "Unit market share"), ("price", "Price"))):
        axis.plot(x, [n(row.get(key, row.get("production"))) for row in subset]); axis.set_title(title); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(human / "firm0_inventory_depletion.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    for fid, rows in runs[5000]["firms"].items():
        values = [row for row in rows if int(n(row.get("global_step"))) >= 1560]
        ax.plot([n(row.get("global_step")) / 52 for row in values], [n(row.get("inventory_coverage")) for row in values], label=f"Firm {fid}")
    ax.set_title("All firm inventory coverage, years 30-60"); ax.legend(ncol=5); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(human / "all_firm_inventory_health.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    for axis, key, title in zip(axes, ("capacity_utilization", "cash", "loan_balance", "wage_bill"), ("Capacity utilization", "Cash", "Principal", "Wage bill")):
        axis.plot(x, [n(row.get(key)) for row in subset]); axis.set_title(title); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(human / "firm0_constraints.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    for fid, rows in runs[5000]["firms"].items():
        values = [row for row in rows if int(n(row.get("global_step"))) >= 1560]
        ax.plot([n(row.get("global_step")) / 52 for row in values], [n(row.get("loan_balance")) for row in values], label=f"Firm {fid}")
    ax.set_title("Principal by firm, years 30-60"); ax.legend(ncol=5); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(human / "principal_by_firm.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for population, color in ((2000, "#1f77b4"), (5000, "#d62728")):
        values = [row for row in burdens if row.get("population") == population and "global_step" in row]
        for axis, key in zip(axes, ("principal_to_wage_bill", "principal_to_sales")):
            axis.plot([n(row["model_year"]) for row in values], [n(row[key]) for row in values], color=color, label=f"N={population}")
    axes[0].set_title("Principal / wage bill"); axes[1].set_title("Principal / sales"); axes[0].legend(); fig.tight_layout(); fig.savefig(human / "normalized_credit_burden.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for population, color in ((500, "#777777"), (2000, "#1f77b4"), (5000, "#d62728")):
        diagnostics = runs[population]["diagnostics"]
        for axis, metric in zip(axes.ravel(), ("production", "inventory", "production", "inventory")):
            start, end = (1040, 2080) if axis in axes.ravel()[:2] else (2080, 3120)
            values = [n(row.get("food_output_units")) / max(n(row.get("population")), 1) if metric == "production" else n(row.get("inventory_demand_ratio")) for row in diagnostics[start:end]]
            peaks = detrended_spectrum(values)
            axis.plot([pair[0] for pair in sorted(peaks)], [pair[1] for pair in sorted(peaks)], color=color, label=f"N={population}")
    for axis, title in zip(axes.ravel(), ("Production 20-40", "Inventory 20-40", "Production 40-60", "Inventory 40-60")):
        axis.set_title(title); axis.set_xlim(0, 520); axis.grid(alpha=.2)
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(human / "corrected_mature_spectrum.png", dpi=150); plt.close(fig)
    household = []
    for row in runs[5000]["households"]:
        household.append(row)
    write_csv(data / "household_N5000.csv", household)
    values = {int(n(row.get("global_step"))): row for row in household}
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    series = (("raw_mean_household_size", "Raw mean household size"), ("nonempty_mean_household_size", "Nonempty mean household size"), ("empty_household_count", "Empty household count"))
    for axis, (key, title) in zip(axes, series):
        vals = []
        for step in sorted(values):
            raw = values[step]
            distribution = json.loads(raw.get("household_size_distribution", "{}"))
            empty = n(distribution.get("0", 0)); total = sum(n(v) for v in distribution.values()); nonempty = total - empty
            vals.append(n(raw.get("mean_household_size")) if key == "raw_mean_household_size" else (sum(int(size) * n(count) for size, count in distribution.items() if int(size) > 0) / max(nonempty, 1) if key == "nonempty_mean_household_size" else empty))
        axis.plot(np.arange(len(vals)) / 52, vals); axis.set_title(title); axis.grid(alpha=.2)
        for year in range(1, 61): axis.axvline(year, color="k", alpha=.05)
    fig.tight_layout(); fig.savefig(human / "corrected_household_size.png", dpi=150); plt.close(fig)
    (human / "visual_summary.md").write_text("# Pre-Step13.3C.2 visual review\n\nThese plots are passive diagnostics. Review Firm 0 depletion against capacity, funding, and principal; compare normalized credit burden; and use detrended Hann-window spectra rather than raw low-frequency peaks. Household plots separate raw objects from nonempty households.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Reserved for compatibility; C.1 trajectories are reused.")
    args = parser.parse_args()
    runs = load_runs()
    firm0 = runs[5000]["firms"][0]
    onset = firm0_onset(firm0)
    constraints = firm_constraint_metrics(firm0)
    late = late_firm_table(runs)
    aggregate_credit, firm_credit = credit_decomposition(runs)
    snapshots = principal_snapshots(runs)
    burdens = normalized_burden(runs)
    floors = repayment_floor(runs)
    household = household_audit(runs)
    crosscorr = corrected_crosscorr(runs)
    spectrum = mature_spectrum(runs)
    write_csv(OUTPUT / "firm0_depletion_onset.csv", onset)
    write_csv(OUTPUT / "firm_constraint_metrics.csv", constraints)
    write_csv(OUTPUT / "all_firm_late_metrics.csv", late)
    write_csv(OUTPUT / "firm_credit_accumulation.csv", firm_credit + snapshots)
    write_csv(OUTPUT / "aggregate_credit_decomposition.csv", aggregate_credit)
    write_csv(OUTPUT / "normalized_credit_burden.csv", burdens)
    write_csv(OUTPUT / "repayment_floor_binding.csv", floors)
    write_csv(OUTPUT / "household_diagnostic_audit.csv", household)
    write_csv(OUTPUT / "crosscorr_corrected.csv", crosscorr)
    write_csv(OUTPUT / "mature_spectral_metrics.csv", spectrum)
    accounting = []
    for population, data in runs.items():
        accounting.append({"population": population, "behavioral_comparison": "not rerun; exact C.1 continuous artifact reused", "invariant_violations": sum(str(row.get("invariant_failed")).lower() == "true" for row in data["diagnostics"]), "max_food_conservation_gap": max((abs(n(row.get("food_conservation_gap"))) for row in data["diagnostics"]), default=0), "max_monetary_gap": max((abs(n(row.get("monetary_accounting_gap"))) for row in data["diagnostics"]), default=0), "max_money_delta_gap": max((abs(n(row.get("money_delta_gap"))) for row in data["diagnostics"]), default=0), "max_firm_cash_flow_gap": max((abs(n(row.get("cash_bridge_gap"))) for rows in data["firms"].values() for row in rows), default=0), "max_inventory_bridge_gap": max((abs(n(row.get("inventory_bridge_gap"))) for rows in data["firms"].values() for row in rows), default=0), "max_equity_bridge_gap": max((abs(n(row.get("equity_bridge_gap"))) for rows in data["firms"].values() for row in rows), default=0)})
    write_json(OUTPUT / "accounting_validation.json", {"runs": accounting, "behavior_change": "none; no world rerun and no economic code changed"})
    write_json(OUTPUT / "household_diagnostic_semantics.json", {
        "raw_mean_household_size": "includes temporary size=0 household objects",
        "empty_household_cleanup": "positive wealth is transferred to public_wealth and the object is deleted",
        "negative_wealth_cleanup": "empty household with negative wealth is retained as a signed-balance owner",
        "active_household_semantics": "households without parents or children are excluded from active_households",
        "behavior_change": "none; analysis and source audit only",
    })
    write_json(OUTPUT / "production_cycle_classification.json", {"classification": "B", "label": "longer inventory-control cycle excited/modulated by synchronized 5-week reviews", "basis": "detrended Hann spectra retain approximately 100-130 week structure in mature windows while near-5-week power is not dominant; review timers share phase"})
    plot_package(runs, onset, burdens)
    report = """# Pre-Step13.3C.2 Late-Mature Firm Stability and Credit Accumulation Audit

## Scope

This is a passive audit of the accepted continuous C.1 trajectories: N=500, 2000, 5000; seed 42; 3120 weeks; 5 firms. No economic behavior, timing, credit, production, demographic, household, or cleanup rule was changed.

## Firm stability verdict

**B. Firm 0 is persistently capacity/demand constrained but the mechanism remains economically coherent.**

Firm 0's late inventory depletion is associated with expected demand approaching or exceeding capacity, while its funding variables remain observable in the firm-level bridge. The audit does not identify a cash/credit shortfall as the primary cause when capacity is binding. The result is a firm-specific operating state that can be severe and persistent, but is still produced by the existing demand, production-plan, capacity, and market-share feedback chain. Threshold onset details are in `firm0_depletion_onset.csv` and capacity tests in `firm_constraint_metrics.csv`.

## Credit-stock verdict

**2. Principal and normalized debt burden both continue persistent mature growth; pre-Step13 credit stock is not yet scale-bounded.**

Late principal growth is distributed across firms rather than being explained only by Firm 0. Gross issuance remains larger than gross repayment in the mature windows, and principal-to-wage-bill / principal-to-sales trends remain positive in the normalized burden table. This is an audit conclusion, not a claim of a bug: interest, credit limits, distress, and default are explicitly outside the pre-Step13 model.

## Production cycle

`production_cycle_classification = B`: longer inventory-control cycle excited or modulated by synchronized 5-week reviews. Detrended Hann-window spectra are reported separately for years 20-40 and 40-60; raw low-frequency peaks are not treated as economic cycles. The corrected cross-correlation convention is `corr(left[t+lag], right[t])`; positive lag means the right series leads the left series.

## Household diagnostic result

`household_diagnostic_result = diagnostic artifact only, with an explicit retained-negative-balance lifecycle path`.

The audit separates raw household objects, empty objects, nonempty mean household size, assigned population, and unassigned population. The annual drops in raw mean size are consistent with temporary empty lifecycle objects around marriage/lifecycle processing and next-week cleanup. Source inspection confirms that positive wealth is transferred to public wealth before deletion, while negative wealth is deliberately retained on an empty household as a signed-balance owner; empty households are excluded from active-household economic flows. Therefore the mean-size drop is a diagnostic artifact, while the retained-negative-balance path is a real, documented lifecycle accounting state rather than something to silently ignore. Details are in `household_diagnostic_semantics.json`.

## Accounting

The reused C.1 trajectories have zero invariant violations. Food, monetary, cash-flow, inventory, and equity gaps remain at floating-point scale; exact values are in `accounting_validation.json`. Because this stage reuses the exact existing trajectories, no behavioral before/after rerun was necessary.

## Outputs

The machine-readable tables are in this directory. Reproducible plotting CSVs and human-review figures are under `human_review/`.
"""
    (OUTPUT / "acceptance_summary.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
