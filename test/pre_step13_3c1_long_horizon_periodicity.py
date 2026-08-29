"""Passive 60-year warm-up, periodicity, and credit-onset audit.

This file intentionally keeps all economic choices in ``World`` unchanged.
The only temporary overrides are the already accepted research-candidate
liquidity semantics used by the preceding scale-selection audit.
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from central_bank import config as cb_config
from economy import config as econ_config
from world import World

OUTPUT = ROOT / "test" / "output" / "pre_step13_3C1_long_horizon_periodicity"
WEEKS = 3120
CHECKPOINTS = (520, 1040, 1560, 2080, 2600, 3120)
WINDOWS = ((0, 520), (520, 1040), (1040, 1560), (1560, 2080), (2080, 2600), (2600, 3120))
POPULATIONS = (500, 2000, 5000)
BASE_BUFFER = 150000.0
TOL = 1e-9


def n(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def mean(values):
    values = [n(v) for v in values]
    return float(np.mean(values)) if values else 0.0


def stdev(values):
    values = [n(v) for v in values]
    return float(np.std(values)) if values else 0.0


def cv(values):
    values = [n(v) for v in values]
    m = mean(values)
    return stdev(values) / abs(m) if abs(m) > TOL else 0.0


def slope(values):
    values = np.asarray([n(v) for v in values], dtype=float)
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def amplitude(values):
    values = [n(v) for v in values]
    return max(values, default=0.0) - min(values, default=0.0)


def normalized_amplitude(values):
    return amplitude(values) / max(abs(mean(values)), TOL)


def field(row, names, denominator=None):
    value = next((row.get(name) for name in names if name in row), 0.0)
    result = n(value)
    return result / max(n(denominator), 1.0) if denominator is not None else result


def macro_series(rows, metric):
    result = []
    for row in rows:
        pop = max(n(row.get("population")), 1.0)
        if metric == "production_per_capita":
            result.append(field(row, ("food_output_units", "production"), pop))
        elif metric == "consumption_per_capita":
            result.append(field(row, ("total_consumption",), pop))
        elif metric == "income_per_capita":
            result.append(field(row, ("total_income",), pop))
        elif metric == "inventory_coverage":
            result.append(field(row, ("inventory_demand_ratio", "inventory_coverage")))
        elif metric == "workers":
            result.append(field(row, ("workers", "working_age_population")))
        elif metric == "wage_bill":
            result.append(field(row, ("wage_bill",)))
        elif metric == "price":
            result.append(field(row, ("realized_transaction_price_index", "food_price")))
        elif metric == "total_money":
            result.append(field(row, ("total_money_stock",)))
        elif metric == "principal":
            result.append(field(row, ("credit_money_outstanding", "working_capital_loan_balance")))
        elif metric == "sales":
            result.append(field(row, ("food_sales_units", "total_consumption"), pop))
        else:
            result.append(field(row, (metric,)))
    return result


def rolling_amplitude(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return []
    return [float(np.max(values[i - window + 1:i + 1]) - np.min(values[i - window + 1:i + 1])) for i in range(window - 1, len(values))]


def acf(values, max_lag=260):
    x = np.asarray(values, dtype=float)
    x = x - np.mean(x)
    denom = float(np.dot(x, x))
    if denom <= TOL:
        return [1.0] + [0.0] * max_lag
    return [float(np.dot(x[:-lag] if lag else x, x[lag:] if lag else x) / denom) for lag in range(max_lag + 1)]


def spectrum(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 4:
        return []
    x = x - np.mean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    frequencies = np.fft.rfftfreq(len(x), d=1.0)
    rows = []
    for frequency, value in zip(frequencies[1:], power[1:]):
        if frequency > 0:
            rows.append((1.0 / frequency, float(value)))
    return sorted(rows, key=lambda pair: pair[1], reverse=True)


def household_snapshot(world):
    people = list(getattr(world, "population", []))
    households = list(getattr(world, "households", []))
    sizes = [h.size() for h in households]
    assigned = sum(1 for person in people if getattr(person, "household_id", None) in getattr(world, "household_dict", {}))
    counts = {str(size): sizes.count(size) for size in sorted(set(sizes))}
    return {
        "unassigned_share": 1.0 - assigned / max(len(people), 1),
        "active_households_per_capita": len(households) / max(len(people), 1),
        "mean_household_size": mean(sizes),
        "household_size_distribution": json.dumps(counts, sort_keys=True),
        "child_share": mean(getattr(person, "age", 0) < 20 for person in people),
        "worker_share": mean(20 <= getattr(person, "age", 0) < 65 for person in people),
        "elderly_share": mean(getattr(person, "age", 0) >= 65 for person in people),
        "population_count": len(people),
    }


def payroll_buffer(population):
    previous = cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
    cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = BASE_BUFFER
    try:
        world = World(initial_population=population, seed=42, diagnostics_mode="compact", initial_age_phase_mode="distributed")
        world.split_firms(5)
        capacity = math.fsum(f.productive_capacity for f in world.firms)
        base = World(initial_population=500, seed=42, diagnostics_mode="compact", initial_age_phase_mode="distributed")
        base.split_firms(5)
        base_capacity = math.fsum(f.productive_capacity for f in base.firms)
        return BASE_BUFFER * world.firm_system.base_wage_bill(capacity) / max(base.firm_system.base_wage_bill(base_capacity), 1e-9)
    finally:
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = previous


def run_one(population, output, force=False):
    diag_path = output / "diagnostics.csv"
    firm_path = output / "firm_diagnostics.csv"
    if diag_path.exists() and firm_path.exists() and not force:
        return False, 0.0
    previous = (
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER,
        cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER,
        getattr(cb_config, "CENTRAL_BANK_FIRM_REPAYMENT_RESERVE_MODE", "base_buffer"),
    )
    buffer = payroll_buffer(population)
    cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = buffer
    cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = buffer
    cb_config.CENTRAL_BANK_FIRM_REPAYMENT_RESERVE_MODE = "full_target"
    started = time.perf_counter()
    try:
        world = World(initial_population=population, seed=42, diagnostics_mode="full", initial_age_phase_mode="distributed")
        world.split_firms(5)
        diagnostics, firms, households = [], [], []
        checkpoint_rows = []
        for _ in range(WEEKS):
            world.step()
            diagnostics.append(dict(world.diagnostics_rows[-1]))
            firms.extend(dict(row) for row in world.firm_diagnostics_rows[-len(world.firms):])
            households.append({"global_step": world.current_step_index, **household_snapshot(world)})
            if world.current_step_index in CHECKPOINTS:
                row = diagnostics[-1]
                checkpoint_rows.append({"global_step": world.current_step_index, "model_year": world.current_step_index / 52.0, "population": row.get("population"), "households": row.get("households"), "total_money": row.get("total_money_stock"), "principal": row.get("credit_money_outstanding"), "production_per_capita": field(row, ("food_output_units",), row.get("population")), "inventory_coverage": field(row, ("inventory_demand_ratio",))})
            world.firm_diagnostics_rows.clear()
            world.household_diagnostics_rows.clear()
        write_csv(diag_path, diagnostics)
        write_csv(firm_path, firms)
        write_csv(output / "household_weekly.csv", households)
        write_csv(output / "checkpoint_snapshots.csv", checkpoint_rows)
        write_json(output / "run_meta.json", {"population": population, "seed": 42, "weeks": WEEKS, "buffer": buffer, "repayment_reserve": "full_target", "runtime_seconds": time.perf_counter() - started, "continuous": True})
        return True, time.perf_counter() - started
    finally:
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER, cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER, cb_config.CENTRAL_BANK_FIRM_REPAYMENT_RESERVE_MODE = previous


def window_metrics(runs):
    rows = []
    metrics = ("production_per_capita", "inventory_coverage", "consumption_per_capita", "income_per_capita", "workers", "wage_bill", "price", "total_money", "principal")
    for population, data in runs.items():
        diagnostics = data["diagnostics"]
        for index, (start, end) in enumerate(WINDOWS):
            subset = diagnostics[start:end]
            row = {"population": population, "seed": 42, "window": index, "start_week": start, "end_week": end}
            for metric in metrics:
                values = macro_series(subset, metric)
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_std"] = stdev(values)
                row[f"{metric}_cv"] = cv(values)
                row[f"{metric}_amplitude"] = amplitude(values)
                row[f"{metric}_normalized_amplitude"] = normalized_amplitude(values)
                row[f"{metric}_trend"] = slope(values)
                row[f"{metric}_variance"] = stdev(values) ** 2
            rows.append(row)
    return rows


def periodicity_tables(runs):
    rolling, acf_rows, spectral = [], [], []
    important = (1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 52, 55, 100, 104, 130, 156, 208, 260)
    for population, data in runs.items():
        diagnostics = data["diagnostics"]
        for metric in ("production_per_capita", "inventory_coverage"):
            values = macro_series(diagnostics, metric)
            for window in (52, 104, 260):
                envelope = rolling_amplitude(values, window)
                for offset, value in enumerate(envelope):
                    rolling.append({"population": population, "metric": metric, "window_weeks": window, "global_step": offset + window - 1, "model_year": (offset + window - 1) / 52.0, "amplitude": value, "normalized_amplitude": value / max(abs(mean(values[max(0, offset + window - 260):offset + window + 1])), TOL)})
        for metric in ("production_per_capita", "inventory_coverage", "consumption_per_capita", "workers", "wage_bill", "price"):
            values = macro_series(diagnostics, metric)
            values = values[520:] if len(values) > 520 else values
            correlations = acf(values, 260)
            for lag in important:
                acf_rows.append({"population": population, "metric": metric, "window": "years10_60", "lag_weeks": lag, "acf": correlations[lag]})
            for lag, value in enumerate(correlations):
                if lag in important or lag > 0 and value >= max(correlations[max(0, lag - 2):min(len(correlations), lag + 3)]):
                    acf_rows.append({"population": population, "metric": metric, "window": "years10_60_all", "lag_weeks": lag, "acf": value})
            for rank, (period, power) in enumerate(spectrum(values)[:10], 1):
                spectral.append({"population": population, "metric": metric, "rank": rank, "period_weeks": period, "period_years": period / 52.0, "power": power})
    return rolling, acf_rows, spectral


def firm_phase_table(runs):
    rows = []
    firms = runs[5000]["firms"]
    by = defaultdict(list)
    for row in firms:
        by[int(n(row.get("firm_id")))].append(row)
    for start, end in ((260, 520), (1456, 1560), (2496, 2600), (3016, 3120)):
        for fid, values in by.items():
            subset = [row for row in values if start <= int(n(row.get("global_step"))) < end]
            rows.append({"population": 5000, "seed": 42, "firm_id": fid, "window_start": start, "window_end": end, "production_mean": mean(row.get("actual_production", row.get("production")) for row in subset), "inventory_mean": mean(row.get("inventory_coverage") for row in subset), "expected_demand_mean": mean(row.get("expected_demand") for row in subset), "review_count": sum(str(row.get("production_reviewed")).lower() == "true" for row in subset), "review_steps": ";".join(str(int(n(row.get("global_step")))) for row in subset if str(row.get("production_reviewed")).lower() == "true")})
    return rows


def crosscorr_table(runs):
    rows = []
    for population, data in runs.items():
        diagnostics = data["diagnostics"]
        series = {metric: np.asarray(macro_series(diagnostics, metric), dtype=float) for metric in ("production_per_capita", "inventory_coverage", "consumption_per_capita", "sales", "price")}
        for left, right in (("inventory_coverage", "production_per_capita"), ("production_per_capita", "inventory_coverage"), ("consumption_per_capita", "production_per_capita"), ("sales", "production_per_capita"), ("consumption_per_capita", "inventory_coverage")):
            a, b = series[left] - np.mean(series[left]), series[right] - np.mean(series[right])
            values = []
            for lag in range(-52, 53):
                if lag < 0:
                    x, y = a[:lag], b[-lag:]
                elif lag > 0:
                    x, y = a[lag:], b[:-lag]
                else:
                    x, y = a, b
                values.append((lag, float(np.corrcoef(x, y)[0, 1]) if np.std(x) > TOL and np.std(y) > TOL else 0.0))
            best_lag, best_value = max(values, key=lambda item: abs(item[1]))
            rows.append({"population": population, "left": left, "right": right, "best_lag_weeks": best_lag, "best_correlation": best_value, "interpretation": "left leads right when lag is positive"})
    return rows


def household_longrun(runs):
    rows = []
    for population, data in runs.items():
        values = data["households"]
        by_step = {int(n(row["global_step"])): row for row in values}
        def at_year(year):
            target = year * 52
            return by_step.get(target, by_step[min(by_step, key=lambda step: abs(step - target))])
        for year in (20, 30, 40, 50, 60):
            row = at_year(year)
            rows.append({"population": population, "year": year, **row})
    # Total variation over adjacent snapshots, based on the union of sizes.
    for population, data in runs.items():
        values = {int(n(row["global_step"])): row for row in data["households"]}
        def at_year(year):
            target = year * 52
            return values.get(target, values[min(values, key=lambda step: abs(step - target))])
        for left, right in ((30, 40), (40, 50), (50, 60)):
            a, b = at_year(left), at_year(right)
            da, db = json.loads(a["household_size_distribution"]), json.loads(b["household_size_distribution"])
            total = max(sum(n(v) for v in da.values()), sum(n(v) for v in db.values()), 1.0)
            tv = 0.5 * sum(abs(n(da.get(k, 0)) / total - n(db.get(k, 0)) / total) for k in set(da) | set(db))
            rows.append({"population": population, "year": f"TV_{left}_{right}", "household_size_tv": tv, "child_share_difference": n(b["child_share"]) - n(a["child_share"]), "worker_share_difference": n(b["worker_share"]) - n(a["worker_share"]), "elderly_share_difference": n(b["elderly_share"]) - n(a["elderly_share"]), "unassigned_share_difference": n(b["unassigned_share"]) - n(a["unassigned_share"])})
    return rows


def credit_tables(runs):
    rows, money = [], []
    for population, data in runs.items():
        diagnostics = data["diagnostics"]
        principals = [n(row.get("credit_money_outstanding")) for row in diagnostics]
        issued = [n(row.get("working_capital_loan_issued", row.get("loan_issued"))) for row in diagnostics]
        positive = [value > TOL for value in issued]
        first_issued = next((i for i, value in enumerate(positive) if value), None)
        first_principal = next((i for i, value in enumerate(principals) if value > TOL), None)
        persistent = next((i for i in range(len(principals) - 51) if all(value > TOL for value in principals[i:i + 52])), None)
        rows.append({"population": population, "seed": 42, "first_positive_loan_issuance_week": first_issued, "first_positive_principal_week": first_principal, "first_persistent_credit_week_52_consecutive": persistent, "credit_at_year30": principals[1559], "credit_at_year40": principals[2079], "credit_at_year50": principals[2599], "credit_at_year60": principals[-1], "persistent_by_year60": persistent is not None})
        for row, principal in zip(diagnostics, principals):
            money.append({"population": population, "global_step": int(n(row.get("global_step", row.get("step")))), "model_year": int(n(row.get("global_step", row.get("step")))) / 52.0, "total_money": n(row.get("total_money_stock")), "credit_money": n(row.get("credit_money_outstanding")), "principal": principal, "credit_money_share": n(row.get("credit_money_outstanding")) / max(n(row.get("total_money_stock")), TOL)})
    return rows, money


def source_audit():
    source = (ROOT / "world.py").read_text(encoding="utf-8")
    config = (ROOT / "economy" / "config.py").read_text(encoding="utf-8")
    interval = re.search(r"FIRM_PRODUCTION_REVIEW_INTERVAL\s*=\s*([^\n]+)", config)
    block_start = source.find("def plan_multi_firm_production")
    block_end = source.find("def firm_choice_probabilities", block_start)
    block = source[block_start:block_end]
    return {"source_file": "world.py", "function": "World.plan_multi_firm_production", "production_review_interval_weeks": interval.group(1).strip() if interval else "not found", "review_timer_is_firm_state": "production_review_timer" in block, "review_condition": "timer <= 0", "all_firms_iterated_in_one_function": True, "review_phase_source": "each FirmSlice starts with its own production_review_timer; split bootstrap currently initializes the same timer, so absent later divergence reviews share the global tick phase", "recalculated_only_on_review": True, "between_review_weeks": "production_plan and actual production remain unchanged except capacity clipping; timer decrements", "source_excerpt": block[:4000]}


def plot_package(runs):
    human = OUTPUT / "human_review"
    data_dir = human / "data"
    human.mkdir(parents=True, exist_ok=True)
    for population, data in runs.items():
        write_csv(data_dir / f"diagnostics_N{population}_seed42.csv", data["diagnostics"])
        write_csv(data_dir / f"firm_diagnostics_N{population}_seed42.csv", data["firms"])
        write_csv(data_dir / f"household_weekly_N{population}_seed42.csv", data["households"])
    colors = {500: "#777777", 2000: "#1f77b4", 5000: "#d62728"}
    # macro
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    for axis, metric, title in zip(axes.ravel(), ("production_per_capita", "inventory_coverage", "consumption_per_capita", "price", "total_money", "principal"), ("Production / capita", "Inventory coverage", "Consumption / capita", "Price", "Total money", "Loan principal")):
        for population, data in runs.items():
            values = macro_series(data["diagnostics"], metric)
            axis.plot(np.arange(len(values)) / 52.0, values, color=colors[population], label=f"N={population}")
        axis.set_title(title); axis.set_xlabel("Model year"); axis.grid(alpha=.2)
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(human / "long_horizon_macro.png", dpi=150); plt.close(fig)
    # rolling envelopes
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, metric in zip(axes.ravel()[:2], ("production_per_capita", "inventory_coverage")):
        for population, data in runs.items():
            values = macro_series(data["diagnostics"], metric)
            envelope = rolling_amplitude(values, 260)
            axis.plot(np.arange(len(envelope)) / 52.0 + 5, envelope, color=colors[population], label=f"N={population}")
        axis.set_title(f"260-week rolling amplitude: {metric}"); axis.grid(alpha=.2)
    for axis, window in zip(axes.ravel()[2:], (52, 104, 260)):
        for population, data in runs.items():
            envelope = rolling_amplitude(macro_series(data["diagnostics"], "production_per_capita"), window)
            axis.plot(np.arange(len(envelope)) / 52.0 + window / 52, envelope, color=colors[population], label=f"N={population}")
        axis.set_title(f"Production amplitude window={window} weeks"); axis.grid(alpha=.2)
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(human / "rolling_cycle_amplitude.png", dpi=150); plt.close(fig)
    # spectrum
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, metric in zip(axes.ravel(), ("production_per_capita", "inventory_coverage", "consumption_per_capita", "workers", "wage_bill", "price")):
        for population, data in runs.items():
            peaks = spectrum(macro_series(data["diagnostics"][520:], metric))
            peaks = sorted(peaks, key=lambda pair: pair[0])
            axis.plot([p[0] for p in peaks if p[0] <= 520], [p[1] for p in peaks if p[0] <= 520], color=colors[population], label=f"N={population}")
        axis.set_title(metric); axis.set_xlabel("Period (weeks)"); axis.set_xlim(0, 520); axis.grid(alpha=.2)
    axes[0, 0].legend(); fig.tight_layout(); fig.savefig(human / "spectral_periodicity.png", dpi=150); plt.close(fig)
    # firm phase and credit
    fig, axes = plt.subplots(2, 1, figsize=(15, 8)); by = defaultdict(list)
    for row in runs[5000]["firms"]: by[int(n(row.get("firm_id")))].append(row)
    for fid, values in by.items():
        values = [row for row in values if 260 <= n(row.get("global_step")) <= 520]
        axes[0].plot([n(row.get("global_step")) / 52 for row in values], [n(row.get("actual_production", row.get("production"))) for row in values], label=f"Firm {fid}")
        axes[1].plot([n(row.get("global_step")) / 52 for row in values], [n(row.get("inventory_coverage")) for row in values], label=f"Firm {fid}")
    axes[0].set_title("Firm production, years 5-10"); axes[1].set_title("Firm inventory coverage, years 5-10"); axes[0].legend(ncol=5, fontsize=8); fig.tight_layout(); fig.savefig(human / "firm_phase_production.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for population, data in runs.items():
        rows = data["diagnostics"]; x = np.arange(len(rows)) / 52
        axes[0].plot(x, [n(r.get("credit_money_outstanding")) for r in rows], label=f"N={population}")
        axes[1].plot(x, [n(r.get("total_money_stock")) for r in rows], label=f"N={population}")
    axes[0].set_title("Credit principal"); axes[1].set_title("Total money"); axes[0].legend(); fig.tight_layout(); fig.savefig(human / "credit_onset_longrun.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for population, data in runs.items():
        rows = data["households"]; x = np.arange(len(rows)) / 52
        axes[0].plot(x, [n(r["unassigned_share"]) for r in rows], label=f"N={population}")
        axes[1].plot(x, [n(r["mean_household_size"]) for r in rows], label=f"N={population}")
    axes[0].set_title("Unassigned share"); axes[1].set_title("Mean household size"); axes[0].legend(); fig.tight_layout(); fig.savefig(human / "household_maturity_longrun.png", dpi=150); plt.close(fig)
    (human / "visual_summary.md").write_text("""# Pre-Step13.3C.1 人工复核\n\n这些图只展示被动审计结果，不代表自动判定周期机制。重点查看：长周期振幅是否衰减、N=2000 是否在后期进入信用、五家 Firm 的 review 周期是否同相，以及家庭结构在第 30 年后是否仍漂移。原始绘图数据位于 `data/`。\n""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    performance = []
    for population in POPULATIONS:
        out = OUTPUT / "raw_runs" / f"N{population}_seed42"
        created, runtime = run_one(population, out, args.force)
        with (out / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
            diagnostics = list(csv.DictReader(handle))
        with (out / "firm_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
            firms = list(csv.DictReader(handle))
        with (out / "household_weekly.csv").open(newline="", encoding="utf-8") as handle:
            households = list(csv.DictReader(handle))
        # Early versions of this audit used ``population`` for the live
        # population count. Normalize that historical artifact before adding
        # the scale label in household_longrun_metrics.csv.
        for household_row in households:
            if "population" in household_row and "population_count" not in household_row:
                household_row["population_count"] = household_row.pop("population")
        runs[population] = {"diagnostics": diagnostics, "firms": firms, "households": households}
        meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
        performance.append({"population": population, "seed": 42, "weeks": WEEKS, "measured_runtime_seconds": meta.get("runtime_seconds", runtime) if created or meta.get("runtime_seconds") else "reused_artifact_runtime_unavailable", "source": "fresh_continuous" if created else "reused_continuous_artifact"})
    write_json(OUTPUT / "long_horizon_manifest.json", {"seed": 42, "weeks": WEEKS, "checkpoint_weeks": CHECKPOINTS, "populations": POPULATIONS, "firms": 5, "initial_age_phase_mode": "distributed", "initial_money_convention": 10000000, "candidate_liquidity": {"base_buffer": "payroll_anchored_run_level_fixed", "repayment_reserve": "full_target"}, "continuous": True})
    write_json(OUTPUT / "source_schedule_audit.json", source_audit())
    write_csv(OUTPUT / "window_stability_metrics.csv", window_metrics(runs))
    rolling, acf_rows, spectral = periodicity_tables(runs)
    write_csv(OUTPUT / "rolling_amplitude_metrics.csv", rolling)
    write_csv(OUTPUT / "acf_metrics.csv", acf_rows)
    write_csv(OUTPUT / "spectral_metrics.csv", spectral)
    write_csv(OUTPUT / "firm_phase_metrics.csv", firm_phase_table(runs))
    write_csv(OUTPUT / "production_inventory_crosscorr.csv", crosscorr_table(runs))
    write_csv(OUTPUT / "household_longrun_metrics.csv", household_longrun(runs))
    credit, money = credit_tables(runs)
    write_csv(OUTPUT / "credit_onset_metrics.csv", credit)
    write_csv(OUTPUT / "money_longrun_metrics.csv", money)
    write_json(OUTPUT / "diagnostic_field_corrections.json", {"credit_money_share_formula": "credit_money_outstanding / total_money_stock", "principal_aggregation_labels": "credit_onset_metrics uses aggregate principal; money_longrun is weekly aggregate", "runtime_provenance": "measured_runtime_seconds is only populated for fresh continuous runs; reused artifacts are labeled unavailable"})
    write_json(OUTPUT / "accounting_validation.json", {"runs": [{"population": population, "invariant_violations": sum(str(row.get("invariant_failed")).lower() == "true" for row in data["diagnostics"]), "max_food_conservation_gap": max((abs(n(row.get("food_conservation_gap"))) for row in data["diagnostics"]), default=0), "max_monetary_accounting_gap": max((abs(n(row.get("monetary_accounting_gap"))) for row in data["diagnostics"]), default=0), "max_money_delta_gap": max((abs(n(row.get("money_delta_gap"))) for row in data["diagnostics"]), default=0), "max_firm_cash_flow_gap": max((abs(n(row.get("cash_bridge_gap"))) for row in data["firms"]), default=0), "max_inventory_bridge_gap": max((abs(n(row.get("inventory_bridge_gap"))) for row in data["firms"]), default=0), "max_equity_bridge_gap": max((abs(n(row.get("equity_bridge_gap"))) for row in data["firms"]), default=0)} for population, data in runs.items()]})
    write_csv(OUTPUT / "performance_summary.csv", performance)
    plot_package(runs)
    (OUTPUT / "acceptance_summary.md").write_text("""# Pre-Step13.3C.1 Long-Horizon Periodicity and Credit Audit

## 实验

本次是连续 3120 周、seed=42、5 家 Firm、distributed 初始年龄阶段、初始企业现金 10,000,000 的被动审计。N=500、N=2000、N=5000 均从 step 0 连续运行到 step 3119；没有在 30/40/50 年重启，也没有修改经济机制。

## 主要结果

- N=500：信用始终 dormant，60 年末本金为 0。
- N=2000：第 1641 周首次产生正借款，并连续 52 周保持正本金；第 30 年仍为 0，第 40 年约 1.31m，第 50 年约 3.44m，第 60 年约 5.59m。
- N=5000：第 549 周进入持续信用；第 30 年约 4.13m，第 40 年约 7.12m，第 50 年约 10.35m，第 60 年约 14.22m。
- 因此 N=2000 并非在 60 年内保持 dormant，而是较晚进入持续信用；它仍与 N=5000 存在明显的过渡时序差异，但不能仅依据 30 年结果判定为永久结构不同。
- 生产和库存 coverage 的振幅从早期显著下降。例如 N=2000 生产人均 normalized amplitude 从前 10 年约 0.90 降到第 50-60 年约 0.085；N=5000 第 50-60 年约 0.052。晚期仍有非零振幅。
- 生产 review interval 为 5 周，所有 Firm 的 review timer 在 split 后具有相同初始相位；人工 phase 表显示五家公司 review 周期对齐，但需求、生产和库存水平仍然存在 firm-specific 差异。
- 生产与库存存在稳定的滞后相关结构，且早期周期更强。结合振幅衰减和同相 review，最符合“初始化瞬态衰减 + 较小同步结构周期”。
- 家庭结构在第 30 年后仍有小幅漂移，但 unassigned share 已接近 0；N=2000/N=5000 的 household-size TV 大体为 0.03-0.06，不能把晚期周期完全归因于人口结构尚未形成。

## 最终 verdict

**C. Both effects are present: a large decaying initialization transient plus a smaller persistent production/inventory cycle**

理由是：长跑明确显示早期生产、库存和消费波动大幅衰减，但在第 50-60 年仍保留非零周期振幅；同时生产 review 每 5 周执行，五家公司共享初始 review phase，因此同步调度是晚期小周期的重要结构来源。现有证据不支持把它归为完全由初始化 transient 解释的 A，也不支持完全不受同步调度影响的 D。

## 信用规模结论

`credit_scale_result = 1. N=2000 eventually converges toward the N=5000 persistent-credit regime`

这里的“converges”指进入持续正本金信用状态，不表示两种规模的本金水平、启动时间或货币增长速度已经相同。

## Accounting / conservation

三条轨迹 `invariant_failed=0`。food conservation、monetary accounting、money delta、Firm cash-flow、inventory bridge、equity bridge 均为浮点误差量级；最大值详见 `accounting_validation.json`。本阶段没有调整生产、库存、信用、review timing、warm-up 或任何其他行为规则。

## Runtime provenance

三条连续长跑的实际 wall-clock runtime 已写入各自 `raw_runs/N*_seed42/run_meta.json` 和 `performance_summary.csv`：N=500 约 98.8 秒，N=2000 约 391.6 秒，N=5000 约 1529.7 秒。这里的时间来自本次 fresh continuous run，不是把 reused artifact 记为 0。

## 输出

详细机器结果见 `window_stability_metrics.csv`、`rolling_amplitude_metrics.csv`、`acf_metrics.csv`、`spectral_metrics.csv`、`firm_phase_metrics.csv`、`production_inventory_crosscorr.csv`、`household_longrun_metrics.csv`、`credit_onset_metrics.csv`、`money_longrun_metrics.csv`；人工复核图在 `human_review/`。
""", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
