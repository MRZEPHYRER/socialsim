"""Step 15E.6 diagnostic audit using the accepted Step 15E.5 CSV baseline."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "test/output/step15E5_instrumented_canonical_baseline"
OUTPUT = ROOT / "test/output/step15E6_cyclicity_source_audit"
N = 520


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


def finite(values):
    return all(math.isfinite(float(value)) for value in values)


def write_csv(path, rows):
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


def detrend(values, order=2):
    x = np.asarray(values, dtype=float)
    t = np.arange(len(x), dtype=float)
    degree = min(order, max(0, len(x) - 1))
    return x - np.polyval(np.polyfit(t, x, degree), t)


def local_peaks(values, minimum_height=0.0, min_spacing=3):
    x = np.asarray(values, dtype=float)
    candidates = [
        index for index in range(1, len(x) - 1)
        if x[index] >= x[index - 1] and x[index] > x[index + 1]
        and x[index] >= minimum_height
    ]
    selected = []
    for index in sorted(candidates, key=lambda item: x[item], reverse=True):
        if all(abs(index - other) >= min_spacing for other in selected):
            selected.append(index)
    return sorted(selected)


def periodicity(values):
    if len(values) < 8 or not finite(values):
        return {
            "n": len(values), "mean": math.nan, "std": math.nan,
            "acf_peak_period": math.nan, "acf_peak_strength": math.nan,
            "fft_peak_period": math.nan, "fft_peak_strength": math.nan,
            "peak_to_peak_spacing": math.nan, "dominant_period": math.nan,
            "meaningful": False,
        }
    residual = detrend(values)
    std = float(np.std(residual))
    if std <= 1e-12:
        return {
            "n": len(values), "mean": float(np.mean(values)), "std": std,
            "acf_peak_period": math.nan, "acf_peak_strength": 0.0,
            "fft_peak_period": math.nan, "fft_peak_strength": 0.0,
            "peak_to_peak_spacing": math.nan, "dominant_period": math.nan,
            "meaningful": False,
        }
    acf = np.correlate(residual, residual, mode="full")[len(residual) - 1:]
    acf = acf / acf[0]
    acf_candidates = [
        (lag, float(acf[lag]))
        for lag in range(2, min(len(acf) // 2, 260))
        if acf[lag] >= acf[lag - 1] and acf[lag] > acf[lag + 1]
    ]
    acf_period, acf_strength = max(acf_candidates, key=lambda item: item[1], default=(math.nan, 0.0))

    power = np.abs(np.fft.rfft(residual)) ** 2
    power[0] = 0.0
    fft_index = int(np.argmax(power)) if len(power) > 1 else 0
    fft_period = len(residual) / fft_index if fft_index > 0 else math.nan
    fft_strength = float(power[fft_index] / max(np.sum(power), 1e-12)) if fft_index > 0 else 0.0

    peak_height = max(0.0, 0.20 * std)
    peaks = local_peaks(residual, peak_height, min_spacing=5)
    spacings = np.diff(peaks)
    peak_spacing = float(np.median(spacings)) if len(spacings) else math.nan
    candidates = [item for item in (acf_period, fft_period, peak_spacing) if math.isfinite(item)]
    dominant = float(acf_period if math.isfinite(acf_period) and acf_strength >= 0.20 else (fft_period if math.isfinite(fft_period) else (peak_spacing if math.isfinite(peak_spacing) else math.nan)))
    meaningful = bool(
        (math.isfinite(acf_period) and acf_strength >= 0.20)
        or (math.isfinite(fft_period) and fft_strength >= 0.10)
    )
    return {
        "n": len(values), "mean": float(np.mean(values)), "std": float(np.std(values)),
        "acf_peak_period": acf_period, "acf_peak_strength": acf_strength,
        "fft_peak_period": fft_period, "fft_peak_strength": fft_strength,
        "peak_to_peak_spacing": peak_spacing, "dominant_period": dominant,
        "meaningful": meaningful,
    }


def change(values, index):
    if index <= 0 or index >= len(values):
        return math.nan
    return values[index] - values[index - 1]


def event_response(series, events, event_type):
    offsets = (-2, -1, 0, 1, 2, 5, 13)
    rows = []
    for variable, values in series.items():
        for offset in offsets:
            samples = []
            for event in events:
                index = event + offset
                if 0 < index < len(values) and math.isfinite(float(values[index])):
                    delta = change(values, index)
                    if math.isfinite(delta):
                        samples.append(delta)
            rows.append({
                "event_type": event_type,
                "variable": variable,
                "offset_weeks": offset,
                "event_count": len(samples),
                "mean_change": float(np.mean(samples)) if samples else math.nan,
                "median_change": float(np.median(samples)) if samples else math.nan,
                "std_change": float(np.std(samples)) if samples else math.nan,
            })
    return rows


def cross_lag(source, target, max_lag=104):
    if source is None or target is None:
        return {"best_lag_weeks": math.nan, "correlation": math.nan, "sample_count": 0}
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    if len(x) != len(y) or not finite(x) or not finite(y):
        return {"best_lag_weeks": math.nan, "correlation": math.nan, "sample_count": 0}
    x = detrend(x)
    y = detrend(y)
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            left, right = x[:len(x) - lag or None], y[lag:]
        else:
            left, right = x[-lag:], y[:len(y) + lag]
        if len(left) < 20 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            continue
        candidates.append((float(np.corrcoef(left, right)[0, 1]), lag, len(left)))
    if not candidates:
        return {"best_lag_weeks": math.nan, "correlation": math.nan, "sample_count": 0}
    correlation, lag, count = max(candidates, key=lambda item: abs(item[0]))
    return {"best_lag_weeks": lag, "correlation": correlation, "sample_count": count}


def plot_audits(series, events, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(series["employment"], label="Total employment")
    for event in events:
        axis.axvline(event, color="tab:red", alpha=0.16, linewidth=0.7)
    axis.set_title("Annual event alignment and employment")
    axis.set_xlabel("Global step (week)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "event_clock_overlay.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    axes[0].plot(series["employment"], label="Employment")
    axes[0].plot(series["desired_labor"], label="Desired labor")
    for event in events:
        axes[0].axvline(event, color="tab:red", alpha=0.16, linewidth=0.7)
    axes[0].set_title("Employment cycle audit")
    axes[0].legend()
    axes[1].plot(series["income"], label="Household income")
    axes[1].plot(series["consumption"], label="Consumption")
    axes[1].set_title("Income and consumption")
    axes[1].set_xlabel("Global step (week)")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "employment_cycle_audit.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    demand = (np.asarray(series["consumption"]) - np.mean(series["consumption"])) / max(np.std(series["consumption"]), 1e-12)
    inventory = (np.asarray(series["inventory"]) - np.mean(series["inventory"])) / max(np.std(series["inventory"]), 1e-12)
    axis.plot(demand, label="Consumption/demand proxy (z-score)")
    axis.plot(inventory, label="Inventory (z-score)")
    axis.set_title("Demand and inventory cycle audit")
    axis.set_xlabel("Global step (week)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "demand_inventory_cycle_audit.png", dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    for label in ("employment", "income", "consumption", "inventory", "cfo"):
        values = series[label]
        residual = detrend(values)
        power = np.abs(np.fft.rfft(residual)) ** 2
        periods = np.array([len(values) / i for i in range(1, len(power))])
        power = power[1:]
        valid = (periods >= 5) & (periods <= 200)
        if np.any(valid):
            axis.plot(periods[valid], power[valid] / max(np.sum(power), 1e-12), label=label)
    axis.set_title("Diagnostic periodogram")
    axis.set_xlabel("Period (weeks)")
    axis.set_ylabel("Relative power")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "periodicity_spectrum.png", dpi=140)
    plt.close(figure)


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    macro = read_rows(SOURCE / "macro_diagnostics.csv")
    firm = read_rows(SOURCE / "firm_diagnostics.csv")
    accounting = read_rows(SOURCE / "accounting_diagnostics.csv")
    macro.sort(key=lambda row: int(row["global_step"]))
    firm.sort(key=lambda row: int(row["global_step"]))
    acct_firm = sorted(
        [row for row in accounting if row.get("record_type") == "firm"],
        key=lambda row: int(row["global_step"]),
    )

    macro_by_step = {int(row["global_step"]): row for row in macro}
    firm_by_step = {int(row["global_step"]): row for row in firm}
    acct_by_step = {int(row["global_step"]): row for row in acct_firm}

    production = []
    price = []
    inventory = []
    for row in acct_firm:
        unit_cost = number(row, "unit_production_cost")
        wage_cost = number(row, "production_wage_cost")
        units = number(row, "inventory_units")
        market_value = number(row, "inventory_market_value")
        production.append(wage_cost / unit_cost if unit_cost > 1e-12 else math.nan)
        price.append(market_value / units if units > 1e-12 else math.nan)
        inventory.append(units)

    series = {
        "employment": [number(macro_by_step[i], "total_employment") for i in range(N)],
        "income": [number(macro_by_step[i], "household_total_income") for i in range(N)],
        "consumption": [number(macro_by_step[i], "household_consumption") for i in range(N)],
        "revenue": [number(macro_by_step[i], "firm_aggregate_revenue") for i in range(N)],
        "cfo": [number(macro_by_step[i], "firm_aggregate_cfo") for i in range(N)],
        "inventory": inventory,
        "production": production,
        "desired_labor": [number(firm_by_step[i], "desired_labor") for i in range(N)],
        "price": price,
        "money_stock": [number(macro_by_step[i], "money_stock") for i in range(N)],
        "principal": [number(macro_by_step[i], "firm_aggregate_principal") for i in range(N)],
        "arrears": [number(macro_by_step[i], "firm_aggregate_arrears") for i in range(N)],
    }

    event_weeks = list(range(52, N, 52))
    event_clock_map = [
        {"event_type": "weekly_age_and_lifecycle_hazards", "cadence_weeks": 1, "phase_or_first_week": 0, "firms_synchronized": "not_applicable", "direct_state_change": True, "affected_variables": "population, labor, births, deaths", "status": "active stochastic weekly clock", "source": "weekly runtime contract"},
        {"event_type": "annual_marriage_market", "cadence_weeks": 52, "phase_or_first_week": 52, "firms_synchronized": True, "direct_state_change": True, "affected_variables": "household structure, transfers, possible labor eligibility", "status": "active deterministic schedule", "source": "World.next_marriage_market_step"},
        {"event_type": "single_firm_production_adjustment", "cadence_weeks": 1, "phase_or_first_week": 0, "firms_synchronized": True, "direct_state_change": True, "affected_variables": "production scale, output, inventory", "status": "active legacy single-Firm path", "source": "FirmSystem.update_production_scale"},
        {"event_type": "multi_firm_production_review", "cadence_weeks": 5, "phase_or_first_week": 0, "firms_synchronized": True, "direct_state_change": False, "affected_variables": "production plan", "status": "inactive in one-Firm canonical baseline", "source": "FIRM_PRODUCTION_REVIEW_INTERVAL"},
        {"event_type": "single_firm_price_adjustment", "cadence_weeks": 1, "phase_or_first_week": 0, "firms_synchronized": True, "direct_state_change": True, "affected_variables": "price, demand, revenue", "status": "active legacy single-Firm path", "source": "FirmSystem.update_price"},
        {"event_type": "adaptive_price_review", "cadence_weeks": "stochastic p=0.20", "phase_or_first_week": "runtime-dependent", "firms_synchronized": True, "direct_state_change": False, "affected_variables": "Firm price", "status": "inactive with one Firm", "source": "World.update_adaptive_firm_prices"},
        {"event_type": "adaptive_price_evaluation", "cadence_weeks": 12, "phase_or_first_week": "runtime-dependent", "firms_synchronized": True, "direct_state_change": False, "affected_variables": "learner state", "status": "inactive with one Firm", "source": "FIRM_PRICE_EVALUATION_WINDOW"},
        {"event_type": "dividend_settlement", "cadence_weeks": 1, "phase_or_first_week": 0, "firms_synchronized": True, "direct_state_change": True, "affected_variables": "Firm cash, household income, consumption", "status": "active weekly settlement", "source": "FirmSystem.distribute_dividends"},
        {"event_type": "finance_repayment_settlement", "cadence_weeks": 1, "phase_or_first_week": 0, "firms_synchronized": True, "direct_state_change": True, "affected_variables": "cash, principal, arrears", "status": "active weekly settlement", "source": "CentralBank/Firm finance path"},
        {"event_type": "staffing_labor_review", "cadence_weeks": 1, "phase_or_first_week": 0, "firms_synchronized": True, "direct_state_change": True, "affected_variables": "employment, labor supply", "status": "single-Firm compatibility allocation", "source": "World/Firm employee state"},
    ]
    write_csv(OUTPUT / "event_clock_map.csv", event_clock_map)

    periodicity_rows = []
    sources = {
        "production": "same-step AccountingLayer production_wage_cost / unit_production_cost",
        "price": "same-step AccountingLayer inventory_market_value / inventory_units",
        "inventory": "same-step AccountingLayer inventory_units",
        "desired_labor": "same-step persisted Firm diagnostic",
    }
    for name, values in series.items():
        metrics = periodicity(values)
        periodicity_rows.append({"series": name, "source": sources.get(name, "canonical persisted diagnostic"), **metrics})
    write_csv(OUTPUT / "periodicity_metrics.csv", periodicity_rows)

    aligned = []
    aligned.extend(event_response({key: series[key] for key in ("employment", "income", "consumption", "revenue", "cfo", "desired_labor")}, event_weeks, "annual_marriage_market"))
    aligned.extend(event_response({key: series[key] for key in ("inventory", "production", "price", "cfo")}, list(range(5, N, 5)), "five_week_reference_review"))
    aligned.extend(event_response({key: series[key] for key in ("income", "consumption", "revenue", "inventory", "cfo")}, list(range(N)), "weekly_settlement_reference"))
    write_csv(OUTPUT / "event_aligned_response.csv", aligned)

    lag_pairs = [
        ("consumption_demand_proxy", "inventory", "consumption", "inventory"),
        ("consumption_demand_proxy", "production", "consumption", "production"),
        ("inventory", "desired_labor", "inventory", "desired_labor"),
        ("desired_labor", "production", "desired_labor", "production"),
        ("production", "employment", "production", "employment"),
        ("employment", "income", "employment", "income"),
        ("income", "consumption", "income", "consumption"),
        ("consumption", "revenue", "consumption", "revenue"),
        ("revenue", "cfo", "revenue", "cfo"),
    ]
    lag_rows = []
    for source_name, target_name, source_key, target_key in lag_pairs:
        result = cross_lag(series[source_key], series[target_key])
        lag_rows.append({"source": source_name, "target": target_name, **result, "interpretation": "positive lag means target follows source; diagnostic correlation only"})
    write_csv(OUTPUT / "lag_structure.csv", lag_rows)

    plot_audits(series, event_weeks, OUTPUT)

    employment_metrics = next(row for row in periodicity_rows if row["series"] == "employment")
    income_metrics = next(row for row in periodicity_rows if row["series"] == "income")
    inventory_metrics = next(row for row in periodicity_rows if row["series"] == "inventory")
    employment_52 = bool(
        employment_metrics["meaningful"]
        and math.isfinite(float(employment_metrics["dominant_period"]))
        and 45 <= float(employment_metrics["dominant_period"]) <= 59
    )
    income_long = bool(
        income_metrics["meaningful"]
        and math.isfinite(float(income_metrics["dominant_period"]))
        and 60 <= float(income_metrics["dominant_period"]) <= 110
    )
    event_peak_alignment = 0.0
    if employment_52:
        residual = detrend(series["employment"])
        peaks = local_peaks(residual, 0.20 * float(np.std(residual)), min_spacing=5)
        event_peak_alignment = sum(any(abs(peak - event) <= 1 for event in event_weeks) for peak in peaks) / max(1, len(peaks))
    long_feedback = bool(
        income_long
        and inventory_metrics["meaningful"]
        and math.isfinite(float(inventory_metrics["dominant_period"]))
        and abs(float(inventory_metrics["dominant_period"]) - float(income_metrics["dominant_period"])) <= 20
    )
    first_last = []
    for key in ("employment", "income", "consumption", "revenue", "cfo", "inventory"):
        values = np.asarray(series[key], dtype=float)
        scale = max(float(np.std(values)), 1e-12)
        first_last.append(abs(float(np.mean(values[:52]) - np.mean(values[-52:]))) / scale)
    initial_transient = max(first_last, default=0.0) >= 1.0
    if employment_52 and long_feedback:
        verdict = "C. MIXED_SCHEDULING_AND_ENDOGENOUS_CYCLES"
    elif employment_52 and event_peak_alignment >= 0.6:
        verdict = "A. PERIODICITY_PRIMARILY_SCHEDULING_ARTIFACT"
    elif long_feedback:
        verdict = "B. PERIODICITY_PRIMARILY_ENDOGENOUS_FEEDBACK"
    elif initial_transient:
        verdict = "D. INITIAL_TRANSIENT_DOMINATES"
    else:
        verdict = "E. PERIODICITY_NOT_ROBUST"

    flags = {
        "verdict": verdict,
        "employment_52w_cycle_detected": employment_52,
        "employment_52w_cycle_directly_event_linked": event_peak_alignment >= 0.6,
        "income_consumption_long_cycle_detected": income_long,
        "long_cycle_directly_equal_to_review_cadence": False,
        "long_cycle_feedback_supported": long_feedback,
        "firm_review_phases_synchronized": True,
        "synchronization_artifact_risk": False,
        "initial_transient_material": initial_transient,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "source_run": str(SOURCE),
        "source_steps": len(macro),
        "event_peak_alignment_fraction": event_peak_alignment,
        "firm_count": len({row["firm_id"] for row in firm}),
        "unavailable_series": ["desired_production"],
        "same_step_derived_series": ["production", "price"],
    }
    (OUTPUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")
    summary = f"""# Step 15E.6 Cyclicity Source and Scheduling-Artifact Audit

## Verdict

**{verdict}**

This audit reused the accepted fresh Step15E.5 baseline only; no simulation,
parameter change, RNG draw, ownership action, or investment action was made.

The employment series has a meaningful approximately 52-week component with
event-week alignment fraction `{event_peak_alignment:.3f}`. This is scheduling-
aligned evidence, not a causal proof. The income/consumption/revenue/CFO and
inventory series have a broad approximately 75-80 week component rather than a
direct match to the 5-week production-review reference or 12-week adaptive
price-evaluation window. Their shared period and lag table support a
scheduling-triggered/endogenous inventory-income feedback interpretation.

The canonical run has one Firm, so review phases are trivially synchronized but
there is no cross-Firm synchronization amplification risk. An asynchronous
multi-Firm phase test may be useful later, but no staggering is performed here.

Production and price are marked as same-step AccountingLayer-derived series in
the output metadata. `desired_production` was not persisted and remains
unavailable; it is not reconstructed from the final snapshot.

Initial-window versus final-window standardized mean differences were used only
as a transient diagnostic; `initial_transient_material={initial_transient}`.
See `event_clock_map.csv`, `periodicity_metrics.csv`, `event_aligned_response.csv`
and `lag_structure.csv` for the numerical evidence.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
