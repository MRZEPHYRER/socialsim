"""Step 15 accounting-scope fix and deterministic phase-staggering screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world import World


OUT = ROOT / "test/output/step15_dynamic_cleanup_scheduler_desync"
TOL = 1e-6
WEEKS = 520
SEED = 42
FOOD_FIRMS = 5


def write_csv(name, rows):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def value(item, default=0.0):
    try:
        return float(item if item not in (None, "") else default)
    except (TypeError, ValueError):
        return float(default)


def slope(values):
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return math.nan
    return float(np.polyfit(np.arange(len(array)), array, 1)[0])


def autocorr(values, lag):
    array = np.asarray(values, dtype=float)
    if len(array) <= lag or np.std(array[:-lag]) <= TOL or np.std(array[lag:]) <= TOL:
        return math.nan
    return float(np.corrcoef(array[:-lag], array[lag:])[0, 1])


def cadence_power(values, cadence):
    array = np.asarray(values, dtype=float)
    if len(array) < 8 or np.std(array) <= TOL:
        return 0.0
    x = np.arange(len(array), dtype=float)
    detrended = array - np.polyval(np.polyfit(x, array, 1), x)
    power = np.abs(np.fft.rfft(detrended)) ** 2
    frequencies = np.fft.rfftfreq(len(array), d=1.0)
    target = 1.0 / cadence
    index = int(np.argmin(np.abs(frequencies - target)))
    denominator = float(power[1:].sum())
    return float(power[index] / denominator) if denominator > TOL else 0.0


def longest_plateau(values):
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return 0
    threshold = TOL * max(1.0, float(np.max(np.abs(array))))
    longest = current = 1
    for unchanged in np.abs(np.diff(array)) <= threshold:
        current = current + 1 if unchanged else 1
        longest = max(longest, current)
    return int(longest)


def accepted_overrides(staggered):
    return {
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "CANONICAL_INVESTMENT_ENABLED": True,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": True,
        "CAPITAL_LIFECYCLE_ENABLED": True,
        "CAPITAL_LIFECYCLE_USEFUL_LIFE_WEEKS": 52.0,
        "CAPITAL_GOOD_FIXTURE_PRODUCTIVITY_PER_LABOR": 3.0,
        "CAPITAL_GOOD_OFFER_PRICE_MODE": "cost_anchored",
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": True,
        "DETERMINISTIC_FIRM_REVIEW_PHASE_STAGGERING_ENABLED": staggered,
    }


def assignment_violations(world):
    seen = defaultdict(int)
    violations = 0
    for firm in world.operating_firms():
        for person_id in getattr(firm, "employee_ids", []):
            seen[person_id] += 1
            person = world.person_dict.get(person_id)
            if person is None or person.firm_id != firm.firm_id:
                violations += 1
    return violations + sum(max(0, count - 1) for count in seen.values())


def run_case(mode, population, weeks):
    staggered = mode == "treatment"
    world = World(
        initial_population=population,
        seed=SEED,
        diagnostics_mode="full",
        scenario_name=f"step15_dynamic_cleanup_{mode}",
        scenario_overrides=accepted_overrides(staggered),
    )
    world.split_firms(FOOD_FIRMS)
    system = world.canonical_investment_system
    system.ensure_firms()
    registry = []
    for firm in world.firms:
        registry.extend([
            {
                "mode": mode,
                "firm_id": firm.firm_id,
                "process": "food_production_review",
                "cadence_weeks": 5,
                "phase_offset": int(getattr(firm, "production_review_phase_offset", 0)),
                "phase_source": "firm_id*7 + process_code modulo cadence",
                "runtime_rng_draws": 0,
            },
            {
                "mode": mode,
                "firm_id": firm.firm_id,
                "process": "investment_review",
                "cadence_weeks": 13,
                "phase_offset": int(getattr(firm, "investment_review_phase_offset", 0)),
                "phase_source": "firm_id*7 + process_code modulo cadence",
                "runtime_rng_draws": 0,
            },
            {
                "mode": mode,
                "firm_id": firm.firm_id,
                "process": "labor_assignment_release",
                "cadence_weeks": 1,
                "phase_offset": 0,
                "phase_source": "weekly settlement; intentionally not staggered",
                "runtime_rng_draws": 0,
            },
        ])
    registry.append({
        "mode": mode,
        "firm_id": 100000,
        "process": "capital_good_backlog_supply_review",
        "cadence_weeks": 1,
        "phase_offset": 0,
        "phase_source": "weekly stock-to-flow update; intentionally not staggered",
        "runtime_rng_draws": 0,
    })

    rows = []
    production_review_weeks = defaultdict(list)
    investment_review_weeks = defaultdict(list)
    for _ in range(weeks):
        world.step()
        week = world.current_step_index
        raw = world.diagnostics_rows[-1]
        cap_week = world.last_canonical_investment_week
        food_firms = list(world.firms)
        capital_firms = list(world.capital_good_firms)
        all_firms = [*food_firms, *capital_firms]
        for firm in food_firms:
            if getattr(firm, "production_reviewed", False):
                production_review_weeks[firm.firm_id].append(week)
            if getattr(firm, "investment_reviewed_this_step", False):
                investment_review_weeks[firm.firm_id].append(week)
        accounting = world.accounting
        recon = accounting.reconciliation_rows[-1]
        firm_accounting = [
            row for row in accounting.rows
            if int(value(row.get("global_step", row.get("step", -1)), -1)) == week
        ]
        liability = math.fsum(value(row.get("customer_advance_liability")) for row in firm_accounting)
        prepaid = math.fsum(value(row.get("prepaid_capital_investment_asset")) for row in firm_accounting)
        active_service = math.fsum(
            value(
                getattr(getattr(firm, "capital_stock", None), "capital_service_capacity", lambda **_: 0.0)(
                    engineering_capacity_per_unit=1.0
                )
            )
            for firm in food_firms
        )
        expansion_backlog = math.fsum(
            max(0.0, value(item))
            for item in system.expansion_backlog_by_firm.values()
        )
        replacement_backlog = math.fsum(
            max(0.0, value(getattr(firm, "pending_replacement_capacity_need", 0.0)))
            for firm in food_firms
        )
        food_employment = sum(len(firm.employee_ids) for firm in food_firms)
        capital_employment = sum(len(firm.employee_ids) for firm in capital_firms)
        firm_cash = math.fsum(value(getattr(firm, "cash", 0.0)) for firm in all_firms)
        food_cash = math.fsum(value(getattr(firm, "cash", 0.0)) for firm in food_firms)
        capital_cash = math.fsum(value(getattr(firm, "cash", 0.0)) for firm in capital_firms)
        rows.append({
            "mode": mode,
            "week": week,
            "population": value(raw.get("population")),
            "total_employment": food_employment + capital_employment,
            "food_employment": food_employment,
            "capital_good_employment": capital_employment,
            "household_income": value(raw.get("total_income")),
            "household_consumption": value(raw.get("total_consumption")),
            "food_demand": value(raw.get("food_demand_units")),
            "food_production": value(raw.get("food_output_units", raw.get("actual_production"))),
            "total_fixed_investment": value(getattr(cap_week, "fixed_investment", 0.0)),
            "capital_good_production": value(getattr(cap_week, "capital_good_production", 0.0)),
            "new_customer_advances": value(getattr(cap_week, "customer_advances_received", 0.0)),
            "backlog": expansion_backlog + replacement_backlog,
            "active_capital_service": active_service,
            "household_cash": value(raw.get("total_household_wealth")),
            "aggregate_firm_cash": firm_cash,
            "food_firm_cash": food_cash,
            "capital_good_firm_cash": capital_cash,
            "legacy_owner_cash": value(getattr(world, "legacy_owner_cash", 0.0)),
            "estate_cash": math.fsum(value(getattr(account, "cash", 0.0)) for account in getattr(world, "estate_accounts", {}).values()),
            "total_non_household_owner_cash": value(recon.get("total_non_household_owner_cash")),
            "total_money_liabilities": value(recon.get("total_money_liabilities")),
            "full_money_location_gap": value(recon.get("full_money_location_gap")),
            "accounting_gap": max(
                [abs(value(row.get(field))) for row in firm_accounting for field in (
                    "cash_flow_gap", "balance_sheet_gap", "inventory_bridge_gap",
                    "capital_book_value_bridge_gap", "customer_advance_liability_bridge_gap",
                    "prepaid_investment_bridge_gap",
                )] or [0.0]
            ),
            "goods_gap": abs(value(raw.get("food_conservation_gap"))),
            "assignment_violations": assignment_violations(world),
            "output_above_feasible_capacity": max(
                [
                    max(
                        0.0,
                        value(getattr(firm, "actual_production", 0.0))
                        - value(getattr(firm, "authoritative_feasible_capacity", getattr(firm, "feasible_capacity", 0.0))),
                    )
                    for firm in food_firms
                ] or [0.0]
            ),
            "customer_advance_liability": liability,
            "prepaid_investment_asset": prepaid,
            "advance_prepaid_gap": liability - prepaid,
        })
    reviews = {
        "production": production_review_weeks,
        "investment": investment_review_weeks,
    }
    return pd.DataFrame(rows), registry, reviews


def review_rows(control_reviews, treatment_reviews):
    rows = []
    for process, cadence in (("production", 5), ("investment", 13)):
        for firm_id in range(FOOD_FIRMS):
            control = control_reviews[process][firm_id]
            treatment = treatment_reviews[process][firm_id]
            for mode, weeks in (("control", control), ("treatment", treatment)):
                intervals = np.diff(weeks)
                rows.append({
                    "process": process,
                    "firm_id": firm_id,
                    "mode": mode,
                    "cadence_weeks": cadence,
                    "review_count": len(weeks),
                    "first_review_week": weeks[0] if weeks else "",
                    "last_review_week": weeks[-1] if weeks else "",
                    "minimum_interval": int(intervals.min()) if len(intervals) else "",
                    "maximum_interval": int(intervals.max()) if len(intervals) else "",
                    "all_intervals_exact": bool(len(intervals) == 0 or np.all(intervals == cadence)),
                    "count_difference_vs_other_mode": len(treatment) - len(control),
                    "boundary_effect_within_one_review": abs(len(treatment) - len(control)) <= 1,
                })
    return rows


def metric_comparisons(control, treatment):
    metrics = (
        "total_employment", "food_employment", "capital_good_employment",
        "household_income", "household_consumption", "food_demand",
        "food_production", "total_fixed_investment", "capital_good_production",
        "new_customer_advances", "backlog", "active_capital_service",
        "household_cash", "aggregate_firm_cash",
    )
    windows = {
        "full": (0, len(control)),
        "late_104": (max(0, len(control) - 104), len(control)),
        "final_quarter": (3 * len(control) // 4, len(control)),
    }
    rows = []
    for metric in metrics:
        for window, (start, end) in windows.items():
            c = control[metric].to_numpy()[start:end]
            t = treatment[metric].to_numpy()[start:end]
            cmean = float(np.mean(c))
            tmean = float(np.mean(t))
            rows.append({
                "metric": metric,
                "window": window,
                "control_mean": cmean,
                "treatment_mean": tmean,
                "relative_mean_shift": (tmean - cmean) / max(abs(cmean), TOL),
                "control_final": float(c[-1]),
                "treatment_final": float(t[-1]),
                "control_slope": slope(c),
                "treatment_slope": slope(t),
            })
    return rows


def periodicity_rows(control, treatment):
    metrics = (
        "total_employment", "capital_good_employment", "household_income",
        "household_consumption", "food_demand", "food_production",
        "total_fixed_investment", "capital_good_production",
        "new_customer_advances", "backlog",
    )
    rows = []
    for mode, frame in (("control", control), ("treatment", treatment)):
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            diff = np.diff(values, prepend=values[0])
            rows.append({
                "mode": mode,
                "metric": metric,
                "lag13_autocorrelation_first_difference": autocorr(diff, 13),
                "lag52_autocorrelation_first_difference": autocorr(diff, 52),
                "spectral_power_13w_first_difference": cadence_power(diff, 13),
                "spectral_power_52w_first_difference": cadence_power(diff, 52),
                "rolling_13w_volatility_mean": float(pd.Series(values).rolling(13, min_periods=2).std().fillna(0).mean()),
                "longest_plateau_weeks": longest_plateau(values),
            })
    return rows


def jump_rows(control, treatment):
    metrics = (
        "total_employment", "capital_good_employment", "household_income",
        "household_consumption", "food_demand", "food_production",
        "total_fixed_investment", "capital_good_production",
        "new_customer_advances", "backlog",
    )
    rows = []
    for mode, frame in (("control", control), ("treatment", treatment)):
        weeks = frame.week.to_numpy(dtype=int)
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            diff = np.diff(values, prepend=values[0])
            absolute = np.abs(diff)
            total = float(absolute.sum())
            top_count = max(1, int(math.ceil(len(absolute) * 0.10)))
            rows.append({
                "mode": mode,
                "metric": metric,
                "largest_absolute_first_difference": float(absolute.max()),
                "p95_absolute_first_difference": float(np.percentile(absolute, 95)),
                "top_10pct_jump_concentration": float(np.sort(absolute)[-top_count:].sum() / total) if total > TOL else 0.0,
                "absolute_change_share_on_13w_boundaries": float(absolute[weeks % 13 == 0].sum() / total) if total > TOL else 0.0,
                "absolute_change_share_on_52w_boundaries": float(absolute[weeks % 52 == 0].sum() / total) if total > TOL else 0.0,
                "longest_plateau_weeks": longest_plateau(values),
            })
    return rows


def cash_rows(control, treatment):
    rows = []
    windows = {
        "full": (0, len(control)),
        "weeks_100_plus": (min(100, len(control)), len(control)),
        "second_half": (len(control) // 2, len(control)),
        "final_quarter": (3 * len(control) // 4, len(control)),
    }
    for mode, frame in (("control", control), ("treatment", treatment)):
        for window, (start, end) in windows.items():
            household = slope(frame.household_cash.to_numpy()[start:end])
            firm = slope(frame.aggregate_firm_cash.to_numpy()[start:end])
            legacy = slope(frame.legacy_owner_cash.to_numpy()[start:end])
            rows.append({
                "mode": mode,
                "window": window,
                "start_week": start,
                "end_week": end - 1,
                "household_cash_slope_per_week": household,
                "firm_cash_slope_per_week": firm,
                "legacy_cash_slope_per_week": legacy,
                "combined_sector_cash_slope": household + firm + legacy,
                "classification": "PERSISTENT_FIRM_CASH_DRAIN_NOT_FIXED_IN_THIS_STAGE",
            })
    return rows


def main(population, weeks, permanent):
    control, control_registry, control_reviews = run_case("control", population, weeks)
    treatment, treatment_registry, treatment_reviews = run_case("treatment", population, weeks)
    review_invariance = review_rows(control_reviews, treatment_reviews)
    comparisons = metric_comparisons(control, treatment)
    periodicity = periodicity_rows(control, treatment)
    jump = jump_rows(control, treatment)
    cash = cash_rows(control, treatment)

    money_rows = []
    for frame in (control, treatment):
        money_rows.extend(frame[[
            "mode", "week", "household_cash", "aggregate_firm_cash",
            "legacy_owner_cash", "estate_cash", "total_non_household_owner_cash",
            "total_money_liabilities", "full_money_location_gap", "accounting_gap",
            "goods_gap", "assignment_violations", "output_above_feasible_capacity",
            "customer_advance_liability", "prepaid_investment_asset", "advance_prepaid_gap",
        ]].to_dict("records"))

    if not permanent:
        print(json.dumps({
            "population": population,
            "weeks": weeks,
            "max_full_money_gap": max(abs(row["full_money_location_gap"]) for row in money_rows),
            "review_invariance": all(row["all_intervals_exact"] for row in review_invariance),
        }, indent=2))
        return

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("full_money_location_reconciliation.csv", money_rows)
    write_csv("scheduler_phase_registry.csv", control_registry + treatment_registry)
    write_csv("review_count_invariance.csv", review_invariance)
    write_csv("control_treatment_dynamic_comparison.csv", comparisons)
    write_csv("periodicity_control_treatment.csv", periodicity)
    write_csv("jump_control_treatment.csv", jump)
    write_csv("macro_cash_slope_recheck.csv", cash)

    max_full_gap = max(abs(row["full_money_location_gap"]) for row in money_rows)
    max_accounting = max(abs(row["accounting_gap"]) for row in money_rows)
    max_goods = max(abs(row["goods_gap"]) for row in money_rows)
    max_advance = max(abs(row["advance_prepaid_gap"]) for row in money_rows)
    max_assignment = max(row["assignment_violations"] for row in money_rows)
    max_feasibility = max(row["output_above_feasible_capacity"] for row in money_rows)
    review_pass = all(
        row["all_intervals_exact"] and row["boundary_effect_within_one_review"]
        for row in review_invariance
    )
    late = {
        row["metric"]: row
        for row in comparisons
        if row["window"] == "late_104"
    }
    core_level_metrics = (
        "total_employment", "household_income", "household_consumption",
        "food_production", "active_capital_service",
    )
    max_core_shift = max(abs(late[metric]["relative_mean_shift"]) for metric in core_level_metrics)
    tracked_level_metrics = (
        "total_employment", "household_income", "household_consumption",
        "food_production", "total_fixed_investment", "active_capital_service",
        "backlog", "household_cash", "aggregate_firm_cash",
    )
    tracked_level_shifts = {
        metric: abs(late[metric]["relative_mean_shift"])
        for metric in tracked_level_metrics
    }
    max_tracked_shift_metric = max(tracked_level_shifts, key=tracked_level_shifts.get)
    max_tracked_shift = tracked_level_shifts[max_tracked_shift_metric]
    jump_lookup = {(row["mode"], row["metric"]): row for row in jump}
    jump_ratios = {
        metric: (
            jump_lookup[("treatment", metric)]["largest_absolute_first_difference"]
            / max(jump_lookup[("control", metric)]["largest_absolute_first_difference"], TOL)
        )
        for metric in (
            "total_employment", "capital_good_employment", "food_production",
            "new_customer_advances", "total_fixed_investment",
        )
    }
    materially_weakened = sum(ratio < 0.80 for ratio in jump_ratios.values()) >= 2
    accounting_pass = (
        max_full_gap <= TOL and max_accounting <= TOL and max_goods <= TOL
        and max_advance <= TOL and max_assignment == 0 and max_feasibility <= TOL
    )
    if not accounting_pass:
        verdict = "E. ACCOUNTING_SCOPE_FIX_BLOCKER"
    elif max_tracked_shift > 0.20:
        verdict = "C. PHASE_STAGGERING_MATERIALLY_DISTORTS_ECONOMIC_LEVELS"
    elif materially_weakened and review_pass:
        verdict = "A. ACCOUNTING_SCOPE_AND_SCHEDULER_DESYNCHRONIZATION_ACCEPTED"
    else:
        verdict = "B. ACCOUNTING_SCOPE_FIXED_BUT_SYNCHRONIZATION_REMAINS"

    cash_lookup = {(row["mode"], row["window"]): row for row in cash}
    flags = {
        "verdict": verdict,
        "population": population,
        "seed": SEED,
        "weeks": weeks,
        "accounting_scope_fixed": accounting_pass,
        "max_full_money_location_gap": max_full_gap,
        "max_accounting_gap": max_accounting,
        "max_goods_gap": max_goods,
        "max_advance_prepaid_gap": max_advance,
        "max_assignment_violations": max_assignment,
        "max_output_above_feasible_capacity": max_feasibility,
        "review_cadence_invariance_pass": review_pass,
        "new_rng_draws": 0,
        "largest_jump_treatment_control_ratios": jump_ratios,
        "staircase_materially_weakened": materially_weakened,
        "maximum_late_core_level_shift": max_core_shift,
        "maximum_late_tracked_level_shift": max_tracked_shift,
        "maximum_late_tracked_level_shift_metric": max_tracked_shift_metric,
        "economic_levels_comparable": max_tracked_shift <= 0.20,
        "macro_cash_closure_fixed": False,
        "economic_behavior_change": "deterministic review phase only",
    }
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8-sig"
    )
    control_cash = cash_lookup[("control", "full")]
    treatment_cash = cash_lookup[("treatment", "full")]
    (OUT / "acceptance_summary.md").write_text(
        f"""# Step 15 Post-Validation Dynamic Cleanup

Verdict: **{verdict}**

The screen used N={population}, seed={SEED}, {weeks} weeks for synchronized control and deterministic phase-staggered treatment. No runtime RNG draws were added. Production remains every 5 weeks per Food Firm and investment review every 13 weeks per Food Firm; only phase differs.

## Accounting scope

LegacyOwner cash and Estate cash are now authoritative money locations. Maximum full money-location gap is `{max_full_gap:.3e}`; maximum accounting, goods and advance/prepaid gaps are `{max_accounting:.3e}`, `{max_goods:.3e}` and `{max_advance:.3e}`. Assignment violations are `{max_assignment}` and maximum output above feasible capacity is `{max_feasibility:.3e}`.

## Scheduler screen

Review cadence/count invariance passes: `{review_pass}`. Largest-jump treatment/control ratios are `{json.dumps(jump_ratios, sort_keys=True)}`. Staircase weakening is classified as `{materially_weakened}`. Maximum late-window mean shift among employment, Household income/consumption, Food production and active capital service is `{max_core_shift:.2%}`.

## Economic-level preservation

The maximum late-window shift across all required tracked economic levels is `{max_tracked_shift:.2%}` in `{max_tracked_shift_metric}`. The narrower core flow/activity set shifts by at most `{max_core_shift:.2%}`.

## Cash slopes

Control weekly slopes: Household `{control_cash['household_cash_slope_per_week']:.2f}`, Firm `{control_cash['firm_cash_slope_per_week']:.2f}`, Legacy `{control_cash['legacy_cash_slope_per_week']:.2f}`.

Treatment weekly slopes: Household `{treatment_cash['household_cash_slope_per_week']:.2f}`, Firm `{treatment_cash['firm_cash_slope_per_week']:.2f}`, Legacy `{treatment_cash['legacy_cash_slope_per_week']:.2f}`.

The macro Firm-cash drain was intentionally not repaired in this stage.
""",
        encoding="utf-8-sig",
    )
    print(json.dumps(flags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=5000)
    parser.add_argument("--weeks", type=int, default=WEEKS)
    parser.add_argument("--permanent", action="store_true")
    args = parser.parse_args()
    main(args.population, args.weeks, args.permanent)
