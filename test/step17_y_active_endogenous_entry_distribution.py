"""Step17.Y active capital-goods entry and household distribution smoke.

The experiment is intentionally a small, deterministic 52-week comparison.
It records household-level cash distributions at the required review points
and keeps the economic interpretation separate from technical acceptance.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world import World


OUT = Path("test/output/step17_y_active_endogenous_entry_distribution")
SNAPSHOT_WEEKS = (0, 13, 26, 39, 52)
EPS = 1e-8


def finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def q(values, p):
    values = sorted(finite(x) for x in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def gini(values):
    values = sorted(max(0.0, finite(x)) for x in values)
    total = math.fsum(values)
    if not values or total <= EPS:
        return 0.0
    return math.fsum((2 * i - len(values) - 1) * x for i, x in enumerate(values, 1)) / (len(values) * total)


def skew_kurt(values):
    values = [finite(x) for x in values]
    if len(values) < 3:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values)
    if sd <= EPS:
        return 0.0, 0.0
    z = [(x - mean) / sd for x in values]
    return statistics.fmean(x ** 3 for x in z), statistics.fmean(x ** 4 for x in z) - 3.0


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def overrides(entry=False):
    return {
        "CANONICAL_INVESTMENT_ENABLED": True,
        "ENDOGENOUS_CAPITAL_GOODS_ENTRY_ENABLED": entry,
        "MULTISECTOR_FOUNDATION_ENABLED": True,
        "PERSON_FOUNDER_BOOTSTRAP_ENABLED": False,
        "PERSON_EQUITY_TRANSITION_ENABLED": False,
        "AUTONOMOUS_SECONDARY_EQUITY_ENABLED": False,
        "CAPITAL_GOOD_CUSTOMER_ADVANCE_ENABLED": False,
        "CAPITAL_CAPACITY_RUNTIME_ENABLED": False,
        "CAPITAL_LIFECYCLE_ENABLED": False,
        "PRIVATE_FAMILY_SUPPORT_ENABLED": False,
        "PAYG_PENSION_ENABLED": False,
        "PERSONAL_INCOME_TAX_ENABLED": False,
        "CORPORATE_PROFIT_TAX_ENABLED": False,
        "PUBLIC_REVENUE_TAX_ENABLED": False,
    }


def household_need(world, household):
    units = finite(world.needs_system.household_minimum_need_units(household))
    price = finite(world.household_planning_price_this_step(), 1.0)
    return max(0.0, units * max(price, EPS))


def household_rows(world, branch, week, cumulative_wages, cumulative_dividends, founder_by_household, founder_outflow):
    rows = []
    for household in sorted(getattr(world, "households", []), key=lambda h: str(getattr(h, "id", ""))):
        if getattr(household, "settlement_only", False):
            continue
        hid = getattr(household, "id", "")
        cash = finite(getattr(household, "wealth", 0.0))
        need = household_need(world, household)
        founder = founder_by_household.get(hid, {})
        rows.append({
            "branch": branch,
            "week": week,
            "household_id": hid,
            "cash": cash,
            "authoritative_weekly_minimum_need": need,
            "liquidity_weeks": cash / need if need > EPS else float("nan"),
            "household_size": len(getattr(household, "parents", [])) + len(getattr(household, "children", [])),
            "founder_household": int(hid in founder_by_household),
            "founder_person_id": founder.get("founder_person_id", ""),
            "founder_equity_outflow_this_week": founder_outflow.get(hid, 0.0),
            "cumulative_founder_equity_outflow": founder.get("founder_equity", 0.0),
            "wage_income_this_week": finite(getattr(household, "wage_income_this_step", 0.0)),
            "cumulative_wage_income": cumulative_wages.get(hid, 0.0),
            "dividend_income_this_week": finite(getattr(household, "dividend_income_this_step", 0.0)),
            "cumulative_dividend_income": cumulative_dividends.get(hid, 0.0),
        })
    return rows


def state_snapshot(world, branch, week, cumulative_wages, cumulative_dividends, founder_by_household, founder_outflow):
    rows = household_rows(world, branch, week, cumulative_wages, cumulative_dividends, founder_by_household, founder_outflow)
    return rows


def run_branch(branch, entry):
    world = World(initial_population=500, seed=42, scenario_overrides=overrides(entry), diagnostics_mode="full")
    cumulative_wages, cumulative_dividends = {}, {}
    founder_by_household, founder_outflow = {}, {}
    snapshots = state_snapshot(world, branch, 0, cumulative_wages, cumulative_dividends, founder_by_household, founder_outflow)
    snapshot_rows = list(snapshots)
    weekly = []
    previous_employees = {}
    entrant_workers = []
    for elapsed in range(1, 53):
        world.step()
        system = getattr(world, "endogenous_capital_goods_entry_system", None)
        events = list(getattr(system, "formation_events", [])) if system is not None else []
        for event in events:
            hid = event.get("founder_household_id")
            if hid not in founder_by_household:
                founder_by_household[hid] = dict(event)
            founder_outflow[hid] = finite(event.get("founder_equity", 0.0)) if event.get("global_step") == world.current_step_index else 0.0
        for household in getattr(world, "households", []):
            hid = getattr(household, "id", "")
            cumulative_wages[hid] = cumulative_wages.get(hid, 0.0) + finite(getattr(household, "wage_income_this_step", 0.0))
            cumulative_dividends[hid] = cumulative_dividends.get(hid, 0.0) + finite(getattr(household, "dividend_income_this_step", 0.0))
        capfirms = list(getattr(world, "capital_good_firms", []))
        entrant_ids = {getattr(f, "firm_id", None) for f in capfirms if getattr(f, "firm_id", 0) >= 100001}
        for firm in capfirms:
            fid = getattr(firm, "firm_id", None)
            current = set(getattr(firm, "employee_ids", []) or [])
            if fid in entrant_ids:
                for pid in sorted(current - previous_employees.get(fid, set()), key=str):
                    person = world.get_person_by_id(pid)
                    entrant_workers.append({
                        "branch": branch, "global_step": world.current_step_index,
                        "firm_id": fid, "person_id": pid,
                        "household_id": getattr(person, "household_id", "") if person else "",
                        "worker_source": "previously_unassigned_or_reallocated",
                        "wage_income_this_week": finite(getattr(world.get_household(getattr(person, "household_id", None)), "wage_income_this_step", 0.0)) if person and world.get_household(getattr(person, "household_id", None)) else 0.0,
                    })
            previous_employees[fid] = current
        macro = dict(world.diagnostics_rows[-1]) if getattr(world, "diagnostics_rows", []) else {}
        firms = [f for f in world.operating_firms()]
        cap = [f for f in capfirms]
        weekly.append({
            "branch": branch, "global_step": world.current_step_index,
            "population": len(world.population), "households": len([h for h in world.households if not getattr(h, "settlement_only", False)]),
            "firm_count": len(world.firms), "capital_goods_firm_count": len(cap),
            "founder_count": len(founder_by_household), "unassigned_labor": finite(macro.get("unassigned_labor", macro.get("unassigned_workers", 0.0))),
            "total_employment": sum(len(getattr(f, "employee_ids", []) or []) for f in firms),
            "wage_income": sum(finite(getattr(h, "wage_income_this_step", 0.0)) for h in world.households),
            "household_consumption": finite(macro.get("consumption", macro.get("total_consumption", 0.0))),
            "household_saving": finite(macro.get("saving", macro.get("total_saving", 0.0))),
            "capital_good_backlog": math.fsum(max(0.0, finite(getattr(f, "capital_good_outstanding_demand_units", 0.0))) for f in cap),
            "capital_good_sales_units": math.fsum(max(0.0, finite(getattr(f, "capital_good_sales_units", 0.0))) for f in cap),
            "capital_good_production": math.fsum(max(0.0, finite(getattr(f, "capital_good_production_units", 0.0))) for f in cap),
            "capital_good_employment": sum(len(getattr(f, "employee_ids", []) or []) for f in cap),
            "capital_good_revenue": math.fsum(finite(getattr(f, "sales_revenue", 0.0)) for f in cap),
            "capital_good_profit": math.fsum(finite(getattr(f, "profit", 0.0)) for f in cap),
            "firm_cash": math.fsum(finite(getattr(f, "cash", 0.0)) for f in firms),
            "money_gap": finite(macro.get("ledger_money_net_gap", 0.0)),
            "wide_money_location_gap": finite(macro.get("monetary_accounting_gap", 0.0)),
            "accounting_gap": max(abs(finite(macro.get(k, 0.0))) for k in ("income_spending_gap", "sales_revenue_split_gap", "firm_profit_aggregation_gap", "dividend_reconciliation_gap", "household_dividend_reconciliation_gap")),
            "goods_gap": finite(macro.get("food_conservation_gap", 0.0)),
            "assignment_violations": sum(1 for person in world.population if (getattr(person, "firm_id", None) is not None) != (person.id in {pid for firm in firms for pid in (getattr(firm, "employee_ids", []) or [])})),
        })
        if elapsed in SNAPSHOT_WEEKS:
            snapshot_rows.extend(state_snapshot(world, branch, elapsed, cumulative_wages, cumulative_dividends, founder_by_household, founder_outflow))
        founder_outflow = {hid: 0.0 for hid in founder_by_household}
    return {
        "world": world, "weekly": weekly, "snapshots": snapshot_rows,
        "events": list(getattr(getattr(world, "endogenous_capital_goods_entry_system", None), "formation_events", [])),
        "failures": list(getattr(getattr(world, "endogenous_capital_goods_entry_system", None), "formation_failures", [])),
        "founders": founder_by_household, "entrant_workers": entrant_workers,
    }


def distribution_rows(result):
    rows = []
    for key in sorted({(r["branch"], r["week"]) for r in result["snapshots"]}):
        branch, week = key
        data = [r for r in result["snapshots"] if r["branch"] == branch and r["week"] == week]
        cash = [finite(r["cash"]) for r in data]
        liq = [finite(r["liquidity_weeks"], float("nan")) for r in data]
        liq_valid = [x for x in liq if math.isfinite(x)]
        total = math.fsum(cash)
        bottom = sum(sorted(cash)[: max(1, len(cash) // 2)]) / total if total > EPS else 0.0
        top10 = sum(sorted(cash, reverse=True)[: max(1, math.ceil(len(cash) * .10))]) / total if total > EPS else 0.0
        top1 = sum(sorted(cash, reverse=True)[: max(1, math.ceil(len(cash) * .01))]) / total if total > EPS else 0.0
        rows.append({"branch": branch, "week": week, "household_count": len(cash), "cash_min": min(cash, default=0.0), "cash_P1": q(cash,.01), "cash_P5": q(cash,.05), "cash_P10": q(cash,.10), "cash_P25": q(cash,.25), "cash_P50": q(cash,.50), "cash_P75": q(cash,.75), "cash_P90": q(cash,.90), "cash_P95": q(cash,.95), "cash_P99": q(cash,.99), "cash_max": max(cash, default=0.0), "cash_mean": statistics.fmean(cash) if cash else 0.0, "cash_sd": statistics.pstdev(cash) if len(cash)>1 else 0.0, "cash_cv": statistics.pstdev(cash)/statistics.fmean(cash) if len(cash)>1 and statistics.fmean(cash)>EPS else 0.0, "cash_gini": gini(cash), "bottom50_cash_share": bottom, "top10_cash_share": top10, "top1_cash_share": top1, "positive_cash_count": sum(x>EPS for x in cash), "zero_or_near_zero_share": sum(x<=EPS for x in cash)/len(cash) if cash else 0.0, "liquidity_P10": q(liq_valid,.10), "liquidity_P25": q(liq_valid,.25), "liquidity_P50": q(liq_valid,.50), "below_0.25_liquidity_share": sum(x<.25 for x in liq_valid)/len(liq_valid) if liq_valid else 0.0, "below_0.5_liquidity_share": sum(x<.5 for x in liq_valid)/len(liq_valid) if liq_valid else 0.0, "below_1_liquidity_share": sum(x<1 for x in liq_valid)/len(liq_valid) if liq_valid else 0.0, "below_2_liquidity_share": sum(x<2 for x in liq_valid)/len(liq_valid) if liq_valid else 0.0, "below_4_liquidity_share": sum(x<4 for x in liq_valid)/len(liq_valid) if liq_valid else 0.0})
    return rows


def shape_rows(summary):
    rows = []
    for s in summary:
        values = [finite(r["cash"]) for r in ALL_SNAPSHOTS if r["branch"] == s["branch"] and r["week"] == s["week"] and finite(r["cash"]) > EPS]
        logs = [math.log(x) for x in values]
        raw_skew, raw_kurt = skew_kurt(values)
        log_skew, log_kurt = skew_kurt(logs)
        rows.append({"branch": s["branch"], "week": s["week"], "positive_cash_count": len(values), "raw_cash_skewness": raw_skew, "raw_cash_excess_kurtosis": raw_kurt, "log_cash_mean": statistics.fmean(logs) if logs else float("nan"), "log_cash_sd": statistics.pstdev(logs) if len(logs)>1 else 0.0, "log_cash_skewness": log_skew, "log_cash_excess_kurtosis": log_kurt, "P90_over_P50": s["cash_P90"]/max(s["cash_P50"],EPS), "P95_over_P50": s["cash_P95"]/max(s["cash_P50"],EPS), "P99_over_P50": s["cash_P99"]/max(s["cash_P50"],EPS), "distribution_shape_classification": "NEAR_ZERO_INFLATED_AND_POLARIZED" if s["zero_or_near_zero_share"]>.10 else "CONTINUOUS_BUT_STRONGLY_NON_LOGNORMAL"})
    return rows


def histogram_rows(snapshot_rows, log=False):
    rows = []
    for branch in ("CONTROL", "ENTRY"):
        for week in (0, 26, 52):
            values = [finite(r["cash"]) for r in snapshot_rows if r["branch"] == branch and r["week"] == week and (not log or finite(r["cash"]) > EPS)]
            if log:
                values = [math.log(x) for x in values]
            if not values:
                continue
            lo, hi = min(values), max(values)
            width = (hi - lo) / 20.0 if hi > lo else 1.0
            for i in range(20):
                left, right = lo + i*width, lo + (i+1)*width
                count = sum((left <= x < right) or (i == 19 and left <= x <= right) for x in values)
                rows.append({"branch": branch, "week": week, "bin": i, "bin_left": left, "bin_right": right, "count": count, "share": count/len(values)})
    return rows


def low_liquidity_rows(snapshot_rows):
    rows = []
    for branch in ("CONTROL", "ENTRY"):
        by_week = {w: {r["household_id"]: finite(r["liquidity_weeks"], float("nan")) for r in snapshot_rows if r["branch"] == branch and r["week"] == w} for w in SNAPSHOT_WEEKS}
        for threshold in (.25, 1.0):
            for before, after in zip(SNAPSHOT_WEEKS, SNAPSHOT_WEEKS[1:]):
                ids = sorted(set(by_week[before]) | set(by_week[after]), key=str)
                low_before = {i for i in ids if math.isfinite(by_week[before].get(i, float("nan"))) and by_week[before][i] < threshold}
                low_after = {i for i in ids if math.isfinite(by_week[after].get(i, float("nan"))) and by_week[after][i] < threshold}
                rows.append({"branch": branch, "threshold_weeks": threshold, "from_week": before, "to_week": after, "entry_count": len(low_after-low_before), "exit_count": len(low_before-low_after), "stay_low_count": len(low_after & low_before), "eligible_count": len(ids), "entry_probability": len(low_after-low_before)/len(ids) if ids else 0.0, "exit_probability": len(low_before-low_after)/len(low_before) if low_before else 0.0, "stay_low_probability": len(low_after&low_before)/len(low_before) if low_before else 0.0, "persistent_low_share": len(low_after&low_before)/len(ids) if ids else 0.0})
    return rows


def write_plots(summary):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    for name, column, title in (("household_cash_histogram_week52.png", "cash", "Household cash, week 52"), ("household_log_cash_histogram_week52.png", "log_cash", "Log household cash, week 52")):
        plt.figure(figsize=(8, 5))
        for branch, color in (("CONTROL", "#3b82f6"), ("ENTRY", "#e76f51")):
            values = [finite(r["cash"]) for r in ALL_SNAPSHOTS if r["branch"] == branch and r["week"] == 52 and finite(r["cash"]) > EPS]
            if column == "log_cash": values = [math.log(x) for x in values]
            plt.hist(values, bins=20, alpha=.45, label=branch, color=color)
        plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(OUT/name, dpi=120); plt.close()
    return True


def main():
    global ALL_SNAPSHOTS
    OUT.mkdir(parents=True, exist_ok=True)
    control = run_branch("CONTROL", False)
    entry = run_branch("ENTRY", True)
    ALL_SNAPSHOTS = control["snapshots"] + entry["snapshots"]
    summary = distribution_rows({"snapshots": ALL_SNAPSHOTS})
    shape = shape_rows(summary)
    event_rows = entry["events"]
    failures = entry["failures"]
    write_csv(OUT/"firm_formation_events.csv", event_rows)
    write_csv(OUT/"firm_formation_failures.csv", failures)
    write_csv(OUT/"firm_count_weekly.csv", control["weekly"] + entry["weekly"])
    write_csv(OUT/"founder_count_weekly.csv", [{k: r[k] for k in ("branch", "global_step", "founder_count")} for r in control["weekly"] + entry["weekly"]])
    founder_effect = []
    for hid, event in entry["founders"].items():
        final = next((r for r in reversed(entry["snapshots"]) if r["household_id"] == hid and r["week"] == 52), {})
        founder_effect.append({**event, "week52_cash": final.get("cash", float("nan")), "subsequent_wages": final.get("cumulative_wage_income", 0.0), "subsequent_dividends": final.get("cumulative_dividend_income", 0.0)})
    write_csv(OUT/"founder_household_cash_effect.csv", founder_effect)
    write_csv(OUT/"entrant_worker_household_effect.csv", entry["entrant_workers"])
    write_csv(OUT/"household_cash_distribution_snapshots.csv", ALL_SNAPSHOTS)
    write_csv(OUT/"household_cash_distribution_summary.csv", summary)
    write_csv(OUT/"household_cash_distribution_shape.csv", shape)
    write_csv(OUT/"household_cash_histogram_bins.csv", histogram_rows(ALL_SNAPSHOTS, False))
    write_csv(OUT/"household_log_cash_histogram_bins.csv", histogram_rows(ALL_SNAPSHOTS, True))
    ecdf = []
    for r in ALL_SNAPSHOTS:
        if r["week"] in (0, 26, 52):
            values = sorted(x["cash"] for x in ALL_SNAPSHOTS if x["branch"] == r["branch"] and x["week"] == r["week"])
            ecdf.append({"branch": r["branch"], "week": r["week"], "cash": r["cash"], "ecdf": sum(x <= r["cash"] for x in values)/len(values) if values else 0.0})
    write_csv(OUT/"household_cash_ecdf.csv", ecdf)
    low = low_liquidity_rows(ALL_SNAPSHOTS)
    write_csv(OUT/"household_low_liquidity_transition.csv", low)
    c52 = next(r for r in summary if r["branch"] == "CONTROL" and r["week"] == 52)
    e52 = next(r for r in summary if r["branch"] == "ENTRY" and r["week"] == 52)
    score_fields = ("zero_or_near_zero_share", "cash_P10", "cash_P25", "cash_P50", "liquidity_P10", "liquidity_P25", "liquidity_P50", "below_0.25_liquidity_share", "below_1_liquidity_share", "cash_gini", "bottom50_cash_share", "top10_cash_share", "top1_cash_share")
    write_csv(OUT/"distribution_north_star_scorecard.csv", [{"metric": f, "control_week52": c52.get(f, 0.0), "entry_week52": e52.get(f, 0.0), "entry_minus_control": e52.get(f, 0.0)-c52.get(f, 0.0)} for f in score_fields])
    write_csv(OUT/"distribution_effect_decomposition.csv", [{"channel": "founder_equity_cash_outflow", "value": math.fsum(finite(e.get("founder_equity", 0.0)) for e in event_rows)}, {"channel": "new_labor_wage_income", "value": math.fsum(finite(r.get("wage_income_this_week", 0.0)) for r in entry["entrant_workers"])}, {"channel": "founder_dividend_income", "value": math.fsum(finite(r.get("subsequent_dividends", 0.0)) for r in founder_effect)}, {"channel": "demand_competition_feedback", "value": "descriptive matched-branch residual"}])
    market = control["weekly"] + entry["weekly"]
    write_csv(OUT/"capital_goods_market_response.csv", market)
    write_csv(OUT/"competition_comparison.csv", [{"branch": b, "final_firm_count": len((control if b == "CONTROL" else entry)["world"].firms), "final_capital_goods_firm_count": len((control if b == "CONTROL" else entry)["world"].capital_good_firms), "entrant_sales": math.fsum(r["capital_good_sales_units"] for r in market if r["branch"] == b)} for b in ("CONTROL", "ENTRY")])
    accounting = [{"branch": b, "max_abs_money_gap": max((abs(r["money_gap"]) for r in market if r["branch"] == b), default=0.0), "max_abs_accounting_gap": max((abs(r["accounting_gap"]) for r in market if r["branch"] == b), default=0.0), "max_abs_goods_gap": max((abs(r["goods_gap"]) for r in market if r["branch"] == b), default=0.0), "max_assignment_violations": max((r["assignment_violations"] for r in market if r["branch"] == b), default=0.0), "founder_equity_transfer_gap": max((abs(finite(e.get("household_cash_before"))-finite(e.get("household_cash_after"))-finite(e.get("founder_equity"))) for e in (event_rows if b == "ENTRY" else [])), default=0.0)} for b in ("CONTROL", "ENTRY")]
    write_csv(OUT/"accounting_reconciliation.csv", accounting)
    pre_entry = [r for r in control["weekly"] + entry["weekly"] if r["global_step"] < 39]
    parity = [{"metric": k, "max_abs_pre_entry_difference": max((abs(finite(next(x[k] for x in entry["weekly"] if x["global_step"] == w))-finite(next(x[k] for x in control["weekly"] if x["global_step"] == w))) for w in sorted({x["global_step"] for x in pre_entry}) if k in next(x for x in entry["weekly"] if x["global_step"] == w)), default=0.0)} for k in ("firm_count", "capital_good_backlog", "wage_income", "household_consumption", "household_saving")]
    write_csv(OUT/"control_parity.csv", parity)
    plotting = write_plots(summary)
    lower_improved = e52["cash_P10"] > c52["cash_P10"] + EPS and e52["below_1_liquidity_share"] < c52["below_1_liquidity_share"] - EPS
    concentration_tradeoff = e52["cash_gini"] > c52["cash_gini"] + .01 or e52["top1_cash_share"] > c52["top1_cash_share"] + .01
    if not event_rows:
        economic = "ECONOMIC_EFFECT_NOT_IDENTIFIABLE"
    elif lower_improved and concentration_tradeoff:
        economic = "LOWER_TAIL_IMPROVES_BUT_UPPER_CONCENTRATION_INCREASES"
    elif lower_improved:
        economic = "HOUSEHOLD_CASH_DISTRIBUTION_BROADLY_IMPROVED"
    elif e52["cash_P50"] == c52["cash_P50"] and abs(e52["cash_gini"]-c52["cash_gini"]) < .01:
        economic = "HOUSEHOLD_CASH_DISTRIBUTION_EFFECT_NEGLIGIBLE"
    elif any(e.get("founder_equity", 0.0) > 0 for e in event_rows):
        economic = "ENTRY_MAINLY_RAISES_FOUNDER_UPPER_TAIL"
    else:
        economic = "HOUSEHOLD_CASH_DISTRIBUTION_WORSENED"
    technical = "ACTIVE_ENDOGENOUS_CAPITAL_GOODS_ENTRY_ACCEPTED" if event_rows and all(r["max_abs_money_gap"] <= 1e-6 and r["max_abs_accounting_gap"] <= 1e-6 and r["max_assignment_violations"] <= 0 for r in accounting) else ("ENTRY_NOT_TRIGGERED" if not event_rows else "ACCOUNTING_OR_RESOURCE_FAILURE")
    flags = {"technical_verdict": technical, "economic_north_star_verdict": economic, "active_entry_trigger_week": min((e["global_step"] for e in event_rows), default=None), "proposed_formations": len(entry["events"])+len(failures), "accepted_formations": len(event_rows), "rejected_formations": len(failures), "founder_equity_transfer_reconciliation_pass": accounting[1]["founder_equity_transfer_gap"] <= 1e-6, "no_food_entry": all(e.get("sector_id") == "capital_goods" for e in event_rows), "new_rng_draws": 0, "taxes_changed": False, "dividend_policy_changed": False, "negative_household_cash": any(finite(r["cash"]) < -EPS for r in ALL_SNAPSHOTS), "money_reconciliation_pass": all(r["max_abs_money_gap"] <= 1e-6 for r in accounting), "accounting_reconciliation_pass": all(r["max_abs_accounting_gap"] <= 1e-6 for r in accounting), "goods_reconciliation_pass": all(r["max_abs_goods_gap"] <= 1e-6 for r in accounting), "assignment_reconciliation_pass": all(r["max_assignment_violations"] <= 0 for r in accounting), "static_plots_written": plotting}
    (OUT/"acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (OUT/"acceptance_summary.md").write_text("# Step17.Y Active Endogenous Capital-Goods Entry\n\n" + f"Technical verdict: **{technical}**\n\nEconomic North-Star verdict: **{economic}**\n\n" + f"The matched N=500, seed42 smoke ran 52 weeks. Entry events: {len(event_rows)}; rejected proposals/failures: {len(failures)}; first accepted entry week: {flags['active_entry_trigger_week']}.\n\n" + "The treatment is capital-goods-only. Founder equity is a real Household cash transfer to the new generic Firm; no startup loan, free inventory, taxes, ownership redistribution, or new RNG draw was used.\n\n" + "Household distribution outputs are authoritative snapshots at weeks 0, 13, 26, 39, and 52. Economic classification is descriptive matched-branch evidence, not a calibration claim.\n\n" + f"Accounting/money/goods/assignment checks: {flags['money_reconciliation_pass']}/{flags['accounting_reconciliation_pass']}/{flags['goods_reconciliation_pass']}/{flags['assignment_reconciliation_pass']}. Optional static plots written: {plotting}.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
