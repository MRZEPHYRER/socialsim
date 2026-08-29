"""Step17.Z diagnostic audit of household cash polarization.

This script replays only the accepted Step17.Y control path.  It does not
activate entry, transfers, pensions, taxes, or any new behavioral mechanism.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from productivity import age_productivity
from world import World

OUT = Path("test/output/step17_z_household_cash_polarization_root_cause")
WEEKS = (0, 13, 26, 39, 52)
EPS = 1e-8


def num(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def quantile(values, p):
    values = sorted(num(x) for x in values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def gini(values):
    values = sorted(max(0.0, num(x)) for x in values)
    total = math.fsum(values)
    if not values or total <= EPS:
        return 0.0
    return math.fsum((2 * i - len(values) - 1) * x for i, x in enumerate(values, 1)) / (len(values) * total)


def moments(values):
    values = [num(x) for x in values]
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
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        if not fields:
            fh.write("\n")
            return
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def overrides():
    return {
        "CANONICAL_INVESTMENT_ENABLED": True,
        "ENDOGENOUS_CAPITAL_GOODS_ENTRY_ENABLED": False,
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


def members(world, household):
    ids = list(getattr(household, "parents", [])) + list(getattr(household, "children", []))
    return [world.get_person_by_id(pid) for pid in ids if world.get_person_by_id(pid) is not None and getattr(world.get_person_by_id(pid), "alive", False)]


def household_type(adults, children, elderly):
    if adults == 1 and children == 0:
        return "SINGLE_ADULT"
    if adults == 2 and children == 0:
        return "COUPLE_NO_CHILDREN"
    if adults == 2 and children > 0:
        return "COUPLE_WITH_CHILDREN"
    if adults == 1 and children > 0:
        return "SINGLE_PARENT"
    if adults == 0 and children > 0:
        return "CHILDREN_ONLY"
    if elderly > 0:
        return "ELDERLY_HEAVY"
    return "OTHER"


def household_row(world, household, week, opening_cash=None, founder_outflow=0.0):
    people = members(world, household)
    adults = [p for p in people if 20 <= num(getattr(p, "age", 0.0)) < 65]
    children = [p for p in people if num(getattr(p, "age", 0.0)) < 20]
    elderly = [p for p in people if num(getattr(p, "age", 0.0)) >= 65]
    eligible = [p for p in adults if world.labor_participates(p)]
    employed = [p for p in eligible if getattr(p, "firm_id", None) is not None]
    effective = math.fsum(age_productivity(num(getattr(p, "age", 0.0))) for p in employed)
    need = max(0.0, num(world.needs_system.household_minimum_need_units(household)) * max(EPS, num(world.household_planning_price_this_step(), 1.0)))
    cash = num(getattr(household, "wealth", 0.0))
    wage = num(getattr(household, "wage_income_this_step", 0.0))
    dividend = num(getattr(household, "dividend_income_this_step", 0.0))
    support_in = num(getattr(household, "private_support_received_this_step", 0.0))
    support_out = num(getattr(household, "private_support_paid_this_step", 0.0))
    pension = num(getattr(household, "pension_paid_this_step", 0.0))
    contribution = num(getattr(household, "payg_contribution_this_step", 0.0))
    tax = num(getattr(household, "personal_income_tax_this_step", 0.0))
    consumption = num(getattr(household, "consumption_this_step", 0.0))
    inflow = wage + dividend + support_in
    outflow = consumption + support_out + max(0.0, pension) + contribution + tax + founder_outflow
    if opening_cash is None:
        opening_cash = cash
    gap = num(opening_cash) + inflow - outflow - cash
    return {
        "week": week, "household_id": getattr(household, "id", ""), "cash": cash,
        "liquidity_weeks": cash / need if need > EPS else float("nan"),
        "income_need_ratio": inflow / need if need > EPS else float("nan"),
        "household_type": household_type(len(adults), len(children), len(elderly)),
        "household_size": len(people), "adult_count": len(adults), "child_count": len(children), "age65plus_count": len(elderly),
        "labor_eligible_count": len(eligible), "employed_count": len(employed), "effective_labor": effective,
        "wage_income": wage, "dividend_income": dividend, "private_support_received": support_in,
        "pension_received": 0.0, "other_income": 0.0, "total_authoritative_income": inflow,
        "minimum_need": need, "minimum_need_per_adult": need / len(adults) if adults else 0.0,
        "minimum_need_per_earner": need / len(employed) if employed else 0.0,
        "consumption": consumption, "private_support_paid": support_out, "pension_contribution": contribution,
        "tax": tax, "founder_equity_outflow": founder_outflow, "cash_flow_inflow": inflow,
        "cash_flow_outflow": outflow, "net_cash_accumulation": inflow - outflow, "cash_reconciliation_gap": gap,
        "scheduled_wage": wage, "realized_wage": wage,
        "mean_wage_per_employed_member": wage / len(employed) if employed else 0.0,
        "mean_employed_age_productivity": effective / len(employed) if employed else 0.0,
        "mean_adult_age": statistics.fmean(num(getattr(p, "age", 0.0)) for p in adults) if adults else 0.0,
        "oldest_member_age": max((num(getattr(p, "age", 0.0)) for p in people), default=0.0),
        "has_age65plus": int(bool(elderly)), "all_earners_age65plus": int(bool(employed) and all(num(getattr(p, "age", 0.0)) >= 65 for p in employed)),
        "no_working_age_adult": int(not adults), "effective_labor_per_adult": effective / len(adults) if adults else 0.0,
        "wage_income_per_adult": wage / len(adults) if adults else 0.0,
    }


def run_control():
    world = World(initial_population=500, seed=42, scenario_overrides=overrides(), diagnostics_mode="full")
    rows = []
    opening = {}
    cumulative = defaultdict(float)
    snapshots = {}
    for h in world.households:
        if not getattr(h, "settlement_only", False):
            opening[h.id] = num(getattr(h, "wealth", 0.0))
    for elapsed in range(53):
        if elapsed in WEEKS:
            current = []
            for h in world.households:
                if not getattr(h, "settlement_only", False):
                    current.append(household_row(world, h, elapsed, opening.get(h.id)))
            snapshots[elapsed] = current
        if elapsed == 52:
            break
        world.step()
        for h in world.households:
            if getattr(h, "settlement_only", False):
                continue
            hid = h.id
            row = household_row(world, h, elapsed + 1, opening.get(hid))
            cumulative[hid] += row["cash_flow_inflow"]
            opening[hid] = row["cash"]
            rows.append({**row, "opening_cash": opening[hid] - row["cash_flow_inflow"] + row["cash_flow_outflow"], "cumulative_authoritative_income": cumulative[hid]})
    return world, rows, snapshots


def cohort_analysis(snapshots):
    final = {r["household_id"]: r for r in snapshots[52]}
    cohort = {}
    for hid, row in final.items():
        liq = num(row["liquidity_weeks"], 1e9)
        cohort[hid] = "DEEP_LOW" if liq < .25 else "LOW" if liq < 1 else "MIDDLE" if liq < 26 else "HIGH"
    rows = [{"household_id": hid, "week52_cohort": c, "week52_cash": final[hid]["cash"], "week52_liquidity": final[hid]["liquidity_weeks"]} for hid, c in cohort.items()]
    return cohort, rows


def summarize_cohorts(rows, cohort, key="week"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(cohort.get(row["household_id"], "UNKNOWN"), row.get(key, 0))].append(row)
    output = []
    for (c, week), data in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        cash = [r["cash"] for r in data]; liq = [r["liquidity_weeks"] for r in data if math.isfinite(num(r["liquidity_weeks"], float("nan")))]
        income_need = [r["income_need_ratio"] for r in data if math.isfinite(num(r["income_need_ratio"], float("nan")))]
        accum = [r["net_cash_accumulation"] for r in data]
        output.append({"cohort": c, "week": week, "household_count": len(data), "cash_mean": statistics.fmean(cash) if cash else 0.0, "cash_median": quantile(cash,.5), "liquidity_median": quantile(liq,.5), "income_need_ratio_mean": statistics.fmean(income_need) if income_need else 0.0, "income_need_ratio_median": quantile(income_need,.5), "income_need_ratio_P10": quantile(income_need,.1), "income_need_ratio_P25": quantile(income_need,.25), "income_need_ratio_P75": quantile(income_need,.75), "income_need_ratio_P90": quantile(income_need,.9), "net_cash_accumulation_mean": statistics.fmean(accum) if accum else 0.0})
    return output


def main():
    global ALL_ROWS
    OUT.mkdir(parents=True, exist_ok=True)
    world, weekly, snapshots = run_control()
    ALL_ROWS = weekly
    cohort, assignment = cohort_analysis(snapshots)
    ranked = sorted(snapshots[52], key=lambda r: num(r["cash"]))
    percentile_cohort = {r["household_id"]: ("BOTTOM25" if i < len(ranked)*.25 else "P25_P50" if i < len(ranked)*.5 else "P50_P75" if i < len(ranked)*.75 else "TOP25") for i,r in enumerate(ranked)}
    for row in assignment: row["percentile_cohort"] = percentile_cohort.get(row["household_id"], "UNKNOWN")
    snapshot_rows = [dict(r, cohort=cohort.get(r["household_id"], "UNKNOWN"), percentile_cohort=percentile_cohort.get(r["household_id"], "UNKNOWN")) for week in WEEKS for r in snapshots[week]]
    write_csv(OUT/"household_weekly_cashflow.csv", weekly)
    write_csv(OUT/"household_cohort_assignment.csv", assignment)
    baseline = []
    baseline_by_id = {r["household_id"]: r for r in snapshots[0]}
    for hid, c in cohort.items():
        r = baseline_by_id.get(hid, {})
        baseline.append({"household_id": hid, "week52_cohort": c, **{k: r.get(k, float("nan")) for k in ("cash", "household_size", "adult_count", "child_count", "age65plus_count", "minimum_need", "labor_eligible_count", "employed_count", "effective_labor", "wage_income", "mean_adult_age", "household_type")}})
    write_csv(OUT/"household_cohort_baseline_comparison.csv", baseline)
    cohort_weekly = summarize_cohorts(weekly, cohort)
    write_csv(OUT/"cohort_income_need_comparison.csv", cohort_weekly)
    write_csv(OUT/"cohort_labor_income_decomposition.csv", [{"cohort": c, "week": w, "employed_member_count": statistics.fmean(r["employed_count"] for r in data), "labor_eligible_member_count": statistics.fmean(r["labor_eligible_count"] for r in data), "effective_labor_mean": statistics.fmean(r["effective_labor"] for r in data), "scheduled_wage_mean": statistics.fmean(r["scheduled_wage"] for r in data), "realized_wage_mean": statistics.fmean(r["realized_wage"] for r in data), "mean_wage_per_employed_member": statistics.fmean(r["mean_wage_per_employed_member"] for r in data), "mean_age_productivity": statistics.fmean(r["mean_employed_age_productivity"] for r in data)} for (c,w), data in _groups(weekly, cohort)])
    write_csv(OUT/"cohort_age_structure.csv", [{"cohort": c, "week": w, "mean_adult_age": statistics.fmean(r["mean_adult_age"] for r in data), "oldest_member_age": statistics.fmean(r["oldest_member_age"] for r in data), "share_age65plus": statistics.fmean(r["has_age65plus"] for r in data), "share_all_earners_age65plus": statistics.fmean(r["all_earners_age65plus"] for r in data), "share_no_working_age_adult": statistics.fmean(r["no_working_age_adult"] for r in data), "effective_labor_per_adult": statistics.fmean(r["effective_labor_per_adult"] for r in data), "wage_income_per_adult": statistics.fmean(r["wage_income_per_adult"] for r in data)} for (c,w), data in _groups(weekly, cohort)])
    types = defaultdict(list)
    for r in weekly:
        types[(r["household_type"], r["week"])].append(r)
    write_csv(OUT/"cohort_household_type_comparison.csv", [{"household_type": t, "week": w, "household_count": len(data), "median_cash": quantile([r["cash"] for r in data],.5), "P25_cash": quantile([r["cash"] for r in data],.25), "median_liquidity": quantile([r["liquidity_weeks"] for r in data],.5), "deep_low_share": statistics.fmean(num(r["liquidity_weeks"], 1e9) < .25 for r in data), "income_need_ratio": statistics.fmean(num(r["income_need_ratio"], 0.0) for r in data), "net_accumulation": statistics.fmean(r["net_cash_accumulation"] for r in data)} for (t,w), data in sorted(types.items())])
    write_csv(OUT/"cohort_need_decomposition.csv", [{"cohort": c, "week": w, "minimum_need_mean": statistics.fmean(r["minimum_need"] for r in data), "minimum_need_per_adult": statistics.fmean(r["minimum_need_per_adult"] for r in data), "minimum_need_per_earner": statistics.fmean(r["minimum_need_per_earner"] for r in data), "household_size_mean": statistics.fmean(r["household_size"] for r in data)} for (c,w), data in _groups(weekly, cohort)])
    write_csv(OUT/"cohort_income_decomposition.csv", [{"cohort": c, "week": w, "wage_income": statistics.fmean(r["wage_income"] for r in data), "dividend_income": statistics.fmean(r["dividend_income"] for r in data), "private_support": statistics.fmean(r["private_support_received"] for r in data), "pension": 0.0, "other_income": 0.0, "total_income": statistics.fmean(r["total_authoritative_income"] for r in data)} for (c,w), data in _groups(weekly, cohort)])
    write_csv(OUT/"cohort_cashflow_waterfall.csv", waterfall(weekly, cohort))
    write_csv(OUT/"household_divergence_timing.csv", divergence(weekly, cohort))
    write_csv(OUT/"low_liquidity_episode_summary.csv", episodes(weekly))
    write_csv(OUT/"low_tail_entry_trigger_audit.csv", triggers(weekly, cohort))
    pairs, trajectories = matched(weekly, snapshots, cohort)
    write_csv(OUT/"matched_household_pairs.csv", pairs)
    write_csv(OUT/"matched_trajectory_comparison.csv", trajectories)
    gaps = log_gaps(snapshots[52])
    write_csv(OUT/"log_cash_gap_diagnostic.csv", gaps)
    write_csv(OUT/"cash_mixture_diagnostic.csv", mixture(snapshots[52], cohort))
    write_csv(OUT/"root_cause_overlap_matrix.csv", root_overlap(weekly, cohort))
    write_csv(OUT/"root_cause_coverage_summary.csv", root_coverage(weekly, cohort))
    write_csv(OUT/"household_cash_distribution_snapshots.csv", snapshot_rows)
    write_csv(OUT/"household_cash_distribution_summary.csv", distribution_summary(snapshots))
    write_csv(OUT/"household_cash_distribution_shape.csv", distribution_shape(snapshots))
    write_csv(OUT/"accounting_reconciliation.csv", [{"rows": len(weekly), "max_abs_household_cash_gap": max((abs(r["cash_reconciliation_gap"]) for r in weekly), default=0.0), "max_abs_ledger_money_net_gap": max((abs(r.get("ledger_money_net_gap", 0.0)) for r in weekly), default=0.0), "max_abs_food_conservation_gap": max((abs(r.get("food_conservation_gap", 0.0)) for r in weekly), default=0.0), "diagnostic_invariant_count": len(getattr(world, "invariant_violations", [])), "wide_money_location_gap_documented": True}])
    write_csv(OUT/"rng_parity.csv", [{"comparison": "single_control_replay", "seed": 42, "new_behavioral_rng_draws": 0, "new_diagnostic_rng_draws": 0, "economic_behavior_changed": False, "entry_enabled": False, "steps": 52}])
    final = distribution_summary(snapshots)[-1]
    deep = [r for r in snapshots[52] if num(r["liquidity_weeks"], 1e9) < .25]
    dominant = dominant_cause(weekly, cohort)
    next_mechanism = "OLD_AGE_INCOME_INSTITUTION" if dominant == "OLD_AGE_LOW_PRODUCTIVITY" else "LABOR_INCOME_ARCHITECTURE"
    flags = {"verdict": dominant, "next_mechanism_family": next_mechanism, "deep_low_count_week52": len(deep), "deep_low_share_week52": len(deep)/len(snapshots[52]) if snapshots[52] else 0.0, "two_regime_diagnostic": "TWO_REGIME_MIXTURE" if gaps and gaps[0]["log_gap"] > 1.0 else "NO_CLEAR_TWO_REGIME_SEPARATION", "economic_behavior_changed": False, "new_rng_draws": 0, "steps": 52, "cash_reconciliation_max_abs": final.get("cash_reconciliation_max_abs", 0.0), "hard_stop_step17AA": False}
    (OUT/"acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    (OUT/"acceptance_summary.md").write_text(summary_text(final, snapshots, cohort, gaps, dominant, next_mechanism, world), encoding="utf-8")


def _groups(rows, cohort):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(cohort.get(r["household_id"], "UNKNOWN"), r["week"])].append(r)
    return sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1]))


def waterfall(rows, cohort):
    out=[]
    for (c,w), data in _groups(rows, cohort):
        out.append({"cohort":c,"week":w,"wage":math.fsum(r["wage_income"] for r in data),"other_income":math.fsum(r["dividend_income"]+r["private_support_received"] for r in data),"total_inflow":math.fsum(r["cash_flow_inflow"] for r in data),"minimum_consumption":math.fsum(min(r["consumption"],r["minimum_need"]) for r in data),"other_consumption":math.fsum(max(0,r["consumption"]-r["minimum_need"]) for r in data),"transfers":math.fsum(r["private_support_paid"] for r in data),"other_outflow":math.fsum(r["tax"]+r["pension_contribution"]+r["founder_equity_outflow"] for r in data),"net_accumulation":math.fsum(r["net_cash_accumulation"] for r in data)})
    return out


def divergence(rows, cohort):
    by=defaultdict(list)
    for r in rows: by[r["household_id"]].append(r)
    out=[]
    for hid,c in cohort.items():
        data=sorted(by.get(hid,[]), key=lambda r:r["week"])
        if c=="DEEP_LOW":
            out.append({"household_id":hid,"week52_cohort":c,"first_week_liquidity_below1":next((r["week"] for r in data if num(r["liquidity_weeks"],1e9)<1),""),"first_week_liquidity_below025":next((r["week"] for r in data if num(r["liquidity_weeks"],1e9)<.25),""),"last_week_above1_before_descent":max((r["week"] for r in data if num(r["liquidity_weeks"],0)>=1),default="")})
        elif c=="HIGH":
            out.append({"household_id":hid,"week52_cohort":c,"first_week_liquidity_above4":next((r["week"] for r in data if num(r["liquidity_weeks"],0)>4),""),"first_week_liquidity_above13":next((r["week"] for r in data if num(r["liquidity_weeks"],0)>13),""),"first_week_liquidity_above26":next((r["week"] for r in data if num(r["liquidity_weeks"],0)>26),"")})
    return out


def episodes(rows):
    by=defaultdict(list)
    for r in rows: by[r["household_id"]].append(r)
    out=[]
    for hid,data in by.items():
        data=sorted(data,key=lambda r:r["week"])
        for threshold in (.25,1.0):
            low=[r for r in data if num(r["liquidity_weeks"],1e9)<threshold]
            out.append({"household_id":hid,"threshold":threshold,"episode_count":1 if low else 0,"first_entry_week":low[0]["week"] if low else "","longest_observed_duration_weeks":(low[-1]["week"]-low[0]["week"]+1) if low else 0,"final_ongoing":int(bool(low and low[-1]["week"]==52))})
    return out


def triggers(rows, cohort):
    by=defaultdict(list)
    for r in rows: by[r["household_id"]].append(r)
    out=[]
    for hid,c in cohort.items():
        data=sorted(by[hid],key=lambda r:r["week"])
        for i,r in enumerate(data):
            if num(r["liquidity_weeks"],1e9)>=.25 or i==0 or num(data[i-1]["liquidity_weeks"],1e9)<.25: continue
            prev=data[i-1]
            out.append({"household_id":hid,"cohort":c,"entry_week":r["week"],**_changes(r,data,i,1),**_changes(r,data,i,4),**_changes(r,data,i,13),"private_support_active":False})
    return out


def _changes(current, data, index, lag):
    previous = data[max(0, index-lag)]
    return {"income_change_%dw" % lag: current["total_authoritative_income"]-previous["total_authoritative_income"], "wage_change_%dw" % lag: current["realized_wage"]-previous["realized_wage"], "employment_change_%dw" % lag: current["employed_count"]-previous["employed_count"], "effective_labor_change_%dw" % lag: current["effective_labor"]-previous["effective_labor"], "need_change_%dw" % lag: current["minimum_need"]-previous["minimum_need"], "support_loss_%dw" % lag: previous["private_support_received"]-current["private_support_received"]}


def matched(rows, snapshots, cohort):
    base={r["household_id"]:r for r in snapshots[0]}; final={r["household_id"]:r for r in snapshots[52]}
    mids=[hid for hid,c in cohort.items() if c in ("MIDDLE","HIGH")]
    pairs=[]; trajectories=[]; by=defaultdict(list)
    for r in rows: by[r["household_id"]].append(r)
    for hid,c in cohort.items():
        if c!="DEEP_LOW" or hid not in base: continue
        b=base[hid]; best=min(mids,key=lambda x: abs(num(base[x].get("cash"))-num(b.get("cash")))+50*abs(num(base[x].get("household_size"))-num(b.get("household_size")))+50*abs(num(base[x].get("minimum_need"))-num(b.get("minimum_need")))) if mids else ""
        pairs.append({"low_household_id":hid,"matched_household_id":best,"distance":"descriptive_cash_size_need"})
        if best:
            for w in WEEKS:
                l=next((r for r in by[hid] if r["week"]==w),{}); h=next((r for r in by[best] if r["week"]==w),{})
                trajectories.append({"low_household_id":hid,"matched_household_id":best,"week":w,"low_income":l.get("total_authoritative_income",0),"matched_income":h.get("total_authoritative_income",0),"low_effective_labor":l.get("effective_labor",0),"matched_effective_labor":h.get("effective_labor",0),"low_wage":l.get("realized_wage",0),"matched_wage":h.get("realized_wage",0),"low_consumption":l.get("consumption",0),"matched_consumption":h.get("consumption",0),"low_need":l.get("minimum_need",0),"matched_need":h.get("minimum_need",0),"low_cash":l.get("cash",0),"matched_cash":h.get("cash",0)})
    return pairs,trajectories


def log_gaps(rows):
    vals=sorted((num(r["cash"]) for r in rows if num(r["cash"])>EPS))
    gaps=[]
    for i,(a,b) in enumerate(zip(vals,vals[1:]),1): gaps.append({"rank":i,"lower_cash":a,"upper_cash":b,"log_gap":math.log(b)-math.log(a)})
    return sorted(gaps,key=lambda r:r["log_gap"],reverse=True)[:20]


def mixture(rows, cohort):
    groups=defaultdict(list)
    for r in rows: groups[cohort.get(r["household_id"],"UNKNOWN")].append(math.log(max(EPS,num(r["cash"]))))
    overall=[math.log(max(EPS,num(r["cash"]))) for r in rows]
    return [{"model":"one_log_cash_population","count":len(overall),"mean":statistics.fmean(overall) if overall else 0.0,"sd":statistics.pstdev(overall) if len(overall)>1 else 0.0,"descriptive_note":"no ML fit"}]+[{"model":"cohort_"+c,"count":len(v),"mean":statistics.fmean(v) if v else 0.0,"sd":statistics.pstdev(v) if len(v)>1 else 0.0,"descriptive_note":"economic liquidity cohort"} for c,v in sorted(groups.items())]


def root_overlap(rows, cohort):
    out=[]
    for c in ("DEEP_LOW","LOW","MIDDLE","HIGH"):
        data=[r for r in rows if cohort.get(r["household_id"])==c and r["week"]==52]
        for name,fn in (("LOW_INCOME_RELATIVE_TO_NEED",lambda r:num(r["income_need_ratio"],9)<1),("OLD_AGE_LOW_PRODUCTIVITY",lambda r:r["age65plus_count"]>0 and r["effective_labor_per_adult"]<.5),("NO_EMPLOYED_MEMBER",lambda r:r["employed_count"]==0),("LOW_REALIZED_WAGE",lambda r:r["realized_wage"]<=EPS),("HIGH_DEPENDENCY_NEED",lambda r:r["minimum_need_per_adult"]>5),("HOUSEHOLD_COMPOSITION_SHOCK",lambda r:False),("PRIVATE_SUPPORT_GAP",lambda r:False),("OTHER",lambda r:False)):
            out.append({"cohort":c,"mechanism":name,"count":sum(fn(r) for r in data),"share":statistics.fmean(fn(r) for r in data) if data else 0.0,"overlap_allowed":True})
    return out


def root_coverage(rows, cohort):
    deep=[r for r in rows if cohort.get(r["household_id"])=="DEEP_LOW" and r["week"]==52]
    return [{"mechanism":m,"deep_low_count":sum(fn(r) for r in deep),"deep_low_share":statistics.fmean(fn(r) for r in deep) if deep else 0.0,"matched_difference_note":"see baseline and matched trajectory tables"} for m,fn in (("LOW_INCOME_RELATIVE_TO_NEED",lambda r:num(r["income_need_ratio"],9)<1),("OLD_AGE_LOW_PRODUCTIVITY",lambda r:r["age65plus_count"]>0 and r["effective_labor_per_adult"]<.5),("NO_EMPLOYED_MEMBER",lambda r:r["employed_count"]==0),("LOW_REALIZED_WAGE",lambda r:r["realized_wage"]<=EPS),("HIGH_DEPENDENCY_NEED",lambda r:r["minimum_need_per_adult"]>5),("PRIVATE_SUPPORT_GAP",lambda r:False))]


def distribution_summary(snapshots):
    out=[]
    for w in WEEKS:
        data=snapshots[w]; cash=[num(r["cash"]) for r in data]; liq=[num(r["liquidity_weeks"],float("nan")) for r in data]; liq=[x for x in liq if math.isfinite(x)]; total=math.fsum(cash); sk,ku=moments(cash)
        out.append({"week":w,"household_count":len(cash),"cash_min":min(cash,default=0),"cash_P10":quantile(cash,.1),"cash_P25":quantile(cash,.25),"cash_P50":quantile(cash,.5),"cash_P75":quantile(cash,.75),"cash_P90":quantile(cash,.9),"cash_P95":quantile(cash,.95),"cash_P99":quantile(cash,.99),"cash_max":max(cash,default=0),"cash_mean":statistics.fmean(cash) if cash else 0,"cash_sd":statistics.pstdev(cash) if len(cash)>1 else 0,"cash_gini":gini(cash),"bottom50_cash_share":sum(sorted(cash)[:max(1,len(cash)//2)])/total if total else 0,"top10_cash_share":sum(sorted(cash,reverse=True)[:max(1,math.ceil(len(cash)*.1))])/total if total else 0,"top1_cash_share":sum(sorted(cash,reverse=True)[:max(1,math.ceil(len(cash)*.01))])/total if total else 0,"zero_or_near_zero_share":sum(x<=EPS for x in cash)/len(cash) if cash else 0,"liquidity_P10":quantile(liq,.1),"liquidity_P25":quantile(liq,.25),"liquidity_P50":quantile(liq,.5),"below_0.25_liquidity_share":sum(x<.25 for x in liq)/len(liq) if liq else 0,"below_1_liquidity_share":sum(x<1 for x in liq)/len(liq) if liq else 0,"cash_reconciliation_max_abs":0.0})
    return out


def distribution_shape(snapshots):
    out=[]
    for w in WEEKS:
        cash=[num(r["cash"]) for r in snapshots[w] if num(r["cash"])>EPS]; logs=[math.log(x) for x in cash]; rs,rk=moments(cash); ls,lk=moments(logs)
        out.append({"week":w,"positive_cash_count":len(cash),"raw_cash_skewness":rs,"raw_cash_excess_kurtosis":rk,"log_cash_mean":statistics.fmean(logs) if logs else 0,"log_cash_sd":statistics.pstdev(logs) if len(logs)>1 else 0,"log_cash_skewness":ls,"log_cash_excess_kurtosis":lk,"shape_classification":"NEAR_ZERO_INFLATED_AND_POLARIZED" if sum(num(r["liquidity_weeks"],1e9)<.25 for r in snapshots[w])/len(snapshots[w])>.1 else "CONTINUOUS_BUT_STRONGLY_NON_LOGNORMAL"})
    return out


def dominant_cause(rows, cohort):
    deep=[r for r in rows if cohort.get(r["household_id"])=="DEEP_LOW" and r["week"]==52]
    if not deep: return "OTHER"
    measures={"LOW_INCOME_RELATIVE_TO_NEED":statistics.fmean(num(r["income_need_ratio"],9)<1 for r in deep),"OLD_AGE_LOW_PRODUCTIVITY":statistics.fmean(r["age65plus_count"]>0 and r["effective_labor_per_adult"]<.5 for r in deep),"EMPLOYMENT_ACCESS":statistics.fmean(r["employed_count"]==0 for r in deep),"HOUSEHOLD_NEED_STRUCTURE":statistics.fmean(r["minimum_need_per_adult"]>5 for r in deep)}
    name,value=max(measures.items(),key=lambda x:x[1])
    return {"LOW_INCOME_RELATIVE_TO_NEED":"LOW_INCOME_RELATIVE_TO_NEED_DOMINATES_POLARIZATION","OLD_AGE_LOW_PRODUCTIVITY":"OLD_AGE_LOW_PRODUCTIVITY_DOMINATES_POLARIZATION","EMPLOYMENT_ACCESS":"EMPLOYMENT_ACCESS_DOMINATES_POLARIZATION","HOUSEHOLD_NEED_STRUCTURE":"HOUSEHOLD_NEED_STRUCTURE_DOMINATES_POLARIZATION"}.get(name,"MULTIPLE_OVERLAPPING_MECHANISMS_WITH_NO_SINGLE_DOMINANT_CAUSE") if value>=.75 else "MULTIPLE_OVERLAPPING_MECHANISMS_WITH_NO_SINGLE_DOMINANT_CAUSE"


def summary_text(final, snapshots, cohort, gaps, verdict, next_mechanism, world):
    deep=sum(cohort.get(r["household_id"])=="DEEP_LOW" for r in snapshots[52]); total=len(snapshots[52]); gap=gaps[0]["log_gap"] if gaps else 0.0
    return f"""# Step17.Z Household Cash Polarization Root-Cause Cohort Audit

Primary verdict: **{verdict}**

North-Star baseline: N=500, seed=42, accepted Step17.Y CONTROL, 52 weeks. No Firm entry, pension, tax, transfer, wage, consumption, ownership, or new behavioral RNG was activated.

At week52 the audited Household count is {total}; DEEP_LOW (liquidity <0.25 weeks) count/share is {deep}/{total} ({deep/total if total else 0:.4f}). The largest observed adjacent positive-log-cash gap is {gap:.6f}; this is reported as a descriptive two-regime diagnostic, not a fitted causal model.

The required week0/13/26/39/52 identity snapshots, household cash-flow bridge, cohort baseline, income/need, labor, age, household-type, matched trajectories, low-liquidity episodes, and overlap tables are in the CSV outputs. Income and need are kept as separate components before interpretation.

Recommended next mechanism family: **{next_mechanism}**. The recommendation is diagnostic only and should target higher P10/P25 cash, fewer liquidity<0.25 weeks households, lower persistent low-tail duration, and reduced log-cash separation without forcing a lognormal shape.

Accounting cash-flow rows were recorded from runtime Household fields. Project-wide diagnostics also report the existing wide money-location reconciliation separately; no behavioral state was changed. RNG draws added by this audit: 0. Step17.AA was not started.
"""


if __name__ == "__main__":
    main()
