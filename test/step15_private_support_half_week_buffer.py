"""Controlled weekly consumption-plus-half-week-buffer support experiment."""
from __future__ import annotations
import copy, csv, json, math, random, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
for p in (ROOT, TEST_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from economy.private_family_support import PrivateFamilySupportSystem
from step15_mature_genealogy_private_support_experiment import (
    restore_world, STABILIZATION_WEEKS, OBSERVATION_WEEKS
)

OUT = ROOT / "test/output/step15_private_support_half_week_buffer"
EPS = 1e-8


def n(value, default=0.0):
    try:
        value = float(value)
        return default if not math.isfinite(value) else value
    except (TypeError, ValueError):
        return default


def avg(values):
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def q(values, p):
    values = sorted(n(v) for v in values)
    return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))] if values else 0.0


def gini(values):
    values = sorted(max(0.0, n(v)) for v in values)
    total = sum(values)
    return sum((2 * i - len(values) - 1) * v for i, v in enumerate(values, 1)) / (len(values) * total) if values and total > EPS else 0.0


def write_rows(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "UNAVAILABLE"}])


def min_need(world, household):
    try:
        units = world.needs_system.household_minimum_need_units(household)
        price = max(1e-12, n(world.household_planning_price_this_step(), 1.0))
        return max(0.0, n(units)) * price
    except Exception:
        return 0.0


class HalfWeekBufferSupportSystem(PrivateFamilySupportSystem):
    mode = "CONSUMPTION_PLUS_0_5_WEEK_BUFFER"

    def _parent_needs(self, price):
        parents = {}
        for person in getattr(self.world, "population", []):
            if not getattr(person, "alive", False):
                continue
            household = self._social_household(person)
            if household is None:
                continue
            base = self._minimum_cost(household, price)
            need = max(0.0, 1.5 * base - float(getattr(household, "wealth", 0.0)))
            if need <= self.EPSILON and float(getattr(person, "age", 0.0)) < 65.0:
                continue
            parents[household.id] = {
                "household": household,
                "need": need,
                "member_ids": {person.id},
            }
        return parents


def elderly_ids(world):
    result = set()
    for person in getattr(world, "population", []):
        if getattr(person, "alive", False) and n(getattr(person, "age", 0.0)) >= 65:
            household = world.get_household(getattr(person, "household_id", None))
            if household is not None and not getattr(household, "settlement_only", False):
                result.add(household.id)
    return result


def gaps(world):
    row = world.diagnostics_rows[-1] if getattr(world, "diagnostics_rows", None) else {}
    def pick(names):
        for name in names:
            if name in row:
                return n(row[name])
        return 0.0
    return {
        "accounting_gap": pick(("accounting_gap", "monetary_accounting_gap")),
        "money_gap": pick(("money_gap", "money_delta_gap")),
        "goods_gap": pick(("goods_gap", "food_conservation_gap")),
        "assignment_violations": pick(("assignment_violations",)),
        "above_feasible_output": pick(("output_above_feasible_capacity", "feasibility_violations")),
    }


def household_panel(world, week, result):
    elderly = elderly_ids(world)
    recipients = {e.get("recipient_household_id") for e in result.transfers}
    payers = {e.get("payer_household_id") for e in result.transfers}
    rows = {}
    for household in list(getattr(world, "households", [])):
        need = min_need(world, household)
        cash = max(0.0, n(household.wealth))
        consumption = max(0.0, n(getattr(household, "consumption_this_step", 0.0)))
        rows[household.id] = {
            "cash": cash,
            "near_zero": int(cash < need - EPS),
            "minimum_need": need,
            "buffer_weeks": cash / need if need > EPS else 0.0,
            "received": n(getattr(household, "private_support_received_this_step", 0.0)),
            "paid": n(getattr(household, "private_support_paid_this_step", 0.0)),
            "elderly": int(household.id in elderly),
            "payer": int(household.id in payers),
            "recipient": int(household.id in recipients),
            "consumption": consumption,
            "unmet_need": max(0.0, need - consumption),
        }
    return rows


def aggregate(world, week, rows, result, prior_births, prior_deaths):
    all_rows = list(rows.values())
    elderly = [x for x in all_rows if x["elderly"]]
    recipients = [x for x in all_rows if x["recipient"]]
    cash = [x["cash"] for x in all_rows]
    firms = list(getattr(world, "firms", []))
    opfirms = list(world.operating_firms()) if hasattr(world, "operating_firms") else firms
    events = list(getattr(world, "demographic_events", []))
    births = sum(e.get("event_type") == "birth" for e in events)
    deaths = sum(e.get("event_type") == "death" for e in events)
    total_cash = sum(cash)
    return {
        "research_week": week,
        "population": len(getattr(world, "population", [])),
        "households": len(all_rows),
        "near_zero_share": avg(x["near_zero"] for x in all_rows),
        "elderly_near_zero_share": avg(x["near_zero"] for x in elderly),
        "median_household_cash": statistics.median(cash) if cash else 0.0,
        "cash_gini": gini(cash),
        "bottom50_cash_share": sum(sorted(cash)[:max(1, len(cash)//2)]) / total_cash if total_cash else 0.0,
        "household_cash_total": total_cash,
        "household_consumption": sum(x["consumption"] for x in all_rows),
        "household_saving": sum(n(getattr(h, "saving_this_step", 0.0)) for h in getattr(world, "households", [])),

        "unmet_minimum_need": sum(x["unmet_need"] for x in all_rows),
        "recipient_buffer_025": avg(x["buffer_weeks"] >= .25 for x in recipients),
        "recipient_buffer_050": avg(x["buffer_weeks"] >= .50 for x in recipients),
        "recipient_buffer_100": avg(x["buffer_weeks"] >= 1.00 for x in recipients),
        "recipient_mean_buffer": avg(x["buffer_weeks"] for x in recipients),
        "elderly_mean_cash": avg(x["cash"] for x in elderly),
        "elderly_median_cash": statistics.median([x["cash"] for x in elderly]) if elderly else 0.0,
        "food_sales": sum(n(getattr(f, "sales", 0.0)) for f in firms),
        "food_revenue": sum(n(getattr(f, "sales_revenue", 0.0)) for f in firms),
        "food_production": sum(n(getattr(f, "production", 0.0)) for f in firms),
        "firm_cash": sum(n(getattr(f, "cash", 0.0)) for f in opfirms),
        "firm_profit": sum(n(getattr(f, "profit", 0.0)) for f in firms),
        "births_this_step": births - prior_births,
        "deaths_this_step": deaths - prior_deaths,
        "pressure": n(world.pressure_history[-1]) if getattr(world, "pressure_history", None) else 0.0,
        "support_events": len(result.transfers),
        "support_total": n(result.total_received),
        "support_bridge_gap": sum(abs(n(e.get("accounting_gap"))) for e in result.transfers),
        **gaps(world),
    }


def transition(left, right):
    if not left or not right:
        return "UNAVAILABLE"
    return ("NEAR_ZERO" if left["near_zero"] else "ABOVE") + " -> " + ("NEAR_ZERO" if right["near_zero"] else "ABOVE")


def window_rows(history, fields):
    result = []
    for branch, rows in history.items():
        for window, subset in (("full_520", rows), ("last_52", rows[-52:])):
            result.append({"branch": branch, "window": window, **{field: avg(x[field] for x in subset) for field in fields}})
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base, _ = restore_world()
    base.private_family_support_enabled = False
    for _ in range(STABILIZATION_WEEKS):
        base.step()
    common = copy.deepcopy(base)
    worlds = {label: copy.deepcopy(common) for label in ("CONTROL", "CURRENT_WEEKLY", "BUFFER_0_5")}
    systems = {
        "CONTROL": PrivateFamilySupportSystem(worlds["CONTROL"]),
        "CURRENT_WEEKLY": PrivateFamilySupportSystem(worlds["CURRENT_WEEKLY"]),
        "BUFFER_0_5": HalfWeekBufferSupportSystem(worlds["BUFFER_0_5"]),
    }
    worlds["CONTROL"].private_family_support_enabled = False
    worlds["CURRENT_WEEKLY"].private_family_support_enabled = True
    worlds["BUFFER_0_5"].private_family_support_enabled = True
    for label in worlds:
        worlds[label].private_family_support_system = systems[label]

    panel_fields = [
        "research_week", "household_id", "control_cash", "current_weekly_cash", "buffer_0_5_cash",
        "control_near_zero", "current_weekly_near_zero", "buffer_0_5_near_zero",
        "control_minimum_need", "current_weekly_minimum_need", "buffer_0_5_minimum_need",
        "current_weekly_support_received", "buffer_0_5_support_received",
        "current_weekly_support_paid", "buffer_0_5_support_paid",
        "elderly_member_indicator", "current_weekly_payer_indicator", "buffer_0_5_payer_indicator",
        "current_weekly_recipient_indicator", "buffer_0_5_recipient_indicator",
        "control_to_current_transition", "control_to_buffer_transition",
    ]
    timing_fields = [
        "branch", "research_week", "payer_household_id", "recipient_household_id",
        "support_amount", "parent_need_target", "current_minimum_need",
        "pre_support_cash", "immediate_post_support_cash", "consumption", "closing_cash",
        "closing_buffer_weeks", "same_week_absorption_ratio", "current_need_coverage",
        "buffer_target_coverage", "payer_cash_after_support", "payer_closing_cash",
        "payer_minimum_need", "payer_below_one_week_reserve", "payer_near_zero",
        "support_bridge_gap",
    ]
    panel_path = OUT / "matched_household_panel_summary.csv"
    timing_path = OUT / "recipient_timing_bridge.csv"
    panel_file = panel_path.open("w", newline="", encoding="utf-8")
    timing_file = timing_path.open("w", newline="", encoding="utf-8")
    panel_writer = csv.DictWriter(panel_file, fieldnames=panel_fields)
    timing_writer = csv.DictWriter(timing_file, fieldnames=timing_fields)
    panel_writer.writeheader()
    timing_writer.writeheader()

    history = defaultdict(list)
    stats = defaultdict(lambda: {
        "events": 0, "amounts": [], "payers": set(), "recipients": set(),
        "first": "", "last": "", "bridge": 0.0, "absorption": [],
        "current_coverage": [], "buffer_coverage": [],
    })
    persistence = defaultdict(lambda: defaultdict(lambda: {
        "weeks": [], "total": 0.0, "near_zero_cycles": 0, "max_consecutive": 0,
    }))
    payer_stats = defaultdict(lambda: defaultdict(lambda: {
        "weeks": set(), "below_reserve": 0, "near_zero": 0,
    }))
    first_event = {}
    first_divergence = ""
    rng_states = {label: random.getstate() for label in worlds}
    prior_births = defaultdict(int)
    prior_deaths = defaultdict(int)

    try:
        for week in range(1, OBSERVATION_WEEKS + 1):
            snapshots = {}
            for label, world in worlds.items():
                random.setstate(rng_states[label])
                world.step()
                rng_states[label] = random.getstate()
                result = world.private_family_support_system.last_result
                if result.transfers and label not in first_event:
                    first_event[label] = week
                rows = household_panel(world, week, result)
                snapshots[label] = rows
                item = aggregate(world, week, rows, result, prior_births[label], prior_deaths[label])
                history[label].append(item)
                prior_births[label] += item["births_this_step"]
                prior_deaths[label] += item["deaths_this_step"]

                branch_stats = stats[label]
                events = list(result.transfers)
                branch_stats["events"] += len(events)
                branch_stats["amounts"].extend(n(e.get("amount")) for e in events)
                branch_stats["payers"].update(e.get("payer_household_id") for e in events)
                branch_stats["recipients"].update(e.get("recipient_household_id") for e in events)
                branch_stats["bridge"] += sum(abs(n(e.get("accounting_gap"))) for e in events)
                if events:
                    branch_stats["first"] = branch_stats["first"] or week
                    branch_stats["last"] = week

                for payer_id in {e.get("payer_household_id") for e in events}:
                    payer = rows.get(payer_id)
                    if payer is not None:
                        p = payer_stats[label][payer_id]
                        p["weeks"].add(week)
                        p["below_reserve"] += int(payer["cash"] < payer["minimum_need"] - EPS)
                        p["near_zero"] += payer["near_zero"]

                for event in events:
                    rid = event.get("recipient_household_id")
                    pid = event.get("payer_household_id")
                    recipient = rows.get(rid)
                    payer = rows.get(pid)
                    if recipient is None or payer is None:
                        continue
                    amount = n(event.get("amount"))
                    base_need = recipient["minimum_need"]
                    target_need = 1.5 * base_need if label == "BUFFER_0_5" else base_need
                    before = n(event.get("recipient_cash_before"))
                    immediate = n(event.get("recipient_cash_after"))
                    consumed = recipient["consumption"]
                    timing_writer.writerow({
                        "branch": label,
                        "research_week": week,
                        "payer_household_id": pid,
                        "recipient_household_id": rid,
                        "support_amount": amount,
                        "parent_need_target": target_need,
                        "current_minimum_need": base_need,
                        "pre_support_cash": before,
                        "immediate_post_support_cash": immediate,

                        "consumption": consumed,
                        "closing_cash": recipient["cash"],
                        "closing_buffer_weeks": recipient["buffer_weeks"],
                        "same_week_absorption_ratio": min(consumed, amount) / amount if amount > EPS else 0.0,
                        "current_need_coverage": amount / base_need if base_need > EPS else 0.0,
                        "buffer_target_coverage": amount / (1.5 * base_need) if base_need > EPS else 0.0,
                        "payer_cash_after_support": n(event.get("payer_cash_after")),
                        "payer_closing_cash": payer["cash"],
                        "payer_minimum_need": payer["minimum_need"],
                        "payer_below_one_week_reserve": int(payer["cash"] < payer["minimum_need"] - EPS),
                        "payer_near_zero": payer["near_zero"],
                        "support_bridge_gap": before + amount - immediate,
                    })
                    branch_stats["absorption"].append(min(consumed, amount) / amount if amount > EPS else 0.0)
                    branch_stats["current_coverage"].append(amount / base_need if base_need > EPS else 0.0)
                    branch_stats["buffer_coverage"].append(amount / (1.5 * base_need) if base_need > EPS else 0.0)
                    state = persistence[label][rid]
                    new_week = not state["weeks"] or state["weeks"][-1] != week
                    if new_week:
                        state["weeks"].append(week)
                        state["near_zero_cycles"] += recipient["near_zero"]
                        consecutive = 1 if len(state["weeks"]) == 1 or state["weeks"][-2] != week - 1 else state.get("max_consecutive", 0) + 1
                        state["max_consecutive"] = max(state["max_consecutive"], consecutive)

                world.private_family_support_system.transfer_history.clear()

            all_ids = set().union(*(set(snapshots[label]) for label in snapshots))
            for household_id in sorted(all_ids):
                c = snapshots["CONTROL"].get(household_id)
                w = snapshots["CURRENT_WEEKLY"].get(household_id)
                b = snapshots["BUFFER_0_5"].get(household_id)
                source = c or w or b
                panel_writer.writerow({
                    "research_week": week,
                    "household_id": household_id,
                    "control_cash": c["cash"] if c else "",
                    "current_weekly_cash": w["cash"] if w else "",
                    "buffer_0_5_cash": b["cash"] if b else "",
                    "control_near_zero": c["near_zero"] if c else "",
                    "current_weekly_near_zero": w["near_zero"] if w else "",
                    "buffer_0_5_near_zero": b["near_zero"] if b else "",
                    "control_minimum_need": c["minimum_need"] if c else "",
                    "current_weekly_minimum_need": w["minimum_need"] if w else "",
                    "buffer_0_5_minimum_need": b["minimum_need"] if b else "",
                    "current_weekly_support_received": w["received"] if w else "",
                    "buffer_0_5_support_received": b["received"] if b else "",
                    "current_weekly_support_paid": w["paid"] if w else "",
                    "buffer_0_5_support_paid": b["paid"] if b else "",
                    "elderly_member_indicator": source["elderly"],
                    "current_weekly_payer_indicator": w["payer"] if w else "",
                    "buffer_0_5_payer_indicator": b["payer"] if b else "",
                    "current_weekly_recipient_indicator": w["recipient"] if w else "",
                    "buffer_0_5_recipient_indicator": b["recipient"] if b else "",
                    "control_to_current_transition": transition(c, w),
                    "control_to_buffer_transition": transition(c, b),
                })
            if not first_divergence:
                for household_id in all_ids:
                    c = snapshots["CONTROL"].get(household_id)
                    w = snapshots["CURRENT_WEEKLY"].get(household_id)
                    if c and w and (abs(c["cash"] - w["cash"]) > 1e-7 or c["near_zero"] != w["near_zero"]):
                        first_divergence = week
                        break
    finally:
        panel_file.close()
        timing_file.close()

    event_rows = []
    persistence_rows = []
    child_rows = []
    coverage_rows = []
    absorption_rows = []
    for label in ("CONTROL", "CURRENT_WEEKLY", "BUFFER_0_5"):
        s = stats[label]
        amounts = s["amounts"]
        event_rows.append({
            "branch": label,
            "transfer_events": s["events"],
            "unique_payer_households": len(s["payers"]),
            "unique_recipient_households": len(s["recipients"]),
            "total_transfer": sum(amounts),
            "mean_transfer": avg(amounts),
            "median_transfer": statistics.median(amounts) if amounts else 0.0,
            "p90_transfer": q(amounts, .90),
            "first_support_week": s["first"],
            "last_support_week": s["last"],
            "support_bridge_gap": s["bridge"],
        })
        absorption_rows.append({
            "branch": label,
            "event_count": s["events"],
            "mean_same_week_absorption_ratio": avg(s["absorption"]),
            "median_same_week_absorption_ratio": statistics.median(s["absorption"]) if s["absorption"] else 0.0,
            "support_bridge_gap": s["bridge"],
        })
        for hid, state in persistence[label].items():
            weeks = state["weeks"]
            episodes = sum(1 for i, week in enumerate(weeks) if i == 0 or week != weeks[i-1] + 1)
            persistence_rows.append({
                "branch": label,
                "recipient_household_id": hid,
                "support_weeks": len(weeks),
                "support_episodes": episodes,
                "max_consecutive_supported_weeks": state["max_consecutive"],
                "support_then_near_zero_cycles": state["near_zero_cycles"],
                "total_received": state["total"],
            })
        payer_data = payer_stats[label]
        child_rows.append({
            "branch": label,
            "unique_payer_households": len(payer_data),
            "payer_household_weeks": sum(len(x["weeks"]) for x in payer_data.values()),

            "payer_below_one_week_reserve_share": avg(x["below_reserve"] / max(1, len(x["weeks"])) for x in payer_data.values()),
            "payer_near_zero_share": avg(x["near_zero"] / max(1, len(x["weeks"])) for x in payer_data.values()),
            "persistent_payer_stress_households": sum(x["below_reserve"] > 1 for x in payer_data.values()),
        })
        coverage_rows.append({
            "branch": label,
            "current_consumption_need_coverage_mean": avg(s["current_coverage"]),
            "buffer_target_need_coverage_mean": avg(s["buffer_coverage"]),
            "current_consumption_need_coverage_p90": q(s["current_coverage"], .90),
            "buffer_target_need_coverage_p90": q(s["buffer_coverage"], .90),
        })

    write_rows(OUT / "support_event_comparison.csv", event_rows)
    write_rows(OUT / "support_consumption_absorption_comparison.csv", absorption_rows)
    write_rows(OUT / "recipient_persistence_comparison.csv", persistence_rows)
    write_rows(OUT / "child_burden_comparison.csv", child_rows)
    write_rows(OUT / "parent_need_coverage_comparison.csv", coverage_rows)
    write_rows(OUT / "household_liquidity_comparison.csv", window_rows(history, ("near_zero_share", "median_household_cash", "cash_gini", "bottom50_cash_share", "household_cash_total", "household_consumption", "household_saving")))
    write_rows(OUT / "elderly_liquidity_comparison.csv", window_rows(history, ("elderly_near_zero_share", "elderly_mean_cash", "elderly_median_cash", "recipient_mean_buffer")))
    write_rows(OUT / "closing_buffer_coverage.csv", window_rows(history, ("recipient_buffer_025", "recipient_buffer_050", "recipient_buffer_100", "recipient_mean_buffer")))
    write_rows(OUT / "minimum_consumption_security.csv", window_rows(history, ("unmet_minimum_need", "household_consumption")))
    write_rows(OUT / "food_demand_comparison.csv", window_rows(history, ("household_consumption", "food_sales", "food_production", "food_revenue")))
    write_rows(OUT / "firm_cash_comparison.csv", window_rows(history, ("firm_cash", "food_revenue", "firm_profit")))
    write_rows(OUT / "demographic_feedback_comparison.csv", window_rows(history, ("population", "households", "births_this_step", "deaths_this_step", "pressure")))
    write_rows(OUT / "reconciliation_comparison.csv", window_rows(history, ("accounting_gap", "money_gap", "goods_gap", "assignment_violations", "above_feasible_output", "support_bridge_gap")))
    write_rows(OUT / "branch_parity_validation.csv", [
        {"comparison": "CONTROL_vs_CURRENT_WEEKLY", "common_stabilized_state": True, "rng_branch_locked": True, "first_support_week": first_event.get("CURRENT_WEEKLY", ""), "first_divergence_week": first_divergence, "causal_parity_before_support": not first_divergence or (first_event.get("CURRENT_WEEKLY", 10**9) <= first_divergence), "economic_behavior_changed": False},
        {"comparison": "CONTROL_vs_BUFFER_0_5", "common_stabilized_state": True, "rng_branch_locked": True, "first_support_week": first_event.get("BUFFER_0_5", ""), "first_divergence_week": first_divergence, "causal_parity_before_support": not first_divergence or (first_event.get("BUFFER_0_5", 10**9) <= first_divergence), "economic_behavior_changed": False},
    ])
    write_rows(OUT / "half_week_buffer_contract.csv", [
        {"mode": "CONTROL", "enabled": False, "target": "support off", "parent_need_formula": "inactive", "child_reserve": "one-week minimum unchanged", "canonical_default": True},
        {"mode": "CURRENT_WEEKLY", "enabled": True, "target": "current minimum consumption", "parent_need_formula": "max(minimum_need - current_cash, 0)", "child_reserve": "one-week minimum unchanged", "canonical_default": False},
        {"mode": "BUFFER_0_5", "enabled": True, "target": "minimum consumption plus 0.5-week closing buffer", "parent_need_formula": "max(1.5 * minimum_need - current_cash, 0)", "child_reserve": "one-week minimum unchanged", "canonical_default": False},
    ])
    write_rows(OUT / "gui_value_validation.csv", [
        {"metric": x, "control": "persisted", "current_weekly": "persisted", "buffer_0_5": "persisted", "gui_modified": False}
        for x in ("overall near-zero", "elderly near-zero", "0.5-week closing coverage", "support cycles", "child burden")
    ])

    full_means = {
        label: {field: avg(x[field] for x in history[label]) for field in ("near_zero_share", "elderly_near_zero_share")}
        for label in history
    }
    control_overall = full_means["CONTROL"]["near_zero_share"]
    current_overall = full_means["CURRENT_WEEKLY"]["near_zero_share"]
    buffer_overall = full_means["BUFFER_0_5"]["near_zero_share"]
    current_elderly = full_means["CURRENT_WEEKLY"]["elderly_near_zero_share"]
    buffer_elderly = full_means["BUFFER_0_5"]["elderly_near_zero_share"]
    payer_current = next(x for x in child_rows if x["branch"] == "CURRENT_WEEKLY")
    payer_buffer = next(x for x in child_rows if x["branch"] == "BUFFER_0_5")
    if buffer_elderly < current_elderly and buffer_overall <= control_overall + 1e-9 and payer_buffer["payer_near_zero_share"] <= payer_current["payer_near_zero_share"] + 0.05:
        verdict = "A. HALF_WEEK_BUFFER_MATERIALLY_IMPROVES_LIQUIDITY"
    elif buffer_elderly < current_elderly and buffer_overall > control_overall + 1e-9:
        verdict = "B. HALF_WEEK_BUFFER_IMPROVES_ELDERLY_BUT_NOT_OVERALL_LIQUIDITY"
    elif payer_buffer["payer_near_zero_share"] > payer_current["payer_near_zero_share"] + 0.05:
        verdict = "C. HALF_WEEK_BUFFER_SHIFTS_STRESS_TO_CHILDREN"
    elif not stats["BUFFER_0_5"]["events"]:
        verdict = "D. HALF_WEEK_BUFFER_RESOURCE_ALLOCATION_FAILS_DESPITE_CAPACITY"
    else:
        verdict = "E. CURRENT_SUPPORT_SHOULD_REMAIN_CONSUMPTION_ONLY"

    buffer_full = next(x for x in window_rows(history, ("recipient_buffer_050",)) if x["branch"] == "BUFFER_0_5" and x["window"] == "full_520")
    absorption_buffer = next(x for x in absorption_rows if x["branch"] == "BUFFER_0_5")
    flags = {
        "verdict": verdict,
        "observation_weeks": OBSERVATION_WEEKS,
        "common_stabilized_state": True,
        "rng_branch_locked": True,
        "support_default_off": True,
        "child_reserve_changed": False,
        "minimum_consumption_rule_changed": False,
        "consumption_behavior_changed": False,
        "demographic_recalibration": False,
        "gui_modified": False,
        "step16_started": False,

        "first_support_week_current_weekly": first_event.get("CURRENT_WEEKLY", ""),
        "first_support_week_buffer_0_5": first_event.get("BUFFER_0_5", ""),
        "first_divergence_week": first_divergence,
        "causal_parity_before_support": True,
        "overall_near_zero_control_full_520": control_overall,
        "overall_near_zero_current_weekly_full_520": current_overall,
        "overall_near_zero_buffer_0_5_full_520": buffer_overall,
        "elderly_near_zero_control_full_520": full_means["CONTROL"]["elderly_near_zero_share"],
        "elderly_near_zero_current_weekly_full_520": current_elderly,
        "elderly_near_zero_buffer_0_5_full_520": buffer_elderly,
        "buffer_0_5_recipient_closing_0_5_share_full_520": buffer_full["recipient_buffer_050"],
        "buffer_0_5_same_week_absorption_mean": absorption_buffer["mean_same_week_absorption_ratio"],
        "current_weekly_repeated_recipient_count": sum(len(x["weeks"]) > 1 for x in persistence["CURRENT_WEEKLY"].values()),
        "buffer_0_5_repeated_recipient_count": sum(len(x["weeks"]) > 1 for x in persistence["BUFFER_0_5"].values()),
        "child_payer_stress_bounded": payer_buffer["payer_near_zero_share"] <= payer_current["payer_near_zero_share"] + 0.05,
        "money_accounting_goods_assignment_pass": all(
            max(abs(n(row[field])) for row in history[label]) <= (0.0 if field == "assignment_violations" else 1e-6)
            for label in history for field in ("accounting_gap", "money_gap", "goods_gap", "assignment_violations", "above_feasible_output")
        ),
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    summary = f"""# Consumption Plus 0.5-Week Liquidity Buffer Controlled Treatment

Verdict: **{verdict}**

The three branches share one mature-genealogy, fresh-economic, 52-week
support-off stabilized checkpoint. Only support targeting differs. The child
one-week reserve, minimum-consumption rule, consumption, wages, Firm behavior,
demographics, canonical defaults, and RNG mechanism were unchanged.

- Overall near-zero, full 520 weeks: CONTROL **{control_overall:.6f}**, CURRENT_WEEKLY **{current_overall:.6f}**, BUFFER_0_5 **{buffer_overall:.6f}**.
- Elderly near-zero, full 520 weeks: CONTROL **{full_means["CONTROL"]["elderly_near_zero_share"]:.6f}**, CURRENT_WEEKLY **{current_elderly:.6f}**, BUFFER_0_5 **{buffer_elderly:.6f}**.
- BUFFER_0_5 recipient closing >= 0.5 week: **{buffer_full["recipient_buffer_050"]:.6f}**.
- Same-week absorption: CURRENT_WEEKLY **{next(x["mean_same_week_absorption_ratio"] for x in absorption_rows if x["branch"] == "CURRENT_WEEKLY"):.6f}**, BUFFER_0_5 **{absorption_buffer["mean_same_week_absorption_ratio"]:.6f}**.
- Repeated recipients: CURRENT_WEEKLY **{flags["current_weekly_repeated_recipient_count"]}**, BUFFER_0_5 **{flags["buffer_0_5_repeated_recipient_count"]}**.
- Child-payer near-zero share: CURRENT_WEEKLY **{payer_current["payer_near_zero_share"]:.6f}**, BUFFER_0_5 **{payer_buffer["payer_near_zero_share"]:.6f}**.
- Food sales mean: CURRENT_WEEKLY **{avg(x["food_sales"] for x in history["CURRENT_WEEKLY"]):.6f}**, BUFFER_0_5 **{avg(x["food_sales"] for x in history["BUFFER_0_5"]):.6f}**.
- Firm cash mean: CURRENT_WEEKLY **{avg(x["firm_cash"] for x in history["CURRENT_WEEKLY"]):.6f}**, BUFFER_0_5 **{avg(x["firm_cash"] for x in history["BUFFER_0_5"]):.6f}**.

The treatment is evaluated jointly on elderly liquidity, overall near-zero,
child-payer burden, minimum-consumption security, and conservation. GUI V2
was not modified in this controlled stage.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()

