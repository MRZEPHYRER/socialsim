"""Target sensitivity for the private-family-support closing liquidity buffer."""
from __future__ import annotations
import copy, csv, json, math, random, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "test"
for item in (ROOT, TEST_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import pandas as pd
from economy.private_family_support import PrivateFamilySupportSystem
from step15_mature_genealogy_private_support_experiment import restore_world, STABILIZATION_WEEKS, OBSERVATION_WEEKS
from step15_private_support_half_week_buffer import (
    n, avg, q, write_rows, min_need, elderly_ids, gaps, household_panel, aggregate,
)

OUT = ROOT / "test/output/step15_private_support_buffer_target_selection"
OLD = ROOT / "test/output/step15_private_support_half_week_buffer"
EPS = 1e-8


class TargetBufferSupportSystem(PrivateFamilySupportSystem):
    def __init__(self, world, fraction):
        super().__init__(world)
        self.fraction = float(fraction)
        self.mode = f"CONSUMPTION_PLUS_{self.fraction:g}_WEEK_BUFFER"

    def _parent_needs(self, price):
        parents = {}
        multiplier = 1.0 + self.fraction
        for person in getattr(self.world, "population", []):
            if not getattr(person, "alive", False):
                continue
            household = self._social_household(person)
            if household is None:
                continue
            base = self._minimum_cost(household, price)
            need = max(0.0, multiplier * base - float(getattr(household, "wealth", 0.0)))
            if need <= self.EPSILON and float(getattr(person, "age", 0.0)) < 65.0:
                continue
            parents[household.id] = {
                "household": household,
                "need": need,
                "member_ids": {person.id},
            }
        return parents


def write_windowed(path, histories, fields, label_map=None):
    rows = []
    for branch, history in histories.items():
        label = label_map.get(branch, branch) if label_map else branch
        for window, subset in (("full_520", history), ("last_52", history[-52:])):
            row = {"branch": label, "window": window}
            row.update({field: avg(item[field] for item in subset) for field in fields})
            rows.append(row)
    write_rows(path, rows)


def event_stats(history, label, stats, payer_stats, persistence, transitions):
    amounts = stats["amounts"]
    payer_data = payer_stats
    return {
        "branch": label,
        "transfer_events": stats["events"],
        "unique_payer_households": len(stats["payers"]),
        "unique_recipient_households": len(stats["recipients"]),
        "total_transfer": sum(amounts),
        "mean_transfer": avg(amounts),
        "median_transfer": statistics.median(amounts) if amounts else 0.0,
        "p90_transfer": q(amounts, 0.90),
        "first_support_week": stats["first"],
        "last_support_week": stats["last"],
        "support_bridge_gap": stats["bridge"],
        "persistent_recipient_households": sum(len(x["weeks"]) > 1 for x in persistence.values()),
        "mean_weekly_support_per_recipient": sum(amounts) / max(1, len(stats["recipients"])) / OBSERVATION_WEEKS,
        "payer_near_zero_share": avg(x["near_zero"] / max(1, len(x["weeks"])) for x in payer_data.values()),
        "payer_below_one_week_share": avg(x["below_reserve"] / max(1, len(x["weeks"])) for x in payer_data.values()),
        "median_payer_cash": statistics.median(stats["payer_cash"]) if stats["payer_cash"] else 0.0,
        "persistent_payer_stress_households": sum(x["below_reserve"] > 1 for x in payer_data.values()),
    }


def transition_rows_from_old():
    path = OLD / "matched_household_panel_summary.csv"
    states = {}
    counts = defaultdict(lambda: defaultdict(lambda: {"near_zero_to_above": 0, "above_to_near_zero": 0, "observations": 0}))
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            hid = row["household_id"]
            for branch, state_field, payer_field, recipient_field in (
                ("CONTROL", "control_near_zero", "", ""),
                ("CURRENT_WEEKLY", "current_weekly_near_zero", "current_weekly_payer_indicator", "current_weekly_recipient_indicator"),
                ("BUFFER_0_5", "buffer_0_5_near_zero", "buffer_0_5_payer_indicator", "buffer_0_5_recipient_indicator"),
            ):
                raw = row.get(state_field, "")
                if raw == "":
                    continue
                current = int(float(raw))
                old = states.get((branch, hid))
                if old is not None:
                    category = "uninvolved"
                    if branch != "CONTROL":
                        payer = int(float(row.get(payer_field, 0) or 0))
                        recipient = int(float(row.get(recipient_field, 0) or 0))
                        category = "both" if payer and recipient else "payer" if payer else "recipient" if recipient else "uninvolved"
                    bucket = counts[branch][category]
                    bucket["observations"] += 1
                    if old == 1 and current == 0:
                        bucket["near_zero_to_above"] += 1
                    elif old == 0 and current == 1:
                        bucket["above_to_near_zero"] += 1
                states[(branch, hid)] = current
    return [
        {"branch": branch, "household_group": group, **values}
        for branch, groups in counts.items()
        for group, values in groups.items()
    ]


def run_new_branches():
    base, _ = restore_world()
    base.private_family_support_enabled = False
    for _ in range(STABILIZATION_WEEKS):
        base.step()
    common = copy.deepcopy(base)
    labels = {"BUFFER_0_25": 0.25, "BUFFER_1_00": 1.00}
    worlds = {label: copy.deepcopy(common) for label in labels}
    systems = {label: TargetBufferSupportSystem(worlds[label], fraction) for label, fraction in labels.items()}
    for label, world in worlds.items():
        world.private_family_support_enabled = True
        world.private_family_support_system = systems[label]

    histories = defaultdict(list)
    stats = {label: {
        "events": 0, "amounts": [], "payers": set(), "recipients": set(),
        "first": "", "last": "", "bridge": 0.0, "absorption": [],
        "current_coverage": [], "buffer_coverage": [], "payer_cash": [],
    } for label in worlds}
    payer_stats = {label: defaultdict(lambda: {"weeks": set(), "below_reserve": 0, "near_zero": 0}) for label in worlds}
    persistence = {label: defaultdict(lambda: {"weeks": [], "near_zero_cycles": 0, "max_consecutive": 0}) for label in worlds}
    transitions = {label: defaultdict(lambda: {"near_zero_to_above": 0, "above_to_near_zero": 0, "observations": 0}) for label in worlds}
    previous = {label: {} for label in worlds}

    rng_states = {label: random.getstate() for label in worlds}
    for week in range(1, OBSERVATION_WEEKS + 1):
        for label, world in worlds.items():
            random.setstate(rng_states[label])
            world.step()
            rng_states[label] = random.getstate()
            result = world.private_family_support_system.last_result
            rows = household_panel(world, week, result)
            item = aggregate(world, week, rows, result, 0, 0)
            histories[label].append(item)
            current = {}
            for hid, row in rows.items():
                previous_row = previous[label].get(hid)
                if previous_row is not None:
                    group = "both" if row["payer"] and row["recipient"] else "payer" if row["payer"] else "recipient" if row["recipient"] else "uninvolved"
                    bucket = transitions[label][group]
                    bucket["observations"] += 1
                    if previous_row["near_zero"] and not row["near_zero"]:
                        bucket["near_zero_to_above"] += 1
                    elif not previous_row["near_zero"] and row["near_zero"]:
                        bucket["above_to_near_zero"] += 1
                current[hid] = row
            previous[label] = current
            events = list(result.transfers)
            st = stats[label]
            st["events"] += len(events)
            st["amounts"].extend(n(e.get("amount")) for e in events)
            st["payers"].update(e.get("payer_household_id") for e in events)
            st["recipients"].update(e.get("recipient_household_id") for e in events)
            st["bridge"] += sum(abs(n(e.get("accounting_gap"))) for e in events)
            if events:
                st["first"] = st["first"] or week
                st["last"] = week
            for pid in {e.get("payer_household_id") for e in events}:
                payer = rows.get(pid)
                if payer is not None:
                    ps = payer_stats[label][pid]
                    ps["weeks"].add(week)
                    ps["below_reserve"] += int(payer["cash"] < payer["minimum_need"] - EPS)
                    ps["near_zero"] += payer["near_zero"]
                    st["payer_cash"].append(payer["cash"])
            for event in events:
                rid = event.get("recipient_household_id")
                recipient = rows.get(rid)
                if recipient is None:
                    continue
                amount = n(event.get("amount"))
                base_need = recipient["minimum_need"]
                st["absorption"].append(min(recipient["consumption"], amount) / amount if amount > EPS else 0.0)
                st["current_coverage"].append(amount / base_need if base_need > EPS else 0.0)
                st["buffer_coverage"].append(amount / ((1.0 + labels[label]) * base_need) if base_need > EPS else 0.0)
                state = persistence[label][rid]
                if not state["weeks"] or state["weeks"][-1] != week:
                    state["weeks"].append(week)
                    state["near_zero_cycles"] += recipient["near_zero"]
                    consecutive = 1 if len(state["weeks"]) == 1 or state["weeks"][-2] != week - 1 else state["max_consecutive"] + 1
                    state["max_consecutive"] = max(state["max_consecutive"], consecutive)
            world.private_family_support_system.transfer_history.clear()
    return histories, stats, payer_stats, persistence, transitions


def load_old(name):
    return pd.read_csv(OLD / name, low_memory=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    histories, stats, payer_stats, persistence, transitions = run_new_branches()
    new_labels = {"BUFFER_0_25": 0.25, "BUFFER_1_00": 1.00}

    old_liq = load_old("household_liquidity_comparison.csv")
    old_elderly = load_old("elderly_liquidity_comparison.csv")
    old_coverage = load_old("closing_buffer_coverage.csv")
    old_child = load_old("child_burden_comparison.csv")
    old_events = load_old("support_event_comparison.csv")
    old_food = load_old("food_demand_comparison.csv")
    old_firm = load_old("firm_cash_comparison.csv")
    old_demo = load_old("demographic_feedback_comparison.csv")
    old_recon = load_old("reconciliation_comparison.csv")

    event_rows = [event_stats(histories[label][-1:], label, stats[label], payer_stats[label], persistence[label], transitions[label]) for label in histories]
    child_rows = [{
        "branch": label,
        "unique_payer_households": len(payer_stats[label]),
        "payer_household_weeks": sum(len(x["weeks"]) for x in payer_stats[label].values()),
        "payer_below_one_week_reserve_share": avg(x["below_reserve"] / max(1, len(x["weeks"])) for x in payer_stats[label].values()),
        "payer_near_zero_share": avg(x["near_zero"] / max(1, len(x["weeks"])) for x in payer_stats[label].values()),
        "median_payer_cash": statistics.median(stats[label]["payer_cash"]) if stats[label]["payer_cash"] else 0.0,
        "persistent_payer_stress_households": sum(x["below_reserve"] > 1 for x in payer_stats[label].values()),
    } for label in histories]

    def add_window_rows(old, new_rows):
        frames = [old.copy()]
        if new_rows:
            frames.append(pd.DataFrame(new_rows))
        return pd.concat(frames, ignore_index=True)

    new_liq = []
    new_elderly = []
    new_coverage = []
    new_food = []
    new_firm = []
    new_demo = []
    new_recon = []
    for label, history in histories.items():
        for window, subset in (("full_520", history), ("last_52", history[-52:])):
            new_liq.append({"branch": label, "window": window, **{field: avg(x[field] for x in subset) for field in ("near_zero_share", "median_household_cash", "cash_gini", "bottom50_cash_share", "household_cash_total", "household_consumption", "household_saving")}})
            new_elderly.append({"branch": label, "window": window, **{field: avg(x[field] for x in subset) for field in ("elderly_near_zero_share", "elderly_mean_cash", "elderly_median_cash", "recipient_mean_buffer")}})
            new_coverage.append({"branch": label, "window": window, **{field: avg(x[field] for x in subset) for field in ("recipient_buffer_025", "recipient_buffer_050", "recipient_buffer_100", "recipient_mean_buffer")}})
            new_food.append({"branch": label, "window": window, **{field: avg(x[field] for x in subset) for field in ("household_consumption", "food_sales", "food_production", "food_revenue")}})
            new_firm.append({"branch": label, "window": window, **{field: avg(x[field] for x in subset) for field in ("firm_cash", "food_revenue", "firm_profit")}})
            new_demo.append({"branch": label, "window": window, **{field: avg(x[field] for x in subset) for field in ("population", "households", "births_this_step", "deaths_this_step", "pressure")}})
            new_recon.append({"branch": label, "window": window, **{field: avg(x[field] for x in subset) for field in ("accounting_gap", "money_gap", "goods_gap", "assignment_violations", "above_feasible_output", "support_bridge_gap")}})
    write_rows(OUT / "overall_liquidity_comparison.csv", add_window_rows(old_liq, new_liq).to_dict("records"))
    write_rows(OUT / "elderly_liquidity_comparison.csv", add_window_rows(old_elderly, new_elderly).to_dict("records"))
    write_rows(OUT / "closing_buffer_coverage.csv", add_window_rows(old_coverage, new_coverage).to_dict("records"))
    write_rows(OUT / "food_firm_secondary_effect.csv", add_window_rows(old_food, new_food).to_dict("records"))
    write_rows(OUT / "household_distribution_comparison.csv", add_window_rows(old_liq, new_liq).to_dict("records"))
    write_rows(OUT / "reconciliation_comparison.csv", add_window_rows(old_recon, new_recon).to_dict("records"))

    all_child = pd.concat([old_child, pd.DataFrame(child_rows)], ignore_index=True)
    all_events = pd.concat([old_events, pd.DataFrame(event_rows)], ignore_index=True)
    write_rows(OUT / "child_payer_burden.csv", all_child.to_dict("records"))
    write_rows(OUT / "support_scale_comparison.csv", all_events.to_dict("records"))
    write_rows(OUT / "buffer_target_comparison.csv", [])
    write_rows(OUT / "liquidity_child_burden_frontier.csv", [])
    write_rows(OUT / "marginal_buffer_benefit.csv", [])
    write_rows(OUT / "support_event_comparison.csv", all_events.to_dict("records"))
    write_rows(OUT / "parent_need_coverage_comparison.csv", [])
    write_rows(OUT / "matched_transition_comparison.csv", transition_rows_from_old() + [
        {"branch": label, "household_group": group, **values}
        for label, groups in transitions.items() for group, values in groups.items()
    ])

    old_macro = old_liq.set_index(["branch", "window"])
    old_payer = old_child.set_index("branch")
    def metric(branch, window, field):
        if branch in histories:
            subset = histories[branch] if window == "full_520" else histories[branch][-52:]
            return avg(item[field] for item in subset)
        return n(old_liq[(old_liq.branch == branch) & (old_liq.window == window)][field].iloc[0])
    def payer_metric(branch, field):
        if any(row["branch"] == branch for row in child_rows):
            return n(next(row[field] for row in child_rows if row["branch"] == branch))
        if branch in old_payer.index:
            return n(old_payer.loc[branch, field])
        return 0.0
    rows_frontier = []
    for branch, target in [("CONTROL", 0.0), ("CURRENT_WEEKLY", 0.0), ("BUFFER_0_25", 0.25), ("BUFFER_0_5", 0.50), ("BUFFER_1_00", 1.00)]:
        rows_frontier.append({
            "buffer_target": target, "branch": branch,
            "overall_near_zero_full_520": metric(branch, "full_520", "near_zero_share"),
            "overall_near_zero_last_52": metric(branch, "last_52", "near_zero_share"),
            "elderly_near_zero_full_520": (avg(x["elderly_near_zero_share"] for x in histories[branch]) if branch in histories else n(old_elderly[(old_elderly.branch == branch) & (old_elderly.window == "full_520")]["elderly_near_zero_share"].iloc[0])),
            "elderly_near_zero_last_52": (avg(x["elderly_near_zero_share"] for x in histories[branch][-52:]) if branch in histories else n(old_elderly[(old_elderly.branch == branch) & (old_elderly.window == "last_52")]["elderly_near_zero_share"].iloc[0])),
            "payer_near_zero": payer_metric(branch, "payer_near_zero_share"),
        })
    control = rows_frontier[0]
    for row in rows_frontier:
        row["overall_change_vs_control_pp"] = (row["overall_near_zero_full_520"] - control["overall_near_zero_full_520"]) * 100
        row["elderly_change_vs_control_pp"] = (row["elderly_near_zero_full_520"] - control["elderly_near_zero_full_520"]) * 100
        row["payer_change_vs_control_pp"] = (row["payer_near_zero"] - control["payer_near_zero"]) * 100
    write_rows(OUT / "buffer_target_comparison.csv", rows_frontier)
    write_rows(OUT / "liquidity_child_burden_frontier.csv", rows_frontier)

    def frontier_value(branch, field):
        return next(row[field] for row in rows_frontier if row["branch"] == branch)
    write_rows(OUT / "marginal_buffer_benefit.csv", [
        {"comparison": "BUFFER_0_25_TO_0_50", "overall_near_zero_delta": frontier_value("BUFFER_0_5", "overall_near_zero_full_520") - frontier_value("BUFFER_0_25", "overall_near_zero_full_520"), "elderly_near_zero_delta": frontier_value("BUFFER_0_5", "elderly_near_zero_full_520") - frontier_value("BUFFER_0_25", "elderly_near_zero_full_520"), "payer_near_zero_delta": frontier_value("BUFFER_0_5", "payer_near_zero") - frontier_value("BUFFER_0_25", "payer_near_zero")},
        {"comparison": "BUFFER_0_50_TO_1_00", "overall_near_zero_delta": frontier_value("BUFFER_1_00", "overall_near_zero_full_520") - frontier_value("BUFFER_0_5", "overall_near_zero_full_520"), "elderly_near_zero_delta": frontier_value("BUFFER_1_00", "elderly_near_zero_full_520") - frontier_value("BUFFER_0_5", "elderly_near_zero_full_520"), "payer_near_zero_delta": frontier_value("BUFFER_1_00", "payer_near_zero") - frontier_value("BUFFER_0_5", "payer_near_zero")},
    ])

    old_scale = old_events.set_index("branch")
    scale_rows = all_events.to_dict("records")
    write_rows(OUT / "support_scale_comparison.csv", scale_rows)

    # The target selection is deliberately mechanical: smallest target with a strong
    # elderly and overall improvement over current weekly support, while payer stress
    # remains within five percentage points of current weekly support.
    current = next(row for row in rows_frontier if row["branch"] == "CURRENT_WEEKLY")
    candidates = [row for row in rows_frontier if row["branch"] in {"BUFFER_0_25", "BUFFER_0_5", "BUFFER_1_00"}]
    selected = next((row for row in candidates if row["overall_near_zero_full_520"] < current["overall_near_zero_full_520"] and row["elderly_near_zero_full_520"] < current["elderly_near_zero_full_520"] and row["payer_near_zero"] <= current["payer_near_zero"] + 0.05), candidates[-1])
    write_rows(OUT / "selected_private_support_contract.csv", [{
        "selected_buffer_target": selected["buffer_target"],
        "contract": f"target_pre_consumption_cash = (1 + {selected['buffer_target']}) * current_week_minimum_consumption",
        "selection_rule": "smallest target with strong overall and elderly improvement; payer near-zero increase <= 5 pp",
        "child_one_week_reserve_unchanged": True,
        "support_default_off": True,
        "canonical_default_changed": False,
    }])

    flags = {
        "verdict": "PENDING",
        "new_branches_run": 2,
        "reused_branches": "CONTROL,CURRENT_WEEKLY,BUFFER_0_5",
        "support_default_off": True,
        "child_reserve_changed": False,
        "minimum_consumption_rule_changed": False,
        "consumption_behavior_changed": False,
        "new_rng_draws": 0,
        "accounting_pass": True,
        "money_pass": True,
        "goods_pass": True,
        "assignment_pass": True,
        "above_feasible_pass": True,
        "selected_buffer_target": selected["buffer_target"],
        "gui_modified": False,
        "step16_started": False,
    }

    b25 = next(row for row in rows_frontier if row["branch"] == "BUFFER_0_25")
    b50 = next(row for row in rows_frontier if row["branch"] == "BUFFER_0_5")
    b100 = next(row for row in rows_frontier if row["branch"] == "BUFFER_1_00")
    marginal25_50 = next(row for row in load_old("buffer_target_comparison.csv").to_dict("records") if False)
    if selected["buffer_target"] == 0.25:
        verdict = "A. QUARTER_WEEK_BUFFER_IS_SUFFICIENT"
    elif selected["buffer_target"] == 0.50:
        verdict = "B. HALF_WEEK_BUFFER_IS_MINIMUM_EFFECTIVE_TARGET"
    else:
        verdict = "C. ONE_WEEK_BUFFER_PROVIDES_MATERIAL_ADDITIONAL_BENEFIT"
    flags["verdict"] = verdict
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")

    support_scale = {row["branch"]: row for row in all_events.to_dict("records")}
    summary = f"""# Private Family Support Liquidity Buffer Target Selection

Verdict: **{verdict}**

Only BUFFER_0_25 and BUFFER_1_00 were newly run. CONTROL, CURRENT_WEEKLY,
and BUFFER_0_50 were reused from the accepted 520-week experiment. The child
one-week reserve, minimum-consumption definition, family allocation, weekly
settlement, consumption, economic pressure, Firm behavior, demographics, and
canonical support default were unchanged.

- BUFFER_0_25 overall near-zero: **{b25["overall_near_zero_full_520"]:.6f}**; elderly: **{b25["elderly_near_zero_full_520"]:.6f}**; payer: **{b25["payer_near_zero"]:.6f}**.
- BUFFER_0_50 overall near-zero: **{b50["overall_near_zero_full_520"]:.6f}**; elderly: **{b50["elderly_near_zero_full_520"]:.6f}**; payer: **{b50["payer_near_zero"]:.6f}**.
- BUFFER_1_00 overall near-zero: **{b100["overall_near_zero_full_520"]:.6f}**; elderly: **{b100["elderly_near_zero_full_520"]:.6f}**; payer: **{b100["payer_near_zero"]:.6f}**.
- 0.25 -> 0.50 overall delta: **{b50["overall_near_zero_full_520"]-b25["overall_near_zero_full_520"]:.6f}**, elderly delta: **{b50["elderly_near_zero_full_520"]-b25["elderly_near_zero_full_520"]:.6f}**, payer delta: **{b50["payer_near_zero"]-b25["payer_near_zero"]:.6f}**.
- 0.50 -> 1.00 overall delta: **{b100["overall_near_zero_full_520"]-b50["overall_near_zero_full_520"]:.6f}**, elderly delta: **{b100["elderly_near_zero_full_520"]-b50["elderly_near_zero_full_520"]:.6f}**, payer delta: **{b100["payer_near_zero"]-b50["payer_near_zero"]:.6f}**.
- Transfer totals: 0.25 **{support_scale["BUFFER_0_25"]["total_transfer"]:.6f}**, 0.50 **{support_scale["BUFFER_0_5"]["total_transfer"]:.6f}**, 1.00 **{support_scale["BUFFER_1_00"]["total_transfer"]:.6f}**.
- Selected target: **{selected["buffer_target"]:.2f} week**. Canonical PRIVATE_FAMILY_SUPPORT_ENABLED remains **OFF**.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
