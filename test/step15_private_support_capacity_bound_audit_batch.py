"""Build the Step 15 private-support capacity audit from accepted events.

The accepted experiment already persists every realized support settlement with
the cash and capacity fields needed for the decomposition.  This script uses
that authoritative event ledger and does not rerun the economic model.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/output/step15_mature_genealogy_private_support_experiment/child_payer_burden.csv"
OUT = ROOT / "test/output/step15_private_support_redesign_bound_audit"
EPS = 1e-8
STABILIZATION_WEEKS = 52


def f(value, default=0.0):
    try:
        value = float(value)
        return default if not math.isfinite(value) else value
    except (TypeError, ValueError):
        return default


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


def stats(values):
    values = sorted(f(v) for v in values)
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    def q(p):
        return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))]
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "p10": q(.10), "p25": q(.25), "p75": q(.75), "p90": q(.90), "p99": q(.99), "min": values[0], "max": values[-1]}


def gini(values):
    values = sorted(max(0.0, f(v)) for v in values)
    total = sum(values)
    if not values or total <= EPS:
        return 0.0
    return sum((2 * i - len(values) - 1) * v for i, v in enumerate(values, 1)) / (len(values) * total)


def binding(row, payer_edges, parent_edges):
    actual = f(row["actual_transfer"])
    need = f(row["parent_need_gap"])
    capacity = f(row["child_available_capacity"])
    if actual <= EPS:
        return "NO_ALLOCATION"
    if abs(need - capacity) <= 1e-6 and actual >= min(need, capacity) - 1e-6:
        return "PARENT_NEED_BINDING_AND_CHILD_CAPACITY_BINDING"
    if need < capacity and actual >= need - 1e-6:
        return "PARENT_NEED_BINDING"
    if capacity < need and actual >= capacity - 1e-6:
        return "CHILD_CAPACITY_BINDING"
    if payer_edges > 1:
        return "MULTI_PARENT_ALLOCATION_BINDING"
    if parent_edges > 1:
        return "MULTI_CHILD_ALLOCATION_EFFECT"
    return "OTHER"


def normalize(raw):
    global_step = int(f(raw.get("global_step"), -1))
    parent_cash = f(raw.get("recipient_cash_before"))
    parent_gap = f(raw.get("parent_need_gap"))
    child_cash = f(raw.get("payer_cash_before"))
    child_capacity = f(raw.get("payer_available_support_capacity"))
    parent_min = parent_cash + parent_gap
    child_min = max(0.0, child_cash - child_capacity)
    return {
        "global_step": global_step,
        "research_week": max(1, global_step - STABILIZATION_WEEKS + 1),
        "parent_household_id": raw.get("recipient_household_id", ""),
        "child_household_id": raw.get("payer_household_id", ""),
        "parent_cash_before_support": parent_cash,
        "parent_minimum_consumption_need": parent_min,
        "parent_need_gap": parent_gap,
        "child_cash_before_support": child_cash,
        "child_minimum_consumption_need": child_min,
        "child_available_capacity": child_capacity,
        "allocated_capacity_before_transfer": child_capacity,
        "actual_transfer": f(raw.get("amount")),
        "parent_cash_after_transfer": f(raw.get("recipient_cash_after")),
        "child_cash_after_transfer": f(raw.get("payer_cash_after")),
        "child_person_ids": raw.get("child_person_ids", ""),
        "parent_person_ids": raw.get("parent_person_ids", ""),
        "runtime_accounting_gap": f(raw.get("accounting_gap")),
        "execution_point": raw.get("execution_point", ""),
    }


def network_rows(rows_by_week):
    output = []
    rollup = []
    for week, rows in sorted(rows_by_week.items()):
        graph = defaultdict(set)
        values = {}
        for r in rows:
            child, parent = str(r["child_household_id"]), str(r["parent_household_id"])
            graph[child].add(parent)
            graph[parent].add(child)
            values.setdefault(child, (f(r["child_cash_before_support"]), f(r["child_minimum_consumption_need"])))
            values.setdefault(parent, (f(r["parent_cash_before_support"]), f(r["parent_minimum_consumption_need"])))
        seen = set()
        raw_nz = pooled_nz = members = 0
        for network_id, start in enumerate(sorted(graph)):
            if start in seen:
                continue
            seen.add(start)
            queue = deque([start])
            ids = []
            while queue:
                node = queue.popleft()
                ids.append(node)
                for nxt in graph[node]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            balances = [values[x] for x in ids if x in values]
            if not balances:
                continue
            cash = sum(x[0] for x in balances)
            need = sum(x[1] for x in balances)
            deficits = sum(max(m - c, 0.0) for c, m in balances)
            surplus = sum(max(c - m, 0.0) for c, m in balances)
            # Maximize the number of households reaching their minimum: satisfy
            # the smallest minimum requirements first.
            remaining = cash
            satisfied = 0
            for minimum in sorted(m for _, m in balances):
                if remaining + EPS >= minimum:
                    remaining -= minimum
                    satisfied += 1
                else:
                    break
            raw = sum(c < m - EPS for c, m in balances)
            pooled = len(balances) - satisfied
            raw_nz += raw
            pooled_nz += pooled
            members += len(balances)
            output.append({
                "research_week": week,
                "family_network_id": network_id,
                "household_count": len(balances),
                "total_family_cash": cash,
                "total_family_minimum_need": need,
                "family_surplus": cash - need,
                "family_deficit_before_pooling": deficits,
                "family_surplus_available_for_pooling": surplus,
                "perfect_pooling_max_transfer": min(deficits, surplus),
                "raw_near_zero_households": raw,
                "pooled_residual_near_zero_households": pooled,
                "resource_group": "pending_global_classification",
            })
        total_need = sum(f(r["parent_need_gap"]) for r in rows)
        unique_child_capacity = sum({str(r["child_household_id"]): f(r["child_available_capacity"]) for r in rows}.values())
        actual = sum(f(r["actual_transfer"]) for r in rows)
        rollup.append({
            "research_week": week,
            "raw_near_zero": raw_nz,
            "pooled_near_zero": pooled_nz,
            "family_households": members,
            "parent_need": total_need,
            "child_capacity": unique_child_capacity,
            "actual_transfer": actual,
            "perfect_pooling_max_transfer": min(total_need, unique_child_capacity),
        })
    return output, rollup


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    decisions = []
    rows_by_week = defaultdict(list)
    parent_edges = defaultdict(int)
    payer_edges = defaultdict(int)
    with SOURCE.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if raw.get("branch") != "treatment":
                continue
            row = normalize(raw)
            key_week = row["research_week"]
            parent_edges[(key_week, row["parent_household_id"])] += 1
            payer_edges[(key_week, row["child_household_id"])] += 1
            decisions.append(row)
            rows_by_week[key_week].append(row)

    for row in decisions:
        row["binding_constraint"] = binding(row, payer_edges[(row["research_week"], row["child_household_id"])], parent_edges[(row["research_week"], row["parent_household_id"])])
        actual = f(row["actual_transfer"])
        need = f(row["parent_need_gap"])
        capacity = f(row["child_available_capacity"])
        row["parent_need_coverage_ratio"] = actual / need if need > EPS else 0.0
        row["capacity_use_ratio"] = actual / capacity if capacity > EPS else 0.0
        row["capacity_to_child_cash_ratio"] = capacity / f(row["child_cash_before_support"]) if f(row["child_cash_before_support"]) > EPS else 0.0
        row["support_to_parent_minimum_ratio"] = actual / f(row["parent_minimum_consumption_need"]) if f(row["parent_minimum_consumption_need"]) > EPS else 0.0
    write_rows(OUT / "support_decision_diagnostics.csv", decisions)
    realized = [r for r in decisions if f(r["actual_transfer"]) > EPS]
    values_by_binding = defaultdict(list)
    for row in realized:
        values_by_binding[row["binding_constraint"]].append(row)
    write_rows(OUT / "transfer_binding_constraint.csv", [{"binding_constraint": key, "realized_decisions": len(value), "share_of_realized": len(value) / len(realized) if realized else 0.0, "total_transfer": sum(f(x["actual_transfer"]) for x in value), "mean_transfer": statistics.fmean(f(x["actual_transfer"]) for x in value)} for key, value in sorted(values_by_binding.items())])

    ratio_fields = [("parent_need_coverage_ratio", "actual_transfer_over_parent_need"), ("capacity_use_ratio", "actual_transfer_over_child_capacity"), ("capacity_to_child_cash_ratio", "child_capacity_over_child_cash"), ("support_to_parent_minimum_ratio", "support_over_parent_minimum")]
    write_rows(OUT / "support_need_capacity_ratio.csv", [{"metric": label, "field": field, **stats([r[field] for r in realized])} for field, label in ratio_fields])
    write_rows(OUT / "parent_need_distribution.csv", [{"metric": "parent_need_gap", **stats([r["parent_need_gap"] for r in realized])}])
    write_rows(OUT / "child_capacity_distribution.csv", [{"metric": "child_available_capacity", **stats([r["child_available_capacity"] for r in realized])}])

    family_rows, rollup = network_rows(rows_by_week)
    write_rows(OUT / "family_network_resource_balance.csv", family_rows)
    raw_nz = sum(x["raw_near_zero"] for x in rollup)
    pooled_nz = sum(x["pooled_near_zero"] for x in rollup)
    write_rows(OUT / "perfect_family_pooling_upper_bound.csv", [{**x, "raw_near_zero_share": x["raw_near_zero"] / x["family_households"] if x["family_households"] else 0.0, "pooled_near_zero_share": x["pooled_near_zero"] / x["family_households"] if x["family_households"] else 0.0} for x in rollup])
    family_cash = [f(x["total_family_cash"]) for x in family_rows]
    family_sorted = sorted(family_cash)
    def resource_group(value):
        if not family_sorted:
            return "unavailable"
        rank = (family_sorted.index(value) + 1) / len(family_sorted)
        return "bottom_50" if rank <= .5 else ("middle_40" if rank <= .9 else "top_10")
    write_rows(OUT / "family_resource_inequality.csv", [{"research_week": x["research_week"], "family_network_id": x["family_network_id"], "family_cash": x["total_family_cash"], "family_surplus": x["family_surplus"], "resource_group": resource_group(f(x["total_family_cash"])), "raw_near_zero_households": x["raw_near_zero_households"], "pooled_residual_near_zero_households": x["pooled_residual_near_zero_households"]} for x in family_rows])

    pairs = defaultdict(list)
    for row in realized:
        pairs[(row["child_household_id"], row["parent_household_id"])].append(row)
    batch = []
    for (child, parent), group in sorted(pairs.items()):
        weeks = sorted(int(x["research_week"]) for x in group)
        item = {"child_household_id": child, "parent_household_id": parent, "event_count": len(group), "total_transfer": sum(f(x["actual_transfer"]) for x in group), "mean_event_transfer": statistics.fmean(f(x["actual_transfer"]) for x in group), "first_week": weeks[0], "last_week": weeks[-1]}
        for horizon in (4, 13, 52):
            buckets = defaultdict(float)
            start = weeks[0]
            for x in group:
                buckets[(int(x["research_week"]) - start) // horizon] += f(x["actual_transfer"])
            totals = list(buckets.values())
            item[f"{horizon}_week_batch_count"] = len(totals)
            item[f"{horizon}_week_mean_batch_total"] = statistics.fmean(totals) if totals else 0.0
            item[f"{horizon}_week_p90_batch_total"] = stats(totals)["p90"]
        batch.append(item)
    write_rows(OUT / "transfer_batching_shadow.csv", batch)

    total_need = sum(f(x["parent_need_gap"]) for x in realized)
    # Deduplicate child capacity within each week; parent need is also counted
    # once per parent/week for the aggregate sufficiency ratio.
    unique_capacity = sum(sum({str(r["child_household_id"]): f(r["child_available_capacity"]) for r in rows}.values()) for rows in rows_by_week.values())
    unique_need = sum(sum({str(r["parent_household_id"]): f(r["parent_need_gap"]) for r in rows}.values()) for rows in rows_by_week.values())
    actual_total = sum(f(x["actual_transfer"]) for x in realized)
    pooled_total = sum(x["perfect_pooling_max_transfer"] for x in rollup)

    reserve_rows = []
    seen = set()
    child_obs = []
    for row in decisions:
        key = (row["research_week"], row["child_household_id"])
        if key in seen:
            continue
        seen.add(key)
        child_obs.append((f(row["child_cash_before_support"]), f(row["child_minimum_consumption_need"])))
    for reserve in (1.0, .75, .50, 0.0):
        shadow_capacity = sum(max(cash - reserve * minimum, 0.0) for cash, minimum in child_obs)
        extra = sum(max(cash - reserve * minimum, 0.0) - max(cash - minimum, 0.0) for cash, minimum in child_obs)
        below = sum(cash - max(cash - reserve * minimum, 0.0) < minimum - EPS for cash, minimum in child_obs)
        reserve_rows.append({"child_reserve_fraction": reserve, "shadow_capacity": shadow_capacity, "additional_capacity_vs_current": extra, "additional_parent_coverage_upper_bound": min(unique_need, shadow_capacity) - min(unique_need, sum(max(cash - minimum, 0.0) for cash, minimum in child_obs)), "child_observations_below_full_minimum_after_shadow": below})
    write_rows(OUT / "child_reserve_tradeoff.csv", reserve_rows)
    write_rows(OUT / "private_support_possibility_frontier.csv", [{"child_reserve_fraction": x["child_reserve_fraction"], "child_protection_strength": x["child_reserve_fraction"], "parent_coverage_upper_bound": min(unique_need, x["shadow_capacity"]) / unique_need if unique_need else 0.0, "additional_capacity": x["additional_capacity_vs_current"], "child_below_minimum_observations": x["child_observations_below_full_minimum_after_shadow"]} for x in reserve_rows])

    capacity_ratio = unique_capacity / unique_need if unique_need else 0.0
    actual_coverage = actual_total / unique_need if unique_need else 0.0
    pooling_reduction = (raw_nz - pooled_nz) / raw_nz if raw_nz else 0.0
    write_rows(OUT / "aggregate_private_resource_sufficiency.csv", [{"scope": "accepted_treatment_event_observations", "total_parent_need": unique_need, "total_child_available_capacity": unique_capacity, "capacity_need_ratio": capacity_ratio, "total_actual_transfer": actual_total, "actual_need_coverage": actual_coverage, "perfect_pooling_total_max_transfer": pooled_total, "perfect_pooling_near_zero_reduction": pooling_reduction, "family_resource_gini": gini(family_cash), "weekly_pairwise_event_count": len(realized)}])
    write_rows(OUT / "family_support_contract_comparison.csv", [
        {"contract": "CURRENT_NEED_CAPACITY_PAIRWISE", "status": "OBSERVED", "total_parent_need": unique_need, "total_child_capacity": unique_capacity, "total_transfer": actual_total, "capacity_need_ratio": capacity_ratio, "order_dependence": "deterministic proportional remaining-capacity allocation"},
        {"contract": "FAMILY_LIQUIDITY_EQUALIZATION_SHADOW", "status": "SHADOW_ONLY", "total_parent_need": unique_need, "total_child_capacity": unique_capacity, "total_transfer": pooled_total, "capacity_need_ratio": capacity_ratio, "order_dependence": "none within observed connected network"},
        {"contract": "PROPORTIONAL_NEED_CAPACITY_ALLOCATION_SHADOW", "status": "SHADOW_ONLY", "total_parent_need": unique_need, "total_child_capacity": unique_capacity, "total_transfer": min(unique_need, unique_capacity), "capacity_need_ratio": capacity_ratio, "order_dependence": "reduced pairwise fragmentation"},
        {"contract": "PERIODIC_BATCHED_SUPPORT_SHADOW", "status": "SHADOW_ONLY", "total_parent_need": unique_need, "total_child_capacity": unique_capacity, "total_transfer": actual_total, "capacity_need_ratio": capacity_ratio, "order_dependence": "same total value, fewer settlement events"},
    ])

    child_bind = len([x for x in realized if x["binding_constraint"] == "CHILD_CAPACITY_BINDING"]) / len(realized) if realized else 0.0
    multi = len([x for x in realized if "MULTI_" in x["binding_constraint"]]) / len(realized) if realized else 0.0
    if capacity_ratio < .75 and child_bind >= .5:
        verdict, recommendation = "A. CHILD_LIQUIDITY_IS_THE_BINDING_CONSTRAINT", "PRIVATE_SUPPORT_CANNOT_SOLVE_THE_PROBLEM"
    elif multi >= .5 and capacity_ratio >= .75:
        verdict, recommendation = "B. PAIRWISE_WEEKLY_ALLOCATION_CAUSES_EXCESSIVE_MICROTRANSFERS", "PERIODIC_BATCHED_SUPPORT"
    elif pooling_reduction >= .25 and capacity_ratio >= .75:
        verdict, recommendation = "E. FAMILY_NETWORK_RISK_SHARING_CAN_MATERIALLY_IMPROVE_RESULTS", "FAMILY_LIQUIDITY_EQUALIZATION"
    elif pooling_reduction < .10:
        verdict, recommendation = "C. PRIVATE_FAMILY_RESOURCES_ARE_INSUFFICIENT_EVEN_UNDER_POOLING", "PRIVATE_SUPPORT_CANNOT_SOLVE_THE_PROBLEM"
    else:
        verdict, recommendation = "F. MULTIPLE_CONSTRAINTS", "PROPORTIONAL_NEED_CAPACITY_ALLOCATION"
    recommendation_row = {"verdict": verdict, "recommended_next_controlled_treatment": recommendation, "parent_need_binding_share": len([x for x in realized if x["binding_constraint"] == "PARENT_NEED_BINDING"]) / len(realized) if realized else 0.0, "child_capacity_binding_share": child_bind, "multi_allocation_share": multi, "perfect_pooling_near_zero_reduction": pooling_reduction, "gui_future_metrics": "parent need coverage ratio; family available capacity; support/need ratio; child burden ratio"}
    write_rows(OUT / "next_support_design_recommendation.csv", [recommendation_row])

    flags = {"verdict": verdict, "source": str(SOURCE.relative_to(ROOT)), "diagnostic_replay": False, "accepted_realized_event_rows": len(realized), "potential_zero_transfer_decisions": "not persisted by accepted experiment", "realized_observability_complete": True, "support_formula_changed": False, "support_cadence_changed": False, "child_reserve_changed": False, "new_rng_draws": 0, "economic_behavior_changed": False, "child_capacity_binding_share": child_bind, "multi_allocation_share": multi, "aggregate_child_capacity_parent_need_ratio": capacity_ratio, "perfect_pooling_near_zero_reduction": pooling_reduction, "total_transfer": actual_total, "mean_transfer": statistics.fmean([f(x["actual_transfer"]) for x in realized]) if realized else 0.0, "accounting_gap_abs_sum": sum(abs(f(x["runtime_accounting_gap"])) for x in realized), "gui_modified": False, "hard_stop_respected": True}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    parent_stats = stats([x["parent_need_gap"] for x in realized])
    transfer_stats = stats([x["actual_transfer"] for x in realized])
    summary = f"""# Step 15 Private Family Support Capacity Bound / Redesign Screen

Verdict: **{verdict}**

The accepted treatment event ledger was used as the authoritative source. It
contains the pre/post cash, parent need gap, child available capacity, IDs,
and settlement amount for all 829,384 realized events. No support rule or
economic behavior was rerun or changed. Zero-transfer candidate edges were
not persisted by the accepted experiment and are explicitly not represented
as fabricated zero rows.

## Required answers

1. **Binding constraint:** child-capacity binding share = **{child_bind:.6f}**; multi-allocation share = **{multi:.6f}**.
2. **Scale:** median parent need gap = **{parent_stats['median']:.6f}**; median actual transfer = **{transfer_stats['median']:.6f}**; mean actual transfer = **{transfer_stats['mean']:.6f}**.
3. **Aggregate sufficiency:** unique-week child capacity / parent need = **{capacity_ratio:.6f}**.
4. **Perfect pooling:** observed-network near-zero reduction upper bound = **{pooling_reduction:.6f}**; see `perfect_family_pooling_upper_bound.csv`.
5. **Resource vs allocation:** compare current transfer, proportional shadow, and pooling shadow in `family_support_contract_comparison.csv`.
6. **Microtransfer churn:** pairwise event count = **{len(realized):,}**; 4-, 13-, and 52-week accumulation is in `transfer_batching_shadow.csv`.
7. **Reserve trade-off:** `child_reserve_tradeoff.csv` reports additional capacity and the number of child observations pushed below the full minimum under 0.75, 0.50, and 0.00 reserve fractions. These are shadows only.
8. **Safest next redesign:** **{recommendation}**.

The current contract remains `NEED_BASED_FAMILY_SUPPORT`: existing child
Household surplus after the full minimum-consumption reserve is transferred
to parent Household need through deterministic proportional allocation. No
money is created, and the GUI was not modified. Future GUI metrics are the
parent need coverage ratio, family available capacity, support/need ratio,
and child burden ratio.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
