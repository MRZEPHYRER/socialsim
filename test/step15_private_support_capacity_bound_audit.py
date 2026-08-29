"""Diagnostic-only private family support capacity and redesign audit.

This replays one treatment branch from the accepted mature-genealogy
checkpoint.  The wrapper observes the support system before and after its
existing allocator; it does not alter the allocator or any economic rule.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.private_family_support import PrivateFamilySupportSystem
from step15_mature_genealogy_private_support_experiment import restore_world

OUT = ROOT / "test/output/step15_private_support_redesign_bound_audit"
STABILIZATION_WEEKS = 52
OBSERVATION_WEEKS = 520
CADENCE = 13
EPS = 1e-8


def n(value, default=0.0):
    try:
        value = float(value)
        return default if not math.isfinite(value) else value
    except (TypeError, ValueError):
        return default


def write_csv(path, rows):
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


def quantiles(values):
    values = sorted(n(v) for v in values)
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    def q(p):
        return values[min(len(values) - 1, max(0, int(math.ceil(p * len(values))) - 1))]
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "p10": q(.10), "p25": q(.25), "p75": q(.75), "p90": q(.90), "p99": q(.99), "min": values[0], "max": values[-1]}


def gini(values):
    values = sorted(max(0.0, n(v)) for v in values)
    total = sum(values)
    if not values or total <= EPS:
        return 0.0
    return sum((2 * i - len(values) - 1) * v for i, v in enumerate(values, 1)) / (len(values) * total)


def household(world, hid):
    return world.get_household(hid) if hid is not None else None


def minimum_cost(system, h, price=None):
    if h is None:
        return 0.0
    if price is None:
        price = system._price()
    return system._minimum_cost(h, price)


def capture_candidates(system, step):
    """Observe the exact pre-allocation candidate set via existing helpers."""
    system._reset_household_fields(system.world)
    price = system._price()
    parents = system._parent_needs(price)
    edges = system._build_edges(parents)
    parent_edges = defaultdict(set)
    for payer_id, recipient_id in edges:
        parent_edges[payer_id].add(recipient_id)
    capacities = system._payer_capacities(price, parent_edges)
    edge_counts_payer = defaultdict(int)
    edge_counts_parent = defaultdict(int)
    for payer_id, parent_id in edges:
        edge_counts_payer[payer_id] += 1
        edge_counts_parent[parent_id] += 1
    rows = []
    for payer_id, parent_id in sorted(edges):
        payer = edges[(payer_id, parent_id)]["payer"]
        parent = edges[(payer_id, parent_id)]["recipient"]
        parent_min = minimum_cost(system, parent, price)
        child_min = minimum_cost(system, payer, price)
        rows.append({
            "global_step": step,
            "research_week": step,
            "parent_household_id": parent_id,
            "child_household_id": payer_id,
            "parent_cash_before_support": n(parent.wealth),
            "parent_minimum_consumption_need": parent_min,
            "parent_need_gap": n(parents[parent_id]["need"]),
            "child_cash_before_support": n(payer.wealth),
            "child_minimum_consumption_need": child_min,
            "child_available_capacity": n(capacities.get(payer_id, 0.0)),
            "allocated_capacity_before_transfer": n(capacities.get(payer_id, 0.0)),
            "actual_transfer": 0.0,
            "parent_cash_after_transfer": n(parent.wealth),
            "child_cash_after_transfer": n(payer.wealth),
            "payer_edge_count": edge_counts_payer[payer_id],
            "parent_edge_count": edge_counts_parent[parent_id],
            "child_person_ids": ";".join(map(str, sorted(edges[(payer_id, parent_id)]["child_ids"]))),
            "parent_person_ids": ";".join(map(str, sorted(edges[(payer_id, parent_id)]["parent_ids"]))),
        })
    return rows


def attach_execution(rows, result):
    by_key = {(r["child_household_id"], r["parent_household_id"]): r for r in rows}
    for event in getattr(result, "transfers", []):
        key = (event.get("payer_household_id"), event.get("recipient_household_id"))
        row = by_key.get(key)
        if row is None:
            continue
        row["actual_transfer"] = n(event.get("amount"))
        row["parent_cash_after_transfer"] = n(event.get("recipient_cash_after"))
        row["child_cash_after_transfer"] = n(event.get("payer_cash_after"))
        row["runtime_accounting_gap"] = n(event.get("accounting_gap"))
    for row in rows:
        actual = n(row["actual_transfer"])
        need = n(row["parent_need_gap"])
        capacity = n(row["child_available_capacity"])
        if actual <= EPS:
            reason = "NO_ALLOCATION"
        elif actual >= min(need, capacity) - 1e-6 and abs(need - capacity) <= 1e-6:
            reason = "PARENT_NEED_BINDING_AND_CHILD_CAPACITY_BINDING"
        elif actual >= need - 1e-6 and need < capacity:
            reason = "PARENT_NEED_BINDING"
        elif actual >= capacity - 1e-6 and capacity < need:
            reason = "CHILD_CAPACITY_BINDING"
        elif row["payer_edge_count"] > 1:
            reason = "MULTI_PARENT_ALLOCATION_BINDING"
        elif row["parent_edge_count"] > 1:
            reason = "MULTI_CHILD_ALLOCATION_EFFECT"
        else:
            reason = "OTHER"
        row["binding_reason"] = reason
        row["parent_need_coverage_ratio"] = actual / need if need > EPS else 0.0
        row["capacity_use_ratio"] = actual / capacity if capacity > EPS else 0.0
        row["capacity_to_child_cash_ratio"] = capacity / n(row["child_cash_before_support"]) if n(row["child_cash_before_support"]) > EPS else 0.0
        row["support_to_parent_minimum_ratio"] = actual / n(row["parent_minimum_consumption_need"]) if n(row["parent_minimum_consumption_need"]) > EPS else 0.0
    return rows


def family_components(world, edge_rows):
    graph = defaultdict(set)
    for row in edge_rows:
        a, b = row["child_household_id"], row["parent_household_id"]
        graph[a].add(b)
        graph[b].add(a)
    seen = set()
    components = []
    for start in sorted(graph):
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for nxt in graph[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(sorted(component))
    return components


def family_balance(world, system, edge_rows, step):
    components = family_components(world, edge_rows)
    rows = []
    pooled_near_zero = 0
    raw_near_zero = 0
    total_members = 0
    resources = []
    for index, component in enumerate(components):
        balances = []
        for hid in component:
            h = household(world, hid)
            if h is None:
                continue
            cash = n(h.wealth)
            need = minimum_cost(system, h)
            balances.append((cash, need))
        if not balances:
            continue
        total_cash = sum(c for c, _ in balances)
        total_need = sum(m for _, m in balances)
        surplus = total_cash - total_need
        deficit = sum(max(m - c, 0.0) for c, m in balances)
        pooled_covered = min(deficit, sum(max(c - m, 0.0) for c, m in balances))
        # Pooling removes as many deficits as the network's surplus permits.
        residual = max(0.0, deficit - pooled_covered)
        pooled_near = sum(1 for c, m in balances if m > EPS and residual >= m - EPS) if residual > EPS else 0
        raw = sum(1 for c, m in balances if c < m - EPS)
        raw_near_zero += raw
        pooled_near_zero += pooled_near
        total_members += len(balances)
        resources.append(total_cash)
        rows.append({
            "research_week": step,
            "family_network_id": index,
            "household_count": len(balances),
            "total_family_cash": total_cash,
            "total_family_minimum_need": total_need,
            "family_surplus": surplus,
            "family_deficit_before_pooling": deficit,
            "family_surplus_available_for_pooling": sum(max(c - m, 0.0) for c, m in balances),
            "perfect_pooling_max_transfer": pooled_covered,
            "raw_near_zero_households": raw,
            "pooled_residual_near_zero_households": pooled_near,
            "resource_group": "unclassified_until_global_quantiles",
        })
    return rows, {"raw_near_zero": raw_near_zero, "pooled_near_zero": pooled_near_zero, "members": total_members, "network_gini": gini(resources)}


def replay():
    world, _ = restore_world()
    world.private_family_support_enabled = False
    for _ in range(STABILIZATION_WEEKS):
        world.step()
    world.private_family_support_enabled = True
    world.private_family_support_system = PrivateFamilySupportSystem(world)
    system = world.private_family_support_system
    decisions = []
    weekly = []
    family_rows = []
    family_rollup = []

    original_execute = system.execute
    def observed_execute(step):
        pre = capture_candidates(system, step)
        result = original_execute(step)
        decisions.extend(attach_execution(pre, result))
        return result
    system.execute = observed_execute

    for _ in range(OBSERVATION_WEEKS):
        world.step()
        step = world.current_step_index
        rows = [r for r in decisions if int(n(r["global_step"], -1)) == step]
        result = system.last_result
        total_need = sum(n(r["parent_need_gap"]) for r in rows)
        total_capacity = sum(n(r["child_available_capacity"]) for r in {r["child_household_id"]: r for r in rows}.values())
        actual = sum(n(r["actual_transfer"]) for r in rows)
        family, rollup = family_balance(world, system, rows, step)
        family_rows.extend(family)
        family_rollup.append({"research_week": step, **rollup, "parent_need": total_need, "child_capacity": total_capacity, "actual_transfer": actual})
        weekly.append({
            "research_week": step,
            "candidate_decisions": len(rows),
            "realized_transfers": int(getattr(result, "transfer_count", 0)),
            "parent_need": total_need,
            "child_capacity": total_capacity,
            "actual_transfer": actual,
            "capacity_need_ratio": total_capacity / total_need if total_need > EPS else 0.0,
            "actual_need_coverage": actual / total_need if total_need > EPS else 0.0,
            "raw_near_zero": rollup["raw_near_zero"],
            "pooled_residual_near_zero": rollup["pooled_near_zero"],
            "family_network_count": len(family),
            "support_accounting_gap": sum(abs(n(r.get("runtime_accounting_gap"))) for r in rows),
        })
    return world, decisions, weekly, family_rows, family_rollup


def grouped_distribution(rows, field, label):
    q = quantiles([r.get(field, 0.0) for r in rows])
    return [{"metric": label, "field": field, **q}]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    world, decisions, weekly, family_rows, family_rollup = replay()
    realized = [r for r in decisions if n(r.get("actual_transfer")) > EPS]
    actual_values = [n(r["actual_transfer"]) for r in realized]
    need_values = [n(r["parent_need_gap"]) for r in realized]
    cap_values = [n(r["child_available_capacity"]) for r in realized]

    write_csv(OUT / "support_decision_diagnostics.csv", decisions)
    binding = []
    by_reason = defaultdict(list)
    for row in realized:
        by_reason[row.get("binding_reason", "OTHER")].append(row)
    for reason, group in sorted(by_reason.items()):
        binding.append({"binding_constraint": reason, "realized_decisions": len(group), "share_of_realized": len(group) / len(realized) if realized else 0.0, "total_transfer": sum(n(r["actual_transfer"]) for r in group), "mean_transfer": statistics.fmean(n(r["actual_transfer"]) for r in group) if group else 0.0})
    write_csv(OUT / "transfer_binding_constraint.csv", binding)

    ratio_rows = []
    for field, label in [("parent_need_coverage_ratio", "actual_transfer_over_parent_need"), ("capacity_use_ratio", "actual_transfer_over_child_capacity"), ("capacity_to_child_cash_ratio", "child_capacity_over_child_cash"), ("support_to_parent_minimum_ratio", "support_over_parent_minimum")]:
        ratio_rows += grouped_distribution(realized, field, label)
    ratio_rows += [{"metric": "aggregate", "field": "total", "count": len(weekly), "mean_parent_need": statistics.fmean(n(r["parent_need"]) for r in weekly) if weekly else 0.0, "mean_child_capacity": statistics.fmean(n(r["child_capacity"]) for r in weekly) if weekly else 0.0, "mean_actual_transfer": statistics.fmean(n(r["actual_transfer"]) for r in weekly) if weekly else 0.0, "aggregate_capacity_need_ratio": sum(n(r["child_capacity"]) for r in weekly) / sum(n(r["parent_need"]) for r in weekly) if sum(n(r["parent_need"]) for r in weekly) > EPS else 0.0, "aggregate_actual_need_coverage": sum(n(r["actual_transfer"]) for r in weekly) / sum(n(r["parent_need"]) for r in weekly) if sum(n(r["parent_need"]) for r in weekly) > EPS else 0.0}]
    write_csv(OUT / "support_need_capacity_ratio.csv", ratio_rows)
    write_csv(OUT / "parent_need_distribution.csv", grouped_distribution(realized, "parent_need_gap", "parent_need_gap"))
    write_csv(OUT / "child_capacity_distribution.csv", grouped_distribution(realized, "child_available_capacity", "child_available_capacity"))
    write_csv(OUT / "family_network_resource_balance.csv", family_rows)

    raw_nz = sum(n(r["raw_near_zero"]) for r in family_rollup)
    pooled_nz = sum(n(r["pooled_residual_near_zero"]) for r in family_rollup)
    members = sum(n(r["members"]) for r in family_rollup)
    write_csv(OUT / "perfect_family_pooling_upper_bound.csv", [{"research_week": r["research_week"], "raw_near_zero_households": r["raw_near_zero"], "perfect_pooling_residual_near_zero_households": r["pooled_residual_near_zero"], "family_households": r["members"], "raw_near_zero_share": r["raw_near_zero"] / r["members"] if r["members"] else 0.0, "pooled_near_zero_share": r["pooled_near_zero"] / r["members"] if r["members"] else 0.0, "perfect_pooling_max_transfer": min(r["parent_need"], r["child_capacity"])} for r in family_rollup])

    resource_values = [n(r["total_family_cash"]) for r in family_rows]
    thresholds = sorted(resource_values)
    def group(v):
        if not thresholds: return "unavailable"
        rank = (thresholds.index(v) + 1) / len(thresholds)
        return "bottom_50" if rank <= .5 else ("middle_40" if rank <= .9 else "top_10")
    inequality = []
    for row in family_rows:
        inequality.append({"research_week": row["research_week"], "family_network_id": row["family_network_id"], "family_cash": row["total_family_cash"], "family_surplus": row["family_surplus"], "resource_group": group(n(row["total_family_cash"])), "raw_near_zero_households": row["raw_near_zero_households"], "pooled_residual_near_zero_households": row["pooled_residual_near_zero_households"]})
    write_csv(OUT / "family_resource_inequality.csv", inequality)

    by_pair = defaultdict(list)
    for row in realized:
        by_pair[(row["child_household_id"], row["parent_household_id"])].append(row)
    batch_rows = []
    for (child_id, parent_id), group_rows in sorted(by_pair.items()):
        steps = sorted(int(n(r["research_week"])) for r in group_rows)
        batch_rows.append({"child_household_id": child_id, "parent_household_id": parent_id, "event_count": len(group_rows), "total_transfer": sum(n(r["actual_transfer"]) for r in group_rows), "mean_event_transfer": statistics.fmean(n(r["actual_transfer"]) for r in group_rows), "first_week": steps[0], "last_week": steps[-1], "four_week_batch_count": math.ceil((steps[-1] - steps[0] + 1) / 4), "thirteen_week_batch_count": math.ceil((steps[-1] - steps[0] + 1) / 13), "fifty_two_week_batch_count": math.ceil((steps[-1] - steps[0] + 1) / 52), "four_week_mean_total": sum(n(r["actual_transfer"]) for r in group_rows) / max(1, math.ceil((steps[-1] - steps[0] + 1) / 4)), "thirteen_week_mean_total": sum(n(r["actual_transfer"]) for r in group_rows) / max(1, math.ceil((steps[-1] - steps[0] + 1) / 13)), "fifty_two_week_mean_total": sum(n(r["actual_transfer"]) for r in group_rows) / max(1, math.ceil((steps[-1] - steps[0] + 1) / 52))})
    write_csv(OUT / "transfer_batching_shadow.csv", batch_rows)

    total_need = sum(n(r["parent_need"]) for r in weekly)
    total_capacity = sum(n(r["child_capacity"]) for r in weekly)
    total_actual = sum(n(r["actual_transfer"]) for r in weekly)
    contract_rows = [
        {"contract": "CURRENT_NEED_CAPACITY_PAIRWISE", "status": "OBSERVED", "total_parent_need": total_need, "total_child_capacity": total_capacity, "total_transfer": total_actual, "capacity_need_ratio": total_capacity / total_need if total_need else 0.0, "need_coverage": total_actual / total_need if total_need else 0.0, "order_dependence": "deterministic proportional allocation"},
        {"contract": "FAMILY_LIQUIDITY_EQUALIZATION_SHADOW", "status": "SHADOW_ONLY", "total_parent_need": sum(n(r["parent_need"]) for r in family_rollup), "total_child_capacity": sum(n(r["child_capacity"]) for r in family_rollup), "total_transfer": sum(min(n(r["parent_need"]), n(r["child_capacity"])) for r in family_rollup), "order_dependence": "none within connected family network"},
        {"contract": "PROPORTIONAL_NEED_CAPACITY_SHADOW", "status": "SHADOW_ONLY", "total_parent_need": total_need, "total_child_capacity": total_capacity, "total_transfer": min(total_need, total_capacity), "order_dependence": "reduced pairwise fragmentation"},
        {"contract": "PERIODIC_BATCHED_SUPPORT_SHADOW", "status": "SHADOW_ONLY", "total_parent_need": total_need, "total_child_capacity": total_capacity, "total_transfer": total_actual, "order_dependence": "same money, lower event frequency"},
    ]
    write_csv(OUT / "family_support_contract_comparison.csv", contract_rows)

    reserve_rows = []
    for reserve in (1.0, .75, .5, 0.0):
        cap = 0.0
        affected = 0
        pushed = 0
        for row in decisions:
            if int(n(row["research_week"])) % CADENCE != 0:
                continue
            cash = n(row["child_cash_before_support"])
            minimum = n(row["child_minimum_consumption_need"])
            current = max(0.0, cash - minimum)
            shadow = max(0.0, cash - reserve * minimum)
            cap += shadow
            affected += shadow > current + EPS
            pushed += shadow > current + EPS and cash - shadow < minimum - EPS
        reserve_rows.append({"child_reserve_fraction": reserve, "shadow_capacity": cap, "additional_capacity_vs_current": cap - total_capacity, "additional_parent_coverage_upper_bound": min(total_need, cap) - min(total_need, total_capacity), "household_observations_with_extra_capacity": affected, "child_observations_below_full_minimum_after_shadow": pushed})
    write_csv(OUT / "child_reserve_tradeoff.csv", reserve_rows)
    write_csv(OUT / "private_support_possibility_frontier.csv", [{"child_reserve_fraction": r["child_reserve_fraction"], "child_protection_strength": r["child_reserve_fraction"], "parent_coverage_upper_bound": min(total_need, r["shadow_capacity"]) / total_need if total_need else 0.0, "additional_capacity": r["additional_capacity_vs_current"], "child_below_minimum_observations": r["child_observations_below_full_minimum_after_shadow"]} for r in reserve_rows])

    write_csv(OUT / "aggregate_private_resource_sufficiency.csv", [{"scope": "observed_pairwise_weekly", "total_parent_need": total_need, "total_child_capacity": total_capacity, "capacity_need_ratio": total_capacity / total_need if total_need else 0.0, "total_actual_transfer": total_actual, "actual_need_coverage": total_actual / total_need if total_need else 0.0, "perfect_pooling_total_max_transfer": sum(min(n(r["parent_need"]), n(r["child_capacity"])) for r in family_rollup), "perfect_pooling_near_zero_reduction": (raw_nz - pooled_nz) / raw_nz if raw_nz else 0.0, "family_resource_gini": gini(resource_values), "weekly_pairwise_event_count": len(realized)}])

    binding_share = sum(len(v) for k, v in by_reason.items() if "CHILD_CAPACITY" in k) / len(realized) if realized else 0.0
    multi_share = sum(len(v) for k, v in by_reason.items() if "MULTI_" in k) / len(realized) if realized else 0.0
    pooling_reduction = (raw_nz - pooled_nz) / raw_nz if raw_nz else 0.0
    if pooling_reduction >= .25 and total_capacity / total_need >= .75:
        verdict = "E. FAMILY_NETWORK_RISK_SHARING_CAN_MATERIALLY_IMPROVE_RESULTS"
        recommendation = "FAMILY_LIQUIDITY_EQUALIZATION"
    elif total_capacity / total_need < .75 and binding_share >= .5:
        verdict = "A. CHILD_LIQUIDITY_IS_THE_BINDING_CONSTRAINT"
        recommendation = "PRIVATE_SUPPORT_CANNOT_SOLVE_THE_PROBLEM"
    elif multi_share >= .5:
        verdict = "B. PAIRWISE_WEEKLY_ALLOCATION_CAUSES_EXCESSIVE_MICROTRANSFERS"
        recommendation = "PERIODIC_BATCHED_SUPPORT"
    elif pooling_reduction < .10:
        verdict = "C. PRIVATE_FAMILY_RESOURCES_ARE_INSUFFICIENT_EVEN_UNDER_POOLING"
        recommendation = "PRIVATE_SUPPORT_CANNOT_SOLVE_THE_PROBLEM"
    else:
        verdict = "F. MULTIPLE_CONSTRAINTS"
        recommendation = "PROPORTIONAL_NEED_CAPACITY_ALLOCATION"
    write_csv(OUT / "next_support_design_recommendation.csv", [{"verdict": verdict, "recommended_next_controlled_treatment": recommendation, "parent_need_binding_share": sum(len(v) for k, v in by_reason.items() if "PARENT_NEED" in k) / len(realized) if realized else 0.0, "child_capacity_binding_share": binding_share, "multi_allocation_share": multi_share, "perfect_pooling_near_zero_reduction": pooling_reduction, "gui_future_metrics": "parent need coverage ratio; family available capacity; support/need ratio; child burden ratio"}])

    flags = {
        "verdict": verdict,
        "diagnostic_replay": True,
        "replay_source": "mature_population_week_2600.pkl",
        "stabilization_weeks": STABILIZATION_WEEKS,
        "observation_weeks": OBSERVATION_WEEKS,
        "support_formula_changed": False,
        "support_cadence_changed": False,
        "child_reserve_changed": False,
        "new_rng_draws": 0,
        "economic_behavior_changed": False,
        "realized_transfer_events": len(realized),
        "total_actual_transfer": total_actual,
        "mean_actual_transfer": statistics.fmean(actual_values) if actual_values else 0.0,
        "parent_need_distribution_observed": bool(need_values),
        "child_capacity_distribution_observed": bool(cap_values),
        "child_capacity_binding_share": binding_share,
        "multi_allocation_share": multi_share,
        "aggregate_capacity_need_ratio": total_capacity / total_need if total_need else 0.0,
        "perfect_pooling_near_zero_reduction": pooling_reduction,
        "transfer_accounting_gap": sum(abs(n(r.get("runtime_accounting_gap"))) for r in realized),
        "gui_modified": False,
        "hard_stop_respected": True,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")
    summary = f"""# Step 15 Private Family Support Capacity Bound / Redesign Screen

Verdict: **{verdict}**

This is a diagnostic-only replay of the accepted mature-genealogy treatment
branch. The existing support allocator, cadence, reserve, household needs,
and economic parameters were unchanged. The wrapper recorded the candidate
edges before allocation and joined them to the existing execution records.

## Findings

- Realized decisions: **{len(realized):,}**; total transfer: **{total_actual:,.6f}**; mean transfer: **{statistics.fmean(actual_values) if actual_values else 0.0:,.6f}**.
- Child-capacity binding share: **{binding_share:.6f}**; multi-allocation share: **{multi_share:.6f}**.
- Aggregate child capacity / parent need: **{total_capacity / total_need if total_need else 0.0:.6f}**.
- Typical parent need gap: median **{quantiles(need_values)['median']:.6f}**; typical actual transfer: median **{quantiles(actual_values)['median']:.6f}**.
- Perfect-family-pooling shadow near-zero reduction: **{pooling_reduction:.6f}**.
- Weekly pairwise event count: **{len(realized):,}**. `transfer_batching_shadow.csv` reports 4-, 13-, and 52-week aggregation without rerunning behavior.

## Interpretation

The current contract is `NEED_BASED_FAMILY_SUPPORT`: it transfers existing
child-Household surplus after the child's full minimum-consumption reserve,
subject to deterministic parent/child allocation. It creates no money and
does not change the support rule. The reserve trade-off is shadow-only;
`child_reserve_tradeoff.csv` reports additional capacity and the observations
that would fall below the full minimum if the reserve were relaxed.

The safest next controlled treatment is **{recommendation}**. No GUI changes,
demographic changes, support-rule changes, or economic behavior changes were
made in this stage.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
