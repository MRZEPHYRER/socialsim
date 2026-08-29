"""Step 17.A shadow audit of household liquidity and family capacity.

The script uses the accepted mature demographic checkpoint only to provide
authoritative living genealogy.  Economic state is freshly initialized by the
existing Step 15 research harness and advanced for one short observation
window.  No transfer is executed and no ledger or production module is
modified.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "test/output/step17_a_household_transfer_capacity"
CHECKPOINT = ROOT / "test/output/mature_population_checkpoint/mature_population_week_2600.pkl"
MATURE_HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
OBSERVATION_WEEKS = 52
EPS = 1e-9
RESERVES = (13, 26, 52)
TARGETS = (0.25, 0.50, 1.00)
LOW_THRESHOLDS = (0.25, 0.50, 1.00)
RICH_THRESHOLDS = (13, 26, 52)


def load_harness():
    spec = importlib.util.spec_from_file_location("step15_mature_harness", MATURE_HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def number(value, default=math.nan):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def write_rows(path, rows, fields=None):
    rows = list(rows)
    inferred = []
    for row in rows:
        for key in row:
            if key not in inferred:
                inferred.append(key)
    fields = list(fields or inferred or ["status"])
    for field in inferred:
        if field not in fields:
            fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, q):
    values = sorted(number(value, 0.0) for value in values)
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def gini(values):
    values = sorted(max(0.0, number(value, 0.0)) for value in values)
    total = sum(values)
    if not values or total <= EPS:
        return 0.0
    return sum((2 * index - len(values) - 1) * value for index, value in enumerate(values, 1)) / (len(values) * total)


def share(value, total):
    return number(value, 0.0) / total if total > EPS else 0.0



def liquidity_band(value):
    if not math.isfinite(number(value)):
        return "UNAVAILABLE"
    value = number(value, 0.0)
    if value < 0.25:
        return "LT_0_25"
    if value < 1:
        return "0_25_TO_1"
    if value < 4:
        return "1_TO_4"
    if value < 13:
        return "4_TO_13"
    if value < 26:
        return "13_TO_26"
    if value < 52:
        return "26_TO_52"
    return "GE_52"

def social_households(world):
    return sorted(
        [h for h in world.households if not getattr(h, "settlement_only", False)],
        key=lambda h: h.id,
    )


def minimum_cost(world, household):
    units = world.needs_system.household_minimum_need_units(household)
    try:
        price = max(1e-12, float(world.household_planning_price_this_step()))
    except (AttributeError, TypeError, ValueError):
        price = 1.0
    return max(0.0, number(units, 0.0)) * price


def household_members(world, households):
    members = defaultdict(list)
    social_ids = {h.id for h in households}
    for person in world.population:
        if not getattr(person, "alive", False):
            continue
        household_id = getattr(person, "household_id", None)
        if household_id in social_ids:
            members[household_id].append(person)
    return members


def life_cycle(member_list):
    adults = [p for p in member_list if number(p.age, 0.0) >= 20]
    children = [p for p in member_list if number(p.age, 0.0) < 20]
    elderly = [p for p in adults if number(p.age, 0.0) >= 65]
    working = [p for p in adults if 20 <= number(p.age, 0.0) < 65]
    if elderly and working:
        group = "MIXED_GENERATION"
    elif elderly:
        group = "ELDERLY"
    elif working and max(number(p.age, 0.0) for p in working) >= 40:
        group = "OLDER_WORKING_AGE"
    elif working:
        group = "WORKING_AGE"
    else:
        group = "YOUNG_ADULT"
    return {
        "adults": adults,
        "children": children,
        "elderly": elderly,
        "working": working,
        "group": group,
    }


def build_household_rows(world, households):
    members = household_members(world, households)
    rows = []
    for household in households:
        people = members.get(household.id, [])
        composition = life_cycle(people)
        minimum = minimum_cost(world, household)
        cash = number(getattr(household, "wealth", math.nan))
        liquidity = cash / minimum if math.isfinite(cash) and minimum > EPS else math.nan
        employed = [p for p in people if getattr(p, "firm_id", None) is not None]
        adult_count = len(composition["adults"])
        income = number(getattr(household, "income_this_step", math.nan))
        saving = number(getattr(household, "saving_this_step", math.nan))
        rows.append({
            "household_id": household.id,
            "cash": cash,
            "weekly_minimum_consumption_cost": minimum,
            "liquidity_weeks": liquidity,
            "liquidity_band": liquidity_band(liquidity),
            "member_count": len(people),
            "adult_count": adult_count,
            "child_count": len(composition["children"]),
            "elderly_count": len(composition["elderly"]),
            "employed_member_count": len(employed),
            "oldest_adult_age": max((number(p.age, math.nan) for p in composition["adults"]), default=math.nan),
            "youngest_adult_age": min((number(p.age, math.nan) for p in composition["adults"]), default=math.nan),
            "dependency_ratio": (len(composition["children"]) + len(composition["elderly"])) / adult_count if adult_count else math.nan,
            "current_week_income": income,
            "current_week_saving": saving,
            "current_equity_assets": number(getattr(household, "equity_asset_value", 0.0), 0.0),
            "financial_net_worth": cash + number(getattr(household, "equity_asset_value", 0.0), 0.0),
            "life_cycle_group": composition["group"],
        })
    return rows


def cash_concentration(rows):
    ranked = sorted(rows, key=lambda row: number(row["cash"], 0.0))
    cash = [number(row["cash"], 0.0) for row in ranked]
    total = sum(cash)
    n = len(cash)
    return {
        "cash_gini": gini(cash),
        "bottom_50_cash_share": share(sum(cash[: max(1, n // 2)]), total),
        "top_10_cash_share": share(sum(cash[-max(1, math.ceil(n * 0.10)):]), total),
        "top_1_cash_share": share(sum(cash[-max(1, math.ceil(n * 0.01)):]), total),
        "total_cash": total,
    }


def genealogy(world, households):
    social_ids = {h.id for h in households}
    people = {p.id: p for p in world.population if getattr(p, "alive", False)}
    valid_parent_people = 0
    valid_child_people = 0
    reciprocal_parent_links = 0
    pair_data = {}
    same_household_links = 0
    for child in people.values():
        parent_ids = set(getattr(child, "parent_ids", []))
        valid_parents = [people[pid] for pid in parent_ids if pid in people and child.id in set(getattr(people[pid], "children_ids", []))]
        if valid_parents:
            valid_parent_people += 1
            reciprocal_parent_links += len(valid_parents)
        valid_children = [people[cid] for cid in set(getattr(child, "children_ids", [])) if cid in people and child.id in set(getattr(people[cid], "parent_ids", []))]
        if valid_children:
            valid_child_people += 1
        child_hid = getattr(child, "household_id", None)
        if child_hid not in social_ids:
            continue
        for parent in valid_parents:
            parent_hid = getattr(parent, "household_id", None)
            if parent_hid not in social_ids:
                continue
            if parent_hid == child_hid:
                same_household_links += 1
                continue
            key = (parent_hid, child_hid)
            entry = pair_data.setdefault(key, {"parent_person_ids": set(), "child_person_ids": set()})
            entry["parent_person_ids"].add(parent.id)
            entry["child_person_ids"].add(child.id)
    edge_rows = []
    edge_roles = {}
    for (parent_hid, child_hid), data in sorted(pair_data.items()):
        edge_roles[(parent_hid, child_hid)] = "PARENT_TO_CHILD"
        edge_roles[(child_hid, parent_hid)] = "CHILD_TO_PARENT"
        edge_rows.append({
            "parent_household_id": parent_hid,
            "adult_child_household_id": child_hid,
            "parent_to_child_valid": True,
            "child_to_parent_valid": True,
            "parent_person_ids": ";".join(map(str, sorted(data["parent_person_ids"]))),
            "adult_child_person_ids": ";".join(map(str, sorted(data["child_person_ids"]))),
            "unique_person_link_count": len(data["parent_person_ids"] | data["child_person_ids"]),
        })
    elderly_households = set()
    elderly_connected = set()
    adult_child_households = set()
    adult_child_with_parent = set()
    for person in people.values():
        hid = getattr(person, "household_id", None)
        if hid not in social_ids:
            continue
        if number(person.age, 0.0) >= 65:
            elderly_households.add(hid)
            if any(key[0] == hid for key in pair_data):
                elderly_connected.add(hid)
        if 20 <= number(person.age, 0.0) < 65:
            adult_child_households.add(hid)
            if any(key[1] == hid for key in pair_data):
                adult_child_with_parent.add(hid)
    coverage = {
        "alive_person_count": len(people),
        "person_with_valid_parent_relationship": valid_parent_people,
        "person_with_valid_child_relationship": valid_child_people,
        "valid_reciprocal_parent_child_links": reciprocal_parent_links,
        "same_household_parent_child_links": same_household_links,
        "distinct_social_household_edge_count": len(edge_rows),
        "elderly_social_households": len(elderly_households),
        "elderly_with_adult_child_household_connection": len(elderly_connected),
        "adult_child_social_households": len(adult_child_households),
        "adult_child_with_living_parent_household": len(adult_child_with_parent),
        "parent_relationship_share": share(valid_parent_people, len(people)),
        "child_relationship_share": share(valid_child_people, len(people)),
        "elderly_connection_share": share(len(elderly_connected), len(elderly_households)),
        "adult_child_parent_connection_share": share(len(adult_child_with_parent), len(adult_child_households)),
        "network_coverage_interpretation": "LOWER_BOUND_INITIAL_POPULATION_RETROSPECTIVE_GENEALOGY_ABSENT",
    }
    return edge_rows, edge_roles, coverage


def max_flow(donor_caps, recipient_caps, edges):
    source, sink = ("source",), ("sink",)
    capacity = defaultdict(float)
    graph = defaultdict(set)
    for donor, amount in donor_caps.items():
        dnode = ("d", donor)
        capacity[source, dnode] = max(0.0, amount)
        graph[source].add(dnode)
        graph[dnode].add(source)
    for recipient, amount in recipient_caps.items():
        rnode = ("r", recipient)
        capacity[rnode, sink] = max(0.0, amount)
        graph[rnode].add(sink)
        graph[sink].add(rnode)
    for donor, recipient in sorted(edges):
        dnode, rnode = ("d", donor), ("r", recipient)
        capacity[dnode, rnode] = math.inf
        graph[dnode].add(rnode)
        graph[rnode].add(dnode)
    flow = defaultdict(float)
    while True:
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for nxt in sorted(graph[node], key=str):
                residual = capacity[node, nxt] - flow[node, nxt]
                if residual > EPS and nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        if sink not in parent:
            break
        amount = math.inf
        node = sink
        while parent[node] is not None:
            amount = min(amount, capacity[parent[node], node] - flow[parent[node], node])
            node = parent[node]
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            flow[previous, node] += amount
            flow[node, previous] -= amount
            node = previous
    allocations = {
        (donor, recipient): max(0.0, flow[("d", donor), ("r", recipient)])
        for donor, recipient in edges
        if flow[("d", donor), ("r", recipient)] > EPS
    }
    donor_used = defaultdict(float)
    recipient_used = defaultdict(float)
    for (donor, recipient), amount in allocations.items():
        donor_used[donor] += amount
        recipient_used[recipient] += amount
    return allocations, donor_used, recipient_used


def network_allocation(rows_by_id, edge_roles, reserve, target, direction="BOTH"):
    recipients = {
        hid: max(0.0, target * number(row["weekly_minimum_consumption_cost"], 0.0) - number(row["cash"], 0.0))
        for hid, row in rows_by_id.items()
        if number(row["liquidity_weeks"], math.inf) < 1.0
    }
    donors = {
        hid: max(0.0, number(row["cash"], 0.0) - reserve * number(row["weekly_minimum_consumption_cost"], 0.0))
        for hid, row in rows_by_id.items()
        if number(row["cash"], 0.0) - reserve * number(row["weekly_minimum_consumption_cost"], 0.0) > EPS
    }
    eligible_edges = []
    for (donor, recipient), role in edge_roles.items():
        if donor not in donors or recipient not in recipients:
            continue
        if direction == "PARENT_TO_CHILD" and role != "PARENT_TO_CHILD":
            continue
        if direction == "CHILD_TO_PARENT" and role != "CHILD_TO_PARENT":
            continue
        eligible_edges.append((donor, recipient))
    allocations, donor_used, recipient_used = max_flow(donors, recipients, eligible_edges)
    return {
        "recipients": recipients,
        "donors": donors,
        "allocations": allocations,
        "donor_used": donor_used,
        "recipient_used": recipient_used,
    }


def profile_threshold(rows, threshold):
    selected = [row for row in rows if number(row["liquidity_weeks"], -math.inf) >= threshold]
    cash = [number(row["cash"], 0.0) for row in selected]
    sizes = [number(row["member_count"], 0.0) for row in selected]
    return {
        "liquidity_threshold_weeks": threshold,
        "household_count": len(selected),
        "household_share": share(len(selected), len(rows)),
        "total_cash": sum(cash),
        "share_of_all_household_cash": share(sum(cash), sum(number(r["cash"], 0.0) for r in rows)),
        "median_cash": quantile(cash, 0.5),
        "median_liquidity_weeks": quantile([row["liquidity_weeks"] for row in selected], 0.5),
        "mean_household_size": sum(sizes) / len(sizes) if sizes else math.nan,
        "mean_oldest_adult_age": sum(number(r["oldest_adult_age"], 0.0) for r in selected) / len(selected) if selected else math.nan,
        "mean_youngest_adult_age": sum(number(r["youngest_adult_age"], 0.0) for r in selected) / len(selected) if selected else math.nan,
        "mean_employed_member_count": sum(number(r["employed_member_count"], 0.0) for r in selected) / len(selected) if selected else math.nan,
        "mean_elderly_member_count": sum(number(r["elderly_count"], 0.0) for r in selected) / len(selected) if selected else math.nan,
        "mean_dependent_children": sum(number(r["child_count"], 0.0) for r in selected) / len(selected) if selected else math.nan,
        "mean_dependency_ratio": sum(number(r["dependency_ratio"], 0.0) for r in selected) / len(selected) if selected else math.nan,
        "mean_current_week_income": sum(number(r["current_week_income"], 0.0) for r in selected) / len(selected) if selected else math.nan,
        "mean_current_week_saving": sum(number(r["current_week_saving"], 0.0) for r in selected) / len(selected) if selected else math.nan,
        "mean_weekly_minimum_consumption": sum(number(r["weekly_minimum_consumption_cost"], 0.0) for r in selected) / len(selected) if selected else math.nan,
        "life_cycle_groups": ";".join(sorted({r["life_cycle_group"] for r in selected})),
    }


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(str(CHECKPOINT))
    harness = load_harness()
    world, checkpoint_state = harness.restore_world()
    initial_population = len(world.population)
    initial_households = len(social_households(world))
    for _ in range(OBSERVATION_WEEKS):
        world.step()
    households = social_households(world)
    rows = build_household_rows(world, households)
    rows_by_id = {row["household_id"]: row for row in rows}
    edge_rows, edge_roles, coverage = genealogy(world, households)
    concentration = cash_concentration(rows)

    snapshot = {
        "snapshot_type": "FINAL_AVAILABLE_REPRESENTATIVE_STATE",
        "research_week": OBSERVATION_WEEKS,
        "absolute_demographic_week": 2600 + OBSERVATION_WEEKS,
        "population": len([p for p in world.population if getattr(p, "alive", False)]),
        "person_count_in_runtime": len(world.population),
        "social_household_count": len(households),
        "settlement_only_household_count": len(world.households) - len(households),
        "initial_checkpoint_population": initial_population,
        "initial_checkpoint_social_households": initial_households,
        "genealogy_coverage_note": coverage["network_coverage_interpretation"],
        "economic_state_source": "fresh_economic_initialization_from_accepted_mature_genealogy_harness",
    }
    write_rows(OUT / "analysis_snapshot.csv", [snapshot])
    write_rows(OUT / "household_liquidity_distribution.csv", rows)
    write_rows(OUT / "high_liquidity_household_profile.csv", [profile_threshold(rows, threshold) for threshold in RICH_THRESHOLDS])

    lifecycle_rows = []
    for group in sorted({row["life_cycle_group"] for row in rows}):
        group_rows = [row for row in rows if row["life_cycle_group"] == group]
        lifecycle_rows.append({
            "life_cycle_group": group,
            "household_count": len(group_rows),
            "share_of_households": share(len(group_rows), len(rows)),
            "total_cash": sum(number(r["cash"], 0.0) for r in group_rows),
            "median_cash": quantile([r["cash"] for r in group_rows], 0.5),
            "median_liquidity_weeks": quantile([r["liquidity_weeks"] for r in group_rows], 0.5),
            "mean_income": sum(number(r["current_week_income"], 0.0) for r in group_rows) / len(group_rows),
            "mean_employed_members": sum(number(r["employed_member_count"], 0.0) for r in group_rows) / len(group_rows),
        })
    write_rows(OUT / "household_lifecycle_profile.csv", lifecycle_rows)
    concentration_rows_by_band = [{"snapshot": "final", "liquidity_group": "ALL", **concentration}]
    band_order = ("LT_0_25", "0_25_TO_1", "1_TO_4", "4_TO_13", "13_TO_26", "26_TO_52", "GE_52", "UNAVAILABLE")
    for band in band_order:
        band_rows = [row for row in rows if row["liquidity_band"] == band]
        band_concentration = cash_concentration(band_rows)
        concentration_rows_by_band.append({
            "snapshot": "final",
            "liquidity_group": band,
            "household_count": len(band_rows),
            **band_concentration,
        })
    write_rows(OUT / "household_cash_concentration.csv", concentration_rows_by_band)

    write_rows(OUT / "genealogy_coverage.csv", [{"metric": key, "value": value} for key, value in coverage.items()])
    write_rows(OUT / "household_family_edges.csv", edge_rows)

    connectivity_rows = []
    for low in LOW_THRESHOLDS:
        low_ids = {row["household_id"] for row in rows if number(row["liquidity_weeks"], math.inf) < low}
        for rich in RICH_THRESHOLDS:
            parent_available = {hid for hid in low_ids if any(donor == parent and recipient == hid and number(rows_by_id[donor]["liquidity_weeks"], -math.inf) >= rich for (donor, recipient), role in edge_roles.items() if role == "PARENT_TO_CHILD" for parent in [donor])}
            child_available = {hid for hid in low_ids if any(donor == child and recipient == hid and number(rows_by_id[donor]["liquidity_weeks"], -math.inf) >= rich for (donor, recipient), role in edge_roles.items() if role == "CHILD_TO_PARENT" for child in [donor])}
            either = parent_available | child_available
            connectivity_rows.append({
                "recipient_liquidity_threshold": low,
                "donor_liquidity_threshold": rich,
                "low_liquidity_households": len(low_ids),
                "richer_parent_household_available": len(parent_available),
                "richer_adult_child_household_available": len(child_available),
                "either_direction_available": len(either),
                "parent_coverage_share": share(len(parent_available), len(low_ids)),
                "adult_child_coverage_share": share(len(child_available), len(low_ids)),
                "either_coverage_share": share(len(either), len(low_ids)),
            })
    write_rows(OUT / "low_liquidity_family_connectivity.csv", connectivity_rows)

    degree_rows = []
    for low in LOW_THRESHOLDS:
        for row in rows:
            if number(row["liquidity_weeks"], math.inf) >= low:
                continue
            hid = row["household_id"]
            neighbors = set()
            rich_neighbors = set()
            for (donor, recipient), role in edge_roles.items():
                if donor == hid:
                    neighbors.add(recipient)
                    if number(rows_by_id[recipient]["liquidity_weeks"], -math.inf) >= 13:
                        rich_neighbors.add(recipient)
                if recipient == hid:
                    neighbors.add(donor)
                    if number(rows_by_id[donor]["liquidity_weeks"], -math.inf) >= 13:
                        rich_neighbors.add(donor)
            degree_rows.append({
                "recipient_liquidity_threshold": low,
                "household_id": hid,
                "connected_social_households": len(neighbors),
                "degree_band": "0" if not neighbors else "1" if len(neighbors) == 1 else "2" if len(neighbors) == 2 else "3+",
                "connected_high_liquidity_donor_candidates_ge_13": len(rich_neighbors),
            })
    write_rows(OUT / "family_network_degree.csv", degree_rows)

    donor_rows = []
    for row in rows:
        for reserve in RESERVES:
            reserve_cash = reserve * number(row["weekly_minimum_consumption_cost"], 0.0)
            donor_rows.append({
                "household_id": row["household_id"],
                "reserve_weeks": reserve,
                "cash": row["cash"],
                "safety_reserve_cash": reserve_cash,
                "excess_cash": max(0.0, number(row["cash"], 0.0) - reserve_cash),
                "is_donor": number(row["cash"], 0.0) - reserve_cash > EPS,
            })
    write_rows(OUT / "donor_excess_capacity.csv", donor_rows)

    recipient_rows = []
    for row in rows:
        for threshold in LOW_THRESHOLDS:
            if number(row["liquidity_weeks"], math.inf) >= threshold:
                continue
            for target in TARGETS:
                target_cash = target * number(row["weekly_minimum_consumption_cost"], 0.0)
                recipient_rows.append({
                    "household_id": row["household_id"],
                    "recipient_liquidity_definition": threshold,
                    "target_weeks": target,
                    "cash": row["cash"],
                    "target_cash": target_cash,
                    "liquidity_gap_cash": max(0.0, target_cash - number(row["cash"], 0.0)),
                })
    write_rows(OUT / "recipient_liquidity_gaps.csv", recipient_rows)

    global_rows, family_rows, directional_rows, allocation_rows = [], [], [], []
    concentration_rows, recipient_composition_rows = [], []
    for reserve in RESERVES:
        donor_excess = sum(max(0.0, number(row["cash"], 0.0) - reserve * number(row["weekly_minimum_consumption_cost"], 0.0)) for row in rows)
        for target in TARGETS:
            recipient_gaps = {
                row["household_id"]: max(0.0, target * number(row["weekly_minimum_consumption_cost"], 0.0) - number(row["cash"], 0.0))
                for row in rows if number(row["liquidity_weeks"], math.inf) < 1.0
            }
            total_gap = sum(recipient_gaps.values())
            global_covered = min(donor_excess, total_gap)
            global_rows.append({
                "reserve_weeks": reserve,
                "target_weeks": target,
                "recipient_definition": "liquidity_lt_1",
                "total_recipient_gap": total_gap,
                "total_donor_excess_cash": donor_excess,
                "global_coverage_cash": global_covered,
                "global_coverage_share": share(global_covered, total_gap),
            })
            both = network_allocation(rows_by_id, edge_roles, reserve, target, "BOTH")
            network_covered = sum(both["allocations"].values())
            family_rows.append({
                "reserve_weeks": reserve,
                "target_weeks": target,
                "direction": "BOTH",
                "total_recipient_gap": sum(both["recipients"].values()),
                "total_donor_excess_cash": sum(both["donors"].values()),
                "family_network_transferable_cash": network_covered,
                "family_network_coverage_share": share(network_covered, sum(both["recipients"].values())),
                "fully_covered_recipients": sum(abs(both["recipient_used"].get(hid, 0.0) - gap) <= 1e-7 for hid, gap in both["recipients"].items()),
                "partially_covered_recipients": sum(0 < both["recipient_used"].get(hid, 0.0) < gap - 1e-7 for hid, gap in both["recipients"].items()),
                "uncovered_recipients": sum(both["recipient_used"].get(hid, 0.0) <= EPS for hid in both["recipients"]),
                "donor_households_used": sum(amount > EPS for amount in both["donor_used"].values()),
                "residual_donor_excess_cash": sum(both["donors"].values()) - network_covered,
            })
            for direction in ("PARENT_TO_CHILD", "CHILD_TO_PARENT"):
                directed = network_allocation(rows_by_id, edge_roles, reserve, target, direction)
                directed_covered = sum(directed["allocations"].values())
                directional_rows.append({
                    "reserve_weeks": reserve,
                    "target_weeks": target,
                    "direction": direction,
                    "total_recipient_gap": sum(directed["recipients"].values()),
                    "family_network_transferable_cash": directed_covered,
                    "family_network_coverage_share": share(directed_covered, sum(directed["recipients"].values())),
                    "recipient_households_covered": sum(amount > EPS for amount in directed["recipient_used"].values()),
                    "donor_households_used": sum(amount > EPS for amount in directed["donor_used"].values()),
                })
            for (donor, recipient), amount in both["allocations"].items():
                allocation_rows.append({
                    "reserve_weeks": reserve,
                    "target_weeks": target,
                    "direction": edge_roles.get((donor, recipient), "UNKNOWN"),
                    "donor_household_id": donor,
                    "recipient_household_id": recipient,
                    "transfer_amount": amount,
                    "donor_excess_cash": both["donors"].get(donor, 0.0),
                    "recipient_gap": both["recipients"].get(recipient, 0.0),
                })
            donor_rank = sorted(rows, key=lambda row: number(row["cash"], 0.0), reverse=True)
            used = both["donor_used"]
            top10 = {row["household_id"] for row in donor_rank[:max(1, math.ceil(len(rows) * 0.10))]}
            top1 = {row["household_id"] for row in donor_rank[:max(1, math.ceil(len(rows) * 0.01))]}
            utilizations = [share(used.get(hid, 0.0), cap) for hid, cap in both["donors"].items()]
            concentration_rows.append({
                "reserve_weeks": reserve,
                "target_weeks": target,
                "direction": "BOTH",
                "top_10_donor_transfer_share": share(sum(amount for hid, amount in used.items() if hid in top10), network_covered),
                "top_1_donor_transfer_share": share(sum(amount for hid, amount in used.items() if hid in top1), network_covered),
                "median_donor_utilization": quantile(utilizations, 0.5),
                "maximum_donor_utilization": max(utilizations, default=0.0),
            })
            for hid, amount in both["recipient_used"].items():
                member_row = rows_by_id[hid]
                recipient_composition_rows.append({
                    "reserve_weeks": reserve,
                    "target_weeks": target,
                    "recipient_household_id": hid,
                    "received_shadow_cash": amount,
                    "elderly": number(member_row["elderly_count"], 0.0) > 0,
                    "no_employed_member": number(member_row["employed_member_count"], 0.0) == 0,
                    "life_cycle_group": member_row["life_cycle_group"],
                    "has_children": number(member_row["child_count"], 0.0) > 0,
                    "household_size": member_row["member_count"],
                })
    write_rows(OUT / "global_capacity_matrix.csv", global_rows)
    write_rows(OUT / "family_network_capacity_matrix.csv", family_rows)
    write_rows(OUT / "directional_capacity_matrix.csv", directional_rows)
    write_rows(OUT / "shadow_transfer_allocation.csv", allocation_rows)
    write_rows(OUT / "transfer_concentration.csv", concentration_rows)
    write_rows(OUT / "recipient_composition.csv", recipient_composition_rows)

    post_rows = []
    for reserve in RESERVES:
        for target in TARGETS:
            both = network_allocation(rows_by_id, edge_roles, reserve, target, "BOTH")
            changes = defaultdict(float)
            for (donor, recipient), amount in both["allocations"].items():
                changes[donor] -= amount
                changes[recipient] += amount
            hypothetical = [number(row["cash"], 0.0) + changes[row["household_id"]] for row in rows]
            for threshold in (0.25, 0.5, 1.0, 2.0):
                liquidities = [cash / number(row["weekly_minimum_consumption_cost"], 0.0) if number(row["weekly_minimum_consumption_cost"], 0.0) > EPS else math.nan for cash, row in zip(hypothetical, rows)]
                post_rows.append({
                    "reserve_weeks": reserve,
                    "recipient_target_weeks": target,
                    "post_transfer_liquidity_threshold": threshold,
                    "household_count_below_threshold": sum(math.isfinite(value) and value < threshold for value in liquidities),
                    "share_below_threshold": share(sum(math.isfinite(value) and value < threshold for value in liquidities), len(rows)),
                    "bottom_50_cash_share": cash_concentration([{**row, "cash": cash} for row, cash in zip(rows, hypothetical)])["bottom_50_cash_share"],
                    "top_10_cash_share": cash_concentration([{**row, "cash": cash} for row, cash in zip(rows, hypothetical)])["top_10_cash_share"],
                    "top_1_cash_share": cash_concentration([{**row, "cash": cash} for row, cash in zip(rows, hypothetical)])["top_1_cash_share"],
                    "cash_gini": gini(hypothetical),
                    "hypothetical_only": True,
                })
    write_rows(OUT / "hypothetical_post_transfer_distribution.csv", post_rows)

    parent_capacity = {(r["reserve_weeks"], r["target_weeks"]): r["family_network_transferable_cash"] for r in directional_rows if r["direction"] == "PARENT_TO_CHILD"}
    child_capacity = {(r["reserve_weeks"], r["target_weeks"]): r["family_network_transferable_cash"] for r in directional_rows if r["direction"] == "CHILD_TO_PARENT"}
    family_coverage = {(r["reserve_weeks"], r["target_weeks"]): r["family_network_coverage_share"] for r in family_rows}
    global_coverage = {(r["reserve_weeks"], r["target_weeks"]): r["global_coverage_share"] for r in global_rows}
    rich_profiles = {threshold: profile_threshold(rows, threshold) for threshold in RICH_THRESHOLDS}
    total_donor_capacity = {reserve: sum(max(0.0, number(row["cash"], 0.0) - reserve * number(row["weekly_minimum_consumption_cost"], 0.0)) for row in rows) for reserve in RESERVES}
    all_family = [value for value in family_coverage.values()]
    classification = "D. AGGREGATE_CASH_IS_SUFFICIENT_BUT_GENEALOGY_NETWORK_IS_THE_BINDING_LIMIT" if max(global_coverage.values(), default=0.0) >= 0.90 and max(family_coverage.values(), default=0.0) < 0.50 else "A. FAMILY_NETWORK_HAS_LARGE_UNUSED_LIQUIDITY_CAPACITY" if all_family and max(all_family) >= 0.50 else "B. FAMILY_NETWORK_HAS_MODERATE_BUT_MEANINGFUL_CAPACITY" if all_family and max(all_family) >= 0.10 else "C. FAMILY_NETWORK_CAPACITY_IS_TOO_SMALL_TO_SOLVE_LOW_LIQUIDITY"
    if coverage["parent_relationship_share"] < 0.05 or coverage["distinct_social_household_edge_count"] == 0:
        classification = "F. GENEALOGY_COVERAGE_TOO_WEAK_FOR_RELIABLE_CONCLUSION"
    parent_max = max(parent_capacity.values(), default=0.0)
    child_max = max(child_capacity.values(), default=0.0)
    summary = [
        "# Step 17.A Household Liquidity and Family Transfer Capacity Audit",
        "",
        f"Verdict: **{classification}**",
        "",
        "## Analysis Snapshot",
        f"The analysis uses research week {snapshot['research_week']} (absolute demographic week {snapshot['absolute_demographic_week']}) after a fresh economic initialization over the accepted mature demographic checkpoint. It contains {snapshot['population']} living Persons and {snapshot['social_household_count']} Social Households. Settlement-only accounts are excluded from the Household distribution.",
        "",
        "## Question 1: Who are the high-liquidity Households?",
        f"Households at or above 13, 26, and 52 liquidity weeks number {rich_profiles[13]['household_count']}, {rich_profiles[26]['household_count']}, and {rich_profiles[52]['household_count']}. They hold {rich_profiles[13]['share_of_all_household_cash']:.2%}, {rich_profiles[26]['share_of_all_household_cash']:.2%}, and {rich_profiles[52]['share_of_all_household_cash']:.2%} of Social-Household cash, respectively. Their lifecycle groups are shown in `high_liquidity_household_profile.csv`; the cross-section is descriptive and does not establish causality.",
        "",
        "## Question 2: Family connectivity",
        f"The authoritative living genealogy yields {coverage['distinct_social_household_edge_count']} distinct Social-Household parent/child edges. Low-liquidity connectivity is reported separately for richer parent and richer adult-child donors in `low_liquidity_family_connectivity.csv`; initial-population retrospective genealogy is absent, so this is an explicit lower bound for the canonical population.",
        "",
        "## Question 3: Shadow coverage",
        f"Aggregate donor excess cash under R13/R26/R52 is {total_donor_capacity[13]:.2f}, {total_donor_capacity[26]:.2f}, and {total_donor_capacity[52]:.2f}. Global and real-family-network coverage for 0.25, 0.50, and 1.00-week recipient targets are in the two capacity matrices. Maximum family-network coverage in this screen is {max(family_coverage.values(), default=0.0):.2%}, versus {max(global_coverage.values(), default=0.0):.2%} for the pooled upper bound.",
        "",
        "## Direction and interpretation",
        f"The larger directional shadow capacity is {'rich parent -> adult child' if parent_max >= child_max else 'rich adult child -> parent'} ({max(parent_max, child_max):.2f} transferable cash versus {min(parent_max, child_max):.2f}). These are capacity bounds only; no Ledger transfer was executed. Liquidity-floor and cash-polarization effects are shown separately in `hypothetical_post_transfer_distribution.csv`.",
        "",
        "## Integrity and limits",
        "The shadow allocator preserves each donor's selected reserve and each recipient's target gap, and mutates no runtime object. The accepted private-support contract remains disabled. No pension, retirement, social assistance, or active transfer mechanism was introduced.",
        "",
        "## Artifacts",
        "All required CSV outputs are in `test/output/step17_a_household_transfer_capacity/`. The calculation uses authoritative `parent_ids`, `children_ids`, `Person.household_id`, Household cash, and `NeedsSystem.household_minimum_need_units`; settlement-only accounts are reported separately rather than mixed into Social-Household statistics.",
    ]
    (OUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    flags = {
        "verdict": classification,
        "snapshot_week": OBSERVATION_WEEKS,
        "social_household_scope_only": True,
        "genealogy_source_authoritative": True,
        "initial_genealogy_missing_lower_bound_warning": True,
        "ledger_transfers_executed": False,
        "private_family_support_changed": False,
        "economic_behavior_changed": False,
        "rng_semantics_changed": False,
        "new_rng_mechanism": False,
        "pension_implemented": False,
        "required_reserves": list(RESERVES),
        "required_recipient_targets": list(TARGETS),
        "family_edge_count": coverage["distinct_social_household_edge_count"],
        "parent_to_child_capacity_max": parent_max,
        "child_to_parent_capacity_max": child_max,
        "global_max_coverage": max(global_coverage.values(), default=0.0),
        "family_network_max_coverage": max(family_coverage.values(), default=0.0),
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
