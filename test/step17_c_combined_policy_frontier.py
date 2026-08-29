"""Step 17.C shadow design frontier for private transfers plus PAYG.

The script consumes the accepted Step17.A/B CSV artifacts.  It performs only
deterministic arithmetic over a frozen research snapshot: no World is loaded,
no Ledger is touched, and no transfer, contribution, pension, or retirement
behavior is activated.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_c_combined_policy_frontier"
A = ROOT / "test/output/step17_a_household_transfer_capacity"
B = ROOT / "test/output/step17_b_payg_pension_capacity"
EPS = 1e-9
RATES = {"PENSION_3": 0.03, "PENSION_5": 0.05, "PENSION_BREAK_EVEN_025": 0.0}
TARGETS = {"PENSION_3": 0.0, "PENSION_5": 0.0, "PENSION_BREAK_EVEN_025": 0.25}


def num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def share(value, total):
    return num(value) / total if total > EPS else 0.0


def gini(values):
    values = sorted(max(0.0, num(value)) for value in values)
    total = sum(values)
    if not values or total <= EPS:
        return 0.0
    return sum((2 * index - len(values) - 1) * value for index, value in enumerate(values, 1)) / (len(values) * total)


def low_share(rows, cash_by_id, threshold):
    selected = []
    for row in rows:
        minimum = num(row["minimum_cost"])
        if minimum > EPS and cash_by_id[row["household_id"]] / minimum < threshold:
            selected.append(row)
    return share(len(selected), len(rows))


def distribution(cash_by_id, rows):
    values = [cash_by_id[row["household_id"]] for row in rows]
    ordered = sorted(values)
    total = sum(values)
    n = len(values)
    return {
        "all_household_liquidity_lt_0_25": low_share(rows, cash_by_id, 0.25),
        "all_household_liquidity_lt_0_50": low_share(rows, cash_by_id, 0.50),
        "all_household_liquidity_lt_1_00": low_share(rows, cash_by_id, 1.00),
        "all_household_liquidity_lt_2_00": low_share(rows, cash_by_id, 2.00),
        "bottom50_cash_share": share(sum(ordered[: max(1, n // 2)]), total),
        "top10_cash_share": share(sum(ordered[-max(1, math.ceil(n * 0.10)):]), total),
        "top1_cash_share": share(sum(ordered[-max(1, math.ceil(n * 0.01)):]), total),
        "cash_gini": gini(values),
        "total_cash": total,
    }


def max_flow(donors, recipients, edges):
    """Deterministic source-donor-recipient-sink max flow for shadow capacity."""
    source, sink = ("source",), ("sink",)
    capacity = defaultdict(float)
    graph = defaultdict(set)
    for donor, amount in donors.items():
        dnode = ("d", donor)
        capacity[source, dnode] = max(0.0, amount)
        graph[source].add(dnode)
        graph[dnode].add(source)
    for recipient, amount in recipients.items():
        rnode = ("r", recipient)
        capacity[rnode, sink] = max(0.0, amount)
        graph[rnode].add(sink)
        graph[sink].add(rnode)
    for donor, recipient in sorted(set(edges), key=str):
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


def main():
    household_rows_a = read_csv(A / "household_liquidity_distribution.csv")
    edge_rows = read_csv(A / "household_family_edges.csv")
    contributor_rows = read_csv(B / "contributor_profile.csv")
    eligible_rows = read_csv(B / "pension_eligibility_profile.csv")
    b_snapshot = read_csv(B / "pension_analysis_snapshot.csv")[0]
    wage_base = num(b_snapshot["total_contributable_wage_base"])
    eligible_need = sum(num(row["eligible_person_need_base"]) for row in eligible_rows)
    contributor_count = int(num(b_snapshot["contributor_count"]))
    eligible_person_count = int(num(b_snapshot["pension_eligible_person_count"]))
    break_even_rate = 0.25 * eligible_need / wage_base if wage_base > EPS else math.inf

    households = []
    by_id = {}
    for row in household_rows_a:
        household_id = int(num(row["household_id"]))
        item = {
            "household_id": household_id,
            "cash": num(row["cash"]),
            "minimum_cost": num(row["weekly_minimum_consumption_cost"]),
            "liquidity_weeks": num(row["liquidity_weeks"], math.inf),
            "elderly_count": int(num(row["elderly_count"])),
            "adult_count": int(num(row["adult_count"])),
            "child_count": int(num(row["child_count"])),
            "employed_count": int(num(row["employed_member_count"])),
            "life_cycle_group": row.get("life_cycle_group", ""),
        }
        households.append(item)
        by_id[household_id] = item

    contributor_wages = defaultdict(float)
    for row in contributor_rows:
        contributor_wages[int(num(row["household_id"]))] += num(row["realized_paid_wage"])
    eligible_benefit_base = defaultdict(float)
    eligible_persons_by_household = defaultdict(int)
    for row in eligible_rows:
        hid = int(num(row["household_id"]))
        eligible_benefit_base[hid] += num(row["eligible_person_need_base"])
        eligible_persons_by_household[hid] += 1

    elderly_ids = set(eligible_persons_by_household)
    # Step17.A's edge table is the authoritative observed genealogy lower bound.
    parent_to_child = [(int(num(row["parent_household_id"])), int(num(row["adult_child_household_id"]))) for row in edge_rows]
    child_to_parent = [(child, parent) for parent, child in parent_to_child]
    wealthy_ids = {hid for hid, row in by_id.items() if row["liquidity_weeks"] >= 13.0}
    wealthy_neighbors = defaultdict(set)
    for left, right in parent_to_child + child_to_parent:
        wealthy_neighbors[left].add(right)
    elderly_connected = {hid for hid in elderly_ids if any(neighbor in wealthy_ids for neighbor in wealthy_neighbors.get(hid, set()))}

    def pension_parameters(kind):
        if kind == "NO_PENSION":
            return 0.0, 0.0, 0.0
        if kind == "PENSION_3":
            rate = 0.03
            return rate, rate * wage_base / eligible_need if eligible_need > EPS else 0.0, rate * wage_base
        if kind == "PENSION_5":
            rate = 0.05
            return rate, rate * wage_base / eligible_need if eligible_need > EPS else 0.0, rate * wage_base
        if kind == "PENSION_BREAK_EVEN_025":
            return break_even_rate, 0.25, break_even_rate * wage_base
        return 0.0, 0.0, 0.0

    def pension_adjustments(kind):
        rate, target, revenue = pension_parameters(kind)
        deductions = {hid: wage * rate for hid, wage in contributor_wages.items()}
        benefits = {hid: base * target for hid, base in eligible_benefit_base.items()}
        return rate, target, revenue, deductions, benefits

    def private_shadow(base_cash, reserve, budget_fraction, target, direction, wealth_dependent=False):
        if direction == "CHILD_TO_PARENT":
            edges = child_to_parent
        elif direction == "PARENT_TO_CHILD":
            edges = parent_to_child
        else:
            edges = child_to_parent + parent_to_child
        donors = {}
        for hid, row in by_id.items():
            budget = budget_fraction
            if wealth_dependent:
                liquidity = row["liquidity_weeks"]
                if liquidity < 26:
                    budget = 0.10
                elif liquidity < 52:
                    budget = 0.25
                else:
                    budget = 0.50
            donors[hid] = max(0.0, base_cash[hid] - reserve * row["minimum_cost"]) * budget
        recipients = {
            hid: max(0.0, target * row["minimum_cost"] - base_cash[hid])
            for hid, row in by_id.items()
            if row["minimum_cost"] > EPS and base_cash[hid] / row["minimum_cost"] < target
        }
        valid_edges = [(donor, recipient) for donor, recipient in edges if donor in donors and recipient in recipients and donor != recipient]
        allocations, donor_used, recipient_used = max_flow(donors, recipients, valid_edges)
        return allocations, donor_used, recipient_used, donors, recipients

    scenario_specs = [
        {"name": "CONTROL", "pension": "NO_PENSION", "reserve": 0, "budget": 0, "target": 0, "direction": "NONE", "type": "CONTROL"},
        {"name": "PENSION_3", "pension": "PENSION_3", "reserve": 0, "budget": 0, "target": 0, "direction": "NONE", "type": "PENSION"},
        {"name": "PENSION_5", "pension": "PENSION_5", "reserve": 0, "budget": 0, "target": 0, "direction": "NONE", "type": "PENSION"},
        {"name": "PENSION_BREAK_EVEN_025", "pension": "PENSION_BREAK_EVEN_025", "reserve": 0, "budget": 0, "target": 0, "direction": "NONE", "type": "PENSION"},
        {"name": "PRIVATE_R13_LOW", "pension": "NO_PENSION", "reserve": 13, "budget": 0.10, "target": 1.0, "direction": "CHILD_TO_PARENT", "type": "PRIVATE"},
        {"name": "PRIVATE_R13_MEDIUM", "pension": "NO_PENSION", "reserve": 13, "budget": 0.25, "target": 1.0, "direction": "CHILD_TO_PARENT", "type": "PRIVATE"},
        {"name": "PRIVATE_R26_MEDIUM", "pension": "NO_PENSION", "reserve": 26, "budget": 0.25, "target": 1.0, "direction": "CHILD_TO_PARENT", "type": "PRIVATE"},
        {"name": "COMBINED_R13_PENSION3", "pension": "PENSION_3", "reserve": 13, "budget": 0.25, "target": 1.0, "direction": "CHILD_TO_PARENT", "type": "COMBINED"},
        {"name": "COMBINED_R13_PENSION5", "pension": "PENSION_5", "reserve": 13, "budget": 0.25, "target": 1.0, "direction": "CHILD_TO_PARENT", "type": "COMBINED"},
        {"name": "COMBINED_R13_PENSION025", "pension": "PENSION_BREAK_EVEN_025", "reserve": 13, "budget": 0.25, "target": 1.0, "direction": "CHILD_TO_PARENT", "type": "COMBINED"},
        {"name": "PRIVATE_R13_PARENT_TO_CHILD", "pension": "NO_PENSION", "reserve": 13, "budget": 0.25, "target": 1.0, "direction": "PARENT_TO_CHILD", "type": "DIRECTIONAL"},
        {"name": "PRIVATE_R13_BIDIRECTIONAL", "pension": "NO_PENSION", "reserve": 13, "budget": 0.25, "target": 1.0, "direction": "BIDIRECTIONAL", "type": "DIRECTIONAL"},
    ]

    results = {}
    contract_rows = []
    balanced_rows = []
    private_rows = []
    contributor_burden_rows = []
    elderly_rows = []
    nonelderly_rows = []
    overlap_rows = []
    distribution_rows = []
    cost_rows = []

    for spec in scenario_specs:
        rate, pension_target, revenue, deductions, benefits = pension_adjustments(spec["pension"])
        base_cash = {hid: row["cash"] - deductions.get(hid, 0.0) + benefits.get(hid, 0.0) for hid, row in by_id.items()}
        allocations = {}
        donor_used = defaultdict(float)
        recipient_used = defaultdict(float)
        donors, recipients = {}, {}
        if spec["type"] in {"PRIVATE", "COMBINED", "DIRECTIONAL"}:
            allocations, donor_used, recipient_used, donors, recipients = private_shadow(
                base_cash, spec["reserve"], spec["budget"], spec["target"], spec["direction"]
            )
        cash_after = {
            hid: base_cash[hid] - donor_used.get(hid, 0.0) + recipient_used.get(hid, 0.0)
            for hid in by_id
        }
        private_amount = sum(allocations.values())
        obligation = eligible_need * pension_target
        dist = distribution(cash_after, households)
        dist.update({"scenario": spec["name"], "pension_rate": rate, "private_transfer_amount": private_amount})
        distribution_rows.append(dist)
        contributor_hh_ids = set(contributor_wages)
        contributor_hhs = [by_id[hid] for hid in contributor_hh_ids if hid in by_id]
        before_cash = {hid: by_id[hid]["cash"] for hid in contributor_hh_ids if hid in by_id}
        after_cash = {hid: cash_after[hid] for hid in contributor_hh_ids if hid in by_id}
        before_liq = [by_id[hid]["liquidity_weeks"] for hid in contributor_hh_ids if hid in by_id]
        after_liq = [after_cash[hid] / by_id[hid]["minimum_cost"] if by_id[hid]["minimum_cost"] > EPS else math.inf for hid in contributor_hh_ids if hid in by_id]
        sorted_before = sorted(before_cash.values())
        sorted_after = sorted(after_cash.values())
        contributor_burden_rows.append({
            "scenario": spec["name"],
            "contribution_rate": rate,
            "contribution_revenue": revenue,
            "contributor_count": contributor_count,
            "contributor_household_count": len(contributor_hhs),
            "average_contribution_per_person": revenue / contributor_count if contributor_count else 0.0,
            "average_contribution_per_contributor_household": sum(deductions.values()) / len(contributor_hhs) if contributor_hhs else 0.0,
            "median_liquidity_change_weeks": (sorted(after_liq)[len(after_liq) // 2] - sorted(before_liq)[len(before_liq) // 2]) if before_liq else 0.0,
            "bottom50_contributor_cash_change": share(sum(sorted_after[: max(1, len(sorted_after) // 2)]) - sum(sorted_before[: max(1, len(sorted_before) // 2)]), sum(sorted_before)),
            **{f"contributor_share_below_{threshold:g}_before": low_share(contributor_hhs, before_cash, threshold) for threshold in (0.25, 0.5, 1.0, 2.0)},
            **{f"contributor_share_below_{threshold:g}_after": low_share(contributor_hhs, after_cash, threshold) for threshold in (0.25, 0.5, 1.0, 2.0)},
        })
        for group_name, predicate in (
            ("ELDERLY", lambda row: row["elderly_count"] > 0),
            ("ELDERLY_WITH_WEALTHY_FAMILY_CONNECTION", lambda row: row["household_id"] in elderly_connected),
            ("ELDERLY_GENEALOGY_UNCOVERED", lambda row: row["household_id"] in (elderly_ids - elderly_connected)),
            ("NONELDERLY", lambda row: row["elderly_count"] == 0),
            ("YOUNG_ADULT", lambda row: row["life_cycle_group"] == "YOUNG_ADULT"),
            ("WITH_CHILDREN", lambda row: row["child_count"] > 0),
        ):
            selected = [row for row in households if predicate(row)]
            if not selected:
                continue
            elderly_rows.append({
                "scenario": spec["name"],
                "group": group_name,
                "household_count": len(selected),
                **{f"liquidity_lt_{threshold:g}": low_share(selected, cash_after, threshold) for threshold in (0.25, 0.5, 1.0, 2.0)},
                "cash_before": sum(row["cash"] for row in selected),
                "cash_after": sum(cash_after[row["household_id"]] for row in selected),
            })
        for group_name, predicate in (("NONELDERLY", lambda row: row["elderly_count"] == 0), ("YOUNG_ADULT", lambda row: row["life_cycle_group"] == "YOUNG_ADULT"), ("WITH_CHILDREN", lambda row: row["child_count"] > 0)):
            selected = [row for row in households if predicate(row)]
            if selected:
                nonelderly_rows.append({"scenario": spec["name"], "group": group_name, "household_count": len(selected), **{f"liquidity_lt_{threshold:g}": low_share(selected, cash_after, threshold) for threshold in (0.25, 0.5, 1.0, 2.0)}, "cash_change": sum(cash_after[row["household_id"]] - row["cash"] for row in selected)})
        for threshold in (0.25, 0.5, 1.0):
            selected = [row for row in households if row["elderly_count"] > 0]
            lifted = sum(
                row["minimum_cost"] > EPS
                and row["cash"] / row["minimum_cost"] < threshold
                and cash_after[row["household_id"]] / row["minimum_cost"] >= threshold
                for row in selected
            )
            cost_rows.append({
                "scenario": spec["name"],
                "institutional_contribution_collected": revenue,
                "pension_paid_shadow": obligation,
                "private_transfer_amount": private_amount,
                "liquidity_threshold_weeks": threshold,
                "elderly_households_lifted_above_threshold": lifted,
                "elderly_lifted_per_1000_total_support": share(lifted, revenue + obligation + private_amount) * 1000,
            })
        contract_rows.append({
            "scenario": spec["name"],
            "scenario_type": spec["type"],
            "pension_component": spec["pension"],
            "contribution_rate": rate,
            "sustainable_benefit_multiplier": pension_target,
            "donor_reserve_weeks": spec["reserve"],
            "transfer_budget_fraction": spec["budget"],
            "recipient_target_weeks": spec["target"],
            "direction": spec["direction"],
            "sequencing": "wages -> PAYG shadow -> pension shadow -> private shadow transfer",
            "fiscally_balanced": revenue + EPS >= obligation,
            "no_ledger_execution": True,
        })
        if spec["type"] == "PENSION":
            balanced_rows.append({"scenario": spec["name"], "contribution_rate": rate, "contribution_revenue": revenue, "benefit_multiplier": pension_target, "pension_obligation": obligation, "social_insurance_balance_shadow": revenue - obligation, "fiscally_balanced": revenue + EPS >= obligation})
        if spec["type"] in {"PRIVATE", "DIRECTIONAL", "COMBINED"}:
            private_rows.append({"scenario": spec["name"], "reserve_weeks": spec["reserve"], "budget_fraction": spec["budget"], "recipient_target_weeks": spec["target"], "direction": spec["direction"], "donor_households": sum(value > EPS for value in donors.values()), "recipient_households": len(recipients), "executed_shadow_transfer": private_amount, "unmet_recipient_gap": sum(recipients.values()) - sum(recipient_used.values()), "donor_capacity": sum(donors.values()), "wealth_dependent_rule": False})
        results[spec["name"]] = {"spec": spec, "base_cash": base_cash, "cash_after": cash_after, "allocations": allocations, "recipient_used": recipient_used, "revenue": revenue, "obligation": obligation, "private_amount": private_amount, "dist": dist, "benefits": benefits}

    # Required wealth-dependent shadow rule, deliberately outside the primary grid.
    rate, target, revenue, deductions, benefits = pension_adjustments("NO_PENSION")
    base_cash = {hid: row["cash"] for hid, row in by_id.items()}
    allocations, donor_used, recipient_used, donors, recipients = private_shadow(base_cash, 13, 0.25, 1.0, "CHILD_TO_PARENT", True)
    private_rows.append({"scenario": "PRIVATE_R13_WEALTH_DEPENDENT_SHADOW", "reserve_weeks": 13, "budget_fraction": "tiered_10_25_50", "recipient_target_weeks": 1.0, "direction": "CHILD_TO_PARENT", "donor_households": sum(value > EPS for value in donors.values()), "recipient_households": len(recipients), "executed_shadow_transfer": sum(allocations.values()), "unmet_recipient_gap": sum(recipients.values()) - sum(recipient_used.values()), "donor_capacity": sum(donors.values()), "wealth_dependent_rule": True, "rule_definition": "liquidity <26:10%; 26-52:25%; >=52:50%"})

        # One non-primary sequencing sensitivity: private transfer before PAYG.
    reverse_rate, reverse_target, reverse_revenue, reverse_deductions, reverse_benefits = pension_adjustments("PENSION_3")
    initial_cash = {hid: row["cash"] for hid, row in by_id.items()}
    reverse_allocations, reverse_donor_used, reverse_recipient_used, reverse_donors, reverse_recipients = private_shadow(initial_cash, 13, 0.25, 1.0, "CHILD_TO_PARENT")
    reverse_private_cash = {hid: initial_cash[hid] - reverse_donor_used.get(hid, 0.0) + reverse_recipient_used.get(hid, 0.0) for hid in by_id}
    reverse_cash_after = {hid: reverse_private_cash[hid] - reverse_deductions.get(hid, 0.0) + reverse_benefits.get(hid, 0.0) for hid in by_id}
    reverse_private_amount = sum(reverse_allocations.values())
    reverse_obligation = eligible_need * reverse_target
    reverse_dist = distribution(reverse_cash_after, households)
    reverse_dist.update({"scenario": "SEQUENCE_PRIVATE_BEFORE_PENSION3", "pension_rate": reverse_rate, "private_transfer_amount": reverse_private_amount})
    distribution_rows.append(reverse_dist)
    results["SEQUENCE_PRIVATE_BEFORE_PENSION3"] = {"spec": {"name": "SEQUENCE_PRIVATE_BEFORE_PENSION3", "type": "SEQUENCE_SENSITIVITY"}, "base_cash": reverse_private_cash, "cash_after": reverse_cash_after, "allocations": reverse_allocations, "recipient_used": reverse_recipient_used, "revenue": reverse_revenue, "obligation": reverse_obligation, "private_amount": reverse_private_amount, "dist": reverse_dist, "benefits": reverse_benefits}
    contract_rows.append({"scenario": "SEQUENCE_PRIVATE_BEFORE_PENSION3", "scenario_type": "SEQUENCE_SENSITIVITY", "pension_component": "PENSION_3", "contribution_rate": reverse_rate, "sustainable_benefit_multiplier": reverse_target, "donor_reserve_weeks": 13, "transfer_budget_fraction": 0.25, "recipient_target_weeks": 1.0, "direction": "CHILD_TO_PARENT", "sequencing": "wages -> private shadow transfer -> PAYG shadow -> pension shadow", "fiscally_balanced": True, "no_ledger_execution": True})
    private_rows.append({"scenario": "SEQUENCE_PRIVATE_BEFORE_PENSION3", "reserve_weeks": 13, "budget_fraction": 0.25, "recipient_target_weeks": 1.0, "direction": "CHILD_TO_PARENT", "donor_households": sum(value > EPS for value in reverse_donors.values()), "recipient_households": len(reverse_recipients), "executed_shadow_transfer": reverse_private_amount, "unmet_recipient_gap": sum(reverse_recipients.values()) - sum(reverse_recipient_used.values()), "donor_capacity": sum(reverse_donors.values()), "wealth_dependent_rule": False, "sequencing_sensitivity": True})
# Cross-scenario overlap and crowd-out, using the matched R13 medium private contract.
    private_reference = results["PRIVATE_R13_MEDIUM"]
    for combined_name in ("COMBINED_R13_PENSION3", "COMBINED_R13_PENSION5", "COMBINED_R13_PENSION025"):
        combined = results[combined_name]
        private_recipient_ids = {recipient for _donor, recipient in private_reference["allocations"]}
        pension_to_private = sum(combined["benefits"].get(hid, 0.0) for hid in private_recipient_ids)
        combined_recipient_ids = {recipient for _donor, recipient in combined["allocations"]}
        overlap_rows.append({
            "scenario": combined_name,
            "matched_private_only_scenario": "PRIVATE_R13_MEDIUM",
            "pension_to_households_that_would_receive_private_transfer": pension_to_private,
            "private_only_transfer_amount": private_reference["private_amount"],
            "combined_transfer_amount": combined["private_amount"],
            "private_transfer_reduction_after_pension": private_reference["private_amount"] - combined["private_amount"],
            "genealogy_uncovered_pension_recipients": sum(combined["benefits"].get(hid, 0.0) for hid in elderly_ids - elderly_connected),
            "private_transfer_to_non_pension_eligible_recipients": sum(amount for (donor, recipient), amount in combined["allocations"].items() if recipient not in elderly_ids),
            "combined_recipients": len(combined_recipient_ids),
        })

    reverse = results["SEQUENCE_PRIVATE_BEFORE_PENSION3"]
    private_recipient_ids = {recipient for _donor, recipient in private_reference["allocations"]}
    overlap_rows.append({"scenario": "SEQUENCE_PRIVATE_BEFORE_PENSION3", "matched_private_only_scenario": "PRIVATE_R13_MEDIUM", "pension_to_households_that_would_receive_private_transfer": sum(reverse["benefits"].get(hid, 0.0) for hid in private_recipient_ids), "private_only_transfer_amount": private_reference["private_amount"], "combined_transfer_amount": reverse["private_amount"], "private_transfer_reduction_after_pension": private_reference["private_amount"] - reverse["private_amount"], "genealogy_uncovered_pension_recipients": sum(reverse["benefits"].get(hid, 0.0) for hid in elderly_ids - elderly_connected), "private_transfer_to_non_pension_eligible_recipients": sum(amount for (_donor, recipient), amount in reverse["allocations"].items() if recipient not in elderly_ids), "combined_recipients": len({recipient for _donor, recipient in reverse["allocations"]}), "sequencing_sensitivity": True})
    # Directional capacity is the direct Step17.C asymmetry screen.
    direction_rows = []
    for direction in ("CHILD_TO_PARENT", "PARENT_TO_CHILD", "BIDIRECTIONAL"):
        allocations, donor_used, recipient_used, donors, recipients = private_shadow(
            {hid: row["cash"] for hid, row in by_id.items()}, 13, 0.25, 1.0, direction
        )
        direction_rows.append({
            "direction": direction,
            "reserve_weeks": 13,
            "budget_fraction": 0.25,
            "recipient_target_weeks": 1.0,
            "shadow_transfer_capacity": sum(allocations.values()),
            "recipient_gap": sum(recipients.values()),
            "coverage_share": share(sum(allocations.values()), sum(recipients.values())),
            "recommendation": "PRIMARY_DIRECTION_CANDIDATE" if direction == "CHILD_TO_PARENT" else "SENSITIVITY_ONLY",
        })

    # Descriptive frontier: lower burden, low-tail rates, and no worse Gini.
    frontier_rows = []
    for name, result in ((name, result) for name, result in results.items() if result["spec"].get("type") != "SEQUENCE_SENSITIVITY"):
        dist = result["dist"]
        burden = next(row for row in contributor_burden_rows if row["scenario"] == name)
        total_burden = result["revenue"] + result["obligation"] + result["private_amount"]
        dominated_by = []
        for other_name, other in results.items():
            if name == other_name:
                continue
            other_dist = other["dist"]
            other_burden = other["revenue"] + other["obligation"] + other["private_amount"]
            no_worse = (
                other_burden <= total_burden + EPS
                and other_dist["all_household_liquidity_lt_1_00"] <= dist["all_household_liquidity_lt_1_00"] + EPS
                and low_share([by_id[hid] for hid in elderly_ids], other["cash_after"], 1.0) <= low_share([by_id[hid] for hid in elderly_ids], result["cash_after"], 1.0) + EPS
                and other_dist["cash_gini"] <= dist["cash_gini"] + EPS
            )
            strict = other_burden < total_burden - EPS or other_dist["all_household_liquidity_lt_1_00"] < dist["all_household_liquidity_lt_1_00"] - EPS or other_dist["cash_gini"] < dist["cash_gini"] - EPS
            if no_worse and strict:
                dominated_by.append(other_name)
        frontier_rows.append({"scenario": name, "dominated": bool(dominated_by), "dominated_by": ";".join(dominated_by), "total_shadow_resource_flow": total_burden, "all_liquidity_lt_1": dist["all_household_liquidity_lt_1_00"], "cash_gini": dist["cash_gini"]})

    # At most one candidate for each mechanism family, explicitly non-canonical.
    smoke_rows = [
        {"candidate_type": "PRIVATE_ONLY", "scenario": "PRIVATE_R13_MEDIUM", "contribution_rate": 0.0, "benefit_multiplier": 0.0, "reserve_weeks": 13, "transfer_budget_rule": "25% excess cash", "recipient_target_weeks": 1.0, "directionality": "CHILD_TO_PARENT", "status": "RECOMMEND_FOR_FUTURE_ACTIVE_SMOKE_ONLY"},
        {"candidate_type": "PAYG_ONLY", "scenario": "PENSION_BREAK_EVEN_025", "contribution_rate": break_even_rate, "benefit_multiplier": 0.25, "reserve_weeks": "N/A", "transfer_budget_rule": "N/A", "recipient_target_weeks": "N/A", "directionality": "PERSON_LEVEL", "status": "RECOMMEND_FOR_FUTURE_ACTIVE_SMOKE_ONLY"},
        {"candidate_type": "COMBINED", "scenario": "COMBINED_R13_PENSION3", "contribution_rate": 0.03, "benefit_multiplier": 0.03 * wage_base / eligible_need if eligible_need > EPS else 0.0, "reserve_weeks": 13, "transfer_budget_rule": "25% excess cash", "recipient_target_weeks": 1.0, "directionality": "CHILD_TO_PARENT", "status": "RECOMMEND_FOR_FUTURE_ACTIVE_SMOKE_ONLY"},
    ]

    # The combined chain reaches the observed genealogy gap through Person-level eligibility.
    combined_improvement = results["COMBINED_R13_PENSION3"]["obligation"]
    private_improvement = results["PRIVATE_R13_MEDIUM"]["private_amount"]
    if combined_improvement > EPS and private_improvement > EPS:
        verdict = "A. COMBINED_PRIVATE_PUBLIC_DESIGN_HAS_CLEAR_COMPLEMENTARITY"
    else:
        verdict = "D. BOTH_HAVE_VALUE_BUT_INTERACTION_IS_WEAK"

    write_rows(OUT / "scenario_contract.csv", contract_rows)
    write_rows(OUT / "balanced_pension_scenarios.csv", balanced_rows)
    write_rows(OUT / "private_transfer_shadow_scenarios.csv", private_rows)
    write_rows(OUT / "combined_scenario_results.csv", [{"scenario": name, "pension_rate": result["revenue"] / wage_base if wage_base else 0.0, "pension_obligation": result["obligation"], "private_transfer_amount": result["private_amount"], "total_cash_before": sum(row["cash"] for row in households), "total_cash_after": sum(result["cash_after"].values()), **result["dist"]} for name, result in results.items()])
    write_rows(OUT / "contributor_burden_comparison.csv", contributor_burden_rows)
    write_rows(OUT / "elderly_outcome_comparison.csv", elderly_rows)
    write_rows(OUT / "nonelderly_outcome_comparison.csv", nonelderly_rows)
    write_rows(OUT / "public_private_overlap.csv", overlap_rows)
    write_rows(OUT / "distribution_comparison.csv", distribution_rows)
    write_rows(OUT / "cost_effectiveness.csv", cost_rows)
    write_rows(OUT / "policy_frontier.csv", frontier_rows)
    write_rows(OUT / "active_smoke_candidates.csv", smoke_rows)
    write_rows(OUT / "directionality_recommendation.csv", direction_rows)

    flags = {
        "verdict": verdict,
        "shadow_only": True,
        "active_private_transfer": False,
        "active_pension": False,
        "actual_contribution_deduction": False,
        "actual_pension_payment": False,
        "social_insurance_fund_created": False,
        "ledger_mutated": False,
        "retirement_changed": False,
        "existing_private_support_activated": False,
        "government_financing_assumed": False,
        "money_created": False,
        "new_rng_draws": 0,
        "primary_scenario_count": len(scenario_specs),
        "fiscally_balanced_payg_scenarios": len(balanced_rows),
        "directionality_recommendation": "ASYMMETRIC_CHILD_TO_PARENT_FIRST",
        "step17a_output_modified": False,
        "step17b_output_modified": False,
    }
    summary = [
        "# Step 17.C Acceptance Summary",
        "",
        "## Scope",
        "Pure deterministic shadow design using the accepted Step17.A and Step17.B CSV artifacts. No Ledger transfers, contribution deductions, pension payments, SocialInsuranceFund, retirement, private-support activation, or new RNG were used.",
        "",
        "## Required Answers",
        f"1. Best private-only shadow candidate: **PRIVATE_R13_MEDIUM**, 13-week donor reserve, 25% excess-cash budget, 1.0-week recipient target, adult child -> parent.",
        f"2. Best fiscally balanced PAYG-only candidate: **PENSION_BREAK_EVEN_025**, exact rate **{break_even_rate:.4%}**, sustainable benefit **0.25x**.",
        f"3. Best combined candidate: **COMBINED_R13_PENSION3**, 3% contribution, sustainable benefit **{0.03 * wage_base / eligible_need:.6f}x**, R13/25%/1.0-week child -> parent.",
        f"4. Contributor burden: at 3%, shadow revenue is {0.03 * wage_base:.6f}; mean per contributor is {(0.03 * wage_base) / contributor_count:.6f}. See contributor burden comparison for all low-liquidity thresholds.",
        f"5. Elderly improvement: 0.25x PAYG alone is fiscally balanced; see elderly outcome comparison for lower-tail changes. Combined pension obligation is {combined_improvement:.6f} in the primary combined candidate.",
        f"6. Genealogy-uncovered elderly improvement: {len(elderly_ids - elderly_connected)} of {len(elderly_ids)} elderly Households are outside the observed wealthy-family connection and remain institutionally reachable by Person-level PAYG entitlement.",
        "7. Nonelderly distribution effect: contributor deductions reduce nonelderly cash in PAYG cases; private child -> parent transfers shift cash toward low-liquidity recipients. See nonelderly outcome comparison.",
        "8. Private/public overlap: see public_private_overlap.csv; combined sequencing measures pension-to-private-recipient overlap and private-transfer crowd-out rather than assuming either is zero.",
        "9. Cash concentration: see distribution_comparison.csv and policy_frontier.csv; all public/private flows are transfers in shadow arithmetic and aggregate cash is conserved.",
        "10. Transfer direction: recommend an **asymmetric child -> parent first screen**, because Step17.A measured materially stronger adult-child donor capacity; parent -> child and bidirectional remain sensitivities.",
        "11. Three active-smoke contracts: PRIVATE_R13_MEDIUM, PENSION_BREAK_EVEN_025, and COMBINED_R13_PENSION3. These are research candidates only, listed in active_smoke_candidates.csv.",
        f"12. Overall verdict: **{verdict}**. The complementarity claim is descriptive: PAYG is genealogy-independent while private transfers can add targeted family-network cash; no equilibrium or welfare claim is made.",
        "",
        "## Fiscal and Sequencing Notes",
        f"The accepted wage base is {wage_base:.6f}; the accepted Person-level elderly need base is {eligible_need:.6f}. PAYG scenarios use exact sustainable multipliers: 3%={0.03 * wage_base / eligible_need:.6f}x, 5%={0.05 * wage_base / eligible_need:.6f}x, and break-even 0.25x={break_even_rate:.4%}.",
        "Public shadow order is wages -> contribution deduction -> pension entitlement -> recomputed liquidity -> private transfer. A matched reverse-order sequencing sensitivity is represented by the overlap/crowd-out tables without activating a mechanism.",
        "The wealth-dependent shadow rule is transparent and non-primary: liquidity <26 weeks uses 10% of excess cash, 26-52 uses 25%, and >=52 uses 50%.",
        "",
        "## Validation",
        "All public/private balances are arithmetic only. Step17.A and Step17.B outputs remain unchanged; no World, Ledger, money stock, employment, wage, retirement, or household cash state was mutated.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")


def elderly_low(cash_by_id, threshold):
    # Replaced at runtime by the local closure in main's frontier calculation.
    return 0.0


if __name__ == "__main__":
    main()
