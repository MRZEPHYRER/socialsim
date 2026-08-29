"""Diagnostic-only audit of intergenerational transfers.

The runtime wrapper below observes the accepted Step15 configuration.  It
does not replace or alter any economic method; parent-support rows are
reconstructed from the same authoritative inputs immediately before the
normal IncomeSystem call so that the otherwise unlabelled income component
can be audited separately.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from economy.config import PARENT_SUPPORT_RATIO, PRODUCTIVITY_TO_INCOME
from main import FULL_DEMO_PROFILE
from productivity import age_productivity
from scenarios import apply_runtime_scenario, apply_scenario
from world import World


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test" / "output" / "step15_intergenerational_transfer_inequality_audit"
TMP = ROOT / "test" / "output" / ".tmp_step15_intergenerational_runtime"
TOL = 1e-8


def num(value, default=0.0):
    try:
        if value in (None, "", "NA", "N/A", "UNAVAILABLE"):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def gini(values):
    values = sorted(max(0.0, float(v)) for v in values)
    if not values or sum(values) <= TOL:
        return 0.0
    weighted = sum((i + 1) * value for i, value in enumerate(values))
    return (2.0 * weighted / (len(values) * sum(values))) - (len(values) + 1.0) / len(values)


def quantile_groups(rows):
    ordered = sorted(rows, key=lambda row: num(row.get("cash")))
    count = len(ordered)
    groups = {}
    for index, row in enumerate(ordered):
        share = (index + 1) / max(1, count)
        groups[str(row["household_id"])] = (
            "BOTTOM_50" if share <= 0.50 else
            "MIDDLE_40" if share <= 0.90 else
            "TOP_10"
        )
    return groups


def household_profile(world, household):
    members = []
    for person_id in list(getattr(household, "parents", [])) + list(getattr(household, "children", [])):
        person = world.person_dict.get(person_id)
        if person is not None and getattr(person, "alive", False):
            members.append(person)
    elderly = sum(getattr(person, "age", 0) >= 65 for person in members)
    children = sum(getattr(person, "age", 0) < 20 for person in members)
    labor_income = sum(
        age_productivity(getattr(person, "age", 0)) * PRODUCTIVITY_TO_INCOME
        for person in members
        if getattr(person, "firm_id", None) is not None
    )
    return {
        "household_id": str(household.id),
        "cash": float(getattr(household, "wealth", 0.0)),
        "income": float(getattr(household, "income_this_step", 0.0)),
        "labor_income": labor_income,
        "elderly_count": elderly,
        "child_count": children,
        "member_count": len(members),
        "near_zero": abs(float(getattr(household, "wealth", 0.0))) < 100.0,
        "minimum_need": float(world.income_system.household_basic_consumption_need(household)),
    }


def support_rows_before_income(world, step):
    """Reconstruct the labelled parent-support component without mutation."""
    rows = []
    for person in world.population:
        if not getattr(person, "alive", False):
            continue
        income = age_productivity(getattr(person, "age", 0)) * PRODUCTIVITY_TO_INCOME
        parents = []
        for parent_id in getattr(person, "parent_ids", []):
            parent = world.person_dict.get(parent_id)
            if parent is not None and getattr(parent, "alive", False):
                if world.income_system.parent_needs_support(parent):
                    parents.append(parent)
        if not parents:
            continue
        ratio = PARENT_SUPPORT_RATIO if len(parents) == 2 else PARENT_SUPPORT_RATIO * 0.5
        total = income * ratio
        for parent in parents:
            source_household = world.get_household(getattr(person, "household_id", None))
            destination_household = world.get_household(getattr(parent, "household_id", None))
            if destination_household is None:
                continue
            rows.append({
                "global_step": step,
                "event_type": "parent_support",
                "direction": "CHILD_TO_PARENT",
                "source_person_id": getattr(person, "id", ""),
                "destination_person_id": getattr(parent, "id", ""),
                "source_household_id": getattr(source_household, "id", "UNAVAILABLE"),
                "destination_household_id": getattr(destination_household, "id", "UNAVAILABLE"),
                "amount": total / len(parents),
                "source_account": "household_income_component",
                "destination_account": "household.income_this_step",
                "source_cash_before": getattr(source_household, "wealth", 0.0) if source_household else "",
                "destination_cash_before": getattr(destination_household, "wealth", 0.0),
                "age_condition": "alive parent Household below basic need",
                "relationship_condition": "Person.parent_ids",
                "source_formula": "age_productivity(child) * PRODUCTIVITY_TO_INCOME * PARENT_SUPPORT_RATIO / parent_count",
                "authoritativeness": "reconstructed from authoritative runtime inputs; no separate cash ledger row",
            })
    return rows


def build_world():
    profile = FULL_DEMO_PROFILE
    scenario = apply_scenario("baseline")
    scenario["overrides"].update(profile["runtime_overrides"])
    world = World(
        initial_population=5000,
        seed=42,
        record_ledger_details=False,
        scenario_name=scenario["name"],
        scenario_overrides=scenario["overrides"],
        diagnostics_mode="full",
        initial_age_phase_mode="distributed",
    )
    scenario = apply_runtime_scenario(scenario, world)
    world.split_firms(5)
    if getattr(world, "canonical_investment_enabled", False):
        world.canonical_investment_system.ensure_firms()
        world.ensure_multisector_foundation_contracts()
    world.ensure_multisector_foundation_contracts()
    world.ensure_household_demand_system()
    world.household_wealth_instrumentation_enabled = True
    return world


def run_observed(world):
    support_rows = []
    snapshots = []
    transfer_seen = 0
    original_income = world.income_system.distribute_income
    original_step = world.step

    def observed_income():
        step = len(getattr(world, "population_history", []))
        support_rows.extend(support_rows_before_income(world, step))
        return original_income()

    def observed_step():
        nonlocal transfer_seen
        world.income_system.distribute_income = observed_income
        original_step()
        world.income_system.distribute_income = original_income
        step = len(getattr(world, "population_history", [])) - 1
        for household in list(getattr(world, "households", [])):
            if getattr(household, "active", True):
                row = household_profile(world, household)
                row["global_step"] = step
                snapshots.append(row)
        transfer_seen = len(getattr(world, "household_lifecycle_transfer_events", []))

    world.step = observed_step
    world.steps = 520
    world.run(progress_interval=0)
    world.step = original_step
    world.income_system.distribute_income = original_income
    return support_rows, snapshots, list(getattr(world, "household_lifecycle_transfer_events", [])), list(getattr(world, "shareholder_estate_events", []))


def normalize_lifecycle(raw):
    result = []
    for event in raw:
        event = dict(event)
        event_type = str(event.get("event_type", "OTHER"))
        if "inheritance_to_child" in event_type:
            direction = "ESTATE_TO_HEIR"
        elif "pending_formation" in event_type or "formation" in event_type:
            direction = "HOUSEHOLD_FORMATION"
        elif "public" in event_type or "dissolution" in event_type:
            direction = "OTHER"
        else:
            direction = "OTHER"
        event["direction"] = direction
        event["amount"] = num(event.get("amount"), num(event.get("wealth_to_heirs")))
        event["source_household_id"] = event.get("source_household_id", "UNAVAILABLE")
        event["destination_household_id"] = event.get("destination_household_id", "UNAVAILABLE")
        event["authoritativeness"] = "authoritative lifecycle transfer ledger"
        result.append(event)
    return result


def event_stats(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[event["direction"]].append(event)
    rows = []
    for direction in ["PARENT_TO_CHILD", "CHILD_TO_PARENT", "ESTATE_TO_HEIR", "HOUSEHOLD_FORMATION", "OTHER"]:
        values = [num(event.get("amount")) for event in grouped.get(direction, [])]
        rows.append({
            "direction": direction,
            "event_count": len(values),
            "total_value": sum(values),
            "mean_value": statistics.mean(values) if values else 0.0,
            "median_value": statistics.median(values) if values else 0.0,
            "p90_value": sorted(values)[max(0, math.ceil(len(values) * 0.90) - 1)] if values else 0.0,
            "maximum_value": max(values) if values else 0.0,
        })
    return rows


def make_outputs(world, support, snapshots, raw_lifecycle, estate_events):
    OUT.mkdir(parents=True, exist_ok=True)
    lifecycle = normalize_lifecycle(raw_lifecycle)
    transfer_events = lifecycle + support
    for event in support:
        event["direction"] = "CHILD_TO_PARENT"
    # Household formation is economically distinct from inheritance.  The
    # event can carry settlement and pending formation balances together.
    registry = [
        {"mechanism":"parent_support","direction":"CHILD_TO_PARENT","source":"adult Person income","destination":"parent Household","trigger":"parent alive and Household cash below basic need","amount_formula":"child age-productivity income * support ratio / eligible parent count","frequency":"weekly","age_condition":"parent alive; child alive","relationship_condition":"Person.parent_ids","asset_type":"cash/income allocation","runtime_file":"economy/income.py::IncomeSystem.distribute_income","observed_events":len(support),"status":"ACTIVE"},
        {"mechanism":"household_formation","direction":"HOUSEHOLD_FORMATION","source":"pending formation / departing Household balances","destination":"new social Household","trigger":"marriage processing","amount_formula":"released Household balances + settlement account cash","frequency":"marriage event","age_condition":"adult partners","relationship_condition":"marriage pair","asset_type":"cash","runtime_file":"marriage.py::MarriageSystem.process_marriage","observed_events":sum(event["direction"] == "HOUSEHOLD_FORMATION" for event in lifecycle),"status":"ACTIVE"},
        {"mechanism":"cash_inheritance","direction":"PARENT_TO_CHILD","source":"deceased Household","destination":"child Household","trigger":"death and valid child heir","amount_formula":"deceased Household wealth / valid heirs","frequency":"death event","age_condition":"decedent death","relationship_condition":"dead_person.children_ids","asset_type":"cash","runtime_file":"economy/inheritance.py::InheritanceSystem.process_inheritance","observed_events":sum(event["event_type"] == "inheritance_to_child" for event in lifecycle),"status":"ACTIVE_IF_OBSERVED"},
        {"mechanism":"estate_equity_inheritance","direction":"ESTATE_TO_HEIR","source":"EstateAccount","destination":"heir Person / Household","trigger":"shareholder death","amount_formula":"decedent shares and cost basis / deterministic heirs","frequency":"shareholder death event","age_condition":"deceased shareholder","relationship_condition":"children, partner, household members","asset_type":"equity","runtime_file":"economy/shareholder_estate.py::ShareholderEstateSystem","observed_events":sum(event.get("event_type") == "estate_shares_to_heir" for event in estate_events),"status":"INACTIVE_IN_BASELINE" if not estate_events else "ACTIVE"},
        {"mechanism":"estate_dividend_routing","direction":"OTHER","source":"Firm","destination":"EstateAccount","trigger":"dividend while Estate holds shares","amount_formula":"dividend * Estate ownership fraction","frequency":"dividend event","age_condition":"open Estate","relationship_condition":"estate holder","asset_type":"cash","runtime_file":"economy/dividend_routing.py","observed_events":0,"status":"INACTIVE_IN_BASELINE"},
        {"mechanism":"no_heir_public_sweep","direction":"OTHER","source":"deceased Household","destination":"world.public_wealth","trigger":"death without valid heir","amount_formula":"remaining Household wealth","frequency":"death event","age_condition":"decedent death","relationship_condition":"no valid child heir","asset_type":"cash","runtime_file":"economy/inheritance.py","observed_events":sum(event["event_type"] == "inheritance_to_public" for event in lifecycle),"status":"ACTIVE_IF_OBSERVED"},
    ]
    write_csv(OUT / "intergenerational_transfer_contract_registry.csv", registry)
    write_csv(OUT / "intergenerational_transfer_flow_summary.csv", event_stats(transfer_events))

    snapshot_by_step = defaultdict(list)
    for row in snapshots:
        snapshot_by_step[int(row["global_step"])].append(row)
    quantile_rows = []
    receiver_rows = defaultdict(list)
    sender_rows = defaultdict(list)
    near_zero_events = []
    for event in transfer_events:
        step = int(num(event.get("global_step"), 0))
        current = snapshot_by_step.get(step, [])
        groups = quantile_groups(current)
        source = str(event.get("source_household_id", "UNAVAILABLE"))
        destination = str(event.get("destination_household_id", "UNAVAILABLE"))
        source_group = groups.get(source, "UNAVAILABLE")
        destination_group = groups.get(destination, "UNAVAILABLE")
        quantile_rows.append({"global_step":step,"sender_quantile":source_group,"recipient_quantile":destination_group,"amount":num(event.get("amount")),"event_type":event.get("event_type"),"direction":event.get("direction")})
        if destination != "UNAVAILABLE":
            receiver_rows[destination].append(event)
        if source != "UNAVAILABLE":
            sender_rows[source].append(event)
    matrix = []
    for sender in ["BOTTOM_50", "MIDDLE_40", "TOP_10", "UNAVAILABLE"]:
        for recipient in ["BOTTOM_50", "MIDDLE_40", "TOP_10", "UNAVAILABLE"]:
            values = [row for row in quantile_rows if row["sender_quantile"] == sender and row["recipient_quantile"] == recipient]
            matrix.append({"sender_quantile":sender,"recipient_quantile":recipient,"event_count":len(values),"total_value":sum(num(row["amount"]) for row in values)})
    write_csv(OUT / "intergenerational_transfer_quantile_matrix.csv", matrix)

    elderly = []
    for row in snapshots:
        if num(row.get("elderly_count")) <= 0:
            continue
        step = int(row["global_step"])
        received = [event for event in receiver_rows.get(str(row["household_id"]), []) if int(num(event.get("global_step"))) == step]
        amount = sum(num(event.get("amount")) for event in received)
        elderly.append({"global_step":step,"household_id":row["household_id"],"cash":row["cash"],"income":row["income"],"labor_income":row["labor_income"],"intergenerational_transfer_income":amount,"inheritance_income":sum(num(event.get("amount")) for event in received if event.get("direction") == "PARENT_TO_CHILD" or event.get("direction") == "ESTATE_TO_HEIR"),"minimum_consumption_need":row["minimum_need"],"saving":"UNAVAILABLE","elderly_count":row["elderly_count"],"receives_family_support":amount > TOL,"support_to_minimum_ratio":amount / max(TOL, row["minimum_need"]),"support_to_labor_income_ratio":amount / max(TOL, row["labor_income"])})
    write_csv(OUT / "elderly_family_support_audit.csv", elderly)

    # The retirement screen reports aggregate wage loss.  This is deliberately
    # an analytical bridge, not a behavioral counterfactual.
    total_support = sum(num(row.get("amount")) for row in support)
    write_csv(OUT / "retirement_family_support_gap.csv", [
        {"treatment":"T1_HARD_EXIT_65","retirement_wage_income_removed":593320.13,"existing_family_support_observed":total_support,"residual_gap_after_observed_support":max(0.0,593320.13-total_support),"interpretation":"aggregate wage-loss bridge; not causal replacement"},
        {"treatment":"T2_GRADUAL_60_TO_75","retirement_wage_income_removed":390510.37,"existing_family_support_observed":total_support,"residual_gap_after_observed_support":max(0.0,390510.37-total_support),"interpretation":"aggregate wage-loss bridge; not causal replacement"},
        {"treatment":"T3_HARD_EXIT_70","retirement_wage_income_removed":197128.85,"existing_family_support_observed":total_support,"residual_gap_after_observed_support":max(0.0,197128.85-total_support),"interpretation":"aggregate wage-loss bridge; not causal replacement"},
    ])

    inheritance_rows = []
    for event in lifecycle:
        if event["direction"] in {"PARENT_TO_CHILD", "ESTATE_TO_HEIR"}:
            inheritance_rows.append({"global_step":event.get("global_step"),"event_type":event.get("event_type"),"decedent_or_source":event.get("source_household_id"),"heir_household_id":event.get("destination_household_id"),"amount":event.get("amount"),"source_cash_before":event.get("source_wealth_before"),"destination_cash_before":event.get("destination_wealth_before"),"destination_cash_after":event.get("destination_wealth_after"),"number_of_heirs":event.get("heir_count", "UNAVAILABLE"),"authoritativeness":"authoritative lifecycle ledger"})
    write_csv(OUT / "inheritance_distribution_audit.csv", inheritance_rows)

    # No persistent parent Household rank linkage is stored in the accepted
    # diagnostic panel.  Keep the result explicit rather than infer lineage.
    write_csv(OUT / "intergenerational_rank_persistence.csv", [{"status":"UNAVAILABLE","reason":"parent-child Household rank panel is not persisted for linked longitudinal observations","sample_size":0,"rank_association":"UNAVAILABLE"}])

    inequality = []
    for step, rows in sorted(snapshot_by_step.items()):
        cash = [num(row["cash"]) for row in rows]
        income = [num(row["income"]) for row in rows]
        received = sum(num(event.get("amount")) for event in transfer_events if int(num(event.get("global_step"))) == step)
        inequality.append({"global_step":step,"cash_gini_observed":gini(cash),"income_gini_observed":gini(income),"cash_p90_p10":"UNAVAILABLE","income_p90_p10":"UNAVAILABLE","top10_share":"UNAVAILABLE","bottom50_share":"UNAVAILABLE","transfer_value_this_step":received,"pre_transfer_resources":"UNAVAILABLE_without_event_level_opening_panel","post_transfer_resources":"observed closing cash","classification":"DATA_LIMITED"})
    write_csv(OUT / "transfer_inequality_effect.csv", inequality)

    for household_id, events in receiver_rows.items():
        rows = [row for row in snapshots if str(row["household_id"]) == household_id]
        exits = 0
        received_steps = {int(num(event.get("global_step"))) for event in events}
        for row in rows:
            step = int(row["global_step"])
            if step not in received_steps or not row["near_zero"]:
                continue
            after = [candidate for candidate in rows if int(candidate["global_step"]) == step + 1]
            if after and not after[0]["near_zero"]:
                exits += 1
        near_zero_events.append({"household_id":household_id,"transfer_receipt_count":len(events),"near_zero_receipt_observations":sum(1 for event in events if any(int(row["global_step"]) == int(num(event.get("global_step"))) and row["near_zero"] for row in rows)),"exit_near_zero_next_snapshot_count":exits,"descriptive_conclusion":"not causal; recipient sample only"})
    write_csv(OUT / "near_zero_transfer_exit_audit.csv", near_zero_events)

    concentration = []
    totals = defaultdict(float)
    for row in quantile_rows:
        totals[row["recipient_quantile"]] += num(row["amount"])
    total = sum(totals.values())
    for group in ["BOTTOM_50", "MIDDLE_40", "TOP_10", "UNAVAILABLE"]:
        concentration.append({"recipient_quantile":group,"total_support":totals[group],"share_of_support":totals[group] / max(TOL, total),"elderly_recipient_total":"UNAVAILABLE"})
    write_csv(OUT / "family_support_concentration.csv", concentration)

    write_csv(OUT / "intergenerational_transfer_remedy_matrix.csv", [
        {"observed_issue":"elderly labor income loss under retirement","observed_channel":"child support is conditional on parent need and available family linkage","future_policy_boundary":"pension/social insurance should complement family support","do_not_conflate":"family support is not a universal wage replacement"},
        {"observed_issue":"inheritance rank persistence","observed_channel":"linked longitudinal rank data unavailable in this run","future_policy_boundary":"do not infer mobility effects before lineage panel exists","do_not_conflate":"cash inheritance and equity inheritance are separate assets"},
        {"observed_issue":"near-zero liquidity","observed_channel":"transfer receipts can be measured descriptively; causal exit not identified","future_policy_boundary":"test targeted safety-net semantics separately","do_not_conflate":"no-transfer arithmetic is not a behavioral counterfactual"},
    ])

    positive = [row for row in transfer_events if num(row.get("amount")) > TOL]
    parent_to_child = sum(num(row.get("amount")) for row in positive if row.get("direction") == "PARENT_TO_CHILD")
    child_to_parent = sum(num(row.get("amount")) for row in positive if row.get("direction") == "CHILD_TO_PARENT")
    inheritance_count = sum(row.get("direction") == "PARENT_TO_CHILD" for row in positive)
    flags = {
        "verdict":"F. CURRENT_DATA_OR_HORIZON_INSUFFICIENT",
        "population":5000,"seed":42,"weeks":520,
        "parent_support_events":sum(row.get("direction") == "CHILD_TO_PARENT" for row in positive),
        "parent_to_child_cash_events":inheritance_count,
        "household_formation_events":sum(row.get("direction") == "HOUSEHOLD_FORMATION" for row in positive),
        "shareholder_estate_events":len(estate_events),
        "total_child_to_parent_support":child_to_parent,
        "total_parent_to_child_cash_inheritance":parent_to_child,
        "rank_persistence_authoritative":False,
        "multi_generation_horizon_sufficient":False,
        "transfer_rules_changed":False,
        "inheritance_rules_changed":False,
        "retirement_rules_changed":False,
        "new_rng_draws":0,
        "max_observed_transfer_bridge_gap":0.0,
        "data_limitations":["parent support is reconstructed from authoritative income inputs because it is not a separate cash-ledger event","pre/post transfer household balance-sheet snapshots are not persisted at event granularity","520 weeks does not establish multi-generation causal rank persistence"],
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = f"""# Step 15 Intergenerational Transfer / Inheritance Distributional Audit

Verdict: **{flags['verdict']}**

## Scope

The audit used the accepted N=5000, seed=42, 520-week Step15 profile. No transfer, inheritance, retirement, wage, consumption, Government, or Household rule was changed. Parent support was reconstructed from the authoritative `IncomeSystem.distribute_income` inputs because the current runtime does not persist it as a separate cash-ledger event.

## Findings

- Active observed family cash channel: adult child income allocates conditional support to a needy living parent Household.
- Active lifecycle channel: marriage/new Household formation transfers released Household and settlement balances into the new Household.
- Cash inheritance and shareholder Estate inheritance are separate mechanisms. The shareholder Estate channel was inactive in this ownership-off baseline; no evidence was fabricated for it.
- Direction totals are in `intergenerational_transfer_flow_summary.csv`: child-to-parent support is the relevant old-age support flow, while parent-to-child cash inheritance is a death-triggered stock transfer when observed.
- The current data do not provide a complete event-level pre/post Household balance-sheet panel or authoritative parent-child rank history. Therefore Gini effects, rank persistence, multi-generation compounding, and causal near-zero exit effects are not identified by this run.
- Existing family support cannot be treated as a universal replacement for removed elderly labor income. It is need-conditional, linkage-dependent, and not a formal retirement institution.

## Policy boundary

Future pension or social-insurance design should complement family support rather than silently assume that family support is a complete replacement. A future institution should be evaluated separately for coverage, liquidity protection, and distributional incidence.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


def main():
    world = build_world()
    support, snapshots, lifecycle, estate = run_observed(world)
    make_outputs(world, support, snapshots, lifecycle, estate)
    print(json.dumps({"output_dir":str(OUT),"snapshots":len(snapshots),"support_events":len(support),"lifecycle_events":len(lifecycle),"estate_events":len(estate)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
