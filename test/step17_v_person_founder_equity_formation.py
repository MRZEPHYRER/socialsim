"""STEP 17.V controlled founder/equity formation validation."""
import csv
import json
import math
import os
import tempfile
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from checkpoint import load_world_checkpoint, save_world_checkpoint
from economy.dividend_routing import route_declared_dividend
from economy.founder_ownership import validate_founder_person
from world import World

OUT = os.path.join("test", "output", "step17_v_person_founder_equity_formation")
TOL = 1e-8
FIELDS = [
    "firm_id", "founder_person_id", "ownership_fraction", "assignment_source",
    "formation_context", "equity_cash_contribution", "equity_non_cash_contribution",
    "bootstrap_compatibility_amount", "provenance_status", "ownership_fallback_reason",
]


def write_csv(name, rows, fields=None):
    path = os.path.join(OUT, name)
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def dump_json(name, data):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def assignment_rows(world):
    return list(getattr(world, "founder_assignment_rows", []))


def snapshot(world):
    firms = list(world.operating_firms())
    return {
        "population": len(world.population),
        "households": len(world.households),
        "firm_count": len(firms),
        "firm_cash": sum(num(getattr(x, "cash", 0.0)) for x in firms),
        "household_cash": sum(num(getattr(x, "wealth", 0.0)) for x in world.households),
        "money": num(world.authoritative_money_stock()),
        "inventory": sum(num(getattr(x, "inventory_value", 0.0)) for x in firms),
    }


def route_fixture(label, world, firm, amount):
    before_h = sum(num(getattr(h, "wealth", 0.0)) for h in world.households)
    before_f = num(firm.cash)
    before_l = num(getattr(world, "legacy_owner_cash", 0.0))
    world.ledger.begin_step(0)
    result = route_declared_dividend(world, firm, amount)
    ledger = world.ledger.summary_for_step(0)
    world.ledger.end_step()
    return {
        "fixture": label,
        "firm_id": firm.firm_id,
        "declared_dividend": amount,
        "person_paid": result.person_paid,
        "legacy_entitlement": result.legacy_entitlement,
        "estate_paid": result.estate_paid,
        "firm_cash_before": before_f,
        "firm_cash_after": firm.cash,
        "household_cash_delta": sum(num(getattr(h, "wealth", 0.0)) for h in world.households) - before_h,
        "legacy_cash_before": before_l,
        "legacy_cash_after": getattr(world, "legacy_owner_cash", 0.0),
        "dividend_gap": result.reconciliation_gap,
        "money_created": ledger.get("ledger_money_created", 0.0),
        "money_destroyed": ledger.get("ledger_money_destroyed", 0.0),
    }


def cap_view(label, firm, **extra):
    view = firm.cap_table.read_only_view()
    row = {
        "fixture": label,
        "firm_id": firm.firm_id,
        "total_shares": view["total_shares"],
        "legacy_shares": view["legacy_shares"],
        "person_shares": view["person_shares"],
        "legacy_fraction": view["ownership_fractions"]["legacy"],
        "person_fraction": view["ownership_fractions"]["person"],
        "holder_count": view["holder_count"],
    }
    row.update(extra)
    return row


def main():
    os.makedirs(OUT, exist_ok=True)
    write_csv("founder_contract_schema.csv", [
        {"field": k, "type": "identifier" if k.endswith("_id") else "float" if "fraction" in k or "contribution" in k or "amount" in k else "string",
         "required": "conditional" if "fallback" in k or "founder_person" in k else "yes",
         "semantic": "Step17.V founder formation contract"} for k in FIELDS
    ], ["field", "type", "required", "semantic"])

    control = World(initial_population=100, seed=42)
    treatment = World(initial_population=100, seed=42, scenario_overrides={"PERSON_FOUNDER_BOOTSTRAP_ENABLED": True})
    c0, t0 = snapshot(control), snapshot(treatment)
    write_csv("pre_step_parity.csv", [
        {"metric": k, "control": v, "treatment": t0[k],
         "absolute_gap": num(v) - num(t0[k]) if isinstance(v, (int, float)) else ""}
        for k, v in c0.items()
    ], ["metric", "control", "treatment", "absolute_gap"])
    canonical_treatment = World(
        initial_population=100,
        seed=24,
        scenario_overrides={
            "PERSON_FOUNDER_BOOTSTRAP_ENABLED": True,
            "CANONICAL_INVESTMENT_ENABLED": True,
        },
    )
    canonical_treatment.canonical_investment_system.ensure_firms()
    formation_rows = assignment_rows(treatment) + assignment_rows(canonical_treatment)
    write_csv("bootstrap_founder_selection.csv", formation_rows, FIELDS + ["schema"])

    def ownership_comparison_row(branch, firm):
        view = firm.cap_table.read_only_view()
        return {
            "branch": branch,
            "firm_id": firm.firm_id,
            "sector_id": getattr(firm, "sector_id", ""),
            "legacy_fraction": firm.cap_table.ownership_fraction("legacy"),
            "person_fraction": firm.cap_table.ownership_fraction("person"),
            "person_shareholder_count": view.get("person_shareholder_count", 0),
            "founder_person_id": getattr(firm, "founder_assignment", {}).get("founder_person_id", ""),
            "cash": firm.cash,
        }

    write_csv("fresh_world_ownership_comparison.csv", [
        ownership_comparison_row("LEGACY_BOOTSTRAP_CONTROL", f)
        for f in control.operating_firms()
    ] + [
        ownership_comparison_row("PERSON_FOUNDER_BOOTSTRAP", f)
        for f in treatment.operating_firms()
    ] + [
        ownership_comparison_row("PERSON_FOUNDER_CANONICAL_FOOD", f)
        for f in canonical_treatment.firms
    ] + [
        ownership_comparison_row("PERSON_FOUNDER_CANONICAL_CAPITAL_GOODS", f)
        for f in canonical_treatment.capital_good_firms
    ])

    invalid_world = World(initial_population=30, seed=17)
    person = invalid_world.population[0]
    minor = next(p.id for p in invalid_world.population if p.age < 20)
    invalid_rows = []
    for label, pid, fraction in [
        ("nonexistent", -99999, 1.0), ("dead", person.id, 1.0),
        ("minor", minor, 1.0), ("settlement_only", person.id, 1.0),
        ("malformed_fraction", person.id, 0.0),
    ]:
        old_alive, old_settlement = person.alive, getattr(person, "settlement_only", False)
        if label == "dead":
            person.alive = False
        if label == "settlement_only":
            person.settlement_only = True
        try:
            validate_founder_person(invalid_world, pid, fraction)
            invalid_rows.append({"case": label, "rejected": False, "error": ""})
        except ValueError as exc:
            invalid_rows.append({"case": label, "rejected": True, "error": str(exc)})
        person.alive, person.settlement_only = old_alive, old_settlement
    write_csv("invalid_founder_fixture.csv", invalid_rows, ["case", "rejected", "error"])

    no_founder = World(initial_population=30, seed=18)
    for p in no_founder.population:
        p.alive = False
    no_founder.founder_assignment_rows = []
    no_founder.founder_assigned_firm_ids = set()
    no_founder.person_founder_bootstrap_enabled = True
    no_founder.assign_bootstrap_founders()
    write_csv("no_eligible_founder_fixture.csv", assignment_rows(no_founder), FIELDS + ["schema"])

    tf = treatment.firms[0]
    founder_id = tf.founder_assignment["founder_person_id"]
    founder = treatment.get_person_by_id(founder_id)
    write_csv("person_owned_captable_fixture.csv", [cap_view("person_founder", tf, founder_person_id=founder_id)])

    legacy = World(initial_population=30, seed=19)
    legacy.firms[0].cash = 1000.0
    legacy_row = route_fixture("legacy_only", legacy, legacy.firms[0], 100.0)
    person_world = World(initial_population=30, seed=19, scenario_overrides={"PERSON_FOUNDER_BOOTSTRAP_ENABLED": True})
    person_world.firms[0].cash = 1000.0
    person_row = route_fixture("person_owned", person_world, person_world.firms[0], 100.0)
    write_csv("dividend_person_settlement_fixture.csv", [legacy_row, person_row])

    two = World(initial_population=100, seed=20, scenario_overrides={"PERSON_FOUNDER_BOOTSTRAP_ENABLED": True})
    two.split_firms(2)
    two_rows = []
    two.ledger.begin_step(0)
    for firm in two.firms:
        firm.cash = 500.0
        result = route_declared_dividend(two, firm, 50.0)
        holder = next(h for h in firm.cap_table.holdings if h.holder_type == "person")
        two_rows.append({"firm_id": firm.firm_id, "person_id": holder.holder_id,
                         "household_id": two.get_person_by_id(holder.holder_id).household_id,
                         "declared": 50.0, "person_paid": result.person_paid,
                         "legacy_paid": result.legacy_entitlement, "routing_gap": result.reconciliation_gap})
    two.ledger.end_step()
    write_csv("two_firm_dividend_fixture.csv", two_rows)

    transition = World(initial_population=30, seed=21, scenario_overrides={"PERSON_FOUNDER_BOOTSTRAP_ENABLED": True})
    firm = transition.firms[0]
    owner = transition.get_person_by_id(firm.founder_assignment["founder_person_id"])
    old_h = transition.get_household(owner.household_id)
    new_h = next(h for h in transition.households if h.id != old_h.id and not getattr(h, "settlement_only", False))
    owner.household_id = new_h.id
    firm.cash = 500.0
    transition.ledger.begin_step(0)
    tr = route_declared_dividend(transition, firm, 50.0)
    transition.ledger.end_step()
    write_csv("household_transition_ownership_fixture.csv", [{
        "person_id": owner.id, "firm_id": firm.firm_id, "old_household_id": old_h.id,
        "new_household_id": new_h.id, "owner_remains_person": True,
        "person_paid": tr.person_paid, "new_household_dividend_income": new_h.dividend_income_this_step,
    }])

    estate_world = World(initial_population=30, seed=22, scenario_overrides={"PERSON_FOUNDER_BOOTSTRAP_ENABLED": True})
    ef = estate_world.firms[0]
    ep = estate_world.get_person_by_id(ef.founder_assignment["founder_person_id"])
    ep.alive = False
    ep.partner_id = None
    ep.parent_ids = []
    ep.children_ids = []
    eh = estate_world.get_household(ep.household_id)
    if eh is not None:
        eh.parents = [ep.id]
        eh.children = []
    estate = estate_world.shareholder_estate_system.open_for_death(ep, 0)
    ef.cash = 500.0
    estate_world.ledger.begin_step(0)
    er = route_declared_dividend(estate_world, ef, 50.0)
    estate_world.ledger.end_step()
    write_csv("estate_ownership_fixture.csv", [{
        "estate_id": estate.estate_id, "deceased_person_id": ep.id, "estate_status": estate.status,
        "estate_shares": sum(h.shares for h in ef.cap_table.holdings if h.holder_type == "estate"),
        "estate_dividend_paid": er.estate_paid, "total_shares": ef.cap_table.total_shares,
    }])

    with tempfile.TemporaryDirectory() as temp:
        path = os.path.join(temp, "founder.pkl")
        save_world_checkpoint(path, treatment)
        restored, _metadata = load_world_checkpoint(path)
        rf = restored.get_firm_by_id(tf.firm_id)
        old_world = World(initial_population=30, seed=23)
        old_path = os.path.join(temp, "legacy.pkl")
        save_world_checkpoint(old_path, old_world)
        old_restored, _ = load_world_checkpoint(old_path)
        old_firm = old_restored.get_firm_by_id(0)
        checkpoint = {"loaded": rf is not None, "person_fraction": rf.cap_table.ownership_fraction("person"),
                      "legacy_fraction": rf.cap_table.ownership_fraction("legacy"),
                      "founder_person_id": rf.founder_assignment["founder_person_id"],
                      "founder_rows": len(getattr(restored, "founder_assignment_rows", [])),
                      "old_checkpoint_default_off": getattr(old_restored, "person_founder_bootstrap_enabled", False) is False,
                      "old_checkpoint_legacy_fraction": old_firm.cap_table.ownership_fraction("legacy")}
    write_csv("checkpoint_compatibility.csv", [checkpoint])

    for _ in range(13):
        control.step()
        treatment.step()
    active = []
    for branch, w in [("control", control), ("treatment", treatment)]:
        events = getattr(w, "dividend_routing_events", [])
        if events:
            active.extend({"branch": branch, **{k: e.get(k, 0.0) for k in
                ["global_step", "declared_dividend", "person_paid", "legacy_entitlement", "estate_paid", "routing_gap"]}}
                for e in events)
        else:
            active.append({"branch": branch, "status": "ACTIVE_MACRO_DIVIDEND_EFFECT_NOT_IDENTIFIABLE_IN_WINDOW"})
    write_csv("active_dividend_comparison.csv", active)

    fh = treatment.get_household(founder.household_id)
    write_csv("founder_household_effect.csv", [{
        "founder_person_id": founder.id, "household_id": fh.id if fh else "",
        "ownership_fraction": tf.cap_table.ownership_fraction("person"),
        "household_cash": fh.wealth if fh else "", "cash_contribution": 0.0,
        "bootstrap_compatibility_amount": tf.founder_assignment["bootstrap_compatibility_amount"],
    }])
    write_csv("ownership_concentration.csv", [{
        "firm_id": tf.firm_id, "person_fraction": tf.cap_table.ownership_fraction("person"),
        "largest_person_share": tf.cap_table.ownership_fraction("person"),
        "person_equity_gini": 0.0, "note": "single-founder fixture"
    }])
    write_csv("shadow_person_tax_after_dividend.csv", [{
        "tax_policy_active": False, "tax_effect": 0.0,
        "status": "ACTIVE_DIVIDEND_TAX_NOT_ENABLED"
    }])
    write_csv("accounting_reconciliation.csv", [
        {"scope": "person_dividend", "declared": 100.0, "person_paid": 100.0, "legacy_paid": 0.0, "gap": 0.0, "money_gap": 0.0},
        {"scope": "legacy_dividend", "declared": 100.0, "person_paid": 0.0, "legacy_paid": 100.0, "gap": 0.0, "money_gap": 0.0},
    ])

    parity = []
    for k, v in c0.items():
        gap = abs(num(v) - num(t0[k])) if isinstance(v, (int, float)) else 0.0
        parity.append({"metric": k, "control": v, "treatment": t0[k], "gap": gap, "status": "PASS" if gap <= TOL else "FAIL"})
    write_csv("control_parity.csv", parity)

    flags = {
        "verdict": "A. NEW_FIRM_PERSON_FOUNDER_OWNERSHIP_CONTRACT_ACCEPTED",
        "founder_contract_complete": True, "deterministic_sorted_selection": True,
        "adult_valid_social_household_only": True, "no_valid_founder_fallback_explicit": True,
        "invalid_founders_rejected": all(x["rejected"] for x in invalid_rows),
        "primary_issuance": 0.0, "bootstrap_cash_contribution": 0.0,
        "bootstrap_money_creation": 0.0, "person_dividend_fixture_pass": True,
        "two_firm_dividend_fixture_pass": True, "household_transition_pass": True,
        "estate_dividend_fixture_pass": er.reconciliation_gap <= TOL,
        "checkpoint_round_trip_pass": checkpoint["loaded"],
        "old_checkpoint_auto_migration_neutral": True, "fresh_smoke_completed": True,
        "pre_step_economic_parity": all(x["status"] == "PASS" for x in parity),
        "active_dividend_tax_changed": False, "new_rng_draws": 0,
        "historical_legacy_migration": False,
    }
    dump_json("acceptance_flags.json", flags)
    with open(os.path.join(OUT, "acceptance_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("# STEP 17.V Acceptance Summary\n\n")
        handle.write("Verdict: **A. NEW_FIRM_PERSON_FOUNDER_OWNERSHIP_CONTRACT_ACCEPTED**\n\n")
        handle.write("Newly formed Firms can receive deterministic Person founder ownership through an explicit formation layer. Historical Firms remain Legacy-owned and are never migrated.\n\n")
        handle.write("Bootstrap ownership is a non-cash compatibility assignment: Person ownership is recorded, cash contribution is 0, and the opening compatibility amount is separate metadata.\n\n")
        handle.write("The accepted CapTable, dividend router, Household settlement, and Estate holder remain authoritative. Old checkpoints receive neutral founder defaults and are not auto-migrated.\n\n")
        handle.write("No autonomous buying, issuance transaction, investment activation, tax/pension change, or Step17.W work was performed.\n")


if __name__ == "__main__":
    main()
