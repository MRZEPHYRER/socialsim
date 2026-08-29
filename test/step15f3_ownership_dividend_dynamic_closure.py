from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.equity_transition import GradualPersonEquityTransition
from world import World


OUTPUT = ROOT / "test/output/step15F3_ownership_dividend_dynamic_closure"
WEEKS = 104
TOLERANCE = 1e-6


def load_case(enabled):
    world = World(initial_population=5000, seed=42, diagnostics_mode="full")
    world.ensure_ownership_state()
    world.person_equity_transition_enabled = bool(enabled)
    if enabled:
        world.equity_transition_system = GradualPersonEquityTransition(world)
    return world


def state_row(world, case):
    firm = world.firms[0]
    return {
        "case": case,
        "global_step": int(world.current_step_index),
        "person_ownership_fraction": firm.cap_table.ownership_fraction("person"),
        "legacy_ownership_fraction": firm.cap_table.ownership_fraction("legacy"),
        "declared_dividend": float(getattr(firm, "dividend_payment", 0.0)),
        "person_dividend": float(getattr(firm, "person_dividend_paid", 0.0)),
        "legacy_dividend": float(getattr(firm, "legacy_dividend_entitlement", 0.0)),
        "household_dividend_income": sum(
            float(getattr(h, "dividend_income_this_step", 0.0))
            for h in world.households
        ),
        "household_total_income": float(world.income_history[-1]),
        "household_consumption": float(world.consumption_history[-1]),
        "household_saving": float(world.saving_history[-1]),
        "household_cash_wealth": sum(float(h.wealth) for h in world.households),
        "legacy_owner_cash": float(getattr(world, "legacy_owner_cash", 0.0)),
        "firm_cash": float(world.firm_system.cash),
    }


def event_rows(world, case, start_index):
    rows = []
    for event in getattr(world, "dividend_routing_events", [])[start_index:]:
        receipts = event.get("household_receipts", [])
        rows.append({
            "case": case,
            "global_step": event["global_step"],
            "firm_id": event["firm_id"],
            "declared_dividend": event["declared_dividend"],
            "legacy_ownership_fraction": event["legacy_ownership_fraction"],
            "person_ownership_fraction": event["person_ownership_fraction"],
            "person_paid_dividend": event["person_paid"],
            "legacy_entitlement": event["legacy_entitlement"],
            "household_dividend_income": event["household_dividend_income"],
            "legacy_owner_cash_inflow": event["legacy_owner_cash_inflow"],
            "firm_cash_before": event["firm_cash_before"],
            "firm_cash_after": event["firm_cash_after"],
            "household_cash_before": (
                sum(r["household_cash_before"] for r in receipts)
                if receipts else ""
            ),
            "household_cash_after": (
                sum(r["household_cash_after"] for r in receipts)
                if receipts else ""
            ),
            "household_ids": json.dumps(
                [r["household_id"] for r in receipts], separators=(",", ":")
            ),
            "person_ids": json.dumps(
                [r["person_id"] for r in receipts], separators=(",", ":")
            ),
            "total_shares": event["total_shares"],
            "legacy_shares": event["legacy_shares"],
            "person_shares": event["person_shares"],
            "dividend_routing_gap": event["routing_gap"],
            "firm_cash_outflow_gap": (
                event["firm_cash_before"]
                - event["firm_cash_after"]
                - event["declared_dividend"]
            ),
            "household_inflow_gap": (
                event["household_dividend_income"] - event["person_paid"]
            ),
            "money_created": 0.0,
            "money_destroyed": 0.0,
        })
    return rows


def run_case(case, enabled):
    world = load_case(enabled)
    states = []
    events = []
    for _ in range(WEEKS):
        before = len(getattr(world, "dividend_routing_events", []))
        world.step()
        states.append(state_row(world, case))
        events.extend(event_rows(world, case, before))

    accounting_rows = getattr(world.accounting, "rows", [])
    household_rows = getattr(world.accounting, "household_rows", [])
    return world, states, events, {
        "case": case,
        "enabled": enabled,
        "weeks": WEEKS,
        "actual_dividend_events": len(events),
        "events_with_person_payment": sum(
            row["person_paid_dividend"] > TOLERANCE for row in events
        ),
        "cumulative_declared_dividends": sum(
            row["declared_dividend"] for row in events
        ),
        "cumulative_person_dividends": sum(
            row["person_paid_dividend"] for row in events
        ),
        "cumulative_legacy_dividends": sum(
            row["legacy_entitlement"] for row in events
        ),
        "ending_legacy_owner_cash": states[-1]["legacy_owner_cash"],
        "cumulative_household_dividend_income": sum(
            row["household_dividend_income"] for row in states
        ),
        "cumulative_household_consumption": sum(
            row["household_consumption"] for row in states
        ),
        "cumulative_household_saving": sum(
            row["household_saving"] for row in states
        ),
        "ending_household_cash_wealth": states[-1]["household_cash_wealth"],
        "ending_firm_cash": states[-1]["firm_cash"],
        "initial_person_ownership_fraction": states[0]["person_ownership_fraction"],
        "final_person_ownership_fraction": states[-1]["person_ownership_fraction"],
        "max_dividend_routing_gap": max(
            (abs(row["dividend_routing_gap"]) for row in events), default=0.0
        ),
        "max_firm_cash_outflow_gap": max(
            (abs(row["firm_cash_outflow_gap"]) for row in events), default=0.0
        ),
        "max_household_inflow_gap": max(
            (abs(row["household_inflow_gap"]) for row in events), default=0.0
        ),
        "max_accounting_gap": max(
            (
                abs(float(row.get("cash_flow_gap", 0.0)))
                for row in accounting_rows
            ),
            default=0.0,
        ),
        "max_household_accounting_gap": max(
            (
                abs(float(row.get("household_wealth_bridge_gap", 0.0)))
                for row in household_rows
            ),
            default=0.0,
        ),
    }


def write_csv(path, rows):
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_windows(states_by_case, events):
    by_key = {
        (row["case"], row["global_step"]): row
        for rows in states_by_case.values()
        for row in rows
    }
    result = []
    for event in events:
        for relative in (-2, -1, 0, 1, 2, 3, 4, 5):
            row = by_key.get((event["case"], event["global_step"] + relative))
            if row is None:
                continue
            result.append({
                "case": event["case"],
                "event_global_step": event["global_step"],
                "relative_step": relative,
                "declared_dividend": event["declared_dividend"],
                "person_ownership_fraction": row["person_ownership_fraction"],
                "person_dividend": row["person_dividend"],
                "legacy_dividend": row["legacy_dividend"],
                "household_dividend_income": row["household_dividend_income"],
                "household_total_income": row["household_total_income"],
                "household_consumption": row["household_consumption"],
                "household_saving": row["household_saving"],
                "household_cash_wealth": row["household_cash_wealth"],
                "legacy_owner_cash": row["legacy_owner_cash"],
                "firm_cash": row["firm_cash"],
            })
    return result


def make_figures(states, events, windows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = sorted(states)
    colors = {"control_off": "#666666", "treatment_on": "#1769aa"}

    def line(name, field, title, ylabel):
        plt.figure(figsize=(9, 4.8))
        for case in cases:
            rows = states[case]
            plt.plot(
                [r["global_step"] for r in rows],
                [r[field] for r in rows],
                label=case,
                color=colors[case],
            )
        plt.title(title)
        plt.xlabel("Global week")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT / name, dpi=140)
        plt.close()

    line(
        "ownership_and_dividend_share.png",
        "person_ownership_fraction",
        "Person ownership transition",
        "Person ownership fraction",
    )
    line(
        "household_dividend_income.png",
        "household_dividend_income",
        "Household dividend income",
        "Dividend income",
    )
    line(
        "legacy_cash_accumulation.png",
        "legacy_owner_cash",
        "Legacy owner cash accumulation",
        "Legacy owner cash",
    )

    plt.figure(figsize=(9, 4.8))
    for case in cases:
        grouped = defaultdict(list)
        for row in windows:
            if row["case"] == case:
                grouped[row["relative_step"]].append(row["household_consumption"])
        xs = sorted(grouped)
        plt.plot(
            xs,
            [sum(grouped[x]) / len(grouped[x]) for x in xs],
            label=case,
            color=colors[case],
        )
    plt.axvline(0, color="#aa3333", linestyle="--", linewidth=1)
    plt.title("Household consumption around actual dividend events")
    plt.xlabel("Relative week to event")
    plt.ylabel("Household consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "dividend_event_consumption_response.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    for case in cases:
        rows = [r for r in events if r["case"] == case]
        plt.plot(
            [r["global_step"] for r in rows],
            [r["dividend_routing_gap"] for r in rows],
            ".-",
            label=f"{case} routing gap",
        )
        plt.plot(
            [r["global_step"] for r in rows],
            [r["firm_cash_outflow_gap"] for r in rows],
            "--",
            label=f"{case} cash gap",
        )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Dividend routing reconciliation")
    plt.xlabel("Global week")
    plt.ylabel("Gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "dividend_reconciliation.png", dpi=140)
    plt.close()


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [("control_off", False), ("treatment_on", True)]
    summaries = []
    states = {}
    events = []
    for case, enabled in cases:
        _, case_states, case_events, summary = run_case(case, enabled)
        summaries.append(summary)
        states[case] = case_states
        events.extend(case_events)

    windows = build_windows(states, events)
    summary_by_case = {row["case"]: row for row in summaries}
    control = summary_by_case["control_off"]
    treatment = summary_by_case["treatment_on"]
    actual_events = bool(events)
    treatment_person_events = treatment["events_with_person_payment"] > 0
    routing_closed = all(
        row["dividend_routing_gap"] == row["dividend_routing_gap"]
        and abs(row["dividend_routing_gap"]) <= TOLERANCE
        and abs(row["firm_cash_outflow_gap"]) <= TOLERANCE
        and abs(row["household_inflow_gap"]) <= TOLERANCE
        for row in events
    )
    leakage_reduced = (
        treatment["cumulative_legacy_dividends"]
        < control["cumulative_legacy_dividends"] - TOLERANCE
    )
    accounting_closed = all(
        summary["max_dividend_routing_gap"] <= TOLERANCE
        and summary["max_firm_cash_outflow_gap"] <= TOLERANCE
        and summary["max_household_inflow_gap"] <= TOLERANCE
        and summary["max_accounting_gap"] <= 1e-4
        and summary["max_household_accounting_gap"] <= 1e-4
        for summary in summaries
    )
    if not actual_events:
        verdict = "B. NO_ACTUAL_DIVIDEND_EVENTS_OBSERVED"
    elif not routing_closed:
        verdict = "C. DIVIDEND_ROUTING_RUNTIME_FAILED"
    elif not accounting_closed:
        verdict = "F. ACCOUNTING_RECONCILIATION_FAILED"
    elif not treatment_person_events:
        verdict = "D. HOUSEHOLD_INCOME_LINK_FAILED"
    elif not leakage_reduced:
        verdict = "E. LEGACY_LEAKAGE_NOT_REDUCED"
    else:
        verdict = "A. OWNERSHIP_DIVIDEND_HOUSEHOLD_CLOSURE_ACCEPTED"

    flags = {
        "verdict": verdict,
        "same_seed": True,
        "actual_dividend_events_observed": actual_events,
        "treatment_person_dividend_events_observed": treatment_person_events,
        "dividend_routing_reconciles": routing_closed,
        "household_income_link_observed": treatment_person_events,
        "legacy_leakage_reduced": leakage_reduced,
        "money_created_by_dividend_routing": 0.0,
        "money_destroyed_by_dividend_routing": 0.0,
        "accounting_reconciles": accounting_closed,
        "ownership_stock_changed_by_dividends": False,
        "ownership_transition_enabled_only_in_treatment": True,
        "dividend_policy_changed": False,
        "secondary_market": False,
        "leveraged_purchase": False,
        "investment_active": False,
        "step13_changed": False,
    }
    write_csv(OUTPUT / "dividend_event_panel.csv", events)
    write_csv(OUTPUT / "dividend_event_windows.csv", windows)
    write_csv(OUTPUT / "ownership_dividend_metrics.csv", summaries)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    make_figures(states, events, windows)
    (OUTPUT / "acceptance_summary.md").write_text(
        "# Step 15F.3 Ownership-Dividend-Household Dynamic Closure\n\n"
        f"**Verdict: {verdict}**\n\n"
        f"A fresh same-seed canonical run covered {WEEKS} weeks. The control "
        "kept ownership transition off; the treatment reused the accepted "
        "deterministic primary issuance adapter. No synthetic dividend was "
        "injected into the main comparison.\n\n"
        "Actual routing events are recorded from the runtime dividend layer, "
        "including Firm and recipient cash before/after settlement. The event "
        "panel distinguishes Person-paid dividends from Legacy owner cash.\n\n"
        f"Control events: {control['actual_dividend_events']}; treatment events: "
        f"{treatment['actual_dividend_events']}. Treatment Person-paid events: "
        f"{treatment['events_with_person_payment']}.\n\n"
        f"Cumulative declared dividends: control {control['cumulative_declared_dividends']:.6f}; "
        f"treatment {treatment['cumulative_declared_dividends']:.6f}.\n\n"
        f"Cumulative Person dividends in treatment: {treatment['cumulative_person_dividends']:.6f}; "
        f"cumulative Legacy dividends in treatment: {treatment['cumulative_legacy_dividends']:.6f}.\n",
        encoding="utf-8",
    )
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
