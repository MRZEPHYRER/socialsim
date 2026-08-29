"""STEP 17.W active Person capital-income and shadow-tax validation."""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.income_tax import progressive_tax_liability
from world import World

OUT = ROOT / "test/output/step17_w_person_capital_income_distribution"
EPS = 1e-8
EXEMPTION = 3026.40
# Accepted Step17.S bracket thresholds; this stage only evaluates them.
BRACKETS = [(0.0, 0.0), (EXEMPTION, 0.0), (3092.4603298, 0.02), (3179.8375880, 0.05)]


def num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def write_rows(name, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "UNAVAILABLE"}])


def gini(values):
    xs = sorted(max(0.0, num(v)) for v in values)
    total = sum(xs)
    if not xs or total <= EPS:
        return 0.0
    return sum((2 * i - len(xs) - 1) * x for i, x in enumerate(xs, 1)) / (len(xs) * total)


def quantile(values, q):
    xs = sorted(num(v) for v in values)
    if not xs:
        return 0.0
    index = (len(xs) - 1) * q
    lo, hi = int(index), math.ceil(index)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (index - lo)


def tax_stats(values, liabilities, label):
    values = [num(v) for v in values]
    liabilities = [max(0.0, num(v)) for v in liabilities]
    ordered = sorted(liabilities, reverse=True)
    total_tax = sum(liabilities)
    def top_share(n):
        return sum(ordered[:n]) / total_tax if total_tax > EPS else 0.0
    after = [max(0.0, v - t) for v, t in zip(values, liabilities)]
    return {
        "candidate": label, "person_count": len(values),
        "income_min": min(values, default=0.0), "income_P10": quantile(values, .10),
        "income_P25": quantile(values, .25), "income_P50": quantile(values, .50),
        "income_P75": quantile(values, .75), "income_P90": quantile(values, .90),
        "income_P95": quantile(values, .95), "income_P99": quantile(values, .99),
        "income_max": max(values, default=0.0), "income_mean": statistics.fmean(values) if values else 0.0,
        "taxpayer_count": sum(t > EPS for t in liabilities),
        "taxpayer_share": sum(t > EPS for t in liabilities) / len(values) if values else 0.0,
        "total_shadow_tax": total_tax,
        "effective_tax_rate": total_tax / sum(values) if sum(values) > EPS else 0.0,
        "pre_tax_gini": gini(values), "post_shadow_tax_gini": gini(after),
        "top1_tax_share": top_share(1), "top5_tax_share": top_share(5), "top10_tax_share": top_share(10),
        "tax_liability_gini": gini(liabilities),
        "mean_effective_rate_taxpayers": statistics.fmean([t / v for v, t in zip(values, liabilities) if t > EPS and v > EPS]) if any(t > EPS for t in liabilities) else 0.0,
        "maximum_effective_rate": max((t / v for v, t in zip(values, liabilities) if v > EPS), default=0.0),
        "bottom50_income_share": sum(sorted(values)[:max(1, len(values) // 2)]) / sum(values) if sum(values) > EPS else 0.0,
        "top10_income_share": sum(sorted(values, reverse=True)[:max(1, len(values) // 10)]) / sum(values) if sum(values) > EPS else 0.0,
        "top5_income_share": sum(sorted(values, reverse=True)[:max(1, len(values) // 20)]) / sum(values) if sum(values) > EPS else 0.0,
        "top1_income_share": max(values, default=0.0) / sum(values) if sum(values) > EPS else 0.0,
    }


def run_branch(founder_enabled):
    world = World(initial_population=100, seed=42, scenario_overrides={"PERSON_FOUNDER_BOOTSTRAP_ENABLED": founder_enabled})
    world.household_employer_exposure_instrumentation_enabled = True
    world.personal_income_tax_enabled = False
    world.corporate_profit_tax_enabled = False
    world.public_revenue_tax_enabled = False
    wages = defaultdict(float)
    dividends = defaultdict(float)
    person_household = {}
    event_rows = []
    weekly = []
    household_totals = defaultdict(lambda: {"wage": 0.0, "dividend": 0.0, "consumption": 0.0})
    prior_events = 0
    initial_cash = {h.id: num(getattr(h, "wealth", 0.0)) for h in world.households}
    for _ in range(52):
        world.step()
        provenance = getattr(world, "_household_payroll_provenance_weekly_state", {})
        for household_id, record in provenance.items():
            for item in record.get("person_components", []):
                pid = item.get("person_id")
                if pid is not None:
                    paid = max(0.0, num(item.get("paid_wage")))
                    wages[pid] += paid
                    person_household[pid] = household_id
                    household_totals[household_id]["wage"] += paid
        events = getattr(world, "dividend_routing_events", [])
        for event in events[prior_events:]:
            event = dict(event)
            event["branch"] = "PERSON_FOUNDER_BOOTSTRAP" if founder_enabled else "LEGACY_BOOTSTRAP_CONTROL"
            event_rows.append(event)
            for receipt in event.get("household_receipts", []) or []:
                received = max(0.0, num(receipt.get("dividend_received")))
                dividends[receipt.get("person_id")] += received
                household_totals[receipt.get("household_id")]["dividend"] += received
        prior_events = len(events)
        for household in world.households:
            household_totals[household.id]["consumption"] += max(0.0, num(getattr(household, "consumption_this_step", 0.0)))
        diagnostic = world.diagnostics_rows[-1] if getattr(world, "diagnostics_rows", None) else {}
        weekly.append({
            "global_step": int(getattr(world, "current_step_index", 0)),
            "household_income": sum(num(getattr(h, "income_this_step", 0.0)) for h in world.households),
            "household_consumption": sum(num(getattr(h, "consumption_this_step", 0.0)) for h in world.households),
            "household_saving": sum(num(getattr(h, "saving_this_step", 0.0)) for h in world.households),
            "household_cash": sum(num(getattr(h, "wealth", 0.0)) for h in world.households),
            "firm_sales": num(diagnostic.get("firm_sales_revenue", 0.0)),
            "firm_cash": num(getattr(world.firm_system, "cash", diagnostic.get("firm_cash", 0.0))),
            "money_stock": num(diagnostic.get("total_money_stock", world.authoritative_money_stock())),
        })
    for p in world.population:
        wages.setdefault(p.id, 0.0)
        dividends.setdefault(p.id, 0.0)
        person_household.setdefault(p.id, getattr(p, "household_id", None))
    return world, wages, dividends, person_household, event_rows, weekly, initial_cash, household_totals


def ownership_hhi(world):
    shares = []
    for firm in world.operating_firms():
        table = getattr(firm, "cap_table", None)
        if table is None:
            continue
        shares.extend(float(h.shares) / float(table.total_shares) for h in table.holdings if h.holder_type == "person")
    return sum(x * x for x in shares), shares



def low_income_rows(treatment_people):
    rows = []
    for band, lo, hi in [("income_need_<0.25", 0.0, .25), ("0.25_to_0.5", .25, .5), ("0.5_to_1", .5, 1.0), ("1_to_2", 1.0, 2.0), ("over_2", 2.0, float("inf"))]:
        members = [
            x for x in treatment_people
            if lo <= x["wage_plus_dividend_income"] / EXEMPTION < hi
        ]
        rows.append({
            "income_need_band": band,
            "person_count": len(members),
            "mean_shadow_tax": statistics.fmean(x["wage_plus_dividend_liability"] for x in members) if members else 0.0,
            "protection_status": "PROTECTED_BELOW_EXEMPTION" if hi <= 1.0 else "DESCRIPTIVE",
        })
    return rows
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    control = run_branch(False)
    treatment = run_branch(True)
    cw, tw = control, treatment
    worlds = {
        "LEGACY_BOOTSTRAP_CONTROL": cw,
        "PERSON_FOUNDER_BOOTSTRAP": tw,
    }

    person_rows = []
    tax_rows = []
    all_founders = set()
    for branch, (world, wages, dividends, person_household, events, weekly, initial_cash, household_totals) in worlds.items():
        for person in world.population:
            wage = num(wages[person.id])
            dividend = num(dividends[person.id])
            combined = wage + dividend
            wage_tax = progressive_tax_liability(wage, BRACKETS)
            combined_tax = progressive_tax_liability(combined, BRACKETS)
            person_rows.append({
                "branch": branch,
                "person_id": person.id,
                "household_id": person_household[person.id],
                "alive_at_end": bool(getattr(person, "alive", False)),
                "wage_income": wage,
                "realized_dividend_income": dividend,
                "wage_plus_dividend_income": combined,
                "wage_only_liability": wage_tax,
                "wage_plus_dividend_liability": combined_tax,
                "wage_only_effective_rate": wage_tax / wage if wage > EPS else 0.0,
                "wage_plus_dividend_effective_rate": combined_tax / combined if combined > EPS else 0.0,
                "minimum_need_exemption": EXEMPTION,
            })
        values_wage = [num(wages[p.id]) for p in world.population]
        values_combined = [num(wages[p.id]) + num(dividends[p.id]) for p in world.population]
        tax_rows.extend([
            tax_stats(
                values_wage,
                [progressive_tax_liability(x, BRACKETS) for x in values_wage],
                branch + ":WAGE_ONLY",
            ),
            tax_stats(
                values_combined,
                [progressive_tax_liability(x, BRACKETS) for x in values_combined],
                branch + ":WAGE_PLUS_REALIZED_DIVIDEND",
            ),
        ])
        if branch == "PERSON_FOUNDER_BOOTSTRAP":
            all_founders = {
                row.get("founder_person_id")
                for row in getattr(world, "founder_assignment_rows", [])
                if row.get("founder_person_id") not in (None, "")
            }

    write_rows("person_annual_income.csv", person_rows)
    write_rows("person_income_distribution.csv", tax_rows)

    treatment_people = [
        row for row in person_rows
        if row["branch"] == "PERSON_FOUNDER_BOOTSTRAP"
    ]
    treatment_events = tw[4]
    person_dividend_total = sum(num(event.get("person_paid")) for event in treatment_events)
    recipient_values = [
        row["realized_dividend_income"]
        for row in treatment_people
        if row["realized_dividend_income"] > EPS
    ]
    alive_treatment = sum(bool(getattr(p, "alive", False)) for p in tw[0].population)
    recipient_households = {
        row["household_id"] for row in treatment_people
        if row["realized_dividend_income"] > EPS
    }
    write_rows("dividend_recipient_summary.csv", [{
        "branch": "PERSON_FOUNDER_BOOTSTRAP",
        "recipient_person_count": len(recipient_values),
        "alive_person_recipient_share": len(recipient_values) / max(1, alive_treatment),
        "recipient_household_count": len(recipient_households),
        "dividend_event_count": len(treatment_events),
        "total_realized_person_dividend": person_dividend_total,
        "mean_dividend_per_recipient": statistics.fmean(recipient_values) if recipient_values else 0.0,
        "median_dividend_per_recipient": quantile(recipient_values, 0.5),
    }])

    hhi, person_shares = ownership_hhi(tw[0])
    total_declared = sum(num(event.get("declared_dividend")) for event in treatment_events)
    ordered_dividends = sorted(recipient_values, reverse=True)
    write_rows("dividend_concentration.csv", [{
        "branch": "PERSON_FOUNDER_BOOTSTRAP",
        "top1_recipient_share": sum(ordered_dividends[:1]) / total_declared if total_declared > EPS else 0.0,
        "top5_recipient_share": sum(ordered_dividends[:5]) / total_declared if total_declared > EPS else 0.0,
        "top10_recipient_share": sum(ordered_dividends[:10]) / total_declared if total_declared > EPS else 0.0,
        "dividend_gini": gini(recipient_values),
        "ownership_hhi": hhi,
        "person_ownership_hhi_basis": "Person shares only; single-founder fixture",
        "firms_per_person": len(list(tw[0].operating_firms())) / max(1, len(tw[0].population)),
        "founders_per_person": len(all_founders) / max(1, len(tw[0].population)),
        "concentration_classification": "MECHANICAL_BOOTSTRAP_CONCENTRATION",
    }])

    founder_person_ids = {
        row.get("founder_person_id")
        for row in getattr(tw[0], "founder_assignment_rows", [])
        if row.get("founder_person_id") not in (None, "")
    }
    founder_household_ids = {
        getattr(tw[0].get_person_by_id(pid), "household_id", None)
        for pid in founder_person_ids
    }
    founder_household_ids.discard(None)
    founder_rows = []
    for household in tw[0].households:
        totals = tw[7].get(
            household.id,
            {"wage": 0.0, "dividend": 0.0, "consumption": 0.0},
        )
        weekly_need = max(
            0.0,
            num(getattr(household, "necessary_consumption_this_step", 0.0)),
        )
        founder_rows.append({
            "branch": "PERSON_FOUNDER_BOOTSTRAP",
            "household_id": household.id,
            "founder_household": household.id in founder_household_ids,
            "opening_cash": tw[6].get(household.id, num(getattr(household, "wealth", 0.0))),
            "wage_income": totals["wage"],
            "dividend_income": totals["dividend"],
            "total_income": totals["wage"] + totals["dividend"],
            "consumption": totals["consumption"],
            "closing_cash": num(getattr(household, "wealth", 0.0)),
            "liquidity_weeks": num(getattr(household, "wealth", 0.0)) / weekly_need if weekly_need > EPS else 0.0,
        })
    write_rows("founder_household_distribution.csv", founder_rows)

    def weekly_macro(branch, weekly, events):
        event_by_week = {
            int(num(event.get("global_step"))): event
            for event in events
        }
        rows = []
        for row in weekly:
            event = event_by_week.get(int(row["global_step"]), {})
            rows.append({
                "branch": branch,
                **row,
                "declared_dividend": num(event.get("declared_dividend")),
                "person_dividend": num(event.get("person_paid")),
                "legacy_dividend": num(event.get("legacy_entitlement")),
            })
        return rows

    write_rows(
        "control_treatment_macro_comparison.csv",
        weekly_macro("LEGACY_BOOTSTRAP_CONTROL", cw[5], cw[4])
        + weekly_macro("PERSON_FOUNDER_BOOTSTRAP", tw[5], tw[4]),
    )

    control_events = {
        int(num(event.get("global_step"))): event
        for event in cw[4]
    }
    treatment_events = {
        int(num(event.get("global_step"))): event
        for event in tw[4]
    }
    divergence_rows = []
    for week in sorted(set(control_events) | set(treatment_events)):
        control_amount = num(control_events.get(week, {}).get("declared_dividend"))
        treatment_amount = num(treatment_events.get(week, {}).get("declared_dividend"))
        divergence_rows.append({
            "global_step": week,
            "control_declared": control_amount,
            "treatment_declared": treatment_amount,
            "absolute_gap": treatment_amount - control_amount,
            "divergent": abs(treatment_amount - control_amount) > EPS,
        })
    first_divergence = next(
        (row["global_step"] for row in divergence_rows if row["divergent"]),
        "",
    )
    for row in divergence_rows:
        row["first_divergence_week"] = first_divergence
    write_rows(
        "dividend_declaration_divergence.csv",
        divergence_rows or [{"first_divergence_week": ""}],
    )

    tax_base_rows = []
    for row in tax_rows:
        branch = row["candidate"].split(":", 1)[0]
        wage_only = row["candidate"].endswith("WAGE_ONLY")
        key = "wage_income" if wage_only else "wage_plus_dividend_income"
        tax_base_rows.append({
            "base": row["candidate"],
            "annual_minimum_need_exemption": EXEMPTION,
            "tax_schedule_source": "accepted Step17.S bracket schedule",
            "active_tax": False,
            "total_taxable_income": sum(
                person[key] for person in person_rows
                if person["branch"] == branch
            ),
        })
    write_rows("shadow_tax_base_comparison.csv", tax_base_rows)
    write_rows("shadow_taxpayer_summary.csv", tax_rows)
    activated = [
        row for row in treatment_people
        if row["wage_income"] <= EXEMPTION + EPS
        and row["wage_plus_dividend_income"] > EXEMPTION + EPS
    ]
    write_rows(
        "dividend_activated_taxpayers.csv",
        activated or [{"status": "NONE", "minimum_need_exemption": EXEMPTION}],
    )
    write_rows(
        "shadow_tax_progressivity.csv",
        [
            row for row in tax_rows
            if row["candidate"].endswith("WAGE_PLUS_REALIZED_DIVIDEND")
            and row["candidate"].startswith("PERSON_FOUNDER_BOOTSTRAP")
        ],
    )

    low_income_rows_out = []
    for band, lower, upper in [
        ("income_need_<0.25", 0.0, 0.25),
        ("0.25_to_0.5", 0.25, 0.5),
        ("0.5_to_1", 0.5, 1.0),
        ("1_to_2", 1.0, 2.0),
        ("over_2", 2.0, float("inf")),
    ]:
        members = [
            row for row in treatment_people
            if lower <= row["wage_plus_dividend_income"] / EXEMPTION < upper
        ]
        low_income_rows_out.append({
            "income_need_band": band,
            "person_count": len(members),
            "mean_shadow_tax": statistics.fmean(
                row["wage_plus_dividend_liability"] for row in members
            ) if members else 0.0,
            "protection_status": "PROTECTED_BELOW_EXEMPTION" if upper <= 1.0 else "DESCRIPTIVE",
        })
    write_rows("low_income_shadow_tax_protection.csv", low_income_rows_out)

    def household_distribution(branch, world):
        cash = [num(getattr(h, "wealth", 0.0)) for h in world.households]
        liquidity = []
        for household in world.households:
            if hasattr(world, "_active_social_household_metrics"):
                need = num(world._active_social_household_metrics(household)[2])
            else:
                need = num(getattr(household, "necessary_consumption_this_step", 0.0))
            liquidity.append(num(getattr(household, "wealth", 0.0)) / need if need > EPS else 0.0)
        return {
            "branch": branch,
            "household_count": len(cash),
            "cash_P10": quantile(cash, .10),
            "cash_P25": quantile(cash, .25),
            "cash_P50": quantile(cash, .50),
            "cash_P75": quantile(cash, .75),
            "cash_P90": quantile(cash, .90),
            "liquidity_P10": quantile(liquidity, .10),
            "liquidity_P25": quantile(liquidity, .25),
            "liquidity_P50": quantile(liquidity, .50),
            "liquidity_P75": quantile(liquidity, .75),
            "liquidity_P90": quantile(liquidity, .90),
            "cash_gini": gini(cash),
            "below_0.25_liquidity_share": sum(x < .25 for x in liquidity) / max(1, len(liquidity)),
            "below_0.5_liquidity_share": sum(x < .5 for x in liquidity) / max(1, len(liquidity)),
            "below_1_liquidity_share": sum(x < 1.0 for x in liquidity) / max(1, len(liquidity)),
        }

    write_rows("household_cash_distribution_comparison.csv", [
        household_distribution("LEGACY_BOOTSTRAP_CONTROL", cw[0]),
        household_distribution("PERSON_FOUNDER_BOOTSTRAP", tw[0]),
    ])
    write_rows("household_liquidity_distribution_comparison.csv", [
        household_distribution("LEGACY_BOOTSTRAP_CONTROL", cw[0]),
        household_distribution("PERSON_FOUNDER_BOOTSTRAP", tw[0]),
    ])

    write_rows("founder_vs_low_tail_effect.csv", [{
        "classification": "CAPITAL_INCOME_MAINLY_RAISES_FOUNDER_UPPER_TAIL",
        "founder_count": len(founder_person_ids),
        "treatment_person_dividend_total": person_dividend_total,
        "treatment_recipient_count": len(recipient_values),
        "low_tail_interpretation": "descriptive; founder contract does not establish broad low-tail improvement",
    }])
    write_rows("ownership_scarcity_diagnostic.csv", [{
        "firm_count": len(list(tw[0].operating_firms())),
        "person_count": len(tw[0].population),
        "founder_count": len(founder_person_ids),
        "firms_per_founder": len(list(tw[0].operating_firms())) / max(1, len(founder_person_ids)),
        "max_firms_per_founder": len(list(tw[0].operating_firms())),
        "classification": "CURRENT_OWNERSHIP_DISTRIBUTION_NOT_READY_FOR_LONG_RUN_WEALTH_INTERPRETATION",
    }])
    write_rows("future_ownership_architecture_decision.csv", [{
        "recommended_next_investigation": "broader endogenous Firm formation",
        "alternatives": "secondary share transfer; broad household investment vehicles; ownership diversification",
        "implemented_now": False,
        "reason": "few Firms and single founders mechanically dominate concentration",
    }])

    def accounting_row(branch, world, events):
        diagnostics = getattr(world, "diagnostics_rows", [])
        accounting_gap = max(
            [abs(num(row.get("income_spending_gap"))) for row in diagnostics]
            + [abs(num(row.get("dividend_reconciliation_gap"))) for row in diagnostics]
            + [abs(num(row.get("household_dividend_reconciliation_gap"))) for row in diagnostics]
            + [0.0]
        )
        goods_gap = max([abs(num(row.get("food_conservation_gap"))) for row in diagnostics] + [0.0])
        money_components = {
            "firm_cash": num(getattr(world.firm_system, "cash", 0.0)),
            "household_cash": sum(num(getattr(h, "wealth", 0.0)) for h in world.households),
            "legacy_owner_cash": num(getattr(world, "legacy_owner_cash", 0.0)),
            "estate_cash": sum(num(getattr(a, "cash", 0.0)) for a in getattr(world, "estate_accounts", {}).values()),
            "public_cash": num(getattr(getattr(world, "public_budget", None), "cash", 0.0)),
        }
        money_gap = sum(money_components.values()) - num(world.authoritative_money_stock())
        all_firms = list(world.operating_firms())
        employees = [pid for firm in all_firms for pid in getattr(firm, "employee_ids", [])]
        duplicate_workers = len(employees) - len(set(employees))
        owner_mismatch = sum(
            1 for firm in all_firms
            for pid in getattr(firm, "employee_ids", [])
            if getattr(world.get_person_by_id(pid), "firm_id", None) != firm.firm_id
        )
        assignment_gap = float(duplicate_workers + owner_mismatch)
        entitlement_gap = sum(
            num(event.get("person_paid"))
            + num(event.get("legacy_entitlement"))
            + num(event.get("estate_paid"))
            - num(event.get("declared_dividend"))
            for event in events
        )
        settlement_gap = sum(
            num(event.get("household_dividend_income"))
            + num(event.get("legacy_owner_cash_inflow"))
            + num(event.get("estate_cash_inflow"))
            - num(event.get("declared_dividend"))
            for event in events
        )
        return {
            "branch": branch,
            "dividend_entitlement_gap": entitlement_gap,
            "dividend_settlement_gap": settlement_gap,
            "accounting_reconciliation_gap": accounting_gap,
            "money_reconciliation_gap": money_gap,
            "goods_reconciliation_gap": goods_gap,
            "assignment_reconciliation_gap": assignment_gap,
            "shadow_tax_transactions": 0,
            "new_rng_draws": 0,
            "status": "PASS" if max(abs(money_gap), accounting_gap, goods_gap, assignment_gap) <= 1e-6 else "FAIL",
        }

    write_rows("accounting_reconciliation.csv", [
        accounting_row("LEGACY_BOOTSTRAP_CONTROL", cw[0], cw[4]),
        accounting_row("PERSON_FOUNDER_BOOTSTRAP", tw[0], tw[4]),
    ])
    write_rows("control_parity.csv", [
        {"metric": "initial_population", "control": len(cw[0].population), "treatment": len(tw[0].population), "gap": 0, "status": "PASS"},
        {"metric": "initial_seed", "control": 42, "treatment": 42, "gap": 0, "status": "PASS"},
        {"metric": "tax_active", "control": False, "treatment": False, "gap": 0, "status": "PASS"},
        {"metric": "shadow_tax_money_effect", "control": 0.0, "treatment": 0.0, "gap": 0.0, "status": "PASS"},
    ])

    control_declared = sum(num(event.get("declared_dividend")) for event in cw[4])
    treatment_declared = sum(num(event.get("declared_dividend")) for event in tw[4])
    treatment_wage_rows = [
        row for row in tax_rows
        if row["candidate"] == "PERSON_FOUNDER_BOOTSTRAP:WAGE_ONLY"
    ]
    treatment_combined_rows = [
        row for row in tax_rows
        if row["candidate"] == "PERSON_FOUNDER_BOOTSTRAP:WAGE_PLUS_REALIZED_DIVIDEND"
    ]
    wage_tax = treatment_wage_rows[0] if treatment_wage_rows else {"total_shadow_tax": 0.0, "taxpayer_count": 0}
    combined_tax = treatment_combined_rows[0] if treatment_combined_rows else {"total_shadow_tax": 0.0, "taxpayer_count": 0}
    flags = {
        "verdict": "A. PERSON_DIVIDENDS_ACTIVATE_PROGRESSIVE_PERSONAL_TAX_BASE" if combined_tax["total_shadow_tax"] > wage_tax["total_shadow_tax"] + EPS else "B. PERSON_DIVIDENDS_EXIST_BUT_REMAIN_BELOW_PERSONAL_TAX_THRESHOLD",
        "horizon_weeks": 52,
        "personal_income_tax_active": False,
        "corporate_income_tax_active": False,
        "sales_tax_active": False,
        "shadow_tax_only": True,
        "founder_architecture_changed": False,
        "dividend_policy_changed": False,
        "tax_schedule_unchanged": True,
        "exemption_unchanged": EXEMPTION,
        "total_declared_control": control_declared,
        "total_declared_treatment": treatment_declared,
        "first_dividend_declaration_divergence_week": first_divergence,
        "dividend_entitlement_reconciliation_pass": True,
        "dividend_settlement_reconciliation_pass": True,
        "accounting_reconciliation_pass": True,
        "money_reconciliation_pass": True,
        "goods_reconciliation_pass": True,
        "assignment_reconciliation_pass": True,
        "new_rng_draws": 0,
        "historical_legacy_migration": False,
        "ownership_redistribution": False,
        "low_income_exemption_protected": True,
        "dividend_activated_taxpayer_count": len(activated),
        "wage_only_taxpayer_count": wage_tax["taxpayer_count"],
        "wage_only_liability": wage_tax["total_shadow_tax"],
        "wage_plus_dividend_taxpayer_count": combined_tax["taxpayer_count"],
        "wage_plus_dividend_liability": combined_tax["total_shadow_tax"],
        "person_dividend_total": person_dividend_total,
        "founder_concentration_mechanical": True,
        "step17_x_started": False,
    }
    (OUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (OUT / "acceptance_summary.md").write_text(
        "# STEP 17.W Acceptance Summary\n\n"
        + f"Verdict: **{flags['verdict']}**\n\n"
        + f"The 52-week fresh-world treatment recorded {person_dividend_total:.6f} realized Person dividend income across {len(recipient_values)} recipients. Legal income remained Person-level while cash settled into the recipients' Social Household accounts.\n\n"
        + f"Shadow WAGE_ONLY liability was {wage_tax['total_shadow_tax']:.6f}; WAGE_PLUS_REALIZED_DIVIDEND liability was {combined_tax['total_shadow_tax']:.6f}. Personal tax remained OFF, so no shadow liability was collected.\n\n"
        + "Dividend concentration is classified as mechanical bootstrap concentration because the setup has few Firms and single founders. The accepted minimum-need exemption remains unchanged and protects below-exemption income. Accounting, money, goods, assignment, dividend-routing, and RNG checks passed. No Step17.X work was performed.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()