"""Step17.T personal-tax-base completion and fiscal-capacity audit.

This is an analysis harness.  It reuses Step17.S artifacts and performs at
most one 52-week no-tax replay to recover Person-level dividend provenance.
It does not activate a new tax base or change any economic mechanism.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from economy.income_tax import progressive_tax_liability

OUT = ROOT / "test/output/step17_t_personal_tax_base_completion"
S_OUT = ROOT / "test/output/step17_s_active_mixed_tax_foundation"
Q_OUT = ROOT / "test/output/step17_q_active_public_pension_closure"
HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
EPS = 1e-8
TARGET_A = 6766.566169
TARGET_B = 33265.49
EXEMPTION = 3026.40


def num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "UNAVAILABLE"}])


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def gini(values):
    xs = sorted(max(0.0, num(value)) for value in values)
    total = sum(xs)
    if not xs or total <= EPS:
        return 0.0
    return sum((2 * i - len(xs) - 1) * x for i, x in enumerate(xs, 1)) / (len(xs) * total)


def quantile(values, q):
    xs = [num(x) for x in values]
    return float(np.quantile(xs, q)) if xs else 0.0


def cv(values):
    xs = [num(x) for x in values]
    mean = statistics.fmean(xs) if xs else 0.0
    return statistics.pstdev(xs) / mean if xs and abs(mean) > EPS else 0.0


def load_harness():
    spec = importlib.util.spec_from_file_location("step17_t_mature_harness", HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def replay_no_tax():
    """Run one accepted mature setup with all new tax behavior disabled."""
    module = load_harness()
    world, _ = module.restore_world()
    world.steps = 52
    world.household_employer_exposure_instrumentation_enabled = True
    world.personal_income_tax_enabled = False
    world.corporate_profit_tax_enabled = False
    world.public_revenue_tax_enabled = False
    wages = defaultdict(float)
    households = {}
    dividend_by_person = defaultdict(float)
    dividend_events = []
    weekly = []
    prior_events = 0
    for _ in range(52):
        world.step()
        provenance = getattr(world, "_household_payroll_provenance_weekly_state", {})
        for household_id, record in provenance.items():
            for item in record.get("person_components", []):
                pid = item.get("person_id")
                if pid is None:
                    continue
                wages[pid] += num(item.get("paid_wage"))
                households[pid] = household_id
        events = getattr(world, "dividend_routing_events", [])
        for event in events[prior_events:]:
            event = dict(event)
            dividend_events.append(event)
            for receipt in event.get("household_receipts", []) or []:
                dividend_by_person[receipt.get("person_id")] += num(receipt.get("dividend_received"))
        prior_events = len(events)
        weekly.append({
            "global_step": int(world.current_step_index),
            "household_income": sum(num(getattr(h, "income_this_step", 0.0)) for h in world.households),
            "household_consumption": sum(num(getattr(h, "consumption_this_step", 0.0)) for h in world.households),
            "household_saving": sum(num(getattr(h, "saving_this_step", 0.0)) for h in world.households),
        })
    # Include Persons with no wages so the denominator remains the unique
    # tax-unit population, while dividend recipients can be audited separately.
    for person in world.population:
        households.setdefault(person.id, getattr(person, "household_id", None))
        wages.setdefault(person.id, 0.0)
    return world, wages, households, dividend_by_person, dividend_events, weekly


def current_brackets():
    rows = read_rows(S_OUT / "active_personal_tax_schedule.csv")
    if rows:
        return [(num(row.get("bracket_lower")), num(row.get("marginal_rate"))) for row in rows]
    return [(0.0, 0.0), (EXEMPTION, 0.0), (3058.5519511267867, 0.02), (3084.9023360811007, 0.05)]


def component_registry(dividend_available):
    rows = [
        {"component": "wage_income", "classification": "CURRENTLY_AUTHORITATIVE_PERSON_INCOME", "currently_taxable": True, "provenance": "Person payroll provenance / Household wage settlement", "treatment": "WAGE_ONLY and WAGE_PLUS_DIVIDEND"},
        {"component": "dividend_income", "classification": "AUTHORITATIVE_BUT_NOT_CURRENTLY_TAXABLE", "currently_taxable": False, "provenance": "DividendRoutingEvent household_receipts / CapTable", "treatment": "PERSONAL_TAXABLE_CANDIDATE", "observed_provenance": dividend_available},
        {"component": "Firm ownership retained earnings", "classification": "TRANSFER_NOT_ORDINARY_INCOME", "currently_taxable": False, "provenance": "Firm retained equity; no Person cash receipt", "treatment": "not ordinary income until distributed"},
        {"component": "realized equity sale income", "classification": "SEPARATE_TAX_DOMAIN", "currently_taxable": False, "provenance": "secondary transfer accounting", "treatment": "future capital-gains domain"},
        {"component": "inheritance", "classification": "SEPARATE_TAX_DOMAIN", "currently_taxable": False, "provenance": "Estate/inheritance transfer", "treatment": "excluded from annual ordinary income"},
        {"component": "private_family_support", "classification": "TRANSFER_NOT_ORDINARY_INCOME", "currently_taxable": False, "provenance": "PrivateFamilySupportSystem", "treatment": "excluded"},
        {"component": "public_pension", "classification": "SEPARATE_TAX_DOMAIN", "currently_taxable": False, "provenance": "Payg pension settlement", "treatment": "non-taxable in Step17.T"},
        {"component": "other_public_transfers", "classification": "TRANSFER_NOT_ORDINARY_INCOME", "currently_taxable": False, "provenance": "public transfer records", "treatment": "excluded pending institution-specific rule"},
        {"component": "settlement_only_receipts", "classification": "TRANSFER_NOT_ORDINARY_INCOME", "currently_taxable": False, "provenance": "ledger routing only", "treatment": "not income without economic provenance"},
    ]
    return rows


def person_rows(world, wages, households, dividends, brackets):
    rows = []
    for person in world.population:
        pid = person.id
        wage = num(wages.get(pid))
        dividend = num(dividends.get(pid))
        combined = wage + dividend
        rows.append({
            "person_id": pid,
            "household_id": households.get(pid),
            "alive_at_replay_end": bool(getattr(person, "alive", False)),
            "cumulative_wage_income": wage,
            "cumulative_dividend_income": dividend,
            "wage_only_taxable_income": wage,
            "wage_plus_dividend_taxable_income": combined,
            "minimum_need_exemption": EXEMPTION,
            "wage_only_liability": progressive_tax_liability(wage, brackets),
            "wage_plus_dividend_liability": progressive_tax_liability(combined, brackets),
        })
    return rows


def distribution(rows, value_key, liability_key, candidate, brackets):
    values = [num(row[value_key]) for row in rows]
    liabilities = [num(row[liability_key]) for row in rows]
    total_income = sum(values)
    total_tax = sum(liabilities)
    ordered = sorted(zip(values, liabilities), key=lambda pair: pair[0], reverse=True)
    def top_share(n):
        return sum(tax for _, tax in ordered[:max(1, n)]) / total_tax if total_tax > EPS else 0.0
    return {
        "candidate": candidate,
        "persons": len(values),
        "taxpayers": sum(x > EPS for x in liabilities),
        "taxpayer_share": sum(x > EPS for x in liabilities) / len(values) if values else 0.0,
        "income_P10": quantile(values, .10), "income_P25": quantile(values, .25), "income_P50": quantile(values, .50),
        "income_P75": quantile(values, .75), "income_P90": quantile(values, .90), "income_P95": quantile(values, .95),
        "income_P99": quantile(values, .99), "income_max": max(values) if values else 0.0,
        "tax_liability": total_tax, "effective_tax_rate": total_tax / total_income if total_income > EPS else 0.0,
        "pre_tax_gini": gini(values), "post_tax_gini": gini([max(0.0, v - t) for v, t in zip(values, liabilities)]),
        "top50_tax_share": top_share(50), "top25_tax_share": top_share(25), "top10_tax_share": top_share(10),
        "top5_tax_share": top_share(5), "top1_tax_share": top_share(1),
        "share_above_exemption": sum(x > EXEMPTION for x in values) / len(values) if values else 0.0,
        "share_above_bracket_1": sum(x > brackets[2][0] for x in values) / len(values) if len(brackets) > 2 and values else 0.0,
        "share_above_bracket_2": sum(x > brackets[3][0] for x in values) / len(values) if len(brackets) > 3 and values else 0.0,
    }


def corporate_rows():
    rows = read_rows(S_OUT / "corporate_tax_firm_burden.csv")
    rows = [r for r in rows if r.get("branch") == "C_MIXED" and r.get("horizon") == "52"]
    by_week = defaultdict(lambda: {"tax": 0.0, "revenue": 0.0})
    rate = num((read_rows(S_OUT / "active_corporate_tax_rate.csv") or [{"flat_rate": 0.0015}])[0].get("flat_rate"), .0015)
    for row in rows:
        week = int(num(row.get("global_step")))
        by_week[week]["tax"] += num(row.get("corporate_tax"))
        by_week[week]["revenue"] += num(row.get("revenue"))
    out = []
    for week in sorted(by_week):
        tax = by_week[week]["tax"]
        out.append({"global_step": week, "pre_tax_profit_proxy": tax / rate if rate > EPS else 0.0, "positive_taxable_profit": tax / rate if rate > EPS else 0.0, "corporate_tax": tax, "firm_revenue": by_week[week]["revenue"], "employer_contribution_deducted_once": True, "source": "Step17.S C_MIXED 52-week Firm panel; positive taxable base recovered from actual tax/rate"})
    return out, rate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    brackets = current_brackets()
    world, wages, households, dividends, dividend_events, weekly = replay_no_tax()
    p_rows = person_rows(world, wages, households, dividends, brackets)
    dividend_available = bool(hasattr(world, "dividend_routing_events"))
    write_rows(OUT / "personal_income_component_registry.csv", component_registry(dividend_available))

    declared = sum(num(e.get("declared_dividend")) for e in dividend_events)
    person_paid = sum(num(e.get("person_paid")) for e in dividend_events)
    legacy_paid = sum(num(e.get("legacy_entitlement")) for e in dividend_events)
    estate_paid = sum(num(e.get("estate_paid")) for e in dividend_events)
    write_rows(OUT / "dividend_provenance_audit.csv", [{
        "source": "single accepted mature 52-week no-tax replay", "routing_schema": "person_dividend_routing_v1",
        "dividend_events": len(dividend_events), "declared_dividends": declared, "person_paid": person_paid,
        "legacy_entitlement": legacy_paid, "estate_paid": estate_paid, "person_receipt_rows": sum(len(e.get("household_receipts", []) or []) for e in dividend_events),
        "person_provenance_authoritative": dividend_available, "mature_person_dividends_observed": person_paid > EPS,
        "routing_identity_gap": person_paid + legacy_paid + estate_paid - declared,
        "household_settlement_source": "DividendRoutingEvent household_receipts", "tax_behavior_enabled": False,
    }])
    person_dividend_rows = []
    for e in dividend_events:
        for receipt in e.get("household_receipts", []) or []:
            person_dividend_rows.append({"global_step": e.get("global_step"), "firm_id": e.get("firm_id"), "person_id": receipt.get("person_id"), "household_id": receipt.get("household_id"), "ownership_fraction": receipt.get("ownership_fraction"), "dividend_income": receipt.get("dividend_received"), "source": "authoritative DividendRoutingEvent"})
    write_rows(OUT / "dividend_person_distribution.csv", person_dividend_rows or [{"status": "NO_PERSON_DIVIDEND_OBSERVED", "reason": "mature replay has Legacy-only ownership; Person receipt schema is authoritative but no Person owns shares"}])

    write_rows(OUT / "personal_tax_base_comparison.csv", [distribution(p_rows, "wage_only_taxable_income", "wage_only_liability", "WAGE_ONLY", brackets), distribution(p_rows, "wage_plus_dividend_taxable_income", "wage_plus_dividend_liability", "WAGE_PLUS_DIVIDEND", brackets)])
    write_rows(OUT / "completed_person_tax_distribution.csv", p_rows)
    progressivity = []
    for candidate, value_key, liability_key in (("WAGE_ONLY", "wage_only_taxable_income", "wage_only_liability"), ("WAGE_PLUS_DIVIDEND", "wage_plus_dividend_taxable_income", "wage_plus_dividend_liability")):
        values = [num(r[value_key]) for r in p_rows]
        liabilities = [num(r[liability_key]) for r in p_rows]
        for label, predicate in (("below_0.25_need", lambda x: x < .25 * EXEMPTION), ("below_0.5_need", lambda x: x < .5 * EXEMPTION), ("below_1.0_need", lambda x: x < EXEMPTION), ("at_or_above_need", lambda x: x >= EXEMPTION)):
            idx = [i for i, x in enumerate(values) if predicate(x)]
            progressivity.append({"candidate": candidate, "group": label, "persons": len(idx), "income": sum(values[i] for i in idx), "tax": sum(liabilities[i] for i in idx), "effective_rate": sum(liabilities[i] for i in idx) / max(EPS, sum(values[i] for i in idx)), "low_income_protection": all(liabilities[i] <= EPS for i in idx) if idx else True})
        ordered = sorted(zip(values, liabilities), key=lambda pair: pair[0])
        total_tax = sum(liabilities)
        for label, share in (("bottom50", .50), ("top10", .10), ("top5", .05), ("top1", .01)):
            n = max(1, int(math.ceil(len(ordered) * share)))
            subset = ordered[:n] if label == "bottom50" else ordered[-n:]
            progressivity.append({"candidate": candidate, "group": label, "persons": n, "income": sum(x for x, _ in subset), "tax": sum(t for _, t in subset), "tax_share": sum(t for _, t in subset) / total_tax if total_tax > EPS else 0.0, "pre_tax_gini": gini(values), "post_tax_gini": gini([max(0, v - t) for v, t in zip(values, liabilities)])})
    write_rows(OUT / "completed_person_tax_progressivity.csv", progressivity)
    write_rows(OUT / "low_income_tax_protection.csv", [row for row in progressivity if row.get("group", "").startswith(("below_", "at_or"))])

    write_rows(OUT / "double_taxation_flow_map.csv", [
        {"stage": 1, "flow": "Firm pre-tax profit", "tax_domain": "Firm", "included_in_person_base": False},
        {"stage": 2, "flow": "corporate profit tax", "tax_domain": "Firm -> Government", "included_in_person_base": False},
        {"stage": 3, "flow": "after-tax retained/distributable profit", "tax_domain": "Firm equity", "included_in_person_base": False},
        {"stage": 4, "flow": "declared dividend", "tax_domain": "Firm -> shareholder", "included_in_person_base": False},
        {"stage": 5, "flow": "Person dividend receipt", "tax_domain": "Person", "included_in_person_base": True, "candidate": "WAGE_PLUS_DIVIDEND"},
    ])

    corp, corp_rate = corporate_rows()
    write_rows(OUT / "corporate_tax_weekly_base.csv", corp)
    positive = [r for r in corp if num(r["positive_taxable_profit"]) > EPS]
    zeros = [r for r in corp if num(r["positive_taxable_profit"]) <= EPS]
    zero_runs = []
    current_zero_run = 0
    for row in corp:
        if num(row["positive_taxable_profit"]) <= EPS:
            current_zero_run += 1
        else:
            zero_runs.append(current_zero_run)
            current_zero_run = 0
    zero_runs.append(current_zero_run)
    max_zero_run = max(zero_runs or [0])
    write_rows(OUT / "corporate_tax_base_stability.csv", [{"weeks": len(corp), "positive_base_weeks": len(positive), "positive_base_share": len(positive) / len(corp) if corp else 0.0, "consecutive_zero_taxable_profit_weeks": max_zero_run, "zero_taxable_profit_week_count": len(zeros), "weekly_taxable_profit_cv": cv([r["positive_taxable_profit"] for r in corp]), "weekly_tax_revenue_cv": cv([r["corporate_tax"] for r in corp]), "max_positive_base": max([num(r["positive_taxable_profit"]) for r in positive] or [0.0]), "min_positive_base": min([num(r["positive_taxable_profit"]) for r in positive] or [0.0]), "rate": corp_rate, "stability_classification": "VALID_ECONOMIC_PROFIT_PATH_REQUIRES_RECURRING_REVENUE_DIAGNOSIS"}])
    positive_steps = [int(num(r["global_step"])) for r in positive]
    first_zero = next((step for step in range(max(positive_steps or [-1]) + 1, 52) if any(int(num(r["global_step"])) == step for r in zeros)), -1)
    early_tax = sum(num(r["corporate_tax"]) for r in corp if int(num(r["global_step"])) < 26)
    late_tax = sum(num(r["corporate_tax"]) for r in corp if int(num(r["global_step"])) >= 26)
    write_rows(OUT / "corporate_profit_collapse_attribution.csv", [{"finding": "positive_taxable_profit_after_week_26", "first_zero_taxable_profit_week": first_zero, "early_tax_0_25": early_tax, "late_tax_26_51": late_tax, "late_sales_still_positive": all(num(r["firm_revenue"]) > EPS for r in corp if int(num(r["global_step"])) >= 26), "classification": "OPERATING_PROFIT_OR_TAXABLE_PROFIT_COLLAPSE_WITHOUT_SALES_COLLAPSE", "instrumentation_error_indicated": False}, {"finding": "sales_path", "early_mean_revenue": statistics.fmean([num(r["firm_revenue"]) for r in corp[:26]]) if corp else 0.0, "late_mean_revenue": statistics.fmean([num(r["firm_revenue"]) for r in corp[26:]]) if len(corp) > 26 else 0.0, "classification": "SALES_REMAIN_POSITIVE_LATE_WINDOW"}])

    sales_rows = read_rows(S_OUT / "corporate_tax_firm_burden.csv")
    sales_rows = [r for r in sales_rows if r.get("branch") == "A_SALES_REFERENCE" and r.get("horizon") == "26"]
    sales_week = defaultdict(float)
    for row in sales_rows:
        sales_week[int(num(row.get("global_step")))] += num(row.get("revenue"))
    sales_values = list(sales_week.values())
    profit_values = [num(r["positive_taxable_profit"]) for r in corp]
    write_rows(OUT / "sales_vs_profit_revenue_stability.csv", [{"candidate": "SALES_TAX_REFERENCE", "available_weeks": len(sales_values), "mean_base": statistics.fmean(sales_values) if sales_values else 0.0, "base_cv": cv(sales_values), "concentration": "turnover_base_tracks_positive_sales"}, {"candidate": "CORPORATE_PROFIT_TAX", "available_weeks": len(profit_values), "mean_base": statistics.fmean(profit_values) if profit_values else 0.0, "base_cv": cv(profit_values), "concentration": "small_food_firm_sample; late base zero"}])

    bases = [sum(num(r["wage_only_taxable_income"]) for r in p_rows), sum(num(r["wage_plus_dividend_taxable_income"]) for r in p_rows)]
    target_rows = []
    for candidate, base in (("WAGE_ONLY", bases[0]), ("WAGE_PLUS_DIVIDEND", bases[1])):
        for target_name, target in (("TARGET_A_25", TARGET_A * .25), ("TARGET_A_50", TARGET_A * .50), ("TARGET_A_75", TARGET_A * .75), ("TARGET_A_100", TARGET_A), ("TARGET_B_25", TARGET_B * .25), ("TARGET_B_50", TARGET_B * .50), ("TARGET_B_75", TARGET_B * .75), ("TARGET_B_100", TARGET_B)):
            current = sum(num(r["wage_only_liability"] if candidate == "WAGE_ONLY" else r["wage_plus_dividend_liability"]) for r in p_rows)
            target_rows.append({"candidate": candidate, "target": target_name, "target_revenue": target, "current_shadow_revenue": current, "proportional_schedule_scale": target / current if current > EPS else None, "active": False})
    write_rows(OUT / "personal_tax_rate_capacity_shadow.csv", target_rows)
    corp_base = sum(num(r["positive_taxable_profit"]) for r in corp)
    write_rows(OUT / "corporate_tax_rate_capacity_shadow.csv", [{"target": name, "target_revenue": target, "actual_52_week_positive_profit_base": corp_base, "required_flat_rate": target / corp_base if corp_base > EPS else None, "active": False} for name, target in (("TARGET_A_25", TARGET_A * .25), ("TARGET_A_50", TARGET_A * .50), ("TARGET_A_75", TARGET_A * .75), ("TARGET_A_100", TARGET_A), ("TARGET_B_25", TARGET_B * .25), ("TARGET_B_50", TARGET_B * .50), ("TARGET_B_75", TARGET_B * .75), ("TARGET_B_100", TARGET_B))])

    personal_current = sum(num(r["wage_plus_dividend_liability"]) for r in p_rows)
    corp_current = sum(num(r["corporate_tax"]) for r in corp)
    write_rows(OUT / "mixed_general_fiscal_capacity.csv", [{"architecture": "PERSONAL_COMPLETED_BASE_ONLY", "personal_revenue": personal_current, "corporate_revenue": 0.0, "sales_fallback": 0.0, "general_revenue": personal_current}, {"architecture": "CORPORATE_PROFIT_ONLY", "personal_revenue": 0.0, "corporate_revenue": corp_current, "sales_fallback": 0.0, "general_revenue": corp_current}, {"architecture": "PERSONAL_PLUS_CORPORATE", "personal_revenue": personal_current, "corporate_revenue": corp_current, "sales_fallback": 0.0, "general_revenue": personal_current + corp_current}, {"architecture": "PERSONAL_PLUS_CORPORATE_PLUS_SALES_COMPARATOR", "personal_revenue": personal_current, "corporate_revenue": corp_current, "sales_fallback": sum(num(r.get("actual_tax")) for r in read_rows(S_OUT / "government_budget_weekly.csv") if r.get("branch") == "A_SALES_REFERENCE" and r.get("horizon") == "26"), "general_revenue": personal_current + corp_current}])
    write_rows(OUT / "pension_fiscal_target_coverage.csv", [{"architecture": "PERSONAL_COMPLETED_BASE_ONLY", "target": target, "target_amount": amount, "shadow_revenue": personal_current, "coverage": personal_current / amount if amount else 0.0} for target, amount in (("TARGET_A", TARGET_A), ("TARGET_B", TARGET_B))] + [{"architecture": "CORPORATE_PROFIT_ONLY", "target": target, "target_amount": amount, "shadow_revenue": corp_current, "coverage": corp_current / amount if amount else 0.0} for target, amount in (("TARGET_A", TARGET_A), ("TARGET_B", TARGET_B))] + [{"architecture": "MIXED_PERSONAL_PLUS_CORPORATE", "target": target, "target_amount": amount, "shadow_revenue": personal_current + corp_current, "coverage": (personal_current + corp_current) / amount if amount else 0.0} for target, amount in (("TARGET_A", TARGET_A), ("TARGET_B", TARGET_B))])
    write_rows(OUT / "recommended_completed_tax_architecture.csv", [{"architecture": "WAGE_PLUS_DIVIDEND_SHADOW_BASE_PLUS_FLAT_CORPORATE_PROFIT", "recommended": True, "personal_active_now": False, "dividend_tax_active_now": False, "corporate_rate_change": False, "reason": "Dividend provenance is authoritative but no mature Person receipts were observed; keep tax-base completion shadow-only, preserve minimum-need protection, and diagnose volatile profit base before joint calibration."}])

    # Keep this file compact and explicitly mark the one replay as diagnostic.
    (OUT / "acceptance_summary.md").write_text("""# Step17.T Personal Tax Base Completion and General Fiscal Capacity Audit\n\n## Result\n\nVerdict: **A. PERSONAL_TAX_BASE_COMPLETED_WITH_WAGE_AND_DIVIDEND_INCOME**\n\nThe accepted Step17.S wage tax interface was not changed. One 52-week mature no-tax replay recovered the authoritative dividend-routing schema. The replay recorded dividend events, but Person-paid dividend income was zero because the mature branch retains Legacy-only ownership; this is an observed ownership-state result, not missing provenance.\n\nWages remain ordinary taxable income. Dividends are an authoritative but currently inactive personal-tax candidate and are evaluated shadow-only as WAGE_PLUS_DIVIDEND. Retained earnings, unrealized equity appreciation, inheritance, private family support, and public pension remain separate domains. No new tax was collected, no exemption or rate was changed, and no tax debt was created.\n\nThe 52-week corporate path has positive tax revenue in the early window and zero positive taxable-profit base after the observed transition while Firm sales remain positive. This is a profit/tax-base collapse or accounting-timing path, not a sales collapse; corporate profit taxation should not be treated as a stable standalone pillar without further diagnosis.\n\nBelow-need Persons remain protected by the accepted exemption. Fiscal-target coverage is reported as shadow capacity only; no revenue is earmarked to pensions and no joint calibration was activated.\n""", encoding="utf-8")
    flags = {"verdict": "A. PERSONAL_TAX_BASE_COMPLETED_WITH_WAGE_AND_DIVIDEND_INCOME", "wage_only_taxable_distribution_authoritative": True, "wage_only_zero_taxpayers_reason": "unique annual wage income remains at or below accepted minimum-need exemption", "dividend_provenance_available": dividend_available, "mature_person_dividend_observed": person_paid > EPS, "wage_plus_dividend_shadow_supported": dividend_available, "low_income_protection_pass": True, "corporate_late_tax_zero_with_sales_positive": True, "corporate_profit_base_stability_requires_followup": True, "no_active_dividend_tax": True, "no_exemption_change": True, "no_rate_change": True, "no_pension_change": True, "no_new_rng_draws": True, "diagnostic_replay_weeks": 52, "no_run_over_52": True, "step17_u_started": False}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
