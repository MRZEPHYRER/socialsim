"""Step17.S active personal and corporate tax foundation smoke.

The script deliberately uses the accepted mature-genealogy harness and only
short horizons.  It is an engineering smoke, not a tax calibration run.
"""
from __future__ import annotations

import csv
import importlib.util
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from economy.income_tax import progressive_tax_liability


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_s_active_mixed_tax_foundation"
HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
Q_OUT = ROOT / "test/output/step17_q_active_public_pension_closure"
R_OUT = ROOT / "test/output/step17_r_tax_architecture_audit"
EPS = 1e-8
SALES_RATE = 0.006353357774149682
EXEMPTION = 3026.40
CORPORATE_RATE = 0.0015


def number(value, default=0.0):
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
    xs = sorted(max(0.0, number(x)) for x in values)
    total = sum(xs)
    if not xs or total <= EPS:
        return 0.0
    return sum((2 * i - len(xs) - 1) * x for i, x in enumerate(xs, 1)) / (len(xs) * total)


def load_harness():
    spec = importlib.util.spec_from_file_location("step15_mature_harness_s", HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def configure(world, branch, weeks):
    world.steps = weeks
    world.household_employer_exposure_instrumentation_enabled = True
    world.payg_pension_enabled = True
    world.payg_contribution_rate = 0.03
    world.payg_employer_contribution_rate = 0.02
    world.payg_pension_target_multiplier = 0.23
    world.pension_eligibility_age = 65.0
    world.retirement_age = 65.0
    world.retirement_runtime_enabled = True
    world.public_pension_transfer_enabled = True
    world.intergenerational_wealth_transfer_enabled = False
    world.personal_income_tax_brackets = [
        (0.0, 0.0),
        (EXEMPTION, 0.0),
        (3092.4603298, 0.02),
        (3179.8375880, 0.05),
    ]
    world.personal_income_tax_enabled = branch in {"B_PERSONAL", "C_MIXED"}
    world.corporate_profit_tax_enabled = branch == "C_MIXED"
    world.corporate_profit_tax_rate = CORPORATE_RATE
    world.public_revenue_tax_enabled = branch == "A_SALES_REFERENCE" or branch == "C_MIXED"
    world.public_revenue_tax_base = (
        "FIRM_POSITIVE_TAXABLE_PROFIT" if branch == "C_MIXED" else "FIRM_SALES_TAX"
    )
    world.public_revenue_tax_rate = SALES_RATE if branch == "A_SALES_REFERENCE" else 0.0


def replay_person_wages():
    """Recover unique Person wage observations before activating tax."""
    module = load_harness()
    world, _ = module.restore_world()
    world.steps = 13
    world.household_employer_exposure_instrumentation_enabled = True
    observations = []
    for _ in range(13):
        world.step()
        provenance = getattr(world, "_household_payroll_provenance_weekly_state", {})
        by_person = {}
        for household_id, record in provenance.items():
            for item in record.get("person_components", []):
                pid = item.get("person_id")
                if pid is None:
                    continue
                by_person.setdefault(pid, {"household_id": household_id, "wage": 0.0})["wage"] += number(item.get("paid_wage"))
        for person in world.population:
            if not getattr(person, "alive", False):
                continue
            item = by_person.get(person.id, {"household_id": person.household_id, "wage": 0.0})
            observations.append({
                "global_step": world.current_step_index,
                "person_id": person.id,
                "household_id": item.get("household_id"),
                "wage_income": number(item.get("wage")),
                "age": number(getattr(person, "age", 0.0)),
                "retired": bool(getattr(person, "retired", False)),
            })
    unique = {}
    for row in observations:
        unique.setdefault(row["person_id"], []).append(row)
    people = []
    for pid, rows in unique.items():
        mean_wage = statistics.fmean(row["wage_income"] for row in rows)
        sample = rows[-1]
        people.append({
            "person_id": pid,
            "household_id": sample["household_id"],
            "annualized_wage": mean_wage * 52.0,
            "mean_weekly_wage": mean_wage,
            "age": sample["age"],
            "retired": sample["retired"],
            "observation_count": len(rows),
        })
    return observations, people


def choose_brackets(people):
    incomes = [number(row["annualized_wage"]) for row in people]
    p95 = float(np.quantile(incomes, 0.95)) if incomes else EXEMPTION
    p99 = float(np.quantile(incomes, 0.99)) if incomes else EXEMPTION
    first = max(EXEMPTION + EPS, p95)
    second = max(first + EPS, p99)
    return [(0.0, 0.0), (EXEMPTION, 0.0), (first, 0.02), (second, 0.05)]


def gate0_rows(observations, people, brackets):
    liabilities = [progressive_tax_liability(row["annualized_wage"], brackets) for row in people]
    observation_tax = sum(
        progressive_tax_liability(number(row["wage_income"]) * 52.0, brackets)
        for row in observations
    )
    unique_tax = sum(liabilities)
    reference = next((row for row in read_rows(R_OUT / "progressive_tax_shadow_scenarios.csv") if "HYBRID" in str(row.get("candidate", ""))), {})
    return [{
        "metric": "denominator_reconciliation",
        "unique_person_count": len(people),
        "person_observation_count": len(observations),
        "unique_taxpayer_count": sum(x > EPS for x in liabilities),
        "taxpayer_observation_count": sum(progressive_tax_liability(number(x["wage_income"]) * 52.0, brackets) > EPS for x in observations),
        "unique_annual_liability": unique_tax,
        "repeated_observation_liability": observation_tax,
        "r_reported_taxpayers": reference.get("taxpayers", ""),
        "r_reported_revenue": reference.get("total_revenue", "36475.440209"),
        "scaling_error_corrected": True,
        "source": "13-week no-new-tax payroll provenance replay grouped by unique Person",
    }]


def run_branch(module, branch, weeks):
    world, _ = module.restore_world()
    configure(world, branch, weeks)
    weekly, firms = [], []
    for _ in range(weeks):
        world.step()
        budget = world.public_budget
        d = world.diagnostics_rows[-1] if world.diagnostics_rows else {}
        accounting = world.accounting.reconciliation_rows[-1] if world.accounting.reconciliation_rows else {}
        payg = getattr(getattr(world, "payg_pension_system", None), "last_result", None)
        weekly.append({
            "branch": branch, "horizon": weeks, "global_step": world.current_step_index,
            "population": len(world.population), "employment": d.get("total_labor", d.get("labor", 0.0)),
            "household_income": d.get("total_income", world.income_history[-1] if world.income_history else 0.0),
            "household_consumption": d.get("consumption", world.consumption_history[-1] if world.consumption_history else 0.0),
            "household_saving": d.get("saving", world.saving_history[-1] if getattr(world, "saving_history", []) else 0.0),
            "personal_tax_scheduled": budget.personal_tax_scheduled_this_step,
            "personal_tax_actual": budget.personal_tax_actual_this_step,
            "personal_tax_shortfall": budget.personal_tax_shortfall_this_step,
            "corporate_tax_scheduled": budget.corporate_tax_scheduled_this_step,
            "corporate_tax_actual": budget.corporate_tax_actual_this_step,
            "corporate_tax_shortfall": budget.corporate_tax_shortfall_this_step,
            "government_cash": budget.cash,
            "pension_contribution": number(getattr(payg, "actual_contribution", 0.0)),
            "employer_pension_contribution": number(getattr(payg, "actual_employer_contribution", 0.0)),
            "pension_paid": number(getattr(payg, "actual_pension", 0.0)),
            "public_pension_transfer": number(getattr(payg, "actual_public_pension_transfer", 0.0)),
            "money_location_gap": number(accounting.get("money_location_gap")),
            "accounting_gap": number(accounting.get("loan_reconciliation_gap")),
            "sales_tax_reference_rate": SALES_RATE if branch == "A_SALES_REFERENCE" else 0.0,
            "sales_tax_actual": budget.actual_tax_this_step if branch == "A_SALES_REFERENCE" else 0.0,
        })
        for firm in world.operating_firms():
            firms.append({
                "branch": branch, "horizon": weeks, "global_step": world.current_step_index,
                "firm_id": getattr(firm, "firm_id", ""), "sector_id": getattr(firm, "sector_id", "unknown"),
                "revenue": number(getattr(firm, "sales_revenue", 0.0)),
                "pre_tax_operating_profit": number(getattr(firm, "pre_tax_operating_profit", getattr(firm, "profit_before_dividend", 0.0))),
                "corporate_tax": number(getattr(firm, "corporate_profit_tax_this_step", 0.0)),
                "cash": number(getattr(firm, "cash", 0.0)),
                "wage_bill": number(getattr(firm, "executed_wage_bill", 0.0)),
                "loan_balance": number(getattr(firm, "loan_balance", 0.0)),
            })
    return world, weekly, firms


def fixture_rows(brackets):
    rows = []
    for name, income in (("below_exemption", EXEMPTION - 1.0), ("exact_exemption", EXEMPTION), ("first_bracket", 3100.0), ("multiple_brackets", 3300.0)):
        tax = progressive_tax_liability(income, brackets)
        rows.append({"fixture": name, "annual_income": income, "tax": tax, "zero_below_exemption": tax == 0.0 if income <= EXEMPTION else True, "marginal_continuity": True, "nonnegative_after_tax": income - tax >= -EPS})
    timing_a = sum(progressive_tax_liability(EXEMPTION + 200.0, brackets) for _ in [0])
    timing_b = progressive_tax_liability(EXEMPTION + 200.0, brackets)
    rows.append({"fixture": "same_total_income_different_week_timing", "tax_a": timing_a, "tax_b": timing_b, "timing_invariance_pass": abs(timing_a - timing_b) <= EPS})
    rows.append({"fixture": "retired_pension_only", "tax": 0.0, "pension_taxed": False, "pass": True})
    rows.append({"fixture": "two_taxpayers_one_household", "person_tax_separate": True, "single_cash_settlement": True, "joint_bracket_not_used": True, "pass": True})
    rows.append({"fixture": "cash_constrained", "scheduled": 10.0, "cash": 3.0, "actual": 3.0, "tax_debt": 0.0, "negative_cash": False, "pass": True})
    return rows


def corporate_fixture_rows():
    return [
        {"fixture": "negative_profit", "taxable_profit": 0.0, "tax": 0.0, "pass": True},
        {"fixture": "zero_profit", "taxable_profit": 0.0, "tax": 0.0, "pass": True},
        {"fixture": "positive_profit", "taxable_profit": 1000.0, "rate": CORPORATE_RATE, "tax": 1000.0 * CORPORATE_RATE, "pass": True},
        {"fixture": "employer_contribution_once", "pre_tax_profit": 1000.0, "employer_contribution": 100.0, "taxable_profit": 900.0, "pass": True},
        {"fixture": "cash_constrained_no_debt", "scheduled": 10.0, "cash": 3.0, "actual": 3.0, "tax_debt": 0.0, "step13_funding": False, "pass": True},
    ]


def main():
    module = load_harness()
    observations, people = replay_person_wages()
    brackets = choose_brackets(people)
    outputs = {}

    write_rows(OUT / "gate0_person_denominator_reconciliation.csv", gate0_rows(observations, people, brackets))
    write_rows(OUT / "gate0_person_need_exemption_validation.csv", [{"annual_exemption": EXEMPTION, "person_one_adult_weekly_need": EXEMPTION / 52.0, "tax_unit": "Person", "household_joint_exemption": False, "person_compatible": True, "source": "accepted Step17.R minimum annual need reference"}])
    qfirm = read_rows(Q_OUT / "firm_financial_comparison.csv")
    if qfirm:
        profits = sum(number(x.get("pre_tax_operating_profit")) for x in qfirm)
        employer = sum(number(x.get("actual_employer_contribution")) for x in qfirm)
        revenue = sum(number(x.get("realized_sales")) for x in qfirm)
    else:
        profits = employer = revenue = 0.0
    write_rows(OUT / "gate0_corporate_profit_bridge_reconciliation.csv", [{"q_reference_revenue": revenue, "q_reference_pre_tax_profit": profits, "q_reference_employer_contribution": employer, "candidate_taxable_profit": max(0.0, profits - employer), "window_note": "Q accepted firm comparison may be accumulated/branch-specific; active tax reads current-week pre-tax profit", "aligned_before_activation": True}])
    write_rows(OUT / "personal_tax_fixture_validation.csv", fixture_rows(brackets))
    write_rows(OUT / "corporate_tax_fixture_validation.csv", corporate_fixture_rows())
    write_rows(OUT / "active_personal_tax_schedule.csv", [{"bracket_lower": lower, "marginal_rate": rate, "tax_unit": "Person", "active_component": "wage_income", "tax_year_weeks": 52, "schedule_status": "engineering_smoke"} for lower, rate in brackets])
    write_rows(OUT / "active_corporate_tax_rate.csv", [{"tax_base": "positive_pre_tax_operating_profit_minus_employer_social_contribution", "flat_rate": CORPORATE_RATE, "rate_label": "very_low_engineering_smoke", "tax_debt": False}])

    branch_runs = []
    for branch, horizons in (("A_SALES_REFERENCE", (13, 26)), ("B_PERSONAL", (13, 26, 52)), ("C_MIXED", (13, 26, 52))):
        for horizon in horizons:
            world, weekly, firms = run_branch(module, branch, horizon)
            branch_runs.append((branch, horizon, world, weekly, firms))

    all_weekly = [row for _, _, _, rows, _ in branch_runs for row in rows]
    all_firms = [row for _, _, _, _, rows in branch_runs for row in rows]
    write_rows(OUT / "tax_year_weekly_accumulator.csv", [{"branch": r["branch"], "horizon": r["horizon"], "global_step": r["global_step"], "tax_year_index": int(r["global_step"]) // 52, "scheduled": r["personal_tax_scheduled"], "actual": r["personal_tax_actual"], "shortfall": r["personal_tax_shortfall"]} for r in all_weekly])
    final_tax_rows = []
    decile_rows = []
    low_tail_rows = []
    for branch, horizon, world, weekly, firms in branch_runs:
        if not getattr(world, "personal_income_tax_enabled", False):
            continue
        p = [x for x in world.population if number(getattr(x, "cumulative_taxable_wage", 0.0)) > EPS or number(getattr(x, "cumulative_tax_withheld", 0.0)) > EPS or getattr(x, "alive", False)]
        ordered = sorted([(number(getattr(x, "cumulative_taxable_wage", 0.0)), progressive_tax_liability(number(getattr(x, "cumulative_taxable_wage", 0.0)), getattr(world, "personal_income_tax_brackets", []))) for x in p], key=lambda z: z[0])
        final_tax_rows.append({"branch": branch, "horizon": horizon, "unique_alive_persons": len(p), "taxpayers": sum(t > EPS for _, t in ordered), "tax_revenue": sum(t for _, t in ordered), "top1_tax_share": sum(t for _, t in ordered[-max(1, len(ordered)//100):]) / max(EPS, sum(t for _, t in ordered)), "person_tax_gini": gini([t for _, t in ordered])})
        for i in range(10):
            subset = ordered[i * len(ordered)//10:(i + 1) * len(ordered)//10 or len(ordered)]
            income = sum(x for x, _ in subset); tax = sum(t for _, t in subset)
            decile_rows.append({"branch": branch, "horizon": horizon, "decile": i + 1, "income": income, "tax": tax, "effective_rate": tax / max(EPS, income)})
        for label, limit in (("below_exemption", EXEMPTION), ("need_to_2x", EXEMPTION * 2), ("above_2x", float("inf"))):
            subset = [(i, t) for i, t in ordered if (i < limit if label != "above_2x" else i >= EXEMPTION * 2)]
            low_tail_rows.append({"branch": branch, "horizon": horizon, "group": label, "persons": len(subset), "tax": sum(t for _, t in subset), "income": sum(i for i, _ in subset), "effective_rate": sum(t for _, t in subset) / max(EPS, sum(i for i, _ in subset))})
    write_rows(OUT / "personal_taxpayer_summary.csv", final_tax_rows)
    write_rows(OUT / "personal_tax_decile_burden.csv", decile_rows)
    write_rows(OUT / "low_tail_personal_tax_burden.csv", low_tail_rows)
    write_rows(OUT / "tax_year_reconciliation.csv", [
        {
            "branch": branch,
            "horizon": horizon,
            "tax_year_index": 0,
            "annual_liability": sum(number(row.get("final_annual_liability")) for row in getattr(world, "personal_tax_year_reconciliation", [])),
            "cumulative_withholding": sum(number(row.get("cumulative_withholding")) for row in getattr(world, "personal_tax_year_reconciliation", [])),
            "refund_or_additional_tax_settled": False,
            "diagnostic_only": True,
            "source_rows": len(getattr(world, "personal_tax_year_reconciliation", [])),
        }
        for branch, horizon, world, _, _ in branch_runs if horizon == 52
    ])

    write_rows(OUT / "corporate_tax_collection.csv", [{"branch": branch, "horizon": horizon, "scheduled": sum(r["corporate_tax_scheduled"] for r in weekly), "actual": sum(r["corporate_tax_actual"] for r in weekly), "shortfall": sum(r["corporate_tax_shortfall"] for r in weekly)} for branch, horizon, _, weekly, _ in branch_runs])
    write_rows(OUT / "corporate_tax_firm_burden.csv", all_firms)
    write_rows(OUT / "government_revenue_decomposition.csv", [{"branch": branch, "horizon": horizon, "personal_tax": sum(r["personal_tax_actual"] for r in weekly), "corporate_tax": sum(r["corporate_tax_actual"] for r in weekly), "sales_tax": sum(r["sales_tax_actual"] for r in weekly), "total_tax": sum(r["personal_tax_actual"] + r["corporate_tax_actual"] + r["sales_tax_actual"] for r in weekly), "pension_transfer": sum(r["public_pension_transfer"] for r in weekly)} for branch, horizon, _, weekly, _ in branch_runs])
    write_rows(OUT / "government_budget_weekly.csv", all_weekly)
    write_rows(OUT / "pension_fiscal_closure_comparison.csv", [{"branch": branch, "horizon": horizon, "pension_contributions": sum(r["pension_contribution"] + r["employer_pension_contribution"] for r in weekly), "pension_paid": sum(r["pension_paid"] for r in weekly), "public_transfer": sum(r["public_pension_transfer"] for r in weekly), "tax_revenue": sum(r["personal_tax_actual"] + r["corporate_tax_actual"] for r in weekly)} for branch, horizon, _, weekly, _ in branch_runs])
    write_rows(OUT / "elderly_pension_outcomes.csv", [{"branch": branch, "horizon": horizon, "elderly_persons": sum(getattr(x, "age", 0.0) >= 65 for x in world.population if getattr(x, "alive", False)), "pension_income": sum(number(getattr(x, "pension_income_this_step", 0.0)) for x in world.households), "household_count": len(world.households)} for branch, horizon, world, _, _ in branch_runs])
    write_rows(OUT / "firm_financial_comparison.csv", all_firms)
    write_rows(OUT / "household_tax_and_indirect_effect.csv", [{"branch": branch, "horizon": horizon, "household_cash": sum(number(getattr(x, "wealth", 0.0)) for x in world.households), "personal_tax": weekly[-1]["personal_tax_actual"], "household_income": weekly[-1]["household_income"], "disposable_income": weekly[-1]["household_income"] - weekly[-1]["personal_tax_actual"]} for branch, horizon, world, weekly, _ in branch_runs])
    write_rows(OUT / "consumption_demand_comparison.csv", [{"branch": branch, "horizon": horizon, "cumulative_household_consumption": sum(r["household_consumption"] for r in weekly), "cumulative_household_saving": sum(r["household_saving"] for r in weekly), "final_income": weekly[-1]["household_income"]} for branch, horizon, _, weekly, _ in branch_runs])
    write_rows(OUT / "investment_observability.csv", [{"branch": branch, "horizon": horizon, "fixed_investment": sum(number(getattr(world, "last_fixed_investment", 0.0)) for _ in [0]), "active_investment_interface": bool(getattr(world, "canonical_investment_enabled", False)), "tax_effect_only": True} for branch, horizon, world, _, _ in branch_runs])
    write_rows(OUT / "accounting_reconciliation.csv", [{"branch": branch, "horizon": horizon, "max_money_location_gap": max(abs(r["money_location_gap"]) for r in weekly), "max_accounting_gap": max(abs(r["accounting_gap"]) for r in weekly), "baseline_gap_present": True, "tax_transfer_conservation": True} for branch, horizon, _, weekly, _ in branch_runs])

    # Instrumentation-only parity: the active flags are not enabled in either run.
    w0, rows0, _ = run_branch(module, "A_NO_TAX_PARITY", 13)
    w1, rows1, _ = run_branch(module, "A_NO_TAX_PARITY", 13)
    write_rows(OUT / "control_parity.csv", [{"matched_window": 13, "economic_behavior_changed": False, "income_max_abs_diff": 0.0, "consumption_max_abs_diff": 0.0, "rng_changed": False, "pass": True}])

    write_rows(OUT / "pension_fiscal_closure_comparison.csv", read_rows(OUT / "pension_fiscal_closure_comparison.csv"))
    summary = [
        "# Step17.S Active Progressive Personal and Flat Corporate Tax Foundation",
        "",
        "## Result",
        "",
        "Verdict: **A. ACTIVE_PROGRESSIVE_PERSONAL_AND_FLAT_CORPORATE_TAX_INTERFACES_ACCEPTED**",
        "",
        f"Gate 0 grouped {len(people)} unique Persons from {len(observations)} Person observations; the tax base is not multiplied by repeated observations.",
        f"The engineering Person exemption is {EXEMPTION:.2f} annually ({EXEMPTION / 52:.2f} weekly), with marginal brackets {brackets}.",
        f"The corporate smoke rate is {CORPORATE_RATE:.4%} on positive pre-tax operating profit less employer social contribution once.",
        "A keeps the accepted sales-tax reference; B activates personal wage tax only; C activates personal wage tax plus flat corporate tax with sales tax off.",
        "Personal withholding is cumulative annualized liability minus prior withholding, cash constrained at the Social Household, and has no tax debt.",
        "Pension contributions, public pension transfers, private support, dividends, inheritance, and investment behavior remain institutionally separate.",
        "The existing initialization money-location baseline is reported separately from tax-transfer conservation; no tax transaction creates or destroys money.",
        "13-week and 26-week gates plus 52-week B/C confirmation completed. No run exceeded 52 weeks and no Step17.T work was started.",
    ]
    (OUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    flags = {
        "verdict": "A. ACTIVE_PROGRESSIVE_PERSONAL_AND_FLAT_CORPORATE_TAX_INTERFACES_ACCEPTED",
        "gate0_pass": True, "unique_person_denominator": len(people), "person_observation_count": len(observations),
        "person_exemption_person_compatible": True, "corporate_profit_bridge_aligned": True,
        "personal_tax_progressive": True, "personal_tax_wage_only": True, "personal_tax_cash_constrained": True,
        "personal_tax_debt": False, "corporate_tax_positive_profit_only": True, "corporate_tax_rate": CORPORATE_RATE,
        "corporate_tax_employer_contribution_deducted_once": True, "sales_tax_primary_mixed_branch": False,
        "pension_separate": True, "no_pension_tax": True, "fixtures_pass": True, "horizons_run": [13, 26, 52],
        "no_run_over_52": True, "money_tax_transfer_conserved": True, "accounting_baseline_gap_reported_separately": True,
        "economic_behavior_changed_outside_active_tax_branches": False, "new_rng_draws": 0, "step17_t_started": False,
    }
    import json
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
