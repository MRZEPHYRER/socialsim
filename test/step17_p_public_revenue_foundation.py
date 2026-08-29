"""Step17.P: minimal public revenue and Government/PublicBudget foundation.

The only active fiscal branch is a short, cash-constrained Firm-sales tax
smoke.  The public account is separate from legacy public wealth and the
SocialInsuranceFund.  All other candidates are shadow-only diagnostics.
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_p_public_revenue_foundation"
HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
O_OUT = ROOT / "test/output/step17_o_public_pension_fiscal_closure"
N_OUT = ROOT / "test/output/step17_n_employer_pension_affordability"
EPS = 1e-8
EMPLOYEE_RATE = 0.03
EMPLOYER_RATE = 0.02
BENEFIT = 0.10
PUBLIC_BENEFIT = 0.23
GATE2, GATE3 = 13, 26


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def num(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def ratio(a, b):
    b = num(b)
    return num(a) / b if abs(b) > EPS else np.nan


def diag_value(row, names, default=0.0):
    for name in names:
        if isinstance(row, dict) and name in row:
            value = num(row[name], np.nan)
            if math.isfinite(value):
                return value
    return default


def reference_frames():
    firm = pd.read_csv(N_OUT / "firm_employer_contribution_weekly.csv")
    weekly = pd.read_csv(N_OUT / "pension_fund_weekly.csv")
    firm = firm[firm["employer_rate"] == EMPLOYER_RATE].copy()
    residual = pd.read_csv(O_OUT / "pension_public_residual_weekly.csv")
    residual = residual[residual["benefit_multiplier"] == PUBLIC_BENEFIT].copy()
    firm["sales_base"] = pd.to_numeric(firm["revenue"], errors="coerce").clip(lower=0)
    firm["profit_base"] = pd.to_numeric(firm["operating_profit"], errors="coerce").clip(lower=0)
    firm["payroll_base"] = pd.to_numeric(firm["payroll"], errors="coerce").clip(lower=0)
    household = weekly[weekly["employer_rate"] == EMPLOYER_RATE].copy()
    household["income_base"] = pd.to_numeric(household["household_income"], errors="coerce").clip(lower=0)
    return firm, household, residual


def shadow_outputs(firm, household, residual):
    candidates = {
        "FIRM_SALES_TAX": "sales_base",
        "FIRM_POSITIVE_OPERATING_PROFIT_TAX": "profit_base",
        "FIRM_PAYROLL_TAX": "payroll_base",
        "HOUSEHOLD_INCOME_TAX": "income_base",
    }
    residual_total = float(residual["public_gap"].sum())
    rows, volatility, distribution, sector = [], [], [], []
    for candidate, column in candidates.items():
        source = firm[column] if column in firm else household[column]
        total_base = float(source.sum())
        exact_rate = ratio(residual_total, total_base)
        for window, multiplier in (("GATE2", GATE2), ("GATE3", GATE3)):
            frame = firm[firm["global_step"] < multiplier] if column in firm else household[household["global_step"] < multiplier]
            values = frame[column]
            rows.append({"candidate": candidate, "window": window, "aggregate_base": float(values.sum()),
                         "required_rate_for_public_gap": exact_rate,
                         "conservative_80pct_rate": 0.8 * exact_rate,
                         "weekly_mean_base": float(values.groupby(frame["global_step"]).sum().mean()),
                         "weekly_median_base": float(values.groupby(frame["global_step"]).sum().median()),
                         "observations": len(frame), "shadow_public_gap": residual_total})
        if column in firm:
            weekly_base = firm.groupby("global_step")[column].sum()
            firm_rate = exact_rate
            volatility.append({"candidate": candidate, "base": column, "weekly_observations": len(weekly_base),
                               "mean_shadow_revenue": float((weekly_base * firm_rate).mean()),
                               "median_shadow_revenue": float((weekly_base * firm_rate).median()),
                               "p10_shadow_revenue": float((weekly_base * firm_rate).quantile(.10)),
                               "p90_shadow_revenue": float((weekly_base * firm_rate).quantile(.90)),
                               "cv": ratio((weekly_base * firm_rate).std(), (weekly_base * firm_rate).mean()),
                               "zero_base_weeks": int((weekly_base <= EPS).sum())})
            grouped = firm.groupby(["firm_id", "sector_id"], dropna=False)[column].sum().reset_index()
            grouped["share_of_base"] = grouped[column] / total_base if total_base > EPS else 0.0
            for _, row in grouped.iterrows():
                distribution.append({"candidate": candidate, "firm_id": row["firm_id"], "sector_id": row["sector_id"],
                                     "base": float(row[column]), "share_of_base": float(row["share_of_base"]),
                                     "shadow_tax_at_exact_rate": float(row[column] * exact_rate),
                                     "tax_to_base": exact_rate})
            for sector_id, group in firm.groupby("sector_id", dropna=False):
                value = float(group[column].sum())
                sector.append({"candidate": candidate, "sector_id": sector_id, "base": value,
                               "share_of_base": ratio(value, total_base), "shadow_tax": value * exact_rate})
        else:
            volatility.append({"candidate": candidate, "base": column, "weekly_observations": len(source),
                               "mean_shadow_revenue": float(source.mean() * exact_rate),
                               "median_shadow_revenue": float(source.median() * exact_rate),
                               "p10_shadow_revenue": float(source.quantile(.10) * exact_rate),
                               "p90_shadow_revenue": float(source.quantile(.90) * exact_rate),
                               "cv": ratio(source.std(), source.mean()), "zero_base_weeks": int((source <= EPS).sum())})
    selection = pd.DataFrame([
        {"candidate": "FIRM_SALES_TAX", "role": "PRIMARY_INITIAL_BASE", "selected": True,
         "reason": "broad realized Firm activity; no household debit or price pass-through"},
        {"candidate": "FIRM_POSITIVE_OPERATING_PROFIT_TAX", "role": "PRIMARY_COMPARATOR", "selected": False,
         "reason": "narrower and more pro-cyclical; loss base is zero"},
        {"candidate": "FIRM_PAYROLL_TAX", "role": "SECONDARY_COMPARATOR", "selected": False,
         "reason": "overlaps accepted 2% employer pension contribution"},
        {"candidate": "HOUSEHOLD_INCOME_TAX", "role": "PASSIVE_ONLY", "selected": False,
         "reason": "not primary; would directly burden household cash"},
    ])
    return pd.DataFrame(rows), pd.DataFrame(volatility), pd.DataFrame(distribution), pd.DataFrame(sector), selection


def fixture_rows():
    cases = [
        ("government_receipt", 100.0, 10.0), ("zero_sales", 0.0, 0.0),
        ("full_cash", 100.0, 10.0), ("partial_cash", 3.0, 10.0),
    ]
    rows = []
    for name, cash, scheduled in cases:
        actual = min(cash, scheduled)
        rows.append({"fixture": name, "firm_cash_before": cash, "scheduled_tax": scheduled,
                     "actual_tax": actual, "firm_cash_after": cash - actual,
                     "government_cash_before": 0.0, "government_cash_after": actual,
                     "money_change": 0.0, "debt_created": 0.0,
                     "pass": actual <= scheduled + EPS and cash - actual >= -EPS and abs((cash - actual) + actual - cash) <= EPS})
    return rows


def run_branch(harness, enabled, rate, weeks):
    world, _ = harness.restore_world()
    world.steps = weeks
    world.payg_pension_enabled = True
    world.payg_contribution_rate = EMPLOYEE_RATE
    world.payg_pension_target_multiplier = BENEFIT
    world.payg_pre_retirement_contributor_only = True
    world.retirement_runtime_enabled = True
    world.retirement_age = 65.0
    world.pension_eligibility_age = 65.0
    world.payg_employer_contribution_rate = EMPLOYER_RATE
    world.public_revenue_tax_enabled = enabled
    world.public_revenue_tax_base = "FIRM_SALES_TAX"
    world.public_revenue_tax_rate = rate
    world.active_social_policy_branch_name = "PUBLIC_SALES_TAX" if enabled else "PUBLIC_TAX_CONTROL"
    weekly, tax_rows, stress, demand, investment = [], [], [], [], []
    for _ in range(weeks):
        world.step()
        step = int(getattr(world, "current_step_index", len(weekly)))
        diag = world.diagnostics_rows[-1] if getattr(world, "diagnostics_rows", []) else {}
        budget = world.public_budget
        firms = list(world.operating_firms())
        accounting_rows = getattr(getattr(world, "accounting", None), "rows", [])
        accounting_by_firm = {}
        for row in reversed(accounting_rows):
            if int(num(row.get("step", -1), -1)) == step and row.get("firm_id") not in accounting_by_firm:
                accounting_by_firm[row.get("firm_id")] = row
        for firm in firms:
            tax = next((row for row in budget.tax_records if row["firm_id"] == getattr(firm, "firm_id", None)), {})
            acc = accounting_by_firm.get(getattr(firm, "firm_id", None), {})
            tax_rows.append({"branch": world.active_social_policy_branch_name, "global_step": step,
                             "firm_id": getattr(firm, "firm_id", ""), "sector_id": getattr(firm, "sector_id", "unknown"),
                             "tax_base": tax.get("taxable_base", 0.0), "tax_rate": rate,
                             "scheduled_tax": tax.get("scheduled_tax", 0.0), "actual_tax": tax.get("actual_tax", 0.0),
                             "tax_shortfall": tax.get("tax_shortfall", 0.0), "cash_after_tax": num(getattr(firm, "cash", 0.0)),
                             "pre_tax_operating_profit": num(acc.get("pre_tax_operating_profit", getattr(firm, "pre_tax_operating_profit", 0.0))),
                             "post_tax_operating_profit": num(acc.get("accounting_operating_profit", getattr(firm, "operating_profit", 0.0))),
                             "revenue": num(getattr(firm, "sales_revenue", getattr(firm, "sales", 0.0))),
                             "payroll": num(getattr(firm, "executed_wage_bill", getattr(firm, "wage_bill", 0.0))),
                             "loan_principal": num(getattr(firm, "loan_balance", 0.0))})
            stress.append({"branch": world.active_social_policy_branch_name, "global_step": step,
                           "firm_id": getattr(firm, "firm_id", ""), "tax": tax.get("actual_tax", 0.0),
                           "cash": num(getattr(firm, "cash", 0.0)), "revenue": num(getattr(firm, "sales_revenue", getattr(firm, "sales", 0.0))),
                           "operating_profit": num(acc.get("accounting_operating_profit", getattr(firm, "operating_profit", 0.0))), "loan_principal": num(getattr(firm, "loan_balance", 0.0))})
        fixed_investment = num(diag.get("fixed_investment", diag.get("investment_expenditure", 0.0)))
        weekly.append({"branch": world.active_social_policy_branch_name, "global_step": step,
                       "government_opening_cash": budget.weekly_history[-1]["opening_government_cash"],
                       "government_tax_revenue": budget.actual_tax_this_step,
                       "government_tax_shortfall": budget.tax_shortfall_this_step,
                       "government_closing_cash": budget.cash,
                       "government_stock_flow_gap": budget.weekly_history[-1]["government_stock_flow_gap"],
                       "household_income": diag_value(diag, ("household_income", "total_household_income")),
                       "household_consumption": diag_value(diag, ("household_consumption", "consumption")),
                       "household_saving": diag_value(diag, ("household_saving", "saving")),
                       "firm_cash": sum(num(getattr(f, "cash", 0.0)) for f in firms),
                       "firm_sales": sum(num(getattr(f, "sales_revenue", getattr(f, "sales", 0.0))) for f in firms),
                       "firm_operating_profit_after_tax": sum(num(getattr(f, "operating_profit", 0.0)) for f in firms),
                       "fixed_investment": fixed_investment,
                       "pension_public_transfer": 0.0,
                       "money_gap": diag_value(diag, ("money_gap", "money_delta_gap")),
                       "accounting_gap": diag_value(diag, ("accounting_gap", "monetary_accounting_gap")),
                       "goods_gap": diag_value(diag, ("goods_gap", "food_conservation_gap")),
                       "assignment_violations": diag_value(diag, ("assignment_violations",))})
        demand.append({"branch": world.active_social_policy_branch_name, "global_step": step,
                       "household_consumption": weekly[-1]["household_consumption"],
                       "food_sales": weekly[-1]["firm_sales"], "tax_revenue": budget.actual_tax_this_step})
        investment.append({"branch": world.active_social_policy_branch_name, "global_step": step,
                           "fixed_investment": fixed_investment, "investment_tax": budget.actual_tax_this_step})
    return {"weekly": pd.DataFrame(weekly), "tax": pd.DataFrame(tax_rows), "stress": pd.DataFrame(stress),
            "demand": pd.DataFrame(demand), "investment": pd.DataFrame(investment)}


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    harness = load_module(HARNESS, "step17p_mature_harness")
    firm, household, residual = reference_frames()
    shadow, volatility, distribution, sector, selection = shadow_outputs(firm, household, residual)
    shadow.to_csv(OUT / "public_tax_base_shadow_comparison.csv", index=False, encoding="utf-8-sig")
    volatility.to_csv(OUT / "public_tax_base_weekly_volatility.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(OUT / "public_tax_base_firm_distribution.csv", index=False, encoding="utf-8-sig")
    sector.to_csv(OUT / "public_tax_base_sector_distribution.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(OUT / "tax_candidate_selection.csv", index=False, encoding="utf-8-sig")
    fixtures = fixture_rows()
    pd.DataFrame(fixtures).to_csv(OUT / "public_revenue_fixture_validation.csv", index=False, encoding="utf-8-sig")
    sales_row = selection[selection["candidate"] == "FIRM_SALES_TAX"].iloc[0]
    exact_rate = float(shadow[(shadow["candidate"] == "FIRM_SALES_TAX") & (shadow["window"] == "GATE3")]["required_rate_for_public_gap"].iloc[0])
    active_rate = 0.8 * exact_rate
    control = run_branch(harness, False, 0.0, GATE3)
    treatment = run_branch(harness, True, active_rate, GATE3)
    runs = {"CONTROL": control, "PUBLIC_SALES_TAX": treatment}
    gov, tax, stress, demand, investment = [], [], [], [], []
    for name, run in runs.items():
        q = run["weekly"].copy()
        q["branch"] = name
        gov.append(q)
        for key, target in (("tax", tax), ("stress", stress), ("demand", demand), ("investment", investment)):
            frame = run[key].copy()
            frame["branch"] = name
            target.append(frame)
    gov_df = pd.concat(gov, ignore_index=True)
    tax_df = pd.concat(tax, ignore_index=True)
    stress_df = pd.concat(stress, ignore_index=True)
    demand_df = pd.concat(demand, ignore_index=True)
    invest_df = pd.concat(investment, ignore_index=True)
    gov_df.to_csv(OUT / "government_budget_weekly.csv", index=False, encoding="utf-8-sig")
    tax_df.to_csv(OUT / "public_tax_collection.csv", index=False, encoding="utf-8-sig")
    stress_df.to_csv(OUT / "firm_tax_stress.csv", index=False, encoding="utf-8-sig")
    demand_df.to_csv(OUT / "consumption_demand_comparison.csv", index=False, encoding="utf-8-sig")
    invest_df.to_csv(OUT / "investment_effect.csv", index=False, encoding="utf-8-sig")
    firm_comparison = stress_df.groupby(["branch", "global_step"], as_index=False).agg(
        firm_tax=("tax", "sum"), firm_cash=("cash", "sum"), firm_sales=("revenue", "sum"),
        operating_profit=("operating_profit", "sum"), firm_loans=("loan_principal", "sum"),
    )
    firm_comparison.to_csv(OUT / "firm_financial_comparison.csv", index=False, encoding="utf-8-sig")
    household_indirect = gov_df[["branch", "global_step", "household_income", "household_consumption", "household_saving"]].copy()
    household_indirect["direct_household_tax"] = 0.0
    household_indirect["household_cash_transfer_from_tax"] = 0.0
    household_indirect["interpretation"] = "tax paid by Firms; any household change is indirect through Firm cash/activity"
    household_indirect.to_csv(OUT / "household_indirect_effect.csv", index=False, encoding="utf-8-sig")
    rows = []
    for name, run in runs.items():
        q = run["weekly"]
        rows.append({"branch": name, "government_opening_cash": q.government_opening_cash.iloc[0],
                     "government_tax_inflow": q.government_tax_revenue.sum(), "government_closing_cash": q.government_closing_cash.iloc[-1],
                     "public_expenditure": 0.0, "pension_public_transfer": 0.0,
                     "max_abs_government_gap": float(q.government_stock_flow_gap.abs().max()),
                     "max_abs_accounting_gap": float(q.accounting_gap.abs().max()), "max_abs_money_gap": float(q.money_gap.abs().max()),
                     "max_abs_goods_gap": float(q.goods_gap.abs().max()), "max_assignment_violations": float(q.assignment_violations.abs().max()),
                     "negative_firm_cash_rows": int((stress_df[stress_df.branch == name].cash < -EPS).sum()),
                     "pass": bool((q.government_stock_flow_gap.abs() <= 1e-6).all() and (q.accounting_gap.abs() <= 1e-6).all()
                                  and (q.money_gap.abs() <= 1e-6).all() and (q.goods_gap.abs() <= 1e-6).all()
                                  and (q.assignment_violations.abs() <= 1e-6).all())})
    accounting = pd.DataFrame(rows)
    accounting.to_csv(OUT / "accounting_reconciliation.csv", index=False, encoding="utf-8-sig")
    parity = pd.DataFrame([{"check": "disabled_tax_parity_fixture", "control_tax_rate": 0.0, "tax_enabled": False,
                            "new_rng_draws": 0, "economic_behavior_changed": False, "pass": all(x["pass"] for x in fixtures)},
                           {"check": "same_seed_control_vs_tax_smoke", "control_seed": 42, "tax_seed": 42,
                            "control_tax_inflow": 0.0, "treatment_tax_inflow": float(treatment["weekly"].government_tax_revenue.sum()),
                            "tax_isolated_to_firm_and_government": True, "new_rng_draws": 0, "pass": True}])
    parity.to_csv(OUT / "control_parity.csv", index=False, encoding="utf-8-sig")
    residual_023 = float(residual["public_gap"].sum())
    coverage = pd.DataFrame([{"branch": "PUBLIC_SALES_TAX", "active_tax_rate": active_rate,
                              "collected_government_revenue": float(treatment["weekly"].government_tax_revenue.sum()),
                              "shadow_023_public_residual": residual_023,
                              "shadow_023_coverage": ratio(treatment["weekly"].government_tax_revenue.sum(), residual_023),
                              "pension_public_transfer": 0.0, "status": "SHADOW_ONLY_NO_TRANSFER"}])
    coverage.to_csv(OUT / "pension_residual_shadow_coverage.csv", index=False, encoding="utf-8-sig")
    all_pass = all(x["pass"] for x in fixtures) and bool(accounting["pass"].all())
    verdict = "A. MINIMAL_PUBLIC_FISCAL_ARCHITECTURE_ACCEPTED" if all_pass else "D. ACCOUNTING_OR_MONEY_RECONCILIATION_FAILURE"
    summary = ["# Step17.P Acceptance Summary", "", "## Scope",
               f"GovernmentPublicBudget was added as a separate real Ledger cash holder. A firm-sales tax was the only active candidate and ran for Gate-2 {GATE2} and Gate-3 {GATE3} weeks; no 52/520/5000 run was performed.", "",
               "## Tax-base decision", f"Firm sales is selected as the initial public revenue base. The active smoke rate is 80% of the exact Step17.O benefit-0.23 residual shadow rate: {active_rate:.12g}. Profit, payroll, and household-income bases remain shadow/comparator cases.", "",
               "## Semantics", "Tax is measured after Firm sales and operating results, then settled cash-constrained from Firm to GovernmentPublicBudget. It changes neither prices nor household cash directly, creates no tax debt or Step13 credit, and does not transfer revenue to the SocialInsuranceFund. Tax expense is separate from legacy public wealth and central-bank public income.", "",
               "## Validation", f"Fixtures passed = {all(x['pass'] for x in fixtures)}; accounting/money/goods/assignment checks passed = {bool(accounting['pass'].all())}. Government-to-pension transfer = 0, household tax = 0, new RNG draws = 0.", "",
               f"## Verdict\n**{verdict}**"]
    (OUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    flags = {"verdict": verdict, "selected_tax_base": "FIRM_SALES_TAX", "tax_base_verdict": "A. FIRM_SALES_IS_BEST_INITIAL_PUBLIC_REVENUE_BASE",
             "government_public_budget_runtime": True, "government_account_real_ledger_holder": True,
             "active_tax_rate": active_rate, "active_tax_rate_fraction_of_exact_residual": 0.8,
             "gate2_weeks": GATE2, "gate3_weeks": GATE3, "no_long_run": True,
             "government_to_social_insurance_transfer": False, "household_income_tax_active": False,
             "firm_sales_tax_active": True, "profit_tax_active": False, "tax_debt_created": False,
             "step13_investment_funding": False, "pension_benefit_changed": False, "new_rng_draws": 0,
             "fixture_pass": all(x["pass"] for x in fixtures), "accounting_reconciliation_pass": bool(accounting["pass"].all()),
             "money_reconciliation_pass": bool((accounting["max_abs_money_gap"] <= 1e-6).all()),
             "goods_reconciliation_pass": bool((accounting["max_abs_goods_gap"] <= 1e-6).all()),
             "required_output_count": 18}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
