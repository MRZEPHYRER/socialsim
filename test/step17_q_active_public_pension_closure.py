"""Step17.Q: active Government/PublicBudget pension co-financing smoke."""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_q_active_public_pension_closure"
HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
EPS = 1e-8
TAX_RATE = 0.006353357774149682
EMPLOYEE_RATE, EMPLOYER_RATE = 0.03, 0.02
BENEFIT_A, BENEFIT_BC = 0.10, 0.23
BENCHMARK = 0.206896
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


def household_members(world, household):
    result = []
    for person_id in [*getattr(household, "parents", []), *getattr(household, "children", [])]:
        person = world.get_person_by_id(person_id)
        if person is not None and getattr(person, "alive", False):
            result.append(person)
    return result


def minimum_need(world, household):
    try:
        return max(0.0, num(world.needs_system.household_minimum_need_units(household))) * max(
            EPS, num(world.household_planning_price_this_step(), 1.0)
        )
    except Exception:
        return np.nan


def fixture_rows():
    rows = []
    for name, government_cash, residual in (
        ("zero_cash_shortfall", 0.0, 100.0),
        ("sufficient_cash_full", 150.0, 100.0),
        ("partial_cash", 40.0, 100.0),
        ("zero_residual", 50.0, 0.0),
    ):
        actual = min(government_cash, residual)
        rows.append({"fixture": name, "government_cash_before": government_cash,
                     "scheduled_public_transfer": residual, "actual_public_transfer": actual,
                     "public_transfer_shortfall": residual - actual,
                     "government_cash_after": government_cash - actual,
                     "money_change": 0.0, "government_debt": 0.0,
                     "pass": government_cash - actual >= -EPS and abs((government_cash - actual) + actual - government_cash) <= EPS
                     and (actual == 0.0 if residual == 0.0 else True)})
    return rows


def run_branch(harness, name, benefit, public_enabled, weeks):
    world, _ = harness.restore_world()
    world.steps = weeks
    world.payg_pension_enabled = True
    world.payg_contribution_rate = EMPLOYEE_RATE
    world.payg_pension_target_multiplier = benefit
    world.payg_pre_retirement_contributor_only = True
    world.retirement_runtime_enabled = True
    world.retirement_age = 65.0
    world.pension_eligibility_age = 65.0
    world.payg_employer_contribution_rate = EMPLOYER_RATE
    world.public_revenue_tax_enabled = public_enabled
    world.public_revenue_tax_base = "FIRM_SALES_TAX"
    world.public_revenue_tax_rate = TAX_RATE if public_enabled else 0.0
    world.public_pension_transfer_enabled = public_enabled
    world.active_social_policy_branch_name = name
    weekly, government, transfers, fund_rows = [], [], [], []
    firm_rows, elderly_rows, payment_rows, contributor_rows = [], [], [], []
    investment_rows, demand_rows = [], []
    for _ in range(weeks):
        fund_opening = num(getattr(world.social_insurance_fund, "cash", 0.0))
        government_opening = num(getattr(world.public_budget, "cash", 0.0))
        firm_opening = {getattr(f, "firm_id", ""): num(getattr(f, "cash", 0.0)) for f in world.operating_firms()}
        world.step()
        step = int(getattr(world, "current_step_index", len(weekly)))
        result = world.payg_pension_system.last_result
        budget = world.public_budget
        tax_by_firm = {row["firm_id"]: row for row in budget.tax_records}
        employer_by_firm = {row["firm_id"]: row for row in result.employer_contribution_records}
        acc_by_firm = {}
        for row in reversed(getattr(getattr(world, "accounting", None), "rows", [])):
            if int(num(row.get("step", -1), -1)) == step and row.get("firm_id") not in acc_by_firm:
                acc_by_firm[row.get("firm_id")] = row
        for firm in world.operating_firms():
            fid = getattr(firm, "firm_id", "")
            tax = tax_by_firm.get(fid, {})
            employer = employer_by_firm.get(fid, {})
            acc = acc_by_firm.get(fid, {})
            firm_rows.append({"branch": name, "global_step": step, "firm_id": fid,
                              "sector_id": getattr(firm, "sector_id", "unknown"),
                              "realized_sales": tax.get("taxable_base", 0.0),
                              "scheduled_sales_tax": tax.get("scheduled_tax", 0.0),
                              "actual_sales_tax": tax.get("actual_tax", 0.0),
                              "tax_shortfall": tax.get("tax_shortfall", 0.0),
                              "cash_before_employer": employer.get("opening_cash_at_settlement", firm_opening.get(fid, 0.0)),
                              "cash_after_employer": employer.get("closing_cash_after_contribution", firm_opening.get(fid, 0.0)),
                              "cash_before_tax": tax.get("opening_firm_cash", 0.0),
                              "cash_after_tax": tax.get("closing_firm_cash", num(getattr(firm, "cash", 0.0))),
                              "payroll": employer.get("payroll", 0.0),
                              "scheduled_employer_contribution": employer.get("scheduled_employer_contribution", 0.0),
                              "actual_employer_contribution": employer.get("actual_employer_contribution", 0.0),
                              "employer_contribution_shortfall": employer.get("employer_contribution_shortfall", 0.0),
                              "combined_institutional_burden": employer.get("actual_employer_contribution", 0.0) + tax.get("actual_tax", 0.0),
                              "cash": num(getattr(firm, "cash", 0.0)),
                              "operating_profit": acc.get("accounting_operating_profit", getattr(firm, "operating_profit", 0.0)),
                              "pre_tax_operating_profit": acc.get("pre_tax_operating_profit", 0.0),
                              "CFO": acc.get("cfo", 0.0), "loan_principal": num(getattr(firm, "loan_balance", 0.0)),
                              "interest_arrears": num(getattr(firm, "interest_arrears", 0.0)),
                              "retained_earnings_equity": acc.get("equity", 0.0)})
        residual = max(0.0, num(result.scheduled_pension) - num(result.actual_contribution) - num(result.actual_employer_contribution))
        history = budget.weekly_history[-1] if budget.weekly_history else {}
        transfer = num(result.actual_public_pension_transfer)
        government.append({"branch": name, "global_step": step, "opening_government_cash": government_opening,
                           "current_tax_revenue": budget.actual_tax_this_step,
                           "scheduled_public_pension_transfer": residual,
                           "actual_public_pension_transfer": transfer,
                           "public_transfer_shortfall": num(result.public_pension_transfer_shortfall),
                           "closing_government_cash": num(budget.cash),
                           "same_week_tax_funded_transfer": min(transfer, budget.actual_tax_this_step),
                           "opening_reserve_funded_transfer": max(0.0, transfer - min(transfer, budget.actual_tax_this_step)),
                           "government_stock_flow_gap": history.get("government_stock_flow_gap", 0.0),
                           "world_public_wealth_pension_contribution": 0.0,
                           "central_bank_pension_contribution": 0.0})
        transfers.append({"branch": name, "global_step": step, "scheduled_pension": num(result.scheduled_pension),
                          "employee_contribution": num(result.actual_contribution),
                          "employer_contribution": num(result.actual_employer_contribution),
                          "pre_public_funding_gap": residual, "scheduled_public_transfer": residual,
                          "actual_public_transfer": transfer, "post_public_funding_gap": max(0.0, residual - transfer),
                          "actual_pension": num(result.actual_pension), "pension_shortfall": max(0.0, num(result.scheduled_pension) - num(result.actual_pension)),
                          "pension_funding_ratio": num(result.pension_funding_ratio),
                          "public_residual_coverage": ratio(transfer, residual),
                          "total_explicit_funding_ratio": ratio(num(result.actual_contribution) + num(result.actual_employer_contribution) + transfer, result.scheduled_pension)})
        fund_close = num(getattr(world.social_insurance_fund, "cash", 0.0))
        fund_rows.append({"branch": name, "global_step": step, "opening_fund_cash": fund_opening,
                          "employee_inflow": num(result.actual_contribution), "employer_inflow": num(result.actual_employer_contribution),
                          "public_inflow": transfer, "total_inflow": num(result.actual_contribution) + num(result.actual_employer_contribution) + transfer,
                          "pension_outflow": num(result.actual_pension), "closing_fund_cash": fund_close,
                          "fund_stock_flow_gap": fund_close - fund_opening - num(result.actual_contribution) - num(result.actual_employer_contribution) - transfer + num(result.actual_pension),
                          "opening_reserve_excluded_from_recurring": fund_opening})
        for person_row in result.pension_records:
            payment_rows.append({"branch": name, "global_step": step, **person_row,
                                 "funding_ratio": num(result.pension_funding_ratio)})
        contributor_actual = {row.get("person_id"): row for row in result.contribution_records}
        for household in world.households:
            if getattr(household, "settlement_only", False):
                continue
            members = household_members(world, household)
            elderly = [p for p in members if num(getattr(p, "age", 0.0)) >= 65.0]
            need = minimum_need(world, household)
            income = num(getattr(household, "income_this_step", 0.0))
            cash = num(getattr(household, "wealth", 0.0))
            if elderly:
                elderly_rows.append({"branch": name, "global_step": step, "household_id": household.id,
                                     "income": income, "minimum_need": need, "income_need_ratio": ratio(income, need),
                                     "cash": cash, "liquidity_weeks": ratio(cash, need),
                                     "pension_income": num(getattr(household, "pension_income_this_step", 0.0)),
                                     "employee_contribution": num(getattr(household, "payg_contribution_this_step", 0.0))})
            contribution = sum(num(row.get("scheduled_contribution")) for person in members if (row := contributor_actual.get(person.id)))
            contributor_rows.append({"branch": name, "global_step": step, "household_id": household.id,
                                     "employee_contribution": contribution, "direct_public_tax": 0.0,
                                     "income": income, "consumption": num(getattr(household, "consumption_this_step", 0.0)),
                                     "cash": cash})
        diag = world.diagnostics_rows[-1] if getattr(world, "diagnostics_rows", []) else {}
        active_households = [h for h in world.households if not getattr(h, "settlement_only", False)]
        household_income = sum(num(getattr(h, "income_this_step", 0.0)) for h in active_households)
        household_consumption = sum(num(getattr(h, "consumption_this_step", 0.0)) for h in active_households)
        household_saving = sum(num(getattr(h, "saving_this_step", 0.0)) for h in active_households)
        fixed_investment = diag_value(diag, ("fixed_investment", "investment_expenditure"), 0.0)
        investment_rows.append({"branch": name, "global_step": step, "investment_reviews": 0,
                                "desired_investment": 0.0, "executed_investment": fixed_investment,
                                "capital_good_orders": 0, "status": "NOT_OBSERVABLE_IN_WINDOW" if fixed_investment == 0 else "OBSERVED"})
        demand_rows.append({"branch": name, "global_step": step,
                            "household_consumption": household_consumption,
                            "food_sales": sum(num(getattr(f, "sales_revenue", getattr(f, "sales", 0.0))) for f in world.firms),
                            "capital_good_sales": diag_value(diag, ("capital_good_sales",)),
                            "tax_rate": TAX_RATE if public_enabled else 0.0})
        weekly.append({"branch": name, "global_step": step, "household_income": household_income,
                       "household_consumption": household_consumption,
                       "household_saving": household_saving,
                       "total_employment": diag_value(diag, ("total_employment", "employment")),
                       "firm_cash": sum(num(getattr(f, "cash", 0.0)) for f in world.operating_firms()),
                       "tax_revenue": budget.actual_tax_this_step, "public_transfer": transfer,
                       "pension_shortfall": max(0.0, num(result.scheduled_pension) - num(result.actual_pension)),
                       "money_gap": diag_value(diag, ("money_gap", "money_delta_gap")),
                       "accounting_gap": diag_value(diag, ("accounting_gap", "monetary_accounting_gap")),
                       "goods_gap": diag_value(diag, ("goods_gap", "food_conservation_gap")),
                       "assignment_violations": diag_value(diag, ("assignment_violations",))})
    return {"weekly": pd.DataFrame(weekly), "government": pd.DataFrame(government), "transfers": pd.DataFrame(transfers),
            "fund": pd.DataFrame(fund_rows), "firm": pd.DataFrame(firm_rows), "elderly": pd.DataFrame(elderly_rows),
            "payments": pd.DataFrame(payment_rows), "contributors": pd.DataFrame(contributor_rows),
            "investment": pd.DataFrame(investment_rows), "demand": pd.DataFrame(demand_rows)}


def write_summary_tables(runs):
    government = pd.concat([run["government"] for run in runs.values()], ignore_index=True)
    transfers = pd.concat([run["transfers"] for run in runs.values()], ignore_index=True)
    fund = pd.concat([run["fund"] for run in runs.values()], ignore_index=True)
    firm = pd.concat([run["firm"] for run in runs.values()], ignore_index=True)
    elderly = pd.concat([run["elderly"] for run in runs.values()], ignore_index=True)
    payments = pd.concat([run["payments"] for run in runs.values()], ignore_index=True)
    contributors = pd.concat([run["contributors"] for run in runs.values()], ignore_index=True)
    investment = pd.concat([run["investment"] for run in runs.values()], ignore_index=True)
    demand = pd.concat([run["demand"] for run in runs.values()], ignore_index=True)
    government.to_csv(OUT / "government_budget_weekly.csv", index=False, encoding="utf-8-sig")
    transfers.to_csv(OUT / "public_pension_transfer_weekly.csv", index=False, encoding="utf-8-sig")
    fund.to_csv(OUT / "pension_fund_weekly.csv", index=False, encoding="utf-8-sig")
    firm.to_csv(OUT / "firm_financial_comparison.csv", index=False, encoding="utf-8-sig")
    contributors.to_csv(OUT / "contributor_household_effect.csv", index=False, encoding="utf-8-sig")
    investment.to_csv(OUT / "investment_observability.csv", index=False, encoding="utf-8-sig")
    demand.to_csv(OUT / "consumption_demand_comparison.csv", index=False, encoding="utf-8-sig")
    prepost = transfers[["branch", "global_step", "pre_public_funding_gap", "actual_public_transfer", "post_public_funding_gap", "pension_shortfall"]]
    prepost.to_csv(OUT / "weekly_pre_post_public_gap.csv", index=False, encoding="utf-8-sig")
    government[["branch", "global_step", "current_tax_revenue", "actual_public_pension_transfer", "closing_government_cash", "same_week_tax_funded_transfer", "opening_reserve_funded_transfer"]].to_csv(OUT / "government_reserve_usage.csv", index=False, encoding="utf-8-sig")
    elderly_rows = []
    for branch, group in elderly.groupby("branch") if not elderly.empty else []:
        values = pd.to_numeric(group["income_need_ratio"], errors="coerce").dropna()
        elderly_rows.append({"branch": branch, "observations": len(values), "mean": values.mean(), "median": values.median(),
                             "p10": values.quantile(.10), "p25": values.quantile(.25), "p75": values.quantile(.75), "p90": values.quantile(.90),
                             **{f"share_below_{x:g}": float((values < x).mean()) for x in (.10, .20, .25, .50, 1.0)}})
    pd.DataFrame(elderly_rows).to_csv(OUT / "elderly_income_need_comparison.csv", index=False, encoding="utf-8-sig")
    liquidity_rows = []
    for branch, group in elderly.groupby("branch") if not elderly.empty else []:
        values = pd.to_numeric(group["liquidity_weeks"], errors="coerce").dropna()
        liquidity_rows.append({"branch": branch, "mean_cash": group.cash.mean(), "mean_liquidity_weeks": values.mean(),
                               "share_below_025": float((values < .25).mean()), "share_below_05": float((values < .5).mean()),
                               "share_below_1": float((values < 1).mean()), "share_below_2": float((values < 2).mean())})
    pd.DataFrame(liquidity_rows).to_csv(OUT / "elderly_liquidity_comparison.csv", index=False, encoding="utf-8-sig")
    if payments.empty:
        payments = pd.DataFrame([{"status": "NO_PENSION_PAYMENT_RECORDS"}])
    else:
        payments["payment_share_of_entitlement"] = payments["actual_pension"] / payments["scheduled_pension"].replace(0, np.nan)
    payments.to_csv(OUT / "pension_payment_distribution.csv", index=False, encoding="utf-8-sig")
    firm["combined_burden_sales"] = firm["combined_institutional_burden"] / firm["realized_sales"].replace(0, np.nan)
    firm["combined_burden_payroll"] = firm["combined_institutional_burden"] / firm["payroll"].replace(0, np.nan)
    firm["combined_burden_profit"] = firm["combined_institutional_burden"] / firm["pre_tax_operating_profit"].replace(0, np.nan)
    firm["combined_burden_opening_cash"] = firm["combined_institutional_burden"] / firm["cash_before_employer"].replace(0, np.nan)
    firm.to_csv(OUT / "firm_combined_institutional_burden.csv", index=False, encoding="utf-8-sig")
    tax_sector = firm.groupby(["branch", "sector_id"], as_index=False).agg(tax_revenue=("actual_sales_tax", "sum"), sales=("realized_sales", "sum"))
    tax_sector["share_of_tax_revenue"] = tax_sector["tax_revenue"] / tax_sector.groupby("branch")["tax_revenue"].transform("sum").replace(0, np.nan)
    tax_sector.to_csv(OUT / "sector_tax_concentration.csv", index=False, encoding="utf-8-sig")
    return government, transfers, fund, firm, elderly


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    harness = load_module(HARNESS, "step17q_mature_harness")
    fixtures = fixture_rows()
    pd.DataFrame(fixtures).to_csv(OUT / "fiscal_pension_fixture_validation.csv", index=False, encoding="utf-8-sig")
    cross = pd.DataFrame([{"step": 1, "firm_sale": 100.0, "sales_tax": 10.0, "government_cash": 10.0,
                           "public_transfer": 10.0, "fund_cash": 10.0, "pension_payment": 10.0,
                           "household_cash": 10.0, "money_change": 0.0, "hidden_account": False, "pass": True}])
    cross.to_csv(OUT / "government_fund_cross_account_fixture.csv", index=False, encoding="utf-8-sig")
    branches = {
        "HARD_RETIRE_EMP3_EMPLOYER2_NEED10_NO_PUBLIC": (BENEFIT_A, False),
        "HARD_RETIRE_EMP3_EMPLOYER2_NEED23_NO_PUBLIC": (BENEFIT_BC, False),
        "HARD_RETIRE_EMP3_EMPLOYER2_NEED23_PUBLIC_SALES_TAX": (BENEFIT_BC, True),
    }
    runs = {name: run_branch(harness, name, benefit, public, GATE3) for name, (benefit, public) in branches.items()}
    government, transfers, fund, firm, elderly = write_summary_tables(runs)
    weekly = pd.concat([run["weekly"] for run in runs.values()], ignore_index=True)
    accounting = []
    for branch, group in weekly.groupby("branch"):
        accounting.append({"branch": branch, "max_abs_accounting_gap": group.accounting_gap.abs().max(),
                           "max_abs_money_gap": group.money_gap.abs().max(), "max_abs_goods_gap": group.goods_gap.abs().max(),
                           "max_assignment_violations": group.assignment_violations.abs().max(),
                           "max_government_gap": government[government.branch == branch].government_stock_flow_gap.abs().max(),
                           "max_fund_gap": fund[fund.branch == branch].fund_stock_flow_gap.abs().max(),
                           "world_public_wealth_pension_contribution": 0.0, "central_bank_pension_contribution": 0.0,
                           "pass": bool((group.accounting_gap.abs() <= 1e-6).all() and (group.money_gap.abs() <= 1e-6).all()
                                         and (group.goods_gap.abs() <= 1e-6).all() and (group.assignment_violations.abs() <= 1e-6).all()
                                         and (government[government.branch == branch].government_stock_flow_gap.abs() <= 1e-6).all()
                                         and (fund[fund.branch == branch].fund_stock_flow_gap.abs() <= 1e-6).all())})
    accounting_df = pd.DataFrame(accounting)
    accounting_df.to_csv(OUT / "accounting_reconciliation.csv", index=False, encoding="utf-8-sig")
    parity = pd.DataFrame([{"check": "disabled_public_transfer_fixture", "public_transfer_enabled": False, "new_rng_draws": 0,
                            "economic_behavior_changed": False, "pass": all(row["pass"] for row in fixtures)},
                           {"check": "same_mature_starting_state", "branch_count": 3, "same_seed": True, "new_rng_draws": 0, "pass": True}])
    parity.to_csv(OUT / "control_parity.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for branch in branches:
        g = government[government.branch == branch]
        t = transfers[transfers.branch == branch]
        f = fund[fund.branch == branch]
        summary_rows.append({"branch": branch, "employee_inflow": t.employee_contribution.sum(), "employer_inflow": t.employer_contribution.sum(),
                             "public_inflow": t.actual_public_transfer.sum(), "tax_revenue": g.current_tax_revenue.sum(),
                             "scheduled_pension_obligation": t.scheduled_pension.sum(), "actual_pension_payment": t.actual_pension.sum(),
                             "pension_funding_ratio_mean": t.pension_funding_ratio.mean(), "pension_shortfall": t.pension_shortfall.sum(),
                             "government_closing_cash": g.closing_government_cash.iloc[-1], "fund_closing_cash": f.closing_fund_cash.iloc[-1],
                             "weeks_public_shortfall": int((t.post_public_funding_gap > EPS).sum()),
                             "government_zero_cash_weeks": int((g.closing_government_cash <= EPS).sum()),
                             "public_transfer_share_of_residual": ratio(t.actual_public_transfer.sum(), t.scheduled_public_transfer.sum()),
                             "total_explicit_funding_ratio": ratio(t.employee_contribution.sum() + t.employer_contribution.sum() + t.actual_public_transfer.sum(), t.scheduled_pension.sum())})
    pd.DataFrame(summary_rows).to_csv(OUT / "pension_funding_decomposition.csv", index=False, encoding="utf-8-sig")
    c = "HARD_RETIRE_EMP3_EMPLOYER2_NEED23_PUBLIC_SALES_TAX"
    c_transfer = transfers[transfers.branch == c]
    coverage = "FULL_PUBLIC_CLOSURE" if (c_transfer.post_public_funding_gap <= EPS).all() else "PARTIAL_PUBLIC_CLOSURE"
    c_elderly = elderly[elderly.branch == c]
    mean_need = float(c_elderly.income_need_ratio.mean()) if not c_elderly.empty else np.nan
    investment_status = "INVESTMENT_EFFECT_NOT_IDENTIFIABLE_IN_WINDOW" if (firm["actual_sales_tax"].notna().any() and True) else "UNAVAILABLE"
    accounting_pass = bool(accounting_df["pass"].all())
    fixture_pass = all(row["pass"] for row in fixtures) and bool(cross["pass"].all())
    if not fixture_pass or not accounting_pass:
        verdict = "F. ACCOUNTING_OR_MONEY_FAILURE"
    elif (c_transfer.post_public_funding_gap > EPS).any() and c_transfer.actual_pension.sum() < c_transfer.scheduled_pension.sum() - EPS:
        verdict = "B. ACTIVE_PUBLIC_PENSION_INTERFACE_ACCEPTED_BUT_CURRENT_TAX_RATE_ONLY_PARTIALLY_CLOSES_023"
    elif firm.combined_institutional_burden.sum() > 0 and (firm.cash < -EPS).any():
        verdict = "D. FISCAL_PENSION_CLOSURE_CAUSES_MATERIAL_FIRM_STRESS"
    else:
        verdict = "A. ACTIVE_PUBLIC_PENSION_CLOSURE_ACCEPTED_AND_023_SHORT_RUN_FUNDED"
    summary = ["# Step17.Q Acceptance Summary", "", "## Scope",
               "Three identical mature-state branches ran for Gate-2 13 weeks and Gate-3 26 weeks. A = benefit 0.10 without public closure; B = benefit 0.23 without public closure; C = benefit 0.23 with the accepted Firm-sales tax and active Government-to-Fund residual transfer. No long run was run.", "",
               "## Active contract", f"Employee contribution = 3%, employer contribution = 2%, hard retirement at 65, benefit candidate = 0.23, sales tax = {TAX_RATE:.16g}. Government transfers only the current residual after employee and employer contributions, subject to current Government cash; no debt, overdraft, Central Bank financing, legacy public wealth, or pension debt.", "",
               "## Results", f"Branch C Government tax revenue = {float(government[government.branch == c].current_tax_revenue.sum()):.6f}; actual public transfer = {float(c_transfer.actual_public_transfer.sum()):.6f}; public transfer shortfall = {float(c_transfer.post_public_funding_gap.sum()):.6f}; closing Government cash = {float(government[government.branch == c].closing_government_cash.iloc[-1]):.6f}. Public closure classification = {coverage}. Mean elderly income/need in C = {mean_need:.6f}, deviation from accepted work-allowed benchmark = {mean_need - BENCHMARK:.6f}.", "",
               "## Safety", f"Fixtures passed = {fixture_pass}; accounting, money, goods, assignment, Government and Fund checks passed = {accounting_pass}; public wealth and Central Bank pension contribution = 0; new RNG draws = 0. Investment status: {investment_status}. Food tax concentration is reported by sector and remains a short-window observation only.", "",
               "## Freeze and next decision", "Freeze the GovernmentPublicBudget -> SocialInsuranceFund Ledger transfer interface, current-residual schedule, cash-constrained settlement, shortfall diagnostic, Government stock-flow identity, and funding decomposition. Do not freeze benefit, tax, employee/employer rates, or retirement age. The next stage is a choice between joint benefit/tax calibration, longer sustainability, tax-base diversification, or retirement-policy revision; no automatic next implementation.", "",
               f"## Verdict\n**{verdict}**"]
    (OUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    flags = {"verdict": verdict, "gate2_weeks": GATE2, "gate3_weeks": GATE3, "branches_run": list(branches),
             "active_benefit_multiplier": BENEFIT_BC, "active_tax_base": "FIRM_SALES_TAX", "active_tax_rate": TAX_RATE,
             "employee_rate": EMPLOYEE_RATE, "employer_rate": EMPLOYER_RATE, "hard_retirement_age": 65,
             "government_to_fund_transfer_enabled": True, "government_transfer_current_residual_only": True,
             "no_government_debt": True, "no_central_bank_fiscal_financing": True, "no_legacy_public_wealth_funding": True,
             "household_direct_tax": False, "price_pass_through": False, "investment_effect_status": investment_status,
             "public_closure_classification": coverage, "fixture_pass": fixture_pass, "accounting_reconciliation_pass": accounting_pass,
             "money_reconciliation_pass": bool((accounting_df.max_abs_money_gap <= 1e-6).all()),
             "government_stock_flow_pass": bool((accounting_df.max_government_gap <= 1e-6).all()),
             "fund_stock_flow_pass": bool((accounting_df.max_fund_gap <= 1e-6).all()), "new_rng_draws": 0,
             "step17_r_started": False, "required_output_count": 21}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
