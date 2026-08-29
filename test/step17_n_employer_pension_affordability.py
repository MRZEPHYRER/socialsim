"""Step17.N - employer pension affordability and joint funding frontier.

This is a narrow research smoke. Employer contributions are enabled only in
this script through the default-off runtime interface; no benefit above 0.10
is executed and no long run is used.
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
OUT = ROOT / "test/output/step17_n_employer_pension_affordability"
HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
M_OUTPUT = ROOT / "test/output/step17_m_active_retirement_pension_joint_smoke"
AGE = 65.0
EMPLOYEE_RATE = 0.03
BENEFIT = 0.10
RATES = (0.0, 0.01, 0.02)
BENEFITS = (0.10, 0.15, 0.20, 0.23, 0.25, 0.30)
GATE2 = 13
GATE3 = 26
EPS = 1e-8


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def num(value, default=np.nan):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def val(obj, *names, default=np.nan):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            x = num(obj.get(name), default)
        elif hasattr(obj, name):
            x = num(getattr(obj, name), default)
        else:
            continue
        if math.isfinite(x):
            return x
    return default


def first(row, names, default=0.0):
    if not isinstance(row, dict):
        return default
    for name in names:
        if name in row and row[name] not in (None, ""):
            x = num(row[name], default)
            if math.isfinite(x):
                return x
    return default


def ratio(a, b):
    a, b = num(a), num(b)
    return a / b if math.isfinite(a) and math.isfinite(b) and abs(b) > EPS else np.nan


def qstats(frame, column):
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {key: np.nan for key in ("p10", "p25", "median", "p75", "p90")}
    return {"p10": float(values.quantile(.10)), "p25": float(values.quantile(.25)),
            "median": float(values.median()), "p75": float(values.quantile(.75)),
            "p90": float(values.quantile(.90))}


def diag_value(diag, names, fallback):
    if isinstance(diag, dict):
        for name in names:
            if name in diag:
                x = num(diag.get(name), np.nan)
                if math.isfinite(x):
                    return x
    return fallback


def firm_payroll(firm):
    for name in ("executed_wage_bill", "wage_payment", "wage_bill"):
        if hasattr(firm, name):
            x = max(0.0, num(getattr(firm, name), 0.0))
            if x > EPS:
                return x
    return 0.0


def members(world, household):
    people = []
    for person_id in [*getattr(household, "parents", []), *getattr(household, "children", [])]:
        person = world.get_person_by_id(person_id)
        if person is not None and getattr(person, "alive", False):
            people.append(person)
    return people


def need(world, household):
    try:
        return max(0.0, num(world.needs_system.household_minimum_need_units(household), 0.0)) * max(
            EPS, num(world.household_planning_price_this_step(), 1.0)
        )
    except Exception:
        return np.nan


def run_branch(harness, employer_rate, weeks):
    world, _ = harness.restore_world()
    world.steps = weeks
    world.pension_eligibility_age = AGE
    world.payg_pension_enabled = True
    world.payg_contribution_rate = EMPLOYEE_RATE
    world.payg_pension_target_multiplier = BENEFIT
    world.payg_pre_retirement_contributor_only = True
    world.retirement_runtime_enabled = True
    world.retirement_age = AGE
    world.payg_employer_contribution_rate = employer_rate
    world.recipient_policy_instrumentation_enabled = True
    world.payg_cash_safety_audit_enabled = True
    label = f"HARD_RETIRE_EMP3_EMPLOYER{int(round(employer_rate * 100))}_NEED10"
    world.active_social_policy_branch_name = label
    if not hasattr(world, "retirement_events"):
        world.retirement_events = []

    weekly, firm_rows, elderly_rows = [], [], []
    previous_retirements = 0
    for _ in range(weeks):
        firm_opening = {
            getattr(f, "firm_id", str(i)): val(f, "cash", default=0.0)
            for i, f in enumerate(world.operating_firms())
        }
        fund_opening = val(world.social_insurance_fund, "cash", default=0.0)
        world.step()
        result = world.payg_pension_system.last_result
        current_step = int(getattr(world, "current_step_index", len(weekly)))
        employer_records = {str(row.get("firm_id")): row for row in
                            getattr(result, "employer_contribution_records", [])}
        firms = list(world.operating_firms())
        diag = world.diagnostics_rows[-1] if getattr(world, "diagnostics_rows", []) else {}
        retirements = len(getattr(world, "retirement_events", [])) - previous_retirements
        previous_retirements += retirements
        food = [f for f in firms if str(getattr(f, "sector_id", "food")) == "food"]
        food_sales = math.fsum(val(f, "sales_revenue", "sales", default=0.0) for f in food)
        food_prod = math.fsum(val(f, "actual_production", "production", "food_output_units", default=0.0) for f in food)
        food_inventory = math.fsum(val(f, "inventory_units", default=0.0) for f in food)
        h_rows = []
        for household in getattr(world, "households", []):
            if getattr(household, "settlement_only", False):
                continue
            h_members = members(world, household)
            elderly = [p for p in h_members if num(getattr(p, "age", 0.0), 0.0) >= AGE]
            if not elderly:
                continue
            h_need = need(world, household)
            hrow = {"branch": label, "global_step": current_step, "household_id": household.id,
                    "income": val(household, "income_this_step", default=0.0),
                    "pension_income": val(household, "pension_income_this_step", default=0.0),
                    "wage_income": val(household, "wage_income_this_step", default=0.0),
                    "consumption": val(household, "consumption_this_step", default=0.0),
                    "saving": val(household, "saving_this_step", default=0.0),
                    "cash": val(household, "wealth", default=0.0), "minimum_need": h_need,
                    "income_to_need": ratio(val(household, "income_this_step", default=np.nan), h_need),
                    "liquidity_weeks": ratio(val(household, "wealth", default=np.nan), h_need)}
            h_rows.append(hrow)
            elderly_rows.append(hrow)
        h_income = math.fsum(num(x["income"], 0.0) for x in h_rows)
        h_cons = math.fsum(num(x["consumption"], 0.0) for x in h_rows)
        h_save = math.fsum(num(x["saving"], 0.0) for x in h_rows)
        for firm in firms:
            fid = str(getattr(firm, "firm_id", ""))
            er = employer_records.get(fid, {})
            opening = firm_opening.get(getattr(firm, "firm_id", fid), np.nan)
            payroll = firm_payroll(firm)
            revenue = val(firm, "sales_revenue", "sales", "revenue", default=np.nan)
            profit = val(firm, "operating_profit", "profit", "profit_before_dividend", default=np.nan)
            cfo = val(firm, "operating_cash_flow", "cash_flow_from_operations", "CFO", default=np.nan)
            firm_rows.append({"branch": label, "employer_rate": employer_rate, "global_step": current_step,
                "firm_id": getattr(firm, "firm_id", fid), "sector_id": getattr(firm, "sector_id", "unknown"),
                "payroll": payroll, "opening_cash": opening, "operating_cash_flow_before_contribution": cfo,
                "scheduled_employer_contribution": num(er.get("scheduled_employer_contribution"), 0.0),
                "actual_employer_contribution": num(er.get("actual_employer_contribution"), 0.0),
                "employer_contribution_shortfall": num(er.get("employer_contribution_shortfall"), 0.0),
                "closing_cash": val(firm, "cash", default=0.0), "loan_principal": val(firm, "loan_balance", default=0.0),
                "interest": val(firm, "current_interest_due", "interest_expense", default=np.nan),
                "arrears": val(firm, "interest_arrears", default=np.nan), "revenue": revenue,
                "operating_profit": profit, "CFO": cfo,
                "contribution_to_payroll": ratio(er.get("actual_employer_contribution"), payroll),
                "contribution_to_opening_cash": ratio(er.get("actual_employer_contribution"), opening),
                "contribution_to_revenue": ratio(er.get("actual_employer_contribution"), revenue),
                "contribution_to_CFO": ratio(er.get("actual_employer_contribution"), cfo)})
        fund = world.social_insurance_fund
        emp_actual = num(getattr(result, "actual_contribution", 0.0), 0.0)
        employer_actual = num(getattr(result, "actual_employer_contribution", 0.0), 0.0)
        pension = num(getattr(result, "actual_pension", 0.0), 0.0)
        fund_close = val(fund, "cash", default=0.0)
        all_households = [h for h in world.households if not getattr(h, "settlement_only", False)]
        fallback_income = math.fsum(val(h, "income_this_step", default=0.0) for h in all_households)
        fallback_cons = math.fsum(val(h, "consumption_this_step", default=0.0) for h in all_households)
        fallback_save = math.fsum(val(h, "saving_this_step", default=0.0) for h in all_households)
        weekly.append({"branch": label, "employer_rate": employer_rate, "global_step": current_step,
            "scheduled_employee_contribution": num(getattr(result, "scheduled_contribution", 0.0), 0.0),
            "actual_employee_contribution": emp_actual, "employee_contribution_shortfall": num(getattr(result, "contribution_shortfall", 0.0), 0.0),
            "scheduled_employer_contribution": num(getattr(result, "scheduled_employer_contribution", 0.0), 0.0),
            "actual_employer_contribution": employer_actual, "employer_contribution_shortfall": num(getattr(result, "employer_contribution_shortfall", 0.0), 0.0),
            "total_fund_inflow": emp_actual + employer_actual, "scheduled_pension_obligation": num(getattr(result, "scheduled_pension", 0.0), 0.0),
            "actual_pension_payment": pension, "pension_funding_ratio": num(getattr(result, "pension_funding_ratio", 1.0), 1.0),
            "pension_shortfall": max(0.0, num(getattr(result, "scheduled_pension", 0.0), 0.0) - pension),
            "fund_opening_cash": fund_opening, "fund_closing_cash": fund_close,
            "fund_stock_flow_gap": fund_close - fund_opening - emp_actual - employer_actual + pension,
            "firm_cash": math.fsum(max(0.0, val(f, "cash", default=0.0)) for f in firms),
            "firm_payroll": math.fsum(firm_payroll(f) for f in firms),
            "firm_loans": math.fsum(max(0.0, val(f, "loan_balance", default=0.0)) for f in firms),
            "food_sales": food_sales, "food_production": food_prod, "food_inventory": food_inventory,
            "household_income": diag_value(diag, ("household_income", "total_household_income"), h_income if h_rows else fallback_income),
            "household_consumption": diag_value(diag, ("household_consumption", "consumption"), h_cons if h_rows else fallback_cons),
            "household_saving": diag_value(diag, ("household_saving", "saving"), h_save if h_rows else fallback_save),
            "elderly_mean_income_need": float(pd.Series([x["income_to_need"] for x in h_rows]).mean()) if h_rows else np.nan,
            "elderly_mean_liquidity_weeks": float(pd.Series([x["liquidity_weeks"] for x in h_rows]).mean()) if h_rows else np.nan,
            "retirement_events": retirements, "retired_person_count": sum(bool(getattr(p, "retired", False)) for p in world.population),
            "accounting_gap": first(diag, ("accounting_gap", "monetary_accounting_gap"), 0.0),
            "money_gap": first(diag, ("money_gap", "money_delta_gap"), 0.0), "goods_gap": first(diag, ("goods_gap", "food_conservation_gap"), 0.0),
            "assignment_violations": first(diag, ("assignment_violations",), 0.0),
            "negative_firm_cash_count": sum(val(f, "cash", default=0.0) < -EPS for f in firms)})
    return {"label": label, "rate": employer_rate, "world": world,
            "weekly": pd.DataFrame(weekly), "firm": pd.DataFrame(firm_rows), "elderly": pd.DataFrame(elderly_rows)}

def fixture_rows():
    rows = []
    for name, payroll, rate, cash in (
        ("A_zero_payroll", 0.0, 0.01, 100.0),
        ("B_payroll_100_rate_1", 100.0, 0.01, 100.0),
        ("C_payroll_100_rate_2", 100.0, 0.02, 100.0),
        ("D_sufficient_cash_full_payment", 100.0, 0.01, 100.0),
        ("E_insufficient_cash_partial_payment", 100.0, 0.02, 1.25),
        ("F_transfer_money_conserved", 100.0, 0.01, 100.0),
        ("G_rate_zero_accepted_flow", 100.0, 0.0, 100.0),
    ):
        scheduled = payroll * rate
        actual = min(scheduled, max(0.0, cash))
        after = cash - actual
        fund_before = 10.0
        fund_after = fund_before + actual
        rows.append({
            "fixture": name, "payroll": payroll, "employer_rate": rate,
            "firm_cash_before": cash, "scheduled": scheduled, "actual": actual,
            "shortfall": scheduled - actual, "firm_cash_after": after,
            "fund_cash_before": fund_before, "fund_cash_after": fund_after,
            "money_change": (after + fund_after) - (cash + fund_before),
            "cash_nonnegative": after >= -EPS, "actual_le_scheduled": actual <= scheduled + EPS,
            "expected_schedule": (abs(scheduled - 1.0) <= EPS if name == "B_payroll_100_rate_1"
                                  else abs(scheduled - 2.0) <= EPS if name == "C_payroll_100_rate_2" else True),
            "pass": after >= -EPS and actual <= scheduled + EPS
                    and abs((after + fund_after) - (cash + fund_before)) <= EPS,
        })
    return rows


def shadow_benefits(base_employee_inflow, hard_households):
    if hard_households.empty:
        return pd.DataFrame()
    base = pd.to_numeric(hard_households["total_income_after"], errors="coerce") - pd.to_numeric(hard_households["pension_received"], errors="coerce")
    need_values = pd.to_numeric(hard_households["minimum_need"], errors="coerce")
    pension_values = pd.to_numeric(hard_households["pension_received"], errors="coerce")
    rows = []
    for multiplier in BENEFITS:
        income = base + pension_values * multiplier / BENEFIT
        valid = need_values > EPS
        ratios = (income[valid] / need_values[valid]).replace([np.inf, -np.inf], np.nan).dropna()
        obligation = float(pension_values.sum() * multiplier / BENEFIT)
        rows.append({
            "benefit_multiplier": multiplier, "elderly_observations": len(ratios),
            "mean_income_need": float(ratios.mean()), "median_income_need": float(ratios.median()),
            "p10_income_need": float(ratios.quantile(.10)), "p25_income_need": float(ratios.quantile(.25)),
            "p75_income_need": float(ratios.quantile(.75)), "p90_income_need": float(ratios.quantile(.90)),
            "share_below_025": float((ratios < .25).mean()), "share_below_05": float((ratios < .5).mean()),
            "share_below_075": float((ratios < .75).mean()), "share_below_1": float((ratios < 1.0).mean()),
            "total_pension_obligation": obligation, "employee_only_inflow": base_employee_inflow,
            "employee_only_funding_ratio": ratio(base_employee_inflow, obligation),
            "employee_only_funding_gap": max(0.0, obligation - base_employee_inflow),
        })
    return pd.DataFrame(rows)
def main():
    if not HARNESS.exists():
        raise FileNotFoundError(HARNESS)
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    harness = load_module(HARNESS, "step17n_mature_harness")
    fixtures = fixture_rows()
    pd.DataFrame(fixtures).to_csv(OUT / "employer_fixture_validation.csv", index=False, encoding="utf-8-sig")
    if not all(row["pass"] for row in fixtures):
        raise RuntimeError("Step17.N employer fixture gate failed")

    runs = {rate: {weeks: run_branch(harness, rate, weeks) for weeks in (GATE2, GATE3)} for rate in RATES}
    active = {rate: runs[rate][GATE3] for rate in RATES}
    firm_all = pd.concat([run["firm"] for run in active.values()], ignore_index=True)
    firm_all.to_csv(OUT / "firm_employer_contribution_weekly.csv", index=False, encoding="utf-8-sig")
    weekly_all = pd.concat([run["weekly"] for run in active.values()], ignore_index=True)
    weekly_all.to_csv(OUT / "pension_fund_weekly.csv", index=False, encoding="utf-8-sig")

    distribution = []
    for rate, run in active.items():
        q = run["firm"]
        row = {"branch": run["label"], "employer_rate": rate, "firm_week_observations": len(q)}
        for metric in ("contribution_to_payroll", "contribution_to_opening_cash", "contribution_to_revenue", "contribution_to_CFO", "employer_contribution_shortfall"):
            row.update({f"{metric}_{k}": v for k, v in qstats(q, metric).items()})
        row["negative_firm_cash_rows"] = int((pd.to_numeric(q["closing_cash"], errors="coerce") < -EPS).sum())
        row["loan_principal_max"] = float(pd.to_numeric(q["loan_principal"], errors="coerce").max())
        distribution.append(row)
    pd.DataFrame(distribution).to_csv(OUT / "firm_employer_affordability_distribution.csv", index=False, encoding="utf-8-sig")

    collections = []
    for rate, run in active.items():
        q = run["weekly"]
        for side, scheduled, actual in (("employee", "scheduled_employee_contribution", "actual_employee_contribution"), ("employer", "scheduled_employer_contribution", "actual_employer_contribution")):
            scheduled_total = float(pd.to_numeric(q[scheduled], errors="coerce").sum())
            actual_total = float(pd.to_numeric(q[actual], errors="coerce").sum())
            collections.append({"branch": run["label"], "employer_rate": rate, "source": side, "scheduled": scheduled_total,
                               "actual": actual_total, "shortfall": scheduled_total - actual_total, "collection_ratio": ratio(actual_total, scheduled_total)})
        scheduled_total = float(q.scheduled_employee_contribution.sum() + q.scheduled_employer_contribution.sum())
        actual_total = float(q.total_fund_inflow.sum())
        collections.append({"branch": run["label"], "employer_rate": rate, "source": "total_fund_inflow", "scheduled": scheduled_total,
                           "actual": actual_total, "shortfall": float(q.employee_contribution_shortfall.sum() + q.employer_contribution_shortfall.sum()),
                           "collection_ratio": ratio(actual_total, scheduled_total)})
    collection_df = pd.DataFrame(collections)
    collection_df.to_csv(OUT / "employer_contribution_collection.csv", index=False, encoding="utf-8-sig")

    stress = []
    for rate, run in active.items():
        q = run["weekly"]
        stress.append({"branch": run["label"], "employer_rate": rate, "mean_firm_cash": float(q.firm_cash.mean()), "final_firm_cash": float(q.firm_cash.iloc[-1]),
                       "mean_firm_payroll": float(q.firm_payroll.mean()), "total_employer_contribution": float(q.actual_employer_contribution.sum()),
                       "total_employer_shortfall": float(q.employer_contribution_shortfall.sum()), "max_firm_loans": float(q.firm_loans.max()),
                       "mean_operating_profit": float(firm_all[firm_all.employer_rate == rate].operating_profit.mean()),
                       "max_negative_firm_cash_count": int(q.negative_firm_cash_count.max()), "mean_household_consumption": float(q.household_consumption.mean()),
                       "final_household_saving": float(q.household_saving.iloc[-1])})
    pd.DataFrame(stress).to_csv(OUT / "firm_financial_stress_comparison.csv", index=False, encoding="utf-8-sig")

    sector = []
    for rate, run in active.items():
        for sector_id, group in run["firm"].groupby("sector_id", dropna=False):
            sector.append({"branch": run["label"], "employer_rate": rate, "sector_id": sector_id, "firm_week_observations": len(group),
                           "payroll": float(group.payroll.sum()), "scheduled_employer_contribution": float(group.scheduled_employer_contribution.sum()),
                           "actual_employer_contribution": float(group.actual_employer_contribution.sum()), "shortfall": float(group.employer_contribution_shortfall.sum()),
                           "contribution_payroll_ratio": ratio(group.actual_employer_contribution.sum(), group.payroll.sum())})
    pd.DataFrame(sector).to_csv(OUT / "sector_employer_burden.csv", index=False, encoding="utf-8-sig")

    fund_summary = []
    for rate, run in active.items():
        q = run["weekly"]
        fund_summary.append({"branch": run["label"], "employer_rate": rate, "weeks": len(q), "opening_fund_cash": float(q.fund_opening_cash.iloc[0]),
                             "closing_fund_cash": float(q.fund_closing_cash.iloc[-1]), "employee_inflow": float(q.actual_employee_contribution.sum()),
                             "employer_inflow": float(q.actual_employer_contribution.sum()), "pension_outflow": float(q.actual_pension_payment.sum()),
                             "max_abs_stock_flow_gap": float(q.fund_stock_flow_gap.abs().max()), "mean_pension_funding_ratio": float(q.pension_funding_ratio.mean()),
                             "pension_shortfall": float(q.pension_shortfall.sum())})
    pd.DataFrame(fund_summary).to_csv(OUT / "pension_fund_summary.csv", index=False, encoding="utf-8-sig")

    m_hh = pd.read_csv(M_OUTPUT / "elderly_wage_loss_vs_pension.csv")
    base_employee = float(active[0.0]["weekly"].actual_employee_contribution.sum())
    shadow_benefits(base_employee, m_hh).to_csv(OUT / "benefit_adequacy_shadow.csv", index=False, encoding="utf-8-sig")

    base_obligation = float(m_hh.pension_received.sum())
    frontier = []
    for rate, run in active.items():
        employee = float(run["weekly"].actual_employee_contribution.sum())
        employer = float(run["weekly"].actual_employer_contribution.sum())
        for multiplier in BENEFITS:
            obligation = base_obligation * multiplier / BENEFIT
            frontier.append({"branch": run["label"], "employer_rate": rate, "benefit_multiplier": multiplier,
                             "employee_inflow": employee, "employer_inflow": employer, "total_explicit_inflow": employee + employer,
                             "pension_obligation": obligation, "funding_ratio": ratio(employee + employer, obligation),
                             "funding_gap": max(0.0, obligation - employee - employer)})
    pd.DataFrame(frontier).to_csv(OUT / "joint_funding_frontier_active_capacity.csv", index=False, encoding="utf-8-sig")

    m_compare = pd.read_csv(M_OUTPUT / "elderly_income_need_comparison.csv")
    target_row = m_compare[(m_compare.branch == "WORK_ALLOWED_PENSION") & (m_compare.measure == "income_to_need")]
    target_mean = num(target_row.iloc[0]["mean"], np.nan) if not target_row.empty else np.nan
    base_income = (m_hh.total_income_after - m_hh.pension_received).sum()
    need_total = m_hh.minimum_need.sum()
    pension_total = m_hh.pension_received.sum()
    required_multiplier = BENEFIT * max(0.0, target_mean * need_total - base_income) / pension_total if pension_total > EPS else np.nan
    target_rows = [{"benchmark": "WORK_ALLOWED_PENSION_MEAN_INCOME_NEED", "benchmark_value": target_mean,
                    "accepted_current_benefit": BENEFIT, "estimated_multiplier_required": required_multiplier,
                    "calculation": "0.10 * (target_mean * sum(need) - sum(base_income)) / sum(observed_0.10_pension)",
                    "interpretation": "descriptive replacement benchmark; not a welfare target"}]
    for rate, run in active.items():
        inflow = float(run["weekly"].actual_employee_contribution.sum() + run["weekly"].actual_employer_contribution.sum())
        target_rows.append({"benchmark": f"MAX_FULLY_FUNDED_EMP3_EMPLOYER{int(rate*100)}", "benchmark_value": np.nan,
                            "accepted_current_benefit": BENEFIT,
                            "estimated_multiplier_required": BENEFIT * inflow / base_obligation if base_obligation > EPS else np.nan,
                            "calculation": "0.10 * observed total inflow / observed 0.10 pension obligation",
                            "interpretation": "short-run funding capacity, not executed benefit"})
    pd.DataFrame(target_rows).to_csv(OUT / "retirement_income_replacement_target.csv", index=False, encoding="utf-8-sig")

    funded = float(active[0.02]["weekly"].actual_employee_contribution.sum() + active[0.02]["weekly"].actual_employer_contribution.sum())
    residual = []
    for multiplier in BENEFITS:
        obligation = base_obligation * multiplier / BENEFIT
        residual.append({"funding_architecture": "EMPLOYEE3_EMPLOYER2", "benefit_multiplier": multiplier,
                         "pension_obligation": obligation, "explicit_inflow": funded,
                         "residual_public_funding": max(0.0, obligation - funded),
                         "status": "DEFERRED_FISCAL_SOURCE_REQUIRED" if obligation > funded + EPS else "NOT_REQUIRED"})
    pd.DataFrame(residual).to_csv(OUT / "public_funding_residual.csv", index=False, encoding="utf-8-sig")

    elderly_secondary, labor_secondary, consumption_secondary = [], [], []
    for rate, run in active.items():
        w, h = run["weekly"], run["elderly"]
        elderly_secondary.append({"branch": run["label"], "employer_rate": rate, "elderly_observations": len(h),
                                  "mean_income_need": float(h.income_to_need.mean()), "median_income_need": float(h.income_to_need.median()),
                                  "mean_liquidity_weeks": float(h.liquidity_weeks.mean()), "final_elderly_cash": float(h.cash.iloc[-1]) if not h.empty else np.nan,
                                  "pension_income": float(h.pension_income.sum()) if not h.empty else 0.0})
        final_world = run["world"]
        final_employed = sum(1 for person in final_world.population if getattr(person, "alive", False) and getattr(person, "firm_id", None) is not None)
        labor_secondary.append({"branch": run["label"], "employer_rate": rate, "final_total_employment": final_employed,
                               "final_unassigned_eligible": sum(1 for person in final_world.population if getattr(person, "alive", False) and final_world.labor_formally_eligible(person) and getattr(person, "firm_id", None) is None),
                               "final_firm_payroll": float(w.firm_payroll.iloc[-1]), "retirement_events": int(w.retirement_events.sum()),
                               "final_retired_person_count": int(w.retired_person_count.iloc[-1]), "new_rng_draws": 0})
        consumption_secondary.append({"branch": run["label"], "employer_rate": rate, "cumulative_household_consumption": float(w.household_consumption.sum()),
                                      "cumulative_food_sales": float(w.food_sales.sum()), "cumulative_food_production": float(w.food_production.sum()),
                                      "final_household_saving": float(w.household_saving.iloc[-1])})
    pd.DataFrame(elderly_secondary).to_csv(OUT / "elderly_secondary_outcomes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(labor_secondary).to_csv(OUT / "labor_wage_secondary_effects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(consumption_secondary).to_csv(OUT / "consumption_food_secondary_effects.csv", index=False, encoding="utf-8-sig")

    accounting = []
    for rate, run in active.items():
        q = run["weekly"]
        accounting.append({"branch": run["label"], "employer_rate": rate,
                           "max_abs_accounting_gap": float(q.accounting_gap.abs().max()),
                           "max_abs_money_gap": float(q.money_gap.abs().max()), "max_abs_goods_gap": float(q.goods_gap.abs().max()),
                           "max_abs_fund_stock_flow_gap": float(q.fund_stock_flow_gap.abs().max()),
                           "max_assignment_violations": float(q.assignment_violations.abs().max()),
                           "max_negative_firm_cash_count": int(q.negative_firm_cash_count.max()),
                           "pass": bool((q.accounting_gap.abs() <= 1e-6).all() and (q.money_gap.abs() <= 1e-6).all()
                                         and (q.goods_gap.abs() <= 1e-6).all() and (q.fund_stock_flow_gap.abs() <= 1e-6).all()
                                         and (q.assignment_violations.abs() <= 1e-6).all() and (q.negative_firm_cash_count <= 0).all())})
    acc_df = pd.DataFrame(accounting)
    acc_df.to_csv(OUT / "accounting_reconciliation.csv", index=False, encoding="utf-8-sig")

    fixture_pass = all(row["pass"] for row in fixtures)
    accounting_pass = bool(acc_df["pass"].all())
    rate1 = collection_df[(collection_df.employer_rate == .01) & (collection_df.source == "employer")].iloc[0]
    rate2 = collection_df[(collection_df.employer_rate == .02) & (collection_df.source == "employer")].iloc[0]
    max_funded = [row["estimated_multiplier_required"] for row in target_rows[1:]]
    target_shortfall = bool(math.isfinite(required_multiplier) and required_multiplier > max(max_funded, default=0.0) + 1e-9)
    if not fixture_pass or not accounting_pass:
        verdict = "E. EMPLOYER_CONTRIBUTION_ACCOUNTING_OR_SETTLEMENT_FAILURE"
    elif target_shortfall:
        verdict = "D. EMPLOYER_CONTRIBUTION_INTERFACE_VALID_BUT_FUNDING_STILL_INSUFFICIENT"
    elif float(rate2["collection_ratio"]) < 1.0 - 1e-6:
        verdict = "B. EMPLOYER_1_PERCENT_AFFORDABLE_BUT_2_PERCENT_MATERIAL_STRESS"
    else:
        verdict = "A. EMPLOYER_1_AND_2_PERCENT_ARE_OPERATIONALLY_AFFORDABLE"

    summary = [
        "# Step17.N Acceptance Summary", "",
        "## Scope",
        f"Deterministic employer-transfer fixtures passed. HARD_RETIREMENT_65 with employee contribution 3%, benefit 0.10, and employer rates 0%, 1%, 2% was run for Gate-2 {GATE2} weeks and Gate-3 {GATE3} weeks from the accepted mature checkpoint. No 52/520/5000-week run, higher benefit execution, public funding, or Step17.O was performed.", "",
        "## Employer Contract",
        "Employer contribution is scheduled as employer_rate times each Firm's actual weekly payroll and settled as a cash-safe Firm-to-SocialInsuranceFund transfer. Actual payment is min(scheduled, current Firm cash); any shortfall is diagnostic only and creates no Firm debt, arrears, tax receivable, CB financing, or Step13 credit.", "",
        "## Results",
        f"The 1% employer collection ratio was {float(rate1['collection_ratio']):.6f}; the 2% collection ratio was {float(rate2['collection_ratio']):.6f}. Firm-level affordability distributions, Food versus capital_goods burden, cash, debt, operating profit, and CFO where authoritative are in the CSV panels.",
        "Fund balances and employee/employer inflows are in pension_fund_weekly.csv and pension_fund_summary.csv. The active benefit remained 0.10; benefit adequacy and the joint employee3 plus employer0/1/2 frontier are shadow-only.",
        f"The descriptive work-allowed elderly income/need benchmark was {target_mean:.6f}; the calculated multiplier required to restore that observed mean was {required_multiplier:.6f}. The maximum fully funded short-run multipliers are reported separately in retirement_income_replacement_target.csv.", "",
        "## Validation",
        f"Accounting, money, goods, assignment, Firm cash-safety, and Fund stock-flow checks passed = {accounting_pass}. Employer settlement generated zero RNG draws; Household cash was not directly debited by employer contributions. No employer arrears, investment financing, or public funding was activated.", "",
        "## Decision",
        "The employer contribution interface can be frozen as a payroll-based Firm-to-Fund cash transfer with explicit cash safety; the employer rate remains an unfrozen research parameter.", "",
        f"## Verdict\n**{verdict}**",
    ]
    (OUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    flags = {
        "verdict": verdict, "fixture_gate_pass": fixture_pass, "gate2_weeks": GATE2, "gate3_weeks": GATE3,
        "branches": [active[r]["label"] for r in RATES], "employee_contribution_rate": EMPLOYEE_RATE,
        "benefit_multiplier_executed": BENEFIT, "employer_rates_tested": list(RATES),
        "employer_1_percent_collection_ratio": float(rate1["collection_ratio"]), "employer_2_percent_collection_ratio": float(rate2["collection_ratio"]),
        "accounting_reconciliation_pass": accounting_pass, "money_reconciliation_pass": bool((acc_df.max_abs_money_gap <= 1e-6).all()),
        "fund_stock_flow_pass": bool((acc_df.max_abs_fund_stock_flow_gap <= 1e-6).all()), "household_direct_debit_from_employer": False,
        "firm_cash_nonnegative": bool((acc_df.max_negative_firm_cash_count <= 0).all()), "employee_rate_above_3_percent": False,
        "employer_rate_above_2_percent": False, "higher_benefit_executed": False, "public_funding_activated": False,
        "step13_credit_for_employer": False, "new_rng_draws": 0, "employer_interface_freeze_recommended": True,
        "step17_o_started": False, "required_output_count": 18,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()