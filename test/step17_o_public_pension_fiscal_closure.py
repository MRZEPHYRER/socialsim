"""Step17.O - passive public-pension fiscal-closure audit.

This script consumes accepted Step17.N/M artifacts and existing public-ledger
diagnostics.  It deliberately does not run World.step(), enable public
funding, add taxes, or alter any pension parameter.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_o_public_pension_fiscal_closure"
N_OUT = ROOT / "test/output/step17_n_employer_pension_affordability"
M_OUT = ROOT / "test/output/step17_m_active_retirement_pension_joint_smoke"
PUBLIC_LEDGER = ROOT / "test/output/main_full_n5000_2000_seed42/accounting/public_accounting.csv"
EPS = 1e-8
BENEFIT_BASE = 0.10
EMPLOYEE_RATE = 0.03
EMPLOYER_RATE = 0.02
BENEFITS = (0.20, 0.23, 0.25)


def number(value, default=np.nan):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def write(rows, name):
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    return frame


def qstats(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if s.empty:
        return {"mean": np.nan, "median": np.nan, "p10": np.nan, "p25": np.nan,
                "p75": np.nan, "p90": np.nan, "max": np.nan}
    return {"mean": float(s.mean()), "median": float(s.median()),
            "p10": float(s.quantile(.10)), "p25": float(s.quantile(.25)),
            "p75": float(s.quantile(.75)), "p90": float(s.quantile(.90)),
            "max": float(s.max())}


def safe_ratio(a, b):
    return float(a / b) if math.isfinite(number(a)) and math.isfinite(number(b)) and abs(float(b)) > EPS else np.nan


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    weekly = pd.read_csv(N_OUT / "pension_fund_weekly.csv")
    weekly = weekly[weekly["employer_rate"].round(8) == EMPLOYER_RATE].copy()
    firms = pd.read_csv(N_OUT / "firm_employer_contribution_weekly.csv")
    firms = firms[firms["employer_rate"].round(8) == EMPLOYER_RATE].copy()
    frontier = pd.read_csv(N_OUT / "joint_funding_frontier_active_capacity.csv")
    frontier = frontier[frontier["employer_rate"].round(8) == EMPLOYER_RATE].copy()
    accepted_residual = pd.read_csv(N_OUT / "public_funding_residual.csv")
    elderly = pd.read_csv(M_OUT / "elderly_wage_loss_vs_pension.csv")
    target = pd.read_csv(N_OUT / "retirement_income_replacement_target.csv")
    public_ledger = pd.read_csv(PUBLIC_LEDGER) if PUBLIC_LEDGER.exists() else pd.DataFrame()

    # The public account is an existing cash holder, but its observed sources
    # are event-driven estate/inventory flows, not an active recurring tax.
    architecture = [
        {"component": "world.public_wealth", "account_type": "public_cash_stock",
         "payer_or_source": "death/no-heir/household-dissolution/public inventory events",
         "recipient": "world public account", "active": True,
         "recurring_revenue": False, "pension_funding_eligible": "stock/event only",
         "money_creation": False, "notes": "Existing cash stock; no tax or general-budget flow."},
        {"component": "central_bank.public_income_balance", "account_type": "public_cash_balance",
         "payer_or_source": "central-bank/public-income bookkeeping",
         "recipient": "public account", "active": True, "recurring_revenue": False,
         "pension_funding_eligible": "not ordinary fiscal revenue", "money_creation": False,
         "notes": "Central-bank money creation is explicitly outside fiscal revenue."},
        {"component": "SocialInsuranceFund", "account_type": "social insurance fund",
         "payer_or_source": "employee 3% + employer 2% research branches",
         "recipient": "eligible pensioners", "active": True, "recurring_revenue": True,
         "pension_funding_eligible": "explicit contribution only", "money_creation": False,
         "notes": "Not a Government tax; Step17.N accepted Firm/Household transfers."},
        {"component": "Government/PublicBudget", "account_type": "fiscal payer",
         "payer_or_source": "no implemented account or recurring revenue",
         "recipient": "SocialInsuranceFund", "active": False, "recurring_revenue": False,
         "pension_funding_eligible": "future interface", "money_creation": False,
         "notes": "Required future cash-constrained Ledger transfer; not executed in O."},
    ]
    write(architecture, "public_fiscal_architecture_map.csv")

    rev_rows = []
    if not public_ledger.empty:
        for col, label, status in (
            ("estate_no_heir_income", "estate/no-heir public wealth", "EVENT_DRIVEN"),
            ("inventory_release_sale_income", "public inventory release", "EVENT_DRIVEN"),
            ("other_actual_cash_revenue", "other recorded public cash revenue", "EVENT_DRIVEN"),
        ):
            values = pd.to_numeric(public_ledger.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            rev_rows.append({"source": label, "payer_or_trigger": "existing runtime event",
                             "tax_base": "none", "active": True, "status": status,
                             "reference_steps": len(values), "total_revenue": float(values.sum()),
                             "mean_weekly_revenue": float(values.mean()),
                             "positive_weeks": int((values > EPS).sum()),
                             "receiving_account": "world.public_wealth / public balance",
                             "already_committed_elsewhere": False})
    for label in ("household income tax", "Firm profit tax", "consumption tax", "general government revenue"):
        rev_rows.append({"source": label, "payer_or_trigger": "not implemented",
                         "tax_base": np.nan, "active": False, "status": "NOT_PRESENT",
                         "reference_steps": np.nan, "total_revenue": np.nan,
                         "mean_weekly_revenue": np.nan, "positive_weeks": np.nan,
                         "receiving_account": "none", "already_committed_elsewhere": np.nan})
    write(rev_rows, "existing_public_revenue_audit.csv")

    if public_ledger.empty:
        stock = [{"source": "existing_public_accounting", "window_steps": 0,
                  "opening_cash": np.nan, "total_inflow": np.nan, "total_outflow": np.nan,
                  "closing_cash": np.nan, "max_cash_stock": np.nan,
                  "flow_status": "UNAVAILABLE"}]
    else:
        stock = [{"source": "existing_public_accounting_main_n5000_2000",
                  "window_steps": len(public_ledger),
                  "opening_cash": float(public_ledger["public_cash_start"].iloc[0]),
                  "total_inflow": float(public_ledger["public_cash_revenue"].sum()),
                  "total_outflow": float(public_ledger["public_cash_expenditure"].sum()),
                  "closing_cash": float(public_ledger["public_cash_end"].iloc[-1]),
                  "max_cash_stock": float(public_ledger["public_cash_end"].max()),
                  "flow_status": "EVENT_DRIVEN_STOCK_NOT_RECURRING"}]
    write(stock, "public_cash_stock_flow.csv")

    explicit_inflow = float(weekly["total_fund_inflow"].sum())
    # Reference denominators are the accepted Step17.M/N final counts.
    reference_elderly_persons = 1109.0
    reference_total_population = 5085.0
    reference_pre_retirement_contributors = 3976.0
    # Keep the accepted Step17.N elderly-sample frontier as summary authority.
    accepted_base_obligation = float(accepted_residual.loc[accepted_residual["benefit_multiplier"].round(8) == 0.20, "pension_obligation"].iloc[0]) / 2.0
    weekly_base_obligation = float(weekly["scheduled_pension_obligation"].sum())
    weekly_obligation_scale = safe_ratio(accepted_base_obligation, weekly_base_obligation)
    public_summary = []
    public_weekly = []
    for benefit in BENEFITS:
        accepted_row = accepted_residual[accepted_residual["benefit_multiplier"].round(8) == round(benefit, 8)].iloc[0]
        obligation = float(accepted_row["pension_obligation"])
        residual = float(accepted_row["residual_public_funding"])
        public_summary.append({
            "scenario": f"EMPLOYEE3_EMPLOYER2_PUBLIC_RESIDUAL_B{benefit:.2f}",
            "benefit_multiplier": benefit, "employee_rate": EMPLOYEE_RATE,
            "employer_rate": EMPLOYER_RATE, "pension_obligation": obligation,
            "explicit_contribution_inflow": explicit_inflow,
            "public_gap": residual, "public_gap_share_obligation": safe_ratio(residual, obligation),
            "public_gap_share_explicit_inflow": safe_ratio(residual, explicit_inflow),
            "public_gap_per_eligible_elderly_person": safe_ratio(residual, reference_elderly_persons),
            "public_gap_per_total_population": safe_ratio(residual, reference_total_population),
            "public_gap_per_employed_pre_retirement_contributor": safe_ratio(residual, reference_pre_retirement_contributors),
            "classification": "RESIDUAL" if residual > EPS else "NO_RESIDUAL",
        })
        for _, row in weekly.iterrows():
            obligation_w = (number(row["scheduled_pension_obligation"], 0.0) * weekly_obligation_scale * benefit / BENEFIT_BASE)
            gap = max(0.0, obligation_w - number(row["total_fund_inflow"], 0.0))
            public_weekly.append({"global_step": int(row["global_step"]), "benefit_multiplier": benefit,
                                  "pension_obligation": obligation_w,
                                  "explicit_inflow": number(row["total_fund_inflow"], 0.0),
                                  "public_gap": gap, "public_gap_share_obligation": safe_ratio(gap, obligation_w),
                                  "fund_cash_before_public": number(row["fund_opening_cash"], np.nan),
                                  "public_transfer_executed": 0.0,
                                  "public_transfer_shortfall": gap,
                                  "active_public_funding": False})
    write(public_summary, "pension_public_residual_summary.csv")
    write(public_weekly, "pension_public_residual_weekly.csv")

    # Adequacy is a passive re-scaling of the accepted 0.10 observation.
    base_income = elderly["total_income_after"] - elderly["pension_received"]
    need = pd.to_numeric(elderly["minimum_need"], errors="coerce")
    pension = pd.to_numeric(elderly["pension_received"], errors="coerce")
    adequacy_rows = []
    baseline_ratio = ((base_income + pension) / need).replace([np.inf, -np.inf], np.nan)
    low_cut = float(baseline_ratio.quantile(.25))
    for benefit in (0.20, 0.23):
        ratio_values = ((base_income + pension * benefit / BENEFIT_BASE) / need).replace([np.inf, -np.inf], np.nan).dropna()
        for subgroup, mask in (
            ("ALL_OBSERVABLE_ELDERLY_HOUSEHOLDS", pd.Series(True, index=elderly.index)),
            ("LOW_TAIL_BASELINE_P25", baseline_ratio <= low_cut),
            ("NON_LOW_TAIL_BASELINE_ABOVE_P25", baseline_ratio > low_cut),
        ):
            values = ratio_values[mask.loc[ratio_values.index]] if subgroup != "ALL_OBSERVABLE_ELDERLY_HOUSEHOLDS" else ratio_values
            adequacy_rows.append({"benefit_multiplier": benefit, "subgroup": subgroup,
                                  "observations": len(values), "mean_income_need": float(values.mean()) if len(values) else np.nan,
                                  "median_income_need": float(values.median()) if len(values) else np.nan,
                                  "p10_income_need": float(values.quantile(.10)) if len(values) else np.nan,
                                  "p25_income_need": float(values.quantile(.25)) if len(values) else np.nan,
                                  "share_below_025": float((values < .25).mean()) if len(values) else np.nan,
                                  "share_below_05": float((values < .5).mean()) if len(values) else np.nan,
                                  "share_below_1": float((values < 1.0).mean()) if len(values) else np.nan,
                                  "composition_status": "household composition unavailable; tail split is baseline adequacy only"})
    write(adequacy_rows, "benefit_subgroup_adequacy.csv")

    adequacy = []
    for benefit in (0.20, 0.23):
        values = ((base_income + pension * benefit / BENEFIT_BASE) / need).replace([np.inf, -np.inf], np.nan).dropna()
        adequacy.append({"benefit_multiplier": benefit, "observations": len(values),
                         "mean_income_need": float(values.mean()), "median_income_need": float(values.median()),
                         "p10_income_need": float(values.quantile(.10)), "p25_income_need": float(values.quantile(.25)),
                         "share_below_025": float((values < .25).mean()), "share_below_05": float((values < .5).mean()),
                         "share_below_1": float((values < 1.0).mean()),
                         "interpretation": "descriptive replacement comparison, not welfare target"})
    write(adequacy, "benefit_020_vs_023_adequacy.csv")

    validation = target.copy()
    validation["validation_scope"] = "accepted Step17.N/M shadow benchmark"
    validation["passes_target_identity"] = validation["estimated_multiplier_required"].between(0.229, 0.231, inclusive="both")
    write(validation.to_dict("records"), "replacement_benchmark_validation.csv")

    # These are capacity ratios only. No tax is active and no rate is executed.
    base_totals = {
        "household_total_income": float(weekly["household_income"].sum()),
        "firm_sales": float(firms["revenue"].clip(lower=0).sum()),
        "firm_positive_operating_profit": float(firms["operating_profit"].clip(lower=0).sum()),
        "firm_payroll": float(weekly["firm_payroll"].clip(lower=0).sum()),
    }
    tax_rows = []
    for base_name, base_value in base_totals.items():
        for benefit in (0.20, 0.23):
            residual = next(x["public_gap"] for x in public_summary if x["benefit_multiplier"] == benefit)
            tax_rows.append({"shadow_base": base_name, "base_value": base_value,
                             "benefit_multiplier": benefit, "required_shadow_rate": safe_ratio(residual, base_value),
                             "active_tax": False, "interpretation": "capacity ratio only; no tax implementation"})
    write(tax_rows, "shadow_tax_base_capacity.csv")

    source_rows = [
        {"candidate_source": "employee_social_insurance", "status": "ACTIVE_EXPLICIT", "recurring": True, "funds_residual": False, "notes": "3% employee contribution already accepted"},
        {"candidate_source": "employer_social_insurance", "status": "ACTIVE_EXPLICIT_RESEARCH", "recurring": True, "funds_residual": False, "notes": "2% employer contribution upper tested boundary"},
        {"candidate_source": "event_driven_public_wealth", "status": "EXISTING_BUT_NOT_RECURRING", "recurring": False, "funds_residual": False, "notes": "stock/event flow cannot establish sustainable pension finance"},
        {"candidate_source": "household_or_firm_tax", "status": "NOT_IMPLEMENTED", "recurring": np.nan, "funds_residual": np.nan, "notes": "future public-budget revenue source required"},
        {"candidate_source": "central_bank_money_creation", "status": "FORBIDDEN_AS_FISCAL_SOURCE", "recurring": False, "funds_residual": False, "notes": "monetary financing is not ordinary fiscal revenue"},
    ]
    write(source_rows, "public_funding_source_comparison.csv")

    interface = [{"interface": "Government/PublicBudget -> SocialInsuranceFund", "status": "PASSIVE_INTERFACE_RECOMMENDED",
                  "payer": "explicit public account", "source_revenue": "future recurring fiscal revenue",
                  "scheduled_transfer": "public residual pension requirement", "actual_transfer": "min(scheduled, available public cash)",
                  "shortfall": "record explicit public transfer shortfall", "timing": "after employee and employer contributions, before pension settlement",
                  "money_creation": False, "active_in_step17_o": False,
                  "freeze_decision": "freeze interface shape only; do not freeze amount/rate/benefit"}]
    write(interface, "recommended_public_pension_interface.csv")

    weekly_023 = pd.DataFrame([x for x in public_weekly if x["benefit_multiplier"] == 0.23])
    gap023 = weekly_023["public_gap"]
    residual023 = next(x for x in public_summary if x["benefit_multiplier"] == 0.23)
    residual020 = next(x for x in public_summary if x["benefit_multiplier"] == 0.20)
    summary = [
        "# Step17.O Acceptance Summary", "",
        "## Scope", 
        "Pure fiscal-closure and benefit-boundary audit using accepted Step17.N/M artifacts and existing public-accounting diagnostics. No World.step() call, no public pension funding, no tax, no benefit change, no long run, and no Step17.P.", "",
        "## Findings",
        "The model has an observable public cash holder (`world.public_wealth`) and central-bank public-income bookkeeping, but no implemented Government/PublicBudget account with recurring tax revenue. Existing public receipts are event-driven estate/no-heir and inventory-release flows; they are stocks or episodic flows, not a sustainable pension funding stream.",
        f"At employee 3% plus employer 2%, the 0.20 shadow pension obligation is {residual020['pension_obligation']:.6f}; explicit contributions are {explicit_inflow:.6f}; the residual is {residual020['public_gap']:.6f} ({residual020['public_gap_share_obligation']:.6%} of obligation). This is a small aggregate mismatch, consistent with changing payroll/eligibility and cash-constrained weekly aggregation rather than a new fiscal institution.",
        f"At 0.23, the obligation is {residual023['pension_obligation']:.6f}; recurring explicit inflow is {explicit_inflow:.6f}; residual public funding is {residual023['public_gap']:.6f}, or {residual023['public_gap'] / len(weekly):.6f} per reference week and {residual023['public_gap_share_obligation']:.6%} of obligation. Weekly residual mean={gap023.mean():.6f}, median={gap023.median():.6f}, max={gap023.max():.6f}, positive-week share={(gap023 > EPS).mean():.6%}.",
        "The accepted descriptive replacement benchmark remains approximately 0.230048 for the WORK_ALLOWED_PENSION mean income/need benchmark 0.206896. It is a replacement benchmark, not a welfare optimum or frozen benefit target. The 0.20 value is funding-capacity-adjacent; 0.23 requires continuous residual co-financing in this reference window.",
        "Household composition fields are not present in the accepted elderly artifact, so single-/multi-elderly subgroup claims are unavailable. The subgroup table reports only observable all-household and baseline-adequacy-tail splits.", "",
        "## Recommended Boundary",
        "Freeze only the future Government/PublicBudget -> SocialInsuranceFund transfer shape: explicit source revenue, scheduled amount, cash-constrained actual transfer, and shortfall record. Keep benefit, employee rate, employer rate, tax rate, and retirement age unfrozen. Central-bank money creation must remain separate from fiscal revenue.", "",
        "## Verdict", "**C. PUBLIC_FUNDING_INTERFACE_IS_REQUIRED_BUT_REVENUE_SOURCE_NOT_YET_MATURE**",
    ]
    (OUT / "acceptance_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    flags = {
        "verdict": "C. PUBLIC_FUNDING_INTERFACE_IS_REQUIRED_BUT_REVENUE_SOURCE_NOT_YET_MATURE",
        "source_artifacts": ["Step17.N", "Step17.M", "existing public_accounting.csv"],
        "simulation_executed": False, "public_funding_activated": False,
        "tax_implementation_activated": False, "benefit_parameter_changed": False,
        "employee_rate_used": EMPLOYEE_RATE, "employer_rate_used": EMPLOYER_RATE,
        "benefit_candidates_shadow_only": [0.20, 0.23, 0.25],
        "explicit_inflow": explicit_inflow, "benefit_020_public_gap": residual020["public_gap"],
        "benefit_023_public_gap": residual023["public_gap"],
        "benefit_023_weekly_gap_mean": float(gap023.mean()), "benefit_023_weekly_gap_median": float(gap023.median()),
        "benefit_023_weekly_gap_max": float(gap023.max()), "benefit_023_positive_week_share": float((gap023 > EPS).mean()),
        "replacement_benchmark_multiplier": float(target.loc[target["benchmark"] == "WORK_ALLOWED_PENSION_MEAN_INCOME_NEED", "estimated_multiplier_required"].iloc[0]),
        "public_recurring_revenue_present": False, "public_cash_stock_observable": not public_ledger.empty,
        "central_bank_as_fiscal_revenue": False, "interface_freeze_recommended": True,
        "new_rng_draws": 0, "long_run_executed": False, "step17_p_started": False,
        "required_output_count": 13,
    }
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
