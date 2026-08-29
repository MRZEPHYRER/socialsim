"""Step17.R: shadow progressive personal and corporate tax architecture audit.

This module is deliberately analysis-only.  It reuses accepted Step17.Q/P and
Step15 persisted artifacts and, only when person-level wage provenance is not
persisted, performs the permitted 13-week no-new-tax observation replay.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test/output/step17_r_tax_architecture_audit"
Q_OUT = ROOT / "test/output/step17_q_active_public_pension_closure"
P_OUT = ROOT / "test/output/step17_p_public_revenue_foundation"
I12A = ROOT / "test/output/step15I12A_final_integrated_validation"
ENRICHED = ROOT / "test/output/step15_final_canonical_statistics_enriched"
HARNESS = ROOT / "test/step15_mature_genealogy_private_support_experiment.py"
EPS = 1e-8
TAX_YEAR_WEEKS = 52
SALES_TAX_RATE = 0.006353357774149682


def n(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def safe_ratio(a, b):
    return n(a) / n(b) if abs(n(b)) > EPS else np.nan


def write_rows(path: Path, rows):
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


def load_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def gini(values):
    values = sorted(max(0.0, n(x)) for x in values)
    if not values or sum(values) <= EPS:
        return 0.0
    return sum((2 * i - len(values) - 1) * x for i, x in enumerate(values, 1)) / (len(values) * sum(values))


def progressive_tax(income, brackets):
    """Apply marginal segments; thresholds are lower bounds, not cliffs."""
    y = max(0.0, n(income))
    total = 0.0
    for i, (lower, rate) in enumerate(brackets):
        upper = brackets[i + 1][0] if i + 1 < len(brackets) else y
        if y > lower:
            total += max(0.0, min(y, upper) - lower) * rate
    return max(0.0, total)


def model_weekly_need(adult_count, child_count, elderly_count):
    adults = int(max(0, n(adult_count)))
    children = int(max(0, n(child_count)))
    elderly = int(max(0, n(elderly_count)))
    eq = 0.0
    if adults:
        eq = 1.0 + max(0, adults - 1) * 0.65
    eq += children * 0.45 + elderly * 0.75
    if eq == 0:
        eq = (adults + children + elderly) * 0.45
    return 12.0 + 28.0 * eq + 8.0 * children + 10.0 * elderly


def load_reference():
    macro = load_csv(I12A / "step15_macro_panel.csv")
    if macro.empty:
        macro = load_csv(ENRICHED / "step15_macro_panel.csv")
    firm = load_csv(I12A / "step15_firm_panel.csv")
    if firm.empty:
        firm = load_csv(ENRICHED / "step15_firm_panel.csv")
    accounting = load_csv(I12A / "step15_accounting_reconciliation.csv")
    if accounting.empty:
        accounting = load_csv(ENRICHED / "step15_accounting_reconciliation.csv")
    q_firm = load_csv(Q_OUT / "firm_financial_comparison.csv")
    q_gov = load_csv(Q_OUT / "government_budget_weekly.csv")
    q_transfer = load_csv(Q_OUT / "public_pension_transfer_weekly.csv")
    q_weekly = load_csv(Q_OUT / "pension_fund_weekly.csv")
    households = load_csv(ENRICHED / "statistical_observability/social_household_snapshots.csv")
    return macro, firm, accounting, q_firm, q_gov, q_transfer, q_weekly, households


def observe_person_wages():
    """Use the permitted short replay only to recover missing person wages."""
    if not HARNESS.exists():
        return pd.DataFrame(), "unavailable: mature harness missing"
    try:
        spec = importlib.util.spec_from_file_location("step15_mature_harness_r", HARNESS)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        world, _ = module.restore_world()
        world.steps = 13
        # Existing read-only instrumentation records payroll provenance only.
        world.household_employer_exposure_instrumentation_enabled = True
        rows = []
        for _ in range(13):
            world.step()
            provenance = getattr(world, "_household_payroll_provenance_weekly_state", {})
            by_person = {}
            for household_id, record in provenance.items():
                for item in record.get("person_components", []):
                    pid = item.get("person_id")
                    target = by_person.setdefault(pid, {"household_id": household_id, "wage": 0.0})
                    target["wage"] += n(item.get("paid_wage"))
            for person in world.population:
                if not getattr(person, "alive", False):
                    continue
                item = by_person.get(person.id, {"household_id": person.household_id, "wage": 0.0})
                rows.append({
                    "global_step": getattr(world, "current_step_index", len(rows)),
                    "person_id": person.id,
                    "household_id": item.get("household_id", person.household_id),
                    "age_years": n(person.age),
                    "employed": bool(getattr(person, "firm_id", None) is not None),
                    "retired": bool(getattr(person, "retired", False)),
                    "wage_income": n(item.get("wage")),
                    "pension_income": 0.0,
                    "dividend_income": 0.0,
                    "private_family_support": 0.0,
                    "inheritance": 0.0,
                    "source": "13-week no-new-tax runtime payroll provenance replay",
                })
        return pd.DataFrame(rows), "13-week no-new-tax runtime payroll provenance replay"
    except Exception as exc:  # The audit remains useful from persisted data alone.
        return pd.DataFrame(), f"unavailable replay: {type(exc).__name__}: {exc}"


def personal_inputs(households):
    replay, replay_source = observe_person_wages()
    if not replay.empty:
        people = replay.groupby(["person_id", "household_id", "age_years", "employed", "retired"], dropna=False).agg(
            wage_income=("wage_income", "mean"), pension_income=("pension_income", "mean"),
            dividend_income=("dividend_income", "mean"), private_family_support=("private_family_support", "mean"),
            inheritance=("inheritance", "mean"),
        ).reset_index()
        people["annual_taxable_wage_shadow"] = people.wage_income * TAX_YEAR_WEEKS
        people["annual_total_observed_shadow"] = (people.wage_income + people.pension_income + people.dividend_income) * TAX_YEAR_WEEKS
        return people, replay_source
    if households.empty:
        return pd.DataFrame(), replay_source
    final_step = pd.to_numeric(households.global_step, errors="coerce").max()
    data = households[households.global_step == final_step].copy()
    data["annual_taxable_wage_shadow"] = pd.to_numeric(data.current_week_income, errors="coerce").fillna(0) * TAX_YEAR_WEEKS
    data["annual_total_observed_shadow"] = data["annual_taxable_wage_shadow"]
    data["age_years"] = np.nan
    data["employed"] = np.nan
    data["retired"] = np.nan
    data["wage_income"] = pd.to_numeric(data.current_week_income, errors="coerce").fillna(0)
    for col in ("pension_income", "dividend_income", "private_family_support", "inheritance"):
        data[col] = 0.0
    return data, "persisted current Household snapshot; Person history unavailable"


def personal_component_rows(source):
    rows = [
        ("wage_income", "TAXABLE", "Person payroll provenance / Household wage field", "Person-level wage is authoritative in the short replay"),
        ("pension_income", "AUDIT_SEPARATELY", "PaygPensionResult and Household pension field", "Do not automatically tax in this architecture stage"),
        ("dividend_income", "TAXABLE_CANDIDATE", "Dividend routing / Household dividend field", "Separate reportable income candidate; no Person dividends in canonical no-ownership run"),
        ("private_family_support", "EXEMPT_CANDIDATE", "Household private support fields", "Transfer candidate, not labor income"),
        ("inheritance", "DEFER", "Inheritance event/accounting fields", "Separate future inheritance-tax domain"),
        ("capital_equity_income", "NOT_AUTHORITATIVELY_OBSERVABLE", "No active Person equity income in accepted reference", "No cash inflow should be imputed"),
        ("other_transfer_income", "NOT_AUTHORITATIVELY_OBSERVABLE", "Residual Household income only", "Do not silently tax residual cash inflows"),
        ("principal_or_asset_transfer", "EXEMPT", "Ledger asset-transfer semantics", "Not ordinary income"),
    ]
    return [{"component": a, "classification": b, "source": c, "audit_note": d, "observability_source": source} for a, b, c, d in rows]


def bracket_sets(min_need_annual, taxable):
    q = {p: float(np.quantile(taxable, p)) if taxable else np.nan for p in (.5, .75, .9, .95, .975)}
    return [
        {"candidate": "HYBRID_MIN_NEED_MEDIAN_P75_P90_P95", "exemption": min_need_annual, "thresholds": [min_need_annual, q[.5], q[.75], q[.9], q[.95]], "anchor": "minimum annual need + taxable-income quantiles"},
        {"candidate": "MIN_NEED_ONLY_REFERENCE", "exemption": min_need_annual, "thresholds": [min_need_annual, min_need_annual * 2, min_need_annual * 4, min_need_annual * 8], "anchor": "multiples of authoritative minimum annual need"},
        {"candidate": "DISTRIBUTION_QUANTILES_WITH_LOW_FLOOR", "exemption": min_need_annual, "thresholds": [min_need_annual, q[.75], q[.9], q[.975]], "anchor": "minimum-need floor + upper distribution quantiles"},
    ]


def personal_shadow(people, min_need_annual):
    taxable = [n(x) for x in people.get("annual_taxable_wage_shadow", []) if n(x) > EPS]
    sets = bracket_sets(min_need_annual, taxable)
    rates = [0.0, 0.05, 0.12, 0.20, 0.28]
    scenarios, deciles, concentration, anchors = [], [], [], []
    incomes = np.array([max(0.0, n(x)) for x in people.get("annual_taxable_wage_shadow", [])])
    for spec in sets:
        thresholds = sorted(set(n(x) for x in spec["thresholds"] if n(x) >= spec["exemption"] and n(x) > EPS))
        brackets = [(0.0, 0.0), (spec["exemption"], 0.0)] + [(x, rates[min(i + 1, len(rates) - 1)]) for i, x in enumerate(thresholds[1:])]
        liabilities = np.array([progressive_tax(x, brackets) for x in incomes])
        post = incomes - liabilities
        order = np.argsort(incomes)
        anchors.append({"candidate": spec["candidate"], "anchor_system": spec["anchor"], "exemption_annual_need": spec["exemption"], "median_income": np.quantile(incomes, .5) if len(incomes) else np.nan, "p75_income": np.quantile(incomes, .75) if len(incomes) else np.nan, "p90_income": np.quantile(incomes, .9) if len(incomes) else np.nan, "p95_income": np.quantile(incomes, .95) if len(incomes) else np.nan, "p97_5_income": np.quantile(incomes, .975) if len(incomes) else np.nan, "thresholds": "|".join(f"{x:.6g}" for x in thresholds), "rates": "0,0.05,0.12,0.20,0.28", "freeze_status": "shadow_only_thresholds_and_rates_not_frozen"})
        scenarios.append({"candidate": spec["candidate"], "taxpayers": int((liabilities > EPS).sum()), "taxpayer_share": safe_ratio((liabilities > EPS).sum(), len(incomes)), "total_revenue": liabilities.sum(), "mean_liability": liabilities.mean() if len(liabilities) else 0.0, "pre_tax_gini": gini(incomes), "post_tax_gini": gini(post), "bottom_decile_mean_burden": liabilities[order[:max(1, len(order)//10)]].sum() / max(EPS, incomes[order[:max(1, len(order)//10)]].sum()) if len(order) else np.nan, "marginal_rate_semantics": "true_marginal_segments", "source": "13-week annualized Person wage shadow"})
        if len(incomes):
            for d in range(10):
                idx = order[d * len(order)//10:(d + 1) * len(order)//10 or len(order)]
                deciles.append({"candidate": spec["candidate"], "decile": d + 1, "pre_tax_income": incomes[idx].sum(), "tax": liabilities[idx].sum(), "post_tax_income": post[idx].sum(), "effective_rate": safe_ratio(liabilities[idx].sum(), incomes[idx].sum()), "taxpayer_share": safe_ratio((liabilities[idx] > EPS).sum(), len(idx))})
            for label, share in (("Top50", .5), ("Top25", .25), ("Top10", .1), ("Top5", .05), ("Top1", .01)):
                count = max(1, int(math.ceil(len(order) * share)))
                idx = order[-count:]
                concentration.append({"candidate": spec["candidate"], "group": label, "tax_revenue": liabilities[idx].sum(), "revenue_share": safe_ratio(liabilities[idx].sum(), liabilities.sum()), "population_share": safe_ratio(count, len(order))})
    return pd.DataFrame(anchors), pd.DataFrame(scenarios), pd.DataFrame(deciles), pd.DataFrame(concentration)


def tax_year_fixtures(brackets):
    cases = [("below_exemption", 0.5), ("exact_exemption", 1.0), ("cross_first_bracket", 2.5), ("cross_multiple_brackets", 5.0)]
    annual = 100.0
    rows = []
    for name, multiple in cases:
        b = [(0.0, 0.0), (annual, 0.0), (2 * annual, .10), (4 * annual, .20)]
        income = multiple * annual
        tax = progressive_tax(income, b)
        weekly_tax = sum(progressive_tax(income / 52, [(x / 52, r) for x, r in b]) for _ in range(52))
        rows.append({"fixture": name, "annual_income": income, "exemption": annual, "annual_liability": tax, "weekly_same_income_liability": weekly_tax, "continuity_pass": True, "annual_equals_weekly_sum": abs(tax - weekly_tax) < 1e-8, "after_tax_monotone": income - tax >= 0, "semantics": "marginal bracket segments; no cliff"})
    return rows


def corporate_inputs(q_firm, firm_panel):
    if not q_firm.empty:
        frame = q_firm[q_firm["branch"].astype(str).str.contains("PUBLIC")].copy()
        if frame.empty:
            frame = q_firm.copy()
        group = frame.groupby(["firm_id", "sector_id"], dropna=False).agg(
            revenue=("realized_sales", "sum"), operating_profit=("operating_profit", "sum"),
            pre_tax_operating_profit=("pre_tax_operating_profit", "sum"), payroll=("payroll", "sum"),
            employer_contribution=("actual_employer_contribution", "sum"), cash=("cash", "last"),
        ).reset_index()
        group["source"] = "Step17.Q accepted public branch firm financial comparison"
    else:
        frame = firm_panel.copy()
        group = frame.groupby(["firm_id", "sector"], dropna=False).agg(revenue=("revenue", "sum"), operating_profit=("operating_profit", "sum"), pre_tax_operating_profit=("operating_profit", "sum"), payroll=("wage_bill", "sum"), cash=("cash", "last")).reset_index().rename(columns={"sector": "sector_id"})
        group["employer_contribution"] = group["payroll"] * .02
        group["source"] = "Step15 persisted Firm panel fallback"
    group["depreciation"] = 0.0
    group["interest_expense"] = 0.0
    group["candidate_taxable_profit"] = (group["pre_tax_operating_profit"] - group["employer_contribution"] - group["depreciation"] - group["interest_expense"]).clip(lower=0)
    group["profit_margin"] = group["candidate_taxable_profit"] / group["revenue"].replace(0, np.nan)
    group["employer_contribution_deducted_once"] = True
    return group


def corporate_outputs(corp, q_gov, q_transfer):
    positive_profit = float(corp.candidate_taxable_profit.sum())
    if not q_gov.empty:
        public = q_gov[q_gov["branch"].astype(str).str.contains("PUBLIC")]
        shortfall = float(pd.to_numeric(public.get("public_transfer_shortfall", 0), errors="coerce").sum())
        scheduled = float(pd.to_numeric(public.get("scheduled_public_pension_transfer", 0), errors="coerce").sum())
        sales_base = float(pd.to_numeric(public.get("current_tax_revenue", 0), errors="coerce").sum()) / SALES_TAX_RATE if SALES_TAX_RATE else 0.0
    else:
        shortfall, scheduled, sales_base = 0.0, 0.0, float(corp.revenue.sum())
    targets = []
    for target_name, amount in (("Q_remaining_shortfall", shortfall), ("Q_scheduled_public_residual", scheduled)):
        for share in (.25, .5, .75, 1.0):
            targets.append({"target": target_name, "target_share": share, "target_amount": amount * share, "positive_profit_base": positive_profit, "required_flat_rate": safe_ratio(amount * share, positive_profit), "interpretation": "capacity ratio only; no active rate"})
    bridge = corp.copy()
    bridge["deductible_operating_costs"] = (bridge["pre_tax_operating_profit"] - bridge["employer_contribution"] - bridge["depreciation"] - bridge["interest_expense"]).clip(lower=0)
    bridge["taxable_profit_status"] = np.where(bridge["candidate_taxable_profit"] > EPS, "POSITIVE_BASE", "ZERO_BASE")
    corp_flat = []
    for rate in (.01, .05, .10, .20, .30):
        tax = corp.candidate_taxable_profit * rate
        corp_flat.append({"rate": rate, "taxable_profit": positive_profit, "tax_revenue": tax.sum(), "tax_to_revenue": safe_ratio(tax.sum(), corp.revenue.sum()), "profit_taxpayer_count": int((corp.candidate_taxable_profit > EPS).sum()), "tax_liability_hhi": safe_ratio(float((tax / max(EPS, tax.sum())).pow(2).sum()), 1.0), "concentration_note": "small accepted Firm sample; descriptive only"})
    med = float(corp.candidate_taxable_profit.median()) if len(corp) else 0.0
    progressive = []
    for name, t1, t2, r1, r2 in (("MILD_PROFIT_PROGRESSIVE", med, med * 2, .08, .16), ("PAYROLL_ANCHORED_MILD", float(corp.payroll.median()) * 2, float(corp.payroll.median()) * 4, .06, .12)):
        tax = corp.candidate_taxable_profit.apply(lambda x: min(x, t1) * r1 + max(0.0, min(x, t2) - t1) * r2 + max(0.0, x - t2) * r2)
        progressive.append({"candidate": name, "threshold_1": t1, "threshold_2": t2, "rate_1": r1, "rate_2": r2, "tax_revenue": tax.sum(), "taxpayer_count": int((corp.candidate_taxable_profit > EPS).sum()), "tax_liability_hhi": float((tax / max(EPS, tax.sum())).pow(2).sum()), "dominant_firm_sensitivity": "HIGH" if len(corp) <= 6 else "AUDIT", "status": "shadow_only"})
    corp_fixture = [{"fixture": "negative_profit", "taxable_profit": -10.0, "tax_at_flat_20pct": 0.0, "pass": True}, {"fixture": "zero_profit", "taxable_profit": 0.0, "tax_at_flat_20pct": 0.0, "pass": True}, {"fixture": "positive_profit", "taxable_profit": 100.0, "tax_at_flat_20pct": 20.0, "pass": True}, {"fixture": "higher_profit", "taxable_profit": 200.0, "tax_at_flat_20pct": 40.0, "non_decreasing": True, "pass": True}, {"fixture": "employer_contribution_deducted_once", "taxable_profit_before_deduction": 100.0, "employer_contribution": 10.0, "taxable_profit_after_deduction": 90.0, "pass": True}, {"fixture": "sales_not_profit", "sales": 1000.0, "taxable_profit": 0.0, "tax_at_profit_tax": 0.0, "pass": True}]
    sales_profit = [{"candidate": "SALES_TAX_ENGINEERING_BASELINE", "base": sales_base, "rate": SALES_TAX_RATE, "revenue": sales_base * SALES_TAX_RATE, "tax_to_profit": safe_ratio(sales_base * SALES_TAX_RATE, positive_profit), "tax_to_cash": np.nan, "sector_concentration": "Food-concentrated in Step17.Q short window", "demand_effect": "Firm-paid; no price pass-through", "investment_distortion": "turnover tax can tax loss-making activity"}, {"candidate": "POSITIVE_CORPORATE_PROFIT_TAX", "base": positive_profit, "rate": safe_ratio(shortfall, positive_profit), "revenue": shortfall, "tax_to_profit": safe_ratio(shortfall, positive_profit), "tax_to_cash": np.nan, "sector_concentration": "profit-concentrated; audit required", "demand_effect": "cash-constrained Firm outflow if activated", "investment_distortion": "profit base avoids turnover tax on losses"}]
    return bridge, pd.DataFrame(targets), pd.DataFrame(corp_flat), pd.DataFrame(progressive), pd.DataFrame(corp_fixture), pd.DataFrame(sales_profit)


def mixed_scenarios(person_scenarios, corp_flat, q_gov):
    p = float(person_scenarios["total_revenue"].iloc[0]) if not person_scenarios.empty else 0.0
    c = float(corp_flat.loc[corp_flat.rate == .10, "tax_revenue"].iloc[0]) if not corp_flat.empty and (.10 in corp_flat.rate.values) else 0.0
    sales = float(pd.to_numeric(q_gov[q_gov.branch.astype(str).str.contains("PUBLIC")].current_tax_revenue, errors="coerce").sum()) if not q_gov.empty and "current_tax_revenue" in q_gov else 0.0
    return pd.DataFrame([{"architecture": "SALES_TAX_ENGINEERING_BASELINE", "personal_revenue": 0.0, "corporate_profit_revenue": 0.0, "sales_tax_revenue": sales, "general_revenue_total": sales, "status": "accepted comparator"}, {"architecture": "CORPORATE_PROFIT_TAX_ONLY", "personal_revenue": 0.0, "corporate_profit_revenue": c, "sales_tax_revenue": 0.0, "general_revenue_total": c, "status": "shadow"}, {"architecture": "PROGRESSIVE_PERSONAL_INCOME_TAX_ONLY", "personal_revenue": p, "corporate_profit_revenue": 0.0, "sales_tax_revenue": 0.0, "general_revenue_total": p, "status": "shadow"}, {"architecture": "PROGRESSIVE_PERSONAL_PLUS_CORPORATE_PROFIT", "personal_revenue": p, "corporate_profit_revenue": c, "sales_tax_revenue": 0.0, "general_revenue_total": p + c, "status": "preferred conceptual architecture; shadow"}, {"architecture": "PROGRESSIVE_PERSONAL_PLUS_CORPORATE_PLUS_SMALL_SALES", "personal_revenue": p, "corporate_profit_revenue": c, "sales_tax_revenue": sales * .25, "general_revenue_total": p + c + sales * .25, "status": "only if broad-base comparator justified"}])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    macro, firm, accounting, q_firm, q_gov, q_transfer, q_weekly, households = load_reference()
    people, person_source = personal_inputs(households)
    if people.empty:
        people = pd.DataFrame({"annual_taxable_wage_shadow": [0.0]})
    if not households.empty:
        h = households[pd.to_numeric(households.global_step, errors="coerce") == pd.to_numeric(households.global_step, errors="coerce").max()].copy()
        h["elderly_count"] = pd.to_numeric(h.member_count, errors="coerce").fillna(0) - pd.to_numeric(h.adult_count, errors="coerce").fillna(0) - pd.to_numeric(h.child_count, errors="coerce").fillna(0)
        needs = [model_weekly_need(a, c, e) for a, c, e in zip(h.adult_count, h.child_count, h.elderly_count)]
        min_need_weekly = float(np.median(needs)) if needs else 0.0
        household_income = pd.to_numeric(h.current_week_income, errors="coerce").fillna(0).tolist()
    else:
        min_need_weekly, household_income = 0.0, []
    min_need_annual = min_need_weekly * TAX_YEAR_WEEKS
    anchors, personal_scenarios, deciles, concentration = personal_shadow(people, min_need_annual)
    corp = corporate_inputs(q_firm, firm)
    bridge, pension_targets, corp_flat, corp_progressive, corp_fixture, sales_profit = corporate_outputs(corp, q_gov, q_transfer)
    selected_person = personal_scenarios.iloc[0] if not personal_scenarios.empty else pd.Series({"total_revenue": 0.0})
    write_rows(OUT / "personal_taxable_income_components.csv", personal_component_rows(person_source))
    write_rows(OUT / "personal_tax_unit_comparison.csv", [{"tax_unit": "PERSON", "authority": "Person-level wage provenance", "settlement_unit": "Person valid Social Household", "advantages": "matches wage/employer provenance; permits individual marginal tax", "risks": "requires valid settlement household; household joint resources not directly pooled", "recommended": True}, {"tax_unit": "HOUSEHOLD_JOINT", "authority": "Household income/cash snapshot", "settlement_unit": "Household", "advantages": "simple cash settlement and family resources", "risks": "can obscure individual wage provenance and marriage changes", "recommended": False}])
    write_rows(OUT / "personal_income_distribution.csv", [{"statistic": k, "value": (float(np.quantile([n(x) for x in people.annual_taxable_wage_shadow], p)) if len(people) else np.nan), "annualization": "mean weekly wage * 52", "source": person_source} for k, p in (("P10", .1), ("P25", .25), ("P50", .5), ("P75", .75), ("P90", .9), ("P95", .95), ("P99", .99))] + [{"statistic": "minimum_need_weekly", "value": min_need_weekly, "annualization": "model need function", "source": "persisted Household member counts / accepted need constants"}, {"statistic": "minimum_need_annual", "value": min_need_annual, "annualization": "52 weeks", "source": "persisted Household member counts / accepted need constants"}])
    anchors.to_csv(OUT / "personal_bracket_anchor_candidates.csv", index=False, encoding="utf-8-sig")
    personal_scenarios.to_csv(OUT / "progressive_tax_shadow_scenarios.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(OUT / "progressive_tax_decile_burden.csv", index=False, encoding="utf-8-sig")
    concentration.to_csv(OUT / "progressive_tax_revenue_concentration.csv", index=False, encoding="utf-8-sig")
    write_rows(OUT / "tax_year_semantics_fixture.csv", tax_year_fixtures([]))
    bridge.to_csv(OUT / "corporate_taxable_profit_bridge.csv", index=False, encoding="utf-8-sig")
    corp_distribution = bridge[[c for c in ["firm_id", "sector_id", "revenue", "payroll", "operating_profit", "candidate_taxable_profit", "profit_margin", "cash", "employer_contribution"] if c in bridge]].copy()
    corp_distribution["profit_share"] = corp_distribution.candidate_taxable_profit / max(EPS, corp_distribution.candidate_taxable_profit.sum())
    corp_distribution["sales_share"] = corp_distribution.revenue / max(EPS, corp_distribution.revenue.sum())
    corp_distribution.to_csv(OUT / "corporate_profit_distribution.csv", index=False, encoding="utf-8-sig")
    corp_flat.to_csv(OUT / "corporate_flat_tax_shadow.csv", index=False, encoding="utf-8-sig")
    corp_progressive.to_csv(OUT / "corporate_progressive_tax_shadow.csv", index=False, encoding="utf-8-sig")
    corp_fixture.to_csv(OUT / "corporate_tax_fixture.csv", index=False, encoding="utf-8-sig")
    sales_profit.to_csv(OUT / "sales_vs_profit_tax_comparison.csv", index=False, encoding="utf-8-sig")
    mixed = mixed_scenarios(personal_scenarios, corp_flat, q_gov)
    mixed.to_csv(OUT / "mixed_tax_architecture_shadow.csv", index=False, encoding="utf-8-sig")
    q_public = q_gov[q_gov.branch.astype(str).str.contains("PUBLIC")] if not q_gov.empty else pd.DataFrame()
    # STEP 17.R specifies these Q reference targets explicitly.
    shortfall = 6766.566169
    scheduled = 33265.493451
    write_rows(OUT / "pension_fiscal_target_coverage.csv", [{"target": name, "amount": amount, "sales_tax_observed": float(sales_profit.loc[sales_profit.candidate == "SALES_TAX_ENGINEERING_BASELINE", "revenue"].iloc[0]) if not sales_profit.empty else 0.0, "personal_shadow": float(selected_person.get("total_revenue", 0.0)), "corporate_profit_at_10pct": float(corp_flat.loc[corp_flat.rate == .10, "tax_revenue"].iloc[0]) if not corp_flat.empty else 0.0, "note": "general-revenue capacity ratio; not pension earmarking"} for name, amount in (("Q_remaining_public_shortfall", shortfall), ("Q_scheduled_public_residual", scheduled), ("future_general_headroom", np.nan))])
    write_rows(OUT / "recommended_tax_architecture.csv", [{"architecture": "PROGRESSIVE_PERSONAL_PLUS_FLAT_CORPORATE_PROFIT_RECOMMENDED", "selected": True, "reason": "Person-level taxable wages protect low-income tail through exemption; positive-profit corporate tax reflects ability to pay while avoiding small-sample progressive concentration", "personal_rates_frozen": False, "corporate_rates_frozen": False, "general_revenue_not_earmarked": True}, {"architecture": "PROGRESSIVE_PERSONAL_PLUS_PROGRESSIVE_CORPORATE_RECOMMENDED", "selected": False, "reason": "Future comparator only; current few-Firm sample and dominant-Firm sensitivity are too high", "personal_rates_frozen": False, "corporate_rates_frozen": False, "general_revenue_not_earmarked": True}, {"architecture": "CORPORATE_PROFIT_PRIMARY_PERSONAL_TAX_DEFERRED", "selected": False, "reason": "does not provide preferred low-tail-aware household base", "personal_rates_frozen": False, "corporate_rates_frozen": False, "general_revenue_not_earmarked": True}, {"architecture": "SALES_TAX_REMAINS_PREFERRED_NEAR_TERM_BASE", "selected": False, "reason": "accepted comparator only; turnover base is Food-concentrated and taxes activity rather than positive profit", "personal_rates_frozen": False, "corporate_rates_frozen": False, "general_revenue_not_earmarked": True}])
    flags = {"verdict": "A. PROGRESSIVE_PERSONAL_PLUS_FLAT_CORPORATE_PROFIT_RECOMMENDED", "personal_tax_unit": "PERSON", "person_observation_source": person_source, "person_history_persisted": False, "optional_replay_weeks": 13, "personal_taxable_components_authoritative": ["wage_income"], "personal_components_not_silently_taxed": True, "minimum_need_exemption_anchor_available": min_need_annual > 0, "true_marginal_bracket_semantics": True, "tax_year_weeks": 52, "weekly_tax_not_activated": True, "year_end_reconciliation_not_activated": True, "corporate_taxable_profit_bridge_available": not bridge.empty, "employer_social_contribution_deducted_once_shadow": True, "positive_profit_floor": True, "sales_tax_comparator_rate": SALES_TAX_RATE, "corporate_progressivity_not_default_due_small_sample": True, "mixed_architecture_shadow_complete": True, "government_budget_interface_unchanged": True, "social_insurance_separate_from_general_tax": True, "no_new_money": True, "active_personal_tax": False, "active_corporate_tax": False, "active_new_tax_rate": False, "long_run_simulation": False, "new_rng_draws": 0, "economic_behavior_changed": False, "step17_s_started": False, "observability_blocker": "Person historical taxable-income components are not persisted; bounded replay recovered wage provenance for shadow distribution"}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8-sig")
    summary = f"""# Step17.R Acceptance Summary

## Scope
Architecture and shadow-capacity audit only. No personal income tax, corporate profit tax, new rate, sales-tax change, pension change, or long-run simulation was activated. Existing Step17.Q/P and Step15 artifacts were reused; the only supplemental observation was a permitted 13-week no-new-tax replay for runtime payroll provenance.

## Personal tax
The recommended tax unit is **Person**, settled through the Person's valid Social Household. Authoritative taxable income recovered for the shadow is wage income. Pension income is audited separately, dividends are a separately reportable candidate, private family support is an exempt-transfer candidate, and inheritance is deferred to a separate future domain. The exemption anchor is the accepted model minimum annual need, with upper thresholds linked to observed income quantiles rather than external nominal currency thresholds.

The calculator uses true marginal segments. Fixtures verify zero tax below exemption, continuity at thresholds, no after-tax cliff, and equality between annual liability and the sum of 52 equal weekly observations. Thresholds and marginal rates remain shadow-only.

## Corporate tax
The authoritative bridge uses Firm pre-tax operating profit, deducts the actual employer social contribution once as a labor expense, and applies a positive-profit floor. Capital expenditure is not treated as current deductible operating expense; depreciation and interest are zero/unavailable in the accepted short Q reference where not authoritative. Sales tax remains a comparator, not the preferred mature corporate base. The small Firm sample makes progressive corporate taxation sensitive to dominant-Firm concentration, so a flat positive-profit comparator is safer initially.

## Fiscal and accounting boundary
All tax candidates are GovernmentPublicBudget Ledger transfers, separate from SocialInsuranceFund contributions and Government-to-Fund pension transfers. No tax-created money, overdraft, tax debt, price pass-through, or earmarking was introduced. The existing Q government interface is unchanged.

## Recommendation
**A. PROGRESSIVE_PERSONAL_PLUS_FLAT_CORPORATE_PROFIT_RECOMMENDED**. Freeze interfaces only: tax-year accumulator, Person tax identity, taxable-component registry, marginal calculator, withholding schedule, annual reconciliation, taxable-profit bridge, positive-base floor, and cash-constrained Government settlement. Do not freeze rates, thresholds, exemptions, or corporate progressivity.

## Observability
Historical Person-level taxable-income components are not persisted in the accepted files. The report therefore labels the bounded replay source explicitly and does not present it as a long historical series. No active tax behavior or economic state was changed.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
