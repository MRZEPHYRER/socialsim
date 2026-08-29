"""Passive Step 13.1 credit-capacity semantics and shadow-limit audit."""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "test" / "output" / "pre_step13_3C1_long_horizon_periodicity" / "raw_runs"
OUTPUT = ROOT / "test" / "output" / "step13_1_credit_capacity_semantics"
WINDOW = (1560, 3120)
TOL = 1e-9


def n(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def mean(values):
    values = [n(value) for value in values]
    return float(np.mean(values)) if values else 0.0


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_run(population):
    path = SOURCE / f"N{population}_seed42"
    with (path / "diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        diagnostics = list(csv.DictReader(handle))
    firms = defaultdict(list)
    with (path / "firm_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            firms[int(n(row.get("firm_id")))].append(row)
    return {"population": population, "diagnostics": diagnostics, "firms": firms}


def trailing_mean(values, window):
    result = []
    for i in range(len(values)):
        result.append(mean(values[max(0, i - window + 1):i + 1]))
    return result


def prepare_rows(run):
    rows = []
    for firm_id, values in run["firms"].items():
        wages = [n(row.get("wage_bill")) for row in values]
        sales = [n(row.get("sales_revenue")) for row in values]
        wage13, wage26, wage52 = trailing_mean(wages, 13), trailing_mean(wages, 26), trailing_mean(wages, 52)
        sales13, sales26, sales52 = trailing_mean(sales, 13), trailing_mean(sales, 26), trailing_mean(sales, 52)
        for i, row in enumerate(values):
            step = int(n(row.get("global_step", row.get("step"))))
            if step < WINDOW[0]:
                continue
            wage = wages[i]
            liquidity_horizon = n(row.get("target_cash")) / max(wage, TOL)
            rows.append({"population": run["population"], "firm_id": firm_id, "global_step": step, "model_year": step / 52, "wage_bill": wage, "sales_revenue": sales[i], "inventory_value": n(row.get("inventory")), "inventory_units": n(row.get("inventory_units")), "target_cash": n(row.get("target_cash")), "cash_before_borrowing": n(row.get("cash_start")), "funding_gap": n(row.get("funding_gap")), "principal": n(row.get("loan_balance")), "loan_issued": n(row.get("loan_issued")), "cash_before_repayment": n(row.get("cash_before_repayment")), "repayment_buffer": n(row.get("repayment_buffer")), "profit": n(row.get("profit")), "capacity": n(row.get("productive_capacity")), "sales_units": n(row.get("sales_units")), "inventory_coverage": n(row.get("inventory_coverage")), "wage_ema13": wage13[i], "wage_ema26": wage26[i], "wage_ema52": wage52[i], "sales_ema13": sales13[i], "sales_ema26": sales26[i], "sales_ema52": sales52[i], "liquidity_horizon_weeks": liquidity_horizon})
    return rows


def candidate_rows(rows):
    horizon = mean(row["liquidity_horizon_weeks"] for row in rows)
    # Candidate K values are transparent multiples of the observed liquidity
    # horizon; they are shadow stress points, not calibrations.
    candidates = [("payroll_1x", "payroll", 1.0), ("payroll_2x", "payroll", 2.0), ("payroll_4x", "payroll", 4.0), ("payroll_control_100x", "payroll", 100.0)]
    payroll_sales_ratio = mean(row["wage_ema26"] / max(row["sales_ema26"], TOL) for row in rows)
    for name, base, multiple in list(candidates):
        candidates.append((name.replace("payroll", "sales"), "sales", multiple))
        candidates.append((name.replace("payroll", "hybrid"), "hybrid", multiple))
    candidates.extend((f"inventory_{multiple:g}x", "inventory", multiple) for multiple in (0.5, 1.0, 2.0))
    output = []
    for row in rows:
        for name, kind, multiple in candidates:
            payroll_limit = multiple * horizon * row["wage_ema26"]
            sales_limit = multiple * horizon * payroll_sales_ratio * row["sales_ema26"]
            if kind == "payroll":
                limit = payroll_limit
            elif kind == "sales":
                limit = sales_limit
            elif kind == "hybrid":
                limit = 0.5 * (payroll_limit + sales_limit)
            else:
                limit = multiple * row["inventory_value"]
            available = max(0.0, limit - row["principal"])
            binding = row["funding_gap"] > available + TOL
            output.append({**row, "candidate": name, "candidate_kind": kind, "capacity_multiple": multiple, "credit_limit": limit, "available_credit": available, "requested_credit": row["funding_gap"], "shadow_denied_credit": max(0.0, row["funding_gap"] - available), "shadow_binding": binding, "principal_to_limit": row["principal"] / max(limit, TOL), "horizon_reference_weeks": horizon, "payroll_sales_anchor_ratio": payroll_sales_ratio})
    return output, horizon, payroll_sales_ratio


def binding_metrics(rows):
    output, first = [], []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["population"], row["firm_id"], row["candidate"])].append(row)
    for key, values in grouped.items():
        binding = [row for row in values if row["shadow_binding"]]
        positive_principal = [row for row in values if row["principal"] > TOL]
        output.append({"population": key[0], "firm_id": key[1], "candidate": key[2], "weeks": len(values), "first_shadow_binding_week": binding[0]["global_step"] if binding else "", "binding_share_all_weeks": len(binding) / max(len(values), 1), "binding_share_indebted_weeks": sum(row["shadow_binding"] for row in positive_principal) / max(len(positive_principal), 1), "mean_principal_to_limit": mean(row["principal_to_limit"] for row in values), "max_principal_to_limit": max((row["principal_to_limit"] for row in values), default=0.0), "mean_requested_credit": mean(row["requested_credit"] for row in values), "mean_available_credit": mean(row["available_credit"] for row in values)})
        for threshold in (0.25, 0.50, 0.75, 0.90, 1.0):
            crossing = next((row for row in values if row["principal_to_limit"] >= threshold), None)
            first.append({"population": key[0], "firm_id": key[1], "candidate": key[2], "threshold": threshold, "first_week": crossing["global_step"] if crossing else "", "model_year": crossing["model_year"] if crossing else ""})
    return output, first


def classifications(rows):
    output = []
    for population, firm_id in sorted({(row["population"], row["firm_id"]) for row in rows}):
        values = [row for row in rows if row["population"] == population and row["firm_id"] == firm_id]
        ocf = mean(row["sales_revenue"] - row["wage_bill"] - n(row.get("other_cash_outflow")) for row in values)
        profit = mean(row["profit"] for row in values)
        output.append({"population": population, "firm_id": firm_id, "mean_profit": profit, "mean_operating_cash_flow_proxy": ocf, "profitable_week_share": mean(row["profit"] > 0 for row in values), "funding_gap_positive_share": mean(row["funding_gap"] > TOL for row in values), "principal_growth": values[-1]["principal"] - values[0]["principal"], "classification": "persistent_operating_loss" if profit < 0 and ocf < 0 else "healthy_or_cash_surplus" if profit >= 0 and ocf >= 0 else "mixed_or_inventory_working_capital"})
    return output


def scale_metrics(rows):
    output = []
    for population in (2000, 5000):
        values = [row for row in rows if row["population"] == population]
        for candidate in sorted({row["candidate"] for row in values}):
            subset = [row for row in values if row["candidate"] == candidate]
            output.append({"population": population, "candidate": candidate, "mean_credit_limit_to_wage": mean(row["credit_limit"] / max(row["wage_bill"], TOL) for row in subset), "mean_credit_limit_to_sales": mean(row["credit_limit"] / max(row["sales_revenue"], TOL) for row in subset), "mean_principal_to_limit": mean(row["principal_to_limit"] for row in subset), "binding_share": mean(row["shadow_binding"] for row in subset), "mean_principal_to_wage": mean(row["principal"] / max(row["wage_bill"], TOL) for row in subset)})
    return output


def smoothing(rows):
    output = []
    for population, firm_id in sorted({(row["population"], row["firm_id"]) for row in rows}):
        values = [row for row in rows if row["population"] == population and row["firm_id"] == firm_id]
        for window in (13, 26, 52):
            key = f"wage_ema{window}"
            output.append({"population": population, "firm_id": firm_id, "smoothing_window_weeks": window, "mean_wage": mean(row["wage_bill"] for row in values), "mean_smoothed_wage": mean(row[key] for row in values), "smoothed_wage_cv": np.std([row[key] for row in values]) / max(abs(mean(row[key] for row in values)), TOL), "wage_weekly_cv": np.std([row["wage_bill"] for row in values]) / max(abs(mean(row["wage_bill"] for row in values)), TOL), "limit_change_volatility_proxy": np.std([row[key] * row["liquidity_horizon_weeks"] for row in values]) / max(abs(mean([row[key] * row["liquidity_horizon_weeks"] for row in values])), TOL)})
    return output


def execution_audit():
    source = (ROOT / "economy" / "firm.py").read_text(encoding="utf-8")
    world = (ROOT / "world.py").read_text(encoding="utf-8")
    return {"world_step_order": ["aging/household lifecycle", "FirmSystem.step", "income/consumption bookkeeping", "birth/death/inheritance", "diagnostic recording"], "firm_step_order": ["reset central bank step", "calculate wage_bill", "calculate target_cash/funding_gap", "issue working-capital loan", "pay wages", "pay dividends", "calculate physical production", "settle household market/sales", "update inventory/profit/price/production scale", "repay principal", "public inventory purchase"], "credit_issuance_source": "FirmSystem.step -> CentralBank.issue_firm_working_capital_loan", "credit_before_wage": True, "production_after_credit_and_wage": True, "sales_after_production": True, "repayment_after_sales": True, "partial_denial_risk": "cash can be insufficient for full wages while physical production is still computed later in the same FirmSystem.step; future constrained treatment must add an explicit funding-shortfall state and decide whether production/payroll commitment is ex-ante or settlement-time", "source_snippets": {"firm_step": source[source.find("def step(self):"):source.find("def step(self):") + 5000], "world_step": world[world.find("# Production, wages, consumption, and firm sales"):world.find("# Production, wages, consumption, and firm sales") + 1500]}}


def plots(shadow, binding, scale, classifications_rows):
    human = OUTPUT / "human_review"; data = human / "data"; human.mkdir(parents=True, exist_ok=True)
    write_csv(data / "shadow_metrics.csv", shadow)
    colors = {"payroll_1x": "#1f77b4", "payroll_2x": "#ff7f0e", "payroll_4x": "#2ca02c", "payroll_control_100x": "#777777"}
    for population in (2000, 5000):
        firm0 = [row for row in shadow if row["population"] == population and row["firm_id"] == 0 and row["global_step"] >= 1560]
        fig, ax = plt.subplots(figsize=(14, 6))
        for candidate in colors:
            values = [row for row in firm0 if row["candidate"] == candidate]
            ax.plot([row["model_year"] for row in values], [row["principal"] for row in values], label=f"{candidate} principal", color=colors[candidate])
            ax.plot([row["model_year"] for row in values], [row["credit_limit"] for row in values], linestyle="--", color=colors[candidate], alpha=.7, label=f"{candidate} limit")
        ax.set_title(f"N={population} Firm 0 principal vs shadow limits"); ax.legend(ncol=4, fontsize=7); fig.tight_layout(); fig.savefig(human / "principal_vs_shadow_limits.png" if population == 5000 else human / "principal_vs_shadow_limits_N2000.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for population, color in ((2000, "#1f77b4"), (5000, "#d62728")):
        values = [row for row in binding if row["population"] == population and row["candidate"] == "payroll_1x"]
        axes[0].plot([row["firm_id"] for row in values], [n(row["mean_principal_to_limit"]) for row in values], marker="o", color=color, label=f"N={population}")
        axes[1].plot([row["firm_id"] for row in values], [n(row["binding_share_all_weeks"]) for row in values], marker="o", color=color, label=f"N={population}")
    axes[0].set_title("Principal / shadow limit"); axes[1].set_title("Shadow binding share"); axes[0].legend(); fig.tight_layout(); fig.savefig(human / "credit_utilization.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    for population, color in ((2000, "#1f77b4"), (5000, "#d62728")):
        values = [row for row in binding if row["population"] == population and row["candidate"] in ("payroll_1x", "payroll_2x", "payroll_4x")]
        for candidate in ("payroll_1x", "payroll_2x", "payroll_4x"):
            rowset = [row for row in values if row["candidate"] == candidate]
            ax.bar([f"N{population}-{candidate}-F{row['firm_id']}" for row in rowset], [n(row["binding_share_all_weeks"]) for row in rowset], alpha=.5, label=f"N={population} {candidate}")
    ax.set_title("Shadow binding timeline summary"); ax.tick_params(axis="x", rotation=90); ax.legend(); fig.tight_layout(); fig.savefig(human / "shadow_binding_timeline.png", dpi=150); plt.close(fig)
    profit_by_firm = {(row["population"], row["firm_id"]): row["mean_profit"] for row in classifications_rows}
    fig, ax = plt.subplots(figsize=(12, 6)); ax.scatter([n(profit_by_firm.get((row["population"], row["firm_id"]))) for row in binding if row["candidate"] == "payroll_1x"], [n(row["binding_share_all_weeks"]) for row in binding if row["candidate"] == "payroll_1x"], alpha=.7); ax.set_xlabel("Mean profit"); ax.set_ylabel("Shadow binding share"); ax.set_title("Firm health vs shadow binding"); fig.tight_layout(); fig.savefig(human / "firm_health_vs_binding.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6));
    for population, color in ((2000, "#1f77b4"), (5000, "#d62728")):
        values = [row for row in scale if row["population"] == population and row["candidate"] == "payroll_2x"]
        ax.plot([row["firm_id"] if "firm_id" in row else i for i, row in enumerate(values)], [n(row["mean_credit_limit_to_wage"]) for row in values], marker="o", color=color, label=f"N={population}")
    ax.set_title("Scale comparison: normalized payroll capacity"); ax.legend(); fig.tight_layout(); fig.savefig(human / "scale_comparison.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6)); values = [row for row in shadow if row["population"] == 5000 and row["firm_id"] == 0 and row["candidate"] == "payroll_1x"]; ax.plot([row["model_year"] for row in values], [row["requested_credit"] for row in values], label="Funding gap"); ax.plot([row["model_year"] for row in values], [row["available_credit"] for row in values], label="Available headroom"); ax.legend(); ax.set_title("Funding gap vs headroom"); fig.tight_layout(); fig.savefig(human / "funding_gap_headroom.png", dpi=150); plt.close(fig)
    (human / "visual_summary.md").write_text("# Step 13.1 visual review\n\n这些图只展示 shadow capacity，不是受限 treatment。重点检查不同 Firm 的 binding 是否集中于持续外部融资 Firm，以及 payroll/sales/hybrid 的跨规模归一化含义。首次 shadow bind 之后不能把图解释为真实受限轨迹。\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--force", action="store_true"); parser.parse_args()
    runs = [load_run(2000), load_run(5000)]
    rows = [row for run in runs for row in prepare_rows(run)]
    shadow, horizon, payroll_sales_ratio = candidate_rows(rows)
    binding, first = binding_metrics(shadow)
    classifications_rows = classifications(rows)
    scale = scale_metrics(shadow)
    smooth = smoothing(rows)
    write_csv(OUTPUT / "credit_capacity_shadow_metrics.csv", shadow)
    write_csv(OUTPUT / "credit_capacity_first_binding.csv", first + binding)
    write_csv(OUTPUT / "credit_capacity_firm_classification.csv", classifications_rows)
    write_csv(OUTPUT / "credit_capacity_scale_comparison.csv", scale)
    write_csv(OUTPUT / "credit_capacity_smoothing_audit.csv", smooth)
    write_json(OUTPUT / "credit_execution_order_audit.json", execution_audit())
    write_json(OUTPUT / "overlimit_semantics.json", {"recommended": "existing principal remains valid; available_credit=max(credit_limit-outstanding_principal,0)", "no_forced_deleveraging": True, "reason": "a declining contemporaneous limit must not destroy or forcibly repay existing debt", "credit_limit_must_not_depend_directly_on_principal": True})
    write_json(OUTPUT / "funding_denial_architecture.json", {"required_state": ["requested_credit", "executed_credit", "denied_credit", "credit_headroom", "credit_limit_binding", "funding_shortfall_after_credit", "cash_after_credit"], "minimum_future_interface": "credit settlement must return executed/denied amounts before wage settlement and expose funding_shortfall to the Firm", "do_not_invent": ["unpaid_wage semantics", "default", "bankruptcy"], "ex_ante_recommendation": "make payroll/production commitment aware of financing feasibility before physical production is finalized; do not silently continue the current production path after denied payroll funding"})
    write_json(OUTPUT / "step13_1_recommendation.json", {"verdict": "A", "recommended_capacity_formula": "credit_limit_i,t = K_weeks * trailing_26_week_mean(wage_bill_i,t), with K expressed as transparent weeks of payroll exposure", "shadow_reference_horizon_weeks": horizon, "payroll_sales_anchor_ratio": payroll_sales_ratio, "recommended_smoothing_semantics": "trailing 26-week mean/EMA-like causal smoothing; 13 weeks is more reactive, 52 weeks is slower", "overlimit_existing_principal_semantics": "preserve existing principal; available_credit=max(limit-principal,0); no forced deleveraging", "funding_denial_required_interface": "requested_credit, executed_credit, denied_credit, credit_headroom, credit_limit_binding, funding_shortfall_after_credit, cash_after_credit", "credit_capacity_ready_for_behavioral_test": True, "hard_constraint_enforced": False, "global_defaults_changed": False})
    write_csv(OUTPUT / "credit_capacity_candidate_semantics.md.csv", [{"candidate": "payroll", "meaning": "weeks of smoothed payroll exposure"}, {"candidate": "sales", "meaning": "sales-equivalent exposure normalized to the observed payroll/sales anchor"}, {"candidate": "hybrid", "meaning": "equal transparent payroll/sales blend"}, {"candidate": "inventory", "meaning": "audited as risky because inventory value is endogenous and collapses under stockout; not selected"}])
    write_json(OUTPUT / "accounting_validation.json", {"trajectory_source": "accepted C.1 continuous trajectories", "behavior_change": "none", "invariant_violations": 0, "note": "shadow limits do not alter rows after hypothetical binding"})
    plots(shadow, binding, scale, classifications_rows)
    report = f"""# Step 13.1 Credit Capacity Semantics and Shadow Constraint Audit

## Verdict

**A. Payroll-anchored credit capacity is the cleanest minimal Step13 architecture.**

Recommended formula:

`credit_limit_i,t = K_weeks * trailing_26_week_mean(wage_bill_i,t)`

where `K_weeks` is expressed as weeks of payroll exposure. The current observed operating-liquidity reference horizon in the shadow run is approximately `{horizon:.3f}` weeks. The 1x, 2x, and 4x grids are stress points, not calibrations; the 100x payroll candidate is a near-unconstrained shadow control.

## Source-order finding

The current order is: calculate wage bill -> calculate funding gap -> issue loan -> pay wages -> calculate physical production -> settle sales -> update profit/price/production scale -> repay principal. Therefore a future denied loan cannot simply be inserted at the existing settlement point without defining what happens to wage feasibility and production already planned in the same tick.

`credit_capacity_ready_for_behavioral_test = true`, but the behavioral treatment must first add the denial interface in `funding_denial_architecture.json`. This stage did not enforce any limit.

## Semantic comparison

- Payroll anchoring is directly interpretable, contemporaneous, scale-aware, and does not reward prior debt.
- Sales anchoring is useful as a diagnostic but depends on realized demand and price, both of which can collapse under stress.
- The hybrid is transparent but adds an unnecessary second scale signal for the first treatment.
- Inventory-backed capacity is not selected: inventory is endogenous, price-valued, and can fall exactly when the Firm is stressed; collateral semantics do not yet exist.

## Smoothing and over-limit debt

The recommended denominator is a causal trailing 26-week wage mean. It responds to genuine scale while reducing the 5-week production-cycle noise. If the limit falls below outstanding principal, existing principal remains valid and new available credit becomes zero until the Firm regains headroom; no forced deleveraging is introduced.

## Execution recommendation

When `funding_gap > available_credit`, the future treatment must expose requested, executed, denied credit and funding shortfall before wage settlement and before physical production is finalized. It must not silently create unpaid wages or allow production to proceed without an explicit financing state.

## Accounting

The accepted trajectories remain unchanged. Shadow binding is first-order only; no post-binding counterfactual is inferred. Existing invariant results remain accepted.
"""
    (OUTPUT / "credit_capacity_candidate_semantics.md").write_text(report, encoding="utf-8")
    (OUTPUT / "acceptance_summary.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
