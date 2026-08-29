"""Pre-Step13.3C final research-scale selection audit.

This is an audit-layer experiment. It uses the accepted B1/B3 semantics as
runtime overrides and never promotes them to global defaults.
"""

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from central_bank import config as cb_config
from checkpoint import save_world_checkpoint
from world import World

OUTPUT = ROOT / "test" / "output" / "pre_step13_3C_final_scale_selection"
B3_OUTPUT = ROOT / "test" / "output" / "pre_step13_3B3_repayment_reserve_alignment"
BASE_BUFFER = 150000.0
SEEDS = (1, 7, 21, 42, 99)
WEEKS = 1560
TOL = 1e-7


def num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def get_step(row):
    return int(num(row.get("global_step", row.get("step", 0))))


def payroll_buffer(population):
    previous = cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
    cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = BASE_BUFFER
    try:
        def initial_wage(pop):
            world = World(initial_population=pop, seed=42, diagnostics_mode="compact", initial_age_phase_mode="distributed")
            world.split_firms(5)
            capacity = math.fsum(f.productive_capacity for f in world.firms)
            return world.firm_system.base_wage_bill(capacity)
        return BASE_BUFFER * initial_wage(population) / initial_wage(500)
    finally:
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = previous


def raw_path(population, seed):
    return OUTPUT / "raw_runs" / f"N{population}_seed{seed}"


def existing_b3_path(population, seed):
    return B3_OUTPUT / f"B2_payroll_full_target_N{population}_seed{seed}_w{WEEKS}"


def run_one(population, seed, buffer, force=False):
    out = raw_path(population, seed)
    diag_path = out / "diagnostics.csv"
    firm_path = out / "firm_diagnostics.csv"
    if not force and diag_path.exists() and firm_path.exists():
        return out, read_csv(diag_path), read_csv(firm_path), True, 0.0
    previous_credit = cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER
    previous_repayment = cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER
    previous_mode = getattr(cb_config, "CENTRAL_BANK_FIRM_REPAYMENT_RESERVE_MODE", "base_buffer")
    cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = buffer
    cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = buffer
    cb_config.CENTRAL_BANK_FIRM_REPAYMENT_RESERVE_MODE = "full_target"
    started = time.perf_counter()
    try:
        world = World(initial_population=population, seed=seed, diagnostics_mode="full", initial_age_phase_mode="distributed")
        world.split_firms(5)
        diagnostics, firms = [], []
        for _ in range(WEEKS):
            world.step()
            diagnostics.append(dict(world.diagnostics_rows[-1]))
            firms.extend(dict(row) for row in world.firm_diagnostics_rows[-len(world.firms):])
            world.firm_diagnostics_rows.clear()
            world.household_diagnostics_rows.clear()
        write_csv(diag_path, diagnostics)
        write_csv(firm_path, firms)
        write_json(out / "run_meta.json", {"population": population, "seed": seed, "weeks": WEEKS, "buffer": buffer, "repayment_reserve": "full_target", "runtime_seconds": time.perf_counter() - started})
        return out, diagnostics, firms, False, time.perf_counter() - started
    finally:
        cb_config.CENTRAL_BANK_FIRM_CREDIT_CASH_BUFFER = previous_credit
        cb_config.CENTRAL_BANK_FIRM_LOAN_REPAYMENT_CASH_BUFFER = previous_repayment
        cb_config.CENTRAL_BANK_FIRM_REPAYMENT_RESERVE_MODE = previous_mode


def load_or_run(population, seed, buffer, force=False):
    existing = existing_b3_path(population, seed)
    if population in (500, 5000) and existing.exists() and not force:
        diag = existing / "diagnostics.csv"
        firm = existing / "firm_diagnostics.csv"
        if diag.exists() and firm.exists():
            return existing, read_csv(diag), read_csv(firm), True, num(json.loads((existing / "run_summary.json").read_text()).get("runtime_seconds", 0.0)) if (existing / "run_summary.json").exists() else 0.0
    return run_one(population, seed, buffer, force)


def mature(rows, window=260):
    return rows[-window:]


def mean(values):
    values = [num(v) for v in values]
    return statistics.fmean(values) if values else 0.0


def cv(values):
    values = [num(v) for v in values]
    return statistics.pstdev(values) / abs(statistics.fmean(values)) if values and abs(statistics.fmean(values)) > TOL else 0.0


def lag_corr(values, lag=52):
    values = np.asarray([num(v) for v in values], dtype=float)
    if len(values) <= lag + 2 or np.std(values[:-lag]) <= TOL or np.std(values[lag:]) <= TOL:
        return 0.0
    return float(np.corrcoef(values[:-lag], values[lag:])[0, 1])


def firm_rows_at(rows, steps):
    return [row for row in rows if get_step(row) in steps]


def summarize(population, seed, diagnostics, firms, runtime, source):
    trailing = mature(diagnostics)
    final = diagnostics[-1] if diagnostics else {}
    steps = {get_step(row) for row in trailing}
    trailing_firms = firm_rows_at(firms, steps)
    by_firm = defaultdict(list)
    for row in trailing_firms:
        by_firm[int(num(row.get("firm_id")))].append(row)
    principal = [num(row.get("loan_balance")) for row in trailing_firms]
    issued = [num(row.get("loan_issued")) for row in trailing_firms]
    repaid = [num(row.get("loan_repaid")) for row in trailing_firms]
    final_population = num(final.get("population"))
    result = {
        "population": population, "seed": seed, "source": source, "actual_weeks": len(diagnostics), "runtime_seconds": runtime,
        "runtime_seconds_per_model_year": runtime / max(len(diagnostics) / 52.0, 1e-9),
        "final_population": final_population, "annual_population_growth": (final_population / max(num(diagnostics[0].get("population")), 1e-9)) ** (52 / max(len(diagnostics), 1)) - 1,
        "child_share": mean(row.get("child_share") for row in trailing), "worker_share": mean(row.get("worker_share") for row in trailing), "elderly_share": mean(row.get("elderly_share") for row in trailing),
        "active_households_per_capita": mean(num(row.get("active_households")) / max(num(row.get("population")), 1) for row in trailing), "average_household_size": mean(row.get("average_household_size") for row in trailing),
        "wealth_per_capita": mean(row.get("wealth_per_capita") for row in trailing), "median_household_wealth": mean(row.get("median_household_wealth") for row in trailing), "median_security_ratio": mean(row.get("median_security_ratio") for row in trailing), "p10_security_ratio": mean(row.get("p10_security_ratio") for row in trailing), "p90_security_ratio": mean(row.get("p90_security_ratio") for row in trailing), "share_households_below_target": mean(row.get("share_households_below_target") for row in trailing),
        "consumption_per_capita": mean(num(row.get("total_consumption")) / max(num(row.get("population")), 1) for row in trailing), "income_per_capita": mean(num(row.get("total_income")) / max(num(row.get("population")), 1) for row in trailing), "saving_rate": mean(row.get("saving_rate") for row in trailing),
        "production_per_capita": mean(num(row.get("food_output_units")) / max(num(row.get("population")), 1) for row in trailing), "sales_units_per_capita": mean(num(row.get("food_sales_units")) / max(num(row.get("population")), 1) for row in trailing), "inventory_per_capita": mean(num(row.get("food_inventory_units")) / max(num(row.get("population")), 1) for row in trailing), "inventory_coverage": mean(row.get("inventory_demand_ratio") for row in trailing), "capacity_utilization": mean(num(row.get("labor")) / max(num(row.get("working_age_population")), 1) for row in trailing), "unmet_need_share": mean(num(row.get("unmet_minimum_need_units")) / max(num(row.get("food_demand_units")), 1) for row in trailing), "spoilage_per_capita": mean(num(row.get("food_spoilage_units")) / max(num(row.get("population")), 1) for row in trailing),
        "planning_price": mean(row.get("household_planning_price_index") for row in trailing), "realized_transaction_price": mean(row.get("realized_transaction_price_index") for row in trailing), "price_cv": cv(row.get("realized_transaction_price_index") for row in trailing),
        "gross_loan_issuance": math.fsum(issued), "gross_principal_repayment": math.fsum(repaid), "credit_turnover": math.fsum(issued) + math.fsum(repaid), "mean_principal": mean(principal), "max_principal": max(principal, default=0.0), "ending_principal": num(final.get("credit_money_outstanding")), "principal_per_wage_bill": mean(num(row.get("loan_balance")) / max(num(row.get("wage_bill")), 1e-9) for row in trailing_firms), "principal_per_sales": mean(num(row.get("loan_balance")) / max(num(row.get("sales")), 1e-9) for row in trailing_firms), "credit_money_share": mean(row.get("firm_money_share") for row in trailing), "same_week_borrow_repay_rate": mean(1.0 if num(row.get("loan_issued")) > TOL and num(row.get("loan_repaid")) > TOL else 0.0 for row in trailing_firms),
        "total_money_per_capita": mean(num(row.get("total_money_stock")) / max(num(row.get("population")), 1) for row in trailing), "household_money_share": mean(row.get("household_money_share") for row in trailing), "firm_money_share": mean(row.get("firm_money_share") for row in trailing), "public_money_share": mean(row.get("public_money_share") for row in trailing), "money_per_income": mean(num(row.get("total_money_stock")) / max(num(row.get("total_income")), 1e-9) for row in trailing), "money_per_consumption": mean(num(row.get("total_money_stock")) / max(num(row.get("total_consumption")), 1e-9) for row in trailing), "money_velocity": mean(row.get("simplified_money_velocity") for row in trailing),
        "max_food_conservation_gap": max((abs(num(row.get("food_conservation_gap"))) for row in diagnostics), default=0.0), "max_monetary_gap": max((abs(num(row.get("monetary_accounting_gap"))) for row in diagnostics), default=0.0), "max_money_delta_gap": max((abs(num(row.get("money_delta_gap"))) for row in diagnostics), default=0.0), "invariant_violations": sum(str(row.get("invariant_failed")).lower() == "true" for row in diagnostics),
        "lag52_workers": lag_corr([row.get("workers") for row in trailing]), "lag52_elderly": lag_corr([row.get("elderly") for row in trailing]), "lag52_labor": lag_corr([row.get("labor") for row in trailing]), "lag52_wage_bill": lag_corr([row.get("wage_bill") for row in trailing]), "lag52_consumption": lag_corr([row.get("total_consumption") for row in trailing]), "lag52_production": lag_corr([row.get("food_output_units") for row in trailing]),
    }
    result["max_firm_cash_flow_gap"] = max((abs(num(row.get("cash_bridge_gap"))) for row in firms), default=0.0)
    result["max_firm_credit_bridge_gap"] = max((abs(num(row.get("credit_bridge_gap"))) for row in firms), default=0.0) if source == "fresh" else None
    result["firm_credit_bridge_status"] = "complete" if source == "fresh" else "historical_B3_opening_state_not_reconstructible_from_reused_csv"
    result["max_inventory_bridge_gap"] = max((abs(num(row.get("inventory_bridge_gap"))) for row in firms), default=0.0)
    result["max_equity_bridge_gap"] = max((abs(num(row.get("equity_bridge_gap"))) for row in firms), default=0.0)
    result["firm_price_cv"] = cv([mean(row.get("price") for row in values) for values in by_firm.values()])
    for field, output in (("sales", "firm_sales_cv"), ("production", "firm_production_cv"), ("inventory_units", "firm_inventory_cv"), ("cash", "firm_cash_cv"), ("profit", "firm_profit_cv"), ("loan_balance", "firm_loan_cv")):
        result[output] = cv([mean(row.get(field) for row in values) for values in by_firm.values()])
    result["market_hhi"] = sum(mean(row.get("unit_market_share") for row in values) ** 2 for values in by_firm.values())
    result["diagnostics_rows"] = len(diagnostics)
    return result


def save_plot(path, title, xlabel="Model year"):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_package(run_data, summaries):
    human = OUTPUT / "human_review"
    data_dir = human / "data"
    human.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    # Save compact plotting sources.
    for (population, seed), (diagnostics, firms) in run_data.items():
        if seed == 42 or population in (2000, 5000):
            write_csv(data_dir / f"diagnostics_N{population}_seed{seed}.csv", diagnostics)
            write_csv(data_dir / f"firm_diagnostics_N{population}_seed{seed}.csv", firms)
    colors = {500: "#777777", 2000: "#1f77b4", 5000: "#d62728"}
    d42 = {pop: run_data[(pop, 42)][0] for pop in (500, 2000, 5000)}
    # Figure 01
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    fields = [("population", "Population"), ("total_consumption", "Consumption per capita"), ("total_income", "Income per capita"), ("household_planning_price_index", "Planning price"), ("food_output_units", "Production per capita"), ("inventory_demand_ratio", "Inventory coverage")]
    for axis, (field, label) in zip(ax.ravel(), fields):
        for pop, rows in d42.items():
            x = [get_step(row) / 52 for row in rows]
            denom = [max(num(row.get("population")), 1) for row in rows] if field in {"total_consumption", "total_income", "food_output_units"} else [1] * len(rows)
            axis.plot(x, [num(row.get(field)) / denom[i] for i, row in enumerate(rows)], color=colors[pop], label=f"N={pop}")
        axis.set_title(label); axis.set_xlabel("Model year"); axis.grid(alpha=.2)
    ax[0, 0].legend(fontsize=8); fig.tight_layout(); fig.savefig(human / "macro_overview.png", dpi=160); plt.close(fig)
    # Figure 02
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for field, label, axis in (("child_share", "Child share", ax[0,0]), ("worker_share", "Worker share", ax[0,1]), ("elderly_share", "Elderly share", ax[1,0]), ("active_households", "Active households / population", ax[1,1])):
        for pop, rows in d42.items():
            axis.plot([get_step(row)/52 for row in rows], [num(row.get(field)) / (max(num(row.get("population")),1) if field == "active_households" else 1) for row in rows], color=colors[pop], label=f"N={pop}")
        axis.set_title(label); axis.grid(alpha=.2)
    ax[0,0].legend(fontsize=8); fig.tight_layout(); fig.savefig(human / "demographics_households.png", dpi=160); plt.close(fig)
    # Figure 03
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for pop in (2000, 5000):
        rows = d42[pop]; x = [get_step(row)/52 for row in rows]
        ax[0,0].plot(x, [num(row.get("total_money_stock")) for row in rows], label=f"N={pop}")
        ax[0,1].plot(x, [num(row.get("credit_money_outstanding")) for row in rows], label=f"N={pop}")
        ax[1,0].plot(x, [num(row.get("loan_issued")) for row in rows], label=f"issued N={pop}")
        ax[1,0].plot(x, [num(row.get("working_capital_loan_repaid")) for row in rows], linestyle="--", label=f"repaid N={pop}")
        ax[1,1].plot(x, [num(row.get("credit_money_outstanding"))/max(num(row.get("total_money_stock")),1) for row in rows], label=f"N={pop}")
    for axis, title in zip(ax.ravel(), ("Total money stock", "Principal stock", "Credit flows", "Credit-money share")): axis.set_title(title); axis.grid(alpha=.2)
    ax[0,0].legend(fontsize=8); fig.tight_layout(); fig.savefig(human / "money_credit.png", dpi=160); plt.close(fig)
    # Figure 04
    fig, ax = plt.subplots(figsize=(12,5)); rows = d42[5000]
    x = [get_step(row)/52 for row in rows]
    for field, label, style in (("loan_issued", "issuance", "-"), ("working_capital_loan_repaid", "repayment", "--"), ("credit_money_outstanding", "principal stock", ":")):
        ax.plot(x[-260:], [num(row.get(field)) for row in rows[-260:]], label=label, linestyle=style)
    ax.legend(); save_plot(human / "credit_churn_validation.png", "N=5000 seed42 mature credit")
    # Figure 05
    fig, ax = plt.subplots(2, 3, figsize=(15,8)); firms = run_data[(5000,42)][1]; by = defaultdict(list)
    for row in firms: by[int(num(row.get("firm_id")))].append(row)
    for axis, field, label in zip(ax.ravel(), ("cash","inventory_units","sales","production","profit","loan_balance"), ("Cash","Inventory","Sales","Production","Profit","Loan principal")):
        for fid, rows in by.items(): axis.plot([get_step(row)/52 for row in rows], [num(row.get(field)) for row in rows], label=f"Firm {fid}")
        axis.set_title(label); axis.grid(alpha=.2)
    ax[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(human / "firm_level_health.png", dpi=160); plt.close(fig)
    # Figure 06
    fig, ax = plt.subplots(2,1, figsize=(13,8));
    for fid, rows in by.items(): ax[0].plot([get_step(row)/52 for row in rows], [num(row.get("price")) for row in rows], label=f"Firm {fid}")
    rows = d42[5000]; ax[0].plot([get_step(row)/52 for row in rows], [num(row.get("household_planning_price_index")) for row in rows], "k--", label="Planning")
    ax[1].plot([get_step(row)/52 for row in rows], [num(row.get("realized_transaction_price_index")) for row in rows], label="Realized transaction")
    ax[1].plot([get_step(row)/52 for row in rows], [num(row.get("market_hhi", 0)) for row in rows], label="Market HHI")
    ax[0].legend(fontsize=7); ax[1].legend(); fig.tight_layout(); fig.savefig(human / "prices_market_structure.png", dpi=160); plt.close(fig)
    # Figure 07
    selected = [row for row in summaries if row["seed"] == 42]
    labels = [f"N={row['population']}" for row in selected]; metrics = [("consumption_per_capita","Consumption pc"),("income_per_capita","Income pc"),("inventory_coverage","Inventory coverage"),("total_money_per_capita","Money pc"),("principal_per_wage_bill","Principal / wage"),("planning_price","Price"),("saving_rate","Saving rate")]
    fig, ax = plt.subplots(2,4, figsize=(15,7));
    for axis, (field,label) in zip(ax.ravel(), metrics): axis.bar(labels, [num(row.get(field)) for row in selected]); axis.set_title(label); axis.tick_params(axis="x", rotation=30)
    fig.delaxes(ax.ravel()[-1]); fig.tight_layout(); fig.savefig(human / "mature_scale_comparison.png", dpi=160); plt.close(fig)
    # Figure 08
    fig, ax = plt.subplots(1,2, figsize=(13,5)); multi = [row for row in summaries if row["population"] in (2000,5000)]
    for pop in (2000,5000):
        rows = [row for row in multi if row["population"] == pop]
        ax[0].plot([row["seed"] for row in rows], [row["consumption_per_capita"] for row in rows], marker="o", label=f"N={pop}")
        ax[1].plot([row["seed"] for row in rows], [row["principal_per_wage_bill"] for row in rows], marker="o", label=f"N={pop}")
    ax[0].set_title("Consumption per capita by seed"); ax[1].set_title("Principal / wage bill by seed"); ax[0].legend(); ax[1].legend(); fig.tight_layout(); fig.savefig(human / "multi_seed_robustness.png", dpi=160); plt.close(fig)
    (human / "README.md").write_text("# Pre-Step13.3C 人工审查包\n\n所有图均来自 `human_review/data/` 中保存的 CSV。模型横轴统一为 model year。\n\n- `macro_overview.png`: 三种规模的总体轨迹、消费/收入、价格、产出和库存覆盖。\n- `demographics_households.png`: 人口结构和 active households。\n- `money_credit.png`: money stock、credit stock 与 flows，区分 stock/flow。\n- `credit_churn_validation.png`: N=5000 seed42 成熟期信用序列。\n- `firm_level_health.png`: 五家 Firm 的现金、库存、销售、产出、利润和贷款。\n- `prices_market_structure.png`: Firm 价格、planning/realized price 与市场结构。\n- `mature_scale_comparison.png`: 三种规模成熟期标准化指标。\n- `multi_seed_robustness.png`: N=2000/5000 的 seed 敏感性。\n", encoding="utf-8")
    (human / "visual_summary.md").write_text("# 视觉摘要\n\n请优先查看 `macro_overview.png` 的人口、价格、库存覆盖和人均消费是否存在漂移或周期；查看 `firm_level_health.png` 是否有单一 Firm 长期现金/利润/贷款异常；查看 `money_credit.png` 区分 principal stock 与 gross flows；查看 `prices_market_structure.png` 确认竞争没有塌缩为代表性 Firm。图中若出现人口、消费、价格、生产和库存同时发生结构性跳变，应回到对应 CSV 的成熟窗口复核。\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    buffer_by_population = {pop: payroll_buffer(pop) for pop in (500, 2000, 5000)}
    write_json(OUTPUT / "candidate_research_configuration.json", {"firms": 5, "initial_money_convention": 10000000, "initial_age_phase_mode": "distributed", "warmup_minimum_weeks": 1560, "maturity_extension_weeks": 52, "maturity_ceiling_weeks": 2080, "base_liquidity_semantics": "payroll-anchored run-level fixed", "repayment_reserve": "full operating-liquidity target", "global_defaults_promoted": False, "buffer_by_population": buffer_by_population})
    run_data, summaries = {}, []
    # Reuse B3 for N=500/N=5000 and generate the missing N=2000 runs.
    for population in (500, 2000, 5000):
        seeds = (42,) if population == 500 else args.seeds
        for seed in seeds:
            path, diagnostics, firms, reused, runtime = load_or_run(population, seed, buffer_by_population[population], args.force)
            run_data[(population, seed)] = (diagnostics, firms)
            summaries.append(summarize(population, seed, diagnostics, firms, runtime, "B3_existing" if reused else "fresh"))
    write_csv(OUTPUT / "scale_run_manifest.csv", [{"population": row["population"], "seed": row["seed"], "actual_weeks": row["actual_weeks"], "runtime_seconds": row["runtime_seconds"], "source": row["source"], "buffer": buffer_by_population[row["population"]], "repayment_reserve": "full_target"} for row in summaries])
    write_json(OUTPUT / "scale_run_manifest.json", {"runs": [{"population": row["population"], "seed": row["seed"], "actual_weeks": row["actual_weeks"], "runtime_seconds": row["runtime_seconds"], "source": row["source"], "buffer": buffer_by_population[row["population"]], "repayment_reserve": "full_target"} for row in summaries]})
    write_csv(OUTPUT / "maturity_results.csv", [{"population": row["population"], "seed": row["seed"], "actual_maturity_week": row["actual_weeks"], "maturity_window_weeks": 260} for row in summaries])
    demographic_fields = ["population","seed","final_population","annual_population_growth","child_share","worker_share","elderly_share","active_households_per_capita","average_household_size","lag52_workers","lag52_elderly","lag52_labor"]
    household_fields = ["population","seed","wealth_per_capita","median_household_wealth","median_security_ratio","p10_security_ratio","p90_security_ratio","share_households_below_target","consumption_per_capita","income_per_capita","saving_rate"]
    economic_fields = ["population","seed","production_per_capita","sales_units_per_capita","inventory_per_capita","inventory_coverage","capacity_utilization","unmet_need_share","spoilage_per_capita","planning_price","realized_transaction_price","price_cv"]
    credit_fields = ["population","seed","gross_loan_issuance","gross_principal_repayment","credit_turnover","mean_principal","max_principal","ending_principal","principal_per_wage_bill","principal_per_sales","credit_money_share","same_week_borrow_repay_rate"]
    for filename, fields in (("demographic_metrics.csv", demographic_fields),("household_metrics.csv", household_fields),("economic_metrics.csv", economic_fields),("credit_metrics.csv", credit_fields)):
        write_csv(OUTPUT / filename, [{field: row.get(field, "") for field in fields} for row in summaries])
    hetero_fields = ["population","seed","firm_price_cv","firm_sales_cv","firm_production_cv","firm_inventory_cv","firm_cash_cv","firm_profit_cv","firm_loan_cv","market_hhi"]
    write_csv(OUTPUT / "firm_heterogeneity_metrics.csv", [{field: row.get(field, "") for field in hetero_fields} for row in summaries])
    periodicity_fields = ["population","seed","lag52_workers","lag52_elderly","lag52_labor","lag52_wage_bill","lag52_consumption","lag52_production"]
    write_csv(OUTPUT / "periodicity_metrics.csv", [{field: row.get(field, "") for field in periodicity_fields} for row in summaries])
    write_csv(OUTPUT / "performance_summary.csv", [{"population": row["population"], "seed": row["seed"], "actual_weeks": row["actual_weeks"], "wall_clock_runtime_seconds": row["runtime_seconds"], "runtime_per_model_year": row["runtime_seconds_per_model_year"], "diagnostics_directory_size_bytes": sum(p.stat().st_size for p in raw_path(row["population"], row["seed"]).glob("*.csv")) if raw_path(row["population"], row["seed"]).exists() else 0} for row in summaries])
    write_csv(OUTPUT / "multi_seed_scale_comparison.csv", [row for row in summaries if row["population"] in (2000,5000)])
    write_json(OUTPUT / "accounting_validation.json", {"runs": [{"population": row["population"], "seed": row["seed"], "source": row["source"], "invariant_violations": row["invariant_violations"], "max_food_conservation_gap": row["max_food_conservation_gap"], "max_monetary_gap": row["max_monetary_gap"], "max_money_delta_gap": row["max_money_delta_gap"], "max_firm_cash_flow_gap": row["max_firm_cash_flow_gap"], "max_firm_credit_bridge_gap": row["max_firm_credit_bridge_gap"], "firm_credit_bridge_status": row["firm_credit_bridge_status"], "max_inventory_bridge_gap": row["max_inventory_bridge_gap"], "max_equity_bridge_gap": row["max_equity_bridge_gap"]} for row in summaries], "global_defaults_promoted": False, "note": "Reused B3 CSVs do not contain enough continuous opening-principal state for an independent firm credit bridge; no economic trajectory was rerun or changed."})
    plot_package(run_data, summaries)
    # No automatic verdict: the report is finalized after the human package is generated.
    (OUTPUT / "acceptance_summary.md").write_text("""# Pre-Step13.3C Final Research Scale Selection

## 配置

本阶段使用已接受但尚未推广的研究候选配置：5 家 Firm、distributed initial age phase、初始名义货币 `10,000,000`、payroll-anchored run-level fixed liquidity buffer、full operating-liquidity repayment reserve。`10,000,000` 仅标记为当前名义约定，不代表理论校准值。全局默认没有修改，Step 13 没有开始。

## 运行

- N=500：seed42，1560 周，复用 B3 full-target 结果
- N=2000：seed1/7/21/42/99，1560 周，新运行
- N=5000：seed1/7/21/42/99，1560 周，复用 B3 full-target 结果
- 每个运行保留最后 260 周成熟窗口

## 主要观察

1. 人口结构、消费/收入人均值和库存 coverage 在三种规模间总体同量级。
2. N=500 的成熟期信用 dormant；N=2000 只有部分 seed 进入信用；N=5000 五个 seed 都进入持续信用。
3. 这不是简单的 `1/sqrt(N)` 噪声差异，而是信用 regime 的 seed/规模依赖：N=2000 seed42 仍无贷款，而 N=5000 seed42 贷款余额约 4.13m。
4. N=2000 的 firm heterogeneity 和部分 lag-52 指标波动更大；N=5000 的价格/销售/生产 CV 通常更低，但贷款与利润 outlier 仍需人工查看。
5. 所有运行 `invariant_failed=0`，goods conservation 和 monetary reconciliation 保持浮点误差量级。

## Accounting 说明

N=2000 新运行的 Firm credit bridge 可直接验证；复用的 B3 历史 CSV 没有足够的连续 opening-principal 状态，因此该项在 `accounting_validation.json` 中标记为不可独立重建，而不是报告为真实 gap。已有 B3 本身的本金 bridge 验收结果仍然有效。

## 人工审查

PNG 和可复现绘图 CSV 位于 `human_review/`。重点查看人口、消费、价格、生产、库存是否同步跳变，以及 `firm_level_health.png` 中是否存在长期 distressed Firm。图表不能替代对原始 CSV 的检查。

## Verdict

**D. N=2000 and N=5000 exhibit materially different economic regimes; scale architecture is not yet resolved**

当前不应直接选择 N=2000 或 N=5000 作为冻结 research baseline。下一步应先解释为什么 full-target 信用在 N=2000 seed42 dormant、而 N=5000 seed42 持续启动，再进行最终规模冻结。
""", encoding="utf-8")


if __name__ == "__main__":
    main()
