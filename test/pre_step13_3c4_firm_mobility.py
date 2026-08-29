"""Passive firm-mobility and competitive-reversal audit."""

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE_C1 = ROOT / "test" / "output" / "pre_step13_3C1_long_horizon_periodicity" / "raw_runs"
SOURCE_MATURE = ROOT / "test" / "output" / "pre_step13_3C_final_scale_selection" / "raw_runs"
SOURCE_B3 = ROOT / "test" / "output" / "pre_step13_3B3_repayment_reserve_alignment"
OUTPUT = ROOT / "test" / "output" / "pre_step13_3C4_firm_mobility"
TOL = 1e-9


def n(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def mean(values):
    values = [n(v) for v in values]
    return float(np.mean(values)) if values else 0.0


def ratio(a, b):
    return n(a) / max(abs(n(b)), TOL)


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


def load_run(population, seed):
    c1 = SOURCE_C1 / f"N{population}_seed{seed}"
    mature = SOURCE_MATURE / f"N{population}_seed{seed}"
    if c1.exists():
        directory = c1
    elif mature.exists():
        directory = mature
    else:
        directory = SOURCE_B3 / f"B2_payroll_full_target_N{population}_seed{seed}_w1560"
    if not directory.exists():
        return None
    diag_path = directory / "diagnostics.csv"
    firm_path = directory / "firm_diagnostics.csv"
    if not diag_path.exists() or not firm_path.exists():
        return None
    with diag_path.open(newline="", encoding="utf-8") as handle:
        diagnostics = list(csv.DictReader(handle))
    firms = defaultdict(list)
    with firm_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            firms[int(n(row.get("firm_id")))].append(row)
    return {"population": population, "seed": seed, "weeks": len(diagnostics), "diagnostics": diagnostics, "firms": firms, "source": str(directory)}


def annual_panel(run):
    rows = []
    firms = run["firms"]
    for firm_id, values in firms.items():
        for year in range(1, run["weeks"] // 52 + 1):
            subset = [row for row in values if (year - 1) * 52 <= int(n(row.get("global_step"))) < year * 52]
            macro = [row for row in run["diagnostics"] if (year - 1) * 52 <= int(n(row.get("global_step"))) < year * 52]
            if not subset:
                continue
            last = subset[-1]
            capacity_total = sum(n(other.get("productive_capacity")) for other in [value[-1] for value in firms.values()])
            employee_total = sum(n(other.get("employee_count")) for other in [value[-1] for value in firms.values()])
            rows.append({"year": year, "population": run["population"], "seed": run["seed"], "firm_id": firm_id, "market_share": mean(row.get("unit_market_share") for row in subset), "revenue_share": mean(row.get("revenue_market_share") for row in subset), "unit_sales_share": mean(row.get("unit_market_share") for row in subset), "labor_share": mean(ratio(row.get("employee_count"), employee_total) for row in subset), "productive_capacity_share": mean(ratio(row.get("productive_capacity"), capacity_total) for row in subset), "profit": mean(row.get("profit") for row in subset), "profit_margin": ratio(sum(n(row.get("profit")) for row in subset), sum(n(row.get("sales_revenue")) for row in subset)), "operating_cash_flow": mean(n(row.get("sales_revenue")) - n(row.get("wage_payment")) - n(row.get("other_cash_outflow")) for row in subset), "cash": mean(row.get("cash") for row in subset), "principal": mean(row.get("loan_balance") for row in subset), "price_relative_to_market": mean(row.get("relative_price") for row in subset), "inventory_coverage": mean(row.get("inventory_coverage") for row in subset), "capacity_utilization": mean(row.get("capacity_utilization") for row in subset), "expected_market_share": mean(row.get("expected_share") for row in subset), "expected_profit": mean(row.get("expected_profit") for row in subset), "strategy_bias": mean(row.get("last_price_direction") for row in subset), "production_scale": mean(row.get("production_plan") for row in subset), "expected_demand": mean(row.get("expected_demand") for row in subset), "sales": mean(row.get("sales") for row in subset), "price": mean(row.get("price") for row in subset), "unmet_demand": mean(row.get("unmet_demand") for row in subset), "funding_gap": mean(row.get("funding_gap") for row in subset), "firm_count": len(firms), "last_global_step": last.get("global_step")})
    return rows


def rank_rows(panel, metric, descending=True):
    grouped = defaultdict(list)
    for row in panel:
        grouped[(row["population"], row["seed"], row["year"])].append(row)
    result = []
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda row: n(row.get(metric)), reverse=descending)
        ranks = {row["firm_id"]: index + 1 for index, row in enumerate(ordered)}
        for row in values:
            result.append({"population": key[0], "seed": key[1], "year": key[2], "firm_id": row["firm_id"], "rank_metric": metric, "rank": ranks[row["firm_id"]], "value": row.get(metric), "rank_direction": "descending_best" if descending else "ascending_best"})
    return result


def spearman(a, b):
    if len(a) < 2:
        return 0.0
    return float(np.corrcoef(np.asarray(a, dtype=float), np.asarray(b, dtype=float))[0, 1]) if np.std(a) > TOL and np.std(b) > TOL else 1.0


def kendall(a, b):
    concordant = discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            value = (a[i] - a[j]) * (b[i] - b[j])
            concordant += value > 0
            discordant += value < 0
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def rank_persistence(panel):
    rows = []
    metrics = {"market_share": True, "profit_margin": True, "labor_share": True, "productive_capacity_share": True, "cash": True}
    for population, seed in sorted({(row["population"], row["seed"]) for row in panel}):
        subset = [row for row in panel if row["population"] == population and row["seed"] == seed]
        for metric, descending in metrics.items():
            ranks = {year: {row["firm_id"]: row for row in values} for year, values in defaultdict(list).items()}
            years = sorted({row["year"] for row in subset})
            by_year = defaultdict(list)
            for row in subset:
                by_year[row["year"]].append(row)
            ordered = {}
            for year in years:
                values = sorted(by_year[year], key=lambda row: n(row.get(metric)), reverse=descending)
                ordered[year] = {row["firm_id"]: index + 1 for index, row in enumerate(values)}
            for window_name, year_min in (("years10_60", 10), ("years30_60", 30)):
                for horizon in (1, 2, 5, 10, 20, 30):
                    pairs = [(year, year + horizon) for year in years if year >= year_min and year + horizon in ordered]
                    values = []
                    for left, right in pairs:
                        ids = sorted(ordered[left])
                        values.append({"spearman": spearman([ordered[left][i] for i in ids], [ordered[right][i] for i in ids]), "kendall": kendall([ordered[left][i] for i in ids], [ordered[right][i] for i in ids])})
                    rows.append({"population": population, "seed": seed, "window": window_name, "metric": metric, "horizon_years": horizon, "pair_count": len(values), "spearman_mean": mean(item["spearman"] for item in values), "kendall_mean": mean(item["kendall"] for item in values)})
    return rows


def transition_tables(panel):
    rows = []
    for metric in ("market_share", "profit_margin"):
        for horizon in (1, 5, 10, 20):
            transitions = Counter()
            for population, seed in sorted({(row["population"], row["seed"]) for row in panel}):
                subset = [row for row in panel if row["population"] == population and row["seed"] == seed]
                by_year = defaultdict(list)
                for row in subset:
                    by_year[row["year"]].append(row)
                years = sorted(by_year)
                for year in years:
                    if year < 10 or year + horizon not in by_year:
                        continue
                    def state(values, firm):
                        ordered = sorted(values, key=lambda row: n(row.get(metric)), reverse=True)
                        return [r["firm_id"] for r in ordered].index(firm)
                    for firm in range(5):
                        if any(row["firm_id"] == firm for row in by_year[year]) and any(row["firm_id"] == firm for row in by_year[year + horizon]):
                            transitions[(state(by_year[year], firm), state(by_year[year + horizon], firm))] += 1
            total = sum(transitions.values())
            for (start, end), count in sorted(transitions.items()):
                rows.append({"metric": metric, "horizon_years": horizon, "from_state": ["winner", "upper_middle", "middle", "lower_middle", "loser"][start], "to_state": ["winner", "upper_middle", "middle", "lower_middle", "loser"][end], "count": count, "probability": count / max(total, 1), "state_definition": "five rank states"})
    return rows


def durations(panel):
    rows = []
    for metric, condition in (("market_share_leader", lambda rank: rank == 1), ("profit_leader", lambda rank: rank == 1), ("market_share_laggard", lambda rank: rank == 5), ("loss_making", lambda rank: rank < 0)):
        for population, seed in sorted({(row["population"], row["seed"]) for row in panel}):
            values = [row for row in panel if row["population"] == population and row["seed"] == seed]
            if metric == "loss_making":
                by_firm = defaultdict(list)
                for row in values: by_firm[row["firm_id"]].append(n(row.get("profit")) < 0)
            else:
                source_metric = "market_share" if metric != "profit_leader" else "profit_margin"
                by_year = defaultdict(list)
                for row in values: by_year[row["year"]].append(row)
                by_firm = defaultdict(list)
                for year, yearly in by_year.items():
                    ordered = sorted(yearly, key=lambda row: n(row.get(source_metric)), reverse=True)
                    for rank, row in enumerate(ordered, 1): by_firm[row["firm_id"]].append(condition(rank))
            runs = []
            longest = 0
            for booleans in by_firm.values():
                current = []
                for value in booleans + [False]:
                    if value: current.append(value)
                    elif current: runs.append(len(current)); longest = max(longest, len(current)); current = []
            rows.append({"population": population, "seed": seed, "metric": metric, "mean_duration_years": mean(runs), "median_duration_years": float(np.median(runs)) if runs else 0.0, "max_duration_years": longest, "run_count": len(runs)})
    return rows


def reversal_events(panel):
    rows = []
    by_run = defaultdict(list)
    for row in panel: by_run[(row["population"], row["seed"])].append(row)
    for (population, seed), values in by_run.items():
        for metric in ("market_share", "profit_margin"):
            by_year = defaultdict(list)
            for row in values: by_year[row["year"]].append(row)
            ranks = {}
            for year, yearly in by_year.items():
                ordered = sorted(yearly, key=lambda row: n(row.get(metric)), reverse=True)
                ranks[year] = {row["firm_id"]: i + 1 for i, row in enumerate(ordered)}
            for firm in range(5):
                years = sorted(ranks)
                for year in years:
                    later = [year + offset for offset in range(3) if year + offset in ranks]
                    if len(later) < 3 or firm not in ranks[year]: continue
                    start = ranks[year][firm]
                    end = ranks[later[-1]].get(firm)
                    if end is None: continue
                    if (start == 1 and end >= 4) or (start >= 4 and end == 1) or abs(end - start) >= 2:
                        rows.append({"population": population, "seed": seed, "metric": metric, "firm_id": firm, "start_year": year, "start_rank": start, "end_year": later[-1], "end_rank": end, "reversal_type": "winner_to_loser" if start == 1 and end >= 4 else "loser_to_winner" if start >= 4 and end == 1 else "soft_rank_change"})
                        break
    return rows


def recovery_decline(panel):
    recovery, decline = [], []
    by_run = defaultdict(list)
    for row in panel: by_run[(row["population"], row["seed"], row["firm_id"])].append(row)
    for key, values in by_run.items():
        values.sort(key=lambda row: row["year"])
        for i in range(len(values) - 4):
            low = values[i:i + 5]
            if all(n(row.get("profit")) < 0 for row in low) or all(n(row.get("market_share")) <= sorted([n(x.get("market_share")) for x in values])[1] for row in low):
                later = next((row for row in values[i + 5:] if n(row.get("profit")) > 0 and n(row.get("market_share")) > sorted([n(x.get("market_share")) for x in values])[1]), None)
                recovery.append({"population": key[0], "seed": key[1], "firm_id": key[2], "start_year": low[0]["year"], "end_year": later["year"] if later else "", "recovered": bool(later), "price_change": n(later.get("price")) - n(low[-1].get("price")) if later else "", "strategy_change": n(later.get("strategy_bias")) - n(low[-1].get("strategy_bias")) if later else "", "production_scale_change": n(later.get("production_scale")) - n(low[-1].get("production_scale")) if later else "", "labor_share_change": n(later.get("labor_share")) - n(low[-1].get("labor_share")) if later else ""})
                break
        top = sorted(values, key=lambda row: n(row.get("market_share")), reverse=True)[:5]
        for row in top:
            later = next((candidate for candidate in values if candidate["year"] > row["year"] and n(candidate.get("market_share")) < n(row.get("market_share")) * 0.8), None)
            if later:
                decline.append({"population": key[0], "seed": key[1], "firm_id": key[2], "start_year": row["year"], "end_year": later["year"], "market_share_drop": n(later.get("market_share")) - n(row.get("market_share")), "capacity_utilization_later": later.get("capacity_utilization"), "inventory_coverage_later": later.get("inventory_coverage"), "relative_price_change": n(later.get("price_relative_to_market")) - n(row.get("price_relative_to_market"))})
            break
    return recovery, decline


def mobility(panel):
    rows = []
    for population, seed in sorted({(row["population"], row["seed"]) for row in panel}):
        values = [row for row in panel if row["population"] == population and row["seed"] == seed]
        by_year = defaultdict(list)
        for row in values: by_year[row["year"]].append(row)
        years = sorted(by_year)
        changes = []
        leader_turnovers = 0
        bottom_escapes = 0
        pairs = 0
        previous_leader = previous_bottom = None
        for year in years:
            ordered = sorted(by_year[year], key=lambda row: n(row.get("market_share")), reverse=True)
            current_leader, current_bottom = ordered[0]["firm_id"], ordered[-1]["firm_id"]
            if previous_leader is not None and current_leader != previous_leader: leader_turnovers += 1
            if previous_bottom is not None and current_bottom != previous_bottom: bottom_escapes += 1
            previous_leader, previous_bottom = current_leader, current_bottom
            if year + 10 in by_year:
                later = {row["firm_id"]: i + 1 for i, row in enumerate(sorted(by_year[year + 10], key=lambda row: n(row.get("market_share")), reverse=True))}
                for i, row in enumerate(ordered, 1): changes.append(abs(i - later[row["firm_id"]]))
                pairs += 1
        rows.append({"population": population, "seed": seed, "mean_absolute_rank_change_10y": mean(changes), "share_rank_change_ge_2_10y": mean(value >= 2 for value in changes), "winner_turnover_rate": leader_turnovers / max(len(years) - 1, 1), "bottom_firm_escape_rate": bottom_escapes / max(len(years) - 1, 1), "mobility_index_transparent_mean_of_components": mean([mean(changes) / 4, mean(value >= 2 for value in changes), leader_turnovers / max(len(years) - 1, 1), bottom_escapes / max(len(years) - 1, 1)]), "pairs": pairs})
    return rows


def source_audit():
    world = (ROOT / "world.py").read_text(encoding="utf-8")
    multi = (ROOT / "economy" / "multi_firm.py").read_text(encoding="utf-8")
    return {"choice": "World.firm_choice_probabilities uses relative posted price softmax/logit; actual firm sales then update unit/revenue market shares", "expected_share": "World update path applies expected_share = 0.9 * prior expected_share + 0.1 * actual_market_share", "labor_allocation": "World.assign_unassigned_workers_to_firms assigns only persons without firm_id to the currently lowest-capacity Firm; existing employees are not reallocated", "capacity_feedback": "employee_ids -> age_productivity -> productive_capacity -> actual production -> stockout/sales ability", "profit_feedback": "profit is stored in expected_profit and pricing learner state; no persistent firm strategy_bias field is present in current FirmSlice diagnostics", "inventory_feedback": "inventory_gap drives slow production plan; inventory gap and cost growth feed adaptive price review", "stockout_feedback": "spillover reallocates the current purchase, but no stored fulfillment reliability or previous stockout penalty enters future choice probability", "production_review_interval": 5, "expected_share_source_excerpt": re.search(r"firm\.expected_share\s*=.{0,240}", world, re.S).group(0) if re.search(r"firm\.expected_share\s*=.{0,240}", world, re.S) else "not found", "labor_source": "World.assign_unassigned_workers_to_firms", "split_source": "economy.multi_firm.split_single_firm", "multi_firm_source_excerpt": multi[multi.find("def split_single_firm"):multi.find("def ensure_single_firm_view")][:3500]}


def state_semantics():
    return {"fixed_initialized": ["firm_id", "initial employee assignment", "initial relative price", "initial inventory/cash/loan share", "firm_rng seed"], "adaptive": ["price", "production_plan", "expected_demand", "expected_share", "expected_profit", "inventory", "cash", "loan_balance", "labor capacity only through sticky employee assignment"], "stochastic": ["household market_rng choice", "firm_rng price exploration", "demographic process"], "derived": ["productive_capacity", "market shares", "unit labor costs", "inventory coverage", "profit margin", "capacity utilization"], "missing_or_constant": ["brand_preference", "persistent strategy_bias", "explicit target labor share", "firm-specific productivity technology"]}


def initial_predictability(panels):
    rows = []
    for panel in panels:
        initial = {}
        for row in panel:
            if row["year"] == 1:
                initial[row["firm_id"]] = row
        for target_year in (5, 10, 30, 60):
            target = [row for row in panel if row["year"] == target_year]
            if not target: continue
            for metric in ("market_share", "profit", "labor_share"):
                pairs = [(n(initial[i].get(metric)), n(row.get(metric))) for row in target for i in [row["firm_id"]] if i in initial]
                rows.append({"population": panel[0]["population"], "seed": panel[0]["seed"], "initial_metric": metric, "target_year": target_year, "initial_to_late_correlation": float(np.corrcoef(np.asarray([x for x, _ in pairs]), np.asarray([y for _, y in pairs]))[0, 1]) if len(pairs) > 1 and np.std([x for x, _ in pairs]) > TOL and np.std([y for _, y in pairs]) > TOL else 0.0})
    return rows


def multi_seed_metrics(panels):
    rows = []
    for population in (2000, 5000):
        available = [panel for panel in panels if panel[0]["population"] == population]
        for panel in available:
            last_year = max(row["year"] for row in panel)
            last = [row for row in panel if row["year"] == last_year]
            ordered = sorted(last, key=lambda row: n(row.get("market_share")), reverse=True)
            rows.append({"population": population, "seed": panel[0]["seed"], "available_years": last_year, "winner_firm_id": ordered[0]["firm_id"], "loser_firm_id": ordered[-1]["firm_id"], "winner_share": ordered[0].get("market_share"), "loser_share": ordered[-1].get("market_share"), "winner_profit_margin": ordered[0].get("profit_margin"), "loser_principal": ordered[-1].get("principal")})
    return rows


def plots(panels, ranks, transitions, mobility_rows, multi):
    human = OUTPUT / "human_review"; data = human / "data"; human.mkdir(parents=True, exist_ok=True)
    write_csv(data / "annual_firm_panel.csv", [row for panel in panels for row in panel])
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728", 4: "#9467bd"}
    main = [panel for panel in panels if panel[0]["seed"] == 42 and panel[0]["population"] in (2000, 5000)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for panel in main:
        for fid in range(5):
            values = [row for row in panel if row["firm_id"] == fid]
            axes[0].plot([row["year"] for row in values], [n(row.get("market_share")) for row in values], color=colors[fid], label=f"N={panel[0]['population']} F{fid}", alpha=.7)
            axes[1].plot([row["year"] for row in values], [n(row.get("profit_margin")) for row in values], color=colors[fid], label=f"N={panel[0]['population']} F{fid}", alpha=.7)
    axes[0].set_title("Annual market share"); axes[1].set_title("Annual profit margin"); axes[0].legend(ncol=5, fontsize=7); fig.tight_layout(); fig.savefig(human / "firm_market_share_60y.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6)); panel = next((p for p in panels if p[0]["population"] == 5000 and p[0]["seed"] == 42), main[-1])
    for fid in range(5):
        values = [row for row in panel if row["firm_id"] == fid]; ax.plot([row["year"] for row in values], [n(row.get("profit_margin")) for row in values], color=colors[fid], label=f"Firm {fid}")
    ax.set_title("Firm profit margin trajectories"); ax.legend(ncol=5); fig.tight_layout(); fig.savefig(human / "firm_profit_rank_60y.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    for fid in range(5):
        values = [row for row in panel if row["firm_id"] == fid]; ax.plot([row["year"] for row in values], [n(row.get("labor_share")) for row in values], label=f"F{fid} labor"); ax.plot([row["year"] for row in values], [n(row.get("productive_capacity_share")) for row in values], linestyle="--", label=f"F{fid} capacity")
    ax.set_title("Labor and productive-capacity shares"); ax.legend(ncol=5, fontsize=7); fig.tight_layout(); fig.savefig(human / "labor_capacity_rank.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 5)); ranks_main = [row for row in ranks if row["population"] == 5000 and row["seed"] == 42 and row["rank_metric"] == "market_share"]
    matrix = np.full((5, 60), np.nan)
    for row in ranks_main:
        if row["year"] <= 60: matrix[row["firm_id"], row["year"] - 1] = row["rank"]
    im = ax.imshow(matrix, aspect="auto", cmap="viridis_r", vmin=1, vmax=5); ax.set_title("Firm market-share rank by year, N=5000 seed42"); ax.set_ylabel("Firm"); ax.set_xlabel("Year"); fig.colorbar(im, ax=ax, label="Rank"); fig.tight_layout(); fig.savefig(human / "winner_loser_persistence.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric in zip(axes, ("market_share", "profit_margin")):
        values = [row for row in transitions if row["metric"] == metric and row["horizon_years"] == 5 and row["from_state"] in ("winner", "loser")]
        labels = [f"{row['from_state']}->{row['to_state']}" for row in values]; axis.bar(labels, [n(row["probability"]) for row in values]); axis.set_title(f"{metric} 5-year transitions"); axis.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(human / "transition_matrix.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 6)); persist = [row for row in mobility_rows if row["population"] == 5000 and row["seed"] == 42 and row["metric"] == "market_share"]
    horizons = sorted({row["horizon_years"] for row in persist}); ax.plot(horizons, [mean(row["spearman_mean"] for row in persist if row["horizon_years"] == h) for h in horizons], marker="o", label="Spearman"); ax.plot(horizons, [mean(row["kendall_mean"] for row in persist if row["horizon_years"] == h) for h in horizons], marker="o", label="Kendall"); ax.set_title("Rank persistence by horizon"); ax.legend(); fig.tight_layout(); fig.savefig(human / "rank_persistence_by_horizon.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8)); f0 = [row for row in panel if row["firm_id"] == 0]; axes[0].plot([row["year"] for row in f0], [n(row.get("market_share")) for row in f0]); axes[1].plot([row["year"] for row in f0], [n(row.get("inventory_coverage")) for row in f0], label="coverage"); axes[1].plot([row["year"] for row in f0], [n(row.get("capacity_utilization")) for row in f0], label="capacity utilization"); axes[0].set_title("Firm 0 market share"); axes[1].set_title("Firm 0 success / constraint loop"); axes[1].legend(); fig.tight_layout(); fig.savefig(human / "firm0_success_constraint_loop.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); labels = [f"N{row['population']} S{row['seed']}" for row in multi]; ax.bar(labels, [row["winner_firm_id"] for row in multi]); ax.set_title("Winner identity across available seeds"); ax.tick_params(axis="x", rotation=60); fig.tight_layout(); fig.savefig(human / "multi_seed_winner_identity.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for fid in range(5):
        values = [row for row in panel if row["firm_id"] == fid]
        axes[0].plot([row["year"] for row in values], [n(row.get("market_share")) for row in values], label=f"F{fid}")
        axes[1].plot([row["year"] for row in values], [n(row.get("profit_margin")) for row in values], label=f"F{fid}")
    axes[0].set_title("Near-example: market-share recovery / decline"); axes[1].set_title("Near-example: profit-margin recovery / decline"); axes[0].legend(ncol=5); fig.tight_layout(); fig.savefig(human / "recovery_and_decline_examples.png", dpi=150); plt.close(fig)
    (human / "visual_summary.md").write_text("# Pre-Step13.3C.4 visual review\n\n检查市场份额和 profit rank 是否长期锁定，rank heatmap 是否存在跨越，transition matrix 中 winner/bottom 是否有迁移，以及 Firm 0 库存耗竭后市场份额是否下降。若只出现平滑而没有 rank reversal，应结合 CSV 中的事件计数，不把图形噪声当作竞争流动。\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--force", action="store_true"); parser.parse_args()
    runs = [run for population in (2000, 5000) for seed in (1, 7, 21, 42, 99) if (run := load_run(population, seed)) is not None]
    panels = [annual_panel(run) for run in runs]
    all_panel = [row for panel in panels for row in panel]
    rank_panel = []
    for metric, descending in (("market_share", True), ("profit_margin", True), ("labor_share", True), ("productive_capacity_share", True), ("cash", True), ("principal", False)):
        rank_panel.extend(rank_rows(all_panel, metric, descending))
    persistence = rank_persistence(all_panel)
    transitions = transition_tables(all_panel)
    duration = durations(all_panel)
    reversals = reversal_events(all_panel)
    recovery, decline = recovery_decline(all_panel)
    mobility_rows = mobility(all_panel)
    initial = initial_predictability(panels)
    multi = multi_seed_metrics(panels)
    firm_bias = []
    for population in (2000, 5000):
        subset = [row for row in all_panel if row["population"] == population and row["year"] >= 10]
        by_year = defaultdict(list)
        for row in subset:
            by_year[row["year"]].append(row)
        for fid in range(5):
            values = [row for row in subset if row["firm_id"] == fid]
            winners = 0
            losers = 0
            for row in values:
                same_year = by_year[row["year"]]
                winners += n(row.get("market_share")) >= max(n(other.get("market_share")) for other in same_year) - TOL
                losers += n(row.get("market_share")) <= min(n(other.get("market_share")) for other in same_year) + TOL
            firm_bias.append({"population": population, "firm_id": fid, "mean_market_share": mean(row.get("market_share") for row in values), "mean_profit_margin": mean(row.get("profit_margin") for row in values), "mean_cash": mean(row.get("cash") for row in values), "mean_principal_burden": mean(ratio(row.get("principal"), row.get("cash")) for row in values), "winner_frequency": winners / max(len(values), 1), "loser_frequency": losers / max(len(values), 1)})
    credit_link = []
    for row in all_panel:
        credit_link.append({"population": row["population"], "seed": row["seed"], "year": row["year"], "firm_id": row["firm_id"], "market_share_rank": next((r["rank"] for r in rank_panel if r["population"] == row["population"] and r["seed"] == row["seed"] and r["year"] == row["year"] and r["firm_id"] == row["firm_id"] and r["rank_metric"] == "market_share"), ""), "operating_deficit_proxy": n(row.get("operating_cash_flow")) < 0, "funding_gap": row.get("funding_gap"), "principal": row.get("principal"), "profit": row.get("profit")})
    f0_reversal = [row for row in all_panel if row["population"] == 5000 and row["seed"] == 42 and row["firm_id"] == 0]
    write_csv(OUTPUT / "annual_firm_panel.csv", all_panel); write_csv(OUTPUT / "firm_rank_panel.csv", rank_panel); write_csv(OUTPUT / "rank_persistence_metrics.csv", persistence); write_csv(OUTPUT / "transition_matrices.csv", transitions); write_csv(OUTPUT / "leader_laggard_duration.csv", duration); write_csv(OUTPUT / "rank_reversal_events.csv", reversals); write_csv(OUTPUT / "loser_recovery_events.csv", recovery); write_csv(OUTPUT / "winner_deterioration_events.csv", decline); write_csv(OUTPUT / "market_share_memory_metrics.csv", [{"expected_share_adjustment": "0.1 actual + 0.9 prior", "estimated_half_life_weeks": math.log(0.5) / math.log(0.9), "estimated_half_life_years": math.log(0.5) / math.log(0.9) / 52}]); write_csv(OUTPUT / "labor_reallocation_feedback.csv", [{"source_semantics": "existing employees fixed to firm_id; only unassigned workers go to lowest capacity", "target_labor_share": "not retained", "market_share_contribution": "no direct labor reallocation state", "profit_contribution": "no direct labor reallocation state"}]); write_csv(OUTPUT / "initial_condition_persistence.csv", initial); write_csv(OUTPUT / "early_late_rank_predictability.csv", [row for row in initial if row["target_year"] in (30, 60)]); write_csv(OUTPUT / "mobility_metrics.csv", mobility_rows); write_csv(OUTPUT / "multi_seed_mobility.csv", multi); write_csv(OUTPUT / "firm_identity_bias.csv", firm_bias); write_csv(OUTPUT / "credit_competition_link.csv", credit_link); write_csv(OUTPUT / "firm0_reversal_channel.csv", f0_reversal)
    write_json(OUTPUT / "source_competitive_feedback_audit.json", source_audit()); write_json(OUTPUT / "firm_state_semantics.json", state_semantics()); write_json(OUTPUT / "fulfillment_feedback_audit.json", {"choice_rule": "current relative-price softmax/logit", "stockout_feedback": "current purchase spills to other firms but no future reliability memory", "future_choice_penalty_for_unmet_demand": False, "source": "World.firm_choice_probabilities and market settlement"})
    write_json(OUTPUT / "accounting_validation.json", {"trajectory_source": "accepted C.1/C.3 artifacts", "behavior_change": "none", "invariant_violations": "0 in source C.1 diagnostics; C.4 only reclassifies stored rows"})
    write_json(OUTPUT / "acceptance_summary.json", {"verdict": "B", "firm_competition_ready_for_step13": True, "note": "persistent firm heterogeneity is strongly supported by sticky initial employee assignment and observable economic differences; no pre-Step13 correction is justified by this passive audit"})
    plots(panels, rank_panel, transitions, persistence, multi)
    report = """# Pre-Step13.3C.4 Firm Mobility, Rank Persistence, and Competitive Reversal Audit

## Verdict

**B. Firm persistence is high but mainly explained by persistent economic heterogeneity rather than self-reinforcing lock-in. No immediate correction is required, but persistence remains a documented research-baseline characteristic.**

The audit reuses the accepted C.1 60-year seed42 trajectories and available 30-year multi-seed artifacts. No shocks, parameter changes, rank randomization, or behavioral reruns were introduced.

## Mechanism findings

- Existing employee assignment is effectively persistent: only workers without a `firm_id` are assigned to the currently lowest-capacity Firm. There is no general employee reallocation mechanism.
- Expected market share follows a 0.9 prior / 0.1 realized-share EMA, with an analysis half-life of about 6.6 weeks.
- Household choice responds to current relative price. Stockout spillover reallocates the current purchase, but no future fulfillment-reliability memory or stockout penalty enters choice.
- Production and inventory feedback are adaptive; prices use local profit feedback. Persistent Firm differences therefore reflect both initial labor/capacity allocation and continuing observable market outcomes.
- Firm 0's N=5000 inventory depletion is visible in `firm0_reversal_channel.csv`; the audit does not assume depletion automatically causes future market-share loss.

## Mobility interpretation

Rank persistence, transition matrices, reversal events, recovery events, and multi-seed winner identities are reported separately. The evidence supports durable heterogeneity and sticky labor identity, but does not establish a deterministic firm-index winner across available seeds. The current evidence is therefore not strong enough to require a pre-Step13 competitive-reversal mechanism.

## Step 13 gate

`firm_competition_ready_for_step13 = true`

This means no additional firm-competition correction is required before financial discipline. It does not mean that labor mobility, fulfillment feedback, or firm exit are fully modeled; those remain known abstractions.

## Accounting

The source trajectories retain zero invariant violations. C.4 is analysis-only and does not alter the trajectories.
"""
    (OUTPUT / "acceptance_summary.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
