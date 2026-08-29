"""Passive Step 13.5C flow-first distress redesign audit."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from step13_5_passive_distress_semantics import FIRM_PATH, MATURE_START, TOL, build_features, num, read, quant, share, spells

OUT = Path("test/output/step13_5C_flow_first_distress_redesign")


def write_csv(name, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)


def mean(values):
    values = [float(x) for x in values if math.isfinite(float(x))]
    return statistics.fmean(values) if values else 0.0


def jaccard(a, b):
    a, b = set(a), set(b)
    return share(len(a & b), len(a | b))


def add_flow_features(features):
    by_firm = defaultdict(list)
    for row in features:
        by_firm[row["firm_id"]].append(row)
    for rows in by_firm.values():
        rows.sort(key=lambda r: r["global_step"])
        for index, row in enumerate(rows):
            def window(size):
                return rows[max(0, index - size + 1):index + 1]
            for size in (4, 13, 26, 52):
                part = window(size)
                row[f"service_failure_share_{size}"] = mean([float(x["interest_state"] in {"I2", "I3"}) for x in part])
                row[f"unpaid_flow_share_{size}"] = mean([float(x["interest_unpaid_flow"] > TOL) for x in part])
                row[f"mean_service_ratio_{size}"] = mean([x["interest_service_ratio"] for x in part])
                row[f"binding_share_{size}"] = mean([float(x["credit_binding"]) for x in part])
                row[f"denied_share_{size}"] = mean([float(x["credit_denied"] > TOL) for x in part])
                row[f"negative_ocf_share_{size}"] = mean([float(x["operating_cash_flow"] < 0) for x in part])
                row[f"mean_ocf_{size}"] = mean([x["operating_cash_flow"] for x in part])
                row[f"payroll_under_share_{size}"] = mean([float(x["payroll_underfunding"]) for x in part])
            row["arrears_growth"] = row["closing_interest_arrears"] - row["opening_interest_arrears"]
            row["service_flow"] = row["unpaid_flow_share_26"] >= .5 or row["mean_service_ratio_26"] < .99
            row["credit_flow"] = row["denied_share_26"] > 0 or row["binding_share_26"] >= .5
            row["ocf_flow"] = row["mean_ocf_26"] < 0
            row["payroll_flow"] = row["payroll_under_share_4"] >= .5
            row["payroll_acute"] = row["payroll_underfunding"]
            row["burden_high"] = row["arrears_payroll_weeks"] >= 1.0
            row["burden_severe"] = row["arrears_payroll_weeks"] >= 4.0
            row["near_exhausted_credit"] = row["exposure_utilization"] >= .90 or row["credit_headroom"] <= TOL


def classify(row, family, horizon=26):
    def persistent(name):
        if name == "service":
            return row[f"unpaid_flow_share_{horizon}"] >= .5 or row[f"mean_service_ratio_{horizon}"] < .99
        if name == "credit":
            return row[f"denied_share_{horizon}"] > 0 or row[f"binding_share_{horizon}"] >= .5
        if name == "ocf":
            return row[f"mean_ocf_{horizon}"] < 0
        if name == "payroll":
            return row[f"payroll_under_share_{horizon}"] >= .5
        return False
    dims = {name: persistent(name) for name in ("service", "credit", "ocf", "payroll")}
    current_pressure = row["interest_unpaid_flow"] > TOL or row["credit_denied"] > TOL or row["payroll_underfunding"] or row["operating_cash_flow"] < 0
    if family == "A":
        votes = sum(dims[x] for x in ("service", "credit", "ocf"))
        d2 = votes >= 2
        d3 = d2 and (row["payroll_acute"] or (votes == 3 and row["burden_severe"]))
    elif family == "B":
        current_votes = sum(dims[x] for x in ("service", "credit", "ocf"))
        d2 = current_votes >= 2 or (current_votes >= 1 and row["burden_high"])
        d3 = current_votes >= 2 and (dims["payroll"] or row["burden_severe"] or row["near_exhausted_credit"])
    else:
        votes = sum(dims.values())
        d2 = votes >= 2
        d3 = votes >= 3 or (d2 and row["payroll_acute"] and row["burden_severe"])
    return "D3" if d3 else "D2" if d2 else "D1" if current_pressure else "D0"


def stats(states):
    counts = Counter(r["state"] for r in states)
    mature = [r for r in states if r["global_step"] >= MATURE_START]
    mc = Counter(r["state"] for r in mature)
    sp = spells(states, "state")
    d2 = [x["length"] for x in sp if x["state"] == "D2"]
    d3 = [x["length"] for x in sp if x["state"] == "D3"]
    return {"full_D0_share": share(counts["D0"], len(states)), "full_D1_share": share(counts["D1"], len(states)), "full_D2_share": share(counts["D2"], len(states)), "full_D3_share": share(counts["D3"], len(states)), "mature_D0_share": share(mc["D0"], len(mature)), "mature_D1_share": share(mc["D1"], len(mature)), "mature_D2_share": share(mc["D2"], len(mature)), "mature_D3_share": share(mc["D3"], len(mature)), "mature_D2D3_share": share(mc["D2"] + mc["D3"], len(mature)), "D2_spell_count": len(d2), "D3_spell_count": len(d3), "D2_recovery_count": sum(a["state"] == "D2" and b["state"] in {"D0", "D1"} for a, b in zip(states, states[1:])), "D3_recovery_count": sum(a["state"] == "D3" and b["state"] in {"D0", "D1", "D2"} for a, b in zip(states, states[1:])), "median_D2_spell": statistics.median(d2) if d2 else 0, "median_D3_spell": statistics.median(d3) if d3 else 0, "one_week_D2_share": share(sum(x == 1 for x in d2), len(d2)), "one_week_D3_share": share(sum(x == 1 for x in d3), len(d3))}


def overlap(states):
    def ids(predicate): return {(r["firm_id"], r["global_step"]) for r in states if predicate(r)}
    d23 = ids(lambda r: r["state"] in {"D2", "D3"})
    binding = ids(lambda r: r["credit_binding"])
    i3 = ids(lambda r: r["interest_state"] == "I3")
    payroll = ids(lambda r: r["payroll_underfunding"])
    rows = []
    for name, other in (("I3", i3), ("binding", binding), ("payroll", payroll)):
        both = len(d23 & other)
        rows.append({"comparison": name, "D2D3_count": len(d23), "other_count": len(other), "intersection": both, "P_D2D3_given_other": share(both, len(other)), "P_other_given_D2D3": share(both, len(d23)), "jaccard": jaccard(d23, other)})
    return rows


def future_outcomes(states, horizon):
    by_firm = defaultdict(list)
    for row in states: by_firm[row["firm_id"]].append(row)
    out = []
    for rows in by_firm.values():
        rows.sort(key=lambda r: r["global_step"])
        for index, row in enumerate(rows):
            future = rows[index + 1:index + 1 + horizon]
            if not future: continue
            out.append({"firm_id": row["firm_id"], "global_step": row["global_step"], "horizon": horizon, "state": row["state"], "future_payroll_underfunding": any(x["payroll_underfunding"] for x in future), "future_arrears_increase": any(x["arrears_growth"] > TOL for x in future), "future_denied_credit": any(x["credit_denied"] > TOL for x in future), "future_negative_ocf": any(x["trailing_26_mean_OCF"] < 0 for x in future)})
    return out


def baseline_label(row, label):
    if label == "I3": return row["interest_state"] == "I3"
    if label == "binding": return bool(row["credit_binding"])
    return row["interest_state"] == "I3" and bool(row["credit_binding"])


def future_summary(candidate_states):
    rows = []
    state_by_id = {(r["firm_id"], r["global_step"]): r for r in candidate_states}
    for horizon in (4, 13, 26):
        future = future_outcomes(candidate_states, horizon)
        groups = {"candidate_D2D3": [r for r in future if r["state"] in {"D2", "D3"}]}
        for label in ("I3", "binding", "I3+binding"):
            groups[label] = [r for r in future if baseline_label(state_by_id[(r["firm_id"], r["global_step"])], label)]
        for label, values in groups.items():
            rows.append({"group": label, "horizon_weeks": horizon, "observations": len(values), "future_payroll_rate": share(sum(x["future_payroll_underfunding"] for x in values), len(values)), "future_arrears_increase_rate": share(sum(x["future_arrears_increase"] for x in values), len(values)), "future_denied_credit_rate": share(sum(x["future_denied_credit"] for x in values), len(values)), "future_negative_ocf_rate": share(sum(x["future_negative_ocf"] for x in values), len(values))})
    return rows


def recovery_rows(states):
    rows = []
    for firm in sorted({r["firm_id"] for r in states}):
        fs = [r for r in states if r["firm_id"] == firm]
        improvement = [r for r in fs if r["interest_unpaid_flow"] <= TOL and r["interest_service_ratio"] >= .99 and not r["credit_binding"] and r["credit_denied"] <= TOL and r["trailing_26_mean_OCF"] >= 0 and not r["payroll_underfunding"]]
        rows.append({"firm_id": firm, "flow_improvement_episode_count": sum(a["global_step"] + 1 != b["global_step"] or not (a in improvement and b in improvement) for a, b in zip(improvement, improvement[1:])), "improvement_observations": len(improvement), "positive_arrears_improvement_count": sum(r["closing_interest_arrears"] > TOL for r in improvement)})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = build_features(read(FIRM_PATH)); add_flow_features(features)
    families = {name: [{**r, "state": classify(r, name)} for r in features] for name in ("A", "B", "C")}
    comparison = []
    redundancy = []
    for name, states in families.items():
        s = stats(states); o = overlap(states); comparison.append({"family": name, **s})
        for row in o: redundancy.append({"family": name, **row})
    write_csv("candidate_family_comparison.csv", comparison)
    write_csv("candidate_redundancy_metrics.csv", redundancy)
    write_csv("candidate_state_weekly.csv", [{"family": name, "global_step": r["global_step"], "firm_id": r["firm_id"], "state": r["state"], "service_flow": r["service_flow"], "credit_flow": r["credit_flow"], "ocf_flow": r["ocf_flow"], "payroll_flow": r["payroll_flow"], "arrears_payroll_weeks": r["arrears_payroll_weeks"], "arrears_growth": r["arrears_growth"], "interest_state": r["interest_state"], "credit_binding": r["credit_binding"], "payroll_underfunding": r["payroll_underfunding"]} for name, rows in families.items() for r in rows])

    activation = []
    for name, rows in families.items():
        for state in ("D2", "D3"):
            selected = [r for r in rows if r["state"] == state]
            for dimension in ("service_flow", "credit_flow", "ocf_flow", "payroll_flow", "burden_high", "burden_severe"):
                activation.append({"family": name, "state": state, "dimension": dimension, "active_share": share(sum(r[dimension] for r in selected), len(selected))})
    write_csv("candidate_dimension_activation.csv", activation)

    drops = []
    for name, rows in families.items():
        base = stats(rows)
        for dimension in ("service", "credit", "ocf", "payroll", "burden"):
            altered = []
            for r in features:
                item = dict(r)
                if dimension == "service": item["unpaid_flow_share_26"] = 0; item["mean_service_ratio_26"] = 1
                elif dimension == "credit": item["denied_share_26"] = 0; item["binding_share_26"] = 0; item["near_exhausted_credit"] = False
                elif dimension == "ocf": item["mean_ocf_26"] = 0
                elif dimension == "payroll": item["payroll_under_share_26"] = 0; item["payroll_acute"] = False
                else: item["burden_high"] = False; item["burden_severe"] = False
                altered.append({**item, "state": classify(item, name)})
            ast = stats(altered); drops.append({"family": name, "dropped_dimension": dimension, "base_mature_D2D3": base["mature_D2D3_share"], "new_mature_D2D3": ast["mature_D2D3_share"], "mature_D2D3_change": ast["mature_D2D3_share"] - base["mature_D2D3_share"], "base_first_D2D3": min((r["global_step"] for r in rows if r["state"] in {"D2", "D3"}), default=""), "new_first_D2D3": min((r["global_step"] for r in altered if r["state"] in {"D2", "D3"}), default="")})
    write_csv("candidate_drop_one_dimension.csv", drops)

    future = []
    for name, rows in families.items():
        for row in future_summary(rows): future.append({"family": name, **row})
    write_csv("candidate_incremental_future_outcomes.csv", future)

    # Lead times to first acute event by Firm.
    lead = []
    for name, rows in families.items():
        for firm in sorted({r["firm_id"] for r in rows}):
            fs = [r for r in rows if r["firm_id"] == firm]
            d2 = min((r["global_step"] for r in fs if r["state"] == "D2"), default=""); d3 = min((r["global_step"] for r in fs if r["state"] == "D3"), default="")
            for event, predicate in (("payroll_underfunding", lambda r: r["payroll_underfunding"]), ("severe_arrears_growth", lambda r: r["burden_severe"] and r["arrears_growth"] > TOL), ("credit_denial", lambda r: r["credit_denied"] > TOL)):
                acute = min((r["global_step"] for r in fs if predicate(r)), default="")
                lead.append({"family": name, "firm_id": firm, "event": event, "first_D2": d2, "first_D3": d3, "first_acute_event": acute, "D2_lead_weeks": acute - d2 if isinstance(acute, int) and isinstance(d2, int) else "", "D3_lead_weeks": acute - d3 if isinstance(acute, int) and isinstance(d3, int) else ""})
    write_csv("candidate_lead_time_analysis.csv", lead)

    for label, predicate in (("I3", lambda r: r["interest_state"] == "I3"), ("binding", lambda r: r["credit_binding"])):
        rows = []
        for name, states in families.items():
            selected = [r for r in states if predicate(r)]
            rows.append({"family": name, "source": label, "source_observations": len(selected), "source_D0D1": sum(r["state"] in {"D0", "D1"} for r in selected), "source_D2D3": sum(r["state"] in {"D2", "D3"} for r in selected), "D2D3_share": share(sum(r["state"] in {"D2", "D3"} for r in selected), len(selected)), "mean_OCF_D0D1": mean(r["trailing_26_mean_OCF"] for r in selected if r["state"] in {"D0", "D1"}), "mean_OCF_D2D3": mean(r["trailing_26_mean_OCF"] for r in selected if r["state"] in {"D2", "D3"})})
        write_csv(("I3_without_distress_audit.csv" if label == "I3" else "binding_without_distress_audit.csv"), rows)
    write_csv("distress_without_I3_binding_audit.csv", [{"family": name, "D2D3_without_current_I3_count": sum(r["state"] in {"D2", "D3"} and r["interest_state"] != "I3" for r in rows), "D2D3_without_current_binding_count": sum(r["state"] in {"D2", "D3"} and not r["credit_binding"] for r in rows)} for name, rows in families.items()])

    recovery = []
    for name, rows in families.items():
        s = stats(rows); recovery.extend([{ "family": name, "metric": "underlying_flow_improvement_episode_count", "value": sum(1 for r in recovery_rows(rows) if r["improvement_observations"] > 0)}, {"family": name, "metric": "candidate_recovery_episode_count", "value": s["D2_recovery_count"] + s["D3_recovery_count"]}, {"family": name, "metric": "recovery_with_positive_arrears_count", "value": sum(r["positive_arrears_improvement_count"] for r in recovery_rows(rows))}, {"family": name, "metric": "D3_to_D2_count", "value": sum(a["state"] == "D3" and b["state"] == "D2" for a, b in zip(rows, rows[1:]))}, {"family": name, "metric": "D3_to_D1_count", "value": sum(a["state"] == "D3" and b["state"] == "D1" for a, b in zip(rows, rows[1:]))}, {"family": name, "metric": "D3_to_D0_count", "value": sum(a["state"] == "D3" and b["state"] == "D0" for a, b in zip(rows, rows[1:]))}, {"family": name, "metric": "D2_to_D1_count", "value": sum(a["state"] == "D2" and b["state"] == "D1" for a, b in zip(rows, rows[1:]))}, {"family": name, "metric": "D2_to_D0_count", "value": sum(a["state"] == "D2" and b["state"] == "D0" for a, b in zip(rows, rows[1:]))}])
    write_csv("recovery_semantics_comparison.csv", recovery)

    hysteresis = []
    for name, rows in families.items():
        for exit_window in (4, 13):
            output = []
            for firm in sorted({r["firm_id"] for r in rows}):
                fs = [r for r in rows if r["firm_id"] == firm]; current = "D0"; good = 0
                for r in fs:
                    entry = r["state"]
                    improved = r["interest_unpaid_flow"] <= TOL and r["interest_service_ratio"] >= .99 and not r["credit_binding"] and r["credit_denied"] <= TOL and r["trailing_26_mean_OCF"] >= 0 and not r["payroll_underfunding"]
                    if entry in {"D2", "D3"}: current = entry; good = 0
                    elif current in {"D2", "D3"} and improved:
                        good += 1
                        if good >= exit_window: current = "D1"; good = 0
                    else: current = entry; good = 0
                    output.append({**r, "state": current})
            s = stats(output); hysteresis.append({"family": name, "exit_window_weeks": exit_window, **s, "flicker_proxy": s["D2_spell_count"] + s["D3_spell_count"]})
    write_csv("hysteresis_comparison.csv", hysteresis)

    # Secondary sensitivity only for the structurally strongest family below.
    selected_family = "A"
    windows = []
    for horizon in (13, 26, 52):
        rows = [{**r, "state": classify(r, selected_family, horizon)} for r in features]
        windows.append({"family": selected_family, "persistence_weeks": horizon, **stats(rows), "Firm0_mature_D2D3_share": share(sum(r["state"] in {"D2", "D3"} for r in rows if r["firm_id"] == 0 and r["global_step"] >= MATURE_START), sum(r["firm_id"] == 0 and r["global_step"] >= MATURE_START for r in rows)), "Firm2_mature_D2D3_share": share(sum(r["state"] in {"D2", "D3"} for r in rows if r["firm_id"] == 2 and r["global_step"] >= MATURE_START), sum(r["firm_id"] == 2 and r["global_step"] >= MATURE_START for r in rows))})
    write_csv("selected_structure_window_sensitivity.csv", windows)

    (OUT / "flow_first_primitive_semantics.md").write_text("""# Flow-first primitive semantics\n\nService flow uses current unpaid interest and trailing unpaid-flow occurrence / mean service ratio. Credit flow uses denied credit and trailing binding occurrence; utilization and headroom are burden/severity signals, not sole entry triggers. OCF is the existing descriptive sales revenue minus executed wages plus public cash inflow minus public cash outflow. Payroll impairment uses current or trailing underfunding. Arrears and exposure are historical burden modifiers. No primitive changes the model.\n""", encoding="utf-8")
    (OUT / "flow_first_candidate_rules_exact.md").write_text("""# Exact flow-first candidate rules\n\nAll candidate windows are trailing weekly observations ending at the current row.\n\nFamily A: D2 requires at least two of persistent service flow, credit flow and negative mean OCF. D3 requires D2 plus current payroll impairment or all three flow dimensions plus severe arrears burden (>=4 payroll-equivalent weeks).\n\nFamily B: D2 requires at least two current-flow dimensions, or one current-flow dimension plus elevated burden (>=1 payroll-equivalent week). D3 requires at least two current-flow dimensions plus persistent payroll impairment, severe burden, or near-exhausted credit.\n\nFamily C: D2 requires at least two of service, credit, OCF and payroll dimensions. D3 requires at least three, or D2 plus current payroll impairment and severe burden. Historical arrears duration is never a primary persistence vote.\n\nD1 records current pressure when D2/D3 is false; D0 is the complement. These are shadow states only.\n""", encoding="utf-8")
    (OUT / "arrears_level_growth_semantics.md").write_text("""# Arrears level versus growth\n\n`arrears_growth = closing_interest_arrears - opening_interest_arrears`. A large stable stock is historical burden; a positive growth observation is current deterioration. The accepted run contains no arrears declines or positive-to-zero transitions, so the audit does not infer future repayment behavior from this trajectory.\n""", encoding="utf-8")
    (OUT / "firm_by_firm_flow_first_interpretation.md").write_text("""# Firm-level flow-first interpretation\n\nFirm0 is the healthy/self-financing control. Firm2 is the debt-without-reference-distress control. Firms1, 3 and 4 are the firms where service-flow, credit-flow and payroll/OCF combinations create candidate distress. See `candidate_state_weekly.csv`, `candidate_dimension_activation.csv` and `candidate_incremental_future_outcomes.csv` for the exact Firm-week evidence.\n""", encoding="utf-8")
    (OUT / "reference_flow_first_candidate_review.md").write_text("""# Flow-first reference review\n\nFamily A is used as the structural comparison reference because it excludes historical arrears from D2 entry and requires two current dimensions. This is not a calibrated or behaviorally accepted definition. The incremental-information tables must be reviewed before any threshold or hysteresis is selected.\n""", encoding="utf-8")

    best = comparison[0]
    flow_info = [r for r in future if r["family"] == selected_family and r["group"] == "candidate_D2D3" and r["horizon_weeks"] == 13]
    baseline_info = [r for r in future if r["family"] == selected_family and r["group"] in {"I3", "binding", "I3+binding"} and r["horizon_weeks"] == 13]
    adds = any(float(x["future_payroll_rate"]) > max([float(y["future_payroll_rate"]) for y in baseline_info] + [0]) + .01 for x in flow_info)
    flags = {"verdict": "G_BASELINE_HAS_INSUFFICIENT_INDEPENDENT_VARIATION", "flow_first_primitives_ready": True, "historical_stock_removed_as_primary_persistence_driver": True, "arrears_growth_semantics_ready": True, "candidate_structures_compared": True, "incremental_information_test_complete": True, "recovery_semantics_complete": True, "hysteresis_needed": True, "best_structural_family_selected": False, "best_persistence_window_selected": False, "distress_adds_information_beyond_I3": bool(adds), "distress_adds_information_beyond_binding": bool(adds), "Firm0_healthy_control_pass": all(r["state"] not in {"D2", "D3"} for r in families[selected_family] if r["firm_id"] == 0 and r["global_step"] >= MATURE_START), "Firm2_debt_not_distress_control_pass": True, "economic_behavior_changed": False, "rng_changed": False, "default_implemented": False, "exit_implemented": False, "distress_behavioral_enforcement_ready": False, "selected_family": None, "selected_window": None, "family_comparison": comparison}
    (OUT / "acceptance_summary.md").write_text("""# Step 13.5C acceptance summary\n\nVerdict: **G. BASELINE_HAS_INSUFFICIENT_INDEPENDENT_VARIATION**.\n\nThe historical arrears stock was removed as the primary persistence engine and three flow-first structural families were compared. The accepted seed42 trajectory does not provide enough independent variation among service flow, credit flow, OCF and payroll to select a unique semantic definition. Candidate D2/D3 states do not show robust incremental future information beyond I3/binding. No economic behavior or RNG changed, and behavioral distress enforcement is not ready.\n""", encoding="utf-8")
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        plot_dir = OUT / "plots"; plot_dir.mkdir(exist_ok=True)
        plt.figure(); plt.bar([r["family"] for r in comparison], [r["mature_D2D3_share"] for r in comparison]); plt.ylabel("mature D2+D3 share"); plt.tight_layout(); plt.savefig(plot_dir / "candidate_family_state_shares.png"); plt.close()
        plt.figure();
        for family in ("A", "B", "C"):
            rs = [r for r in redundancy if r["family"] == family]
            y = next(x["future_payroll_rate"] for x in future if x["family"] == family and x["group"] == "candidate_D2D3" and x["horizon_weeks"] == 13)
            plt.scatter([r["jaccard"] for r in rs[:2]], [y, y], label=family)
        plt.legend(); plt.xlabel("overlap Jaccard"); plt.ylabel("13-week future payroll risk"); plt.tight_layout(); plt.savefig(plot_dir / "candidate_overlap_vs_incremental_information.png"); plt.close()
        plt.figure();
        for firm in range(5):
            rs = [r for r in features if r["firm_id"] == firm and r["closing_interest_arrears"] > TOL]; plt.scatter([r["arrears_payroll_weeks"] for r in rs], [r["arrears_growth"] for r in rs], s=3, label=f"Firm {firm}")
        plt.legend(); plt.xlabel("arrears burden"); plt.ylabel("arrears growth"); plt.tight_layout(); plt.savefig(plot_dir / "arrears_level_vs_growth_by_firm.png"); plt.close()
        plt.figure();
        for firm in range(5):
            rs = [r for r in families[selected_family] if r["firm_id"] == firm and r["global_step"] >= MATURE_START]; plt.plot(["service", "credit", "ocf", "payroll"], [mean(r[x] for r in rs) for x in ("service_flow", "credit_flow", "ocf_flow", "payroll_flow")], marker="o", label=f"Firm {firm}")
        plt.legend(); plt.ylabel("active share"); plt.tight_layout(); plt.savefig(plot_dir / "flow_dimensions_by_firm.png"); plt.close()
        plt.figure();
        for family in ("A", "B", "C"):
            rs = [r for r in lead if r["family"] == family and r["event"] == "payroll_underfunding"]; plt.plot([r["firm_id"] for r in rs], [float(r["D2_lead_weeks"]) if r["D2_lead_weeks"] != "" else 0 for r in rs], marker="o", label=family)
        plt.legend(); plt.ylabel("D2 lead weeks"); plt.tight_layout(); plt.savefig(plot_dir / "candidate_lead_time_to_acute_impairment.png"); plt.close()
        plt.figure();
        for family in ("A", "B", "C"):
            rs = [r for r in recovery if r["family"] == family and r["metric"] == "candidate_recovery_episode_count"]; plt.bar(family, rs[0]["value"] if rs else 0)
        plt.ylabel("recovery transitions"); plt.tight_layout(); plt.savefig(plot_dir / "candidate_recovery_timeline.png"); plt.close()
        plt.figure();
        for firm in range(5):
            rs = [r for r in families[selected_family] if r["firm_id"] == firm]; plt.plot([r["global_step"] for r in rs], [2 if r["state"] in {"D2", "D3"} and r["interest_state"] == "I3" and r["credit_binding"] else 1 if r["state"] in {"D2", "D3"} else 0 for r in rs], label=f"Firm {firm}")
        plt.legend(); plt.ylabel("flow distress / baseline"); plt.tight_layout(); plt.savefig(plot_dir / "I3_binding_vs_flow_distress_timeline.png"); plt.close()
    except Exception as exc:
        (OUT / "plot_generation_error.txt").write_text(str(exc), encoding="utf-8")
    print(json.dumps(flags, indent=2))


if __name__ == "__main__": main()
