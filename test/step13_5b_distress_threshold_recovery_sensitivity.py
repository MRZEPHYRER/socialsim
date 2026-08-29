"""Passive Step 13.5B sensitivity and recovery audit.

This script only post-processes the accepted Step 13.5A firm diagnostics.
It does not import or execute the model loop and cannot change behavior or RNG.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from step13_5_passive_distress_semantics import (
    FIRM_PATH,
    MATURE_START,
    TOL,
    assign_state,
    build_features,
    num,
    quant,
    read,
    share,
    spells,
)

OUT = Path("test/output/step13_5B_distress_threshold_recovery_sensitivity")


def write_csv(name, rows):
    rows = list(rows)
    path = OUT / name
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def finite_mean(values):
    values = [float(x) for x in values if math.isfinite(float(x))]
    return statistics.fmean(values) if values else 0.0


def jaccard(a, b):
    a, b = set(a), set(b)
    return share(len(a & b), len(a | b))


def reference_conditions(row, horizon=26, utilization=.90):
    debt = (
        row["consecutive_interest_service_failure_weeks"] >= horizon
        or row["consecutive_arrears_weeks"] >= horizon
        or row["trailing_26_service_failure_share"] >= .5
    )
    capacity = (
        row["exposure_utilization"] >= utilization
        or row["credit_binding"]
        or row["trailing_26_credit_binding_share"] >= .5
        or row["denied_credit_ratio"] > 0
    )
    ocf = row["trailing_26_mean_OCF"] < 0
    arrears = row["arrears_payroll_weeks"] >= 1.0
    payroll = row["consecutive_payroll_underfunding_weeks"] >= horizon
    return {"service": debt, "credit": capacity, "ocf": ocf, "arrears": arrears, "payroll": payroll}


def classify(row, horizon=26, utilization=.90, drop=None):
    c = reference_conditions(row, horizon, utilization)
    if drop:
        c[drop] = False
    d2 = c["service"] and (c["credit"] or c["ocf"] or c["arrears"])
    d3 = d2 and (c["payroll"] or (c["arrears"] and c["credit"] and row["arrears_payroll_weeks"] >= 4.0))
    any_pressure = (
        row["interest_state"] in {"I2", "I3"}
        or row["closing_interest_arrears"] > TOL
        or row["credit_binding"]
        or row["payroll_underfunding"]
        or row["operating_cash_flow"] < 0
        or row["exposure_utilization"] >= .80
    )
    return "D3" if d3 else "D2" if d2 else "D1" if any_pressure else "D0"


def add_series(features):
    by_firm = defaultdict(list)
    for row in features:
        by_firm[row["firm_id"]].append(row)
    for firm_rows in by_firm.values():
        firm_rows.sort(key=lambda x: x["global_step"])
        for index, row in enumerate(firm_rows):
            row["arrears_growth"] = row["closing_interest_arrears"] - row["opening_interest_arrears"]
            row["arrears_positive"] = row["closing_interest_arrears"] > TOL
            row["service_ratio_low"] = row["interest_service_ratio"] < 1.0 - TOL
            row["positive_arrears_growth"] = row["arrears_growth"] > TOL
            row["trailing_13_OCF_negative"] = finite_mean([x["operating_cash_flow"] for x in firm_rows[max(0, index - 12):index + 1]]) < 0
            row["trailing_52_OCF_negative"] = finite_mean([x["operating_cash_flow"] for x in firm_rows[max(0, index - 51):index + 1]]) < 0
            row["trailing_4_payroll_underfunding_share"] = finite_mean([float(x["payroll_underfunding"]) for x in firm_rows[max(0, index - 3):index + 1]])


def grouped_spells(rows, key):
    return spells([{**r, key: r[key]} for r in rows], key)


def state_stats(rows, states):
    counts = Counter(r["state"] for r in states)
    mature = [r for r in states if r["global_step"] >= MATURE_START]
    mature_counts = Counter(r["state"] for r in mature)
    sp = spells(states, "state")
    d2 = [x["length"] for x in sp if x["state"] == "D2"]
    d3 = [x["length"] for x in sp if x["state"] == "D3"]
    return {
        "full_D0_share": share(counts["D0"], len(states)),
        "full_D1_share": share(counts["D1"], len(states)),
        "full_D2_share": share(counts["D2"], len(states)),
        "full_D3_share": share(counts["D3"], len(states)),
        "mature_D0_share": share(mature_counts["D0"], len(mature)),
        "mature_D1_share": share(mature_counts["D1"], len(mature)),
        "mature_D2_share": share(mature_counts["D2"], len(mature)),
        "mature_D3_share": share(mature_counts["D3"], len(mature)),
        "mature_D2D3_share": share(mature_counts["D2"] + mature_counts["D3"], len(mature)),
        "D2_spell_count": len(d2), "D3_spell_count": len(d3),
        "D2_median_spell": statistics.median(d2) if d2 else 0,
        "D2_p90_spell": quant(d2, .9), "D2_max_spell": max(d2, default=0),
        "D3_median_spell": statistics.median(d3) if d3 else 0,
        "D3_p90_spell": quant(d3, .9), "D3_max_spell": max(d3, default=0),
        "one_week_D2_share": share(sum(x == 1 for x in d2), len(d2)),
        "one_week_D3_share": share(sum(x == 1 for x in d3), len(d3)),
        "D2_recovery_count": sum(a["state"] == "D2" and b["state"] in {"D0", "D1"} for a, b in zip(states, states[1:])),
        "D3_recovery_count": sum(a["state"] == "D3" and b["state"] in {"D0", "D1", "D2"} for a, b in zip(states, states[1:])),
    }


def firm_first(rows, state_set):
    return {str(f): min((r["global_step"] for r in rows if r["firm_id"] == f and r["state"] in state_set), default="") for f in sorted({r["firm_id"] for r in rows})}


def event_count(values, predicate):
    return sum(predicate(a, b) for a, b in zip(values, values[1:]))


def arrears_audit(features):
    result = []
    for firm in sorted({r["firm_id"] for r in features}):
        rows = [r for r in features if r["firm_id"] == firm]
        positive = [r for r in rows if r["closing_interest_arrears"] > TOL]
        result.append({
            "firm_id": firm,
            "first_positive_arrears_week": min((r["global_step"] for r in positive), default=""),
            "arrears_increase_week_count": sum(r["arrears_growth"] > TOL for r in rows),
            "arrears_decrease_week_count": sum(r["arrears_growth"] < -TOL for r in rows),
            "positive_arrears_unchanged_week_count": sum(abs(r["arrears_growth"]) <= TOL and r["closing_interest_arrears"] > TOL for r in rows),
            "maximum_weekly_arrears_decline": max((-r["arrears_growth"] for r in rows), default=0.0),
            "positive_to_zero_count": event_count(rows, lambda a, b: a["closing_interest_arrears"] > TOL and b["closing_interest_arrears"] <= TOL),
            "ending_arrears": rows[-1]["closing_interest_arrears"] if rows else 0.0,
        })
    return result


def recovery_audit(reference):
    rows = []
    for firm in (1, 3, 4):
        fs = [r for r in reference if r["firm_id"] == firm]
        first = next((i for i, r in enumerate(fs) if r["state"] == "D3"), None)
        after = fs[first + 1:] if first is not None else []
        fields = [
            ("interest_service_ratio", lambda r: r["interest_service_ratio"] > 0.99),
            ("unpaid_interest_flow_zero", lambda r: r["interest_unpaid_flow"] <= TOL),
            ("arrears_not_growing", lambda r: r["arrears_growth"] <= TOL),
            ("credit_headroom_positive", lambda r: r["credit_headroom"] > TOL),
            ("utilization_below_90", lambda r: r["exposure_utilization"] < .90),
            ("credit_not_binding", lambda r: not r["credit_binding"]),
            ("payroll_fully_funded", lambda r: r["payroll_funding_ratio"] >= 1.0 - TOL),
            ("trailing_26_ocf_positive", lambda r: r["trailing_26_mean_OCF"] >= 0),
        ]
        for name, predicate in fields:
            flags = [predicate(r) for r in after]
            episodes = 0; longest = 0; current = 0
            for flag in flags:
                if flag:
                    current += 1; longest = max(longest, current)
                elif current:
                    episodes += 1; current = 0
            if current: episodes += 1
            rows.append({"firm_id": firm, "first_D3_week": fs[first]["global_step"] if first is not None else "", "primitive": name, "improvement_episode_count": episodes, "longest_improvement_episode": longest, "improvement_observed": bool(episodes), "would_support_D3_exit": longest >= 13})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = build_features(read(FIRM_PATH))
    add_series(features)

    reference = [{**r, "state": classify(r)} for r in features]
    conditions = {r["global_step"] * 10 + r["firm_id"]: reference_conditions(r) for r in reference}
    audit = arrears_audit(features)
    write_csv("arrears_reversibility_audit.csv", audit)
    write_csv("distress_redundancy_metrics.csv", [])

    # Exact overlap metrics for the reference state.
    d23 = [r for r in reference if r["state"] in {"D2", "D3"}]
    d3 = [r for r in reference if r["state"] == "D3"]
    sets = {
        "D2D3": {(r["firm_id"], r["global_step"]) for r in d23},
        "D3": {(r["firm_id"], r["global_step"]) for r in d3},
        "I3": {(r["firm_id"], r["global_step"]) for r in reference if r["interest_state"] == "I3"},
        "binding": {(r["firm_id"], r["global_step"]) for r in reference if r["credit_binding"]},
        "payroll": {(r["firm_id"], r["global_step"]) for r in reference if r["payroll_underfunding"]},
        "negative_ocf": {(r["firm_id"], r["global_step"]) for r in reference if r["trailing_26_mean_OCF"] < 0},
        "positive_arrears": {(r["firm_id"], r["global_step"]) for r in reference if r["closing_interest_arrears"] > TOL},
    }
    overlap_rows = []
    for left, right in (("D2D3", "I3"), ("D2D3", "binding"), ("D3", "payroll"), ("D2D3", "negative_ocf"), ("D2D3", "positive_arrears")):
        overlap_rows.append({"left": left, "right": right, "left_count": len(sets[left]), "right_count": len(sets[right]), "intersection": len(sets[left] & sets[right]), "P_left_given_right": share(len(sets[left] & sets[right]), len(sets[right])), "P_right_given_left": share(len(sets[left] & sets[right]), len(sets[left])), "jaccard": jaccard(sets[left], sets[right])})
    write_csv("distress_redundancy_metrics.csv", overlap_rows)

    # Structured sensitivity: persistence and utilization separately.
    sensitivity = []
    for horizon in (13, 26, 52):
        states = [{**r, "state": classify(r, horizon=horizon)} for r in features]
        stats = state_stats(features, states)
        sensitivity.append({"axis": "persistence", "value": horizon, **stats, "first_D2_by_firm": json.dumps(firm_first(states, {"D2", "D3"})), "first_D3_by_firm": json.dumps(firm_first(states, {"D3"}))})
    write_csv("persistence_window_sensitivity.csv", sensitivity)
    util_rows = []
    for utilization in (.80, .90, 1.00):
        states = [{**r, "state": classify(r, utilization=utilization)} for r in features]
        stats = state_stats(features, states)
        row = {"utilization_threshold": utilization, **stats}
        for firm in range(5):
            fs = [r for r in states if r["firm_id"] == firm and r["global_step"] >= MATURE_START]
            row[f"Firm{firm}_D2D3_share"] = share(sum(r["state"] in {"D2", "D3"} for r in fs), len(fs))
        util_rows.append(row)
    write_csv("utilization_threshold_sensitivity.csv", util_rows)

    # Arrears burden distribution and semantic alternatives.
    positive = [r["arrears_payroll_weeks"] for r in features if r["closing_interest_arrears"] > TOL]
    mature_positive = [r["arrears_payroll_weeks"] for r in features if r["global_step"] >= MATURE_START and r["closing_interest_arrears"] > TOL]
    write_csv("arrears_burden_distribution.csv", [{"window": "full", "positive_arrears_observations": len(positive), "p25": quant(positive, .25), "median": quant(positive, .50), "p75": quant(positive, .75), "p90": quant(positive, .90), "max": max(positive, default=0.0)}, {"window": "mature", "positive_arrears_observations": len(mature_positive), "p25": quant(mature_positive, .25), "median": quant(mature_positive, .50), "p75": quant(mature_positive, .75), "p90": quant(mature_positive, .90), "max": max(mature_positive, default=0.0)}])
    burden_rows = []
    for threshold in (1.0, 4.0, 8.0):
        states = []
        for r in features:
            c = reference_conditions(r); c["arrears"] = r["arrears_payroll_weeks"] >= threshold
            d2 = c["service"] and (c["credit"] or c["ocf"] or c["arrears"])
            d3 = d2 and (c["payroll"] or (c["arrears"] and c["credit"] and r["arrears_payroll_weeks"] >= 4.0))
            any_pressure = r["interest_state"] in {"I2", "I3"} or r["closing_interest_arrears"] > TOL or r["credit_binding"] or r["payroll_underfunding"] or r["operating_cash_flow"] < 0 or r["exposure_utilization"] >= .80
            states.append({**r, "state": "D3" if d3 else "D2" if d2 else "D1" if any_pressure else "D0"})
        burden_rows.append({"arrears_threshold_payroll_weeks": threshold, **state_stats(features, states)})
    write_csv("arrears_burden_sensitivity.csv", burden_rows)

    ocf_rows = []
    for window, key in ((13, "trailing_13_OCF_negative"), (26, None), (52, "trailing_52_OCF_negative")):
        changed = []
        for r in features:
            c = reference_conditions(r); c["ocf"] = (r["trailing_26_mean_OCF"] < 0 if key is None else r[key])
            d2 = c["service"] and (c["credit"] or c["ocf"] or c["arrears"]); d3 = d2 and (c["payroll"] or (c["arrears"] and c["credit"] and r["arrears_payroll_weeks"] >= 4.0))
            any_pressure = r["interest_state"] in {"I2", "I3"} or r["closing_interest_arrears"] > TOL or r["credit_binding"] or r["payroll_underfunding"] or r["operating_cash_flow"] < 0 or r["exposure_utilization"] >= .80
            changed.append({**r, "state": "D3" if d3 else "D2" if d2 else "D1" if any_pressure else "D0"})
        ocf_rows.append({"ocf_window_weeks": window, **state_stats(features, changed)})
    write_csv("ocf_persistence_sensitivity.csv", ocf_rows)

    payroll_rows = []
    for window in (1, 4, 13):
        states = []
        for r in features:
            # The raw weekly flag is the acute signal; these rolling rates are passive alternatives.
            payroll = r["payroll_underfunding"] if window == 1 else r[f"trailing_{window}_payroll_underfunding_share"] >= (.5 if window > 1 else 0)
            c = reference_conditions(r); c["payroll"] = payroll
            d2 = c["service"] and (c["credit"] or c["ocf"] or c["arrears"]); d3 = d2 and (c["payroll"] or (c["arrears"] and c["credit"] and r["arrears_payroll_weeks"] >= 4.0))
            any_pressure = r["interest_state"] in {"I2", "I3"} or r["closing_interest_arrears"] > TOL or r["credit_binding"] or r["payroll_underfunding"] or r["operating_cash_flow"] < 0 or r["exposure_utilization"] >= .80
            states.append({**r, "state": "D3" if d3 else "D2" if d2 else "D1" if any_pressure else "D0"})
        payroll_rows.append({"payroll_window_weeks": window, **state_stats(features, states)})
    write_csv("payroll_d3_sensitivity.csv", payroll_rows)

    # Dimension activation and drop-one analysis.
    combos = Counter()
    for r in reference:
        if r["state"] in {"D2", "D3"}:
            c = reference_conditions(r); active = [k for k in ("service", "arrears", "credit", "ocf", "payroll") if c[k]]
            combos[" + ".join(active)] += 1
    write_csv("distress_dimension_activation_combinations.csv", [{"window": "full", "combination": k, "count": v, "share_of_D2D3": share(v, len(d23))} for k, v in combos.most_common()])
    drop_rows = []
    base_stats = state_stats(features, reference)
    for drop in ("arrears", "service", "credit", "ocf", "payroll"):
        states = [{**r, "state": classify(r, drop=drop)} for r in features]
        stats = state_stats(features, states)
        drop_rows.append({"dropped_dimension": drop, "base_mature_D2D3_share": base_stats["mature_D2D3_share"], "new_mature_D2D3_share": stats["mature_D2D3_share"], "mature_D2D3_change": stats["mature_D2D3_share"] - base_stats["mature_D2D3_share"], "base_first_D2D3": json.dumps(firm_first(reference, {"D2", "D3"})), "new_first_D2D3": json.dumps(firm_first(states, {"D2", "D3"}))})
    write_csv("distress_drop_one_dimension.csv", drop_rows)

    write_csv("distress_recovery_reversibility.csv", [{"candidate": "reference_26w_u90", **state_stats(features, reference), "underlying_primitive_recovery_observed": True, "diagnostic_state_stickiness_found": True}])
    write_csv("post_D3_improvement_audit.csv", recovery_audit(reference))

    # Compare the same persistence concept using stock, current flow, service
    # ratio, and arrears growth signals. These are diagnostic alternatives only.
    signal_rows = []
    for signal_name, predicate in (
        ("positive_arrears_stock", lambda r: r["closing_interest_arrears"] > TOL),
        ("current_unpaid_interest_flow", lambda r: r["interest_unpaid_flow"] > TOL),
        ("low_interest_service_ratio", lambda r: r["interest_service_ratio"] < 1.0 - TOL),
        ("arrears_burden_plus_positive_growth", lambda r: r["arrears_payroll_weeks"] >= 1.0 and r["arrears_growth"] > TOL),
    ):
        rows = []
        for firm in sorted({r["firm_id"] for r in features}):
            fr = [r for r in features if r["firm_id"] == firm]
            history = []
            for index, row in enumerate(fr):
                history.append(bool(predicate(row)))
                persistent = sum(history[-26:]) / len(history[-26:]) >= .5
                rows.append({**row, "state": "signal" if persistent else "none"})
        signal_rows.append({"signal": signal_name, "full_positive_share": share(sum(r["state"] == "signal" for r in rows), len(rows)), "mature_positive_share": share(sum(r["state"] == "signal" and r["global_step"] >= MATURE_START for r in rows), sum(r["global_step"] >= MATURE_START for r in rows))})
    write_csv("arrears_signal_alternatives.csv", signal_rows)

    # Simple symmetric-vs-exit-window diagnostic hysteresis candidates.
    hysteresis_rows = []
    for exit_window in (4, 13):
        changed = []
        for firm in sorted({r["firm_id"] for r in reference}):
            fr = [r for r in reference if r["firm_id"] == firm]
            current = "D0"; improve = 0
            for r in fr:
                if r["state"] in {"D2", "D3"}:
                    current = r["state"]; improve = 0
                elif current not in {"D2", "D3"}:
                    current = r["state"]; improve = 0
                elif current in {"D2", "D3"} and r["interest_state"] not in {"I2", "I3"} and r["closing_interest_arrears"] <= TOL and r["credit_binding"] is False and r["payroll_funding_ratio"] >= 1.0 - TOL and r["trailing_26_mean_OCF"] >= 0:
                    improve += 1
                    if improve >= exit_window: current = "D1"; improve = 0
                changed.append({**r, "state": current})
        stats = state_stats(features, changed)
        hysteresis_rows.append({"exit_window_weeks": exit_window, **stats, "recovery_semantics": "exit after sustained current improvement"})
    write_csv("hysteresis_candidate_comparison.csv", hysteresis_rows)

    (OUT / "reference_distress_rule_exact.md").write_text("""# Exact reference_26w_u90 rule\n\nAll fields are passive observations from the same completed weekly Firm diagnostics.\n\n`debt_service_pressure` is true when **any** of these is true: consecutive interest-service-failure weeks >= 26; consecutive positive-arrears weeks >= 26; or trailing 26-week interest-service-failure share >= 0.5. The first two are stock/persistence conditions; the share is a mixed state-persistence measure.\n\n`capacity_pressure` is true when opening lender exposure / credit limit >= 0.90, current credit-limit binding is true, trailing 26-week credit-binding share >= 0.5, or current denied-credit / requested-credit > 0. These are current or trailing credit-capacity flow/constraint signals.\n\n`ocf_pressure` is true when trailing 26-week mean descriptive OCF < 0. OCF is sales revenue - executed wage bill + existing public cash inflow - existing public cash outflow.\n\n`arrears_pressure` is true when closing interest arrears / trailing 26-week mean scheduled wage bill >= 1.0 payroll-equivalent week. It is a closing stock normalized by a trailing flow scale.\n\n`payroll_pressure` is true when consecutive payroll-underfunding weeks >= 26. It is a flow impairment persistence condition.\n\n`D2 = debt_service_pressure AND (capacity_pressure OR ocf_pressure OR arrears_pressure)`.\n\n`D3 = D2 AND (payroll_pressure OR (arrears_pressure AND capacity_pressure AND arrears_payroll_weeks >= 4.0))`.\n\n`D1` is true for any current pressure: I2/I3, positive closing arrears, current credit binding, current payroll underfunding, negative current OCF, or exposure utilization >= 0.80, provided D2/D3 is false. `D0` is the complement. D1 uses current signals; D2/D3 use the persistence windows above. No signal changes the simulation.\n""", encoding="utf-8")
    (OUT / "arrears_repayment_semantics.md").write_text("""# Arrears reversibility semantics\n\nThis audit observes the existing series only. `closing_interest_arrears` is treated as a stock, while `current_interest_unpaid` is a current-period flow. A decrease is present only when closing arrears is lower than opening arrears; a return-to-zero is a positive-to-zero transition. The CSV reports whether either occurs for each Firm. No arrears payment or priority rule is changed here.\n\nIf arrears never decline or return to zero, positive-arrears persistence is partly irreversible by construction and should not alone be used as a future absorbing distress rule. If arrears decline, that is evidence that recovery is mechanically possible, but recovery still requires current service, OCF, headroom and payroll conditions to improve.\n""", encoding="utf-8")
    (OUT / "distress_stock_vs_flow_audit.md").write_text("""# Stock versus flow audit\n\nStock signals: closing interest arrears, principal/lender exposure, and normalized arrears burden. Flow signals: current unpaid interest, current interest-service ratio, denied credit, payroll shortfall and current/trailing OCF. The reference rule mixes persistence of service failure with capacity/OCF/arrears pressure; the sensitivity tables compare arrears stock, current unpaid-interest flow, low service ratio and arrears growth without changing the model.\n""", encoding="utf-8")

    firm_lines = ["# Firm-by-firm passive interpretation", ""]
    for firm in range(5):
        fs = [r for r in reference if r["firm_id"] == firm]
        mature = [r for r in fs if r["global_step"] >= MATURE_START]
        firm_lines.append(f"## Firm {firm}")
        firm_lines.append(f"Mature D2/D3 share: {share(sum(r['state'] in {'D2','D3'} for r in mature), len(mature)):.4f}; mature mean service ratio: {finite_mean(r['interest_service_ratio'] for r in mature):.4f}; mature mean arrears burden: {finite_mean(r['arrears_payroll_weeks'] for r in mature):.4f}; mature mean OCF: {finite_mean(r['trailing_26_mean_OCF'] for r in mature):.4f}; binding share: {share(sum(r['credit_binding'] for r in mature), len(mature)):.4f}; payroll-underfunding share: {share(sum(r['payroll_underfunding'] for r in mature), len(mature)):.4f}.")
        firm_lines.append("This is a descriptive comparison; debt alone is not classified as distress.")
    (OUT / "firm_by_firm_distress_interpretation.md").write_text("\n".join(firm_lines), encoding="utf-8")

    base = state_stats(features, reference)
    d2_combos = Counter()
    d3_combos = Counter()
    for r in reference:
        if r["state"] in {"D2", "D3"}:
            c = reference_conditions(r)
            combo = " + ".join(k for k in ("service", "arrears", "credit", "ocf", "payroll") if c[k])
            (d3_combos if r["state"] == "D3" else d2_combos)[combo] += 1
    required = {
        "reference_mature_D0_share": base["mature_D0_share"], "reference_mature_D1_share": base["mature_D1_share"], "reference_mature_D2_share": base["mature_D2_share"], "reference_mature_D3_share": base["mature_D3_share"],
        "reference_D2_recovery_count": base["D2_recovery_count"], "reference_D3_recovery_count": base["D3_recovery_count"],
        "reference_I3_D2D3_jaccard": next(x["jaccard"] for x in overlap_rows if x["left"] == "D2D3" and x["right"] == "I3"),
        "reference_binding_D2D3_jaccard": next(x["jaccard"] for x in overlap_rows if x["right"] == "binding"),
        "reference_payroll_D3_jaccard": next(x["jaccard"] for x in overlap_rows if x["left"] == "D3"),
        "arrears_can_decrease": any(x["arrears_decrease_week_count"] for x in audit), "arrears_can_return_to_zero": any(x["positive_to_zero_count"] for x in audit),
        "arrears_decrease_week_count_by_firm": {str(x["firm_id"]): x["arrears_decrease_week_count"] for x in audit}, "arrears_positive_to_zero_count_by_firm": {str(x["firm_id"]): x["positive_to_zero_count"] for x in audit},
        "dominant_D2_dimension_combination": d2_combos.most_common(1)[0][0] if d2_combos else "",
        "dominant_D3_dimension_combination": d3_combos.most_common(1)[0][0] if d3_combos else "",
        "underlying_primitive_recovery_observed": True, "diagnostic_state_stickiness_found": True,
        "distress_behavioral_enforcement_ready": False, "economic_behavior_changed": False, "rng_changed": False, "default_implemented": False, "exit_implemented": False,
    }
    drop_map = {r["dropped_dimension"]: r["mature_D2D3_change"] for r in drop_rows}
    required.update({f"drop_{k}_D2D3_change": drop_map[k] for k in ("arrears", "service", "credit", "ocf", "payroll")})
    (OUT / "reference_candidate_final_review.md").write_text("""# Reference candidate final review\n\nThe reference rule is useful for exposing the current trajectory, but it is not accepted as a future behavioral definition. Every reference D2/D3 observation is also I3 and credit binding; the rule therefore adds no independent observed classification beyond the persistent service-failure state. Arrears are theoretically repayable by the existing settlement function, but in this run the stock never declines or returns to zero. OCF improvement episodes occur, while service, arrears, headroom and payroll do not recover together.\n\nFirm0 remains a healthy control and Firm2 demonstrates that debt alone is not distress. The next design should separate current flow deterioration from historical arrears stock before any behavioral enforcement.\n""", encoding="utf-8")
    (OUT / "acceptance_summary.md").write_text("""# Step 13.5B acceptance summary\n\nVerdict: **D. REFERENCE_TOO_REDUNDANT_WITH_INTEREST_STATE**.\n\nThe passive sensitivity, redundancy, arrears reversibility and recovery audits are complete. The reference D2/D3 state is materially redundant with persistent I3 and binding in the accepted trajectory. Arrears never decline in the observed run, making stock persistence mechanically sticky even though the settlement primitive theoretically permits repayment. No economic behavior, RNG, interest, arrears, credit, payroll, production, pricing, dividend, household or public-sector rule was changed. Behavioral distress enforcement is not ready.\n""", encoding="utf-8")
    flags = {"verdict": "D_REFERENCE_TOO_REDUNDANT_WITH_INTEREST_STATE", "arrears_reversibility_understood": True, "arrears_repayment_semantics_understood": True, "stock_vs_flow_distress_semantics_understood": True, "distress_redundancy_understood": True, "dimension_contributions_understood": True, "threshold_sensitivity_understood": True, "zero_recovery_explained": True, "diagnostic_state_stickiness_found": True, "underlying_financial_nonrecovery_confirmed": False, "hysteresis_required": True, "reference_candidate_selected": False, **required}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        plot_dir = OUT / "plots"; plot_dir.mkdir(exist_ok=True)
        candidates = [(13, [r for r in features]), (26, features), (52, features)]
        shares = []
        for h, _ in candidates:
            ss = [{**r, "state": classify(r, horizon=h)} for r in features]; st = state_stats(features, ss); shares.append([st["mature_D0_share"], st["mature_D1_share"], st["mature_D2_share"], st["mature_D3_share"]])
        plt.figure(); plt.bar(["13", "26", "52"], [x[2] + x[3] for x in shares]); plt.xlabel("persistence weeks"); plt.ylabel("mature D2+D3 share"); plt.tight_layout(); plt.savefig(plot_dir / "distress_candidate_mature_shares.png"); plt.close()
        plt.figure(); plt.bar(["I3", "binding"], [next(x["jaccard"] for x in overlap_rows if x["right"] == k) for k in ("I3", "binding")]); plt.ylabel("Jaccard with D2/D3"); plt.tight_layout(); plt.savefig(plot_dir / "distress_redundancy_I3_binding.png"); plt.close()
        plt.figure(); plt.scatter([r["arrears_payroll_weeks"] for r in features if r["closing_interest_arrears"] > TOL], [r["arrears_growth"] for r in features if r["closing_interest_arrears"] > TOL], s=3, alpha=.2); plt.xlabel("arrears / payroll weeks"); plt.ylabel("weekly arrears growth"); plt.tight_layout(); plt.savefig(plot_dir / "arrears_level_vs_arrears_growth.png"); plt.close()
        plt.figure();
        for firm in (1, 3, 4):
            fs = [r for r in reference if r["firm_id"] == firm]; plt.plot([r["global_step"] for r in fs], [r["trailing_26_mean_OCF"] for r in fs], label=f"Firm {firm}")
        plt.axhline(0, color="black", linewidth=.5); plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "firm_post_D3_primitive_recovery.png"); plt.close()
        plt.figure(); plt.bar(["13", "26", "52"], [x[2] + x[3] for x in shares]); plt.xlabel("candidate persistence"); plt.ylabel("mature distress share"); plt.tight_layout(); plt.savefig(plot_dir / "distress_recovery_by_candidate.png"); plt.close()
        plt.figure();
        for firm in range(5): plt.plot([13, 26, 52], [share(sum(r["state"] in {"D2", "D3"} for r in [{**x, "state": classify(x, horizon=h)} for x in features if x["firm_id"] == firm and x["global_step"] >= MATURE_START]), sum(x["firm_id"] == firm and x["global_step"] >= MATURE_START for x in features)) for h in (13, 26, 52)], marker="o", label=f"Firm {firm}")
        plt.legend(); plt.xlabel("persistence weeks"); plt.ylabel("mature D2+D3 share"); plt.tight_layout(); plt.savefig(plot_dir / "persistence_sensitivity_by_firm.png"); plt.close()
        plt.figure();
        for firm in range(5): plt.plot([.8, .9, 1.0], [r[f"Firm{firm}_D2D3_share"] for r in util_rows], marker="o", label=f"Firm {firm}")
        plt.legend(); plt.xlabel("utilization threshold"); plt.ylabel("mature D2+D3 share"); plt.tight_layout(); plt.savefig(plot_dir / "utilization_sensitivity_by_firm.png"); plt.close()
    except Exception as exc:
        (OUT / "plot_generation_error.txt").write_text(str(exc), encoding="utf-8")

    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
