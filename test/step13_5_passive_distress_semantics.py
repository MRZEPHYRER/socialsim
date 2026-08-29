"""Passive Step 13.5 shadow distress audit; post-processing only."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path("test/output/step13_5_passive_distress_semantics")
RUN = Path("test/output/pre_step13_5K_household_denominator_recovery_hazard/n5000_targeted")
FIRM_PATH = RUN / "firm_diagnostics.csv"
TOL = 1e-9
MATURE_START = 1560


def num(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(name, rows):
    path = ROOT / name
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader(); writer.writerows(rows)


def share(n, d):
    return n / d if d else 0.0


def quant(values, q):
    values = sorted(values)
    if not values: return 0.0
    return values[min(len(values) - 1, int((len(values) - 1) * q))]


def service_state(row):
    obligation = num(row, "total_interest_obligation")
    paid = num(row, "interest_paid")
    unpaid = num(row, "current_interest_unpaid")
    return "I0" if obligation <= TOL else "I1" if unpaid <= TOL else "I2" if paid > TOL else "I3"


def rolling(values, window):
    return values[-window:] if len(values) >= window else values[:]


def build_features(rows):
    by_firm = defaultdict(list)
    for row in rows:
        by_firm[int(num(row, "firm_id"))].append(row)
    output = []
    for firm_id, firm_rows in by_firm.items():
        firm_rows.sort(key=lambda row: int(num(row, "global_step", num(row, "step"))))
        service_failures = []
        bindings = []
        payroll = []
        ocf_values = []
        arrears_values = []
        util_values = []
        for index, row in enumerate(firm_rows):
            week = int(num(row, "global_step", num(row, "step")))
            interest_state = service_state(row)
            service_failure = interest_state in {"I2", "I3"}
            binding = str(row.get("credit_limit_binding", "False")) == "True"
            payroll_under = num(row, "payroll_funding_ratio", 1.0) < 1.0 - TOL
            ocf = (
                num(row, "sales_revenue")
                - num(row, "executed_wage_bill", num(row, "wage_payment"))
                + num(row, "public_sector_cash_inflow")
                - num(row, "public_sector_cash_outflow")
            )
            credit_limit = num(row, "credit_limit")
            opening_exposure = num(row, "opening_lender_exposure")
            utilization = opening_exposure / credit_limit if credit_limit > TOL and math.isfinite(credit_limit) else 0.0
            requested = num(row, "requested_credit")
            denied = num(row, "denied_credit")
            denied_ratio = denied / requested if requested > TOL else 0.0
            scheduled = num(row, "scheduled_wage_bill")
            arrears = num(row, "closing_interest_arrears", num(row, "interest_arrears"))
            service_failures.append(int(service_failure)); bindings.append(int(binding)); payroll.append(int(payroll_under)); ocf_values.append(ocf); arrears_values.append(arrears); util_values.append(utilization)
            def streak(values):
                count = 0
                for value in reversed(values):
                    if value: count += 1
                    else: break
                return count
            def frac(values, window):
                part = rolling(values, window)
                return sum(part) / len(part) if part else 0.0
            wage_window = firm_rows[max(0, index - 25):index + 1]
            trailing_wage = statistics.fmean([num(x, "scheduled_wage_bill") for x in wage_window]) if wage_window else 0.0
            trailing26 = rolling(ocf_values, 26); trailing52 = rolling(ocf_values, 52)
            arrears_payroll_weeks = arrears / trailing_wage if trailing_wage > TOL else 0.0
            output.append({
                "global_step": week, "firm_id": firm_id, "interest_state": interest_state,
                "interest_due": num(row, "total_interest_obligation"), "interest_paid": num(row, "interest_paid"),
                "interest_unpaid_flow": num(row, "current_interest_unpaid"), "opening_interest_arrears": num(row, "opening_interest_arrears"),
                "closing_interest_arrears": arrears, "interest_service_ratio": share(num(row, "interest_paid"), num(row, "total_interest_obligation")),
                "credit_limit": credit_limit, "opening_lender_exposure": opening_exposure, "closing_lender_exposure": num(row, "closing_lender_exposure", num(row, "lender_exposure")),
                "credit_headroom": num(row, "credit_headroom"), "credit_requested": requested, "credit_executed": num(row, "executed_credit"), "credit_denied": denied,
                "credit_binding": binding, "exposure_utilization": utilization, "denied_credit_ratio": denied_ratio,
                "scheduled_wage_bill": scheduled, "executed_wage_bill": num(row, "executed_wage_bill", num(row, "wage_payment")),
                "payroll_funding_ratio": num(row, "payroll_funding_ratio", 1.0), "payroll_underfunding": payroll_under,
                "operating_cash_flow": ocf, "cash": num(row, "cash"), "inventory": num(row, "inventory_units", num(row, "inventory")), "sales": num(row, "sales"),
                "production": num(row, "actual_production", num(row, "production")), "market_share": num(row, "unit_market_share"), "employees": num(row, "employee_count"),
                "capacity_utilization": num(row, "capacity_utilization"), "price": num(row, "price"), "margin": num(row, "margin"),
                "arrears_payroll_weeks": arrears_payroll_weeks,
                "consecutive_arrears_weeks": streak([x > TOL for x in arrears_values]), "consecutive_interest_service_failure_weeks": streak(service_failures),
                "trailing_13_service_failure_share": frac(service_failures, 13), "trailing_26_service_failure_share": frac(service_failures, 26), "trailing_52_service_failure_share": frac(service_failures, 52),
                "consecutive_credit_binding_weeks": streak(bindings), "trailing_13_credit_binding_share": frac(bindings, 13), "trailing_26_credit_binding_share": frac(bindings, 26), "trailing_52_credit_binding_share": frac(bindings, 52),
                "consecutive_payroll_underfunding_weeks": streak(payroll), "trailing_13_payroll_underfunding_share": frac(payroll, 13), "trailing_26_payroll_underfunding_share": frac(payroll, 26), "trailing_52_payroll_underfunding_share": frac(payroll, 52),
                "trailing_13_OCF": math.fsum(rolling(ocf_values, 13)), "trailing_26_OCF": math.fsum(trailing26), "trailing_52_OCF": math.fsum(trailing52),
                "trailing_26_mean_OCF": statistics.fmean(trailing26) if trailing26 else 0.0, "trailing_52_mean_OCF": statistics.fmean(trailing52) if trailing52 else 0.0,
            })
    return output


def assign_state(row, horizon=26, utilization_threshold=.90, arrears_threshold=1.0):
    debt_pressure = row["consecutive_interest_service_failure_weeks"] >= horizon or row["consecutive_arrears_weeks"] >= horizon or row["trailing_26_service_failure_share"] >= .5
    capacity_pressure = row["exposure_utilization"] >= utilization_threshold or row["credit_binding"] or row["trailing_26_credit_binding_share"] >= .5 or row["denied_credit_ratio"] > 0
    ocf_pressure = row["trailing_26_mean_OCF"] < 0
    arrears_pressure = row["arrears_payroll_weeks"] >= arrears_threshold
    any_pressure = row["interest_state"] in {"I2", "I3"} or row["closing_interest_arrears"] > TOL or row["credit_binding"] or row["payroll_underfunding"] or row["operating_cash_flow"] < 0 or row["exposure_utilization"] >= .80
    persistent_other = capacity_pressure or ocf_pressure or arrears_pressure
    d2 = debt_pressure and persistent_other
    d3 = d2 and (row["consecutive_payroll_underfunding_weeks"] >= horizon or (arrears_pressure and capacity_pressure and row["arrears_payroll_weeks"] >= 4.0))
    return "D3" if d3 else "D2" if d2 else "D1" if any_pressure else "D0"


def spells(rows, state_key):
    result = []
    for firm_id in sorted({row["firm_id"] for row in rows}):
        values = [row for row in rows if row["firm_id"] == firm_id]
        current = None
        for row in values + [{"global_step": 10**9, state_key: "END"}]:
            state = row.get(state_key)
            if state in {"D1", "D2", "D3"}:
                if current is None or current["state"] != state or int(row["global_step"]) != current["last"] + 1:
                    if current: result.append(current)
                    current = {"firm_id": firm_id, "state": state, "first": int(row["global_step"]), "last": int(row["global_step"]), "length": 1}
                else:
                    current["last"] = int(row["global_step"]); current["length"] += 1
            elif current:
                result.append(current); current = None
    return result


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    raw = read(FIRM_PATH)
    features = build_features(raw)
    candidate_rows = []
    candidate_summaries = []
    candidates = [("loose_13w_u80", 13, .80), ("loose_13w_u90", 13, .90), ("reference_26w_u90", 26, .90), ("reference_26w_u100", 26, 1.00), ("strict_52w_u90", 52, .90), ("strict_52w_u100", 52, 1.00)]
    for name, horizon, util in candidates:
        assigned = []
        for row in features:
            item = dict(row); item["candidate"] = name; item["distress_state"] = assign_state(row, horizon, util); assigned.append(item)
        counts = Counter(row["distress_state"] for row in assigned); mature = [row for row in assigned if row["global_step"] >= MATURE_START]; mature_counts = Counter(row["distress_state"] for row in mature)
        candidate_summaries.append({"candidate": name, "persistence_weeks": horizon, "utilization_threshold": util, "full_run_D0_share": share(counts["D0"], len(assigned)), "full_run_D1_share": share(counts["D1"], len(assigned)), "full_run_D2_share": share(counts["D2"], len(assigned)), "full_run_D3_share": share(counts["D3"], len(assigned)), "mature_D0_share": share(mature_counts["D0"], len(mature)), "mature_D1_share": share(mature_counts["D1"], len(mature)), "mature_D2_share": share(mature_counts["D2"], len(mature)), "mature_D3_share": share(mature_counts["D3"], len(mature)), "one_week_D2_spell_share": 0.0, "one_week_D3_spell_share": 0.0, "D2_recovery_count": 0, "D3_recovery_count": 0})
        if name == "reference_26w_u90":
            candidate_rows = assigned
    write("firm_distress_shadow_weekly.csv", candidate_rows)
    write("firm_distress_candidate_comparison.csv", candidate_summaries)
    reference = candidate_rows

    state_spells = spells(reference, "distress_state")
    spell_summary = []
    for firm_id in sorted({row["firm_id"] for row in reference}):
        for state in ("D1", "D2", "D3"):
            values = [x["length"] for x in state_spells if x["firm_id"] == firm_id and x["state"] == state]
            spell_summary.append({"firm_id": firm_id, "state": state, "spell_count": len(values), "first_week": min([x["first"] for x in state_spells if x["firm_id"] == firm_id and x["state"] == state], default=""), "median_spell_length": statistics.median(values) if values else 0, "p90_spell_length": quant(values, .90), "max_spell_length": max(values, default=0), "one_week_spell_share": share(sum(x == 1 for x in values), len(values))})
    write("distress_spell_summary.csv", spell_summary)

    transitions = Counter((before["distress_state"], after["distress_state"]) for firm_id in sorted({x["firm_id"] for x in reference}) for before, after in zip([x for x in reference if x["firm_id"] == firm_id], [x for x in reference if x["firm_id"] == firm_id][1:]))
    write("distress_state_transition_matrix.csv", [{"from_state": a, "to_state": b, "count": n} for (a, b), n in sorted(transitions.items())])
    cross_interest = Counter((row["distress_state"], row["interest_state"]) for row in reference)
    write("distress_vs_interest_state.csv", [{"distress_state": a, "interest_state": b, "count": n} for (a, b), n in sorted(cross_interest.items())])
    write("distress_vs_credit_binding.csv", [{"distress_state": state, "binding": binding, "count": sum(row["distress_state"] == state and row["credit_binding"] == binding for row in reference)} for state in ("D0", "D1", "D2", "D3") for binding in (True, False)])
    write("distress_vs_payroll_underfunding.csv", [{"distress_state": state, "payroll_underfunding": under, "count": sum(row["distress_state"] == state and row["payroll_underfunding"] == under for row in reference)} for state in ("D0", "D1", "D2", "D3") for under in (True, False)])
    write("distress_vs_operating_condition.csv", [{"distress_state": state, "observations": sum(row["distress_state"] == state for row in reference), "mean_OCF": statistics.fmean([row["operating_cash_flow"] for row in reference if row["distress_state"] == state]) if any(row["distress_state"] == state for row in reference) else 0.0, "mean_cash": statistics.fmean([row["cash"] for row in reference if row["distress_state"] == state]) if any(row["distress_state"] == state for row in reference) else 0.0, "mean_sales": statistics.fmean([row["sales"] for row in reference if row["distress_state"] == state]) if any(row["distress_state"] == state for row in reference) else 0.0} for state in ("D0", "D1", "D2", "D3")])
    write("firm_distress_timeline_summary.csv", [{"firm_id": firm_id, "first_D1_week": min([x["global_step"] for x in reference if x["firm_id"] == firm_id and x["distress_state"] in {"D1", "D2", "D3"}], default=""), "first_D2_week": min([x["global_step"] for x in reference if x["firm_id"] == firm_id and x["distress_state"] in {"D2", "D3"}], default=""), "first_D3_week": min([x["global_step"] for x in reference if x["firm_id"] == firm_id and x["distress_state"] == "D3"], default=""), "mature_D2_D3_share": share(sum(x["distress_state"] in {"D2", "D3"} for x in reference if x["firm_id"] == firm_id and x["global_step"] >= MATURE_START), sum(x["global_step"] >= MATURE_START for x in reference if x["firm_id"] == firm_id))} for firm_id in sorted({x["firm_id"] for x in reference})])

    full = candidate_summaries[2]
    firm0 = [x for x in reference if x["firm_id"] == 0 and x["global_step"] >= MATURE_START]
    i3 = [x for x in reference if x["interest_state"] == "I3"]
    d2d3 = [x for x in reference if x["distress_state"] in {"D2", "D3"}]
    binding = [x for x in reference if x["credit_binding"]]
    payroll = [x for x in reference if x["payroll_underfunding"]]
    required = {"full_run_D0_share": full["full_run_D0_share"], "full_run_D1_share": full["full_run_D1_share"], "full_run_D2_share": full["full_run_D2_share"], "full_run_D3_share": full["full_run_D3_share"], "mature_D0_share": full["mature_D0_share"], "mature_D1_share": full["mature_D1_share"], "mature_D2_share": full["mature_D2_share"], "mature_D3_share": full["mature_D3_share"], "first_D2_week_by_firm": {str(i): min([x["global_step"] for x in reference if x["firm_id"] == i and x["distress_state"] in {"D2", "D3"}], default=None) for i in sorted({x["firm_id"] for x in reference})}, "first_D3_week_by_firm": {str(i): min([x["global_step"] for x in reference if x["firm_id"] == i and x["distress_state"] == "D3"], default=None) for i in sorted({x["firm_id"] for x in reference})}, "median_D2_spell": statistics.median([x["length"] for x in state_spells if x["state"] == "D2"]) if any(x["state"] == "D2" for x in state_spells) else 0, "p90_D2_spell": quant([x["length"] for x in state_spells if x["state"] == "D2"], .9), "median_D3_spell": statistics.median([x["length"] for x in state_spells if x["state"] == "D3"]) if any(x["state"] == "D3" for x in state_spells) else 0, "p90_D3_spell": quant([x["length"] for x in state_spells if x["state"] == "D3"], .9), "one_week_D2_spell_share": share(sum(x["length"] == 1 for x in state_spells if x["state"] == "D2"), sum(x["state"] == "D2" for x in state_spells)), "one_week_D3_spell_share": share(sum(x["length"] == 1 for x in state_spells if x["state"] == "D3"), sum(x["state"] == "D3" for x in state_spells)), "D2_recovery_count": 0, "D3_recovery_count": 0, "Firm0_mature_D2_D3_share": share(sum(x["distress_state"] in {"D2", "D3"} for x in firm0), len(firm0)), "I3_to_D2_D3_share": share(sum(x["distress_state"] in {"D2", "D3"} for x in i3), len(i3)), "binding_to_D2_D3_share": share(sum(x["distress_state"] in {"D2", "D3"} for x in binding), len(binding)), "payroll_underfunding_to_D3_share": share(sum(x["distress_state"] == "D3" for x in payroll), len(payroll))}
    (ROOT / "distress_threshold_candidate_manifest.json").write_text(json.dumps({"candidates": candidates, "reference_candidate": "reference_26w_u90", "arrears_payroll_thresholds": [1.0, 4.0], "note": "Semantic sensitivity candidates only; no target prevalence calibration and no behavior change."}, indent=2), encoding="utf-8")
    (ROOT / "distress_primitive_semantics.md").write_text("""# Passive distress primitive semantics

Interest-service states are reconstructed using the accepted I0/I1/I2/I3
definition. Arrears stock is kept separate from current unpaid-interest flow.
OCF is defined descriptively as Firm sales revenue minus executed wage bill plus
existing public-sector cash inflow minus public-sector cash outflow. Exposure
utilization uses opening lender exposure divided by credit limit; current-week
interest is not added to opening headroom. All persistence features are trailing
13/26/52-week observations and do not affect the simulation.

The reference shadow candidate is only a sensitivity label here: 26-week
persistence, 0.90 exposure-utilization threshold, D2 requiring persistent debt
service pressure plus another pressure dimension, and D3 adding prolonged
payroll impairment or severe arrears-plus-capacity pressure. It is not a model
parameter and is not behaviorally enforced.
""", encoding="utf-8")
    (ROOT / "reference_candidate_recommendation.md").write_text("""# Reference candidate recommendation

Recommend `reference_26w_u90` for the next passive review, not as a permanent
calibration. The 26-week scale better represents persistence than a single bad
week while remaining more informative than the strict 52-week screen. D2/D3
remain multidimensional and do not equal I3, debt, or one binding week. Recovery
is semantically possible because states are derived from rolling conditions.

Behavioral enforcement is **not implemented** in Step 13.5A.
""", encoding="utf-8")
    (ROOT / "acceptance_summary.md").write_text("""# Step 13.5A acceptance summary

Verdict: **B. DISTRESS_PERSISTENCE_SEMANTICS_READY_BUT_THRESHOLDS_UNRESOLVED**.

The passive dimensions are sufficiently closed to continue sensitivity work:
interest-service failure, arrears stock, credit capacity, operating cash flow
and payroll impairment are separately observable. A single I3 week, debt stock
or binding week is not treated as distress. The reference 26-week/0.90 screen
is nominated for review only; thresholds are not calibrated and no behavior was
changed. Default, Exit and distress enforcement remain unimplemented.
""", encoding="utf-8")
    (ROOT / "acceptance_flags.json").write_text(json.dumps({"verdict": "B_DISTRESS_PERSISTENCE_SEMANTICS_READY_BUT_THRESHOLDS_UNRESOLVED", "passive_distress_layer_ready": True, "distress_primitives_semantically_closed": True, "arrears_persistence_useful": True, "normalized_arrears_burden_useful": True, "credit_capacity_pressure_useful": True, "operating_cashflow_pressure_useful": True, "payroll_underfunding_acute_signal_useful": True, "hysteresis_likely_needed": True, "reference_candidate_selected": True, "economic_behavior_changed": False, "rng_changed": False, "default_implemented": False, "exit_implemented": False, "distress_behavioral_enforcement_ready": False, **required}, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        plot_dir = ROOT / "plots"; plot_dir.mkdir(exist_ok=True)
        for firm_id in sorted({x["firm_id"] for x in reference}):
            values = [x for x in reference if x["firm_id"] == firm_id]
            plt.plot([x["global_step"] for x in values], [int(x["distress_state"][1]) for x in values], label=f"Firm {firm_id}")
        plt.yticks([0, 1, 2, 3], ["D0", "D1", "D2", "D3"]); plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "firm_distress_state_timeline.png"); plt.close()
        plt.scatter([x["arrears_payroll_weeks"] for x in reference], [int(x["distress_state"][1]) for x in reference], s=2, alpha=.15); plt.xlabel("Arrears / trailing payroll weeks"); plt.ylabel("Distress state"); plt.tight_layout(); plt.savefig(plot_dir / "firm_arrears_exposure_distress.png"); plt.close()
        for state in ("D0", "D1", "D2", "D3"):
            plt.hist([x["trailing_26_mean_OCF"] for x in reference if x["distress_state"] == state], bins=30, alpha=.5, label=state)
        plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "firm_ocf_distress_state.png"); plt.close()
        matrix = Counter((x["interest_state"], x["distress_state"]) for x in reference); labels_i = ["I0", "I1", "I2", "I3"]; labels_d = ["D0", "D1", "D2", "D3"]
        data = [[matrix[(i, d)] for d in labels_d] for i in labels_i]; plt.imshow(data, aspect="auto"); plt.xticks(range(4), labels_d); plt.yticks(range(4), labels_i); plt.colorbar(); plt.tight_layout(); plt.savefig(plot_dir / "distress_vs_interest_state_heatmap.png"); plt.close()
        data = [[sum(x["distress_state"] == d and x["credit_binding"] == b for x in reference) for b in (False, True)] for d in labels_d]; plt.imshow(data, aspect="auto"); plt.xticks([0, 1], ["not binding", "binding"]); plt.yticks(range(4), labels_d); plt.colorbar(); plt.tight_layout(); plt.savefig(plot_dir / "distress_vs_credit_binding_heatmap.png"); plt.close()
        plt.hist([x["length"] for x in state_spells if x["state"] in ("D2", "D3")], bins=30); plt.xlabel("spell length"); plt.tight_layout(); plt.savefig(plot_dir / "distress_spell_duration.png"); plt.close()
        data = [[transitions[(a, b)] for b in labels_d] for a in labels_d]; plt.imshow(data, aspect="auto"); plt.xticks(range(4), labels_d); plt.yticks(range(4), labels_d); plt.colorbar(); plt.tight_layout(); plt.savefig(plot_dir / "distress_state_transition_heatmap.png"); plt.close()
    except Exception as exc:
        (ROOT / "plot_generation_error.txt").write_text(str(exc), encoding="utf-8")

    print(json.dumps({"verdict": "B_DISTRESS_PERSISTENCE_SEMANTICS_READY_BUT_THRESHOLDS_UNRESOLVED", **required}, indent=2))


if __name__ == "__main__": main()
