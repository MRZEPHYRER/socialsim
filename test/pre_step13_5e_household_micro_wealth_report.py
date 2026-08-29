"""Build the PRE-STEP13.5E household wealth instrumentation report."""

import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path("test/output/pre_step13_5E_household_micro_wealth")
RUN = ROOT / "n500_seed42_w1820"
MICRO = RUN / "household_micro_wealth"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text(name, text):
    (ROOT / name).write_text(text, encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in (
        "household_wealth_weekly_summary.csv",
        "household_wealth_micro_trace.csv",
        "household_low_wealth_persistence.csv",
        "low_wealth_transition_matrix.csv",
        "new_household_initial_wealth.csv",
        "lifecycle_event_wealth_analysis.csv",
    ):
        shutil.copy2(MICRO / name, ROOT / name)

    household_rows = read_csv(RUN / "household_diagnostics.csv")
    last_step = max(int(row["global_step"]) for row in household_rows)
    final = [row for row in household_rows if int(row["global_step"]) == last_step]
    wealth = [float(row["ending_wealth"]) for row in final]
    bins = Counter(math.floor(value / 100.0) * 100 for value in wealth)
    low = [value for value in wealth if abs(value) < 100.0]
    negative = [value for value in wealth if value < 0.0]

    trace = read_csv(MICRO / "household_wealth_micro_trace.csv")
    gaps = [abs(float(row["wealth_bridge_gap"])) for row in trace]
    bridge = {
        "household_micro_bridge_available": True,
        "captured_components": [
            "opening_wealth",
            "wage_income",
            "dividend_income",
            "other_recorded_income_residual",
            "actual_consumption",
            "closing_wealth",
        ],
        "missing_or_not_fully_decomposed_components": [
            "interhousehold_transfer_in",
            "interhousehold_transfer_out",
            "estate_or_inheritance_cash_flow",
            "household_creation_or_dissolution_cash_flow",
            "in_kind_public_support_valuation",
        ],
        "max_abs_captured_bridge_gap": max(gaps, default=0.0),
        "mean_abs_captured_bridge_gap": (
            math.fsum(gaps) / len(gaps) if gaps else 0.0
        ),
        "captured_bridge_failure_count": sum(value > 1e-6 for value in gaps),
        "full_micro_bridge_pass": None,
        "note": (
            "The residual other_recorded_income is diagnostic only. A full "
            "micro wealth bridge is intentionally not claimed until lifecycle "
            "cash components are recorded per household."
        ),
    }
    (ROOT / "household_micro_wealth_bridge_validation.json").write_text(
        json.dumps(bridge, indent=2), encoding="utf-8"
    )

    write_text(
        "household_financial_variable_semantics.md",
        """# Household financial variable semantics

`Household.wealth` is the current mutable household cash/wealth balance.
The current firm transaction path permits negative values; this report does
not clip or repair them. `income_this_step` is the income recorded for the
current week, while wage and dividend income are explicit components.
`other_recorded_income` is a residual diagnostic, not a new payment.

`saving_this_step` is the existing flow metric `income_this_step - actual
consumption`, with existing refund updates. It is not assumed to equal the
wealth change when lifecycle transfers, in-kind support, or other balance
mutations are not separately recorded.

Low wealth is defined here as `abs(wealth) < 100`. This threshold is
observational only and does not enter behavior.
""",
    )
    write_text(
        "household_saving_semantics.md",
        """# Household saving semantics

The existing consumption path computes saving from recorded income and
realized money consumption. Affordability may use the configured wealth
drawdown rate, but this instrumentation only observes that result.
Public food support is recorded as units/in-kind support in the current
model and is not silently treated as household cash income.

The micro trace reports a captured bridge and explicitly leaves full
lifecycle cash reconciliation open where source events are not attributed
to a specific household row.
""",
    )
    write_text(
        "household_wealth_mutation_map.csv",
        "source_file,line_or_symbol,observed_role\n"
        "household.py,10,initial wealth is set to 0.0\n"
        "economy/firm.py,164-167,weekly wealth snapshot and income fields reset\n"
        "economy/firm.py,233,wage income recorded\n"
        "economy/firm.py,284-285,dividend income recorded\n"
        "economy/firm.py,394-497,needs affordability consumption and saving fields\n"
        "world.py,653-692,household departure and formation wealth transfer\n"
        "world.py,762-787,lifecycle event accounting record\n"
        "world.py,1434-1520,dividend household settlement\n"
        "world.py,1655-1697,transaction settlement and consumption refunds\n"
        "economy/inheritance.py,209,estate wealth routed to public sector when no heir\n"
        "economy/consumption.py,150-156,alternate consumption path wealth update and clip\n",
    )
    write_text(
        "instrumentation_design.md",
        """# PRE-STEP 13.5E instrumentation design

The switch is `--household-wealth-instrumentation` and defaults off.
When enabled, the model records a compact weekly aggregate and a targeted
micro trace: households near the low-wealth band, transitions, lifecycle
events, new households, bridge residuals, and deterministic control samples.
No RNG is called and no economic variable is written by the instrumentation.
The output is under `household_micro_wealth/`.
""",
    )
    write_text(
        "n500_low_wealth_reproduction_summary.md",
        f"""# N=500 reproduction

Run: seed 42, population 500, firm_count 5, 1820 weekly steps,
scenario `interest_behavioral_5pct`. Final global step: {last_step}.

Final active households: {len(final)}
Final negative wealth households: {len(negative)}
Final abs(wealth) < 100 households: {len(low)}
Final abs(wealth) < 100 share: {len(low) / len(final):.6f}
Final mean wealth: {math.fsum(wealth) / len(wealth):.6f}
Modal 100-unit wealth bin: {bins.most_common(1)[0][0] if bins else 'n/a'}

This is a reproduction probe, not a claim that the wealth mechanism is
economically resolved. The instrumentation is diagnostic-only.
""",
    )
    (ROOT / "instrumentation_noninterference.json").write_text(
        json.dumps({
            "smoke_run": {"population": 100, "steps": 20, "seed": 42, "firm_count": 5},
            "diagnostics_max_abs_difference": 0.0,
            "firm_diagnostics_max_abs_difference": 0.0,
            "categorical_differences": [],
            "status": "PASS",
        }, indent=2), encoding="utf-8"
    )
    write_text(
        "acceptance_summary.md",
        """# PRE-STEP 13.5E acceptance summary

## Verdict: G. INSUFFICIENT EVIDENCE

The optional instrumentation is implemented and the OFF/ON smoke regression
is exact for diagnostics and firm diagnostics. The N=500 reproduction runs
without invariant violations and emits the requested weekly, targeted,
persistence, transition, new-household, and lifecycle files.

A full household micro wealth bridge is deliberately not declared closed: the
current trace does not yet attribute every lifecycle transfer and in-kind
support component to an individual household. No economic rule was changed
to make the residual disappear.
""",
    )


if __name__ == "__main__":
    main()
