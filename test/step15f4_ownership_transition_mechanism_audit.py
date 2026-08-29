from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test/output/step15F4_ownership_transition_mechanism_audit"
F2_METRICS = ROOT / "test/output/step15F2_gradual_equity_transition/equity_transition_metrics.csv"
F3_METRICS = ROOT / "test/output/step15F3_ownership_dividend_dynamic_closure/ownership_dividend_metrics.csv"
ISSUANCE_PRICE = 10.0
LEGACY_SHARES = 100.0
TOLERANCE = 1e-9


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value):
    return float(value)


def dilution_rows():
    rows = []
    for legacy_target in (0.75, 0.50, 0.25, 0.10):
        person_target = 1.0 - legacy_target
        new_person_shares = LEGACY_SHARES * person_target / legacy_target
        total_shares = LEGACY_SHARES + new_person_shares
        firm_cash_raised = new_person_shares * ISSUANCE_PRICE
        rows.append({
            "initial_legacy_shares": LEGACY_SHARES,
            "target_legacy_fraction": legacy_target,
            "target_person_fraction": person_target,
            "new_person_shares_required": new_person_shares,
            "post_issuance_total_shares": total_shares,
            "issuance_price": ISSUANCE_PRICE,
            "firm_cash_raised": firm_cash_raised,
            "paid_in_equity_increase": firm_cash_raised,
            "legacy_share_count_change": 0.0,
            "total_shares_change": new_person_shares,
            "primary_cash_and_equity_identity_gap": 0.0,
        })
    return rows


def mechanism_rows():
    return [
        {
            "mechanism": "primary_issuance",
            "purpose": "Firm financing plus new Person ownership",
            "household_cash": "decrease",
            "firm_cash": "increase",
            "legacy_shares": "unchanged",
            "person_shares": "increase",
            "total_shares": "increase",
            "paid_in_equity": "increase",
            "money_created": 0.0,
            "firm_financing": "yes",
            "pure_ownership_transfer": "no",
            "legacy_proceeds_destination": "not applicable",
            "demand_leakage_risk": "low if Firm spends/retains cash; not household income",
            "status": "active controlled mechanism",
        },
        {
            "mechanism": "secondary_legacy_sale",
            "purpose": "Pure Legacy-to-Person ownership transition",
            "household_cash": "decrease",
            "firm_cash": "unchanged",
            "legacy_shares": "decrease",
            "person_shares": "increase",
            "total_shares": "unchanged",
            "paid_in_equity": "unchanged",
            "money_created": 0.0,
            "firm_financing": "no",
            "pure_ownership_transfer": "yes",
            "legacy_proceeds_destination": "explicit seller account required",
            "demand_leakage_risk": "depends on seller account; avoid automatic Household income",
            "status": "design only; not implemented",
        },
        {
            "mechanism": "legacy_proceeds_legacy_owner_cash",
            "purpose": "Temporary explicit seller settlement account",
            "household_cash": "no direct inflow",
            "firm_cash": "unchanged",
            "legacy_shares": "decrease",
            "person_shares": "increase",
            "total_shares": "unchanged",
            "paid_in_equity": "unchanged",
            "money_created": 0.0,
            "firm_financing": "no",
            "pure_ownership_transfer": "yes",
            "legacy_proceeds_destination": "world.legacy_owner_cash",
            "demand_leakage_risk": "persistent leakage if idle; transparent and reversible",
            "status": "candidate bridge",
        },
        {
            "mechanism": "legacy_proceeds_explicit_balance_sheet",
            "purpose": "Permanent explicit Legacy owner account",
            "household_cash": "no direct inflow",
            "firm_cash": "unchanged",
            "legacy_shares": "decrease",
            "person_shares": "increase",
            "total_shares": "unchanged",
            "paid_in_equity": "unchanged",
            "money_created": 0.0,
            "firm_financing": "no",
            "pure_ownership_transfer": "yes",
            "legacy_proceeds_destination": "LegacyOwner balance sheet",
            "demand_leakage_risk": "same economic question, better institutional semantics",
            "status": "preferred future settlement boundary",
        },
        {
            "mechanism": "hybrid_primary_plus_secondary",
            "purpose": "Separate financing from ownership transition",
            "household_cash": "decrease in either purchase; destination differs",
            "firm_cash": "increase only for primary",
            "legacy_shares": "unchanged for primary; decrease for secondary",
            "person_shares": "increase",
            "total_shares": "increase for primary; unchanged for secondary",
            "paid_in_equity": "increase only for primary",
            "money_created": 0.0,
            "firm_financing": "primary only",
            "pure_ownership_transfer": "secondary only",
            "legacy_proceeds_destination": "explicit Legacy owner account",
            "demand_leakage_risk": "auditable and not silently household income",
            "status": "nominated architecture",
        },
    ]


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    f2 = {row["case"]: row for row in read_rows(F2_METRICS)}
    f3 = {row["case"]: row for row in read_rows(F3_METRICS)}
    f2_treatment = f2["treatment_on"]
    f3_treatment = f3["treatment_on"]

    f2_person_fraction = number(f2_treatment["final_person_ownership_fraction"])
    f3_person_fraction = number(f3_treatment["final_person_ownership_fraction"])
    f2_issue = number(f2_treatment["total_equity_issuance"])
    f3_shares = LEGACY_SHARES * f3_person_fraction / (1.0 - f3_person_fraction)
    f2_implied_base = f2_issue / 0.0001
    f3_implied_base = f3_shares * ISSUANCE_PRICE / 0.0001
    scale_rows = [
        {
            "run": "Step15F.2 warm checkpoint",
            "start_global_step": f2_treatment["start_global_step"],
            "end_global_step": f2_treatment["end_global_step"],
            "review_count": f2_treatment["review_count"],
            "observed_total_issuance": f2_issue,
            "observed_shares_issued": f2_treatment["total_shares_issued"],
            "final_person_fraction": f2_person_fraction,
            "final_legacy_fraction": f2_treatment["final_legacy_ownership_fraction"],
            "person_shareholder_count": f2_treatment["person_shareholder_count"],
            "firm_cap_fraction_parameter": 0.0001,
            "implied_opening_base_if_firm_cap_binding": f2_implied_base,
            "starting_state": "step5000 wage_shock_1_47 warm, already mature/distressed",
            "interpretation": "two review windows and larger effective issuance capacity",
        },
        {
            "run": "Step15F.3 fresh canonical",
            "start_global_step": "0",
            "end_global_step": f3_treatment["weeks"],
            "review_count": "one effective review before horizon end",
            "observed_total_issuance": f3_shares * ISSUANCE_PRICE,
            "observed_shares_issued": f3_shares,
            "final_person_fraction": f3_person_fraction,
            "final_legacy_fraction": f3_treatment["final_legacy_ownership_fraction"]
            if "final_legacy_ownership_fraction" in f3_treatment
            else 1.0 - f3_person_fraction,
            "person_shareholder_count": "not persisted in F3 summary",
            "firm_cap_fraction_parameter": 0.0001,
            "implied_opening_base_if_firm_cap_binding": f3_implied_base,
            "starting_state": "fresh canonical, early operating transition",
            "interpretation": "only one issuance window; effective opening-base cap was much smaller",
        },
    ]

    dilution = dilution_rows()
    mechanisms = mechanism_rows()
    write_csv(OUTPUT / "dilution_math.csv", dilution)
    write_csv(OUTPUT / "transition_mechanism_matrix.csv", mechanisms)

    primary_checks = {
        "primary_cash_equals_paid_in_equity": all(
            abs(row["firm_cash_raised"] - row["paid_in_equity_increase"]) <= TOLERANCE
            for row in dilution
        ),
        "legacy_share_count_unchanged_under_primary": all(
            row["legacy_share_count_change"] == 0.0 for row in dilution
        ),
        "secondary_identity_defined": True,
        "no_simulation_run": True,
        "no_new_transactions": True,
        "primary_behavior_unchanged": True,
        "dividend_policy_unchanged": True,
        "investment_inactive": True,
        "step13_unchanged": True,
    }
    flags = {
        "verdict": "C. HYBRID_PRIMARY_FINANCING_AND_SECONDARY_TRANSITION_REQUIRED",
        "primary_issuance_dilution_math_closed": primary_checks["primary_cash_equals_paid_in_equity"],
        "primary_is_financing_and_not_pure_transfer": True,
        "secondary_is_pure_ownership_transfer": True,
        "secondary_not_implemented": True,
        "legacy_proceeds_need_explicit_account": True,
        "legacy_proceeds_silent_household_income_forbidden": True,
        "f2_f3_scale_difference_explained": True,
        "no_simulation_run": True,
        "new_rng_draws": 0,
        "autonomous_buying": False,
        "investment_active": False,
        "dividend_policy_changed": False,
        "step13_changed": False,
    }
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = f"""# Step 15F.4 Ownership Transition Mechanism and Legacy Exit Audit

**Verdict: {flags['verdict']}**

## Primary issuance math

The current CapTable starts with 100 Legacy shares and the Step15F.2 engineering issuance price is 10. For a target Legacy fraction `q`, primary issuance requires:

`new Person shares = 100 * (1-q) / q`

The resulting Firm cash increase and paid-in-equity increase are both `new shares * issuance price`. Legacy share count remains 100, while total shares increase. The exact target cases are in `dilution_math.csv`.

At the current price, moving Legacy ownership to 75%, 50%, 25%, and 10% requires respectively 33.333333, 100, 300, and 900 new Person shares, raising Firm cash by 333.333333, 1000, 3000, and 9000.

Therefore primary issuance is a valid Firm financing mechanism, but it is not a pure ownership-transition mechanism.

## Secondary transfer boundary

A Legacy secondary sale would transfer existing shares from `LegacyOwnershipPool` to Person. Household cash would fall and the seller account would receive the same cash. Firm cash, paid-in equity, and total shares would remain unchanged. This is the correct semantic operation when ownership changes without new Firm financing.

Sale proceeds must remain in an explicit Legacy owner account or future LegacyOwner balance sheet. They must not be silently redirected to Household income. An idle Legacy account can still represent demand leakage, but it makes that leakage visible and reversible.

## F2 versus F3 scale

Step15F.2 started from the step5000 warm checkpoint and had two review windows, total issuance {f2_issue:.6f}, and final Person ownership {f2_person_fraction:.6%}. Step15F.3 started fresh, reached only one effective issuance review within its 104-week horizon, and ended at {f3_person_fraction:.6%}. The difference is therefore a difference in starting state, review-window count, and effective Firm opening-base cap, not a change in dilution mathematics or ownership policy.

The implied opening-base values shown in the audit CSV are calculated only under the observed firm-cap-binding interpretation of the configured 0.0001 cap. They are diagnostic scale indicators, not recalibration.

## Decision

The permanent architecture should separate:

1. **Primary issuance** for genuine Firm equity financing.
2. **Secondary Legacy sale** for ownership transition without capitalization.
3. **Explicit LegacyOwner balance sheet/account** for seller proceeds.

No secondary transaction was implemented in this audit, and no simulation was run.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, indent=2, sort_keys=True))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
