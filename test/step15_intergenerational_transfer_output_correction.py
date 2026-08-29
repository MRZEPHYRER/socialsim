"""Tighten labels in the already generated diagnostic-only audit output."""

from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test" / "output" / "step15_intergenerational_transfer_inequality_audit"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    registry_path = OUT / "intergenerational_transfer_contract_registry.csv"
    registry = read_rows(registry_path)
    for row in registry:
        if row.get("mechanism") == "parent_support" and row.get("observed_events") == "0":
            row["status"] = "INACTIVE_IN_BASELINE"
            row["notes"] = "rule exists in economy/income.py but no eligible support event was observed"
    write_rows(registry_path, registry)

    concentration_path = OUT / "family_support_concentration.csv"
    concentration = read_rows(concentration_path)
    for row in concentration:
        if row.get("recipient_quantile") == "UNAVAILABLE":
            row["total_support"] = "0.0"
            row["share_of_support"] = "0.0"
            row["notes"] = "household-formation balances are excluded from family-support concentration"
    write_rows(concentration_path, concentration)

    flags_path = OUT / "acceptance_flags.json"
    flags = json.loads(flags_path.read_text(encoding="utf-8"))
    flags["parent_support_rule_present_but_realized"] = False
    flags["cash_inheritance_realized"] = False
    flags["shareholder_estate_inheritance_realized"] = False
    flags["household_formation_is_not_family_support"] = True
    flags_path.write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = """# Step 15 Intergenerational Transfer / Inheritance Distributional Audit

Verdict: **F. CURRENT_DATA_OR_HORIZON_INSUFFICIENT**

## Scope

The audit used the accepted N=5000, seed=42, 520-week Step15 profile. No transfer, inheritance, retirement, wage, consumption, Government, or Household rule was changed. Parent support was reconstructed from the authoritative `IncomeSystem.distribute_income` inputs because the current runtime does not persist it as a separate cash-ledger event.

## Findings

- The code contains a conditional child-to-parent support rule, but zero eligible support events occurred in this canonical run.
- No cash inheritance event and no shareholder Estate-to-heir event occurred in this ownership-off baseline.
- Realized lifecycle cash flows were Household-formation settlement transfers and two no-heir public-wealth sweeps. Household formation is not counted as family support.
- The current data do not provide a complete event-level pre/post Household balance-sheet panel or authoritative parent-child rank history. Therefore Gini effects, rank persistence, multi-generation compounding, and causal near-zero exit effects are not identified by this run.
- Existing family support cannot be treated as a universal replacement for removed elderly labor income. In this run, observed support is zero, so it replaces none of the retirement wage loss.

## Policy boundary

Future pension or social-insurance design should complement family support rather than silently assume that family support is a complete replacement. A future institution should be evaluated separately for coverage, liquidity protection, and distributional incidence.
"""
    (OUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
