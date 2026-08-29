"""Step 15.0 aggregate-demand closure architecture audit.

This script is deliberately non-simulating.  It records the architecture
comparison requested by Step 15.0 and cites the accepted Step 14E.3 result.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test/output/step15_0_aggregate_demand_closure_architecture"
STEP14E3_OUTPUT = ROOT / "test/output/step14E3_dynamic_demand_closure_audit"

VERDICT = "E. COMBINED_PRODUCTION_INVESTMENT_ARCHITECTURE_REQUIRED"

MATRIX = [
    {
        "candidate_id": "1",
        "mechanism": "Firm investment demand",
        "demand_class": "Final demand when purchasing capital goods; intermediate if the purchase is an operating input",
        "spender": "Investing Firm, using retained cash, equity proceeds, or a loan",
        "receiver": "Capital-good Firm or supplying sector",
        "money_semantics": "Cash/equity finance is a transfer; loan finance creates money and Firm debt; principal repayment destroys money",
        "absorbs_household_saving": "Conditionally: directly through Person equity/bond ownership or financial intermediation, not through Firm retained cash alone",
        "independent_demand": "Yes, if a real capital good is produced and sold; no if the Firm merely relabels internal spending",
        "step13_finance_compatibility": "High, with loan proceeds tied to an identified investment transaction and debt service",
        "future_person_equity_compatibility": "High; equity issuance and dividends can later connect Firms and Persons",
        "capital_stock_required": "Yes",
        "multigood_multisector_support": "Required for capital goods or an explicit capital-accounting layer",
        "implementation_complexity": "High",
        "main_accounting_identities": "Buyer cash decrease = seller cash increase; capital asset increase = payment; loan asset/liability pair if credit-financed; investment is not household consumption",
        "double_counting_risk": "Count capital formation once; do not also count the same capital-good sale as intermediate consumption or household final demand",
        "free_money_risk": "Loan-created money must carry Firm debt and a real purchased asset; no grant-like cash injection",
        "main_risk": "Pseudo-investment, arbitrary capital productivity, or debt-financed demand without a capital asset",
        "readiness": "Design-ready, implementation-blocked on capital stock",
        "recommendation_rank": "1",
    },
    {
        "candidate_id": "2",
        "mechanism": "Intermediate-input demand",
        "demand_class": "Intermediate demand, not final demand",
        "spender": "Downstream Firm",
        "receiver": "Upstream Firm",
        "money_semantics": "Normally a monetary transfer; working-capital loan finance can create money and debt; repayment destroys principal money",
        "absorbs_household_saving": "Indirectly through production and income; not a direct counterpart to household saving unless financed through a financial claim",
        "independent_demand": "Yes as input demand, but it requires final demand, inventory accumulation, or investment to avoid merely moving the shortfall upstream",
        "step13_finance_compatibility": "High; supplier invoices and working-capital loans fit existing Firm-specific credit",
        "future_person_equity_compatibility": "Medium-high; ownership is compatible but not required for the transaction",
        "capital_stock_required": "No capital stock required initially; input inventories and production-use contracts are required",
        "multigood_multisector_support": "Required to identify inputs and prevent Food/Service transactions from being treated as one good",
        "implementation_complexity": "Medium-high",
        "main_accounting_identities": "Downstream input expense = upstream revenue; buyer cash decrease = seller cash increase; input inventory/use bridge must close",
        "double_counting_risk": "Do not add intermediate sales to final GDP demand on top of the final product; consolidate across Firms",
        "free_money_risk": "Supplier credit or loans must create a matching liability; unpaid invoices cannot become free income",
        "main_risk": "Circular input loops and larger gross sales without any additional final buyer",
        "readiness": "Architecture-ready after a distinct input/accounting contract; not sufficient as sole closure",
        "recommendation_rank": "2",
    },
    {
        "candidate_id": "3",
        "mechanism": "Government final demand",
        "demand_class": "Final demand",
        "spender": "Public Sector/Government",
        "receiver": "Firm supplying the procured good or service",
        "money_semantics": "Tax-funded spending transfers money; bond-funded spending transfers existing money; central-bank-funded spending creates money and a public/CB liability unless separately retired",
        "absorbs_household_saving": "Yes through taxes or government securities, but the financing path must be explicit; CB money creation is not absorption of saving",
        "independent_demand": "Yes",
        "step13_finance_compatibility": "Medium-high; public procurement can use current ledger, but fiscal funding and public liabilities are not yet a policy contract",
        "future_person_equity_compatibility": "Neutral-compatible; ownership is not required",
        "capital_stock_required": "No for current procurement; yes only for public capital projects",
        "multigood_multisector_support": "Useful and eventually required to target sectors without hiding Food inventory release semantics",
        "implementation_complexity": "Medium",
        "main_accounting_identities": "Public cash start + revenue - expenditure = public cash end; Firm revenue = public purchase payment; public debt or CB liability must fund deficits",
        "double_counting_risk": "Separate public procurement from later public inventory release and in-kind support; do not count both purchase and release as new final demand",
        "free_money_risk": "High if Government spending is added without tax, bond, or explicit CB balance-sheet counterpart",
        "main_risk": "Arbitrary fiscal demand can mask weak private closure and become a hidden subsidy",
        "readiness": "Not ready as a neutral closure rule; current public Food operation is limited and strategic",
        "recommendation_rank": "3",
    },
    {
        "candidate_id": "4",
        "mechanism": "External demand",
        "demand_class": "Final demand (exports)",
        "spender": "Foreign household, Firm, or Government",
        "receiver": "Domestic exporting Firm",
        "money_semantics": "Foreign payment enters through an external/FX settlement; domestic money creation depends on the monetary regime and must be paired with a foreign-asset or external-liability entry",
        "absorbs_household_saving": "Yes in aggregate through a trade surplus/current-account counterpart",
        "independent_demand": "Yes",
        "step13_finance_compatibility": "Low initially; requires open-economy settlement, foreign assets/liabilities, and import treatment",
        "future_person_equity_compatibility": "Compatible but not a prerequisite",
        "capital_stock_required": "No initially",
        "multigood_multisector_support": "Required to identify exportable goods/services and imports",
        "implementation_complexity": "High",
        "main_accounting_identities": "Exports - imports = current-account balance; domestic cash/FX asset change must equal external settlement; Firm revenue must match foreign payment",
        "double_counting_risk": "Do not count an export both as domestic final demand and external demand; imports are leakage, not domestic production",
        "free_money_risk": "High if foreign buyers appear without an external balance sheet or FX counterpart",
        "main_risk": "An arbitrary export demand shock can hide domestic demand weakness and create untracked foreign wealth",
        "readiness": "Not ready in the closed-economy architecture",
        "recommendation_rank": "4",
    },
]


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_step14e3_flags():
    path = STEP14E3_OUTPUT / "acceptance_flags.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    step14e3_flags = load_step14e3_flags()

    manifest = {
        "architecture_version": "step15.0",
        "verdict": VERDICT,
        "audit_only": True,
        "simulation_run": False,
        "economic_behavior_changed": False,
        "step13_parameters_changed": False,
        "capital_stock_implemented": False,
        "government_implemented": False,
        "external_sector_implemented": False,
        "ownership_implemented": False,
        "taxes_implemented": False,
        "candidate_count": len(MATRIX),
        "candidates": [row["mechanism"] for row in MATRIX],
        "selected_architectural_stage": "combined_production_investment_architecture",
        "first_concrete_substage": "firm_investment_demand_contract",
        "supporting_substage": "intermediate_input_contract_and_consolidated_accounting",
        "government_and_external_status": "deferred_until_funding_and_external_balance_sheets_exist",
        "source_artifacts": {
            "step14e3_summary": str(
                STEP14E3_OUTPUT / "acceptance_summary.md"
            ),
            "step14e3_flags": str(
                STEP14E3_OUTPUT / "acceptance_flags.json"
            ),
            "step14e3_metrics": str(
                STEP14E3_OUTPUT / "step14E3_closure_metrics.csv"
            ),
            "ledger": "economy/ledger.py",
            "accounting": "economy/accounting.py",
            "public_and_credit_runtime": "world.py",
        },
        "observed_step14e3_flags": step14e3_flags,
        "nomination_reason": (
            "No single candidate is behavior-ready. Firm investment is the safest first module, "
            "but it needs a real capital-good/capital-stock contract; intermediate demand must be "
            "designed alongside it and kept out of final-demand totals."
        ),
    }

    macro_closure = """# Current Macro Closure

The current model has the following sector identities:

`Household income = wages + dividends + transfers`

`Household income = household consumption + household saving`

Positive household saving is money/wealth retained by households and is therefore a leakage from current goods demand. The Step14E.3 audit found that Service demand is only a split of the existing Household consumption budget; it does not create a new purchasing-power source.

For Firms, the cash bridge is approximately:

`delta Firm cash = sales collections + public receipts + loan issued - wages - interest - principal repaid - dividends - other cash outflows`

A working-capital loan creates money and a matching Firm liability. It can finance payroll or a cash buffer, but loan creation alone is not final demand. If the Firm retains the proceeds, goods demand does not rise.

For the public/Central Bank side:

`delta public cash = public revenue + inventory-release income + other public receipts - public expenditure`

`delta credit-money liabilities = gross loans created - principal destroyed`

`Firm loan liabilities = Central Bank loan assets`

The existing public Food inventory operation is a bounded observed flow. It is not a general Government final-demand rule. Consequently, when households save and Firms do not make an independent investment or input purchase, there is no scalable counterpart expenditure. The result is falling sales, lower labor demand, lower wages, and a reinforcing fall in Household consumption.
"""

    candidate_table = "\n".join(
        f"| {row['candidate_id']} | {row['mechanism']} | {row['demand_class']} | {row['absorbs_household_saving']} | {row['capital_stock_required']} | {row['implementation_complexity']} | {row['readiness']} |"
        for row in MATRIX
    )
    summary = f"""# Step 15.0 Aggregate Demand Closure Architecture

## Verdict

**{VERDICT}**

This is an audit-only result. No simulation was run and no Government,
investment, capital stock, tax, export, ownership, or Step13 parameter was
implemented.

## Demand boundary

Final demand consists of Household consumption, Government purchases,
investment/capital formation, and exports. Intermediate inputs are purchases
between Firms. They generate gross sales and production requirements, but must
not be added again to final GDP demand after the final product is counted.

| candidate | mechanism | demand class | absorbs Household saving? | capital stock | complexity | status |
|---|---|---|---|---|---|---|
{candidate_table}

The full accounting and implementation comparison is in
`step15_candidate_matrix.csv`.

{macro_closure}

## Candidate conclusions

### Firm investment demand

This is the safest first genuine-demand module. A Firm would spend retained
cash, equity proceeds, or a loan on a real capital good supplied by another Firm
or sector. Cash finance is a transfer; loan finance creates money and debt;
principal repayment destroys money. It can later connect naturally to Person
equity ownership. It is not ready for behavioral implementation until capital
goods, capital stock, depreciation, and the investment-versus-intermediate
accounting boundary exist.

### Intermediate-input demand

This is the necessary production-network companion. It extends generalized
Firms and Step13 working-capital finance cleanly, but it is not final demand.
It can move demand upstream without closing the aggregate shortfall if no final
buyer or investment buyer exists. It must be consolidated to prevent double
counting.

### Government final demand

It is a genuine final-demand source, but the funding rule is decisive. Tax,
bond, and Central Bank financing have different money and balance-sheet
semantics. Adding procurement without an explicit funding counterpart would
turn the existing public Food operation into arbitrary free demand.

### External demand

Exports can provide an independent final-demand counterpart to domestic saving,
but only with an external balance sheet, FX/settlement convention, imports, and
current-account accounting. It is not compatible with the current closed
economy as an immediate next module.

## Nomination

The safest next architectural stage is a combined production-investment
architecture, with Firm investment as the first concrete substage and a
separate intermediate-input contract designed at the same boundary. This keeps
the first new demand inside the generalized Firm architecture, gives Step13
credit a real transaction to finance, preserves ledger conservation, and leaves
Government, external demand, and Person ownership as compatible later layers.

The nomination is architectural only. Step 15.0 stops before implementing any
of those mechanisms.
"""

    (OUTPUT / "architecture_summary.md").write_text(summary, encoding="utf-8")
    (OUTPUT / "architecture_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(OUTPUT / "step15_candidate_matrix.csv", MATRIX)
    print(VERDICT)
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
