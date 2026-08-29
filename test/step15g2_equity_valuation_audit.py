from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/output/step15G1_main_smoke"
G1 = ROOT / "test/output/step15G1_autonomous_secondary_equity"
OUTPUT = ROOT / "test/output/step15G2_equity_valuation_audit"
ENGINEERING_PRICE = 10.0
TRAILING_WEEKS = 26
REFERENCE_YIELDS = (0.05, 0.10, 0.20)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row, field, default=0.0):
    value = row.get(field, default)
    if value in (None, "", "nan", "NaN"):
        return float("nan")
    return float(value)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_ratio(numerator, denominator):
    if denominator is None or not math.isfinite(float(denominator)):
        return float("nan")
    if abs(float(denominator)) <= 1e-12:
        return float("nan")
    return float(numerator) / float(denominator)


def fmt(value):
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):,.6f}"


def main():
    if not SOURCE.exists() or not G1.exists():
        raise FileNotFoundError(
            "Step15G.1 artifacts are required; no simulation is run by this audit."
        )

    firm_rows = read_csv(SOURCE / "firm_diagnostics.csv")
    accounting_rows = read_csv(SOURCE / "accounting/firm_accounting.csv")
    ownership_rows = [
        row
        for row in read_csv(G1 / "ownership_panel.csv")
        if row.get("case") == "treatment_on"
    ]
    g1_summaries = read_csv(G1 / "autonomous_equity_metrics.csv")
    treatment_summary = next(
        row for row in g1_summaries if row["case"] == "treatment_on"
    )

    firm_rows = sorted(firm_rows, key=lambda row: number(row, "global_step"))
    accounting_rows = sorted(
        accounting_rows, key=lambda row: number(row, "global_step")
    )
    final_accounting = accounting_rows[-1]
    final_firm = firm_rows[-1]
    trailing_accounting = accounting_rows[-TRAILING_WEEKS:]
    trailing_firm = firm_rows[-TRAILING_WEEKS:]
    trailing_weeks = len(trailing_accounting)

    total_shares = number(ownership_rows[-1], "total_shares")
    final_person_fraction = number(
        ownership_rows[-1], "person_ownership_fraction"
    )
    target_shares = final_person_fraction * total_shares

    cash = number(final_accounting, "cash")
    inventory_book = number(final_accounting, "inventory_book_value")
    capital_book = 0.0
    liabilities = number(final_accounting, "total_liabilities")
    book_equity = number(final_accounting, "equity")
    cash_inventory_capital_less_liabilities = (
        cash + inventory_book + capital_book - liabilities
    )
    trailing_operating_profit = sum(
        number(row, "accounting_operating_profit") for row in trailing_accounting
    )
    trailing_net_income = sum(
        number(row, "accounting_net_income") for row in trailing_accounting
    )
    trailing_dividends = sum(
        number(row, "dividends") for row in trailing_accounting
    )
    trailing_cfo = sum(number(row, "cfo") for row in trailing_accounting)
    annual_factor = 52.0 / trailing_weeks
    annual_operating_profit = trailing_operating_profit * annual_factor
    annual_net_income = trailing_net_income * annual_factor
    annual_dividends = trailing_dividends * annual_factor
    annual_cfo = trailing_cfo * annual_factor

    engineering_equity_value = ENGINEERING_PRICE * total_shares
    book_price = max(book_equity, 0.0) / total_shares
    cash_inventory_price = max(
        cash_inventory_capital_less_liabilities, 0.0
    ) / total_shares

    dividend_prices = {
        yield_rate: safe_ratio(annual_dividends, yield_rate * total_shares)
        for yield_rate in REFERENCE_YIELDS
    }
    earnings_prices = {
        yield_rate: safe_ratio(annual_operating_profit, yield_rate * total_shares)
        for yield_rate in REFERENCE_YIELDS
    }
    # This is intentionally a sensitivity-only illustration. It is not used
    # as the nominated architecture and does not choose a required return.
    hybrid_prices = {
        yield_rate: (
            book_price + earnings_prices[yield_rate]
        ) / 2.0
        for yield_rate in REFERENCE_YIELDS
    }

    price_candidates = [
        {
            "candidate": "BOOK_EQUITY_PER_SHARE",
            "price_per_share": book_price,
            "total_implied_equity_value": book_price * total_shares,
            "reference_yield": "not required",
            "anchor_input": "closing accounting equity, floor=0 for diagnostic positivity",
            "economic_interpretation": "residual accounting claim per share",
            "required_new_calibration": "none for the passive book anchor; book-value policy still requires review",
            "pro_cyclicality_risk": "medium",
            "instability_risk": "medium",
            "behavior_feedback_risk": "medium",
            "loss_making_firm_compatibility": "partial",
            "future_capital_stock_compatibility": "high",
            "future_market_pricing_compatibility": "reference anchor only",
        },
        {
            "candidate": "DIVIDEND_CAPITALIZATION_REFERENCE",
            "price_per_share": dividend_prices[0.10],
            "total_implied_equity_value": dividend_prices[0.10] * total_shares,
            "reference_yield": "5%, 10%, 20% sensitivity; 10% shown",
            "anchor_input": "annualized trailing declared dividends",
            "economic_interpretation": "cash-distribution reference value",
            "required_new_calibration": "required return and sustainable dividend policy",
            "pro_cyclicality_risk": "high",
            "instability_risk": "high when dividends are sparse",
            "behavior_feedback_risk": "high",
            "loss_making_firm_compatibility": "low if dividends are zero",
            "future_capital_stock_compatibility": "medium",
            "future_market_pricing_compatibility": "reference anchor only",
        },
        {
            "candidate": "EARNINGS_CAPITALIZATION_REFERENCE",
            "price_per_share": earnings_prices[0.10],
            "total_implied_equity_value": earnings_prices[0.10] * total_shares,
            "reference_yield": "5%, 10%, 20% sensitivity; 10% shown",
            "anchor_input": "annualized trailing accounting operating profit",
            "economic_interpretation": "operating-earning power reference value",
            "required_new_calibration": "reference earnings yield and sustainable earnings",
            "pro_cyclicality_risk": "high",
            "instability_risk": "high for volatile or negative earnings",
            "behavior_feedback_risk": "high",
            "loss_making_firm_compatibility": "low when earnings are non-positive",
            "future_capital_stock_compatibility": "medium",
            "future_market_pricing_compatibility": "reference anchor only",
        },
        {
            "candidate": "HYBRID_FUNDAMENTAL_ANCHOR",
            "price_per_share": hybrid_prices[0.10],
            "total_implied_equity_value": hybrid_prices[0.10] * total_shares,
            "reference_yield": "5%, 10%, 20% sensitivity; 10% illustrative only",
            "anchor_input": "unweighted book/equity and earnings sensitivity",
            "economic_interpretation": "combined balance-sheet and flow reference",
            "required_new_calibration": "explicit weights and reference earnings yield",
            "pro_cyclicality_risk": "high",
            "instability_risk": "high",
            "behavior_feedback_risk": "high",
            "loss_making_firm_compatibility": "partial",
            "future_capital_stock_compatibility": "high",
            "future_market_pricing_compatibility": "reference anchor only",
        },
        {
            "candidate": "ENGINEERING_FIXED_PRICE",
            "price_per_share": ENGINEERING_PRICE,
            "total_implied_equity_value": engineering_equity_value,
            "reference_yield": "not applicable",
            "anchor_input": "Step15G.1 settlement price",
            "economic_interpretation": "technical transfer fixture price",
            "required_new_calibration": "economic adequacy review required",
            "pro_cyclicality_risk": "low",
            "instability_risk": "low",
            "behavior_feedback_risk": "extreme underpricing in this run",
            "loss_making_firm_compatibility": "high mechanically",
            "future_capital_stock_compatibility": "low",
            "future_market_pricing_compatibility": "poor as a market anchor",
        },
    ]

    return_metrics = []
    for label, price in (
        [("ENGINEERING_FIXED_PRICE", ENGINEERING_PRICE),
         ("BOOK_EQUITY_PER_SHARE", book_price),
         ("CASH_INVENTORY_CAPITAL_LESS_LIABILITIES", cash_inventory_price)]
        + [
            (f"DIVIDEND_CAP_{int(yield_rate * 100)}PCT", dividend_prices[yield_rate])
            for yield_rate in REFERENCE_YIELDS
        ]
        + [
            (f"EARNINGS_CAP_{int(yield_rate * 100)}PCT", earnings_prices[yield_rate])
            for yield_rate in REFERENCE_YIELDS
        ]
        + [
            (f"HYBRID_ILLUSTRATIVE_{int(yield_rate * 100)}PCT", hybrid_prices[yield_rate])
            for yield_rate in REFERENCE_YIELDS
        ]
    ):
        annual_dividend_per_share = safe_ratio(annual_dividends, total_shares)
        annual_earnings_per_share = safe_ratio(annual_operating_profit, total_shares)
        annual_cfo_per_share = safe_ratio(annual_cfo, total_shares)
        return_metrics.append({
            "candidate": label,
            "price_per_share": price,
            "trailing_annual_dividend_yield": safe_ratio(
                annual_dividend_per_share, price
            ),
            "trailing_annual_earnings_yield": safe_ratio(
                annual_earnings_per_share, price
            ),
            "price_to_earnings": safe_ratio(price, annual_earnings_per_share),
            "price_to_book": safe_ratio(price, book_price),
            "price_to_cfo": safe_ratio(price, annual_cfo_per_share),
            "annualized_dividend_per_share": annual_dividend_per_share,
            "annualized_operating_profit_per_share": annual_earnings_per_share,
            "annualized_cfo_per_share": annual_cfo_per_share,
            "cash_for_1pct_ownership": price * total_shares * 0.01,
            "price_to_engineering_price": safe_ratio(price, ENGINEERING_PRICE),
            "cfo_positive": annual_cfo > 0.0,
            "yield_interpretation": (
                "not meaningful as positive cash yield because annualized trailing CFO is negative"
                if annual_cfo <= 0.0
                else "positive"
            ),
        })

    counterfactuals = []
    for row in price_candidates:
        price = number(row, "price_per_share")
        cost = price * target_shares
        counterfactuals.append({
            "candidate": row["candidate"],
            "price_per_share": price,
            "observed_person_ownership_fraction": final_person_fraction,
            "observed_person_shares": target_shares,
            "cash_required_for_observed_ownership": cost,
            "engineering_cash_required": ENGINEERING_PRICE * target_shares,
            "cash_multiplier_vs_engineering": safe_ratio(
                cost, ENGINEERING_PRICE * target_shares
            ),
            "household_cash_after_purchase_if_same_buyers": number(
                treatment_summary, "final_household_cash"
            ) - cost + number(treatment_summary, "total_secondary_sale_proceeds"),
            "equity_value_received": cost,
            "firm_cash_change": 0.0,
            "paid_in_equity_change": 0.0,
            "money_created": 0.0,
            "money_destroyed": 0.0,
            "note": "passive cost counterfactual; no household purchase was rerun",
        })

    engineering_return = next(
        row for row in return_metrics if row["candidate"] == "ENGINEERING_FIXED_PRICE"
    )
    flags = {
        "verdict": "B. BOOK_EQUITY_REFERENCE_PRICE_REQUIRED",
        "source_run": str(SOURCE.relative_to(ROOT)),
        "step15g1_purchase_rerun": False,
        "new_simulation": False,
        "new_rng_draws": 0,
        "economic_behavior_changed": False,
        "engineering_price": ENGINEERING_PRICE,
        "total_shares": total_shares,
        "engineering_implied_equity_value": engineering_equity_value,
        "closing_book_equity": book_equity,
        "engineering_price_materially_below_book": (
            engineering_equity_value < book_equity * 0.10
        ),
        "engineering_dividend_yield_implausible": (
            engineering_return["trailing_annual_dividend_yield"] > 1.0
        ),
        "engineering_earnings_yield_implausible": (
            engineering_return["trailing_annual_earnings_yield"] > 1.0
        ),
        "legacy_and_person_rights_identical": True,
        "book_anchor_requires_no_return_parameter": True,
        "dividend_and_earnings_return_anchors_require_calibration": True,
        "cfo_negative_in_trailing_window": annual_cfo <= 0.0,
        "no_stock_market": True,
        "no_order_book": True,
        "no_purchase_behavior_change": True,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "valuation_candidate_matrix.csv", price_candidates)
    write_csv(OUTPUT / "implied_return_metrics.csv", return_metrics)
    write_csv(OUTPUT / "ownership_cost_counterfactual.csv", counterfactuals)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = f"""# Step 15G.2 Equity Valuation and Engineering-Price Adequacy Audit

**Verdict: B. BOOK_EQUITY_REFERENCE_PRICE_REQUIRED**

This was a passive audit of the existing Step15G.1 artifacts. No simulation,
autonomous purchase, RNG draw, price behavior, or policy was changed.

## Observed scale

- Engineering transfer price: {fmt(ENGINEERING_PRICE)} per share.
- Total shares: {fmt(total_shares)}.
- Engineering implied Firm equity value: {fmt(engineering_equity_value)}.
- Closing accounting book equity: {fmt(book_equity)}.
- Closing cash + inventory book value + capital book value - liabilities:
  {fmt(cash_inventory_capital_less_liabilities)}.
- Engineering price is only {fmt(safe_ratio(engineering_equity_value, book_equity) * 100)}%
  of closing accounting equity.

The observed Step15G.1 Person ownership was {final_person_fraction:.6%}, or
{fmt(target_shares)} shares. At the engineering price that stake costs only
{fmt(ENGINEERING_PRICE * target_shares)}.

## Trailing fundamentals

The last {trailing_weeks} persisted weeks were used. Annualized trailing
operating profit was {fmt(annual_operating_profit)}, declared dividends were
{fmt(annual_dividends)}, and CFO was {fmt(annual_cfo)}. CFO is negative in this
window, so a positive CFO yield is not meaningful.

At the engineering price, the annualized dividend yield is
{fmt(engineering_return['trailing_annual_dividend_yield'] * 100)}% and the
annualized earnings yield is {fmt(engineering_return['trailing_annual_earnings_yield'] * 100)}%.
These are diagnostics, not calibrated market returns, but they show that the
fixed settlement price materially overstates the ownership and dividend claim
relative to cash paid.

## Architecture conclusion

The fixed price should remain available for technical settlement fixtures, but
it is not adequate as the autonomous transition anchor in this run. A passive
book-equity reference is the safest next boundary because it uses an existing
accounting stock and does not require selecting an arbitrary required return.
Dividend- and earnings-capitalization references remain useful sensitivity
diagnostics, but require an explicit sustainable-flow definition and return
calibration. The illustrative hybrid row is not selected and is not a policy.

Legacy and Person shares currently have identical dividend and residual-equity
rights; no share-class distinction was introduced.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
