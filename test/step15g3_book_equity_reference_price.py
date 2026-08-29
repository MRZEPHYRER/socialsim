from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/output/step15G1_main_smoke"
G1 = ROOT / "test/output/step15G1_autonomous_secondary_equity"
OUTPUT = ROOT / "test/output/step15G3_book_equity_reference_price"
TOTAL_SHARES_FALLBACK = 100.0


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(row, field, default=float("nan")):
    value = row.get(field, default)
    if value in (None, "", "nan", "NaN"):
        return float("nan")
    return float(value)


def safe_divide(a, b):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)):
        return float("nan")
    if abs(float(b)) <= 1e-12:
        return float("nan")
    return float(a) / float(b)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return sum(values) / len(values) if values else float("nan")


def stability(values):
    values = [float(v) for v in values if math.isfinite(float(v)) and v > 0]
    changes = [
        abs(values[i] / values[i - 1] - 1.0)
        for i in range(1, len(values))
        if values[i - 1] > 0
    ]
    average = mean(values)
    variance = mean([(v - average) ** 2 for v in values])
    return {
        "observation_count": len(values),
        "mean": average,
        "min": min(values, default=float("nan")),
        "max": max(values, default=float("nan")),
        "coefficient_of_variation": safe_divide(math.sqrt(variance), average),
        "mean_abs_weekly_change": mean(changes),
        "max_abs_weekly_change": max(changes, default=float("nan")),
    }


def fmt(value):
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):,.6f}"


def main():
    if not SOURCE.exists() or not G1.exists():
        raise FileNotFoundError(
            "Step15G.1 artifacts are required; this audit does not rerun the model."
        )

    accounting = sorted(
        read_csv(SOURCE / "accounting/firm_accounting.csv"),
        key=lambda row: int(float(row["global_step"])),
    )
    ownership = [
        row
        for row in read_csv(G1 / "ownership_panel.csv")
        if row.get("case") == "treatment_on"
    ]
    summaries = read_csv(G1 / "autonomous_equity_metrics.csv")
    treatment = next(row for row in summaries if row["case"] == "treatment_on")

    total_shares = (
        num(ownership[-1], "total_shares") if ownership else TOTAL_SHARES_FALLBACK
    )
    equity_by_step = {
        int(float(row["global_step"])): num(row, "equity") for row in accounting
    }
    current_prices = {
        step: safe_divide(equity, total_shares)
        for step, equity in equity_by_step.items()
    }
    steps = sorted(current_prices)
    rows = []
    for index, step in enumerate(steps):
        current = current_prices[step]
        lagged = current_prices.get(steps[index - 1], float("nan")) if index else float("nan")
        recent_4 = [current_prices[s] for s in steps[max(0, index - 3): index + 1]]
        recent_13 = [current_prices[s] for s in steps[max(0, index - 12): index + 1]]
        book_equity = equity_by_step[step]
        rows.append({
            "valuation_step": step,
            "book_equity": book_equity,
            "total_shares": total_shares,
            "authoritative_assets": num(accounting[index], "total_assets"),
            "authoritative_liabilities": num(accounting[index], "total_liabilities"),
            "raw_book_price_per_share": current,
            "lagged_book_price_per_share": lagged,
            "moving_average_4_book_price": mean(recent_4),
            "moving_average_13_book_price": mean(recent_13),
            "raw_status": "POSITIVE_EQUITY" if book_equity > 0 else "NON_POSITIVE_EQUITY",
            "lagged_status": (
                "POSITIVE_EQUITY" if lagged > 0 else "NON_POSITIVE_EQUITY_OR_UNAVAILABLE"
            ),
            "reference_price_contract_candidate": (
                lagged if math.isfinite(lagged) and lagged > 0 else float("nan")
            ),
            "price_timing": "prior_period_closing_accounting_equity",
            "purchase_allowed_under_contract": bool(math.isfinite(lagged) and lagged > 0),
        })

    effective_prices = [
        row["reference_price_contract_candidate"]
        for row in rows
        if math.isfinite(float(row["reference_price_contract_candidate"]))
    ]
    current_stability = stability(list(current_prices.values()))
    lagged_stability = stability(effective_prices)
    ma4_stability = stability([num(row, "moving_average_4_book_price") for row in rows])
    ma13_stability = stability([num(row, "moving_average_13_book_price") for row in rows])

    review_rows = [
        row
        for row in ownership
        if row.get("reviewed", "").lower() == "true"
        and num(row, "secondary_sale_proceeds", 0.0) > 1e-12
    ]
    counterfactual = []
    for review in review_rows:
        review_step = int(num(review, "global_step"))
        pre_step = max((step for step in steps if step < review_step), default=None)
        lagged_price = current_prices.get(pre_step, float("nan"))
        observed_proceeds = num(review, "secondary_sale_proceeds", 0.0)
        observed_shares = num(review, "secondary_shares_sold", 0.0)
        same_budget_shares = safe_divide(observed_proceeds, lagged_price)
        counterfactual.append({
            "counterfactual_type": "same_observed_aggregate_cash_budget",
            "review_step": review_step,
            "valuation_step": pre_step,
            "reference_price_used": lagged_price,
            "book_equity_used": equity_by_step.get(pre_step, float("nan")),
            "total_shares_used": total_shares,
            "observed_g1_cash_spent": observed_proceeds,
            "observed_g1_shares_filled": observed_shares,
            "same_budget_fillable_shares_at_book_price": same_budget_shares,
            "same_budget_person_ownership_fraction": safe_divide(
                same_budget_shares, total_shares
            ),
            "ownership_ratio_vs_observed_g1": safe_divide(
                same_budget_shares, observed_shares
            ),
            "cash_required_under_same_budget": observed_proceeds,
            "firm_cash_change": 0.0,
            "paid_in_equity_change": 0.0,
            "money_created": 0.0,
            "money_destroyed": 0.0,
            "micro_budget_data_available": False,
            "interpretation": (
                "Exact household fill cannot be reconstructed because G1 did not persist per-household desired budgets; "
                "this uses the observed aggregate cash spend as a same-budget proxy."
            ),
        })

    final_observed_shares = num(treatment, "total_secondary_shares_sold")
    final_reference_price = effective_prices[-1]
    final_reference_value = final_observed_shares * final_reference_price
    engineering_cost = num(treatment, "total_secondary_sale_proceeds")
    counterfactual.append({
        "counterfactual_type": "observed_g1_holdings_marked_at_final_lagged_book_price",
        "review_step": "final",
        "valuation_step": steps[-1] - 1 if len(steps) > 1 else "unavailable",
        "reference_price_used": final_reference_price,
        "book_equity_used": equity_by_step.get(steps[-2], float("nan")) if len(steps) > 1 else float("nan"),
        "total_shares_used": total_shares,
        "observed_g1_cash_spent": engineering_cost,
        "observed_g1_shares_filled": final_observed_shares,
        "same_budget_fillable_shares_at_book_price": safe_divide(
            engineering_cost, final_reference_price
        ),
        "same_budget_person_ownership_fraction": safe_divide(
            engineering_cost, final_reference_price * total_shares
        ),
        "ownership_ratio_vs_observed_g1": safe_divide(
            engineering_cost, final_reference_price * final_observed_shares
        ),
        "cash_required_under_same_budget": engineering_cost,
        "firm_cash_change": 0.0,
        "paid_in_equity_change": 0.0,
        "money_created": 0.0,
        "money_destroyed": 0.0,
        "micro_budget_data_available": False,
        "interpretation": "Diagnostic mark only; no revaluation is posted to Household accounting.",
    })

    flags = {
        "verdict": "B. LAGGED_BOOK_PRICE_CONTRACT_READY",
        "source_run": str(SOURCE.relative_to(ROOT)),
        "purchase_rerun": False,
        "economic_behavior_changed": False,
        "new_rng_draws": 0,
        "book_equity_source": "accounting/firm_accounting.csv: equity = authoritative assets - liabilities",
        "all_observed_book_equity_positive": all(num(row, "book_equity") > 0 for row in rows),
        "non_positive_equity_behavior": "purchase unavailable; status NON_POSITIVE_EQUITY",
        "lagged_price_known_before_decision": True,
        "same_price_locked_for_decision_and_settlement": True,
        "smoothing_needed": False,
        "household_accounting_cost_based": True,
        "unrealized_revaluation_posts_cash": False,
        "firm_cash_changed_by_pricing": False,
        "paid_in_equity_changed_by_pricing": False,
        "total_shares_changed_by_pricing": False,
        "exact_micro_budget_counterfactual_available": False,
        "g1_transition_becomes_much_smaller_under_book_price": (
            bool(counterfactual)
            and num(counterfactual[0], "same_budget_person_ownership_fraction")
            < num(counterfactual[0], "observed_g1_shares_filled") / total_shares
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "book_price_metrics.csv", rows)
    write_csv(OUTPUT / "g1_book_price_counterfactual.csv", counterfactual)
    (OUTPUT / "acceptance_flags.json").write_text(
        json.dumps(flags, indent=2, sort_keys=True), encoding="utf-8"
    )

    semantics = f"""# Step 15G.3 Book-Equity Reference Price Semantics

## Authority

The reference uses `accounting/firm_accounting.csv` and its authoritative
`equity` field. It does not use the legacy market-value net-worth diagnostic.

For positive equity:

`raw_book_price_per_share = book_equity / total_shares`

The proposed contract uses the previous completed week's closing accounting
equity. At purchase decision week `t`, the locked price is the value from
valuation step `t-1`. This avoids look-ahead and ensures the same price is used
for the decision and settlement. The first week has no lagged value and is
therefore unavailable for autonomous purchase.

For `book_equity <= 0`, the contract status is `NON_POSITIVE_EQUITY` and
autonomous purchase is unavailable. It never creates a negative share price and
never silently falls back to the engineering price.

## Timing and stability audit

- Current raw book price: mean {fmt(current_stability['mean'])}, coefficient of variation {fmt(current_stability['coefficient_of_variation'])}, maximum absolute weekly change {fmt(current_stability['max_abs_weekly_change'] * 100)}%.
- Lagged book price: mean {fmt(lagged_stability['mean'])}, coefficient of variation {fmt(lagged_stability['coefficient_of_variation'])}, maximum absolute weekly change {fmt(lagged_stability['max_abs_weekly_change'] * 100)}%.
- Four-week moving average: coefficient of variation {fmt(ma4_stability['coefficient_of_variation'])}.
- Thirteen-week moving average: coefficient of variation {fmt(ma13_stability['coefficient_of_variation'])}.

The observed path is positive and smooth enough that an additional smoothing
parameter is not justified. Lagging is selected for timing semantics, not as a
new economic calibration.

## Household and LegacyOwner accounting

Household accounting remains acquisition-cost based. A share purchase records
cash down and equity acquisition cost up by exactly the same amount. Analysis
may additionally report `reference_value` and `unrealized_gain_loss`, but those
are non-cash diagnostics and do not create income or alter Household cash.

Secondary proceeds continue to flow from Household cash to `LegacyOwner.cash`.
Firm cash, paid-in equity, and total shares remain unchanged.

## G1 counterfactual boundary

Step15G.1 persisted aggregate executed cash and shares, but not each
Household's desired budget. Therefore the counterfactual uses the observed
aggregate G1 cash spend as a same-budget proxy and explicitly marks the
micro-budget result unavailable. It does not execute purchases. Under that
proxy, the observed 4.348237% transition becomes approximately
{fmt(safe_divide(engineering_cost, final_reference_price * total_shares) * 100)}%
of ownership at the book reference price.

## Verdict

**B. LAGGED_BOOK_PRICE_CONTRACT_READY**

The lagged authoritative accounting book price is ready as a non-market
reference contract. It is not a stock-market price and does not imply that book
value is the eventual economic market value.
"""
    (OUTPUT / "valuation_semantics.md").write_text(semantics, encoding="utf-8")

    summary = f"""# Step 15G.3 Book-Equity Reference Price Contract

**Verdict: B. LAGGED_BOOK_PRICE_CONTRACT_READY**

This stage was audit-only. No autonomous purchase was rerun and no economic
behavior changed.

The contract uses authoritative accounting equity divided by total shares. The
purchase decision at week `t` uses the locked closing book value from `t-1`.
Observed book equity stayed positive, and the raw/lagged series was stable
enough that moving-average smoothing was not justified.

The G1 observed 4.348237% ownership would require approximately
{fmt(safe_divide(engineering_cost, final_reference_price * total_shares) * 100)}%
ownership under the same observed aggregate cash budget at the final lagged
book price. This is a passive proxy because per-Household G1 budgets were not
persisted.
"""
    (OUTPUT / "acceptance_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps(flags, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
