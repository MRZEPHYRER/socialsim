import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


import config
from test_competing_food_firms import run, slope


def segment(rows, start, end=None):
    return [
        row
        for row in rows
        if row["step"] >= start and (end is None or row["step"] <= end)
    ]


def values(rows, metric):
    return [row[metric] for row in rows]


def mean(values_):
    return float(np.mean(values_)) if values_ else 0.0


def scenario_args(
    base_args,
    adaptation_rate,
    cash_pressure_weight,
    cb_release_rate,
    target_markup,
    cost_anchor_weight,
    normal_floor_factor,
):
    return SimpleNamespace(
        seed=base_args.seed,
        population=base_args.population,
        warmup_steps=base_args.warmup_steps,
        competition_steps=base_args.competition_steps,
        firms=base_args.firms,
        tail_window=base_args.tail_window,
        late_start=base_args.late_start,
        segment_boundaries=[base_args.late_start],
        progress_interval=0,
        price_sensitivity=config.PRICE_SENSITIVITY,
        loyalty_weight=config.LOYALTY_WEIGHT,
        price_adjustment_rate=config.FIRM_PRICE_ADJUSTMENT_RATE,
        production_adjustment_rate=config.FIRM_PRODUCTION_ADJUSTMENT_RATE,
        strategy_exploration_noise=config.STRATEGY_EXPLORATION_NOISE,
        labor_reallocation_rate=config.LABOR_REALLOCATION_RATE,
        profit_labor_weight=config.PROFIT_LABOR_WEIGHT,
        strategy_adaptation_rate=adaptation_rate,
        strategy_cash_pressure_weight=cash_pressure_weight,
        cb_release_rate=cb_release_rate,
        cb_purchase_rate=config.CB_PURCHASE_RATE,
        cb_max_new_money_issue_per_step=config.CB_MAX_NEW_MONEY_ISSUE_PER_STEP,
        target_markup=target_markup,
        cost_anchor_weight=cost_anchor_weight,
        normal_floor_factor=normal_floor_factor,
        skip_standard_analysis=True,
        no_plots=True,
    )


def summarize(label, args, rows):
    late = segment(rows, args.late_start)
    tail = rows[-min(args.tail_window, len(rows)):]
    final = rows[-1]

    return {
        "label": label,
        "adapt": args.strategy_adaptation_rate,
        "cash_pressure": args.strategy_cash_pressure_weight,
        "cb_release": args.cb_release_rate,
        "markup": args.target_markup,
        "anchor": args.cost_anchor_weight,
        "floor": args.normal_floor_factor,
        "final_price": final["price_mean"],
        "final_firm_cash": final["firm_cash"],
        "final_net_worth": final["firm_net_worth"],
        "final_money": final["money_supply"],
        "final_hhi": final["market_share_hhi"],
        "late_price_slope": slope(values(late, "price_mean")),
        "tail_price_slope": slope(values(tail, "price_mean")),
        "late_cash_slope": slope(values(late, "firm_cash")),
        "tail_cash_slope": slope(values(tail, "firm_cash")),
        "late_net_worth_slope": slope(values(late, "firm_net_worth")),
        "tail_net_worth_slope": slope(values(tail, "firm_net_worth")),
        "late_money_slope": slope(values(late, "money_supply")),
        "tail_money_slope": slope(values(tail, "money_supply")),
        "late_cb_inventory_slope": slope(values(late, "cb_inventory")),
        "tail_cb_inventory_slope": slope(values(tail, "cb_inventory")),
        "late_price_amplitude": max(values(late, "price_mean")) - min(values(late, "price_mean")),
        "tail_price_amplitude": max(values(tail, "price_mean")) - min(values(tail, "price_mean")),
        "late_unmet_food": mean(values(late, "unmet_food")),
        "accounting_gap": final["accounting_gap"],
    }


def stability_score(summary):
    return -(
        abs(summary["tail_price_slope"]) * 30_000
        +
        abs(summary["late_price_slope"]) * 20_000
        +
        abs(summary["tail_cash_slope"]) / 500
        +
        abs(summary["tail_net_worth_slope"]) / 500
        +
        abs(summary["tail_money_slope"]) / 100
        +
        summary["tail_price_amplitude"] * 20
        +
        max(0.0, summary["late_unmet_food"]) / 100
    )


def print_summary(summaries, args):
    print("Late price stability sweep")
    print("==========================")
    print(f"Seed: {args.seed}")
    print(f"Population: {args.population:,}")
    print(f"Warmup / competition steps: {args.warmup_steps:,} / {args.competition_steps:,}")
    print(f"Late window starts at competition step: {args.late_start}")
    print(f"Tail window: {args.tail_window}")
    print(f"Scenarios: {len(summaries)}")
    print()

    print("Outcome summary")
    print("---------------")
    print(
        f"{'label':<36} {'score':>8} {'p':>6} {'p late':>9} {'p tail':>9} "
        f"{'pampt':>7} {'cash tail':>11} {'nw tail':>10} {'money':>8} "
        f"{'cb inv':>9} {'hhi':>6} {'gap':>9}"
    )
    for item in summaries:
        print(
            f"{item['label']:<36} "
            f"{stability_score(item):>8.2f} "
            f"{item['final_price']:>6.3f} "
            f"{item['late_price_slope']:>9.6f} "
            f"{item['tail_price_slope']:>9.6f} "
            f"{item['tail_price_amplitude']:>7.4f} "
            f"{item['tail_cash_slope']:>11,.0f} "
            f"{item['tail_net_worth_slope']:>10,.0f} "
            f"{item['tail_money_slope']:>8,.2f} "
            f"{item['tail_cb_inventory_slope']:>9,.0f} "
            f"{item['final_hhi']:>6.3f} "
            f"{item['accounting_gap']:>9.1e}"
        )

    print()
    print("Top candidates by late-price stability")
    print("--------------------------------------")
    for item in sorted(summaries, key=stability_score, reverse=True)[:6]:
        print(
            f"{item['label']}: score={stability_score(item):.2f}, "
            f"tail price slope={item['tail_price_slope']:.6f}, "
            f"late price slope={item['late_price_slope']:.6f}, "
            f"tail cash slope={item['tail_cash_slope']:,.2f}, "
            f"tail net worth slope={item['tail_net_worth_slope']:,.2f}, "
            f"money slope={item['tail_money_slope']:,.2f}"
        )

    print()
    print("Reading guide")
    print("-------------")
    print("The best candidate is not necessarily the highest-price case.")
    print("Prefer near-zero late/tail price slopes, small tail price amplitude,")
    print("non-collapsing firm cash/net worth, near-zero money slope, and no unmet food.")


def plot_summary(summaries):
    labels = [item["label"] for item in summaries]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(2, 2, figsize=(16, 9))

    ax[0, 0].bar(x, [item["tail_price_slope"] for item in summaries], label="Tail")
    ax[0, 0].bar(x, [item["late_price_slope"] for item in summaries], alpha=0.55, label="Late")
    ax[0, 0].axhline(0, linestyle="--", linewidth=1)
    ax[0, 0].set_title("Price Slopes")
    ax[0, 0].legend()

    ax[0, 1].bar(x, [item["tail_price_amplitude"] for item in summaries])
    ax[0, 1].set_title("Tail Price Amplitude")

    ax[1, 0].bar(x, [item["tail_cash_slope"] for item in summaries], label="Cash")
    ax[1, 0].bar(x, [item["tail_net_worth_slope"] for item in summaries], alpha=0.55, label="Net Worth")
    ax[1, 0].axhline(0, linestyle="--", linewidth=1)
    ax[1, 0].set_title("Firm Tail Slopes")
    ax[1, 0].legend()

    ax[1, 1].bar(x, [item["tail_money_slope"] for item in summaries], label="Money")
    ax[1, 1].bar(x, [item["tail_cb_inventory_slope"] for item in summaries], alpha=0.55, label="CB Inventory")
    ax[1, 1].axhline(0, linestyle="--", linewidth=1)
    ax[1, 1].set_title("Central Bank Tail Slopes")
    ax[1, 1].legend()

    for axis in ax.ravel():
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep adaptive strategy parameters with emphasis on late-stage price stability."
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--population", type=int, default=1200)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--competition-steps", type=int, default=900)
    parser.add_argument("--firms", type=int, default=config.FIRM_COUNT)
    parser.add_argument("--late-start", type=int, default=600)
    parser.add_argument("--tail-window", type=int, default=200)
    parser.add_argument(
        "--adaptation-rates",
        type=float,
        nargs="*",
        default=[0.0015, 0.0025, 0.0040],
    )
    parser.add_argument(
        "--cash-pressure-weights",
        type=float,
        nargs="*",
        default=[0.10, 0.20, 0.30],
    )
    parser.add_argument(
        "--cb-release-rates",
        type=float,
        nargs="*",
        default=[0.45, 0.60],
    )
    parser.add_argument(
        "--target-markups",
        type=float,
        nargs="*",
        default=[config.FIRM_TARGET_MARKUP],
    )
    parser.add_argument(
        "--cost-anchor-weights",
        type=float,
        nargs="*",
        default=[config.FIRM_PRICE_COST_ANCHOR_WEIGHT],
    )
    parser.add_argument(
        "--normal-floor-factors",
        type=float,
        nargs="*",
        default=[config.FIRM_NORMAL_PRICE_FLOOR_FACTOR],
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    base_args = parse_args()
    scenarios = []
    for adaptation_rate in base_args.adaptation_rates:
        for cash_pressure_weight in base_args.cash_pressure_weights:
            for cb_release_rate in base_args.cb_release_rates:
                for target_markup in base_args.target_markups:
                    for cost_anchor_weight in base_args.cost_anchor_weights:
                        for normal_floor_factor in base_args.normal_floor_factors:
                            label = (
                                f"a={adaptation_rate:.4f} "
                                f"cw={cash_pressure_weight:.2f} "
                                f"r={cb_release_rate:.2f} "
                                f"m={target_markup:.2f} "
                                f"aw={cost_anchor_weight:.2f} "
                                f"fl={normal_floor_factor:.2f}"
                            )
                            scenarios.append(
                                (
                                    label,
                                    scenario_args(
                                        base_args,
                                        adaptation_rate,
                                        cash_pressure_weight,
                                        cb_release_rate,
                                        target_markup,
                                        cost_anchor_weight,
                                        normal_floor_factor,
                                    ),
                                )
                            )

    summaries = []
    original_values = (
        config.FIRM_TARGET_MARKUP,
        config.FIRM_PRICE_COST_ANCHOR_WEIGHT,
        config.FIRM_NORMAL_PRICE_FLOOR_FACTOR,
    )
    try:
        for label, args in scenarios:
            print(f"Running {label}")
            config.FIRM_TARGET_MARKUP = args.target_markup
            config.FIRM_PRICE_COST_ANCHOR_WEIGHT = args.cost_anchor_weight
            config.FIRM_NORMAL_PRICE_FLOOR_FACTOR = args.normal_floor_factor
            _, _, rows = run(args)
            summaries.append(summarize(label, args, rows))
    finally:
        (
            config.FIRM_TARGET_MARKUP,
            config.FIRM_PRICE_COST_ANCHOR_WEIGHT,
            config.FIRM_NORMAL_PRICE_FLOOR_FACTOR,
        ) = original_values

    print_summary(summaries, base_args)

    if not base_args.no_plots:
        plot_summary(summaries)


if __name__ == "__main__":
    main()
