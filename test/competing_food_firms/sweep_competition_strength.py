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


def mean(values):
    return float(np.mean(values)) if values else 0.0


def summarize(label, args, rows):
    final = rows[-1]
    tail = rows[-min(args.tail_window, len(rows)):]
    return {
        "label": label,
        "price_sensitivity": args.price_sensitivity,
        "labor_reallocation_rate": args.labor_reallocation_rate,
        "profit_labor_weight": args.profit_labor_weight,
        "strategy_exploration_noise": args.strategy_exploration_noise,
        "population": final["population"],
        "firm_cash": final["firm_cash"],
        "firm_cash_slope": slope([row["firm_cash"] for row in tail]),
        "money_supply": final["money_supply"],
        "money_slope": slope([row["money_supply"] for row in tail]),
        "price_std": final["price_std"],
        "price_std_slope": slope([row["price_std"] for row in tail]),
        "price_range": final["price_max"] - final["price_min"],
        "cash_gini": final["cash_gini"],
        "hhi": final["market_share_hhi"],
        "hhi_slope": slope([row["market_share_hhi"] for row in tail]),
        "cb_inventory": final["cb_inventory"],
        "cb_inventory_slope": slope([row["cb_inventory"] for row in tail]),
        "unmet_food": mean([row["unmet_food"] for row in tail]),
        "accounting_gap": final["accounting_gap"],
    }


def score(summary):
    competition = (
        100.0 * summary["price_std"]
        +
        8.0 * max(0.0, summary["hhi"] - 0.20)
        +
        30.0 * summary["cash_gini"]
    )
    instability = (
        abs(summary["firm_cash_slope"]) / 2000.0
        +
        abs(summary["money_slope"]) / 2000.0
        +
        max(0.0, summary["unmet_food"]) / 1000.0
        +
        abs(summary["accounting_gap"]) / 1000.0
    )
    return competition - instability


def stability_score(summary):
    competition_floor = (
        40.0 * summary["price_std"]
        +
        4.0 * max(0.0, summary["hhi"] - 0.20)
        +
        10.0 * summary["cash_gini"]
    )
    instability = (
        abs(summary["firm_cash_slope"]) / 500.0
        +
        abs(summary["money_slope"]) / 1000.0
        +
        abs(summary["cb_inventory_slope"]) / 1000.0
        +
        max(0.0, summary["unmet_food"]) / 100.0
        +
        abs(summary["accounting_gap"]) / 1000.0
    )
    return competition_floor - instability


def scenario_args(base_args, price_sensitivity, labor_rate, profit_weight, noise):
    return SimpleNamespace(
        seed=base_args.seed,
        population=base_args.population,
        warmup_steps=base_args.warmup_steps,
        competition_steps=base_args.competition_steps,
        firms=base_args.firms,
        tail_window=base_args.tail_window,
        progress_interval=0,
        price_sensitivity=price_sensitivity,
        loyalty_weight=base_args.loyalty_weight,
        price_adjustment_rate=base_args.price_adjustment_rate,
        production_adjustment_rate=base_args.production_adjustment_rate,
        strategy_exploration_noise=noise,
        labor_reallocation_rate=labor_rate,
        profit_labor_weight=profit_weight,
        standard_analysis=False,
        no_plots=True,
    )


def print_summary(summaries):
    print("Competing food firms strength sweep")
    print("===================================")
    print(f"Scenarios: {len(summaries)}")
    print()

    print("Outcome summary")
    print("---------------")
    print(
        f"{'label':<24} {'comp':>7} {'stable':>7} {'pstd':>7} {'range':>7} "
        f"{'hhi':>7} {'cgini':>7} {'cashsl':>9} {'msl':>9} "
        f"{'cbisl':>9} {'unmet':>8} {'gap':>9}"
    )
    for summary in summaries:
        print(
            f"{summary['label']:<24} "
            f"{score(summary):>7.2f} "
            f"{stability_score(summary):>7.2f} "
            f"{summary['price_std']:>7.4f} "
            f"{summary['price_range']:>7.4f} "
            f"{summary['hhi']:>7.3f} "
            f"{summary['cash_gini']:>7.3f} "
            f"{summary['firm_cash_slope']:>9,.0f} "
            f"{summary['money_slope']:>9,.0f} "
            f"{summary['cb_inventory_slope']:>9,.0f} "
            f"{summary['unmet_food']:>8,.2f} "
            f"{summary['accounting_gap']:>9.1e}"
        )

    print()
    print("Top candidates by competition score")
    print("-----------------------------------")
    for summary in sorted(summaries, key=score, reverse=True)[:5]:
        print(
            f"{summary['label']}: score={score(summary):.2f}, "
            f"price_std={summary['price_std']:.4f}, "
            f"HHI={summary['hhi']:.3f}, cash_gini={summary['cash_gini']:.3f}, "
            f"cash_slope={summary['firm_cash_slope']:,.2f}, "
            f"money_slope={summary['money_slope']:,.2f}"
        )

    print()
    print("Top candidates by stability-first score")
    print("---------------------------------------")
    for summary in sorted(summaries, key=stability_score, reverse=True)[:5]:
        print(
            f"{summary['label']}: stable={stability_score(summary):.2f}, "
            f"competition={score(summary):.2f}, "
            f"price_std={summary['price_std']:.4f}, HHI={summary['hhi']:.3f}, "
            f"cash_slope={summary['firm_cash_slope']:,.2f}, "
            f"money_slope={summary['money_slope']:,.2f}, "
            f"cb_inventory_slope={summary['cb_inventory_slope']:,.2f}"
        )

    print()
    print("Reading guide")
    print("-------------")
    print("Competition score rewards visible differentiation between firms.")
    print("Stability-first score is stricter about firm cash, money supply, and")
    print("central-bank inventory slopes. Use it when deciding whether a candidate")
    print("is safe enough for longer runs.")


def plot_summary(summaries):
    labels = [summary["label"] for summary in summaries]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    ax[0, 0].bar(x, [summary["price_std"] for summary in summaries])
    ax[0, 0].set_title("Final Price Std")

    ax[0, 1].bar(x, [summary["hhi"] for summary in summaries])
    ax[0, 1].axhline(0.20, linestyle="--", linewidth=1)
    ax[0, 1].set_title("Market Share HHI")

    ax[1, 0].bar(x, [summary["cash_gini"] for summary in summaries])
    ax[1, 0].set_title("Firm Cash Gini")

    ax[1, 1].bar(x, [summary["firm_cash_slope"] for summary in summaries], label="Firm Cash")
    ax[1, 1].bar(x, [summary["money_slope"] for summary in summaries], alpha=0.65, label="Money")
    ax[1, 1].axhline(0, linestyle="--", linewidth=1)
    ax[1, 1].set_title("Tail Slopes")
    ax[1, 1].legend()

    for axis in ax.ravel():
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep stronger competition parameters for the split food-firm model."
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--population", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--competition-steps", type=int, default=300)
    parser.add_argument("--firms", type=int, default=config.FIRM_COUNT)
    parser.add_argument("--tail-window", type=int, default=120)
    parser.add_argument("--loyalty-weight", type=float, default=config.LOYALTY_WEIGHT)
    parser.add_argument(
        "--price-adjustment-rate",
        type=float,
        default=config.FIRM_PRICE_ADJUSTMENT_RATE,
    )
    parser.add_argument(
        "--production-adjustment-rate",
        type=float,
        default=config.FIRM_PRODUCTION_ADJUSTMENT_RATE,
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    base_args = parse_args()
    scenarios = []

    for price_sensitivity in [2.5, 4.0, 5.5]:
        for labor_rate in [0.12, 0.20, 0.30]:
            for profit_weight in [0.20, 0.35]:
                noise = config.STRATEGY_EXPLORATION_NOISE
                label = (
                    f"ps={price_sensitivity:.1f} "
                    f"lr={labor_rate:.2f} "
                    f"pw={profit_weight:.2f}"
                )
                scenarios.append(
                    (
                        label,
                        scenario_args(
                            base_args,
                            price_sensitivity,
                            labor_rate,
                            profit_weight,
                            noise,
                        ),
                    )
                )

    summaries = []
    for label, args in scenarios:
        print(f"Running {label}")
        _, _, rows = run(args)
        summaries.append(summarize(label, args, rows))

    print_summary(summaries)

    if not base_args.no_plots:
        plot_summary(summaries)


if __name__ == "__main__":
    main()
