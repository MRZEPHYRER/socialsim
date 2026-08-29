"""Step17.H - structural low-tail Household root-cause decomposition."""
from __future__ import annotations
import json, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GDIR = ROOT / "test/output/step17_g_long_horizon_liquidity_warmup"
OUT = ROOT / "test/output/step17_h_structural_low_tail_decomposition"
EPS = 1e-8
LOW = .25


def write(name, frame):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def qstats(series):
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {"count": 0, "mean": np.nan, "median": np.nan, "p10": np.nan, "p25": np.nan, "p75": np.nan, "p90": np.nan}
    return {"count": int(len(x)), "mean": float(x.mean()), "median": float(x.median()), "p10": float(x.quantile(.1)), "p25": float(x.quantile(.25)), "p75": float(x.quantile(.75)), "p90": float(x.quantile(.9))}


def group_rows(frame, group_col="liquidity_group"):
    rows = []
    for group, q in frame.groupby(group_col, sort=False):
        for metric, series in (("income_to_minimum_need", q.income_to_minimum_need), ("wage_income_to_minimum_need", q.wage_income_to_minimum_need), ("accumulation_margin", q.accumulation_margin), ("accumulation_margin_weeks", q.accumulation_margin_weeks), ("minimum_consumption_cost", q.weekly_minimum_consumption_cost), ("realized_consumption", q.consumption)):
            item = {"group": group, "metric": metric, **qstats(series)}
            rows.append(item)
    return pd.DataFrame(rows)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    gpath = GDIR / "long_horizon_household_snapshots.csv"
    if not gpath.exists():
        raise FileNotFoundError(gpath)
    f = pd.read_csv(gpath, low_memory=False)
    for c in ("global_step", "household_id", "closing_cash", "weekly_minimum_consumption_cost", "liquidity_weeks", "household_size", "wage_income", "total_income", "consumption", "saving"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    for c in ("elderly_household", "employed_member", "genealogy_uncovered_elderly"):
        f[c] = f[c].fillna(False).astype(bool)
    f = f.sort_values(["household_id", "global_step"]).drop_duplicates(["global_step", "household_id"]).reset_index(drop=True)
    f = f[(f.global_step >= 3960) & (f.global_step <= 4999)].copy()
    f["liquidity_group"] = pd.cut(f.liquidity_weeks, bins=[-np.inf, .25, 1, 13, np.inf], labels=["DEEP_LOW", "LOW", "SECURE", "HIGH_LIQUIDITY"], right=False).astype(object)
    f["income_to_minimum_need"] = f.total_income / f.weekly_minimum_consumption_cost.replace(0, np.nan)
    f["wage_income_to_minimum_need"] = f.wage_income / f.weekly_minimum_consumption_cost.replace(0, np.nan)
    f["accumulation_margin"] = f.total_income - f.consumption
    f["accumulation_margin_weeks"] = f.accumulation_margin / f.weekly_minimum_consumption_cost.replace(0, np.nan)
    f["employment_category"] = np.where(f.employed_member, "EMPLOYED_COUNT_UNAVAILABLE", "NO_EMPLOYED_MEMBER")
    f["size_category"] = f.household_size.map(lambda x: "SIZE_1" if x == 1 else "SIZE_2" if x == 2 else "SIZE_3" if x == 3 else "SIZE_4_PLUS" if x >= 4 else "UNAVAILABLE")
    # G does not persist opening cash or exact adult/child counts. The first
    # observed row is retained as a panel entry, not relabeled as formation.
    first_seen = f.groupby("household_id").global_step.min().rename("first_observed_step")
    f = f.merge(first_seen, on="household_id", how="left")
    f["newly_observed_in_window"] = f.first_observed_step >= 3960

    deep = f[f.liquidity_group == "DEEP_LOW"].copy()
    nondeep = f[f.liquidity_group != "DEEP_LOW"].copy()
    high = f[f.liquidity_group == "HIGH_LIQUIDITY"].copy()
    obs = f.groupby("household_id").agg(observed_weeks=("global_step", "nunique"), low_weeks=("liquidity_group", lambda x: int((x == "DEEP_LOW").sum())), first_observed_step=("global_step", "min"), last_observed_step=("global_step", "max"), mean_size=("household_size", "mean"), elderly_share=("elderly_household", "mean"), employed_share=("employed_member", "mean"), mean_income=("total_income", "mean"), mean_need=("weekly_minimum_consumption_cost", "mean"), mean_margin=("accumulation_margin", "mean"), mean_income_need=("income_to_minimum_need", "mean"), mean_wage_need=("wage_income_to_minimum_need", "mean"), mean_cash=("closing_cash", "mean"), max_cash=("closing_cash", "max")).reset_index()
    obs["low_share"] = obs.low_weeks / obs.observed_weeks.replace(0, np.nan)
    # The mature window is sampled every 13 weeks, so 52 observed weeks means
    # at least four snapshot observations; do not call short panels persistent.
    obs["classification"] = np.where((obs.observed_weeks >= 4) & (obs.low_share >= .75), "PERSISTENT_DEEP_LOW", np.where(obs.low_weeks == 0, "NEVER_DEEP_LOW", np.where(obs.observed_weeks <= 2, "TRANSIENT_DEEP_LOW", "RECURRENT_DEEP_LOW")))
    persistent_ids = set(obs.loc[obs.classification == "PERSISTENT_DEEP_LOW", "household_id"])
    obs["lifecycle_observation_note"] = "disappearance is observation loss, not recovery"
    write("persistent_low_households.csv", obs[obs.classification == "PERSISTENT_DEEP_LOW"].sort_values("household_id"))

    summary = []
    for group, q in f.groupby("liquidity_group", observed=False):
        item = {"group": str(group), "household_weeks": len(q), "unique_households": q.household_id.nunique(), "share_of_mature_household_weeks": len(q) / len(f), "mean_size": q.household_size.mean(), "elderly_share": q.elderly_household.mean(), "employed_member_share": q.employed_member.mean(), "mean_cash": q.closing_cash.mean(), "mean_minimum_need": q.weekly_minimum_consumption_cost.mean(), "mean_income": q.total_income.mean(), "mean_consumption": q.consumption.mean(), "mean_saving": q.saving.mean(), "mean_accumulation_margin": q.accumulation_margin.mean(), "mean_income_to_need": q.income_to_minimum_need.mean(), "median_income_to_need": q.income_to_minimum_need.median()}
        summary.append(item)
    write("mature_household_group_summary.csv", summary)
    write("income_need_comparison.csv", group_rows(f[f.liquidity_group.notna()]))
    write("accumulation_margin_comparison.csv", group_rows(f[f.liquidity_group.notna()]))

    # Employment, size and elderly stratifications remain descriptive. Exact
    # employed-member counts and adult/child counts were not persisted by G.
    def stratify(column, name):
        rows = []
        for key, q in f.groupby(column, dropna=False, observed=False):
            for group, z in q.groupby("liquidity_group", observed=False):
                rows.append({"stratifier": name, "stratum": str(key), "group": str(group), "household_weeks": len(z), "unique_households": z.household_id.nunique(), "mean_income": z.total_income.mean(), "mean_need": z.weekly_minimum_consumption_cost.mean(), "mean_income_to_need": z.income_to_minimum_need.mean(), "mean_wage_to_need": z.wage_income_to_minimum_need.mean(), "mean_margin": z.accumulation_margin.mean(), "mean_cash": z.closing_cash.mean()})
        return rows
    write("employment_stratified_comparison.csv", stratify("employment_category", "employment_category"))
    write("household_size_stratified_comparison.csv", stratify("size_category", "household_size"))
    write("elderly_stratified_comparison.csv", stratify("elderly_household", "elderly_presence"))
    pw = f[f.employed_member].groupby("liquidity_group", observed=False).agg(household_weeks=("household_id", "size"), unique_households=("household_id", "nunique"), mean_realized_wage_per_employed_household=("wage_income", "mean"), mean_income_to_need=("income_to_minimum_need", "mean"), median_income_to_need=("income_to_minimum_need", "median"), mean_effective_labor=("household_id", lambda x: np.nan)).reset_index()
    pw["worker_age_productivity"] = np.nan
    pw["effective_labor_per_earner"] = np.nan
    pw["realized_wage_per_earner"] = np.nan
    pw["unavailable_fields"] = "adult/earner count, age-productivity and effective labor were not persisted"
    write("productivity_wage_comparison.csv", pw)

    # Snapshot entry/exit event studies. These are 13-week observed transitions;
    # G's authoritative online weekly transition table is used for rates, while
    # household feature windows are available only at snapshot cadence.
    f2 = f.sort_values(["household_id", "global_step"]).copy()
    f2["prev_liquidity"] = f2.groupby("household_id").liquidity_weeks.shift(1)
    f2["entry_deep_low"] = (f2.liquidity_weeks < LOW) & (f2.prev_liquidity >= LOW)
    f2["exit_deep_low"] = (f2.liquidity_weeks >= LOW) & (f2.prev_liquidity < LOW)
    def event_summary(events, kind):
        rows = []
        for _, row in events.iterrows():
            before = f[(f.household_id == row.household_id) & (f.global_step == row.global_step - 13)]
            b = before.iloc[0] if len(before) else None
            rows.append({"event": kind, "household_id": row.household_id, "event_week": int(row.global_step), "income_change": row.total_income - b.total_income if b is not None else np.nan, "need_change": row.weekly_minimum_consumption_cost - b.weekly_minimum_consumption_cost if b is not None else np.nan, "employment_changed": bool(row.employed_member != b.employed_member) if b is not None else np.nan, "size_change": row.household_size - b.household_size if b is not None else np.nan, "elderly_changed": bool(row.elderly_household != b.elderly_household) if b is not None else np.nan, "margin_change": row.accumulation_margin - b.accumulation_margin if b is not None else np.nan, "feature_window_note": "13-week observed event; exact weekly trigger unavailable"})
        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame([{"event": kind, "event_count": 0, "income_loss_share": np.nan, "need_increase_share": np.nan, "employment_change_share": np.nan, "size_increase_share": np.nan, "elderly_transition_share": np.nan}])
        return pd.DataFrame([{"event": kind, "event_count": len(out), "income_loss_share": float((out.income_change < -EPS).mean()), "need_increase_share": float((out.need_change > EPS).mean()), "employment_change_share": float(out.employment_changed.fillna(False).mean()), "size_increase_share": float((out.size_change > 0).mean()), "elderly_transition_share": float(out.elderly_changed.fillna(False).mean()), "mean_income_change": out.income_change.mean(), "mean_need_change": out.need_change.mean(), "mean_margin_change": out.margin_change.mean(), "trigger_evidence": "descriptive timing only"}])
    write("low_tail_entry_event_summary.csv", event_summary(f2[f2.entry_deep_low], "ENTRY_INTO_DEEP_LOW"))
    write("low_tail_exit_event_summary.csv", event_summary(f2[f2.exit_deep_low], "EXIT_FROM_DEEP_LOW"))

    # Aggregate formation evidence: identities first observed in the window are
    # a lower bound on new observed households, not authoritative formation.
    first_rows = f[f.global_step == f.first_observed_step].groupby("global_step").agg(newly_observed_households=("household_id", "nunique"), newly_observed_deep_low=("liquidity_group", lambda x: int((x == "DEEP_LOW").sum())), mean_opening_cash=("closing_cash", "mean"), mean_opening_liquidity=("liquidity_weeks", "mean")).reset_index()
    first_rows["formation_status"] = "UNAVAILABLE; first observed in selected panel is only a coverage proxy"
    write("household_formation_low_tail.csv", first_rows)

    # Deterministic same-week match by persisted size, elderly flag and employed
    # boolean; then household id. This is a descriptive matched contrast.
    latest = f.sort_values("global_step").groupby("household_id", as_index=False).tail(1)
    persistent = obs[obs.classification == "PERSISTENT_DEEP_LOW"].copy()
    candidates = latest[latest.classification.isin([])] if False else latest[~latest.household_id.isin(persistent_ids) & (latest.liquidity_group != "DEEP_LOW")]
    match_rows = []
    for _, lowrow in persistent.sort_values("household_id").iterrows():
        pool = candidates[(candidates.household_size == round(lowrow.mean_size)) & (candidates.elderly_household == (lowrow.elderly_share >= .5)) & (candidates.employed_member == (lowrow.employed_share >= .5))]
        if pool.empty:
            pool = candidates[candidates.household_size == round(lowrow.mean_size)]
        if pool.empty:
            continue
        match = pool.sort_values(["household_id"]).iloc[0]
        match_rows.append({"persistent_household_id": lowrow.household_id, "comparison_household_id": match.household_id, "match_size": match.household_size, "match_elderly": match.elderly_household, "match_employment": match.employed_member, "persistent_mean_income": lowrow.mean_income, "comparison_income": match.total_income, "persistent_mean_need": lowrow.mean_need, "comparison_need": match.weekly_minimum_consumption_cost, "persistent_income_to_need": lowrow.mean_income_need, "comparison_income_to_need": match.income_to_minimum_need, "persistent_margin": lowrow.mean_margin, "comparison_margin": match.accumulation_margin, "matching_note": "latest mature snapshot; deterministic id tie-break"})
    write("matched_household_comparison.csv", match_rows)

    # Interpretable descriptive logistic regression on household-level mature
    # aggregates. Avoid black-box models and label coefficients as descriptive.
    features = obs.copy()
    features["target"] = (features.classification == "PERSISTENT_DEEP_LOW").astype(int)
    predictor_names = ["employed_share", "mean_size", "elderly_share", "mean_income_need", "mean_wage_need", "mean_margin"]
    model_rows = []
    try:
        from sklearn.linear_model import LogisticRegression
        x = features[predictor_names].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        y = features.target.to_numpy(int)
        if len(np.unique(y)) > 1:
            means = x.mean(axis=0); scales = x.std(axis=0); scales[scales < EPS] = 1.0
            z = (x - means) / scales
            model = LogisticRegression(max_iter=1000, random_state=0).fit(z, y)
            for name, coef in zip(predictor_names, model.coef_[0]):
                model_rows.append({"predictor": name, "coefficient_standardized": float(coef), "odds_ratio_per_sd": float(np.exp(coef)), "model": "descriptive household-level logistic; not causal", "target": "P(PERSISTENT_DEEP_LOW)"})
            model_rows.append({"predictor": "MODEL_INTERCEPT", "coefficient_standardized": float(model.intercept_[0]), "odds_ratio_per_sd": np.nan, "model": "descriptive; discrimination secondary", "target": "P(PERSISTENT_DEEP_LOW)"})
        else:
            raise ValueError("single target class")
    except Exception as exc:
        model_rows.append({"predictor": "UNAVAILABLE", "coefficient_standardized": np.nan, "odds_ratio_per_sd": np.nan, "model": f"not fit: {type(exc).__name__}", "target": "P(PERSISTENT_DEEP_LOW)"})
    write("persistent_low_logistic_model.csv", model_rows)

    # Mechanism overlap and deterministic primary classification.
    all_persistent = f[f.household_id.isin(persistent_ids)].groupby("household_id").agg(observed_weeks=("global_step", "nunique"), low_weeks=("liquidity_group", lambda x: int((x == "DEEP_LOW").sum())), no_earner=("employed_member", lambda x: not bool(x.any())), mean_income_need=("income_to_minimum_need", "mean"), mean_margin=("accumulation_margin", "mean"), elderly=("elderly_household", "mean"), mean_need=("weekly_minimum_consumption_cost", "mean"), mean_cash=("closing_cash", "mean"), first_observed=("global_step", "min")).reset_index()
    all_persistent["income_gap"] = all_persistent.mean_income_need < 1.0
    all_persistent["elderly_gap"] = (all_persistent.elderly > .5) & (all_persistent.mean_income_need < 1.5)
    all_persistent["high_need_burden"] = (all_persistent.mean_income_need < 1.5) & ~all_persistent.income_gap
    all_persistent["repeated_negative_margin"] = all_persistent.mean_margin <= EPS
    all_persistent["new_observed_proxy"] = all_persistent.first_observed >= 3960
    overlaps = []
    for flag in ("no_earner", "income_gap", "high_need_burden", "elderly_gap", "repeated_negative_margin", "new_observed_proxy"):
        overlaps.append({"mechanism": flag, "persistent_households": int(all_persistent[flag].sum()), "share_persistent_households": float(all_persistent[flag].mean()) if len(all_persistent) else np.nan, "persistent_household_weeks": int(all_persistent.loc[all_persistent[flag], "observed_weeks"].sum()), "share_persistent_household_weeks": float(all_persistent.loc[all_persistent[flag], "observed_weeks"].sum() / all_persistent.observed_weeks.sum()) if len(all_persistent) and all_persistent.observed_weeks.sum() else np.nan, "rule": "no earner; mean income/need<1; elderly+income/need<1.5; nonnegative-income-gap need burden; mean margin<=0; first observed proxy"})
    write("mechanism_overlap.csv", overlaps)
    def primary(row):
        if row.no_earner: return "NO_EARNER"
        if row.income_gap: return "LOW_EARNER_INCOME_RELATIVE_TO_NEED"
        if row.elderly_gap: return "ELDERLY_INCOME_GAP"
        if row.high_need_burden: return "HIGH_DEPENDENT_OR_MINIMUM_NEED_BURDEN"
        if row.repeated_negative_margin: return "REPEATED_NEGATIVE_ACCUMULATION_MARGIN"
        return "MIXED_OR_UNCLASSIFIED"
    all_persistent["primary_mechanism"] = all_persistent.apply(primary, axis=1)
    write("mechanism_primary_classification.csv", all_persistent)
    contrib = all_persistent.groupby("primary_mechanism", dropna=False).agg(persistent_households=("household_id", "nunique"), persistent_household_weeks=("observed_weeks", "sum")).reset_index()
    contrib["share_persistent_households"] = contrib.persistent_households / max(1, len(all_persistent))
    contrib["share_persistent_household_weeks"] = contrib.persistent_household_weeks / max(1, all_persistent.observed_weeks.sum())
    write("mechanism_contribution_summary.csv", contrib)

    # Polarization comparison: persistent low identity aggregates vs high-liquidity identities.
    high_ids = set(f[f.liquidity_group == "HIGH_LIQUIDITY"].household_id.unique())
    pol_rows = []
    for label, ids in (("PERSISTENT_DEEP_LOW", persistent_ids), ("HIGH_LIQUIDITY", high_ids)):
        q = f[f.household_id.isin(ids)]
        pol_rows.append({"group": label, "households": len(ids), "household_weeks": len(q), "mean_income": q.total_income.mean(), "mean_consumption": q.consumption.mean(), "mean_saving": q.saving.mean(), "mean_margin": q.accumulation_margin.mean(), "mean_size": q.household_size.mean(), "earners_proxy_share": q.employed_member.mean(), "elderly_share": q.elderly_household.mean(), "mean_minimum_need": q.weekly_minimum_consumption_cost.mean(), "mean_income_to_need": q.income_to_minimum_need.mean(), "mean_liquidity": q.liquidity_weeks.mean()})
    write("polarization_comparison.csv", pol_rows)

    # Capacity-only policy-neutral screens; no transfers or runtime changes.
    screens = []
    p = f[f.household_id.isin(persistent_ids)].copy()
    for name, amount in (("A_ELIMINATE_NO_EARNER_INCOME_GAP", np.where((~p.employed_member) & (p.total_income < p.weekly_minimum_consumption_cost), p.weekly_minimum_consumption_cost - p.total_income, 0.0)), ("B_RAISE_LOW_INCOME_TO_MINIMUM_NEED", np.maximum(p.weekly_minimum_consumption_cost - p.total_income, 0.0)), ("C_OLD_AGE_INCOME_FLOOR", np.where(p.elderly_household, np.maximum(p.weekly_minimum_consumption_cost - p.total_income, 0.0), 0.0)), ("D_FORMATION_OPENING_CASH_FLOOR", np.where(p.newly_observed_in_window, np.maximum(p.weekly_minimum_consumption_cost - p.closing_cash, 0.0), 0.0)), ("E_GENERIC_SOCIAL_ASSISTANCE_FLOOR", np.maximum(p.weekly_minimum_consumption_cost - p.closing_cash, 0.0))):
        screens.append({"screen": name, "persistent_households": len(persistent_ids), "household_weeks": len(p), "total_shadow_cash_capacity": float(np.nansum(amount)), "mean_per_persistent_household_week": float(np.nanmean(amount)) if len(amount) else np.nan, "positive_capacity_household_weeks": int(np.sum(np.asarray(amount) > EPS)), "runtime_transfer_executed": False, "interpretation": "capacity screen only; no policy implementation"})
    write("shadow_policy_capacity_screen.csv", screens)

    # Acceptance flags and summary.
    transition = pd.read_csv(GDIR / "long_horizon_transition_rates.csv")
    mature_transition = transition[transition.window.isin(["2601-4999", "3960-4999"])] if "window" in transition else transition.tail(1)
    primary_counts = contrib.set_index("primary_mechanism").persistent_households.to_dict() if not contrib.empty else {}
    top = max(primary_counts, key=primary_counts.get) if primary_counts else "MIXED_OR_UNCLASSIFIED"
    distinct = sum(value > 0 for value in primary_counts.values())
    verdict = "F. STRUCTURAL_LOW_TAIL_HAS_MULTIPLE_DISTINCT_MECHANISMS" if distinct >= 2 else ({"NO_EARNER": "A. STRUCTURAL_LOW_TAIL_IS_PRIMARILY_NO_EARNER", "LOW_EARNER_INCOME_RELATIVE_TO_NEED": "B. STRUCTURAL_LOW_TAIL_IS_PRIMARILY_LOW_INCOME_RELATIVE_TO_NEED", "ELDERLY_INCOME_GAP": "C. STRUCTURAL_LOW_TAIL_IS_PRIMARILY_ELDERLY_INCOME_GAP", "HIGH_DEPENDENT_OR_MINIMUM_NEED_BURDEN": "E. STRUCTURAL_LOW_TAIL_IS_PRIMARILY_PERSISTENT_CONSUMPTION_BURDEN"}.get(top, "H. OTHER"))
    flags = {"verdict": verdict, "analysis_window": "3960-4999", "sensitivity_window": "2600-4999 available in G but primary outputs use 3960-4999", "persistent_household_count": len(persistent_ids), "mature_household_week_rows": len(f), "deep_low_household_weeks": len(deep), "high_liquidity_household_count": len(high_ids), "weekly_transition_source": "Step17.G online weekly transition table; household feature event studies are 13-week snapshots", "exact_adult_child_counts": False, "exact_employed_member_count": False, "age_productivity_composition": False, "authoritative_formation_events_by_household": False, "inheritance_events_available": False, "cash_flow_other_actual_outflows": False, "opening_cash_persisted": False, "disappearance_not_recovery": True, "logistic_is_descriptive_not_causal": True, "policy_screens_are_shadow_only": True, "economic_behavior_changed": False, "new_rng_draws": 0, "new_long_runs": 0, "step17_h_rerun": False, "g_source": str(GDIR), "g_weekly_rows": int(pd.read_csv(GDIR / "long_horizon_weekly_liquidity.csv").shape[0])}
    (OUT / "acceptance_flags.json").write_text(json.dumps(flags, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    text = ["# Step17.H Acceptance Summary", "", f"Verdict: {verdict}", "", f"Primary analysis uses the mature window 3960-4999 from the accepted Step17.G CONTROL run. It contains {len(f):,} authoritative Household-snapshot rows and {len(persistent_ids):,} persistently classified DEEP_LOW Household identities.", "", "Classification uses at least four 13-week observations in the selected window and at least 75% DEEP_LOW observations; disappeared Households are not treated as recovered. The underlying online entry/exit rates remain sourced from Step17.G weekly transitions.", "", f"The leading primary mechanism is {top}; {distinct} distinct primary mechanisms have positive mass. Income/need, employment boolean, size and elderly stratifications are descriptive. Exact adult/child counts, exact earner counts, effective labor, worker age productivity, authoritative formation identity, inheritance event history, opening cash and other outflows were not persisted by G and are marked unavailable.", "", "Cash-flow interpretation uses total income minus realized consumption as the observed accumulation margin. In CONTROL, mandatory policy outflows are inactive; this is not a reconstructed full ledger identity where other outflows are unavailable.", "", "The logistic model is a standardized, interpretable descriptive model for P(PERSISTENT_DEEP_LOW), not causal evidence. Shadow policy rows are capacity screens only; no transfers or policy behavior were executed.", "", "No simulation rerun, policy implementation, RNG change or economic behavior change occurred. Artifacts: test/output/step17_h_structural_low_tail_decomposition/"]
    (OUT / "acceptance_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()