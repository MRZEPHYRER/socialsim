from economy.investment_planning import ShadowInvestmentPlanner


def decide(**overrides):
    values = {
        "firm_id": 1,
        "expected_demand": 100.0,
        "desired_output": 100.0,
        "current_effective_capacity": 80.0,
        "desired_capacity": 100.0,
        "non_capital_capacity": 0.0,
        "current_capital_capacity": 80.0,
        "replacement_capacity_loss": 0.0,
        "replacement_cost_per_capacity": 1.0,
        "expansion_cost_per_capacity": 1.0,
        "cash": 1_000.0,
        "operating_liquidity_floor": 0.0,
    }
    values.update(overrides)
    return ShadowInvestmentPlanner().decide(**values)


def test_committed_expansion_reduces_only_uncommitted_gap():
    decision = decide(committed_expansion_capacity=15.0)

    assert decision.expansion_investment_need == 5.0
    assert decision.committed_expansion_capacity == 15.0
    assert decision.effective_future_capital_capacity == 95.0
    assert decision.residual_uncommitted_capacity_gap == 5.0


def test_committed_replacement_restores_retired_service_without_expansion():
    decision = decide(
        current_capital_capacity=80.0,
        replacement_capacity_loss=20.0,
        committed_replacement_capacity=20.0,
    )

    assert decision.replacement_investment_need == 0.0
    assert decision.expansion_investment_need == 0.0
    assert decision.committed_replacement_capacity == 20.0
    assert decision.effective_future_capital_capacity == 100.0


def test_default_zero_commitment_preserves_installed_capital_formula():
    decision = decide(
        current_capital_capacity=80.0,
        replacement_capacity_loss=20.0,
    )

    assert decision.replacement_investment_need == 20.0
    assert decision.expansion_investment_need == 0.0
    assert decision.residual_uncommitted_capacity_gap == 20.0
