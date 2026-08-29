# Step 10 完成记录

Step 10 已结束，当前不再为了降低库存或触发贷款而修改 baseline 经济机制。

已确认的限制：

- 当前主模型仍是单商品、单企业结构。
- 企业库存主要只能依靠最终家庭消费变现，外加已有公共库存工具。
- 模型还没有跨商品替代、企业间交易、资产转换或中间品市场。
- 因此库存占用现金的问题应先作为当前结构限制记录，不应通过临时调低库存目标、修改生产规则、价格规则或贷款规则来消除。

Step 10.6 的最后诊断要求也已落实：后续 diagnostics 会直接记录 `wage_bill` / `wage_payment`，不再要求通过 `firm_cash / firm_cash_to_wage_bill` 反推工资账单。

Step 10.9 已加入完整 warm-start/checkpoint 机制。正式 warm world 使用：

```text
scenario=wage_shock_1_47
initial_population=5000
seed=42
step 0-999: WAGE_MULTIPLIER=1.0
step 1000 起: WAGE_MULTIPLIER=1.47
saved_global_step=5000
```

正式 checkpoint 路径：

```text
test/output/step10_9_warm_checkpoint/wage_shock_1_47_seed_42_pop_5000_step_5000/world_step_5000.pkl
```

checkpoint 保存完整 `World`，包括人口、家庭、企业、央行/公共部门、现金财富、库存、价格、贷款余额、预期、家庭关系、ID/counter、global step，以及 Python/NumPy RNG state。加载时会校验 checkpoint version 和结构，不兼容则报错。

下一步进入 Step 11：多企业。
