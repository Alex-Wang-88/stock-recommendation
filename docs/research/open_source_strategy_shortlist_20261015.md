# 开源选股方案筛选（2026-10-15）

结论：下一轮优先采用 Qlib 的 Alpha158 量价特征，以 LightGBM 为主候选、线性模型为简单对照。推荐依据是公式、训练配置、组合规则公开，且不需要历史财报或行业标签；尚未证明它在本项目的数据与成交约束下能产生稳定正超额。

本次完成公开资料及部分源码核对，没有安装外部项目或训练新模型。下列“适配建议”属于本项目的研究设计，不是原作者回测结果。

## 候选与取舍

| 优先级 | 方案 | 核对内容 | 本项目适配判断 |
| --- | --- | --- | --- |
| 1 | Qlib Alpha158 + LightGBM + 排名选股 | 因子公式、标签、训练/验证/测试配置、费用、TopkDropout 参数公开 | 最适合先验证。让模型在训练期学习量价信号的组合，替代继续手工枚举权重；仍需控制过拟合 |
| 2 | Alpha158 + 线性模型 | Qlib 有 OLS 基线；vnpy.alpha 有 Lasso 实现 | 与第一项使用相同特征，属于模型对照，不应算成独立的策略发现。建议采用带正则的线性模型作简单对照 |
| 备用 | QuantConnect 残差动量 | 用历史收益对 Fama-French 三因子回归，按标准化残差排序；原实现含多空组合 | 需要匹配 A 股的因子序列及较长历史，且原版做空腿无法直接迁移到现有只做多组合；暂缓 |
| 工程参考 | vnpy.alpha、Hikyuu | 前者有因子、模型及研究工作流，后者提供策略部件和回测框架 | 可借鉴接口和自动化。采用框架本身不能证明选股信号有效，目前没有必要为此整体重写本项目回测器 |

## Qlib 源码确认及差异

- Alpha158 由 K 线、价格、成交量及滚动统计构成，默认滚动窗口为 5/10/20/30/60 日。VWAP 也是输入；没有使用财报或行业字段。
- 官方 LightGBM 示例为沪深300股票池，TopkDropout 的 topk=50、n_drop=5，按收盘价成交，开仓费率 0.0005、平仓 0.0015、最低费用 5。配置里的训练期为 2008—2014，验证期为 2015—2016，测试期为 2017—2020。
- 默认预测标签是后一交易日至再后一交易日的收盘收益，与本项目次日开盘交易、5/10 日持有口径不同。因此迁移后应称“借鉴公开基线的适配实验”，不能宣称复现了原仓库收益。
- 官方示例 num_threads=20 不符合本机资源政策；后续实施必须改为预检后允许的线程数，并以 BelowNormal 运行。
- Qlib 使用 MIT 许可证。实际复制源码时需保留相应版权及许可证；本次仅记录链接与研究结论。

来源：[Alpha158 公式](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)、[标签定义](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py)、[LightGBM 完整配置](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)、[线性模型配置](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/Linear/workflow_config_linear_Alpha158.yaml)、[许可证](https://github.com/microsoft/qlib/blob/main/LICENSE)。链接为检索当日分支内容，实施时还需固定版本。

## 数据能否支持

现有冻结行情包含 OHLC、volume、amount，足以支持大部分量价公式；尚不能直接宣称完整 Alpha158 已可复现：

1. 核对成交量的股/手单位，以及 amount/volume 与 OHLC 的币种、单位和复权是否一致，再计算 VWAP。无法对齐时应明确删去依赖项，记录特征数变化，不能拿收盘价冒充 VWAP。
2. 核对停牌、ST、涨跌停、上市时间、历史股票池和退市覆盖，避免当前成分股倒填历史。
3. 较完整的横截面行情主要从 2023 年开始；不能照搬原作者十余年的训练数据设置。复权和历史股票池的时点正确性也仍需审计。
4. 继续排除无法按当时可知版本复现的历史财报及行业数据。

本地证据：[上一轮研究报告](market_direction_patterns_20261015.md)、冻结行情 `logs/historical-weight-validation-20261014/inputs/market.parquet`。本次未重新扫描全量行情。

## 更有效率的下一轮设计

1. 先固定一个预测期限、股票池和成交规则，比较现有评分、正则线性模型、LightGBM。线性和树模型使用相同特征、标签与数据切分；候选模型少量固定，所有尝试均记录。
2. 按时间滚动训练和预测；预处理参数只在训练期拟合。训练/验证/测试边界移除跨界的未来收益标签；只有到预测时已经完整可见的标签才能进入训练。
3. 先检查预测排名与未来收益的相关性、分组收益和跨期稳定性，淘汰无预测力的方案，再进行完整账户回放。特征相关、重要性高或训练误差低，均不能替代净超额验证。
4. 在同一引擎、同一股票池、同一仓位和调仓约束下回放，保留单边 10/20bps、四个价格指数及时间对齐的全收益指数、同池基准，分开报告选股收益和仓位控制效果。
5. 已查看的 15 段只用于开发比较。最终验收需要此前未参与筛选的数据或后续模拟跟踪；时间切分本身不能抹去我们已经看过这些行情的事实。

重点是建立可重复的信号比较流程。机器学习能减少手工调权重，仍可能过拟合；本次检索没有支持提高多少超额的可靠量化承诺。

## 其他公开来源的核对结果

- [vnpy.alpha 官方说明](https://github.com/vnpy/vnpy#ai-powered)提供 Alpha158、Lasso、LightGBM 及完整工作流；[Lasso 源码](https://github.com/vnpy/vnpy/blob/master/vnpy/alpha/model/models/lasso_model.py)已查看。其 Alpha158 来源于 Qlib，不能把两者当成两套独立信号。[因子实现](https://github.com/vnpy/vnpy/blob/master/vnpy/alpha/dataset/datasets/alpha_158.py)已查看；较大的 LightGBM notebook 正文未成功读取，未审计其全部回测输出。
- [QuantConnect 残差动量](https://www.quantconnect.com/research/15304/residual-momentum/)公开回归及组合步骤，依赖约 36 个月历史和匹配市场的 Fama-French 因子。移植为 A 股只做多版本需要重新验证。
- [QuantConnect 短期反转](https://www.quantconnect.com/research/15311/short-term-reversal-strategy-in-stocks/)按近期跌幅买入、涨幅做空，并限制流动性；作者明确报告该实现除 2020 年崩盘期外，所测各期跑输标普500。因此仅作为公开反例和机制参考，不列为优先收益候选。
- [Hikyuu](https://github.com/fasiondog/hikyuu)提供因子、资金分配、交易信号等可组合部件，使用 Apache-2.0 许可证。本次核对了项目说明，没有审计其整个策略部件库。

未依据 GitHub 星数、宣传年化收益或回测截图排名；未将美股多空表现换算为 A 股只做多收益。网络结果不构成该策略已通过本项目验收的证据。
