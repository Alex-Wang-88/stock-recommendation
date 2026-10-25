# 复杂性看板公开数据接入

本地入口：`site/complexity.html`。当前是公开日频快照，使用真实来源输入计算，没有虚拟数值兜底。

## 通俗阅读

页面默认显示通俗解读：首屏概括样本强弱、成交分布与历史联系，指标卡提供“怎么读/这是什么意思”注解，并附术语小词典。成交占比从实际成交额计算，均线数量从真实比例与样本数计算；不把HHI直接转成成交百分比，也不把相关系数或历史百分位称为上涨概率。历史联系文字随20/60/120日窗口更新，仅描述样本中的关系。

“展开专业明细”会展开关系图、来源表与计算口径；也可单独展开某一块。通俗版不改变原始快照和指标算法，原始指标仍可在注解、历史图与下载文件中核对。

## 数据源

| 系列 | 接入方式 | 实际品种/口径 |
| --- | --- | --- |
| 纳斯达克100、KOSPI | Yahoo Finance 公开 chart 端点 | `^NDX`、`^KS11`，完整日收盘，不是盘前分钟行情 |
| 黄金、布伦特 | 同上 | `GC=F`、`BZ=F`，供应商近月期货序列，存在换月影响，不是现货 |
| USD/CNY | Frankfurter v2 `/providers/ecb/rates` | ECB参考汇率，不是离岸人民币 USD/CNH |
| 美国10年期国债 | 美国财政部每日 XML | `daily_treasury_yield_curve` 的 `BC_10YEAR`，平价收益率，变化以 bp 表示 |
| A股行业代理ETF | 腾讯 `newfqkline/get` | 512480、512400、159945、516160、159928；统一请求未复权 `day`，成交额万元转元 |

参考资料：[Frankfurter文档](https://frankfurter.dev/)、[财政部XML说明](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed)、[Yahoo历史行情页面](https://finance.yahoo.com/quote/%5ENDX/history/)、[腾讯标的页面](https://gu.qq.com/sh512480)。Yahoo和腾讯是公开页面行情端点，不是有稳定性保证的官方API契约；公开可访问不构成任意再分发许可证明。

选择公开免密钥源后，原来的“A股全市场广度/行业集中度”改为明确的五只ETF代理篮子指标。五只ETF不能替代全市场；不将免费可得的代理值冒充原先计划字段。ETF使用未复权价格收益，非含分红再投资的总回报；除息/拆分、跟踪误差和主题重叠会影响指标。

## 更新与构建

Windows 可双击项目根目录 `更新复杂性看板.cmd`。也可在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe scripts/build_complexity_snapshot.py
```

脚本串行抓取，1 worker，Windows自身设置并核验 BelowNormal，不使用 NumPy/Pandas 或全市场重算。缓存位于被忽略的 `data/complexity/`，网站只读取生成的公开快照。

输出：

- `site/complexity-data.json`：来源摘要、请求收据/SHA-256、清洗后的输入序列、计算结果、口径。
- `site/complexity-data.js`：相同快照的脚本封装；避免本地 `file://` 的 fetch/CORS 限制。

每次成功抓取后保存该源缓存；传输故障、429或服务端错误最多重试一次。抓取失败保留原来的成功抓取时间，并明确标注 `cached` 和本次错误。首次云端运行可用仓库附带的真实快照初始化空缓存，保留原抓取时间，不覆盖已有缓存。不存在有效缓存时，显示 `unavailable` 和缺失数值。任何来源都不回退到随机数或原演示数据。旧源日期超过4个自然日显示较旧；页面生成超过36小时也会提示，这些是保守的新鲜度提示，不替代交易日历。

原有 GitHub Actions 流程已接入快照刷新：先构建 `_site`，再刷新 `_site/complexity-data.*`，缓存各源成功结果，之后随原有 Pages 工件一起发布。此次只修改本地代码，没有推送或触发远程部署；现有定时流程何时运行仍由仓库实际配置决定。

手工构建到指定输出目录：

```powershell
.\.venv\Scripts\python.exe scripts/build_public_site.py --data-dir data --source-dir site --output-dir _site
.\.venv\Scripts\python.exe scripts/build_complexity_snapshot.py --output-dir _site
```

## 计算约定

- 所有ETF收益使用相同的共同交易日网格，遇到缺失日期不会混用一天和多天的收益。
- 站上20日均线比例、收益总体标准差、ETF成交额HHI、ETF间60日相关网络密度均由五只ETF的真实输入计算。
- HHI等额基准为0.20。分位采用此前最多252个观测的中秩，至少60个历史样本；不把当前值加入自己的参照样本。
- 跨市场每个A股日期只取严格早于该日的外部源日期，先对齐水平再计算变化，因此休市日不重复使用前日收益。超过7天的外部旧值不配对。
- 对齐规则是日期滞后描述统计，历史发布时点未知，`published_at` 保留为空；没有声称通过点时回测验证。期货跨日后纳入，其他系列也使用保守完整日截断，可能比实时可得报价更旧。
- 20/60/120日窗内，至少20个有效配对且两端有波动才估计Pearson相关。画边阈值为 |ρ|≥0.30。密度、最强关系、热力表与线条均由同一份有符号系数计算。
- Fisher近似95%区间仅用于读数，不声称校正了时间依赖或多重比较，不提供预测或收益保证。

## 核验

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_complexity.py
.\.venv\Scripts\ruff.exe check src/recommend/complexity.py scripts/build_complexity_snapshot.py tests/test_complexity.py
node --test tests/complexity_frontend.test.cjs
node docs/audits/audit_complexity.cjs
```

Python检查覆盖价格清洗、单位转换、完整日截断、假日对齐、收益率bp、缺失/零方差、统一收益期间、分位、网络汇总和失败缓存。Node检查验证脚本与JSON相同、窗口联动、初始状态重复选择、各图符号一致及缺失渲染。Node使用最小document桩，不是浏览器渲染测试。

首次完整接入实测：11/11系列成功，五只ETF共同完整交易日为2026-10-19；30个外部×ETF关系在20/60/120窗均有足够样本。旧的演示核验记录保留在 `docs/audits/complexity-2026-10-19-review.md`；其JSON证据属于修复前版本。

浏览器安全策略禁止本会话打开 `file://` 页面；未绕过该限制，也未声称完成真实截图或移动端布局验证。新页面提供至少12px辅助文字、44px按钮、表格与图表局部横向滚动、原始读数表和CSV下载，这些已做源码/事件逻辑检查。
