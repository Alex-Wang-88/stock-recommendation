const byId = (id) => document.getElementById(id);
const numberFormat = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 });

function numericValue(value) {
  if (value == null || value === "") return Number.NaN;
  return Number(value);
}

function number(value, digits = 2) {
  const parsed = numericValue(value);
  if (!Number.isFinite(parsed)) return "—";
  return parsed.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function scoreOutOf100(value) {
  const parsed = numericValue(value);
  return Number.isFinite(parsed) ? `${number(parsed * 100, 1)} / 100` : "—";
}

function price(value) {
  const parsed = numericValue(value);
  return Number.isFinite(parsed) ? `¥${numberFormat.format(parsed)}` : "—";
}

function percent(value, digits = 1) {
  const parsed = numericValue(value);
  if (!Number.isFinite(parsed)) return "—";
  const formatted = `${(parsed * 100).toFixed(digits)}%`;
  return parsed > 0 ? `+${formatted}` : formatted;
}

function makeNode(tag, className = "", textValue = "") {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = textValue == null ? "" : String(textValue);
  return element;
}

function signalClass(signal) {
  if (signal === "今天可分批") return "buy";
  if (signal === "等回落再买") return "wait";
  return "pause";
}

let selectedSymbol = null;

function renderCandidateOverview(rows) {
  const container = byId("candidate-overview");
  container.replaceChildren();

  const signalCounts = new Map();
  const industryCounts = new Map();
  for (const row of rows) {
    const signal = row.operation_signal || "未标注";
    signalCounts.set(signal, (signalCounts.get(signal) || 0) + 1);
    const industry = row.industry || "未同步";
    industryCounts.set(industry, (industryCounts.get(industry) || 0) + 1);
  }

  const summary = makeNode("div", "candidate-status-grid");
  const commonSignals = ["今天可分批", "等回落再买", "暂缓，先确认"];
  const otherSignals = [...signalCounts.keys()].filter((signal) => !commonSignals.includes(signal));
  for (const signal of [...commonSignals, ...otherSignals]) {
    const card = makeNode("div", `candidate-status-item ${signalClass(signal)}`);
    card.append(
      makeNode("span", "candidate-status-label", signal),
      makeNode("strong", "candidate-status-value", `${signalCounts.get(signal) || 0} 只`),
    );
    summary.append(card);
  }

  const industries = makeNode("div", "candidate-industry-line");
  industries.append(makeNode("span", "industry-caption", "候选行业 · "));
  const topIndustries = [...industryCounts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0], "zh-CN"))
    .slice(0, 4);
  for (const [industry, count] of topIndustries) {
    const chip = makeNode("span", "candidate-industry-chip");
    chip.append(makeNode("span", "", industry), makeNode("strong", "", `${count} 只`));
    industries.append(chip);
  }
  industries.append(makeNode("span", "candidate-scope-note", "仅统计当前候选清单"));
  container.append(summary, industries);
}

function renderCandidate(item, rank, selected) {
  const button = makeNode("button", `candidate-row${selected ? " selected" : ""}`);
  button.type = "button";
  if (selected) button.setAttribute("aria-current", "true");
  button.setAttribute("aria-label", `${item.name || "未命名"} ${item.symbol || ""}，综合得分 ${scoreOutOf100(item.score)}，${item.operation_signal || "未标注"}`);

  const identity = makeNode("span", "candidate-identity");
  identity.append(
    makeNode("span", "candidate-rank", String(rank).padStart(2, "0")),
    makeNode("span", "candidate-name", item.name || "未命名"),
    makeNode("span", "candidate-symbol", item.symbol || ""),
  );
  const signal = makeNode("span", `candidate-signal ${signalClass(item.operation_signal)}`, item.operation_signal || "未标注");
  const score = makeNode("span", "candidate-score", number(numericValue(item.score) * 100, 1));
  score.title = "综合得分，满分 100 分";
  const ret = makeNode("span", `candidate-return${numericValue(item.ret_20) < 0 ? " negative" : ""}`, percent(item.ret_20));
  button.append(identity, signal, score, ret);
  button.addEventListener("click", () => {
    selectedSymbol = item.symbol;
    const data = window.recommendationSnapshot;
    if (data) updateRecommendations(data);
  });
  const row = makeNode("div", "candidate-row-wrap");
  row.setAttribute("role", "listitem");
  row.append(button);
  return row;
}

function addDetailStat(parent, label, value) {
  const cell = makeNode("div", "detail-stat");
  cell.append(makeNode("span", "detail-stat-label", label), makeNode("strong", "detail-stat-value", value));
  parent.append(cell);
}

function addPriceCell(parent, label, value) {
  const cell = makeNode("div", "detail-price-cell");
  cell.append(makeNode("span", "detail-price-label", label), makeNode("strong", "detail-price-value", value));
  parent.append(cell);
}

function svgNode(tag, attributes = {}, textValue = "") {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  if (textValue) element.textContent = textValue;
  return element;
}

function renderCandlestickChart(item) {
  const section = makeNode("section", "detail-chart-section");
  const sourceBars = Array.isArray(item.bars) ? item.bars : [];
  const history = sourceBars.map((row) => {
    if (!Array.isArray(row) || row.length < 5) return null;
    const [date, open, high, low, close, volume] = row;
    const values = [open, high, low, close].map((value) => value == null ? Number.NaN : Number(value));
    if (!values.every(Number.isFinite)) return null;
    return { date: String(date || ""), open: values[0], high: values[1], low: values[2], close: values[3], volume };
  }).filter(Boolean);
  const bars = history.slice(-60);

  const heading = makeNode("div", "detail-chart-heading");
  heading.append(makeNode("strong", "detail-chart-title", "日线走势"));
  const latestDate = bars.length ? bars[bars.length - 1].date : "";
  heading.append(makeNode("span", "detail-chart-range", bars.length ? `近 ${bars.length} 日 · 均线基于 ${history.length} 日 · 截至 ${latestDate}` : "日线数据暂不可用"));

  section.append(heading);
  if (!bars.length) {
    section.append(makeNode("p", "detail-chart-empty", "该股票暂无可绘制的近期日线数据。"));
    return section;
  }

  const width = 680;
  const height = 236;
  const margin = { left: 60, right: 14, top: 18, bottom: 27 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maPeriods = [5, 10, 20, 60];
  const movingAverages = new Map(maPeriods.map((period) => [period, []]));
  const movingTotals = new Map(maPeriods.map((period) => [period, 0]));
  history.forEach((bar, index) => {
    for (const period of maPeriods) {
      let total = movingTotals.get(period) + bar.close;
      if (index >= period) total -= history[index - period].close;
      movingTotals.set(period, total);
      movingAverages.get(period).push(index + 1 >= period ? total / period : Number.NaN);
    }
  });
  const visibleStart = history.length - bars.length;
  const visibleMovingAverages = maPeriods.flatMap((period) =>
    movingAverages.get(period).slice(visibleStart).filter(Number.isFinite),
  );
  const priceBounds = [
    ...bars.map((bar) => bar.low),
    ...bars.map((bar) => bar.high),
    ...visibleMovingAverages,
  ];
  let minPrice = Math.min(...priceBounds);
  let maxPrice = Math.max(...priceBounds);
  const spread = Math.max(maxPrice - minPrice, Math.abs(maxPrice) * 0.02, 0.02);
  minPrice -= spread * 0.04;
  maxPrice += spread * 0.04;
  const yPosition = (value) => margin.top + ((maxPrice - value) / (maxPrice - minPrice)) * plotHeight;
  const step = plotWidth / bars.length;
  const candleWidth = Math.min(7, Math.max(2.4, step * 0.62));

  const svg = svgNode("svg", {
    class: "detail-candlestick-chart",
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `${item.name || "股票"} ${item.symbol || ""} 最近 ${bars.length} 个交易日日线 K 线和 MA5、MA10、MA20、MA60 均线`,
    preserveAspectRatio: "none",
  });
  svg.append(svgNode("title", {}, `${item.name || "股票"} ${item.symbol || ""} 日线 K 线`));
  svg.append(svgNode("rect", { class: "chart-background", x: 0, y: 0, width, height, rx: 8 }));

  for (let tick = 0; tick < 4; tick += 1) {
    const value = maxPrice - ((maxPrice - minPrice) * tick) / 3;
    const y = yPosition(value);
    svg.append(svgNode("line", {
      class: "chart-grid-line",
      x1: margin.left,
      y1: y.toFixed(2),
      x2: width - margin.right,
      y2: y.toFixed(2),
    }));
    svg.append(svgNode("text", {
      class: "chart-axis-label",
      x: margin.left - 7,
      y: (y + 4).toFixed(2),
      "text-anchor": "end",
    }, value.toFixed(2)));
  }

  const maPoints = new Map(maPeriods.map((period) => [period, []]));
  bars.forEach((bar, index) => {
    const x = margin.left + (index + 0.5) * step;
    const openY = yPosition(bar.open);
    const closeY = yPosition(bar.close);
    const group = svgNode("g", { class: bar.close >= bar.open ? "candle-up" : "candle-down" });
    const volumeText = bar.volume == null ? "" : ` · 成交量 ${number(bar.volume, 0)}`;
    group.append(svgNode("title", {}, `${bar.date} 开 ${price(bar.open)} 高 ${price(bar.high)} 低 ${price(bar.low)} 收 ${price(bar.close)}${volumeText}`));
    group.append(svgNode("line", {
      class: "candle-wick",
      x1: x.toFixed(2),
      y1: yPosition(bar.high).toFixed(2),
      x2: x.toFixed(2),
      y2: yPosition(bar.low).toFixed(2),
    }));
    group.append(svgNode("rect", {
      class: "candle-body",
      x: (x - candleWidth / 2).toFixed(2),
      y: Math.min(openY, closeY).toFixed(2),
      width: candleWidth.toFixed(2),
      height: Math.max(Math.abs(closeY - openY), 1.2).toFixed(2),
    }));
    svg.append(group);

    for (const period of maPeriods) {
      const average = movingAverages.get(period)[visibleStart + index];
      if (Number.isFinite(average)) {
        maPoints.get(period).push(`${x.toFixed(2)},${yPosition(average).toFixed(2)}`);
      }
    }
  });
  for (const period of maPeriods) {
    const points = maPoints.get(period);
    const lineClass = `chart-ma-line chart-ma-${period}`;
    if (points.length >= 2) svg.append(svgNode("polyline", { class: lineClass, points: points.join(" ") }));
    else if (points.length === 1) {
      const [cx, cy] = points[0].split(",");
      svg.append(svgNode("circle", { class: lineClass, cx, cy, r: 2.4 }));
    }
  }
  svg.append(svgNode("text", {
    class: "chart-axis-label",
    x: margin.left,
    y: height - 8,
    "text-anchor": "start",
  }, bars[0].date));
  svg.append(svgNode("text", {
    class: "chart-axis-label",
    x: width - margin.right,
    y: height - 8,
    "text-anchor": "end",
  }, bars[bars.length - 1].date));

  const chartLegend = makeNode("div", "detail-chart-legend");
  const legendItems = [
    ["阳线", "chart-legend-up"],
    ["阴线", "chart-legend-down"],
    ...maPeriods.map((period) => [`MA${period}`, `chart-legend-ma-${period}`]),
  ];
  for (const [label, swatchClass] of legendItems) {
    const legendItem = makeNode("span", "chart-legend-item");
    legendItem.append(makeNode("i", swatchClass), makeNode("span", "", label));
    chartLegend.append(legendItem);
  }
  section.append(chartLegend, svg);
  return section;
}

function renderStockDetail(item, rank) {
  const panel = byId("selected-detail");
  panel.replaceChildren();
  panel.setAttribute("aria-label", `${item.name || "未命名"} ${item.symbol || ""} 详情`);

  const header = makeNode("div", "detail-header");
  const titleGroup = makeNode("div", "detail-title-group");
  titleGroup.append(
    makeNode("span", "detail-rank", `策略排名 ${String(rank).padStart(2, "0")}`),
    makeNode("h3", "detail-title", item.name || "未命名"),
  );
  const symbol = makeNode("span", "detail-symbol", item.symbol || "");
  header.append(titleGroup, symbol);
  const meta = makeNode("p", "detail-meta", [item.industry || "行业未同步", item.expectation_state || ""].filter(Boolean).join(" · "));

  const stats = makeNode("div", "detail-stats");
  addDetailStat(stats, "收盘价", price(item.close));
  addDetailStat(stats, "综合得分", scoreOutOf100(item.score));
  addDetailStat(stats, "权重覆盖", item.score_coverage == null ? "—" : percent(item.score_coverage, 0));
  addDetailStat(stats, "近 20 日", percent(item.ret_20));

  const signal = makeNode("p", `detail-signal ${signalClass(item.operation_signal)}`);
  signal.append(makeNode("span", "", "操作信号"), makeNode("strong", "", item.operation_signal || "未标注"));

  const prices = makeNode("div", "detail-price-grid");
  const buyRange = item.buy_low != null || item.buy_high != null
    ? `${price(item.buy_low)} – ${price(item.buy_high)}`
    : "—";
  addPriceCell(prices, "参考价", price(item.close));
  addPriceCell(prices, "分批买入区间", buyRange);
  addPriceCell(prices, "回落触发参考", price(item.pullback_price));
  addPriceCell(prices, "不追价参考", price(item.no_chase_price));
  addPriceCell(prices, "逻辑失效参考", price(item.invalidation_price));
  addPriceCell(prices, "目标一 / 目标二", `${price(item.target_1)} / ${price(item.target_2)}`);

  panel.append(header, meta, stats, signal, prices, renderCandlestickChart(item));
  if (item.announcement_alert) {
    panel.append(makeNode("p", "announcement-note", `公告提醒：${item.announcement_alert}`));
  }

  if (item.operation_reason || item.reason) {
    const details = makeNode("details", "details-toggle");
    details.append(makeNode("summary", "", "筛选与操作说明"));
    if (item.operation_reason) details.append(makeNode("p", "", item.operation_reason));
    if (item.reason) details.append(makeNode("p", "", item.reason));
    panel.append(details);
  }

  const insight = makeNode("p", "detail-insight");
  const pe = numericValue(item.pe_ttm);
  const peText = Number.isFinite(pe) && pe > 0 ? number(pe, 1) : "—";
  insight.textContent = `相对大盘 ${percent(item.relative_ret_20)} · PE(TTM) ${peText}`;
  panel.append(insight);
}

function renderEvaluations(rows) {
  const container = byId("evaluation-list");
  container.replaceChildren();
  const horizons = [1, 5, 20];
  const found = new Map((rows || []).map((row) => [Number(row.horizon), row]));
  for (const horizon of horizons) {
    const row = found.get(horizon) || {};
    const card = makeNode("article", "evaluation-card");
    card.append(makeNode("h3", "", `${horizon} 个交易日`));
    const excess = Number(row.excess_mean);
    const measured = Number(row.measured) || 0;
    const runs = Number(row.runs) || 0;
    const ready = Number.isFinite(excess) && measured > 0;
    card.append(makeNode("p", "evaluation-state", ready ? `已完成 ${measured} / ${runs} 个推荐批次` : "样本积累中，暂无成熟结果"));
    card.append(makeNode("p", `evaluation-excess${ready ? (excess >= 0 ? " positive" : " negative") : ""}`, ready ? percent(excess, 2) : "待验证"));
    const details = ready
      ? `候选平均 ${percent(row.pick_ret, 2)} · 全市场等权 ${percent(row.universe_ret, 2)}`
      : "持有期尚未走完或有效样本不足。";
    card.append(makeNode("p", "evaluation-sub", ready ? `平均超额收益 · ${details}` : details));
    container.append(card);
  }
}

function updateRecommendations(data) {
  const query = byId("search-input").value.trim().toLocaleLowerCase("zh-CN");
  const selectedSignal = byId("signal-filter").value;
  const selectedIndustry = byId("industry-filter").value;
  const sortBy = byId("sort-select").value;
  let rows = [...(data.recommendations || [])].filter((item) => {
    const searchText = [item.symbol, item.name, item.industry, item.announcement_alert].join(" ").toLocaleLowerCase("zh-CN");
    return (!query || searchText.includes(query))
      && (!selectedSignal || (item.operation_signal || "未标注") === selectedSignal)
      && (!selectedIndustry || (item.industry || "未同步") === selectedIndustry);
  });

  if (sortBy === "score") rows.sort((a, b) => {
    const left = numericValue(b.score);
    const right = numericValue(a.score);
    return (Number.isFinite(left) ? left : -Infinity) - (Number.isFinite(right) ? right : -Infinity);
  });
  if (sortBy === "ret20") rows.sort((a, b) => {
    const left = numericValue(b.ret_20);
    const right = numericValue(a.ret_20);
    return (Number.isFinite(left) ? left : -Infinity) - (Number.isFinite(right) ? right : -Infinity);
  });

  const list = byId("recommendation-list");
  list.replaceChildren();
  list.setAttribute("aria-busy", "false");
  byId("visible-count").textContent = `显示 ${rows.length} / ${data.recommendations.length} 只`;
  if (!rows.length) {
    list.append(makeNode("p", "empty-message", "没有符合当前搜索条件的候选。"));
    byId("selected-detail").replaceChildren(makeNode("p", "empty-message", "当前筛选下没有候选详情。"));
    selectedSymbol = null;
    return;
  }

  if (!rows.some((item) => item.symbol === selectedSymbol)) selectedSymbol = rows[0].symbol;
  rows.forEach((item) => {
    const rank = data.recommendations.findIndex((candidate) => candidate.symbol === item.symbol) + 1;
    list.append(renderCandidate(item, rank, item.symbol === selectedSymbol));
  });
  const selected = rows.find((item) => item.symbol === selectedSymbol);
  renderStockDetail(selected, data.recommendations.findIndex((item) => item.symbol === selectedSymbol) + 1);
}

function populateFilter(id, placeholder, values) {
  const select = byId(id);
  const previous = select.value;
  select.replaceChildren();
  const allOption = makeNode("option", "", placeholder);
  allOption.value = "";
  select.append(allOption);
  for (const value of values) {
    const option = makeNode("option", "", value);
    option.value = value;
    select.append(option);
  }
  if (values.includes(previous)) select.value = previous;
}

function renderNotes(notes) {
  const section = byId("data-notes");
  const list = byId("notes-list");
  const count = byId("notes-count");
  list.replaceChildren();
  if (!notes || !notes.length) {
    section.hidden = true;
    return;
  }
  notes.forEach((note) => list.append(makeNode("li", "", note)));
  count.textContent = `${notes.length} 项，点击展开`;
  section.hidden = false;
}

function renderPage(data) {
  window.recommendationSnapshot = data;
  byId("as-of-value").textContent = data.as_of || "暂无快照";
  byId("candidate-count").textContent = number(data.recommendations?.length ?? 0, 0);
  byId("pool-count").textContent = data.qualified_count != null
    ? `规则通过 ${number(data.qualified_count, 0)} 只 · 展示前 ${number(data.recommendations?.length ?? 0, 0)} 只`
    : "按当前筛选规则排序";
  byId("spec-hash").textContent = data.spec_hash || "—";
  byId("updated-at").textContent = (data.published_at || "").replace("T", " ").slice(0, 16) || "—";
  renderNotes(data.notes || []);
  renderEvaluations(data.evaluation || []);
  renderCandidateOverview(data.recommendations || []);
  populateFilter(
    "signal-filter",
    "全部信号",
    [...new Set((data.recommendations || []).map((item) => item.operation_signal || "未标注"))].sort((a, b) => a.localeCompare(b, "zh-CN")),
  );
  populateFilter(
    "industry-filter",
    "全部行业",
    [...new Set((data.recommendations || []).map((item) => item.industry || "未同步"))].sort((a, b) => a.localeCompare(b, "zh-CN")),
  );
  updateRecommendations(data);
  for (const id of ["search-input", "signal-filter", "industry-filter", "sort-select"]) {
    byId(id).addEventListener("input", () => updateRecommendations(data));
    byId(id).addEventListener("change", () => updateRecommendations(data));
  }
  byId("reload-button").addEventListener("click", () => window.location.reload());
}

async function loadSnapshot() {
  try {
    const response = await fetch(`./data.json?cache=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderPage(await response.json());
  } catch (error) {
    const errorBox = byId("load-error");
    errorBox.textContent = `暂时无法读取推荐快照（${error.message}）。请稍后重新载入，或查看 GitHub Actions 最近一次运行。`;
    errorBox.hidden = false;
    byId("recommendation-list").setAttribute("aria-busy", "false");
    byId("recommendation-list").replaceChildren(makeNode("p", "empty-message", "推荐数据暂不可用。"));
    byId("evaluation-list").replaceChildren();
    byId("reload-button").addEventListener("click", () => window.location.reload());
  }
}

loadSnapshot();
