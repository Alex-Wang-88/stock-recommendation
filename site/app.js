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

function addPriceCell(parent, label, value) {
  const cell = makeNode("div", "price-cell");
  cell.append(makeNode("span", "", label), makeNode("strong", "", value));
  parent.append(cell);
}

function signalClass(signal) {
  if (signal === "今天可分批") return "buy";
  if (signal === "等回落再买") return "wait";
  return "pause";
}

function renderStock(item, index) {
  const card = makeNode("article", "stock-card");
  const head = makeNode("div", "stock-card-head");
  const identity = makeNode("div", "identity");
  identity.append(makeNode("span", "rank-number", String(index + 1).padStart(2, "0")));
  const nameWrap = makeNode("div", "stock-name-wrap");
  nameWrap.append(makeNode("h3", "stock-name", item.name || "未命名"));
  const meta = makeNode("div", "stock-meta");
  meta.append(makeNode("span", "stock-code", item.symbol || ""));
  if (item.industry) meta.append(makeNode("span", "", item.industry));
  nameWrap.append(meta);
  identity.append(nameWrap);
  const scoreBox = makeNode("div", "score-box");
  scoreBox.append(makeNode("span", "score-label", "综合分"), makeNode("strong", "score-value", number(item.score, 3)));
  head.append(identity, scoreBox);

  const signalRow = makeNode("div", "signal-row");
  signalRow.append(makeNode("span", `signal-pill ${signalClass(item.operation_signal)}`, item.operation_signal || "未标注"));
  if (item.expectation_state) signalRow.append(makeNode("span", "expectation", item.expectation_state));

  const prices = makeNode("div", "price-grid");
  addPriceCell(prices, "参考收盘价", price(item.close));
  addPriceCell(prices, "分批买入区间", item.buy_low != null || item.buy_high != null ? `${price(item.buy_low)} – ${price(item.buy_high)}` : "—");
  addPriceCell(prices, "回落触发参考", price(item.pullback_price));
  addPriceCell(prices, "不追价参考", price(item.no_chase_price));
  addPriceCell(prices, "逻辑失效参考", price(item.invalidation_price));
  addPriceCell(prices, "目标一 / 目标二", `${price(item.target_1)} / ${price(item.target_2)}`);

  const insights = makeNode("div", "stock-insights");
  const pe = numericValue(item.pe_ttm);
  const peText = Number.isFinite(pe) && pe > 0 ? number(pe, 1) : "—";
  insights.append(makeNode("span", "", "近 20 日 "));
  insights.lastChild.append(makeNode("strong", "", percent(item.ret_20)));
  insights.append(makeNode("span", "", "相对大盘 "));
  insights.lastChild.append(makeNode("strong", "", percent(item.relative_ret_20)));
  insights.append(makeNode("span", "", "PE(TTM) "));
  insights.lastChild.append(makeNode("strong", "", peText));

  card.append(head, signalRow, prices, insights);
  if (item.announcement_alert) card.append(makeNode("p", "announcement-note", `公告提醒：${item.announcement_alert}`));
  if (item.operation_reason || item.reason) {
    const details = makeNode("details", "details-toggle");
    details.append(makeNode("summary", "", "查看筛选与操作说明"));
    if (item.operation_reason) details.append(makeNode("p", "", item.operation_reason));
    if (item.reason) details.append(makeNode("p", "", item.reason));
    card.append(details);
  }
  return card;
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
  const sortBy = byId("sort-select").value;
  let rows = [...(data.recommendations || [])].filter((item) => {
    const searchText = [item.symbol, item.name, item.industry, item.announcement_alert].join(" ").toLocaleLowerCase("zh-CN");
    return (!query || searchText.includes(query)) && (!selectedSignal || item.operation_signal === selectedSignal);
  });

  if (sortBy === "score") rows.sort((a, b) => (Number(b.score) || 0) - (Number(a.score) || 0));
  if (sortBy === "ret20") rows.sort((a, b) => (Number(b.ret_20) || -Infinity) - (Number(a.ret_20) || -Infinity));

  const list = byId("recommendation-list");
  list.replaceChildren();
  list.setAttribute("aria-busy", "false");
  byId("visible-count").textContent = `显示 ${rows.length} / ${data.recommendations.length} 只`;
  if (!rows.length) {
    list.append(makeNode("p", "empty-message", "没有符合当前搜索条件的候选。"));
    return;
  }
  rows.forEach((item, index) => list.append(renderStock(item, index)));
}

function renderNotes(notes) {
  const section = byId("data-notes");
  const list = byId("notes-list");
  list.replaceChildren();
  if (!notes || !notes.length) {
    section.hidden = true;
    return;
  }
  notes.forEach((note) => list.append(makeNode("li", "", note)));
  section.hidden = false;
}

function renderPage(data) {
  byId("as-of-value").textContent = data.as_of || "暂无快照";
  byId("candidate-count").textContent = number(data.recommendations?.length ?? 0, 0);
  byId("pool-count").textContent = data.qualified_count != null
    ? `规则通过 ${number(data.qualified_count, 0)} 只 · 展示前 ${number(data.recommendations?.length ?? 0, 0)} 只`
    : "按当前筛选规则排序";
  byId("spec-hash").textContent = data.spec_hash || "—";
  byId("updated-at").textContent = (data.published_at || "").replace("T", " ").slice(0, 16) || "—";
  renderNotes(data.notes || []);
  renderEvaluations(data.evaluation || []);
  updateRecommendations(data);
  for (const id of ["search-input", "signal-filter", "sort-select"]) {
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
