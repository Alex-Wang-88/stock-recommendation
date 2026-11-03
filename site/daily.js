"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const escape = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const pct = (n, digits = 2) => Number.isFinite(n) ? `${n > 0 ? "+" : ""}${(n * 100).toFixed(digits)}%` : "—";
  const number = (n, digits = 1) => Number.isFinite(n) ? n.toFixed(digits) : "—";
  const color = n => Number.isFinite(n) ? n > 0 ? "positive" : n < 0 ? "negative" : "" : "";
  const statuses = {waiting_entry:"等待入场日留证",holding:"观察期内",matured:"已完成",exit_blocked:"退出受限，继续跟踪",data_gap:"数据待核对"};
  let payload = null;
  let refreshing = false;
  let lastRefreshAt = 0;
  let selectedCohort = null;
  function spark(history) {
    const values = (history || []).map(p => p.close).filter(Number.isFinite);
    if (values.length < 2) return "";
    const low = Math.min(...values), range = Math.max(...values) - low || 1;
    const points = values.map((v, i) => `${(i / (values.length - 1) * 180).toFixed(1)},${(52 - (v - low) / range * 45).toFixed(1)}`).join(" ");
    return `<svg class="spark" viewBox="0 0 180 58" role="img" aria-label="最近60根收盘价走势，不表示未来预测"><polyline points="${points}"></polyline></svg>`;
  }
  function outdated(data) {
    const end = new Date(`${data.as_of}T15:30:00+08:00`);
    end.setUTCDate(end.getUTCDate() + 1);
    while ([0,6].includes(new Date(end.getTime() + 8 * 3600000).getUTCDay())) end.setUTCDate(end.getUTCDate() + 1);
    return Date.now() >= end.getTime();
  }
  function render(data) {
    if (data?.schema_version !== 3 || !data.selection || !data.validation) throw new Error("每日五选快照尚未就绪，请等待首次新版同步。");
    payload = data;
    const selection = data.selection, p = data.protocol, validation = data.validation;
    $("signal-date").textContent = data.as_of || "—";
    $("publish-time").textContent = "页面更新 " + new Date(data.published_at).toLocaleString("zh-CN", {timeZone:"Asia/Shanghai",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false});
    const live = data.state === "live" && !outdated(data);
    $("live-status").textContent = live ? "已冻结 · 等待后续验证" : "历史快照 · 以来源日期为准";
    $("live-status").className = `status ${live ? "" : "stale"}`;
    const issues = data.issues || [];
    $("notice").className = `notice ${live && !issues.length ? "" : "warn"}`;
    $("notice").textContent = !live ? `当前为 ${data.as_of} 的历史快照，不作为今天的新推荐。${issues[0] || "等待下一次完整同步。"}` : issues.length ? issues.join("；") : "本期名单与规则已冻结。次日开盘模拟执行，20个交易日后核对净收益与中证全指超额。";
    $("pick-count").textContent = selection.picks.length;
    $("record-count").textContent = validation.summary.signals;
    $("selection-summary").textContent = selection.reason;
    $("signal-download").hidden = !data.signal_file;
    if (data.signal_file) $("signal-download").href = data.signal_file;
    $("picks").innerHTML = selection.picks.length ? selection.picks.map((stock,i) => `
      <article class="pick-card"><div class="stock-heading"><div class="stock-identity"><span class="rank-number">${String(i+1).padStart(2,"0")}</span><div><h3>${escape(stock.name)}</h3><div class="stock-sub">${escape(stock.symbol)} · ${escape(stock.industry)}</div></div></div><div class="stock-score">${number(stock.score*100)}<small>相对评分 / 100</small></div></div>
      <div class="price-row"><div class="close-price"><span>${escape(data.as_of)} 收盘参考价</span><strong>${number(stock.close,2)}</strong><small>元</small></div>${spark(stock.history)}</div>
      <div class="stock-metrics"><div><span>中期趋势</span><strong class="${color(stock.trend_return)}">${pct(stock.trend_return,1)}</strong></div><div><span>60日单日波动</span><strong>${pct(stock.volatility_60,2).replace("+","")}</strong></div><div><span>近5日涨跌</span><strong class="${color(stock.return_5)}">${pct(stock.return_5,1)}</strong></div></div>
      <details><summary>为什么入选 · 查看评分拆解</summary><p>${escape(stock.reason)}</p>${stock.parts.map(part=>`<div class="part"><span>${escape(part.label)} · 权重 ${Math.round(part.weight*100)}%</span><strong>池内分位 ${Math.round(part.percentile*100)}% · 贡献 ${number(part.contribution*100)} 分</strong></div>`).join("")}</details></article>`).join("") : `<div class="empty"><h3>本期没有新增候选</h3><p>${escape(selection.reason)}</p><p>不为凑满数量放宽规则。空仓批次也会留证，并在观察期后与指数比较。</p></div>`;
    $("weights").innerHTML = (data.strategy || []).map(w=>`<div class="weight-row"><span>${escape(w.label)}</span><div class="weight-track" aria-hidden="true"><i style="width:${w.weight*100}%"></i></div><strong>${Math.round(w.weight*100)}%</strong></div>`).join("");
    $("method").innerHTML = `<ul><li>范围：沪深主板，按信号日名称排除ST和退市风险；上市满365天、至少250根日线。</li><li>收盘价≥3元，近20日日均成交额≥1亿元；中期收益为正，价格站上120日均线。</li><li>中期趋势＝20日前收盘÷120日前收盘−1；低波动＝60日收益标准差的反向分位；短期回撤＝5日收益的反向分位。</li><li>每个行业最多1只；市场站上120日均线的比例低于40%时不新增候选。</li><li>每批模拟资金10万元，按实际候选数等分；100股整手；单边成本20bps，每笔至少5元。</li><li>本期完整股票池 ${selection.pool_count} 只；市场均线广度 ${pct(selection.market_breadth,1).replace("+","")}；可交易股票行情覆盖 ${pct(selection.inputs?.bar_coverage,1).replace("+","")}；报价响应覆盖 ${pct(selection.inputs?.quote_coverage,1).replace("+","")}。</li></ul><p>版本 ${escape(p.version)} · 指纹 <code>${escape(data.protocol_hash)}</code></p>${issues.map(x=>`<p>${escape(x)}</p>`).join("")}`;
    const summary=validation.summary;
    $("validation-metrics").innerHTML = [
      ["已完成批次",String(summary.matured),`${summary.pending} 批观察中 · ${summary.data_gaps} 批待核对`,null],
      ["20日平均净收益",pct(summary.mean_net_return),"独立批次均值 · 已扣费用",summary.mean_net_return],
      ["平均指数超额",pct(summary.mean_excess),`${summary.paired} 批与中证全指同期间配对`,summary.mean_excess],
      ["跑赢指数比例",pct(summary.win_rate,1).replace("+",""),"仅统计已完成且基准齐全的批次",null]
    ].map(([title,value,note,n])=>`<div class="metric"><span>${title}</span><strong class="${color(n)}">${value}</strong><small>${escape(note)}</small></div>`).join("");
    $("acceptance").textContent = `${summary.acceptance}。观察门槛为60批成熟记录，且平均指数超额与同池超额均为正；重叠批次并非独立样本。`;
    $("acceptance").className = `notice ${summary.data_gaps || summary.blocked ? "warn" : ""}`;
    $("cohort-rows").innerHTML = validation.cohorts.length ? [...validation.cohorts].reverse().map(c=>`<tr><td>${escape(c.as_of)}</td><td>${c.picks.length}</td><td>${escape(statuses[c.status] || c.status)}</td><td class="${color(c.net_return)}">${pct(c.net_return)}</td><td class="${color(c.benchmark_return)}">${pct(c.benchmark_return)}</td><td class="${color(c.excess)}">${pct(c.excess)}</td><td><button type="button" data-cohort="${escape(c.signal_hash)}" aria-label="查看${escape(c.as_of)}批次明细">查看</button></td></tr>`).join("") : '<tr><td colspan="7" class="table-empty">尚无前向留证。首次完整同步后开始记录，不补造历史业绩。</td></tr>';
    $("validation-method").innerHTML = `<p>固定基准：${escape(p.benchmark_name)}，实际入场日开盘至实际全部退出日收盘。超额＝批次净收益−同期指数收益，单位为百分点。</p><p>平均同池超额：${pct(summary.mean_pool_excess)}；最差已完成批次：${pct(summary.worst_cohort)}。</p><ul>${validation.limitations.map(t=>`<li>${escape(t)}</li>`).join("")}</ul>`;
    $("version").textContent = `${p.version} · 无大模型依赖`;
    $("load-error").hidden = true;
    if(selectedCohort) detail(selectedCohort);
  }
  function detail(hash) {
    const c=payload.validation.cohorts.find(row=>row.signal_hash===hash);
    if(!c){selectedCohort=null;$("cohort-detail").innerHTML="";return;}
    selectedCohort = hash;
    $("cohort-detail").innerHTML = `<div class="cohort-detail"><h3>${escape(c.as_of)} · ${escape(statuses[c.status])}</h3><p>入场：${escape(c.entry_day || "等待次日")} · 目标退出：${escape(c.target_exit_day || "观察期未走完")} · 实际退出：${escape(c.exit_day || "尚未完成")}</p><p>实际模拟成交 ${c.filled} 只 · 已记费用 ${number(c.fees,2)} 元 · 同池近似超额 ${pct(c.pool_excess)}</p><ul>${(c.positions||[]).map(pos=>`<li>${escape(pos.symbol)}：${pos.quantity} 股；${escape(pos.reason)}；${pos.exit_day ? `已于 ${escape(pos.exit_day)} 退出` : escape(pos.status)}</li>`).join("")}${(c.issues||[]).map(issue=>`<li>${escape(issue)}</li>`).join("")}</ul><p>冻结指纹 <code>${escape(c.signal_hash)}</code></p></div>`;
    const names = c.picks.length ? c.picks.map(p=>`${escape(p.name || p.symbol)}（${escape(p.symbol)}）`).join("、") : "本期为空仓观察";
    $("cohort-detail").innerHTML += `<div class="cohort-detail"><p>当期候选：${names}</p><a class="text-link" href="signals/${encodeURIComponent(c.as_of)}__${encodeURIComponent((c.protocol_hash || payload.protocol_hash).slice(0,16))}.json" download>下载该期冻结推荐 ↗</a></div>`;
  }
  const tabs=Array.from(document.querySelectorAll("[data-tab]"));
  function activate(tab) {
    tabs.forEach(button=>{const selected=button===tab;button.setAttribute("aria-selected",String(selected));button.tabIndex=selected?0:-1;$(`${button.dataset.tab}-panel`).hidden=!selected;});
  }
  tabs.forEach((tab,i)=>{tab.addEventListener("click",()=>activate(tab));tab.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;event.preventDefault();const next=event.key==="Home"?0:event.key==="End"?tabs.length-1:(i+(event.key==="ArrowRight"?1:-1)+tabs.length)%tabs.length;activate(tabs[next]);tabs[next].focus();});});
  $("cohort-rows").addEventListener("click",event=>{const button=event.target.closest("[data-cohort]");if(button)detail(button.dataset.cohort);});
  async function refresh({silent = false} = {}) {
    if(refreshing) return;
    refreshing = true;
    lastRefreshAt = Date.now();
    const button=$("refresh");button.disabled=true;button.textContent="读取中…";
    try {const response=await fetch(`data.json?at=${Date.now()}`,{cache:"no-store"});if(!response.ok)throw new Error(`读取失败 HTTP ${response.status}`);render(await response.json());}
    catch(error){if(!silent || !payload){$("load-error").textContent=`数据刷新失败：${error.message}。已加载的快照仍保留。`;$("load-error").hidden=false;}}
    finally{refreshing=false;button.disabled=false;button.textContent="刷新数据";}
  }
  function refreshOnReturn() {
    if(!document.hidden && Date.now() - lastRefreshAt >= 60000){
      if(payload) render(payload);
      refresh({silent:true});
    }
  }
  $("refresh").addEventListener("click",()=>refresh());
  document.addEventListener("visibilitychange",refreshOnReturn);
  window.addEventListener("focus",refreshOnReturn);
  try {if(window.DAILY_PICKS)render(window.DAILY_PICKS);refresh({silent:!!payload});}
  catch(error){$("notice").textContent=error.message;$("notice").className="notice warn";}
})();
