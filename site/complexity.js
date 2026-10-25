/* All views use the same precomputed signed relations; no invented fallbacks. */
(() => {
  "use strict";
  const data = window.CX_SNAPSHOT;
  const $ = s => document.querySelector(s);
  const number = (v,d=2) => Number.isFinite(v) ? v.toFixed(d) : "—";
  const signed = (v,d=2) => Number.isFinite(v) ? `${v>0?"+":v<0?"−":""}${Math.abs(v).toFixed(d)}` : "—";
  const stamp = v => v && Number.isFinite(Date.parse(v)) ? new Date(v).toLocaleString("zh-CN",{timeZone:"Asia/Shanghai",hour12:false}) : "未知";
  function el(tag,text,cls) {
    const n=document.createElement(tag);
    if(text!==undefined)n.textContent=text;
    if(cls)n.className=cls;
    return n;
  }
  function svg(tag,attrs={},text) {
    const n=document.createElementNS("http://www.w3.org/2000/svg",tag);
    Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,String(v)));
    if(text!==undefined)n.textContent=text;
    return n;
  }
  function columns(table,labels) {
    const row=el("tr");
    labels.forEach(label=>{const n=el("th",label);n.scope="col";row.append(n);});
    const head=el("thead");head.append(row);table.append(head);
    const body=el("tbody");table.append(body);return body;
  }
  function exportCsv(filename,rows) {
    const quote=v=>`"${String(v??"").replaceAll('"','""')}"`;
    const blob=new Blob(["\uFEFF"+rows.map(r=>r.map(quote).join(",")).join("\r\n")],{type:"text/csv;charset=utf-8"});
    const url=URL.createObjectURL(blob),a=el("a");a.href=url;a.download=filename;a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  if(!data || data.schema_version!==1 || !data.windows || !Array.isArray(data.factors)) {
    $("#cx-build-state").textContent="快照缺失";
    $("#cx-generated").textContent="请先生成 complexity-data.js";
    $("#cx-message").textContent="没有可用快照，未显示任何示例行情。请运行项目快照更新脚本后刷新页面。";
    return;
  }
  let windowDays=60;
  const names=Object.fromEntries([...data.factors,...data.sectors].map(s=>[s.id,s.label]));
  function explanation(title,text) {
    const detail=el("details",undefined,"cx-explanation");
    detail.append(el("summary",title),el("p",text));return detail;
  }
  const warnings=[...data.warnings];
  if(Date.now()-Date.parse(data.generated_at)>36*3600*1000)warnings.unshift("页面快照生成已超过36小时，请检查更新任务。");
  $("#cx-build-state").textContent=warnings.length?"日频快照 · 含延迟或缺失":"日频快照已接入";
  $("#cx-generated").textContent=`生成于 ${stamp(data.generated_at)}（中国时间）`;
  $("#cx-message").textContent=warnings.length?warnings.join("；"):"已加载真实来源日值；完整源日期、抓取时间和口径见下方。";
  const factorReading={
    ndx:{label:"美国股市参考",help:"看纳斯达克100指数相对前一有效交易日的涨跌。它可以提供海外市场背景，不能直接换算成A股下一天的涨跌。"},
    kospi:{label:"韩国股市参考",help:"看KOSPI指数相对前一有效交易日的涨跌。请看清本卡的日期，它可能与美股卡片的日期不同。"},
    cny:{label:"人民币汇率",help:"这个数字表示1美元能换多少人民币。数字下降，表示人民币相对美元走强；数字上升则相反。这是每日参考汇率。"},
    gold:{label:"黄金价格（期货）",help:"看国际黄金近月期货的涨跌，报价单位为美元/盎司。它与金店零售价口径不同，合约换月也可能影响变化。"},
    brent:{label:"原油价格（期货）",help:"看布伦特近月原油期货的涨跌，报价单位为美元/桶。它与加油站零售价口径不同，合约换月也可能影响变化。"},
    ust10y:{label:"美国国债收益率",help:"这里看的是收益率的升降。1个基点等于0.01个百分点，例如4.00%到4.01%是上升1个基点。"},
  };
  data.factors.forEach(s=>{
    const reading=factorReading[s.id]||{label:s.label,help:"先看来源日期，再比较前一个有效日的变化。"};
    const card=el("article",undefined,"cx-pulse-card");
    let headline=Number.isFinite(s.change)?`${signed(s.change)}%`:"—";
    if(s.kind==="yield")headline=Number.isFinite(s.change)?`${s.change>0?"上升":s.change<0?"下降":"变动"}${number(Math.abs(s.change),0)}个基点`:"—";
    if(s.kind==="fx")headline=number(s.value,4);
    card.append(el("h3",reading.label,"cx-pulse-heading"),el("strong",headline,"cx-pulse-value"));
    const quote=s.kind==="fx"?"1美元兑换的人民币金额":`${s.label}：${number(s.value)} ${s.unit}`;
    card.append(el("span",quote,"cx-card-unit"));
    const change=s.kind==="fx"?`美元兑人民币报价较上次 ${signed(s.change)}%`:`与 ${s.previous_date||"未知日期"} 比较`;
    card.append(el("p",change,"cx-card-change"),el("p",`数据日期 ${s.as_of||"缺失"}`,"cx-card-date"));
    card.append(el("p",s.status==="unavailable"?"不可用":s.status==="cached"?"上次成功缓存":s.stale?"源日期较旧":"完整日值","cx-card-state"));
    card.append(explanation("怎么读？",reading.help));
    $("#cx-pulses").append(card);
  });
  const latest=data.basket.latest,previous=data.basket.previous;
  $("#cx-basket-date").textContent=`${data.basket.as_of||"无共同完整日"} · 覆盖 ${data.basket.available}/${data.basket.expected}`;
  const basketReadings=data.sectors.map(s=>{
    const rows=data.source_observations?.[s.id]||[];
    const current=rows.find(r=>r.date===latest?.date),prior=rows.find(r=>r.date===previous?.date);
    return {id:s.id,label:s.label,amount:current?.amount,
      change:current&&prior&&prior.close>0?(current.close/prior.close-1)*100:null};
  });
  const amountsComplete=basketReadings.length===data.basket.expected&&basketReadings.every(s=>Number.isFinite(s.amount)&&s.amount>=0);
  const totalAmount=amountsComplete?basketReadings.reduce((sum,s)=>sum+s.amount,0):0;
  const busiest=totalAmount>0?[...basketReadings].sort((a,b)=>b.amount-a.amount)[0]:null;
  const leadingShare=busiest?busiest.amount/totalAmount*100:null;
  const countAbove=Number.isFinite(latest?.breadth)?Math.round(latest.breadth*latest.sample/100):null;
  const syncCount=Number.isFinite(latest?.synchrony)?Math.round(latest.synchrony*latest.synchrony_pairs):null;
  const validChanges=basketReadings.filter(s=>Number.isFinite(s.change)).sort((a,b)=>b.change-a.change);
  const gap=validChanges.length===data.basket.expected?validChanges[0].change-validChanges.at(-1).change:null;
  const metrics=[
    {key:"breadth",label:"有几个方向站上平均价？",unit:"%",digits:1,
      headline:countAbove===null?"—":`${countAbove} / ${latest.sample}个`,
      note:countAbove===null?"缺少完整样本，暂时无法判断。":`五只ETF中，有${countAbove}只高于自己最近20个交易日的平均收盘价。`,
      help:"原指标叫“篮子站上20日均线比例”。比例越高，说明样本中站上这条平均线的方向越多；它描述当前价格位置，不是下一天的上涨概率。"},
    {key:"dispersion",label:"几个方向的涨跌差别大吗？",unit:" 个百分点",digits:2,
      note:Number.isFinite(gap)?`本期表现最高与最低的ETF，价格收益相差${number(gap)}个百分点。`:"缺少完整的涨跌对比数据。",
      help:"原指标叫“日收益离散度”，用五只ETF价格收益的标准差衡量差别。下面显示的标准差与最高、最低之间的差值不是同一个数。数值小也可能是一起下跌。",percentile:"dispersion_percentile"},
    {key:"hhi",label:"成交更集中在哪里？",unit:"",digits:3,
      headline:busiest?`${number(leadingShare,1)}%`:"—",
      note:busiest?`${busiest.label}占这五只ETF当日成交额的${number(leadingShare,1)}%，占比最大。`:"成交额信息不完整，暂不比较集中位置。",
      help:"卡片大字是成交最多的一只ETF所占的份额。原指标“HHI”把五只ETF的成交份额平方后相加，越高表示越集中。成交额不能直接当作资金净流入。",percentile:"hhi_percentile"},
    {key:"synchrony",label:"这几个方向的走势有多少联系？",unit:"",digits:2,
      headline:syncCount===null?"—":`${syncCount} / ${latest.synchrony_pairs}组`,
      note:syncCount===null?"历史样本不足，暂不判断。":`最近60个交易日的${latest.synchrony_pairs}组两两比较中，${syncCount}组达到本页的连线门槛。`,
      help:"原指标叫“60日篮子同步密度”。联系包括同向和反向，按相关系数绝对值至少0.30来计数；这只是观察标准，不是上涨胜率。"},
  ];
  metrics.forEach(m=>{
    const card=el("article",undefined,"cx-metric-card");
    const main=m.headline??`${number(latest?.[m.key],m.digits)}${m.unit}`;
    card.append(el("h3",m.label,"cx-metric-label"),el("strong",main,"cx-metric-value"),el("p",m.note));
    const delta=latest&&previous&&Number.isFinite(latest[m.key])&&Number.isFinite(previous[m.key])?signed(latest[m.key]-previous[m.key],m.digits):"—";
    const help=explanation("这是什么意思？",m.help);
    help.append(el("p",`原始指标：${number(latest?.[m.key],m.digits)}${m.unit}；较前一共同交易日 ${delta}${m.key==="breadth"?" 个百分点":m.unit}。`,"cx-metric-foot"));
    if(m.percentile)help.append(el("p",Number.isFinite(latest?.[m.percentile])?`原始指标在此前最多252日：第 ${number(latest[m.percentile],0)} 百分位。含义见下方小词典。`:"历史位置的参照记录还不够（至少需60个）。","cx-metric-foot"));
    card.append(help);
    $("#cx-metrics").append(card);
  });
  function renderPlainReading(network) {
    const overview=$("#cx-overview");overview.replaceChildren();
    $("#cx-overview-date").textContent=`A股样本日期：${data.basket.as_of||"暂无完整数据"}；历史联系回看${windowDays}个交易日。`;
    const connected=network.relations.filter(r=>r.connected&&Number.isFinite(r.rho)).sort((a,b)=>Math.abs(b.rho)-Math.abs(a.rho));
    const pieces=[
      ["样本强弱",countAbove===null?"样本暂不完整":`${latest.sample}个方向，${countAbove}个站上平均价`,countAbove===null?"缺少完整数据时不补猜测值。":"这里的平均价指各自最近20个交易日的平均收盘价。"],
      ["成交分布",busiest?`${busiest.label}成交占比最大`:"成交数据暂不完整",busiest?`占这五只ETF成交额的${number(leadingShare,1)}%。只能说明样本里的交易集中位置。`:"等五只ETF的成交信息齐全后再比较。"],
      ["历史联系",network.valid_pairs?`${network.valid_pairs}组比较，${connected.length}组达到观察门槛`:"历史配对暂不够",`回看${windowDays}个交易日。门槛是本页的研究设定，不能解读为买入信号。`],
    ];
    pieces.forEach(([label,headline,note])=>{const card=el("article",undefined,"cx-overview-card");card.append(el("p",label,"cx-overview-label"),el("h3",headline),el("p",note));overview.append(card);});
    const summary=$("#cx-link-summary");summary.replaceChildren();
    if(!connected.length){
      summary.append(el("p",network.valid_pairs?`回看${windowDays}个交易日，没有关系达到本页的连线门槛。较弱的关系仍可在专业明细中查看。`:"可配对的历史样本不足，暂时无法比较海外与这些ETF的联系。"));return;
    }
    connected.slice(0,3).forEach(r=>{
      const card=el("article",undefined,"cx-link-card");
      const factor=r.factor==="cny"?"美元兑人民币报价":names[r.factor];
      card.append(el("span",r.rho>0?"历史同向线索":"历史反向线索",r.rho>0?"cx-key-positive":"cx-key-negative"),el("h3",`${factor} → ${names[r.sector]}`));
      card.append(el("p",`前者较早时段的变化，与后者之后的涨跌，在这${r.n}组历史样本中偏${r.rho>0?"同向":"反向"}变化。`));
      const uncertain=r.ci95&&r.ci95[0]<=0&&r.ci95[1]>=0;
      card.append(el("p",uncertain?"估计区间还包含零，关系方向仍有不确定性。":"这是样本里的统计联系，不能保证下一次仍会出现。","cx-reading-caution"));
      card.append(explanation("查看数字依据",`相关系数 ${signed(r.rho)}；${r.n}个有效配对；近似95%区间 ${r.ci95?`${signed(r.ci95[0])} 至 ${signed(r.ci95[1])}`:"不可估计"}。未校正时间依赖和多重比较。`));
      summary.append(card);
    });
    summary.append(el("p",`显示达到门槛的${connected.length}组关系中，按系数绝对值排序的前${Math.min(3,connected.length)}组；全部结果见专业明细。`,"cx-chart-note"));
  }
  function renderNetwork(network) {
    const graph=$("#cx-network");graph.querySelectorAll("g").forEach(n=>n.remove());
    const factorY=Object.fromEntries(data.factors.map((s,i)=>[s.id,35+i*70]));
    const sectorY=Object.fromEntries(data.sectors.map((s,i)=>[s.id,60+i*80]));
    const lines=svg("g",{class:"cx-network-lines",fill:"none"});
    network.relations.filter(r=>r.connected).forEach(r=>{
      const a=factorY[r.factor],b=sectorY[r.sector];
      const edge=svg("path",{d:`M260 ${a} C410 ${a} 500 ${b} 640 ${b}`,class:r.rho>0?"cx-edge-positive":"cx-edge-negative","stroke-width":1+Math.abs(r.rho)*7,"data-pair":`${r.factor}-${r.sector}`});
      edge.append(svg("title",{},`${names[r.factor]} → ${names[r.sector]}：${signed(r.rho)}，n=${r.n}`));lines.append(edge);
    });graph.append(lines);
    const nodes=svg("g");
    [...data.factors.map(s=>({...s,x:10,y:factorY[s.id],external:true})),...data.sectors.map(s=>({...s,x:640,y:sectorY[s.id],external:false}))].forEach(s=>{
      const n=svg("g",{class:`cx-network-node ${s.external?"cx-node-external":"cx-node-sector"}`,transform:`translate(${s.x} ${s.y-20})`});
      n.append(svg("rect",{width:250,height:40,rx:9}),svg("text",{x:12,y:25},`${s.label} ${s.external?"":s.symbol.slice(2)}`));nodes.append(n);
    });graph.append(nodes);
    const dl=$("#cx-readouts");dl.replaceChildren();const top=network.strongest;
    [["有效关系",`${network.valid_pairs} / ${network.possible_pairs}`],["超过阈值的关系",String(network.connected_pairs)],["跨市场网络密度",number(network.density)],["连边平均 |ρ|",number(network.average_strength)],["最强关系（按 |ρ|）",top?`${names[top.factor]} → ${names[top.sector]}`:"样本不足"],["最强关系系数 / 样本",top?`${signed(top.rho)} / n=${top.n}`:"—"]].forEach(([k,v])=>dl.append(el("dt",k),el("dd",v)));
  }
  function renderHeatmap(network) {
    const table=$("#cx-heatmap");table.replaceChildren(el("caption",`${network.days}日窗 · ${network.start||"—"} 至 ${network.end||"—"}`));
    const body=columns(table,["A股代理ETF",...data.factors.map(s=>s.label)]);
    const interval=$("#cx-interval-table");interval.replaceChildren(el("caption","Fisher近似95%区间，未校正自相关与多重比较"));
    const details=columns(interval,["外部因素","代理ETF","ρ","n","近似95%区间"]);
    data.sectors.forEach(s=>{
      const tr=el("tr"),th=el("th",`${s.label} ${s.symbol.slice(2)}`);th.scope="row";tr.append(th);
      data.factors.forEach(f=>{
        const r=network.relations.find(v=>v.sector===s.id&&v.factor===f.id);
        const cls=r.rho===null||Math.abs(r.rho)<.1?"cx-heat-neutral":r.rho>0?"cx-heat-positive":"cx-heat-negative";
        const td=el("td",undefined,cls);td.setAttribute("data-pair",`${r.factor}-${r.sector}`);
        td.append(el("strong",signed(r.rho)),el("small",`n=${r.n}`));
        const ci=r.ci95?`${signed(r.ci95[0])} 至 ${signed(r.ci95[1])}`:"不可估计";
        td.title=`${f.label} × ${s.label}；ρ=${signed(r.rho)}；n=${r.n}；近似95%区间 ${ci}`;tr.append(td);
        const row=el("tr");[f.label,s.label,signed(r.rho),r.n,ci].forEach(v=>row.append(el("td",String(v))));details.append(row);
      });body.append(tr);
    });
  }
  function renderHistory() {
    const key=$("#cx-series").value,m=metrics.find(v=>v.key===key),history=data.basket.history.slice(-windowDays);
    const guides={breadth:"线越高，表示五只ETF中，高于自己近20个交易日平均价的样本越多。",dispersion:"线越高，表示几个方向的价格收益差别越大；线低也可能是一起下跌。",hhi:"线越高，表示这五只ETF的成交额越集中在少数几只上；五只成交额相等时为0.20。",synchrony:"线越高，表示这几个方向中达到连线门槛的历史关系越多，包括同向和反向；它不是上涨概率。"};
    $("#cx-chart-help").textContent=guides[key];
    const caption=history.length?`${history[0].date} 至 ${history.at(-1).date} · ${history.length} 个有效观测 · ${m.label}（${m.unit.trim()||"0–1"}）`:"没有完整篮子历史";
    $("#cx-chart-caption").textContent=caption;
    const chart=$("#cx-history-chart");chart.replaceChildren(svg("title",{id:"cx-chart-title"},caption+"；精确值见后续数据表"));
    const values=history.map(r=>r[key]).filter(Number.isFinite);
    if(values.length){
      let max=key==="breadth"?100:["hhi","synchrony"].includes(key)?1:Math.max(...values)*1.15;if(max===0)max=1;
      const x=i=>60+i*810/Math.max(1,history.length-1),y=v=>235-v/max*210;
      for(let i=0;i<=4;i++){
        const v=max*i/4;
        chart.append(svg("line",{x1:60,y1:y(v),x2:870,y2:y(v),stroke:"#31445b","stroke-dasharray":"3 5"}),svg("text",{x:4,y:y(v)+4,fill:"#b7c5d8","font-size":13},number(v,key==="breadth"?0:2)));
      }
      let d="",pen=false;history.forEach((r,i)=>{if(!Number.isFinite(r[key])){pen=false;return;}d+=`${pen?"L":"M"}${x(i).toFixed(1)} ${y(r[key]).toFixed(1)} `;pen=true;});
      chart.append(svg("path",{d,fill:"none",stroke:"#6fd3b6","stroke-width":2.5,"vector-effect":"non-scaling-stroke"}));
      chart.append(svg("text",{x:60,y:266,fill:"#b7c5d8","font-size":13},history[0].date),svg("text",{x:870,y:266,"text-anchor":"end",fill:"#b7c5d8","font-size":13},history.at(-1).date));
    }
    const table=$("#cx-history-table");table.replaceChildren(el("caption",caption));
    const body=columns(table,["日期","站上20日均线比例（%）","收益离散度（百分点）","成交额HHI","60日同步密度"]);
    history.forEach(r=>{const tr=el("tr");[r.date,number(r.breadth,1),number(r.dispersion,3),number(r.hhi,4),number(r.synchrony,3)].forEach(v=>tr.append(el("td",v)));body.append(tr);});
  }
  function setWindow(days) {
    if(!data.windows[String(days)])return;
    windowDays=Number(days);const n=data.windows[String(days)];
    document.querySelectorAll("[data-window]").forEach(b=>b.setAttribute("aria-pressed",String(Number(b.dataset.window)===windowDays)));
    $("#cx-window-range").textContent=`${windowDays}日窗：${n.start||"—"} 至 ${n.end||"—"}，回看${n.observed_days}个共同交易日；每组能比较的天数可能不同。`;
    renderNetwork(n);renderHeatmap(n);renderHistory();renderPlainReading(n);
  }
  const sourceBody=columns($("#cx-sources"),["系列 / 标识","来源","最近源日期","前一源日期","日线条数","最近成功抓取（中国时间）","状态"]);
  [...data.factors,...data.sectors].forEach(s=>{
    const tr=el("tr"),th=el("th",`${s.label} / ${s.symbol}`);th.scope="row";tr.append(th);
    const td=el("td"),link=el("a",s.provider);link.href=s.source_url;link.target="_blank";link.rel="noreferrer";td.append(link);tr.append(td);
    const state=s.status==="unavailable"?"不可用":s.status==="cached"?"旧缓存（本次抓取失败）":s.stale?"可用但源日期较旧":"已接入 / 日频";
    [s.as_of||"—",s.previous_date||"—",String(s.rows),stamp(s.fetched_at),state].forEach(v=>tr.append(el("td",v)));sourceBody.append(tr);
  });
  document.querySelectorAll("[data-window]").forEach(b=>{b.disabled=false;b.addEventListener("click",()=>setWindow(b.dataset.window));});
  $("#cx-series").addEventListener("change",renderHistory);
  $("#cx-export-correlations").disabled=false;
  $("#cx-export-correlations").addEventListener("click",()=>{
    const n=data.windows[String(windowDays)];
    exportCsv(`complexity-correlations-${windowDays}d.csv`,[["window","start","end","factor","symbol","ETF","ETF_symbol","rho","n","ci95_low","ci95_high","connected","threshold"],...n.relations.map(r=>[windowDays,n.start,n.end,names[r.factor],data.factors.find(f=>f.id===r.factor).symbol,names[r.sector],data.sectors.find(s=>s.id===r.sector).symbol,r.rho,r.n,r.ci95?.[0],r.ci95?.[1],r.connected,data.method.threshold])]);
  });
  $("#cx-export-history").disabled=false;
  $("#cx-export-history").addEventListener("click",()=>exportCsv(`complexity-basket-${windowDays}d.csv`,[["date","breadth_pct","dispersion_percentage_points","ETF_amount_HHI","synchrony_density"],...data.basket.history.slice(-windowDays).map(r=>[r.date,r.breadth,r.dispersion,r.hhi,r.synchrony])]));
  function setReadingView(view) {
    document.querySelectorAll("[data-view]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.view===view)));
    document.querySelectorAll("[data-expert]").forEach(d=>{d.open=view==="expert";});
    $("#cx-reading-hint").textContent=view==="expert"?"专业关系图、来源与计算公式已展开，通俗解读仍保留在前面。":"先看文字解读；专业明细已收起，也可单独展开。";
  }
  document.querySelectorAll("[data-view]").forEach(b=>{b.disabled=false;b.addEventListener("click",()=>setReadingView(b.dataset.view));});
  setReadingView("plain");
  setWindow(60);
})();
