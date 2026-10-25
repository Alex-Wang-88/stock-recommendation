// Minimal DOM stubs for render/event unit checks. No browser or layout simulation.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const code = fs.readFileSync(path.join(root, 'site/complexity.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'site/complexity.html'), 'utf8');
const snapshot = JSON.parse(fs.readFileSync(path.join(root, 'site/complexity-data.json'), 'utf8'));
test('local script bundle and downloadable JSON contain the same snapshot', () => {
  const context = vm.createContext({window:{}});
  vm.runInContext(fs.readFileSync(path.join(root,'site/complexity-data.js'),'utf8'),context,{timeout:1000});
  assert.equal(JSON.stringify(context.window.CX_SNAPSHOT),JSON.stringify(snapshot));
});

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this.dataset = {}; this.events = {}; this._text = ''; this.className = ''; }
  append(...nodes) { for (const n of nodes) { n.parent = this; this.children.push(n); } }
  replaceChildren(...nodes) { this.children = []; this._text = ''; this.append(...nodes); }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  get firstChild() { return this.children[0]; }
  setAttribute(k,v) { this.attrs[k] = v; if (k === 'class') this.className = v; }
  addEventListener(k,fn) { this.events[k] = fn; }
  click() { this.events.click?.(); }
  remove() { if(this.parent) this.parent.children = this.parent.children.filter(c => c !== this); }
  querySelectorAll(selector) {
    const all = this.children.flatMap(c => [c, ...c.querySelectorAll('*')]);
    return selector === '*' ? all : all.filter(c => c.tag === selector);
  }
}
function render(payload = structuredClone(snapshot)) {
  const ids = Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m => [m[1], new Element('div')]));
  const buttons = [...html.matchAll(/data-window="(\d+)"/g)].map(m => { const n = new Element('button'); n.dataset.window = m[1]; return n; });
  const viewButtons = [...html.matchAll(/data-view="([^"]+)"/g)].map(m => { const n = new Element('button'); n.dataset.view = m[1]; return n; });
  const expertBlocks = [...html.matchAll(/<details\s+data-expert\b/g)].map(() => new Element('details'));
  ids['cx-series'].value = 'breadth';
  const document = { querySelector: s => ids[s.slice(1)], querySelectorAll: s => {
    const selections={'[data-window]':buttons,'[data-view]':viewButtons,'[data-expert]':expertBlocks};
    assert.ok(selections[s],s); return selections[s];
  }, createElement: tag => new Element(tag), createElementNS: (_,tag) => new Element(tag) };
  const context = vm.createContext({ window:{CX_SNAPSHOT:payload}, document, Blob, URL, setTimeout });
  vm.runInContext(code, context, {timeout:3000});
  return { ids, buttons, viewButtons, expertBlocks };
}
function assertWindow(ui, days) {
  const n = snapshot.windows[String(days)];
  assert.equal(ui.buttons.filter(b => b.attrs['aria-pressed'] === 'true').length, 1);
  assert.equal(ui.buttons.find(b => b.attrs['aria-pressed'] === 'true').dataset.window, String(days));
  const paths = ui.ids['cx-network'].querySelectorAll('path');
  assert.equal(paths.length, n.connected_pairs);
  const cells = ui.ids['cx-heatmap'].querySelectorAll('td');
  assert.equal(cells.length, n.possible_pairs);
  for (const r of n.relations) {
    const key = `${r.factor}-${r.sector}`;
    const cell = cells.find(c => c.attrs['data-pair'] === key);
    const cls = r.rho === null || Math.abs(r.rho) < .1 ? 'cx-heat-neutral' : r.rho > 0 ? 'cx-heat-positive' : 'cx-heat-negative';
    assert.equal(cell.className, cls);
    if(r.connected) assert.equal(paths.find(p => p.attrs['data-pair'] === key).className, r.rho > 0 ? 'cx-edge-positive' : 'cx-edge-negative');
  }
  assert.equal(ui.ids['cx-history-table'].querySelectorAll('tbody')[0].children.length, Math.min(days, snapshot.basket.history.length));
  assert.match(ui.ids['cx-window-range'].textContent, new RegExp(`${days}日窗`));
  if(n.strongest) assert.match(ui.ids['cx-readouts'].textContent, new RegExp(`n=${n.strongest.n}`));
}
test('initial render and every window keep network, heat cells and history consistent', () => {
  const ui = render();
  assert.equal(ui.ids['cx-pulses'].children.length,6);
  assert.equal(ui.ids['cx-metrics'].children.length,4);
  assertWindow(ui,60);
  const initial = ui.ids['cx-heatmap'].textContent;
  ui.buttons.find(b=>b.dataset.window==='60').click();
  assert.equal(ui.ids['cx-heatmap'].textContent,initial);
  for(const days of [20,120,60]) { ui.buttons.find(b=>b.dataset.window===String(days)).click(); assertWindow(ui,days); }
  for(const metric of ['dispersion','hhi','synchrony','breadth']) {
    ui.ids['cx-series'].value=metric; ui.ids['cx-series'].events.change();
    assert.ok(ui.ids['cx-chart-caption'].textContent.includes(snapshot.basket.as_of));
  }
});
test('absent or unsupported snapshot shows missing state without fake prices', () => {
  for(const payload of [undefined,{schema_version:999}]) {
    const ui=render(payload===undefined?null:payload);
    assert.equal(ui.ids['cx-build-state'].textContent,'快照缺失');
    assert.equal(ui.ids['cx-pulses'].children.length,0);
  }
});
test('unavailable sources and empty basket render honestly without crashing', () => {
  const payload=structuredClone(snapshot);
  payload.warnings=['纳斯达克100：暂无可用数据'];
  payload.factors.forEach(s=>Object.assign(s,{value:null,change:null,status:'unavailable',as_of:null,previous_date:null,rows:0}));
  Object.assign(payload.basket,{history:[],latest:null,previous:null,as_of:null,available:0});
  Object.values(payload.windows).forEach(n=>{
    Object.assign(n,{valid_pairs:0,connected_pairs:0,density:null,average_strength:null,strongest:null});
    n.relations.forEach(r=>Object.assign(r,{rho:null,n:0,ci95:null,connected:false}));
  });
  const ui=render(payload);
  assert.match(ui.ids['cx-build-state'].textContent,/缺失/);
  assert.match(ui.ids['cx-pulses'].textContent,/不可用/);
  assert.equal(ui.ids['cx-network'].querySelectorAll('path').length,0);
  assert.match(ui.ids['cx-chart-caption'].textContent,/没有完整/);
  assert.match(ui.ids['cx-overview'].textContent,/暂不完整/);
  assert.match(ui.ids['cx-link-summary'].textContent,/不足/);
});
test('plain overview explains actual counts, and mode switching preserves the window', () => {
  const ui=render();
  const count=Math.round(snapshot.basket.latest.breadth*snapshot.basket.latest.sample/100);
  assert.ok(ui.ids['cx-overview'].textContent.includes(`${snapshot.basket.latest.sample}个方向，${count}个站上平均价`));
  assert.equal(ui.expertBlocks.length,3);
  assert.ok(ui.expertBlocks.every(d=>d.open===false));
  ui.viewButtons.find(b=>b.dataset.view==='expert').click();
  assert.ok(ui.expertBlocks.every(d=>d.open===true));
  ui.buttons.find(b=>b.dataset.window==='120').click();
  assert.ok(ui.expertBlocks.every(d=>d.open===true));
  assert.ok(ui.ids['cx-overview-date'].textContent.includes('120个交易日'));
  if(snapshot.windows['120'].connected_pairs===0) assert.ok(ui.ids['cx-link-summary'].textContent.includes('没有关系达到'));
  else assert.ok(ui.ids['cx-link-summary'].textContent.includes('历史样本'));
  ui.viewButtons.find(b=>b.dataset.view==='plain').click();
  assert.ok(ui.expertBlocks.every(d=>d.open===false));
  assert.equal(ui.buttons.find(b=>b.attrs['aria-pressed']==='true').dataset.window,'120');
});
test('turnover explanation derives a share rather than treating HHI as a percentage', () => {
  const ui=render();
  const amounts=snapshot.sectors.map(s=>({label:s.label,amount:snapshot.source_observations[s.id].find(r=>r.date===snapshot.basket.as_of).amount}));
  const top=amounts.sort((a,b)=>b.amount-a.amount)[0];
  const share=(top.amount/amounts.reduce((n,r)=>n+r.amount,0)*100).toFixed(1);
  assert.ok(ui.ids['cx-metrics'].textContent.includes(`${top.label}占这五只ETF当日成交额的${share}%`));
  assert.ok(ui.ids['cx-metrics'].textContent.includes('成交额不能直接当作资金净流入'));
  assert.ok(html.includes('不等于40%的上涨概率'));
  assert.ok(ui.ids['cx-chart-help'].textContent.includes('20个交易日'));
});
