// Read-only source audit with a minimal document stub. This is not a browser render test.
// Run from the project root: node docs/audits/audit_complexity.cjs
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '../..');
const html = fs.readFileSync(path.join(root, 'site/complexity.html'), 'utf8');
const js = fs.readFileSync(path.join(root, 'site/complexity.js'), 'utf8');
const css = fs.readFileSync(path.join(root, 'site/complexity.css'), 'utf8');
// Preserve the legacy audit below for its recorded baseline; summarize the new data schema.
if (js.includes('window.CX_SNAPSHOT')) {
  const payload = JSON.parse(fs.readFileSync(path.join(root,'site/complexity-data.json'),'utf8'));
  const checks = Object.values(payload.windows).map(w => {
    const valid = w.relations.filter(r => r.rho !== null);
    const connected = valid.filter(r => r.connected);
    const strongest = valid.reduce((best,r) => !best || Math.abs(r.rho) > Math.abs(best.rho) ? r : best,null);
    return {days:w.days, valid:valid.length, connected:connected.length,
      densityMatches: w.density === (valid.length ? connected.length/valid.length : null),
      strongestMatches: JSON.stringify(w.strongest) === JSON.stringify(strongest)};
  });
  process.stdout.write(JSON.stringify({schema:payload.schema_version,generated_at:payload.generated_at,
    basket_as_of:payload.basket.as_of,warnings:payload.warnings,checks,
    note:'Data consistency only. Run node --test tests/complexity_frontend.test.cjs for render/event stub checks; no browser layout verification.'},null,2)+'\n');
  process.exit(checks.every(c => c.densityMatches && c.strongestMatches) ? 0 : 1);
}
function attributes(tag) {
  return Object.fromEntries([...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map(m => [m[1], m[2]]));
}
function element(a, text = '') {
  const classes = new Set((a.class || '').split(/\s+/).filter(Boolean));
  return {
    attrs: {...a}, textContent: text, listeners: {},
    dataset: Object.fromEntries(Object.entries(a).filter(([k]) => k.startsWith('data-')).map(([k,v]) => [k.slice(5).replace(/-([a-z])/g, (_,c) => c.toUpperCase()), v])),
    classList: {remove(...items) { items.forEach(x => classes.delete(x)); }, add(x) { classes.add(x); }},
    setAttribute(k,v) { this.attrs[k] = v; },
    addEventListener(k,fn) { this.listeners[k] = fn; },
    classes,
  };
}
const nodes = [...html.matchAll(/<([a-z][\w-]*)\b([^>]*)>/gi)].map(m => element(attributes(m[0])));
const cells = [...html.matchAll(/<td\b([^>]*data-cell="[^"]+"[^>]*)>([^<]*)<\/td>/g)].map(m => element(attributes(m[1]), m[2]));
const ids = Object.fromEntries(nodes.filter(n => n.attrs.id).map(n => [n.attrs.id, n]));
const readouts = Object.fromEntries(nodes.filter(n => n.dataset.readout).map(n => [n.dataset.readout, n]));
const buttons = nodes.filter(n => n.dataset.window);
const edges = nodes.filter(n => n.dataset.edge);
const series = nodes.filter(n => n.dataset.series);
function query(selector) {
  if (selector.startsWith('#')) return ids[selector.slice(1)];
  const m = selector.match(/^\[data-readout="([^"]+)"\]$/);
  if (m) return readouts[m[1]];
  throw new Error(`Unsupported selector: ${selector}`);
}
const document = {
  querySelector: query,
  querySelectorAll(s) {
    const list = {'[data-window]':buttons, '[data-edge]':edges, '[data-cell]':cells}[s];
    if (!list) throw new Error(`Unsupported selector: ${s}`);
    return list;
  },
};
const context = vm.createContext({document});
vm.runInContext(js, context, {timeout: 1000});
const periods = vm.runInContext('cxPeriodData', context);
const initialCells = cells.map(c => ({key:c.dataset.cell, text:c.textContent, classes:[...c.classes]}));
const pairs = {'us-electronics':'electronics-us','us-newenergy':'newenergy-us','kr-electronics':'electronics-kr','kr-consumer':'consumer-kr','fx-electronics':'electronics-fx','fx-consumer':'consumer-fx','rates-newenergy':'newenergy-rates'};
const windows = [];
for (const days of [60,20,120,60]) {
  const data = periods[days];
  buttons.find(b => b.dataset.window === String(days)).listeners.click();
  const signedEdges = Object.fromEntries(edges.map(e => [e.dataset.edge, data.edges[e.dataset.edge] * (e.classes.has('cx-edge-negative') ? -1 : 1)]));
  const mismatches = Object.entries(pairs).filter(([edge,cell]) => Math.abs(signedEdges[edge]-data.heat[cell]) > 1e-9).map(([edge,cell]) => ({edge,cell,network:signedEdges[edge],heatmap:data.heat[cell]}));
  const changedColors = cells.filter(c => initialCells.find(i => i.key === c.dataset.cell).classes.join() !== [...c.classes].join()).map(c => ({key:c.dataset.cell, value:c.textContent, classes:[...c.classes]}));
  windows.push({days, activeButtons:buttons.filter(b => b.attrs['aria-pressed'] === 'true').map(b => b.dataset.window), readouts:Object.fromEntries(Object.entries(readouts).map(([k,n]) => [k,n.textContent])), edgeMaximum:Object.entries(signedEdges).sort((a,b) => Math.abs(b[1])-Math.abs(a[1]))[0], mismatches, cellClassesDifferentFromInitial:changedColors, uniqueEdgeCount:edges.length, unchangedSeriesPaths:series.every(n => n.attrs.d === html.match(new RegExp(`data-series="${n.dataset.series}" d="([^"]+)"`))[1])});
}
function seriesDetails(n) {
  const points = [...n.attrs.d.matchAll(/[ML](\d+) (\d+)/g)].map(m => [Number(m[1]),Number(m[2])]);
  return {key:n.dataset.series, plottedVertices:points.length, lastY:points.at(-1)[1], lastValueOnShown0to100Axis:(235-points.at(-1)[1])/2};
}
const output = {
  scope:'Local source and minimal-document-stub execution; no rendered browser, real DOM, layout, or provider account test.',
  sha256:Object.fromEntries([['html',html],['js',js],['css',css]].map(([k,v]) => [k,crypto.createHash('sha256').update(v).digest('hex')])),
  externalDataApiCallsInPageScript: /\b(fetch\s*\(|XMLHttpRequest|WebSocket)/.test(js),
  demoDisclosure:html.includes('均为虚拟样例'),
  windows,
  chart:series.map(seriesDetails),
  ratesEdgePath:edges.find(e => e.dataset.edge === 'rates-newenergy').attrs.d,
  ratesNode:html.match(/cx-node-rates[^>]*transform="([^"]+)"/)[1],
  smallFontDeclarations:[...css.matchAll(/font-size:\s*([89]|10)px/g)].length,
};
process.stdout.write(JSON.stringify(output,null,2)+'\n');
