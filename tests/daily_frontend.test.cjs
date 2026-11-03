const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const script = fs.readFileSync(path.join(__dirname, '../site/daily.js'), 'utf8');

function fixture() {
  const pick = {symbol:'600001',name:'测试公司',industry:'行业A',close:10,score:.78,
    trend_return:.12,volatility_60:.01,return_5:-.03,reason:'固定规则',
    history:[{close:9},{close:10}],parts:[{key:'trend',label:'中期趋势',weight:.4,percentile:.8,contribution:.32}]};
  return {schema_version:3,as_of:'2026-11-02',published_at:'2026-11-02T17:00:00+08:00',state:'live',
    protocol:{version:'daily-test',benchmark_name:'中证全指价格指数'},protocol_hash:'abc',
    signal_file:'signals/frozen.json',issues:[],strategy:[{label:'中期趋势',weight:.4}],
    selection:{picks:[pick],reason:'入选1只',pool_count:100,market_breadth:.6,
      inputs:{bar_coverage:1,quote_coverage:1}},
    validation:{summary:{signals:1,matured:0,pending:1,data_gaps:0,blocked:0,paired:0,
      mean_net_return:null,mean_excess:null,mean_pool_excess:null,win_rate:null,worst_cohort:null,
      acceptance:'积累样本中'},cohorts:[{as_of:'2026-11-02',picks:[pick],status:'waiting_entry',
      signal_hash:'abc',net_return:null,benchmark_return:null,excess:null,positions:[],issues:[],
      fees:0,filled:0,pool_excess:null}],limitations:['批次不能复利年化']}};
}

function run(data, clock='2026-11-02T18:00:00+08:00', options={}) {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id,{id,innerHTML:'',textContent:'',hidden:false,
      dataset:{},attributes:{},events:{},setAttribute(k,v){this.attributes[k]=v;},
      addEventListener(k,f){this.events[k]=f;},focus(){this.focused=true;}});
    return elements.get(id);
  }
  const tabs=['today','validation'].map(name=>{const el=element(name+'-tab');el.dataset.tab=name;return el;});
  const windowEvents={},documentEvents={},calls=[];
  const document={getElementById:element,querySelectorAll:()=>tabs,hidden:false,
    addEventListener(k,f){documentEvents[k]=f;}};
  const window={DAILY_PICKS:data,addEventListener(k,f){windowEvents[k]=f;}};
  class Clock extends Date {static now(){return new Date(clock).getTime();}}
  vm.runInNewContext(script,{window,document,Date:Clock,fetch:async(url,init)=>{
    calls.push({url,init});
    if(options.fetch) return options.fetch(url,init);
    throw Error('offline');
  }});
  return {get:element,tabs,document,windowEvents,documentEvents,calls,setClock(value){clock=value;}};
}

test('candidate scores, reasons, weights and unavailable returns render honestly',()=>{
  const {get}=run(fixture());
  assert.match(get('picks').innerHTML,/测试公司/);
  assert.match(get('picks').innerHTML,/相对评分/);
  assert.match(get('picks').innerHTML,/78.0/);
  assert.match(get('validation-metrics').innerHTML,/—/);
  assert.doesNotMatch(get('validation-metrics').innerHTML,/NaN|undefined/);
  assert.match(get('live-status').textContent,/已冻结/);
});

test('cash signals and waiting observations do not appear as completed zero returns',()=>{
  const data=fixture();data.selection.picks=[];data.selection.reason='市场门槛不足';
  const {get}=run(data);
  assert.match(get('picks').innerHTML,/本期没有新增候选/);
  assert.match(get('cohort-rows').innerHTML,/等待入场日留证/);
  assert.doesNotMatch(get('cohort-rows').innerHTML,/0.00%/);
});

test('old page snapshots become visibly stale without waiting for the next build',()=>{
  const {get}=run(fixture(),'2026-11-04T12:00:00+08:00');
  assert.match(get('live-status').textContent,/历史快照/);
  assert.match(get('notice').textContent,/不作为今天的新推荐/);
});

test('tab keyboard navigation preserves selection and exposes the ledger',()=>{
  const {get,tabs}=run(fixture());
  let prevented=false;
  tabs[0].events.keydown({key:'ArrowRight',preventDefault(){prevented=true;}});
  assert.equal(prevented,true);
  assert.equal(get('today-panel').hidden,true);
  assert.equal(get('validation-panel').hidden,false);
  assert.equal(tabs[1].attributes['aria-selected'],'true');
  assert.equal(tabs[1].focused,true);
});

test('source-controlled company names are escaped before HTML insertion',()=>{
  const data=fixture();data.selection.picks[0].name='<img src=x onerror=alert(1)>';
  const {get}=run(data);
  assert.doesNotMatch(get('picks').innerHTML,/<img/);
  assert.match(get('picks').innerHTML,/&lt;img/);
});

test('clicking a cohort reveals fill and evidence details',()=>{
  const {get}=run(fixture());
  get('cohort-rows').events.click({target:{closest:()=>({dataset:{cohort:'abc'}})}});
  assert.match(get('cohort-detail').innerHTML,/等待入场日留证/);
  assert.match(get('cohort-detail').innerHTML,/冻结指纹/);
  assert.match(get('cohort-detail').innerHTML,/当期候选：测试公司（600001）/);
  assert.match(get('cohort-detail').innerHTML,/下载该期冻结推荐/);
});

const settled=()=>new Promise(resolve=>setImmediate(resolve));
test('opening a cached page immediately fetches its latest JSON without browser cache',async()=>{
  const latest=fixture();latest.as_of='2026-11-03';latest.selection.picks[0].name='新快照公司';
  const view=run(fixture(),undefined,{fetch:async()=>({ok:true,json:async()=>latest})});
  assert.match(view.get('picks').innerHTML,/测试公司/);
  await settled();
  assert.equal(view.calls.length,1);
  assert.equal(view.calls[0].init.cache,'no-store');
  assert.match(view.get('picks').innerHTML,/新快照公司/);
  assert.equal(view.get('signal-date').textContent,'2026-11-03');
});

test('returning after several days refreshes once and keeps the validation tab selected',async()=>{
  let latest=fixture();
  const view=run(fixture(),undefined,{fetch:async()=>({ok:true,json:async()=>latest})});
  await settled();
  view.tabs[1].events.click();
  view.document.hidden=true;view.setClock('2026-11-05T18:00:00+08:00');
  view.documentEvents.visibilitychange();
  assert.equal(view.calls.length,1);
  latest=fixture();latest.as_of='2026-11-05';
  view.document.hidden=false;view.documentEvents.visibilitychange();view.windowEvents.focus();
  await settled();
  assert.equal(view.calls.length,2);
  assert.equal(view.get('signal-date').textContent,'2026-11-05');
  assert.equal(view.get('validation-panel').hidden,false);
  view.windowEvents.focus();
  assert.equal(view.calls.length,2);
});

test('offline automatic refresh keeps the snapshot and manual retry explains the failure',async()=>{
  const view=run(fixture());
  await settled();
  assert.equal(view.get('load-error').hidden,true);
  assert.match(view.get('picks').innerHTML,/测试公司/);
  view.setClock('2026-11-05T18:00:00+08:00');view.windowEvents.focus();
  await settled();
  assert.match(view.get('live-status').textContent,/历史快照/);
  await view.get('refresh').events.click();
  assert.equal(view.get('load-error').hidden,false);
  assert.match(view.get('load-error').textContent,/已加载的快照仍保留/);
  assert.equal(view.get('refresh').disabled,false);
});
