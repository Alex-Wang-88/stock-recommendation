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

function run(data, clock='2026-11-02T18:00:00+08:00') {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id,{id,innerHTML:'',textContent:'',hidden:false,
      dataset:{},attributes:{},events:{},setAttribute(k,v){this.attributes[k]=v;},
      addEventListener(k,f){this.events[k]=f;},focus(){this.focused=true;}});
    return elements.get(id);
  }
  const tabs=['today','validation'].map(name=>{const el=element(name+'-tab');el.dataset.tab=name;return el;});
  class Clock extends Date {static now(){return new Date(clock).getTime();}}
  vm.runInNewContext(script,{window:{DAILY_PICKS:data},document:{getElementById:element,
    querySelectorAll:()=>tabs},Date:Clock,fetch:async()=>{throw Error('offline');}});
  return {get:element,tabs};
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
});
