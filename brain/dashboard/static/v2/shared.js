/* ===== dashboardV2 HONEST-RENDER TOOLKIT — window.V2 (static, no restart) =====
 * One source of truth for the honesty primitives every /v2 page depends on:
 * fail-CLOSED gate logic, self/grounded/earned tagging, as-of stamps, band colors.
 * If a number is null/missing/stale we render UNKNOWN, never green, never zero. */
window.V2 = (function(){
  "use strict";
  function fetchJSON(u){ return fetch(u).then(function(r){ if(!r.ok) throw new Error(u+' '+r.status); return r.json(); }); }
  function num(v){ return (v===null||v===undefined||(typeof v==='number'&&isNaN(v)))?null:+v; }
  function pct1(v){ v=num(v); return v===null?'—':v.toFixed(1); }
  function f2(v){ v=num(v); return v===null?'—':v.toFixed(2); }
  function f3(v){ v=num(v); return v===null?'—':v.toFixed(3); }
  function el(id){ return document.getElementById(id); }
  // fail-CLOSED: explicit true => 'ok'; explicit false => 'fail'; anything else => 'unk'.
  function gateState(v){ if(v===true) return 'ok'; if(v===false) return 'fail'; return 'unk'; }
  // relative age from epoch-seconds. (client clock; acceptable for as-of display)
  function ago(ts){ ts=num(ts); if(!ts) return '—'; var s=Math.max(0,(Date.now()/1000)-ts);
    if(s<90) return Math.round(s)+'s ago'; if(s<5400) return Math.round(s/60)+'m ago'; return (s/3600).toFixed(1)+'h ago'; }
  // a value -> CSS color by good/mid thresholds; null => dim (not green).
  function bandColor(v,good,mid){ v=num(v); if(v===null) return 'var(--dim)';
    if(v>=good) return 'var(--green)'; if(v>=mid) return 'var(--amber)'; return 'var(--red)'; }
  function tag(cls,txt){ return '<span class="tag '+cls+'">'+txt+'</span>'; }
  function fmtUptime(s){ s=num(s); if(s===null) return '—'; return s/3600>=24?(s/86400).toFixed(1)+'d':(s/3600).toFixed(1)+'h'; }
  function cap(str){ return String(str||'').replace(/_/g,' '); }
  // ---- centralized nav: add a page here once; every page picks it up ----
  // [href, label, stub?] — drop the stub flag when a page ships.
  var NAVPAGES=[
    ['/static/v2/integrity.html','integrity'],
    ['/static/v2/maturity.html','maturity'],
    ['/static/v2/yardsticks.html','yardsticks'],
    ['/static/v2/memory.html','memory'],
    ['/static/v2/capability.html','capability'],
    ['/static/v2/cognition.html','cognition'],
    ['/static/v2/identity.html','identity'],
    ['/static/v2/spatial.html','spatial'],
    ['/static/v2/ops.html','ops']
  ];
  // render the shared nav into <nav id="v2nav"></nav>, marking `active`.
  function renderNav(active){
    var nav=document.getElementById('v2nav'); if(!nav) return;
    var html=NAVPAGES.map(function(p){
      var on=(p[0]===active), stub=p[2];
      return '<a class="'+(on?'on':(stub?'stub':''))+'" href="'+(stub?'#':p[0])+'">'+p[1]+'</a>';
    }).join('');
    html+='<a href="/mind">/mind ↗</a><a href="/">← v1 dashboard</a>';
    nav.innerHTML=html;
  }
  // legacy: highlight an already-rendered nav by href match.
  function markNav(route){ var as=document.querySelectorAll('nav a'); for(var i=0;i<as.length;i++){ if(as[i].getAttribute('href')===route) as[i].className='on'; } }
  // generic progress bar row (label | track | value). pct 0..100; color optional.
  function barRow(label, pct, valText, color){
    pct=num(pct);
    if(pct===null){ return '<div class="barrow"><span class="bl">'+label+'</span><div class="track hatch"></div><span class="bv unpop">'+(valText||'—')+'</span></div>'; }
    var w=Math.max(0,Math.min(100,pct));
    return '<div class="barrow"><span class="bl">'+label+'</span><div class="track"><div class="fill" style="width:'+w.toFixed(0)+'%;background:'+(color||'var(--cyan)')+'"></div></div><span class="bv">'+(valText||'')+'</span></div>';
  }
  return { fetchJSON, num, pct1, f2, f3, el, gateState, ago, bandColor, tag, fmtUptime, cap, renderNav, markNav, barRow };
})();
