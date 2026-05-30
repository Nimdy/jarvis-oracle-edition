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
    ['#sep','core'],
    ['/static/v2/awakening.html','awakening'],
    ['/static/v2/integrity.html','integrity'],
    ['/static/v2/maturity.html','maturity'],
    ['/static/v2/yardsticks.html','yardsticks'],
    ['/static/v2/memory.html','memory'],
    ['/static/v2/cognition.html','cognition'],
    ['#sep','senses'],
    ['/static/v2/identity.html','identity'],
    ['/static/v2/voice.html','voice'],
    ['/static/v2/camera.html','camera'],
    ['/static/v2/spatial.html','spatial'],
    ['#sep','ops'],
    ['/static/v2/capability.html','capability'],
    ['/static/v2/autonomy.html','autonomy'],
    ['/static/v2/synthetic.html','synthetic'],
    ['/static/v2/ops.html','ops'],
    ['/static/v2/training.html','training'],
    ['#sep','lab'],
    ['/static/v2/lab.html','lab'],
    ['#sep','insight'],
    ['/static/v2/timeline.html','timeline'],
    ['/static/v2/immune.html','immune'],
    ['/static/v2/provenance.html','provenance'],
    ['/static/v2/emergence.html','emergence']
  ];
  // render the shared nav into <nav id="v2nav"></nav>, marking `active`.
  function renderNav(active){
    var nav=document.getElementById('v2nav'); if(!nav) return;
    var html=NAVPAGES.map(function(p){
      if(p[0]==='#sep') return '<span class="navsep">'+p[1]+'</span>';
      var on=(p[0]===active), stub=p[2];
      return '<a class="'+(on?'on':(stub?'stub':''))+'" href="'+(stub?'#':p[0])+'">'+p[1]+'</a>';
    }).join('');
    html+='<span class="navsep">ext</span><a href="/mind">/mind ↗</a><a href="/">← v1</a>';
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
  // ================= CRUD / ACTION LAYER =================
  // Operator-initiated writes. The OPERATOR's browser POSTs with the api_key
  // (read from /api/config, exactly like v1's window._apiKey). Claude never
  // fires these — they run only when a human clicks. Destructive actions go
  // through confirm() (typed gate).
  var _apiKey='';
  function loadKey(){ return fetchJSON('/api/config').then(function(c){ _apiKey=(c&&c.api_key)||''; return _apiKey; }).catch(function(){ return ''; }); }
  function _authHeaders(json){ var h={}; if(json) h['Content-Type']='application/json'; if(_apiKey) h['Authorization']='Bearer '+_apiKey; return h; }
  function _handle(r){ return r.json().catch(function(){return {};}).then(function(d){ if(!r.ok) throw new Error((d&&d.detail)||('HTTP '+r.status)); return d; }); }
  function post(url, body){ return fetch(url,{method:'POST',headers:_authHeaders(!!body),body:body?JSON.stringify(body):undefined}).then(_handle); }
  function del(url){ return fetch(url,{method:'DELETE',headers:_authHeaders(false)}).then(_handle); }

  // ---- modal ----
  function _ensureModalRoot(){
    var r=document.getElementById('v2-overlay');
    if(!r){ r=document.createElement('div'); r.id='v2-overlay'; r.className='v2-overlay'; r.style.display='none';
      r.innerHTML='<div class="v2-modal"><div class="v2-modal-hd"><h3 id="v2-modal-title"></h3><button class="v2-modal-x" id="v2-modal-x">×</button></div><div class="v2-modal-bd" id="v2-modal-bd"></div></div>';
      document.body.appendChild(r);
      r.addEventListener('click',function(e){ if(e.target===r) closeModal(); });
      r.querySelector('#v2-modal-x').addEventListener('click', closeModal);
    }
    return r;
  }
  function modal(title, bodyHtml){ var r=_ensureModalRoot(); r.querySelector('#v2-modal-title').textContent=title; r.querySelector('#v2-modal-bd').innerHTML=bodyHtml; r.style.display='flex'; return r.querySelector('#v2-modal-bd'); }
  function closeModal(){ var r=document.getElementById('v2-overlay'); if(r) r.style.display='none'; }

  // ---- confirm (danger gate; opts.typed = word the operator must type) ----
  function confirm(title, msg, onYes, opts){
    opts=opts||{}; var typed=opts.typed;
    var bd=modal(title,
      '<div class="cfm-msg">'+msg+'</div>'+
      (typed?'<input id="cfm-in" class="v2-field" autocomplete="off" placeholder="type '+typed+' to confirm">':'')+
      '<div class="cfm-btns"><button class="btn-act" id="cfm-no">Cancel</button>'+
      '<button class="btn-danger" id="cfm-yes"'+(typed?' disabled':'')+'>'+(opts.yesLabel||'Confirm')+'</button></div>');
    if(typed){ bd.querySelector('#cfm-in').addEventListener('input', function(){ bd.querySelector('#cfm-yes').disabled = this.value.trim()!==typed; }); }
    bd.querySelector('#cfm-no').addEventListener('click', closeModal);
    bd.querySelector('#cfm-yes').addEventListener('click', function(){ closeModal(); try{ onYes(); }catch(e){ toast('action failed: '+e.message, false); } });
  }

  // ---- toast ----
  function toast(msg, ok){
    var t=document.createElement('div'); t.className='v2-toast '+(ok===false?'bad':'good'); t.textContent=msg;
    document.body.appendChild(t);
    setTimeout(function(){ t.classList.add('show'); }, 10);
    setTimeout(function(){ t.classList.remove('show'); setTimeout(function(){ t.remove(); }, 300); }, 3200);
  }
  // convenience: run an action with toast feedback (NOT auto-fired — call from a click handler)
  function act(promise, okMsg){ return promise.then(function(d){ toast(okMsg||'done', true); return d; }).catch(function(e){ toast('failed: '+e.message, false); throw e; }); }

  loadKey();  // warm the api_key so operator clicks are ready
  // =======================================================

  return { fetchJSON, num, pct1, f2, f3, el, gateState, ago, bandColor, tag, fmtUptime, cap, renderNav, markNav, barRow,
           loadKey, post, del, modal, closeModal, confirm, toast, act };
})();
