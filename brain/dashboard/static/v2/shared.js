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
    ['/static/v2/cockpit.html','cockpit'],
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
    ['/static/v2/pi5.html','pi5 body'],
    ['/static/v2/spatial.html','spatial'],
    ['#sep','ops'],
    ['/static/v2/capability.html','capability'],
    ['/static/v2/autonomy.html','autonomy'],
    ['/static/v2/synthetic.html','synthetic'],
    ['/static/v2/ops.html','ops'],
    ['/static/v2/training.html','training'],
    ['/static/v2/matrix.html','matrix'],
    ['/static/v2/domains.html','domains'],
    ['#sep','lab'],
    ['/static/v2/lab.html','lab'],
    ['#sep','insight'],
    ['/static/v2/timeline.html','timeline'],
    ['/static/v2/immune.html','immune'],
    ['/static/v2/provenance.html','provenance'],
    ['/static/v2/grounding.html','grounding'],
    ['/static/v2/emergence.html','emergence'],
    ['/static/v2/prove.html','prove']
  ];
  // render the shared nav into <nav id="v2nav"></nav>, marking `active`.
  function renderNav(active){
    var nav=document.getElementById('v2nav'); if(!nav) return;
    var html=NAVPAGES.map(function(p){
      if(p[0]==='#sep') return '<span class="navsep">'+p[1]+'</span>';
      var on=(p[0]===active), stub=p[2];
      return '<a class="'+(on?'on':(stub?'stub':''))+'" href="'+(stub?'#':p[0])+'">'+p[1]+'</a>';
    }).join('');
    html+='<span class="navsep">act</span><a href="#" onclick="window.V2&&V2.palette();return false;" title="⌘K / Ctrl-K">⌘K jump</a><a href="#" onclick="window.V2&&V2.chat();return false;">💬 chat</a><a href="#" onclick="window.V2&&V2.legend();return false;">honesty</a><span class="navsep">ext</span><a href="/mind">/mind ↗</a><a href="/">← v1</a>';
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

  // ---- chat with JARVIS (cross-cutting; operator-fired only) ----
  function chat(){
    var bd=modal('Chat with JARVIS',
      '<div id="v2chat-log" style="max-height:46vh;overflow:auto;font-size:11px;line-height:1.5;margin-bottom:8px;"></div>'+
      '<div style="display:flex;gap:8px;"><input id="v2chat-in" class="v2-field" style="margin:0;flex:1" placeholder="Talk to JARVIS…" autocomplete="off"><button class="btn-act" id="v2chat-send">Send</button></div>'+
      '<div class="opnote" style="margin-top:7px;">POST /api/chat — the LLM articulates over grounded state, subject to the L0 capability-gate. This is a live conversation turn (it writes to memory).</div>');
    var log=bd.querySelector('#v2chat-log'), inp=bd.querySelector('#v2chat-in');
    function esc(s){ return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
    function send(){
      var msg=inp.value.trim(); if(!msg) return;
      log.innerHTML+='<div style="color:var(--cyan);margin:5px 0;">▸ '+esc(msg)+'</div>';
      inp.value=''; inp.disabled=true; log.scrollTop=log.scrollHeight;
      post('/api/chat',{message:msg}).then(function(d){
        var reply=d.response||d.reply||d.text||d.message||JSON.stringify(d).slice(0,500);
        log.innerHTML+='<div style="color:var(--text);margin:2px 0 9px;">'+esc(reply)+'</div>';
      }).catch(function(e){ log.innerHTML+='<div style="color:var(--red);margin:2px 0 9px;">error: '+esc(e.message)+'</div>'; })
      .then(function(){ inp.disabled=false; inp.focus(); log.scrollTop=log.scrollHeight; });
    }
    bd.querySelector('#v2chat-send').addEventListener('click', send);
    inp.addEventListener('keydown', function(e){ if(e.key==='Enter') send(); });
    inp.focus();
  }

  // ================= COMMAND PALETTE (⌘K) + HONESTY LEGEND =================
  // Every surface jumpable: V2 pages + reference docs + HRR worlds + v1 + actions.
  function _surfaces(){
    var s=[];
    NAVPAGES.forEach(function(p){ if(p[0]!=='#sep') s.push({href:p[0], label:p[1], group:'v2'}); });
    [['/docs','system reference'],['/science','scientific spec'],['/showcase','tech showcase'],
     ['/history','build history'],['/api-reference','API reference'],['/learning','companion training']]
      .forEach(function(d){ s.push({href:d[0], label:d[1], group:'docs'}); });
    [['/mind','mind’s-eye · spatial world'],['/hrr-scene','HRR scene graph']]
      .forEach(function(d){ s.push({href:d[0], label:d[1], group:'worlds'}); });
    s.push({href:'/', label:'v1 dashboard', group:'legacy'});
    s.push({act:'chat', label:'chat with JARVIS', group:'action'});
    s.push({act:'legend', label:'honesty legend', group:'action'});
    return s;
  }
  var _pal={open:false, sel:0, items:[]};
  function _palRoot(){
    var r=document.getElementById('v2-pal');
    if(!r){
      r=document.createElement('div'); r.id='v2-pal'; r.className='v2-pal'; r.style.display='none';
      r.innerHTML='<div class="v2-pal-box"><input id="v2-pal-in" class="v2-pal-in" placeholder="jump to… (page, doc, world, action)" autocomplete="off"><div class="v2-pal-list" id="v2-pal-list"></div><div class="v2-pal-hint">↑↓ move · ↵ open · esc close</div></div>';
      document.body.appendChild(r);
      r.addEventListener('click', function(e){ if(e.target===r) closePalette(); });
      var inp=r.querySelector('#v2-pal-in');
      inp.addEventListener('input', function(){ _palRender(this.value); });
      inp.addEventListener('keydown', _palKey);
    }
    return r;
  }
  function _palRender(q){
    q=(q||'').toLowerCase().trim();
    var items=_surfaces().filter(function(s){ return !q || (s.label+' '+s.group).toLowerCase().indexOf(q)>=0; });
    _pal.items=items; if(_pal.sel>=items.length) _pal.sel=0;
    var list=document.getElementById('v2-pal-list');
    list.innerHTML=items.map(function(s,i){
      return '<div class="v2-pal-item'+(i===_pal.sel?' sel':'')+'" data-i="'+i+'"><span class="v2-pal-g">'+s.group+'</span><span>'+s.label+'</span></div>';
    }).join('')||'<div class="v2-pal-empty">no match</div>';
    Array.prototype.forEach.call(list.querySelectorAll('[data-i]'), function(node){
      node.addEventListener('click', function(){ _palGo(items[+this.getAttribute('data-i')]); });
      node.addEventListener('mousemove', function(){ _pal.sel=+this.getAttribute('data-i'); _palHi(); });
    });
  }
  function _palHi(){ var list=document.getElementById('v2-pal-list'); if(!list) return;
    Array.prototype.forEach.call(list.children, function(c,i){ if(c.classList) c.classList.toggle('sel', i===_pal.sel); }); }
  function _palGo(s){ if(!s) return; closePalette(); if(s.href) location.href=s.href; else if(s.act==='chat') chat(); else if(s.act==='legend') legend(); }
  function _palKey(e){
    var n=_pal.items.length||1;
    if(e.key==='ArrowDown'){ e.preventDefault(); _pal.sel=(_pal.sel+1)%n; _palHi(); }
    else if(e.key==='ArrowUp'){ e.preventDefault(); _pal.sel=(_pal.sel-1+n)%n; _palHi(); }
    else if(e.key==='Enter'){ e.preventDefault(); _palGo(_pal.items[_pal.sel]); }
    else if(e.key==='Escape'){ closePalette(); }
  }
  function palette(){ var r=_palRoot(); _pal.open=true; _pal.sel=0; r.style.display='flex'; var inp=r.querySelector('#v2-pal-in'); inp.value=''; _palRender(''); inp.focus(); }
  function closePalette(){ var r=document.getElementById('v2-pal'); if(r) r.style.display='none'; _pal.open=false; }

  function legend(){
    modal('Honesty legend — how to read this dashboard',
      '<div class="lg-note">Every number here is labelled by <b>provenance</b>, and anything not measurable is shown <b>UNAVAILABLE</b>, never faked. That discipline is the point.</div>'+
      '<div class="lg-key">'+
      '<div>'+tag('self','self-scored')+' graded by JARVIS against its own rubric — no external comparator (e.g. the Oracle benchmark).</div>'+
      '<div>'+tag('grounded','grounded')+' verified from lived runtime evidence (process contracts, observed events).</div>'+
      '<div>'+tag('earned','earned')+' a maturity gate actually crossed by accumulated evidence.</div>'+
      '<div>'+tag('derived','derived/templated')+' computed/assembled from real data with a human-written structure — no free-form LLM.</div>'+
      '<div><span class="pillbig" style="color:var(--magenta);border-color:var(--magenta)">shadow</span> a learned model running telemetry-only — NOT driving live decisions yet.</div>'+
      '<div><span class="s-unk">UNKNOWN</span> / hatched / "—" = not measurable / not yet instrumented. Gates fail <b>closed</b> — never green on missing data.</div>'+
      '</div>'+
      '<div class="lg-note" style="margin-top:11px">Evaluating the system? Open <a href="/static/v2/prove.html">Prove It</a> (falsifiable claims + live evidence) and <a href="/static/v2/yardsticks.html">Yardsticks</a> (where the system’s own metrics are self-referential).</div>');
  }

  // global hotkeys: ⌘K / Ctrl-K palette; Esc closes overlays.
  document.addEventListener('keydown', function(e){
    if((e.metaKey||e.ctrlKey) && (e.key==='k'||e.key==='K')){ e.preventDefault(); _pal.open?closePalette():palette(); }
    else if(e.key==='Escape'){ closePalette(); closeModal(); }
  });

  loadKey();  // warm the api_key so operator clicks are ready
  // =======================================================

  return { fetchJSON, num, pct1, f2, f3, el, gateState, ago, bandColor, tag, fmtUptime, cap, renderNav, markNav, barRow,
           loadKey, post, del, modal, closeModal, confirm, toast, act, chat, palette, closePalette, legend };
})();
