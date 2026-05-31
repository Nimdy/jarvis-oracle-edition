/* ===== dashboardV2 SHARED DRILL-DOWN MODULE — window.V2D (static, no restart) =====
 * Reusable detail modals, ported verbatim from v1 (interactives.js) field rendering.
 * Every modal opens through V2.modal(title, html) and renders ONLY fields that map to
 * a real endpoint+path. null/missing -> '—'/UNKNOWN, never faked, never green-on-null,
 * never zero-as-success. READ-ONLY: every fetch here is a GET.
 *
 * SHARED CONTRACT (agrees with cockpit.html + shared.js window.V2):
 *   cockpit.html runs one 2s poll loop caching a merged object into window._lastSnap:
 *     ._lastSnap.full     = /api/full-snapshot       (autonomy, episodes, eval, core, …)
 *     ._lastSnap.kperf    = /api/kernel/performance
 *     ._lastSnap.thoughts = /api/consciousness/thoughts
 *     ._lastSnap._ts      = epoch-seconds stamp
 *   GUARD: if window._lastSnap (or the slice we need) is missing, we V2.fetchJSON fresh.
 *
 * Cached reads here use the .full / .thoughts slices; endpoint-backed modals
 * (trace, memory, debug) fetch their own GET. */
window.V2D = (function(){
  "use strict";

  var V = window.V2;  // honest-render toolkit (fetchJSON,num,f3,pct1,ago,bandColor,gateState,tag,cap,modal,…)

  // ---- tiny local helpers (mirror v1's esc/fmt where V2 has no exact twin) ----
  function esc(s){ return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
  // dash-on-null number formatter (v1 fmtNum, honest: null -> '—', never 0)
  function fnum(v, dp){ v=V.num(v); if(v===null) return '—'; return dp!=null?v.toFixed(dp):String(v); }
  // dash-on-null integer-with-commas (counts)
  function fint(v){ v=V.num(v); if(v===null) return '—'; return v.toLocaleString(); }
  // dash-on-null percent (0..1 -> NN.N%); null -> '—'
  function fpct(v){ v=V.num(v); if(v===null) return '—'; return (v*100).toFixed(1)+'%'; }
  // relative-age via the shared clock; null/0 -> '—'
  function fago(ts){ ts=V.num(ts); if(!ts) return '—'; return V.ago(ts); }
  // byte sizes for the eval store
  function fbytes(b){ b=V.num(b); if(b===null) return '—'; if(b<1024) return b+'B'; if(b<1048576) return (b/1024).toFixed(1)+'K'; return (b/1048576).toFixed(1)+'M'; }
  function isObj(o){ return o && typeof o==='object'; }
  function sq(s){ return esc(String(s)).replace(/'/g,"\\'"); }  // safe single-quote for inline onclick

  // ---- snapshot slice accessors (GUARD: fall back to a fresh GET if cache absent) ----
  // .full = /api/full-snapshot. Returns a Promise (already-cached resolves immediately).
  function full(){
    var s = window._lastSnap;
    if(s && isObj(s.full)) return Promise.resolve(s.full);
    return V.fetchJSON('/api/full-snapshot');
  }
  // .thoughts = /api/consciousness/thoughts
  function thoughts(){
    var s = window._lastSnap;
    if(s && isObj(s.thoughts)) return Promise.resolve(s.thoughts);
    return V.fetchJSON('/api/consciousness/thoughts');
  }

  // small UNKNOWN block for empty panels (honest: name the gap, never fake)
  function unknownBox(msg){
    return '<div style="padding:10px 12px;background:var(--panel,#0d0d1a);border:1px solid #2a2a44;border-radius:6px;color:var(--dim,#6a6a80);font-size:11px;">'+esc(msg||'UNKNOWN — no data for this view.')+'</div>';
  }
  // explicit honest caveat strip (italic, dim) — used for labels the operator must not misread
  function caveat(txt){
    return '<div style="font-size:10px;color:var(--dim,#6a6a80);font-style:italic;margin:3px 0 7px;">'+esc(txt)+'</div>';
  }

  // =====================================================================
  // Thought Detail (Meta-Thought)  — window._lastSnap.thoughts.recent[idx]
  // ports v1 openThoughtDetail (interactives.js:1261). Adds trigger / evidence_refs /
  // confidence per the cockpit plan (all honest: missing -> '—', not faked).
  // =====================================================================
  function thoughtDetail(idx){
    thoughts().then(function(th){
      var recent = (th && th.recent) || [];
      var t = recent[idx];
      if(!t){ V.modal('Meta-Thought', unknownBox('Thought not found in snapshot (idx '+esc(idx)+').')); return; }

      var depthColors = { surface:'#6a6a80', moderate:'#0cf', deep:'#c0f', profound:'#f0f' };
      var dc = depthColors[t.depth] || '#6a6a80';
      var html = '';

      // type badge + depth badge (color-graded)
      html += '<div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">'+
        '<span style="padding:2px 8px;background:#0cf22;border:1px solid #0cf44;color:#0cf;border-radius:4px;font-size:11px;">'+esc(t.type||'—')+'</span>'+
        '<span style="padding:2px 8px;background:'+dc+'22;border:1px solid '+dc+'44;color:'+dc+';border-radius:4px;font-size:11px;">'+esc(t.depth||'—')+'</span>'+
        '</div>';

      // full text
      html += '<div style="padding:10px 12px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:6px;margin-bottom:10px;">'+
        '<div style="font-size:13px;color:#e0e0e8;line-height:1.5;white-space:pre-wrap;">'+esc(t.text||'—')+'</div></div>';

      // trigger (source) + confidence
      html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px;">';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Trigger:</span> <span style="color:#c0f;">'+esc(t.trigger||'—')+'</span></div>';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Confidence:</span> <span style="color:#8a8aa0;">'+(t.confidence!=null?fpct(t.confidence):'—')+'</span></div>';
      html += '</div>';

      // id + timestamp
      html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px;">';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">ID:</span> <span style="color:#484860;">'+esc(t.id||'—')+'</span></div>';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Generated:</span> <span style="color:#8a8aa0;">'+fago(t.time)+'</span></div>';
      html += '</div>';

      // evidence refs (honest: empty list -> explicit none)
      html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Evidence Refs</div>';
      var refs = t.evidence_refs || [];
      if(refs.length){
        html += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">';
        refs.forEach(function(r){ html += '<span style="padding:1px 6px;background:#181828;border:1px solid #2a2a44;border-radius:3px;font-size:10px;color:#0cf;">'+esc(r)+'</span>'; });
        html += '</div>';
      } else {
        html += '<div style="font-size:10px;color:#484860;margin-bottom:8px;">— none recorded</div>';
      }

      // tags
      var tags = t.tags || [];
      if(tags.length){
        html += '<div style="display:flex;flex-wrap:wrap;gap:4px;">';
        tags.forEach(function(tg){ html += '<span style="padding:1px 6px;background:#1a1a2e;border:1px solid #2a2a44;border-radius:3px;font-size:10px;color:#c0f;">#'+esc(tg)+'</span>'; });
        html += '</div>';
      }

      V.modal('Meta-Thought', html);
    }).catch(function(e){ V.modal('Meta-Thought', unknownBox('Error loading thought: '+esc(e.message))); });
  }

  // =====================================================================
  // Research Job Detail — window._lastSnap.full.autonomy.completed[idx]
  //   (+ .recent_learnings / .recent_deltas matched by intent_id)
  // ports v1 openResearchDetail/openResearchEpisodeDetail (interactives.js:1095).
  // priority is LABELLED 'pre-research estimate, not a quality score'.
  // pseudo-score composed from reason + source_event + result.success + #findings +
  //   net_improvement (clearly labelled DERIVED, never a server metric).
  // chains to deltaDetail when a matching delta exists.
  // =====================================================================
  function researchDetail(idx){
    full().then(function(fs){
      var auto = (fs && fs.autonomy) || {};
      var completed = auto.completed || [];
      var learnings = auto.recent_learnings || [];
      var deltas = auto.recent_deltas || [];
      var item = completed[idx];
      if(!item){ V.modal('Research Episode', unknownBox('Research episode not found in snapshot (idx '+esc(idx)+').')); return; }

      var html = '';

      // question headline
      html += '<div style="margin-bottom:8px;">'+
        '<div style="font-size:10px;color:#6a6a80;">Question</div>'+
        '<div style="font-size:14px;color:#e0e0e8;margin-bottom:4px;">'+esc(item.question||'—')+'</div></div>';

      // status + priority (LABELLED) + scope
      var st = item.status;
      var sc = st==='completed' ? '#0f9' : st==='failed' ? '#f44' : st ? '#ff0' : '#6a6a80';
      html += '<div style="display:flex;gap:8px;align-items:center;margin-bottom:4px;flex-wrap:wrap;">'+
        '<span style="padding:1px 6px;background:'+sc+'22;border:1px solid '+sc+'44;color:'+sc+';border-radius:3px;font-size:10px;">'+esc(st||'UNKNOWN')+'</span>'+
        '<span style="font-size:10px;color:#6a6a80;">priority: '+fnum(item.priority,3)+'</span>'+
        '<span style="font-size:10px;color:#6a6a80;">scope: '+esc(item.scope||'—')+'</span>'+
        '</div>';
      html += caveat("priority is a pre-research estimate, not a quality score.");

      // scoring reason
      if(item.reason){
        html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;margin-bottom:8px;">'+
          '<div style="font-size:10px;color:#6a6a80;">Scoring Reason</div>'+
          '<div style="font-size:11px;color:#8a8aa0;white-space:pre-wrap;">'+esc(item.reason)+'</div></div>';
      }

      // source event + hint
      html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:6px;">';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Source event:</span> <span style="color:#c0f;">'+esc(item.source_event||'—')+'</span></div>';
      if(item.source_hint) html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Tool hint:</span> <span style="color:#0cf;">'+esc(item.source_hint)+'</span></div>';
      html += '</div>';

      // timestamps
      if(item.created_at || item.started_at || item.completed_at){
        html += '<div style="display:flex;gap:8px;font-size:9px;color:#484860;margin-bottom:6px;flex-wrap:wrap;">';
        if(item.created_at)   html += '<span>created: '+fago(item.created_at)+'</span>';
        if(item.started_at)   html += '<span>started: '+fago(item.started_at)+'</span>';
        if(item.completed_at) html += '<span>completed: '+fago(item.completed_at)+'</span>';
        if(item.started_at && item.completed_at) html += '<span>duration: '+fnum(item.completed_at-item.started_at,1)+'s</span>';
        html += '</div>';
      }

      // result block (tool_used + summary + findings with provenance/confidence/doi/authors/venue)
      var result = item.result || {};
      var findings = result.findings || [];
      html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Result</div>';
      html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;margin-bottom:8px;">';
      // honest gate: explicit true/false only; anything else UNKNOWN (never green-on-null)
      var rg = V.gateState(result.success);
      var rgC = rg==='ok'?'#0f9':rg==='fail'?'#f44':'#6a6a80';
      var rgLbl = rg==='ok'?'yes':rg==='fail'?'no':'UNKNOWN';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Tool used:</span> <span style="color:#0cf;">'+esc(result.tool_used||'—')+'</span>'+
        ' <span style="color:#6a6a80;">success:</span> <span style="color:'+rgC+';">'+rgLbl+'</span></div>';
      if(result.summary) html += '<div style="font-size:11px;color:#8a8aa0;margin-top:4px;max-height:150px;overflow-y:auto;white-space:pre-wrap;">'+esc(result.summary)+'</div>';
      html += '</div>';

      if(findings.length){
        html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Findings ('+findings.length+')</div>';
        html += '<div style="max-height:200px;overflow-y:auto;border:1px solid #1a1a2e;border-radius:4px;padding:4px;margin-bottom:8px;">';
        findings.forEach(function(fd){
          html += '<div style="padding:4px;border-bottom:1px solid #0d0d1a;font-size:10px;">';
          html += '<div style="color:#e0e0e8;white-space:pre-wrap;">'+esc(fd.content||'—')+'</div>';
          html += '<div style="color:#484860;margin-top:2px;">';
          html += 'provenance: <span style="color:#8a8aa0;">'+esc(fd.provenance||fd.source_type||'—')+'</span>';
          html += ' · confidence: <span style="color:#8a8aa0;">'+(fd.confidence!=null?fpct(fd.confidence):'—')+'</span>';
          if(fd.doi)     html += ' · doi: <span style="color:#0cf;">'+esc(fd.doi)+'</span>';
          if(fd.authors) html += ' · '+esc(Array.isArray(fd.authors)?fd.authors.join(', '):fd.authors);
          if(fd.venue)   html += ' · '+esc(fd.venue);
          html += '</div></div>';
        });
        html += '</div>';
      } else {
        html += '<div style="font-size:10px;color:#484860;margin-bottom:8px;">— no findings recorded</div>';
      }

      // matched learning (by intent_id)
      var learning = null;
      for(var i=0;i<learnings.length;i++){ if(learnings[i].intent_id===item.id){ learning=learnings[i]; break; } }
      if(learning){
        html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Learning Outcome</div>';
        html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;margin-bottom:8px;">';
        html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Tool:</span> <span style="color:#0cf;">'+esc(learning.tool||'—')+'</span>'+
          ' <span style="color:#6a6a80;">Findings:</span> <span style="color:#0f9;">'+fint(learning.findings)+'</span>'+
          ' <span style="color:#6a6a80;">Memories:</span> <span style="color:#c0f;">'+fint(learning.memories_created)+'</span></div>';
        if(learning.summary) html += '<div style="font-size:11px;color:#8a8aa0;margin-top:4px;max-height:150px;overflow-y:auto;white-space:pre-wrap;">'+esc(learning.summary)+'</div>';
        html += '</div>';
      }

      // matched delta (by intent_id) -> impact + chain button to deltaDetail
      var matchedDeltaIdx = -1, matchedDelta = null;
      for(var j=0;j<deltas.length;j++){ if(deltas[j].intent_id===item.id){ matchedDelta=deltas[j]; matchedDeltaIdx=j; break; } }

      // DERIVED pseudo-score (clearly labelled — NOT a server metric)
      // composed from: reason present + source_event present + result.success + #findings + net_improvement
      var pscore = 0, parts = [];
      if(item.reason){ pscore += 1; parts.push('reason'); }
      if(item.source_event){ pscore += 1; parts.push('source_event'); }
      if(result.success===true){ pscore += 2; parts.push('result.success'); }
      if(findings.length){ pscore += Math.min(2, findings.length); parts.push(findings.length+' findings'); }
      var net = matchedDelta ? V.num(matchedDelta.net_improvement) : null;
      if(net!==null && net>0){ pscore += 2; parts.push('net_improvement +'+fnum(net,4)); }
      html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Composite Signal (DERIVED)</div>';
      html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px dashed #2a2a44;border-radius:4px;margin-bottom:4px;">'+
        '<div style="font-size:13px;color:#0cf;font-weight:600;">'+pscore+'</div>'+
        '<div style="font-size:10px;color:#8a8aa0;">'+ (parts.length?esc(parts.join(' + ')):'— no positive signals') +'</div></div>';
      html += caveat("DERIVED tally of present signals — NOT a server-side score. The system does not emit a post-research quality score.");

      // impact delta + chain
      if(matchedDelta){
        var nc = net>0.01 ? '#0f9' : net<-0.01 ? '#f44' : '#6a6a80';
        html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Impact Delta</div>'+
          '<div style="display:flex;align-items:center;gap:10px;font-size:11px;margin-bottom:4px;">'+
          '<span>Net improvement: <span style="color:'+nc+';font-weight:600;">'+(net!=null?(net>0?'+':'')+fnum(net,4):'—')+'</span></span>'+
          '<button class="btn-act" onclick="window.V2D.deltaDetail('+matchedDeltaIdx+')">delta detail →</button>'+
          '</div>';
      }

      V.modal('Research Episode', html);
    }).catch(function(e){ V.modal('Research Episode', unknownBox('Error loading research episode: '+esc(e.message))); });
  }

  // =====================================================================
  // Autonomy Delta Detail — window._lastSnap.full.autonomy.recent_deltas[idx]
  // ports v1 openDeltaDetail (interactives.js:1438).
  // attribution: 'no measured impact' when absent (NOT 0).
  // =====================================================================
  function deltaDetail(idx){
    full().then(function(fs){
      var auto = (fs && fs.autonomy) || {};
      var deltas = auto.recent_deltas || [];
      var d = deltas[idx];
      if(!d){ V.modal('Autonomy Delta', unknownBox('Delta not found in snapshot (idx '+esc(idx)+').')); return; }

      var html = '';

      // net improvement headline (color-coded; null -> dim '—', never green)
      var net = V.num(d.net_improvement);
      var netC = net===null ? '#6a6a80' : net>0.01 ? '#0f9' : net<-0.01 ? '#f44' : '#6a6a80';
      html += '<div style="text-align:center;margin-bottom:8px;">'+
        '<div style="font-size:18px;font-weight:700;color:'+netC+';">'+(net!=null?(net>0?'+':'')+fnum(net,4):'—')+'</div>'+
        '<div style="font-size:10px;color:#6a6a80;">Net Improvement</div></div>';

      // intent id + matched question
      html += '<div style="font-size:10px;color:#484860;margin-bottom:6px;">Intent: '+esc(d.intent_id||'—')+'</div>';
      var completed = auto.completed || [];
      var matchedQ = '';
      for(var i=0;i<completed.length;i++){ if(completed[i].id===d.intent_id){ matchedQ=completed[i].question||''; break; } }
      if(matchedQ){
        html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;margin-bottom:8px;">'+
          '<div style="font-size:10px;color:#6a6a80;">Research Question</div>'+
          '<div style="font-size:11px;color:#e0e0e8;">'+esc(matchedQ)+'</div></div>';
      }

      var deltaFields = d.deltas || {};
      var cfDeltas = d.counterfactual || d.counterfactual_deltas || {};
      var attrib = d.attribution || {};
      var baseline = d.baseline || {};
      var post = d.post || {};

      // metrics table: metric | before | after | delta | attribution
      // attribution missing -> 'no measured impact' (NOT 0)
      if(Object.keys(deltaFields).length){
        html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Metrics</div>';
        html += '<table style="width:100%;font-size:10px;border-collapse:collapse;margin-bottom:8px;">';
        html += '<tr style="border-bottom:1px solid #2a2a44;">'+
          '<th style="text-align:left;color:#6a6a80;padding:2px 4px;">Metric</th>'+
          '<th style="text-align:right;color:#6a6a80;padding:2px 4px;">Before</th>'+
          '<th style="text-align:right;color:#6a6a80;padding:2px 4px;">After</th>'+
          '<th style="text-align:right;color:#6a6a80;padding:2px 4px;">Delta</th>'+
          '<th style="text-align:right;color:#6a6a80;padding:2px 4px;">Attribution</th></tr>';
        Object.keys(deltaFields).forEach(function(name){
          var dv = V.num(deltaFields[name]);
          var bv = baseline[name], pv = post[name];
          var hasAttr = Object.prototype.hasOwnProperty.call(attrib, name);
          var av = hasAttr ? V.num(attrib[name]) : null;
          var dc = dv===null ? '#6a6a80' : dv>0.001 ? '#0f9' : dv<-0.001 ? '#f44' : '#6a6a80';
          var attrCell;
          if(!hasAttr || av===null){
            attrCell = '<span style="color:#484860;font-style:italic;">no measured impact</span>';
          } else {
            var ac = av>0.001 ? '#0f9' : av<-0.001 ? '#f44' : '#484860';
            attrCell = '<span style="color:'+ac+';">'+(av>0?'+':'')+fnum(av,4)+'</span>';
          }
          html += '<tr style="border-bottom:1px solid #0d0d1a;">'+
            '<td style="padding:2px 4px;color:#8a8aa0;">'+esc(V.cap(name))+'</td>'+
            '<td style="text-align:right;padding:2px 4px;color:#484860;">'+fnum(bv,3)+'</td>'+
            '<td style="text-align:right;padding:2px 4px;color:#484860;">'+fnum(pv,3)+'</td>'+
            '<td style="text-align:right;padding:2px 4px;color:'+dc+';">'+(dv!=null?(dv>0?'+':'')+fnum(dv,4):'—')+'</td>'+
            '<td style="text-align:right;padding:2px 4px;">'+attrCell+'</td></tr>';
        });
        html += '</table>';
      } else {
        html += unknownBox('No per-metric deltas recorded for this intent.');
      }

      // counterfactual summary
      var cfKeys = Object.keys(cfDeltas);
      if(cfKeys.length){
        html += '<div style="font-size:10px;color:#6a6a80;margin:6px 0 2px;">Counterfactual (what would have happened without this research)</div>';
        html += '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">';
        cfKeys.forEach(function(name){
          var v = V.num(cfDeltas[name]);
          if(v!==null && Math.abs(v)>0.0001){
            var cc = v>0 ? '#0f9' : '#f44';
            html += '<span style="padding:1px 4px;border:1px solid '+cc+'33;color:'+cc+';border-radius:2px;font-size:9px;">'+esc(V.cap(name))+': '+(v>0?'+':'')+fnum(v,4)+'</span>';
          }
        });
        html += '</div>';
      }

      V.modal('Autonomy Delta', html);
    }).catch(function(e){ V.modal('Autonomy Delta', unknownBox('Error loading delta: '+esc(e.message))); });
  }

  // =====================================================================
  // Trace Explorer — GET /api/trace/explorer?root_limit=25&run_limit=25&tool_limit=60
  // ports v1 openTraceExplorer (interactives.js:781).
  // reconstructability -> 'unknown' on empty (never faked green).
  // =====================================================================
  function traceExplorer(){
    V.modal('Trace Explorer', '<div id="v2d-trace-body" style="max-height:440px;overflow-y:auto;font-size:11px;color:#8a8aa0;">Loading trace explorer…</div>');
    V.fetchJSON('/api/trace/explorer?root_limit=25&run_limit=25&tool_limit=60').then(function(data){
      var body = document.getElementById('v2d-trace-body'); if(!body) return;
      var roots = data.root_chains || [];
      var runs = data.agent_runs || [];
      var tools = data.tool_lineage || [];
      var rec = data.reconstructability || {};
      var recStatus = rec.reconstructability || 'unknown';
      var recColor = recStatus==='reconstructable' ? '#0f9' : recStatus==='partial' ? '#ff0' : recStatus==='non_reconstructable' ? '#f44' : '#6a6a80';

      var out = '';
      out += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px;">'+
        '<div style="padding:6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;text-align:center;"><div style="font-size:16px;color:#0cf;font-weight:700;">'+fint(data.entry_count)+'</div><div style="font-size:9px;color:#6a6a80;">Ledger Entries</div></div>'+
        '<div style="padding:6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;text-align:center;"><div style="font-size:16px;color:#0f9;font-weight:700;">'+roots.length+'</div><div style="font-size:9px;color:#6a6a80;">Root Chains</div></div>'+
        '<div style="padding:6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;text-align:center;"><div style="font-size:16px;color:#c0f;font-weight:700;">'+runs.length+'</div><div style="font-size:9px;color:#6a6a80;">Agent Runs</div></div>'+
        '<div style="padding:6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;text-align:center;"><div style="font-size:13px;color:'+recColor+';font-weight:700;">'+esc(V.cap(recStatus))+'</div><div style="font-size:9px;color:#6a6a80;">Reconstructability</div></div>'+
        '</div>';

      out += '<div style="font-size:10px;color:#6a6a80;margin-bottom:4px;">Root Chains</div>';
      if(!roots.length){
        out += '<div style="font-size:10px;color:#484860;margin-bottom:8px;">— no root chains in cache window.</div>';
      } else {
        roots.slice(0,12).forEach(function(r){
          var rid = r.root_entry_id || '—';
          var subs = (r.subsystems||[]).slice(0,4).join(', ');
          out += '<div style="display:flex;align-items:center;gap:6px;padding:3px 4px;border-bottom:1px solid #1a1a2e;font-size:10px;">'+
            '<span style="color:#0cf;min-width:84px;">'+esc(String(rid).substring(0,20))+'</span>'+
            '<span style="min-width:46px;color:#e0e0e8;">'+fint(r.entry_count)+' ev</span>'+
            '<span style="min-width:52px;color:#6a6a80;">'+(r.duration_s!=null?fnum(r.duration_s,2)+'s':'—')+'</span>'+
            '<span style="flex:1;color:#8a8aa0;">'+esc(subs||'—')+'</span>'+
            '<button class="btn-act" onclick="window.V2D.traceChain(\''+sq(rid)+'\')">chain</button>'+
            '</div>';
        });
      }

      out += '<div style="font-size:10px;color:#6a6a80;margin-top:8px;margin-bottom:4px;">Per-Agent Runs</div>';
      if(!runs.length){
        out += '<div style="font-size:10px;color:#484860;margin-bottom:8px;">— no autonomy runs in cache window.</div>';
      } else {
        runs.slice(0,10).forEach(function(run){
          out += '<div style="padding:3px 4px;border-bottom:1px solid #1a1a2e;font-size:10px;">'+
            '<span style="color:#c0f;">'+esc(String(run.intent_id||'—').substring(0,24))+'</span>'+
            ' <span style="color:#6a6a80;">events:</span> '+fint(run.event_count)+
            (run.tools && run.tools.length ? ' <span style="color:#6a6a80;">tools:</span> '+esc(run.tools.slice(0,3).join(', ')) : '')+
            (run.goal_id ? ' <span style="color:#484860;">goal:'+esc(String(run.goal_id).substring(0,12))+'</span>' : '')+
            '</div>';
        });
      }

      out += '<div style="font-size:10px;color:#6a6a80;margin-top:8px;margin-bottom:4px;">Tool Lineage</div>';
      if(!tools.length){
        out += '<div style="font-size:10px;color:#484860;">— no tool lineage in cache window.</div>';
      } else {
        tools.slice(0,20).forEach(function(t){
          out += '<div style="padding:2px 0;border-bottom:1px solid #1a1a2e;font-size:9px;">'+
            '<span style="color:#0f9;">'+esc(t.tool||'—')+'</span> '+
            '<span style="color:#6a6a80;">('+esc(t.source||'—')+')</span>'+
            (t.trace_id ? ' <span style="color:#484860;">'+esc(String(t.trace_id).substring(0,16))+'</span>' : '')+
            '</div>';
        });
      }

      body.innerHTML = out;
    }).catch(function(e){
      var body = document.getElementById('v2d-trace-body');
      if(body) body.innerHTML = unknownBox('Error loading trace explorer: '+esc(e.message));
    });
  }

  // =====================================================================
  // Trace Chain — GET /api/trace/explorer/chain/{rootId}
  // ports v1 openTraceChain (interactives.js:858).
  // outcome ladder: pending (dim) / success (green) / failed (red).
  // =====================================================================
  function traceChain(rootId){
    if(!rootId){ V.modal('Trace Chain', unknownBox('No root id supplied.')); return; }
    V.modal('Trace Chain '+esc(String(rootId).substring(0,20)), '<div id="v2d-chain-body" style="max-height:460px;overflow-y:auto;font-size:11px;color:#8a8aa0;">Loading chain '+esc(rootId)+'…</div>');
    V.fetchJSON('/api/trace/explorer/chain/'+encodeURIComponent(rootId)).then(function(data){
      var body = document.getElementById('v2d-chain-body'); if(!body) return;
      var nodes = data.nodes || [];
      if(!nodes.length){ body.innerHTML = unknownBox('No chain nodes found for this root.'); return; }

      var byId = {};
      nodes.forEach(function(n){ byId[n.entry_id]=n; });
      function _depth(node){ var d=0,p=node.parent_entry_id,g=0; while(p && byId[p] && g<32){ d+=1; p=byId[p].parent_entry_id; g+=1; } return d; }

      var out = '';
      out += '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px;font-size:9px;color:#6a6a80;">'+
        '<span>nodes: <span style="color:#0cf;">'+nodes.length+'</span></span>'+
        '<span>roots: <span style="color:#0f9;">'+((data.root_nodes||[]).length||0)+'</span></span>'+
        '<span>tool path: <span style="color:#c0f;">'+esc((data.tool_path||[]).join(' → ')||'—')+'</span></span>'+
        '</div>';

      var sorted = nodes.slice().sort(function(a,b){ return (V.num(a.ts)||0)-(V.num(b.ts)||0); });
      sorted.forEach(function(n){
        var indent = _depth(n)*12;
        // outcome ladder: success green, failed red, pending/unknown dim
        var oc = n.outcome;
        var outcomeColor = oc==='success' ? '#0f9' : (oc==='failed'||oc==='failure') ? '#f44' : '#6a6a80';
        var outcomeLbl = oc || 'pending';
        out += '<div style="padding:2px 0 2px '+indent+'px;border-bottom:1px solid #1a1a2e;font-size:9px;">'+
          '<span style="color:#484860;">'+fago(n.ts)+'</span> '+
          '<span style="color:#0cf;">'+esc((n.subsystem||'—')+':'+(n.event_type||'—'))+'</span> '+
          '<span style="color:'+outcomeColor+';">'+esc(outcomeLbl)+'</span>'+
          (n.tool ? ' <span style="color:#0f9;">tool='+esc(n.tool)+'</span>' : '')+
          (n.trace_id ? ' <span style="color:#484860;">'+esc(String(n.trace_id).substring(0,14))+'</span>' : '')+
          '</div>';
      });
      body.innerHTML = out;
    }).catch(function(e){
      var body = document.getElementById('v2d-chain-body');
      if(body) body.innerHTML = unknownBox('Error loading chain: '+esc(e.message));
    });
  }

  // =====================================================================
  // Flight Recorder drill-down — window._lastSnap.full.episodes[idx]
  // INPUT -> ROUTE -> IDENTITY -> MEMORIES -> GATES -> [LLM implicit] -> OUTPUT -> LATENCY -> FEEDBACK
  // LLM stage is LABELLED 'implicit (inferred from route)' — there is no explicit LLM field.
  // =====================================================================
  function flightDetail(idx){
    full().then(function(fs){
      var episodes = (fs && fs.episodes) || [];
      var ep = episodes[idx];
      if(!ep){ V.modal('Flight Episode', unknownBox('Episode not found in snapshot (idx '+esc(idx)+').')); return; }

      var html = '';

      // INPUT: user_input / speaker / emotion
      html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Input</div>';
      html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;margin-bottom:8px;">'+
        '<div style="font-size:12px;color:#e0e0e8;white-space:pre-wrap;">'+esc(ep.user_input||'—')+'</div>'+
        '<div style="font-size:10px;color:#484860;margin-top:3px;">speaker: <span style="color:#8a8aa0;">'+esc(ep.speaker||'—')+'</span> · emotion: <span style="color:#8a8aa0;">'+esc(ep.emotion||'—')+'</span> · '+fago(ep.timestamp)+'</div></div>';

      // ROUTE + LLM stage (LABELLED implicit)
      html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px;">';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Tool route:</span> <span style="color:#0cf;">'+esc(ep.tool_route||'—')+'</span></div>';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">LLM stage:</span> <span style="color:#8a8aa0;font-style:italic;">implicit (inferred from route)</span></div>';
      html += '</div>';

      // IDENTITY: resolved / confidence (fail-CLOSED)
      var ident = ep.identity_state || {};
      var ig = V.gateState(ident.resolved);
      var idC = ig==='ok'?'#0f9':ig==='fail'?'#f44':'#6a6a80';
      html += '<div style="font-size:10px;margin-bottom:8px;"><span style="color:#6a6a80;">Identity resolved:</span> <span style="color:'+idC+';">'+(ig==='ok'?'yes':ig==='fail'?'no':'UNKNOWN')+'</span>'+
        ' <span style="color:#6a6a80;">confidence:</span> <span style="color:#8a8aa0;">'+(ident.confidence!=null?fpct(ident.confidence):'—')+'</span></div>';

      // MEMORIES retrieved
      var mem = ep.memories_retrieved || {};
      html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Memories Retrieved</div>';
      html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;margin-bottom:8px;font-size:10px;">'+
        '<span style="color:#6a6a80;">count:</span> <span style="color:#c0f;">'+fint(mem.count)+'</span>'+
        ' <span style="color:#6a6a80;">route type:</span> <span style="color:#8a8aa0;">'+esc(mem.route_type||'—')+'</span>';
      var subjects = mem.subjects || [];
      if(subjects.length) html += '<div style="margin-top:2px;color:#8a8aa0;">subjects: '+esc(subjects.slice(0,8).join(', '))+'</div>';
      var types = mem.types || {};
      if(isObj(types) && Object.keys(types).length){
        html += '<div style="margin-top:2px;color:#484860;">types: '+Object.keys(types).map(function(k){ return esc(k)+':'+fint(types[k]); }).join(' · ')+'</div>';
      }
      html += '</div>';

      // GATES: epistemic_flags (fail-CLOSED rendering)
      var ef = ep.epistemic_flags || {};
      html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Epistemic Gates</div>';
      html += '<div style="display:flex;gap:8px;flex-wrap:wrap;font-size:10px;margin-bottom:8px;">';
      [['contradiction_touched','contradiction'],['provisional','provisional'],['hallucination_risk','hallucination risk']].forEach(function(pair){
        var g = V.gateState(ef[pair[0]]);
        var c, lbl;
        if(g==='ok'){ c='#0f9'; lbl='clear'; }           // explicit false -> clear
        else if(g==='fail'){ c='#f90'; lbl='RAISED'; }    // explicit true -> raised (warn)
        else { c='#6a6a80'; lbl='UNKNOWN'; }
        html += '<span style="padding:1px 6px;border:1px solid '+c+'44;color:'+c+';border-radius:3px;">'+esc(pair[1])+': '+lbl+'</span>';
      });
      html += '</div>';

      // OUTPUT
      html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Response</div>';
      html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;margin-bottom:8px;font-size:11px;color:#e0e0e8;white-space:pre-wrap;max-height:160px;overflow-y:auto;">'+esc(ep.response_text||'—')+'</div>';

      // LATENCY + FEEDBACK
      html += '<div style="display:flex;gap:12px;flex-wrap:wrap;font-size:10px;margin-bottom:8px;">'+
        '<span><span style="color:#6a6a80;">latency:</span> <span style="color:#0cf;">'+(ep.response_latency_ms!=null?fnum(ep.response_latency_ms,0)+'ms':'—')+'</span></span>'+
        '<span><span style="color:#6a6a80;">follow-up:</span> <span style="color:#8a8aa0;">'+(ep.follow_up!=null?esc(String(ep.follow_up)):'—')+'</span></span>'+
        '<span><span style="color:#6a6a80;">barged-in:</span> <span style="color:#8a8aa0;">'+(ep.barged_in!=null?esc(String(ep.barged_in)):'—')+'</span></span>'+
        '<span><span style="color:#6a6a80;">disagreements:</span> <span style="color:#8a8aa0;">'+fint(ep.disagreement_count)+'</span></span>'+
        '</div>';

      // golden status
      var golden = ep.golden || {};
      if(golden.status || golden.authority_class){
        html += '<div style="font-size:10px;color:#6a6a80;margin-bottom:6px;">golden: <span style="color:#0f9;">'+esc(golden.status||'—')+'</span> · authority: <span style="color:#8a8aa0;">'+esc(golden.authority_class||'—')+'</span></div>';
      }

      // identity/trace ids -> chain into trace
      html += '<div style="font-size:9px;color:#484860;margin-top:4px;">';
      ['conversation_id','trace_id','request_id','output_id'].forEach(function(k){ if(ep[k]) html += k+': '+esc(ep[k])+'  '; });
      html += '</div>';
      if(ep.trace_id){
        html += '<div style="margin-top:6px;"><button class="btn-act" onclick="window.V2D.traceChain(\''+sq(ep.trace_id)+'\')">trace chain →</button></div>';
      }

      V.modal('Flight Episode', html);
    }).catch(function(e){ V.modal('Flight Episode', unknownBox('Error loading episode: '+esc(e.message))); });
  }

  // =====================================================================
  // Debug Snapshot Inspector — GET /api/full-snapshot
  // key dropdown -> pretty JSON of the selected key (+ copy-to-clipboard).
  // =====================================================================
  function debugSnapshot(){
    V.modal('Debug Snapshot Inspector',
      '<div style="display:flex;gap:6px;align-items:center;margin-bottom:8px;">'+
        '<select id="v2d-dbg-key" class="v2-field" style="margin:0;flex:1;"><option>loading…</option></select>'+
        '<button class="btn-act" id="v2d-dbg-copy">copy</button>'+
      '</div>'+
      '<div class="opnote" style="margin-bottom:6px;">GET /api/full-snapshot — raw, read-only. Pick a top-level key to inspect its JSON.</div>'+
      '<pre id="v2d-dbg-out" style="max-height:420px;overflow:auto;padding:8px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;font-size:10px;color:#aaa;white-space:pre-wrap;">Loading…</pre>');
    V.fetchJSON('/api/full-snapshot').then(function(snap){
      var sel = document.getElementById('v2d-dbg-key');
      var out = document.getElementById('v2d-dbg-out');
      var copy = document.getElementById('v2d-dbg-copy');
      if(!sel||!out) return;
      var keys = Object.keys(snap||{}).sort();
      if(!keys.length){ out.textContent = 'snapshot empty'; sel.innerHTML='<option>—</option>'; return; }
      sel.innerHTML = keys.map(function(k){ return '<option value="'+esc(k)+'">'+esc(k)+'</option>'; }).join('');
      function render(){
        var k = sel.value;
        try { out.textContent = JSON.stringify(snap[k], null, 2); }
        catch(e){ out.textContent = String(snap[k]); }
      }
      sel.addEventListener('change', render);
      if(copy) copy.addEventListener('click', function(){
        try { if(navigator.clipboard) navigator.clipboard.writeText(out.textContent); V.toast('copied '+sel.value, true); }
        catch(e){ V.toast('copy failed', false); }
      });
      render();
    }).catch(function(e){
      var out = document.getElementById('v2d-dbg-out');
      if(out) out.textContent = 'Error: '+e.message;
    });
  }

  // =====================================================================
  // Eval Sidecar Detail — window._lastSnap.full.eval
  // ports v1 openEvalDetail (interactives.js:1305): dream buffer/promotion, scoreboard, oracle.
  // =====================================================================
  function evalDetail(){
    full().then(function(fs){
      var ev = (fs && fs.eval) || {};
      if(!Object.keys(ev).length){ V.modal('Eval Sidecar — Full Details', unknownBox('No eval data in snapshot.')); return; }

      var banner = ev.banner || {};
      var sm = ev.store_meta || {};
      var tap = ev.tap || {};
      var coll = ev.collector || {};
      var files = ev.store_file_sizes || {};

      var html = '';

      // banner headline (freshness honest: <30s green else amber; null -> dim)
      var fresh = V.num(banner.data_freshness_s);
      var freshC = fresh===null ? '#6a6a80' : fresh<30 ? '#0f9' : '#ff0';
      html += '<div style="display:flex;gap:12px;margin-bottom:10px;padding:8px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:6px;flex-wrap:wrap;">'+
        '<div><div style="font-size:9px;color:#6a6a80;">Mode</div><div style="font-size:12px;color:#0cf;">'+esc(banner.mode||'—')+'</div></div>'+
        '<div><div style="font-size:9px;color:#6a6a80;">Version</div><div style="font-size:12px;color:#8a8aa0;">'+esc(banner.scoring_version||'—')+'</div></div>'+
        '<div><div style="font-size:9px;color:#6a6a80;">Uptime</div><div style="font-size:12px;color:#8a8aa0;">'+(banner.uptime_s!=null?V.fmtUptime(banner.uptime_s):'—')+'</div></div>'+
        '<div><div style="font-size:9px;color:#6a6a80;">Freshness</div><div style="font-size:12px;color:'+freshC+';">'+(fresh!=null?fnum(fresh,1)+'s':'—')+'</div></div>'+
        '</div>';

      // store metadata
      html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Store Metadata</div>';
      html += '<table style="width:100%;font-size:10px;border-collapse:collapse;margin-bottom:8px;">';
      [['Events Written', fint(sm.total_events_written)],
       ['Snapshots Written', fint(sm.total_snapshots_written)],
       ['Scorecards Written', fint(sm.total_scorecards_written)],
       ['Dropped Events', fint(sm.dropped_event_count)],
       ['Schema Version', sm.schema_version || '—'],
       ['Created', sm.created_at ? fago(sm.created_at) : '—'],
       ['Last Flush', sm.last_flush_ts ? fago(sm.last_flush_ts) : '—'],
       ['Last Scorecard', sm.last_scorecard_ts ? fago(sm.last_scorecard_ts) : '—']
      ].forEach(function(r){
        html += '<tr style="border-bottom:1px solid #0d0d1a;"><td style="padding:2px 4px;color:#6a6a80;">'+r[0]+'</td><td style="padding:2px 4px;text-align:right;">'+esc(String(r[1]))+'</td></tr>';
      });
      html += '</table>';

      // file sizes
      if(Object.keys(files).length){
        html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">File Sizes</div>';
        html += '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-bottom:8px;">';
        Object.keys(files).forEach(function(k){
          html += '<div style="padding:4px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;text-align:center;">'+
            '<div style="font-size:9px;color:#6a6a80;">'+esc(k)+'</div>'+
            '<div style="font-size:11px;color:#0cf;">'+fbytes(files[k])+'</div></div>';
        });
        html += '</div>';
      }

      // rotation counts
      var rotations = sm.rotation_counts || {};
      if(Object.keys(rotations).length){
        html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Rotation Counts</div>';
        html += '<div style="display:flex;gap:8px;font-size:10px;margin-bottom:8px;flex-wrap:wrap;">';
        Object.keys(rotations).forEach(function(k){ var c=(V.num(rotations[k])||0)>0?'#f90':'#484860'; html += '<span style="color:'+c+';">'+esc(k)+': '+fint(rotations[k])+'</span>'; });
        html += '</div>';
      }

      // event tap (wired fail-CLOSED)
      var wg = V.gateState(tap.wired);
      html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Event Tap</div>';
      html += '<div style="display:flex;gap:12px;font-size:10px;margin-bottom:8px;flex-wrap:wrap;">'+
        '<span>wired: <span style="color:'+(wg==='ok'?'#0f9':wg==='fail'?'#f44':'#6a6a80')+';">'+(wg==='ok'?'yes':wg==='fail'?'no':'UNKNOWN')+'</span></span>'+
        '<span>buffer: '+fint(tap.buffer_size)+'</span>'+
        '<span>total buffered: '+fint(tap.total_buffered)+'</span>'+
        '<span>event types: '+fint(tap.tapped_event_count)+'</span>'+
        '<span>mode: <span style="color:#0cf;">'+esc(tap.current_mode||'—')+'</span></span>'+
        '</div>';

      // collector
      var collErr = V.num(coll.collect_errors);
      html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Collector</div>';
      html += '<div style="display:flex;gap:12px;font-size:10px;margin-bottom:8px;flex-wrap:wrap;">'+
        '<span>snapshots: '+fint(coll.snapshots_collected)+'</span>'+
        '<span>interval: '+(coll.interval_s!=null?fnum(coll.interval_s,0)+'s':'—')+'</span>'+
        '<span>errors: <span style="color:'+(collErr&&collErr>0?'#f44':'#0f9')+';">'+fint(coll.collect_errors)+'</span></span>'+
        '<span>last: '+(coll.last_collect_ts?fago(coll.last_collect_ts):'—')+'</span>'+
        '</div>';

      // all event counts (scoreboard)
      var ec = ev.event_counts || {};
      var ecEntries = Object.keys(ec).map(function(k){ return [k, V.num(ec[k])]; }).sort(function(a,b){ return (b[1]||0)-(a[1]||0); });
      if(ecEntries.length){
        html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">All Event Counts ('+ecEntries.length+' types)</div>';
        html += '<div style="max-height:200px;overflow-y:auto;border:1px solid #1a1a2e;border-radius:4px;padding:4px;margin-bottom:8px;">';
        ecEntries.forEach(function(e){
          html += '<div style="display:flex;justify-content:space-between;padding:1px 4px;font-size:9px;border-bottom:1px solid #0d0d1a;">'+
            '<span style="color:#8a8aa0;">'+esc(e[0])+'</span><span style="color:#0cf;">'+fint(e[1])+'</span></div>';
        });
        html += '</div>';
      }

      // dream artifacts (buffer / promotion / avg conf / avg coh)
      var dream = ev.dream || {};
      if(dream.buffer_size!=null || Object.keys(dream).length){
        html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Dream Artifacts</div>';
        html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:4px;">';
        html += '<div style="padding:3px;background:#0d0d1a;border-radius:3px;text-align:center;"><div style="font-size:9px;color:#6a6a80;">buffer</div><div style="font-size:11px;color:#c0f;">'+fint(dream.buffer_size)+'</div></div>';
        html += '<div style="padding:3px;background:#0d0d1a;border-radius:3px;text-align:center;"><div style="font-size:9px;color:#6a6a80;">promo rate</div><div style="font-size:11px;color:'+(dream.promotion_rate_color==='green'?'#0f9':'#ff0')+';">'+fpct(dream.promotion_rate)+'</div></div>';
        html += '<div style="padding:3px;background:#0d0d1a;border-radius:3px;text-align:center;"><div style="font-size:9px;color:#6a6a80;">avg conf</div><div style="font-size:11px;">'+fpct(dream.avg_confidence)+'</div></div>';
        html += '<div style="padding:3px;background:#0d0d1a;border-radius:3px;text-align:center;"><div style="font-size:9px;color:#6a6a80;">avg coh</div><div style="font-size:11px;">'+fpct(dream.avg_coherence)+'</div></div>';
        html += '</div>';
        var byType = dream.by_type || {};
        if(Object.keys(byType).length){
          html += '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:8px;">';
          Object.keys(byType).forEach(function(k){ html += '<span style="padding:1px 4px;border:1px solid #c0f33;border-radius:2px;font-size:9px;color:#c0f;">'+esc(V.cap(k))+': '+fint(byType[k])+'</span>'; });
          html += '</div>';
        }
      }

      // oracle / behavioral checks (self-report honesty + emotional independence)
      var srh = ev.self_report_honesty || {};
      var ei = ev.emotional_independence || {};
      var oracle = ev.oracle || {};
      if(Object.keys(srh).length || Object.keys(ei).length || Object.keys(oracle).length){
        html += '<div style="font-size:11px;color:#6a6a80;margin-bottom:3px;">Oracle / Behavioral Checks</div>';
        html += '<div style="display:flex;gap:12px;font-size:10px;margin-bottom:4px;flex-wrap:wrap;">';
        if(srh.score!=null) html += '<span>self-report honesty: <span style="color:'+(srh.color==='green'?'#0f9':'#ff0')+';">'+fpct(srh.score)+'</span></span>';
        if(ei.score!=null) html += '<span>emotional independence: <span style="color:'+(ei.color==='green'?'#0f9':'#ff0')+';">'+fpct(ei.score)+'</span></span>';
        html += '</div>';
        if(Object.keys(oracle).length){
          html += '<pre style="max-height:160px;overflow:auto;padding:6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;font-size:9px;color:#8a8aa0;white-space:pre-wrap;">'+esc(JSON.stringify(oracle,null,2))+'</pre>';
        }
      }

      V.modal('Eval Sidecar — Full Details', html);
    }).catch(function(e){ V.modal('Eval Sidecar — Full Details', unknownBox('Error loading eval: '+esc(e.message))); });
  }

  // =====================================================================
  // Memory Detail — GET /api/memories/{id}
  // ports v1 openMemoryDetail (interactives.js:232).
  // associations[] clickable -> recursive memoryDetail.
  // =====================================================================
  function memoryDetail(memId){
    if(!memId){ V.modal('Memory Detail', unknownBox('No memory id supplied.')); return; }
    V.modal('Memory Detail', '<div id="v2d-mem-body" style="font-size:11px;color:#6a6a80;">Loading memory '+esc(memId)+'…</div>');
    V.fetchJSON('/api/memories/'+encodeURIComponent(memId)).then(function(m){
      var body = document.getElementById('v2d-mem-body'); if(!body) return;
      var w = V.num(m.weight);
      var wc = w===null ? '#6a6a80' : w>0.5 ? '#0f9' : w>0.2 ? '#ff0' : '#6a6a80';
      var html = '';
      html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px;">';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Type:</span> <span style="color:#0cf;">'+esc(m.type||'—')+'</span></div>';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Weight:</span> <span style="color:'+wc+';">'+fnum(w,3)+'</span></div>';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Provenance:</span> '+esc(m.provenance||'—')+'</div>';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Created:</span> '+(m.timestamp?fago(m.timestamp):'—')+'</div>';
      if(m.speaker) html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Speaker:</span> '+esc(m.speaker)+'</div>';
      html += '<div style="font-size:10px;"><span style="color:#6a6a80;">Accessed:</span> '+(m.access_count!=null?fint(m.access_count)+'x':'—')+'</div>';
      html += '</div>';

      html += '<div style="background:#0d0d1a;padding:8px;border-radius:4px;border:1px solid #1a1a2e;margin-bottom:8px;font-size:11px;white-space:pre-wrap;max-height:200px;overflow-y:auto;">'+esc(m.payload||m.content||'—')+'</div>';

      var tags = m.tags || [];
      if(tags.length){
        html += '<div style="margin-bottom:6px;display:flex;flex-wrap:wrap;gap:3px;">';
        tags.forEach(function(tg){ html += '<span style="padding:1px 5px;background:#181828;border:1px solid #2a2a44;border-radius:3px;font-size:9px;color:#c0f;">#'+esc(tg)+'</span>'; });
        html += '</div>';
      }

      var assoc = m.associations || [];
      if(assoc.length){
        html += '<div style="font-size:10px;color:#6a6a80;margin-bottom:3px;">Associations</div>';
        assoc.slice(0,10).forEach(function(a){
          var aId = (a && typeof a==='object') ? (a.target_id || a.id || '') : String(a);
          html += '<div style="font-size:9px;padding:1px 0;cursor:pointer;color:#0cf;" onclick="window.V2D.memoryDetail(\''+sq(aId)+'\')">'+esc(String(aId).substring(0,40))+'</div>';
        });
      }

      html += '<div style="margin-top:8px;font-size:9px;color:#484860;">ID: '+esc(m.id||memId)+'</div>';
      body.innerHTML = html;
    }).catch(function(e){
      var body = document.getElementById('v2d-mem-body');
      if(body) body.innerHTML = (String(e.message||'').indexOf('404')>-1)
        ? unknownBox('Memory not found.')
        : unknownBox('Error: '+esc(e.message));
    });
  }

  // =====================================================================
  // Memory Search — form -> GET /api/memories/search?q=&limit=30(&type=) -> list -> memoryDetail
  // ports v1 openMemorySearch (interactives.js:189).
  // =====================================================================
  function memorySearch(){
    V.modal('Memory Search',
      '<form id="v2d-mem-search-form" style="display:flex;gap:6px;margin-bottom:8px;">'+
        '<input id="v2d-mem-q" type="text" placeholder="Search memories…" class="v2-field" style="margin:0;flex:1;" autocomplete="off">'+
        '<select id="v2d-mem-type" class="v2-field" style="margin:0;width:auto;">'+
          '<option value="">All types</option><option value="observation">observation</option><option value="conversation">conversation</option><option value="core">core</option><option value="insight">insight</option><option value="research">research</option><option value="identity">identity</option>'+
        '</select>'+
        '<button type="submit" class="btn-act">Search</button>'+
      '</form>'+
      '<div id="v2d-mem-results" style="max-height:380px;overflow-y:auto;font-size:11px;"></div>');
    var form = document.getElementById('v2d-mem-search-form');
    var qIn = document.getElementById('v2d-mem-q');
    if(qIn) qIn.focus();
    if(form) form.addEventListener('submit', function(e){
      e.preventDefault();
      var q = (document.getElementById('v2d-mem-q')||{}).value || '';
      var type = (document.getElementById('v2d-mem-type')||{}).value || '';
      var out = document.getElementById('v2d-mem-results');
      if(out) out.innerHTML = '<div style="color:#6a6a80;">Searching…</div>';
      var url = '/api/memories/search?q='+encodeURIComponent(q)+'&limit=30';
      if(type) url += '&type='+encodeURIComponent(type);
      V.fetchJSON(url).then(function(results){
        if(!out) return;
        results = results || [];
        if(!results.length){ out.innerHTML = '<div style="color:#484860;">No results</div>'; return; }
        var h = '';
        results.forEach(function(m){
          var w = V.num(m.weight);
          var wc = w===null ? '#6a6a80' : w>0.5 ? '#0f9' : w>0.2 ? '#ff0' : '#6a6a80';
          h += '<div style="padding:5px;border-bottom:1px solid #1a1a2e;cursor:pointer;" onclick="window.V2D.memoryDetail(\''+sq(m.id)+'\')">'+
            '<div style="display:flex;justify-content:space-between;align-items:center;">'+
              '<span style="font-size:9px;color:#0cf;padding:1px 4px;border:1px solid #0cf33;border-radius:2px;">'+esc(m.type||'—')+'</span>'+
              '<span style="font-size:9px;color:'+wc+';">w:'+fnum(w,2)+'</span>'+
            '</div>'+
            '<div style="margin-top:2px;">'+esc(String(m.payload||m.content||'').substring(0,200))+'</div>'+
            '<div style="font-size:9px;color:#484860;margin-top:1px;">'+
              ((m.tags&&m.tags.length)?m.tags.map(function(t){ return '#'+esc(t); }).join(' '):'')+
              (m.timestamp?' · '+fago(m.timestamp):'')+
            '</div></div>';
        });
        out.innerHTML = h;
      }).catch(function(err){ if(out) out.innerHTML = unknownBox('Error: '+esc(err.message)); });
    });
  }

  // expose every modal at the end
  return {
    thoughtDetail: thoughtDetail,
    researchDetail: researchDetail,
    deltaDetail: deltaDetail,
    traceExplorer: traceExplorer,
    traceChain: traceChain,
    flightDetail: flightDetail,
    debugSnapshot: debugSnapshot,
    evalDetail: evalDetail,
    memoryDetail: memoryDetail,
    memorySearch: memorySearch
  };
})();
