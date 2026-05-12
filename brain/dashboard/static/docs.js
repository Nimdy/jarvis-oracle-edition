/* ═══════════════════════════════════════════════════════════════════════════
   JARVIS System Reference — docs.js
   Static reference page: architecture diagram, sidebar scroll tracking,
   smooth navigation. No WebSocket, no live data.
   ═══════════════════════════════════════════════════════════════════════════ */

(function() {
  'use strict';

  // ═══════════════════════════════════════════════════════════════════════════
  // 1. Sidebar scroll spy
  // ═══════════════════════════════════════════════════════════════════════════

  var navLinks = document.querySelectorAll('.d-nav-link[href^="#"]');
  var sections = [];
  navLinks.forEach(function(link) {
    var id = link.getAttribute('href').slice(1);
    var sec = document.getElementById(id);
    if (sec) sections.push({ el: sec, link: link });
  });

  function updateActiveNav() {
    var scrollY = window.scrollY + 120;
    var active = null;
    for (var i = sections.length - 1; i >= 0; i--) {
      if (sections[i].el.offsetTop <= scrollY) {
        active = sections[i];
        break;
      }
    }
    navLinks.forEach(function(l) { l.classList.remove('active'); });
    if (active) active.link.classList.add('active');
  }

  var _scrollRaf = 0;
  window.addEventListener('scroll', function() {
    if (_scrollRaf) return;
    _scrollRaf = requestAnimationFrame(function() {
      _scrollRaf = 0;
      updateActiveNav();
    });
  });
  updateActiveNav();

  // ═══════════════════════════════════════════════════════════════════════════
  // 2. Architecture diagram (static version — no live data)
  // ═══════════════════════════════════════════════════════════════════════════

  function renderArchDiagram() {
    var container = document.getElementById('arch-diagram-container');
    if (!container) return;

    var svgW = 900, svgH = 620;
    var svg = '';
    svg += '<svg id="arch-svg" viewBox="0 0 ' + svgW + ' ' + svgH + '" ' +
      'xmlns="http://www.w3.org/2000/svg" style="font-family:monospace;">';

    svg += '<defs>' +
      '<marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">' +
      '<path d="M0,0 L8,3 L0,6 Z" fill="#4a4a6a"/></marker>' +
      '<marker id="ah-green" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">' +
      '<path d="M0,0 L8,3 L0,6 Z" fill="#0f9"/></marker>' +
      '<marker id="ah-cyan" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">' +
      '<path d="M0,0 L8,3 L0,6 Z" fill="#0cf"/></marker>' +
      '<marker id="ah-amber" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">' +
      '<path d="M0,0 L8,3 L0,6 Z" fill="#f90"/></marker>' +
      '</defs>';

    svg += '<rect width="' + svgW + '" height="' + svgH + '" fill="#0a0a14" rx="6"/>';

    // Section backgrounds
    svg += '<rect x="20" y="15" width="860" height="95" rx="4" fill="#0f0f1f" stroke="#2a2a4a" stroke-width="0.5"/>';
    svg += '<text x="30" y="32" fill="#6a6a80" font-size="9" font-weight="700">PERCEPTION &amp; CONVERSATION</text>';

    svg += '<rect x="20" y="120" width="420" height="180" rx="4" fill="#0c1a0f" stroke="#1a3a2a" stroke-width="0.5"/>';
    svg += '<text x="30" y="137" fill="#2a6a4a" font-size="9" font-weight="700">PHASE 5 — CONTINUOUS IMPROVEMENT</text>';

    svg += '<rect x="460" y="120" width="420" height="180" rx="4" fill="#0f0f1a" stroke="#2a2a4a" stroke-width="0.5"/>';
    svg += '<text x="470" y="137" fill="#4a4a8a" font-size="9" font-weight="700">NEURAL NETWORKS &amp; DISTILLATION</text>';

    svg += '<rect x="20" y="310" width="420" height="130" rx="4" fill="#1a0f0c" stroke="#3a2a1a" stroke-width="0.5"/>';
    svg += '<text x="30" y="327" fill="#8a6a4a" font-size="9" font-weight="700">MEMORY &amp; KNOWLEDGE</text>';

    svg += '<rect x="460" y="310" width="420" height="130" rx="4" fill="#1a0c1a" stroke="#3a1a3a" stroke-width="0.5"/>';
    svg += '<text x="470" y="327" fill="#8a4a8a" font-size="9" font-weight="700">WORLD MODEL &amp; COGNITION</text>';

    svg += '<rect x="20" y="450" width="860" height="155" rx="4" fill="#0c0c18" stroke="#2a2a4a" stroke-width="0.5"/>';
    svg += '<text x="30" y="467" fill="#4a4a8a" font-size="9" font-weight="700">EPISTEMIC INTEGRITY STACK (L0–L12 + L3A/L3B)</text>';

    // Node helper
    var _nIdx = 0;
    function node(x, y, w, h, label, sublabel, color, tip, navTarget) {
      var nid = 'dn-' + (_nIdx++);
      var r = '<g class="arch-node" id="' + nid + '" style="cursor:pointer;" ' +
        'data-tip="' + (tip || '').replace(/"/g, '&quot;') + '" ' +
        'data-nav="' + (navTarget || '') + '" ' +
        'data-cx="' + (x + w/2) + '" data-cy="' + y + '">';
      r += '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="3" ' +
        'fill="#0a0a14" stroke="' + color + '" stroke-width="1.2" class="arch-node-bg"/>';
      r += '<text x="' + (x + w/2) + '" y="' + (y + h/2 - (sublabel ? 3 : 0)) + '" ' +
        'fill="' + color + '" font-size="9" font-weight="600" text-anchor="middle" dominant-baseline="middle">' + label + '</text>';
      if (sublabel) {
        r += '<text x="' + (x + w/2) + '" y="' + (y + h/2 + 9) + '" ' +
          'fill="#6a6a80" font-size="7" text-anchor="middle" dominant-baseline="middle">' + sublabel + '</text>';
      }
      r += '</g>';
      return r;
    }

    function arrow(x1, y1, x2, y2, color, dashed) {
      color = color || '#4a4a6a';
      var marker = color === '#0f9' ? 'ah-green' : color === '#0cf' ? 'ah-cyan' : color === '#f90' ? 'ah-amber' : 'ah';
      var dashAttr = dashed ? ' stroke-dasharray="4,3"' : '';
      return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" ' +
        'stroke="' + color + '" stroke-width="1"' + dashAttr + ' marker-end="url(#' + marker + ')"/>';
    }

    function curvedArrow(x1, y1, cx, cy, x2, y2, color) {
      color = color || '#4a4a6a';
      var marker = color === '#0f9' ? 'ah-green' : color === '#0cf' ? 'ah-cyan' : color === '#f90' ? 'ah-amber' : 'ah';
      return '<path d="M' + x1 + ',' + y1 + ' Q' + cx + ',' + cy + ' ' + x2 + ',' + y2 + '" ' +
        'fill="none" stroke="' + color + '" stroke-width="1" marker-end="url(#' + marker + ')"/>';
    }

    // --- PERCEPTION ROW ---
    svg += node(35, 42, 100, 55, 'Pi Senses', 'vision+audio', '#0cf',
      'Raspberry Pi 5 + Hailo-10H AI HAT+: camera vision, mic capture, cyberpunk particle display', 'two-device-split');
    svg += node(160, 42, 110, 55, 'Wake/VAD/STT', 'speech → text', '#0cf',
      'openWakeWord detection, Silero VAD speech segmentation, faster-whisper GPU transcription', 'flow-voice');
    svg += node(295, 42, 110, 55, 'Tool Router', '13 intent classes', '#0cf',
      'Keyword + regex intent dispatch: time, system, status, memory, vision, introspection, identity, skill, web, etc.', 'flow-conversation');
    svg += node(430, 42, 115, 55, 'Conversation', 'LLM + response', '#0cf',
      'Ollama LLM streaming response with cancel-token, TTS synthesis via Kokoro, audio sent to Pi', 'flow-conversation');
    svg += node(570, 42, 115, 55, 'Personal Intel', 'fact capture', '#f90',
      'Extracts interests, dislikes, facts, preferences. Filtered by _is_unstable_personal_fact(). Corrections downweight junk.', 'flow-personal-intel');
    svg += node(710, 42, 135, 55, 'Friction Miner', 'corrections/rephrase', '#f44',
      'Detects corrections, rephrases, annoyance in conversation. Feeds friction_rate metric trigger for Phase 5.', 'flow-phase5');

    svg += arrow(135, 70, 160, 70, '#0cf');
    svg += arrow(270, 70, 295, 70, '#0cf');
    svg += arrow(405, 70, 430, 70, '#0cf');
    svg += arrow(545, 70, 570, 70, '#f90');
    svg += arrow(545, 55, 710, 55, '#f44');

    // --- PHASE 5 ---
    svg += node(35, 150, 120, 50, 'Metric Triggers', '7 deficit dimensions', '#0f9',
      '7 deficit dimensions (retrieval, friction, autonomy, etc.). Spawns research intents with tool rotation.', 'flow-phase5');
    svg += node(175, 150, 120, 50, 'Research Intent', 'tool rotation', '#0f9',
      'Research question + tool hint (introspection/codebase/web/academic). Queued by triggers or drives.', 'flow-phase5');
    svg += node(315, 150, 110, 50, 'Knowledge', 'integration', '#0f9',
      'Pre-research knowledge check, conflict detection. Registers sources in ledger. Provenance-gated memory writes.', 'flow-phase5');

    svg += node(35, 220, 120, 50, 'Source Ledger', 'usefulness track', '#0f9',
      'Tracks per-source retrieval count and usefulness. log_outcome() is single authority. Feeds back to scorer.', 'flow-phase5');
    svg += node(175, 220, 120, 50, 'Interventions', 'shadow → measure', '#0f9',
      'Shadow-evaluates proposed changes for 24h. Captures baseline_value at activation, computes measured_delta.', 'flow-phase5');
    svg += node(315, 220, 110, 50, 'Opp. Scorer', 'priority queue', '#0f9',
      'Score = Impact × Evidence × Confidence - Risk - Cost. Policy memory + source ledger adjust ±0.3.', 'flow-phase5');

    svg += arrow(155, 175, 175, 175, '#0f9');
    svg += arrow(295, 175, 315, 175, '#0f9');
    svg += arrow(370, 200, 370, 220, '#0f9');
    svg += arrow(315, 245, 295, 245, '#0f9');
    svg += arrow(175, 245, 155, 245, '#0f9');
    svg += arrow(95, 220, 95, 200, '#0f9');
    svg += curvedArrow(710, 97, 400, 130, 155, 150, '#f44');
    svg += arrow(155, 235, 175, 235, '#0f9');
    svg += curvedArrow(95, 270, 200, 290, 325, 270, '#0f9');

    // --- NEURAL NETWORKS ---
    svg += node(475, 150, 130, 50, 'Distillation', 'teacher → specialist', '#c0f',
      'GPU models, routers, validators, and lifecycle outcomes teach Tier-1 specialists via fidelity-weighted loss.', 'flow-distillation');
    svg += node(625, 150, 125, 50, 'Tier-1 Specialists', 'perception+claims+skills', '#c0f',
      'Perception specialists plus language_style, plan_evaluator, diagnostic, code_quality, claim_classifier, dream_synthesis, skill_acquisition, and HRR encoder stub.', 'flow-distillation');
    svg += node(770, 150, 95, 50, 'Tier-2', 'hemispheres', '#c0f',
      'Dynamic architecture NNs. Focuses: MEMORY, MOOD, TRAITS, GENERAL + Matrix specialists.', 'flow-distillation');

    svg += node(475, 220, 130, 50, 'Z-Score Norm', 'audio_features*', '#c0f',
      'Normalizes mixed-scale audio features (spectral ~2000, RMS ~0.05, ECAPA ~±1) before training.', 'flow-distillation');
    svg += node(625, 220, 125, 50, 'Policy NN', 'shadow A/B', '#c0f',
      '20-dim state → 8 behavioral knobs. Shadow A/B evaluation. >55% decisive win rate to promote.', 'flow-policy');
    svg += node(770, 220, 95, 50, 'Broadcast', '4-6 slots', '#c0f',
      'Global Broadcast Slots feed Tier-2 outputs into Policy NN state encoder (dims 16-19).', 'flow-distillation');

    svg += arrow(605, 175, 625, 175, '#c0f');
    svg += arrow(750, 175, 770, 175, '#c0f');
    svg += arrow(605, 245, 625, 245, '#c0f');
    svg += arrow(750, 245, 770, 245, '#c0f');
    svg += arrow(540, 200, 540, 220, '#c0f');
    svg += arrow(817, 200, 817, 220, '#c0f');

    // --- MEMORY ---
    svg += node(35, 340, 120, 50, 'Memory Store', 'unified write path', '#f90',
      'engine.remember() → quarantine → salience → storage.add() → index → MEMORY_WRITE event.', 'flow-memory');
    svg += node(175, 340, 120, 50, 'Vector Search', 'embedding extract', '#f90',
      '_extract_embedding_text() pulls human-readable fields from structured payloads instead of repr().', 'flow-memory');
    svg += node(315, 340, 110, 50, 'Retrieval Log', 'single authority', '#f90',
      'log_outcome() is the ONLY call site for source_ledger.record_retrieval(). Prevents double-counting.', 'flow-memory');

    svg += node(35, 400, 120, 26, 'CueGate', 'memory safety', '#f90',
      'Single authority for memory access policy. Blocks observation writes during dream/sleep/reflective.', 'flow-memory');
    svg += node(175, 400, 120, 26, 'Cortex NNs', 'ranker + salience', '#f90',
      'MemoryRanker (MLP 12→32→16→1) + SalienceModel (MLP 11→24→12→3). Trained during dream cycles.', 'flow-memory');
    svg += node(315, 400, 110, 26, 'Consolidation', 'dream cycle', '#f90',
      'Associate, reinforce, decay, summaries. Uses begin/end_consolidation() window.', 'flow-memory');

    svg += arrow(155, 365, 175, 365, '#f90');
    svg += arrow(295, 365, 315, 365, '#f90');
    svg += curvedArrow(627, 97, 300, 300, 95, 340, '#f90');
    svg += curvedArrow(370, 340, 420, 290, 95, 270, '#0f9');

    // --- WORLD MODEL ---
    svg += node(475, 340, 130, 50, 'World Model', 'belief state', '#c8f',
      'Fuses 9 subsystem snapshots into unified WorldState. Shadow → advisory → active promotion.', 'flow-world-model');
    svg += node(625, 340, 125, 50, 'Causal Engine', '18 rules', '#c8f',
      'Heuristic rules producing predicted state deltas. Priority-based conflict resolution.', 'flow-world-model');
    svg += node(770, 340, 95, 50, 'Simulator', 'what-if', '#c8f',
      'Hypothetical projections. Read-only, max 3 steps. Shadow → advisory promotion.', 'flow-world-model');

    svg += node(475, 400, 130, 26, 'WM Delta Events', 'emitted per tick', '#c8f',
      'WORLD_MODEL_DELTA for each detected change. Consumed by EvalSidecar, CuriosityQuestionBuffer.', 'flow-world-model');

    svg += arrow(605, 365, 625, 365, '#c8f');
    svg += arrow(750, 365, 770, 365, '#c8f');
    svg += arrow(540, 390, 540, 400, '#c8f');

    // --- EPISTEMIC STACK ---
    var layers = [
      { label: 'L0', name: 'Cap Gate', color: '#0f9', tip: 'Capability Gate: 7 enforcement layers, 15 claim patterns + action confabulation guard. Unverified claims rewritten.' },
      { label: 'L1', name: 'Attrib', color: '#0f9', tip: 'Attribution Ledger: append-only JSONL event truth, causal chains, outcome resolution.' },
      { label: 'L2', name: 'Provenance', color: '#0f9', tip: 'Every memory carries provenance. Retrieval boost by source type.' },
      { label: 'L3', name: 'Identity', color: '#0f9', tip: 'Identity Boundary Engine: retrieval policy matrix prevents cross-identity leaks.' },
      { label: 'L3A', name: 'Persist', color: '#0f9', tip: 'Identity Persistence: voice and face continuity with decaying confidence across short recognition gaps.' },
      { label: 'L3B', name: 'Scene', color: '#ff0', tip: 'Persistent Scene Model (shadow): entity tracking, region mapping, display classification.' },
      { label: 'L4', name: 'Outcomes', color: '#0f9', tip: 'Delayed Outcome Attribution: counterfactual baselines for causal credit.' },
      { label: 'L5', name: 'Contradict', color: '#0f9', tip: 'Typed Contradiction Engine: 6 conflict classes, debt management.' },
      { label: 'L6', name: 'Calibrate', color: '#0f9', tip: 'Truth Calibration: 8-domain scoring, Brier/ECE, drift detection.' },
      { label: 'L7', name: 'Beliefs', color: '#0f9', tip: 'Belief Confidence Graph: weighted evidence edges, 6 sacred invariants.' },
      { label: 'L8', name: 'Quarantine', color: '#ff0', tip: 'Cognitive Quarantine (Active-Lite): 5 anomaly categories, EMA pressure.' },
      { label: 'L9', name: 'Audit', color: '#0f9', tip: 'Reflective Audit: 6-dimension scan, severity-weighted scoring.' },
      { label: 'L10', name: 'Soul', color: '#0f9', tip: 'Soul Integrity Index: 10-dimension weighted composite.' },
      { label: 'L11', name: 'Compact', color: '#0f9', tip: 'Epistemic Compaction: weight caps, subject-version collapsing, edge budgets.' },
      { label: 'L12', name: 'Intent', color: '#0f9', tip: 'Intention Truth Layer: tracks backed commitments and keeps IntentionResolver delivery scoring shadow-only until promoted.' }
    ];

    var lx = 35;
    for (var li = 0; li < layers.length; li++) {
      var lw = 54;
      var lyr = layers[li];
      var eid = 'de-' + li;
      svg += '<g class="arch-node" id="' + eid + '" style="cursor:pointer;" ' +
        'data-tip="' + lyr.tip.replace(/"/g, '&quot;') + '" data-nav="epistemic-stack" ' +
        'data-cx="' + (lx + lw/2) + '" data-cy="480">';
      svg += '<rect x="' + lx + '" y="480" width="' + lw + '" height="40" rx="2" ' +
        'fill="#0a0a14" stroke="' + lyr.color + '" stroke-width="0.8" class="arch-node-bg"/>';
      svg += '<text x="' + (lx + lw/2) + '" y="494" fill="' + lyr.color + '" ' +
        'font-size="8" font-weight="700" text-anchor="middle">' + lyr.label + '</text>';
      svg += '<text x="' + (lx + lw/2) + '" y="508" fill="#6a6a80" ' +
        'font-size="7" text-anchor="middle">' + lyr.name + '</text>';
      svg += '</g>';
      lx += lw + 4;
    }

    // Footer text
    svg += '<text x="450" y="560" fill="#4a4a6a" font-size="8" text-anchor="middle" font-style="italic">' +
      'Every outgoing response passes through L0 Capability Gate. Every memory write passes through CueGate.' +
      '</text>';
    svg += '<text x="450" y="575" fill="#4a4a6a" font-size="8" text-anchor="middle" font-style="italic">' +
      'Beliefs are extracted (L5), calibrated (L6), graph-linked (L7), audited (L9-L10), and commitments tracked (L12).' +
      '</text>';

    // Tooltip group (last so it renders on top)
    svg += '<g id="arch-tip-g" style="display:none;pointer-events:none;">' +
      '<rect id="arch-tip-bg" rx="4" fill="#10101e" fill-opacity="0.96" stroke="#4a4a6a" stroke-width="0.5"/>' +
      '<text id="arch-tip-text" fill="#ddd" font-size="8" dominant-baseline="hanging"></text>' +
      '</g>';

    svg += '</svg>';

    // Legend
    svg += '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;font-size:0.68rem;">' +
      '<span style="color:#0cf;">\u25cf Perception</span>' +
      '<span style="color:#0f9;">\u25cf Phase 5 Loop</span>' +
      '<span style="color:#f44;">\u25cf Friction Detection</span>' +
      '<span style="color:#f90;">\u25cf Memory/Knowledge</span>' +
      '<span style="color:#c0f;">\u25cf Neural Networks</span>' +
      '<span style="color:#c8f;">\u25cf World Model</span>' +
      '<span style="color:#4a4a6a;">\u25cf Epistemic Stack</span>' +
      '</div>';

    container.innerHTML = svg;

    wireArchEvents();
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // 3. Diagram interactivity (tooltips + click-to-scroll)
  // ═══════════════════════════════════════════════════════════════════════════

  function wireArchEvents() {
    var svgEl = document.getElementById('arch-svg');
    if (!svgEl) return;
    var tipG = document.getElementById('arch-tip-g');
    var tipBg = document.getElementById('arch-tip-bg');
    var tipText = document.getElementById('arch-tip-text');
    if (!tipG || !tipBg || !tipText) return;

    var nodes = svgEl.querySelectorAll('.arch-node');
    nodes.forEach(function(g) {
      g.addEventListener('mouseenter', function() {
        var tip = g.getAttribute('data-tip');
        if (!tip) return;
        var cx = parseFloat(g.getAttribute('data-cx')) || 0;
        var cy = parseFloat(g.getAttribute('data-cy')) || 0;

        var lines = [];
        var words = tip.split(' ');
        var line = '';
        for (var i = 0; i < words.length; i++) {
          var test = line ? line + ' ' + words[i] : words[i];
          if (test.length > 60 && line) { lines.push(line); line = words[i]; }
          else { line = test; }
        }
        if (line) lines.push(line);

        while (tipText.firstChild) tipText.removeChild(tipText.firstChild);
        var lineH = 11;
        for (var j = 0; j < lines.length; j++) {
          var tspan = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
          tspan.setAttribute('x', '0');
          tspan.setAttribute('dy', j === 0 ? '0' : '' + lineH);
          tspan.textContent = lines[j];
          tipText.appendChild(tspan);
        }

        var padX = 8, padY = 6;
        var textW = 0;
        for (var k = 0; k < lines.length; k++) {
          textW = Math.max(textW, lines[k].length * 4.8);
        }
        var textH = lines.length * lineH;

        var tipX = Math.max(5, Math.min(cx - textW / 2, 900 - textW - padX * 2 - 5));
        var tipY = cy - textH - padY * 2 - 8;
        if (tipY < 5) tipY = cy + 55;

        tipBg.setAttribute('x', tipX);
        tipBg.setAttribute('y', tipY);
        tipBg.setAttribute('width', textW + padX * 2);
        tipBg.setAttribute('height', textH + padY * 2);
        tipText.setAttribute('x', tipX + padX);
        tipText.setAttribute('y', tipY + padY);
        tipText.querySelectorAll('tspan').forEach(function(ts) { ts.setAttribute('x', tipX + padX); });

        tipG.style.display = '';
        var bgRect = g.querySelector('.arch-node-bg');
        if (bgRect) bgRect.setAttribute('fill', '#16162a');
      });

      g.addEventListener('mouseleave', function() {
        tipG.style.display = 'none';
        var bgRect = g.querySelector('.arch-node-bg');
        if (bgRect) bgRect.setAttribute('fill', '#0a0a14');
      });

      g.addEventListener('click', function() {
        var nav = g.getAttribute('data-nav');
        if (nav) {
          var target = document.getElementById(nav);
          if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            target.style.outline = '2px solid rgba(0, 204, 255, 0.4)';
            setTimeout(function() { target.style.outline = 'none'; }, 2000);
          }
        }
      });
    });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // 4. Init
  // ═══════════════════════════════════════════════════════════════════════════

  renderArchDiagram();

})();
