'use strict';

/* ═══════════════════════════════════════════════════════════════════════════
   JARVIS Dashboard — Interactive Features (CRUD, modals, forms, camera)
   Loaded AFTER dashboard.js. Uses window._apiKey, _authHeaders, esc, etc.
   ═══════════════════════════════════════════════════════════════════════════ */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _fmtUptime(s) {
  if (s == null) return '--';
  var d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d > 0) return d + 'd ' + h + 'h';
  if (h > 0) return h + 'h ' + m + 'm';
  return m + 'm ' + Math.floor(s % 60) + 's';
}

function _fmtBytes(b) {
  if (b == null) return '--';
  if (b < 1024) return b + 'B';
  if (b < 1048576) return (b / 1024).toFixed(1) + 'KB';
  if (b < 1073741824) return (b / 1048576).toFixed(1) + 'MB';
  return (b / 1073741824).toFixed(1) + 'GB';
}

function _authHeaders(extra) {
  var h = Object.assign({'Content-Type': 'application/json'}, extra || {});
  if (window._apiKey) h['Authorization'] = 'Bearer ' + window._apiKey;
  return h;
}

function _apiPost(url, body) {
  return fetch(url, { method: 'POST', headers: _authHeaders(), body: body ? JSON.stringify(body) : undefined });
}

function _apiDelete(url) {
  return fetch(url, { method: 'DELETE', headers: _authHeaders() });
}

function _toast(msg, color) {
  var t = document.createElement('div');
  t.className = 'j-toast';
  t.style.cssText = 'position:fixed;bottom:20px;right:20px;padding:8px 16px;border-radius:6px;font-size:0.75rem;z-index:9999;color:#fff;background:' + (color || '#0f9') + ';opacity:0;transition:opacity 0.3s;max-width:340px;';
  t.textContent = msg;
  document.body.appendChild(t);
  requestAnimationFrame(function() { t.style.opacity = '1'; });
  setTimeout(function() { t.style.opacity = '0'; setTimeout(function() { t.remove(); }, 300); }, 3000);
}

// ---------------------------------------------------------------------------
// Modal system
// ---------------------------------------------------------------------------

function openModal(titleHtml, bodyHtml, opts) {
  opts = opts || {};
  var bd = document.getElementById('modal-backdrop');
  var mc = document.getElementById('modal-content');
  if (!bd || !mc) return;
  mc.innerHTML =
    '<div class="j-modal-header">' +
      '<h3>' + titleHtml + '</h3>' +
      '<button class="j-modal-close" onclick="closeModal()">&times;</button>' +
    '</div>' +
    '<div class="j-modal-body">' + bodyHtml + '</div>' +
    (opts.footer ? '<div class="j-modal-footer">' + opts.footer + '</div>' : '');
  mc.className = 'j-modal';
  if (opts.xl) mc.classList.add('j-modal-xl');
  else if (opts.wide) mc.classList.add('j-modal-wide');
  bd.style.display = 'flex';
}
window.openModal = openModal;

function closeModal() {
  var bd = document.getElementById('modal-backdrop');
  if (bd) bd.style.display = 'none';
}
window.closeModal = closeModal;

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeModal();
});

var bd = document.getElementById('modal-backdrop');
if (bd) bd.addEventListener('click', function(e) {
  if (e.target === bd) closeModal();
});


// ---------------------------------------------------------------------------
// Chat (Diagnostics tab or global)
// ---------------------------------------------------------------------------

window.openChat = function() {
  // Load prior conversation history from snapshot
  var snap = window._lastSnap || {};
  var convHist = snap.conversation_history || snap.episodes || {};
  var recentContext = convHist.recent_context || convHist.recent_turns || [];

  var historyHtml = '';
  if (recentContext.length) {
    historyHtml = '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:4px;">Recent Conversation</div>';
    recentContext.slice(-8).forEach(function(turn) {
      var role = turn.role || turn.speaker || 'unknown';
      var text = turn.text || turn.content || turn.message || '';
      var color = role === 'user' ? '#0cf' : '#0f9';
      var label = role === 'user' ? 'You' : 'JARVIS';
      historyHtml += '<div style="padding:2px 0;border-bottom:1px solid #0d0d1a;">' +
        '<span style="color:' + color + ';font-weight:600;font-size:0.65rem;">' + label + ':</span> ' +
        '<span style="font-size:0.68rem;">' + esc(text.substring(0, 300)) + '</span></div>';
    });
    historyHtml += '<div style="border-bottom:1px solid #2a2a44;margin:6px 0;"></div>';
  }

  var html =
    '<div id="chat-log" style="max-height:360px;overflow-y:auto;margin-bottom:8px;font-size:0.72rem;">' + historyHtml + '</div>' +
    '<form id="chat-form" onsubmit="return window._sendChat(event)" style="display:flex;gap:6px;">' +
      '<input id="chat-input" type="text" placeholder="Talk to JARVIS..." style="flex:1;padding:6px 10px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;" autocomplete="off">' +
      '<button type="submit" class="j-btn-sm" style="white-space:nowrap;">Send</button>' +
    '</form>' +
    '<div style="margin-top:6px;display:flex;gap:6px;align-items:center;">' +
      '<button class="j-btn-sm j-btn-green" onclick="window._sendFeedback(\'positive\')" title="Good response">&#128077;</button>' +
      '<button class="j-btn-sm j-btn-red" onclick="window._sendFeedback(\'negative\')" title="Bad response">&#128078;</button>' +
      '<span id="chat-meta" style="flex:1;text-align:right;font-size:0.58rem;color:#484860;"></span>' +
    '</div>';
  openModal('Chat with JARVIS', html, { wide: true });
  var inp = document.getElementById('chat-input');
  if (inp) inp.focus();
};

window._sendChat = function(e) {
  e.preventDefault();
  var inp = document.getElementById('chat-input');
  var log = document.getElementById('chat-log');
  var meta = document.getElementById('chat-meta');
  if (!inp || !inp.value.trim()) return false;
  var msg = inp.value.trim();
  inp.value = '';
  log.innerHTML += '<div style="padding:4px 0;border-bottom:1px solid #0d0d1a;">' +
    '<span style="color:#0cf;font-weight:600;">You:</span> ' + esc(msg) + '</div>';
  log.scrollTop = log.scrollHeight;

  _apiPost('/api/chat', { message: msg }).then(function(r) { return r.json(); }).then(function(d) {
    var latency = d.latency_ms || 0;
    var tags = d.memory_tags || [];
    var route = d.route || '';
    var text = d.text || d.error || '--';

    // Response with metadata
    var responseHtml = '<div style="padding:4px 0;border-bottom:1px solid #0d0d1a;">' +
      '<span style="color:#0f9;font-weight:600;">JARVIS:</span> ' + esc(text);
    if (tags.length) {
      responseHtml += '<div style="margin-top:3px;display:flex;flex-wrap:wrap;gap:3px;">';
      tags.forEach(function(tag) {
        responseHtml += '<span style="padding:0 4px;background:#181828;border:1px solid #2a2a44;border-radius:2px;font-size:0.55rem;color:#c0f;">#' + esc(tag) + '</span>';
      });
      responseHtml += '</div>';
    }
    responseHtml += '</div>';
    log.innerHTML += responseHtml;
    log.scrollTop = log.scrollHeight;

    // Update meta
    if (meta) {
      var metaParts = [];
      if (latency) metaParts.push(latency + 'ms');
      if (route) metaParts.push('route: ' + route);
      if (tags.length) metaParts.push(tags.length + ' memory tags');
      meta.textContent = metaParts.join(' \u2022 ');
    }
  }).catch(function(err) {
    log.innerHTML += '<div style="color:#f44;padding:2px 0;">Error: ' + esc(err.message) + '</div>';
  });
  return false;
};

window._sendFeedback = function(signal) {
  _apiPost('/api/feedback', { signal: signal }).then(function() {
    _toast('Feedback sent: ' + signal, signal === 'positive' ? '#0f9' : '#f44');
  }).catch(function() { _toast('Feedback failed', '#f44'); });
};


// ---------------------------------------------------------------------------
// Memory search
// ---------------------------------------------------------------------------

window.openMemorySearch = function() {
  var html =
    '<form id="mem-search-form" onsubmit="return window._searchMemories(event)" style="display:flex;gap:6px;margin-bottom:8px;">' +
      '<input id="mem-q" type="text" placeholder="Search memories..." style="flex:1;padding:6px 10px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;">' +
      '<select id="mem-type" style="padding:4px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.7rem;">' +
        '<option value="">All types</option><option value="observation">observation</option><option value="conversation">conversation</option><option value="core">core</option><option value="insight">insight</option><option value="research">research</option><option value="identity">identity</option>' +
      '</select>' +
      '<button type="submit" class="j-btn-sm">Search</button>' +
    '</form>' +
    '<div id="mem-results" style="max-height:360px;overflow-y:auto;font-size:0.72rem;"></div>';
  openModal('Memory Search', html, { wide: true });
  document.getElementById('mem-q').focus();
};

window._searchMemories = function(e) {
  e.preventDefault();
  var q = document.getElementById('mem-q').value;
  var type = document.getElementById('mem-type').value;
  var out = document.getElementById('mem-results');
  out.innerHTML = '<div style="color:#6a6a80;">Searching...</div>';
  var url = '/api/memories/search?q=' + encodeURIComponent(q) + '&limit=30';
  if (type) url += '&type=' + encodeURIComponent(type);
  fetch(url).then(function(r) { return r.json(); }).then(function(results) {
    if (!results.length) { out.innerHTML = '<div style="color:#484860;">No results</div>'; return; }
    var h = '';
    results.forEach(function(m) {
      var wc = m.weight > 0.5 ? '#0f9' : m.weight > 0.2 ? '#ff0' : '#6a6a80';
      h += '<div class="j-mem-row" onclick="window.openMemoryDetail(\'' + esc(m.id) + '\')" style="cursor:pointer;">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;">' +
          '<span style="font-size:0.6rem;color:#0cf;padding:1px 4px;border:1px solid #0cf33;border-radius:2px;">' + esc(m.type || '--') + '</span>' +
          '<span style="font-size:0.6rem;color:' + wc + ';">w:' + (m.weight || 0).toFixed(2) + '</span>' +
        '</div>' +
        '<div style="margin-top:2px;">' + esc((m.payload || '').substring(0, 200)) + '</div>' +
        '<div style="font-size:0.58rem;color:#484860;margin-top:1px;">' +
          (m.tags && m.tags.length ? m.tags.map(function(t) { return '#' + t; }).join(' ') : '') +
          (m.timestamp ? ' &middot; ' + window.timeAgo(m.timestamp) : '') +
        '</div></div>';
    });
    out.innerHTML = h;
  }).catch(function(err) { out.innerHTML = '<div style="color:#f44;">Error: ' + esc(err.message) + '</div>'; });
  return false;
};

window.openMemoryDetail = function(memId) {
  if (!memId) return;
  var out = '<div style="font-size:0.72rem;color:#6a6a80;">Loading memory ' + esc(memId) + '...</div>';
  openModal('Memory Detail', out);

  fetch('/api/memories/' + encodeURIComponent(memId)).then(function(r) {
    if (r.status === 404) throw new Error('Memory not found');
    return r.json();
  }).then(function(m) {
    var html = '';
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px;">';
    html += '<div style="font-size:0.65rem;"><span style="color:#6a6a80;">Type:</span> <span style="color:#0cf;">' + esc(m.type || '--') + '</span></div>';
    html += '<div style="font-size:0.65rem;"><span style="color:#6a6a80;">Weight:</span> <span style="color:' + (m.weight > 0.5 ? '#0f9' : '#ff0') + ';">' + (m.weight || 0).toFixed(3) + '</span></div>';
    html += '<div style="font-size:0.65rem;"><span style="color:#6a6a80;">Provenance:</span> ' + esc(m.provenance || '--') + '</div>';
    html += '<div style="font-size:0.65rem;"><span style="color:#6a6a80;">Created:</span> ' + (m.timestamp ? window.timeAgo(m.timestamp) : '--') + '</div>';
    if (m.speaker) html += '<div style="font-size:0.65rem;"><span style="color:#6a6a80;">Speaker:</span> ' + esc(m.speaker) + '</div>';
    if (m.access_count != null) html += '<div style="font-size:0.65rem;"><span style="color:#6a6a80;">Accessed:</span> ' + m.access_count + 'x</div>';
    html += '</div>';

    html += '<div style="background:#0d0d1a;padding:8px;border-radius:4px;border:1px solid #1a1a2e;margin-bottom:8px;font-size:0.72rem;white-space:pre-wrap;max-height:200px;overflow-y:auto;">' + esc(m.payload || m.content || '--') + '</div>';

    if (m.tags && m.tags.length) {
      html += '<div style="margin-bottom:6px;display:flex;flex-wrap:wrap;gap:3px;">';
      m.tags.forEach(function(tag) {
        html += '<span style="padding:1px 5px;background:#181828;border:1px solid #2a2a44;border-radius:3px;font-size:0.6rem;color:#c0f;">#' + esc(tag) + '</span>';
      });
      html += '</div>';
    }

    if (m.associations && m.associations.length) {
      html += '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Associations</div>';
      m.associations.slice(0, 10).forEach(function(a) {
        var aId = typeof a === 'object' ? (a.target_id || a.id || '') : String(a);
        html += '<div style="font-size:0.6rem;padding:1px 0;cursor:pointer;color:#0cf;" onclick="window.openMemoryDetail(\'' + esc(aId) + '\')">' + esc(aId.substring(0, 40)) + '</div>';
      });
    }

    html += '<div style="margin-top:8px;font-size:0.55rem;color:#484860;">ID: ' + esc(m.id || memId) + '</div>';

    var body = document.querySelector('.j-modal-body');
    if (body) body.innerHTML = html;
  }).catch(function(err) {
    var body = document.querySelector('.j-modal-body');
    if (body) body.innerHTML = '<div style="color:#f44;">Error: ' + esc(err.message) + '</div>';
  });
};


// ---------------------------------------------------------------------------
// Memory browser (recent memories list)
// ---------------------------------------------------------------------------

window.openRecentMemories = function() {
  var html = '<div id="recent-mem-list" style="max-height:400px;overflow-y:auto;font-size:0.72rem;">Loading...</div>';
  openModal('Recent Memories', html, { wide: true });
  fetch('/api/memories?count=30').then(function(r) { return r.json(); }).then(function(results) {
    var out = document.getElementById('recent-mem-list');
    if (!results.length) { out.innerHTML = '<div style="color:#484860;">No memories yet</div>'; return; }
    var h = '';
    results.forEach(function(m) {
      var wc = m.weight > 0.5 ? '#0f9' : m.weight > 0.2 ? '#ff0' : '#6a6a80';
      h += '<div class="j-mem-row">' +
        '<div style="display:flex;justify-content:space-between;">' +
          '<span style="font-size:0.6rem;color:#0cf;">' + esc(m.type || '--') + '</span>' +
          '<span style="font-size:0.6rem;color:' + wc + ';">w:' + (m.weight || 0).toFixed(2) + '</span>' +
        '</div>' +
        '<div style="margin-top:2px;">' + esc((m.payload || '').substring(0, 200)) + '</div>' +
        '<div style="font-size:0.58rem;color:#484860;margin-top:1px;">' +
          (m.tags && m.tags.length ? m.tags.slice(0, 5).map(function(t) { return '#' + t; }).join(' ') : '') +
          (m.timestamp ? ' &middot; ' + window.timeAgo(m.timestamp) : '') +
        '</div></div>';
    });
    out.innerHTML = h;
  }).catch(function(err) {
    document.getElementById('recent-mem-list').innerHTML = '<div style="color:#f44;">Error: ' + esc(err.message) + '</div>';
  });
};


// ---------------------------------------------------------------------------
// Skill detail
// ---------------------------------------------------------------------------

window.openSkillDetail = function(skillId) {
  openModal('Skill: ' + esc(skillId), '<div id="skill-detail-body">Loading...</div>', { wide: true });
  fetch('/api/skills/' + encodeURIComponent(skillId), { headers: _authHeaders() }).then(function(r) { return r.json(); }).then(function(data) {
    var sk = data.skill || {};
    var job = data.learning_job;
    var h = '';
    h += '<div style="margin-bottom:8px;">' +
      '<span style="font-size:0.75rem;font-weight:600;">' + esc(sk.skill_id || sk.name || skillId) + '</span> ' +
      window._statusBadge(sk.status || '--') +
      '</div>';
    h += '<div class="metric-row"><span class="metric-label">Type</span><span class="metric-value">' + esc(sk.capability_type || '--') + '</span></div>';
    h += '<div class="metric-row"><span class="metric-label">Category</span><span class="metric-value">' + esc(sk.category || '--') + '</span></div>';
    if (sk.summary) h += '<div style="font-size:0.7rem;color:#aaa;margin:6px 0;">' + esc(sk.summary) + '</div>';
    if (sk.evidence_history && sk.evidence_history.length) {
      h += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:8px;">Evidence History (' + sk.evidence_history.length + ')</div>';
      sk.evidence_history.slice(-5).reverse().forEach(function(ev) {
        h += '<div style="font-size:0.6rem;padding:2px 0;border-bottom:1px solid #1a1a2e;">' + esc(JSON.stringify(ev).substring(0, 200)) + '</div>';
      });
    }
    if (job) {
      h += '<div style="font-size:0.68rem;color:#ff0;margin-top:10px;">Learning Job</div>';
      h += '<div class="metric-row"><span class="metric-label">Phase</span><span class="metric-value">' + esc(job.phase || '--') + '</span></div>';
      h += '<div class="metric-row"><span class="metric-label">Status</span><span class="metric-value">' + esc(job.status || '--') + '</span></div>';
      h += '<div class="metric-row"><span class="metric-label">Priority</span><span class="metric-value">' + (job.priority || 0) + '</span></div>';
    }
    h += '<div style="margin-top:10px;display:flex;gap:6px;">' +
      '<button class="j-btn-sm j-btn-red" onclick="window._deleteSkill(\'' + esc(skillId) + '\')">Forget Skill</button>' +
      '</div>';
    document.getElementById('skill-detail-body').innerHTML = h;
  }).catch(function(err) {
    document.getElementById('skill-detail-body').innerHTML = '<div style="color:#f44;">Error: ' + esc(err.message) + '</div>';
  });
};

window._deleteSkill = function(skillId) {
  if (!confirm('Remove skill "' + skillId + '"?')) return;
  _apiDelete('/api/skills/' + encodeURIComponent(skillId)).then(function(r) {
    if (r.ok) { _toast('Skill removed', '#0f9'); closeModal(); }
    else r.json().then(function(d) { _toast(d.detail || 'Failed', '#f44'); });
  });
};


// ---------------------------------------------------------------------------
// Goal actions
// ---------------------------------------------------------------------------

window.goalAction = function(goalId, action) {
  var reasonMap = { complete: 'Manual completion', abandon: 'Manual abandonment', pause: '', resume: '' };
  var body = action === 'resume' ? undefined : { reason: reasonMap[action] || '' };
  _apiPost('/api/goals/' + encodeURIComponent(goalId) + '/' + action, body).then(function(r) { return r.json(); }).then(function(d) {
    _toast('Goal ' + action + ': ' + (d.status || d.error || 'done'), d.status ? '#0f9' : '#f44');
  }).catch(function() { _toast('Failed', '#f44'); });
};

window.openGoalObserve = function() {
  var html =
    '<form id="goal-obs-form" onsubmit="return window._submitGoalObservation(event)">' +
      '<textarea id="goal-obs-content" rows="3" placeholder="Observation content..." style="width:100%;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;resize:vertical;"></textarea>' +
      '<div style="display:flex;gap:6px;margin-top:6px;">' +
        '<select id="goal-obs-scope" style="padding:4px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.7rem;">' +
          '<option value="user">user</option><option value="system">system</option><option value="world">world</option>' +
        '</select>' +
        '<button type="submit" class="j-btn-sm">Submit</button>' +
      '</div>' +
    '</form>';
  openModal('Goal Observation', html);
};

window._submitGoalObservation = function(e) {
  e.preventDefault();
  var content = document.getElementById('goal-obs-content').value.trim();
  if (!content) return false;
  var scope = document.getElementById('goal-obs-scope').value;
  _apiPost('/api/goals/observe', { content: content, source_scope: scope }).then(function(r) { return r.json(); }).then(function(d) {
    _toast('Observation: ' + (d.outcome || 'submitted'), '#0f9');
    closeModal();
  }).catch(function() { _toast('Failed', '#f44'); });
  return false;
};


// ---------------------------------------------------------------------------
// Speaker / Face enrollment
// ---------------------------------------------------------------------------

window.openEnrollment = function() {
  var html =
    '<div style="margin-bottom:12px;">' +
      '<div style="font-size:0.72rem;color:#6a6a80;margin-bottom:4px;">Speaker Enrollment</div>' +
      '<form id="enroll-form" onsubmit="return window._quickEnroll(event)" style="display:flex;gap:6px;">' +
        '<input id="enroll-name" type="text" placeholder="Person name" style="flex:1;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;">' +
        '<button type="submit" class="j-btn-sm">Quick Enroll</button>' +
        '<button type="button" class="j-btn-sm" onclick="window._recordEnroll()">Record</button>' +
      '</form>' +
      '<div id="enroll-status" style="font-size:0.65rem;margin-top:4px;"></div>' +
    '</div>' +
    '<div>' +
      '<div style="font-size:0.72rem;color:#6a6a80;margin-bottom:4px;">Face Enrollment</div>' +
      '<form id="face-enroll-form" onsubmit="return window._faceEnroll(event)" style="display:flex;gap:6px;">' +
        '<input id="face-name" type="text" placeholder="Person name" style="flex:1;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;">' +
        '<button type="submit" class="j-btn-sm">Capture Face</button>' +
      '</form>' +
    '</div>' +
    '<div style="margin-top:12px;">' +
      '<div style="font-size:0.72rem;color:#6a6a80;margin-bottom:4px;">Danger Zone</div>' +
      '<form id="forget-form" onsubmit="return window._forgetAll(event)" style="display:flex;gap:6px;">' +
        '<input id="forget-name" type="text" placeholder="Name to forget" style="flex:1;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;">' +
        '<button type="submit" class="j-btn-sm j-btn-red">Forget All</button>' +
      '</form>' +
    '</div>';
  openModal('Identity Enrollment', html);
};

var _enrollRecorder = null;

window._quickEnroll = function(e) {
  e.preventDefault();
  var name = document.getElementById('enroll-name').value.trim();
  if (!name) return false;
  var status = document.getElementById('enroll-status');
  status.textContent = 'Enrolling from last audio...';
  status.style.color = '#ff0';
  _apiPost('/api/speakers/enroll', { name: name, clips: [] }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.status === 'enrolled') {
      status.textContent = 'Enrolled: ' + d.name + ' (' + d.clips + ' clips)';
      status.style.color = '#0f9';
    } else {
      status.textContent = d.detail || d.error || 'Failed';
      status.style.color = '#f44';
    }
  }).catch(function(err) { status.textContent = err.message; status.style.color = '#f44'; });
  return false;
};

window._recordEnroll = function() {
  var status = document.getElementById('enroll-status');
  if (_enrollRecorder) {
    _enrollRecorder.stop();
    status.textContent = 'Processing...';
    return;
  }
  var name = document.getElementById('enroll-name').value.trim();
  if (!name) { status.textContent = 'Enter a name first'; status.style.color = '#f44'; return; }

  navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
    status.textContent = 'Recording 8 seconds...';
    status.style.color = '#ff0';
    var recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    var chunks = [];
    recorder.ondataavailable = function(e) { chunks.push(e.data); };
    recorder.onstop = function() {
      _enrollRecorder = null;
      stream.getTracks().forEach(function(t) { t.stop(); });
      var blob = new Blob(chunks, { type: 'audio/webm' });
      var reader = new FileReader();
      reader.onload = function() {
        var b64 = reader.result.split(',')[1];
        _apiPost('/api/speakers/enroll', { name: name, clips: [b64] }).then(function(r) { return r.json(); }).then(function(d) {
          if (d.status === 'enrolled') {
            status.textContent = 'Enrolled: ' + d.name;
            status.style.color = '#0f9';
          } else {
            status.textContent = d.detail || 'Failed';
            status.style.color = '#f44';
          }
        }).catch(function(err) { status.textContent = err.message; status.style.color = '#f44'; });
      };
      reader.readAsDataURL(blob);
    };
    recorder.start();
    _enrollRecorder = recorder;
    setTimeout(function() { if (_enrollRecorder) _enrollRecorder.stop(); }, 8000);
  }).catch(function(err) { status.textContent = 'Mic error: ' + err.message; status.style.color = '#f44'; });
};

window._faceEnroll = function(e) {
  e.preventDefault();
  var name = document.getElementById('face-name').value.trim();
  if (!name) return false;
  _apiPost('/api/faces/enroll', { name: name, crops: [] }).then(function(r) { return r.json(); }).then(function(d) {
    _toast(d.status === 'enrolled' ? 'Face enrolled: ' + d.name : (d.detail || 'Failed'), d.status === 'enrolled' ? '#0f9' : '#f44');
  }).catch(function() { _toast('Face enrollment failed', '#f44'); });
  return false;
};

window._forgetAll = function(e) {
  e.preventDefault();
  var name = document.getElementById('forget-name').value.trim();
  if (!name) return false;
  if (!confirm('Forget ALL biometric data for "' + name + '"? This cannot be undone.')) return false;
  _apiPost('/api/identity/forget-all', { name: name }).then(function(r) { return r.json(); }).then(function(d) {
    _toast('Forgot: voice=' + d.voice_removed + ' face=' + d.face_removed, '#0f9');
    closeModal();
  }).catch(function() { _toast('Failed', '#f44'); });
  return false;
};

window.removeSpeaker = function(name) {
  if (!confirm('Remove speaker "' + name + '"?')) return;
  _apiDelete('/api/speakers/' + encodeURIComponent(name)).then(function(r) {
    _toast(r.ok ? 'Removed: ' + name : 'Failed', r.ok ? '#0f9' : '#f44');
  });
};

window.removeFace = function(name) {
  if (!confirm('Remove face "' + name + '"?')) return;
  _apiDelete('/api/faces/' + encodeURIComponent(name)).then(function(r) {
    _toast(r.ok ? 'Removed: ' + name : 'Failed', r.ok ? '#0f9' : '#f44');
  });
};


// ---------------------------------------------------------------------------
// Library ingest
// ---------------------------------------------------------------------------

window.openLibraryIngest = function() {
  var html =
    '<form id="ingest-form" onsubmit="return window._ingestSource(event)">' +
      '<div style="margin-bottom:6px;">' +
        '<label style="font-size:0.7rem;color:#6a6a80;">Mode</label>' +
        '<div style="display:flex;gap:6px;margin-top:2px;">' +
          '<label style="font-size:0.7rem;"><input type="radio" name="ingest-mode" value="url" checked onchange="document.getElementById(\'ingest-url-row\').style.display=\'\';document.getElementById(\'ingest-paste-row\').style.display=\'none\';"> URL</label>' +
          '<label style="font-size:0.7rem;"><input type="radio" name="ingest-mode" value="paste" onchange="document.getElementById(\'ingest-url-row\').style.display=\'none\';document.getElementById(\'ingest-paste-row\').style.display=\'\';"> Paste</label>' +
        '</div>' +
      '</div>' +
      '<div id="ingest-url-row" style="margin-bottom:6px;">' +
        '<input id="ingest-url" type="url" placeholder="https://..." style="width:100%;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;">' +
      '</div>' +
      '<div id="ingest-paste-row" style="display:none;margin-bottom:6px;">' +
        '<textarea id="ingest-content" rows="4" placeholder="Paste content..." style="width:100%;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;resize:vertical;"></textarea>' +
      '</div>' +
      '<input id="ingest-title" type="text" placeholder="Title (optional)" style="width:100%;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;margin-bottom:6px;">' +
      '<input id="ingest-tags" type="text" placeholder="Tags (comma-separated)" style="width:100%;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;margin-bottom:6px;">' +
      '<label style="font-size:0.7rem;color:#6a6a80;"><input type="checkbox" id="ingest-study"> Study immediately</label>' +
      '<div style="margin-top:8px;"><button type="submit" class="j-btn-sm">Ingest</button></div>' +
    '</form>' +
    '<div id="ingest-status" style="font-size:0.65rem;margin-top:6px;"></div>';
  openModal('Library Ingest', html);
};

window._ingestSource = function(e) {
  e.preventDefault();
  var mode = document.querySelector('input[name="ingest-mode"]:checked').value;
  var body = {};
  if (mode === 'url') body.url = document.getElementById('ingest-url').value.trim();
  else body.content = document.getElementById('ingest-content').value.trim();
  body.title = document.getElementById('ingest-title').value.trim();
  body.domain_tags = document.getElementById('ingest-tags').value.trim();
  body.study_now = document.getElementById('ingest-study').checked;
  if (!body.url && !body.content) { _toast('Provide URL or content', '#f44'); return false; }
  var status = document.getElementById('ingest-status');
  status.textContent = 'Ingesting...';
  status.style.color = '#ff0';
  _apiPost('/api/library/ingest', body).then(function(r) { return r.json(); }).then(function(d) {
    status.textContent = d.status === 'error' ? (d.error || 'Failed') : 'Ingested: ' + (d.source_id || d.title || 'done');
    status.style.color = d.status === 'error' ? '#f44' : '#0f9';
  }).catch(function(err) { status.textContent = err.message; status.style.color = '#f44'; });
  return false;
};


// ---------------------------------------------------------------------------
// Library source browser
// ---------------------------------------------------------------------------

window.openLibrarySources = function() {
  var html = '<div id="lib-sources" style="max-height:400px;overflow-y:auto;font-size:0.72rem;">Loading...</div>' +
    '<div style="margin-top:6px;display:flex;gap:6px;">' +
      '<button class="j-btn-sm" onclick="window._loadSources(0)">Refresh</button>' +
      '<button class="j-btn-sm" onclick="window.openLibraryIngest()">Ingest New</button>' +
    '</div>';
  openModal('Library Sources', html, { wide: true });
  window._loadSources(0);
};

window._loadSources = function(offset) {
  var out = document.getElementById('lib-sources');
  if (!out) return;
  fetch('/api/library/sources?limit=20&offset=' + (offset || 0), { headers: _authHeaders() }).then(function(r) { return r.json(); }).then(function(data) {
    var sources = data.sources || [];
    if (!sources.length) { out.innerHTML = '<div style="color:#484860;">No sources</div>'; return; }
    var h = '';
    sources.forEach(function(s) {
      var depth = s.content_depth || 'unknown';
      var depthColors = { full_text: '#0cf', tldr: '#0f9', abstract: '#f90', metadata_only: '#f44', title_only: '#f44' };
      var dc = depthColors[depth] || '#6a6a80';
      h += '<div class="j-mem-row" onclick="window._openSourceDetail(\'' + esc(s.source_id) + '\')" style="cursor:pointer;">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;">' +
          '<span style="font-weight:600;font-size:0.7rem;">' + esc((s.title || 'Untitled').substring(0, 60)) + '</span>' +
          '<span style="font-size:0.58rem;color:' + dc + ';padding:1px 4px;border:1px solid ' + dc + '33;border-radius:2px;">' + esc(depth) + '</span>' +
        '</div>' +
        '<div style="font-size:0.58rem;color:#484860;margin-top:1px;">' +
          esc(s.source_type || '') + (s.quality_score ? ' &middot; q:' + s.quality_score.toFixed(2) : '') +
          (s.studied ? ' &middot; <span style="color:#0f9;">studied</span>' : '') +
        '</div></div>';
    });
    if (data.count > offset + 20) {
      h += '<div style="text-align:center;margin-top:6px;"><button class="j-btn-sm" onclick="window._loadSources(' + (offset + 20) + ')">Load More</button></div>';
    }
    out.innerHTML = h;
  }).catch(function(err) { out.innerHTML = '<div style="color:#f44;">Error: ' + esc(err.message) + '</div>'; });
};

window._openSourceDetail = function(sourceId) {
  fetch('/api/library/sources/' + encodeURIComponent(sourceId), { headers: _authHeaders() }).then(function(r) { return r.json(); }).then(function(s) {
    var h = '';
    h += '<div class="metric-row"><span class="metric-label">Type</span><span class="metric-value">' + esc(s.source_type || '--') + '</span></div>';
    h += '<div class="metric-row"><span class="metric-label">Depth</span><span class="metric-value">' + esc(s.content_depth || '--') + '</span></div>';
    h += '<div class="metric-row"><span class="metric-label">Quality</span><span class="metric-value">' + (s.quality_score != null ? s.quality_score.toFixed(2) : '--') + '</span></div>';
    h += '<div class="metric-row"><span class="metric-label">Studied</span><span class="metric-value">' + (s.studied ? 'Yes' : 'No') + '</span></div>';
    if (s.url) h += '<div class="metric-row"><span class="metric-label">URL</span><span class="metric-value"><a href="' + esc(s.url) + '" target="_blank" style="color:#0cf;">' + esc((s.url || '').substring(0, 60)) + '</a></span></div>';
    if (s.authors) h += '<div class="metric-row"><span class="metric-label">Authors</span><span class="metric-value">' + esc(String(s.authors).substring(0, 100)) + '</span></div>';
    if (s.content_text) {
      h += '<div style="margin-top:8px;padding:8px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;max-height:250px;overflow-y:auto;font-size:0.65rem;white-space:pre-wrap;color:#aaa;">' + esc(s.content_text.substring(0, 3000)) + '</div>';
    }
    openModal('Source: ' + esc((s.title || 'Untitled').substring(0, 50)), h, { wide: true });
  }).catch(function(err) { _toast('Error loading source', '#f44'); });
};


// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

window.openSettings = function() {
  var html = '<div id="settings-body">Loading...</div>';
  openModal('Settings', html);
  fetch('/api/settings', { headers: _authHeaders() }).then(function(r) { return r.json(); }).then(function(data) {
    var h = '<form id="settings-form" onsubmit="return window._saveSettings(event)">';
    h += '<div style="margin-bottom:8px;">' +
      '<label style="font-size:0.7rem;color:#6a6a80;">Tone</label>' +
      '<select id="set-tone" style="display:block;width:100%;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;margin-top:2px;">';
    ['professional', 'casual', 'urgent', 'empathetic', 'playful'].forEach(function(t) {
      h += '<option value="' + t + '"' + (data.tone === t ? ' selected' : '') + '>' + t + '</option>';
    });
    h += '</select></div>';
    var budget = (data.kernel || {}).budget_ms || 16;
    h += '<div style="margin-bottom:8px;">' +
      '<label style="font-size:0.7rem;color:#6a6a80;">Kernel Budget (ms)</label>' +
      '<input id="set-budget" type="number" min="5" max="100" value="' + budget + '" style="display:block;width:100%;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;margin-top:2px;">' +
      '</div>';
    h += '<button type="submit" class="j-btn-sm">Save Settings</button>';
    h += '</form>';
    document.getElementById('settings-body').innerHTML = h;
  }).catch(function(err) {
    document.getElementById('settings-body').innerHTML = '<div style="color:#f44;">Error loading settings</div>';
  });
};

window._saveSettings = function(e) {
  e.preventDefault();
  var tone = document.getElementById('set-tone').value;
  var budget = parseFloat(document.getElementById('set-budget').value);
  _apiPost('/api/settings', { tone: tone, budget_ms: budget }).then(function(r) { return r.json(); }).then(function(d) {
    _toast('Settings saved: ' + (d.updated || []).join(', '), '#0f9');
    closeModal();
  }).catch(function() { _toast('Failed', '#f44'); });
  return false;
};


// ---------------------------------------------------------------------------
// Voice test
// ---------------------------------------------------------------------------

window.openVoiceTest = function() {
  var html =
    '<form onsubmit="return window._voiceTest(event)" style="display:flex;gap:6px;">' +
      '<input id="voice-text" type="text" placeholder="Text to speak..." style="flex:1;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;">' +
      '<button type="submit" class="j-btn-sm">Speak</button>' +
    '</form>';
  openModal('Voice Test', html);
  document.getElementById('voice-text').focus();
};

window._voiceTest = function(e) {
  e.preventDefault();
  var text = document.getElementById('voice-text').value.trim();
  if (!text) return false;
  _apiPost('/api/voice-test', { text: text }).then(function(r) { return r.json(); }).then(function(d) {
    _toast(d.sent ? 'Sent: ' + d.sent.substring(0, 40) : (d.detail || 'Failed'), d.sent ? '#0f9' : '#f44');
  }).catch(function() { _toast('Failed', '#f44'); });
  return false;
};


// ---------------------------------------------------------------------------
// System controls
// ---------------------------------------------------------------------------

window.systemSave = function() {
  _apiPost('/api/system/save').then(function(r) { return r.json(); }).then(function(d) {
    _toast('Saved: ' + (d.saved || []).join(', '), '#0f9');
  }).catch(function() { _toast('Save failed', '#f44'); });
};

window.systemRestart = function() {
  if (!confirm('Restart JARVIS brain? Active conversations will be interrupted.')) return;
  _apiPost('/api/system/restart').then(function(r) { return r.json(); }).then(function(d) {
    _toast('Restarting...', '#ff0');
  }).catch(function() { _toast('Restart failed', '#f44'); });
};

window.systemShutdown = function() {
  if (!confirm('Shutdown JARVIS brain? This will stop all processes.')) return;
  if (!confirm('Are you sure? JARVIS will go offline.')) return;
  _apiPost('/api/system/shutdown').then(function(r) { return r.json(); }).then(function(d) {
    _toast('Shutting down...', '#f44');
  }).catch(function() { _toast('Shutdown failed', '#f44'); });
};


// ---------------------------------------------------------------------------
// Debug snapshot
// ---------------------------------------------------------------------------

window.openDebugSnapshot = function() {
  var html =
    '<div style="margin-bottom:6px;display:flex;gap:6px;">' +
      '<select id="snap-key" style="flex:1;padding:6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.75rem;">' +
        '<option value="">Full snapshot</option>' +
      '</select>' +
      '<button class="j-btn-sm" onclick="window._loadSnapshotKeys()">Load Keys</button>' +
      '<button class="j-btn-sm" onclick="window._fetchSnapshot()">Fetch</button>' +
      '<button class="j-btn-sm" onclick="window._copySnapshot()">Copy</button>' +
    '</div>' +
    '<pre id="snap-output" style="max-height:400px;overflow:auto;padding:8px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;font-size:0.6rem;color:#aaa;white-space:pre-wrap;word-break:break-all;">Click Fetch to load</pre>';
  openModal('Debug Snapshot', html, { wide: true });
  window._loadSnapshotKeys();
};

window._loadSnapshotKeys = function() {
  var sel = document.getElementById('snap-key');
  if (!sel) return;
  fetch('/api/full-snapshot').then(function(r) { return r.json(); }).then(function(snap) {
    window._debugSnap = snap;
    sel.innerHTML = '<option value="">Full snapshot (' + JSON.stringify(snap).length + ' chars)</option>';
    Object.keys(snap).sort().forEach(function(k) {
      sel.innerHTML += '<option value="' + k + '">' + k + '</option>';
    });
  });
};

window._fetchSnapshot = function() {
  var key = document.getElementById('snap-key').value;
  var out = document.getElementById('snap-output');
  var snap = window._debugSnap;
  if (!snap) { out.textContent = 'No snapshot loaded'; return; }
  var data = key ? snap[key] : snap;
  out.textContent = JSON.stringify(data, null, 2);
};

window._copySnapshot = function() {
  var out = document.getElementById('snap-output');
  if (out) {
    navigator.clipboard.writeText(out.textContent).then(function() { _toast('Copied to clipboard', '#0f9'); });
  }
};


// ---------------------------------------------------------------------------
// Trace explorer
// ---------------------------------------------------------------------------

window.openTraceExplorer = function() {
  var html = '<div id="trace-explorer-body" style="max-height:420px;overflow-y:auto;font-size:0.7rem;color:#8a8aa0;">Loading trace explorer...</div>';
  openModal('Trace Explorer', html, { wide: true });

  fetch('/api/trace/explorer?root_limit=25&run_limit=25&tool_limit=60', { headers: _authHeaders() })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var body = document.getElementById('trace-explorer-body');
      if (!body) return;

      var roots = data.root_chains || [];
      var runs = data.agent_runs || [];
      var tools = data.tool_lineage || [];
      var rec = data.reconstructability || {};
      var recStatus = rec.reconstructability || 'unknown';
      var recColor = recStatus === 'reconstructable' ? '#0f9' : recStatus === 'partial' ? '#ff0' : recStatus === 'non_reconstructable' ? '#f44' : '#6a6a80';

      var out = '';
      out += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px;">' +
        '<div style="padding:6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;text-align:center;"><div style="font-size:1rem;color:#0cf;font-weight:700;">' + (data.entry_count || 0) + '</div><div style="font-size:0.56rem;color:#6a6a80;">Ledger Entries</div></div>' +
        '<div style="padding:6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;text-align:center;"><div style="font-size:1rem;color:#0f9;font-weight:700;">' + roots.length + '</div><div style="font-size:0.56rem;color:#6a6a80;">Root Chains</div></div>' +
        '<div style="padding:6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;text-align:center;"><div style="font-size:1rem;color:#c0f;font-weight:700;">' + runs.length + '</div><div style="font-size:0.56rem;color:#6a6a80;">Agent Runs</div></div>' +
        '<div style="padding:6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;text-align:center;"><div style="font-size:1rem;color:' + recColor + ';font-weight:700;">' + esc(recStatus.replace(/_/g, ' ')) + '</div><div style="font-size:0.56rem;color:#6a6a80;">Reconstructability</div></div>' +
        '</div>';

      out += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:4px;">Root Chains</div>';
      if (!roots.length) {
        out += '<div style="font-size:0.62rem;color:#484860;margin-bottom:8px;">No root chains in cache window.</div>';
      } else {
        roots.slice(0, 12).forEach(function(r) {
          var rid = r.root_entry_id || '--';
          var subs = (r.subsystems || []).slice(0, 4).join(', ');
          out += '<div style="display:flex;align-items:center;gap:6px;padding:3px 4px;border-bottom:1px solid #1a1a2e;font-size:0.58rem;">' +
            '<span style="color:#0cf;min-width:84px;">' + esc(rid.substring(0, 20)) + '</span>' +
            '<span style="min-width:46px;color:#e0e0e8;">' + (r.entry_count || 0) + ' ev</span>' +
            '<span style="min-width:52px;color:#6a6a80;">' + (r.duration_s != null ? Number(r.duration_s).toFixed(2) + 's' : '--') + '</span>' +
            '<span style="flex:1;color:#8a8aa0;">' + esc(subs || '--') + '</span>' +
            '<button class="j-btn-xs" onclick="window.openTraceChain(\'' + esc(rid) + '\')">chain</button>' +
            '</div>';
        });
      }

      out += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:8px;margin-bottom:4px;">Per-Agent Runs</div>';
      if (!runs.length) {
        out += '<div style="font-size:0.62rem;color:#484860;margin-bottom:8px;">No autonomy runs in cache window.</div>';
      } else {
        runs.slice(0, 10).forEach(function(run) {
          out += '<div style="padding:3px 4px;border-bottom:1px solid #1a1a2e;font-size:0.58rem;">' +
            '<span style="color:#c0f;">' + esc((run.intent_id || '--').substring(0, 24)) + '</span>' +
            ' <span style="color:#6a6a80;">events:</span> ' + (run.event_count || 0) +
            (run.tools && run.tools.length ? ' <span style="color:#6a6a80;">tools:</span> ' + esc(run.tools.slice(0, 3).join(', ')) : '') +
            (run.goal_id ? ' <span style="color:#484860;">goal:' + esc(run.goal_id.substring(0, 12)) + '</span>' : '') +
            '</div>';
        });
      }

      out += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:8px;margin-bottom:4px;">Tool Lineage</div>';
      if (!tools.length) {
        out += '<div style="font-size:0.62rem;color:#484860;">No tool lineage in cache window.</div>';
      } else {
        tools.slice(0, 20).forEach(function(t) {
          out += '<div style="padding:2px 0;border-bottom:1px solid #1a1a2e;font-size:0.55rem;">' +
            '<span style="color:#0f9;">' + esc(t.tool || '--') + '</span> ' +
            '<span style="color:#6a6a80;">(' + esc(t.source || '--') + ')</span>' +
            (t.trace_id ? ' <span style="color:#484860;">' + esc(t.trace_id.substring(0, 16)) + '</span>' : '') +
            '</div>';
        });
      }

      body.innerHTML = out;
    })
    .catch(function(err) {
      var body = document.getElementById('trace-explorer-body');
      if (body) body.innerHTML = '<div style="color:#f44;">Error loading trace explorer: ' + esc(err.message) + '</div>';
    });
};

window.openTraceChain = function(rootId) {
  if (!rootId) return;
  var html = '<div id="trace-chain-body" style="max-height:440px;overflow-y:auto;font-size:0.68rem;color:#8a8aa0;">Loading chain ' + esc(rootId) + '...</div>';
  openModal('Trace Chain ' + esc(rootId.substring(0, 20)), html, { wide: true });

  fetch('/api/trace/explorer/chain/' + encodeURIComponent(rootId), { headers: _authHeaders() })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var body = document.getElementById('trace-chain-body');
      if (!body) return;
      var nodes = data.nodes || [];
      if (!nodes.length) {
        body.innerHTML = '<div style="color:#484860;">No chain nodes found for this root.</div>';
        return;
      }

      var byId = {};
      nodes.forEach(function(n) { byId[n.entry_id] = n; });
      function _depth(node) {
        var d = 0;
        var p = node.parent_entry_id;
        var guard = 0;
        while (p && byId[p] && guard < 32) {
          d += 1;
          p = byId[p].parent_entry_id;
          guard += 1;
        }
        return d;
      }

      var out = '';
      out += '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px;font-size:0.56rem;color:#6a6a80;">' +
        '<span>nodes: <span style="color:#0cf;">' + nodes.length + '</span></span>' +
        '<span>roots: <span style="color:#0f9;">' + ((data.root_nodes || []).length || 0) + '</span></span>' +
        '<span>tool path: <span style="color:#c0f;">' + esc((data.tool_path || []).join(' → ') || '--') + '</span></span>' +
        '</div>';

      var sorted = nodes.slice().sort(function(a, b) {
        return (a.ts || 0) - (b.ts || 0);
      });
      sorted.forEach(function(n) {
        var depth = _depth(n);
        var indent = depth * 12;
        var outcomeColor = n.outcome === 'success' ? '#0f9' : n.outcome === 'failure' ? '#f44' : '#6a6a80';
        out += '<div style="padding:2px 0 2px ' + indent + 'px;border-bottom:1px solid #1a1a2e;font-size:0.56rem;">' +
          '<span style="color:#484860;">' + (window.timeAgo ? window.timeAgo(n.ts) : '--') + '</span> ' +
          '<span style="color:#0cf;">' + esc((n.subsystem || '--') + ':' + (n.event_type || '--')) + '</span> ' +
          '<span style="color:' + outcomeColor + ';">' + esc(n.outcome || '--') + '</span>' +
          (n.tool ? ' <span style="color:#0f9;">tool=' + esc(n.tool) + '</span>' : '') +
          (n.trace_id ? ' <span style="color:#484860;">' + esc(n.trace_id.substring(0, 14)) + '</span>' : '') +
          '</div>';
      });
      body.innerHTML = out;
    })
    .catch(function(err) {
      var body = document.getElementById('trace-chain-body');
      if (body) body.innerHTML = '<div style="color:#f44;">Error loading chain: ' + esc(err.message) + '</div>';
    });
};


// ---------------------------------------------------------------------------
// Log viewer
// ---------------------------------------------------------------------------

var _logInterval = null;

window.openLogViewer = function() {
  var html =
    '<div style="margin-bottom:6px;display:flex;gap:6px;align-items:center;">' +
      '<select id="log-level" style="padding:4px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.7rem;">' +
        '<option value="">All</option><option value="ERROR">ERROR</option><option value="WARNING">WARNING</option><option value="INFO">INFO</option>' +
      '</select>' +
      '<input id="log-filter" type="text" placeholder="Filter..." style="flex:1;padding:4px 6px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;color:#e0e0e8;font-size:0.7rem;">' +
      '<label style="font-size:0.65rem;color:#6a6a80;"><input type="checkbox" id="log-auto" checked> Auto-refresh</label>' +
    '</div>' +
    '<pre id="log-output" style="max-height:400px;overflow:auto;padding:8px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;font-size:0.58rem;color:#aaa;white-space:pre-wrap;">Loading...</pre>';
  openModal('Log Viewer', html, { wide: true });
  window._refreshLogs();
  _logInterval = setInterval(function() {
    var auto = document.getElementById('log-auto');
    if (auto && auto.checked) window._refreshLogs();
  }, 3000);
};

window._refreshLogs = function() {
  var out = document.getElementById('log-output');
  if (!out) { if (_logInterval) clearInterval(_logInterval); return; }
  fetch('/api/logs?lines=80').then(function(r) { return r.json(); }).then(function(d) {
    var lines = d.lines || [];
    var level = (document.getElementById('log-level') || {}).value || '';
    var filter = (document.getElementById('log-filter') || {}).value || '';
    if (level) lines = lines.filter(function(l) { return l.indexOf(level) !== -1; });
    if (filter) lines = lines.filter(function(l) { return l.toLowerCase().indexOf(filter.toLowerCase()) !== -1; });
    out.textContent = lines.join('\n') || 'No logs';
    out.scrollTop = out.scrollHeight;
  }).catch(function() {});
};

var _origCloseModal = closeModal;
window.closeModal = function() {
  if (_logInterval) { clearInterval(_logInterval); _logInterval = null; }
  _origCloseModal();
};


// ---------------------------------------------------------------------------
// Self-improvement actions
// ---------------------------------------------------------------------------

window.approvePatch = function(patchId, approved) {
  _apiPost('/api/self-improve/approve', { patch_id: patchId, approved: approved }).then(function(r) { return r.json(); }).then(function(d) {
    _toast('Patch ' + (approved ? 'approved' : 'rejected'), approved ? '#0f9' : '#f44');
  }).catch(function() { _toast('Failed', '#f44'); });
};

window.triggerDryRun = function() {
  _apiPost('/api/self-improve/dry-run', {}).then(function(r) { return r.json(); }).then(function(d) {
    var h = '<div class="metric-row"><span class="metric-label">Status</span><span class="metric-value">' + esc(d.status || '--') + '</span></div>';
    if (d.iterations) h += '<div class="metric-row"><span class="metric-label">Iterations</span><span class="metric-value">' + d.iterations + '</span></div>';
    if (d.description) h += '<div style="font-size:0.7rem;color:#aaa;margin:6px 0;">' + esc(d.description) + '</div>';
    if (d.diffs && d.diffs.length) {
      h += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;">Diffs</div>';
      d.diffs.forEach(function(diff) {
        h += '<div style="font-size:0.6rem;color:#0cf;margin-top:4px;">' + esc(diff.path) + '</div>';
        h += '<pre style="font-size:0.55rem;color:#aaa;background:#0d0d1a;padding:4px;border-radius:3px;max-height:200px;overflow:auto;white-space:pre-wrap;">' + esc(diff.diff) + '</pre>';
      });
    }
    openModal('Dry Run Result', h, { wide: true });
  }).catch(function(err) { _toast('Dry run failed: ' + err.message, '#f44'); });
};


// ---------------------------------------------------------------------------
// Learning job actions
// ---------------------------------------------------------------------------

window.deleteLearningJob = function(jobId, skillId, isDefault) {
  if (isDefault) {
    if (!confirm(
      '⚠️ WARNING: "' + skillId + '" is a DEFAULT SYSTEM SKILL.\n\n' +
      'Deleting this job may affect core system capabilities like speaker recognition, ' +
      'emotion detection, or other foundational subsystems.\n\n' +
      'Are you sure you want to proceed?'
    )) return;
    if (!confirm(
      '⚠️ FINAL CONFIRMATION\n\n' +
      'You are about to delete a job for the default skill "' + skillId + '".\n' +
      'Job ID: ' + jobId + '\n\n' +
      'This action cannot be undone.\n\n' +
      'Type OK to confirm.'
    )) return;
  } else {
    var msg = 'Delete learning job "' + jobId + '"';
    if (skillId) msg += ' (skill: ' + skillId + ')';
    msg += '?\n\nThe skill record will be repointed to any surviving completed job, or cleared.';
    if (!confirm(msg)) return;
  }
  _apiDelete('/api/learning-jobs/' + encodeURIComponent(jobId)).then(function(r) {
    if (r.ok) {
      r.json().then(function(d) {
        _toast('Deleted job ' + jobId + (d.skill_removed ? ' (skill also removed)' : ''), '#0f9');
      });
    } else {
      r.json().then(function(d) {
        _toast('Delete failed: ' + (d.error || 'unknown'), '#f44');
      }).catch(function() { _toast('Delete failed', '#f44'); });
    }
  }).catch(function(err) { _toast('Delete failed: ' + err.message, '#f44'); });
};

window.cleanupLearningJobs = function() {
  if (!confirm('Clean up stale learning jobs?')) return;
  _apiPost('/api/learning-jobs/cleanup').then(function(r) { return r.json(); }).then(function(d) {
    _toast('Cleaned ' + (d.deleted_count || 0) + ' jobs', '#0f9');
  }).catch(function() { _toast('Failed', '#f44'); });
};


// ---------------------------------------------------------------------------
// Camera feed
// ---------------------------------------------------------------------------

window.openCameraFeed = function() {
  var html = '<div id="cam-container" style="text-align:center;">Loading camera...</div>';
  openModal('Camera Feed', html, { wide: true });
  fetch('/api/config').then(function(r) { return r.json(); }).then(function(cfg) {
    var container = document.getElementById('cam-container');
    if (!container) return;
    var url = cfg.pi_video_url;
    if (url) {
      container.innerHTML =
        '<img id="cam-feed" src="' + esc(url) + '" style="max-width:100%;border-radius:4px;border:1px solid #2a2a44;" onerror="this.style.display=\'none\';document.getElementById(\'cam-error\').style.display=\'\';">' +
        '<div id="cam-error" style="display:none;color:#f44;font-size:0.72rem;padding:20px;">Camera feed unavailable — check Pi connection</div>' +
        '<div style="margin-top:8px;font-size:0.6rem;color:#484860;">' + esc(url) + '</div>';
    } else {
      container.innerHTML = '<div style="color:#6a6a80;font-size:0.72rem;padding:20px;">Pi host not configured. Set PI_HOST env var.</div>';
    }
  }).catch(function() {
    var container = document.getElementById('cam-container');
    if (container) container.innerHTML = '<div style="color:#f44;">Failed to load config</div>';
  });
};


// ---------------------------------------------------------------------------
// Soul export
// ---------------------------------------------------------------------------

window.exportSoul = function() {
  _apiPost('/api/soul/export').then(function(r) { return r.json(); }).then(function(d) {
    var blob = new Blob([JSON.stringify(d, null, 2)], { type: 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'jarvis_soul_export.json';
    a.click();
    _toast('Soul exported', '#0f9');
  }).catch(function() { _toast('Export failed', '#f44'); });
};


// ---------------------------------------------------------------------------
// Onboarding
// ---------------------------------------------------------------------------

window.startOnboarding = function() {
  if (!confirm('Start companion training onboarding?')) return;
  _apiPost('/api/onboarding/start').then(function(r) { return r.json(); }).then(function(d) {
    _toast('Onboarding started: day ' + (d.day || 0), '#0f9');
  }).catch(function() { _toast('Failed', '#f44'); });
};


// ---------------------------------------------------------------------------
// Research Episode Drill-down
// ---------------------------------------------------------------------------

window.openResearchDetail = function(idx) {
  var snap = window._lastSnap || {};
  var auto = snap.autonomy || {};
  var completed = auto.completed || [];
  var learnings = auto.recent_learnings || [];
  var item = completed[idx];
  if (!item) { _toast('Research episode not found', '#f44'); return; }

  var html = '';

  // Question headline
  html += '<div style="margin-bottom:8px;">' +
    '<div style="font-size:0.55rem;color:#6a6a80;">Question</div>' +
    '<div style="font-size:0.78rem;color:#e0e0e8;margin-bottom:4px;">' + esc(item.question || '--') + '</div>' +
    '</div>';

  // Status + priority
  var sc = item.status === 'completed' ? '#0f9' : item.status === 'failed' ? '#f44' : '#ff0';
  html += '<div style="display:flex;gap:8px;margin-bottom:6px;">' +
    '<span style="padding:1px 6px;background:' + sc + '22;border:1px solid ' + sc + '44;color:' + sc + ';border-radius:3px;font-size:0.58rem;">' + esc(item.status || '--') + '</span>' +
    '<span style="font-size:0.55rem;color:#6a6a80;">priority: ' + fmtNum(item.priority || 0, 3) + '</span>' +
    '<span style="font-size:0.55rem;color:#6a6a80;">scope: ' + esc(item.scope || '--') + '</span>' +
    '</div>';

  // Reason / scoring breakdown
  if (item.reason) {
    html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;margin-bottom:8px;">' +
      '<div style="font-size:0.55rem;color:#6a6a80;">Scoring Reason</div>' +
      '<div style="font-size:0.58rem;color:#8a8aa0;white-space:pre-wrap;">' + esc(item.reason) + '</div></div>';
  }

  // Source event + hint
  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:6px;">';
  if (item.source_event) html += '<div style="font-size:0.55rem;"><span style="color:#6a6a80;">Source event:</span> <span style="color:#c0f;">' + esc(item.source_event) + '</span></div>';
  if (item.source_hint) html += '<div style="font-size:0.55rem;"><span style="color:#6a6a80;">Tool hint:</span> <span style="color:#0cf;">' + esc(item.source_hint) + '</span></div>';
  html += '</div>';

  // Timestamps
  if (item.created_at || item.started_at || item.completed_at) {
    html += '<div style="display:flex;gap:8px;font-size:0.5rem;color:#484860;margin-bottom:6px;">';
    if (item.created_at) html += '<span>created: ' + window.timeAgo(item.created_at) + '</span>';
    if (item.started_at) html += '<span>started: ' + window.timeAgo(item.started_at) + '</span>';
    if (item.completed_at) html += '<span>completed: ' + window.timeAgo(item.completed_at) + '</span>';
    if (item.started_at && item.completed_at) html += '<span>duration: ' + fmtNum(item.completed_at - item.started_at, 1) + 's</span>';
    html += '</div>';
  }

  // Find matching learning
  var learning = null;
  for (var i = 0; i < learnings.length; i++) {
    if (learnings[i].intent_id === item.id) { learning = learnings[i]; break; }
  }
  if (learning) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Learning Outcome</div>';
    html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;margin-bottom:8px;">';
    html += '<div style="font-size:0.55rem;"><span style="color:#6a6a80;">Tool:</span> <span style="color:#0cf;">' + esc(learning.tool || '--') + '</span>' +
      ' <span style="color:#6a6a80;">Findings:</span> <span style="color:#0f9;">' + (learning.findings || 0) + '</span>' +
      ' <span style="color:#6a6a80;">Memories:</span> <span style="color:#c0f;">' + (learning.memories_created || 0) + '</span></div>';
    if (learning.summary) {
      html += '<div style="font-size:0.58rem;color:#8a8aa0;margin-top:4px;max-height:150px;overflow-y:auto;white-space:pre-wrap;">' + esc(learning.summary) + '</div>';
    }
    html += '</div>';
  }

  // Find matching delta
  var deltas = auto.recent_deltas || [];
  var matchedDelta = null;
  for (var j = 0; j < deltas.length; j++) {
    if (deltas[j].intent_id === item.id) { matchedDelta = deltas[j]; break; }
  }
  if (matchedDelta) {
    var net = matchedDelta.net_improvement;
    var nc = net > 0.01 ? '#0f9' : net < -0.01 ? '#f44' : '#6a6a80';
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Impact Delta</div>' +
      '<div style="font-size:0.55rem;margin-bottom:4px;">Net improvement: <span style="color:' + nc + ';font-weight:600;">' + (net > 0 ? '+' : '') + fmtNum(net, 4) + '</span></div>';
  }

  openModal('Research Episode', html, { wide: true });
};


// ---------------------------------------------------------------------------
// Goal Detail Drill-down
// ---------------------------------------------------------------------------

window.openGoalDetail = function(goalId) {
  var snap = window._lastSnap || {};
  var goals = snap.goals || {};
  var active = goals.active_goals || [];
  var goal = null;
  for (var i = 0; i < active.length; i++) {
    if ((active[i].goal_id || active[i].id) === goalId) { goal = active[i]; break; }
  }
  if (!goal) { _toast('Goal not found in snapshot', '#f44'); return; }

  var html = '';
  html += '<div style="margin-bottom:8px;">';
  html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">';
  html += window._statusBadge ? window._statusBadge(goal.status || 'active') : '<span>' + esc(goal.status) + '</span>';
  html += '<span style="font-size:0.85rem;font-weight:600;color:#e0e0e8;">' + esc(goal.title || goal.description || goalId) + '</span>';
  html += '</div>';

  if (goal.progress != null) {
    var pctVal = (goal.progress * 100).toFixed(0);
    html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">' +
      '<div style="flex:1;height:8px;background:#1a1a2e;border-radius:4px;overflow:hidden;">' +
      '<div style="height:100%;width:' + pctVal + '%;background:#0cf;border-radius:4px;"></div></div>' +
      '<span style="font-size:0.7rem;color:#0cf;">' + pctVal + '%</span></div>';
  }
  html += '</div>';

  var fields = [
    ['Created', goal.created_at ? window.timeAgo(goal.created_at) : null],
    ['Priority', goal.priority],
    ['Source', goal.source],
    ['Type', goal.goal_type || goal.type],
    ['Score', goal.score != null ? goal.score.toFixed(2) : null],
    ['Criteria', goal.criteria ? JSON.stringify(goal.criteria).substring(0, 200) : null],
    ['Linked intent', goal.dispatched_intent_id]
  ];
  fields.forEach(function(f) {
    if (f[1] != null) {
      html += '<div style="display:flex;justify-content:space-between;padding:2px 0;font-size:0.65rem;">' +
        '<span style="color:#6a6a80;">' + esc(f[0]) + '</span>' +
        '<span>' + esc(String(f[1])) + '</span></div>';
    }
  });

  var effects = goal.effects || goal.goal_effects || [];
  if (effects.length) {
    html += '<div style="font-size:0.65rem;color:#6a6a80;margin-top:8px;margin-bottom:3px;">Recent Effects</div>';
    effects.slice(-5).forEach(function(e) {
      var text = typeof e === 'object' ? (e.effect || e.text || JSON.stringify(e)) : String(e);
      html += '<div style="font-size:0.6rem;padding:1px 0;color:#aaa;">' + esc(text.substring(0, 120)) + '</div>';
    });
  }

  var evidence = goal.evidence || [];
  if (evidence.length) {
    html += '<div style="font-size:0.65rem;color:#6a6a80;margin-top:8px;margin-bottom:3px;">Evidence</div>';
    evidence.slice(-5).forEach(function(ev) {
      var text = typeof ev === 'object' ? (ev.signal || ev.text || JSON.stringify(ev)) : String(ev);
      html += '<div style="font-size:0.6rem;padding:1px 0;color:#aaa;">' + esc(text.substring(0, 120)) + '</div>';
    });
  }

  if (goalId && window.goalAction) {
    html += '<div style="margin-top:10px;display:flex;gap:6px;">';
    if (goal.status === 'active' || goal.status === 'promoted') {
      html += '<button class="j-btn-sm j-btn-green" onclick="window.goalAction(\'' + esc(goalId) + '\',\'complete\');window.closeModal();">Complete</button>' +
        '<button class="j-btn-sm" onclick="window.goalAction(\'' + esc(goalId) + '\',\'pause\');window.closeModal();">Pause</button>' +
        '<button class="j-btn-sm j-btn-red" onclick="window.goalAction(\'' + esc(goalId) + '\',\'abandon\');window.closeModal();">Abandon</button>';
    } else if (goal.status === 'paused') {
      html += '<button class="j-btn-sm j-btn-green" onclick="window.goalAction(\'' + esc(goalId) + '\',\'resume\');window.closeModal();">Resume</button>';
    }
    html += '</div>';
  }

  openModal('Goal: ' + esc((goal.title || '').substring(0, 40)), html, { wide: true });
};


// ---------------------------------------------------------------------------
// Thought Detail Drill-down
// ---------------------------------------------------------------------------

window.openThoughtDetail = function(idx) {
  var snap = window._lastSnap || {};
  var th = snap.thoughts || {};
  var recent = th.recent || [];
  var t = recent[idx];
  if (!t) { _toast('Thought not found', '#f44'); return; }

  var html = '';

  // Type badge + depth
  var depthColors = { surface: '#6a6a80', moderate: '#0cf', deep: '#c0f', profound: '#f0f' };
  var dc = depthColors[t.depth] || '#6a6a80';
  html += '<div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">' +
    '<span style="padding:2px 8px;background:#0cf22;border:1px solid #0cf44;color:#0cf;border-radius:4px;font-size:0.68rem;">' + esc(t.type || '--') + '</span>' +
    '<span style="padding:2px 8px;background:' + dc + '22;border:1px solid ' + dc + '44;color:' + dc + ';border-radius:4px;font-size:0.68rem;">' + esc(t.depth || '--') + '</span>' +
    '</div>';

  // Full text
  html += '<div style="padding:10px 12px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:6px;margin-bottom:10px;">' +
    '<div style="font-size:0.78rem;color:#e0e0e8;line-height:1.5;">' + esc(t.text || '--') + '</div></div>';

  // ID + timestamp
  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px;">';
  html += '<div style="font-size:0.55rem;"><span style="color:#6a6a80;">ID:</span> <span style="color:#484860;">' + esc(t.id || '--') + '</span></div>';
  if (t.time) html += '<div style="font-size:0.55rem;"><span style="color:#6a6a80;">Generated:</span> <span style="color:#8a8aa0;">' + window.timeAgo(t.time) + '</span></div>';
  html += '</div>';

  // Tags
  if (t.tags && t.tags.length) {
    html += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;">';
    t.tags.forEach(function(tag) {
      html += '<span style="padding:1px 6px;background:#1a1a2e;border:1px solid #2a2a44;border-radius:3px;font-size:0.55rem;color:#0cf;">' + esc(tag) + '</span>';
    });
    html += '</div>';
  }

  openModal('Meta-Thought', html);
};


// ---------------------------------------------------------------------------
// Eval Sidecar Drill-down
// ---------------------------------------------------------------------------

window.openEvalDetail = function() {
  var snap = window._lastSnap || {};
  var ev = snap.eval || {};
  if (!Object.keys(ev).length) { _toast('No eval data', '#f44'); return; }

  var banner = ev.banner || {};
  var sm = ev.store_meta || {};
  var tap = ev.tap || {};
  var coll = ev.collector || {};
  var files = ev.store_file_sizes || {};

  var html = '';

  // Banner headline
  html += '<div style="display:flex;gap:12px;margin-bottom:10px;padding:8px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:6px;">' +
    '<div><div style="font-size:0.5rem;color:#6a6a80;">Mode</div><div style="font-size:0.75rem;color:#0cf;">' + esc(banner.mode || '--') + '</div></div>' +
    '<div><div style="font-size:0.5rem;color:#6a6a80;">Version</div><div style="font-size:0.75rem;color:#8a8aa0;">' + esc(banner.scoring_version || '--') + '</div></div>' +
    '<div><div style="font-size:0.5rem;color:#6a6a80;">Uptime</div><div style="font-size:0.75rem;color:#8a8aa0;">' + (banner.uptime_s ? _fmtUptime(banner.uptime_s) : '--') + '</div></div>' +
    '<div><div style="font-size:0.5rem;color:#6a6a80;">Freshness</div><div style="font-size:0.75rem;color:' + ((banner.data_freshness_s || 0) < 30 ? '#0f9' : '#ff0') + ';">' + fmtNum(banner.data_freshness_s, 1) + 's</div></div>' +
    '</div>';

  // Store metadata
  html += '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Store Metadata</div>';
  html += '<table style="width:100%;font-size:0.55rem;border-collapse:collapse;margin-bottom:8px;">';
  var storeMeta = [
    ['Events Written', fmtNum(sm.total_events_written || 0)],
    ['Snapshots Written', fmtNum(sm.total_snapshots_written || 0)],
    ['Scorecards Written', fmtNum(sm.total_scorecards_written || 0)],
    ['Dropped Events', fmtNum(sm.dropped_event_count || 0)],
    ['Schema Version', sm.schema_version || '--'],
    ['Created', sm.created_at ? window.timeAgo(sm.created_at) : '--'],
    ['Last Flush', sm.last_flush_ts ? window.timeAgo(sm.last_flush_ts) : '--'],
    ['Last Scorecard', sm.last_scorecard_ts ? window.timeAgo(sm.last_scorecard_ts) : '--']
  ];
  storeMeta.forEach(function(r) {
    html += '<tr style="border-bottom:1px solid #0d0d1a;"><td style="padding:2px 4px;color:#6a6a80;">' + r[0] + '</td><td style="padding:2px 4px;text-align:right;">' + esc(String(r[1])) + '</td></tr>';
  });
  html += '</table>';

  // File sizes
  if (Object.keys(files).length) {
    html += '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">File Sizes</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-bottom:8px;">';
    Object.entries(files).forEach(function(e) {
      html += '<div style="padding:4px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;text-align:center;">' +
        '<div style="font-size:0.5rem;color:#6a6a80;">' + esc(e[0]) + '</div>' +
        '<div style="font-size:0.62rem;color:#0cf;">' + _fmtBytes(e[1]) + '</div></div>';
    });
    html += '</div>';
  }

  // Rotations
  var rotations = sm.rotation_counts || {};
  if (Object.keys(rotations).length) {
    html += '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Rotation Counts</div>';
    html += '<div style="display:flex;gap:8px;font-size:0.55rem;margin-bottom:8px;">';
    Object.entries(rotations).forEach(function(e) {
      html += '<span style="color:' + (e[1] > 0 ? '#f90' : '#484860') + ';">' + esc(e[0]) + ': ' + e[1] + '</span>';
    });
    html += '</div>';
  }

  // Tap details
  html += '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Event Tap</div>';
  html += '<div style="display:flex;gap:12px;font-size:0.55rem;margin-bottom:8px;">' +
    '<span>wired: <span style="color:' + (tap.wired ? '#0f9' : '#f44') + ';">' + (tap.wired ? 'yes' : 'no') + '</span></span>' +
    '<span>buffer: ' + (tap.buffer_size || 0) + '</span>' +
    '<span>total buffered: ' + (tap.total_buffered || 0) + '</span>' +
    '<span>event types: ' + (tap.tapped_event_count || 0) + '</span>' +
    '<span>mode: <span style="color:#0cf;">' + esc(tap.current_mode || '--') + '</span></span>' +
    '</div>';

  // Collector details
  html += '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Collector</div>';
  html += '<div style="display:flex;gap:12px;font-size:0.55rem;margin-bottom:8px;">' +
    '<span>snapshots: ' + (coll.snapshots_collected || 0) + '</span>' +
    '<span>interval: ' + fmtNum(coll.interval_s || 60, 0) + 's</span>' +
    '<span>errors: <span style="color:' + ((coll.collect_errors || 0) > 0 ? '#f44' : '#0f9') + ';">' + (coll.collect_errors || 0) + '</span></span>' +
    '<span>last: ' + (coll.last_collect_ts ? window.timeAgo(coll.last_collect_ts) : '--') + '</span>' +
    '</div>';

  // Full event counts
  var ec = ev.event_counts || {};
  var ecEntries = Object.entries(ec).sort(function(a, b) { return b[1] - a[1]; });
  if (ecEntries.length) {
    html += '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">All Event Counts (' + ecEntries.length + ' types)</div>';
    html += '<div style="max-height:200px;overflow-y:auto;border:1px solid #1a1a2e;border-radius:4px;padding:4px;">';
    ecEntries.forEach(function(e) {
      html += '<div style="display:flex;justify-content:space-between;padding:1px 4px;font-size:0.5rem;border-bottom:1px solid #0d0d1a;">' +
        '<span style="color:#8a8aa0;">' + esc(e[0]) + '</span><span style="color:#0cf;">' + e[1] + '</span></div>';
    });
    html += '</div>';
  }

  // Dream artifacts
  var dream = ev.dream || {};
  if (dream.buffer_size) {
    html += '<div style="font-size:0.65rem;color:#6a6a80;margin-top:8px;margin-bottom:3px;">Dream Artifacts</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:4px;">';
    html += '<div style="padding:3px;background:#0d0d1a;border-radius:3px;text-align:center;"><div style="font-size:0.48rem;color:#6a6a80;">buffer</div><div style="font-size:0.62rem;color:#c0f;">' + dream.buffer_size + '</div></div>';
    html += '<div style="padding:3px;background:#0d0d1a;border-radius:3px;text-align:center;"><div style="font-size:0.48rem;color:#6a6a80;">promo rate</div><div style="font-size:0.62rem;color:' + (dream.promotion_rate_color === 'green' ? '#0f9' : '#ff0') + ';">' + fmtPct(dream.promotion_rate) + '</div></div>';
    html += '<div style="padding:3px;background:#0d0d1a;border-radius:3px;text-align:center;"><div style="font-size:0.48rem;color:#6a6a80;">avg conf</div><div style="font-size:0.62rem;">' + fmtPct(dream.avg_confidence) + '</div></div>';
    html += '<div style="padding:3px;background:#0d0d1a;border-radius:3px;text-align:center;"><div style="font-size:0.48rem;color:#6a6a80;">avg coh</div><div style="font-size:0.62rem;">' + fmtPct(dream.avg_coherence) + '</div></div>';
    html += '</div>';
    var byType = dream.by_type || {};
    if (Object.keys(byType).length) {
      html += '<div style="display:flex;flex-wrap:wrap;gap:3px;">';
      Object.entries(byType).forEach(function(e) {
        html += '<span style="padding:1px 4px;border:1px solid #c0f33;border-radius:2px;font-size:0.48rem;color:#c0f;">' + esc(e[0].replace(/_/g, ' ')) + ': ' + e[1] + '</span>';
      });
      html += '</div>';
    }
  }

  // Self-report honesty + emotional independence
  var srh = ev.self_report_honesty || {};
  var ei = ev.emotional_independence || {};
  if (Object.keys(srh).length || Object.keys(ei).length) {
    html += '<div style="font-size:0.65rem;color:#6a6a80;margin-top:8px;margin-bottom:3px;">Behavioral Checks</div>';
    html += '<div style="display:flex;gap:12px;font-size:0.55rem;margin-bottom:4px;">';
    if (srh.score != null) html += '<span>self-report honesty: <span style="color:' + (srh.color === 'green' ? '#0f9' : '#ff0') + ';">' + fmtPct(srh.score) + '</span></span>';
    if (ei.score != null) html += '<span>emotional independence: <span style="color:' + (ei.color === 'green' ? '#0f9' : '#ff0') + ';">' + fmtPct(ei.score) + '</span></span>';
    html += '</div>';
  }

  openModal('Eval Sidecar — Full Details', html, { wide: true });
};


// ---------------------------------------------------------------------------
// Autonomy Delta Drill-down
// ---------------------------------------------------------------------------

window.openDeltaDetail = function(idx) {
  var snap = window._lastSnap || {};
  var auto = snap.autonomy || {};
  var deltas = auto.recent_deltas || [];
  var d = deltas[idx];
  if (!d) { _toast('Delta not found', '#f44'); return; }

  var html = '';

  // Net improvement headline
  var net = d.net_improvement;
  var netC = net > 0.01 ? '#0f9' : net < -0.01 ? '#f44' : '#6a6a80';
  html += '<div style="text-align:center;margin-bottom:8px;">' +
    '<div style="font-size:1rem;font-weight:700;color:' + netC + ';">' + (net != null ? (net > 0 ? '+' : '') + fmtNum(net, 4) : '--') + '</div>' +
    '<div style="font-size:0.55rem;color:#6a6a80;">Net Improvement</div></div>';

  // Intent ID + find matching question
  html += '<div style="font-size:0.55rem;color:#484860;margin-bottom:6px;">Intent: ' + esc(d.intent_id || '--') + '</div>';
  var completed = auto.completed || [];
  var matchedQ = '';
  for (var i = 0; i < completed.length; i++) {
    if (completed[i].id === d.intent_id) { matchedQ = completed[i].question || ''; break; }
  }
  if (matchedQ) {
    html += '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;margin-bottom:8px;">' +
      '<div style="font-size:0.55rem;color:#6a6a80;">Research Question</div>' +
      '<div style="font-size:0.68rem;color:#e0e0e8;">' + esc(matchedQ) + '</div></div>';
  }

  // Deltas table — before/after comparison
  var deltaFields = d.deltas || {};
  var cfDeltas = d.counterfactual_deltas || {};
  var attrib = d.attribution || {};
  var baseline = d.baseline || {};
  var post = d.post || {};

  if (Object.keys(deltaFields).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Metrics</div>';
    html += '<table style="width:100%;font-size:0.55rem;border-collapse:collapse;margin-bottom:8px;">';
    html += '<tr style="border-bottom:1px solid #2a2a44;"><th style="text-align:left;color:#6a6a80;padding:2px 4px;">Metric</th>' +
      '<th style="text-align:right;color:#6a6a80;padding:2px 4px;">Before</th>' +
      '<th style="text-align:right;color:#6a6a80;padding:2px 4px;">After</th>' +
      '<th style="text-align:right;color:#6a6a80;padding:2px 4px;">Delta</th>' +
      '<th style="text-align:right;color:#6a6a80;padding:2px 4px;">Attribution</th></tr>';

    Object.entries(deltaFields).forEach(function(e) {
      var name = e[0], dv = e[1];
      var bv = baseline[name];
      var pv = post[name];
      var av = attrib[name];
      var dc = dv > 0.001 ? '#0f9' : dv < -0.001 ? '#f44' : '#6a6a80';
      var ac = av > 0.001 ? '#0f9' : av < -0.001 ? '#f44' : '#484860';
      html += '<tr style="border-bottom:1px solid #0d0d1a;">' +
        '<td style="padding:2px 4px;color:#8a8aa0;">' + esc(name.replace(/_/g, ' ')) + '</td>' +
        '<td style="text-align:right;padding:2px 4px;color:#484860;">' + (bv != null ? fmtNum(bv, 3) : '--') + '</td>' +
        '<td style="text-align:right;padding:2px 4px;color:#484860;">' + (pv != null ? fmtNum(pv, 3) : '--') + '</td>' +
        '<td style="text-align:right;padding:2px 4px;color:' + dc + ';">' + (dv != null ? (dv > 0 ? '+' : '') + fmtNum(dv, 4) : '--') + '</td>' +
        '<td style="text-align:right;padding:2px 4px;color:' + ac + ';">' + (av != null ? (av > 0 ? '+' : '') + fmtNum(av, 4) : '--') + '</td></tr>';
    });
    html += '</table>';
  }

  // Counterfactual summary
  if (Object.keys(cfDeltas).length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Counterfactual Deltas (what would have happened without this research)</div>';
    html += '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">';
    Object.entries(cfDeltas).forEach(function(e) {
      if (Math.abs(e[1]) > 0.0001) {
        var cc = e[1] > 0 ? '#0f9' : '#f44';
        html += '<span style="padding:1px 4px;border:1px solid ' + cc + '33;color:' + cc + ';border-radius:2px;font-size:0.48rem;">' +
          esc(e[0].replace(/_/g, ' ')) + ': ' + (e[1] > 0 ? '+' : '') + fmtNum(e[1], 4) + '</span>';
      }
    });
    html += '</div>';
  }

  openModal('Autonomy Delta', html, { wide: true });
};


// ---------------------------------------------------------------------------
// Skill Detail Drill-down (enhanced)
// ---------------------------------------------------------------------------

window.openSkillDetail = function(skillId) {
  openModal('Skill: ' + esc(String(skillId).substring(0, 40)), '<div id="skill-detail-body">Loading audit packet...</div>', { wide: true });
  fetch('/api/skills/' + encodeURIComponent(skillId), { headers: _authHeaders() })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      var body = document.getElementById('skill-detail-body');
      if (!body) return;
      if (data.error) {
        body.innerHTML = '<div style="color:#f44;">' + esc(data.error) + '</div>';
        return;
      }
      body.innerHTML = _renderSkillAuditDetail(skillId, data);
    })
    .catch(function(err) {
      var body = document.getElementById('skill-detail-body');
      if (body) body.innerHTML = '<div style="color:#f44;">Error: ' + esc(err.message) + '</div>';
    });
};

function _renderSkillAuditDetail(skillId, data) {
  var sk = data.skill || {};
  var audit = data.audit_packet || {};
  var decision = audit.decision_summary || {};
  var request = audit.request_context || {};
  var contract = audit.resolver_contract || {};
  var classes = audit.evidence_classes || {};
  var handoff = audit.operational_handoff || {};
  var missing = audit.missing_proof || [];
  var html = '';

  html += '<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;">' +
    '<div style="display:flex;align-items:center;gap:8px;">' +
    (window._statusBadge ? window._statusBadge(sk.status || audit.status || '--') : '') +
    '<span style="font-size:0.85rem;font-weight:600;color:#e0e0e8;">' + esc(sk.name || sk.skill_id || skillId) + '</span>' +
    '</div>' +
    '<span style="font-size:0.55rem;color:#6a6a80;">audit schema v' + esc(audit.schema_version || 1) + '</span>' +
    '</div>';

  if (decision.message) {
    html += '<div style="padding:7px 8px;border:1px solid #24344a;background:#0b1020;border-radius:4px;font-size:0.62rem;color:#cfd8e8;line-height:1.4;margin-bottom:8px;">' +
      esc(decision.message) + '</div>';
  }

  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px;">' +
    _skillAuditCard('Status', sk.status || audit.status || '--') +
    _skillAuditCard('Verified', audit.verified ? 'Yes' : 'No', audit.verified ? '#0f9' : '#ff0') +
    _skillAuditCard('Phase', decision.current_phase || '--') +
    _skillAuditCard('Missing Proof', String(decision.missing_count || 0), (decision.missing_count || 0) ? '#f90' : '#0f9') +
    '</div>';

  html += _skillAuditSection('Request Context', _skillAuditRows([
    ['Job', request.job_id],
    ['User text', request.user_text],
    ['Speaker', request.speaker],
    ['Risk', request.risk_level],
    ['Matrix', request.matrix_protocol ? ('yes ' + (request.protocol_id || '')) : 'no'],
    ['Created', request.created_at]
  ]));

  var cap = contract.capability_contract || {};
  html += _skillAuditSection('Contract / Resolver', _skillAuditRows([
    ['Capability type', contract.capability_type],
    ['Required evidence', (contract.required_evidence || []).join(', ')],
    ['Contract ID', cap.execution_contract_id],
    ['Required executor', cap.required_executor_kind],
    ['Acquisition eligible', cap.acquisition_eligible === true ? 'yes' : 'no'],
    ['Plan', contract.plan_summary]
  ]));

  html += _skillAuditSection('Evidence Classes', _skillAuditTags(classes));

  html += _skillAuditSection('Operational Handoff', _renderSkillOperationalHandoff(handoff));

  html += _skillAuditSection('Acquisition Proof Chain', _renderSkillAcquisitionChain(audit.acquisition_chain || {}));

  if (missing.length) {
    html += _skillAuditSection('Why This Is Not Verified Yet', missing.map(function(m) {
      return '<div class="skill-detail-limitation"><strong>' + esc(m.name || '') + '</strong>: ' + esc(m.reason || '') + '</div>';
    }).join(''));
  }

  html += _skillAuditSection('Evidence Checks', _renderSkillAuditEvidence(audit.evidence_history || []));
  html += _skillAuditSection('Artifacts', _renderSkillAuditArtifacts(audit.artifacts || []));
  html += _skillAuditSection('Timeline', _renderSkillAuditTimeline(audit.timeline || []));

  if ((audit.integrity_notes || []).length) {
    html += _skillAuditSection('Integrity Notes', (audit.integrity_notes || []).map(function(n) {
      return '<div style="font-size:0.58rem;color:#8a8aa0;line-height:1.35;margin:2px 0;">' + esc(n) + '</div>';
    }).join(''));
  }

  html += '<div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap;">';
  if (handoff.status === 'acquisition_failed' || handoff.status === 'acquisition_cancelled') {
    html += '<button class="j-btn-sm j-btn-primary" onclick="window._retrySkillHandoff(\'' + esc(skillId) + '\')">Retry Operational Build</button>' +
      '<div style="font-size:0.58rem;color:#f88;line-height:1.35;max-width:520px;">Linked acquisition ended: ' + esc(handoff.last_error || handoff.status) + '</div>';
  } else if (handoff.acquisition_id) {
    html += '<a class="j-btn-sm j-btn-primary" href="/capability-pipeline#acquisition:' + encodeURIComponent(handoff.acquisition_id) + '">Open Acquisition Review</a>' +
      '<div style="font-size:0.58rem;color:#ff0;line-height:1.35;max-width:520px;">Acquisition has started. Review, approve, reject/revise, or cancel the plan in Capability Acquisition.</div>';
  } else if (handoff.status === 'awaiting_operator_approval') {
    html += '<button class="j-btn-sm j-btn-primary" onclick="window._approveSkillHandoff(\'' + esc(skillId) + '\')">Approve Operational Build</button>' +
      '<button class="j-btn-sm j-btn-red" onclick="window._rejectSkillHandoff(\'' + esc(skillId) + '\')">Reject Operational Build</button>';
  }
  html += '<button class="j-btn-sm j-btn-red" onclick="window._deleteSkill(\'' + esc(skillId) + '\')">Delete Skill</button>' +
    '</div>';
  return html;
}

function _skillAuditCard(label, value, color) {
  return '<div style="background:#0b0b16;border:1px solid #1f2740;border-radius:4px;padding:6px;">' +
    '<div style="font-size:0.5rem;color:#6a6a80;text-transform:uppercase;">' + esc(label) + '</div>' +
    '<div style="font-size:0.72rem;color:' + (color || '#e0e0e8') + ';font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + esc(value || '--') + '</div>' +
    '</div>';
}

function _skillAuditSection(title, body) {
  if (!body) body = '<div style="font-size:0.58rem;color:#6a6a80;">No data.</div>';
  return '<div class="skill-detail-section"><h3>' + esc(title) + '</h3>' + body + '</div>';
}

function _skillAuditRows(rows) {
  return rows.filter(function(r) { return r[1] != null && r[1] !== ''; }).map(function(r) {
    return '<div class="skill-detail-row"><span class="label">' + esc(r[0]) + '</span><span class="value" style="max-width:70%;text-align:right;">' + esc(String(r[1]).substring(0, 300)) + '</span></div>';
  }).join('');
}

function _skillAuditTags(obj) {
  return Object.keys(obj || {}).map(function(k) {
    var val = !!obj[k];
    return '<span class="skill-detail-tag" style="border:1px solid ' + (val ? '#0f966' : '#6a6a8033') + ';color:' + (val ? '#0f9' : '#6a6a80') + ';">' +
      esc(k) + ': ' + (val ? 'yes' : 'no') + '</span>';
  }).join('');
}

function _renderSkillOperationalHandoff(handoff) {
  if (!handoff || !handoff.status) {
    return '<div style="font-size:0.58rem;color:#6a6a80;">No operational handoff request.</div>';
  }
  return _skillAuditRows([
    ['Status', handoff.status],
    ['Approval required', handoff.approval_required ? 'yes' : 'no'],
    ['Contract ID', handoff.contract_id],
    ['Required executor', handoff.required_executor_kind],
    ['Acquisition ID', handoff.acquisition_id],
    ['Outcome class', handoff.outcome_class],
    ['Risk tier', handoff.risk_tier],
    ['Approved by', handoff.approved_by],
    ['Approved at', handoff.approved_at],
    ['Rejected by', handoff.rejected_by],
    ['Rejected at', handoff.rejected_at],
    ['Rejection reason', handoff.rejection_reason],
    ['Last error type', handoff.last_error_type],
    ['Last error', handoff.last_error],
    ['Updated', handoff.updated_at]
  ]);
}

function _renderSkillAcquisitionChain(chain) {
  if (!chain || !chain.acquisition_id) {
    return '<div style="font-size:0.58rem;color:#6a6a80;">No linked acquisition proof job.</div>';
  }
  var plugin = chain.plugin || {};
  var verification = chain.verification || {};
  var html = _skillAuditRows([
    ['Acquisition ID', chain.acquisition_id],
    ['Exists', chain.exists === false ? 'no' : 'yes'],
    ['Status', chain.status],
    ['Outcome class', chain.outcome_class],
    ['Risk tier', chain.risk_tier],
    ['Plugin', chain.plugin_name],
    ['Plugin state', plugin.state],
    ['Execution mode', plugin.execution_mode],
    ['Verification ID', chain.verification_id],
    ['Sandbox ref', verification.sandbox_result_ref],
    ['Verification passed', verification.overall_passed === true ? 'yes' : (verification.overall_passed === false ? 'no' : '')]
  ]);
  var lanes = chain.lanes || {};
  var laneNames = Object.keys(lanes);
  if (laneNames.length) {
    html += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;">' + laneNames.map(function(name) {
      var lane = lanes[name] || {};
      var done = lane.status === 'completed';
      var blocked = lane.status === 'blocked' || lane.status === 'failed';
      var color = done ? '#0f9' : (blocked ? '#f66' : '#f90');
      return '<span class="skill-detail-tag" style="border:1px solid ' + color + '66;color:' + color + ';">' +
        esc(name + ': ' + (lane.status || '--')) + '</span>';
    }).join('') + '</div>';
  }
  if (verification.lane_verdicts) {
    html += '<pre style="white-space:pre-wrap;font-size:0.5rem;color:#aaa;background:#050510;border:1px solid #151522;padding:5px;border-radius:3px;max-height:140px;overflow:auto;margin-top:6px;">' +
      esc(_jsonCompact(verification.lane_verdicts)) + '</pre>';
  }
  return html;
}

function _renderSkillAuditEvidence(history) {
  if (!history.length) return '<div style="font-size:0.58rem;color:#6a6a80;">No evidence recorded yet.</div>';
  var html = '';
  history.slice(-12).forEach(function(ev) {
    html += '<div style="border:1px solid #1a1a2e;border-radius:4px;padding:6px;margin-bottom:6px;background:#0b0b16;">' +
      '<div style="display:flex;justify-content:space-between;gap:8px;font-size:0.58rem;color:#8a8aa0;margin-bottom:4px;">' +
      '<span>' + esc(ev.evidence_id || ev.source || 'evidence') + '</span>' +
      '<span>' + esc((ev.is_current ? 'current' : 'historical') + ' · ' + (ev.result || '--')) + '</span>' +
      '</div>';
    (ev.tests || []).forEach(function(t) {
      html += '<div class="skill-detail-test">' +
        '<span class="test-status ' + (t.passed ? 'pass' : 'fail') + '">' + (t.passed ? 'PASS' : 'FAIL') + '</span>' +
        '<span style="flex:1;">' + esc(t.name || '') + '<br><span style="color:#8a8aa0;">' + esc(t.details || '') + '</span></span>' +
        '</div>';
      if (t.expected != null || t.actual != null) {
        html += '<pre style="white-space:pre-wrap;font-size:0.52rem;color:#aaa;background:#050510;border:1px solid #151522;padding:5px;border-radius:3px;max-height:180px;overflow:auto;">expected: ' +
          esc(_jsonCompact(t.expected)) + '\nactual: ' + esc(_jsonCompact(t.actual)) + '</pre>';
      }
    });
    html += '</div>';
  });
  return html;
}

function _renderSkillAuditArtifacts(artifacts) {
  if (!artifacts.length) return '<div style="font-size:0.58rem;color:#6a6a80;">No artifacts recorded yet.</div>';
  return artifacts.slice(-12).map(function(a) {
    var preview = a.preview != null ? _jsonCompact(a.preview).substring(0, 800) : '';
    return '<div style="border-bottom:1px solid #1a1a2e;padding:5px 0;font-size:0.58rem;">' +
      '<div style="display:flex;justify-content:space-between;gap:8px;">' +
      '<span style="color:#e0e0e8;">' + esc(a.type || a.id || 'artifact') + '</span>' +
      '<span style="color:' + (a.exists ? '#0f9' : '#f90') + ';">' + (a.exists ? 'exists' : 'missing') + (a.is_current_job ? ' · current' : ' · historical') + '</span>' +
      '</div>' +
      (a.path ? '<div style="color:#6a6a80;word-break:break-all;">' + esc(a.path) + '</div>' : '') +
      (preview ? '<pre style="white-space:pre-wrap;font-size:0.5rem;color:#aaa;background:#050510;border:1px solid #151522;padding:5px;border-radius:3px;max-height:140px;overflow:auto;">' + esc(preview) + '</pre>' : '') +
      '</div>';
  }).join('');
}

function _renderSkillAuditTimeline(timeline) {
  if (!timeline.length) return '<div style="font-size:0.58rem;color:#6a6a80;">No timeline events yet.</div>';
  return timeline.slice(-40).map(function(e) {
    return '<div style="display:grid;grid-template-columns:128px 120px 1fr;gap:6px;font-size:0.56rem;border-bottom:1px solid #151522;padding:3px 0;">' +
      '<span style="color:#6a6a80;">' + esc(e.ts || '') + '</span>' +
      '<span style="color:' + (e.is_current_job ? '#0cf' : '#8a8aa0') + ';">' + esc(e.type || '') + '</span>' +
      '<span style="color:#ccc;">' + esc(e.message || '') + '</span>' +
      '</div>';
  }).join('');
}

function _jsonCompact(value) {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  try { return JSON.stringify(value, null, 2); } catch (e) { return String(value); }
}

window._deleteSkill = function(skillId) {
  if (!confirm('Delete skill "' + skillId + '"?')) return;
  _apiDelete('/api/skills/' + encodeURIComponent(skillId)).then(function(r) {
    _toast(r.ok ? 'Skill deleted' : 'Failed', r.ok ? '#0f9' : '#f44');
    if (r.ok) window.closeModal();
  });
};

window._approveSkillHandoff = function(skillId) {
  var notes = prompt('Approve Jarvis to build an operational plugin/tool for "' + skillId + '"? Optional notes:', 'Operator approved operational proof build.');
  if (notes === null) return;
  _apiPost('/api/skills/' + encodeURIComponent(skillId) + '/handoff/approve', {
    approved_by: 'dashboard',
    notes: notes || ''
  }).then(function(r) {
    return r.json().then(function(d) { return { ok: r.ok, data: d }; });
  }).then(function(res) {
    if (!res.ok || res.data.error || res.data.ok === false) {
      _toast(res.data.error || res.data.reason || 'Approval failed', '#f44');
      return;
    }
    _toast('Operational build approved: ' + (res.data.acquisition_id || 'acquisition started'), '#0f9');
    window.openSkillDetail(skillId);
  }).catch(function() { _toast('Approval failed', '#f44'); });
};

window._rejectSkillHandoff = function(skillId) {
  var reason = prompt('Why reject the operational build for "' + skillId + '"?', '');
  if (reason === null) return;
  reason = (reason || '').trim();
  if (reason.length < 5) {
    _toast('Rejection reason is required', '#f44');
    return;
  }
  _apiPost('/api/skills/' + encodeURIComponent(skillId) + '/handoff/reject', {
    rejected_by: 'dashboard',
    reason: reason
  }).then(function(r) {
    return r.json().then(function(d) { return { ok: r.ok, data: d }; });
  }).then(function(res) {
    if (!res.ok || res.data.error || res.data.ok === false) {
      _toast(res.data.error || res.data.reason || 'Rejection failed', '#f44');
      return;
    }
    _toast('Operational build rejected', '#ff0');
    window.openSkillDetail(skillId);
  }).catch(function() { _toast('Rejection failed', '#f44'); });
};

window._retrySkillHandoff = function(skillId) {
  var notes = prompt('Retry operational build for "' + skillId + '"? Optional notes:', 'Retry with a concrete implementation plan and sandbox-backed proof.');
  if (notes === null) return;
  _apiPost('/api/skills/' + encodeURIComponent(skillId) + '/handoff/retry', {
    approved_by: 'dashboard',
    notes: notes || ''
  }).then(function(r) {
    return r.json().then(function(d) { return { ok: r.ok, data: d }; });
  }).then(function(res) {
    if (!res.ok || res.data.error || res.data.ok === false) {
      _toast(res.data.error || res.data.reason || 'Retry failed', '#f44');
      return;
    }
    _toast('Operational build retried: ' + (res.data.acquisition_id || 'acquisition started'), '#0f9');
    window.openSkillDetail(skillId);
  }).catch(function() { _toast('Retry failed', '#f44'); });
};
