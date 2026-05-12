/* ═══════════════════════════════════════════════════════════════════════════
   JARVIS Dashboard — Tab Renderers (Activity, Learning, Diagnostics)
   Loaded via <script> BEFORE dashboard.js, AFTER charts.js.
   Exposes: window.renderActivity, window.renderLearning, window.renderDiagnostics
   ═══════════════════════════════════════════════════════════════════════════ */

var esc = window.esc || function(s) { if (!s) return ''; var d = document.createElement('div'); d.textContent = s; return d.innerHTML; };
var fmt = window.fmt || function(v, d) { if (v == null) return '--'; return Number(v).toFixed(d); };
var pct = window.pct || function(v) { return v != null ? (v * 100).toFixed(1) + '%' : '--'; };
var fmtNum = window.fmtNum || function(v, d) { if (v == null || Number.isNaN(v)) return '--'; return Number(v).toFixed(d != null ? d : 3); };
var fmtPct = window.fmtPct || function(v, d) { if (v == null || Number.isNaN(v)) return '--'; return (v * 100).toFixed(d != null ? d : 0) + '%'; };
var timeAgo = window.timeAgo || function(ts) { var s = Math.floor(Date.now() / 1000 - ts); if (s < 60) return s + 's ago'; if (s < 3600) return Math.floor(s / 60) + 'm ago'; if (s < 86400) return Math.floor(s / 3600) + 'h ago'; return Math.floor(s / 86400) + 'd ago'; };

function _rnd_sc(status) {
  var map = { active: 'active', done: 'done', waiting: 'waiting', queued: 'queued', blocked: 'blocked', idle: 'idle', completed: 'done', error: 'error' };
  return map[status] || 'idle';
}

function _rnd_ageStr(s) {
  if (!s || s <= 0) return '';
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return (s / 60).toFixed(0) + 'm';
  return (s / 3600).toFixed(1) + 'h';
}

function _rnd_fmtBytes(b) {
  if (!b) return '0 B';
  if (b < 1024) return b + ' B';
  if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
  return (b / (1024 * 1024)).toFixed(2) + ' MB';
}

function _rnd_fmtUptime(s) {
  if (!s && s !== 0) return '--';
  var mins = Math.floor(s / 60);
  var hrs = Math.floor(mins / 60);
  if (hrs > 0) return hrs + 'h ' + (mins % 60) + 'm';
  return mins + 'm';
}

function _statusBadge(status) {
  var colors = {
    active: '#0f9', running: '#0f9', done: '#0cf', completed: '#0cf',
    idle: '#6a6a80', waiting: '#ff0', queued: '#ff0', blocked: '#f44',
    error: '#f44', shadow: '#c0f', live: '#0f9', disabled: '#f44',
    learning: '#ff0', verified: '#0f9', unknown: '#6a6a80',
    pass: '#0f9', fail: '#f44', warning: '#ff0',
    promoted: '#0f9', candidate: '#ff0', proposed: '#ff0',
    stalled: '#f90', abandoned: '#f44', paused: '#ff0'
  };
  var c = colors[(status || '').toLowerCase()] || '#6a6a80';
  return '<span class="status-badge" style="color:' + c + ';border-color:' + c + '33;background:' + c + '15;padding:1px 6px;border:1px solid;border-radius:3px;font-size:0.65rem;font-weight:600;">' + esc(status || '--') + '</span>';
}

function _barFill(value, max, color) {
  var pctVal = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return '<div style="height:6px;background:#1a1a2e;border-radius:3px;overflow:hidden;flex:1;">' +
    '<div style="width:' + pctVal + '%;height:100%;background:' + (color || '#0f9') + ';border-radius:3px;transition:width 0.3s;"></div>' +
    '</div>';
}

function _metricRow(label, value) {
  return '<div class="metric-row" style="display:flex;justify-content:space-between;align-items:center;padding:2px 0;">' +
    '<span class="metric-label" style="color:#6a6a80;font-size:0.72rem;">' + esc(label) + '</span>' +
    '<span class="metric-value" style="font-size:0.72rem;">' + esc(String(value != null ? value : '--')) + '</span></div>';
}

function _statCard(label, value, color) {
  var c = color || '#e0e0e8';
  return '<div class="stat-card" style="text-align:center;padding:6px 8px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
    '<div class="stat-value" style="font-size:1.1rem;font-weight:700;color:' + c + ';">' + esc(String(value != null ? value : '--')) + '</div>' +
    '<div class="stat-label" style="font-size:0.6rem;color:#6a6a80;margin-top:2px;">' + esc(label) + '</div></div>';
}

function _rendererPanelDocLink(opts, title) {
  if (window.panelDocLink) return window.panelDocLink(opts || {}, title);
  var maps = window.DASHBOARD_DOC_LINKS_BY_TITLE || {};
  var href = (opts && opts.docsHref) || maps[title] || '';
  if (!href || href === false) return '';
  return '<a class="panel-doc-link" href="' + esc(href) + '" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation();" title="Open docs for ' + esc(title) + '">Docs</a>';
}

function _panelHdr(title, badge, opts) {
  var docsLink = _rendererPanelDocLink(opts || {}, title);
  return '<div class="panel-header" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">' +
    '<h3 style="margin:0;font-size:0.85rem;color:#e0e0e8;">' + esc(title) + '</h3>' +
    '<span class="panel-header-actions">' + docsLink + (badge || '') + '</span></div>';
}

function _panel(title, body, badge, opts) {
  opts = opts || {};
  var attrs = '';
  if (window._panelAttrs) {
    attrs = window._panelAttrs(opts, title);
  } else {
    var fallbackId = opts.panelId || (window.resolvePanelId ? window.resolvePanelId(title) : '');
    if (fallbackId) attrs += ' data-panel-id="' + esc(fallbackId) + '"';
    if (opts.ownerTab) attrs += ' data-owner-tab="' + esc(opts.ownerTab) + '"';
  }
  return '<div class="panel"' + attrs + ' style="background:#12121e;border:1px solid #1a1a2e;border-radius:6px;padding:12px;margin-bottom:10px;">' +
    _panelHdr(title, badge, opts) + body + '</div>';
}

function _renderMovedToOwnerPanel(title, ownerPanelId, detail, summaryPanelId, summaryOwnerTab) {
  var owner = (window.getPanelOwnership && window.getPanelOwnership(ownerPanelId)) || {};
  var ownerTab = owner.ownerTab || 'trust';
  var currentTab = summaryOwnerTab || 'diagnostics';
  var body = '<div style="font-size:0.62rem;color:#8a8aa0;line-height:1.45;margin-bottom:8px;">' + esc(detail || '') + '</div>' +
    '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">' +
    '<button class="j-btn-sm" onclick="window.openOwnedPanel && window.openOwnedPanel(\'' + esc(ownerPanelId) + '\')">Open in ' + esc(ownerTab) + '</button>' +
    '<span style="font-size:0.56rem;color:#6a6a80;">owner: ' + esc(ownerTab) + '</span>' +
    '</div>';
  return _panel(title, body, '', { panelId: summaryPanelId || (ownerPanelId + '-summary'), ownerTab: currentTab });
}

function _emptyMsg(text) {
  return '<div style="color:#484860;font-size:0.7rem;padding:4px 0;">' + esc(text) + '</div>';
}

function _tagGrid(obj, emptyText) {
  var entries = Object.entries(obj || {}).sort(function(a, b) { return b[1] - a[1]; });
  if (!entries.length) return '<span style="color:#484860;font-size:0.7rem;">' + esc(emptyText || 'none') + '</span>';
  return entries.map(function(e) {
    return '<span style="display:inline-block;padding:1px 6px;margin:1px 2px;border:1px solid #2a2a44;border-radius:3px;font-size:0.65rem;">' + esc(e[0]) + ' <b>' + e[1] + '</b></span>';
  }).join('');
}


/* ═══════════════════════════════════════════════════════════════════════════
   TAB: ACTIVITY — "What is JARVIS doing right now?"
   ═══════════════════════════════════════════════════════════════════════════ */

window.renderActivity = function(snap) {
  var root = document.getElementById('activity-root');
  if (!root) return;

  // Preserve live camera element across re-renders to avoid MJPEG stream restart flicker
  var cameraHiddenForRender = _isActivityCameraHidden();
  var savedCam = document.getElementById('activity-cam-container');
  if (!cameraHiddenForRender && savedCam && savedCam.querySelector('img')) {
    savedCam.parentNode.removeChild(savedCam);
  } else {
    savedCam = null;
  }

  var html = '';
  if (window._renderTabWayfinding) html += window._renderTabWayfinding('activity');

  // Activity toolbar
  html += '<div class="j-toolbar">' +
    '<button class="j-btn-sm" onclick="window.openCameraFeed && window.openCameraFeed()">Camera Feed</button>' +
    '<button class="j-btn-sm" onclick="window.openGoalObserve && window.openGoalObserve()">Add Goal Signal</button>' +
    '</div>';

  // Operations spans full width
  html += _renderOpsPanel(snap);

  // Flight Recorder — conversation history with drill-down
  html += _renderFlightRecorderPanel(snap);

  // World & Perception section (full width with camera + data side by side)
  html += _renderWorldPerceptionSection(snap);

  // 2-column grid for the rest
  html += '<div class="j-panel-grid">';
  html += _renderAutonomyPanel(snap);
  html += _renderGoalsPanel(snap);
  html += _renderDrivesPanel(snap);
  html += _renderMovedToOwnerPanel(
    'Simulator Detail (Consolidated)',
    'activity-world-perception',
    'Simulator and world-model trace details are consolidated in World & Perception above to avoid duplicate surfaces.',
    'activity-simulator-summary',
    'activity'
  );
  html += _renderMovedToOwnerPanel(
    'Self-Improvement (Moved)',
    'learning-self-improvement',
    'Deep self-improvement status now lives in Learning so Activity can stay focused on live runtime flow.',
    'activity-self-improvement-summary',
    'activity'
  );
  html += '</div>';

  root.innerHTML = html;

  // Re-attach preserved camera element or load fresh
  if (savedCam) {
    var placeholder = document.getElementById('activity-cam-container');
    if (placeholder && placeholder.parentNode) {
      placeholder.parentNode.replaceChild(savedCam, placeholder);
    }
  } else {
    _loadActivityCameraFeed();
  }
};


var _activityCamLoaded = false;
var _ACTIVITY_CAMERA_HIDDEN_KEY = 'jarvis-hide-activity-camera-feed-v1';

function _isActivityCameraHidden() {
  try {
    return localStorage.getItem(_ACTIVITY_CAMERA_HIDDEN_KEY) === 'true';
  } catch (_) {
    return false;
  }
}

function _setActivityCameraHidden(hidden) {
  try {
    localStorage.setItem(_ACTIVITY_CAMERA_HIDDEN_KEY, hidden ? 'true' : 'false');
  } catch (_) {}
}

window.toggleActivityCameraFeedPrivacy = function() {
  _setActivityCameraHidden(!_isActivityCameraHidden());
  _activityCamLoaded = false;
  if (window._lastSnap && window.renderActivity) {
    window.renderActivity(window._lastSnap);
  } else {
    _loadActivityCameraFeed();
  }
};

function _loadActivityCameraFeed() {
  var container = document.getElementById('activity-cam-container');
  if (!container) return;
  if (_isActivityCameraHidden()) {
    _activityCamLoaded = false;
    container.innerHTML =
      '<div style="text-align:center;color:#6a6a80;font-size:0.68rem;padding:24px;">' +
      '<div style="color:#ff0;margin-bottom:4px;">Camera feed hidden on this browser</div>' +
      '<div style="font-size:0.58rem;">Perception still runs; only the dashboard preview is hidden.</div>' +
      '</div>';
    return;
  }
  if (_activityCamLoaded && container.querySelector('img')) return;

  fetch('/api/config').then(function(r) { return r.json(); }).then(function(cfg) {
    var ctr = document.getElementById('activity-cam-container');
    if (!ctr) return;
    var url = cfg.pi_video_url;
    if (url) {
      _activityCamLoaded = true;
      ctr.innerHTML =
        '<img src="' + esc(url) + '" style="max-width:100%;border-radius:4px;" ' +
        'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'\';">' +
        '<div style="display:none;color:#f44;font-size:0.68rem;padding:16px;">Camera offline \u2014 check Pi connection<br>' +
        '<span style="color:#484860;font-size:0.6rem;">' + esc(url) + '</span></div>';
    } else {
      ctr.innerHTML = '<div style="color:#484860;font-size:0.68rem;padding:16px;">PI_HOST not configured</div>';
    }
  }).catch(function() {
    var ctr = document.getElementById('activity-cam-container');
    if (ctr) ctr.innerHTML = '<div style="color:#484860;font-size:0.68rem;padding:16px;">Config unavailable</div>';
  });
}


function _renderWorldPerceptionSection(snap) {
  var wm = snap.world_model || {};
  var scene = snap.scene || {};
  var attn = snap.attention || {};
  var promo = wm.promotion || {};
  var causal = wm.causal || {};
  var acc = causal.overall_accuracy;
  var userState = wm.user || {};
  var convState = wm.conversation || {};
  var sysState = wm.system || {};
  var staleness = wm.staleness || {};
  var uncertainty = wm.uncertainty || {};
  var simPromo = wm.simulator_promotion || {};
  var simStats = wm.simulator || {};

  var panelAttrs = window._panelAttrs
    ? window._panelAttrs({ panelId: 'activity-world-perception', ownerTab: 'activity' }, 'World & Perception')
    : ' data-panel-id="activity-world-perception" data-owner-tab="activity"';
  var html = '<div class="panel"' + panelAttrs + ' style="margin-bottom:10px;">' +
    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">' +
    '<h3 style="margin:0;font-size:0.9rem;color:#e0e0e8;border-left:3px solid #c0f;padding-left:8px;">World & Perception</h3>' +
    _statusBadge(promo.level_name || 'shadow') + '</div>';

  // ── Row 1: Camera + User/Attention side by side ──
  html += '<div class="j-world-grid">';

  // LEFT: Camera Feed
  html += '<div>';
  var cameraHidden = _isActivityCameraHidden();
  html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">' +
    '<div style="font-size:0.68rem;color:#6a6a80;">Camera Feed</div>' +
    '<button class="j-btn-sm" style="font-size:0.55rem;padding:2px 6px;" onclick="window.toggleActivityCameraFeedPrivacy && window.toggleActivityCameraFeedPrivacy()">' +
    (cameraHidden ? 'Show Feed' : 'Hide Feed') + '</button></div>';
  html += '<div id="activity-cam-container" style="background:#0a0a14;border:1px solid #1a1a2e;border-radius:4px;min-height:200px;display:flex;align-items:center;justify-content:center;">' +
    '<span style="color:#484860;font-size:0.7rem;">' + (cameraHidden ? 'Camera feed hidden' : 'Loading camera...') + '</span></div>';
  html += '</div>';

  // RIGHT: User State + Attention
  html += '<div>';

  // User State (from world model)
  html += '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">User State</div>';
  html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:3px;margin-bottom:6px;">';
  var emotionC = userState.emotion === 'frustrated' ? '#f44' : userState.emotion === 'happy' ? '#0f9' : '#ff0';
  html += _statCard('Speaker', userState.speaker || attn.speaker_identity || 'unknown', '#0cf');
  html += _statCard('Emotion', userState.emotion || attn.user_emotion || '--', emotionC);
  html += _statCard('Gesture', userState.gesture || attn.gesture || '--', '#c0f');
  html += '</div>';

  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-bottom:6px;">';
  var eng = attn.engagement_level != null ? attn.engagement_level : userState.engagement;
  html += _statCard('Engagement', eng != null ? (eng * 100).toFixed(0) + '%' : '--', eng > 0.5 ? '#0f9' : '#6a6a80');
  html += _statCard('Present', userState.present ? 'Yes' : 'No', userState.present ? '#0f9' : '#f44');
  html += _statCard('Presence', attn.presence_confidence != null ? (attn.presence_confidence * 100).toFixed(0) + '%' : '--', '#6a6a80');
  html += _statCard('Interrupt', attn.interruption_allowed ? 'OK' : 'No', attn.interruption_allowed ? '#0f9' : '#f90');
  html += '</div>';

  // Attention detail
  html += '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Attention</div>';
  var focusTarget = attn.focus_target || attn.primary_target || '--';
  html += '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:4px;font-size:0.62rem;">' +
    '<span>Focus: <span style="color:#0cf;">' + esc(String(focusTarget)) + '</span></span>' +
    '<span>Ambient: <span style="color:#6a6a80;">' + esc(attn.ambient_state || '--') + '</span></span>' +
    '<span>Speaker conf: <span style="color:#6a6a80;">' + (attn.speaker_confidence != null ? (attn.speaker_confidence * 100).toFixed(0) + '%' : '--') + '</span></span>' +
    '<span>Emotion conf: <span style="color:#6a6a80;">' + (attn.emotion_confidence != null ? (attn.emotion_confidence * 100).toFixed(0) + '%' : '--') + '</span></span>' +
    '</div>';

  var reasons = attn.reasons || [];
  if (reasons.length) {
    html += '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:4px;">';
    reasons.forEach(function(r) {
      var text = typeof r === 'string' ? r : (r.reason || r.label || '');
      html += '<span style="padding:1px 5px;background:#181828;border:1px solid #2a2a44;border-radius:3px;font-size:0.52rem;color:#e0e0e8;">' + esc(text) + '</span>';
    });
    html += '</div>';
  }

  // Modality weights
  var weights = attn.modality_weights || {};
  var wEntries = Object.entries(weights);
  if (wEntries.length) {
    wEntries.forEach(function(w) {
      var wVal = w[1] || 0;
      html += '<div style="display:flex;align-items:center;gap:4px;padding:1px 0;">' +
        '<span style="min-width:50px;font-size:0.52rem;color:#6a6a80;">' + esc(w[0]) + '</span>' +
        _barFill(wVal, 1, '#0cf') +
        '<span style="min-width:28px;text-align:right;font-size:0.52rem;">' + (wVal * 100).toFixed(0) + '%</span></div>';
    });
  }

  // Conversation state
  if (convState.active || convState.topic) {
    html += '<div style="margin-top:6px;padding-top:4px;border-top:1px solid #1a1a2e;font-size:0.6rem;">';
    html += '<span style="color:#6a6a80;">Conversation:</span> ' +
      '<span style="color:' + (convState.active ? '#0f9' : '#6a6a80') + ';">' + (convState.active ? 'Active' : 'Inactive') + '</span>' +
      (convState.topic ? ' · <span style="color:#e0e0e8;">' + esc(convState.topic) + '</span>' : '') +
      (convState.turn_count ? ' · <span style="color:#6a6a80;">' + convState.turn_count + ' turns</span>' : '') +
      (convState.follow_up ? ' · <span style="color:#ff0;">follow-up</span>' : '') +
      '</div>';
  }

  html += '</div>'; // end right
  html += '</div>'; // end j-world-grid

  // ── Row 2: Scene Entities + Display Surfaces ──
  var entityList = Array.isArray(scene.entities) ? scene.entities : [];
  var entityCount = scene.entity_count != null ? scene.entity_count : entityList.length;
  var displaySurfaces = scene.display_surfaces || [];
  var displayContent = scene.display_content || [];
  var regionVis = scene.region_visibility || {};

  html += '<div class="j-panel-grid" style="margin-top:8px;">';

  // LEFT: Scene Entities
  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;">';
  html += '<div style="font-size:0.65rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">Scene Entities (' + entityCount + ')</div>';
  if (entityList.length) {
    html += '<div style="max-height:200px;overflow-y:auto;">';
    var visible = entityList.filter(function(e) { return e.state === 'visible'; });
    var other = entityList.filter(function(e) { return e.state !== 'visible'; });
    var sorted = visible.concat(other);
    sorted.slice(0, 20).forEach(function(e) {
      var stateColors = { visible: '#0f9', removed: '#f44', occluded: '#ff0', missing: '#f90', candidate: '#6a6a80' };
      var sc = stateColors[e.state] || '#484860';
      html += '<div style="display:flex;align-items:center;gap:4px;padding:2px 0;border-bottom:1px solid #0a0a14;font-size:0.58rem;">' +
        '<span style="width:6px;height:6px;border-radius:50%;background:' + sc + ';flex-shrink:0;"></span>' +
        '<span style="font-weight:600;min-width:60px;">' + esc(e.label || e.entity_id || '--') + '</span>' +
        '<span style="color:#6a6a80;">' + esc(e.region || '') + '</span>' +
        '<span style="color:#484860;margin-left:auto;">' + (e.confidence != null ? (e.confidence * 100).toFixed(0) + '%' : '') + '</span>' +
        (e.is_display_surface ? '<span style="color:#c0f;font-size:0.48rem;">DISP</span>' : '') +
        '</div>';
    });
    html += '</div>';
  }
  html += '</div>';

  // RIGHT: Displays + Regions
  html += '<div>';

  // Display Surfaces
  if (displaySurfaces.length) {
    html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;margin-bottom:6px;">';
    html += '<div style="font-size:0.65rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">Display Surfaces (' + displaySurfaces.length + ')</div>';
    displaySurfaces.forEach(function(ds) {
      var content = displayContent.filter(function(dc) { return dc.surface_id === ds.surface_id; })[0] || {};
      html += '<div style="padding:3px 0;border-bottom:1px solid #0a0a14;font-size:0.58rem;">' +
        '<div style="display:flex;gap:6px;align-items:center;">' +
        '<span style="font-weight:600;color:#c0f;">' + esc(ds.kind || 'display') + '</span>' +
        '<span style="color:#6a6a80;">' + (ds.confidence != null ? (ds.confidence * 100).toFixed(0) + '% conf' : '') + '</span>' +
        '<span style="color:#484860;">' + (ds.stable_for_s > 0 ? _rnd_ageStr(ds.stable_for_s) + ' stable' : '') + '</span>' +
        '</div>' +
        (content.activity_label ? '<div style="font-size:0.52rem;color:#0cf;margin-top:1px;">' +
          esc(content.activity_label) + ' (' + (content.activity_confidence != null ? (content.activity_confidence * 100).toFixed(0) + '% conf' : '') + ')' +
          (content.content_type ? ' · ' + esc(content.content_type) : '') + '</div>' : '') +
        '</div>';
    });
    html += '</div>';
  }

  // Region Visibility
  var regionEntries = Object.entries(regionVis);
  if (regionEntries.length) {
    html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;">';
    html += '<div style="font-size:0.65rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">Region Visibility</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(' + Math.min(regionEntries.length, 5) + ',1fr);gap:3px;">';
    regionEntries.forEach(function(rv) {
      var vis = rv[1] || 0;
      var vc = vis >= 0.8 ? '#0f9' : vis >= 0.4 ? '#ff0' : '#f44';
      html += '<div style="text-align:center;padding:2px;">' +
        '<div style="font-size:0.6rem;font-weight:600;color:' + vc + ';">' + (vis * 100).toFixed(0) + '%</div>' +
        '<div style="font-size:0.45rem;color:#6a6a80;">' + esc(rv[0].replace(/_/g, ' ')) + '</div></div>';
    });
    html += '</div>';
    html += '</div>';
  }

  html += '</div>'; // end right
  html += '</div>'; // end j-panel-grid

  // ── Row 3: World Model + Simulator + Staleness/Uncertainty ──
  html += '<div class="j-panel-grid-3" style="margin-top:8px;">';

  // World Model
  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;">';
  html += '<div style="font-size:0.65rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">World Model</div>';
  html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:3px;margin-bottom:4px;">' +
    _statCard('Level', promo.level_name || 'shadow', '#c0f') +
    _statCard('Accuracy', acc != null ? (acc * 100).toFixed(1) + '%' : '--', acc > 0.7 ? '#0f9' : acc > 0.4 ? '#ff0' : '#f44') +
    _statCard('Validated', causal.total_validated || 0, '#0cf') +
    _statCard('Misses', causal.total_misses || 0, (causal.total_misses || 0) > 0 ? '#f44' : '#0f9') +
    '</div>';

  // Causal rules
  var perRule = causal.per_rule || {};
  var rulesSorted = Object.entries(perRule).sort(function(a, b) { return (b[1].accuracy || 0) - (a[1].accuracy || 0); });
  if (rulesSorted.length) {
    html += '<div style="font-size:0.55rem;color:#6a6a80;margin-bottom:2px;">Causal Rules</div>';
    rulesSorted.slice(0, 6).forEach(function(e) {
      var name = e[0], info = e[1];
      var rAcc = info.accuracy || 0;
      var c = rAcc >= 0.7 ? '#0f9' : rAcc >= 0.3 ? '#ff0' : '#f44';
      html += '<div style="display:flex;align-items:center;gap:3px;padding:1px 0;">' +
        '<span style="min-width:80px;font-size:0.5rem;color:#6a6a80;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="' + esc(name) + '">' + esc(name.replace(/_/g, ' ').substring(0, 22)) + '</span>' +
        _barFill(rAcc, 1, c) +
        '<span style="min-width:28px;text-align:right;font-size:0.5rem;color:' + c + ';">' + (rAcc * 100).toFixed(0) + '%</span></div>';
    });
  }

  // Active predictions
  var predictions = wm.predictions || [];
  if (predictions.length) {
    html += '<div style="font-size:0.55rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Active Predictions (' + predictions.length + ')</div>';
    predictions.slice(0, 4).forEach(function(p) {
      html += '<div style="font-size:0.5rem;padding:1px 0;color:#e0e0e8;">' +
        esc((p.label || p.rule_id || '').replace(/_/g, ' ')) +
        ' <span style="color:#0cf;">' + (p.confidence != null ? (p.confidence * 100).toFixed(0) + '%' : '') + '</span></div>';
    });
  }
  html += '</div>';

  // Simulator
  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;">';
  html += '<div style="font-size:0.65rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">Mental Simulator</div>';
  html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:3px;margin-bottom:4px;">' +
    _statCard('Level', simPromo.level_name || 'shadow', '#c0f') +
    _statCard('Accuracy', simPromo.rolling_accuracy != null ? (simPromo.rolling_accuracy * 100).toFixed(1) + '%' : '--', '#6a6a80') +
    _statCard('Sims', simStats.total_simulations || 0, '#0cf') +
    _statCard('Avg Depth', simStats.avg_depth != null ? fmtNum(simStats.avg_depth, 1) : '--', '#6a6a80') +
    '</div>';
  html += '<div style="display:flex;gap:8px;font-size:0.5rem;color:#6a6a80;">' +
    '<span>Steps: ' + (simStats.total_steps || 0) + '</span>' +
    '<span>Avg ms: ' + (simStats.avg_elapsed_ms != null ? fmtNum(simStats.avg_elapsed_ms, 1) : '--') + '</span>' +
    '<span>Conf: ' + (simStats.avg_confidence != null ? fmtNum(simStats.avg_confidence, 2) : '--') + '</span>' +
    '</div>';

  var recentSims = wm.recent_simulations || [];
  if (recentSims.length) {
    html += '<div style="font-size:0.55rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Recent Simulations</div>';
    recentSims.slice(0, 4).forEach(function(s) {
      html += '<div style="font-size:0.5rem;padding:1px 0;">' +
        '<span style="color:#0cf;">' + esc((s.delta_event || '').replace(/_/g, ' ')) + '</span>' +
        ' <span style="color:#6a6a80;">depth:' + (s.depth || 0) + '</span>' +
        ' <span style="color:#484860;">conf:' + (s.total_confidence != null ? fmtNum(s.total_confidence, 2) : '--') + '</span>' +
        '</div>';
    });
  }

  var recentVal = wm.recent_validated || [];
  if (recentVal.length) {
    html += '<div style="font-size:0.55rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Validated (' + recentVal.length + ')</div>';
    recentVal.slice(0, 4).forEach(function(v) {
      var oc = v.outcome === 'hit' ? '#0f9' : '#f44';
      html += '<div style="font-size:0.5rem;padding:1px 0;">' +
        '<span style="color:' + oc + ';">' + esc(v.outcome || '--') + '</span> ' +
        '<span style="color:#6a6a80;">' + esc((v.label || v.rule_id || '').replace(/_/g, ' ')) + '</span></div>';
    });
  }
  html += '</div>';

  // Staleness + Uncertainty + Recent Deltas
  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;">';
  html += '<div style="font-size:0.65rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">State Health</div>';

  // Staleness
  var staleEntries = Object.entries(staleness);
  if (staleEntries.length) {
    html += '<div style="font-size:0.55rem;color:#6a6a80;margin-bottom:2px;">Staleness</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:2px;margin-bottom:4px;">';
    staleEntries.forEach(function(se) {
      var val = se[1] || 0;
      var c = val === 0 ? '#0f9' : val < 0.3 ? '#ff0' : '#f44';
      html += '<div style="display:flex;justify-content:space-between;padding:1px 3px;font-size:0.5rem;">' +
        '<span style="color:#6a6a80;">' + esc(se[0]) + '</span>' +
        '<span style="color:' + c + ';">' + fmtNum(val, 2) + '</span></div>';
    });
    html += '</div>';
  }

  // Uncertainty
  var uncEntries = Object.entries(uncertainty);
  if (uncEntries.length) {
    html += '<div style="font-size:0.55rem;color:#6a6a80;margin-bottom:2px;">Uncertainty</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:2px;margin-bottom:4px;">';
    uncEntries.forEach(function(ue) {
      var val = ue[1] || 0;
      var c = val < 0.2 ? '#0f9' : val < 0.5 ? '#ff0' : '#f44';
      html += '<div style="display:flex;justify-content:space-between;padding:1px 3px;font-size:0.5rem;">' +
        '<span style="color:#6a6a80;">' + esc(ue[0]) + '</span>' +
        '<span style="color:' + c + ';">' + fmtNum(val, 2) + '</span></div>';
    });
    html += '</div>';
  }

  // Recent world model deltas
  var wmDeltas = wm.recent_deltas || [];
  if (wmDeltas.length) {
    html += '<div style="font-size:0.55rem;color:#6a6a80;margin-bottom:2px;">Recent Deltas (' + wmDeltas.length + ')</div>';
    html += '<div style="max-height:120px;overflow-y:auto;">';
    wmDeltas.slice(0, 10).forEach(function(d) {
      html += '<div style="font-size:0.48rem;padding:1px 0;border-bottom:1px solid #0a0a14;">' +
        '<span style="color:#0cf;">' + esc((d.event || '').replace(/_/g, ' ')) + '</span>' +
        ' <span style="color:#6a6a80;">' + esc(d.facet || '') + '</span>' +
        (d.details ? ' <span style="color:#484860;">' + esc(JSON.stringify(d.details).substring(0, 50)) + '</span>' : '') +
        '</div>';
    });
    html += '</div>';
  }

  // Scene deltas
  var sceneDeltas = scene.recent_deltas || scene.recent_changes || scene.recent_events || [];
  if (sceneDeltas.length) {
    html += '<div style="font-size:0.55rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Scene Changes</div>';
    sceneDeltas.slice(0, 5).forEach(function(c) {
      var text = typeof c === 'object' ? (c.description || c.event || c.type || JSON.stringify(c).substring(0, 60)) : String(c);
      html += '<div style="font-size:0.48rem;padding:1px 0;color:#aaa;">' + esc(text.substring(0, 80)) + '</div>';
    });
  }

  html += '</div>'; // end state health
  html += '</div>'; // end 3-col row

  html += '</div>'; // end panel

  return html;
}


function _renderOpsPanel(snap) {
  var ops = snap.operations || {};
  var cur = ops.current || {};
  var bg = ops.background || {};
  var items = bg.items || [];
  var path = ops.interactive_path || {};
  var stages = path.stages || [];
  var subs = ops.subsystems || {};

  var heroHtml = '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">' +
    _statusBadge(cur.status || 'idle') +
    '<div>' +
    '<div style="font-weight:600;font-size:0.85rem;">' + esc(cur.label || cur.name || 'Idle') + '</div>' +
    (cur.detail ? '<div style="color:#6a6a80;font-size:0.68rem;">' + esc(cur.detail) + '</div>' : '') +
    '</div>' +
    (cur.duration_s > 0 ? '<span style="margin-left:auto;color:#6a6a80;font-size:0.65rem;">' + _rnd_ageStr(cur.duration_s) + '</span>' : '') +
    '</div>';

  var stackHtml = '';
  if (ops.stack && ops.stack.length) {
    stackHtml = '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">';
    ops.stack.forEach(function(s) {
      var val = typeof s === 'object' ? s.value : s;
      stackHtml += '<span style="padding:1px 5px;background:#181828;border:1px solid #2a2a44;border-radius:3px;font-size:0.6rem;">' + esc(val) + '</span>';
    });
    stackHtml += '</div>';
  }

  var pipeHtml = '';
  if (stages.length) {
    pipeHtml = '<div style="display:flex;align-items:center;gap:4px;margin-bottom:8px;overflow-x:auto;">';
    stages.forEach(function(s, i) {
      var colors = { active: '#0f9', done: '#0cf', waiting: '#ff0', idle: '#484860', error: '#f44' };
      var c = colors[s.status] || '#484860';
      pipeHtml += '<div style="text-align:center;">' +
        '<div style="width:8px;height:8px;border-radius:50%;background:' + c + ';margin:0 auto 2px;"></div>' +
        '<div style="font-size:0.55rem;color:' + c + ';">' + esc(s.label) + '</div></div>';
      if (i < stages.length - 1) pipeHtml += '<div style="width:12px;height:1px;background:' + c + '44;"></div>';
    });
    pipeHtml += '</div>';
  }

  var bgHtml = '';
  if (items.length) {
    var activeItems = items.filter(function(it) { return it.status === 'active' || it.status === 'waiting'; });
    var idleItems = items.filter(function(it) { return it.status !== 'active' && it.status !== 'waiting'; });
    bgHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:4px;">Background (' + (bg.active_count || 0) + ' active)</div>';
    bgHtml += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:4px;margin-bottom:6px;">';
    var allBg = activeItems.concat(idleItems);
    allBg.forEach(function(item) {
      var colors = { active: '#0f9', done: '#0cf', waiting: '#ff0', idle: '#484860', blocked: '#f44' };
      var c = colors[item.status] || '#484860';
      var border = item.status === 'active' ? c + '66' : item.status === 'waiting' ? c + '44' : '#1a1a2e';
      bgHtml += '<div style="padding:4px 6px;background:#0d0d1a;border:1px solid ' + border + ';border-radius:3px;">' +
        '<div style="display:flex;align-items:center;gap:4px;margin-bottom:1px;">' +
        '<span style="width:6px;height:6px;border-radius:50%;background:' + c + ';flex-shrink:0;"></span>' +
        '<span style="font-size:0.62rem;font-weight:600;color:#e0e0e8;">' + esc(item.label) + '</span>' +
        '<span style="margin-left:auto;font-size:0.5rem;color:#484860;">' + _rnd_ageStr(item.age_s) + '</span></div>' +
        (item.detail ? '<div style="font-size:0.5rem;color:#6a6a80;padding-left:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + esc(item.detail) + '</div>' : '') +
        '</div>';
    });
    bgHtml += '</div>';
  }

  var subsHtml = '';
  var subEntries = Object.entries(subs);
  if (subEntries.length) {
    var userFacing = subEntries.filter(function(e) { return e[1] && e[1].user_facing; });
    var internal = subEntries.filter(function(e) { return !e[1] || !e[1].user_facing; });

    if (userFacing.length) {
      subsHtml += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Voice Pipeline</div>';
      subsHtml += '<div style="display:grid;grid-template-columns:repeat(' + Math.min(userFacing.length, 7) + ',1fr);gap:3px;margin-bottom:6px;">';
      userFacing.forEach(function(e) {
        var name = e[0], info = e[1];
        var status = info.status || 'idle';
        var detail = info.detail || '';
        var label = info.label || name.replace(/_/g, ' ');
        var colors = { active: '#0f9', done: '#0cf', waiting: '#ff0', idle: '#484860', blocked: '#f44' };
        var c = colors[status] || '#484860';
        subsHtml += '<div style="padding:4px;background:#0d0d1a;border:1px solid ' + (status !== 'idle' ? c + '44' : '#1a1a2e') + ';border-radius:3px;text-align:center;">' +
          '<div style="font-size:0.55rem;font-weight:600;color:' + c + ';">' + esc(label) + '</div>' +
          _statusBadge(status) +
          (detail ? '<div style="font-size:0.45rem;color:#484860;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="' + esc(detail) + '">' + esc(detail.substring(0, 30)) + '</div>' : '') +
          '</div>';
      });
      subsHtml += '</div>';
    }

    if (internal.length) {
      subsHtml += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Internal Systems</div>';
      subsHtml += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:3px;">';
      internal.forEach(function(e) {
        var name = e[0], info = e[1] || {};
        var status = info.status || 'idle';
        var detail = info.detail || '';
        var label = info.label || name.replace(/_/g, ' ');
        var colors = { active: '#0f9', done: '#0cf', waiting: '#ff0', idle: '#484860', blocked: '#f44' };
        var c = colors[status] || '#484860';
        subsHtml += '<div style="padding:3px 5px;background:#0d0d1a;border:1px solid ' + (status !== 'idle' ? c + '44' : '#1a1a2e') + ';border-radius:3px;">' +
          '<div style="display:flex;justify-content:space-between;align-items:center;">' +
          '<span style="font-size:0.55rem;color:#e0e0e8;">' + esc(label) + '</span>' +
          _statusBadge(status) + '</div>' +
          (detail ? '<div style="font-size:0.45rem;color:#484860;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="' + esc(detail) + '">' + esc(detail.substring(0, 50)) + '</div>' : '') +
          '</div>';
      });
      subsHtml += '</div>';
    }
  }

  return _panel('Operations', heroHtml + stackHtml + pipeHtml + bgHtml + subsHtml, _statusBadge(cur.status || 'idle'));
}


function _renderAutonomyPanel(snap) {
  var auto = snap.autonomy || {};
  if (!auto.enabled && !auto.started) return _panel('Autonomy', _emptyMsg('Autonomy disabled'));

  var level = auto.autonomy_level != null ? auto.autonomy_level : (auto.level != null ? auto.level : '--');
  var levelNames = { 0: 'L0 propose', 1: 'L1 research', 2: 'L2 safe-apply', 3: 'L3 full' };
  var levelLabel = auto.autonomy_level_name ? auto.autonomy_level_name.replace(/_/g, ' ') : (levelNames[level] || ('L' + level));
  var promo = auto.promotion || {};

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Level', 'L' + level + ' ' + levelLabel, '#c0f') +
    _statCard('Queue', auto.queue_size || 0, '#0cf') +
    _statCard('Win Rate', promo.win_rate != null ? (promo.win_rate * 100).toFixed(0) + '%' : '--', (promo.win_rate || 0) > 0.4 ? '#0f9' : '#f90') +
    _statCard('Completed', auto.completed_total_session || auto.completed_total || 0, '#6a6a80') +
    '</div>';

  var recentHtml = '';
  var recent = auto.completed || auto.recent_research || [];
  if (recent.length) {
    recentHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">Recent Research (' + recent.length + ')</div>';
    recent.slice(0, 6).forEach(function(r, idx) {
      var question = r.question || r.topic || '--';
      var sc = r.status === 'completed' ? '#0f9' : r.status === 'failed' ? '#f44' : '#ff0';
      var hint = r.source_hint || r.tool || '';
      recentHtml += '<div style="padding:3px 0;border-bottom:1px solid #1a1a2e;cursor:pointer;" onclick="window.openResearchDetail && window.openResearchDetail(' + idx + ')">' +
        '<div style="display:flex;gap:6px;align-items:center;">' +
        '<span style="width:6px;height:6px;border-radius:50%;background:' + sc + ';flex-shrink:0;"></span>' +
        '<span style="font-size:0.62rem;flex:1;">' + esc(question.substring(0, 100)) + '</span>' +
        (hint ? '<span style="font-size:0.48rem;color:#0cf;padding:0 3px;border:1px solid #0cf33;border-radius:2px;">' + esc(hint) + '</span>' : '') +
        '<span style="color:#0cf;font-size:0.55rem;">\u25B6</span>' +
        '</div></div>';
    });
  }

  var deltasHtml = '';
  var deltas = auto.recent_deltas || [];
  if (deltas.length) {
    deltasHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Recent Deltas (' + deltas.length + ')</div>';
    deltas.slice(0, 6).forEach(function(d, idx) {
      var net = d.net_improvement;
      var netC = net > 0.01 ? '#0f9' : net < -0.01 ? '#f44' : '#6a6a80';
      var changed = d.deltas || {};
      var nonZero = Object.entries(changed).filter(function(e) { return Math.abs(e[1]) > 0.0001; });
      var topMetrics = nonZero.sort(function(a, b) { return Math.abs(b[1]) - Math.abs(a[1]); }).slice(0, 3);
      var summary = topMetrics.map(function(e) { return e[0].replace(/_/g, ' ') + ':' + (e[1] > 0 ? '+' : '') + fmtNum(e[1], 3); }).join(', ');
      deltasHtml += '<div style="display:flex;gap:6px;padding:2px 0;font-size:0.58rem;cursor:pointer;border-bottom:1px solid #0d0d1a;" onclick="window.openDeltaDetail && window.openDeltaDetail(' + idx + ')">' +
        '<span style="min-width:40px;text-align:right;color:' + netC + ';font-weight:600;">' + (net != null ? (net > 0 ? '+' : '') + fmtNum(net, 4) : '--') + '</span>' +
        '<span style="flex:1;color:#6a6a80;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(summary || 'no significant changes') + '</span>' +
        '<span style="color:#0cf;font-size:0.55rem;">\u25B6</span></div>';
    });
  }

  // Recent Learnings
  var learnings = auto.recent_learnings || [];
  var learnHtml = '';
  if (learnings.length) {
    learnHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Recent Learnings (' + learnings.length + ')</div>';
    learnings.slice(0, 5).forEach(function(l) {
      var toolC = l.tool === 'academic_search' ? '#c0f' : l.tool === 'web_search' ? '#0cf' : '#6a6a80';
      learnHtml += '<div style="padding:2px 0;border-bottom:1px solid #0d0d1a;font-size:0.55rem;">' +
        '<span style="color:' + toolC + ';">[' + esc(l.tool || '--') + ']</span> ' +
        '<span style="color:#8a8aa0;">' + esc((l.question || '').substring(0, 80)) + '</span>' +
        ' <span style="color:#0f9;">' + (l.findings || 0) + ' findings</span>' +
        ' <span style="color:#c0f;">' + (l.memories_created || 0) + ' memories</span>' +
        '</div>';
    });
  }

  // Governor
  var gov = auto.governor || {};
  var govHtml = '';
  if (gov.hourly_limit || gov.total_allowed) {
    govHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Governor</div>' +
      '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:4px;">' +
      _statCard('Hourly', (gov.hourly_used || 0) + '/' + (gov.hourly_limit || 0), '#0cf') +
      _statCard('Daily', (gov.daily_used || 0) + '/' + (gov.daily_limit || 0), '#0cf') +
      _statCard('Allowed', gov.total_allowed || 0, '#0f9') +
      _statCard('Blocked', gov.total_blocked || 0, (gov.total_blocked || 0) > 0 ? '#f44' : '#0f9') +
      '</div>';
    if (gov.web_hourly_used || gov.academic_hourly_used) {
      govHtml += '<div style="display:flex;gap:8px;font-size:0.5rem;color:#484860;margin-bottom:4px;">' +
        '<span>web: ' + (gov.web_hourly_used || 0) + '/' + (gov.web_hourly_limit || 3) + '/h</span>' +
        '<span>academic: ' + (gov.academic_hourly_used || 0) + '/' + (gov.academic_hourly_limit || 10) + '/h</span>' +
        '<span>topics: ' + (gov.active_topics || 0) + '</span>' +
        '<span>running: ' + (gov.running_count || 0) + '</span>' +
        '</div>';
    }
  }

  // Delta tracker
  var dt = auto.delta_tracker || {};
  var dtHtml = '';
  if (dt.total_measured) {
    dtHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Delta Tracker</div>' +
      '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:4px;">' +
      _statCard('Measured', dt.total_measured || 0) +
      _statCard('Improved', dt.total_improved || 0, '#0f9') +
      _statCard('Regressed', dt.total_regressed || 0, (dt.total_regressed || 0) > 0 ? '#f44' : '#0f9') +
      _statCard('Rate', dt.improvement_rate != null ? (dt.improvement_rate * 100).toFixed(0) + '%' : '--', (dt.improvement_rate || 0) > 0.5 ? '#0f9' : '#ff0') +
      '</div>';
  }

  // Metric triggers
  var mt = auto.metric_triggers || {};
  var mtHtml = '';
  if (mt.total_triggers || mt.active_deficits) {
    var activeDeficits = mt.active_deficits || {};
    mtHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Metric Triggers</div>' +
      '<div style="display:flex;gap:6px;font-size:0.5rem;margin-bottom:4px;">' +
      '<span style="color:#0cf;">triggers: ' + (mt.total_triggers || 0) + '</span>' +
      '<span style="color:#f90;">vetoed: ' + (mt.total_vetoed || 0) + '</span>' +
      '<span style="color:#6a6a80;">rotated: ' + (mt.total_rotated || 0) + '</span>' +
      '</div>';
    if (Object.keys(activeDeficits).length) {
      mtHtml += '<div style="display:flex;flex-wrap:wrap;gap:3px;">';
      Object.entries(activeDeficits).forEach(function(e) {
        mtHtml += '<span style="padding:1px 4px;border:1px solid #f9044;color:#f90;border-radius:2px;font-size:0.48rem;">' + esc(e[0].replace(/_/g, ' ')) + '</span>';
      });
      mtHtml += '</div>';
    }
  }

  // Integrator
  var intg = auto.integrator || {};
  var intHtml = '';
  if (intg.total_memories_created || intg.total_evidence_fed) {
    intHtml = '<div style="display:flex;gap:8px;font-size:0.5rem;color:#6a6a80;margin-top:4px;">' +
      '<span>memories: ' + (intg.total_memories_created || 0) + '</span>' +
      '<span>evidence: ' + (intg.total_evidence_fed || 0) + '</span>' +
      '<span>known: ' + (intg.total_skipped_known || 0) + '</span>' +
      '<span>conflicts: ' + (intg.total_conflicts_detected || 0) + '</span>' +
      '</div>';
  }

  // Policy memory
  var pmHtml = '';
  var pm = auto.policy_memory || {};
  if (pm.total_outcomes) {
    pmHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Policy Memory</div>' +
      '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:4px;">' +
      _statCard('Outcomes', pm.total_outcomes || 0) +
      _statCard('Wins', pm.total_wins || 0, '#0f9') +
      _statCard('Losses', pm.total_losses || 0, (pm.total_losses || 0) > 0 ? '#f44' : '#0f9') +
      _statCard('Win Rate', pm.overall_win_rate != null ? (pm.overall_win_rate * 100).toFixed(0) + '%' : '--', (pm.overall_win_rate || 0) > 0.5 ? '#0f9' : '#ff0') +
      '</div>';
    if (pm.in_warmup) {
      pmHtml += '<div style="font-size:0.5rem;color:#ff0;">In warmup · remaining: ' + fmtNum(pm.warmup_remaining_s || 0, 0) + 's</div>';
    }
  }

  // Episode recorder + bridge
  var ep = auto.episode_recorder || {};
  var br = auto.bridge || {};
  var miscHtml = '';
  if (ep.total_episodes || br.events_processed) {
    miscHtml = '<div style="display:flex;gap:8px;font-size:0.5rem;color:#484860;margin-top:4px;">' +
      (ep.total_episodes ? '<span>episodes: ' + ep.total_episodes + ' (success: ' + (ep.successful_episodes || 0) + ')</span>' : '') +
      (br.events_processed ? '<span>bridge: ' + br.events_processed + ' events \u2192 ' + (br.intents_generated || 0) + ' intents</span>' : '') +
      '</div>';
  }

  return _panel('Autonomy', statsHtml + recentHtml + deltasHtml + learnHtml + govHtml + dtHtml + mtHtml + intHtml + pmHtml + miscHtml,
    _statusBadge(auto.enabled ? (auto.started ? 'active' : 'starting') : 'disabled'));
}


function _renderDrivesPanel(snap) {
  var auto = snap.autonomy || {};
  var drivesObj = (auto.drives || {}).drives || auto.drives || {};
  var driveEntries = [];

  if (Array.isArray(drivesObj)) {
    drivesObj.forEach(function(d) { driveEntries.push([d.name || d.drive || '?', d]); });
  } else if (typeof drivesObj === 'object') {
    driveEntries = Object.entries(drivesObj);
  }
  if (!driveEntries.length) return _panel('Motive Drives', _emptyMsg('No drive data'));

  var body = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px;">';
  var totalActions = 0;
  var activeCount = 0;
  driveEntries.forEach(function(e) {
    var drv = e[1] || {};
    totalActions += drv.action_count || 0;
    if ((drv.urgency || 0) > 0) activeCount++;
  });
  body += _statCard('Drives', driveEntries.length, '#0cf');
  body += _statCard('Active', activeCount, activeCount > 0 ? '#0f9' : '#6a6a80');
  body += _statCard('Total Actions', totalActions, '#6a6a80');
  body += _statCard('Suppressed', driveEntries.filter(function(e) { return !!(e[1] || {}).suppression; }).length, '#f90');
  body += '</div>';

  driveEntries.forEach(function(entry) {
    var name = entry[0];
    var drv = entry[1] || {};
    var urgency = drv.urgency || 0;
    var c = urgency > 0.5 ? '#f44' : urgency > 0.2 ? '#ff0' : urgency > 0 ? '#0f9' : '#484860';
    var outcomeC = drv.last_outcome === 'positive' ? '#0f9' : drv.last_outcome === 'negative' ? '#f44' : '#6a6a80';
    var suppressed = drv.suppression ? true : false;

    body += '<div style="display:flex;align-items:center;gap:6px;padding:3px 0;' + (suppressed ? 'opacity:0.5;' : '') + '">' +
      '<span style="min-width:70px;font-size:0.65rem;font-weight:600;color:#e0e0e8;">' + esc(name) + '</span>' +
      _barFill(urgency, 1, c) +
      '<span style="min-width:30px;text-align:right;font-size:0.6rem;color:' + c + ';">' + fmtNum(urgency, 2) + '</span>' +
      '<span style="min-width:50px;font-size:0.5rem;color:#6a6a80;">' + esc(drv.strategy || '') + '</span>' +
      '<span style="font-size:0.5rem;color:' + outcomeC + ';">' + (drv.last_outcome || 'none') + '</span>' +
      '<span style="font-size:0.5rem;color:#484860;">' + (drv.action_count || 0) + ' acts</span>' +
      '</div>';
    if (suppressed) {
      body += '<div style="padding-left:76px;font-size:0.48rem;color:#f90;margin-top:-2px;">\u26a0 ' + esc(drv.suppression) + '</div>';
    }
    if (drv.tools && drv.tools.length) {
      body += '<div style="padding-left:76px;font-size:0.45rem;color:#484860;">tools: ' + drv.tools.join(', ') + '</div>';
    }
  });

  return _panel('Motive Drives', body);
}


function _renderGoalsPanel(snap) {
  var goals = snap.goals || {};
  var stats = goals.stats || {};
  var active = goals.active_goals || [];
  var focus = goals.current_focus || null;
  var log = goals.promotion_log || [];
  var byStatus = stats.by_status || {};

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Total', stats.total || 0) +
    _statCard('Active', active.length, '#0f9') +
    _statCard('Completed', byStatus.completed || 0, '#0cf') +
    _statCard('Stalled', byStatus.stalled || 0, byStatus.stalled > 0 ? '#f90' : '#6a6a80') +
    '</div>';

  var focusHtml = '';
  if (focus) {
    var g = typeof focus === 'object' ? focus : {};
    focusHtml = '<div style="padding:6px 8px;background:#0d0d1a;border:1px solid #2a2a44;border-radius:4px;margin-bottom:8px;">' +
      '<div style="font-size:0.68rem;color:#6a6a80;">Current Focus</div>' +
      '<div style="font-size:0.78rem;font-weight:600;color:#0cf;">' + esc(g.title || g.description || g.goal_id || String(focus)) + '</div>' +
      (g.progress != null ? '<div style="margin-top:3px;">' + _barFill(g.progress, 1, '#0cf') + '</div>' : '') +
      '</div>';
  }

  var activeHtml = '';
  if (active.length) {
    activeHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">Active Goals</div>';
    active.slice(0, 8).forEach(function(g) {
      var title = g.title || g.description || g.goal_id || '--';
      var gid = g.goal_id || g.id || '';
      var status = g.status || 'active';
      var progress = g.progress || g.score || 0;
      activeHtml += '<div style="padding:3px 0;border-bottom:1px solid #1a1a2e;display:flex;align-items:center;gap:6px;">' +
        _statusBadge(status) +
        '<span style="flex:1;font-size:0.68rem;cursor:pointer;text-decoration:underline;text-decoration-color:#2a2a44;" onclick="window.openGoalDetail && window.openGoalDetail(\'' + esc(gid) + '\')">' + esc(title.substring(0, 60)) + '</span>' +
        '<span style="font-size:0.6rem;color:#6a6a80;">' + (progress * 100).toFixed(0) + '%</span>';
      if (gid && window.goalAction) {
        if (status === 'active' || status === 'promoted') {
          activeHtml += '<button class="j-btn-xs" onclick="window.goalAction(\'' + esc(gid) + '\',\'complete\')" title="Complete">&#10003;</button>' +
            '<button class="j-btn-xs" onclick="window.goalAction(\'' + esc(gid) + '\',\'pause\')" title="Pause">&#10074;&#10074;</button>' +
            '<button class="j-btn-xs j-btn-red" onclick="window.goalAction(\'' + esc(gid) + '\',\'abandon\')" title="Abandon">&times;</button>';
        } else if (status === 'paused') {
          activeHtml += '<button class="j-btn-xs" onclick="window.goalAction(\'' + esc(gid) + '\',\'resume\')" title="Resume">&#9654;</button>';
        }
      }
      activeHtml += '</div>';
    });
  }

  var logHtml = '';
  if (log.length) {
    logHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Promotion Log</div>';
    log.slice(-5).reverse().forEach(function(entry) {
      var text = typeof entry === 'object' ? (entry.message || entry.text || JSON.stringify(entry).substring(0, 80)) : String(entry);
      logHtml += '<div style="font-size:0.6rem;color:#484860;padding:1px 0;">' + esc(text.substring(0, 120)) + '</div>';
    });
  }

  // Completed recent
  var completedHtml = '';
  var completed = goals.completed_recent || [];
  if (completed.length) {
    completedHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Recently Completed</div>';
    completed.slice(0, 3).forEach(function(g) {
      completedHtml += '<div style="font-size:0.55rem;padding:1px 0;color:#0f9;">\u2713 ' + esc((g.title || g.goal_id || '--').substring(0, 60)) + '</div>';
    });
  }

  // Dispatch / execution status
  var dispatchHtml = '';
  if (goals.focus_reason || goals.why_not_executing) {
    dispatchHtml = '<div style="margin-top:4px;padding:3px 6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;">';
    if (goals.focus_reason) dispatchHtml += '<div style="font-size:0.5rem;color:#6a6a80;">Focus: ' + esc(goals.focus_reason) + '</div>';
    if (goals.why_not_executing) dispatchHtml += '<div style="font-size:0.5rem;color:#f90;">Not executing: ' + esc(goals.why_not_executing) + '</div>';
    if (goals.dispatch_block_reason) dispatchHtml += '<div style="font-size:0.5rem;color:#484860;">Block: ' + esc(goals.dispatch_block_reason) + '</div>';
    dispatchHtml += '</div>';
  }

  // Producer health
  var ph = goals.producer_health || {};
  var phHtml = '';
  if (Object.keys(ph).length) {
    phHtml = '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;font-size:0.48rem;color:#484860;">';
    Object.entries(ph).forEach(function(e) {
      if (e[1] > 0) phHtml += '<span>' + esc(e[0].replace(/_/g, ' ')) + ': ' + e[1] + '</span>';
    });
    phHtml += '</div>';
  }

  // By kind / by horizon
  var byKind = stats.by_kind || {};
  var byHorizon = stats.by_horizon || {};
  var breakdownHtml = '';
  if (Object.keys(byKind).length || Object.keys(byHorizon).length) {
    breakdownHtml = '<div style="display:flex;gap:10px;margin-top:4px;">';
    if (Object.keys(byKind).length) {
      breakdownHtml += '<div style="display:flex;flex-wrap:wrap;gap:2px;">';
      Object.entries(byKind).forEach(function(e) {
        breakdownHtml += '<span style="padding:1px 3px;border:1px solid #0cf22;color:#0cf;border-radius:2px;font-size:0.45rem;">' + esc(e[0]) + ':' + e[1] + '</span>';
      });
      breakdownHtml += '</div>';
    }
    if (Object.keys(byHorizon).length) {
      breakdownHtml += '<div style="display:flex;flex-wrap:wrap;gap:2px;">';
      Object.entries(byHorizon).forEach(function(e) {
        breakdownHtml += '<span style="padding:1px 3px;border:1px solid #c0f22;color:#c0f;border-radius:2px;font-size:0.45rem;">' + esc(e[0]) + ':' + e[1] + '</span>';
      });
      breakdownHtml += '</div>';
    }
    breakdownHtml += '</div>';
  }

  var actionsHtml = '<div style="margin-top:4px;display:flex;gap:6px;">' +
    (window.openGoalObserve ? '<button class="j-btn-sm" onclick="window.openGoalObserve()">Submit Observation</button>' : '') +
    '</div>';

  return _panel('Goals', statsHtml + focusHtml + activeHtml + completedHtml + dispatchHtml + breakdownHtml + phHtml + logHtml + actionsHtml,
    '<span style="font-size:0.6rem;color:#6a6a80;">' + (stats.total || 0) + ' total</span>');
}


function _renderWorldModelPanel(snap) {
  var wm = snap.world_model || {};
  if (!Object.keys(wm).length) return _panel('World Model', _emptyMsg('Not active'));

  var promo = wm.promotion || {};
  var causal = wm.causal || {};
  var acc = causal.overall_accuracy;

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Level', promo.level_name || 'shadow', '#c0f') +
    _statCard('Accuracy', acc != null ? (acc * 100).toFixed(1) + '%' : '--', acc > 0.7 ? '#0f9' : acc > 0.4 ? '#ff0' : '#f44') +
    _statCard('Validated', causal.total_validated || 0, '#0cf') +
    _statCard('Misses', causal.total_misses || 0, (causal.total_misses || 0) > 0 ? '#f44' : '#0f9') +
    '</div>';

  var promoHtml = '';
  if (promo.level_name && promo.level_name !== 'active') {
    promoHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">Promotion Progress</div>' +
      _metricRow('Ready', promo.promotion_ready ? 'YES' : 'No') +
      _metricRow('Hours in shadow', promo.hours_in_shadow != null ? promo.hours_in_shadow + 'h / 24h' : '--') +
      _metricRow('Validated', (causal.total_validated || 0) + ' / 50');
  }

  var rulesHtml = '';
  var perRule = causal.per_rule || {};
  var sorted = Object.entries(perRule).sort(function(a, b) { return (b[1].accuracy || 0) - (a[1].accuracy || 0); });
  if (sorted.length) {
    rulesHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Causal Rules</div>';
    sorted.slice(0, 8).forEach(function(e) {
      var name = e[0], info = e[1];
      var rAcc = info.accuracy || 0;
      var c = rAcc >= 0.7 ? '#0f9' : rAcc >= 0.3 ? '#ff0' : '#f44';
      rulesHtml += '<div style="display:flex;align-items:center;gap:6px;padding:1px 0;">' +
        '<span style="min-width:120px;font-size:0.6rem;color:#6a6a80;">' + esc(name.replace(/_/g, ' ')) + '</span>' +
        _barFill(rAcc, 1, c) +
        '<span style="min-width:50px;text-align:right;font-size:0.6rem;color:' + c + ';">' + (rAcc * 100).toFixed(0) + '% (' + (info.total || 0) + ')</span></div>';
    });
  }

  return _panel('World Model', statsHtml + promoHtml + rulesHtml, _statusBadge(promo.level_name || 'shadow'));
}


function _renderSimulatorPanel(snap) {
  var wm = snap.world_model || {};
  var sim = wm.simulator || {};
  var promo = wm.simulator_promotion || {};
  var traces = wm.recent_simulations || [];
  var validated = wm.recent_validated || [];

  if (!sim.total_simulations && !traces.length) return _panel('Simulator', _emptyMsg('No simulations yet'));

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Total Sims', sim.total_simulations || 0) +
    _statCard('Steps', sim.total_steps || 0) +
    _statCard('Avg Depth', sim.avg_depth != null ? fmtNum(sim.avg_depth, 1) : '--') +
    _statCard('Avg Conf', sim.avg_confidence != null ? (sim.avg_confidence * 100).toFixed(1) + '%' : '--', sim.avg_confidence > 0.5 ? '#0f9' : '#ff0') +
    '</div>';

  // Promotion status
  var promoHtml = '';
  if (promo.level_name || promo.level != null) {
    var promoAcc = promo.rolling_accuracy != null ? promo.rolling_accuracy : promo.accuracy;
    promoHtml = '<div style="display:flex;flex-wrap:wrap;gap:8px;font-size:0.55rem;margin-bottom:6px;">' +
      '<span style="color:#c0f;">Level: <b>' + esc(promo.level_name || 'shadow') + '</b></span>' +
      '<span style="color:#6a6a80;">Accuracy: ' + (promoAcc != null ? (promoAcc * 100).toFixed(1) + '%' : '--') + '</span>' +
      '<span style="color:#6a6a80;">Validated: ' + (promo.total_validated || 0) + '</span>' +
      '<span style="color:#6a6a80;">Hours: ' + (promo.hours_in_shadow != null ? fmtNum(promo.hours_in_shadow, 1) : '--') + 'h</span>' +
      (promo.promotion_ready ? '<span style="color:#0f9;">READY TO PROMOTE</span>' : '') +
      '</div>';
  }

  // Avg elapsed
  if (sim.avg_elapsed_ms != null) {
    statsHtml += '<div style="font-size:0.5rem;color:#484860;margin-bottom:4px;">avg elapsed: ' + fmtNum(sim.avg_elapsed_ms, 1) + 'ms · buffer: ' + (sim.trace_buffer_size || 0) + '</div>';
  }

  // Recent simulations
  var traceHtml = '';
  if (traces.length) {
    traceHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Recent Simulations</div>';
    traces.slice(0, 6).forEach(function(t) {
      var event = t.delta_event || t.summary || t.trigger || '';
      var facet = t.delta_facet || '';
      var depth = t.depth || (Array.isArray(t.steps) ? t.steps.length : 0);
      var conf = t.total_confidence || t.confidence || 0;
      var elapsed = t.elapsed_ms;
      traceHtml += '<div style="padding:2px 0;border-bottom:1px solid #1a1a2e;display:flex;gap:6px;align-items:center;">' +
        '<span style="font-size:0.5rem;color:#6a6a80;min-width:16px;">d' + depth + '</span>' +
        '<span style="flex:1;font-size:0.55rem;">' + esc(event.replace(/_/g, ' ')) +
        (facet ? ' <span style="color:#484860;">[' + esc(facet) + ']</span>' : '') + '</span>' +
        '<span style="font-size:0.5rem;color:' + (conf > 0.5 ? '#0f9' : '#ff0') + ';">' + (conf * 100).toFixed(0) + '%</span>' +
        (elapsed != null ? '<span style="font-size:0.42rem;color:#484860;">' + fmtNum(elapsed, 1) + 'ms</span>' : '') +
        '</div>';
    });
  }

  // Validated predictions
  var valHtml = '';
  if (validated.length) {
    valHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Validated Predictions (' + validated.length + ')</div>';
    validated.slice(0, 6).forEach(function(v) {
      var isHit = v.outcome === 'hit' || v.correct === true;
      var c = isHit ? '#0f9' : '#f44';
      var ruleId = v.rule_id || v.rule || '--';
      var label = v.label || v.description || v.summary || '';
      var conf = v.confidence;
      valHtml += '<div style="font-size:0.5rem;padding:1px 0;border-bottom:1px solid #0d0d1a;">' +
        '<span style="color:' + c + ';">' + (isHit ? '\u2713' : '\u2717') + '</span> ' +
        '<span style="color:#0cf;">' + esc(ruleId.replace(/_/g, ' ')) + '</span> ' +
        '<span style="color:#6a6a80;">' + esc(label.replace(/_/g, ' ').substring(0, 40)) + '</span>' +
        (conf != null ? ' <span style="color:#484860;">' + (conf * 100).toFixed(0) + '%</span>' : '') +
        '</div>';
    });
  }

  return _panel('Simulator', statsHtml + promoHtml + traceHtml + valHtml, _statusBadge(promo.level_name || promo.level || 'shadow'));
}


function _renderScenePanel(snap) {
  var scene = snap.scene || {};
  if (!Object.keys(scene).length) return _panel('Scene', _emptyMsg('No scene data'));

  var entityCount = scene.entity_count != null ? scene.entity_count : (Array.isArray(scene.entities) ? scene.entities.length : 0);
  var entityList = Array.isArray(scene.entities) ? scene.entities : [];
  var persons = scene.visible_persons != null ? scene.visible_persons : '--';
  var regions = scene.regions || [];
  var changes = scene.recent_changes || scene.recent_events || [];

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Entities', entityCount) +
    _statCard('Persons', persons, persons > 0 ? '#0f9' : '#6a6a80') +
    _statCard('Regions', Array.isArray(regions) ? regions.length : (regions || 0)) +
    '</div>';

  var regionHtml = '';
  if (Array.isArray(regions) && regions.length) {
    regionHtml = '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">';
    regions.forEach(function(r) {
      var name = typeof r === 'object' ? (r.name || r.id || '') : String(r);
      regionHtml += '<span style="padding:1px 5px;background:#181828;border:1px solid #2a2a44;border-radius:3px;font-size:0.6rem;">' + esc(name) + '</span>';
    });
    regionHtml += '</div>';
  }

  var entityHtml = '';
  if (entityList.length) {
    entityHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">Entities</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">';
    entityList.slice(0, 20).forEach(function(e) {
      entityHtml += '<span style="padding:1px 5px;background:#181828;border:1px solid #2a2a44;border-radius:3px;font-size:0.6rem;">' +
        esc(e.label || e.class_name || e.entity_id || 'entity') + '</span>';
    });
    entityHtml += '</div>';
  }

  var changesHtml = '';
  if (changes.length) {
    changesHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">Recent Changes</div>';
    changes.slice(0, 5).forEach(function(c) {
      var text = typeof c === 'object' ? (c.description || c.event || c.type || JSON.stringify(c).substring(0, 80)) : String(c);
      var ts = typeof c === 'object' && c.timestamp ? timeAgo(c.timestamp) : '';
      changesHtml += '<div style="font-size:0.6rem;padding:1px 0;display:flex;gap:6px;">' +
        '<span style="flex:1;">' + esc(text.substring(0, 100)) + '</span>' +
        (ts ? '<span style="color:#484860;">' + ts + '</span>' : '') + '</div>';
    });
  }

  return _panel('Scene', statsHtml + regionHtml + entityHtml + changesHtml);
}


function _renderAttentionPanel(snap) {
  var attn = snap.attention || {};
  if (!Object.keys(attn).length) return _panel('Attention', _emptyMsg('No attention data'));

  var engagement = attn.engagement_level;
  var focusTarget = attn.focus_target || attn.primary_target || '--';
  var weights = attn.modality_weights || {};

  var body = '';
  body += _metricRow('Engagement', engagement != null ? (engagement * 100).toFixed(0) + '%' : '--');
  body += _metricRow('Focus Target', focusTarget);

  if (Object.keys(weights).length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Modality Weights</div>';
    Object.entries(weights).forEach(function(e) {
      var name = e[0], val = e[1];
      body += '<div style="display:flex;align-items:center;gap:6px;padding:1px 0;">' +
        '<span style="min-width:70px;font-size:0.65rem;color:#6a6a80;">' + esc(name) + '</span>' +
        _barFill(val || 0, 1, '#0cf') +
        '<span style="min-width:36px;text-align:right;font-size:0.6rem;">' + fmtNum(val, 2) + '</span></div>';
    });
  }

  return _panel('Attention', body);
}


/* ── Flight Recorder — conversation history with drill-down ──────────── */

function _renderFlightRecorderPanel(snap) {
  var episodes = snap.episodes || [];
  var gate = snap.capability_gate || {};

  if (!episodes.length) {
    return _panel('Flight Recorder', _emptyMsg('No conversations recorded yet.'),
      '<span style="font-size:0.6rem;color:#6a6a80;">0 episodes</span>');
  }

  var sorted = episodes.slice().sort(function(a, b) { return (b.timestamp || 0) - (a.timestamp || 0); });

  var routeColors = {
    NONE: '#6a6a80', STATUS: '#0cf', INTROSPECTION: '#c0f', SKILL: '#f90',
    MEMORY: '#0f9', VISION: '#ff0', TIME: '#0cf', SYSTEM_STATUS: '#0cf',
    IDENTITY: '#f0f', PERFORM: '#ff0', SELF_IMPROVE: '#f44',
    WEB_SEARCH: '#0cf', ACADEMIC_SEARCH: '#0cf', CAMERA_CONTROL: '#ff0',
    LIBRARY_INGEST: '#f90'
  };

  var body = '';

  // Summary stats
  var routeCounts = {};
  sorted.forEach(function(ep) {
    var r = ep.tool_route || 'NONE';
    routeCounts[r] = (routeCounts[r] || 0) + 1;
  });
  body += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">';
  Object.entries(routeCounts).sort(function(a, b) { return b[1] - a[1]; }).forEach(function(e) {
    var rc = routeColors[e[0]] || '#6a6a80';
    body += '<span style="padding:1px 6px;border:1px solid ' + rc + '33;color:' + rc + ';border-radius:3px;font-size:0.6rem;">' +
      esc(e[0]) + ' <b>' + e[1] + '</b></span>';
  });
  if (gate.narration_rewrites > 0) {
    body += '<span style="padding:1px 6px;border:1px solid #f4433;color:#f44;border-radius:3px;font-size:0.6rem;" title="Confabulation narrations blocked on NONE route">' +
      'narration blocks <b>' + gate.narration_rewrites + '</b></span>';
  }
  if (gate.claims_blocked > 0) {
    body += '<span style="padding:1px 6px;border:1px solid #f9033;color:#f90;border-radius:3px;font-size:0.6rem;">' +
      'claims blocked <b>' + gate.claims_blocked + '</b></span>';
  }
  body += '</div>';

  // Episode list
  body += '<div style="max-height:400px;overflow-y:auto;">';
  sorted.forEach(function(ep, idx) {
    var ts = ep.timestamp ? new Date(ep.timestamp * 1000) : null;
    var timeStr = ts ? ts.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) : '--';
    var dateStr = ts ? ts.toLocaleDateString([], {month: 'short', day: 'numeric'}) : '';
    var route = ep.tool_route || 'NONE';
    var rc = routeColors[route] || '#6a6a80';
    var speaker = ep.speaker || 'unknown';
    var speakerColor = speaker === 'unknown' ? '#6a6a80' : '#0f9';
    var emotion = ep.emotion || '';
    var latency = ep.response_latency_ms || 0;
    var input = ep.user_input || '';
    var response = ep.response_text || '';
    var mem = ep.memories_retrieved || {};
    var ident = ep.identity_state || {};
    var flags = ep.epistemic_flags || {};
    var epId = 'flight-ep-' + idx;

    body += '<div class="flight-episode" onclick="var el=document.getElementById(\'' + epId + '\');if(el){el.parentElement.classList.toggle(\'expanded\');}">';

    // Header row
    body += '<div class="flight-episode-header">';
    body += '<span style="font-size:0.55rem;color:' + rc + ';font-weight:700;min-width:50px;">' + esc(route) + '</span>';
    body += '<span class="flight-episode-time">' + esc(dateStr + ' ' + timeStr) + '</span>';
    body += '<span style="color:' + speakerColor + ';font-size:0.6rem;font-weight:600;">' + esc(speaker) + '</span>';
    if (emotion && emotion !== 'neutral' && emotion !== 'calm') {
      body += '<span style="font-size:0.55rem;color:#ff0;">' + esc(emotion) + '</span>';
    }
    body += '<span class="flight-episode-input">' + esc(input) + '</span>';
    body += '<span style="font-size:0.55rem;color:#6a6a80;min-width:40px;text-align:right;">' + latency + 'ms</span>';
    body += '</div>';

    // Meta row
    body += '<div class="flight-episode-meta">';
    if (mem.count > 0) {
      body += '<span>memories: ' + mem.count + '</span>';
    }
    if (ident.resolved && ident.resolved !== 'unknown') {
      body += '<span>id: ' + esc(ident.resolved) + ' (' + (ident.confidence * 100).toFixed(0) + '%)</span>';
    }
    if (ep.follow_up) body += '<span>follow-up</span>';
    if (ep.barged_in) body += '<span style="color:#f44;">barged-in</span>';
    body += '</div>';

    // Expandable detail
    body += '<div class="flight-episode-detail" id="' + epId + '">';

    // Response text
    if (response) {
      body += '<div style="margin-bottom:6px;">';
      body += '<div style="font-size:0.6rem;color:#6a6a80;margin-bottom:2px;">Response</div>';
      body += '<div style="font-size:0.65rem;color:#e0e0e8;padding:4px 8px;background:#0a0a14;border-radius:3px;border-left:2px solid ' + rc + ';">' + esc(response) + '</div>';
      body += '</div>';
    }

    // Routing pipeline
    body += '<div class="flight-pipeline">';
    var steps = ['wake', 'listen', 'stt', 'route', 'reason', 'tts', 'playback'];
    steps.forEach(function(step, si) {
      var isRoute = step === 'route';
      var stepColor = isRoute ? rc : '#0cf';
      body += '<span class="flight-step" style="color:' + stepColor + ';">' + esc(step) + (isRoute ? ': ' + esc(route) : '') + '</span>';
      if (si < steps.length - 1) body += '<span class="flight-connector">→</span>';
    });
    body += '</div>';

    // Memory retrieval details
    if (mem.count > 0) {
      body += '<div style="margin-top:4px;">';
      body += '<div style="font-size:0.6rem;color:#6a6a80;">Memory Retrieval</div>';
      body += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:2px;">';
      body += '<span style="padding:1px 5px;border:1px solid #2a2a44;border-radius:2px;font-size:0.55rem;">' +
        'count: ' + mem.count + '</span>';
      if (mem.route_type) {
        body += '<span style="padding:1px 5px;border:1px solid #2a2a44;border-radius:2px;font-size:0.55rem;">' +
          'route: ' + esc(mem.route_type) + '</span>';
      }
      if (mem.search_scope) {
        body += '<span style="padding:1px 5px;border:1px solid #2a2a44;border-radius:2px;font-size:0.55rem;">' +
          'scope: ' + esc(mem.search_scope) + '</span>';
      }
      var subjects = Object.entries(mem.subjects || {});
      subjects.forEach(function(s) {
        body += '<span style="padding:1px 5px;border:1px solid #0f933;color:#0f9;border-radius:2px;font-size:0.55rem;">' +
          esc(s[0]) + ': ' + s[1] + '</span>';
      });
      var types = Object.entries(mem.types || {});
      types.forEach(function(t) {
        body += '<span style="padding:1px 5px;border:1px solid #0cf33;color:#0cf;border-radius:2px;font-size:0.55rem;">' +
          esc(t[0]) + ': ' + t[1] + '</span>';
      });
      body += '</div></div>';
    }

    // Epistemic flags
    var hasFlags = flags.contradiction_touched || flags.provisional;
    if (hasFlags) {
      body += '<div class="flight-flags">';
      if (flags.contradiction_touched) {
        body += '<span class="flight-flag flight-flag-conflicted">contradiction</span>';
      }
      if (flags.provisional) {
        body += '<span class="flight-flag flight-flag-provisional">provisional</span>';
      }
      body += '</div>';
    }

    // Conversation ID
    if (ep.conversation_id) {
      body += '<div style="font-size:0.5rem;color:#484860;margin-top:4px;">conv: ' + esc(ep.conversation_id) + '</div>';
    }

    body += '</div>'; // detail
    body += '</div>'; // episode
  });
  body += '</div>'; // scrollable container

  var badge = '<span style="font-size:0.6rem;color:#c0f;">' + sorted.length + ' episodes</span>';
  return _panel('Flight Recorder', body, badge);
}


function _renderSelfImprovePanel(snap) {
  var si = snap.self_improve || {};
  var prov = si.provider || {};
  var codegen = snap.codegen || {};
  var coder = codegen.coder || si.coder || {};
  var scanner = si.scanner || {};

  var stageLabel = si.stage_label || (si.active ? 'active' : 'disabled');
  var stageColor = stageLabel === 'dry_run' ? '#ff0' : stageLabel === 'human_approval' ? '#f90' : si.active ? '#0f9' : '#6a6a80';

  var body = '';
  body += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Stage', stageLabel.replace('_', ' '), stageColor) +
    _statCard('Proposals', (si.recent_proposals || []).length || si.total_improvements || 0, '#0cf') +
    _statCard('Rollbacks', si.total_rollbacks || 0, (si.total_rollbacks || 0) > 0 ? '#f44' : '#0f9') +
    _statCard('Scans', scanner.total_scans || 0, '#c0f') +
    '</div>';

  // Provider + coder badges
  body += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px;">';
  var coderAvail = coder.available;
  var coderColor = coderAvail ? '#0f9' : '#f44';
  var coderLabel = coderAvail ? 'Shared CodeGen: ready' : 'Shared CodeGen: unavailable';
  if (coder.running) { coderColor = '#0cf'; coderLabel = 'Shared CodeGen: running'; }
  body += '<span style="padding:1px 5px;border:1px solid ' + coderColor + '44;color:' + coderColor + ';border-radius:2px;font-size:0.52rem;">' + coderLabel + '</span>';
  body += '<span style="padding:1px 5px;border:1px solid #6a6a8044;color:#8a8aa0;border-radius:2px;font-size:0.52rem;">infrastructure only</span>';
  if (prov.claude_available) body += '<span style="padding:1px 5px;border:1px solid #0f944;color:#0f9;border-radius:2px;font-size:0.52rem;">Claude</span>';
  if (prov.openai_available) body += '<span style="padding:1px 5px;border:1px solid #0f944;color:#0f9;border-radius:2px;font-size:0.52rem;">OpenAI</span>';
  if (si.effective_dry_run || si.dry_run_mode) body += '<span style="padding:1px 5px;border:1px solid #ff044;color:#ff0;border-radius:2px;font-size:0.52rem;">DRY RUN</span>';
  if (si.effective_write_policy) body += '<span style="padding:1px 5px;border:1px solid #0cf44;color:#0cf;border-radius:2px;font-size:0.52rem;">' + esc(si.effective_write_policy) + '</span>';
  if (si.auto_frozen) body += '<span style="padding:1px 5px;border:1px solid #f4444;color:#f44;border-radius:2px;font-size:0.52rem;">AUTO-FROZEN</span>';
  if (si.has_reliable_provider) body += '<span style="padding:1px 5px;border:1px solid #0f944;color:#0f9;border-radius:2px;font-size:0.52rem;">Provider OK</span>';
  body += '</div>';

  // Scanner summary
  if (scanner.total_scans > 0 || scanner.total_opportunities > 0) {
    var dailyUsed = scanner.daily_attempts_used || 0;
    var dailyCap = scanner.daily_cap || 6;
    body += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:6px;">' +
      _statCard('Opportunities', scanner.total_opportunities || 0, '#c0f') +
      _statCard('Daily Gen', dailyUsed + '/' + dailyCap, dailyUsed >= dailyCap ? '#f44' : '#0f9') +
      _statCard('Fingerprints', scanner.unique_fingerprints || 0, '#6af') +
      '</div>';
  }

  if (si.pending_approvals > 0) {
    body += '<div style="padding:4px 8px;background:rgba(255,153,0,0.08);border:1px solid rgba(255,153,0,0.3);border-radius:4px;margin-bottom:4px;">' +
      '<span style="font-size:0.62rem;color:#f90;font-weight:600;">' + si.pending_approvals + ' pending approval(s)</span>';
    if (window.approvePatch) {
      body += ' <button class="j-btn-xs" onclick="window.approvePatch(\'latest\',true)">Approve</button>' +
        ' <button class="j-btn-xs j-btn-red" onclick="window.approvePatch(\'latest\',false)">Reject</button>';
    }
    body += '</div>';
  }

  // Recent proposals
  var proposals = si.recent_proposals || [];
  var recent = proposals.length ? proposals : (si.recent_history || si.recent_patches || []);
  if (recent.length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Recent Proposals</div>';
    recent.slice(0, 5).forEach(function(p) {
      var desc = typeof p === 'object' ? (p.description || p.summary || p.request || '') : String(p);
      var status = typeof p === 'object' ? (p.status || p.result || '') : '';
      var fp = typeof p === 'object' && p.fingerprint ? p.fingerprint : '';
      body += '<div style="padding:2px 0;border-bottom:1px solid #1a1a2e;display:flex;gap:6px;align-items:center;">' +
        (status ? _statusBadge(status) : '') +
        '<span style="font-size:0.6rem;flex:1;">' + esc(desc.substring(0, 100)) + '</span>' +
        (fp ? '<span style="font-size:0.45rem;color:#6a6a80;font-family:monospace;">' + esc(fp.substring(0, 12)) + '</span>' : '') +
        '</div>';
    });
  }

  body += '<div style="margin-top:6px;display:flex;gap:6px;">';
  if (window.triggerDryRun) {
    body += '<button class="j-btn-sm" onclick="window.triggerDryRun()">Trigger Dry Run</button>';
  }
  body += '<a href="/capability-pipeline" class="j-btn-sm" style="text-decoration:none;color:#0cf;border-color:#0cf4;">Capability Pipeline &rarr;</a>';
  body += '</div>';

  return _panel('Self-Improvement', body, _statusBadge(si.active ? stageLabel.replace('_', ' ') : 'disabled'));
}


/* ═══════════════════════════════════════════════════════════════════════════
   TAB: LEARNING — "How is JARVIS improving?"
   ═══════════════════════════════════════════════════════════════════════════ */

window.renderLearning = function(snap) {
  var root = document.getElementById('learning-root');
  if (!root) return;

  var html = '';
  if (window._renderTabWayfinding) html += window._renderTabWayfinding('learning');

  // Learning toolbar
  html += '<div class="j-toolbar">' +
    '<button class="j-btn-sm" onclick="window.openLibraryIngest && window.openLibraryIngest()">Ingest Source</button>' +
    '<button class="j-btn-sm" onclick="window.openLibrarySources && window.openLibrarySources()">Browse Library</button>' +
    '<button class="j-btn-sm" onclick="window.cleanupLearningJobs && window.cleanupLearningJobs()">Cleanup Jobs</button>' +
    '</div>';

  // Skills spans full width
  html += _renderSkillsPanel(snap);

  // Data-heavy learning substrates get full-width rows.
  html += _renderLibraryPanel(snap);
  html += _renderHemispherePanel(snap);
  html += _renderLanguagePanel(snap);

  // Paired rows: live acquisition, then learned policy/matrix,
  // then self-improvement surfaces, then inspection/diagnostic panels.
  html += '<div class="j-panel-grid">';
  html += _renderAcquisitionPanel(snap);
  html += _renderMatrixPanel(snap);
  html += _renderPolicyPanel(snap);
  html += _renderSelfImprovePanel(snap);
  html += _renderImprovementHistoryPanel(snap);
  html += _renderPluginRegistryPanel(snap);
  html += _renderExplainabilityPanel(snap);
  html += _renderMLChartsPanel(snap);
  html += _renderCogGapsPanel(snap);
  html += '</div>';

  root.innerHTML = html;
};


function _renderSkillsPanel(snap) {
  var registry = snap.skills || {};
  var jobs = snap.learning_jobs || {};
  var gate = snap.capability_gate || {};
  var byStatus = registry.by_status || {};
  var skills = registry.skills || [];

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Total Skills', registry.total || 0) +
    _statCard('Verified', byStatus.verified || 0, '#0f9') +
    _statCard('Learning', byStatus.learning || 0, '#ff0') +
    _statCard('Blocked', byStatus.blocked || 0, (byStatus.blocked || 0) > 0 ? '#f44' : '#6a6a80') +
    _statCard('Jobs Total', jobs.total_count || 0, '#0cf') +
    _statCard('Jobs Done', jobs.completed_count || 0, '#0f9') +
    '</div>';

  // Capability gate
  var gateHtml = '<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px;padding:4px 8px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
    '<span style="font-size:0.55rem;color:#6a6a80;">Capability Gate:</span>' +
    '<span style="font-size:0.55rem;color:#0f9;">' + (gate.claims_passed || 0) + ' passed</span>' +
    '<span style="font-size:0.55rem;color:#f44;">' + (gate.claims_blocked || 0) + ' blocked</span>' +
    '<span style="font-size:0.55rem;color:#0cf;">' + (gate.claims_conversational || 0) + ' conv</span>' +
    (gate.affect_rewrites > 0 ? '<span style="font-size:0.55rem;color:#c0f;">' + gate.affect_rewrites + ' affect rw</span>' : '') +
    (gate.self_state_rewrites > 0 ? '<span style="font-size:0.55rem;color:#c0f;">' + gate.self_state_rewrites + ' self-state rw</span>' : '') +
    (gate.learning_rewrites > 0 ? '<span style="font-size:0.55rem;color:#c0f;">' + gate.learning_rewrites + ' learning rw</span>' : '') +
    (gate.honesty_failures > 0 ? '<span style="font-size:0.55rem;color:#f44;">' + gate.honesty_failures + ' honesty fail</span>' : '') +
    (gate.jobs_auto_created > 0 ? '<span style="font-size:0.55rem;color:#ff0;">' + gate.jobs_auto_created + ' auto jobs</span>' : '') +
    (gate.identity_confirmed ? '<span style="font-size:0.55rem;color:#0f9;">ID confirmed</span>' : '') +
    (jobs.failed_count > 0 ? '<span style="font-size:0.55rem;color:#f44;">Jobs failed: ' + jobs.failed_count + '</span>' : '') +
    '</div>';

  // Active learning jobs
  var jobsHtml = '';
  var activeJobs = jobs.active_jobs || [];
  if (activeJobs.length) {
    jobsHtml = '<div style="font-size:0.68rem;color:#ff0;font-weight:600;margin-bottom:4px;">Active Learning Jobs (' + (jobs.active_count || activeJobs.length) + ')</div>';
    jobsHtml += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px;">';
    activeJobs.slice(0, 8).forEach(function(j) {
      var matrixTag = j.matrix_protocol ? '<span style="padding:0 4px;background:#0f9;color:#000;border-radius:2px;font-size:0.5rem;">MATRIX</span> ' : '';
      var phases = ['assess', 'research', 'acquire', 'integrate', 'collect', 'train', 'verify', 'register'];
      var curPhase = j.phase || '';
      var phaseIdx = phases.indexOf(curPhase);

      jobsHtml += '<div style="padding:6px;background:#0d0d1a;border:1px solid #ff044;border-radius:4px;">';
      jobsHtml += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">' +
        '<span style="font-weight:600;font-size:0.68rem;color:#ff0;">' + matrixTag + esc(j.skill_id || '--') + '</span>' +
        _statusBadge(j.status) + '</div>';

      // Phase progress bar
      jobsHtml += '<div style="display:flex;gap:1px;margin-bottom:4px;">';
      phases.forEach(function(p, i) {
        var pc = i < phaseIdx ? '#0f9' : i === phaseIdx ? '#ff0' : '#1a1a2e';
        jobsHtml += '<div style="flex:1;height:3px;background:' + pc + ';border-radius:1px;" title="' + p + '"></div>';
      });
      jobsHtml += '</div>';

      jobsHtml += '<div style="font-size:0.55rem;color:#6a6a80;">' +
        'Phase: <span style="color:#0cf;">' + esc(curPhase || '--') + '</span>' +
        ' · Type: ' + esc(j.capability_type || '') +
        ' · Priority: ' + (j.priority || 0) +
        '</div>';

      if (j.needs_approval) {
        jobsHtml += '<div style="font-size:0.55rem;color:#f44;margin-top:2px;">Needs approval ' +
          '<button class="j-btn-xs" onclick="window.approveLearningJob && window.approveLearningJob(\'' + esc(j.job_id || j.skill_id || '') + '\')">Approve</button> ' +
          '<button class="j-btn-xs j-btn-red" onclick="window.rejectLearningJob && window.rejectLearningJob(\'' + esc(j.job_id || j.skill_id || '') + '\')">Reject</button></div>';
      }

      var canDelete = j.status === 'blocked' || j.status === 'failed' || j.stale;
      if (canDelete) {
        var isDef = j.is_default_skill ? 'true' : 'false';
        var btnLabel = j.is_default_skill ? '⚠ Delete Job (System Skill)' : 'Delete Job';
        var btnStyle = j.is_default_skill ? 'background:#600;border:1px solid #f44;' : '';
        jobsHtml += '<div style="display:flex;gap:4px;margin-top:3px;">' +
          '<button class="j-btn-xs j-btn-red" style="' + btnStyle + '" onclick="window.deleteLearningJob && window.deleteLearningJob(\'' + esc(j.job_id || '') + '\', \'' + esc(j.skill_id || '') + '\', ' + isDef + ')">' + btnLabel + '</button>';
        if (j.stale && j.status !== 'blocked') {
          jobsHtml += '<span style="font-size:0.48rem;color:#f44;align-self:center;">stale (' + Math.round((j.phase_age_s || 0) / 3600) + 'h)</span>';
        }
        jobsHtml += '</div>';
      }
      jobsHtml += '</div>';
    });
    jobsHtml += '</div>';
  }

  // Skills list — 2-column
  var skillsHtml = '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">Skill Registry</div>';
  skillsHtml += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:3px 10px;max-height:300px;overflow-y:auto;">';
  if (skills.length) {
    var statusColors = { verified: '#0f9', learning: '#ff0', blocked: '#f44', degraded: '#f90', unknown: '#6a6a80' };
    skills.forEach(function(s) {
      var c = statusColors[s.status] || '#6a6a80';
      var sid = s.skill_id || s.name || '';
      var evSum = s.evidence_summary || {};
      var evMethod = evSum.verification_method || evSum.summary || '';
      skillsHtml += '<div style="display:flex;align-items:center;gap:4px;padding:3px 0;border-bottom:1px solid #1a1a2e;cursor:pointer;min-width:0;" onclick="window.openSkillDetail && window.openSkillDetail(\'' + esc(sid) + '\')">' +
        '<span style="width:6px;height:6px;border-radius:50%;background:' + c + ';flex-shrink:0;"></span>' +
        '<div style="flex:1;min-width:0;">' +
        '<div style="font-size:0.62rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + esc(s.name || s.skill_id) + '</div>' +
        '<div style="font-size:0.48rem;color:#484860;">' + esc(s.capability_type || '') +
        (evMethod ? ' · ' + esc(String(evMethod).substring(0, 30)) : '') +
        (s.learning_job_id ? ' · <span style="color:#0cf;">job linked</span>' : '') +
        '</div></div>' +
        '<span style="font-size:0.5rem;color:' + c + ';flex-shrink:0;">' + esc(s.status || '--') + '</span></div>';
    });
  }
  skillsHtml += '</div>';

  return _panel('Skills & Learning', statsHtml + gateHtml + jobsHtml + skillsHtml,
    '<span style="font-size:0.6rem;color:#6a6a80;">' + (registry.total || 0) + ' skills · ' + (jobs.total_count || 0) + ' jobs</span>');
}


function _renderLibraryPanel(snap) {
  var lib = snap.library || {};
  if (!Object.keys(lib).length) return _panel('Library', _emptyMsg('Not initialized'));

  var chunks = lib.chunks || {};
  var concepts = lib.concepts || {};
  var retrieval = lib.retrieval || {};

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Sources', lib.total || 0) +
    _statCard('Studied', lib.studied || 0, '#0f9') +
    _statCard('Unstudied', lib.unstudied || 0, (lib.unstudied || 0) > 0 ? '#ff0' : '#0f9') +
    _statCard('Failed', lib.failed || 0, (lib.failed || 0) > 0 ? '#f44' : '#0f9') +
    '</div>';

  statsHtml += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Chunks', chunks.total || 0, '#0cf') +
    _statCard('With Concepts', chunks.with_concepts || 0, '#c0f') +
    _statCard('Concepts', concepts.total_concepts || 0, '#c0f') +
    _statCard('Edges', concepts.total_edges || 0, '#6a6a80') +
    '</div>';

  // Source types + ingested by
  var typeHtml = '';
  var byType = lib.by_type || {};
  var byIngest = lib.by_ingested_by || {};
  if (Object.keys(byType).length || Object.keys(byIngest).length) {
    typeHtml = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:6px;">';
    if (Object.keys(byType).length) {
      typeHtml += '<div><div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">By Type</div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:3px;">';
      Object.entries(byType).forEach(function(e) {
        typeHtml += '<span style="padding:1px 4px;border:1px solid #0cf33;color:#0cf;border-radius:2px;font-size:0.5rem;">' + esc(e[0]) + ': ' + e[1] + '</span>';
      });
      typeHtml += '</div></div>';
    }
    if (Object.keys(byIngest).length) {
      typeHtml += '<div><div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Ingested By</div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:3px;">';
      Object.entries(byIngest).forEach(function(e) {
        var c = e[0] === 'gestation' ? '#c0f' : '#0f9';
        typeHtml += '<span style="padding:1px 4px;border:1px solid ' + c + '33;color:' + c + ';border-radius:2px;font-size:0.5rem;">' + esc(e[0]) + ': ' + e[1] + '</span>';
      });
      typeHtml += '</div></div>';
    }
    typeHtml += '</div>';
  }

  // Content depth
  var depthHtml = '';
  var byDepth = lib.by_content_depth || {};
  if (Object.keys(byDepth).length) {
    var depthColors = { title_only: '#f44', metadata_only: '#f66', abstract: '#f90', tldr: '#0f9', full_text: '#0cf', unknown: '#6a6a80' };
    depthHtml = '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Content Depth</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">';
    Object.entries(byDepth).forEach(function(e) {
      var c = depthColors[e[0]] || '#6a6a80';
      depthHtml += '<span style="padding:1px 4px;border:1px solid ' + c + '33;color:' + c + ';border-radius:2px;font-size:0.5rem;">' + esc(e[0]) + ': ' + e[1] + '</span>';
    });
    depthHtml += '</div>';
  }

  // Top domains
  var domainHtml = '';
  var topDomains = lib.top_domains || [];
  if (topDomains.length) {
    domainHtml = '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Top Domains</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">';
    topDomains.forEach(function(td) {
      domainHtml += '<span style="padding:1px 4px;border:1px solid #c0f33;color:#c0f;border-radius:2px;font-size:0.5rem;">' + esc(td.tags || td.name || '') + ': ' + (td.count || 0) + '</span>';
    });
    domainHtml += '</div>';
  }

  // Top concepts
  var conceptHtml = '';
  var topConcepts = concepts.top_concepts || [];
  if (topConcepts.length) {
    conceptHtml = '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Top Concepts</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:6px;">';
    topConcepts.slice(0, 12).forEach(function(tc) {
      var size = Math.max(0.4, Math.min(0.6, 0.4 + (tc.count / 100) * 0.2));
      conceptHtml += '<span style="padding:1px 3px;color:#e0e0e8;font-size:' + size + 'rem;">' + esc(tc.name) + '<sup style="color:#484860;">' + tc.count + '</sup></span>';
    });
    conceptHtml += '</div>';
  }

  // Retrieval
  var retHtml = '';
  if (retrieval.total_starts || retrieval.total_outcomes) {
    retHtml = '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Retrieval</div>' +
      '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:4px;">' +
      _statCard('Starts', retrieval.total_starts || 0) +
      _statCard('Outcomes', retrieval.total_outcomes || 0) +
      _statCard('Success', retrieval.recent_total > 0 ? ((retrieval.recent_ok / retrieval.recent_total) * 100).toFixed(0) + '%' : '--', '#0f9') +
      _statCard('Log Size', retrieval.log_size_kb ? fmtNum(retrieval.log_size_kb, 1) + ' KB' : '--', '#6a6a80') +
      '</div>';
  }

  // Backoff pending
  var backoffHtml = '';
  if (lib.backoff_pending > 0) {
    backoffHtml = '<div style="padding:3px 6px;background:rgba(255,153,0,0.08);border:1px solid rgba(255,153,0,0.3);border-radius:3px;margin-bottom:4px;font-size:0.55rem;color:#f90;">' +
      lib.backoff_pending + ' source(s) in backoff</div>';
  }

  var libActions = '<div style="margin-top:4px;display:flex;gap:6px;">' +
    (window.openLibrarySources ? '<button class="j-btn-sm" onclick="window.openLibrarySources()">Browse Sources</button>' : '') +
    (window.openLibraryIngest ? '<button class="j-btn-sm" onclick="window.openLibraryIngest()">Ingest New</button>' : '') +
    '</div>';

  return _panel('Library', statsHtml + typeHtml + depthHtml + domainHtml + conceptHtml + retHtml + backoffHtml + libActions,
    '<span style="font-size:0.6rem;color:#6a6a80;">' + (lib.total || 0) + ' sources · ' + (chunks.total || 0) + ' chunks</span>');
}


function _manualGatePromptDeck(responseClass) {
  var prompts = {
    recent_learning: [
      'Jarvis, what have you learned recently?',
      'Jarvis, what did you learn from our recent conversations?',
      'Jarvis, what skill did you just finish learning?'
    ],
    recent_research: [
      'Jarvis, what have you researched recently?',
      'Jarvis, what did you study recently?',
      'Jarvis, what recent sources did you learn from?'
    ],
    identity_answer: [
      'Jarvis, who am I?',
      'Jarvis, do you recognize me?',
      'Jarvis, do you know who I am?',
      'Jarvis, who is speaking?'
    ],
    capability_status: [
      'Jarvis, what can you do?',
      'Jarvis, what are your capabilities?',
      'Jarvis, what are your current skills?',
      'Jarvis, which capabilities are verified?'
    ]
  };
  return prompts[responseClass] || [
    'Use a natural lived conversation that should produce this response class.'
  ];
}

function _manualGateExpectedRoute(responseClass) {
  var routes = {
    recent_learning: 'INTROSPECTION -> recent_learning',
    recent_research: 'INTROSPECTION -> recent_research',
    identity_answer: 'IDENTITY -> identity_answer',
    capability_status: 'INTROSPECTION -> capability_status'
  };
  return routes[responseClass] || '--';
}


function _onboardingOperatorPromptDeck(stage) {
  var decks = {
    1: [
      'Jarvis, my name is David.',
      'Jarvis, do you recognize me?',
      'Jarvis, remember that I prefer direct, honest answers.',
      'Jarvis, tell me how your memory works.'
    ],
    2: [
      'Jarvis, I prefer detailed answers when we are working on code.',
      'Jarvis, when I ask for a quick answer, keep it to one short paragraph.',
      'Jarvis, I like reliability, honesty, and clear progress reports.',
      'Jarvis, do not bring up private family or medical details proactively.',
      'Jarvis, I prefer that you tell me when you are uncertain.'
    ],
    3: [
      'Jarvis, my family and household details are private unless I say otherwise.',
      'Jarvis, if someone else asks about my family, do not share private details.',
      'Jarvis, if another person is speaking, verify who they are before using memories.',
      'Jarvis, tell me what you are allowed to share about me with someone else.'
    ],
    4: [
      'Jarvis, my usual morning routine is ...',
      'Jarvis, during focused work, stay quiet unless I ask you something.',
      'Jarvis, interrupt me only for safety, urgent system issues, or something I explicitly asked you to watch.',
      'Jarvis, my top priorities right now are ...'
    ],
    5: [
      'Jarvis, correction: that is not what I meant.',
      'Jarvis, correction: update that preference.',
      'Jarvis, do not learn from that last thing.',
      'Jarvis, tell me something you are uncertain about, and I will correct it.'
    ],
    6: [
      'Jarvis, what are my top preferences?',
      'Jarvis, what do you know about my routine?',
      'Jarvis, what corrections have I made recently?',
      'Jarvis, tell me something you remember about me and include your confidence.'
    ],
    7: [
      'Jarvis, make a low-stakes suggestion based on what you know about my preferences.',
      'Jarvis, give me your honest companion training self-assessment.',
      'Jarvis, how has your understanding of me changed since training started?'
    ]
  };
  return decks[Number(stage)] || [];
}

function _onboardingMetricValueLabel(value, target) {
  if (value == null || value === '') return '--';
  if (typeof value === 'number') {
    if (typeof target === 'number' && target <= 1) return fmtNum(value, 2);
    return Number.isInteger(value) ? String(value) : fmtNum(value, 2);
  }
  return String(value);
}

function _renderOnboardingSayThis(stageInfo, currentStage, missingCount) {
  if (!stageInfo) return '';
  var prompted = Number(stageInfo.exercises_prompted || 0);
  var exercises = Array.isArray(stageInfo.exercises) ? stageInfo.exercises : [];
  var operatorPrompts = _onboardingOperatorPromptDeck(currentStage);
  var prompts = operatorPrompts.length ? operatorPrompts : exercises;
  if (!prompts.length) return '';
  var remaining = Math.max(0, prompts.length - prompted);
  var start = Math.min(prompted, Math.max(0, prompts.length - 1));
  var selected = prompts.slice(start, start + 4);
  if (!selected.length) selected = prompts.slice(0, 4);
  var title = missingCount > 0 ? 'Say This Next' : 'Practice Prompts';
  var html = '<div style="margin:6px 0;padding:7px;background:#0d0d1a;border:1px solid #1a2a44;border-radius:5px;">' +
    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">' +
    '<div style="font-size:0.62rem;color:#0cf;font-weight:700;">' + title + '</div>' +
    '<div style="font-size:0.48rem;color:#6a6a80;">stage prompts used: ' + prompted + ' · deck left: ' + remaining + '</div>' +
    '</div>' +
    '<div style="display:flex;flex-direction:column;gap:3px;">';
  selected.forEach(function(prompt) {
    html += '<div style="padding:4px 6px;background:#090914;border:1px solid #182844;border-radius:3px;font-size:0.54rem;color:#d8f6ff;line-height:1.35;">' +
      esc(prompt) + '</div>';
  });
  html += '</div>' +
    '<div style="margin-top:4px;font-size:0.48rem;color:#6a6a80;">Use natural conversation after the exact starter if needed. The gate counts lived evidence, not button clicks.</div>' +
    '</div>';
  return html;
}

function _renderOnboardingCurrentTargets(ob, stageInfo) {
  if (!stageInfo) return '';
  var targets = stageInfo.checkpoint_targets || {};
  var met = stageInfo.checkpoints_met || {};
  var live = ob.live_metrics || {};
  var keys = Object.keys(targets);
  if (!keys.length) return '';
  var missing = keys.filter(function(k) { return !met[k]; });
  var html = '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:3px;">Current Stage Targets</div>' +
    '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:3px;margin-bottom:6px;">';
  keys.forEach(function(k) {
    var target = targets[k];
    var value = live[k];
    var passed = !!met[k];
    var border = passed ? '#0f9' : '#f90';
    var color = passed ? '#0f9' : '#ff0';
    html += '<div style="padding:4px 5px;background:#0d0d1a;border:1px solid ' + border + '55;border-radius:3px;font-size:0.5rem;">' +
      '<div style="display:flex;justify-content:space-between;gap:6px;">' +
      '<span style="color:#8a8aa0;">' + esc(k.replace(/_/g, ' ')) + '</span>' +
      '<span style="color:' + color + ';">' + (passed ? 'met' : 'needed') + '</span>' +
      '</div>' +
      '<div style="color:#e0e0e8;margin-top:2px;">' +
      esc(_onboardingMetricValueLabel(value, target)) + ' / ' + esc(_onboardingMetricValueLabel(target, target)) +
      '</div></div>';
  });
  html += '</div>';
  if (missing.length) {
    html += '<div style="font-size:0.5rem;color:#f90;margin-bottom:5px;">Blocking now: ' +
      missing.map(function(k) { return esc(k.replace(/_/g, ' ')); }).join(', ') + '</div>';
  }
  return html;
}


function _renderManualGateWork(snap) {
  var ev = snap.eval || {};
  var vp = ev.validation_pack || {};
  var targets = vp.language_evidence_targets || [];
  var baselines = vp.language_route_class_baselines || [];
  if (!targets.length && !baselines.length) return '';

  var baselineByClass = {};
  baselines.forEach(function(row) {
    if (!row || !row.response_class) return;
    baselineByClass[row.response_class] = row;
  });

  var rows = targets.filter(function(row) {
    return row && row.response_class && !row.current_ok;
  });
  if (!rows.length) return '';

  var totalGap = rows.reduce(function(acc, row) {
    return acc + Math.max(0, Number(row.gap || 0));
  }, 0);

  var html = '<div style="margin:8px 0;padding:8px;background:#120f0a;border:1px solid #4a3514;border-radius:5px;">';
  html += '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:5px;">' +
    '<div style="font-size:0.66rem;color:#ff0;font-weight:700;">Manual Gate Work Needed</div>' +
    '<div style="font-size:0.5rem;color:#f90;">lived examples remaining: ' + totalGap + '</div>' +
    '</div>';
  html += '<div style="font-size:0.52rem;color:#8a8aa0;line-height:1.35;margin-bottom:6px;">' +
    'These gates do not open just because JARVIS runs overnight. They need real operator conversations that naturally produce each response class. Synthetic or passive runtime does not count here.' +
    '</div>';
  html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:5px;">';
  rows.forEach(function(row) {
    var rc = row.response_class;
    var target = Number(row.target_count || 30);
    var count = Number(row.count || 0);
    var gap = Math.max(0, Number(row.gap || (target - count)));
    var base = baselineByClass[rc] || {};
    var baseCount = Number(base.count || 0);
    var baseOk = !!base.current_ok;
    var border = gap > 0 ? '#4a3514' : '#184430';
    var accent = baseOk ? '#0cf' : '#f44';
    html += '<div style="background:#0d0d1a;border:1px solid ' + border + ';border-radius:4px;padding:6px;">' +
      '<div style="display:flex;justify-content:space-between;gap:4px;align-items:center;margin-bottom:4px;">' +
      '<span style="font-size:0.58rem;color:#e0e0e8;">' + esc(rc) + '</span>' +
      '<span style="font-size:0.5rem;color:#ff0;">' + count + '/' + target + '</span>' +
      '</div>' +
      _barFill(count, target, gap > 0 ? '#ff0' : '#0f9') +
      '<div style="font-size:0.5rem;color:#6a6a80;line-height:1.35;margin-top:5px;">' +
      '<div>Need <span style="color:#ff0;">' + gap + '</span> more lived example' + (gap === 1 ? '' : 's') + '.</div>' +
      '<div>Baseline route/class: <span style="color:' + accent + ';">' +
      esc((base.route || '--') + ' -> ' + rc + ' (' + baseCount + ')') + '</span></div>' +
      '<div style="margin-top:3px;color:#8a8aa0;">Expected: <span style="color:#0cf;">' + esc(_manualGateExpectedRoute(rc)) + '</span></div>' +
      '<div style="margin-top:4px;color:#ff0;font-weight:700;">Say this exactly:</div>';
    _manualGatePromptDeck(rc).slice(0, Math.min(4, Math.max(1, gap))).forEach(function(prompt) {
      html += '<div style="margin-top:2px;padding:3px 5px;background:#090914;border:1px solid #24243a;border-radius:3px;color:#e0e0e8;font-family:monospace;font-size:0.52rem;">' + esc(prompt) + '</div>';
    });
    html += '<div style="margin-top:4px;color:#8a8aa0;">Counts only after JARVIS produces the intended grounded/native response class. Do not use third-person variants like "what can she do" for this gate pass yet.</div>' +
      '</div></div>';
  });
  html += '</div></div>';
  return html;
}
window._renderManualGateWork = _renderManualGateWork;


function _renderLanguagePanel(snap) {
  var lang = snap.language || {};
  if (!Object.keys(lang).length) return _panel('Language Substrate', _emptyMsg('No data'));

  var quality = lang.quality || {};
  var phasec = lang.phase_c || {};
  var phasecTokenizer = phasec.tokenizer || {};
  var phasecDataset = phasec.dataset || {};
  var phasecSplit = phasec.split || {};
  var phasecStudent = phasec.student || {};
  var phasecLastRun = phasec.last_run || {};
  var phasecBaseline = phasec.baseline || {};
  var shadowCmp = quality.shadow_comparisons || {};
  var amb = quality.ambiguous_intent || {};
  var byRoute = lang.counts_by_route || {};
  var byClass = lang.counts_by_response_class || {};
  var byFeedback = lang.counts_by_feedback || {};
  var byProvenance = lang.counts_by_provenance || {};
  var bySafety = lang.counts_by_safety_flag || {};
  var nativeClasses = lang.native_response_classes || [];

  function langStat(label, value, color) {
    var c = color || '#e0e0e8';
    return '<div class="stat-card lang-stat-card" style="text-align:left;padding:3px 5px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;min-width:0;">' +
      '<div class="stat-value" style="font-size:0.78rem;line-height:1;font-weight:700;color:' + c + ';white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + esc(String(value != null ? value : '--')) + '</div>' +
      '<div class="stat-label" style="font-size:0.45rem;line-height:1.1;color:#6a6a80;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + esc(label) + '</div></div>';
  }
  function langMetricChip(label, value, color) {
    var c = color || '#8a8aa0';
    return '<span style="display:flex;justify-content:space-between;gap:4px;padding:1px 4px;border:1px solid #222238;border-radius:2px;font-size:0.45rem;line-height:1.25;color:#6a6a80;min-width:0;">' +
      '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(label) + '</span>' +
      '<b style="color:' + c + ';font-weight:700;">' + esc(String(value != null ? value : '--')) + '</b></span>';
  }
  function langMiniTitle(title) {
    return '<div style="font-size:0.5rem;color:#6a6a80;margin:1px 0 2px;text-transform:uppercase;letter-spacing:0.03em;">' + esc(title) + '</div>';
  }

  var statsHtml = '<div class="language-compact-grid language-top-grid" style="display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:3px;margin-bottom:4px;">' +
    langStat('Corpus', lang.total_examples || 0) +
    langStat('Recent', lang.recent_example_count || 0, '#0cf') +
    langStat('MF Previews', lang.examples_with_meaning_frame_preview || 0, '#c0f') +
    langStat('File Size', lang.file_size_bytes ? fmtNum(lang.file_size_bytes / 1024, 1) + ' KB' : '--', '#6a6a80') +
    langStat('Native Rate', Object.keys(quality).length && quality.total_events ? fmtPct(quality.native_usage_rate || 0, 1) : '--', '#0f9') +
    langStat('Fail-Closed', Object.keys(quality).length && quality.total_events ? fmtPct(quality.fail_closed_rate || 0, 1) : '--', '#ff0') +
    langStat('Events', Object.keys(quality).length && quality.total_events ? quality.total_events || 0 : '--') +
    langStat('Retained', lang.retained_file_count || 0, '#6a6a80') +
    '</div>';

  if (shadowCmp.total) {
    statsHtml += '<div class="language-compact-grid" style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:3px;margin-bottom:4px;">' +
      langStat('Shadow Cmp', shadowCmp.total || 0, '#0cf') +
      langStat('Cmp Chosen B', (shadowCmp.by_choice || {}).bounded || 0, '#0f9') +
      langStat('Cmp Chosen L', (shadowCmp.by_choice || {}).llm || 0, '#f90') +
      langStat('Last Cmp', shadowCmp.last_ts ? timeAgo(shadowCmp.last_ts) : '--', '#6a6a80') +
      '</div>';
  }
  if (amb.total) {
    var bySel = amb.by_selected_route || {};
    var byCand = amb.by_candidate_intent || {};
    var byFb = amb.by_feedback || {};
    var noneCt = (bySel.NONE || 0) + (bySel.none || 0);
    var corrCt = (byFb.correction || 0);
    statsHtml += '<div class="language-compact-grid" style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:3px;margin-bottom:4px;">' +
      langStat('Ambiguous', amb.total || 0, '#c0f') +
      langStat('Route NONE', noneCt, noneCt > 0 ? '#ff0' : '#0f9') +
      langStat('Cand Research', byCand.recent_research_or_processing || 0, '#0cf') +
      langStat('Corrections', corrCt, corrCt > 0 ? '#f90' : '#6a6a80') +
      '</div>';
  }

  // Native response classes
  var nativeHtml = '';
  if (nativeClasses.length) {
    nativeHtml = '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Native Response Classes</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:4px;">';
    nativeClasses.forEach(function(nc) {
      nativeHtml += '<span style="padding:1px 4px;border:1px solid #0f933;color:#0f9;border-radius:2px;font-size:0.48rem;">' + esc(nc) + '</span>';
    });
    nativeHtml += '</div>';
  }

  var routeHtml = '';
  if (Object.keys(byRoute).length) {
    routeHtml = '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">By Route</div>' +
      '<div style="margin-bottom:4px;">' + _tagGrid(byRoute, 'no routes') + '</div>';
  }

  var classHtml = '';
  if (Object.keys(byClass).length) {
    classHtml = '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">By Response Class</div>' +
      '<div style="margin-bottom:4px;">' + _tagGrid(byClass, 'no classes') + '</div>';
  }

  // Feedback, Provenance, Safety
  var extraHtml = '';
  if (Object.keys(byFeedback).length) {
    extraHtml += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">By Feedback</div>' +
      '<div style="margin-bottom:4px;">' + _tagGrid(byFeedback, 'none') + '</div>';
  }
  if (Object.keys(byProvenance).length) {
    extraHtml += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">By Provenance</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:4px;">';
    Object.entries(byProvenance).forEach(function(e) {
      extraHtml += '<span style="padding:1px 3px;border:1px solid #c0f22;color:#c0f;border-radius:2px;font-size:0.45rem;">' + esc(e[0]) + ':' + e[1] + '</span>';
    });
    extraHtml += '</div>';
  }
  if (Object.keys(bySafety).length) {
    extraHtml += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Safety Flags</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:4px;">';
    Object.entries(bySafety).forEach(function(e) {
      extraHtml += '<span style="padding:1px 3px;border:1px solid #ff022;color:#ff0;border-radius:2px;font-size:0.45rem;">' + esc(e[0]) + ':' + e[1] + '</span>';
    });
    extraHtml += '</div>';
  }

  // Phase D: Eval Gates
  var gateHtml = '';
  var gateScores = lang.gate_scores || {};
  var gateColor = lang.gate_color || '';
  var gateScoresByClass = lang.gate_scores_by_class || {};
  if (Object.keys(gateScores).length) {
    var colorMap = {green: '#0f9', yellow: '#ff0', red: '#f44'};
    var gcol = colorMap[gateColor] || '#6a6a80';
    gateHtml += langMiniTitle('Phase D Eval Gates') +
      '<div style="font-size:0.55rem;color:#6a6a80;margin:-15px 0 2px 120px;">' +
      '<span style="color:' + gcol + ';font-weight:bold;">' + (gateColor || '--').toUpperCase() + '</span></div>';
    gateHtml += '<div class="language-compact-grid" style="display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:3px;margin-bottom:4px;">';
    var gateLabels = {
      sample_count: 'Samples', provenance_fidelity: 'Provenance', exactness: 'Exactness',
      hallucination_rate: 'Anti-Halluc', fail_closed_correctness: 'Fail-Closed',
      native_usage_rate: 'Native Use', style_quality: 'Style'
    };
    Object.entries(gateScores).forEach(function(e) {
      var label = gateLabels[e[0]] || e[0];
      var val = e[1];
      var col = val >= 0.85 ? '#0f9' : val >= 0.6 ? '#ff0' : '#f44';
      gateHtml += langStat(label, fmtPct(val, 1), col);
    });
    gateHtml += '</div>';
  }
  if (Object.keys(gateScoresByClass).length) {
    gateHtml += langMiniTitle('Per-Class Gate Diagnostics');
    gateHtml += '<div class="language-class-grid" style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:3px;margin-bottom:4px;">';
    Object.entries(gateScoresByClass).forEach(function(e) {
      var rc = e[0];
      var gs = e[1] || {};
      var s = gs.scores || {};
      var c = gs.color || 'red';
      var border = c === 'green' ? '#184430' : c === 'yellow' ? '#564c1e' : '#4a1f26';
      var cc = c === 'green' ? '#0f9' : c === 'yellow' ? '#ff0' : '#f44';
      gateHtml += '<div style="background:#0d0d1a;border:1px solid ' + border + ';border-radius:3px;padding:3px 4px;min-width:0;">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;gap:4px;">' +
        '<span style="font-size:0.48rem;color:#e0e0e8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(rc) + '</span>' +
        '<span style="font-size:0.5rem;color:' + cc + ';font-weight:bold;">' + esc(c.toUpperCase()) + '</span>' +
        '</div>' +
        '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:2px;">' +
        langMetricChip('Smp', fmtPct(s.sample_count || 0, 0), '#6a6a80') +
        langMetricChip('Prov', fmtPct(s.provenance_fidelity || 0, 0), '#0cf') +
        langMetricChip('Exact', fmtPct(s.exactness || 0, 0), '#0f9') +
        langMetricChip('Anti-H', fmtPct(s.hallucination_rate || 0, 0), '#ff0') +
        langMetricChip('Fail', fmtPct(s.fail_closed_correctness || 0, 0), '#f90') +
        langMetricChip('Native', fmtPct(s.native_usage_rate || 0, 0), '#0af') +
        '</div></div>';
    });
    gateHtml += '</div>';
  }

  // Phase D: Promotion State
  var promoHtml = '';
  var promo = lang.promotion || {};
  var promoAgg = lang.promotion_aggregate || {};
  var redQuality = lang.promotion_red_quality_classes || promoAgg.red_quality_classes || 0;
  var redDataLimited = lang.promotion_red_data_limited_classes || promoAgg.red_data_limited_classes || 0;
  if (Object.keys(promo).length) {
    promoHtml += langMiniTitle('Promotion Governor');
    var levels = promoAgg.levels || {};
    var colors = promoAgg.colors || {};
    if (Object.keys(promoAgg).length) {
      promoHtml += '<div class="language-compact-grid" style="display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:3px;margin-bottom:4px;">' +
        langStat('Shadow', levels.shadow || 0, '#6a6a80') +
        langStat('Canary', levels.canary || 0, '#ff0') +
        langStat('Live', levels.live || 0, '#0f9') +
        langStat('Red Quality', redQuality, redQuality > 0 ? '#f44' : '#0f9') +
        langStat('Evals', promoAgg.total_evaluations || 0, '#0cf') +
        langStat('Max Red', promoAgg.max_consecutive_red || 0, (promoAgg.max_consecutive_red || 0) > 0 ? '#f90' : '#6a6a80') +
        langStat('Max Green', promoAgg.max_consecutive_green || 0, '#0f9') +
        '</div>';
      promoHtml += '<div style="font-size:0.5rem;color:#6a6a80;margin-bottom:6px;">' +
        'data-limited reds: <span style="color:' + (redDataLimited > 0 ? '#ff0' : '#6a6a80') + ';">' + redDataLimited + '</span>' +
        ' · total reds: <span style="color:' + ((colors.red || 0) > 0 ? '#f44' : '#6a6a80') + ';">' + (colors.red || 0) + '</span>' +
        '</div>';
    }
    promoHtml += '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:4px;">';
    var levelColors = {shadow: '#6a6a80', canary: '#ff0', live: '#0f9'};
    Object.entries(promo).forEach(function(e) {
      var rc = e[0];
      var st = e[1] || {};
      var lvl = st.level || 'shadow';
      var lc = levelColors[lvl] || '#6a6a80';
      var gc = st.color === 'green' ? '#0f9' : st.color === 'yellow' ? '#ff0' : st.color === 'red' ? '#f44' : '#6a6a80';
      promoHtml += '<span style="padding:2px 4px;border:1px solid ' + lc + '33;color:' + lc + ';border-radius:2px;font-size:0.45rem;">' +
        esc(rc) + ' <span style="color:' + gc + ';">' + lvl.toUpperCase() + '</span>' +
        (st.consecutive_green ? ' (' + st.consecutive_green + 'g)' : '') +
        (st.consecutive_red ? ' (' + st.consecutive_red + 'r)' : '') + '</span>';
    });
    promoHtml += '</div>';
    promoHtml += '<div class="language-class-grid" style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:3px;margin-bottom:4px;">';
    Object.entries(promo).forEach(function(e) {
      var rc = e[0];
      var st = e[1] || {};
      var lvl = st.level || 'shadow';
      var lvlColor = lvl === 'live' ? '#0f9' : lvl === 'canary' ? '#ff0' : '#6a6a80';
      var gColor = st.color === 'green' ? '#0f9' : st.color === 'yellow' ? '#ff0' : '#f44';
      var reason = st.gate_reason || '';
      var border = lvl === 'live' ? '#184430' : lvl === 'canary' ? '#564c1e' : '#2a2a44';
      var trans = st.last_transition_to ? ((st.last_transition_from || '--') + '→' + st.last_transition_to) : '--';
      promoHtml += '<div style="background:#0d0d1a;border:1px solid ' + border + ';border-radius:3px;padding:4px;min-width:0;">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;gap:4px;">' +
        '<span style="font-size:0.48rem;color:#e0e0e8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(rc) + '</span>' +
        '<span style="font-size:0.48rem;color:' + lvlColor + ';">' + esc(lvl.toUpperCase()) + '</span></div>' +
        '<div style="margin-top:2px;font-size:0.45rem;color:#6a6a80;line-height:1.25;">' +
        '<div><span style="color:#8a8aa0;">gate:</span> <span style="color:' + gColor + ';">' + esc((st.color || '--').toUpperCase()) + '</span></div>' +
        '<div><span style="color:#8a8aa0;">streak:</span> g' + (st.consecutive_green || 0) + ' / r' + (st.consecutive_red || 0) + '</div>' +
        '<div><span style="color:#8a8aa0;">dwell:</span> ' + fmtNum((st.dwell_s || 0), 1) + 's</div>' +
        (reason && reason !== 'ok' ? '<div><span style="color:#8a8aa0;">reason:</span> ' + esc(reason) + '</div>' : '') +
        '<div><span style="color:#8a8aa0;">transition:</span> ' + esc(trans) + '</div>' +
        '</div></div>';
    });
    promoHtml += '</div>';
  }

  // Phase D: Runtime guard consumption
  var runtimeHtml = '';
  var runtimeMode = (lang.runtime_rollout_mode || 'off').toLowerCase();
  var runtimeEnabled = !!lang.runtime_bridge_enabled;
  var runtimeGuardTotal = lang.runtime_guard_total || 0;
  var runtimeLiveTotal = lang.runtime_live_total || 0;
  var runtimeBlocked = lang.runtime_blocked_by_guard_count || 0;
  var runtimeUnpromoted = lang.runtime_unpromoted_live_attempts || 0;
  var runtimeRedLive = lang.runtime_live_red_classes || 0;
  var runtimeModeColor = runtimeMode === 'full' ? '#0f9' : runtimeMode === 'canary' ? '#ff0' : '#6a6a80';
  var runtimeBadge = runtimeEnabled ? runtimeMode.toUpperCase() : 'OFF';
  if (runtimeEnabled || runtimeGuardTotal || runtimeLiveTotal || runtimeBlocked || runtimeUnpromoted || runtimeRedLive) {
    runtimeHtml += langMiniTitle('Runtime Guard') +
      '<div style="font-size:0.55rem;color:#6a6a80;margin:-15px 0 2px 92px;">' +
      '<span style="color:' + runtimeModeColor + ';font-weight:bold;">' + esc(runtimeBadge) + '</span></div>';
    runtimeHtml += '<div class="language-compact-grid" style="display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:3px;margin-bottom:4px;">' +
      langStat('Guard Events', runtimeGuardTotal, '#0cf') +
      langStat('Live Native', runtimeLiveTotal, '#0f9') +
      langStat('Blocked', runtimeBlocked, runtimeBlocked > 0 ? '#ff0' : '#6a6a80') +
      langStat('Unpromoted', runtimeUnpromoted, runtimeUnpromoted > 0 ? '#f44' : '#0f9') +
      langStat('Live Red', runtimeRedLive, runtimeRedLive > 0 ? '#f44' : '#0f9') +
      langStat('Live Rate', fmtPct(lang.runtime_live_rate || 0, 1), '#0af') +
      langStat('Blocked Rate', fmtPct(lang.runtime_blocked_rate || 0, 1), '#ff0') +
      langStat('Runtime Last', lang.runtime_last_ts ? timeAgo(lang.runtime_last_ts) : '--', '#6a6a80') +
      '</div>';
    var runtimeLiveByClass = lang.runtime_live_by_class || {};
    var runtimeBlockedByClass = lang.runtime_blocked_by_class || {};
    if (Object.keys(runtimeLiveByClass).length || Object.keys(runtimeBlockedByClass).length) {
      runtimeHtml += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Runtime By Class</div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:6px;">';
      Object.keys(runtimeLiveByClass).forEach(function(rc) {
        runtimeHtml += '<span style="padding:1px 4px;border:1px solid #0f933;color:#0f9;border-radius:2px;font-size:0.45rem;">' +
          esc(rc) + ':live ' + (runtimeLiveByClass[rc] || 0) + '</span>';
      });
      Object.keys(runtimeBlockedByClass).forEach(function(rc) {
        runtimeHtml += '<span style="padding:1px 4px;border:1px solid #ff022;color:#ff0;border-radius:2px;font-size:0.45rem;">' +
          esc(rc) + ':blocked ' + (runtimeBlockedByClass[rc] || 0) + '</span>';
      });
      runtimeHtml += '</div>';
    }
  }

  // Phase C: Shadow Language Model
  var shadowHtml = '';
  var sm = lang.shadow_model || {};
  if (sm.trained) {
    shadowHtml += langMiniTitle('Shadow Style Model') + '<div style="font-size:0.55rem;color:#6a6a80;margin:-15px 0 2px 118px;">' +
      '<span style="color:#0f9;font-weight:bold;">TRAINED</span></div>';
    shadowHtml += '<div class="language-compact-grid" style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:3px;margin-bottom:4px;">';
    shadowHtml += langStat('Corpus', sm.corpus_size || 0, '#0af');
    var classCt = sm.class_counts || {};
    shadowHtml += langStat('Classes', Object.keys(classCt).length + '/7', '#0af');
    var age = sm.built_at ? Math.round((Date.now() / 1000 - sm.built_at) / 3600) + 'h ago' : '--';
    shadowHtml += langStat('Trained', age, '#6a6a80');
    shadowHtml += '</div>';
  } else {
    shadowHtml += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:4px;">Shadow Style Model ' +
      '<span style="color:#6a6a80;">NOT TRAINED</span>' +
      (sm.corpus_size ? ' (' + sm.corpus_size + ' examples)' : '') + '</div>';
  }

  var phasecHtml = '';
  if (Object.keys(phasec).length) {
    var baselineVer = phasecBaseline.baseline_version || '--';
    var tokStrat = phasecTokenizer.strategy || '--';
    var tokVocab = phasecTokenizer.estimated_vocab_size || 0;
    var dsCount = phasecDataset.total_samples || 0;
    var splitTrain = phasecSplit.train_count || 0;
    var splitVal = phasecSplit.val_count || 0;
    var studentState = phasecStudent.available ? 'READY' : (phasecStudent.reason || 'NOT_READY');
    var studentColor = phasecStudent.available ? '#0f9' : '#f90';
    phasecHtml += langMiniTitle('Phase C Harness') + '<div style="font-size:0.55rem;color:#6a6a80;margin:-15px 0 2px 104px;">' +
      '<span style="color:#0cf;">' + esc(baselineVer) + '</span></div>';
    phasecHtml += '<div class="language-compact-grid" style="display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:3px;margin-bottom:4px;">' +
      langStat('Tokenizer', tokStrat, '#0af') +
      langStat('Vocab Est', tokVocab, '#0af') +
      langStat('Dataset', dsCount, '#c0f') +
      langStat('Split T/V', splitTrain + '/' + splitVal, '#6a6a80') +
      langStat('Student', studentState, studentColor) +
      langStat('PPL(val)', phasecStudent.val_perplexity != null ? fmtNum(phasecStudent.val_perplexity, 3) : '--', '#ff0') +
      langStat('Epochs', phasecStudent.epochs || 0, '#6a6a80') +
      langStat('Last Train', phasecLastRun.ts ? timeAgo(phasecLastRun.ts) : '--', '#6a6a80') +
      '</div>';
  }

  return _panel('Language Substrate', '<div class="language-compact-panel">' + statsHtml + nativeHtml + gateHtml + promoHtml + runtimeHtml + shadowHtml + phasecHtml + routeHtml + classHtml + extraHtml + '</div>',
    '<span style="font-size:0.6rem;color:#6a6a80;">' + (lang.total_examples || 0) + ' examples</span>');
}


function _renderExplainabilityPanel(snap) {
  var expl = snap.explainability || {};
  var traces = expl.recent_traces || [];
  if (!traces.length) return _panel('Explainability', _emptyMsg('No provenance traces yet'));

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Traces', expl.trace_count || 0) +
    _statCard('Latest', traces.length ? timeAgo(traces[0].ts) : '--', '#0cf') +
    _statCard('Sources', _countUniqueSources(traces), '#c0f') +
    '</div>';

  var traceHtml = '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Recent Provenance Traces</div>';
  traceHtml += '<div style="max-height:180px;overflow-y:auto;">';
  traces.forEach(function(t) {
    var prov = t.provenance || {};
    var verdict = prov.provenance || prov.source || '--';
    var conf = prov.confidence != null ? fmtPct(prov.confidence, 0) : '--';
    var native = prov.native ? 'bounded' : 'llm';
    var rc = prov.response_class || '--';
    var outcomeCol = t.outcome === 'success' ? '#0f9' : t.outcome === 'failure' ? '#f44' : '#6a6a80';
    var nativeCol = prov.native ? '#0f9' : '#ff0';

    traceHtml += '<div style="display:flex;align-items:center;gap:4px;padding:3px 4px;border-bottom:1px solid #1a1a2e;font-size:0.55rem;">' +
      '<span style="color:#6a6a80;min-width:36px;">' + timeAgo(t.ts) + '</span>' +
      '<span style="color:' + nativeCol + ';min-width:44px;font-weight:600;">' + esc(native) + '</span>' +
      '<span style="color:#e0e0e8;flex:1;">' + esc(rc) + '</span>' +
      '<span style="color:#c0f;">' + esc(conf) + '</span>' +
      '<span style="color:' + outcomeCol + ';">' + esc(t.outcome || '--') + '</span>' +
      '</div>';
  });
  traceHtml += '</div>';

  // Provenance verdict distribution
  var verdictCounts = {};
  traces.forEach(function(t) {
    var v = (t.provenance || {}).provenance || 'unknown';
    verdictCounts[v] = (verdictCounts[v] || 0) + 1;
  });
  var distHtml = '';
  if (Object.keys(verdictCounts).length > 1) {
    distHtml = '<div style="font-size:0.58rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Verdict Distribution</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:2px;">';
    Object.entries(verdictCounts).forEach(function(e) {
      distHtml += '<span style="padding:1px 3px;border:1px solid #c0f22;color:#c0f;border-radius:2px;font-size:0.45rem;">' + esc(e[0]) + ':' + e[1] + '</span>';
    });
    distHtml += '</div>';
  }

  return _panel('Explainability', statsHtml + traceHtml + distHtml,
    _statusBadge('Phase 6.4'));
}

function _countUniqueSources(traces) {
  var seen = {};
  traces.forEach(function(t) {
    var v = (t.provenance || {}).provenance || 'unknown';
    seen[v] = true;
  });
  return Object.keys(seen).length;
}


function _renderPolicyPanel(snap) {
  var pol = snap.policy || {};
  var expBuf = snap.experience_buffer || {};

  var mode = pol.mode || pol.status || 'not loaded';
  var arch = pol.arch || pol.architecture || '--';
  var winRate = pol.nn_win_rate != null ? pol.nn_win_rate : (pol.win_rate || null);
  var decisiveWR = pol.nn_decisive_win_rate;

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Mode', mode, mode === 'live' ? '#0f9' : '#c0f') +
    _statCard('Architecture', arch, '#0cf') +
    _statCard('Version', 'v' + (pol.model_version || pol.registry_active_version || '--'), '#6a6a80') +
    _statCard('Win Rate', winRate != null ? (winRate * 100).toFixed(1) + '%' : '--', winRate > 0.5 ? '#0f9' : '#ff0') +
    '</div>';

  statsHtml += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Decisions', pol.decisions_total || 0, '#0cf') +
    _statCard('Shadow A/B', pol.shadow_ab_total || 0, '#6a6a80') +
    _statCard('NN Wins', pol.shadow_nn_wins || 0, '#0f9') +
    _statCard('Kernel Wins', pol.shadow_kernel_wins || 0, '#f90') +
    '</div>';

  // Training stats
  var trainHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Training</div>';
  trainHtml += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Runs', pol.train_runs_total || 0) +
    _statCard('Loss', pol.last_train_loss != null ? fmtNum(pol.last_train_loss, 4) : '--', '#6a6a80') +
    _statCard('Epochs', pol.last_train_epochs || '--', '#6a6a80') +
    _statCard('Eval Score', pol.last_eval_score != null ? (pol.last_eval_score * 100).toFixed(1) + '%' : '--', pol.last_eval_score > 0.8 ? '#0f9' : '#ff0') +
    '</div>';

  // Feature flags
  var ff = pol.feature_flags || {};
  var ffEntries = Object.entries(ff);
  var ffHtml = '';
  if (ffEntries.length) {
    ffHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Feature Flags</div>';
    ffHtml += '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">';
    ffEntries.forEach(function(e) {
      var name = e[0].replace(/_/g, ' '), enabled = e[1];
      var c = enabled ? '#0f9' : '#484860';
      ffHtml += '<span style="padding:1px 5px;border:1px solid ' + c + '44;color:' + c + ';border-radius:3px;font-size:0.52rem;">' +
        (enabled ? '\u2713' : '\u2717') + ' ' + esc(name) + '</span>';
    });
    ffHtml += '</div>';
  }

  // Performance
  var perfHtml = '<div style="display:flex;flex-wrap:wrap;gap:8px;font-size:0.55rem;color:#6a6a80;margin-bottom:4px;">' +
    '<span>p50: ' + (pol.decision_ms_p50_ema != null ? fmtNum(pol.decision_ms_p50_ema, 2) + 'ms' : '--') + '</span>' +
    '<span>p95: ' + (pol.decision_ms_p95_ema != null ? fmtNum(pol.decision_ms_p95_ema, 2) + 'ms' : '--') + '</span>' +
    '<span>dec/s: ' + (pol.decisions_per_s_ema != null ? fmtNum(pol.decisions_per_s_ema, 2) : '--') + '</span>' +
    '<span>margin: ' + (pol.win_margin_ema != null ? fmtNum(pol.win_margin_ema, 4) : '--') + '</span>' +
    '<span>passes: ' + (pol.passes_total || 0) + '</span>' +
    '<span>blocks: ' + (pol.blocks_total || 0) + '</span>' +
    '<span>overruns: ' + (pol.overruns_total || 0) + '</span>' +
    '</div>';

  // Recent events
  var eventsHtml = '';
  var events = pol.recent_events || [];
  if (events.length) {
    eventsHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Recent Events (' + events.length + ')</div>';
    eventsHtml += '<div style="max-height:100px;overflow-y:auto;">';
    events.slice(-6).reverse().forEach(function(e) {
      var typeColors = { train_complete: '#0f9', model_registered: '#0cf', promoted: '#c0f', enable: '#0f9', disable: '#f44', rollback: '#f44' };
      var ec = typeColors[e.type] || '#6a6a80';
      eventsHtml += '<div style="font-size:0.5rem;padding:1px 0;border-bottom:1px solid #0d0d1a;">' +
        '<span style="color:' + ec + ';">[' + esc(e.type || '--') + ']</span> ' +
        '<span style="color:#8a8aa0;">' + esc((e.msg || '').substring(0, 60)) + '</span></div>';
    });
    eventsHtml += '</div>';
  }

  // Experience buffer
  var expBuf = snap.experience_buffer || {};
  var bufHtml = '';
  if (expBuf.size || expBuf.max_size) {
    bufHtml = '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Experience Buffer</div>' +
      '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:4px;">' +
      _statCard('Size', expBuf.size || 0, '#0cf') +
      _statCard('Max', expBuf.max_size || 0, '#6a6a80') +
      _statCard('Top Reward', expBuf.top_reward_magnitude != null ? fmtNum(expBuf.top_reward_magnitude, 2) : '--', '#0f9') +
      _statCard('Unflushed', expBuf.unflushed || 0, (expBuf.unflushed || 0) > 0 ? '#ff0' : '#0f9') +
      '</div>';
  }

  return _panel('Policy NN', statsHtml + trainHtml + ffHtml + perfHtml + bufHtml + eventsHtml, _statusBadge(pol.active ? 'active' : mode));
}


function _renderHemispherePanel(snap) {
  var h = snap.hemisphere || {};
  var hs = h.hemisphere_state || h;
  var totalNets = hs.total_networks || 0;
  var hemispheres = hs.hemispheres || [];
  var distill = (h.distillation || {});
  var teachers = distill.teachers || {};
  var broadcastSlots = h.broadcast_slots || [];
  var gapDet = h.gap_detector || {};
  var tier1Gating = h.tier1_gating || {};
  var expansion = h.expansion || {};

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Networks', totalNets) +
    _statCard('Parameters', hs.total_parameters ? hs.total_parameters.toLocaleString() : '0') +
    _statCard('Substrate', hs.active_substrate || 'rule-based', '#c0f') +
    _statCard('Migration', hs.overall_migration_readiness != null ? ((hs.overall_migration_readiness * 100).toFixed(1) + '%') : '--') +
    '</div>';

  statsHtml += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Cycles', h.cycle_count || 0, '#6a6a80') +
    _statCard('Broadcast', h.num_broadcast_slots || 4, '#0cf') +
    _statCard('Distill Signals', distill.total_signals || 0, '#c0f') +
    _statCard('Quarantined', distill.total_quarantined || 0, (distill.total_quarantined || 0) > 0 ? '#f90' : '#0f9') +
    '</div>';

  // Hemispheres
  var hemiHtml = '';
  if (hemispheres.length) {
    hemiHtml = '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Hemispheres</div>';
    hemiHtml += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:4px;margin-bottom:6px;">';
    hemispheres.forEach(function(hp) {
      var st = hp.status || 'idle';
      var stColors = { active: '#0f9', training: '#ff0', inactive: '#484860', idle: '#6a6a80' };
      var sc = stColors[st] || '#6a6a80';
      hemiHtml += '<div style="padding:4px 6px;background:#0d0d1a;border:1px solid ' + (st === 'active' ? sc + '44' : '#1a1a2e') + ';border-radius:3px;">' +
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">' +
        '<span style="font-weight:600;font-size:0.62rem;color:#0cf;">' + esc(hp.focus || '--') + '</span>' +
        '<span style="font-size:0.5rem;color:' + sc + ';">' + st + '</span></div>' +
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:2px;font-size:0.5rem;">' +
        '<span style="color:#6a6a80;">nets: ' + (hp.network_count || 0) + '</span>' +
        '<span style="color:#6a6a80;">gen: ' + (hp.evolution_generations || 0) + '</span>' +
        '<span style="color:#0f9;">val: ' + (hp.best_validation_accuracy != null ? (hp.best_validation_accuracy * 100).toFixed(1) + '%' : '--') + '</span>' +
        '<span style="color:#6a6a80;">loss: ' + (hp.best_loss != null ? fmtNum(hp.best_loss, 4) : '--') + '</span>' +
        '</div></div>';
    });
    hemiHtml += '</div>';
  }

  // Broadcast Slots
  var bsHtml = '';
  if (broadcastSlots.length) {
    bsHtml = '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Broadcast Slots</div>';
    bsHtml += '<div style="display:grid;grid-template-columns:repeat(' + Math.min(broadcastSlots.length, 4) + ',1fr);gap:4px;margin-bottom:6px;">';
    broadcastSlots.forEach(function(bs) {
      var val = bs.value != null ? bs.value : 0;
      var c = val > 0.8 ? '#0f9' : val > 0.4 ? '#ff0' : '#6a6a80';
      bsHtml += '<div style="padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;text-align:center;">' +
        '<div style="font-size:0.58rem;font-weight:600;color:' + c + ';">' + fmtNum(val, 3) + '</div>' +
        '<div style="font-size:0.45rem;color:#6a6a80;">' + esc(bs.name || '--') + '</div>' +
        '<div style="font-size:0.42rem;color:#484860;">score:' + fmtNum(bs.score, 3) + ' dwell:' + (bs.dwell || 0) + '</div></div>';
    });
    bsHtml += '</div>';
  }

  // Distillation Teachers
  var distillHtml = '';
  var teacherEntries = Object.entries(teachers);
  if (teacherEntries.length) {
    distillHtml = '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Distillation Teachers</div>';
    distillHtml += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:3px;margin-bottom:6px;">';
    teacherEntries.forEach(function(e) {
      var name = e[0], info = e[1];
      var total = info.total || 0;
      var buf = info.buffer_size || 0;
      var quar = info.quarantined || 0;
      var c = total > 100 ? '#0f9' : total > 0 ? '#ff0' : '#6a6a80';
      var barLabel = buf > 0 && buf !== total ? 'buffer ' + buf + '/500' : 'buffer';
      var meta = [];
      if (buf > 0 && buf !== total) meta.push('buf ' + buf);
      if (quar > 0) meta.push('<span style="color:#f90;">q ' + quar + '</span>');
      if (info.last_seen_s != null) meta.push(_rnd_ageStr(info.last_seen_s));
      distillHtml += '<div style="display:grid;grid-template-columns:minmax(0,1fr) 54px;gap:4px;align-items:center;padding:2px 4px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;min-width:0;">' +
        '<div style="min-width:0;">' +
        '<div style="display:flex;justify-content:space-between;gap:4px;align-items:center;margin-bottom:1px;">' +
        '<span title="' + esc(name) + '" style="font-size:0.48rem;color:#8a8aa0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(name) + '</span>' +
        '<span title="total signals" style="font-size:0.45rem;color:' + c + ';white-space:nowrap;">' + total + '</span>' +
        '</div>' +
        _barFill(buf, 500, c) +
        '</div>' +
        '<div style="text-align:right;line-height:1.15;">' +
        '<div style="font-size:0.46rem;color:#484860;">' + esc(barLabel) + '</div>' +
        '<div style="font-size:0.42rem;color:#6a6a80;white-space:nowrap;">' + (meta.length ? meta.join(' · ') : '--') + '</div>' +
        '</div></div>';
    });
    distillHtml += '</div>';
  }

  // Tier-1 Gating
  var t1Html = '';
  var failCounts = tier1Gating.failure_counts || {};
  var disabledSpecs = tier1Gating.disabled_for_session || [];
  if (Object.keys(failCounts).length) {
    t1Html = '<div style="font-size:0.65rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Tier-1 Gating (floor: ' + ((tier1Gating.accuracy_floor || 0) * 100) + '%, max fails: ' + (tier1Gating.max_failures || 3) + ')</div>';
    t1Html += '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:4px;">';
    Object.entries(failCounts).forEach(function(fc) {
      var isDisabled = disabledSpecs.indexOf(fc[0]) !== -1;
      var c = isDisabled ? '#f44' : fc[1] > 0 ? '#f90' : '#0f9';
      t1Html += '<span style="padding:1px 5px;border:1px solid ' + c + '44;color:' + c + ';border-radius:3px;font-size:0.52rem;">' +
        esc(fc[0]) + ': ' + fc[1] + (isDisabled ? ' DISABLED' : '') + '</span>';
    });
    t1Html += '</div>';
  }

  // Gap Detector
  var gapHtml = '';
  var gapDims = gapDet.dimensions || {};
  if (Object.keys(gapDims).length) {
    gapHtml = '<div style="font-size:0.65rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Cognitive Gaps (total emitted: ' + (gapDet.total_gaps_emitted || 0) + ')</div>';
    gapHtml += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:3px;">';
    Object.entries(gapDims).forEach(function(gd) {
      var name = gd[0], info = gd[1];
      var ema = info.ema || 0;
      var thresh = info.threshold || 0;
      var c = ema > thresh ? '#0f9' : '#f90';
      gapHtml += '<div style="padding:2px 4px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:2px;">' +
        '<div style="font-size:0.5rem;color:#e0e0e8;">' + esc(name.replace(/_/g, ' ')) + '</div>' +
        '<div style="font-size:0.48rem;color:' + c + ';">ema:' + fmtNum(ema, 2) + ' thresh:' + fmtNum(thresh, 2) + '</div></div>';
    });
    gapHtml += '</div>';
  }

  // Expansion
  var expHtml = '';
  if (expansion.triggered) {
    expHtml = '<div style="margin-top:4px;padding:4px;background:#0f922;border:1px solid #0f944;border-radius:4px;font-size:0.58rem;color:#0f9;">M6 Expansion Triggered · Slots: ' + (expansion.slot_count || 4) + '</div>';
  }

  return _panel('Hemisphere NNs', statsHtml + hemiHtml + bsHtml + distillHtml + t1Html + gapHtml + expHtml,
    '<span style="font-size:0.6rem;color:#6a6a80;">' + totalNets + ' nets · ' + (h.cycle_count || 0) + ' cycles</span>');
}


function _renderMatrixPanel(snap) {
  var mx = snap.matrix || {};
  var hem = snap.hemisphere || {};
  var specialists = hem.matrix_specialists || mx.specialists || [];
  var jobs = mx.jobs || [];
  var protocols = mx.protocols_used || {};
  var claimability = mx.claimability_summary || {};
  var hemExp = hem.expansion || {};

  var activeJobs = mx.active_matrix_jobs || 0;
  var completedJobs = mx.completed_matrix_jobs || 0;

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Active Jobs', activeJobs, activeJobs > 0 ? '#0f9' : '#6a6a80') +
    _statCard('Completed', completedJobs, '#0cf') +
    _statCard('Specialists', specialists.length) +
    _statCard('Broadcast Slots', hemExp.slot_count || hem.num_broadcast_slots || 4, '#c0f') +
    '</div>';

  // Expansion state
  var expHtml = '';
  if (hemExp.triggered) {
    expHtml = '<div style="padding:4px 8px;background:rgba(0,255,153,0.08);border:1px solid rgba(0,255,153,0.3);border-radius:4px;margin-bottom:6px;">' +
      '<span style="font-size:0.62rem;color:#0f9;font-weight:600;">M6 Expansion Active</span> · ' +
      '<span style="font-size:0.55rem;color:#6a6a80;">Slots expanded to ' + (hemExp.slot_count || 6) + '</span></div>';
  } else {
    expHtml = '<div style="font-size:0.55rem;color:#484860;margin-bottom:4px;">M6 Expansion: not triggered</div>';
  }

  // Matrix Specialists
  var specHtml = '';
  if (specialists.length) {
    specHtml = '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Matrix Specialists</div>';
    specHtml += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:4px;margin-bottom:6px;">';
    specialists.forEach(function(sp) {
      var impact = sp.impact || 0;
      var c = impact > 0.1 ? '#0f9' : impact > 0 ? '#ff0' : '#6a6a80';
      specHtml += '<div style="padding:4px 6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;">' +
        '<div style="font-weight:600;font-size:0.62rem;color:#0cf;">' + esc(sp.focus || '--') + '</div>' +
        '<div style="display:flex;gap:6px;font-size:0.5rem;margin-top:2px;">' +
        '<span style="color:' + c + ';">impact:' + fmtNum(impact, 3) + '</span>' +
        '<span style="color:#6a6a80;">' + esc(sp.lifecycle || sp.status || '--') + '</span>' +
        (sp.accuracy != null ? '<span>acc:' + (sp.accuracy * 100).toFixed(1) + '%</span>' : '') +
        '</div></div>';
    });
    specHtml += '</div>';
  } else {
    specHtml = '<div style="font-size:0.58rem;color:#484860;margin-bottom:4px;">No matrix specialists yet (requires Tier-1 promoted models)</div>';
  }

  // Matrix Jobs
  var jobHtml = '';
  if (jobs.length) {
    jobHtml = '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Matrix Jobs</div>';
    jobs.slice(0, 6).forEach(function(j) {
      var phases = ['assess', 'research', 'acquire', 'integrate', 'collect', 'train', 'verify', 'register'];
      var curPhase = j.phase || '';
      var phaseIdx = phases.indexOf(curPhase);
      jobHtml += '<div style="padding:4px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;margin-bottom:3px;">' +
        '<div style="display:flex;gap:6px;align-items:center;margin-bottom:3px;">' +
        '<span style="font-weight:600;font-size:0.62rem;color:#c0f;">' + esc(j.skill_name || j.skill_id || '--') + '</span>' +
        _statusBadge(j.status) + '</div>' +
        '<div style="display:flex;gap:1px;margin-bottom:2px;">';
      phases.forEach(function(p, i) {
        var pc = i < phaseIdx ? '#0f9' : i === phaseIdx ? '#ff0' : '#1a1a2e';
        jobHtml += '<div style="flex:1;height:3px;background:' + pc + ';border-radius:1px;" title="' + p + '"></div>';
      });
      jobHtml += '</div>' +
        '<div style="font-size:0.5rem;color:#6a6a80;">Phase: ' + esc(curPhase || '--') +
        (j.protocol_id ? ' · Protocol: ' + esc(j.protocol_id) : '') +
        (j.claimability ? ' · Claim: ' + esc(j.claimability) : '') + '</div></div>';
    });
  }

  // Protocols Used
  var protoHtml = '';
  var protoEntries = Object.entries(protocols);
  if (protoEntries.length) {
    protoHtml = '<div style="font-size:0.65rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Protocols Used</div>';
    protoHtml += '<div style="display:flex;flex-wrap:wrap;gap:3px;">';
    protoEntries.forEach(function(pe) {
      protoHtml += '<span style="padding:1px 5px;border:1px solid #0cf44;color:#0cf;border-radius:3px;font-size:0.52rem;">' +
        esc(pe[0]) + ': ' + pe[1] + '</span>';
    });
    protoHtml += '</div>';
  }

  // Claimability summary
  var claimHtml = '';
  var claimEntries = Object.entries(claimability);
  if (claimEntries.length) {
    claimHtml = '<div style="font-size:0.65rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Claimability</div>';
    claimHtml += '<div style="display:flex;flex-wrap:wrap;gap:3px;">';
    claimEntries.forEach(function(ce) {
      claimHtml += '<span style="padding:1px 5px;border:1px solid #c0f44;color:#c0f;border-radius:3px;font-size:0.52rem;">' +
        esc(ce[0]) + ': ' + esc(String(ce[1])) + '</span>';
    });
    claimHtml += '</div>';
  }

  return _panel('Matrix Protocol', statsHtml + expHtml + specHtml + jobHtml + protoHtml + claimHtml);
}


function _renderMLChartsPanel(snap) {
  var pt = snap.policy_training || {};
  var pol = snap.policy || {};
  var lossHistory = pt.loss_history || pol.training_loss_history || [];
  var rewardHistory = pt.reward_history || pol.reward_history || [];
  var winHistory = pt.win_rate_history || pol.win_rate_history || [];

  if (!lossHistory.length && !rewardHistory.length && !winHistory.length) {
    return _panel('ML Training', _emptyMsg('No training data yet'));
  }

  var body = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:6px;">';

  if (lossHistory.length) {
    var lastLoss = lossHistory[lossHistory.length - 1];
    var lossVal = typeof lastLoss === 'object' ? lastLoss.final_loss : lastLoss;
    body += _statCard('Loss', typeof lossVal === 'number' ? lossVal.toFixed(4) : '--', '#6a6a80');
  } else {
    body += _statCard('Loss', '--');
  }

  if (rewardHistory.length) {
    var lastReward = rewardHistory[rewardHistory.length - 1];
    var rewVal = typeof lastReward === 'object' ? lastReward.value : lastReward;
    body += _statCard('Reward', typeof rewVal === 'number' ? rewVal.toFixed(3) : '--', rewVal > 0 ? '#0f9' : '#6a6a80');
  } else {
    body += _statCard('Reward', '--');
  }

  if (winHistory.length) {
    var lastWin = winHistory[winHistory.length - 1];
    var winVal = typeof lastWin === 'object' ? lastWin.value : lastWin;
    body += _statCard('Win Rate', typeof winVal === 'number' ? (winVal * 100).toFixed(1) + '%' : '--', winVal > 0.5 ? '#0f9' : '#ff0');
  } else {
    body += _statCard('Win Rate', '--');
  }

  body += '</div>';

  body += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Loss Samples', lossHistory.length, '#6a6a80') +
    _statCard('Reward Samples', rewardHistory.length, '#6a6a80') +
    _statCard('Win Rate Samples', winHistory.length, '#6a6a80') +
    '</div>';

  // Training runs detail
  if (lossHistory.length && typeof lossHistory[0] === 'object') {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Training Runs</div>';
    body += '<div style="max-height:100px;overflow-y:auto;">';
    lossHistory.slice().reverse().forEach(function(run) {
      var epochs = run.epoch_losses ? run.epoch_losses.length : '--';
      body += '<div style="font-size:0.5rem;padding:1px 0;border-bottom:1px solid #0d0d1a;">' +
        '<span style="color:#0cf;">loss:' + (run.final_loss != null ? fmtNum(run.final_loss, 4) : '--') + '</span>' +
        ' <span style="color:#6a6a80;">epochs:' + epochs + '</span>' +
        ' <span style="color:#484860;">' + (run.timestamp ? timeAgo(run.timestamp) : '') + '</span></div>';
    });
    body += '</div>';
  }

  return _panel('ML Training', body);
}


function _renderAcquisitionPanel(snap) {
  var acq = snap.acquisition || {};
  if (!acq.enabled && !acq.total_count) return '';

  var stats = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Active', acq.active_count || 0, '#0cf') +
    _statCard('Total', acq.total_count || 0) +
    _statCard('Completed', acq.completed_count || 0, '#0f9') +
    _statCard('Failed', acq.failed_count || 0, (acq.failed_count || 0) > 0 ? '#f44' : '#6a6a80') +
    '</div>';

  // Scheduler status
  var sched = acq.scheduler || {};
  var schedHtml = '';
  if (sched.current_mode) {
    schedHtml = '<div style="font-size:0.55rem;color:#6a6a80;margin-bottom:6px;">Mode: <span style="color:#0cf;">' + esc(sched.current_mode) + '</span>';
    var pLevel = sched.pressure_level || 'normal';
    var pColor = pLevel === 'high' ? '#f44' : pLevel === 'elevated' ? '#ff0' : '#0f9';
    var pVal = sched.quarantine_pressure != null ? sched.quarantine_pressure.toFixed(2) : '--';
    schedHtml += ' &middot; Pressure: <span style="color:' + pColor + ';">' + pVal + ' (' + esc(pLevel) + ')</span>';
    var deferred = sched.deferred_lanes || [];
    var suppressed = sched.suppressed_lanes || [];
    var combined = deferred.concat(suppressed.filter(function(s) { return deferred.indexOf(s) === -1; }));
    if (combined.length) {
      schedHtml += ' &middot; Blocked: <span style="color:#ff0;">' + esc(combined.join(', ')) + '</span>';
    }
    schedHtml += '</div>';
  }

  // Pending approvals banner — shows full plan/verification detail before approve buttons
  var pendingHtml = '';
  var pending = acq.pending_approvals || [];
  if (pending.length) {
    pendingHtml = '<div style="background:#1a1a0a;border:1px solid #ff0;border-radius:4px;padding:8px;margin-bottom:8px;">';
    pendingHtml += '<div style="font-size:0.65rem;color:#ff0;font-weight:700;margin-bottom:6px;">&#9888; Pending Approvals (' + pending.length + ')</div>';
    pending.forEach(function(p) {
      var gateLabel = p.gate === 'plan_review' ? 'Plan Review' : 'Deploy Approval';
      pendingHtml += '<div style="border-top:1px solid #2a2a1a;padding:6px 0;">';
      pendingHtml += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">' +
        '<span style="font-size:0.6rem;color:#e0e0e8;flex:1;font-weight:600;">' + esc(p.title || '--') + '</span>' +
        '<span style="font-size:0.5rem;color:#8a8aaa;">T' + (p.risk_tier || 0) + ' &middot; ' + esc(p.outcome_class || '') + ' &middot; ' + esc(gateLabel) + '</span></div>';

      // Plan summary if available
      var ps = p.plan_summary;
      if (ps) {
        pendingHtml += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;padding:6px;margin-bottom:6px;font-size:0.55rem;">';
        pendingHtml += '<div style="color:#0cf;font-weight:600;margin-bottom:4px;">Plan Details</div>';
        // Show prior rejection feedback if this is a revision
        var pr = ps.prior_rejection;
        if (pr && pr.notes) {
          pendingHtml += '<div style="background:#1a0a0a;border:1px solid #f44;border-radius:3px;padding:5px;margin-bottom:6px;">';
          pendingHtml += '<div style="color:#f88;font-weight:600;font-size:0.5rem;margin-bottom:2px;">&#9888; Prior Rejection Feedback (addressed in this revision)</div>';
          pendingHtml += '<div style="color:#f88;font-size:0.5rem;margin-bottom:2px;">Category: ' + esc(pr.category || 'unknown') + '</div>';
          pendingHtml += '<div style="color:#e0e0e8;font-size:0.5rem;">' + esc(pr.notes) + '</div>';
          if (pr.changes && pr.changes.length) {
            pr.changes.forEach(function(ch) {
              var desc = typeof ch === 'string' ? ch : (ch.description || JSON.stringify(ch));
              pendingHtml += '<div style="color:#f88;font-size:0.5rem;padding-left:6px;">&bull; ' + esc(desc) + '</div>';
            });
          }
          pendingHtml += '</div>';
        }
        if (ps.technical_approach) {
          pendingHtml += '<div style="margin-bottom:3px;"><span style="color:#6a6a80;">Approach:</span> <span style="color:#e0e0e8;">' + esc(ps.technical_approach) + '</span></div>';
        } else {
          pendingHtml += '<div style="margin-bottom:3px;color:#f44;">&#9888; No technical approach specified — plan is empty</div>';
        }
        if (ps.risk_analysis) {
          pendingHtml += '<div style="margin-bottom:3px;"><span style="color:#6a6a80;">Risk:</span> <span style="color:#ff0;">' + esc(ps.risk_analysis) + '</span></div>';
        }
        if (ps.risk_level) {
          pendingHtml += '<div style="margin-bottom:3px;"><span style="color:#6a6a80;">Risk Level:</span> <span style="color:' + (ps.risk_level === 'high' ? '#f44' : ps.risk_level === 'medium' ? '#ff0' : '#0f9') + ';">' + esc(ps.risk_level) + '</span></div>';
        }
        if (ps.dependencies && ps.dependencies.length) {
          pendingHtml += '<div style="margin-bottom:3px;"><span style="color:#6a6a80;">Dependencies:</span> <span style="color:#e0e0e8;">' + esc(ps.dependencies.join(', ')) + '</span></div>';
        }
        if (ps.test_cases && ps.test_cases.length) {
          pendingHtml += '<div style="margin-bottom:3px;"><span style="color:#6a6a80;">Tests (' + ps.test_cases.length + '):</span>';
          ps.test_cases.forEach(function(tc) { pendingHtml += '<div style="padding-left:8px;color:#e0e0e8;">&bull; ' + esc(typeof tc === 'string' ? tc : tc.description || JSON.stringify(tc)) + '</div>'; });
          pendingHtml += '</div>';
        } else {
          pendingHtml += '<div style="margin-bottom:3px;color:#ff0;">&#9888; No test cases defined</div>';
        }
        if (ps.implementation_sketch) {
          pendingHtml += '<div style="margin-bottom:3px;"><span style="color:#6a6a80;">Implementation:</span> <pre style="color:#e0e0e8;margin:2px 0;padding:4px;background:#0a0a14;border-radius:2px;font-size:0.5rem;white-space:pre-wrap;max-height:120px;overflow-y:auto;">' + esc(ps.implementation_sketch) + '</pre></div>';
        }
        pendingHtml += '<div style="color:#484860;">v' + (ps.version || 1) + ' &middot; ' + (ps.doc_count || 0) + ' doc artifacts</div>';
        pendingHtml += '</div>';
      }
      var diag = p.planning_diagnostics || {};
      if (diag.failure_reason) {
        pendingHtml += '<div style="background:#1a0a0a;border:1px solid #f44;border-radius:3px;padding:6px;margin-bottom:6px;font-size:0.55rem;">' +
          '<div style="color:#f88;font-weight:600;margin-bottom:3px;">Planning Diagnostics</div>' +
          '<div><span style="color:#6a6a80;">Reason:</span> <span style="color:#f88;">' + esc(diag.failure_reason) + '</span></div>' +
          '<div><span style="color:#6a6a80;">Raw output:</span> <span style="color:#e0e0e8;">' + esc(String(diag.raw_output_length || 0)) + ' chars</span></div>' +
          '<div><span style="color:#6a6a80;">Missing:</span> <span style="color:#e0e0e8;">' + esc((diag.missing_fields || []).join(', ') || 'none') + '</span></div>' +
          '</div>';
      }

      // Verification summary for deploy approvals
      var vs = p.verification_summary;
      if (vs) {
        pendingHtml += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;padding:6px;margin-bottom:6px;font-size:0.55rem;">';
        var passColor = vs.overall_passed ? '#0f9' : '#f44';
        pendingHtml += '<div style="color:' + passColor + ';font-weight:600;margin-bottom:4px;">' + (vs.overall_passed ? '&#10003; Verification Passed' : '&#10007; Verification Failed') + '</div>';
        var verdicts = vs.lane_verdicts || {};
        Object.keys(verdicts).forEach(function(ln) {
          var ok = verdicts[ln];
          pendingHtml += '<div style="display:flex;gap:6px;"><span style="color:#6a6a80;min-width:120px;">' + esc(ln) + '</span><span style="color:' + (ok ? '#0f9' : '#f44') + ';">' + (ok ? 'pass' : 'FAIL') + '</span></div>';
        });
        pendingHtml += '</div>';
      }

      // Action buttons
      var acqEsc = esc(p.acquisition_id);
      pendingHtml += '<div style="display:flex;flex-direction:column;gap:4px;">';
      if (p.gate === 'plan_review') {
        var planVer = (ps && ps.version) || 1;
        var planInvalid = ps && (
          !ps.technical_approach ||
          ps.technical_approach === 'Coder model returned empty response.' ||
          !ps.implementation_sketch ||
          !(ps.test_cases && ps.test_cases.length)
        );
        if (planVer > 1) {
          pendingHtml += '<div style="color:#ff0;font-size:0.5rem;margin-bottom:2px;">&#9888; Revision v' + planVer + ' — re-planned with operator feedback</div>';
        }
        if (planInvalid) {
          pendingHtml += '<div style="color:#f88;font-size:0.5rem;margin-bottom:2px;">&#9888; Plan is incomplete. Reject &amp; Revise before approval.</div>';
        }
        pendingHtml += '<div style="display:flex;gap:4px;align-items:center;">';
        pendingHtml += '<button ' + (planInvalid ? 'disabled title="Plan is incomplete; use Reject & Revise."' : 'onclick="window._acqApprovePlan(\'' + acqEsc + '\',\'approved_as_is\')"') + ' ' +
          'style="font-size:0.55rem;padding:3px 12px;background:#0a2a0a;border:1px solid #0f9;color:#0f9;border-radius:3px;cursor:' + (planInvalid ? 'not-allowed;opacity:0.45;' : 'pointer;') + '">&#10003; Approve Plan</button>';
        pendingHtml += '<button onclick="window._acqShowRejectForm(\'' + acqEsc + '\')" ' +
          'style="font-size:0.55rem;padding:3px 12px;background:#2a0a0a;border:1px solid #f44;color:#f44;border-radius:3px;cursor:pointer;">&#10007; Reject &amp; Revise</button>';
        pendingHtml += '</div>';
        // Hidden rejection form — toggled by _acqShowRejectForm
        pendingHtml += '<div id="acq-reject-form-' + acqEsc + '" style="display:none;background:#1a0a0a;border:1px solid #f44;border-radius:3px;padding:6px;margin-top:4px;">';
        pendingHtml += '<div style="color:#f88;font-size:0.55rem;font-weight:600;margin-bottom:4px;">Reject with Feedback</div>';
        pendingHtml += '<div style="margin-bottom:4px;"><label style="color:#8a8aaa;font-size:0.5rem;">Category:</label> ';
        pendingHtml += '<select id="acq-reject-cat-' + acqEsc + '" style="font-size:0.5rem;background:#0a0a14;color:#ccc;border:1px solid #333;border-radius:2px;padding:2px;">';
        pendingHtml += '<option value="incomplete_design">Incomplete Design</option>';
        pendingHtml += '<option value="wrong_approach">Wrong Approach</option>';
        pendingHtml += '<option value="missing_tests">Missing Tests</option>';
        pendingHtml += '<option value="security_concern">Security Concern</option>';
        pendingHtml += '<option value="scope_too_large">Scope Too Large</option>';
        pendingHtml += '<option value="needs_more_detail">Needs More Detail</option>';
        pendingHtml += '<option value="other">Other</option>';
        pendingHtml += '</select></div>';
        pendingHtml += '<div style="margin-bottom:4px;"><label style="color:#8a8aaa;font-size:0.5rem;">Feedback (what to keep, what to change, what to add):</label>';
        pendingHtml += '<textarea id="acq-reject-notes-' + acqEsc + '" rows="3" style="display:block;width:100%;font-size:0.5rem;background:#0a0a14;color:#e0e0e8;border:1px solid #333;border-radius:2px;padding:4px;resize:vertical;" placeholder="e.g. Keep the API endpoint design but add input validation and rate limiting. Also add unit tests for error cases."></textarea></div>';
        pendingHtml += '<div style="display:flex;gap:4px;">';
        pendingHtml += '<button onclick="window._acqSubmitReject(\'' + acqEsc + '\')" ' +
          'style="font-size:0.5rem;padding:3px 10px;background:#2a0a0a;border:1px solid #f44;color:#f44;border-radius:3px;cursor:pointer;">Submit Rejection</button>';
        pendingHtml += '<button onclick="document.getElementById(\'acq-reject-form-' + acqEsc + '\').style.display=\'none\';" ' +
          'style="font-size:0.5rem;padding:3px 10px;background:#0a0a14;border:1px solid #555;color:#888;border-radius:3px;cursor:pointer;">Cancel</button>';
        pendingHtml += '</div></div>';
      } else {
        pendingHtml += '<div style="display:flex;gap:4px;">';
        pendingHtml += '<button onclick="window._acqApproveDeploy(\'' + acqEsc + '\',true)" ' +
          'style="font-size:0.55rem;padding:3px 12px;background:#0a2a0a;border:1px solid #0f9;color:#0f9;border-radius:3px;cursor:pointer;">&#10003; Approve Deploy</button>' +
          '<button onclick="window._acqApproveDeploy(\'' + acqEsc + '\',false)" ' +
          'style="font-size:0.55rem;padding:3px 12px;background:#2a0a0a;border:1px solid #f44;color:#f44;border-radius:3px;cursor:pointer;">&#10007; Deny Deploy</button>';
        pendingHtml += '</div>';
      }
      pendingHtml += '</div></div>';
    });
    pendingHtml += '</div>';
  }

  // Recent jobs with expandable artifact detail
  var recentHtml = '';
  var recent = acq.recent || [];
  if (recent.length) {
    recentHtml = '<div style="font-size:0.6rem;color:#8a8aaa;margin-bottom:4px;">Recent Jobs</div>';
    recent.forEach(function(j, idx) {
      var stColor = j.status === 'completed' ? '#0f9' : j.status === 'failed' ? '#f44' :
        (j.status === 'awaiting_plan_review' || j.status === 'awaiting_approval') ? '#ff0' : '#0cf';
      var shortId = (j.acquisition_id || '').slice(0, 10);
      recentHtml += '<div style="border:1px solid #1a1a2e;border-radius:4px;margin-bottom:4px;background:#0d0d1a;">';
      recentHtml += '<div onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display===\'none\'?\'block\':\'none\'" ' +
        'style="display:flex;align-items:center;gap:6px;padding:4px 8px;cursor:pointer;font-size:0.6rem;">' +
        '<span style="color:#484860;">&#9654;</span>' +
        '<span style="font-family:monospace;font-size:0.55rem;color:#6a6a80;">' + esc(shortId) + '</span>' +
        '<span style="flex:1;color:#e0e0e8;">' + esc(j.title || '--') + '</span>' +
        '<span style="color:' + stColor + ';">' + esc(j.status || '--') + '</span>' +
        '<span style="color:#6a6a80;">T' + (j.risk_tier != null ? j.risk_tier : '-') + '</span>' +
        '<span style="color:#484860;font-size:0.5rem;">' + esc(j.outcome_class || '') + '</span>' +
        '</div>';
      // Expandable detail section
      recentHtml += '<div style="display:none;padding:6px 8px 8px 16px;font-size:0.55rem;">';

      // Lane progress table
      var lanes = j.lanes || {};
      var laneNames = Object.keys(lanes);
      if (laneNames.length) {
        recentHtml += '<div style="margin-bottom:6px;">';
        recentHtml += '<div style="color:#6a6a80;font-weight:600;margin-bottom:2px;">Lane Progress</div>';
        laneNames.forEach(function(ln) {
          var ls = lanes[ln];
          var lStatus = typeof ls === 'string' ? ls : (ls && ls.status ? ls.status : 'unknown');
          var lColor = lStatus === 'completed' ? '#0f9' : lStatus === 'failed' ? '#f44' :
            lStatus === 'running' ? '#0cf' : lStatus === 'skipped' ? '#484860' : '#8a8aaa';
          var lError = (typeof ls === 'object' && ls && ls.error) ? ls.error : '';
          var childId = (typeof ls === 'object' && ls && ls.child_id) ? ls.child_id : '';
          recentHtml += '<div style="display:flex;gap:6px;padding:1px 0;">' +
            '<span style="color:#6a6a80;min-width:120px;">' + esc(ln) + '</span>' +
            '<span style="color:' + lColor + ';min-width:60px;">' + esc(lStatus) + '</span>';
          if (childId) recentHtml += '<span style="color:#484860;font-family:monospace;">' + esc(childId.slice(0,16)) + '</span>';
          if (lError) recentHtml += '<span style="color:#f44;margin-left:4px;">' + esc(lError.slice(0, 80)) + '</span>';
          recentHtml += '</div>';
        });
        recentHtml += '</div>';
      }

      // Plan summary
      var ps = j.plan_summary;
      if (ps) {
        recentHtml += '<div style="background:#0a0a14;border:1px solid #1a1a2e;border-radius:3px;padding:6px;margin-bottom:6px;">';
        recentHtml += '<div style="color:#0cf;font-weight:600;margin-bottom:3px;">Plan</div>';
        if (ps.technical_approach) {
          recentHtml += '<div style="margin-bottom:2px;"><span style="color:#6a6a80;">Approach:</span> <span style="color:#e0e0e8;">' + esc(ps.technical_approach) + '</span></div>';
        } else {
          recentHtml += '<div style="color:#f44;margin-bottom:2px;">&#9888; No technical approach (plan not enriched)</div>';
        }
        if (ps.risk_analysis) {
          recentHtml += '<div style="margin-bottom:2px;"><span style="color:#6a6a80;">Risk:</span> <span style="color:#ff0;">' + esc(ps.risk_analysis) + '</span></div>';
        }
        if (ps.dependencies && ps.dependencies.length) {
          recentHtml += '<div><span style="color:#6a6a80;">Deps:</span> <span style="color:#e0e0e8;">' + esc(ps.dependencies.join(', ')) + '</span></div>';
        }
        if (ps.test_cases && ps.test_cases.length) {
          recentHtml += '<div><span style="color:#6a6a80;">Tests:</span>';
          ps.test_cases.forEach(function(tc) { recentHtml += ' <span style="color:#e0e0e8;">&bull; ' + esc(typeof tc === 'string' ? tc : tc.description || JSON.stringify(tc)) + '</span>'; });
          recentHtml += '</div>';
        }
        if (ps.implementation_sketch) {
          recentHtml += '<div style="margin-top:3px;"><span style="color:#6a6a80;">Implementation:</span><pre style="color:#e0e0e8;margin:2px 0;padding:4px;background:#080812;border-radius:2px;font-size:0.5rem;white-space:pre-wrap;max-height:100px;overflow-y:auto;">' + esc(ps.implementation_sketch) + '</pre></div>';
        }
        recentHtml += '</div>';
      }

      var diag = j.planning_diagnostics || {};
      if (diag.failure_reason) {
        recentHtml += '<div style="background:#1a0a0a;border:1px solid #f44;border-radius:3px;padding:6px;margin-bottom:6px;">' +
          '<div style="color:#f88;font-weight:600;margin-bottom:3px;">Planning Diagnostics</div>' +
          '<div><span style="color:#6a6a80;">Reason:</span> <span style="color:#f88;">' + esc(diag.failure_reason) + '</span></div>' +
          '<div><span style="color:#6a6a80;">Raw output:</span> <span style="color:#e0e0e8;">' + esc(String(diag.raw_output_length || 0)) + ' chars</span></div>' +
          '<div><span style="color:#6a6a80;">Missing:</span> <span style="color:#e0e0e8;">' + esc((diag.missing_fields || []).join(', ') || 'none') + '</span></div>' +
          '</div>';
      }

      // Code bundle summary
      var cb = j.code_bundle_summary;
      if (cb) {
        recentHtml += '<div style="background:#0a0a14;border:1px solid #1a1a2e;border-radius:3px;padding:6px;margin-bottom:6px;">';
        recentHtml += '<div style="color:#0cf;font-weight:600;margin-bottom:3px;">Generated Code</div>';
        if (cb.code_files && cb.code_files.length) {
          cb.code_files.forEach(function(f) {
            recentHtml += '<div style="color:#e0e0e8;font-family:monospace;">&bull; ' + esc(f) + '</div>';
          });
        } else {
          recentHtml += '<div style="color:#f44;">&#9888; No code files generated</div>';
        }
        if (cb.code_hash) {
          recentHtml += '<div style="color:#484860;margin-top:2px;">hash: ' + esc(cb.code_hash.slice(0,16)) + '&hellip;</div>';
        }
        recentHtml += '<button onclick="window._acqViewCode(\'' + esc(j.acquisition_id) + '\')" style="font-size:0.5rem;padding:2px 8px;margin-top:4px;background:#0d0d1a;border:1px solid #0cf;color:#0cf;border-radius:3px;cursor:pointer;">View Full Code</button>';
        recentHtml += '</div>';
      }

      // Verification summary
      var vs = j.verification_summary;
      if (vs) {
        var passColor = vs.overall_passed ? '#0f9' : '#f44';
        recentHtml += '<div style="background:#0a0a14;border:1px solid #1a1a2e;border-radius:3px;padding:6px;margin-bottom:6px;">';
        recentHtml += '<div style="color:' + passColor + ';font-weight:600;margin-bottom:3px;">' + (vs.overall_passed ? '&#10003; Verification Passed' : '&#10007; Verification Failed') + '</div>';
        var verdicts = vs.lane_verdicts || {};
        Object.keys(verdicts).forEach(function(ln) {
          var ok = verdicts[ln];
          recentHtml += '<div style="display:flex;gap:6px;"><span style="color:#6a6a80;min-width:120px;">' + esc(ln) + '</span><span style="color:' + (ok ? '#0f9' : '#f44') + ';">' + (ok ? 'pass' : 'FAIL') + '</span></div>';
        });
        recentHtml += '</div>';
      }

      // Plugin ID reference
      if (j.plugin_id) {
        recentHtml += '<div style="color:#484860;">Plugin: <span style="color:#8a8aaa;font-family:monospace;">' + esc(j.plugin_id) + '</span></div>';
      }

      // Cancel/dismiss button for failed, cancelled, or stale jobs
      if (j.status === 'failed' || j.status === 'cancelled' || j.status === 'blocked') {
        recentHtml += '<div style="margin-top:6px;padding-top:6px;border-top:1px solid #1a1a2e;">';
        recentHtml += '<button onclick="window._acqCancelJob(\'' + esc(j.acquisition_id) + '\')" ' +
          'style="font-size:0.5rem;padding:3px 10px;background:#2a0a0a;border:1px solid #f44;color:#f44;border-radius:3px;cursor:pointer;">&#10007; Remove Job</button>';
        recentHtml += '</div>';
      }

      recentHtml += '</div></div>';
    });
  }

  // Stall detection alerts
  var stallHtml = '';
  var stalls = acq.stalled_jobs || [];
  if (stalls.length) {
    stallHtml = '<div style="margin-bottom:8px;">';
    stalls.forEach(function(s) {
      var sevColor = s.severity === 'error' ? '#f44' : s.severity === 'warn' ? '#ff0' : '#0cf';
      var sevIcon = s.severity === 'error' ? '&#9888;' : s.severity === 'warn' ? '&#9888;' : '&#9432;';
      var elapsed = s.elapsed_s > 3600 ? (s.elapsed_s / 3600).toFixed(1) + 'h' : s.elapsed_s > 60 ? Math.round(s.elapsed_s / 60) + 'm' : s.elapsed_s + 's';
      stallHtml += '<div style="display:flex;align-items:center;gap:6px;padding:4px 8px;background:#0d0d1a;border:1px solid ' + sevColor + ';border-radius:3px;margin-bottom:3px;font-size:0.55rem;">' +
        '<span style="color:' + sevColor + ';">' + sevIcon + '</span>' +
        '<span style="color:#e0e0e8;flex:1;">' + esc(s.title || '--') + '</span>' +
        '<span style="color:#6a6a80;">' + esc(s.current_lane || '') + '</span>' +
        '<span style="color:' + sevColor + ';">' + elapsed + '</span>';
      if (s.blocked_reason) {
        stallHtml += '<span style="color:#8a8aaa;">' + esc(s.blocked_reason.split(':')[0]) + '</span>';
      }
      if (s.next_expected_lane) {
        stallHtml += '<span style="color:#484860;">&rarr; ' + esc(s.next_expected_lane) + '</span>';
      }
      stallHtml += '</div>';
    });
    stallHtml += '</div>';
  }

  return _panel('Capability Acquisition', stats + schedHtml + pendingHtml + stallHtml + recentHtml, 'learning-acquisition');
}


function _renderSkillAcquisitionWeightRoomPanel(snap) {
  var wr = snap.skill_acquisition_weight_room || {};
  var sp = snap.skill_acquisition_specialist || {};
  if (!wr.enabled && !sp.enabled) return '';

  var run = wr.run || {};
  var latest = wr.latest_report || {};
  var gates = wr.gates || {};
  var profiles = wr.profiles || {};
  var dist = wr.distillation || {};
  var boundary = [
    'authority=' + (wr.authority || 'telemetry_only'),
    'synthetic_only=' + (wr.synthetic_only !== false),
    'live_influence=' + (wr.live_influence === true),
    'promotion_eligible=' + (wr.promotion_eligible === true)
  ].join(' | ');

  var progress = '';
  if (run.status === 'running') {
    var done = run.episodes_done || 0;
    var total = run.target_episodes || 0;
    progress = '<div style="margin-bottom:8px;">' +
      '<div style="display:flex;justify-content:space-between;font-size:0.58rem;color:#8a8aa0;margin-bottom:2px;">' +
      '<span>Running ' + esc(run.profile || '--') + '</span><span>' + done + '/' + total + ' episodes</span></div>' +
      _barFill(done, total || 1, '#c0f') +
      '</div>';
  }

  var stats = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Episodes', wr.synthetic_episodes_total || 0, '#c0f') +
    _statCard('Features', dist.features || wr.synthetic_features_total || 0, '#0cf') +
    _statCard('Labels', dist.labels || wr.synthetic_labels_total || 0, '#0f9') +
    _statCard('Maturity', sp.maturity || 'bootstrap', '#ff0') +
    '</div>';

  var gain = '';
  if (run.feature_gain || run.label_gain || run.report_path) {
    gain = '<div style="font-size:0.58rem;color:#8a8aa0;margin-bottom:6px;">' +
      'Last run gains: <span style="color:#0cf;">+' + (run.feature_gain || 0) + ' features</span>, ' +
      '<span style="color:#0f9;">+' + (run.label_gain || 0) + ' labels</span>' +
      (run.report_path ? ' · report: <span style="color:#6a6a80;">' + esc(run.report_path) + '</span>' : '') +
      '</div>';
  }

  var profileHtml = '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-bottom:8px;">';
  Object.keys(profiles).forEach(function(name) {
    var p = profiles[name] || {};
    var gate = gates[name] || {};
    var allowed = gate.allowed === true;
    var reasons = gate.blocked_reasons || [];
    var disabled = !allowed || run.status === 'running';
    var title = allowed ? p.description || '' : reasons.join(', ');
    profileHtml += '<div style="padding:6px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">';
    profileHtml += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">' +
      '<span style="font-size:0.66rem;color:#e0e0e8;font-weight:600;">' + esc(name) + '</span>' +
      _statusBadge(allowed ? 'allowed' : 'blocked') + '</div>';
    profileHtml += '<div style="font-size:0.52rem;color:#6a6a80;margin-bottom:4px;">' +
      (p.episode_count || 0) + ' episodes · records=' + (p.record_signals ? 'yes' : 'no') + '</div>';
    if (reasons.length) {
      profileHtml += '<div style="font-size:0.5rem;color:#f88;margin-bottom:4px;">Blocked: ' + esc(reasons.join(', ')) + '</div>';
    } else {
      profileHtml += '<div style="font-size:0.5rem;color:#8a8aa0;margin-bottom:4px;">' + esc(p.description || '') + '</div>';
    }
    profileHtml += '<button class="j-btn-sm" ' +
      (disabled ? 'disabled title="' + esc(title) + '" style="opacity:0.45;cursor:not-allowed;"' : 'onclick="window._runSkillAcqWeightRoom(\'' + esc(name) + '\')"') +
      '>Run ' + esc(name) + '</button>';
    profileHtml += '</div>';
  });
  profileHtml += '</div>';

  var latestHtml = '';
  if (latest && latest.profile_name) {
    latestHtml += '<div style="font-size:0.58rem;color:#8a8aa0;margin-bottom:6px;">Latest report: ' +
      '<span style="color:' + (latest.passed ? '#0f9' : '#f44') + ';">' + esc(latest.profile_name) + '</span>' +
      ' · episodes ' + (latest.episodes || 0) +
      ' · passed=' + (latest.passed ? 'true' : 'false') + '</div>';
    latestHtml += '<div style="font-size:0.52rem;color:#6a6a80;margin-bottom:6px;">Outcomes: ' +
      esc(JSON.stringify(latest.outcomes || {})) + '</div>';
    if ((latest.invariant_failures || []).length) {
      latestHtml += '<div style="font-size:0.52rem;color:#f88;margin-bottom:6px;">Invariant failures: ' +
        esc((latest.invariant_failures || []).join(', ')) + '</div>';
    }
  }

  var body = stats + progress + gain + profileHtml + latestHtml +
    '<div style="font-size:0.55rem;color:#8a8aa0;padding:6px;background:#0a0a14;border:1px solid #2a2a44;border-radius:4px;">' +
    '<b style="color:#ff0;">Truth boundary:</b> ' + esc(boundary) +
    '<br>Synthetic workouts can grow specialist telemetry only. They cannot verify skills, promote plugins, or unlock capability claims.' +
    '</div>';

  return _panel('Skill Acquisition Weight Room', body, _statusBadge(run.status || 'idle'), {
    panelId: 'training-skill-acquisition-weight-room',
    ownerTab: 'training'
  });
}


window._runSkillAcqWeightRoom = function(profile) {
  if (!confirm('Run synthetic skill-acquisition workout profile "' + profile + '"? Synthetic samples are telemetry-only and cannot verify capabilities.')) return;
  var btn = event && event.target;
  if (btn) { btn.disabled = true; btn.textContent = 'Starting...'; }
  fetch('/api/synthetic/skill-acquisition/run', {
    method: 'POST',
    headers: _authHeaders(),
    body: JSON.stringify({ profile: profile })
  }).then(function(r) {
    if (!r.ok) {
      return r.text().then(function(txt) {
        try { var d = JSON.parse(txt); return { ok: false, data: d }; }
        catch(e) { return { ok: false, data: { error: txt || ('HTTP ' + r.status) } }; }
      });
    }
    return r.json().then(function(data) { return { ok: true, data: data }; });
  }).then(function(res) {
    var data = res.data || {};
    if (!res.ok || data.started === false) {
      alert('Weight room blocked: ' + ((data.blocked_reasons || [data.error || data.detail || 'unknown']).join(', ')));
      return;
    }
    if (btn) { btn.textContent = 'Running...'; btn.style.color = '#0f9'; }
  }).catch(function(err) {
    alert('Weight room request failed: ' + err);
  });
};

// Acquisition action handlers
window._acqApprovePlan = function(acqId, verdict) {
  fetch('/api/acquisition/' + acqId + '/approve-plan', {
    method: 'POST', headers: _authHeaders(),
    body: JSON.stringify({verdict: verdict, notes: '', reason_category: 'other'})
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.error) alert('Error: ' + d.error);
    else alert('Plan approved — proceeding to implementation.');
  }).catch(function(e) { alert('Request failed: ' + e); });
};

window._acqShowRejectForm = function(acqId) {
  var form = document.getElementById('acq-reject-form-' + acqId);
  if (form) form.style.display = form.style.display === 'none' ? 'block' : 'none';
};

window._acqSubmitReject = function(acqId) {
  var notes = (document.getElementById('acq-reject-notes-' + acqId) || {}).value || '';
  var category = (document.getElementById('acq-reject-cat-' + acqId) || {}).value || 'other';
  if (!notes.trim()) {
    alert('Please provide feedback so the planner knows what to change.');
    return;
  }
  var payload = {
    verdict: 'rejected',
    notes: notes.trim(),
    reason_category: category,
    suggested_changes: [{description: notes.trim()}]
  };
  fetch('/api/acquisition/' + acqId + '/approve-plan', {
    method: 'POST', headers: _authHeaders(),
    body: JSON.stringify(payload)
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.error) { alert('Error: ' + d.error); return; }
    alert('Plan rejected — re-planning with your feedback. Next review will be a revision.');
    var form = document.getElementById('acq-reject-form-' + acqId);
    if (form) form.style.display = 'none';
  }).catch(function(e) { alert('Request failed: ' + e); });
};

window._acqApproveDeploy = function(acqId, approved) {
  fetch('/api/acquisition/' + acqId + '/approve-deploy', {
    method: 'POST', headers: _authHeaders(),
    body: JSON.stringify({approved: approved})
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.error) alert('Error: ' + d.error);
  }).catch(function(e) { alert('Request failed: ' + e); });
};

window._acqCancelJob = function(acqId) {
  if (!confirm('Remove acquisition job ' + acqId + '?')) return;
  fetch('/api/acquisition/' + acqId + '/cancel', {
    method: 'POST', headers: _authHeaders(),
    body: JSON.stringify({reason: 'operator_removed'})
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.error) alert('Error: ' + d.error);
    else alert('Job removed.');
  }).catch(function(e) { alert('Request failed: ' + e); });
};

window._acqViewCode = function(acqId) {
  fetch('/api/acquisition/' + acqId, { headers: _authHeaders() })
    .then(function(r) { return r.json(); })
    .then(function(d) {
      var cb = d.code_bundle;
      if (!cb) { alert('No code bundle found for this acquisition.'); return; }
      var files = cb.code_files || {};
      var fileNames = Object.keys(files);
      if (!fileNames.length) { alert('Code bundle exists but contains no files.'); return; }
      var content = '=== Acquisition: ' + acqId + ' ===\n';
      content += '=== Bundle: ' + (cb.bundle_id || '?') + ' ===\n';
      content += '=== Hash: ' + (cb.code_hash || '?') + ' ===\n\n';
      fileNames.forEach(function(fn) {
        content += '--- ' + fn + ' ---\n' + files[fn] + '\n\n';
      });
      if (cb.manifest) {
        content += '--- manifest ---\n' + JSON.stringify(cb.manifest, null, 2) + '\n';
      }
      var w = window.open('', '_blank', 'width=800,height=600');
      if (w) {
        w.document.title = 'Code: ' + acqId;
        var pre = w.document.createElement('pre');
        pre.style.cssText = 'background:#0d0d1a;color:#e0e0e8;padding:16px;margin:0;font-family:monospace;font-size:13px;white-space:pre-wrap;';
        pre.textContent = content;
        w.document.body.style.cssText = 'margin:0;background:#0d0d1a;';
        w.document.body.appendChild(pre);
      }
    })
    .catch(function(e) { alert('Failed to fetch code: ' + e); });
};


function _renderPluginRegistryPanel(snap) {
  var plugs = snap.plugins || {};
  var list = plugs.plugins || [];
  if (!list.length && !plugs.total) return '';

  var stats = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Total', plugs.total || 0) +
    _statCard('Active', plugs.active || 0, '#0f9') +
    _statCard('Quarantined', plugs.quarantined || 0, '#ff0') +
    _statCard('Disabled', plugs.disabled || 0, '#f44') +
    '</div>';

  var tableHtml = '';
  if (list.length) {
    tableHtml = '<table class="j-mini-table"><thead><tr><th>Plugin</th><th>State</th><th>Risk</th><th>Invocations</th><th>Success</th><th>Latency</th><th>Actions</th></tr></thead><tbody>';
    list.forEach(function(p) {
      var stColor = p.state === 'active' ? '#0f9' : p.state === 'quarantined' ? '#ff0' : p.state === 'shadow' ? '#0cf' : p.state === 'supervised' ? '#c0f' : '#f44';
      var successRate = (p.invoke_count || 0) > 0 ? ((p.success_count || 0) / p.invoke_count * 100).toFixed(0) + '%' : '--';
      var latency = (p.avg_latency_ms != null && p.avg_latency_ms > 0) ? p.avg_latency_ms.toFixed(0) + 'ms' : '--';
      var name = p.name || '--';
      var actions = '';
      if (p.state === 'quarantined') {
        actions = '<button onclick="window._pluginAction(\'' + esc(name) + '\',\'activate\')" ' +
          'style="font-size:0.5rem;padding:1px 6px;background:#0a1a2a;border:1px solid #0cf;color:#0cf;border-radius:3px;cursor:pointer;">Activate</button>';
      } else if (p.state === 'shadow' || p.state === 'supervised') {
        actions = '<button onclick="window._pluginAction(\'' + esc(name) + '\',\'promote\')" ' +
          'style="font-size:0.5rem;padding:1px 6px;background:#0a2a0a;border:1px solid #0f9;color:#0f9;border-radius:3px;cursor:pointer;margin-right:2px;">Promote</button>' +
          '<button onclick="window._pluginAction(\'' + esc(name) + '\',\'disable\')" ' +
          'style="font-size:0.5rem;padding:1px 6px;background:#2a0a0a;border:1px solid #f44;color:#f44;border-radius:3px;cursor:pointer;">Disable</button>';
      } else if (p.state === 'active') {
        actions = '<button onclick="window._pluginAction(\'' + esc(name) + '\',\'rollback\')" ' +
          'style="font-size:0.5rem;padding:1px 6px;background:#1a1a0a;border:1px solid #ff0;color:#ff0;border-radius:3px;cursor:pointer;margin-right:2px;">Rollback</button>' +
          '<button onclick="window._pluginAction(\'' + esc(name) + '\',\'disable\')" ' +
          'style="font-size:0.5rem;padding:1px 6px;background:#2a0a0a;border:1px solid #f44;color:#f44;border-radius:3px;cursor:pointer;">Disable</button>';
      } else if (p.state === 'disabled') {
        actions = '<button onclick="window._pluginAction(\'' + esc(name) + '\',\'activate\')" ' +
          'style="font-size:0.5rem;padding:1px 6px;background:#0a1a2a;border:1px solid #0cf;color:#0cf;border-radius:3px;cursor:pointer;">Re-enable</button>';
      }
      tableHtml += '<tr>' +
        '<td>' + esc(name) + '</td>' +
        '<td style="color:' + stColor + ';">' + esc(p.state || '--') + '</td>' +
        '<td>' + (p.risk_tier != null ? 'T' + p.risk_tier : '--') + '</td>' +
        '<td>' + (p.invoke_count || 0) + '</td>' +
        '<td>' + successRate + '</td>' +
        '<td>' + latency + '</td>' +
        '<td>' + actions + '</td>' +
        '</tr>';
    });
    tableHtml += '</tbody></table>';
  }

  return _panel('Plugin Registry', stats + tableHtml, 'learning-plugins');
}

window._pluginAction = function(name, action) {
  fetch('/api/plugins/' + encodeURIComponent(name) + '/' + action, {
    method: 'POST', headers: _authHeaders(), body: '{}'
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.error) alert('Error: ' + d.error);
  }).catch(function(e) { alert('Request failed: ' + e); });
};


function _renderCogGapsPanel(snap) {
  var hem = snap.hemisphere || {};
  var gd = hem.gap_detector || snap.gap_detector || {};
  var gaps = gd.recent_gaps || gd.gaps || gd.detected_gaps || [];

  if (!gaps.length && !gd.total_gaps_emitted && !Object.keys(gd.dimensions || {}).length) {
    return _panel('Cognitive Gaps', _emptyMsg('No gaps detected'));
  }

  var body = '';
  body += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Total Emitted', gd.total_gaps_emitted || 0, '#0cf') +
    _statCard('Dimensions', Object.keys(gd.dimensions || {}).length, '#6a6a80') +
    _statCard('Cooldowns', Object.keys(gd.cooldowns || {}).length, '#6a6a80') +
    '</div>';

  var dims = gd.dimensions || {};
  if (Object.keys(dims).length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Dimensions</div>';
    Object.entries(dims).forEach(function(e) {
      var name = e[0], val = e[1];
      var ema = typeof val === 'object' ? (val.ema || 0) : (val || 0);
      var thresh = typeof val === 'object' ? (val.threshold || 0) : 0;
      var dataPoints = typeof val === 'object' ? (val.data_points || 0) : 0;
      var consBelow = typeof val === 'object' ? (val.consecutive_below || 0) : 0;
      var c = ema > thresh ? '#0f9' : '#f90';
      body += '<div style="display:flex;align-items:center;gap:4px;padding:2px 0;">' +
        '<span style="min-width:110px;font-size:0.58rem;color:#6a6a80;">' + esc(name.replace(/_/g, ' ')) + '</span>' +
        _barFill(ema, 1, c) +
        '<span style="min-width:75px;text-align:right;font-size:0.5rem;color:' + c + ';">ema:' + fmtNum(ema, 2) + ' th:' + fmtNum(thresh, 2) + '</span>' +
        '<span style="font-size:0.42rem;color:#484860;">n=' + dataPoints + (consBelow > 0 ? ' \u2193' + consBelow : '') + '</span></div>';
    });
  }

  if (gaps.length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Recent Gaps</div>';
    gaps.slice(0, 6).forEach(function(g) {
      var dim = typeof g === 'object' ? (g.dimension || g.name || '') : String(g);
      var severity = typeof g === 'object' ? (g.severity || g.score || 0) : 0;
      body += '<div style="display:flex;gap:6px;padding:2px 0;font-size:0.58rem;">' +
        '<span>' + esc(dim) + '</span>' +
        (severity ? '<span style="color:' + (severity > 0.5 ? '#f44' : '#ff0') + ';">' + fmtNum(severity, 2) + '</span>' : '') +
        '</div>';
    });
  }

  return _panel('Cognitive Gaps', body);
}


function _renderOnboardingPanel(snap) {
  var ob = snap.onboarding || {};
  if (!ob.enabled && !ob.active) return '';

  var readiness = ob.readiness_latest || {};
  var readinessComposite = readiness.composite;
  var readinessMetrics = readiness.metrics || {};
  var currentStage = Number(ob.current_stage || ob.current_day || 0);
  var stages = ob.stages || ob.days || {};
  var currentStageInfo = stages[String(currentStage)] || stages[currentStage] || null;
  var currentTargets = currentStageInfo ? (currentStageInfo.checkpoint_targets || {}) : {};
  var currentMet = currentStageInfo ? (currentStageInfo.checkpoints_met || {}) : {};
  var currentMissingCount = Object.keys(currentTargets).filter(function(k) { return !currentMet[k]; }).length;

  var body = '';
  body += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Day', ob.current_day || 0, '#0cf') +
    _statCard('Stage', ob.current_stage || '--', '#c0f') +
    _statCard('Graduated', ob.graduated ? 'Yes' : 'No', ob.graduated ? '#0f9' : '#6a6a80') +
    _statCard('Readiness', readinessComposite != null ? (readinessComposite * 100).toFixed(0) + '%' : '--', readinessComposite > 0.8 ? '#0f9' : readinessComposite > 0.5 ? '#ff0' : '#f44') +
    '</div>';

  // Current stage info
  if (ob.current_stage_label) {
    body += '<div style="font-size:0.62rem;margin-bottom:4px;">' +
      '<span style="color:#6a6a80;">Current: </span><span style="color:#e0e0e8;">' + esc(ob.current_stage_label) + '</span>' +
      ' <span style="color:#484860;">· prompts: ' + (ob.prompts_this_stage || 0) + ' today: ' + (ob.prompts_today || 0) + '</span></div>';
  }

  body += _renderOnboardingCurrentTargets(ob, currentStageInfo);
  body += _renderOnboardingSayThis(currentStageInfo, currentStage, currentMissingCount);

  // 7-day progress bar
  var dayLabels = ob.day_labels || {};
  var stageLabels = ob.stage_labels || {};
  var daysCompleted = ob.days_completed || [];
  var stagesCompleted = ob.stages_completed || [];
  if (Object.keys(dayLabels).length || Object.keys(stageLabels).length) {
    body += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:3px;">Training Phases</div>';
    body += '<div style="display:grid;grid-template-columns:repeat(7,1fr);gap:2px;margin-bottom:6px;">';
    for (var d = 1; d <= 7; d++) {
      var ds = String(d);
      var dayDone = daysCompleted.indexOf(d) !== -1 || daysCompleted.indexOf(ds) !== -1;
      var isCurrent = (ob.current_day || 0) === d;
      var label = dayLabels[ds] || stageLabels[ds] || 'Phase ' + d;
      var borderC = isCurrent ? '#0cf' : dayDone ? '#0f9' : '#1a1a2e';
      var bgC = isCurrent ? 'rgba(0,204,255,0.06)' : '#0d0d1a';
      body += '<div style="padding:3px 2px;text-align:center;background:' + bgC + ';border:1px solid ' + borderC + ';border-radius:3px;">' +
        '<div style="font-size:0.5rem;font-weight:600;color:' + (dayDone ? '#0f9' : isCurrent ? '#0cf' : '#484860') + ';">P' + d + '</div>' +
        '<div style="font-size:0.38rem;color:#6a6a80;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%;" title="' + esc(label) + '">' + esc(label) + '</div></div>';
    }
    body += '</div>';
  }

  // Readiness metrics
  if (Object.keys(readinessMetrics).length) {
    body += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:3px;">Readiness Metrics</div>';
    body += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:2px;margin-bottom:4px;">';
    Object.entries(readinessMetrics).forEach(function(rm) {
      var val = rm[1];
      var isNum = typeof val === 'number' && val !== null;
      var numVal = isNum ? val : 0;
      var c = val == null ? '#484860' : numVal >= 0.8 ? '#0f9' : numVal >= 0.5 ? '#ff0' : '#f44';
      var displayVal = val == null ? '--' : isNum ? fmtNum(val, 2) : esc(String(val));
      body += '<div style="display:flex;justify-content:space-between;padding:1px 4px;font-size:0.5rem;background:#0d0d1a;border-radius:2px;">' +
        '<span style="color:#6a6a80;">' + esc(rm[0].replace(/_/g, ' ')) + '</span>' +
        '<span style="color:' + c + ';">' + displayVal + '</span></div>';
    });
    body += '</div>';
  }

  // Checkpoints met per phase
  var checkpoints = ob.checkpoints_met || {};
  if (Object.keys(checkpoints).length) {
    body += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:3px;">Checkpoints Met</div>';
    body += '<div style="display:flex;flex-direction:column;gap:3px;">';
    Object.entries(checkpoints).forEach(function(cp) {
      var phase = cp[0];
      var metricMap = cp[1];
      var phaseLabel = dayLabels[phase] || stageLabels[phase] || 'Phase ' + phase;
      if (typeof metricMap === 'object' && metricMap !== null) {
        var metricNames = Object.keys(metricMap);
        var passCount = metricNames.filter(function(k) { return metricMap[k]; }).length;
        body += '<div style="font-size:0.48rem;"><span style="color:#6a6a80;">P' + esc(phase) + ' ' + esc(phaseLabel) + ': </span>';
        metricNames.forEach(function(mk) {
          var passed = metricMap[mk];
          body += '<span style="padding:0 3px;margin-left:2px;border:1px solid ' + (passed ? '#0f9' : '#f44') + ';color:' + (passed ? '#0f9' : '#f44') + ';border-radius:2px;font-size:0.42rem;">' +
            esc(mk.replace(/_/g, ' ')) + (passed ? ' \u2713' : '') + '</span>';
        });
        body += '</div>';
      } else {
        body += '<span style="padding:1px 4px;border:1px solid #0f9;color:#0f9;border-radius:2px;font-size:0.48rem;">' +
          'Phase ' + esc(phase) + ': ' + esc(String(metricMap)) + '</span>';
      }
    });
    body += '</div>';
  }

  return _panel('Companion Training', body, _statusBadge(ob.graduated ? 'graduated' : (ob.active ? 'active' : 'inactive')));
}


function _renderImprovementHistoryPanel(snap) {
  var si = snap.self_improve || {};
  var convos = si.recent_conversations || snap.improvement_conversations || [];

  if (!si.total_improvements && !si.total_rollbacks && !convos.length) return '';

  var body = '';
  body += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Improvements', si.total_improvements || 0, '#0f9') +
    _statCard('Rollbacks', si.total_rollbacks || 0, (si.total_rollbacks || 0) > 0 ? '#f44' : '#0f9') +
    _statCard('Failures', si.total_failures || 0, '#6a6a80') +
    _statCard('Pending', si.pending_approvals || 0, (si.pending_approvals || 0) > 0 ? '#f90' : '#6a6a80') +
    '</div>';

  if (si.last_verification) {
    body += '<div style="font-size:0.55rem;color:#6a6a80;">Last verification: ' + (typeof si.last_verification === 'string' ? esc(si.last_verification) : timeAgo(si.last_verification)) + '</div>';
  }
  if (si.last_dry_run) {
    body += '<div style="font-size:0.55rem;color:#6a6a80;">Last dry run: ' + (typeof si.last_dry_run === 'string' ? esc(si.last_dry_run) : timeAgo(si.last_dry_run)) + '</div>';
  }

  if (convos.length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Improvement Conversations (' + convos.length + ')</div>';
    convos.slice(0, 5).forEach(function(c) {
      var desc = typeof c === 'object' ? (c.request || c.description || c.summary || c.id || '') : String(c);
      var status = typeof c === 'object' ? (c.status || '') : '';
      body += '<div style="padding:2px 0;border-bottom:1px solid #1a1a2e;font-size:0.6rem;">' +
        (status ? _statusBadge(status) + ' ' : '') +
        esc(desc.substring(0, 120)) + '</div>';
    });
  }

  return _panel('Improvement History', body);
}


/* ═══════════════════════════════════════════════════════════════════════════
   TAB: TRAINING — Operator-facing maturation guidance
   ═══════════════════════════════════════════════════════════════════════════ */

function _renderTrainingHero(snap) {
  var ob = snap.onboarding || {};
  var readiness = ob.readiness_latest || {};
  var composite = readiness.composite;
  var metrics = readiness.metrics || {};
  var se = snap.synthetic_exercise || {};
  var wr = snap.skill_acquisition_weight_room || {};
  var ev = snap.eval || {};
  var vp = ev.validation_pack || {};
  var targets = vp.language_evidence_targets || [];
  var gatesMissing = targets.filter(function(r) { return r && !r.current_ok; }).length;

  var pct = composite != null ? Math.round(composite * 100) : 0;
  var barColor = pct >= 80 ? '#0f9' : pct >= 50 ? '#ff0' : '#f44';

  var html = '<div style="margin-bottom:12px;padding:14px 16px;background:linear-gradient(135deg,#0a0a1a 0%,#0d1025 100%);border:1px solid #1a2a44;border-radius:8px;">';

  html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">' +
    '<div style="font-size:0.9rem;color:#e0e0e8;font-weight:700;">Training Progress</div>' +
    '<div style="font-size:0.62rem;color:#6a6a80;">' +
    (ob.graduated ? '<span style="color:#0f9;font-weight:700;">GRADUATED</span>' : 'Stage ' + (ob.current_stage || 0) + ' — ' + esc(ob.current_stage_label || 'Not started')) +
    '</div></div>';

  html += '<div style="position:relative;height:22px;background:#12121f;border-radius:6px;overflow:hidden;margin-bottom:10px;">' +
    '<div style="position:absolute;left:0;top:0;height:100%;width:' + pct + '%;background:linear-gradient(90deg,' + barColor + ',' + barColor + '88);border-radius:6px;transition:width 0.5s;"></div>' +
    '<div style="position:absolute;left:0;top:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:0.62rem;font-weight:700;color:#fff;">' +
    pct + '% Readiness</div></div>';

  html += '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;">' +
    _statCard('Stage', (ob.current_stage || 0) + '/7', '#c0f') +
    _statCard('Conversations', ob.conversation_count || ob.prompts_today || 0, '#0cf') +
    _statCard('Preferences', (metrics.preference_memories != null ? fmtNum(metrics.preference_memories, 0) : '--'), '#ff0') +
    _statCard('Gate Work', gatesMissing > 0 ? gatesMissing + ' needed' : 'Clear', gatesMissing > 0 ? '#f90' : '#0f9') +
    _statCard('Distillation', (se.distillation_records || 0) + (wr.synthetic_episodes_total || 0), '#c0f') +
    '</div>';

  html += '</div>';
  return html;
}


function _renderSyntheticTextExercisesPanel(snap) {
  var se = snap.synthetic_exercises || {};
  var exercises = se.exercises || {};
  var running = snap._synthetic_running || {};
  var exerciseCount = Object.keys(exercises).length;
  var runningCount = Object.keys(running).length;

  var allExercises = [
    {id: 'commitment', label: 'Commitment', desc: 'Intention truth layer regression'},
    {id: 'claim', label: 'Claim', desc: 'CapabilityGate accuracy'},
    {id: 'retrieval', label: 'Retrieval', desc: 'Memory ranker/salience pairs'},
    {id: 'world_model', label: 'World Model', desc: 'Causal engine rule coverage'},
    {id: 'contradiction', label: 'Contradiction', desc: 'Conflict classifier coverage'},
    {id: 'diagnostic', label: 'Diagnostic', desc: 'Detector encoder signals'},
    {id: 'plan_evaluator', label: 'Plan Evaluator', desc: 'Plan verdict predictions'}
  ];

  var body = '<div style="font-size:0.55rem;color:#8a8aa0;margin-bottom:8px;">' +
    'Text-only synthetic exercises that train specialist NNs without requiring live audio or user interaction.' +
    '</div>';

  body += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:6px;">';
  allExercises.forEach(function(ex) {
    var data = exercises[ex.id] || {};
    var isRunning = !!(running[ex.id] && running[ex.id].running);
    var hasRun = !!data.last_run;
    var passed = data.passed;
    var border = isRunning ? '#c0f' : hasRun ? (passed ? '#0f9' : passed === false ? '#f44' : '#1a1a2e') : '#1a1a2e';
    var statusColor = isRunning ? '#c0f' : hasRun ? (passed ? '#0f9' : passed === false ? '#f44' : '#6a6a80') : '#484860';
    var statusText = isRunning ? 'RUNNING' : hasRun ? (passed ? 'PASS' : passed === false ? 'FAIL' : 'RAN') : 'NOT RUN';

    body += '<div style="padding:8px;background:#0d0d1a;border:1px solid ' + border + ';border-radius:5px;">';
    body += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">' +
      '<span style="font-size:0.62rem;color:#e0e0e8;font-weight:600;">' + esc(ex.label) + '</span>' +
      '<span style="font-size:0.48rem;padding:1px 5px;border:1px solid ' + statusColor + ';color:' + statusColor + ';border-radius:2px;">' + statusText + '</span>' +
      '</div>';
    body += '<div style="font-size:0.48rem;color:#6a6a80;margin-bottom:6px;">' + esc(ex.desc) + '</div>';

    if (isRunning) {
      var runInfo = running[ex.id];
      var elapsed = Math.round((Date.now() / 1000 - (runInfo.started_at || 0)));
      body += '<div style="font-size:0.48rem;color:#c0f;margin-bottom:4px;">' +
        'Profile: <span style="color:#e0e0e8;">' + esc(runInfo.profile || '--') + '</span>' +
        ' · Elapsed: <span style="color:#e0e0e8;">' + elapsed + 's</span>' +
        '</div>';
    } else if (hasRun) {
      body += '<div style="font-size:0.48rem;color:#484860;margin-bottom:4px;">' +
        'Profile: <span style="color:#8a8aa0;">' + esc(data.profile || '--') + '</span>' +
        ' · Episodes: <span style="color:#8a8aa0;">' + (data.episodes || 0) + '</span>' +
        '</div>';
      if (data.last_run) {
        var ago = Math.round((Date.now() / 1000 - data.last_run) / 60);
        var agoLabel = ago < 60 ? ago + 'm ago' : Math.round(ago / 60) + 'h ago';
        body += '<div style="font-size:0.44rem;color:#484860;">Last: ' + agoLabel + '</div>';
      }
    }

    var btnDisabled = isRunning ? ' disabled style="opacity:0.45;cursor:not-allowed;"' : '';
    body += '<div style="margin-top:6px;display:flex;gap:4px;">' +
      '<button class="j-btn-sm"' + btnDisabled + ' onclick="window._runSyntheticExercise(\'' + esc(ex.id) + '\',\'smoke\')">Smoke</button>' +
      '<button class="j-btn-sm"' + btnDisabled + ' onclick="window._runSyntheticExercise(\'' + esc(ex.id) + '\',\'coverage\')">Coverage</button>' +
      '</div>';

    body += '</div>';
  });
  body += '</div>';

  var badge = runningCount > 0 ? _statusBadge('running') : exerciseCount > 0 ? _statusBadge('active') : _statusBadge('idle');
  return _panel('Text-Only Exercises', body, badge);
}

window._runSyntheticExercise = function(exercise, profile) {
  if (!confirm('Run synthetic ' + exercise + ' exercise with profile "' + profile + '"?')) return;
  var btn = event && event.target;
  if (btn) { btn.disabled = true; btn.textContent = 'Starting...'; }
  fetch('/api/synthetic/exercises/run', {
    method: 'POST',
    headers: _authHeaders(),
    body: JSON.stringify({ exercise: exercise, profile: profile })
  }).then(function(r) {
    if (!r.ok) {
      return r.text().then(function(txt) {
        try { var d = JSON.parse(txt); return { ok: false, data: d }; }
        catch(e) { return { ok: false, data: { error: txt || ('HTTP ' + r.status) } }; }
      });
    }
    return r.json().then(function(data) { return { ok: true, data: data }; });
  }).then(function(res) {
    if (!res.ok) {
      alert('Exercise blocked: ' + (res.data.error || res.data.detail || 'unknown'));
      return;
    }
    if (btn) { btn.textContent = 'Running...'; btn.style.color = '#0f9'; }
  }).catch(function(err) {
    alert('Request failed: ' + err);
  });
};


window.renderTraining = function(snap) {
  var root = document.getElementById('training-root');
  if (!root) return;

  fetch('/api/synthetic/exercises/running').then(function(r) {
    return r.ok ? r.json() : {};
  }).then(function(running) {
    snap._synthetic_running = running || {};
    _doRenderTraining(snap, root);
  }).catch(function() {
    snap._synthetic_running = {};
    _doRenderTraining(snap, root);
  });
};

function _doRenderTraining(snap, root) {
  var html = '';
  if (window._renderTabWayfinding) html += window._renderTabWayfinding('training');

  var ob = snap.onboarding || {};
  html += '<div class="j-toolbar">' +
    (ob.enabled && !ob.graduated && !ob.active ? '<button class="j-btn-sm" onclick="window.startOnboarding && window.startOnboarding()">Start Companion Training</button>' : '') +
    '<a class="j-btn-sm" href="/docs#companion-training" target="_blank" style="text-decoration:none;">Training Docs</a>' +
    '</div>';

  html += _renderTrainingHero(snap);

  html += '<div style="font-size:0.72rem;color:#c0f;font-weight:700;margin:12px 0 6px 0;border-bottom:1px solid #1a1a2e;padding-bottom:4px;">Interactive Training</div>';
  html += '<div style="font-size:0.5rem;color:#6a6a80;margin-bottom:8px;">Real conversations with Jarvis. Each interaction builds the identity, preference, and routing baselines that all neural networks learn from.</div>';

  html += _renderOnboardingPanel(snap);

  var manualGate = _renderManualGateWork(snap);
  if (manualGate) {
    html += _panel('Language Evidence Gates', manualGate +
      '<div style="font-size:0.48rem;color:#6a6a80;margin-top:6px;">These gates require real operator conversations that naturally produce each response class. Synthetic or passive runtime does not count.</div>');
  }

  html += '<div style="font-size:0.72rem;color:#0cf;font-weight:700;margin:16px 0 6px 0;border-bottom:1px solid #1a1a2e;padding-bottom:4px;">Synthetic Training</div>';
  html += '<div style="font-size:0.5rem;color:#6a6a80;margin-bottom:8px;">Automated exercises that train specialist NNs, test truth boundaries, and improve routing accuracy — no live user interaction required.</div>';

  if (window._renderSyntheticExercise) {
    html += window._renderSyntheticExercise(snap);
  }

  html += _renderSkillAcquisitionWeightRoomPanel(snap);

  html += _renderSyntheticTextExercisesPanel(snap);

  root.innerHTML = html;
}


/* ═══════════════════════════════════════════════════════════════════════════
   TAB: DIAGNOSTICS — Engineering deep-dive
   ═══════════════════════════════════════════════════════════════════════════ */

window.renderDiagnostics = function(snap) {
  var root = document.getElementById('diagnostics-root');
  if (!root) return;

  var html = '';
  if (window._renderTabWayfinding) html += window._renderTabWayfinding('diagnostics');

  // Diagnostic action toolbar
  html += '<div class="j-toolbar">' +
    '<button class="j-btn-sm" onclick="window.openLogViewer && window.openLogViewer()">Log Viewer</button>' +
    '<button class="j-btn-sm" onclick="window.openDebugSnapshot && window.openDebugSnapshot()">Debug Snapshot</button>' +
    '<button class="j-btn-sm" onclick="window.openTraceExplorer && window.openTraceExplorer()">Trace Explorer</button>' +
    '<button class="j-btn-sm" onclick="window.openChat && window.openChat()">Chat</button>' +
    '<button class="j-btn-sm" onclick="window.exportSoul && window.exportSoul()">Export Soul</button>' +
    '<button class="j-btn-sm" onclick="window.systemSave && window.systemSave()">Save State</button>' +
    '<button class="j-btn-sm j-btn-red" onclick="window.systemRestart && window.systemRestart()">Restart</button>' +
    '<button class="j-btn-sm j-btn-red" onclick="window.systemShutdown && window.systemShutdown()">Shutdown</button>' +
    '</div>';

  // 2-column grid for diagnostic panels
  html += '<div class="j-panel-grid">';
  html += _renderMovedTrustSurfacesPanel();
  html += _renderKernelPerfPanel(snap);
  html += _renderEventReliabilityPanel(snap);
  html += _renderEventValidationPanel(snap);
  html += _renderConsciousnessReportsPanel(snap);
  html += _renderCodebasePanel(snap);
  html += _renderTraitValidationPanel(snap);
  html += _renderMemoryRoutePanel(snap);
  html += _renderObserverPanel(snap);
  html += _renderEmergenceEvidencePanel(snap);
  html += _renderThoughtsPanel(snap);
  html += _renderMutationsPanel(snap);
  html += _renderEpistemicPanel(snap);
  html += _renderEvalSidecarPanel(snap);
  html += _renderSpatialDiagnosticsPanel(snap);
  html += _renderGestationPanel(snap);
  html += _renderHardwarePanel(snap);
  html += '</div>';

  root.innerHTML = html;
};

function _renderMovedTrustSurfacesPanel() {
  var links = [
    { id: 'l0-capability-gate', label: 'L0 Capability Gate' },
    { id: 'l1-attribution-ledger', label: 'L1 Attribution Ledger' },
    { id: 'l9-reflective-audit', label: 'L9 Reflective Audit' },
    { id: 'l10-soul-integrity', label: 'L10 Soul Integrity' },
    { id: 'trust-trace-explorer', label: 'Trace Explorer' },
    { id: 'trust-reconstructability', label: 'Reconstructability' }
  ];
  var body = '<div style="font-size:0.62rem;color:#8a8aa0;line-height:1.45;margin-bottom:8px;">' +
    'Epistemic-layer panels moved to Trust as canonical owner surfaces. Diagnostics now focuses on engineering internals.' +
    '</div>';
  body += '<div style="display:flex;gap:6px;flex-wrap:wrap;">';
  links.forEach(function(link) {
    body += '<button class="j-btn-xs j-owner-jump" onclick="window.openOwnedPanel && window.openOwnedPanel(\'' + esc(link.id) + '\')">' +
      esc(link.label) + ' \u2192 trust</button>';
  });
  body += '</div>';
  return _panel('Trust Surfaces (Moved)', body, '', {
    panelId: 'diagnostics-trust-moved-summary',
    ownerTab: 'diagnostics'
  });
}


function _renderSoulIntegrityPanel(snap) {
  var si = snap.soul_integrity || {};
  if (!Object.keys(si).length) return _panel('Soul Integrity', _emptyMsg('Not computed yet'));

  var idx = si.current_index;
  var c = idx > 0.7 ? '#0f9' : idx > 0.4 ? '#ff0' : '#f44';

  var body = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Index', idx != null ? (idx * 100).toFixed(1) + '%' : '--', c) +
    _statCard('Computations', si.total_computations || 0, '#6a6a80') +
    _statCard('Repairs', si.total_repairs_triggered || 0, (si.total_repairs_triggered || 0) > 0 ? '#f44' : '#0f9') +
    _statCard('Critical', si.critical ? 'YES' : 'No', si.critical ? '#f44' : '#0f9') +
    '</div>';

  var dims = si.dimensions || [];
  if (Array.isArray(dims) && dims.length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Dimensions (' + dims.length + ')</div>';
    var sorted = dims.slice().sort(function(a, b) { return (a.score || 0) - (b.score || 0); });
    sorted.forEach(function(d) {
      var score = d.score || 0;
      var dimC = score > 0.7 ? '#0f9' : score > 0.4 ? '#ff0' : '#f44';
      var staleTag = d.stale ? ' <span style="color:#f90;font-size:0.45rem;">[stale]</span>' : '';
      body += '<div style="display:flex;align-items:center;gap:6px;padding:1px 0;">' +
        '<span style="min-width:120px;font-size:0.55rem;color:#6a6a80;">' + esc((d.name || '').replace(/_/g, ' ')) + staleTag + '</span>' +
        _barFill(score, 1, dimC) +
        '<span style="min-width:36px;text-align:right;font-size:0.55rem;color:' + dimC + ';">' + (score * 100).toFixed(0) + '%</span>' +
        '<span style="min-width:18px;font-size:0.42rem;color:#484860;">w:' + fmtNum(d.weight || 0, 2) + '</span></div>';
      if (d.source) {
        body += '<div style="padding-left:126px;font-size:0.42rem;color:#484860;margin-top:-1px;">' + esc(d.source.substring(0, 80)) + '</div>';
      }
    });
  } else if (typeof dims === 'object' && !Array.isArray(dims) && Object.keys(dims).length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Dimensions</div>';
    Object.entries(dims).sort(function(a, b) {
      var va = typeof a[1] === 'object' ? (a[1].score || 0) : (a[1] || 0);
      var vb = typeof b[1] === 'object' ? (b[1].score || 0) : (b[1] || 0);
      return va - vb;
    }).forEach(function(e) {
      var score = typeof e[1] === 'object' ? (e[1].score || 0) : (e[1] || 0);
      var dimC = score > 0.7 ? '#0f9' : score > 0.4 ? '#ff0' : '#f44';
      body += '<div style="display:flex;align-items:center;gap:6px;padding:1px 0;">' +
        '<span style="min-width:120px;font-size:0.55rem;color:#6a6a80;">' + esc(e[0].replace(/_/g, ' ')) + '</span>' +
        _barFill(score, 1, dimC) +
        '<span style="min-width:36px;text-align:right;font-size:0.55rem;color:' + dimC + ';">' + (score * 100).toFixed(0) + '%</span></div>';
    });
  }

  if (si.weakest_dimension) {
    body += '<div style="margin-top:4px;font-size:0.55rem;color:#f44;">Weakest: ' + esc(si.weakest_dimension.replace(/_/g, ' ')) + ' (' + fmtPct(si.weakest_score) + ')</div>';
  }

  // Trend
  var trend = si.trend || {};
  if (trend.direction) {
    var trendC = trend.direction === 'improving' ? '#0f9' : trend.direction === 'declining' ? '#f44' : '#6a6a80';
    body += '<div style="font-size:0.55rem;margin-top:4px;">' +
      '<span style="color:' + trendC + ';">' + (trend.direction === 'improving' ? '\u2191' : trend.direction === 'declining' ? '\u2193' : '\u2192') + ' ' + esc(trend.direction) + '</span>' +
      (trend.delta != null ? ' <span style="color:#484860;">(\u0394 ' + fmtNum(trend.delta, 4) + ')</span>' : '') + '</div>';
  }

  // History sparkline
  var history = si.history || [];
  if (history.length > 2) {
    body += '<div style="font-size:0.52rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">History (' + history.length + ' snapshots)</div>';
    body += '<div style="display:flex;gap:1px;align-items:flex-end;height:20px;">';
    var maxH = Math.max.apply(null, history.map(function(h) { return typeof h === 'number' ? h : (h.value || h.index || 0); })) || 1;
    history.forEach(function(h) {
      var v = typeof h === 'number' ? h : (h.value || h.index || 0);
      var barH = Math.max(2, (v / maxH) * 18);
      var bc = v > 0.7 ? '#0f9' : v > 0.4 ? '#ff0' : '#f44';
      body += '<div style="flex:1;height:' + barH + 'px;background:' + bc + ';border-radius:1px;opacity:0.7;" title="' + fmtNum(v, 3) + '"></div>';
    });
    body += '</div>';
  }

  return _panel('Soul Integrity (L10)', body, '<span style="font-size:0.85rem;font-weight:700;color:' + c + ';">' + (idx != null ? (idx * 100).toFixed(0) + '%' : '--') + '</span>');
}


function _renderReflectiveAuditPanel(snap) {
  var ra = snap.reflective_audit || {};
  if (!Object.keys(ra).length) return _panel('Reflective Audit (L9)', _emptyMsg('Not active'));

  var lastScore = 0;
  var reports = ra.recent_reports || [];
  if (reports.length) lastScore = reports[0].score || 0;
  var sc = lastScore >= 0.8 ? '#0f9' : lastScore >= 0.5 ? '#ff0' : '#f44';

  var body = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Score', lastScore ? (lastScore * 100).toFixed(0) + '%' : '--', sc) +
    _statCard('Total Audits', ra.total_audits || 0, '#6a6a80') +
    _statCard('Findings', ra.total_findings || 0, (ra.total_findings || 0) > 0 ? '#f90' : '#0f9') +
    _statCard('Last', ra.last_audit_ts ? timeAgo(ra.last_audit_ts) : 'never', '#6a6a80') +
    '</div>';

  var findCounts = ra.finding_counts || {};
  if (Object.keys(findCounts).length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Finding Categories</div>';
    body += '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px;">';
    Object.entries(findCounts).forEach(function(e) {
      body += '<span style="font-size:0.52rem;padding:1px 5px;background:#1a1a2e;border-radius:3px;color:#f90;">' + esc(e[0].replace(/_/g,' ')) + ': ' + e[1] + '</span>';
    });
    body += '</div>';
  }

  if (reports.length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Recent Reports</div>';
    reports.slice(0, 3).forEach(function(r) {
      var rsc = (r.score || 0) >= 0.8 ? '#0f9' : '#ff0';
      body += '<div style="padding:3px 0;border-bottom:1px solid #1a1a2e;">';
      body += '<div style="display:flex;justify-content:space-between;font-size:0.55rem;">' +
        '<span style="color:' + rsc + ';">' + ((r.score || 0) * 100).toFixed(0) + '% · ' + (r.finding_count || 0) + ' findings · ' + (r.categories_scanned || 0) + ' categories</span>' +
        '<span style="color:#484860;">' + (r.duration_ms ? fmtNum(r.duration_ms, 1) + 'ms' : '') + '</span></div>';
      var findings = r.findings || [];
      findings.forEach(function(f) {
        var sevC = f.severity === 'critical' ? '#f44' : f.severity === 'warning' ? '#f90' : '#6a6a80';
        body += '<div style="font-size:0.5rem;padding:1px 0;margin-left:8px;">' +
          '<span style="color:' + sevC + ';">[' + esc(f.severity || 'info') + ']</span> ' +
          '<span style="color:#8a8aa0;">' + esc(f.category || '') + ':</span> ' +
          esc((f.description || '').substring(0, 100)) + '</div>';
        if (f.recommendation) {
          body += '<div style="font-size:0.45rem;padding-left:16px;color:#484860;">\u2192 ' + esc(f.recommendation.substring(0, 80)) + '</div>';
        }
      });
      body += '</div>';
    });
  }

  return _panel('Reflective Audit (L9)', body);
}


function _renderKernelPerfPanel(snap) {
  var k = snap.kernel || {};

  var body = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Avg Tick', k.avg_tick_ms != null ? k.avg_tick_ms + 'ms' : '--') +
    _statCard('P95 Tick', k.p95_tick_ms != null ? k.p95_tick_ms + 'ms' : '--', k.p95_tick_ms > 50 ? '#f44' : '#0f9') +
    _statCard('Max Tick', k.max_tick_ms != null ? k.max_tick_ms + 'ms' : '--') +
    '</div>';

  body += _metricRow('Tick Count', k.tick_count || 0);
  body += _metricRow('Budget Overruns', k.budget_overruns || 0);
  body += _metricRow('Deferred Backlog', k.deferred_backlog || 0);
  body += _metricRow('Slow Ticks', k.slow_ticks || 0);

  return _panel('Kernel Performance', body);
}


function _renderEventReliabilityPanel(snap) {
  var er = snap.event_reliability || {};
  if (!Object.keys(er).length) return _panel('Event Reliability', _emptyMsg('No data'));

  var body = '';
  body += _metricRow('Total Emits', er.total_emits || er.emit_count || 0);
  body += _metricRow('Errors', er.error_count || er.errors || 0);
  body += _metricRow('Circuit Breaker Trips', er.circuit_breaker_trips || er.cb_trips || 0);
  body += _metricRow('Active Breakers', er.active_breakers || 0);
  body += _metricRow('Recursive Guards', er.recursive_guard_count || er.recursive_guards || 0);

  var cb = er.circuit_breakers || {};
  if (Object.keys(cb).length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Circuit Breakers</div>';
    Object.entries(cb).forEach(function(e) {
      var name = e[0], info = e[1];
      var state = typeof info === 'object' ? (info.state || 'closed') : info;
      var sc = state === 'open' ? '#f44' : state === 'half_open' ? '#ff0' : '#0f9';
      body += '<div style="display:flex;gap:6px;padding:1px 0;font-size:0.6rem;">' +
        '<span style="min-width:140px;color:#6a6a80;">' + esc(name) + '</span>' +
        '<span style="color:' + sc + ';">' + esc(state) + '</span></div>';
    });
  }

  return _panel('Event Reliability', body);
}


function _renderEventValidationPanel(snap) {
  var ev = snap.event_validation || {};
  if (!Object.keys(ev).length) return _panel('Event Validation', _emptyMsg('No data'));

  var body = '';
  body += _metricRow('Total Validated', ev.total_validated || ev.total || 0);
  body += _metricRow('Passed', ev.passed || 0);
  body += _metricRow('Failed', ev.failed || ev.violations || 0);
  body += _metricRow('Warnings', ev.warnings || 0);

  return _panel('Event Validation', body);
}


function _renderConsciousnessReportsPanel(snap) {
  var cr = snap.consciousness_reports || {};
  var state = cr.state || {};
  var recent = cr.recent || [];

  if (!recent.length) return _panel('Consciousness Reports', _emptyMsg('No reports yet'));

  var body = _metricRow('Total Reports', state.total_reports || state.report_count || 0);

  recent.slice(0, 5).forEach(function(r) {
    var summary = r.summary || r.text || r.content || JSON.stringify(r).substring(0, 120);
    var ts = r.timestamp ? timeAgo(r.timestamp) : '';
    body += '<div style="padding:3px 0;border-bottom:1px solid #1a1a2e;font-size:0.65rem;">' +
      (ts ? '<span style="color:#484860;margin-right:6px;">' + ts + '</span>' : '') +
      esc(String(summary).substring(0, 150)) + '</div>';
  });

  return _panel('Consciousness Reports', body);
}


function _renderCodebasePanel(snap) {
  var cb = snap.codebase || {};
  if (!Object.keys(cb).length) return _panel('Codebase Index', _emptyMsg('Not indexed'));

  var symbols = cb.total_symbols || cb.symbol_count || cb.symbols || 0;
  var files = cb.total_modules || cb.file_count || cb.files || 0;
  var lines = cb.total_lines || 0;
  var edges = cb.import_edges || 0;

  var body = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Symbols', symbols, '#0cf') +
    _statCard('Files', files, '#c0f') +
    _statCard('Lines', lines ? lines.toLocaleString() : '--', '#0f9') +
    _statCard('Imports', edges, '#ff0') +
    '</div>';

  body += _metricRow('Index Freshness', cb.last_indexed ? timeAgo(cb.last_indexed) : (cb.index_age || '--'));

  var kinds = cb.symbol_kinds || {};
  if (Object.keys(kinds).length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Symbol Breakdown</div>';
    body += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(80px,1fr));gap:3px;">';
    Object.entries(kinds).sort(function(a, b) { return b[1] - a[1]; }).forEach(function(e) {
      body += '<div style="text-align:center;padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;">' +
        '<div style="font-size:0.75rem;font-weight:600;color:#0cf;">' + e[1] + '</div>' +
        '<div style="font-size:0.48rem;color:#6a6a80;">' + esc(e[0]) + '</div></div>';
    });
    body += '</div>';
  }

  return _panel('Codebase Index', body);
}


function _renderTraitValidationPanel(snap) {
  var tv = snap.trait_validation || {};
  var pr = snap.personality_rollback || {};

  var body = '';
  if (Object.keys(tv).length) {
    body += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:8px;">' +
      _statCard('Validations', tv.total_validations || tv.issues_count || 0) +
      _statCard('Stability', tv.last_stability != null ? (tv.last_stability * 100).toFixed(1) + '%' : '--', tv.last_stability > 0.8 ? '#0f9' : '#ff0') +
      _statCard('Has Baseline', tv.has_baseline ? 'Yes' : 'No', tv.has_baseline ? '#0f9' : '#6a6a80') +
      '</div>';
    if (tv.last_validation) body += _metricRow('Last Validation', typeof tv.last_validation === 'number' ? timeAgo(tv.last_validation) : '--');
    body += _metricRow('Snapshot History', tv.snapshot_history_len || 0);
  } else {
    body += _emptyMsg('No trait validation data');
  }

  if (Object.keys(pr).length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Personality Rollback</div>';
    body += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:4px;margin-bottom:4px;">' +
      _statCard('Emergency', pr.in_emergency ? 'YES' : 'No', pr.in_emergency ? '#f44' : '#0f9') +
      _statCard('Snapshots', pr.snapshot_count || 0) +
      _statCard('Rollbacks', pr.rollback_count || 0, (pr.rollback_count || 0) > 0 ? '#f90' : '#0f9') +
      '</div>';
    if (pr.current_stability != null) body += _metricRow('Current Stability', (pr.current_stability * 100).toFixed(1) + '%');
    if (pr.stability_source) body += _metricRow('Stability Source', pr.stability_source);

    var traits = pr.current_traits || {};
    if (Object.keys(traits).length) {
      body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Current Traits</div>';
      body += '<div style="display:flex;flex-wrap:wrap;gap:3px;">';
      Object.entries(traits).forEach(function(e) {
        var val = typeof e[1] === 'number' ? e[1].toFixed(2) : String(e[1]);
        body += '<span style="padding:1px 5px;background:#181828;border:1px solid #2a2a44;border-radius:3px;font-size:0.55rem;">' + esc(e[0]) + ': <b>' + esc(val) + '</b></span>';
      });
      body += '</div>';
    }
  }

  return _panel('Trait Validation / Personality', body);
}


function _renderMemoryRoutePanel(snap) {
  var mr = snap.memory_route || {};
  if (!Object.keys(mr).length) return _panel('Memory Route', _emptyMsg('No route data'));

  var body = '';
  body += _metricRow('Route Type', mr.route_type || '--');
  body += _metricRow('Scope', mr.search_scope || '--');
  body += _metricRow('Subject', mr.subject || (mr.referenced_entities && mr.referenced_entities.length ? mr.referenced_entities.join(', ') : '--'));
  body += _metricRow('Preference Injection', mr.allow_preference_injection ? 'allowed' : 'blocked');
  body += _metricRow('3rd Party', mr.allow_thirdparty_injection ? 'allowed' : 'blocked');
  body += _metricRow('Autonomy Recall', mr.allow_autonomy_recall ? 'allowed' : 'blocked');

  return _panel('Memory Route', body);
}


function _renderObserverPanel(snap) {
  var obs = snap.observer || {};
  if (!Object.keys(obs).length) return _panel('Observer', _emptyMsg('No data'));

  var body = '';
  body += _metricRow('Awareness', obs.awareness_level != null ? (obs.awareness_level * 100).toFixed(1) + '%' : '--');
  body += _metricRow('Observations', obs.observation_count || 0);
  body += _metricRow('Self-Modifications', obs.self_modification_events || 0);

  var types = obs.types || {};
  if (Object.keys(types).length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Types</div>';
    body += '<div style="margin-bottom:4px;">' + _tagGrid(types, 'none') + '</div>';
  }

  var recent = obs.recent || [];
  if (recent.length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">Recent</div>';
    recent.slice(0, 5).forEach(function(o) {
      body += '<div style="font-size:0.6rem;padding:1px 0;">' +
        '<span style="color:#0cf;">[' + esc(o.type || '') + ']</span> ' +
        esc((o.target || '').substring(0, 60)) +
        ' <span style="color:#6a6a80;">' + fmtNum(o.confidence, 2) + '</span></div>';
    });
  }

  return _panel('Observer', body);
}


function _renderThoughtsPanel(snap) {
  var th = snap.thoughts || {};
  var recent = th.recent || [];

  if (!recent.length) return _panel('Meta-Thoughts', _emptyMsg('No thoughts generated'));

  var body = '<div style="font-size:0.58rem;color:#8a8aa0;line-height:1.35;margin-bottom:6px;">' +
    'Raw thought feed. Template-generated meta-thoughts are one source of internal activity, alongside autonomy, dreams, recall, existential reasoning, philosophical dialogue, mutation, and evolution loops.' +
    '</div>';
  body += _metricRow('Total Generated', th.total_generated || 0);

  recent.slice(0, 8).forEach(function(t, idx) {
    var depthColors = { profound: '#c0f', deep: '#0cf', surface: '#6a6a80' };
    var dc = depthColors[t.depth] || '#6a6a80';
    body += '<div style="padding:2px 0;border-bottom:1px solid #1a1a2e;cursor:pointer;" onclick="window.openThoughtDetail && window.openThoughtDetail(' + idx + ')">' +
      '<div style="display:flex;gap:6px;align-items:center;">' +
      '<span style="width:6px;height:6px;border-radius:50%;background:' + dc + ';"></span>' +
      '<span style="font-size:0.65rem;color:#0cf;">[' + esc(t.type || '') + ']</span>' +
      '<span style="font-size:0.55rem;color:#484860;margin-left:auto;">\u25B6</span>' +
      '</div>' +
      '<div style="font-size:0.6rem;padding-left:12px;color:#e0e0e8;">' + esc((t.text || '').substring(0, 120)) + '</div>' +
      (t.tags && t.tags.length ? '<div style="padding-left:12px;margin-top:1px;">' + t.tags.map(function(tg) { return '<span style="font-size:0.5rem;padding:0 3px;background:#181828;border:1px solid #2a2a44;border-radius:2px;margin-right:2px;">' + esc(tg) + '</span>'; }).join('') + '</div>' : '') +
      '</div>';
  });

  return _panel('Meta-Thoughts', body, '<span style="font-size:0.6rem;color:#6a6a80;">' + (th.total_generated || 0) + ' total</span>');
}


function _renderEmergenceEvidencePanel(snap) {
  var ev = snap.emergence_evidence || {};
  var summary = ev.summary || {};
  var levels = ev.levels || [];

  if (!levels.length) {
    return _panel('Emergence Evidence', _emptyMsg('No emergence evidence surface available'));
  }

  var maxLevel = summary.max_supported_level;
  var body = '<div style="padding:8px;border:1px solid #2a2a44;background:#0d0d1a;border-radius:4px;margin-bottom:8px;">' +
    '<div style="font-size:0.72rem;color:#0cf;font-weight:700;">Operational emergence evidence, not proof of sentience.</div>' +
    '<div style="font-size:0.62rem;color:#e0e0e8;margin-top:3px;">Real substrate evidence, not roleplay. Level 7 remains empty unless an event survives known-mechanism elimination.</div>' +
    '</div>';

  body += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Supported Levels', summary.supported_levels || 0, '#0cf') +
    _statCard('Max Level', maxLevel != null && maxLevel >= 0 ? 'L' + maxLevel : '--', '#c0f') +
    _statCard('Level 7', summary.level7_claimed ? 'CLAIMED' : 'empty', summary.level7_claimed ? '#f44' : '#0f9') +
    '</div>';

  levels.forEach(function(level) {
    var status = level.status || 'unknown';
    var color = status === 'supported' ? '#0f9' : (status === 'partial' ? '#ff0' : (status === 'not_claimed' ? '#6a6a80' : '#f90'));
    var examples = level.representative_examples || [];
    var sources = level.source_paths || [];
    body += '<div style="padding:7px 0;border-top:1px solid #1a1a2e;">' +
      '<div style="display:flex;justify-content:space-between;gap:8px;align-items:center;">' +
      '<div style="font-size:0.68rem;color:#e0e0e8;font-weight:700;">L' + esc(level.level) + ' — ' + esc(level.name || '') + '</div>' +
      '<span style="font-size:0.56rem;color:' + color + ';border:1px solid ' + color + '55;border-radius:3px;padding:1px 5px;">' + esc(status) + '</span>' +
      '</div>' +
      _metricRow('Evidence Count', level.evidence_count || 0) +
      '<div style="font-size:0.56rem;color:#8a8aa0;line-height:1.35;margin-top:3px;">' + esc(level.limitations || '') + '</div>' +
      '<div style="font-size:0.54rem;color:#6a6a80;line-height:1.35;margin-top:3px;"><b>Falsification:</b> ' + esc(level.falsification_notes || '') + '</div>';
    if (examples.length) {
      body += '<div style="font-size:0.52rem;color:#0cf;margin-top:3px;">Examples: ' + examples.slice(0, 2).map(function(x) { return esc(String(x)); }).join(' · ') + '</div>';
    }
    if (sources.length) {
      body += '<div style="font-size:0.5rem;color:#484860;margin-top:2px;">Sources: ' + sources.slice(0, 3).map(function(x) { return '<code>' + esc(String(x)) + '</code>'; }).join(', ') + '</div>';
    }
    body += '</div>';
  });

  var exclusions = ev.known_mechanism_exclusions || [];
  if (exclusions.length) {
    body += '<div style="font-size:0.54rem;color:#8a8aa0;margin-top:8px;">Known mechanisms excluded before Level 7: ' +
      exclusions.map(function(x) { return '<code>' + esc(String(x)) + '</code>'; }).join(', ') + '</div>';
  }

  return _panel('Emergence Evidence', body, '<span style="font-size:0.58rem;color:#0cf;">truth surface</span>', { panelId: 'emergence-evidence', ownerTab: 'diagnostics' });
}


function _renderMutationsPanel(snap) {
  var mut = snap.mutations || {};

  var body = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Count', mut.count || 0) +
    _statCard('Rollbacks', mut.rollback_count || 0, (mut.rollback_count || 0) > 0 ? '#f44' : '#0f9') +
    _statCard('Hourly', (mut.mutations_this_hour || 0) + '/' + (mut.hourly_cap || 12)) +
    _statCard('Session Cap', mut.session_cap || 400) +
    '</div>';

  body += _metricRow('Config Version', mut.config_version || '--');
  body += _metricRow('Monitoring', mut.active_monitor ? 'ACTIVE' : 'idle');
  body += _metricRow('Total Rejections', mut.total_rejections || 0);

  var history = (mut.history || []).slice().reverse();
  if (history.length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Recent History</div>';
    history.slice(0, 8).forEach(function(h) {
      body += '<div style="font-size:0.6rem;padding:1px 0;border-bottom:1px solid #1a1a2e;">' + esc(String(h).substring(0, 120)) + '</div>';
    });
  }

  var rejections = mut.recent_rejections || [];
  if (rejections.length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Recent Rejections</div>';
    rejections.forEach(function(r) {
      body += '<div style="font-size:0.6rem;padding:1px 0;color:#f44;">' + esc(String(r).substring(0, 100)) + '</div>';
    });
  }

  return _panel('Mutations', body);
}


function _renderLedgerPanel(snap) {
  var ledger = snap.ledger || {};
  if (!Object.keys(ledger).length) return _panel('Attribution Ledger (L1)', _emptyMsg('No data'));

  var integrity = ledger.integrity || {};

  var body = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Entries', ledger.total_recorded || 0) +
    _statCard('Pending', integrity.pending_entries || 0, (integrity.pending_entries || 0) > 10 ? '#ff0' : '#6a6a80') +
    _statCard('Orphaned', integrity.orphaned_entries || 0, (integrity.orphaned_entries || 0) > 0 ? '#f44' : '#0f9') +
    '</div>';

  body += _metricRow('Total Outcomes', ledger.total_outcomes || 0);

  var sched = ledger.outcome_scheduler || {};
  if (Object.keys(sched).length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Outcome Scheduler</div>';
    body += _metricRow('Pending', sched.pending || 0);
    body += _metricRow('Resolved', sched.resolved || 0);
    body += _metricRow('Inconclusive', sched.inconclusive || 0);
    body += _metricRow('Errors', sched.errors || 0);
    body += _metricRow('Evicted', sched.evicted || 0);
  }

  var scopeCounts = ledger.scope_counts || {};
  if (Object.keys(scopeCounts).length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">By Scope</div>';
    body += '<div>' + _tagGrid(scopeCounts, 'no scope data') + '</div>';
  }

  var orphans = integrity.orphaned_details || [];
  if (orphans.length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Orphaned Entries</div>';
    orphans.slice(0, 5).forEach(function(o) {
      body += '<div style="font-size:0.6rem;padding:1px 0;color:#f44;">[' + esc(o.subsystem || '') + '] ' + esc(o.event_type || '') + ' — ' + (o.age_s || 0) + 's old</div>';
    });
  }

  return _panel('Attribution Ledger (L1)', body);
}


function _renderTraceExplorerPanel(snap) {
  var tx = snap.trace_explorer || {};
  var roots = tx.root_chains || [];
  var runs = tx.agent_runs || [];
  var lineage = tx.tool_lineage || [];
  var rec = (snap.reconstructability || {}).trace_explorer || {};
  var recStatus = rec.reconstructability || 'unknown';
  var recColor = recStatus === 'reconstructable' ? '#0f9' : recStatus === 'partial' ? '#ff0' : recStatus === 'non_reconstructable' ? '#f44' : '#6a6a80';
  var recBadge = '<span style="font-size:0.56rem;color:' + recColor + ';">' + esc(recStatus.replace(/_/g, ' ')) + '</span>';

  if (!roots.length && !runs.length && !lineage.length) {
    var emptyBody = _emptyMsg('No lineage data in current cache window.');
    if (window.openTraceExplorer) {
      emptyBody += '<div style="margin-top:6px;"><button class="j-btn-sm" onclick="window.openTraceExplorer()">Open Trace Explorer</button></div>';
    }
    return _panel('Trace Explorer', emptyBody, recBadge);
  }

  var body = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Ledger Entries', tx.entry_count || 0, '#0cf') +
    _statCard('Root Chains', roots.length || 0, '#0f9') +
    _statCard('Agent Runs', runs.length || 0, '#c0f') +
    _statCard('Tool Hops', lineage.length || 0, '#ff0') +
    '</div>';

  if (roots.length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:2px;">Recent Root Chains</div>';
    roots.slice(0, 5).forEach(function(r) {
      var rootId = r.root_entry_id || '--';
      var dur = r.duration_s != null ? fmtNum(r.duration_s, 2) + 's' : '--';
      var sys = (r.subsystems || []).slice(0, 3).join(', ') || '--';
      body += '<div style="display:flex;gap:6px;align-items:center;padding:2px 0;border-bottom:1px solid #1a1a2e;font-size:0.56rem;">' +
        '<span style="color:#0cf;min-width:56px;">' + esc(rootId.substring(0, 16)) + '</span>' +
        '<span style="color:#e0e0e8;min-width:48px;">' + (r.entry_count || 0) + ' ev</span>' +
        '<span style="color:#6a6a80;min-width:40px;">' + esc(dur) + '</span>' +
        '<span style="color:#8a8aa0;flex:1;">' + esc(sys) + '</span>' +
        '<button class="j-btn-xs" onclick="window.openTraceChain && window.openTraceChain(\'' + esc(rootId) + '\')">view</button>' +
        '</div>';
    });
  }

  if (runs.length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:6px;margin-bottom:2px;">Per-Agent Runs</div>';
    runs.slice(0, 5).forEach(function(run) {
      var rid = run.intent_id || '--';
      body += '<div style="padding:2px 0;border-bottom:1px solid #1a1a2e;font-size:0.56rem;">' +
        '<span style="color:#c0f;">' + esc(rid.substring(0, 18)) + '</span>' +
        ' <span style="color:#6a6a80;">events:</span> ' + (run.event_count || 0) +
        (run.tools && run.tools.length ? ' <span style="color:#6a6a80;">tools:</span> ' + esc(run.tools.slice(0, 2).join(', ')) : '') +
        '</div>';
    });
  }

  if (lineage.length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:6px;margin-bottom:2px;">Tool Lineage</div>';
    lineage.slice(0, 6).forEach(function(h) {
      body += '<div style="padding:1px 0;font-size:0.55rem;border-bottom:1px solid #1a1a2e;">' +
        '<span style="color:#0f9;">' + esc(h.tool || '--') + '</span> ' +
        '<span style="color:#6a6a80;">(' + esc(h.source || '--') + ')</span>' +
        (h.trace_id ? ' <span style="color:#484860;">' + esc(h.trace_id.substring(0, 14)) + '</span>' : '') +
        '</div>';
    });
  }

  if (window.openTraceExplorer) {
    body += '<div style="margin-top:6px;"><button class="j-btn-sm" onclick="window.openTraceExplorer()">Open Full Explorer</button></div>';
  }

  return _panel('Trace Explorer', body, recBadge);
}


function _renderReconstructabilityPanel(snap) {
  var rec = snap.reconstructability || {};
  var items = Object.entries(rec);
  if (!items.length) return _panel('Reconstructability', _emptyMsg('No metadata'));

  var counts = { reconstructable: 0, partial: 0, non_reconstructable: 0, unknown: 0 };
  items.forEach(function(e) {
    var status = (e[1] || {}).reconstructability || 'unknown';
    if (!counts.hasOwnProperty(status)) counts.unknown += 1;
    else counts[status] += 1;
  });

  var body = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Reconstructable', counts.reconstructable, '#0f9') +
    _statCard('Partial', counts.partial, '#ff0') +
    _statCard('Non-Recon', counts.non_reconstructable, '#f44') +
    _statCard('Surfaces', items.length, '#0cf') +
    '</div>';

  items.sort(function(a, b) {
    var rank = { reconstructable: 0, partial: 1, non_reconstructable: 2, unknown: 3 };
    var ra = rank[(a[1] || {}).reconstructability || 'unknown'];
    var rb = rank[(b[1] || {}).reconstructability || 'unknown'];
    return ra - rb;
  }).forEach(function(pair) {
    var name = pair[0];
    var meta = pair[1] || {};
    var status = meta.reconstructability || 'unknown';
    var c = status === 'reconstructable' ? '#0f9' : status === 'partial' ? '#ff0' : status === 'non_reconstructable' ? '#f44' : '#6a6a80';
    var source = (meta.source_of_truth || []).join(', ');
    var derived = (meta.derived_fields || []).join(', ');
    var evidence = meta.evidence_link || '--';
    body += '<div style="padding:4px 0;border-bottom:1px solid #1a1a2e;">' +
      '<div style="display:flex;justify-content:space-between;gap:6px;font-size:0.58rem;">' +
      '<span style="color:#e0e0e8;">' + esc(name.replace(/_/g, ' ')) + '</span>' +
      '<span style="color:' + c + ';">' + esc(status.replace(/_/g, ' ')) + '</span></div>' +
      '<div style="font-size:0.48rem;color:#6a6a80;margin-top:2px;">source: ' + esc(source.substring(0, 120)) + '</div>' +
      '<div style="font-size:0.48rem;color:#484860;">derived: ' + esc(derived.substring(0, 120)) + '</div>' +
      '<div style="font-size:0.48rem;color:#484860;">evidence: ' + esc(evidence) + '</div>' +
      '</div>';
  });

  return _panel('Reconstructability', body);
}


function _renderEpistemicPanel(snap) {
  var ep = snap.epistemic || {};
  if (!Object.keys(ep).length) return _panel('Epistemic Reasoning', _emptyMsg('No data'));

  var models = ep.active_models || ep.models || [];

  var body = '';
  body += _metricRow('Causal Models', ep.causal_model_count || ep.model_count || models.length || 0);
  body += _metricRow('Predictions', ep.prediction_count || ep.predictions || ep.total_chains || 0);
  body += _metricRow('Cascades', ep.cascade_count || ep.cascades || 0);

  var acc = ep.accuracy || ep.prediction_accuracy;
  if (acc != null) body += _metricRow('Accuracy', (acc * 100).toFixed(1) + '%');

  if (models.length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Active Models</div>';
    models.slice(0, 6).forEach(function(m) {
      var name = typeof m === 'string' ? m : (m.phenomenon || m.name || m.id || 'unnamed');
      var conf = typeof m === 'object' ? fmtNum(m.confidence, 2) : '';
      body += '<div style="font-size:0.6rem;padding:1px 0;">' +
        '<span style="color:#0cf;">' + esc(name) + '</span>' +
        (conf ? ' <span style="color:#6a6a80;">' + conf + '</span>' : '') + '</div>';
    });
  }

  return _panel('Epistemic Reasoning', body);
}


function _renderEvalSidecarPanel(snap) {
  var ev = snap.eval || {};
  if (!Object.keys(ev).length) return _panel('Eval Sidecar', _emptyMsg('Not active'));

  var banner = ev.banner || {};
  var sm = ev.store_meta || {};
  var tap = ev.tap || {};
  var coll = ev.collector || {};
  var files = ev.store_file_sizes || ev.file_sizes || {};

  // Summary stats row
  var body = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Events', fmtNum(sm.total_events_written || tap.total_buffered || 0)) +
    _statCard('Snapshots', fmtNum(sm.total_snapshots_written || coll.snapshots_collected || 0)) +
    _statCard('Scorecards', fmtNum(sm.total_scorecards_written || 0)) +
    _statCard('Freshness', banner.data_freshness_s != null ? fmtNum(banner.data_freshness_s, 1) + 's' : '--',
      (banner.data_freshness_s || 0) < 30 ? '#0f9' : '#ff0') +
    '</div>';

  // Banner info
  body += '<div style="display:flex;gap:8px;font-size:0.5rem;color:#484860;margin-bottom:4px;">' +
    '<span>mode: <span style="color:#0cf;">' + esc(banner.mode || '--') + '</span></span>' +
    '<span>version: ' + esc(banner.scoring_version || '--') + '</span>' +
    '<span>uptime: ' + (banner.uptime_s ? _rnd_fmtUptime(banner.uptime_s) : '--') + '</span>' +
    '<span>pvl: ' + (banner.pvl_enabled ? '<span style="color:#0f9;">on</span>' : '<span style="color:#f44;">off</span>') + '</span>' +
    '</div>';

  // Tap + Collector
  body += '<div style="display:flex;gap:8px;font-size:0.5rem;color:#484860;margin-bottom:4px;">' +
    '<span>tap: ' + (tap.wired ? '<span style="color:#0f9;">wired</span>' : '<span style="color:#f44;">off</span>') +
      ' (buf:' + (tap.buffer_size || 0) + ', types:' + (tap.tapped_event_count || 0) + ')</span>' +
    '<span>collector: every ' + fmtNum(coll.interval_s || 60, 0) + 's, errors:' + (coll.errors || coll.collect_errors || 0) + '</span>' +
    '</div>';

  // Event counts
  var ec = ev.event_counts || {};
  var ecEntries = Object.entries(ec).sort(function(a, b) { return b[1] - a[1]; });
  if (ecEntries.length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Event Counts (' + ecEntries.length + ' types)</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:4px;">';
    ecEntries.slice(0, 12).forEach(function(e) {
      body += '<span style="padding:1px 4px;border:1px solid #2a2a44;border-radius:2px;font-size:0.45rem;color:#8a8aa0;">' +
        esc(e[0].replace(/:/g, ':')) + ' <span style="color:#0cf;">' + e[1] + '</span></span>';
    });
    if (ecEntries.length > 12) body += '<span style="font-size:0.42rem;color:#484860;cursor:pointer;" onclick="window.openEvalDetail && window.openEvalDetail()">+' + (ecEntries.length - 12) + ' more \u25B6</span>';
    body += '</div>';
  }

  // Dream artifacts
  var dream = ev.dream || {};
  if (dream.buffer_size) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Dream Artifacts</div>' +
      '<div style="display:flex;gap:8px;font-size:0.5rem;margin-bottom:3px;">' +
      '<span style="color:#c0f;">buf: ' + dream.buffer_size + '</span>' +
      '<span style="color:#0f9;">promo: ' + fmtPct(dream.promotion_rate) + '</span>' +
      '<span style="color:#6a6a80;">validator: ' + (dream.validator_runs || 0) + ' runs</span>' +
      '</div>';
    var byType = dream.by_type || {};
    if (Object.keys(byType).length) {
      body += '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:3px;">';
      Object.entries(byType).forEach(function(e) {
        body += '<span style="padding:1px 4px;border:1px solid #c0f33;border-radius:2px;font-size:0.45rem;color:#c0f;">' +
          esc(e[0].replace(/_/g, ' ')) + ': ' + e[1] + '</span>';
      });
      body += '</div>';
    }
    var byState = dream.by_state || {};
    if (Object.keys(byState).length) {
      body += '<div style="display:flex;gap:6px;font-size:0.48rem;color:#484860;">';
      Object.entries(byState).forEach(function(e) {
        var sc = e[0] === 'promoted' ? '#0f9' : e[0] === 'held' ? '#ff0' : '#6a6a80';
        body += '<span style="color:' + sc + ';">' + esc(e[0]) + ': ' + e[1] + '</span>';
      });
      body += '</div>';
    }
  }

  // File sizes
  if (Object.keys(files).length) {
    body += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Store Sizes</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:6px;font-size:0.48rem;color:#484860;">';
    Object.entries(files).forEach(function(e) {
      body += '<span>' + esc(e[0]) + ': <span style="color:#8a8aa0;">' + _rnd_fmtBytes(e[1]) + '</span></span>';
    });
    body += '</div>';
  }

  // Rotations
  var rotations = sm.rotation_counts || {};
  if (Object.keys(rotations).length) {
    var hasRotations = Object.values(rotations).some(function(v) { return v > 0; });
    if (hasRotations) {
      body += '<div style="display:flex;gap:6px;font-size:0.48rem;color:#484860;margin-top:3px;">';
      body += '<span>rotations:</span>';
      Object.entries(rotations).forEach(function(e) {
        if (e[1] > 0) body += '<span style="color:#f90;">' + esc(e[0]) + ':' + e[1] + '</span>';
      });
      body += '</div>';
    }
  }

  // Drill-down link
  body += '<div style="text-align:right;margin-top:4px;">' +
    '<span style="font-size:0.55rem;color:#0cf;cursor:pointer;" onclick="window.openEvalDetail && window.openEvalDetail()">Full details \u25B6</span></div>';

  return _panel('Eval Sidecar', body,
    '<span style="font-size:0.55rem;color:' + (banner.mode === 'shadow' ? '#0cf' : '#0f9') + ';">' + esc(banner.mode || 'shadow') + '</span>');
}


function _renderSpatialDiagnosticsPanel(snap) {
  var sp = snap.spatial || {};
  if (sp.status !== 'active') return _panel('Spatial Intelligence', _emptyMsg('Not initialized'));

  var cal = sp.calibration || {};
  var est = sp.estimator || {};
  var val = sp.validation || {};

  var calColor = cal.state === 'valid' ? '#0f0' : cal.state === 'stale' ? '#ff0' : '#f44';
  var calLabel = (cal.state || 'unknown').toUpperCase();

  var statsHtml = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Calibration', calLabel, calColor) +
    _statCard('Cal Version', cal.version || 0, '#0af') +
    _statCard('Tracks', est.track_count || 0, '#0f0') +
    _statCard('Stable', est.stable_tracks || 0, '#0af') +
    '</div>';

  statsHtml += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('Observations', est.observation_count || 0, '#aaa') +
    _statCard('Anchors', est.anchor_count || 0, '#c0f') +
    _statCard('Promoted', val.total_promoted || 0, '#0f0') +
    _statCard('Rejected', val.total_rejected || 0, '#f44') +
    '</div>';

  var detailHtml = '';

  var tracks = est.tracks || {};
  var trackKeys = Object.keys(tracks);
  if (trackKeys.length > 0) {
    detailHtml += '<div style="margin-top:6px;font-size:0.7rem;color:#8a8aa0;">';
    detailHtml += '<div style="font-weight:600;margin-bottom:4px;color:#0af;">Spatial Tracks</div>';
    for (var i = 0; i < Math.min(trackKeys.length, 10); i++) {
      var tid = trackKeys[i];
      var t = tracks[tid];
      var pos = t.position_room_m || [0, 0, 0];
      var statusColor = t.track_status === 'stable' ? '#0f0' : t.track_status === 'provisional' ? '#ff0' : '#888';
      detailHtml += '<div style="display:flex;gap:8px;padding:1px 0;">' +
        '<span style="color:' + statusColor + ';">●</span> ' +
        '<span style="width:80px;">' + esc(t.label || tid) + '</span>' +
        '<span style="color:#0af;">~' + fmtNum(pos[2], 1) + 'm</span>' +
        '<span style="color:#888;">' + esc(t.track_status || '') + '</span>' +
        '<span style="color:#666;">conf:' + fmtNum(t.confidence, 2) + '</span>' +
        '</div>';
    }
    detailHtml += '</div>';
  }

  var rejections = val.rejection_counts || {};
  var rejKeys = Object.keys(rejections);
  if (rejKeys.length > 0) {
    detailHtml += '<div style="margin-top:6px;font-size:0.7rem;color:#8a8aa0;">';
    detailHtml += '<div style="font-weight:600;margin-bottom:4px;color:#f44;">Rejection Reasons</div>';
    for (var j = 0; j < rejKeys.length; j++) {
      detailHtml += '<div>' + esc(rejKeys[j]) + ': <span style="color:#f44;">' + rejections[rejKeys[j]] + '</span></div>';
    }
    detailHtml += '</div>';
  }

  if (cal.age_s != null) {
    detailHtml += '<div style="margin-top:6px;font-size:0.65rem;color:#666;">Cal age: ' + timeAgo(Date.now()/1000 - cal.age_s) + '</div>';
  }

  return _panel('Spatial Intelligence', statsHtml + detailHtml, _statusBadge(cal.state || 'invalid'));
}

function _renderGestationPanel(snap) {
  var g = snap.gestation || {};
  if (!g.active) return '';

  var body = '';
  body += _metricRow('Phase', g.phase != null ? 'Phase ' + g.phase : '--');
  body += _metricRow('Readiness', g.readiness != null ? (g.readiness * 100).toFixed(0) + '%' : '--');
  body += _metricRow('Elapsed', g.elapsed_s ? _rnd_fmtUptime(g.elapsed_s) : '--');
  body += _metricRow('Directives', g.directives_completed ? g.directives_completed + '/' + (g.directives_total || 0) : '--');
  body += _metricRow('Research Jobs', g.research_jobs || 0);

  if (g.readiness_scores && Object.keys(g.readiness_scores).length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Readiness Scores</div>';
    Object.entries(g.readiness_scores).forEach(function(e) {
      var val = e[1] || 0;
      var c = val >= 0.8 ? '#0f9' : val >= 0.5 ? '#ff0' : '#f44';
      body += '<div style="display:flex;align-items:center;gap:6px;padding:1px 0;">' +
        '<span style="min-width:100px;font-size:0.6rem;color:#6a6a80;">' + esc(e[0].replace(/_/g, ' ')) + '</span>' +
        _barFill(val, 1, c) +
        '<span style="min-width:36px;text-align:right;font-size:0.6rem;color:' + c + ';">' + (val * 100).toFixed(0) + '%</span></div>';
    });
  }

  return _panel('Gestation', body, _statusBadge('active'));
}


function _renderHardwarePanel(snap) {
  var hw = snap.hardware || {};
  if (!Object.keys(hw).length) return _panel('Hardware', _emptyMsg('No hardware data'));

  var body = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:8px;">' +
    _statCard('GPU Tier', hw.tier || '--', '#c0f') +
    _statCard('VRAM', hw.vram_mb ? Math.round(hw.vram_mb / 1024) + ' GB' : '--') +
    _statCard('CPU Tier', hw.cpu_tier || '--') +
    '</div>';

  body += _metricRow('GPU', hw.gpu_name || '--');
  body += _metricRow('CPU', hw.cpu_model || '--');
  body += _metricRow('Cores', hw.cpu_cores && hw.cpu_threads ? hw.cpu_cores + 'c/' + hw.cpu_threads + 't' : '--');
  body += _metricRow('RAM', hw.cpu_ram_gb ? hw.cpu_ram_gb + ' GB' : '--');
  body += _metricRow('STT Model', hw.stt_model || '--');
  body += _metricRow('LLM Model', hw.llm_model || '--');
  body += _metricRow('Compute Type', hw.stt_compute || '--');

  var workload = [];
  if (hw.stt_device) workload.push('STT=' + hw.stt_device);
  if (hw.emotion_device) workload.push('Emotion=' + hw.emotion_device);
  if (hw.speaker_id_device) workload.push('Speaker=' + hw.speaker_id_device);
  if (hw.embedding_device) workload.push('Embed=' + hw.embedding_device);
  if (hw.hemisphere_device) workload.push('Hemi=' + hw.hemisphere_device);
  if (workload.length) {
    body += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Device Workload</div>';
    body += '<div style="font-size:0.6rem;color:#484860;">' + esc(workload.join('  ')) + '</div>';
  }

  return _panel('Hardware', body);
}
