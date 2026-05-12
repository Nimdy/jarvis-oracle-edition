'use strict';

// ═══════════════════════════════════════════════════════════════════════════
// 1. Utilities
// ═══════════════════════════════════════════════════════════════════════════

var $ = function(s) { return document.querySelector(s); };
var el = function(id) { return document.getElementById(id); };

window.esc = function(s) {
  if (s == null) return '';
  var d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
};
window.fmt = function(v, d) { return v != null ? Number(v).toFixed(d != null ? d : 0) : '--'; };
window.pct = function(v) { return v != null ? (v * 100).toFixed(1) + '%' : '--'; };
window.fmtNum = function(v, d) { return (v != null && !isNaN(v)) ? Number(v).toFixed(d != null ? d : 0) : '--'; };
window.fmtPct = function(v, d) { return (v != null && !isNaN(v)) ? (Number(v) * 100).toFixed(d != null ? d : 1) + '%' : '--'; };
window.timeAgo = function(ts) {
  if (!ts) return '--';
  var s = Math.floor((Date.now() / 1000) - ts);
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.floor(s / 60) + 'm ago';
  if (s < 86400) return Math.floor(s / 3600) + 'h ago';
  return Math.floor(s / 86400) + 'd ago';
};
window.fmtBytes = function(b) {
  if (b == null) return '--';
  if (b < 1024) return b + 'B';
  if (b < 1048576) return (b / 1024).toFixed(1) + 'KB';
  return (b / 1048576).toFixed(1) + 'MB';
};
window.fmtUptime = function(s) {
  if (s == null) return '--';
  s = Math.round(s);
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60) + 's';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
};
window.kvString = function(obj) {
  if (!obj || typeof obj !== 'object') return '--';
  return Object.entries(obj).map(function(e) { return e[0] + ': ' + e[1]; }).join(' \u00b7 ');
};
window._sc = function(status) {
  var m = {
    active: 'badge-pass', pass: 'badge-pass', completed: 'badge-pass', verified: 'badge-pass',
    fail: 'badge-fail', failed: 'badge-fail', error: 'badge-fail', blocked: 'badge-fail',
    warning: 'badge-warning', learning: 'badge-warning', pending: 'badge-warning',
    running: 'badge-info', in_progress: 'badge-info', shadow: 'badge-info'
  };
  return m[status] || '';
};
window._ageStr = function(s) {
  if (s == null) return '--';
  s = Math.round(s);
  if (s < 60) return s + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
};


// ═══════════════════════════════════════════════════════════════════════════
// 2. API Key + Auth
// ═══════════════════════════════════════════════════════════════════════════

window._apiKey = '';

function _loadApiKey() {
  fetch('/api/config').then(function(r) { return r.json(); }).then(function(d) {
    window._apiKey = d.api_key || '';
  }).catch(function() {});
}

function _authHeaders(extra) {
  var h = Object.assign({'Content-Type': 'application/json'}, extra || {});
  if (window._apiKey) h['Authorization'] = 'Bearer ' + window._apiKey;
  return h;
}
window._authHeaders = _authHeaders;


// ═══════════════════════════════════════════════════════════════════════════
// 3. Tab Switching
// ═══════════════════════════════════════════════════════════════════════════

var TAB_NAMES = ['cockpit', 'trust', 'memory', 'activity', 'learning', 'training', 'diagnostics'];
var _activeTab = 'cockpit';
var _tabUiState = {};
var _pendingPanelFocusId = '';
var _panelFocusAttempts = 0;

// Stable panel IDs and ownership metadata drive collapse state,
// tab wayfinding, and cross-tab navigation.
var PANEL_OWNERSHIP = {
  'l0-capability-gate': { ownerTab: 'trust', label: 'L0 Capability Gate', sourceKeys: ['capability_gate'] },
  'l1-attribution-ledger': { ownerTab: 'trust', label: 'L1 Attribution Ledger', sourceKeys: ['ledger'] },
  'l2-provenance': { ownerTab: 'trust', label: 'L2 Provenance & Validation', sourceKeys: ['memory', 'validation'] },
  'l3-identity-boundary': { ownerTab: 'trust', label: 'L3 Identity Boundary', sourceKeys: ['identity_boundary'] },
  'l3a-identity-persistence': { ownerTab: 'trust', label: 'L3A Identity Persistence', sourceKeys: ['identity'] },
  'l3b-scene-continuity': { ownerTab: 'trust', label: 'L3B Scene Continuity', sourceKeys: ['scene'] },
  'l4-delayed-outcomes': { ownerTab: 'trust', label: 'L4 Delayed Outcomes', sourceKeys: ['eval.scorecards', 'ledger.outcome_scheduler'] },
  'l5-contradiction-engine': { ownerTab: 'trust', label: 'L5 Contradiction Engine', sourceKeys: ['contradiction'] },
  'l6-truth-calibration': { ownerTab: 'trust', label: 'L6 Truth Calibration', sourceKeys: ['truth_calibration'] },
  'l7-belief-graph': { ownerTab: 'trust', label: 'L7 Belief Graph', sourceKeys: ['belief_graph'] },
  'l8-quarantine': { ownerTab: 'trust', label: 'L8 Quarantine', sourceKeys: ['quarantine'] },
  'l9-reflective-audit': { ownerTab: 'trust', label: 'L9 Reflective Audit', sourceKeys: ['reflective_audit'] },
  'l10-soul-integrity': { ownerTab: 'trust', label: 'L10 Soul Integrity', sourceKeys: ['soul_integrity'] },
  'trust-overview': { ownerTab: 'trust', label: 'Trust Overview', sourceKeys: ['trust_state'] },
  'trust-oracle-benchmark': { ownerTab: 'trust', label: 'Oracle Benchmark', sourceKeys: ['eval.oracle_benchmark'] },
  'trust-pvl': { ownerTab: 'trust', label: 'Process Verification (PVL)', sourceKeys: ['eval.pvl'] },
  'trust-maturity-gates': { ownerTab: 'trust', label: 'Maturity Gates', sourceKeys: ['eval.maturity_tracker'] },
  'trust-validation-pack': { ownerTab: 'trust', label: 'Runtime Validation Pack', sourceKeys: ['eval.validation_pack'] },
  'trust-language-governance': { ownerTab: 'trust', label: 'Language Governance', sourceKeys: ['eval.language'] },
  'trust-trace-explorer': { ownerTab: 'trust', label: 'Trace Explorer', sourceKeys: ['trace_explorer'] },
  'trust-reconstructability': { ownerTab: 'trust', label: 'Reconstructability', sourceKeys: ['reconstructability'] },
  'trust-rolling-comparisons': { ownerTab: 'trust', label: 'Rolling Comparisons', sourceKeys: ['eval.scorecards'] },
  'trust-stability-chart': { ownerTab: 'trust', label: 'Stability Chart', sourceKeys: ['eval.stability'] },
  'trust-category-scoreboard': { ownerTab: 'trust', label: 'Category Scoreboard', sourceKeys: ['eval.scoreboard'] },
  'trust-golden-commands': { ownerTab: 'trust', label: 'Golden Commands', sourceKeys: ['golden_commands'] },
  'learning-skills': { ownerTab: 'learning', label: 'Skills & Learning', sourceKeys: ['skills', 'learning_jobs', 'capability_gate'] },
  'learning-library': { ownerTab: 'learning', label: 'Library', sourceKeys: ['library'] },
  'learning-hemisphere': { ownerTab: 'learning', label: 'Hemisphere NNs', sourceKeys: ['hemisphere', 'skill_acquisition_specialist'] },
  'learning-language': { ownerTab: 'learning', label: 'Language Substrate', sourceKeys: ['language', 'language_phasec'] },
  'learning-self-improvement': { ownerTab: 'learning', label: 'Self-Improvement', sourceKeys: ['self_improve'] },
  'learning-acquisition': { ownerTab: 'learning', label: 'Capability Acquisition', sourceKeys: ['acquisition'] },
  'learning-skill-acquisition-weight-room': { ownerTab: 'learning', label: 'Skill Acquisition Weight Room', sourceKeys: ['skill_acquisition_weight_room', 'skill_acquisition_specialist'] },
  'learning-plugins': { ownerTab: 'learning', label: 'Plugin Registry', sourceKeys: ['plugins'] },
  'activity-world-perception': { ownerTab: 'activity', label: 'World & Perception', sourceKeys: ['world_model', 'scene', 'attention'] }
};

var PANEL_ID_BY_TITLE = {
  'Trust Overview': 'trust-overview',
  'Oracle Benchmark': 'trust-oracle-benchmark',
  'Process Verification (PVL)': 'trust-pvl',
  'Maturity Gates': 'trust-maturity-gates',
  'Runtime Validation Pack': 'trust-validation-pack',
  'Language Governance': 'trust-language-governance',
  'Attribution Ledger (L1)': 'l1-attribution-ledger',
  'Identity Boundary (L3)': 'l3-identity-boundary',
  'Contradiction Engine (L5)': 'l5-contradiction-engine',
  'Truth Calibration (L6)': 'l6-truth-calibration',
  'Belief Graph (L7)': 'l7-belief-graph',
  'Quarantine (L8)': 'l8-quarantine',
  'Reflective Audit (L9)': 'l9-reflective-audit',
  'Soul Integrity (L10)': 'l10-soul-integrity',
  'Trace Explorer': 'trust-trace-explorer',
  'Reconstructability': 'trust-reconstructability',
  'Rolling Comparisons': 'trust-rolling-comparisons',
  'Stability Chart': 'trust-stability-chart',
  'Category Scoreboard': 'trust-category-scoreboard',
  'Golden Commands': 'trust-golden-commands',
  'Skills & Learning': 'learning-skills',
  'Library': 'learning-library',
  'Hemisphere NNs': 'learning-hemisphere',
  'Language Substrate': 'learning-language',
  'Self-Improvement': 'learning-self-improvement',
  'Capability Acquisition': 'learning-acquisition',
  'Skill Acquisition Weight Room': 'learning-skill-acquisition-weight-room',
  'Plugin Registry': 'learning-plugins',
  'World & Perception': 'activity-world-perception'
};

var DOC_LINKS_BY_PANEL_ID = {
  'l0-capability-gate': '/docs#safety-guards',
  'l1-attribution-ledger': '/docs#epistemic-stack',
  'l2-provenance': '/docs#epistemic-stack',
  'l3-identity-boundary': '/docs#identity-boundary',
  'l3a-identity-persistence': '/science#identity-math',
  'l3b-scene-continuity': '/docs#flow-perception',
  'l4-delayed-outcomes': '/docs#epistemic-stack',
  'l5-contradiction-engine': '/docs#epistemic-stack',
  'l6-truth-calibration': '/science#score-truth',
  'l7-belief-graph': '/science#belief-propagation',
  'l8-quarantine': '/docs#epistemic-stack',
  'l9-reflective-audit': '/docs#epistemic-stack',
  'l10-soul-integrity': '/science#score-soul',
  'trust-overview': '/docs#epistemic-stack',
  'trust-oracle-benchmark': '/docs#eval-sidecar',
  'trust-pvl': '/docs#eval-sidecar',
  'trust-maturity-gates': '/maturity',
  'trust-validation-pack': '/docs#maturity-gates',
  'trust-language-governance': '/docs#flow-conversation',
  'trust-trace-explorer': '/docs#eval-sidecar',
  'trust-reconstructability': '/docs#restart-resilience',
  'trust-rolling-comparisons': '/docs#eval-sidecar',
  'trust-stability-chart': '/docs#eval-sidecar',
  'trust-category-scoreboard': '/docs#eval-sidecar',
  'trust-golden-commands': '/docs#safety-guards',
  'learning-skills': '/docs#flow-self-improve',
  'learning-library': '/docs#flow-memory',
  'learning-hemisphere': '/science#nn-hemisphere',
  'learning-language': '/docs#flow-conversation',
  'learning-self-improvement': '/docs#flow-self-improve',
  'learning-acquisition': '/capability-pipeline#acquisition',
  'learning-skill-acquisition-weight-room': '/docs#synthetic-training',
  'learning-plugins': '/docs#flow-self-improve',
  'activity-world-perception': '/docs#flow-world-model'
};

var DOC_LINKS_BY_TITLE = {
  'Operations': '/docs#flow-consciousness',
  'Autonomy': '/docs#flow-phase5',
  'Motive Drives': '/docs#flow-phase5',
  'Goals': '/docs#goal-continuity',
  'World Model': '/science#world-model',
  'Simulator': '/science#simulator',
  'Scene': '/docs#flow-perception',
  'Attention': '/docs#flow-perception',
  'Flight Recorder': '/docs#flow-conversation',
  'Self-Improvement': '/docs#flow-self-improve',
  'Skills & Learning': '/docs#flow-self-improve',
  'Library': '/docs#flow-memory',
  'Language Substrate': '/docs#flow-conversation',
  'Explainability': '/docs#eval-sidecar',
  'Policy NN': '/science#nn-policy',
  'Hemisphere NNs': '/science#nn-hemisphere',
  'Matrix Protocol': '/science#nn-distillation',
  'ML Training': '/science#rl-training',
  'Capability Acquisition': '/capability-pipeline#acquisition',
  'Skill Acquisition Weight Room': '/docs#synthetic-training',
  'Plugin Registry': '/docs#flow-self-improve',
  'Cognitive Gaps': '/docs#flow-self-improve',
  'Companion Training': '/docs#synthetic-training',
  'Improvement History': '/docs#flow-self-improve',
  'Soul Integrity': '/science#score-soul',
  'Soul Integrity (L10)': '/science#score-soul',
  'Reflective Audit (L9)': '/docs#epistemic-stack',
  'Kernel Performance': '/docs#flow-consciousness',
  'Event Reliability': '/docs#flow-consciousness',
  'Event Validation': '/docs#flow-consciousness',
  'Consciousness Reports': '/docs#flow-consciousness',
  'Codebase Index': '/docs#flow-self-improve',
  'Trait Validation / Personality': '/docs#flow-personal-intel',
  'Memory Route': '/docs#flow-memory',
  'Observer': '/docs#flow-consciousness',
  'Meta-Thoughts': '/docs#flow-consciousness',
  'Mutations': '/docs#flow-self-improve',
  'Attribution Ledger (L1)': '/docs#epistemic-stack',
  'Trace Explorer': '/docs#eval-sidecar',
  'Reconstructability': '/docs#restart-resilience',
  'Epistemic Reasoning': '/docs#epistemic-stack',
  'Eval Sidecar': '/docs#eval-sidecar',
  'Spatial Intelligence': '/docs#flow-perception',
  'Gestation': '/docs#restart-resilience',
  'Hardware': '/docs#hardware-tiers',
  'Memory Overview': '/docs#flow-memory',
  'Core Memories': '/docs#flow-memory',
  'Memory Density': '/docs#flow-memory',
  'Memory Cortex': '/science#nn-cortex',
  'CueGate — Memory Access Policy': '/docs#flow-memory',
  'Fractal Recall': '/science#score-resonance',
  'Synthetic Perception Exercise': '/docs#synthetic-training',
  'Identity & Recognition': '/science#identity-math',
  'Rapport': '/docs#flow-personal-intel',
  'Identity Boundary (L3)': '/docs#identity-boundary',
  'Memory Analytics': '/docs#flow-memory',
  'Memory Maintenance': '/docs#flow-memory',
  'Dream Processing': '/docs#maturity-gates',
  'HRR / VSA Shadow': '/science#holographic-cognition-hrr'
};

window.DASHBOARD_PANEL_OWNERSHIP = PANEL_OWNERSHIP;
window.DASHBOARD_DOC_LINKS_BY_PANEL_ID = DOC_LINKS_BY_PANEL_ID;
window.DASHBOARD_DOC_LINKS_BY_TITLE = DOC_LINKS_BY_TITLE;

function _resolvePanelId(title) {
  return PANEL_ID_BY_TITLE[title] || '';
}
window.resolvePanelId = _resolvePanelId;

function _panelOwnershipMeta(panelId) {
  return panelId ? (PANEL_OWNERSHIP[panelId] || null) : null;
}
window.getPanelOwnership = _panelOwnershipMeta;

function _panelAttrs(opts, title) {
  opts = opts || {};
  var panelId = opts.panelId || _resolvePanelId(title);
  var ownership = _panelOwnershipMeta(panelId) || {};
  var ownerTab = opts.ownerTab || ownership.ownerTab || '';
  var attrs = '';
  if (panelId) attrs += ' data-panel-id="' + esc(panelId) + '"';
  if (ownerTab) attrs += ' data-owner-tab="' + esc(ownerTab) + '"';
  if (opts.panelKind) attrs += ' data-panel-kind="' + esc(opts.panelKind) + '"';
  return attrs;
}
window._panelAttrs = _panelAttrs;

function _panelDocsHref(opts, title) {
  opts = opts || {};
  if (opts.docsHref === false) return '';
  if (opts.docsHref) return opts.docsHref;
  var panelId = opts.panelId || _resolvePanelId(title);
  return (panelId && DOC_LINKS_BY_PANEL_ID[panelId]) || DOC_LINKS_BY_TITLE[title] || '';
}

function _panelDocLink(opts, title) {
  var href = _panelDocsHref(opts, title);
  if (!href) return '';
  var label = (opts && opts.docsLabel) || 'Docs';
  return '<a class="panel-doc-link" href="' + esc(href) + '" target="_blank" rel="noopener noreferrer" onclick="event.stopPropagation();" title="Open docs for ' + esc(title) + '">' + esc(label) + '</a>';
}

window.panelDocLink = _panelDocLink;

var TAB_WAYFINDING = {
  cockpit: {
    title: 'Cockpit owns high-level system state.',
    desc: 'Use this tab for top-level health, mode, and alert triage.',
    links: [{ panelId: 'l0-capability-gate', label: 'Trust Layers' }]
  },
  trust: {
    title: 'Trust owns epistemic layers and governance.',
    desc: 'Canonical L0-L10 sequence plus validation, traceability, and benchmark support.',
    links: [
      { panelId: 'l0-capability-gate', label: 'L0' },
      { panelId: 'l5-contradiction-engine', label: 'L5' },
      { panelId: 'l10-soul-integrity', label: 'L10' },
      { panelId: 'trust-trace-explorer', label: 'Trace' }
    ]
  },
  memory: {
    title: 'Memory owns storage, retrieval, and recall behavior.',
    desc: 'Identity boundary enforcement is canonical in Trust.',
    links: [{ panelId: 'l3-identity-boundary', label: 'Open L3 in Trust' }]
  },
  activity: {
    title: 'Activity owns live runtime and perception flow.',
    desc: 'Self-improvement detail is canonical in Learning.',
    links: [{ panelId: 'learning-self-improvement', label: 'Open Self-Improve' }]
  },
  learning: {
    title: 'Learning owns skills, jobs, policy, hemispheres, and self-improvement.',
    desc: 'Trust governance context stays centralized in Trust.',
    links: [{ panelId: 'trust-language-governance', label: 'Open Language Governance' }]
  },
  training: {
    title: 'Training owns operator-guided maturation: companion training, language evidence gates, and synthetic exercises.',
    desc: 'ML internals (hemispheres, policy, matrix) are in Learning.',
    links: [{ panelId: 'learning-hemisphere', label: 'Open Hemispheres in Learning' }]
  },
  diagnostics: {
    title: 'Diagnostics owns engineering internals and runtime debugging.',
    desc: 'Epistemic layers were moved to Trust to remove duplication.',
    links: [{ panelId: 'l0-capability-gate', label: 'Open Trust Layers' }]
  }
};

function _renderTabWayfinding(tabName) {
  var spec = TAB_WAYFINDING[tabName];
  if (!spec) return '';
  var linksHtml = '';
  (spec.links || []).forEach(function(link) {
    var meta = _panelOwnershipMeta(link.panelId) || {};
    var owner = meta.ownerTab || '';
    var text = link.label || meta.label || link.panelId;
    linksHtml += '<button class="j-btn-xs j-owner-jump" onclick="window.openOwnedPanel && window.openOwnedPanel(\'' + esc(link.panelId) + '\')">' +
      esc(text) + (owner ? ' \u2192 ' + esc(owner) : '') + '</button>';
  });
  return '<div class="j-tab-wayfinding">' +
    '<div class="j-tab-wayfinding-title">' + esc(spec.title) + '</div>' +
    '<div class="j-tab-wayfinding-desc">' + esc(spec.desc) + '</div>' +
    (linksHtml ? '<div class="j-tab-wayfinding-links">' + linksHtml + '</div>' : '') +
    '</div>';
}
window._renderTabWayfinding = _renderTabWayfinding;

function _focusPendingPanel() {
  if (!_pendingPanelFocusId) return;
  var section = _tabSection(_activeTab);
  if (!section) return;
  var panel = section.querySelector('.panel[data-panel-id="' + _pendingPanelFocusId + '"]');
  if (!panel) {
    _panelFocusAttempts += 1;
    if (_panelFocusAttempts < 4) {
      requestAnimationFrame(_focusPendingPanel);
    } else {
      _pendingPanelFocusId = '';
      _panelFocusAttempts = 0;
    }
    return;
  }
  if (panel.classList.contains('collapsed')) panel.classList.remove('collapsed');
  panel.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
  panel.classList.add('panel-focus-ring');
  setTimeout(function() { panel.classList.remove('panel-focus-ring'); }, 1200);
  _pendingPanelFocusId = '';
  _panelFocusAttempts = 0;
}

window.openOwnedPanel = function(panelId) {
  if (!panelId) return;
  var meta = _panelOwnershipMeta(panelId) || {};
  var targetTab = meta.ownerTab || _activeTab;
  _pendingPanelFocusId = panelId;
  _panelFocusAttempts = 0;
  if (targetTab !== _activeTab) {
    switchTab(targetTab);
  } else {
    requestAnimationFrame(_focusPendingPanel);
  }
};

function _tabSection(tabName) {
  return document.querySelector('.j-tab-content[data-tab="' + tabName + '"]');
}

function _panelStateKey(panelEl, index) {
  if (!panelEl) return 'panel::' + index;
  var stableId = panelEl.getAttribute('data-panel-id');
  if (stableId) return 'id::' + stableId;
  var titleEl = panelEl.querySelector('.panel-header h3');
  var title = titleEl ? String(titleEl.textContent || '').trim() : '';
  return (title || 'panel') + '::' + index;
}

function _rowStateKey(rowEl, index) {
  if (!rowEl) return 'row::' + index;
  var detailEl = rowEl.querySelector('.dream-cycle-detail,.dream-artifact-detail');
  if (detailEl && detailEl.id) return detailEl.id;
  var cls = rowEl.classList.contains('dream-cycle-row') ? 'dream-cycle-row' :
    (rowEl.classList.contains('dream-artifact-row') ? 'dream-artifact-row' : 'row');
  return cls + '::' + index;
}

function _persistUiState() {
  try {
    localStorage.setItem('jarvis-ui-state-v1', JSON.stringify(_tabUiState));
  } catch (_) {}
}

function _captureTabUiState(tabName) {
  var section = _tabSection(tabName);
  if (!section) return;

  var panelNodes = section.querySelectorAll('.panel');
  var maturityNodes = section.querySelectorAll('[id^="mt-"]');
  var rowNodes = section.querySelectorAll('.dream-cycle-row,.dream-artifact-row');
  if (!panelNodes.length && !maturityNodes.length && !rowNodes.length) return;

  var state = { panels: {}, maturity: {}, rows: {} };

  panelNodes.forEach(function(panelEl, idx) {
    state.panels[_panelStateKey(panelEl, idx)] = panelEl.classList.contains('collapsed');
  });

  maturityNodes.forEach(function(node) {
    if (node.id) state.maturity[node.id] = node.classList.contains('collapsed');
  });

  rowNodes.forEach(function(rowEl, idx) {
    state.rows[_rowStateKey(rowEl, idx)] = rowEl.classList.contains('expanded');
  });

  _tabUiState[tabName] = state;
  _persistUiState();
}

function _restoreTabUiState(tabName) {
  var section = _tabSection(tabName);
  var state = _tabUiState[tabName];
  if (!section || !state) return;

  var panels = section.querySelectorAll('.panel');
  panels.forEach(function(panelEl, idx) {
    var key = _panelStateKey(panelEl, idx);
    if (Object.prototype.hasOwnProperty.call(state.panels, key)) {
      panelEl.classList.toggle('collapsed', !!state.panels[key]);
    }
  });

  Object.keys(state.maturity || {}).forEach(function(id) {
    var node = document.getElementById(id);
    if (node && section.contains(node)) {
      node.classList.toggle('collapsed', !!state.maturity[id]);
    }
  });

  var rows = section.querySelectorAll('.dream-cycle-row,.dream-artifact-row');
  rows.forEach(function(rowEl, idx) {
    var key = _rowStateKey(rowEl, idx);
    if (Object.prototype.hasOwnProperty.call(state.rows || {}, key)) {
      rowEl.classList.toggle('expanded', !!state.rows[key]);
    }
  });
}

try {
  var _savedUiState = localStorage.getItem('jarvis-ui-state-v1');
  if (_savedUiState) {
    var parsed = JSON.parse(_savedUiState);
    if (parsed && typeof parsed === 'object') _tabUiState = parsed;
  }
} catch (_) {}

function switchTab(name) {
  if (TAB_NAMES.indexOf(name) === -1) return;
  _activeTab = name;
  document.querySelectorAll('.j-tab').forEach(function(t) {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.querySelectorAll('.j-tab-content').forEach(function(s) {
    s.classList.toggle('active', s.dataset.tab === name);
  });
  localStorage.setItem('jarvis-tab', name);
  if (_lastSnap) renderActiveTab(_lastSnap);
}

document.querySelectorAll('.j-tab').forEach(function(t) {
  t.addEventListener('click', function() { switchTab(t.dataset.tab); });
});

document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  var n = parseInt(e.key);
  if (n >= 1 && n <= 7) { switchTab(TAB_NAMES[n - 1]); return; }
  if (e.key === 'p' || e.key === 'P') { _togglePause(); return; }
  if (e.key === '?') { el('help-overlay').style.display = 'flex'; return; }
  if (e.key === 'Escape') { el('help-overlay').style.display = 'none'; return; }
});

var savedTab = localStorage.getItem('jarvis-tab');
if (savedTab && TAB_NAMES.indexOf(savedTab) !== -1) switchTab(savedTab);

if (el('help-btn')) {
  el('help-btn').addEventListener('click', function() {
    el('help-overlay').style.display = 'flex';
  });
}


// ═══════════════════════════════════════════════════════════════════════════
// 4. WebSocket + Pause/Throttle
// ═══════════════════════════════════════════════════════════════════════════

var ws = null;
var _lastSnap = null;
window._lastSnap = null;
var _wsUrl = 'ws://' + location.host + '/ws';

var _paused = false;
var _pendingSnap = null;
var _lastRenderTime = 0;
var _MIN_RENDER_INTERVAL_MS = 3000;
var _pendingSnapCount = 0;
var _refreshTickId = 0;

function _togglePause() {
  _paused = !_paused;
  var btn = el('pause-btn');
  if (btn) {
    btn.innerHTML = _paused ? '&#9654;' : '&#9646;&#9646;';
    btn.title = _paused ? 'Resume auto-refresh (P)' : 'Pause auto-refresh (P)';
    btn.classList.toggle('active', _paused);
  }
  if (!_paused && _pendingSnap) {
    _doRender(_pendingSnap);
    _pendingSnap = null;
    _pendingSnapCount = 0;
  }
  _updateRefreshIndicator();
}

function _forceRefresh() {
  var snap = _pendingSnap || _lastSnap;
  if (snap) _doRender(snap);
  _pendingSnap = null;
  _pendingSnapCount = 0;
}

function _doRender(snap) {
  _lastSnap = snap;
  window._lastSnap = snap;
  _lastRenderTime = Date.now();
  updateGlobalUI(snap);
  renderActiveTab(snap);
  _updateRefreshIndicator();
}

function _updateRefreshIndicator() {
  var ind = el('refresh-indicator');
  if (!ind) return;
  if (_paused) {
    ind.textContent = _pendingSnapCount > 0 ? _pendingSnapCount + ' queued' : 'PAUSED';
    ind.style.color = '#f90';
  } else {
    var ago = _lastRenderTime ? Math.round((Date.now() - _lastRenderTime) / 1000) : 0;
    ind.textContent = ago + 's ago';
    ind.style.color = ago > 10 ? '#f90' : '#6a6a80';
  }
}

setInterval(_updateRefreshIndicator, 1000);

function connectWS() {
  ws = new WebSocket(_wsUrl);
  var dot = el('ws-dot');

  ws.onopen = function() {
    if (dot) { dot.classList.remove('disconnected'); dot.classList.add('connected'); dot.title = 'Connected'; }
  };
  ws.onclose = function() {
    if (dot) { dot.classList.remove('connected'); dot.classList.add('disconnected'); dot.title = 'Disconnected'; }
    setTimeout(connectWS, 3000);
  };
  ws.onmessage = function(msg) {
    try {
      var snap = JSON.parse(msg.data);
      _lastSnap = snap;
      window._lastSnap = snap;

      if (_paused) {
        _pendingSnap = snap;
        _pendingSnapCount++;
        _updateRefreshIndicator();
        return;
      }

      var now = Date.now();
      if (now - _lastRenderTime < _MIN_RENDER_INTERVAL_MS) {
        _pendingSnap = snap;
        if (!_refreshTickId) {
          _refreshTickId = setTimeout(function() {
            _refreshTickId = 0;
            if (!_paused && _pendingSnap) {
              _doRender(_pendingSnap);
              _pendingSnap = null;
              _pendingSnapCount = 0;
            }
          }, _MIN_RENDER_INTERVAL_MS - (now - _lastRenderTime));
        }
        return;
      }

      _doRender(snap);
      _pendingSnap = null;
      _pendingSnapCount = 0;
    } catch (err) {
      console.error('[dashboard] render error:', err);
    }
  };
}

(function _wirePauseControls() {
  var btn = el('pause-btn');
  if (btn) btn.addEventListener('click', _togglePause);
  var ind = el('refresh-indicator');
  if (ind) {
    ind.addEventListener('click', _forceRefresh);
    ind.style.cursor = 'pointer';
  }
})();


// ═══════════════════════════════════════════════════════════════════════════
// 5. Render Dispatcher
// ═══════════════════════════════════════════════════════════════════════════

function renderActiveTab(snap) {
  _captureTabUiState(_activeTab);
  switch (_activeTab) {
    case 'cockpit': renderCockpit(snap); break;
    case 'trust': renderTrust(snap); break;
    case 'memory': renderMemory(snap); break;
    case 'activity': if (window.renderActivity) window.renderActivity(snap); break;
    case 'learning': if (window.renderLearning) window.renderLearning(snap); break;
    case 'training': if (window.renderTraining) window.renderTraining(snap); break;
    case 'diagnostics': if (window.renderDiagnostics) window.renderDiagnostics(snap); break;
  }
  _restoreTabUiState(_activeTab);
  _focusPendingPanel();
  if (window.updateSparklines) window.updateSparklines(snap);
}


// ═══════════════════════════════════════════════════════════════════════════
// 6. Global UI Updates
// ═══════════════════════════════════════════════════════════════════════════

function updateGlobalUI(snap) {
  var summary = snap.summary || {};
  var health = snap.health || {};
  var mode = snap.mode || {};
  var identity = snap.identity || {};

  // Boot banner
  var bootBanner = el('boot-banner');
  var uptime = (health.uptime_s != null) ? health.uptime_s : (summary.uptime_s || 999);
  if (bootBanner) {
    bootBanner.style.display = uptime < 120 ? '' : 'none';
  }

  // Action-needed strip
  var strip = el('action-strip');
  var actions = summary.actions || [];
  if (strip) {
    if (actions.length) {
      var sevColors = { error: '#f44', warning: '#f90', info: '#0cf' };
      strip.innerHTML = actions.map(function(a) {
        var c = sevColors[a.severity] || '#6a6a80';
        return '<span style="color:' + c + ';">\u25cf ' + esc(a.text) + '</span>';
      }).join(' &nbsp; ');
      strip.style.display = '';
    } else {
      strip.style.display = 'none';
    }
  }

  // Footer
  var fm = el('footer-mode');
  var fp = el('footer-phase');
  var fu = el('footer-uptime');
  var fi = el('footer-identity');
  var fc = el('footer-cache-age');
  if (fm) fm.textContent = summary.mode || (mode.mode || '--');
  if (fp) fp.textContent = summary.activity || (snap.core || {}).phase || '--';
  if (fu) fu.textContent = summary.uptime || fmtUptime(health.uptime_s);
  if (fi) fi.textContent = summary.who || (identity.identity || '--');
  if (fc) fc.textContent = snap._ts ? timeAgo(snap._ts) : '--';
}


// ═══════════════════════════════════════════════════════════════════════════
// Stale-code banner — polls /api/system/code-freshness every 60s and shows
// a non-blocking "newer code on disk, click Restart" nudge after sync.
// Independent from the main snapshot loop so a filesystem walk never taxes
// the 2s dashboard refresh.
// ═══════════════════════════════════════════════════════════════════════════

var _STALE_BANNER_DISMISSED_UNTIL = 0;  // epoch seconds; banner hidden if now < this

function _fmtStaleAge(s) {
  s = Math.max(0, Math.round(s || 0));
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.round(s / 60) + 'm ago';
  if (s < 86400) return (s / 3600).toFixed(1) + 'h ago';
  return (s / 86400).toFixed(1) + 'd ago';
}

function _renderStaleCodeBanner(data) {
  var banner = el('stale-code-banner');
  if (!banner) return;
  if (!data || !data.is_stale) {
    banner.style.display = 'none';
    banner.innerHTML = '';
    return;
  }
  if (Date.now() / 1000 < _STALE_BANNER_DISMISSED_UNTIL) {
    banner.style.display = 'none';
    return;
  }
  var file = data.newest_file ? String(data.newest_file) : '';
  var age = _fmtStaleAge(data.stale_age_s);
  banner.innerHTML =
    '<div class="j-stale-msg">' +
      '<span>\u26A0</span>' +
      '<span><strong>Newer code on disk</strong> \u2014 edited ' + esc(age) +
      ', after this process started. Restart to load the new code.</span>' +
      (file ? '<span class="j-stale-file" title="' + esc(file) + '">' + esc(file) + '</span>' : '') +
    '</div>' +
    '<div class="j-stale-actions">' +
      '<button class="j-stale-restart" onclick="window._staleRestart && window._staleRestart()">Restart</button>' +
      '<button onclick="window._staleDismiss && window._staleDismiss()">Dismiss 1h</button>' +
    '</div>';
  banner.style.display = '';
}

window._staleRestart = function() {
  if (window.systemRestart) {
    window.systemRestart();
  }
};

window._staleDismiss = function() {
  _STALE_BANNER_DISMISSED_UNTIL = (Date.now() / 1000) + 3600;
  var banner = el('stale-code-banner');
  if (banner) banner.style.display = 'none';
};

function _pollStaleCode() {
  fetch('/api/system/code-freshness', { cache: 'no-store' })
    .then(function(r) { return r.ok ? r.json() : null; })
    .then(function(data) { _renderStaleCodeBanner(data); })
    .catch(function() { /* silent — this is a nudge, not critical */ });
}

// First check shortly after load, then every 60s.
setTimeout(_pollStaleCode, 3000);
setInterval(_pollStaleCode, 60000);


// ═══════════════════════════════════════════════════════════════════════════
// Shared panel builders
// ═══════════════════════════════════════════════════════════════════════════

function _panelOpen(title, opts) {
  opts = opts || {};
  var badge = opts.badge || '';
  var collapsed = opts.collapsed ? ' collapsed' : '';
  var attrs = _panelAttrs(opts, title);
  var docsLink = _panelDocLink(opts, title);
  return '<div class="panel' + collapsed + '"' + attrs + '>' +
    '<div class="panel-header" onclick="window._togglePanel && window._togglePanel(this.parentElement)">' +
    '<span class="panel-chevron">\u25bc</span>' +
    '<h3>' + esc(title) + '</h3>' +
    '<span class="panel-header-actions">' + docsLink + (badge ? '<span>' + badge + '</span>' : '') + '</span>' +
    '</div>' +
    '<div class="panel-body">';
}
function _panelClose() { return '</div></div>'; }

window._togglePanel = function(panelEl) {
  if (!panelEl) return;
  var wasCollapsed = panelEl.classList.contains('collapsed');
  panelEl.classList.toggle('collapsed');

  // Charts rendered while hidden can end up blank; redraw on expand.
  if (wasCollapsed && !panelEl.classList.contains('collapsed')) {
    var stabCanvas = panelEl.querySelector('#trust-stability-chart');
    if (stabCanvas && window.renderStabilityChart) {
      var snap = window._lastSnap || null;
      var stability = snap && snap.eval ? snap.eval.stability : null;
      if (stability) {
        requestAnimationFrame(function() {
          window.renderStabilityChart(stabCanvas, stability);
        });
      }
    }
  }
};

function _statusBadge(status) {
  var colors = {
    active: '#0f9', running: '#0f9', done: '#0cf', completed: '#0cf',
    idle: '#6a6a80', waiting: '#ff0', queued: '#ff0', blocked: '#f44',
    error: '#f44', shadow: '#c0f', live: '#0f9', disabled: '#f44',
    learning: '#ff0', verified: '#0f9', unknown: '#6a6a80',
    pass: '#0f9', fail: '#f44', warning: '#ff0',
    promoted: '#0f9', candidate: '#ff0', proposed: '#ff0',
    stalled: '#f90', abandoned: '#f44', paused: '#ff0',
    grounded: '#0f9', provisional: '#0cf', conflicted: '#ff0', degraded: '#f44'
  };
  var c = colors[(status || '').toLowerCase()] || '#6a6a80';
  return '<span class="status-badge" style="color:' + c + ';border-color:' + c + '33;background:' + c + '15;">' + esc(status || '--') + '</span>';
}
window._statusBadge = _statusBadge;

function _metricRow(label, value) {
  return '<div class="metric-row"><span class="metric-label">' + esc(label) +
    '</span><span class="metric-value">' + esc(String(value != null ? value : '--')) + '</span></div>';
}

function _statCard(label, value, color) {
  var c = color || '#e0e0e8';
  return '<div class="stat-card"><div class="stat-value" style="color:' + c + ';">' +
    esc(String(value != null ? value : '--')) + '</div><div class="stat-label">' + esc(label) + '</div></div>';
}

function _barFill(value, max, color) {
  var p = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return '<div class="bar-track"><div class="bar-fill" style="width:' + p + '%;background:' + (color || '#0f9') + ';"></div></div>';
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


// ═══════════════════════════════════════════════════════════════════════════
// 7. Cockpit Renderer
// ═══════════════════════════════════════════════════════════════════════════

var _prevSnap = null;
var _transitions = [];

function _renderGestationProtocol(g) {
  var html = '';

  // Phase progress banner
  var phaseIdx = g.phase || 0;
  var phases = ['Self Discovery', 'Knowledge Foundation', 'Autonomy Bootcamp', 'Identity Formation'];
  var phaseColors = ['#0cf', '#c0f', '#f90', '#0f9'];
  var phaseName = g.phase_name ? g.phase_name.replace(/_/g, ' ') : (phases[phaseIdx] || 'Unknown');

  // Elapsed time
  var elapsed = g.elapsed_s || 0;
  var hours = Math.floor(elapsed / 3600);
  var mins = Math.floor((elapsed % 3600) / 60);
  var elapsedStr = hours > 0 ? hours + 'h ' + mins + 'm' : mins + 'm';

  // Overall readiness
  var readiness = g.readiness || {};
  var overall = readiness.overall || 0;
  var rec = readiness.recommendation || 'continue';
  var recColor = rec === 'ready' ? '#0f9' : rec === 'ready_person_waiting' ? '#ff0' : '#6a6a80';
  var recLabel = rec === 'ready' ? 'READY TO GRADUATE' : rec === 'ready_person_waiting' ? 'READY (Person Waiting)' : 'Continuing...';

  // Header
  html += '<div style="background:linear-gradient(135deg,#0a0a20,#12122a);border:2px solid ' + phaseColors[phaseIdx] + ';border-radius:8px;padding:16px;margin-bottom:12px;">';
  html += '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">';
  html += '<div style="display:flex;align-items:center;gap:10px;">';
  html += '<div style="font-size:1.8rem;color:' + phaseColors[phaseIdx] + ';">\u2726</div>';
  html += '<div><div style="font-size:1.1rem;font-weight:700;color:#e0e0e8;">Birthing Protocol</div>';
  html += '<div style="font-size:0.7rem;color:#6a6a80;">Phase ' + phaseIdx + ' of 3 &mdash; ' + esc(phaseName) + '</div></div>';
  html += '</div>';
  html += '<div style="text-align:right;">';
  html += '<div style="font-size:1.4rem;font-weight:700;color:' + (overall >= 0.8 ? '#0f9' : overall >= 0.5 ? '#ff0' : '#f44') + ';">' + (overall * 100).toFixed(0) + '%</div>';
  html += '<div style="font-size:0.6rem;color:#6a6a80;">Overall Readiness</div>';
  html += '</div></div>';

  // Phase timeline
  html += '<div style="display:flex;gap:4px;margin-bottom:14px;">';
  for (var p = 0; p < 4; p++) {
    var isActive = p === phaseIdx;
    var isDone = p < phaseIdx;
    var pc = isDone ? '#0f9' : isActive ? phaseColors[p] : '#1a1a2e';
    var border = isActive ? '2px solid ' + phaseColors[p] : '1px solid ' + (isDone ? '#0f9' : '#2a2a44');
    html += '<div style="flex:1;padding:6px 8px;background:' + (isActive ? '#0d0d1a' : 'transparent') + ';border:' + border + ';border-radius:4px;text-align:center;">';
    html += '<div style="font-size:0.55rem;color:' + pc + ';font-weight:' + (isActive ? '700' : '400') + ';">' + phases[p] + '</div>';
    if (isDone) html += '<div style="font-size:0.5rem;color:#0f9;">&#10003; Done</div>';
    else if (isActive) html += '<div style="font-size:0.5rem;color:' + phaseColors[p] + ';">In Progress</div>';
    html += '</div>';
  }
  html += '</div>';

  // Stats row
  html += '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-bottom:12px;">' +
    _statCard('Elapsed', elapsedStr, '#6a6a80') +
    _statCard('Directives', (g.directives_completed || 0) + '/' + (g.directives_issued || 0), '#0cf') +
    _statCard('Research Jobs', g.research_jobs_completed || 0, '#c0f') +
    _statCard('Status', recLabel, recColor) +
    _statCard('Min Duration', readiness.met_minimum_duration ? 'Met' : 'Not yet', readiness.met_minimum_duration ? '#0f9' : '#f90') +
    '</div>';

  // Readiness components — the core of the birthing dashboard
  var comps = readiness.components || {};
  var compKeys = Object.keys(comps);
  if (compKeys.length) {
    html += '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:6px;font-weight:600;">Readiness Components</div>';
    html += '<div style="display:grid;gap:6px;">';
    var compLabels = {
      self_knowledge: 'Self Knowledge', knowledge_foundation: 'Knowledge Foundation',
      memory_mass: 'Memory Mass', consciousness_stage: 'Consciousness Stage',
      hemisphere_training: 'Hemisphere Training', personality_emergence: 'Personality Emergence',
      policy_experience: 'Policy Experience', loop_integrity: 'Loop Integrity'
    };
    compKeys.forEach(function(k) {
      var v = comps[k] || 0;
      var pct = Math.min(100, v * 100);
      var label = compLabels[k] || k.replace(/_/g, ' ');
      var barC = v >= 0.8 ? '#0f9' : v >= 0.5 ? '#ff0' : v >= 0.2 ? '#f90' : '#f44';
      html += '<div style="display:flex;align-items:center;gap:8px;">';
      html += '<div style="width:130px;font-size:0.62rem;color:#aaa;">' + esc(label) + '</div>';
      html += '<div style="flex:1;height:8px;background:#1a1a2e;border-radius:4px;overflow:hidden;">';
      html += '<div style="width:' + pct + '%;height:100%;background:' + barC + ';border-radius:4px;transition:width 0.5s ease;"></div></div>';
      html += '<div style="width:38px;text-align:right;font-size:0.62rem;font-weight:700;color:' + barC + ';">' + pct.toFixed(0) + '%</div>';
      html += '</div>';
    });
    html += '</div>';
  }

  // Phase-specific queue info
  var pc_ = g.phase_completions || {};
  var selfRemain = g.self_study_remaining || 0;
  var knowRemain = g.knowledge_remaining || 0;
  var bootRemain = g.bootcamp_remaining || 0;
  var selfDone = pc_.self_study || 0;
  var knowDone = pc_.knowledge || 0;
  var bootDone = pc_.bootcamp || 0;

  html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:12px;">';
  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;text-align:center;">';
  html += '<div style="font-size:0.55rem;color:#0cf;margin-bottom:2px;">Self Study</div>';
  html += '<div style="font-size:0.85rem;font-weight:700;color:#e0e0e8;">' + selfDone + '</div>';
  html += '<div style="font-size:0.48rem;color:#6a6a80;">' + selfRemain + ' remaining</div></div>';

  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;text-align:center;">';
  html += '<div style="font-size:0.55rem;color:#c0f;margin-bottom:2px;">Knowledge</div>';
  html += '<div style="font-size:0.85rem;font-weight:700;color:#e0e0e8;">' + knowDone + '</div>';
  html += '<div style="font-size:0.48rem;color:#6a6a80;">' + knowRemain + ' remaining</div></div>';

  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;text-align:center;">';
  html += '<div style="font-size:0.55rem;color:#f90;margin-bottom:2px;">Bootcamp</div>';
  html += '<div style="font-size:0.85rem;font-weight:700;color:#e0e0e8;">' + bootDone + '</div>';
  html += '<div style="font-size:0.48rem;color:#6a6a80;">' + bootRemain + ' remaining</div></div>';
  html += '</div>';

  // Sensor / network status
  html += '<div style="display:flex;gap:12px;margin-top:10px;font-size:0.58rem;">';
  var netC = g.network_healthy ? '#0f9' : '#f44';
  html += '<span style="color:' + netC + ';">' + (g.network_healthy ? '\u25cf' : '\u25cb') + ' Pi Connected</span>';
  var personC = g.person_detected ? '#0f9' : '#6a6a80';
  html += '<span style="color:' + personC + ';">' + (g.person_detected ? '\u25cf' : '\u25cb') + ' Person Detected' +
    (g.person_sustained_s > 0 ? ' (' + g.person_sustained_s.toFixed(0) + 's)' : '') + '</span>';
  var fcC = g.first_contact_armed ? '#ff0' : '#6a6a80';
  html += '<span style="color:' + fcC + ';">' + (g.first_contact_armed ? '\u25cf Armed' : '\u25cb Not armed') + ' First Contact</span>';
  if (g.backpressure_active) html += '<span style="color:#f90;">\u26a0 Backpressure Active</span>';
  html += '</div>';

  html += '</div>';  // close main container
  return html;
}

function _renderBirthReadinessSnapshot(g) {
  var birth = g.birth_snapshot || {};
  var readiness = g.readiness || {};
  var post = g.post_birth_progress || {};
  var overall = readiness.overall || 0;
  var policy = post.policy_experience || 0;
  var personality = post.personality_emergence || 0;

  var html = '<div style="background:linear-gradient(135deg,#0a0a20,#12122a);border:1px solid #2a2a44;border-radius:8px;padding:12px;margin-bottom:12px;">';
  html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">';
  html += '<div style="font-size:0.9rem;font-weight:700;color:#e0e0e8;">Birth Readiness Snapshot</div>';
  html += '<div style="font-size:0.58rem;color:#6a6a80;">Frozen at graduation</div>';
  html += '</div>';

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:6px;">' +
    _statCard('Birth Readiness', (overall * 100).toFixed(0) + '%', overall >= 0.8 ? '#0f9' : '#ff0') +
    _statCard('Policy Progress', (policy * 100).toFixed(0) + '%', policy >= 0.5 ? '#0f9' : '#ff0') +
    _statCard('Personality Progress', (personality * 100).toFixed(0) + '%', personality >= 0.5 ? '#0f9' : '#ff0') +
    _statCard('Birth Duration', birth.duration_s ? fmtUptime(birth.duration_s) : '--', '#6a6a80') +
    '</div>';

  html += '<div style="font-size:0.55rem;color:#6a6a80;">Live trust uses Truth Calibration and Soul Integrity. Birth readiness is historical context.</div>';
  html += '</div>';
  return html;
}

function renderCockpit(snap) {
  var root = el('cockpit-root');
  if (!root) return;

  var s = snap.summary || {};
  var ts = snap.trust_state || {};
  var health = snap.health || {};
  var con = snap.consciousness || {};
  var mode = snap.mode || {};
  var auto = snap.autonomy || {};
  var att = snap.attention || {};

  var html = '';
  html += _renderTabWayfinding('cockpit');

  // Gestation panel — shown when birthing protocol is active
  var g = snap.gestation || {};
  if (g.active) {
    html += _renderGestationProtocol(g);
  } else if (g.graduated && g.readiness_source === 'birth_certificate') {
    html += _renderBirthReadinessSnapshot(g);
  }

  // Row 1: Compact Status Bar (trust + activity + mode + stage + uptime)
  var trustColors = { grounded: '#0f9', provisional: '#0cf', conflicted: '#ff0', degraded: '#f44', unknown: '#6a6a80' };
  var tc_ = trustColors[ts.state] || '#6a6a80';
  var reasons = ts.reasons || [];
  var reasonText = reasons.length ? reasons.join(', ') : '';

  html += '<div style="display:flex;align-items:center;gap:12px;padding:8px 12px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:6px;margin-bottom:10px;flex-wrap:wrap;">';
  html += '<div style="display:flex;align-items:center;gap:6px;">' +
    '<div style="width:8px;height:8px;border-radius:50%;background:' + tc_ + ';"></div>' +
    '<span style="font-size:0.85rem;font-weight:700;color:' + tc_ + ';">' + esc(s.trust_label || ts.state || 'Unknown') + '</span>' +
    (reasonText ? '<span style="font-size:0.6rem;color:#6a6a80;">(' + esc(reasonText) + ')</span>' : '') +
    '</div>';
  html += '<div style="display:flex;gap:6px;margin-left:auto;">';
  html += '<span style="font-size:0.62rem;padding:2px 8px;background:#1a1a2e;border-radius:3px;color:#e0e0e8;">' + esc(s.activity || '--') + '</span>';
  html += '<span style="font-size:0.62rem;padding:2px 8px;background:#1a1a2e;border-radius:3px;color:#c0f;">' + esc(s.mode || '--') + '</span>';
  if (con.boot_stabilization_active) {
    var remain = Number(con.boot_stabilization_remaining_s || 0);
    var remainText = remain > 0 ? fmtUptime(remain) : '--';
    html += '<span style="font-size:0.62rem;padding:2px 8px;background:#1a1a2e;border-radius:3px;color:#f90;">Stabilizing ' + esc(remainText) + '</span>';
  }
  if (con.stage) html += '<span style="font-size:0.62rem;padding:2px 8px;background:#1a1a2e;border-radius:3px;color:#0cf;">' + esc(con.stage) + '</span>';
  html += '<span style="font-size:0.62rem;padding:2px 8px;background:#1a1a2e;border-radius:3px;color:#6a6a80;">' + esc(s.uptime || '--') + '</span>';
  html += '</div></div>';

  // Row 2: Identity
  var presenceIcon = s.user_present ? '\u25cf' : '\u25cb';
  var presenceColor = s.user_present ? '#0f9' : '#6a6a80';
  var confText = s.who_confidence ? ' (' + (s.who_confidence * 100).toFixed(0) + '% sure)' : '';
  html += '<div class="j-cockpit-identity">' +
    '<span style="color:' + presenceColor + ';font-size:1rem;margin-right:6px;">' + presenceIcon + '</span>' +
    '<span class="j-cockpit-who">' + esc(s.who || 'Nobody') + '</span>' +
    '<span class="j-cockpit-who-conf">' + esc(confText) + '</span>';
  if (att.user_emotion && att.emotion_confidence > 0.3) {
    html += '<span style="margin-left:12px;font-size:0.6rem;color:#6a6a80;">mood: <span style="color:#f90;">' + esc(att.user_emotion) + '</span></span>';
  }
  html += '</div>';

  // Row 3: Vitals (6 cards)
  var errorCount = (health.error_count || 0);
  var errorColor = errorCount > 3 ? '#f44' : errorCount > 0 ? '#f90' : '#0f9';
  var si = snap.soul_integrity || {};
  var tcal = snap.truth_calibration || {};
  html += '<div class="section-grid" style="grid-template-columns:repeat(6,1fr);margin-bottom:10px;">' +
    _statCard('Memories', s.memory_count || 0, '#0cf') +
    _statCard('Core', s.core_memory_count || 0, '#c0f') +
    _statCard('Engagement', s.engagement != null ? (s.engagement * 100).toFixed(0) + '%' : '--', '#f90') +
    _statCard('Soul', si.current_index != null ? (si.current_index * 100).toFixed(0) + '%' : '--', si.current_index > 0.7 ? '#0f9' : '#ff0') +
    _statCard('Truth (Live)', tcal.truth_score != null ? (tcal.truth_score * 100).toFixed(0) + '%' : '--', tcal.truth_score >= 0.6 ? '#0f9' : '#ff0') +
    _statCard('Errors', errorCount, errorColor) +
    '</div>';

  // Row 4: Actions Needed
  var actions = s.actions || [];
  html += '<div class="j-cockpit-actions">';
  if (actions.length) {
    var sevColors = { error: '#f44', warning: '#f90', info: '#0cf' };
    actions.forEach(function(a) {
      var c = sevColors[a.severity] || '#6a6a80';
      html += '<div class="j-cockpit-action" style="border-left:3px solid ' + c + ';">' +
        '<span style="color:' + c + ';font-weight:600;">' + esc(a.severity || 'info') + '</span> ' +
        esc(a.text) + '</div>';
    });
  } else {
    html += '<div class="j-cockpit-action j-cockpit-action-ok">Everything looks good</div>';
  }
  html += '</div>';

  // Row 5: Latest Thought
  if (s.recent_insight) {
    html += '<div class="j-cockpit-insight">' +
      '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:2px;">Latest thought</div>' +
      '<div style="font-size:0.78rem;color:#e0e0e8;">' + esc(s.recent_insight) + '</div></div>';
  }

  // Row 6: Two-column: Quick Cards + Drives
  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;">';

  // Left: Quick access cards
  var lj = snap.learning_jobs || {};
  html += '<div>';
  var ob = snap.onboarding || {};
  var obReadiness = (ob.readiness_latest || {}).composite;
  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:6px;">' +
    '<div class="j-cockpit-qcard" onclick="switchTab(\'trust\')">' +
    '<div class="stat-value" style="color:#0f9;">' + (si.current_index != null ? (si.current_index * 100).toFixed(0) + '%' : '--') + '</div>' +
    '<div class="stat-label">Trust</div>' +
    '<div class="j-cockpit-qlink">View \u2192</div></div>' +
    '<div class="j-cockpit-qcard" onclick="switchTab(\'memory\')">' +
    '<div class="stat-value" style="color:#0cf;">' + (s.memory_count || 0) + '</div>' +
    '<div class="stat-label">Memories</div>' +
    '<div class="j-cockpit-qlink">View \u2192</div></div>' +
    '<div class="j-cockpit-qcard" onclick="switchTab(\'learning\')">' +
    '<div class="stat-value" style="color:#ff0;">' + (lj.active_count || 0) + '</div>' +
    '<div class="stat-label">Learning</div>' +
    '<div class="j-cockpit-qlink">View \u2192</div></div>' +
    '<div class="j-cockpit-qcard" onclick="switchTab(\'training\')">' +
    '<div class="stat-value" style="color:#c0f;">' + (obReadiness != null ? (obReadiness * 100).toFixed(0) + '%' : '--') + '</div>' +
    '<div class="stat-label">Training</div>' +
    '<div class="j-cockpit-qlink">View \u2192</div></div>' +
    '</div>';

  // Autonomy summary under quick cards
  var autoLevel = auto.autonomy_level != null ? auto.autonomy_level : (auto.level != null ? auto.level : '--');
  var autoName = auto.autonomy_level_name || '';
  var promo = auto.promotion || {};
  html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;">' +
    _statCard('Auto Level', autoLevel + (autoName ? ' (' + autoName.replace(/_/g,' ') + ')' : ''), '#0cf') +
    _statCard('Win Rate', promo.win_rate != null ? (promo.win_rate * 100).toFixed(0) + '%' : '--', promo.win_rate > 0.5 ? '#0f9' : '#ff0') +
    _statCard('Completed', auto.completed_total || auto.completed_total_session || 0, '#6a6a80') +
    '</div>';
  html += '</div>';

  // Right: Drives
  var drivesObj = (auto.drives || {}).drives || {};
  var driveKeys = Object.keys(drivesObj);
  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:6px;padding:8px;">';
  html += '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:6px;">Motive Drives (' + driveKeys.length + ')</div>';
  if (driveKeys.length) {
    driveKeys.forEach(function(dk) {
      var drv = drivesObj[dk] || {};
      var urg = drv.urgency || 0;
      var uc = urg > 0.5 ? '#f44' : urg > 0.2 ? '#ff0' : urg > 0 ? '#0f9' : '#484860';
      var outcomeC = drv.last_outcome === 'positive' ? '#0f9' : drv.last_outcome === 'negative' ? '#f44' : '#6a6a80';
      var suppressed = drv.suppression ? true : false;
      html += '<div style="display:flex;align-items:center;gap:6px;padding:2px 0;font-size:0.58rem;' + (suppressed ? 'opacity:0.5;' : '') + '">' +
        '<div style="width:50px;color:#e0e0e8;font-weight:600;">' + esc(dk) + '</div>' +
        '<div style="flex:1;height:4px;background:#1a1a2e;border-radius:2px;overflow:hidden;">' +
        '<div style="width:' + Math.min(100, urg * 100) + '%;height:100%;background:' + uc + ';border-radius:2px;"></div></div>' +
        '<div style="width:28px;text-align:right;color:' + uc + ';">' + fmtNum(urg, 2) + '</div>' +
        '<div style="width:16px;text-align:center;color:' + outcomeC + ';">' +
        (drv.last_outcome === 'positive' ? '\u2713' : drv.last_outcome === 'negative' ? '\u2717' : '\u25cf') + '</div>' +
        '<div style="width:20px;text-align:right;color:#484860;">' + (drv.action_count || 0) + '</div>' +
        (suppressed ? '<span style="color:#f90;font-size:0.45rem;" title="' + esc(drv.suppression || '') + '">\u26a0</span>' : '') +
        '</div>';
    });
  } else {
    html += '<div style="font-size:0.6rem;color:#484860;">No drive data available</div>';
  }
  html += '</div>';
  html += '</div>';

  // Row 7: What Changed
  if (_prevSnap) {
    var ps = _prevSnap.summary || {};
    var prevTs = _prevSnap.trust_state || {};
    if (ps.mode !== s.mode) _transitions.push({ time: Date.now(), text: 'Mode: ' + (ps.mode || '--') + ' \u2192 ' + (s.mode || '--') });
    if (ps.activity !== s.activity) _transitions.push({ time: Date.now(), text: 'Phase: ' + (ps.activity || '--') + ' \u2192 ' + (s.activity || '--') });
    if (prevTs.state !== ts.state) _transitions.push({ time: Date.now(), text: 'Trust: ' + (prevTs.state || '--') + ' \u2192 ' + (ts.state || '--') });
    if (ps.who !== s.who) _transitions.push({ time: Date.now(), text: 'Identity: ' + (ps.who || '--') + ' \u2192 ' + (s.who || '--') });
    while (_transitions.length > 10) _transitions.shift();
  }
  if (_transitions.length) {
    html += '<div class="j-cockpit-transitions">' +
      '<div style="font-size:0.65rem;color:#6a6a80;margin-bottom:3px;">Recent changes</div>';
    _transitions.slice().reverse().slice(0, 5).forEach(function(t) {
      var ago = Math.floor((Date.now() - t.time) / 1000);
      html += '<div style="font-size:0.6rem;padding:1px 0;color:#aaa;">' +
        '<span style="color:#484860;min-width:36px;display:inline-block;">' + ago + 's</span> ' +
        esc(t.text) + '</div>';
    });
    html += '</div>';
  }
  _prevSnap = snap;

  // Row 8: Personality
  html += _renderPersonalityPanel(snap);

  // Row 9: Quick Actions toolbar
  html += '<div class="j-cockpit-toolbar">' +
    '<button class="j-btn-sm" onclick="window.openChat()">Chat</button>' +
    '<button class="j-btn-sm" onclick="window.openMemorySearch()">Search Memories</button>' +
    '<button class="j-btn-sm" onclick="window.openCameraFeed()">Camera</button>' +
    '<button class="j-btn-sm" onclick="window.openEnrollment()">Enroll</button>' +
    '<button class="j-btn-sm" onclick="window.openSettings()">Settings</button>' +
    '<button class="j-btn-sm" onclick="window.openVoiceTest()">Voice Test</button>' +
    '<button class="j-btn-sm" onclick="window.systemSave()">Save</button>' +
    '<button class="j-btn-sm j-btn-red" onclick="window.systemRestart()">Restart</button>' +
    '</div>';

  root.innerHTML = html;
}


// ═══════════════════════════════════════════════════════════════════════════
// 7b. Personality Panel
// ═══════════════════════════════════════════════════════════════════════════

function _renderPersonalityPanel(snap) {
  var p = snap.personality || {};
  var archetypes = p.archetypes || [];
  if (!archetypes.length) return '';

  var rb = p.rollback || {};
  var vd = p.validation || {};
  var dims = p.soul_dims || {};

  var html = _panelOpen('Personality', {
    badge: '<span style="font-size:0.6rem;color:#c0f;">' + esc(p.dominant || '--') + '</span>'
  });

  // Top info bar
  var ageTxt = p.age_days != null ? p.age_days.toFixed(1) + 'd' : '--';
  html += '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;font-size:0.65rem;color:#6a6a80;">' +
    '<span>Age: <b style="color:#e0e0e8;">' + ageTxt + '</b></span>' +
    '<span>Mood: <b style="color:#f90;">' + esc(p.mood || 'neutral') + '</b></span>' +
    '<span>Relationships: <b style="color:#0cf;">' + (p.relationships || 0) + '</b></span>' +
    '<span>Interactions: <b style="color:#0f9;">' + (p.interaction_count || 0) + '</b></span>' +
    '<span>Stability: <b style="color:' + (rb.current_stability > 0.5 ? '#0f9' : rb.current_stability > 0.3 ? '#f90' : '#f44') + ';">' + (rb.current_stability != null ? rb.current_stability.toFixed(2) : '--') + '</b></span>' +
    (rb.in_emergency ? '<span style="color:#f44;font-weight:700;">EMERGENCY</span>' : '') +
    (vd.in_post_birth_grace ? '<span style="color:#c0f;">post-birth grace</span>' : '') +
    '</div>';

  // Archetype bars
  html += '<div style="display:flex;gap:10px;flex-wrap:wrap;">';

  // Left: Bar chart
  html += '<div style="flex:1;min-width:220px;">';
  var trendIcons = { rising: '\u25b2', declining: '\u25bc', stable: '\u25cf' };
  var trendColors = { rising: '#0f9', declining: '#f44', stable: '#6a6a80' };
  archetypes.forEach(function(a) {
    var pct = Math.round(a.score * 100);
    var isActive = a.score >= 0.15;
    var barColor = isActive ? '#c0f' : '#2a2a3e';
    var textColor = isActive ? '#e0e0e8' : '#484860';
    var trend = trendIcons[a.trend] || '';
    var tCol = trendColors[a.trend] || '#6a6a80';
    html += '<div style="margin-bottom:4px;">' +
      '<div style="display:flex;justify-content:space-between;font-size:0.62rem;color:' + textColor + ';">' +
      '<span>' + esc(a.name) + ' <span style="color:' + tCol + ';font-size:0.5rem;">' + trend + '</span></span>' +
      '<span>' + pct + '%</span></div>' +
      '<div style="height:6px;background:#1a1a2e;border-radius:3px;overflow:hidden;">' +
      '<div style="height:100%;width:' + pct + '%;background:' + barColor + ';border-radius:3px;transition:width 0.5s;"></div>' +
      '</div></div>';
  });
  html += '</div>';

  // Right: Radar chart (SVG)
  html += '<div style="flex:0 0 160px;display:flex;align-items:center;justify-content:center;">';
  html += _renderTraitRadar(archetypes);
  html += '</div>';

  html += '</div>';

  // Soul dimensions row
  var dimKeys = Object.keys(dims);
  if (dimKeys.length) {
    html += '<div style="margin-top:8px;padding-top:6px;border-top:1px solid #1a1a2e;">' +
      '<div style="font-size:0.6rem;color:#6a6a80;margin-bottom:4px;">Soul Dimensions (identity.json)</div>' +
      '<div style="display:flex;gap:8px;flex-wrap:wrap;">';
    dimKeys.forEach(function(d) {
      var val = dims[d];
      var pct = Math.round(val * 100);
      html += '<div style="flex:1;min-width:80px;max-width:120px;">' +
        '<div style="font-size:0.55rem;color:#6a6a80;text-transform:capitalize;">' + esc(d) + '</div>' +
        '<div style="height:4px;background:#1a1a2e;border-radius:2px;overflow:hidden;margin-top:2px;">' +
        '<div style="height:100%;width:' + pct + '%;background:#0cf;border-radius:2px;"></div>' +
        '</div>' +
        '<div style="font-size:0.5rem;color:#484860;text-align:right;">' + pct + '%</div>' +
        '</div>';
    });
    html += '</div></div>';
  }

  html += _panelClose();
  return html;
}

function _renderTraitRadar(archetypes) {
  if (!archetypes || archetypes.length < 3) return '';
  var cx = 75, cy = 75, r = 55;
  var n = archetypes.length;
  var svg = '<svg width="150" height="150" viewBox="0 0 150 150">';

  // Background rings
  [0.25, 0.5, 0.75, 1.0].forEach(function(ring) {
    var pts = [];
    for (var i = 0; i < n; i++) {
      var angle = (Math.PI * 2 * i / n) - Math.PI / 2;
      pts.push((cx + r * ring * Math.cos(angle)).toFixed(1) + ',' + (cy + r * ring * Math.sin(angle)).toFixed(1));
    }
    svg += '<polygon points="' + pts.join(' ') + '" fill="none" stroke="#1a1a2e" stroke-width="0.5"/>';
  });

  // Axis lines
  for (var i = 0; i < n; i++) {
    var angle = (Math.PI * 2 * i / n) - Math.PI / 2;
    var ex = cx + r * Math.cos(angle);
    var ey = cy + r * Math.sin(angle);
    svg += '<line x1="' + cx + '" y1="' + cy + '" x2="' + ex.toFixed(1) + '" y2="' + ey.toFixed(1) + '" stroke="#1a1a2e" stroke-width="0.5"/>';
  }

  // Data polygon
  var dataPts = [];
  archetypes.forEach(function(a, i) {
    var angle = (Math.PI * 2 * i / n) - Math.PI / 2;
    var val = Math.min(1.0, a.score);
    dataPts.push((cx + r * val * Math.cos(angle)).toFixed(1) + ',' + (cy + r * val * Math.sin(angle)).toFixed(1));
  });
  svg += '<polygon points="' + dataPts.join(' ') + '" fill="rgba(192,0,255,0.15)" stroke="#c0f" stroke-width="1.5"/>';

  // Data dots + labels
  archetypes.forEach(function(a, i) {
    var angle = (Math.PI * 2 * i / n) - Math.PI / 2;
    var val = Math.min(1.0, a.score);
    var dx = cx + r * val * Math.cos(angle);
    var dy = cy + r * val * Math.sin(angle);
    svg += '<circle cx="' + dx.toFixed(1) + '" cy="' + dy.toFixed(1) + '" r="2.5" fill="#c0f"/>';
    var lx = cx + (r + 12) * Math.cos(angle);
    var ly = cy + (r + 12) * Math.sin(angle);
    var anchor = Math.cos(angle) < -0.1 ? 'end' : Math.cos(angle) > 0.1 ? 'start' : 'middle';
    var shortName = a.name.replace('-Oriented', '').replace('-Adaptive', '').replace('-Conscious', '');
    svg += '<text x="' + lx.toFixed(1) + '" y="' + (ly + 3).toFixed(1) + '" text-anchor="' + anchor + '" font-size="6" fill="#6a6a80">' + shortName + '</text>';
  });

  svg += '</svg>';
  return svg;
}


// ═══════════════════════════════════════════════════════════════════════════
// 8. Trust Renderer
// ═══════════════════════════════════════════════════════════════════════════

var _lastBenchmark = null;

window._exportBenchmarkJSON = function() {
  if (!_lastBenchmark) return;
  var blob = new Blob([JSON.stringify(_lastBenchmark, null, 2)], { type: 'application/json' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'oracle_benchmark.json';
  a.click();
};

window._exportBenchmarkMarkdown = function() {
  if (!_lastBenchmark) return;
  var bm = _lastBenchmark;
  var lines = ['# Oracle Benchmark Report', '', '**Score:** ' + fmtNum(bm.composite_score, 1) + ' / 100'];
  lines.push('**Rank:** ' + (bm.benchmark_rank_display || bm.benchmark_rank || bm.rank || '--'));
  lines.push('**Seal:** ' + (bm.seal || 'none'));
  lines.push('**Credibility:** ' + (bm.credibility_status === 'pass' ? 'PASS' : (bm.credibility_status || bm.credibility || '--')));
  lines.push('');
  var domains = Object.entries(bm.domains || {});
  if (domains.length) {
    lines.push('## Domains');
    domains.forEach(function(entry) {
      var key = entry[0];
      var d = entry[1] || {};
      var name = key.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
      lines.push('- **' + name + '**: ' + fmtNum(d.raw != null ? d.raw : (d.score || 0), 1) + ' / ' + (d.max || d.max_score || '--'));
    });
  }
  var blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
  var a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'oracle_benchmark.md';
  a.click();
};

window._mtToggle = function(catId) {
  var el = document.getElementById('mt-' + catId);
  if (el) el.classList.toggle('collapsed');
};

function renderTrust(snap) {
  var root = el('trust-root');
  if (!root) return;

  var html = '';

  html += _renderTabWayfinding('trust');
  html += _renderTrustOverview(snap);
  html += _renderStageLadder(snap);
  html += _renderArchLinkCard();

  html += '<div class="j-layer-section-title">Epistemic Layers (Canonical Order)</div>';
  html += '<div class="j-panel-grid">';
  html += _renderCapabilityGateLayer(snap);
  html += _renderExternalTrustPanel('_renderLedgerPanel', snap, 'L1 Attribution Ledger', 'l1-attribution-ledger');
  html += _renderProvenanceLayer(snap);
  html += _renderIdentityBoundary(snap);
  html += _renderIdentityPersistenceLayer(snap);
  html += _renderSceneContinuityLayer(snap);
  html += _renderDelayedOutcomeLayer(snap);
  html += _renderContradictionEngine(snap);
  html += _renderTruthCalibration(snap);
  html += _renderBeliefGraph(snap);
  html += _renderQuarantine(snap);
  html += _renderExternalTrustPanel('_renderReflectiveAuditPanel', snap, 'L9 Reflective Audit', 'l9-reflective-audit');
  html += _renderExternalTrustPanel('_renderSoulIntegrityPanel', snap, 'L10 Soul Integrity', 'l10-soul-integrity');
  html += _renderIntentionRegistry(snap);
  html += _renderIntentionResolver(snap);
  html += '</div>';

  html += '<div class="j-layer-section-title">Support Surfaces (Trace / Validation / Benchmark)</div>';
  html += '<div class="j-panel-grid">';
  html += _renderOracleBenchmark(snap);
  html += _renderPVL(snap);
  html += _renderMaturityGates(snap);
  html += _renderValidationPack(snap);
  html += _renderLanguageGovernance(snap);
  html += _renderWindowDeltas(snap);
  html += _renderStabilityChartPanel(snap);
  html += _renderScoreboard(snap);
  html += _renderGoldenCommands(snap);
  html += _renderExternalTrustPanel('_renderTraceExplorerPanel', snap, 'Trace Explorer', 'trust-trace-explorer');
  html += _renderExternalTrustPanel('_renderReconstructabilityPanel', snap, 'Reconstructability', 'trust-reconstructability');
  html += '</div>';

  root.innerHTML = html;

  // Draw charts after DOM is ready
  requestAnimationFrame(function() {
    var ev = snap.eval || {};
    if (ev.stability) {
      var stabCanvas = document.getElementById('trust-stability-chart');
      if (stabCanvas && window.renderStabilityChart) {
        window.renderStabilityChart('trust-stability-chart', ev.stability);
      }
    }
    if (ev.scoreboard && (ev.scoreboard.bars || []).length) {
      var sbCanvas = document.getElementById('trust-scoreboard-chart');
      if (sbCanvas && window.drawBarChart) {
        var sbBars = ev.scoreboard.bars || [];
        var vals = sbBars.map(function(b) { return { label: b.category || b.name || '--', value: b.score != null ? b.score : 0 }; });
        window.drawBarChart('trust-scoreboard-chart', vals, '#0cf');
      }
    }
  });
}

function _renderExternalTrustPanel(fnName, snap, fallbackTitle, panelId) {
  var fn = window[fnName];
  if (typeof fn === 'function') {
    return fn(snap);
  }
  var html = _panelOpen(fallbackTitle, { panelId: panelId, ownerTab: 'trust' });
  html += _emptyMsg('Panel unavailable in current renderer build.');
  html += _panelClose();
  return html;
}

function _renderCapabilityGateLayer(snap) {
  var gate = snap.capability_gate || {};
  if (!Object.keys(gate).length) {
    return _panelOpen('L0 Capability Gate', { panelId: 'l0-capability-gate', ownerTab: 'trust' }) +
      _emptyMsg('No capability gate telemetry yet.') + _panelClose();
  }

  var html = _panelOpen('L0 Capability Gate', {
    panelId: 'l0-capability-gate',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.72rem;color:#0f9;">' + (gate.claims_passed || 0) + ' passed</span>'
  });
  html += '<div class="section-grid j-grid-4" style="margin-bottom:8px;">' +
    _statCard('Passed', gate.claims_passed || 0, '#0f9') +
    _statCard('Blocked', gate.claims_blocked || 0, (gate.claims_blocked || 0) > 0 ? '#f44' : '#0f9') +
    _statCard('Conversational', gate.claims_conversational || 0, '#0cf') +
    _statCard('Honesty Failures', gate.honesty_failures || 0, (gate.honesty_failures || 0) > 0 ? '#f44' : '#6a6a80') +
    '</div>';
  html += '<div style="display:flex;gap:6px;flex-wrap:wrap;font-size:0.56rem;color:#6a6a80;">' +
    '<span>affect rewrites: ' + (gate.affect_rewrites || 0) + '</span>' +
    '<span>self-state rewrites: ' + (gate.self_state_rewrites || 0) + '</span>' +
    '<span>learning rewrites: ' + (gate.learning_rewrites || 0) + '</span>' +
    '<span>auto jobs: ' + (gate.jobs_auto_created || 0) + '</span>' +
    '</div>';
  html += _panelClose();
  return html;
}

function _renderProvenanceLayer(snap) {
  var mem = snap.memory || {};
  var byProv = mem.by_provenance || {};
  var val = ((snap.eval || {}).validation_pack || {});
  var hasProv = Object.keys(byProv).length > 0;
  var hasVal = (val.checks_total || 0) > 0 || (val.checks || []).length > 0;
  if (!hasProv && !hasVal) {
    return _panelOpen('L2 Provenance & Validation', { panelId: 'l2-provenance', ownerTab: 'trust' }) +
      _emptyMsg('No provenance or validation data yet.') + _panelClose();
  }

  var html = _panelOpen('L2 Provenance & Validation', {
    panelId: 'l2-provenance',
    ownerTab: 'trust'
  });
  html += '<div class="section-grid j-grid-4" style="margin-bottom:8px;">' +
    _statCard('Observed', byProv.observed || 0, '#0f9') +
    _statCard('User Claim', byProv.user_claim || 0, '#0cf') +
    _statCard('External', byProv.external_source || 0, '#c0f') +
    _statCard('Model Inference', byProv.model_inference || 0, '#f90') +
    '</div>';
  if (hasVal) {
    html += '<div class="section-grid j-grid-4" style="margin-bottom:6px;">' +
      _statCard('Validation Current', (val.checks_passing || 0) + '/' + (val.checks_total || 0), '#0f9') +
      _statCard('Validation Ever', val.checks_ever_met || 0, '#0cf') +
      _statCard('Regressed', val.checks_regressed || 0, (val.checks_regressed || 0) > 0 ? '#ff0' : '#6a6a80') +
      _statCard('Critical', (val.critical_passing || 0) + '/' + (val.critical_total || 0), (val.critical_passing || 0) === (val.critical_total || 0) ? '#0f9' : '#f44') +
      '</div>';
  }
  html += _panelClose();
  return html;
}

function _renderIdentityPersistenceLayer(snap) {
  var identity = snap.identity || {};
  var trustState = identity.voice_trust_state || 'unknown';
  if (!Object.keys(identity).length) {
    return _panelOpen('L3A Identity Persistence', { panelId: 'l3a-identity-persistence', ownerTab: 'trust' }) +
      _emptyMsg('No identity persistence data yet.') + _panelClose();
  }
  var trustColors = { trusted: '#0f9', tentative: '#ff0', degraded: '#f90', conflicted: '#f44', unknown: '#6a6a80' };
  var trustColor = trustColors[trustState] || '#6a6a80';
  var html = _panelOpen('L3A Identity Persistence', {
    panelId: 'l3a-identity-persistence',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.72rem;color:' + trustColor + ';">' + esc(trustState) + '</span>'
  });
  html += '<div class="section-grid j-grid-4" style="margin-bottom:8px;">' +
    _statCard('Identity', identity.identity || 'unknown', identity.is_known ? '#0f9' : '#6a6a80') +
    _statCard('Confidence', identity.confidence != null ? fmtPct(identity.confidence, 0) : '--', '#0cf') +
    _statCard('Persistence', identity.persisted ? 'active' : 'idle', identity.persisted ? '#0f9' : '#6a6a80') +
    _statCard('Flip Count', identity.flip_count || 0, (identity.flip_count || 0) > 2 ? '#ff0' : '#0f9') +
    '</div>';
  html += '<div style="font-size:0.56rem;color:#6a6a80;">' +
    'basis: ' + esc(identity.resolution_basis || '--') +
    ' \u00b7 trust reason: ' + esc(identity.trust_reason || '--') +
    (identity.persist_remaining_s ? ' \u00b7 persist ttl: ' + fmtNum(identity.persist_remaining_s, 0) + 's' : '') +
    '</div>';
  html += _panelClose();
  return html;
}

function _renderSceneContinuityLayer(snap) {
  var scene = snap.scene || {};
  var entities = Array.isArray(scene.entities) ? scene.entities : [];
  var changes = scene.recent_changes || scene.recent_events || [];
  if (!Object.keys(scene).length && !entities.length && !changes.length) {
    return _panelOpen('L3B Scene Continuity', { panelId: 'l3b-scene-continuity', ownerTab: 'trust' }) +
      _emptyMsg('No scene continuity data yet.') + _panelClose();
  }

  var entityCount = scene.entity_count != null ? scene.entity_count : entities.length;
  var html = _panelOpen('L3B Scene Continuity', {
    panelId: 'l3b-scene-continuity',
    ownerTab: 'trust'
  });
  html += '<div class="section-grid j-grid-4" style="margin-bottom:8px;">' +
    _statCard('Entities', entityCount || 0, '#0cf') +
    _statCard('Visible Persons', scene.visible_persons || 0, '#0f9') +
    _statCard('Display Surfaces', (scene.display_surfaces || []).length || 0, '#c0f') +
    _statCard('Recent Changes', changes.length || 0, '#ff0') +
    '</div>';
  if (changes.length) {
    html += '<div style="max-height:120px;overflow-y:auto;">';
    changes.slice(0, 5).forEach(function(ch) {
      var text = typeof ch === 'object' ? (ch.description || ch.event || ch.type || '') : String(ch);
      html += '<div style="font-size:0.55rem;color:#8a8aa0;padding:1px 0;border-bottom:1px solid #1a1a2e;">' +
        esc(text.substring(0, 110)) + '</div>';
    });
    html += '</div>';
  }
  html += _panelClose();
  return html;
}

function _renderDelayedOutcomeLayer(snap) {
  var ledger = snap.ledger || {};
  var sched = ledger.outcome_scheduler || {};
  var wins = (((snap.eval || {}).scorecards || {}).windows || {});
  var windowKeys = ['15m', '1h', '6h', '24h'];
  var available = windowKeys.filter(function(k) { return wins[k] && wins[k].available; }).length;
  var hasSched = Object.keys(sched).length > 0;
  if (!hasSched && available === 0) {
    return _panelOpen('L4 Delayed Outcomes', { panelId: 'l4-delayed-outcomes', ownerTab: 'trust' }) +
      _emptyMsg('No delayed-outcome telemetry yet.') + _panelClose();
  }
  var html = _panelOpen('L4 Delayed Outcomes', {
    panelId: 'l4-delayed-outcomes',
    ownerTab: 'trust'
  });
  html += '<div class="section-grid j-grid-4" style="margin-bottom:8px;">' +
    _statCard('Pending', sched.pending || 0, (sched.pending || 0) > 0 ? '#ff0' : '#6a6a80') +
    _statCard('Resolved', sched.resolved || 0, '#0f9') +
    _statCard('Inconclusive', sched.inconclusive || 0, '#f90') +
    _statCard('Windows', available + '/4', available > 0 ? '#0cf' : '#6a6a80') +
    '</div>';
  html += '<div style="display:flex;gap:6px;flex-wrap:wrap;font-size:0.56rem;color:#6a6a80;">';
  windowKeys.forEach(function(key) {
    var win = wins[key] || {};
    if (!win.available) return;
    var soulDelta = ((win.deltas || {}).soul_integrity || 0);
    var color = soulDelta > 0 ? '#0f9' : soulDelta < 0 ? '#f44' : '#6a6a80';
    html += '<span style="color:' + color + ';">' + key + ': ' + (soulDelta >= 0 ? '+' : '') + fmtNum(soulDelta * 100, 1) + '%</span>';
  });
  html += '</div>';
  html += _panelClose();
  return html;
}

function _renderMovedPanelSummary(title, ownerPanelId, detail, summaryPanelId) {
  var meta = _panelOwnershipMeta(ownerPanelId) || {};
  var ownerTab = meta.ownerTab || 'trust';
  var html = _panelOpen(title, {
    panelId: summaryPanelId || (ownerPanelId + '-summary'),
    ownerTab: _activeTab
  });
  html += '<div style="font-size:0.62rem;color:#8a8aa0;line-height:1.45;margin-bottom:8px;">' + esc(detail || '') + '</div>';
  html += '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">' +
    '<button class="j-btn-sm" onclick="window.openOwnedPanel && window.openOwnedPanel(\'' + esc(ownerPanelId) + '\')">Open in ' + esc(ownerTab) + '</button>' +
    '<span style="font-size:0.56rem;color:#6a6a80;">owner: ' + esc(ownerTab) + '</span>' +
    '</div>';
  html += _panelClose();
  return html;
}

function _renderTrustOverview(snap) {
  var ts = snap.trust_state || {};
  var si = snap.soul_integrity || {};
  var tc = snap.truth_calibration || {};
  var ev = snap.eval || {};
  var pvl = ev.pvl || {};
  var bm = ev.oracle_benchmark || {};

  var html = _panelOpen('Trust Overview', {
    panelId: 'trust-overview',
    ownerTab: 'trust',
    badge: _statusBadge(ts.state || 'unknown')
  });

  var trustColor = { grounded: '#0f9', provisional: '#0cf', conflicted: '#ff0', degraded: '#f44', unknown: '#6a6a80' };
  var tc_ = trustColor[ts.state] || '#6a6a80';
  html += '<div style="text-align:center;padding:12px 0;">' +
    '<div style="font-size:1.6rem;font-weight:700;color:' + tc_ + ';">' + esc(ts.label || 'Unknown') + '</div>';
  var reasons = ts.reasons || [];
  if (reasons.length) {
    html += '<div style="font-size:0.72rem;color:#6a6a80;margin-top:4px;">' + reasons.map(function(r) { return esc(r); }).join(' \u2022 ') + '</div>';
  }
  html += '</div>';

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);">' +
    _statCard('Soul Integrity', si.current_index != null ? (si.current_index * 100).toFixed(0) + '%' : '--', si.current_index > 0.7 ? '#0f9' : '#ff0') +
    _statCard('Truth Score', tc.truth_score != null ? (tc.truth_score * 100).toFixed(0) + '%' : '--', tc.truth_score >= 0.6 ? '#0f9' : '#ff0') +
    _statCard('PVL Coverage', pvl.coverage_pct != null ? pvl.coverage_pct.toFixed(0) + '%' : '--', '#0cf') +
    _statCard('Oracle Score', bm.composite_score != null ? fmtNum(bm.composite_score, 1) : '--', bm.composite_score >= 80 ? '#0f9' : '#ff0') +
    '</div>';

  html += _panelClose();
  return html;
}

// ═══════════════════════════════════════════════════════════════════════════
// Stage Ladder — single-panel answer to "where is JARVIS on every promotion
// gate right now?"  Pure read from existing snapshot fields; does not emit,
// mutate, or train anything.  If a field is missing, the row says "--" and
// stays that way.  Never invents a gate position.
// ═══════════════════════════════════════════════════════════════════════════

function _sl_bandColor(band) {
  return ({
    green: '#0f9', yellow: '#ff0', orange: '#f90', red: '#f44',
  })[band] || '#6a6a80';
}

function _sl_promotedBadge(ok, label) {
  var color = ok ? '#0f9' : '#c0f';
  var icon = ok ? '\u2713' : '\u29B8';  // check or ring
  return '<span style="display:inline-flex;align-items:center;gap:4px;padding:1px 8px;border-radius:3px;font-size:0.66rem;font-weight:600;color:' + color + ';background:' + color + '14;border:1px solid ' + color + '33;">' + icon + ' ' + esc(label || '--') + '</span>';
}

function _sl_row(name, status, progress, next, hint) {
  return '<div class="j-sl-row">' +
    '<div class="j-sl-name">' + esc(name) + '</div>' +
    '<div class="j-sl-status">' + status + '</div>' +
    '<div class="j-sl-progress">' + progress + '</div>' +
    '<div class="j-sl-next">' + (next || '<span style="color:#4a4a60;">\u2014</span>') + '</div>' +
    (hint ? '<div class="j-sl-hint">' + esc(hint) + '</div>' : '') +
    '</div>';
}

function _renderStageLadder(snap) {
  var gest = snap.gestation || {};
  var wm = (snap.world_model || {}).promotion || {};
  var sim = (snap.world_model || {}).simulator_promotion || {};
  var pol = snap.policy || {};
  var ap = (snap.autonomy || {}).promotion || {};
  var ig = (snap.intentions || {}).graduation || {};
  var bm = ((snap.eval || {}).oracle_benchmark) || {};
  var si = snap.soul_integrity || {};
  var pvl = ((snap.eval || {}).pvl) || {};

  // Badge in panel header: Oracle seal + composite + evolution stage.
  var oracleBits = [];
  if (bm.seal) oracleBits.push('<span style="color:#0cf;font-weight:600;">' + esc(bm.seal) + '</span>');
  if (bm.composite_score != null) oracleBits.push('<span style="color:#6a6a80;">' + fmtNum(bm.composite_score, 1) + '/100</span>');
  if (bm.evolution_stage) oracleBits.push('<span style="color:#c0f;">' + esc(bm.evolution_stage) + '</span>');
  var badge = oracleBits.length ? '<span style="font-size:0.70rem;display:inline-flex;gap:8px;">' + oracleBits.join('<span style="color:#2a2a40;">\u00b7</span>') + '</span>' : '';

  var html = _panelOpen('Stage Ladder', {
    panelId: 'trust-stage-ladder',
    ownerTab: 'trust',
    badge: badge,
  });

  // --- ladder header row ---
  html += '<div class="j-sl-grid">';
  html += '<div class="j-sl-row j-sl-head">' +
    '<div class="j-sl-name">Subsystem</div>' +
    '<div class="j-sl-status">Status</div>' +
    '<div class="j-sl-progress">Progress</div>' +
    '<div class="j-sl-next">Next gate</div>' +
    '</div>';

  // --- Gestation ---
  (function() {
    var graduated = !!gest.graduated;
    var active = !!gest.active;
    var directives = (gest.directives_completed || 0) + '/' + (gest.directives_issued || 0);
    var research = gest.research_jobs_completed != null ? (' \u00b7 ' + gest.research_jobs_completed + ' research') : '';
    var status = graduated
      ? _sl_promotedBadge(true, 'graduated')
      : (active ? _sl_promotedBadge(false, 'phase ' + (gest.phase != null ? gest.phase : '?')) : _sl_promotedBadge(false, 'idle'));
    var progress = '<span>' + esc(directives) + esc(research) + '</span>';
    var next = graduated ? '' : (gest.phase_name ? esc(gest.phase_name) : 'in progress');
    html += _sl_row('Gestation', status, progress, next);
  })();

  // --- World Model ---
  (function() {
    var lvl = wm.level != null ? wm.level : '--';
    var lname = wm.level_name || 'unknown';
    var active = lvl === 2;
    var status = _sl_promotedBadge(active, 'L' + lvl + ' ' + lname);
    var acc = wm.rolling_accuracy != null ? fmtPct(wm.rolling_accuracy, 1) : '--';
    var validated = wm.total_validated || 0;
    var progress = validated + ' validated \u00b7 ' + acc + ' acc';
    var next = active ? '' : (lvl === 0 ? 'advisory' : 'active');
    html += _sl_row('World Model', status, progress, next);
  })();

  // --- Mental Simulator ---
  (function() {
    var lvl = sim.level != null ? sim.level : '--';
    var lname = sim.level_name || 'unknown';
    var active = lvl >= 1;
    var status = _sl_promotedBadge(active, 'L' + lvl + ' ' + lname);
    var acc = sim.rolling_accuracy != null ? fmtPct(sim.rolling_accuracy, 1) : '--';
    var validated = sim.total_validated || 0;
    var hours = sim.hours_in_shadow != null ? fmtNum(sim.hours_in_shadow, 1) : '0';
    var progress = validated + ' sims \u00b7 ' + acc + ' acc \u00b7 ' + hours + 'h shadow';
    // Known gate: >= 100 validated, >= 48h shadow, >= 70% accuracy for advisory.
    var needHours = Math.max(0, 48 - (sim.hours_in_shadow || 0));
    var needSims = Math.max(0, 100 - validated);
    var parts = [];
    if (!active) {
      if (needSims > 0) parts.push(needSims + ' sims');
      if (needHours > 0) parts.push(fmtNum(needHours, 1) + 'h');
      if (!parts.length) parts.push('accuracy gate');
    }
    var next = active ? '' : ('advisory: need ' + parts.join(' + '));
    html += _sl_row('Mental Simulator', status, progress, next);
  })();

  // --- Policy NN ---
  (function() {
    var mode = pol.mode || 'unknown';
    var eligible = !!pol.eligible_for_control;
    var status = _sl_promotedBadge(mode === 'live', mode);
    var ab = pol.shadow_ab_total || 0;
    var win = pol.nn_win_rate != null ? fmtPct(pol.nn_win_rate, 1) : '--';
    var progress = ab + ' decisions \u00b7 ' + win + ' win';
    // Promotion gate: >= 100 shadow decisions at > 55% decisive win rate.
    var needDecisions = Math.max(0, 100 - ab);
    var next;
    if (mode === 'live') {
      next = '';
    } else if (eligible) {
      next = 'eligible';
    } else {
      next = needDecisions > 0 ? ('control: ' + needDecisions + ' more decisions') : 'win-rate gate';
    }
    html += _sl_row('Policy NN', status, progress, next);
  })();

  // --- Autonomy ---
  (function() {
    var lvl = ap.current_level != null ? ap.current_level : ((snap.autonomy || {}).autonomy_level);
    var name = (snap.autonomy || {}).autonomy_level_name || ('L' + lvl);
    var active = lvl >= 2;
    var status = _sl_promotedBadge(active, 'L' + (lvl != null ? lvl : '?') + ' ' + name);
    var wins = ap.wins != null ? ap.wins : 0;
    var total = ap.total_outcomes != null ? ap.total_outcomes : 0;
    var rate = ap.win_rate != null ? fmtPct(ap.win_rate, 1) : '--';
    var progress = wins + '/' + total + ' wins \u00b7 ' + rate;
    var next = ap.l3_reason ? esc(ap.l3_reason.replace(/^Need /, 'L3: need ')) : '';
    if (ap.eligible_for_l3) next = 'L3 eligible';

    // Phase 6.5: three-axis L3 badges sourced from the live snapshot
    // cache. current_ok is live-only; prior_attested_ok comes from the
    // attestation ledger; activation_ok is live_level >= 3.
    // verified vs archived_missing carry distinct colors;
    // hash_mismatch / hash_unverifiable are rejected (do not count
    // toward prior_attested_ok) and rendered elsewhere in the
    // self-improve UI.
    var l3 = (snap.autonomy || {}).l3 || {};
    if (l3.available) {
      var currentOk = !!l3.current_ok;
      var priorOk = !!l3.prior_attested_ok;
      var activationOk = !!l3.activation_ok;
      var strength = l3.attestation_strength || 'none';

      var _chip = function(label, ok, color, extra) {
        var bg = 'rgba(' + color + ',0.13)';
        var bd = 'rgba(' + color + ',0.45)';
        var fg = 'rgb(' + color + ')';
        var t = extra ? (' title="' + esc(extra) + '"') : '';
        return '<span' + t + ' style="display:inline-block;padding:1px 6px;border-radius:3px;'
          + 'background:' + bg + ';color:' + fg + ';border:1px solid ' + bd + ';">'
          + esc(label) + '</span>';
      };
      // Color palette: green for true/verified, amber for
      // archived_missing, red for false on live gates, muted for
      // false on non-live gates.
      var greenRgb = '0,255,153';
      var redRgb   = '255,68,68';
      var amberRgb = '255,153,0';
      var mutedRgb = '106,106,128';

      var axes = [];
      axes.push(_chip(
        'current_ok: ' + (currentOk ? 'true' : 'false'),
        currentOk,
        currentOk ? greenRgb : redRgb
      ));
      var priorRgb = priorOk
        ? (strength === 'verified' ? greenRgb : amberRgb)
        : mutedRgb;
      axes.push(_chip(
        'prior_attested_ok: ' + (priorOk ? 'true' : 'false')
          + (priorOk ? (' \u00b7 ' + strength) : ''),
        priorOk,
        priorRgb,
        'attestation_strength=' + strength
      ));
      axes.push(_chip(
        'activation_ok: ' + (activationOk ? 'true' : 'false'),
        activationOk,
        activationOk ? greenRgb : mutedRgb
      ));

      var axesHtml = '<div style="margin-top:4px;display:flex;gap:6px;flex-wrap:wrap;font-size:0.68rem;">'
        + axes.join('') + '</div>';
      next = next ? (next + axesHtml) : axesHtml;
    }
    html += _sl_row('Autonomy', status, progress, next);
  })();

  // --- Intentions Stage 0/1 ---
  (function() {
    var stage = ig.stage != null ? ig.stage : 0;
    var ready = !!ig.stage1_ready;
    var gates = ig.gates || [];
    var passed = gates.filter(function(g) { return g && g.status === 'pass'; }).length;
    var status = _sl_promotedBadge(ready, 'Stage ' + stage);
    var progress = passed + '/' + gates.length + ' gates';
    var next = ready ? '' : 'Stage ' + (ig.next_stage != null ? ig.next_stage : (stage + 1));
    html += _sl_row('Intentions', status, progress, next);
  })();

  html += '</div>';  // /j-sl-grid

  // --- supporting strips: Oracle domains, Soul Integrity, PVL ---
  html += '<div class="j-sl-supports">';

  // Oracle domain band strip
  (function() {
    var doms = bm.domains || {};
    if (!doms || !Object.keys(doms).length) return;
    html += '<div class="j-sl-sub">';
    html += '<div class="j-sl-sub-label">Oracle domains</div>';
    html += '<div class="j-sl-bands">';
    var order = ['restart_integrity', 'epistemic_integrity', 'memory_continuity', 'operational_maturity', 'autonomy_attribution', 'world_model_coherence', 'learning_adaptation'];
    order.forEach(function(k) {
      var d = doms[k];
      if (!d) return;
      var color = _sl_bandColor(d.band);
      var raw = d.raw != null ? fmtNum(d.raw, 1) : '--';
      var mx = d.max != null ? d.max : '--';
      var label = k.replace(/_/g, ' ');
      html += '<div class="j-sl-band" title="' + esc(label) + ': ' + raw + '/' + mx + ' (' + (d.band || '?') + ')" style="border-color:' + color + '55;">' +
        '<div style="background:' + color + '22;color:' + color + ';">' + esc(label) + '</div>' +
        '<div>' + raw + '<span style="color:#4a4a60;">/' + mx + '</span></div>' +
        '</div>';
    });
    html += '</div>';
    html += '</div>';
  })();

  // Soul Integrity weakest dimension
  (function() {
    if (si.current_index == null) return;
    var idx = si.current_index;
    var color = idx >= 0.85 ? '#0f9' : (idx >= 0.75 ? '#9f9' : (idx >= 0.50 ? '#ff0' : '#f44'));
    var weakest = si.weakest_dimension || '--';
    var weakScore = si.weakest_score != null ? fmtNum(si.weakest_score, 3) : '--';
    html += '<div class="j-sl-sub">';
    html += '<div class="j-sl-sub-label">Soul Integrity</div>';
    html += '<div class="j-sl-kv">' +
      '<span style="color:' + color + ';font-weight:600;">' + fmtNum(idx, 3) + '</span>' +
      '<span style="color:#6a6a80;"> \u00b7 weakest: </span>' +
      '<span style="color:#ff0;">' + esc(weakest) + '</span>' +
      '<span style="color:#6a6a80;"> @ ' + weakScore + '</span>' +
      '</div>';
    html += '</div>';
  })();

  // PVL coverage
  (function() {
    var groups = pvl.groups || [];
    if (!groups.length) return;
    var pass = 0, total = 0;
    groups.forEach(function(g) {
      pass += (g.passing || 0);
      total += (g.total || 0);
    });
    if (!total) return;
    var rate = pass / total;
    var color = rate >= 0.90 ? '#0f9' : (rate >= 0.75 ? '#9f9' : (rate >= 0.50 ? '#ff0' : '#f44'));
    html += '<div class="j-sl-sub">';
    html += '<div class="j-sl-sub-label">PVL coverage</div>';
    html += '<div class="j-sl-kv">' +
      '<span style="color:' + color + ';font-weight:600;">' + pass + '/' + total + '</span>' +
      '<span style="color:#6a6a80;"> (' + fmtPct(rate, 0) + ') \u00b7 ' + groups.length + ' groups</span>' +
      '</div>';
    html += '</div>';
  })();

  html += '</div>';  // /j-sl-supports

  html += _panelClose();
  return html;
}


function _renderArchLinkCard() {
  var html = _panelOpen('System Architecture', {
    panelId: 'trust-architecture-link',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.62rem;color:#0cf;">reference</span>'
  });
  html += '<div style="text-align:center;padding:16px 0;">' +
    '<p style="color:#6a6a80;font-size:0.82rem;margin-bottom:12px;">Interactive architecture diagram, data flow docs, maturity gates, and epistemic stack reference.</p>' +
    '<a href="/docs" style="display:inline-block;padding:8px 20px;background:rgba(0,204,255,0.08);border:1px solid rgba(0,204,255,0.25);border-radius:4px;color:#0cf;text-decoration:none;font-size:0.82rem;font-weight:600;transition:background 0.2s;" ' +
    'onmouseover="this.style.background=\'rgba(0,204,255,0.15)\'" onmouseout="this.style.background=\'rgba(0,204,255,0.08)\'">' +
    '\u{1F4D6} Open System Reference</a>' +
    '</div>';
  html += _panelClose();
  return html;
}

function _renderSystemArchitectureDiagram(snap) {
  var auto = snap.autonomy || {};
  var frict = snap.friction || {};
  var src = snap.source_usefulness || snap.source_ledger || {};
  var interv = snap.interventions || {};
  var hemi = snap.hemispheres || {};
  var wm = snap.world_model || {};

  var html = _panelOpen('System Architecture — Data Flow Map', {
    panelId: 'trust-architecture-diagram',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.62rem;color:#8a8aa0;">interactive</span>'
  });

  html += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:10px;">' +
    'How the major loops connect. Hover a node for details. Live counters from the running system.' +
    '</div>';

  var svgW = 900, svgH = 620;
  html += '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">';
  html += '<svg id="arch-svg" viewBox="0 0 ' + svgW + ' ' + svgH + '" ' +
    'style="width:100%;max-width:900px;height:auto;min-height:320px;font-family:monospace;" ' +
    'xmlns="http://www.w3.org/2000/svg">';

  html += '<defs>' +
    '<marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">' +
    '<path d="M0,0 L8,3 L0,6 Z" fill="#4a4a6a"/>' +
    '</marker>' +
    '<marker id="ah-green" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">' +
    '<path d="M0,0 L8,3 L0,6 Z" fill="#0f9"/>' +
    '</marker>' +
    '<marker id="ah-cyan" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">' +
    '<path d="M0,0 L8,3 L0,6 Z" fill="#0cf"/>' +
    '</marker>' +
    '<marker id="ah-amber" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">' +
    '<path d="M0,0 L8,3 L0,6 Z" fill="#f90"/>' +
    '</marker>' +
    '</defs>';

  html += '<rect width="' + svgW + '" height="' + svgH + '" fill="#0e0e1a" rx="6"/>';

  // --- SECTION BACKGROUNDS ---
  html += '<rect x="20" y="15" width="860" height="95" rx="4" fill="#12122a" stroke="#2a2a4a" stroke-width="0.5"/>';
  html += '<text x="30" y="32" fill="#6a6a80" font-size="9" font-weight="700">PERCEPTION &amp; CONVERSATION</text>';

  html += '<rect x="20" y="120" width="420" height="180" rx="4" fill="#0f1a12" stroke="#1a3a2a" stroke-width="0.5"/>';
  html += '<text x="30" y="137" fill="#2a6a4a" font-size="9" font-weight="700">PHASE 5 — CONTINUOUS IMPROVEMENT</text>';

  html += '<rect x="460" y="120" width="420" height="180" rx="4" fill="#12121f" stroke="#2a2a4a" stroke-width="0.5"/>';
  html += '<text x="470" y="137" fill="#4a4a8a" font-size="9" font-weight="700">NEURAL NETWORKS &amp; DISTILLATION</text>';

  html += '<rect x="20" y="310" width="420" height="130" rx="4" fill="#1a120f" stroke="#3a2a1a" stroke-width="0.5"/>';
  html += '<text x="30" y="327" fill="#8a6a4a" font-size="9" font-weight="700">MEMORY &amp; KNOWLEDGE</text>';

  html += '<rect x="460" y="310" width="420" height="130" rx="4" fill="#1a0f1a" stroke="#3a1a3a" stroke-width="0.5"/>';
  html += '<text x="470" y="327" fill="#8a4a8a" font-size="9" font-weight="700">WORLD MODEL &amp; COGNITION</text>';

  html += '<rect x="20" y="450" width="860" height="155" rx="4" fill="#0f0f1a" stroke="#2a2a4a" stroke-width="0.5"/>';
  html += '<text x="30" y="467" fill="#4a4a8a" font-size="9" font-weight="700">EPISTEMIC INTEGRITY STACK (L0–L12 + L3A/L3B)</text>';

  // --- NODE HELPER ---
  var _nodeIdx = 0;
  function node(x, y, w, h, label, sublabel, color, live, tipText, navTarget) {
    var nid = 'arch-n-' + (_nodeIdx++);
    var r = '<g class="arch-node" id="' + nid + '" style="cursor:pointer;" ' +
      'data-tip="' + (tipText || '').replace(/"/g, '&quot;') + '" ' +
      'data-nav="' + (navTarget || '') + '" ' +
      'data-cx="' + (x + w/2) + '" data-cy="' + y + '">';
    r += '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="3" ' +
      'fill="#0e0e1a" stroke="' + color + '" stroke-width="1.2" class="arch-node-bg"/>';
    r += '<text x="' + (x + w/2) + '" y="' + (y + h/2 - (sublabel ? 3 : 0)) + '" ' +
      'fill="' + color + '" font-size="9" font-weight="600" text-anchor="middle" dominant-baseline="middle">' + label + '</text>';
    if (sublabel) {
      r += '<text x="' + (x + w/2) + '" y="' + (y + h/2 + 9) + '" ' +
        'fill="#6a6a80" font-size="7" text-anchor="middle" dominant-baseline="middle">' + sublabel + '</text>';
    }
    if (live) {
      r += '<text x="' + (x + w - 4) + '" y="' + (y + 10) + '" ' +
        'fill="#0f9" font-size="7" text-anchor="end" font-weight="700">' + live + '</text>';
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

  // --- PERCEPTION & CONVERSATION ROW ---
  var frictionCount = (frict.total_events != null) ? frict.total_events : (frict.count || '--');
  html += node(35, 42, 100, 55, 'Pi Senses', 'vision+audio', '#0cf', '',
    'Raspberry Pi 5 + Hailo-10H AI HAT+: camera vision, mic capture, cyberpunk particle display', '');
  html += node(160, 42, 110, 55, 'Wake/VAD/STT', 'speech → text', '#0cf', '',
    'openWakeWord detection, Silero VAD speech segmentation, faster-whisper GPU transcription', '');
  html += node(295, 42, 110, 55, 'Tool Router', '13 intent classes', '#0cf', '',
    'Keyword + regex intent dispatch: time, system, status, memory, vision, introspection, identity, skill, web, etc.', '');
  html += node(430, 42, 115, 55, 'Conversation', 'LLM + response', '#0cf', '',
    'Ollama LLM streaming response with cancel-token, TTS synthesis via Kokoro, audio sent to Pi', '');
  html += node(570, 42, 115, 55, 'Personal Intel', 'fact capture', '#f90', '',
    'Extracts interests, dislikes, facts, preferences from speech. Filtered by _is_unstable_personal_fact(). Corrections downweight recent junk.', '');
  html += node(710, 42, 135, 55, 'Friction Miner', 'corrections/rephrase', '#f44', '' + frictionCount,
    'Detects corrections, rephrases, annoyance in conversation. Feeds friction_rate metric trigger for Phase 5 research.', '');

  html += arrow(135, 70, 160, 70, '#0cf');
  html += arrow(270, 70, 295, 70, '#0cf');
  html += arrow(405, 70, 430, 70, '#0cf');
  html += arrow(545, 70, 570, 70, '#f90');
  html += arrow(545, 55, 710, 55, '#f44');

  // --- PHASE 5 LOOP ---
  var trigCount = (auto.completed_total || auto.session_completed || '--');
  var intervCount = (interv.proposed_count != null) ? (interv.proposed_count + '+' + (interv.shadow_active_count || 0) + '+' + (interv.completed_count || 0)) : '--';
  var srcCount = (src.total_sources != null) ? src.total_sources : '--';

  html += node(35, 150, 120, 50, 'Metric Triggers', '7 deficit dimensions', '#0f9', '' + trigCount,
    '7 deficit dimensions (retrieval, friction, autonomy, etc.). Spawns research intents with tool rotation.', '');
  html += node(175, 150, 120, 50, 'Research Intent', 'tool rotation', '#0f9', '',
    'Research question + tool hint (introspection/codebase/web/academic). Queued by triggers or drives.', '');
  html += node(315, 150, 110, 50, 'Knowledge', 'integration', '#0f9', '',
    'Pre-research knowledge check, conflict detection. Registers sources in ledger. Provenance-gated memory writes.', '');

  html += node(35, 220, 120, 50, 'Source Ledger', 'usefulness track', '#0f9', '' + srcCount,
    'Tracks per-source retrieval count and usefulness. log_outcome() is single authority for counting. Feeds back to scorer.', '');
  html += node(175, 220, 120, 50, 'Interventions', 'shadow → measure', '#0f9', '' + intervCount,
    'Shadow-evaluates proposed changes for 24h. Captures baseline_value at activation, computes measured_delta at check.', '');
  html += node(315, 220, 110, 50, 'Opp. Scorer', 'priority queue', '#0f9', '',
    'Score = Impact x Evidence x Confidence - Risk - Cost. Policy memory + source ledger usefulness adjust +/-0.3.', '');

  html += arrow(155, 175, 175, 175, '#0f9');
  html += arrow(295, 175, 315, 175, '#0f9');
  html += arrow(370, 200, 370, 220, '#0f9');
  html += arrow(315, 245, 295, 245, '#0f9');
  html += arrow(175, 245, 155, 245, '#0f9');
  html += arrow(95, 220, 95, 200, '#0f9');

  // Friction → Triggers
  html += curvedArrow(710, 97, 400, 130, 155, 150, '#f44');

  // Source Ledger → Scorer feedback
  html += arrow(155, 235, 175, 235, '#0f9');
  html += curvedArrow(95, 270, 200, 290, 325, 270, '#0f9');

  // --- NEURAL NETWORKS ---
  var distill = (snap.distillation || hemi.distillation || {});
  var specCount = (distill.active_count != null) ? distill.active_count : (distill.specialists ? Object.keys(distill.specialists).length : '');
  html += node(475, 150, 130, 50, 'Distillation', 'teacher → specialist', '#c0f', '' + specCount,
    'GPU models, routers, validators, and lifecycle outcomes teach Tier-1 specialists via fidelity-weighted loss.', '');
  html += node(625, 150, 125, 50, 'Tier-1 Specialists', 'perception+claims+skills', '#c0f', '',
    'Perception specialists plus language_style, plan_evaluator, diagnostic, code_quality, claim_classifier, dream_synthesis, skill_acquisition, and HRR encoder stub. Accuracy-gated.', '');
  html += node(770, 150, 95, 50, 'Tier-2', 'hemispheres', '#c0f', '',
    'Dynamic architecture NNs designed by NeuralArchitect. Focuses: MEMORY, MOOD, TRAITS, GENERAL + Matrix specialists.', '');

  html += node(475, 220, 130, 50, 'Z-Score Norm', 'audio_features*', '#c0f', '',
    'Normalizes mixed-scale audio features (spectral ~2000, RMS ~0.05, ECAPA ~+/-1) before training. Uses startswith match.', '');
  html += node(625, 220, 125, 50, 'Policy NN', 'shadow A/B', '#c0f', '',
    '20-dim state -> 8 behavioral knobs. Shadow A/B evaluation. Promotion requires >55% decisive win rate.', '');
  html += node(770, 220, 95, 50, 'Broadcast', '4-6 slots', '#c0f', '',
    'Global Broadcast Slots feed Tier-2 outputs into Policy NN state encoder (dims 16-19). Hysteresis gated.', '');

  html += arrow(605, 175, 625, 175, '#c0f');
  html += arrow(750, 175, 770, 175, '#c0f');
  html += arrow(605, 245, 625, 245, '#c0f');
  html += arrow(750, 245, 770, 245, '#c0f');
  html += arrow(540, 200, 540, 220, '#c0f');
  html += arrow(817, 200, 817, 220, '#c0f');

  // --- MEMORY & KNOWLEDGE ---
  html += node(35, 340, 120, 50, 'Memory Store', 'unified write path', '#f90', '',
    'engine.remember() -> quarantine soft-gate -> salience -> storage.add() -> index -> MEMORY_WRITE event.', 'memory');
  html += node(175, 340, 120, 50, 'Vector Search', 'embedding extract', '#f90', '',
    '_extract_embedding_text() pulls human-readable fields from structured payloads instead of Python repr().', 'memory');
  html += node(315, 340, 110, 50, 'Retrieval Log', 'single authority', '#f90', '',
    'log_outcome() is the ONLY call site for source_ledger.record_retrieval(). Prevents double-counting.', 'memory');

  html += node(35, 400, 120, 26, 'CueGate', 'memory safety', '#f90', '',
    'Single authority for memory access policy. Blocks observation writes during dream/sleep/reflective modes.', 'memory');
  html += node(175, 400, 120, 26, 'Cortex NNs', 'ranker + salience', '#f90', '',
    'MemoryRanker (MLP 12->32->16->1) + SalienceModel (MLP 11->24->12->3). Trained during dream cycles.', 'memory');
  html += node(315, 400, 110, 26, 'Consolidation', 'dream cycle', '#f90', '',
    'Dream cycle: associate, reinforce, decay, consolidation summaries. Uses begin/end_consolidation() window.', 'memory');

  html += arrow(155, 365, 175, 365, '#f90');
  html += arrow(295, 365, 315, 365, '#f90');

  // Personal Intel → Memory
  html += curvedArrow(627, 97, 300, 300, 95, 340, '#f90');

  // Retrieval → Source Ledger feedback
  html += curvedArrow(370, 340, 420, 290, 95, 270, '#0f9');

  // --- WORLD MODEL & COGNITION ---
  var wmLevel = (wm.level || wm.promotion_level || '--');
  html += node(475, 340, 130, 50, 'World Model', 'belief state', '#c8f', '' + wmLevel,
    'Fuses 9 subsystem snapshots into unified WorldState. Shadow -> advisory -> active promotion. Emits WORLD_MODEL_DELTA.', '');
  html += node(625, 340, 125, 50, 'Causal Engine', '18 rules', '#c8f', '',
    'Heuristic rules producing predicted state deltas. Priority-based conflict resolution. Feeds simulator.', '');
  html += node(770, 340, 95, 50, 'Simulator', 'what-if', '#c8f', '',
    'Hypothetical projections using CausalEngine rules. Read-only, max 3 steps. Shadow -> advisory promotion.', '');

  html += node(475, 400, 130, 26, 'WM Delta Events', 'emitted per tick', '#c8f', '',
    'event_bus.emit(WORLD_MODEL_DELTA) for each detected change. Consumed by EvalSidecar, CuriosityQuestionBuffer.', '');

  html += arrow(605, 365, 625, 365, '#c8f');
  html += arrow(750, 365, 770, 365, '#c8f');
  html += arrow(540, 390, 540, 400, '#c8f');

  // --- EPISTEMIC STACK (compact) ---
  var layers = [
    { label: 'L0', name: 'Cap Gate', color: '#0f9', tip: 'Capability Gate: scans all outgoing text with 7 enforcement layers and 15 claim patterns + action confabulation guard. Unverified claims rewritten.' },
    { label: 'L1', name: 'Attrib', color: '#0f9', tip: 'Attribution Ledger: append-only JSONL event truth, causal chains, outcome resolution, scope/blame.' },
    { label: 'L2', name: 'Provenance', color: '#0f9', tip: 'Every memory carries provenance (observed, user_claim, model_inference, etc.). Retrieval boost by source type.' },
    { label: 'L3', name: 'Identity', color: '#0f9', tip: 'Identity Boundary Engine: retrieval policy matrix prevents cross-identity memory leaks.' },
    { label: 'L3B', name: 'Scene', color: '#ff0', tip: 'Persistent Scene Model (shadow): entity tracking, region mapping, display classification. Physical world continuity.' },
    { label: 'L4', name: 'Outcomes', color: '#0f9', tip: 'Delayed Outcome Attribution: before/after + counterfactual baselines for causal credit assignment.' },
    { label: 'L5', name: 'Contradict', color: '#0f9', tip: 'Typed Contradiction Engine: 6 conflict classes, belief extraction, debt management, corpus scanning.' },
    { label: 'L6', name: 'Calibrate', color: '#0f9', tip: 'Truth Calibration: 8-domain scoring, Brier/ECE confidence tracking, drift detection, correction detection.' },
    { label: 'L7', name: 'Beliefs', color: '#0f9', tip: 'Belief Confidence Graph: weighted evidence edges, support/contradiction propagation, 6 sacred invariants.' },
    { label: 'L8', name: 'Quarantine', color: '#ff0', tip: 'Cognitive Quarantine (Active-Lite): 5 anomaly categories, EMA pressure, friction contract helpers.' },
    { label: 'L9', name: 'Audit', color: '#0f9', tip: 'Reflective Audit: 6-dimension scan (learning, identity, trust, autonomy, skills, memory), severity scoring.' },
    { label: 'L10', name: 'Soul', color: '#0f9', tip: 'Soul Integrity Index: 10-dimension weighted composite over all subsystem health metrics.' },
    { label: 'L11', name: 'Compact', color: '#0f9', tip: 'Epistemic Compaction: weight economy caps, subject-version collapsing, per-subject edge budgets.' }
  ];

  var lx = 35;
  for (var li = 0; li < layers.length; li++) {
    var lw = 62;
    var lyr = layers[li];
    var eid = 'arch-e-' + li;
    html += '<g class="arch-node" id="' + eid + '" style="cursor:pointer;" ' +
      'data-tip="' + lyr.tip.replace(/"/g, '&quot;') + '" data-nav="trust" ' +
      'data-cx="' + (lx + lw/2) + '" data-cy="480">';
    html += '<rect x="' + lx + '" y="' + 480 + '" width="' + lw + '" height="40" rx="2" ' +
      'fill="#0e0e1a" stroke="' + lyr.color + '" stroke-width="0.8" class="arch-node-bg"/>';
    html += '<text x="' + (lx + lw/2) + '" y="' + 494 + '" fill="' + lyr.color + '" ' +
      'font-size="8" font-weight="700" text-anchor="middle">' + lyr.label + '</text>';
    html += '<text x="' + (lx + lw/2) + '" y="' + 508 + '" fill="#6a6a80" ' +
      'font-size="7" text-anchor="middle">' + lyr.name + '</text>';
    html += '</g>';
    lx += lw + 4;
  }

  // Guard rail arrows from epistemic stack upward
  html += '<text x="450" y="560" fill="#4a4a6a" font-size="8" text-anchor="middle" font-style="italic">' +
    'Every outgoing response passes through L0 Capability Gate. Every memory write passes through CueGate.' +
    '</text>';
  html += '<text x="450" y="575" fill="#4a4a6a" font-size="8" text-anchor="middle" font-style="italic">' +
    'Beliefs are extracted (L5), calibrated (L6), graph-linked (L7), and audited (L9-L10).' +
    '</text>';
  html += '<text x="450" y="590" fill="#4a4a6a" font-size="8" text-anchor="middle" font-style="italic">' +
    'No subsystem can bypass epistemic integrity — all data flows are gated.' +
    '</text>';

  // --- TOOLTIP GROUP (rendered last so it's on top) ---
  html += '<g id="arch-tip-g" style="display:none;pointer-events:none;">' +
    '<rect id="arch-tip-bg" rx="4" fill="#1a1a2e" fill-opacity="0.95" stroke="#4a4a6a" stroke-width="0.5"/>' +
    '<text id="arch-tip-text" fill="#ddd" font-size="8" dominant-baseline="hanging"></text>' +
    '</g>';

  html += '</svg></div>';

  // Legend
  html += '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;font-size:0.58rem;">' +
    '<span style="color:#0cf;">\u25cf Perception</span>' +
    '<span style="color:#0f9;">\u25cf Phase 5 Loop</span>' +
    '<span style="color:#f44;">\u25cf Friction Detection</span>' +
    '<span style="color:#f90;">\u25cf Memory/Knowledge</span>' +
    '<span style="color:#c0f;">\u25cf Neural Networks</span>' +
    '<span style="color:#c8f;">\u25cf World Model</span>' +
    '<span style="color:#4a4a6a;">\u25cf Epistemic Stack</span>' +
    '</div>';

  html += _panelClose();
  return html;
}

function _wireArchDiagramEvents() {
  var svg = document.getElementById('arch-svg');
  if (!svg) return;
  var tipG = document.getElementById('arch-tip-g');
  var tipBg = document.getElementById('arch-tip-bg');
  var tipText = document.getElementById('arch-tip-text');
  if (!tipG || !tipBg || !tipText) return;

  var nodes = svg.querySelectorAll('.arch-node');
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
        if (test.length > 60 && line) {
          lines.push(line);
          line = words[i];
        } else {
          line = test;
        }
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
      var tspans = tipText.querySelectorAll('tspan');
      tspans.forEach(function(ts) { ts.setAttribute('x', tipX + padX); });

      tipG.style.display = '';

      var bgRect = g.querySelector('.arch-node-bg');
      if (bgRect) bgRect.setAttribute('fill', '#1a1a2e');
    });

    g.addEventListener('mouseleave', function() {
      tipG.style.display = 'none';
      var bgRect = g.querySelector('.arch-node-bg');
      if (bgRect) bgRect.setAttribute('fill', '#0e0e1a');
    });

    g.addEventListener('click', function() {
      var nav = g.getAttribute('data-nav');
      if (nav) {
        _archNavigate(nav);
      }
    });
  });
}

function _archNavigate(target) {
  var panelMap = {
    'l0-capability-gate': 'trust',
    'l1-attribution-ledger': 'trust',
    'l2-provenance': 'trust',
    'l3-identity-boundary': 'trust',
    'l3a-identity-persistence': 'trust',
    'l3b-scene-continuity': 'trust',
    'l4-delayed-outcomes': 'trust',
    'l5-contradiction': 'trust',
    'l6-truth-calibration': 'trust',
    'l7-belief-graph': 'trust',
    'l8-quarantine': 'trust',
    'l9-reflective-audit': 'trust',
    'l10-soul-integrity': 'trust',
    'memory': 'memory',
    'activity': 'activity',
    'learning': 'learning',
    'diagnostics': 'diagnostics',
    'training': 'training'
  };

  var tabName = panelMap[target] || target;
  if (['cockpit', 'trust', 'memory', 'activity', 'learning', 'training', 'diagnostics'].indexOf(tabName) !== -1) {
    if (_activeTab !== tabName) switchTab(tabName);
  }
}

function _renderOracleBenchmark(snap) {
  var ev = snap.eval || {};
  var bm = ev.oracle_benchmark || {};
  _lastBenchmark = bm;

  if (!bm.composite_score && bm.composite_score !== 0) {
    return _panelOpen('Oracle Benchmark', { panelId: 'trust-oracle-benchmark', ownerTab: 'trust' }) + _emptyMsg('Not computed yet') + _panelClose();
  }

  var html = _panelOpen('Oracle Benchmark', {
    panelId: 'trust-oracle-benchmark',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.75rem;font-weight:700;color:#0cf;">' + fmtNum(bm.composite_score, 1) + '/100</span>'
  });

  var bmRank = bm.benchmark_rank_display || bm.benchmark_rank || bm.rank || '--';
  var bmCred = bm.credibility_status === 'pass' ? 'PASS' : (bm.credibility_status || bm.credibility || '--');
  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:10px;">' +
    _statCard('Score', fmtNum(bm.composite_score, 1), bm.composite_score >= 80 ? '#0f9' : bm.composite_score >= 60 ? '#ff0' : '#f44') +
    _statCard('Rank', bmRank, '#c0f') +
    _statCard('Seal', bm.seal || 'none', bm.seal === 'Gold' ? '#ff0' : bm.seal === 'Silver' ? '#aaa' : '#c0f') +
    _statCard('Credibility', bmCred, bmCred === 'PASS' || bmCred === 'credible' ? '#0f9' : '#f44') +
    '</div>';

  // Domain cards — compact 2-column grid with expandable subcriteria
  var domainEntries = Object.entries(bm.domains || {});
  if (domainEntries.length) {
    html += '<div style="font-size:0.72rem;color:#6a6a80;margin-bottom:6px;">Domains</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:6px;">';
    domainEntries.forEach(function(entry) {
      var domainKey = entry[0];
      var d = entry[1] || {};
      var name = domainKey.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
      var score = d.raw != null ? d.raw : (d.score != null ? d.score : 0);
      var maxScore = d.max || d.max_score || 20;
      var bandColors = { green: '#0f9', yellow: '#ff0', red: '#f44' };
      var c = d.band ? (bandColors[d.band] || '#6a6a80') : (score / maxScore >= 0.8 ? '#0f9' : score / maxScore >= 0.5 ? '#ff0' : '#f44');
      var subs = d.subcriteria || d.sub_criteria || [];
      var domId = 'oracle-dom-' + domainKey.replace(/\W+/g, '');
      html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px 8px;">' +
        '<div style="display:flex;justify-content:space-between;font-size:0.68rem;margin-bottom:3px;">' +
        '<span style="font-weight:600;">' + esc(name) + '</span>' +
        '<span style="color:' + c + ';font-weight:700;">' + fmtNum(score, 1) + '/' + maxScore + '</span></div>' +
        _barFill(score, maxScore, c);
      if (subs.length) {
        html += '<div id="' + domId + '" style="margin-top:4px;padding-top:3px;border-top:1px solid #1a1a2e;">';
        subs.forEach(function(sub) {
          html += '<div style="display:flex;justify-content:space-between;font-size:0.55rem;color:#6a6a80;padding:0 0 1px;">' +
            '<span>' + esc(sub.label || sub.name || sub.id || '') + '</span>' +
            '<span>' + fmtNum(sub.score, 2) + '/' + (sub.max || '--') +
            (sub.value != null ? ' (' + esc(String(sub.value).substring(0, 25)) + ')' : '') +
            '</span></div>';
        });
        html += '</div>';
      }
      html += '</div>';
    });
    html += '</div>';
  }

  // Hard-fail checks — hard_fail_reasons is a string array of failure names
  var hardFailReasons = bm.hard_fail_reasons || [];
  var knownChecks = [
    'missing_restore_trust_fields', 'insufficient_runtime_sample', 'insufficient_event_count',
    'missing_epistemic_evidence', 'missing_contradiction_data', 'stage_requirements_not_met'
  ];
  var failSet = {};
  hardFailReasons.forEach(function(r) { failSet[r] = true; });
  var hasAnyCheck = hardFailReasons.length > 0 || Object.keys(bm.domains || {}).length > 0;
  if (hasAnyCheck) {
    html += '<div style="font-size:0.72rem;color:#6a6a80;margin-top:8px;margin-bottom:4px;">Hard-Fail Checks</div>' +
      '<table style="width:100%;font-size:0.65rem;border-collapse:collapse;">' +
      '<tr style="color:#6a6a80;"><th style="text-align:left;padding:2px 4px;">Check</th><th style="text-align:center;padding:2px 4px;">Status</th></tr>';
    knownChecks.forEach(function(checkName) {
      var failed = !!failSet[checkName];
      var c = failed ? '#f44' : '#0f9';
      var icon = failed ? '\u2717' : '\u2713';
      var label = checkName.replace(/_/g, ' ');
      html += '<tr><td style="padding:2px 4px;">' + esc(label) + '</td>' +
        '<td style="text-align:center;padding:2px 4px;color:' + c + ';">' + icon + '</td></tr>';
    });
    hardFailReasons.forEach(function(r) {
      if (knownChecks.indexOf(r) === -1) {
        html += '<tr><td style="padding:2px 4px;">' + esc(r.replace(/_/g, ' ')) + '</td>' +
          '<td style="text-align:center;padding:2px 4px;color:#f44;">\u2717</td></tr>';
      }
    });
    html += '</table>';
  }

  var evidence = bm.evidence || {};
  if (typeof evidence === 'object' && Object.keys(evidence).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:8px;margin-bottom:4px;">Evidence Provenance</div>';
    var provTypes = ['live_proven', 'test_proven', 'unexercised'];
    provTypes.forEach(function(pt) {
      var items = evidence[pt];
      var count = Array.isArray(items) ? items.length : (typeof items === 'number' ? items : 0);
      var c = pt === 'live_proven' ? '#0f9' : pt === 'test_proven' ? '#0cf' : '#6a6a80';
      html += '<span style="display:inline-block;margin-right:10px;font-size:0.6rem;">' +
        '<span style="color:' + c + ';">\u25cf</span> ' + pt.replace(/_/g, ' ') + ': <b>' + count + '</b></span>';
    });
  }

  // Strengths / Weaknesses
  var strengths = bm.strengths || [];
  var weaknesses = bm.weaknesses || [];
  if (strengths.length || weaknesses.length) {
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px;">';
    if (strengths.length) {
      html += '<div><div style="font-size:0.68rem;color:#0f9;margin-bottom:3px;">Strengths</div>';
      strengths.forEach(function(s) { html += '<div style="font-size:0.6rem;color:#aaa;">+ ' + esc(s) + '</div>'; });
      html += '</div>';
    }
    if (weaknesses.length) {
      html += '<div><div style="font-size:0.68rem;color:#f44;margin-bottom:3px;">Weaknesses</div>';
      weaknesses.forEach(function(w) { html += '<div style="font-size:0.6rem;color:#aaa;">- ' + esc(w) + '</div>'; });
      html += '</div>';
    }
    html += '</div>';
  }

  // Export buttons
  html += '<div style="margin-top:10px;display:flex;gap:8px;">' +
    '<button class="j-btn-sm" onclick="window._exportBenchmarkJSON()">Export JSON</button>' +
    '<button class="j-btn-sm" onclick="window._exportBenchmarkMarkdown()">Export Markdown</button>' +
    '</div>';

  html += _panelClose();
  return html;
}

function _renderPVL(snap) {
  var ev = snap.eval || {};
  var pvl = ev.pvl || {};

  if (!pvl.coverage_pct && pvl.coverage_pct !== 0) {
    return _panelOpen('Process Verification (PVL)', { panelId: 'trust-pvl', ownerTab: 'trust' }) + _emptyMsg('Not active') + _panelClose();
  }

  var html = _panelOpen('Process Verification (PVL)', {
    panelId: 'trust-pvl',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.72rem;color:#0cf;">' + fmtNum(pvl.coverage_pct, 1) + '%</span>'
  });

  html += _barFill(pvl.coverage_pct || 0, 100, '#0cf');
  var pvlPassing = pvl.passing_contracts || pvl.passing || 0;
  var pvlEverPassing = pvl.ever_passing_contracts || 0;
  var pvlFailing = pvl.failing_contracts || pvl.failing || 0;
  var pvlAwaiting = pvl.awaiting_contracts || pvl.awaiting || 0;
  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin:8px 0;">' +
    _statCard('Passing', pvlPassing, '#0f9') +
    _statCard('Ever Pass', pvlEverPassing, pvlEverPassing > pvlPassing ? '#0cf' : '#6a6a80') +
    _statCard('Failing', pvlFailing, pvlFailing > 0 ? '#f44' : '#6a6a80') +
    _statCard('Awaiting', pvlAwaiting, '#6a6a80') +
    '</div>';

  var groups = pvl.groups || pvl.contract_groups || [];
  if (groups.length) {
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:6px;align-items:start;">';
    groups.forEach(function(g) {
      var gName = g.label || g.name || g.group_id || g.group || '--';
      var contracts = g.contracts || [];
      var gPass = contracts.filter(function(c) { return c.status === 'pass'; }).length;
      var gEver = contracts.filter(function(c) { return !!c.ever_passed; }).length;
      var gFail = contracts.filter(function(c) { return c.status === 'fail'; }).length;
      var gTotal = contracts.length;
      var pct = gTotal > 0 ? (gPass / gTotal) * 100 : 0;
      var borderColor = gFail > 0 ? '#f44' : pct >= 100 ? '#0f9' : '#1a1a2e';

      html += '<div style="background:#0d0d1a;border:1px solid ' + borderColor + ';border-radius:4px;padding:6px;min-width:0;overflow:hidden;">';
      html += '<div style="font-size:0.62rem;font-weight:600;color:#e0e0e8;margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="' + esc(gName) + '">\u25bc ' + esc(gName) + '</div>';
      html += '<div style="font-size:0.5rem;color:#6a6a80;margin-bottom:4px;">' + gPass + '/' + gTotal + (gEver > gPass ? (' \u00b7 ' + gEver + 'e') : '') + '</div>';
      contracts.forEach(function(c) {
        var icons = { pass: '\u2713', fail: '\u2717', awaiting: '\u25cb', not_applicable: '\u2014' };
        var colors = { pass: '#0f9', fail: '#f44', awaiting: '#6a6a80', not_applicable: '#484860' };
        var st = c.status || 'awaiting';
        var cLabel = c.label || c.name || c.contract_id || c.id || '--';
        var cEver = !!c.ever_passed;
        var cEvidence = (c.evidence || '').toString();
        var cLastPassEvidence = (c.last_pass_evidence || '').toString();
        var icon = (icons[st] || '\u25cb');
        var color = (colors[st] || '#6a6a80');
        if (st !== 'pass' && cEver) {
          icon = '\u25c6';
          color = '#0cf';
        }
        var cTitle = cLabel;
        if (cEvidence) cTitle += ' | now: ' + cEvidence;
        if (st !== 'pass' && cEver && cLastPassEvidence) cTitle += ' | ever: ' + cLastPassEvidence;
        var cDetailParts = [];
        if (st !== 'pass' && cEvidence) cDetailParts.push('now: ' + cEvidence);
        if (st !== 'pass' && cEver && cLastPassEvidence) cDetailParts.push('met: ' + cLastPassEvidence);
        var cDetail = cDetailParts.join(' | ');
        html += '<div style="display:flex;gap:3px;padding:0;font-size:0.5rem;align-items:baseline;min-width:0;line-height:1.4;">' +
          '<span style="color:' + color + ';flex-shrink:0;">' + icon + '</span>' +
          '<span style="display:block;flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="' + esc(cTitle) + '">' + esc(cLabel) + '</span>' +
          '</div>';
        if (cDetail) {
          html += '<div style="margin-left:10px;max-width:100%;font-size:0.45rem;color:#6a6a80;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.2;" title="' + esc(cDetail) + '">' + esc(cDetail) + '</div>';
        }
      });
      html += '</div>';
    });
    html += '</div>';
  }

  html += _panelClose();
  return html;
}

function _renderMaturityGates(snap) {
  var ev = snap.eval || {};
  var mt = ev.maturity_tracker || {};
  var categories = mt.categories || [];

  if (!categories.length) {
    return _panelOpen('Maturity Gates', { panelId: 'trust-maturity-gates', ownerTab: 'trust' }) + _emptyMsg('No maturity data') + _panelClose();
  }

  var totalActive = 0;
  var totalGates = 0;
  var totalEver = (typeof mt.ever_active_gates === 'number') ? mt.ever_active_gates : 0;
  var computedEver = 0;
  categories.forEach(function(cat) {
    var gates = cat.gates || [];
    totalGates += gates.length;
    gates.forEach(function(g) { if (g.status === 'active') totalActive++; });
    gates.forEach(function(g) { if (g.ever_met) computedEver++; });
  });
  if (!totalEver && computedEver) totalEver = computedEver;

  var html = _panelOpen('Maturity Gates', {
    panelId: 'trust-maturity-gates',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.6rem;color:' + (totalActive === totalGates ? '#0f9' : '#ff0') + ';">' + totalActive + '/' + totalGates + ' active</span>' +
      (totalEver > totalActive ? '<span style="font-size:0.56rem;color:#0cf;margin-left:6px;">' + totalEver + '/' + totalGates + ' ever</span>' : '')
  });

  categories.forEach(function(cat) {
    var catName = cat.label || cat.id || '--';
    var catId = (cat.id || catName || '').replace(/\W+/g, '_').toLowerCase();
    var gates = cat.gates || [];
    var catPass = gates.filter(function(g) { return g.status === 'active'; }).length;
    var catEver = gates.filter(function(g) { return !!g.ever_met; }).length;
    var catPct = gates.length > 0 ? (catPass / gates.length) * 100 : 0;
    var catTail = catPass + '/' + gates.length + (catEver > catPass ? (' · ' + catEver + 'e') : '');

    html += '<div class="panel" id="mt-' + catId + '" style="margin-bottom:4px;">' +
      '<div class="panel-header" onclick="window._togglePanel && window._togglePanel(this.parentElement)" style="padding:4px 0;">' +
      '<span class="panel-chevron">\u25bc</span>' +
      '<span style="font-size:0.65rem;">' + esc(catName) + '</span>' +
      '<span style="margin-left:auto;font-size:0.55rem;color:#6a6a80;">' + catTail + '</span></div>' +
      '<div class="panel-body">' +
      _barFill(catPct, 100, catPct >= 80 ? '#0f9' : catPct >= 50 ? '#ff0' : '#f44');

    html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:2px 8px;">';
    gates.forEach(function(g) {
      var st = g.status || 'locked';
      var ever = !!g.ever_met;
      var icons = { active: '\u2713', progress: '\u25b6', locked: '\u25cb' };
      var colors = { active: '#0f9', progress: '#ff0', locked: '#484860' };
      if (st !== 'active' && ever) {
        icons[st] = '\u25c6';
        colors[st] = '#0cf';
      }
      var gLabel = g.label || '--';
      var gDisplay = g.display || '';
      html += '<div style="display:flex;gap:3px;padding:1px 0;font-size:0.55rem;align-items:center;min-width:0;">' +
        '<span style="color:' + (colors[st] || '#484860') + ';flex-shrink:0;">' + (icons[st] || '\u25cb') + '</span>' +
        '<span style="flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + esc(gLabel) + '</span>' +
        (gDisplay ? '<span style="color:#6a6a80;font-size:0.48rem;max-width:9em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex-shrink:1;">' + esc(String(gDisplay).substring(0, 32)) + '</span>' : '') +
        '</div>';
    });
    html += '</div>';

    html += '</div></div>';
  });

  html += _panelClose();
  return html;
}

function _renderValidationPack(snap) {
  var ev = snap.eval || {};
  var vp = ev.validation_pack || {};
  var checks = vp.checks || [];

  if (!checks.length) {
    return _panelOpen('Runtime Validation Pack', { panelId: 'trust-validation-pack', ownerTab: 'trust' }) + _emptyMsg('No validation checks yet') + _panelClose();
  }

  var status = (vp.status || 'unknown').toLowerCase();
  var statusColor = status === 'ready' ? '#0f9' : status === 'blocked' ? '#f44' : '#ff0';
  var html = _panelOpen('Runtime Validation Pack', {
    panelId: 'trust-validation-pack',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.72rem;color:' + statusColor + ';">' + esc(status.toUpperCase()) + '</span>'
  });

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin:8px 0;">' +
    _statCard('Current', (vp.checks_passing || 0) + '/' + (vp.checks_total || 0), '#0f9') +
    _statCard('Ever Met', vp.checks_ever_met || 0, '#0cf') +
    _statCard('Regressed', vp.checks_regressed || 0, (vp.checks_regressed || 0) > 0 ? '#ff0' : '#6a6a80') +
    _statCard('Critical', (vp.critical_passing || 0) + '/' + (vp.critical_total || 0), (vp.critical_passing || 0) === (vp.critical_total || 0) ? '#0f9' : '#f44') +
    '</div>';

  var ready = !!vp.ready_for_next_items;
  var continuationReady = !!vp.ready_for_continuation;
  var continuationMode = (vp.continuation_mode || '').toString();
  html += '<div style="font-size:0.62rem;color:' + (ready ? '#0f9' : '#f90') + ';margin-bottom:6px;">' +
    (ready ? '\u2713 Ready for next roadmap items' : '\u26a0 Resolve blocked critical checks before next roadmap items') +
    '</div>';
  if (vp.next_action) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:8px;">' + esc(vp.next_action) + '</div>';
  }
  html += '<div style="font-size:0.6rem;color:' + (continuationReady ? '#0cf' : '#f44') + ';margin-bottom:4px;">' +
    (continuationReady ? '\u25c6 Continuation allowed (historical proof present)' : '\u2717 Continuation blocked (critical gates never met)') +
    (continuationMode ? ' \u00b7 mode=' + esc(continuationMode) : '') +
    '</div>';
  if (vp.continuation_action && vp.continuation_action !== vp.next_action) {
    html += '<div style="font-size:0.56rem;color:#6a6a80;margin-bottom:8px;">' + esc(vp.continuation_action) + '</div>';
  }

  html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:6px;">';
  checks.forEach(function(c) {
    var currentOk = !!c.current_ok;
    var everOk = !!c.ever_ok;
    var critical = !!c.critical;
    var icon = currentOk ? '\u2713' : (everOk ? '\u25c6' : '\u2717');
    var color = currentOk ? '#0f9' : (everOk ? '#0cf' : '#f44');
    var border = currentOk ? '#184430' : (everOk ? '#12425a' : '#4a1f26');

    html += '<div style="background:#0d0d1a;border:1px solid ' + border + ';border-radius:4px;padding:6px;">';
    html += '<div style="display:flex;align-items:center;gap:6px;min-width:0;">' +
      '<span style="color:' + color + ';flex-shrink:0;">' + icon + '</span>' +
      '<span style="font-size:0.6rem;color:#e0e0e8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="' + esc(c.label || c.id || '--') + '">' + esc(c.label || c.id || '--') + '</span>' +
      (critical ? '<span style="margin-left:auto;font-size:0.48rem;color:#f90;">critical</span>' : '') +
      '</div>';

    html += '<div style="margin-top:4px;font-size:0.5rem;color:#6a6a80;line-height:1.35;">' +
      '<div title="' + esc(c.current_detail || '--') + '"><span style="color:#8a8aa0;">now:</span> ' + esc(c.current_detail || '--') + '</div>' +
      '<div title="' + esc(c.ever_detail || '--') + '"><span style="color:#8a8aa0;">ever:</span> ' + esc(c.ever_detail || '--') + '</div>' +
      '</div>';
    html += '</div>';
  });
  html += '</div>';

  html += _panelClose();
  return html;
}

function _renderLanguageGovernance(snap) {
  var ev = snap.eval || {};
  var lang = ev.language || {};
  if (!Object.keys(lang).length) {
    return _panelOpen('Language Governance', { panelId: 'trust-language-governance', ownerTab: 'trust' }) + _emptyMsg('No language eval data') + _panelClose();
  }

  var gateColor = (lang.gate_color || 'unknown').toLowerCase();
  var gateColorHex = gateColor === 'green' ? '#0f9' : gateColor === 'yellow' ? '#ff0' : gateColor === 'red' ? '#f44' : '#6a6a80';
  var phasec = lang.phase_c || {};
  var promoAgg = lang.promotion_aggregate || {};
  var levels = promoAgg.levels || {};
  var colors = promoAgg.colors || {};
  var redQuality = lang.promotion_red_quality_classes || promoAgg.red_quality_classes || 0;
  var redDataLimited = lang.promotion_red_data_limited_classes || promoAgg.red_data_limited_classes || 0;
  var summary = lang.promotion_summary || {};
  var gatesByClass = lang.gate_scores_by_class || {};
  var runtimeMode = (lang.runtime_rollout_mode || 'off').toLowerCase();
  var runtimeEnabled = !!lang.runtime_bridge_enabled;
  var runtimeModeColor = runtimeMode === 'full' ? '#0f9' : runtimeMode === 'canary' ? '#ff0' : '#6a6a80';

  var html = _panelOpen('Language Governance', {
    panelId: 'trust-language-governance',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.72rem;color:' + gateColorHex + ';">gate ' + esc(gateColor.toUpperCase()) + '</span>'
  });

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin:8px 0;">' +
    _statCard('Corpus', lang.corpus_total_examples || 0, '#0af') +
    _statCard('Events', lang.quality_total_events || 0, '#c0f') +
    _statCard('Native', fmtPct(lang.native_usage_rate || 0, 1), (lang.native_usage_rate || 0) >= 0.7 ? '#0f9' : '#ff0') +
    _statCard('Fail-Closed', fmtPct(lang.fail_closed_rate || 0, 1), (lang.fail_closed_rate || 0) <= 0.25 ? '#0f9' : '#f90') +
    '</div>';

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:8px;">' +
    _statCard('Shadow', levels.shadow || 0, '#6a6a80') +
    _statCard('Canary', levels.canary || 0, '#ff0') +
    _statCard('Live', levels.live || 0, '#0f9') +
    _statCard('Red (Quality)', redQuality, redQuality > 0 ? '#f44' : '#0f9') +
    '</div>';
  html += '<div style="font-size:0.56rem;color:#6a6a80;margin:-4px 0 8px 0;">' +
    'data-limited reds: <span style="color:' + (redDataLimited > 0 ? '#ff0' : '#6a6a80') + ';">' + redDataLimited + '</span>' +
    ' · total reds: <span style="color:' + ((colors.red || 0) > 0 ? '#f44' : '#6a6a80') + ';">' + (colors.red || 0) + '</span>' +
    '</div>';

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:8px;">' +
    _statCard('Total Evals', promoAgg.total_evaluations || 0, '#0cf') +
    _statCard('Max Red', promoAgg.max_consecutive_red || 0, (promoAgg.max_consecutive_red || 0) > 0 ? '#f90' : '#6a6a80') +
    _statCard('Max Green', promoAgg.max_consecutive_green || 0, '#0f9') +
    _statCard('Phase C Student', phasec.student_available ? 'READY' : 'NO', phasec.student_available ? '#0f9' : '#f90') +
    '</div>';

  html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Runtime Guard ' +
    '<span style="color:' + runtimeModeColor + ';font-weight:bold;">' + (runtimeEnabled ? esc(runtimeMode.toUpperCase()) : 'OFF') + '</span></div>';
  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:8px;">' +
    _statCard('Guard Events', lang.runtime_guard_total || 0, '#0cf') +
    _statCard('Live Native', lang.runtime_live_total || 0, '#0f9') +
    _statCard('Blocked', lang.runtime_blocked_by_guard_count || 0, (lang.runtime_blocked_by_guard_count || 0) > 0 ? '#ff0' : '#6a6a80') +
    _statCard('Unpromoted', lang.runtime_unpromoted_live_attempts || 0, (lang.runtime_unpromoted_live_attempts || 0) > 0 ? '#f44' : '#0f9') +
    '</div>';
  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:8px;">' +
    _statCard('Live Red', lang.runtime_live_red_classes || 0, (lang.runtime_live_red_classes || 0) > 0 ? '#f44' : '#0f9') +
    _statCard('Live Rate', fmtPct(lang.runtime_live_rate || 0, 1), '#0af') +
    _statCard('Blocked Rate', fmtPct(lang.runtime_blocked_rate || 0, 1), '#ff0') +
    _statCard('Runtime Last', lang.runtime_last_ts ? timeAgo(lang.runtime_last_ts) : '--', '#6a6a80') +
    '</div>';
  var runtimeClassTags = '';
  var runtimeLiveByClass = lang.runtime_live_by_class || {};
  var runtimeBlockedByClass = lang.runtime_blocked_by_class || {};
  Object.keys(runtimeLiveByClass).forEach(function(rc) {
    runtimeClassTags += '<span style="font-size:0.48rem;padding:1px 4px;border-radius:3px;border:1px solid #0f933;color:#0f9;">' +
      esc(rc) + ':live ' + (runtimeLiveByClass[rc] || 0) + '</span>';
  });
  Object.keys(runtimeBlockedByClass).forEach(function(rc) {
    runtimeClassTags += '<span style="font-size:0.48rem;padding:1px 4px;border-radius:3px;border:1px solid #ff022;color:#ff0;">' +
      esc(rc) + ':blocked ' + (runtimeBlockedByClass[rc] || 0) + '</span>';
  });
  if (runtimeClassTags) {
    html += '<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px;">' + runtimeClassTags + '</div>';
  }

  var classRows = Object.keys(summary).length ? summary : gatesByClass;
  if (Object.keys(classRows).length) {
    html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:6px;">';
    Object.entries(classRows).forEach(function(entry) {
      var rc = entry[0];
      var row = entry[1] || {};
      var level = row.level || 'shadow';
      var color = row.color || (gatesByClass[rc] ? gatesByClass[rc].color : 'unknown');
      var reason = row.gate_reason || (gatesByClass[rc] ? gatesByClass[rc].gate_reason : '');
      var lc = level === 'live' ? '#0f9' : level === 'canary' ? '#ff0' : '#6a6a80';
      var gc = color === 'green' ? '#0f9' : color === 'yellow' ? '#ff0' : color === 'red' ? '#f44' : '#6a6a80';
      html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;">' +
        '<div style="display:flex;align-items:center;gap:6px;">' +
        '<span style="font-size:0.6rem;color:#e0e0e8;">' + esc(rc) + '</span>' +
        '<span style="margin-left:auto;font-size:0.5rem;color:' + lc + ';">' + esc(level.toUpperCase()) + '</span>' +
        '<span style="font-size:0.5rem;color:' + gc + ';">' + esc(color.toUpperCase()) + '</span>' +
        '</div>' +
        '<div style="margin-top:4px;font-size:0.5rem;color:#6a6a80;line-height:1.35;">' +
        '<div>g/r streak: ' + (row.consecutive_green || 0) + '/' + (row.consecutive_red || 0) + '</div>' +
        '<div>dwell: ' + fmtNum(row.dwell_s || 0, 1) + 's</div>' +
        '<div>evals: ' + (row.total_evaluations || 0) + '</div>' +
        (reason && reason !== 'ok' ? '<div>reason: ' + esc(reason) + '</div>' : '') +
        '</div></div>';
    });
    html += '</div>';
  }

  html += _panelClose();
  return html;
}

function _renderTruthCalibration(snap) {
  var tc = snap.truth_calibration || {};
  var routeBrier = tc.route_brier_scores || tc.route_brier || {};
  var activeDrifts = tc.active_drift_alerts || tc.drift_alerts || [];

  if (!Object.keys(tc).length) {
    return _panelOpen('Truth Calibration (L6)', { panelId: 'l6-truth-calibration', ownerTab: 'trust' }) + _emptyMsg('Not active') + _panelClose();
  }

  var truthScore = tc.truth_score;
  var maturity = tc.maturity;
  var c = truthScore >= 0.7 ? '#0f9' : truthScore >= 0.4 ? '#ff0' : '#f44';

  var html = _panelOpen('Truth Calibration (L6)', {
    panelId: 'l6-truth-calibration',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.75rem;font-weight:700;color:' + c + ';">' + (truthScore != null ? (truthScore * 100).toFixed(0) + '%' : '--') + '</span>'
  });

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:6px;">' +
    _statCard('Truth Score', truthScore != null ? (truthScore * 100).toFixed(1) + '%' : '--', c) +
    _statCard('Maturity', maturity != null ? (maturity * 100).toFixed(0) + '%' : '--', maturity >= 1.0 ? '#0f9' : '#ff0') +
    _statCard('Brier', tc.brier_score != null ? fmtNum(tc.brier_score, 3) : '--', tc.brier_score != null && tc.brier_score < 0.2 ? '#0f9' : '#ff0') +
    _statCard('ECE', tc.ece != null ? fmtNum(tc.ece, 3) : '--', tc.ece != null && tc.ece < 0.1 ? '#0f9' : '#ff0') +
    '</div>';

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:6px;">' +
    _statCard('Provisional', tc.provisional_count || 0, '#f90') +
    _statCard('Corrections', tc.correction_count || 0, (tc.correction_count || 0) > 0 ? '#f44' : '#0f9') +
    _statCard('Route Brier', Object.keys(routeBrier).length + ' routes', '#6a6a80') +
    _statCard('Drift Alerts', activeDrifts.length || 0, activeDrifts.length > 0 ? '#f90' : '#0f9') +
    '</div>';

  var domains = tc.domain_scores || {};
  var domProv = tc.domain_provisional || {};
  if (typeof domains === 'object' && Object.keys(domains).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Domain Scores</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">';
    Object.entries(domains).forEach(function(e) {
      var name = e[0];
      var val = typeof e[1] === 'object' ? (e[1].score || 0) : (e[1] || 0);
      var isProv = domProv[name] === true;
      var dc = val >= 0.7 ? '#0f9' : val >= 0.4 ? '#ff0' : '#f44';
      html += '<div style="text-align:center;padding:4px;background:#0d0d1a;border:1px solid ' + (isProv ? '#f90' : '#1a1a2e') + ';border-radius:4px;">' +
        '<div style="font-size:0.82rem;font-weight:700;color:' + dc + ';">' + (val * 100).toFixed(0) + '%' + (isProv ? '<span style="font-size:0.5rem;color:#f90;"> P</span>' : '') + '</div>' +
        '<div style="font-size:0.48rem;color:#6a6a80;margin-top:1px;">' + esc(name.replace(/_/g, ' ')) + '</div></div>';
    });
    html += '</div>';
  }

  var coverage = tc.data_coverage || {};
  if (Object.keys(coverage).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Data Coverage</div>';
    html += '<div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px;">';
    Object.entries(coverage).forEach(function(e) {
      var covered = e[1] === true;
      html += '<span style="font-size:0.5rem;padding:1px 4px;border-radius:3px;background:' + (covered ? 'rgba(0,255,153,0.1)' : 'rgba(255,68,68,0.1)') + ';border:1px solid ' + (covered ? '#0f9' : '#f44') + ';color:' + (covered ? '#0f9' : '#f44') + ';">' + esc(e[0]) + '</span>';
    });
    html += '</div>';
  }

  if (typeof routeBrier === 'object' && Object.keys(routeBrier).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Route-Level Brier</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:4px;">';
    Object.entries(routeBrier).forEach(function(e) {
      var brierVal = typeof e[1] === 'number' ? e[1] : 0;
      var bc = brierVal < 0.15 ? '#0f9' : brierVal < 0.25 ? '#ff0' : '#f44';
      html += '<div style="text-align:center;padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
        '<div style="font-size:0.75rem;font-weight:700;color:' + bc + ';">' + fmtNum(brierVal, 4) + '</div>' +
        '<div style="font-size:0.45rem;color:#6a6a80;">' + esc(e[0].replace(/_/g, ' ')) + '</div></div>';
    });
    html += '</div>';
  }

  // Over/underconfidence
  if (tc.overconfidence_error != null || tc.underconfidence_error != null) {
    html += '<div style="display:flex;gap:8px;font-size:0.55rem;margin-bottom:4px;">' +
      '<span style="color:#6a6a80;">Overconfidence: <span style="color:' + (tc.overconfidence_error > 0.1 ? '#f44' : '#0f9') + ';">' + fmtNum(tc.overconfidence_error || 0, 3) + '</span></span>' +
      '<span style="color:#6a6a80;">Underconfidence: <span style="color:' + (tc.underconfidence_error > 0.1 ? '#f44' : '#0f9') + ';">' + fmtNum(tc.underconfidence_error || 0, 3) + '</span></span>' +
      '<span style="color:#6a6a80;">Outcomes: ' + (tc.confidence_outcome_count || 0) + '</span>' +
      '<span style="color:#6a6a80;">Ticks: ' + (tc.tick_count || 0) + '</span>' +
      '</div>';
  }

  // Calibration curve
  var curve = tc.calibration_curve || {};
  if (Object.keys(curve).length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Calibration Curve</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(' + Math.min(Object.keys(curve).length, 5) + ',1fr);gap:3px;margin-bottom:6px;">';
    Object.entries(curve).forEach(function(e) {
      var bucket = e[0];
      var info = e[1] || {};
      var avgAcc = typeof info === 'object' ? (info.avg_accuracy || 0) : (info || 0);
      var avgConf = typeof info === 'object' ? (info.avg_confidence || 0) : 0;
      var count = typeof info === 'object' ? (info.count || 0) : 0;
      var gap = count > 0 ? Math.abs(avgAcc - avgConf) : 0;
      var bc = count === 0 ? '#484860' : gap < 0.1 ? '#0f9' : gap < 0.25 ? '#ff0' : '#f44';
      html += '<div style="text-align:center;padding:2px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;">' +
        '<div style="font-size:0.6rem;font-weight:600;color:' + bc + ';">' + fmtNum(avgAcc, 2) + '</div>' +
        '<div style="font-size:0.4rem;color:#484860;">' + esc(bucket) + (count ? ' n=' + count : '') + '</div></div>';
    });
    html += '</div>';
  }

  // Outcome bridges
  var bridges = tc.outcome_bridges || {};
  if (Object.keys(bridges).length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Outcome Bridges</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:4px;">';
    Object.entries(bridges).forEach(function(e) {
      html += '<span style="padding:1px 4px;border:1px solid #0cf22;color:#0cf;border-radius:2px;font-size:0.48rem;">' + esc(e[0].replace(/_/g,' ')) + ': ' + (typeof e[1] === 'number' ? fmtNum(e[1], 3) : e[1]) + '</span>';
    });
    html += '</div>';
  }

  // Per-provenance accuracy
  var provAcc = tc.per_provenance_accuracy || {};
  if (Object.keys(provAcc).length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Per-Provenance Accuracy</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:4px;">';
    Object.entries(provAcc).forEach(function(e) {
      var acc = typeof e[1] === 'object' ? (e[1].accuracy || 0) : e[1];
      var accNum = typeof acc === 'number' ? acc : 0;
      var pc = accNum >= 0.7 ? '#0f9' : accNum >= 0.4 ? '#ff0' : '#f44';
      html += '<span style="padding:1px 4px;border:1px solid ' + pc + '33;color:' + pc + ';border-radius:2px;font-size:0.48rem;">' +
        esc(e[0].replace(/_/g,' ')) + ': ' + (typeof acc === 'number' ? (acc * 100).toFixed(0) + '%' : esc(String(acc))) + '</span>';
    });
    html += '</div>';
  }

  // Domain trends
  var domTrends = tc.domain_trends || {};
  if (Object.keys(domTrends).length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Domain Trends</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:4px;">';
    Object.entries(domTrends).forEach(function(e) {
      var trend = e[1];
      var dir = typeof trend === 'object' ? (trend.direction || trend.slope_sign || '') : (trend > 0 ? 'up' : trend < 0 ? 'down' : 'flat');
      var tc2 = dir === 'improving' || dir === 'up' ? '#0f9' : dir === 'declining' || dir === 'down' ? '#f44' : '#6a6a80';
      var arrow = dir === 'improving' || dir === 'up' ? '\u2191' : dir === 'declining' || dir === 'down' ? '\u2193' : '\u2192';
      html += '<span style="padding:1px 3px;color:' + tc2 + ';font-size:0.48rem;">' + arrow + ' ' + esc(e[0]) + '</span>';
    });
    html += '</div>';
  }

  // Route sample counts
  var routeSamples = tc.route_sample_counts || {};
  if (Object.keys(routeSamples).length) {
    html += '<div style="display:flex;flex-wrap:wrap;gap:3px;font-size:0.48rem;color:#484860;margin-bottom:4px;">';
    html += '<span>Samples: </span>';
    Object.entries(routeSamples).forEach(function(e) {
      html += '<span>' + esc(e[0]) + ':' + e[1] + '</span>';
    });
    html += '</div>';
  }

  var drifts = activeDrifts;
  if (drifts.length) {
    html += '<div style="font-size:0.62rem;color:#f90;margin-top:4px;margin-bottom:3px;">Drift Alerts</div>';
    drifts.forEach(function(d) {
      var text = typeof d === 'object' ? (d.message || d.description || d.domain || JSON.stringify(d).substring(0, 80)) : String(d);
      var sev = typeof d === 'object' ? (d.severity || '') : '';
      html += '<div style="font-size:0.55rem;padding:1px 0;color:#f90;">' +
        (sev ? '[' + esc(sev) + '] ' : '') + esc(text) + '</div>';
    });
  }

  html += _panelClose();
  return html;
}

function _renderGoldenCommands(snap) {
  var gc = snap.golden_commands || {};
  var counts = gc.counts || {};
  var recent = gc.recent || [];
  var last = gc.last || {};

  if (!Object.keys(counts).length && !recent.length) {
    return _panelOpen('Golden Commands', { panelId: 'trust-golden-commands', ownerTab: 'trust' }) + _emptyMsg('No Golden command events yet.') + _panelClose();
  }

  var status = (last.status || 'none').toLowerCase();
  var statusColor = status === 'executed' ? '#0f9' :
    status === 'invalid' ? '#f44' :
    status === 'blocked' ? '#ff0' :
    status === 'unauthorized' ? '#f90' : '#6a6a80';

  var html = _panelOpen('Golden Commands', {
    panelId: 'trust-golden-commands',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.72rem;color:' + statusColor + ';">' + esc(status) + '</span>'
  });

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:6px;">' +
    _statCard('Executed', counts.executed || 0, '#0f9') +
    _statCard('Blocked', counts.blocked || 0, '#ff0') +
    _statCard('Unauthorized', counts.unauthorized || 0, '#f90') +
    _statCard('Invalid', counts.invalid || 0, '#f44') +
    '</div>';

  if (last && Object.keys(last).length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:4px;">' +
      'Last: <span style="color:#e0e0e8;">' + esc(last.canonical_body || '--') + '</span>' +
      ' \u2022 route <span style="color:#0cf;">' + esc(last.tool_route || '--') + '</span>' +
      (last.block_reason ? ' \u2022 reason <span style="color:#f90;">' + esc(last.block_reason) + '</span>' : '') +
      '</div>';
  }

  if (recent.length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Recent Outcomes</div>';
    html += '<div style="max-height:150px;overflow:auto;">';
    recent.slice().reverse().slice(0, 10).forEach(function(item) {
      var st = (item.status || 'none').toLowerCase();
      var c = st === 'executed' ? '#0f9' :
        st === 'invalid' ? '#f44' :
        st === 'blocked' ? '#ff0' :
        st === 'unauthorized' ? '#f90' : '#6a6a80';
      html += '<div style="display:flex;gap:6px;font-size:0.55rem;padding:2px 0;border-bottom:1px solid #121226;">' +
        '<span style="min-width:70px;color:' + c + ';">' + esc(st) + '</span>' +
        '<span style="flex:1;color:#c0f;">' + esc(item.canonical_body || '--') + '</span>' +
        '<span style="color:#6a6a80;">' + esc(item.tool_route || '--') + '</span>' +
        '</div>';
    });
    html += '</div>';
  }

  html += _panelClose();
  return html;
}

function _renderContradictionEngine(snap) {
  var contra = snap.contradiction || {};

  if (!Object.keys(contra).length) {
    return _panelOpen('Contradiction Engine (L5)', { panelId: 'l5-contradiction-engine', ownerTab: 'trust' }) + _emptyMsg('Not active') + _panelClose();
  }

  var debt = contra.contradiction_debt || contra.debt || 0;
  var debtColor = debt > 0.1 ? '#f44' : debt > 0.05 ? '#ff0' : '#0f9';

  var html = _panelOpen('Contradiction Engine (L5)', {
    panelId: 'l5-contradiction-engine',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.72rem;color:' + debtColor + ';">debt ' + fmtNum(debt, 3) + '</span>'
  });

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:8px;">' +
    _statCard('Debt', fmtNum(debt, 4), debtColor) +
    _statCard('Beliefs', contra.total_beliefs || 0, '#0cf') +
    _statCard('Active', contra.active_beliefs || 0, '#0cf') +
    _statCard('Tensions', contra.active_tensions || 0, '#c0f') +
    '</div>';

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:6px;">' +
    _statCard('Paradoxes', contra.stable_paradoxes || 0, '#c0f') +
    _statCard('Near Misses', contra.near_miss_count || 0, '#f90') +
    _statCard('Resolved', contra.resolved_count || 0, '#0f9') +
    _statCard('Collapses', contra.version_collapses || 0, '#6a6a80') +
    '</div>';

  var debtTrend = contra.debt_trend || [];
  if (debtTrend.length > 1) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">Debt Trend</div>';
    html += '<div style="display:flex;gap:3px;align-items:flex-end;height:24px;margin-bottom:6px;">';
    var maxD = Math.max.apply(null, debtTrend) || 1;
    debtTrend.forEach(function(v) {
      var h = Math.max(2, (v / maxD) * 22);
      var bc = v > 0.1 ? '#f44' : v > 0.05 ? '#ff0' : '#0f9';
      html += '<div style="flex:1;height:' + h + 'px;background:' + bc + ';border-radius:1px;opacity:0.7;" title="' + fmtNum(v, 4) + '"></div>';
    });
    html += '</div>';
  }

  // By type and by resolution
  var byType = contra.by_type || {};
  var byRes = contra.by_resolution || {};
  if (Object.keys(byType).length || Object.keys(byRes).length) {
    html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:6px;">';
    if (Object.keys(byType).length) {
      html += '<div><div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">By Type</div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:2px;">';
      Object.entries(byType).forEach(function(e) {
        html += '<span style="padding:1px 4px;border:1px solid #c0f33;color:#c0f;border-radius:2px;font-size:0.48rem;">' + esc(e[0].replace(/_/g,' ')) + ': ' + e[1] + '</span>';
      });
      html += '</div></div>';
    }
    if (Object.keys(byRes).length) {
      html += '<div><div style="font-size:0.58rem;color:#6a6a80;margin-bottom:2px;">By Resolution</div>' +
        '<div style="display:flex;flex-wrap:wrap;gap:2px;">';
      Object.entries(byRes).forEach(function(e) {
        html += '<span style="padding:1px 4px;border:1px solid #0f933;color:#0f9;border-radius:2px;font-size:0.48rem;">' + esc(e[0].replace(/_/g,' ')) + ': ' + e[1] + '</span>';
      });
      html += '</div></div>';
    }
    html += '</div>';
  }

  // Near miss info
  if (contra.near_miss_rate || contra.extraction_discard_count) {
    html += '<div style="display:flex;gap:8px;font-size:0.5rem;color:#484860;margin-bottom:4px;">' +
      '<span>near miss rate: ' + fmtPct(contra.near_miss_rate || 0) + '</span>' +
      '<span>extraction discards: ' + (contra.extraction_discard_count || 0) + '</span>' +
      '</div>';
  }

  var tensions = contra.tension_records || [];
  if (tensions.length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Active Tensions</div>';
    tensions.forEach(function(t) {
      var mat = t.maturation_score || 0;
      var mc = mat >= 0.8 ? '#0f9' : mat >= 0.5 ? '#ff0' : '#f44';
      html += '<div style="font-size:0.55rem;padding:2px 0;border-bottom:1px solid #1a1a2e;">' +
        '<span style="color:#c0f;">' + esc(t.topic || '--') + '</span>' +
        ' <span style="color:#6a6a80;">beliefs:' + (t.belief_count || 0) + ' revisits:' + (t.revisit_count || 0) + '</span>' +
        ' <span style="color:' + mc + ';">mat:' + fmtNum(mat, 2) + '</span>' +
        '</div>';
    });
  }

  // Recent near misses
  var nearMisses = contra.recent_near_misses || [];
  if (nearMisses.length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Recent Near Misses</div>';
    nearMisses.slice(0, 3).forEach(function(nm) {
      html += '<div style="font-size:0.5rem;padding:1px 0;color:#f90;">' +
        esc((nm.topic || nm.description || nm.claim || '').substring(0, 80)) +
        (nm.type ? ' <span style="color:#6a6a80;">[' + esc(nm.type) + ']</span>' : '') + '</div>';
    });
  }

  html += _panelClose();
  return html;
}

function _renderBeliefGraph(snap) {
  var bg = snap.belief_graph || {};

  if (!Object.keys(bg).length) {
    return _panelOpen('Belief Graph (L7)', { panelId: 'l7-belief-graph', ownerTab: 'trust' }) + _emptyMsg('Not active') + _panelClose();
  }

  var intObj = (typeof bg.integrity === 'object' && bg.integrity) ? bg.integrity : {};
  var healthScore = intObj.health_score || 0;
  var ic = healthScore > 0.7 ? '#0f9' : healthScore > 0.4 ? '#ff0' : '#f44';
  var orphanRate = intObj.orphan_rate || 0;
  var nodeCount = bg.involved_belief_count || intObj.graph_beliefs || 0;
  var edgeCount = bg.total_edges || intObj.total_edges || 0;

  var html = _panelOpen('Belief Graph (L7)', {
    panelId: 'l7-belief-graph',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.72rem;color:' + ic + ';">' + fmtNum(healthScore * 100, 0) + '%</span>'
  });

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:8px;">' +
    _statCard('Beliefs', nodeCount, '#0cf') +
    _statCard('Edges', edgeCount, '#c0f') +
    _statCard('Health', fmtNum(healthScore * 100, 0) + '%', ic) +
    _statCard('Orphan Rate', fmtPct(orphanRate), orphanRate > 0.1 ? '#f44' : '#0f9') +
    '</div>';

  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">';
  if (intObj.fragmentation != null) html += '<div style="text-align:center;padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
    '<div style="font-size:0.75rem;font-weight:700;color:#6a6a80;">' + fmtNum(intObj.fragmentation, 3) + '</div>' +
    '<div style="font-size:0.48rem;color:#484860;">fragmentation</div></div>';
  if (intObj.cycle_count != null) html += '<div style="text-align:center;padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
    '<div style="font-size:0.75rem;font-weight:700;color:#6a6a80;">' + intObj.cycle_count + '</div>' +
    '<div style="font-size:0.48rem;color:#484860;">cycles</div></div>';
  if (intObj.component_count != null) html += '<div style="text-align:center;padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
    '<div style="font-size:0.75rem;font-weight:700;color:#6a6a80;">' + intObj.component_count + '</div>' +
    '<div style="font-size:0.48rem;color:#484860;">components</div></div>';
  if (intObj.dangling_dependency_count != null) html += '<div style="text-align:center;padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
    '<div style="font-size:0.75rem;font-weight:700;color:#6a6a80;">' + intObj.dangling_dependency_count + '</div>' +
    '<div style="font-size:0.48rem;color:#484860;">dangling deps</div></div>';
  html += '</div>';

  var byType = bg.by_type || {};
  if (Object.keys(byType).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Edge Types</div>';
    html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px;">';
    Object.entries(byType).forEach(function(e) {
      html += '<span style="font-size:0.58rem;color:#8a8aa0;">' + esc(e[0].replace(/_/g,' ')) + ': <span style="color:#0cf;">' + e[1] + '</span></span>';
    });
    html += '</div>';
  }

  var prop = bg.propagation || {};
  if (prop.ran_once) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Propagation</div>';
    html += '<div style="display:flex;gap:10px;font-size:0.55rem;color:#8a8aa0;">' +
      '<span>boosted: <span style="color:#0f9;">' + (prop.boosted || 0) + '</span></span>' +
      '<span>diminished: <span style="color:#f44;">' + (prop.diminished || 0) + '</span></span>' +
      '<span>max delta: ' + fmtNum(prop.max_delta || 0, 2) + '</span>' +
      '</div>';
  }

  var topo = bg.topology || {};
  if (topo.root_count != null) {
    var topCStr = '';
    if (topo.top_centrality && topo.top_centrality.length) {
      topCStr = ' · top: ' + topo.top_centrality.slice(0, 3).map(function(tc) {
        if (typeof tc === 'object') return esc((tc.belief_id || tc.id || '?').substring(0, 12)) + '(' + fmtNum(tc.centrality || 0, 2) + ')';
        return esc(String(tc));
      }).join(', ');
    }
    html += '<div style="font-size:0.55rem;color:#484860;margin-top:4px;">roots: ' + topo.root_count + ' · leaves: ' + (topo.leaf_count || 0) + topCStr + '</div>';
  }

  // Bridge stats
  var bridge = bg.bridge || {};
  if (bridge.edges_created || bridge.gates_rejected) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Bridge</div>' +
      '<div style="display:flex;gap:8px;font-size:0.5rem;color:#484860;">' +
      '<span>edges created: <span style="color:#0f9;">' + (bridge.edges_created || 0) + '</span></span>' +
      '<span>gates rejected: <span style="color:#f44;">' + (bridge.gates_rejected || 0) + '</span></span>' +
      '<span>budget suppressed: ' + (bridge.budget_suppressed || 0) + '</span>' +
      '</div>';
  }

  // By basis
  var byBasis = bg.by_basis || {};
  if (Object.keys(byBasis).length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Edge Basis</div>' +
      '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-bottom:4px;">';
    Object.entries(byBasis).forEach(function(e) {
      html += '<span style="padding:1px 4px;border:1px solid #0cf22;color:#0cf;border-radius:2px;font-size:0.48rem;">' + esc(e[0].replace(/_/g,' ')) + ': ' + e[1] + '</span>';
    });
    html += '</div>';
  }

  // Recent edges
  var recentEdges = bg.recent_edges || [];
  if (recentEdges.length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Recent Edges (' + recentEdges.length + ')</div>';
    recentEdges.slice(0, 5).forEach(function(re) {
      var eType = re.type || re.basis || '--';
      var src = re.source || re.source_id || re.from || '?';
      var tgt = re.target || re.target_id || re.to || '?';
      var str = re.strength != null ? re.strength : re.weight;
      html += '<div style="font-size:0.48rem;padding:1px 0;border-bottom:1px solid #0d0d1a;">' +
        '<span style="color:#c0f;">[' + esc(eType) + ']</span> ' +
        '<span style="color:#8a8aa0;">' + esc(String(src).substring(0, 16)) + '</span>' +
        ' <span style="color:#484860;">\u2192</span> ' +
        '<span style="color:#8a8aa0;">' + esc(String(tgt).substring(0, 16)) + '</span>' +
        (str != null ? ' <span style="color:#484860;">s:' + fmtNum(str, 2) + '</span>' : '') +
        (re.basis ? ' <span style="color:#484860;">' + esc(re.basis) + '</span>' : '') +
        (re.age_s != null ? ' <span style="color:#484860;">' + _rnd_ageStr(re.age_s) + '</span>' : '') +
        '</div>';
    });
  }

  // Total created + tick count
  html += '<div style="font-size:0.48rem;color:#484860;margin-top:4px;">total created: ' + (bg.total_created || 0) + ' · ticks: ' + (bg.tick_count || 0) + '</div>';

  html += _panelClose();
  return html;
}

function _renderIntentionResolver(snap) {
  var rs = snap.intention_resolver || {};
  var stage = rs.stage || 'shadow_only';
  var total = rs.total_evaluated || 0;
  var vc = rs.verdict_counts || {};
  var rc = rs.reason_counts || {};
  var sm = rs.shadow_metrics || {};

  var stageColor = stage === 'shadow_only' ? '#888'
    : stage === 'shadow_advisory' ? '#ff0'
    : stage === 'advisory_canary' ? '#f90'
    : stage === 'advisory' ? '#0cf'
    : '#0f9';
  var badge = '<span style="font-size:0.72rem;color:' + stageColor + ';">' + stage + '</span>';

  var html = _panelOpen('Intention Resolver (Stage 1, Shadow)', {
    panelId: 'intention-resolver',
    ownerTab: 'trust',
    badge: badge,
  });

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:8px;">' +
    _statCard('Stage', stage, stageColor) +
    _statCard('Evaluated', total) +
    _statCard('Shadow acc', sm.sufficient_data ? ((sm.shadow_accuracy*100)|0) + '%' : '—',
      sm.sufficient_data && sm.shadow_accuracy >= 0.6 ? '#0f9' : '#888') +
    _statCard('Uptime', rs.uptime_s
      ? (rs.uptime_s < 60 ? (rs.uptime_s|0) + 's'
        : rs.uptime_s < 3600 ? ((rs.uptime_s/60)|0) + 'm'
        : (rs.uptime_s/3600).toFixed(1) + 'h')
      : '—') +
    '</div>';

  var decisions = ['deliver_now', 'deliver_on_next_turn', 'suppress', 'defer'];
  var dColors = { deliver_now: '#0f9', deliver_on_next_turn: '#0cf', suppress: '#f44', defer: '#f90' };
  if (total > 0) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Verdict Distribution</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-bottom:6px;">';
    decisions.forEach(function(d) {
      var v = vc[d] || 0;
      var c = v > 0 ? (dColors[d] || '#888') : '#484860';
      html += '<div style="text-align:center;padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
        '<div style="font-size:0.8rem;font-weight:700;color:' + c + ';">' + v + '</div>' +
        '<div style="font-size:0.48rem;color:#6a6a80;">' + d.replace(/_/g, ' ') + '</div></div>';
    });
    html += '</div>';
  }

  var nonZeroReasons = [];
  Object.keys(rc).forEach(function(k) { if (rc[k] > 0) nonZeroReasons.push([k, rc[k]]); });
  nonZeroReasons.sort(function(a, b) { return b[1] - a[1]; });
  if (nonZeroReasons.length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Reason Code Histogram</div>';
    nonZeroReasons.forEach(function(pair) {
      html += '<div style="font-size:0.5rem;padding:1px 0;color:#8a8aa0;">' +
        '<span style="color:#0cf;font-family:monospace;">' + esc(pair[0]) + '</span> ' +
        '<span style="color:#ff0;">' + pair[1] + '</span></div>';
    });
  }

  var rv = rs.recent_verdicts || [];
  if (rv.length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Recent Verdicts (' + rv.length + ')</div>';
    rv.slice(0, 8).forEach(function(v) {
      var dc = dColors[v.decision] || '#888';
      html += '<div style="font-size:0.5rem;padding:1px 0;color:#8a8aa0;">' +
        '<span style="color:' + dc + ';">[' + (v.decision || '—') + ']</span> ' +
        '<span style="color:#6a6a80;">' + esc(v.reason_code || '') + '</span> ' +
        '<span style="color:#888;">score=' + ((v.score || 0).toFixed(2)) + '</span>' +
        '</div>';
    });
  }

  if (total === 0) {
    html += _emptyMsg('No resolver evaluations yet (shadow-only)');
  }

  return html + _panelClose();
}

function _renderIntentionRegistry(snap) {
  var intent = snap.intentions || {};
  var status = intent.status || {};
  var open = intent.open || [];
  var recent = intent.recent_resolved || [];

  var openCount = status.open_count || 0;
  var stale7d = (status.outcome_histogram_7d || {}).stale || 0;
  var oldestAge = status.oldest_open_intention_age_s || 0;
  var ageTxt = oldestAge < 60 ? (oldestAge|0) + 's'
    : oldestAge < 3600 ? ((oldestAge/60)|0) + 'm'
    : oldestAge < 86400 ? (oldestAge/3600).toFixed(1) + 'h'
    : (oldestAge/86400).toFixed(1) + 'd';

  var badgeColor = openCount === 0 ? '#0f9' : openCount < 5 ? '#ff0' : '#f90';
  var badge = '<span style="font-size:0.72rem;color:' + badgeColor + ';">' + openCount + ' open</span>';

  var html = _panelOpen('L12 Intention Truth Layer (Stage 0)', {
    panelId: 'l12-intention-truth',
    ownerTab: 'trust',
    badge: badge,
  });

  var hist = status.outcome_histogram_7d || {};
  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:8px;">' +
    _statCard('Open', openCount, openCount === 0 ? '#0f9' : '#ff0') +
    _statCard('Oldest age', openCount ? ageTxt : '—') +
    _statCard('Stale (7d)', stale7d, stale7d ? '#f90' : '#6a6a80') +
    _statCard('Resolved (7d)', hist.resolved || 0, '#0f9') +
    '</div>';

  if (Object.keys(hist).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">7-Day Outcome Histogram</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-bottom:6px;">';
    var colorMap = { resolved: '#0f9', failed: '#f44', stale: '#f90', abandoned: '#888' };
    ['resolved', 'failed', 'stale', 'abandoned'].forEach(function(k) {
      var v = hist[k] || 0;
      var cc = v > 0 ? colorMap[k] : '#484860';
      html += '<div style="text-align:center;padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
        '<div style="font-size:0.8rem;font-weight:700;color:' + cc + ';">' + v + '</div>' +
        '<div style="font-size:0.48rem;color:#6a6a80;">' + k + '</div></div>';
    });
    html += '</div>';
  }

  if (open.length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Open Intentions</div>';
    open.slice(0, 6).forEach(function(rec) {
      var kind = rec.backing_job_kind || '—';
      var phrase = (rec.commitment_phrase || '').substring(0, 80);
      var ctype = rec.commitment_type || 'generic';
      var age = (Date.now()/1000) - (rec.created_at || 0);
      var ageT = age < 60 ? (age|0) + 's' : age < 3600 ? ((age/60)|0) + 'm' : (age/3600).toFixed(1) + 'h';
      html += '<div style="font-size:0.55rem;padding:2px 0;border-bottom:1px solid #1a1a2e;">' +
        '<span style="color:#0cf;">[' + esc(kind) + ']</span> ' +
        '<span style="color:#6a6a80;">(' + ageT + ')</span> ' +
        '<span style="color:#888;">[' + esc(ctype) + ']</span> ' +
        esc(phrase) + '</div>';
    });
  }

  if (recent.length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Recently Resolved (' + recent.length + ')</div>';
    recent.slice(0, 5).forEach(function(rec) {
      var oc = rec.outcome || '—';
      var oClr = oc === 'resolved' ? '#0f9' : oc === 'failed' ? '#f44' : oc === 'stale' ? '#f90' : '#888';
      var phrase = (rec.commitment_phrase || '').substring(0, 80);
      var reason = (rec.resolution_reason || '').substring(0, 60);
      html += '<div style="font-size:0.5rem;padding:1px 0;color:#8a8aa0;">' +
        '<span style="color:' + oClr + ';">[' + oc + ']</span> ' +
        esc(phrase) +
        (reason ? ' <span style="color:#6a6a80;">— ' + esc(reason) + '</span>' : '') +
        '</div>';
    });
  }

  if (!open.length && !recent.length) {
    html += _emptyMsg('No intentions tracked yet');
  }

  // Stage-0 → Stage-1 graduation checklist (observability only; never gates behavior)
  var grad = intent.graduation || {};
  var gates = grad.gates || [];
  if (gates.length) {
    var passCount = 0;
    var pendingCount = 0;
    var unknownCount = 0;
    gates.forEach(function(g) {
      if (g.status === 'pass') passCount++;
      else if (g.status === 'unknown') unknownCount++;
      else pendingCount++;
    });
    var headerColor = passCount === gates.length ? '#0f9'
      : pendingCount === 0 ? '#ff0' : '#888';
    html += '<div style="margin-top:10px;padding:6px 8px;background:#0b0b16;border:1px solid #1a1a2e;border-radius:4px;">';
    html += '<div style="font-size:0.62rem;color:' + headerColor + ';font-weight:700;margin-bottom:4px;">' +
      'Stage-1 Graduation Gates ' +
      '<span style="color:#6a6a80;font-weight:400;">(' +
      passCount + ' pass · ' + pendingCount + ' pending · ' + unknownCount + ' external' +
      ')</span>' +
      '</div>';

    var statusIcon = { pass: '[OK]', pending: '[--]', unknown: '[ext]' };
    var statusColor = { pass: '#0f9', pending: '#ff0', unknown: '#888' };

    gates.forEach(function(g) {
      var icon = statusIcon[g.status] || '[?]';
      var col = statusColor[g.status] || '#888';
      var obs = g.observed == null ? '—' : g.observed;
      var req = g.required == null ? '—' : g.required;
      var progress = '';
      if (g.status !== 'unknown' && typeof g.observed === 'number' && typeof g.required === 'number') {
        progress = '<span style="color:#6a6a80;">(' + obs + '/' + req + ')</span>';
      } else if (g.status === 'unknown') {
        progress = '<span style="color:#6a6a80;">(external)</span>';
      }
      html += '<div style="font-size:0.56rem;padding:1px 0;color:#8a8aa0;line-height:1.5;">' +
        '<span style="color:' + col + ';font-family:monospace;font-weight:700;">' + icon + '</span> ' +
        '<span style="color:#aac;">' + esc(g.label || g.id) + '</span> ' +
        progress +
        '</div>';
    });

    var note = grad.stage1_readiness_note || '';
    if (note) {
      html += '<div style="margin-top:4px;font-size:0.5rem;color:#6a6a80;font-style:italic;">' +
        esc(note) + '</div>';
    }
    var docPath = grad.design_doc || '';
    if (docPath) {
      html += '<div style="margin-top:2px;font-size:0.5rem;color:#6a6a80;">Design: <code style="color:#88c;">' +
        esc(docPath) + '</code></div>';
    }
    html += '</div>';
  }

  return html + _panelClose();
}


function _renderQuarantine(snap) {
  var q = snap.quarantine || {};

  if (!Object.keys(q).length) {
    return _panelOpen('Quarantine (L8)', { panelId: 'l8-quarantine', ownerTab: 'trust' }) + _emptyMsg('Not active') + _panelClose();
  }

  var pressure = q.pressure || {};
  var ema = pressure.composite != null ? pressure.composite : (pressure.ema_pressure || 0);
  var band = pressure.band || 'normal';
  var pc = ema > 0.5 ? '#f44' : ema > 0.2 ? '#ff0' : '#0f9';

  var html = _panelOpen('Quarantine (L8)', {
    panelId: 'l8-quarantine',
    ownerTab: 'trust',
    badge: '<span style="font-size:0.72rem;color:' + pc + ';">pressure ' + fmtNum(ema, 3) + '</span>'
  });

  var catCounts = q.category_counts || {};
  html += '<div class="section-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:8px;">' +
    _statCard('Pressure', fmtNum(ema, 3), pc) +
    _statCard('Signals', q.total_signals || 0) +
    _statCard('Categories', Object.keys(catCounts).length) +
    '</div>';

  var byCat = pressure.by_category || {};
  if (Object.keys(byCat).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Pressure by Category</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:3px;margin-bottom:6px;">';
    Object.entries(byCat).forEach(function(e) {
      var v = e[1] || 0;
      var cc = v > 0.2 ? '#f44' : v > 0 ? '#ff0' : '#0f9';
      html += '<div style="text-align:center;padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
        '<div style="font-size:0.7rem;font-weight:700;color:' + cc + ';">' + fmtNum(v, 3) + '</div>' +
        '<div style="font-size:0.45rem;color:#6a6a80;">' + esc(e[0].replace(/_/g,' ')) + '</div></div>';
    });
    html += '</div>';
  }

  if (Object.keys(catCounts).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Signal Counts</div>';
    html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:4px;">';
    Object.entries(catCounts).forEach(function(e) {
      html += '<span style="font-size:0.55rem;padding:1px 5px;background:#1a1a2e;border-radius:3px;color:#8a8aa0;">' + esc(e[0].replace(/_/g,' ')) + ': ' + e[1] + '</span>';
    });
    html += '</div>';
  }

  // Pressure details
  if (pressure.memories_tagged || pressure.promotions_blocked || pressure.chronic_count) {
    html += '<div style="display:flex;gap:8px;font-size:0.5rem;color:#484860;margin-bottom:4px;">' +
      (pressure.chronic_count ? '<span>chronic: <span style="color:#f44;">' + pressure.chronic_count + '</span></span>' : '') +
      (pressure.memories_tagged ? '<span>memories tagged: ' + pressure.memories_tagged + '</span>' : '') +
      (pressure.promotions_blocked ? '<span>promotions blocked: <span style="color:#f44;">' + pressure.promotions_blocked + '</span></span>' : '') +
      '</div>';
  }

  var recent = q.recent_signals || [];
  if (recent.length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Recent Signals</div>';
    recent.slice(0, 5).forEach(function(s) {
      var reason = typeof s === 'object' ? (s.reason || s.description || s.category || '') : String(s);
      var cat = typeof s === 'object' ? (s.category || '') : '';
      var score = typeof s === 'object' && s.score != null ? fmtNum(s.score, 2) : '';
      html += '<div style="font-size:0.55rem;padding:2px 0;border-bottom:1px solid #1a1a2e;">' +
        (cat ? '<span style="color:#f90;">[' + esc(cat.replace(/_/g,' ')) + ']</span> ' : '') +
        (score ? '<span style="color:#6a6a80;">(' + score + ')</span> ' : '') +
        esc(reason.substring(0, 120)) + '</div>';
    });
  }

  // Chronic signals
  var chronic = q.chronic_signals || [];
  if (chronic.length) {
    html += '<div style="font-size:0.58rem;color:#6a6a80;margin-top:4px;margin-bottom:2px;">Chronic Signals (' + chronic.length + ')</div>';
    chronic.slice(0, 3).forEach(function(cs) {
      html += '<div style="font-size:0.5rem;padding:1px 0;color:#f44;">' + esc((cs.reason || cs.category || '').substring(0, 80)) + '</div>';
    });
  }

  // Log stats, cooldowns, suppressed
  var logStats = q.log_stats || {};
  html += '<div style="display:flex;gap:8px;font-size:0.48rem;color:#484860;margin-top:4px;">' +
    '<span>ticks: ' + (q.tick_count || 0) + '</span>' +
    '<span>cooldowns: ' + (q.active_cooldowns || 0) + '</span>' +
    '<span>suppressed: ' + (q.suppressed_duplicates || 0) + '</span>' +
    (logStats.total_logged ? '<span>log: ' + logStats.total_logged + ' entries</span>' : '') +
    '</div>';

  html += _panelClose();
  return html;
}

function _renderWindowDeltas(snap) {
  var ev = snap.eval || {};
  var sc = ev.scorecards || snap.oracle_story_windows || {};
  var wins = sc.windows || {};

  if (!Object.keys(wins).length && !sc.current) {
    return _panelOpen('Rolling Comparisons', { panelId: 'trust-rolling-comparisons', ownerTab: 'trust' }) + _emptyMsg('No scorecard data') + _panelClose();
  }

  var windowKeys = ['15m', '1h', '6h', '24h'];
  var html = _panelOpen('Rolling Comparisons', {
    panelId: 'trust-rolling-comparisons',
    ownerTab: 'trust'
  });

  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">';
  windowKeys.forEach(function(w) {
    var win = wins[w] || {};
    if (!win.available) {
      html += '<div class="stat-card"><div class="stat-value" style="color:#484860;">--</div><div class="stat-label">' + w + '</div></div>';
      return;
    }
    var d = win.deltas || {};
    var soulDelta = d.soul_integrity || 0;
    var dc = soulDelta > 0 ? '#0f9' : soulDelta < 0 ? '#f44' : '#6a6a80';
    var arrow = soulDelta > 0 ? '\u25b2' : soulDelta < 0 ? '\u25bc' : '\u25cf';
    html += '<div class="stat-card" style="cursor:pointer;" onclick="this.nextElementSibling && this.parentElement.parentElement.querySelector(\'[data-win=' + w + ']\').classList.toggle(\'j-hidden\')">' +
      '<div class="stat-value" style="color:' + dc + ';">' + arrow + ' ' + (soulDelta >= 0 ? '+' : '') + fmtNum(soulDelta * 100, 1) + '%</div>' +
      '<div class="stat-label">' + w + '</div>' +
      '<div style="font-size:0.48rem;color:#6a6a80;margin-top:2px;">' + esc((win.headline || '').substring(0, 50)) + '</div>' +
      '</div>';
  });
  html += '</div>';

  windowKeys.forEach(function(w) {
    var win = wins[w] || {};
    if (!win.available) return;
    var proofs = win.proof_points || [];
    html += '<div data-win="' + w + '" style="margin-bottom:6px;">';
    html += '<div style="font-size:0.62rem;color:#0cf;margin-bottom:2px;">' + esc(w) + ' Window</div>';
    if (proofs.length) {
      html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:1px 10px;">';
      proofs.forEach(function(p) {
        var text = typeof p === 'string' ? p : (p.text || '');
        html += '<div style="font-size:0.55rem;padding:1px 0;color:#8a8aa0;">' + esc(text) + '</div>';
      });
      html += '</div>';
    }
    html += '</div>';
  });

  html += _panelClose();
  return html;
}

function _renderStabilityChartPanel(snap) {
  var ev = snap.eval || {};
  if (!ev.stability) {
    return _panelOpen('Stability Chart', { panelId: 'trust-stability-chart', ownerTab: 'trust' }) + _emptyMsg('No stability data') + _panelClose();
  }
  var html = _panelOpen('Stability Chart', {
    panelId: 'trust-stability-chart',
    ownerTab: 'trust'
  });
  html += '<canvas id="trust-stability-chart" style="width:100%;height:160px;"></canvas>';
  html += _panelClose();
  return html;
}

function _renderScoreboard(snap) {
  var ev = snap.eval || {};
  var sb = ev.scoreboard || {};
  var bars = sb.bars || [];

  if (!bars.length) {
    return _panelOpen('Category Scoreboard', { panelId: 'trust-category-scoreboard', ownerTab: 'trust' }) + _emptyMsg('No scoreboard data') + _panelClose();
  }

  var html = _panelOpen('Category Scoreboard', {
    panelId: 'trust-category-scoreboard',
    ownerTab: 'trust'
  });

  if (sb.composite != null) {
    html += '<div style="display:flex;gap:12px;align-items:center;margin-bottom:6px;font-size:0.62rem;">' +
      '<span>Composite: <span style="color:#0cf;font-weight:700;">' + fmtNum(sb.composite, 2) + '</span></span>' +
      (sb.scoring_version ? '<span style="color:#484860;">v' + esc(sb.scoring_version) + '</span>' : '') +
      (sb.badge ? '<span style="color:#f90;font-size:0.5rem;">' + esc(sb.badge) + '</span>' : '') +
      '</div>';
  }

  html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:4px;">';
  bars.forEach(function(bar) {
    var name = bar.category || bar.name || '--';
    var val = bar.score;
    var hasScore = val != null;
    var c = !hasScore ? '#484860' : val >= 0.7 ? '#0f9' : val >= 0.4 ? '#ff0' : '#f44';
    var displayVal = hasScore ? (val * 100).toFixed(0) + '%' : '--';
    var sampleSize = bar.sample_size;
    html += '<div style="text-align:center;padding:5px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;">' +
      '<div style="font-size:0.82rem;font-weight:700;color:' + c + ';">' + displayVal + '</div>' +
      '<div style="font-size:0.48rem;color:#6a6a80;margin-top:1px;">' + esc(name.replace(/_/g, ' ')) + '</div>' +
      (sampleSize != null ? '<div style="font-size:0.42rem;color:#484860;">n=' + sampleSize + '</div>' : '') +
      '</div>';
  });
  html += '</div>';

  if (!sb.composite_enabled) {
    html += '<div style="font-size:0.5rem;color:#484860;margin-top:4px;">Composite scoring not yet enabled</div>';
  }

  html += _panelClose();
  return html;
}


// ═══════════════════════════════════════════════════════════════════════════
// 9. Memory Renderer
// ═══════════════════════════════════════════════════════════════════════════

function renderMemory(snap) {
  var root = el('memory-root');
  if (!root) return;

  var html = '';

  html += _renderTabWayfinding('memory');

  // Memory action toolbar
  html += '<div class="j-toolbar">' +
    '<button class="j-btn-sm" onclick="window.openMemorySearch()">Search Memories</button>' +
    '<button class="j-btn-sm" onclick="window.openRecentMemories()">Browse Recent</button>' +
    '<button class="j-btn-sm" onclick="window.openEnrollment()">Enroll Identity</button>' +
    '</div>';

  // Row 1: Memory Overview (full width)
  html += _renderMemoryOverview(snap);

  // Row 1.5: Dream Processing (full width)
  html += _renderDreamProcessing(snap);

  // Row 1.75: CueGate (full width)
  html += _renderCueGate(snap);

  // Row 1.8: Fractal Recall (full width)
  html += _renderFractalRecall(snap);

  // Row 1.85: HRR / VSA Shadow Substrate (full width; dormant / non-authoritative).
  // Sits directly under Fractal Recall because the Recall Advisory shadow
  // rides on top of Fractal Recall outputs. Authority flags always = false.
  html += _renderHRRShadow(snap);

  // Row 1.9: Synthetic Exercise (full width, only if active or has data)
  html += _renderSyntheticExercise(snap);

  // Row 2: 2-column pairs — left: Cortex, right: Density + Maintenance stacked
  html += '<div class="j-panel-grid">';
  html += '<div>' + _renderMemoryCortex(snap) + '</div>';
  html += '<div>' +
    _renderMemoryDensity(snap) +
    _renderMemoryMaintenance(snap) +
    _renderMovedPanelSummary(
      'Identity Boundary Policy (Moved)',
      'l3-identity-boundary',
      'L3 identity scope policy and block/quarantine enforcement now live in Trust for canonical layer ordering.',
      'memory-identity-boundary-summary'
    ) +
    '</div>';
  html += '</div>';

  // Row 3: 2-column — left: Identity & Recognition, right: Analytics + Rapport stacked
  html += '<div class="j-panel-grid">';
  html += '<div>' + _renderIdentityRecognition(snap) + _renderRapport(snap) + '</div>';
  html += '<div>' + _renderMemoryAnalytics(snap) + '</div>';
  html += '</div>';

  // Row 4: Core Memories (full width, scrollable)
  html += _renderCoreMemories(snap);

  root.innerHTML = html;

  requestAnimationFrame(function() {
    var weightCanvas = document.getElementById('mem-weight-histogram');
    if (weightCanvas && window.drawWeightHistogram) {
      var mem = snap.memory || {};
      var weights = mem.weight_distribution || mem.weights || [];
      if (weights.length) {
        var bins = window.buildHistogram ? window.buildHistogram(weights, 10) : [];
        if (bins.length) window.drawWeightHistogram('mem-weight-histogram', bins);
      }
    }
  });
}

function _renderMemoryOverview(snap) {
  var mem = snap.memory || {};
  var byType = mem.by_type || {};
  var byProv = mem.by_provenance || {};

  var html = _panelOpen('Memory Overview', {
    badge: '<span style="font-size:0.72rem;color:#0cf;">' + (mem.total || mem.total_memories || mem.count || 0) + ' memories</span>'
  });

  html += '<div class="section-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:8px;">' +
    _statCard('Total', mem.total || mem.total_memories || mem.count || 0, '#0cf') +
    _statCard('Observations', byType.observation || 0) +
    _statCard('Conversations', byType.conversation || 0) +
    _statCard('Core', byType.core || 0, '#c0f') +
    '</div>';

  if (Object.keys(byType).length) {
    html += '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">By Type</div>' +
      '<div style="margin-bottom:6px;">' + _tagGrid(byType, 'none') + '</div>';
  }
  if (Object.keys(byProv).length) {
    html += '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">By Provenance</div>' +
      '<div style="margin-bottom:6px;">' + _tagGrid(byProv, 'none') + '</div>';
  }

  html += '<div style="margin-top:6px;"><canvas id="mem-weight-histogram" style="width:100%;height:80px;"></canvas></div>';

  html += _panelClose();
  return html;
}

function _renderCoreMemories(snap) {
  var cm = snap.core_memories || {};
  var items = cm.items || [];

  if (!items.length) {
    return _panelOpen('Core Memories') + _emptyMsg('No core memories yet') + _panelClose();
  }

  var html = _panelOpen('Core Memories', {
    badge: '<span style="font-size:0.6rem;color:#c0f;">' + (cm.total || items.length) + ' memories</span>'
  });

  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
    _statCard('Total', cm.total || 0) +
    _statCard('Explicit', cm.explicit_count || 0, '#0f9') +
    _statCard('High Conf', cm.high_confidence_count || 0, '#0cf') +
    _statCard('User Scoped', cm.user_scoped_count || 0, '#f90') +
    '</div>';

  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;max-height:240px;overflow-y:auto;">';
  items.slice(0, 20).forEach(function(m) {
    var kindColors = { birthday: '#ff0', name: '#0cf', preferred_name: '#0f9', favorite: '#f90', preference: '#c0f', relationship: '#f44', core_note: '#aaa', user_preference: '#c0f', core: '#0cf' };
    var kc = kindColors[m.kind || m.type] || '#6a6a80';
    html += '<div style="padding:3px 0;border-bottom:1px solid #1a1a2e;min-width:0;">' +
      '<div style="display:flex;align-items:baseline;gap:4px;">' +
      '<span style="padding:0 3px;border:1px solid ' + kc + '33;color:' + kc + ';border-radius:2px;font-size:0.48rem;flex-shrink:0;">' + esc(m.kind || m.type || '--') + '</span>' +
      '<span style="font-size:0.58rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + esc((m.payload || '').substring(0, 60)) + '</span>' +
      '</div>' +
      '<div style="font-size:0.45rem;color:#484860;">w:' + fmtNum(m.weight, 2) + ' · ' + esc(m.provenance || '') +
      (m.explicit ? ' · <span style="color:#0f9;">explicit</span>' : '') + '</div></div>';
  });
  html += '</div>';

  html += _panelClose();
  return html;
}

function _renderMemoryDensity(snap) {
  var md = snap.memory_density || {};

  if (!Object.keys(md).length || !md.overall) {
    return _panelOpen('Memory Density') + _emptyMsg('Not computed') + _panelClose();
  }

  var html = _panelOpen('Memory Density', {
    badge: '<span style="font-size:0.72rem;color:#0cf;">' + fmtNum(md.overall, 2) + '</span>'
  });

  var axes = [
    { name: 'Associative Richness', key: 'associative_richness' },
    { name: 'Temporal Coherence', key: 'temporal_coherence' },
    { name: 'Semantic Clustering', key: 'semantic_clustering' },
    { name: 'Distribution', key: 'distribution_score' }
  ];
  axes.forEach(function(ax) {
    var val = md[ax.key] || 0;
    var c = val > 0.6 ? '#0f9' : val > 0.3 ? '#ff0' : '#f44';
    html += '<div style="display:flex;align-items:center;gap:6px;padding:2px 0;">' +
      '<span style="min-width:130px;font-size:0.62rem;color:#6a6a80;">' + ax.name + '</span>' +
      _barFill(val, 1, c) +
      '<span style="min-width:36px;text-align:right;font-size:0.6rem;color:' + c + ';">' + fmtNum(val * 100, 0) + '%</span></div>';
  });

  html += _metricRow('Overall', fmtNum(md.overall, 3));
  html += _metricRow('Memory Count', md.memory_count || 0);

  html += _panelClose();
  return html;
}

function _renderMemoryCortex(snap) {
  var mc = snap.memory_cortex || {};

  if (!Object.keys(mc).length) {
    return _panelOpen('Memory Cortex') + _emptyMsg('Not active') + _panelClose();
  }

  var html = _panelOpen('Memory Cortex');

  var ranker = mc.ranker || {};
  var salience = mc.salience || {};
  var retLog = mc.retrieval_log || {};
  var evalM = mc.eval_metrics || {};

  var rankerStatus = ranker.permanently_disabled ? 'Disabled' : ranker.ready ? 'Active' : ranker.enabled ? 'Enabled' : 'Inactive';
  var rankerC = rankerStatus === 'Active' ? '#0f9' : rankerStatus === 'Disabled' ? '#f44' : '#6a6a80';
  var salStatus = salience.ready ? 'Active' : salience.enabled ? 'Enabled' : 'Inactive';
  var salC = salStatus === 'Active' ? '#0f9' : '#6a6a80';

  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">';

  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;">';
  html += '<div style="font-size:0.65rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">Ranker</div>';
  html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:3px;margin-bottom:4px;">' +
    _statCard('Status', rankerStatus, rankerC) +
    _statCard('Accuracy', ranker.last_accuracy != null ? (ranker.last_accuracy * 100).toFixed(1) + '%' : '--', ranker.last_accuracy > 0.7 ? '#0f9' : '#ff0') +
    _statCard('Trained', ranker.train_count || 0, '#6a6a80') +
    _statCard('Loss', ranker.last_loss != null ? fmtNum(ranker.last_loss, 3) : '--', '#6a6a80') +
    '</div>';
  if (ranker.disable_count > 0) html += '<div style="font-size:0.5rem;color:#f90;">Auto-disables: ' + ranker.disable_count + (ranker.disable_reason ? ' (' + esc(ranker.disable_reason) + ')' : '') + '</div>';
  html += '</div>';

  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;">';
  html += '<div style="font-size:0.65rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">Salience</div>';
  html += '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:3px;margin-bottom:4px;">' +
    _statCard('Status', salStatus, salC) +
    _statCard('Accuracy', salience.last_accuracy != null ? (salience.last_accuracy * 100).toFixed(1) + '%' : '--', salience.last_accuracy > 0.7 ? '#0f9' : '#ff0') +
    _statCard('Trained', salience.train_count || 0, '#6a6a80') +
    _statCard('Blend', salience.model_blend != null ? fmtNum(salience.model_blend, 2) : '--', '#6a6a80') +
    '</div>';
  html += '</div>';

  html += '</div>';

  if (Object.keys(evalM).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Eval Metrics</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-bottom:6px;">' +
      _statCard('Success', evalM.overall_success_rate != null ? (evalM.overall_success_rate * 100).toFixed(0) + '%' : '--', '#0f9') +
      _statCard('Ranker SR', evalM.ranker_success_rate != null ? (evalM.ranker_success_rate * 100).toFixed(0) + '%' : '--', '#0cf') +
      _statCard('Lift', evalM.lift != null ? fmtNum(evalM.lift, 2) : '--', evalM.lift > 0 ? '#0f9' : '#f44') +
      _statCard('Coverage', evalM.coverage != null ? (evalM.coverage * 100).toFixed(0) + '%' : '--', '#6a6a80') +
      '</div>';
  }

  if (Object.keys(retLog).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Retrieval Log</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-bottom:4px;">' +
      _statCard('Events', retLog.total_events || 0, '#6a6a80') +
      _statCard('Outcomes', retLog.total_outcomes || 0, '#6a6a80') +
      _statCard('Success', retLog.retrieval_success_rate != null ? (retLog.retrieval_success_rate * 100).toFixed(0) + '%' : '--', '#0f9') +
      _statCard('Rehydrated', retLog.rehydrated_count || 0, retLog.rehydrated ? '#0f9' : '#6a6a80') +
      '</div>';
    var os = retLog.outcome_stats || {};
    if (Object.keys(os).length) {
      html += '<div style="display:flex;gap:6px;font-size:0.5rem;color:#6a6a80;">';
      Object.entries(os).forEach(function(e) { html += '<span>' + esc(e[0]) + ': ' + e[1] + '</span>'; });
      html += '</div>';
    }
  }

  html += _panelClose();
  return html;
}

function _renderCueGate(snap) {
  var mg = snap.memory_gate || {};
  if (!Object.keys(mg).length) {
    return _panelOpen('CueGate — Memory Access Policy') + _emptyMsg('Not active') + _panelClose();
  }

  var html = _panelOpen('CueGate — Memory Access Policy');

  var obsAllowed = mg.observation_writes_allowed;
  var conActive = mg.consolidation_active;
  var obsC = obsAllowed ? '#0f9' : '#f44';
  var conC = conActive ? '#0cf' : '#6a6a80';
  var modeC = '#6a6a80';
  if (mg.current_mode === 'dreaming' || mg.current_mode === 'sleep') modeC = '#c0f';
  else if (mg.current_mode === 'conversational' || mg.current_mode === 'focused') modeC = '#0f9';
  else if (mg.current_mode === 'reflective' || mg.current_mode === 'deep_learning') modeC = '#ff0';

  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Mode', mg.current_mode || '--', modeC) +
    _statCard('Obs Writes', obsAllowed ? 'Allowed' : 'Blocked', obsC) +
    _statCard('Consolidation', conActive ? 'Active' : 'Idle', conC) +
    _statCard('Read Sessions', (mg.depth || 0) + ' open', mg.is_open ? '#0f9' : '#6a6a80') +
    '</div>';

  html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Total Opens', mg.total_opens || 0, '#6a6a80') +
    _statCard('Last Opened', mg.last_opened_at ? timeAgo(mg.last_opened_at) : 'never', '#6a6a80') +
    _statCard('Last Closed', mg.last_closed_at ? timeAgo(mg.last_closed_at) : 'never', '#6a6a80') +
    '</div>';

  var transitions = mg.recent_transitions || [];
  if (transitions.length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:4px;">Recent Transitions (' + transitions.length + ')</div>';
    html += '<div style="max-height:160px;overflow-y:auto;background:#0a0a15;border:1px solid #1a1a2e;border-radius:4px;padding:4px;">';
    for (var i = transitions.length - 1; i >= Math.max(0, transitions.length - 15); i--) {
      var t = transitions[i];
      var ac = '#6a6a80';
      if (t.action === 'obs_writes_blocked') ac = '#f44';
      else if (t.action === 'obs_writes_enabled') ac = '#0f9';
      else if (t.action === 'consolidation_begin') ac = '#0cf';
      else if (t.action === 'consolidation_end') ac = '#6af';
      else if (t.action === 'open') ac = '#8a8aa0';
      else if (t.action === 'close') ac = '#8a8aa0';
      var age = t.ts ? timeAgo(t.ts) : '--';
      html += '<div style="display:flex;gap:6px;font-size:0.5rem;line-height:1.6;border-bottom:1px solid #12121f;">' +
        '<span style="min-width:44px;color:#555;">' + age + '</span>' +
        '<span style="min-width:110px;color:' + ac + ';font-weight:600;">' + esc(t.action) + '</span>' +
        '<span style="color:#8a8aa0;">' + esc(t.reason || '') + '</span>' +
        (t.actor ? '<span style="color:#555;margin-left:auto;">' + esc(t.actor) + '</span>' : '') +
        '</div>';
    }
    html += '</div>';
  }

  html += _panelClose();
  return html;
}

function _renderFractalRecall(snap) {
  var fr = snap.fractal_recall || {};
  if (!fr.enabled) {
    return _panelOpen('Fractal Recall') + _emptyMsg('Not active') + _panelClose();
  }

  var totalC = fr.total_count || 0;
  var hourly = fr.count_1h || 0;
  var avgR = fr.avg_resonance || 0;
  var avgCL = fr.avg_chain_length || 0;
  var lastTs = fr.last_recall_ts || 0;

  var html = _panelOpen('Fractal Recall', {
    badge: '<span style="font-size:0.72rem;color:#c0f;">' + totalC + ' recalls</span>'
  });

  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Total', totalC, '#c0f') +
    _statCard('This Hour', hourly, hourly >= 5 ? '#f44' : '#0cf') +
    _statCard('Avg Resonance', fmtNum(avgR, 3), avgR >= 0.55 ? '#0f9' : '#ff0') +
    _statCard('Avg Chain Len', fmtNum(avgCL, 1), '#0cf') +
    '</div>';

  html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Mode Skips', fr.blocked_mode_skips || 0, '#6a6a80') +
    _statCard('Cooldown Skips', fr.cooldown_skips || 0, '#6a6a80') +
    _statCard('Low Signal Skips', fr.low_signal_skips || 0, '#6a6a80') +
    '</div>';

  if (lastTs > 0) {
    html += '<div style="font-size:0.55rem;color:#6a6a80;margin-bottom:6px;">Last recall: ' + timeAgo(lastTs) + '</div>';
  }

  var gov = fr.governance_outcomes || {};
  var govKeys = Object.keys(gov);
  if (govKeys.length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:4px;">Governance Outcomes</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(' + Math.min(govKeys.length, 4) + ',1fr);gap:4px;margin-bottom:8px;">';
    var govColors = { ignore: '#f44', hold_for_curiosity: '#ff0', eligible_for_proactive: '#0f9', reflective_only: '#c0f' };
    govKeys.forEach(function(k) {
      html += _statCard(k.replace(/_/g, ' '), gov[k], govColors[k] || '#6a6a80');
    });
    html += '</div>';
  }

  var prov = fr.provenance_mix || {};
  var provKeys = Object.keys(prov);
  if (provKeys.length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:4px;">Provenance Mix</div>';
    html += '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">';
    provKeys.forEach(function(k) {
      html += '<span style="font-size:0.5rem;padding:2px 6px;background:#12121f;border:1px solid #1a1a2e;border-radius:3px;">' +
        '<b style="color:#0cf;">' + prov[k] + '</b> ' + esc(k) + '</span>';
    });
    html += '</div>';
  }

  var recent = fr.recent_recalls || [];
  if (recent.length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:4px;">Recent Recalls (' + recent.length + ')</div>';
    html += '<div style="max-height:200px;overflow-y:auto;background:#0a0a15;border:1px solid #1a1a2e;border-radius:4px;padding:4px;">';
    for (var i = recent.length - 1; i >= 0; i--) {
      var r = recent[i];
      var gc = { ignore: '#f44', hold_for_curiosity: '#ff0', eligible_for_proactive: '#0f9', reflective_only: '#c0f' };
      var govColor = gc[r.governance_action] || '#6a6a80';
      html += '<div style="display:flex;gap:6px;font-size:0.5rem;line-height:1.6;border-bottom:1px solid #12121f;padding:2px 0;">' +
        '<span style="min-width:44px;color:#555;">' + (r.ts ? timeAgo(r.ts) : '--') + '</span>' +
        '<span style="min-width:80px;color:#c0f;">' + esc(r.cue_class || '') + '</span>' +
        '<span style="min-width:90px;color:' + govColor + ';">' + esc((r.governance_action || '').replace(/_/g, ' ')) + '</span>' +
        '<span style="color:#0cf;min-width:30px;">' + fmtNum(r.governance_confidence || 0, 2) + '</span>' +
        '<span style="color:#6a6a80;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' +
          esc((r.reason_codes || []).join(', ')) + '</span>' +
        '</div>';
    }
    html += '</div>';
  }

  html += _panelClose();
  return html;
}

/* ─────────────────────────────────────────────────────────────────────────
 * HRR / VSA Shadow Substrate (memory tab)
 *
 * Pure visualisation. No writes. Authority flags are expected to remain
 * `false` across the whole lifetime of the substrate — rendered as green
 * "OK" chips. If one flips to `true` the validation-pack has already failed;
 * this panel just mirrors the payload.
 *
 * Reads from snap.hrr when available; otherwise from a background fetch of
 * /api/hrr/status cached on window._hrrStatusCache so the panel works
 * without a brain restart that would add "hrr" to the snapshot payload.
 * ──────────────────────────────────────────────────────────────────────── */
window._hrrStatusCache = window._hrrStatusCache || null;
window._hrrStatusInflight = window._hrrStatusInflight || false;

function _kickHRRStatusFetch() {
  if (window._hrrStatusInflight) return;
  window._hrrStatusInflight = true;
  fetch('/api/hrr/status').then(function(r){ return r.ok ? r.json() : null; })
    .then(function(js){
      window._hrrStatusCache = js || window._hrrStatusCache;
      window._hrrStatusInflight = false;
      // Re-render the currently-visible memory tab if we just got the first payload.
      var active = document.querySelector('.j-tab.active');
      if (active && active.getAttribute('data-tab') === 'memory' && window._lastSnap) {
        if (typeof renderMemory === 'function') renderMemory(window._lastSnap);
      }
    })
    .catch(function(){ window._hrrStatusInflight = false; });
}

function _renderHRRShadow(snap) {
  // Prefer snap.hrr (fed by dashboard.snapshot.build_cache post-restart).
  // Otherwise fall back to a self-refreshing cache of /api/hrr/status so the
  // panel stays live even before the next engine restart picks up the
  // snapshot change.
  var h = (snap && snap.hrr) ? snap.hrr : (window._hrrStatusCache || null);
  if (!snap || !snap.hrr) _kickHRRStatusFetch();
  var title = 'HRR / VSA Shadow Substrate';
  var stageBadgeHtml = '<span style="font-size:0.68rem;color:#ff9900;border:1px solid #ff9900;padding:2px 8px;border-radius:3px;letter-spacing:0.06em;">PRE-MATURE</span>';
  var enabledBadgeHtml;
  if (!h) {
    enabledBadgeHtml = '<span style="font-size:0.68rem;color:#6a6a80;border:1px solid #444466;padding:2px 8px;border-radius:3px;margin-left:6px;">loading…</span>';
    var emptyHtml = _panelOpen(title, { badge: stageBadgeHtml + enabledBadgeHtml });
    emptyHtml += _emptyMsg('Fetching /api/hrr/status…');
    emptyHtml += '<div style="margin-top:8px;font-size:0.7rem;"><a href="/hrr" target="_blank" style="color:#00ccff;">Open dedicated HRR dashboard →</a></div>';
    emptyHtml += _panelClose();
    return emptyHtml;
  }

  var enabled = !!h.enabled;
  enabledBadgeHtml = enabled
    ? '<span style="font-size:0.68rem;color:#00ff99;border:1px solid #00ff99;padding:2px 8px;border-radius:3px;margin-left:6px;">ENABLED</span>'
    : '<span style="font-size:0.68rem;color:#6a6a80;border:1px solid #444466;padding:2px 8px;border-radius:3px;margin-left:6px;">DORMANT</span>';

  var html = _panelOpen(title, { badge: stageBadgeHtml + enabledBadgeHtml });

  var w = h.world_shadow || {};
  var s = h.simulation_shadow || {};
  var r = h.recall_advisory || {};

  // Runtime summary
  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Stage', (h.stage || '—').replace(/_/g, ' '), enabled ? '#00ccff' : '#6a6a80') +
    _statCard('Dim', h.dim || '—', '#00ccff') +
    _statCard('Sample Every', h.sample_every_ticks ? (h.sample_every_ticks + 't') : '—', '#00ccff') +
    _statCard('Backend', h.backend || '—', '#6a6a80') +
    '</div>';

  // Three shadow mini-panels side-by-side: World (cyan) / Simulation (magenta) / Recall (green)
  html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;">';

  // — World Shadow —
  html += '<div style="background:#0a0a15;border:1px solid #00ccff55;border-radius:5px;padding:8px;">' +
    '<div style="font-size:0.62rem;color:#00ccff;letter-spacing:0.08em;margin-bottom:6px;font-weight:700;">WORLD SHADOW</div>' +
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:3px;font-size:0.55rem;">' +
      '<div style="color:#6a6a80;">samples</div><div style="text-align:right;color:#0cf;font-family:monospace;">' + (w.samples_total || 0) + '</div>' +
      '<div style="color:#6a6a80;">retained</div><div style="text-align:right;color:#e0e0f0;font-family:monospace;">' + (w.samples_retained || 0) + ' / ' + (w.ring_capacity || 500) + '</div>' +
      '<div style="color:#6a6a80;">bind clean</div><div style="text-align:right;color:#0f9;font-family:monospace;">' + fmtNum(w.binding_cleanliness, 3) + '</div>' +
      '<div style="color:#6a6a80;">cleanup acc</div><div style="text-align:right;color:#0f9;font-family:monospace;">' + fmtNum(w.cleanup_accuracy, 3) + '</div>' +
      '<div style="color:#6a6a80;">Δ prev</div><div style="text-align:right;color:#0cf;font-family:monospace;">' + fmtNum(w.similarity_to_previous, 3) + '</div>' +
    '</div></div>';

  // — Simulation Shadow —
  html += '<div style="background:#0a0a15;border:1px solid #cc00ff55;border-radius:5px;padding:8px;">' +
    '<div style="font-size:0.62rem;color:#cc00ff;letter-spacing:0.08em;margin-bottom:6px;font-weight:700;">SIMULATION SHADOW</div>' +
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:3px;font-size:0.55rem;">' +
      '<div style="color:#6a6a80;">traces</div><div style="text-align:right;color:#c0f;font-family:monospace;">' + (s.samples_total || 0) + '</div>' +
      '<div style="color:#6a6a80;">retained</div><div style="text-align:right;color:#e0e0f0;font-family:monospace;">' + (s.samples_retained || 0) + ' / ' + (s.ring_capacity || 200) + '</div>' +
      '<div style="color:#6a6a80;">last bind</div><div style="text-align:right;color:#0f9;font-family:monospace;">' + fmtNum(s.last_cleanliness_after !== undefined ? s.last_cleanliness_after : s.last_binding_cleanliness, 3) + '</div>' +
      '<div style="color:#6a6a80;">last Δ sim</div><div style="text-align:right;color:#0cf;font-family:monospace;">' + fmtNum(s.last_delta_similarity !== undefined ? s.last_delta_similarity : s.last_similarity_delta, 3) + '</div>' +
      '<div style="color:#6a6a80;">side effects</div><div style="text-align:right;font-family:monospace;color:' + ((s.side_effects || 0) === 0 ? '#0f9' : '#f44') + ';">' + (s.side_effects || 0) + '</div>' +
    '</div></div>';

  // — Recall Advisory —
  html += '<div style="background:#0a0a15;border:1px solid #00ff9955;border-radius:5px;padding:8px;">' +
    '<div style="font-size:0.62rem;color:#00ff99;letter-spacing:0.08em;margin-bottom:6px;font-weight:700;">RECALL ADVISORY</div>' +
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:3px;font-size:0.55rem;">' +
      '<div style="color:#6a6a80;">observations</div><div style="text-align:right;color:#0f9;font-family:monospace;">' + (r.samples_total || 0) + '</div>' +
      '<div style="color:#6a6a80;">retained</div><div style="text-align:right;color:#e0e0f0;font-family:monospace;">' + (r.samples_retained || 0) + ' / ' + (r.ring_capacity || 500) + '</div>' +
      '<div style="color:#6a6a80;">LRU cache</div><div style="text-align:right;color:#0cf;font-family:monospace;">' + (r.cache_size || 0) + ' / ' + (r.cache_capacity || 2000) + '</div>' +
      '<div style="color:#6a6a80;">help rate</div><div style="text-align:right;color:#0f9;font-family:monospace;">' + fmtNum(r.help_rate, 3) + '</div>' +
      '<div style="color:#6a6a80;">fs writes</div><div style="text-align:right;font-family:monospace;color:' + ((r.fs_writes || 0) === 0 ? '#0f9' : '#f44') + ';">' + (r.fs_writes || 0) + '</div>' +
    '</div></div>';

  html += '</div>';

  // Authority flags — these MUST remain false. Render every flag as a chip.
  var authKeys = [
    'policy_influence',
    'belief_write_enabled',
    'canonical_memory',
    'autonomy_influence',
    'llm_raw_vector_exposure',
    'soul_integrity_influence'
  ];
  var chips = authKeys.map(function(k) {
    var v = h[k];
    var ok = v === false;
    var color = ok ? '#00ff99' : '#ff4444';
    var bg = ok ? 'rgba(0,255,136,.08)' : 'rgba(255,68,68,.15)';
    return '<span style="font-size:0.55rem;font-family:monospace;padding:3px 7px;border-radius:3px;background:' + bg +
      ';border:1px solid ' + color + '44;color:' + color + ';">' + esc(k) + ' = ' + String(v) + '</span>';
  }).join('');
  html += '<div style="font-size:0.6rem;color:#6a6a80;margin-bottom:4px;letter-spacing:0.04em;">AUTHORITY FLAGS (all must be <code style="color:#00ff99;">false</code>)</div>';
  html += '<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;">' + chips + '</div>';

  // Footer: links + non-negotiable reminder
  html += '<div style="display:flex;gap:10px;align-items:center;font-size:0.62rem;color:#6a6a80;margin-top:6px;">' +
    '<a href="/hrr" target="_blank" style="color:#00ccff;">Open full HRR dashboard →</a>' +
    '<a href="/api/hrr/status" target="_blank" style="color:#6a6a80;">raw /api/hrr/status</a>' +
    '<a href="/api/hrr/samples" target="_blank" style="color:#6a6a80;">raw /api/hrr/samples</a>' +
    '<span style="margin-left:auto;color:#ff9900;">dormant · read-only · non-authoritative</span>' +
    '</div>';

  html += _panelClose();
  return html;
}

function _renderSyntheticExercise(snap) {
  var se = snap.synthetic_exercise || {};
  var hasSessions = (se.utterances_stt || 0) > 0 || (se.total_runs || 0) > 0;
  if (!se.active && !hasSessions) {
    return '';
  }

  var html = _panelOpen('Synthetic Perception Exercise', {
    badge: se.active
      ? '<span style="font-size:0.72rem;color:#0f9;">ACTIVE</span>'
      : '<span style="font-size:0.72rem;color:#6a6a80;">idle (' + (se.total_runs || 0) + ' runs)</span>'
  });

  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('STT Processed', se.utterances_stt || 0, '#0cf') +
    _statCard('Routes', se.routes_produced || 0, '#0f9') +
    _statCard('Distillation', se.distillation_records || 0, '#c0f') +
    _statCard('Blocked', se.blocked_side_effects || 0, '#f44') +
    '</div>';

  // Invariant leak indicators
  var leakKeys = ['llm_leaks', 'tts_leaks', 'transcription_emit_leaks', 'memory_side_effects', 'identity_side_effects'];
  var hasLeaks = false;
  for (var li = 0; li < leakKeys.length; li++) {
    if ((se[leakKeys[li]] || 0) > 0) hasLeaks = true;
  }
  html += '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;">';
  for (var li = 0; li < leakKeys.length; li++) {
    var lk = leakKeys[li];
    var lv = se[lk] || 0;
    var lColor = lv === 0 ? '#0f9' : '#f44';
    var lLabel = lk.replace(/_/g, ' ');
    html += '<span style="font-size:0.55rem;padding:2px 6px;background:#0a0a15;border:1px solid ' +
      (lv === 0 ? '#1a1a2e' : '#f44') + ';border-radius:3px;color:' + lColor + ';">' +
      lLabel + ': ' + lv + '</span>';
  }
  html += '</div>';
  if (hasLeaks) {
    html += '<div style="padding:4px 8px;background:#2a0000;border:1px solid #f44;border-radius:4px;color:#f44;font-size:0.65rem;margin-bottom:8px;">' +
      'INVARIANT VIOLATION — synthetic exercise leaked into protected subsystems</div>';
  }

  // Route histogram
  var rh = se.route_histogram || {};
  var rhKeys = Object.keys(rh);
  if (rhKeys.length > 0) {
    rhKeys.sort(function(a, b) { return rh[b] - rh[a]; });
    var maxRh = rh[rhKeys[0]] || 1;
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:4px;">Route Histogram</div>';
    html += '<div style="background:#0a0a15;border:1px solid #1a1a2e;border-radius:4px;padding:6px;margin-bottom:8px;">';
    for (var ri = 0; ri < rhKeys.length; ri++) {
      var rk = rhKeys[ri];
      var rv = rh[rk];
      var pct = Math.round((rv / maxRh) * 100);
      html += '<div style="display:flex;align-items:center;gap:6px;font-size:0.55rem;margin-bottom:2px;">' +
        '<span style="min-width:100px;color:#0cf;text-align:right;">' + esc(rk) + '</span>' +
        '<div style="flex:1;height:10px;background:#12121f;border-radius:2px;overflow:hidden;">' +
        '<div style="width:' + pct + '%;height:100%;background:linear-gradient(90deg,#0cf,#0f9);border-radius:2px;"></div>' +
        '</div>' +
        '<span style="min-width:24px;color:#6a6a80;text-align:right;">' + rv + '</span>' +
        '</div>';
    }
    html += '</div>';
  }

  // Recent route examples
  var examples = se.recent_route_examples || [];
  if (examples.length > 0) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:4px;">Recent Examples (' + examples.length + ')</div>';
    html += '<div style="max-height:150px;overflow-y:auto;background:#0a0a15;border:1px solid #1a1a2e;border-radius:4px;padding:4px;">';
    for (var ei = examples.length - 1; ei >= Math.max(0, examples.length - 10); ei--) {
      var ex = examples[ei];
      html += '<div style="display:flex;gap:6px;font-size:0.5rem;line-height:1.6;border-bottom:1px solid #12121f;padding:1px 0;">' +
        '<span style="min-width:100px;color:#c0f;">' + esc(ex.route || '') + '</span>' +
        '<span style="color:#6a6a80;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(ex.text || '') + '</span>' +
        '</div>';
    }
    html += '</div>';
  }

  html += _panelClose();
  return html;
}
window._renderSyntheticExercise = _renderSyntheticExercise;

function _renderIdentityRecognition(snap) {
  var identity = snap.identity || {};
  var speakers = snap.speakers || {};
  var faces = snap.faces || {};

  var html = _panelOpen('Identity & Recognition');

  // --- Fused identity status ---
  var idName = identity.identity || 'unknown';
  var idColor = identity.is_known ? '#0f9' : '#6a6a80';
  var methodLabel = identity.method || '--';
  var confPct = identity.confidence != null ? fmtPct(identity.confidence) : '--';
  var recState = identity.recognition_state || '--';
  var recColor = recState === 'confirmed_match' ? '#0f9' : recState === 'tentative_match' ? '#ff0' : recState === 'unknown_present' ? '#f80' : '#6a6a80';

  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:8px;">';
  html += '<div style="grid-column:1/-1;display:flex;align-items:center;gap:8px;padding:4px 0;">' +
    '<span style="font-size:1.1rem;font-weight:700;color:' + idColor + ';">' + esc(idName) + '</span>' +
    '<span style="font-size:0.6rem;color:#484860;background:#1a1a2e;padding:2px 6px;border-radius:3px;">' + esc(methodLabel) + '</span>' +
    '<span style="font-size:0.65rem;color:' + recColor + ';">' + esc(recState) + '</span>' +
    (identity.conflict ? '<span style="font-size:0.55rem;color:#f44;font-weight:600;">CONFLICT</span>' : '') +
    (identity.cold_start_active ? '<span style="font-size:0.55rem;color:#ff0;">COLD START</span>' : '') +
    '</div>';
  html += '</div>';

  // --- Per-modality signals ---
  var voice = identity.voice_signal || {};
  var face = identity.face_signal || {};
  html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px;">';

  var voiceConf = voice.confidence != null ? voice.confidence : 0;
  var voiceColor = voiceConf >= 0.5 ? '#0cf' : voiceConf > 0 ? '#484860' : '#2a2a40';
  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;">' +
    '<div style="font-size:0.6rem;color:#6a6a80;margin-bottom:2px;">Voice</div>' +
    '<div style="font-size:0.85rem;font-weight:600;color:' + voiceColor + ';">' + esc(voice.name || 'none') + '</div>' +
    '<div style="font-size:0.6rem;color:#484860;">' + fmtPct(voiceConf) + (voice.age_s ? ' &middot; ' + fmtNum(voice.age_s, 0) + 's ago' : '') + '</div>' +
    '</div>';

  var faceConf = face.confidence != null ? face.confidence : 0;
  var faceColor = faceConf >= 0.55 ? '#c0f' : faceConf > 0 ? '#484860' : '#2a2a40';
  html += '<div style="background:#0d0d1a;border:1px solid #1a1a2e;border-radius:4px;padding:6px;">' +
    '<div style="font-size:0.6rem;color:#6a6a80;margin-bottom:2px;">Face</div>' +
    '<div style="font-size:0.85rem;font-weight:600;color:' + faceColor + ';">' + esc(face.name || 'none') + '</div>' +
    '<div style="font-size:0.6rem;color:#484860;">' + fmtPct(faceConf) + (face.age_s ? ' &middot; ' + fmtNum(face.age_s, 0) + 's ago' : '') + '</div>' +
    '</div>';

  html += '</div>';

  // --- Status indicators ---
  html += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;font-size:0.6rem;">';

  var flipCount = identity.flip_count || 0;
  var flipColor = flipCount === 0 ? '#0f9' : flipCount <= 3 ? '#ff0' : '#f44';
  html += '<span style="color:' + flipColor + ';background:#0d0d1a;padding:2px 6px;border-radius:3px;">flips: ' + flipCount + '</span>';

  if (identity.persisted) {
    html += '<span style="color:#0cf;background:#0d0d1a;padding:2px 6px;border-radius:3px;">persisted ' +
      (identity.persist_remaining_s ? fmtNum(identity.persist_remaining_s, 0) + 's left' : '') + '</span>';
  }

  if (identity.tentative_name) {
    html += '<span style="color:#ff0;background:#0d0d1a;padding:2px 6px;border-radius:3px;">tentative: ' +
      esc(identity.tentative_name) + ' (' + fmtPct(identity.tentative_confidence || 0) + ')</span>';
  }

  html += '<span style="color:#484860;background:#0d0d1a;padding:2px 6px;border-radius:3px;">conf: ' + confPct + '</span>';

  // Trust state badge
  var trustState = identity.voice_trust_state || 'unknown';
  var trustColors = {trusted: '#0f9', tentative: '#ff0', degraded: '#f80', conflicted: '#f44', unknown: '#484860'};
  var trustColor = trustColors[trustState] || '#484860';
  html += '<span style="color:' + trustColor + ';background:#0d0d1a;padding:2px 6px;border-radius:3px;font-weight:600;">trust: ' + esc(trustState) + '</span>';

  if (identity.visible_person_count > 1) {
    html += '<span style="color:#f80;background:#0d0d1a;padding:2px 6px;border-radius:3px;">multi-person (' + identity.visible_person_count + ')</span>';
  }

  if (identity.multi_person_suppression_active) {
    html += '<span style="color:#f44;background:#0d0d1a;padding:2px 6px;border-radius:3px;">voice suppressed</span>';
  }

  if (identity.threshold_assist_active) {
    html += '<span style="color:#0cf;background:#0d0d1a;padding:2px 6px;border-radius:3px;">threshold assist: ' +
      esc(identity.threshold_assist_name || '') + '</span>';
  }

  if (identity.resolution_basis) {
    html += '<span style="color:#484860;background:#0d0d1a;padding:2px 6px;border-radius:3px;">basis: ' + esc(identity.resolution_basis) + '</span>';
  }

  if (identity.unknown_voice_count > 0) {
    html += '<span style="color:#f80;background:#0d0d1a;padding:2px 6px;border-radius:3px;">unknown voices: ' + identity.unknown_voice_count + '</span>';
  }

  html += '</div>';

  // Trust reason
  if (identity.trust_reason) {
    html += '<div style="font-size:0.55rem;color:#484860;margin-bottom:6px;padding:2px 4px;">reason: ' + esc(identity.trust_reason) + '</div>';
  }

  // Speaker profiles
  var speakerProfiles = speakers.profiles || [];
  if (speakerProfiles.length) {
    html += '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">Speaker Profiles (' + speakerProfiles.length + ')</div>';
    speakerProfiles.forEach(function(sp) {
      var name = sp.name || sp.id || '--';
      html += '<div style="display:flex;gap:6px;padding:2px 0;border-bottom:1px solid #1a1a2e;font-size:0.65rem;align-items:center;">' +
        '<span style="font-weight:600;color:#0cf;min-width:80px;">' + esc(name) + '</span>' +
        '<span style="color:#6a6a80;">clips:' + (sp.enrollment_clips || sp.sample_count || sp.samples || 0) + '</span>' +
        '<span style="color:#6a6a80;">hits:' + (sp.interaction_count || 0) + '</span>' +
        (sp.last_seen ? '<span style="color:#484860;">' + timeAgo(sp.last_seen) + '</span>' : '') +
        '<button class="j-btn-xs j-btn-red" style="margin-left:auto;" onclick="window.removeSpeaker(\'' + esc(name) + '\')">forget</button>' +
        '</div>';
    });
  }

  // Face profiles
  var faceProfiles = faces.profiles || [];
  if (faceProfiles.length) {
    html += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Face Profiles (' + faceProfiles.length + ')</div>';
    faceProfiles.forEach(function(fp) {
      var name = fp.name || fp.id || '--';
      html += '<div style="display:flex;gap:6px;padding:2px 0;border-bottom:1px solid #1a1a2e;font-size:0.65rem;align-items:center;">' +
        '<span style="font-weight:600;color:#c0f;min-width:80px;">' + esc(name) + '</span>' +
        '<span style="color:#6a6a80;">clips:' + (fp.enrollment_crops || fp.sample_count || fp.samples || 0) + '</span>' +
        '<span style="color:#6a6a80;">hits:' + (fp.interaction_count || 0) + '</span>' +
        (fp.last_seen ? '<span style="color:#484860;">' + timeAgo(fp.last_seen) + '</span>' : '') +
        '<button class="j-btn-xs j-btn-red" style="margin-left:auto;" onclick="window.removeFace(\'' + esc(name) + '\')">forget</button>' +
        '</div>';
    });
  }

  html += _panelClose();
  return html;
}

function _renderRapport(snap) {
  var rapport = snap.rapport || {};
  var rels = rapport.relationships || [];
  var intel = rapport.personal_intel || [];

  if (!rels.length && !intel.length) {
    return _panelOpen('Rapport') + _emptyMsg('No relationship data yet') + _panelClose();
  }

  var html = _panelOpen('Rapport');

  if (rels.length) {
    html += '<div style="font-size:0.68rem;color:#6a6a80;margin-bottom:3px;">Relationships</div>';
    rels.forEach(function(r) {
      var name = r.name || r.person || '--';
      var role = r.role || r.type || '';
      var trust = r.trust_level || r.trust || 0;
      html += '<div style="display:flex;gap:6px;padding:2px 0;border-bottom:1px solid #1a1a2e;font-size:0.65rem;">' +
        '<span style="font-weight:600;min-width:80px;">' + esc(name) + '</span>' +
        (role ? '<span style="color:#6a6a80;">' + esc(role) + '</span>' : '') +
        '<span style="color:#0cf;">trust:' + fmtNum(trust, 2) + '</span></div>';
    });
  }

  if (intel.length) {
    html += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:6px;margin-bottom:3px;">Personal Intel (' + intel.length + ')</div>';
    html += '<div style="max-height:250px;overflow-y:auto;">';
    intel.slice(0, 20).forEach(function(i) {
      html += '<div style="font-size:0.6rem;padding:2px 0;border-bottom:1px solid #1a1a2e;">' +
        esc((i.payload || '').substring(0, 80)) +
        ' <span style="color:#484860;">w:' + fmtNum(i.weight, 2) + '</span></div>';
    });
    html += '</div>';
  }

  html += _panelClose();
  return html;
}

function _renderIdentityBoundary(snap) {
  var ib = snap.identity_boundary || {};

  if (!Object.keys(ib).length) {
    return _panelOpen('Identity Boundary (L3)', { panelId: 'l3-identity-boundary', ownerTab: 'trust' }) + _emptyMsg('No data') + _panelClose();
  }

  var html = _panelOpen('Identity Boundary (L3)', {
    panelId: 'l3-identity-boundary',
    ownerTab: 'trust'
  });

  html += '<div class="section-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:8px;">' +
    _statCard('Scoped', ib.scope_count || ib.total_scoped || 0, '#0cf') +
    _statCard('Blocked', ib.block_count || ib.total_blocked || 0, (ib.block_count || 0) > 0 ? '#f44' : '#0f9') +
    _statCard('Quarantined', ib.quarantine_count || 0, (ib.quarantine_count || 0) > 0 ? '#ff0' : '#0f9') +
    '</div>';

  if (ib.referenced_allow_count) html += _metricRow('Referenced-Allow', ib.referenced_allow_count);
  if (ib.recent_blocks) {
    html += '<div style="font-size:0.68rem;color:#6a6a80;margin-top:4px;margin-bottom:3px;">Recent Blocks</div>';
    var blocks = Array.isArray(ib.recent_blocks) ? ib.recent_blocks : [];
    blocks.slice(0, 5).forEach(function(b) {
      var text = typeof b === 'object' ? (b.reason || b.description || JSON.stringify(b).substring(0, 80)) : String(b);
      html += '<div style="font-size:0.6rem;color:#f44;padding:1px 0;">' + esc(text.substring(0, 100)) + '</div>';
    });
  }

  html += _panelClose();
  return html;
}

function _renderMemoryAnalytics(snap) {
  var ma = snap.memory_analytics || {};
  var assoc = snap.memory_associations || {};

  if (!Object.keys(ma).length && !Object.keys(assoc).length) {
    return _panelOpen('Memory Analytics') + _emptyMsg('No analytics data') + _panelClose();
  }

  var html = _panelOpen('Memory Analytics');

  if (Object.keys(ma).length) {
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
      _statCard('Total', ma.total || 0, '#0cf') +
      _statCard('Strong', ma.strong_count || 0, '#0f9') +
      _statCard('Weak', ma.weak_count || 0, (ma.weak_count || 0) > 0 ? '#f44' : '#0f9') +
      _statCard('Integrity', ma.integrity_score != null ? (ma.integrity_score * 100).toFixed(0) + '%' : '--', (ma.integrity_score || 0) >= 0.9 ? '#0f9' : '#ff0') +
      '</div>';

    var byType = ma.by_type || {};
    if (Object.keys(byType).length) {
      html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">By Type</div>';
      html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:3px;margin-bottom:6px;">';
      Object.entries(byType).forEach(function(e) {
        html += '<div style="text-align:center;padding:3px;background:#0d0d1a;border:1px solid #1a1a2e;border-radius:3px;">' +
          '<div style="font-size:0.7rem;font-weight:700;color:#0cf;">' + e[1] + '</div>' +
          '<div style="font-size:0.45rem;color:#6a6a80;">' + esc(e[0].replace(/_/g, ' ')) + '</div></div>';
      });
      html += '</div>';
    }
  }

  if (Object.keys(assoc).length) {
    html += '<div style="font-size:0.62rem;color:#6a6a80;margin-bottom:3px;">Associations</div>';
    html += '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:3px;">' +
      _statCard('Total', assoc.total_connections || 0, '#c0f') +
      _statCard('Avg/Memory', assoc.avg_per_memory != null ? fmtNum(assoc.avg_per_memory, 1) : '--', '#6a6a80') +
      _statCard('Isolated', assoc.isolated_count || 0, (assoc.isolated_count || 0) > 0 ? '#f90' : '#0f9') +
      '</div>';
  }

  html += _panelClose();
  return html;
}

function _renderMemoryMaintenance(snap) {
  var mm = snap.memory_maintenance || {};

  if (!Object.keys(mm).length) {
    return _panelOpen('Memory Maintenance') + _emptyMsg('No maintenance data') + _panelClose();
  }

  var html = _panelOpen('Memory Maintenance');

  html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:4px;">' +
    _statCard('GC Runs', mm.total_gc_runs || 0, '#6a6a80') +
    _statCard('Repaired', mm.total_repaired || 0, (mm.total_repaired || 0) > 0 ? '#f90' : '#0f9') +
    _statCard('Last GC', mm.last_gc_time ? timeAgo(mm.last_gc_time) : 'never', '#6a6a80') +
    _statCard('Integrity', mm.last_integrity_check ? timeAgo(mm.last_integrity_check) : 'never', '#6a6a80') +
    '</div>';

  html += _panelClose();
  return html;
}


// ═══════════════════════════════════════════════════════════════════════════
// 9b. Dream Processing Panel
// ═══════════════════════════════════════════════════════════════════════════

var _DREAM_TYPE_COLORS = {
  bridge_candidate: '#0cf', symbolic_summary: '#c0f', tension_flag: '#f44',
  consolidation_proposal: '#0f9', waking_question: '#ff0', shadow_scenario: '#6a6a80'
};

var _DREAM_STATE_COLORS = {
  promoted: '#0f9', held: '#ff0', discarded: '#f44', quarantined: '#c0f', pending: '#6a6a80'
};

var _DREAM_STANCE_COLORS = {
  waking: '#0f9', dreaming: '#c0f', reflective: '#0cf'
};

function _renderDreamProcessing(snap) {
  var da = snap.dream_artifacts || {};
  var buf = da.buffer || {};
  var val = da.validator || {};
  var artifacts = da.recent_artifacts || [];
  var cycles = da.cycle_history || [];
  var stance = da.observer_stance || {};
  var clusters = (snap.memory_clusters || {}).clusters || [];

  var hasData = buf.buffer_size > 0 || cycles.length > 0 || artifacts.length > 0;

  if (!hasData) {
    return _panelOpen('Dream Processing', {
      badge: '<span class="dream-stance-badge" style="color:' + (_DREAM_STANCE_COLORS[stance.stance] || '#6a6a80') + ';">' + esc(stance.stance || 'waking') + '</span>'
    }) + _emptyMsg('No dream data yet. Dreams run when the system enters sleep/dreaming mode or every 10 minutes with 20+ memories.') + _panelClose();
  }

  var html = _panelOpen('Dream Processing', {
    badge: '<span class="dream-stance-badge" style="color:' + (_DREAM_STANCE_COLORS[stance.stance] || '#6a6a80') + ';border-color:' + (_DREAM_STANCE_COLORS[stance.stance] || '#6a6a80') + '33;">' + esc((stance.stance || 'waking').toUpperCase()) + '</span>'
  });

  // --- A. Header Stats ---
  var lastCycleTs = cycles.length ? cycles[cycles.length - 1].timestamp : 0;
  html += '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-bottom:8px;">' +
    _statCard('Cycles', cycles.length, '#e0e0e8') +
    _statCard('Last Cycle', lastCycleTs ? timeAgo(lastCycleTs) : 'never', '#6a6a80') +
    _statCard('Buffer', (buf.buffer_size || 0) + '/' + (buf.buffer_capacity || 200), '#0cf') +
    _statCard('Validated', val.validation_count || 0, '#c0f') +
    _statCard('Last Validation', val.last_validation ? timeAgo(val.last_validation) : 'never', '#6a6a80') +
    '</div>';

  // --- B. Validation Funnel ---
  var byState = buf.by_state || {};
  var totalCreated = buf.total_created || 0;
  var totalPromoted = buf.total_promoted || 0;
  var promoRate = totalCreated > 0 ? ((totalPromoted / totalCreated) * 100).toFixed(1) : '0.0';

  html += '<div class="dream-funnel">';
  html += '<div class="dream-funnel-title">Artifact Validation Funnel</div>';
  html += '<div class="dream-funnel-flow">';
  html += '<span class="dream-funnel-node" style="border-color:#e0e0e833;">Created<b>' + totalCreated + '</b></span>';
  html += '<span class="dream-funnel-arrow">&rarr;</span>';
  html += '<span class="dream-funnel-node" style="border-color:#6a6a8033;">Pending<b>' + (byState.pending || 0) + '</b></span>';
  html += '<span class="dream-funnel-arrow">&rarr;</span>';
  html += '<div class="dream-funnel-branches">';
  html += '<span class="dream-funnel-node" style="border-color:#0f933;color:#0f9;">Promoted<b>' + (buf.total_promoted || 0) + '</b></span>';
  html += '<span class="dream-funnel-node" style="border-color:#ff033;color:#ff0;">Held<b>' + (buf.total_held || 0) + '</b></span>';
  html += '<span class="dream-funnel-node" style="border-color:#f4433;color:#f44;">Discarded<b>' + (buf.total_discarded || 0) + '</b></span>';
  html += '<span class="dream-funnel-node" style="border-color:#c0f33;color:#c0f;">Quarantined<b>' + (buf.total_quarantined || 0) + '</b></span>';
  html += '</div>';
  html += '</div>';
  html += '<div style="font-size:0.6rem;color:#6a6a80;margin-top:4px;">Promotion Rate: <b style="color:' + (parseFloat(promoRate) > 60 ? '#f44' : parseFloat(promoRate) > 30 ? '#ff0' : '#0f9') + ';">' + promoRate + '%</b>';
  if (buf.avg_confidence != null) html += ' &middot; Avg Confidence: <b>' + fmtNum(buf.avg_confidence, 2) + '</b>';
  if (buf.avg_coherence != null) html += ' &middot; Avg Coherence: <b>' + fmtNum(buf.avg_coherence, 2) + '</b>';
  html += '</div>';
  html += '</div>';

  // --- C. Cycle History Log ---
  html += '<div style="margin-top:8px;">';
  html += '<div style="font-size:0.7rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">Cycle History (' + cycles.length + ')</div>';
  if (cycles.length) {
    html += '<div class="dream-cycle-scroll">';
    var sortedCycles = cycles.slice().reverse();
    sortedCycles.forEach(function(c, idx) {
      var cId = 'dream-cycle-' + idx;
      html += '<div class="dream-cycle-row" onclick="var el=document.getElementById(\'' + cId + '\');if(el){el.parentElement.classList.toggle(\'expanded\');}">';
      html += '<div class="dream-cycle-header">';
      html += '<span class="dream-cycle-time">' + esc(timeAgo(c.timestamp)) + '</span>';
      html += '<span class="dream-cycle-badge">scanned <b>' + (c.memories_scanned || 0) + '</b></span>';
      html += '<span class="dream-cycle-badge" style="color:#0cf;">' + (c.clusters_found || 0) + ' clusters</span>';
      if (c.associations_made > 0) html += '<span class="dream-cycle-badge" style="color:#0f9;">' + c.associations_made + ' linked</span>';
      if (c.reinforced > 0) html += '<span class="dream-cycle-badge" style="color:#ff0;">' + c.reinforced + ' reinforced</span>';
      if (c.decayed > 0) html += '<span class="dream-cycle-badge" style="color:#f44;">' + c.decayed + ' decayed</span>';
      if (c.artifacts_created > 0) html += '<span class="dream-cycle-badge" style="color:#c0f;">' + c.artifacts_created + ' artifacts</span>';
      if (c.topics_noted > 0) html += '<span class="dream-cycle-badge" style="color:#ff0;">' + c.topics_noted + ' topics</span>';
      if (c.cortex_trained) html += '<span class="dream-cycle-badge" style="color:#0f9;">cortex trained</span>';
      html += '</div>';
      html += '<div class="dream-cycle-detail" id="' + cId + '">';
      html += '<div style="font-size:0.6rem;color:#a0a0b0;padding:4px 0;white-space:pre-wrap;">' + esc(c.summary || '') + '</div>';
      html += '</div>';
      html += '</div>';
    });
    html += '</div>';
  } else {
    html += _emptyMsg('No cycles recorded yet.');
  }
  html += '</div>';

  // --- D. Artifact Inspector ---
  html += '<div style="margin-top:8px;">';
  html += '<div style="font-size:0.7rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">Artifact Inspector (' + artifacts.length + ')</div>';

  // Type breakdown badges
  var byType = buf.by_type || {};
  var typeEntries = Object.entries(byType).sort(function(a, b) { return b[1] - a[1]; });
  if (typeEntries.length) {
    html += '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">';
    typeEntries.forEach(function(e) {
      var tc = _DREAM_TYPE_COLORS[e[0]] || '#6a6a80';
      html += '<span style="display:inline-block;padding:1px 6px;border:1px solid ' + tc + '33;border-radius:3px;font-size:0.6rem;color:' + tc + ';">' + esc(e[0]) + ' <b>' + e[1] + '</b></span>';
    });
    html += '</div>';
  }

  if (artifacts.length) {
    html += '<div class="dream-artifact-scroll">';
    var sortedArtifacts = artifacts.slice().sort(function(a, b) { return (b.timestamp || 0) - (a.timestamp || 0); });
    sortedArtifacts.forEach(function(a, idx) {
      var aId = 'dream-art-' + idx;
      var tc = _DREAM_TYPE_COLORS[a.type] || '#6a6a80';
      var sc = _DREAM_STATE_COLORS[a.state] || '#6a6a80';
      html += '<div class="dream-artifact-row" onclick="var el=document.getElementById(\'' + aId + '\');if(el){el.parentElement.classList.toggle(\'expanded\');}">';
      html += '<div class="dream-artifact-header">';
      html += '<span class="dream-artifact-type" style="color:' + tc + ';border-color:' + tc + '33;">' + esc(a.type || '--') + '</span>';
      html += '<span class="dream-artifact-state" style="color:' + sc + ';border-color:' + sc + '33;">' + esc(a.state || '--') + '</span>';
      html += '<span class="dream-artifact-content">' + esc((a.content || '').substring(0, 80)) + (a.content && a.content.length > 80 ? '...' : '') + '</span>';
      html += '<span style="font-size:0.55rem;color:#6a6a80;white-space:nowrap;">' + fmtNum(a.confidence, 2) + ' conf</span>';
      html += '<span style="font-size:0.55rem;color:#6a6a80;white-space:nowrap;">' + fmtNum(a.coherence, 2) + ' coh</span>';
      html += '<span class="dream-artifact-time">' + esc(timeAgo(a.timestamp)) + '</span>';
      html += '</div>';

      // Expandable detail
      html += '<div class="dream-artifact-detail" id="' + aId + '">';
      html += '<div style="padding:6px 0;">';
      html += '<div style="font-size:0.62rem;color:#a0a0b0;margin-bottom:4px;white-space:pre-wrap;">' + esc(a.content || '') + '</div>';
      if (a.notes) html += '<div style="font-size:0.6rem;color:#ff0;margin-bottom:3px;"><b>Validator:</b> ' + esc(a.notes) + '</div>';
      html += '<div style="display:flex;gap:8px;font-size:0.58rem;color:#6a6a80;">';
      html += '<span>Confidence: <b>' + fmtNum(a.confidence, 3) + '</b></span>';
      html += '<span>Coherence: <b>' + fmtNum(a.coherence, 3) + '</b></span>';
      if (a.promoted_at) html += '<span>Promoted: <b>' + esc(timeAgo(a.promoted_at)) + '</b></span>';
      if (a.discarded_at) html += '<span>Discarded: <b>' + esc(timeAgo(a.discarded_at)) + '</b></span>';
      html += '</div>';
      if (a.source_ids && a.source_ids.length) {
        html += '<div style="font-size:0.55rem;color:#484860;margin-top:3px;">Sources: ' + a.source_ids.map(function(id) { return esc(id.substring(0, 12)); }).join(', ') + '</div>';
      }
      html += '<div style="font-size:0.55rem;color:#484860;margin-top:2px;">ID: ' + esc(a.id || '') + '</div>';
      html += '</div>';
      html += '</div>';
      html += '</div>';
    });
    html += '</div>';
  } else {
    html += _emptyMsg('No artifacts generated yet.');
  }
  html += '</div>';

  // --- E. Dream Specialist (Distillation) ---
  var distill = da.distillation || {};
  var dFeatCount = distill.feature_count || 0;
  var dLabelCount = distill.label_count || 0;
  var dQuarantined = distill.quarantined || 0;
  var dLastSignal = distill.last_signal_s;
  var reasonDist = distill.reason_distribution || {};
  var reasonKeys = Object.keys(reasonDist);

  html += '<div style="margin-top:8px;">';
  html += '<div style="font-size:0.7rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">Dream Specialist <span style="font-size:0.55rem;font-weight:400;color:#6a6a80;margin-left:4px;">DREAM_SYNTHESIS (shadow-only)</span></div>';

  if (dFeatCount === 0 && dLabelCount === 0) {
    html += '<div style="font-size:0.6rem;color:#6a6a80;padding:6px 0;">No distillation signals yet. Signals accumulate when the ReflectiveValidator runs during dream cycles.</div>';
  } else {
    html += '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px;">' +
      _statCard('Features', dFeatCount, '#c0f') +
      _statCard('Labels', dLabelCount, '#0cf') +
      _statCard('Quarantined', dQuarantined, dQuarantined > 0 ? '#f44' : '#6a6a80') +
      _statCard('Last Signal', dLastSignal != null ? fmtNum(dLastSignal, 0) + 's ago' : 'never', '#6a6a80') +
      '</div>';

    if (reasonKeys.length > 0) {
      html += '<div style="font-size:0.6rem;color:#a0a0b0;margin-bottom:3px;">Reason Distribution</div>';
      html += '<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px;">';
      var reasonColors = {
        'coherence_pass': '#0f9', 'novel_insight': '#0cf', 'redundant': '#ff0',
        'low_coherence': '#f44', 'low_confidence': '#fa0', 'stale': '#888',
        'anomalous': '#c0f', 'uncategorized': '#6a6a80'
      };
      reasonKeys.sort(function(a, b) { return (reasonDist[b] || 0) - (reasonDist[a] || 0); });
      reasonKeys.forEach(function(k) {
        var rc = reasonColors[k] || '#6a6a80';
        html += '<span style="display:inline-block;padding:1px 6px;border:1px solid ' + rc + '33;border-radius:3px;font-size:0.58rem;color:' + rc + ';">' + esc(k) + ' <b>' + reasonDist[k] + '</b></span>';
      });
      html += '</div>';
    }
  }
  html += '</div>';

  // --- F. Cluster Summary ---
  html += '<div style="margin-top:8px;">';
  html += '<div style="font-size:0.7rem;font-weight:600;color:#e0e0e8;margin-bottom:4px;">Memory Clusters (' + clusters.length + ')</div>';
  if (clusters.length) {
    html += '<div class="dream-cluster-grid">';
    clusters.forEach(function(cl) {
      html += '<div class="dream-cluster-card">';
      html += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">';
      html += '<span style="font-size:0.65rem;font-weight:600;color:#e0e0e8;">' + esc(cl.topic || 'untitled') + '</span>';
      html += '<span style="font-size:0.55rem;color:#6a6a80;">' + (cl.size || 0) + ' memories</span>';
      html += '</div>';
      html += '<div style="font-size:0.58rem;color:#6a6a80;margin-bottom:4px;">type: ' + esc(cl.type || '--') + '</div>';
      var coh = cl.coherence || 0;
      var cohColor = coh >= 0.65 ? '#0f9' : coh >= 0.45 ? '#ff0' : '#f44';
      html += '<div style="display:flex;align-items:center;gap:6px;">';
      html += '<span style="font-size:0.58rem;color:#6a6a80;">coherence</span>';
      html += '<div class="bar-track" style="flex:1;"><div class="bar-fill" style="width:' + Math.min(100, coh * 100) + '%;background:' + cohColor + ';"></div></div>';
      html += '<span style="font-size:0.58rem;color:' + cohColor + ';font-weight:600;">' + fmtNum(coh, 2) + '</span>';
      html += '</div>';
      html += '</div>';
    });
    html += '</div>';
  } else {
    html += _emptyMsg('No clusters formed yet.');
  }
  html += '</div>';

  html += _panelClose();
  return html;
}


// ═══════════════════════════════════════════════════════════════════════════
// 10. Initialization
// ═══════════════════════════════════════════════════════════════════════════

_loadApiKey();
connectWS();

var _savedRestore = localStorage.getItem('jarvis-tab');
if (_savedRestore && TAB_NAMES.indexOf(_savedRestore) !== -1) switchTab(_savedRestore);

setInterval(function() {
  if (ws && ws.readyState === WebSocket.OPEN) return;
  fetch('/api/full-snapshot').then(function(r) { return r.json(); }).then(function(snap) {
    _lastSnap = snap;
    window._lastSnap = snap;
    updateGlobalUI(snap);
    renderActiveTab(snap);
  }).catch(function() {});
}, 5000);
