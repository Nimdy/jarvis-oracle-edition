/**
 * Shared status-marker renderer for prose dashboard pages (P1.7).
 *
 * Any page can opt in by:
 *   1. Including this script: <script src="/static/status-markers.js"></script>
 *   2. Tagging architectural claims with:
 *        <span data-status-marker="phase_6_5_l3_governance"></span>
 *
 * This script fetches /api/meta/status-markers once and fills every
 * matching span with a colored badge. It is the architecture-contract
 * enforcement surface for prose pages.
 *
 * Safe to include on pages that have zero markers (the fetch is cheap
 * and the DOM walk is a no-op).
 */
(function() {
  'use strict';

  var COLORS = {
    'SHIPPED': { bg: 'rgba(46, 213, 115, 0.16)', fg: '#2ed573', border: '#2ed573' },
    'PARTIAL': { bg: 'rgba(255, 184, 0, 0.16)', fg: '#ffb800', border: '#ffb800' },
    'PRE-MATURE': { bg: 'rgba(140, 149, 159, 0.18)', fg: '#b7c1cb', border: '#8c959f' },
    'DEFERRED': { bg: 'rgba(148, 85, 255, 0.18)', fg: '#c5a6ff', border: '#9455ff' },
    'UNKNOWN': { bg: 'rgba(255, 71, 87, 0.16)', fg: '#ff4757', border: '#ff4757' },
  };

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function(c) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
    });
  }

  function renderBadge(status, legend) {
    var c = COLORS[status] || COLORS.UNKNOWN;
    var title = legend && legend[status] ? legend[status] : status;
    return '<span style="display:inline-block;padding:1px 8px;border-radius:10px;' +
      'background:' + c.bg + ';color:' + c.fg + ';border:1px solid ' + c.border + ';' +
      'font-size:0.68rem;font-weight:600;letter-spacing:0.04em;vertical-align:middle;' +
      'font-family:monospace;" title="' + esc(title) + '">' + esc(status) + '</span>';
  }

  function render(markers, legend) {
    var nodes = document.querySelectorAll('[data-status-marker]');
    nodes.forEach(function(n) {
      var key = n.getAttribute('data-status-marker');
      var status = markers[key] || (COLORS[key] ? key : 'UNKNOWN');
      n.innerHTML = renderBadge(status, legend);
    });
  }

  function load() {
    var nodes = document.querySelectorAll('[data-status-marker]');
    if (!nodes.length) return;
    fetch('/api/meta/status-markers', { cache: 'no-store' })
      .then(function(r) { return r.ok ? r.json() : null; })
      .then(function(data) {
        if (!data || !data.markers) return;
        render(data.markers, data.legend || {});
      })
      .catch(function() { /* silent */ });
  }

  document.addEventListener('DOMContentLoaded', function() {
    setTimeout(load, 150);
  });
})();
