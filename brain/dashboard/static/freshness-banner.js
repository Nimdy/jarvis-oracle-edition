/**
 * Shared static-page freshness helper for JARVIS dashboard pages.
 *
 * Usage: include this script on any static page and add
 *   <div class="j-freshness-banner" id="j-freshness-banner"></div>
 * somewhere near the top of <body>.
 *
 * Static HTML/CSS/JS assets are served from disk on each request, so editing
 * a prose page after process start does NOT make the running brain stale.
 * This helper intentionally keeps the legacy banner hidden. Runtime-code
 * freshness is handled separately by dashboard.js via /api/system/code-freshness
 * on the live index page, where a restart may actually be required.
 *
 * Never modifies state. Safe to fail silently.
 */
(function() {
  'use strict';

  function _hideBanner() {
    var banner = document.getElementById('j-freshness-banner');
    if (!banner) return;
    banner.style.display = 'none';
    banner.innerHTML = '';
  }

  document.addEventListener('DOMContentLoaded', function() {
    _hideBanner();
  });
})();
