/**
 * System Truth Score strip for the top of index.html (P1.7).
 *
 * Renders an at-a-glance aggregate of:
 *   - /api/self-test verdict (validation pack + cache shape + attestation)
 *   - /api/autonomy/level three-axis state (current_ok / ever_ok /
 *     prior_attested_ok)
 *   - /api/meta/status-markers roll-up counts
 *
 * Strict rules (architecture-contract compliance):
 *   - Never fabricates a score. If any upstream call fails, the strip
 *     shows "unavailable" rather than a green number.
 *   - Only reads; never mutates state or cache.
 *   - Hides the strip entirely on failure so it can never misinform.
 */
(function() {
  'use strict';

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function(c) {
      return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
    });
  }

  function pill(label, value, colorVar) {
    var color = colorVar || '--text-primary';
    return '<span style="display:inline-flex;align-items:center;gap:6px;' +
      'padding:4px 10px;border:1px solid rgba(255,255,255,0.12);' +
      'border-radius:12px;font-size:0.72rem;margin-right:8px;">' +
      '<span style="color:var(--text-muted);">' + esc(label) + '</span>' +
      '<strong style="color:var(' + color + ');font-family:monospace;">' +
        esc(String(value)) + '</strong>' +
      '</span>';
  }

  function render(strip, self_test, autonomy, markers) {
    if (!strip) return;
    if (!self_test || !autonomy || !markers) {
      strip.style.display = 'none';
      return;
    }

    var verdict = self_test.status || (self_test.ok ? 'ok' : 'unknown');
    var verdictColor = (self_test.ok === true || verdict === 'ok' ||
      verdict === 'proven' || verdict === 'mature' || verdict === 'pass')
      ? '--green'
      : (verdict === 'blocked' || verdict === 'fail' || verdict === 'degraded'
          ? '--red' : '--amber');

    var level = (autonomy.current_level != null
      ? autonomy.current_level
      : (autonomy.autonomy_level != null ? autonomy.autonomy_level : '?'));
    var currentOk = !!autonomy.current_ok;
    var priorOk = !!autonomy.prior_attested_ok;
    var requestOk = !!autonomy.request_ok;
    var activationOk = !!autonomy.activation_ok;

    var mMap = markers.markers || {};
    var counts = { SHIPPED: 0, PARTIAL: 0, 'PRE-MATURE': 0, DEFERRED: 0 };
    Object.keys(mMap).forEach(function(k) {
      var v = mMap[k];
      if (counts[v] != null) counts[v] += 1;
    });

    strip.innerHTML =
      '<div style="display:flex;flex-wrap:wrap;align-items:center;' +
        'gap:4px;padding:10px 16px;background:rgba(0,0,0,0.25);' +
        'border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.72rem;">' +
        '<strong style="color:var(--cyan);margin-right:12px;font-size:0.8rem;">' +
          'System Truth Score</strong>' +
        pill('Self-Test', verdict, verdictColor) +
        pill('Autonomy L', level, '--cyan') +
        pill('current_ok', currentOk ? 'true' : 'false',
             currentOk ? '--green' : '--text-muted') +
        pill('prior_attested_ok', priorOk ? 'true' : 'false',
             priorOk ? '--green' : '--text-muted') +
        pill('request_ok', requestOk ? 'true' : 'false',
             requestOk ? '--green' : '--text-muted') +
        pill('activation_ok', activationOk ? 'true' : 'false',
             activationOk ? '--green' : '--text-muted') +
        '<span style="color:var(--text-muted);margin-left:auto;">' +
          'SHIPPED ' + counts.SHIPPED + ' · ' +
          'PARTIAL ' + counts.PARTIAL + ' · ' +
          'PRE-MATURE ' + counts['PRE-MATURE'] + ' · ' +
          'DEFERRED ' + counts.DEFERRED +
        '</span>' +
      '</div>';
    strip.style.display = '';
  }

  function poll() {
    var strip = document.getElementById('j-truth-strip');
    if (!strip) return;
    var self_test = null, autonomy = null, markers = null;
    var done = 0;
    function maybeRender() {
      done += 1;
      if (done >= 3) render(strip, self_test, autonomy, markers);
    }
    fetch('/api/self-test', { cache: 'no-store' })
      .then(function(r) { return r.ok ? r.json() : null; })
      .then(function(d) { self_test = d; maybeRender(); })
      .catch(function() { maybeRender(); });
    fetch('/api/autonomy/level', { cache: 'no-store' })
      .then(function(r) { return r.ok ? r.json() : null; })
      .then(function(d) { autonomy = d; maybeRender(); })
      .catch(function() { maybeRender(); });
    fetch('/api/meta/status-markers', { cache: 'no-store' })
      .then(function(r) { return r.ok ? r.json() : null; })
      .then(function(d) { markers = d; maybeRender(); })
      .catch(function() { maybeRender(); });
  }

  document.addEventListener('DOMContentLoaded', function() {
    setTimeout(poll, 1200);
    setInterval(poll, 30000);
  });
})();
