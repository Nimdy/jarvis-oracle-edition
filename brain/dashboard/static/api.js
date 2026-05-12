/* ═══════════════════════════════════════════════════════════════════════════
   JARVIS Oracle Edition — API Reference (api.js)
   Sidebar nav, endpoint toggle, search/filter, copy-to-clipboard, deep links
   ═══════════════════════════════════════════════════════════════════════════ */

(function() {
  'use strict';

  // 1. Sidebar scroll spy (reuse from docs pattern)
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
      if (sections[i].el.offsetTop <= scrollY) { active = sections[i]; break; }
    }
    navLinks.forEach(function(l) { l.classList.remove('active'); });
    if (active) active.link.classList.add('active');
  }

  var _raf = 0;
  window.addEventListener('scroll', function() {
    if (_raf) return;
    _raf = requestAnimationFrame(function() { _raf = 0; updateActiveNav(); });
  });
  updateActiveNav();

  // 2. Endpoint expand/collapse
  window.toggleEndpoint = function(headerEl) {
    var body = headerEl.nextElementSibling;
    var chevron = headerEl.querySelector('.a-ep-chevron');
    var isOpen = body.classList.contains('open');
    body.classList.toggle('open');
    if (chevron) chevron.classList.toggle('open');
    if (!isOpen) {
      var card = headerEl.closest('.a-endpoint');
      if (card && card.id) history.replaceState(null, '', '#' + card.id);
    }
  };

  // 3. Copy to clipboard
  window.copyCode = function(btn) {
    var codeEl = btn.closest('.a-code');
    var text = codeEl.textContent.replace(/^Copy(ed!)?/, '').trim();
    navigator.clipboard.writeText(text).then(function() {
      btn.textContent = 'Copied!';
      btn.classList.add('copied');
      setTimeout(function() { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
    });
  };

  // 4. Search / filter
  var searchInput = document.getElementById('api-search');
  var filterBtns = document.querySelectorAll('.a-filter-btn');
  var endpoints = document.querySelectorAll('.a-endpoint');
  var categories = document.querySelectorAll('.a-category');
  var activeFilter = 'all';

  function applyFilters() {
    var q = (searchInput ? searchInput.value : '').toLowerCase().trim();
    var visibleCats = {};

    endpoints.forEach(function(ep) {
      var method = ep.getAttribute('data-method') || '';
      var path = ep.getAttribute('data-path') || '';
      var desc = ep.getAttribute('data-desc') || '';
      var cat = ep.getAttribute('data-cat') || '';
      var text = (method + ' ' + path + ' ' + desc).toLowerCase();

      var matchFilter = (activeFilter === 'all') || (method === activeFilter);
      var matchSearch = !q || text.indexOf(q) !== -1;

      if (matchFilter && matchSearch) {
        ep.style.display = '';
        visibleCats[cat] = true;
      } else {
        ep.style.display = 'none';
      }
    });

    categories.forEach(function(c) {
      var catId = c.getAttribute('data-cat-id') || '';
      c.style.display = visibleCats[catId] ? '' : 'none';
    });
  }

  if (searchInput) {
    searchInput.addEventListener('input', applyFilters);
  }

  filterBtns.forEach(function(btn) {
    btn.addEventListener('click', function() {
      filterBtns.forEach(function(b) { b.classList.remove('active'); });
      btn.classList.add('active');
      activeFilter = btn.getAttribute('data-filter') || 'all';
      applyFilters();
    });
  });

  // 5. Deep link: open endpoint from hash
  function openFromHash() {
    var hash = location.hash.slice(1);
    if (!hash) return;
    var el = document.getElementById(hash);
    if (!el) return;
    if (el.classList.contains('a-endpoint')) {
      var body = el.querySelector('.a-ep-body');
      var chevron = el.querySelector('.a-ep-chevron');
      if (body) body.classList.add('open');
      if (chevron) chevron.classList.add('open');
    }
    setTimeout(function() { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 100);
  }
  openFromHash();
  window.addEventListener('hashchange', openFromHash);

  // 6. Expand all / collapse all
  window.toggleAllEndpoints = function(expand) {
    endpoints.forEach(function(ep) {
      var body = ep.querySelector('.a-ep-body');
      var chevron = ep.querySelector('.a-ep-chevron');
      if (body) { if (expand) body.classList.add('open'); else body.classList.remove('open'); }
      if (chevron) { if (expand) chevron.classList.add('open'); else chevron.classList.remove('open'); }
    });
  };

})();
