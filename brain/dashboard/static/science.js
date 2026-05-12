/* ═══════════════════════════════════════════════════════════════════════════
   JARVIS Oracle Edition — Scientific Reference (science.js)
   Sidebar nav, collapsible sections, LaTeX rendering, scroll spy
   ═══════════════════════════════════════════════════════════════════════════ */

(function() {
  'use strict';

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

  window.toggleSection = function(headerEl) {
    var body = headerEl.nextElementSibling;
    var chevron = headerEl.querySelector('.s-card-chevron');
    body.classList.toggle('open');
    if (chevron) chevron.classList.toggle('open');
  };

  window.expandAllSections = function(expand) {
    document.querySelectorAll('.s-card-body').forEach(function(b) {
      if (expand) b.classList.add('open'); else b.classList.remove('open');
    });
    document.querySelectorAll('.s-card-chevron').forEach(function(c) {
      if (expand) c.classList.add('open'); else c.classList.remove('open');
    });
  };

  function openFromHash() {
    var hash = location.hash.slice(1);
    if (!hash) return;
    var el = document.getElementById(hash);
    if (!el) return;
    var body = el.querySelector('.s-card-body');
    var chevron = el.querySelector('.s-card-chevron');
    if (body) body.classList.add('open');
    if (chevron) chevron.classList.add('open');
    setTimeout(function() { el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 100);
  }
  openFromHash();
  window.addEventListener('hashchange', openFromHash);

  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
      ],
      throwOnError: false,
    });
  }

})();
