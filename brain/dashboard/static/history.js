/* ═══════════════════════════════════════════════════════════════════════════
   JARVIS Build History — history.js
   Collapsible entries, sidebar scroll spy, deep-link handling.
   ═══════════════════════════════════════════════════════════════════════════ */

(function() {
  'use strict';

  // ═══════════════════════════════════════════════════════════════════════════
  // 1. Entry toggle
  // ═══════════════════════════════════════════════════════════════════════════

  window.toggleEntry = function(headerEl) {
    var entry = headerEl.closest('.h-entry');
    if (!entry) return;

    var details = entry.querySelector('.h-details');
    if (!details) return;

    var isOpen = entry.classList.contains('open');
    if (isOpen) {
      details.style.display = 'none';
      entry.classList.remove('open');
    } else {
      details.style.display = '';
      entry.classList.add('open');
    }
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // 2. Sidebar scroll spy
  // ═══════════════════════════════════════════════════════════════════════════

  var navLinks = document.querySelectorAll('.d-nav-link[href^="#"]');
  var sections = [];
  navLinks.forEach(function(link) {
    var id = link.getAttribute('href').slice(1);
    var sec = document.getElementById(id);
    if (sec) sections.push({ el: sec, link: link });
  });

  function updateActiveNav() {
    var scrollY = window.scrollY + 140;
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
  // 3. Deep-link: auto-open entry on hash navigation
  // ═══════════════════════════════════════════════════════════════════════════

  function openFromHash() {
    var hash = window.location.hash;
    if (!hash || hash.length < 2) return;
    var target = document.getElementById(hash.slice(1));
    if (!target || !target.classList.contains('h-entry')) return;

    var details = target.querySelector('.h-details');
    if (details) {
      details.style.display = '';
      target.classList.add('open');
    }

    setTimeout(function() {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }

  openFromHash();
  window.addEventListener('hashchange', openFromHash);

  // ═══════════════════════════════════════════════════════════════════════════
  // 4. Expand/Collapse All toggle
  // ═══════════════════════════════════════════════════════════════════════════

  window.toggleAll = function(expand) {
    var entries = document.querySelectorAll('.h-entry');
    entries.forEach(function(entry) {
      var details = entry.querySelector('.h-details');
      if (!details) return;
      if (expand) {
        details.style.display = '';
        entry.classList.add('open');
      } else {
        details.style.display = 'none';
        entry.classList.remove('open');
      }
    });
  };

})();
