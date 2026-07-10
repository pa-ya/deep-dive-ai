/* theme.js — dark/light toggle, persisted, respects system on first load */
(function () {
  "use strict";
  const KEY = "fwdeck.theme";
  const root = document.documentElement;

  function apply(t) { root.setAttribute("data-theme", t); }

  function init() {
    let saved = null;
    try { saved = localStorage.getItem(KEY); } catch (e) {}
    if (!saved) {
      const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
      saved = prefersLight ? "light" : "dark";
    }
    apply(saved);

    const btn = document.getElementById("themeToggle");
    if (btn) btn.addEventListener("click", toggle);
  }

  function toggle() {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    apply(next);
    try { localStorage.setItem(KEY, next); } catch (e) {}
    window.Toast && window.Toast(next === "dark" ? "🌙 Dark theme" : "☀️ Light theme");
  }

  window.Theme = { toggle, init };
  init();
})();
