/* app.js — bootstrap after all content scripts have registered */
(function () {
  "use strict";

  // Toast helper
  let toastTimer = null;
  window.Toast = function (msg) {
    const t = document.getElementById("toast");
    t.textContent = msg; t.hidden = false;
    requestAnimationFrame(() => t.classList.add("show"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      t.classList.remove("show");
      setTimeout(() => (t.hidden = true), 260);
    }, 1600);
  };

  // Compact "at a glance" line in the sidebar footer.
  function renderDeckStats(fws) {
    const box = document.getElementById("deckStats");
    if (!box) return;
    const frameworks = fws.filter((f) => f.id !== "sample-projects");
    const sections = fws.reduce((n, f) => n + (f.sections || []).length, 0);
    const minutes = fws.reduce((n, f) => n + (f.readMinutes || 0), 0);
    const hrs = Math.floor(minutes / 60), mins = minutes % 60;
    const time = hrs ? (hrs + "h " + (mins ? mins + "m" : "")).trim() : mins + "m";
    box.innerHTML = "<b>" + frameworks.length + "</b> frameworks &middot; <b>" + sections + "</b> sections &middot; <b>~" + time + "</b> to review";
  }

  function boot() {
    const fws = window.FRAMEWORKS || [];
    if (!fws.length) {
      document.getElementById("content").innerHTML =
        '<div class="sr-empty" style="padding:60px">No content loaded. Ensure content/*.js files are present.</div>';
      return;
    }

    window.Nav.buildSidebar();
    renderDeckStats(fws);
    window.Search.init();
    window.Flashcards.init();
    window.Nav.initShortcuts();

    // wire chrome
    document.getElementById("menuToggle").addEventListener("click", window.Nav.toggleMobileNav);
    document.getElementById("scrim").addEventListener("click", window.Nav.closeMobileNav);

    // pick initial framework: hash > last visited > first.
    // hash may be a framework id ("#fastapi") or a deep section anchor ("#fastapi--routing").
    const raw = (location.hash || "").slice(1);
    let sectionAnchor = null;
    let start = raw;
    if (raw.indexOf("--") !== -1) { sectionAnchor = raw; start = raw.split("--")[0]; }
    if (!start || !fws.find((f) => f.id === start)) {
      sectionAnchor = null;
      try { start = localStorage.getItem("fwdeck.last"); } catch (e) {}
    }
    if (!start || !fws.find((f) => f.id === start)) start = fws[0].id;

    window.Nav.select(start, false);

    // scroll to a deep-linked section once it has rendered
    if (sectionAnchor) {
      setTimeout(() => {
        const target = document.getElementById(sectionAnchor);
        if (target) {
          if (target.tagName === "DETAILS") target.open = true;
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 160);
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
