/* ux.js — reading progress, back-to-top, keyboard-shortcut overlay.
   Small quality-of-life layer; self-inits on load (DOM chrome is static). */
(function () {
  "use strict";

  /* ---------- Reading progress + back-to-top ---------- */
  const bar = document.querySelector("#readingProgress span");
  const btt = document.getElementById("backToTop");
  let ticking = false;

  function onScroll() {
    const doc = document.documentElement;
    const max = doc.scrollHeight - window.innerHeight;
    const y = window.scrollY || doc.scrollTop || 0;
    const pct = max > 0 ? Math.min(100, (y / max) * 100) : 0;
    if (bar) bar.style.width = pct.toFixed(1) + "%";
    if (btt) btt.classList.toggle("show", y > 600);
    ticking = false;
  }
  function requestScroll() {
    if (!ticking) { ticking = true; requestAnimationFrame(onScroll); }
  }
  window.addEventListener("scroll", requestScroll, { passive: true });
  window.addEventListener("resize", requestScroll, { passive: true });

  function toTop() { window.scrollTo({ top: 0, behavior: "smooth" }); }
  if (btt) btt.addEventListener("click", toTop);

  // recompute after a framework switch (content height changes)
  window.UX = { refresh: requestScroll };

  /* ---------- Keyboard-shortcut overlay ---------- */
  const help = document.getElementById("helpModal");
  const helpBtn = document.getElementById("helpToggle");
  function openHelp() { if (help) help.hidden = false; }
  function closeHelp() { if (help) help.hidden = true; }
  function toggleHelp() { if (help) help.hidden = !help.hidden; }
  if (helpBtn) helpBtn.addEventListener("click", toggleHelp);
  if (help) help.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeHelp));

  /* ---------- Extra keys: ? (help), gg (top) ---------- */
  let lastG = 0;
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName);
    if (e.key === "Escape") { closeHelp(); return; }
    if (typing) return;
    if (e.key === "?") { e.preventDefault(); toggleHelp(); }
    else if (e.key === "g") {
      const now = e.timeStamp || 0;
      if (now - lastG < 450) { toTop(); lastG = 0; } else { lastG = now; }
    }
  });

  /* ---------- Copy deep link from a section number ---------- */
  document.addEventListener("click", (e) => {
    const num = e.target.closest(".section-num");
    if (!num) return;
    const sec = num.closest(".section");
    if (!sec || !sec.id || !navigator.clipboard) return;
    const url = location.href.split("#")[0] + "#" + sec.id;
    navigator.clipboard.writeText(url).then(() => {
      num.classList.add("copied");
      window.Toast && window.Toast("🔗 Section link copied");
      setTimeout(() => num.classList.remove("copied"), 1200);
    });
  });

  onScroll();
})();
