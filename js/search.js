/* search.js — global search across all frameworks/sections */
(function () {
  "use strict";
  let index = [];

  function textOf(sec) {
    let t = sec.title + " ";
    (sec.body || []).forEach((b) => {
      if (b.text) t += b.text + " ";
      if (b.code) t += b.code + " ";
      if (b.items) t += b.items.join(" ") + " ";
      if (b.rows) t += b.rows.flat().join(" ") + " ";
    });
    return t;
  }

  function build() {
    index = [];
    (window.FRAMEWORKS || []).forEach((fw) => {
      (fw.sections || []).forEach((sec) => {
        index.push({ fw: fw.id, fwName: fw.name, color: fw.color, secId: sec.id, title: sec.title, text: textOf(sec).toLowerCase() });
      });
      (fw.gotchas || []).forEach((g, i) => {
        index.push({ fw: fw.id, fwName: fw.name, color: fw.color, secId: "gotchas", title: "Gotcha", text: g.toLowerCase() });
      });
      (fw.packages || []).forEach((p) => {
        index.push({ fw: fw.id, fwName: fw.name, color: fw.color, secId: "packages", title: "📦 " + p.name, text: ((p.name || "") + " " + (p.why || "")).toLowerCase() });
      });
      (fw.cheatsheet || []).forEach((c) => {
        index.push({ fw: fw.id, fwName: fw.name, color: fw.color, secId: "cheat", title: "⚡ " + (c.label || "Cheat"), text: ((c.label || "") + " " + (c.code || "")).toLowerCase() });
      });
    });
  }

  function esc(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

  function search(q) {
    q = q.trim().toLowerCase();
    if (q.length < 2) return [];
    const terms = q.split(/\s+/);
    const scored = [];
    index.forEach((it) => {
      let score = 0;
      terms.forEach((term) => {
        if (it.title.toLowerCase().includes(term)) score += 5;
        if (it.fwName.toLowerCase().includes(term)) score += 3;
        const idx = it.text.indexOf(term);
        if (idx >= 0) score += 1;
      });
      if (score > 0) {
        const first = it.text.indexOf(terms[0]);
        let snip = it.text.slice(Math.max(0, first - 30), first + 60).trim();
        scored.push({ it, score, snip });
      }
    });
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, 12);
  }

  function highlightTerm(s, q) {
    const term = q.trim().split(/\s+/)[0];
    if (!term) return s;
    try { return s.replace(new RegExp("(" + esc(term) + ")", "ig"), "<mark>$1</mark>"); }
    catch (e) { return s; }
  }

  function render(results, q) {
    const box = document.getElementById("searchResults");
    box.innerHTML = "";
    if (!results.length) {
      box.hidden = false;
      box.appendChild(Object.assign(document.createElement("div"), { className: "sr-empty", textContent: "No matches" }));
      return;
    }
    results.forEach((r, i) => {
      const a = document.createElement("a");
      a.className = "sr-item" + (i === 0 ? " is-active" : "");
      a.innerHTML =
        `<div class="sr-fw" style="color:${r.it.color}">${r.it.fwName}</div>` +
        `<div class="sr-title">${window.Render.inline(r.it.title)}</div>` +
        `<div class="sr-snippet">${highlightTerm(r.snip, q)}…</div>`;
      a.addEventListener("click", (e) => { e.preventDefault(); go(r.it); });
      box.appendChild(a);
    });
    box.hidden = false;
  }

  function go(it) {
    window.Nav.select(it.fw);
    document.getElementById("searchResults").hidden = true;
    document.getElementById("searchInput").value = "";
    setTimeout(() => {
      const target = document.getElementById(it.fw + "--" + it.secId);
      if (target) { if (target.tagName === "DETAILS") target.open = true; target.scrollIntoView({ behavior: "smooth", block: "start" }); }
    }, 200);
  }

  function init() {
    build();
    const input = document.getElementById("searchInput");
    const box = document.getElementById("searchResults");
    let curResults = [];
    let activeIdx = 0;

    input.addEventListener("input", () => {
      const q = input.value;
      if (q.trim().length < 2) { box.hidden = true; return; }
      curResults = search(q); activeIdx = 0;
      render(curResults, q);
    });
    input.addEventListener("keydown", (e) => {
      if (box.hidden) return;
      const items = Array.from(box.querySelectorAll(".sr-item"));
      if (e.key === "ArrowDown") { e.preventDefault(); activeIdx = Math.min(activeIdx + 1, items.length - 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); activeIdx = Math.max(activeIdx - 1, 0); }
      else if (e.key === "Enter") { e.preventDefault(); if (curResults[activeIdx]) go(curResults[activeIdx].it); return; }
      else return;
      items.forEach((it, i) => it.classList.toggle("is-active", i === activeIdx));
      items[activeIdx] && items[activeIdx].scrollIntoView({ block: "nearest" });
    });
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".search-wrap")) box.hidden = true;
    });
  }

  window.Search = { init, build };
})();
