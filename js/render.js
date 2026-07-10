/* ============================================================
   render.js — content object -> DOM
   ============================================================ */
(function () {
  "use strict";

  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  // inline formatting: $math$, `code`, **bold**, *em*, [text](url)
  function inline(str) {
    if (str == null) return "";
    let s = String(str);
    // Pull inline math ($...$) out BEFORE HTML-escaping & markdown, so LaTeX
    // (which contains <, >, &, backslashes, *, _) is handed to KaTeX untouched.
    const maths = [];
    s = s.replace(/\$([^$\n]+)\$/g, (_, tex) => {
      let html;
      try { html = window.katex ? window.katex.renderToString(tex, { throwOnError: false, displayMode: false }) : tex; }
      catch (e) { html = tex; }
      maths.push(html);
      return "M" + (maths.length - 1) + "";   // private-use sentinel
    });
    s = s
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    s = s.replace(/`([^`]+)`/g, '<code class="inline">$1</code>');
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // single-asterisk *emphasis* — guarded so spaced math ("N * M") isn't matched
    s = s.replace(/\*(?=\S)([^*\n]+?)(?<=\S)\*/g, "<em>$1</em>");
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,
      '<a class="inline-link" href="$2" target="_blank" rel="noopener">$1</a>');
    // splice the rendered math back in
    s = s.replace(/M(\d+)/g, (_, i) => maths[+i]);
    return s;
  }
  window.__inline = inline;

  function codeBlock(lang, code) {
    const wrap = el("div", "codeblock");
    const head = el("div", "code-head");
    head.appendChild(el("span", "code-lang", lang || "text"));
    const btn = el("button", "copy-btn",
      '<svg viewBox="0 0 24 24" width="13" height="13"><rect x="9" y="9" width="11" height="11" rx="2" stroke="currentColor" stroke-width="2" fill="none"/><path d="M5 15V5a2 2 0 0 1 2-2h10" stroke="currentColor" stroke-width="2" fill="none"/></svg><span>Copy</span>');
    btn.addEventListener("click", () => {
      navigator.clipboard.writeText(code).then(() => {
        btn.classList.add("copied");
        btn.querySelector("span").textContent = "Copied";
        setTimeout(() => { btn.classList.remove("copied"); btn.querySelector("span").textContent = "Copy"; }, 1400);
      });
    });
    head.appendChild(btn);
    const pre = el("pre");
    const codeEl = el("code");
    codeEl.innerHTML = window.Highlight(code, lang);
    pre.appendChild(codeEl);
    wrap.appendChild(head);
    wrap.appendChild(pre);
    return wrap;
  }

  const CALLOUT_ICON = { tip: "💡", gotcha: "⚠️", warn: "🛑", good: "✅", note: "📌" };

  function block(b) {
    switch (b.type) {
      case "p": return el("p", null, inline(b.text));
      case "code": return codeBlock(b.lang, b.code);
      case "list": {
        const ul = el(b.ordered ? "ol" : "ul");
        (b.items || []).forEach((it) => ul.appendChild(el("li", null, inline(it))));
        return ul;
      }
      case "callout": {
        const c = el("div", "callout " + (b.variant || "note"));
        c.appendChild(el("span", "callout-icon", CALLOUT_ICON[b.variant] || "📌"));
        c.appendChild(el("div", null, inline(b.text)));
        return c;
      }
      case "table": {
        const wrap = el("div", "tbl-wrap");
        const t = el("table", "data");
        const thead = el("thead");
        const htr = el("tr");
        (b.headers || []).forEach((h) => htr.appendChild(el("th", null, inline(h))));
        thead.appendChild(htr);
        const tbody = el("tbody");
        (b.rows || []).forEach((r) => {
          const tr = el("tr");
          r.forEach((cell) => tr.appendChild(el("td", null, inline(cell))));
          tbody.appendChild(tr);
        });
        t.appendChild(thead); t.appendChild(tbody);
        wrap.appendChild(t);
        return wrap;
      }
      case "link": {
        const c = el("div", "callout tip");
        c.appendChild(el("span", "callout-icon", "🔗"));
        c.appendChild(el("div", null, `<a class="inline-link" href="${b.url}" target="_blank" rel="noopener">${inline(b.text || b.url)}</a>`));
        return c;
      }
      case "heading": return el("h3", "sub-heading", inline(b.text));
      case "math": {
        const wrap = el("div", "mathblock");
        if (window.katex) {
          try { window.katex.render(b.tex, wrap, { displayMode: true, throwOnError: false }); }
          catch (e) { wrap.textContent = b.tex; }
        } else { wrap.textContent = b.tex; }
        if (b.caption) wrap.appendChild(el("div", "math-caption", inline(b.caption)));
        return wrap;
      }
      default: return el("p", null, inline(b.text || ""));
    }
  }

  function bodyBlocks(container, blocks) {
    (blocks || []).forEach((b) => container.appendChild(block(b)));
  }

  function sectionEl(fw, sec, index) {
    const isDeep = sec.level === "deep";
    const anchor = `${fw.id}--${sec.id}`;

    if (isDeep) {
      const det = el("details", "collapsible section");
      det.id = anchor;
      det.dataset.secId = sec.id;
      const sum = el("summary");
      sum.innerHTML =
        '<svg class="chev" viewBox="0 0 24 24" width="16" height="16"><path d="M9 6l6 6-6 6" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
        `<span>${inline(sec.title)}</span><span class="deep-badge">Deep dive</span>`;
      det.appendChild(sum);
      const body = el("div", "collapsible-body");
      bodyBlocks(body, sec.body);
      det.appendChild(body);
      return det;
    }

    const s = el("section", "section");
    s.id = anchor;
    s.dataset.secId = sec.id;
    const head = el("div", "section-head");
    head.appendChild(el("span", "section-num", String(index).padStart(2, "0")));
    head.appendChild(el("h2", null, inline(sec.title)));
    s.appendChild(head);
    bodyBlocks(s, sec.body);
    return s;
  }

  function hero(fw) {
    const h = el("div", "hero");
    h.style.setProperty("--hero-color", fw.color || "var(--accent)");
    h.style.setProperty("--hero-glow", `color-mix(in srgb, ${fw.color || "var(--accent)"} 22%, transparent)`);
    h.innerHTML = `
      <span class="hero-lang"><span class="dot" style="width:8px;height:8px;border-radius:50%;background:${fw.color}"></span>${fw.language}</span>
      <h1>${fw.name}</h1>
      <p class="hero-tagline">${inline(fw.tagline || "")}</p>
      <div class="hero-meta">
        <span class="meta-chip">⏱ <b>${fw.readMinutes || 15} min</b> read</span>
        <span class="meta-chip">📚 <b>${(fw.sections || []).length}</b> sections</span>
        ${(fw.flashcards || []).length ? `<span class="meta-chip">🃏 <b>${fw.flashcards.length}</b> flashcards</span>` : ""}
      </div>
      <div class="hero-actions"></div>`;
    const actions = h.querySelector(".hero-actions");
    if ((fw.flashcards || []).length) {
      const fcBtn = el("button", "btn btn-primary", "🃏 Study flashcards");
      fcBtn.addEventListener("click", () => window.Flashcards && window.Flashcards.open(fw.id));
      actions.appendChild(fcBtn);
    }
    return h;
  }

  function packagesCard(fw) {
    if (!fw.packages || !fw.packages.length) return null;
    const card = el("div", "card");
    card.appendChild(el("div", "card-title", "📦 Most-used packages"));
    const grid = el("div", "pkg-grid");
    fw.packages.forEach((p) => {
      const pk = el("div", "pkg");
      pk.appendChild(el("span", "pkg-name", p.name));
      pk.appendChild(el("span", "pkg-why", inline(p.why || "")));
      grid.appendChild(pk);
    });
    card.appendChild(grid);
    return card;
  }

  function cheatCard(fw) {
    if (!fw.cheatsheet || !fw.cheatsheet.length) return null;
    const card = el("div", "card");
    card.appendChild(el("div", "card-title", "⚡ Cheat card"));
    const grid = el("div", "cheat-grid");
    fw.cheatsheet.forEach((c) => {
      const ch = el("div", "cheat");
      ch.appendChild(el("span", "cheat-label", c.label));
      ch.appendChild(el("span", "cheat-code", c.code));
      grid.appendChild(ch);
    });
    card.appendChild(grid);
    return card;
  }

  function gotchaCard(fw) {
    if (!fw.gotchas || !fw.gotchas.length) return null;
    const card = el("div", "card");
    card.appendChild(el("div", "card-title", "🕳 Gotchas & hard points"));
    const ul = el("ul", "gotcha-list");
    fw.gotchas.forEach((g) => ul.appendChild(el("li", null, `<div>${inline(g)}</div>`)));
    card.appendChild(ul);
    return card;
  }

  // Prev / Next framework navigation (linear, wraps around — matches j/k order)
  function fwNavFooter(fw) {
    const list = window.FRAMEWORKS || [];
    const idx = list.findIndex((f) => f.id === fw.id);
    if (idx < 0 || list.length < 2) return null;
    const prev = list[(idx - 1 + list.length) % list.length];
    const next = list[(idx + 1) % list.length];
    const nav = el("nav", "fw-nav");
    nav.setAttribute("aria-label", "Framework navigation");

    function card(target, dir) {
      const c = el("button", "fw-nav-card " + dir);
      c.style.setProperty("--fw-color", target.color || "var(--accent)");
      const arrow = `<span class="fw-nav-arrow">${dir === "prev" ? "←" : "→"}</span>`;
      const meta =
        `<span>` +
        `<span class="fw-nav-dir">${dir === "prev" ? "Previous" : "Next"}</span><br>` +
        `<span class="fw-nav-name"><span class="dot"></span>${target.name}</span>` +
        `</span>`;
      c.innerHTML = dir === "prev" ? arrow + meta : meta + arrow;
      c.addEventListener("click", () => window.Nav && window.Nav.select(target.id));
      return c;
    }
    nav.appendChild(card(prev, "prev"));
    nav.appendChild(card(next, "next"));
    return nav;
  }

  // Render a whole framework into #content, return list of TOC entries
  function renderFramework(fw) {
    const root = document.getElementById("content");
    root.innerHTML = "";
    root.appendChild(hero(fw));

    const toc = [];
    let n = 0;
    (fw.sections || []).forEach((sec) => {
      n++;
      root.appendChild(sectionEl(fw, sec, n));
      toc.push({ id: `${fw.id}--${sec.id}`, title: sec.title, deep: sec.level === "deep" });
    });

    const pk = packagesCard(fw); if (pk) { pk.id = `${fw.id}--packages`; root.appendChild(pk); toc.push({ id: pk.id, title: "Packages" }); }
    const ch = cheatCard(fw);    if (ch) { ch.id = `${fw.id}--cheat`; root.appendChild(ch); toc.push({ id: ch.id, title: "Cheat card" }); }
    const gc = gotchaCard(fw);   if (gc) { gc.id = `${fw.id}--gotchas`; root.appendChild(gc); toc.push({ id: gc.id, title: "Gotchas" }); }

    const fn = fwNavFooter(fw);  if (fn) root.appendChild(fn);

    return toc;
  }

  window.Render = { renderFramework, block, inline };
})();
