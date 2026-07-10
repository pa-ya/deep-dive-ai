/* nav.js — sidebar build, framework switching, TOC, scroll-spy, shortcuts */
(function () {
  "use strict";
  let current = null;
  let scrollSpyObserver = null;

  function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }

  // Corner-arrow icon marking a nav item as a cross-link (shortcut) into another group.
  const SHORTCUT_SVG = '<svg class="shortcut-ico" viewBox="0 0 24 24" width="12" height="12" fill="none" aria-hidden="true"><path d="M7 17L17 7M17 7H9M17 7v8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  // Language groups get shortcut sub-items linking to the matching Solana sub-section,
  // so e.g. a Solana/C++ program is discoverable from the C++ group (where people look for it).
  const SHORTCUTS = {
    "C++":        [{ id: "solana-cpp", label: "Solana · C / C++" }],
    "Go":         [{ id: "solana-go", label: "Solana · Go" }],
    "TypeScript": [{ id: "solana-ts", label: "Solana · TypeScript" }],
    "Rust":       [{ id: "rust-anchor", label: "Solana · Anchor" },
                   { id: "solana-native", label: "Solana · native Rust" }],
  };

  function buildSidebar() {
    const nav = document.getElementById("nav");
    nav.innerHTML = "";
    const fws = window.FRAMEWORKS || [];
    const sample = fws.find((f) => f.id === "sample-projects");
    const main = fws.filter((f) => f.id !== "sample-projects");

    nav.appendChild(el("div", "nav-group-label", "Frameworks"));

    // Walk in order, folding consecutive same-group frameworks into a collapsible group.
    let i = 0;
    while (i < main.length) {
      const fw = main[i];
      if (fw.group) {
        const name = fw.group;
        const members = [];
        while (i < main.length && main[i].group === name) { members.push(main[i]); i++; }
        nav.appendChild(navGroup(name, members));
      } else {
        nav.appendChild(navItem(fw, i));
        i++;
      }
    }

    if (sample) {
      nav.appendChild(el("div", "nav-group-label", "Practice"));
      nav.appendChild(navItem(sample, main.length));
    }
  }

  function navItem(fw, i) {
    const item = el("div", "nav-item");
    item.dataset.fw = fw.id;
    item.style.animationDelay = (i * 0.03) + "s";
    item.innerHTML =
      `<span class="dot" style="background:${fw.color}"></span>` +
      `<span class="nav-name">${fw.name}</span>` +
      (fw.language ? `<span class="nav-lang">${fw.language}</span>` : "");
    item.addEventListener("click", () => { select(fw.id); closeMobileNav(); });
    return item;
  }

  // A collapsible parent (Go / Rust / C++) wrapping sub-items.
  function navGroup(name, members) {
    const wrap = el("div", "nav-group");
    wrap.dataset.group = name;
    const head = el("div", "nav-group-head");
    head.innerHTML =
      `<span class="dot" style="background:${members[0].color}"></span>` +
      `<span class="nav-name">${name}</span>` +
      `<span class="nav-group-count">${members.length}</span>` +
      '<svg class="chev" viewBox="0 0 24 24" width="15" height="15"><path d="M9 6l6 6-6 6" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
    head.addEventListener("click", () => wrap.classList.toggle("is-open"));
    wrap.appendChild(head);

    const body = el("div", "nav-group-body");
    const inner = el("div", "nav-group-inner");
    members.forEach((m) => inner.appendChild(navSubItem(m)));
    (SHORTCUTS[name] || []).forEach((sc) => {
      // only render a shortcut if its target exists and isn't already a real member of this group
      if (members.some((m) => m.id === sc.id)) return;
      if ((window.FRAMEWORKS || []).some((f) => f.id === sc.id)) inner.appendChild(navShortcut(sc));
    });
    body.appendChild(inner);
    wrap.appendChild(body);
    return wrap;
  }

  // A cross-link sub-item that jumps to a framework living in another group (e.g. Solana).
  function navShortcut(sc) {
    const fw = (window.FRAMEWORKS || []).find((f) => f.id === sc.id);
    const item = el("div", "nav-item is-sub is-shortcut");
    item.dataset.shortcut = sc.id;                 // NOT data-fw, so it stays out of active-state toggling
    item.title = "Jump to " + (fw ? fw.name : sc.id) + " in the Solana section";
    item.innerHTML =
      `<span class="dot" style="background:${fw ? fw.color : "#9945FF"}"></span>` +
      `<span class="nav-name">${sc.label}</span>` +
      SHORTCUT_SVG;
    item.addEventListener("click", () => { select(sc.id); closeMobileNav(); });
    return item;
  }

  function navSubItem(fw) {
    const item = el("div", "nav-item is-sub");
    item.dataset.fw = fw.id;
    const label = fw.navLabel || fw.name.replace(/^(Go|Rust|C\+\+)\s+/i, "");
    item.innerHTML =
      `<span class="dot" style="background:${fw.color}"></span>` +
      `<span class="nav-name">${label}</span>`;
    item.addEventListener("click", () => { select(fw.id); closeMobileNav(); });
    return item;
  }

  function select(id, push) {
    const fw = (window.FRAMEWORKS || []).find((f) => f.id === id);
    if (!fw) return;
    current = id;

    document.querySelectorAll(".nav-item").forEach((it) => it.classList.toggle("is-active", it.dataset.fw === id || it.dataset.shortcut === id));

    // keep the group containing the active item expanded
    const activeItem = document.querySelector('.nav-item[data-fw="' + id + '"]');
    if (activeItem) {
      const grp = activeItem.closest(".nav-group");
      if (grp) grp.classList.add("is-open");
      // scroll the sidebar (only) so the active item stays visible
      requestAnimationFrame(() => {
        const sidebar = document.getElementById("sidebar");
        if (!sidebar) return;
        const ir = activeItem.getBoundingClientRect();
        const sr = sidebar.getBoundingClientRect();
        if (ir.top < sr.top + 8 || ir.bottom > sr.bottom - 8) {
          sidebar.scrollTop += (ir.top - sr.top) - sr.height / 2 + ir.height / 2;
        }
      });
    }

    const content = document.getElementById("content");
    content.classList.add("switching");
    setTimeout(() => {
      const toc = window.Render.renderFramework(fw);
      buildTOC(toc);
      setupScrollSpy();
      content.classList.remove("switching");
      window.scrollTo({ top: 0, behavior: "auto" });
      window.UX && window.UX.refresh();
    }, 90);

    if (push !== false) {
      try { history.replaceState(null, "", "#" + id); } catch (e) {}
      try { localStorage.setItem("fwdeck.last", id); } catch (e) {}
    }
  }

  function buildTOC(entries) {
    const toc = document.getElementById("toc");
    toc.innerHTML = "";
    toc.appendChild(el("div", "toc-title", "On this page"));
    entries.forEach((en) => {
      const a = el("a", en.deep ? "deep-toc" : "", window.Render.inline(en.title));
      a.href = "#" + en.id;
      a.addEventListener("click", (e) => {
        e.preventDefault();
        const target = document.getElementById(en.id);
        if (target) {
          if (target.tagName === "DETAILS") target.open = true;
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
      toc.appendChild(a);
    });
  }

  function setupScrollSpy() {
    if (scrollSpyObserver) scrollSpyObserver.disconnect();
    const links = Array.from(document.querySelectorAll(".toc a"));
    const map = new Map();
    links.forEach((l) => map.set(l.getAttribute("href").slice(1), l));
    scrollSpyObserver = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          links.forEach((l) => l.classList.remove("is-active"));
          const active = map.get(en.target.id);
          if (active) active.classList.add("is-active");
        }
      });
    }, { rootMargin: "-80px 0px -70% 0px", threshold: 0 });
    document.querySelectorAll(".section, .card").forEach((s) => { if (s.id) scrollSpyObserver.observe(s); });
  }

  function next(dir) {
    const fws = (window.FRAMEWORKS || []);
    const idx = fws.findIndex((f) => f.id === current);
    let n = idx + dir;
    if (n < 0) n = fws.length - 1;
    if (n >= fws.length) n = 0;
    select(fws[n].id);
  }

  function closeMobileNav() { document.body.classList.remove("nav-open"); }
  function toggleMobileNav() { document.body.classList.toggle("nav-open"); }

  function initShortcuts() {
    document.addEventListener("keydown", (e) => {
      const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName);
      if (e.key === "/" && !typing) { e.preventDefault(); document.getElementById("searchInput").focus(); }
      else if (e.key === "Escape") { closeMobileNav(); window.Flashcards && window.Flashcards.close(); document.getElementById("searchResults").hidden = true; }
      if (typing) return;
      if (e.key === "j") next(1);
      else if (e.key === "k") next(-1);
      else if (e.key === "t") window.Theme.toggle();
      else if (e.key === "f") window.Flashcards && window.Flashcards.open(current);
    });
  }

  window.Nav = { buildSidebar, select, initShortcuts, toggleMobileNav, closeMobileNav, get current() { return current; } };
})();
