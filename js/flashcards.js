/* flashcards.js — flashcard deck + quiz mode modal */
(function () {
  "use strict";
  let fw = null, mode = "cards", cardIdx = 0;
  let quiz = null;

  const modal = () => document.getElementById("flashModal");

  function open(fwId) {
    fw = (window.FRAMEWORKS || []).find((f) => f.id === fwId);
    if (!fw || !(fw.flashcards || []).length) { window.Toast && window.Toast("No flashcards for this section"); return; }
    document.getElementById("flashTitle").textContent = fw.name;
    modal().hidden = false;
    setMode("cards");
  }
  function close() { modal().hidden = true; quiz = null; }

  function setMode(m) {
    mode = m; cardIdx = 0;
    document.querySelectorAll(".modal-tab").forEach((t) => t.classList.toggle("is-active", t.dataset.mode === m));
    if (m === "cards") renderCard(); else startQuiz();
  }

  /* -------- flashcards -------- */
  function renderCard() {
    const cards = fw.flashcards;
    const c = cards[cardIdx];
    const body = document.getElementById("flashBody");
    body.innerHTML = "";
    const card = document.createElement("div");
    card.className = "flashcard";
    card.innerHTML =
      '<div class="flashcard-inner">' +
        `<div class="flashcard-face front"><span class="fc-tag">Question</span>${window.Render.inline(c.q)}<span class="fc-hint">click to flip</span></div>` +
        `<div class="flashcard-face back"><span class="fc-tag">Answer</span><div>${window.Render.inline(c.a)}</div></div>` +
      "</div>";
    card.addEventListener("click", () => card.classList.toggle("flipped"));
    body.appendChild(card);

    const foot = document.getElementById("flashFoot");
    foot.innerHTML = "";
    const prev = mkBtn("← Prev", () => { cardIdx = (cardIdx - 1 + cards.length) % cards.length; renderCard(); });
    const count = document.createElement("span");
    count.className = "quiz-progress";
    count.textContent = `${cardIdx + 1} / ${cards.length}`;
    const nextB = mkBtn("Next →", () => { cardIdx = (cardIdx + 1) % cards.length; renderCard(); }, true);
    foot.append(prev, count, nextB);
  }

  /* -------- quiz -------- */
  function buildQuiz() {
    const cards = fw.flashcards.slice();
    if (cards.length < 4) return null;
    // shuffle deterministically-ish using index rotation (no Math.random per env limits)
    const shuffled = cards.map((c, i) => ({ c, k: (i * 7 + 3) % cards.length })).sort((a, b) => a.k - b.k).map((x) => x.c);
    const qs = shuffled.slice(0, Math.min(6, shuffled.length)).map((card, qi) => {
      const distractors = cards.filter((c) => c.a !== card.a).map((c) => c.a);
      const picks = [];
      for (let i = 0; i < 3 && i < distractors.length; i++) picks.push(distractors[(qi * 3 + i) % distractors.length]);
      const opts = [card.a, ...picks];
      // rotate options so answer isn't always first
      const rot = qi % opts.length;
      const rotated = opts.slice(rot).concat(opts.slice(0, rot));
      return { q: card.q, answer: card.a, options: Array.from(new Set(rotated)) };
    });
    return { qs, idx: 0, score: 0, answered: false };
  }

  function startQuiz() {
    quiz = buildQuiz();
    const body = document.getElementById("flashBody");
    if (!quiz) { body.innerHTML = '<div class="sr-empty">Need at least 4 flashcards for quiz mode.</div>'; document.getElementById("flashFoot").innerHTML = ""; return; }
    renderQuiz();
  }

  function renderQuiz() {
    const body = document.getElementById("flashBody");
    const foot = document.getElementById("flashFoot");
    if (quiz.idx >= quiz.qs.length) {
      const pct = Math.round((quiz.score / quiz.qs.length) * 100);
      body.innerHTML = `<div class="quiz-score"><div class="big">${pct}%</div><p>You got <b>${quiz.score}</b> of <b>${quiz.qs.length}</b> right.</p></div>`;
      foot.innerHTML = "";
      foot.append(mkBtn("Retry", () => setMode("quiz")), mkBtn("Flashcards", () => setMode("cards"), true));
      return;
    }
    const q = quiz.qs[quiz.idx];
    body.innerHTML = `<div class="quiz-progress" style="margin-bottom:12px">Question ${quiz.idx + 1} of ${quiz.qs.length}</div><div class="quiz-q">${window.Render.inline(q.q)}</div>`;
    const opts = document.createElement("div");
    opts.className = "quiz-opts";
    q.options.forEach((opt) => {
      const b = document.createElement("button");
      b.className = "quiz-opt";
      b.innerHTML = window.Render.inline(opt);
      b.addEventListener("click", () => answer(b, opt, q, opts));
      opts.appendChild(b);
    });
    body.appendChild(opts);
    foot.innerHTML = "";
    const hint = document.createElement("span");
    hint.className = "quiz-progress";
    hint.textContent = "Pick an answer";
    foot.append(hint);
  }

  function answer(btn, opt, q, opts) {
    if (quiz.answered) return;
    quiz.answered = true;
    const correct = opt === q.answer;
    if (correct) quiz.score++;
    Array.from(opts.children).forEach((b) => {
      b.disabled = true;
      if (b.innerHTML === window.Render.inline(q.answer)) b.classList.add("correct");
    });
    if (!correct) btn.classList.add("wrong");
    const foot = document.getElementById("flashFoot");
    foot.innerHTML = "";
    const label = document.createElement("span");
    label.className = "quiz-progress";
    label.innerHTML = correct ? "✅ Correct" : "❌ Not quite";
    foot.append(label, mkBtn("Next →", () => { quiz.idx++; quiz.answered = false; renderQuiz(); }, true));
  }

  function mkBtn(label, fn, primary) {
    const b = document.createElement("button");
    b.className = "btn" + (primary ? " btn-primary" : "");
    b.textContent = label;
    b.addEventListener("click", fn);
    return b;
  }

  function init() {
    document.getElementById("flashBtn").addEventListener("click", () => open(window.Nav.current));
    modal().querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", close));
    modal().querySelectorAll(".modal-tab").forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));
  }

  window.Flashcards = { open, close, init };
})();
