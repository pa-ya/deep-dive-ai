# Content authoring schema — Deep Dive AI

Each file registers ONE topic (rendered as one "framework" page):

```js
(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "math-linear-algebra",   // unique, url-safe (used in #hash + anchors)
  name: "Linear Algebra",       // page H1
  language: "Math Foundations", // shown as the little category chip in the hero
  group: "Math Foundations",    // nav grouping key — MUST be contiguous (see below)
  navLabel: "Linear Algebra",   // sidebar label (falls back to name)
  tagline: "one line, supports **bold** / `code` / $math$",
  color: "#6366F1",             // accent (nav dot + hero glow)
  readMinutes: 52,
  sections: [ /* see below */ ],
  packages: [ { name: "numpy", why: "…" } ],
  gotchas: [ "text with **bold**/`code`/$math$/[links](url)" ],
  flashcards: [ { q: "…", a: "… `code` / $math$ ok" } ],
  cheatsheet: [ { label: "SVD", code: "np.linalg.svd(A)" } ],
});
```

## Section

```js
{ id: "svd", title: "Singular Value Decomposition", level: "core", body: [ /* blocks */ ] }
```

`level`: `"core"` (visible) or `"deep"` (collapsed accordion — use for optional depth / references).

## Blocks

| type | fields |
|------|--------|
| `p` | `text` — supports `` `code` ``, `**bold**`, `*em*`, `[text](url)`, and inline `$math$` |
| `math` | `tex` — **display** equation (KaTeX). Optional `caption`. |
| `code` | `lang` (py/bash/json/text/…), `code` |
| `list` | `items: []`, optional `ordered: true` (items support inline formatting + `$math$`) |
| `callout` | `variant`: tip \| gotcha \| warn \| good \| note, `text` |
| `table` | `headers: []`, `rows: [[]]` — every row length MUST equal headers length; cells support inline `$math$` |
| `link` | `url`, `text` — renders as a 🔗 callout (use for "go deeper" references) |
| `heading` | `text` — a sub-heading inside a section |

## Math (IMPORTANT — this deck adds KaTeX)

- **Display math** → use a `math` block with a **`String.raw` template literal** so LaTeX backslashes stay literal:
  ```js
  { type: "math", tex: String.raw`\nabla_\theta J(\theta) = \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})\,x^{(i)}` }
  ```
  `String.raw` is mandatory: in a normal template literal `\frac` becomes a form-feed + "rac". Do NOT put a backtick or `${` inside a `String.raw` math block (math never needs them).

- **Inline math** inside `p` / `list` / `table` / `tagline` / flashcards → normal double-quoted strings, so **every LaTeX backslash is doubled**:
  ```js
  { type: "p", text: "The gradient $\\nabla_\\theta J$ points uphill; we step against it." }
  ```
  Inline `` `code` `` (single backticks) still works in the same string — no conflict.

## Escaping (the #1 source of load errors)

- `code` blocks are stored in template literals: escape a literal backtick `` ` `` → `` \` `` and a `${` → `\${`.
- In double-quoted `text`/`q`/`a` strings, a literal backslash must be doubled (`\\`). This is exactly what inline LaTeX needs.

## Nav grouping (MUST be contiguous)

`nav.js` folds *consecutive* items sharing the same `group` into one collapsible parent. All files of a group must be listed **back-to-back** in `index.html`. If a group's files are interleaved with another group's, the sidebar breaks. Group order in this deck:

`Math Foundations → Classical ML → Neural Networks → NLP → LLMs → Reinforcement Learning → MLOps → Practice`

## Validate every file after editing

```bash
node -e "global.window={FRAMEWORKS:[]};require('./content/<file>.js');console.log('ok', window.FRAMEWORKS.at(-1).sections.length,'sections')"
```
Then add a `<script src="content/<file>.js"></script>` line in `index.html`, keeping it inside its group block.
