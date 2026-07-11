(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "llm-rag",
  name: "RAG, Embeddings & Vector Search",
  language: "Large Language Models",
  group: "Large Language Models",
  navLabel: "RAG & Embeddings",
  tagline: "Give a frozen LLM an open book: retrieve the right documents, stuff them into the prompt, and generate grounded, cited answers — built from cosine similarity up to production reranking.",
  color: "#BE185D",
  readMinutes: 50,
  sections: [
    {
      id: "why",
      title: "Why RAG: giving a frozen model an open book",
      level: "core",
      body: [
        { type: "p", text: "A trained LLM is a **compressed, frozen snapshot** of its training data. Two facts follow, and both are business-critical. First, its knowledge stops at a **cutoff date** — ask about last week's earnings call or your company's internal wiki and it simply cannot know. Second, when it doesn't know, it does not say so; it **hallucinates** a fluent, confident, wrong answer, because next-token prediction has no built-in notion of 'I am uncertain about this fact.'" },
        { type: "p", text: "**Retrieval-Augmented Generation (RAG)** is the dominant fix. The idea is almost embarrassingly simple: before you ask the model a question, *look up* relevant documents from an external store and paste them into the prompt. The model then answers from the text in front of it — an **open-book exam** instead of a closed-book one. It was introduced by [Lewis et al. (2020)](https://arxiv.org/abs/2005.11401) at Facebook AI, originally as a jointly-trained retriever + generator; the modern 'frozen LLM + vector search' recipe is the pragmatic descendant that took over the industry from 2023 onward." },
        { type: "table",
          headers: ["Problem with a bare LLM", "How RAG fixes it"],
          rows: [
            ["Knowledge frozen at training cutoff", "Retrieve fresh docs at query time — update the store, not the weights"],
            ["Hallucinates facts it never saw", "Grounds the answer in retrieved text you control"],
            ["No access to private/internal data", "Index your own docs; nothing leaves your store until retrieval"],
            ["Can't cite where an answer came from", "Each retrieved chunk carries a source — attach citations"],
            ["Expensive to re-train on new facts", "Retrieval is a database write, not a training run"],
          ]
        },
        { type: "heading", text: "RAG vs fine-tuning — the question everyone asks first" },
        { type: "p", text: "These are **not competitors**; they change different things. Fine-tuning adjusts the model's **weights** to change its *behaviour, style, or skill* — how it formats output, what tone it uses, how it reasons about a domain. RAG changes the model's **context** to give it *knowledge* it can read at inference time. A useful slogan: **fine-tuning teaches the model a skill; RAG gives it a fact.**" },
        { type: "table",
          headers: ["Need", "Reach for", "Why"],
          rows: [
            ["Answer from a large, changing knowledge base", "**RAG**", "Facts churn; you don't want to re-train weekly"],
            ["Cite exact sources / audit answers", "**RAG**", "Retrieved chunks are traceable evidence"],
            ["Enforce a house style / output format", "**Fine-tune**", "Behaviour lives in the weights"],
            ["Teach a narrow reasoning skill or jargon", "**Fine-tune**", "Pattern, not lookupable fact"],
            ["Private data, strict freshness, low re-train budget", "**RAG**", "Update the index, not the model"],
            ["Both facts *and* behaviour", "**Both**", "Fine-tune the style, RAG the knowledge — common in production"],
          ]
        },
        { type: "callout", variant: "note", text: "**Where this sits in the stack.** RAG is built on top of everything earlier in the deck: the retriever is powered by **embeddings** (NLP Foundations, 'from tokens to meaning') and **cosine similarity** (Linear Algebra, the dot product), the generator is a **Transformer** LLM (NLP → Transformers), and the whole thing is judged with the **evaluation** mindset from Classical ML. RAG is where all of it finally clicks together into a product." },
        { type: "callout", variant: "tip", text: "**RAG is the highest-leverage applied-LLM pattern.** Most 'AI features' shipped in real companies — chat-with-your-docs, support-ticket assistants, internal search, coding copilots over a private repo — are RAG systems with a nice UI. Master this section and you can build the thing people actually pay for." },
      ]
    },

    {
      id: "embeddings",
      title: "Embeddings: turning meaning into geometry",
      level: "core",
      body: [
        { type: "p", text: "Retrieval needs a way to measure *'is this document about the same thing as the question?'* — and keyword overlap isn't enough (a query for 'how do I reset my password' should match a doc titled 'account recovery' with zero shared keywords). The trick is to map text to **vectors** such that *semantically similar text lands nearby in space*. Those vectors are **embeddings**." },
        { type: "p", text: "You met word embeddings (word2vec, GloVe) in **NLP Foundations**. RAG uses their grown-up cousin: **sentence / passage embeddings**, where an entire chunk of text becomes one dense vector, typically of dimension $d \\in \\{384, 768, 1024, 3072\\}$. A good embedding model places 'how do I reset my password' and 'steps for account recovery' close together even though they share no words." },
        { type: "math", tex: String.raw`\text{embed}: \text{text} \longmapsto \mathbf{v} \in \mathbb{R}^{d}, \qquad \text{similar meaning} \Rightarrow \text{small angle between vectors}` },
        { type: "heading", text: "Cosine similarity — the ruler of semantic space" },
        { type: "p", text: "We compare two embeddings with **cosine similarity**: the cosine of the angle between them, which is exactly the normalized dot product you derived in Linear Algebra. It ignores magnitude and measures pure direction — *alignment of meaning*." },
        { type: "math", tex: String.raw`\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q}^\top \mathbf{d}}{\|\mathbf{q}\|\,\|\mathbf{d}\|} \;\in\; [-1, 1]` },
        { type: "p", text: "In practice you **L2-normalize** every embedding once (divide by its norm), after which cosine similarity is just a dot product $\\mathbf{q}^\\top\\mathbf{d}$ — one `@` in NumPy — and ranking by cosine becomes ranking by dot product. This is why vector databases store normalized vectors: the whole search reduces to fast matrix multiplication." },
        { type: "callout", variant: "gotcha", text: "**Cosine vs Euclidean vs dot product.** For *normalized* vectors, ranking by cosine similarity, by (negated) Euclidean distance, and by dot product all give the **same order** — because $\\|q-d\\|^2 = 2 - 2\\,q^\\top d$ when both are unit-length. For *un-normalized* vectors they differ, and a dot product then rewards longer vectors. Pick a metric, normalize consistently, and stick to it; mixing them silently is a classic RAG bug." },
        { type: "heading", text: "Choosing an embedding model (2025–2026)" },
        { type: "p", text: "The public scoreboard is the **[MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)** (Massive Text Embedding Benchmark) on HuggingFace — thousands of models scored on retrieval, clustering, reranking, and more. The landscape as of 2026:" },
        { type: "table",
          headers: ["Model / family", "Who", "Notes"],
          rows: [
            ["**text-embedding-3-small / -large**", "OpenAI", "The safe API default; `-large` is 3072-dim, supports dimension shortening"],
            ["**E5** (`e5-large-v2`, `multilingual-e5`)", "Microsoft", "Strong open weights; needs `query:` / `passage:` prefixes"],
            ["**BGE** (`bge-large`, `bge-m3`)", "BAAI", "Open-weight production favourite; BGE-M3 does dense+sparse+multi-vec"],
            ["**GTE / Qwen3-Embedding**", "Alibaba", "Top of recent MTEB multilingual boards, open weights"],
            ["**Voyage / Cohere embed-v4**", "Voyage, Cohere", "High-quality commercial APIs, strong for retrieval"],
            ["**all-MiniLM-L6-v2**", "sentence-transformers", "Tiny (384-dim), fast, the classic 'good enough & free' starter"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Two rules that save you.** (1) *Embed queries and documents with the exact same model* — mixing models puts them in incompatible spaces and retrieval returns noise. (2) *Respect the model's instructions*: E5/BGE want you to prefix a search query with `query:` and a document with `passage:` (or a task instruction). Skipping the prefix quietly tanks recall by 5–15 points." },
        { type: "code", lang: "py", code: "from sentence_transformers import SentenceTransformer\nimport numpy as np\n\nmodel = SentenceTransformer(\"BAAI/bge-small-en-v1.5\")  # 384-dim, fast, free\n\ndocs = [\n    \"To reset your password, open Settings and click 'Account recovery'.\",\n    \"Our office is open Monday to Friday, 9am to 5pm.\",\n    \"You can change your billing plan under the Subscription tab.\",\n]\nquery = \"I forgot my login and can't get in\"\n\n# normalize_embeddings=True -> cosine similarity == dot product\nD = model.encode(docs, normalize_embeddings=True)          # (3, 384)\nq = model.encode(query, normalize_embeddings=True)          # (384,)\n\nscores = D @ q                                              # cosine sims\nbest = int(np.argmax(scores))\nprint(round(float(scores[best]), 3), \"->\", docs[best])\n# ~0.62 -> the password-reset sentence, despite zero shared keywords." },
        { type: "callout", variant: "note", text: "**Why embeddings even work** is the *distributional hypothesis* from NLP Foundations: words (and sentences) that appear in similar contexts get similar vectors. Sentence-transformers push this further by *fine-tuning* an encoder with a contrastive loss on pairs of (question, correct-passage), explicitly pulling matching pairs together and pushing mismatches apart. That training objective is exactly the retrieval task, which is why they retrieve so well." },
      ]
    },

    {
      id: "pipeline",
      title: "The RAG pipeline, stage by stage",
      level: "core",
      body: [
        { type: "p", text: "Every RAG system is two phases. **Indexing** happens once (or on a schedule) and is offline: turn your documents into a searchable vector index. **Querying** happens per request and is online: find relevant chunks and let the LLM answer from them. Here is the whole flow, and we'll take each box in turn." },
        { type: "code", lang: "text", code: "INDEXING (offline, batch)\n  raw docs ─► load ─► chunk ─► embed ─► store in vector index\n  (PDFs,      (parse    (split    (each    (FAISS / Chroma /\n   HTML,       text)     into      chunk    Qdrant / pgvector)\n   md, db)               pieces)   -> vec)\n\nQUERYING (online, per request)\n  user question\n    ─► embed the question           (same model as indexing!)\n    ─► retrieve top-k similar chunks (vector search)\n    ─► [optional] rerank / filter\n    ─► build augmented prompt        (question + retrieved chunks)\n    ─► LLM generates grounded answer + citations" },
        { type: "heading", text: "1. Ingest / load" },
        { type: "p", text: "Pull text out of wherever it lives — PDFs, HTML, Markdown, Confluence, Notion, a database, source code. The unglamorous truth: **most RAG quality problems are born here.** A PDF table parsed into scrambled text, headers glued to body copy, or dropped code blocks all poison everything downstream. Garbage in, confidently-cited garbage out." },
        { type: "heading", text: "2. Chunk" },
        { type: "p", text: "Documents are too long to embed whole (and you want to retrieve the *relevant paragraph*, not a 40-page manual), so split them into **chunks** of a few hundred tokens. This is subtle enough to get its own section next." },
        { type: "heading", text: "3. Embed" },
        { type: "p", text: "Run every chunk through the embedding model to get one vector per chunk (previous section). Batch this — encoding 100k chunks one at a time is painfully slow; `model.encode(chunks, batch_size=64)` is the difference between minutes and hours." },
        { type: "heading", text: "4. Index / store" },
        { type: "p", text: "Put `(vector, chunk_text, metadata)` into a **vector database** so that, given a query vector, it can return the nearest chunks fast even over millions of vectors. Metadata (source URL, title, date, section) rides along for filtering and citations. Covered in the Vector Databases section." },
        { type: "heading", text: "5. Retrieve" },
        { type: "p", text: "Embed the incoming question with the **same** model, then ask the index for the top-$k$ nearest chunks by cosine similarity. $k$ is typically 3–10 for the final prompt (often more before reranking)." },
        { type: "heading", text: "6. Augment the prompt" },
        { type: "p", text: "Assemble a prompt that hands the retrieved chunks to the LLM as context and instructs it to answer *from that context only*. The template matters a lot — the Generation section is entirely about getting this right." },
        { type: "heading", text: "7. Generate" },
        { type: "p", text: "The LLM reads question + context and produces the answer, ideally with inline citations back to the chunk sources. That's RAG." },
        { type: "callout", variant: "gotcha", text: "**The #1 silent failure: mismatched embedding models between indexing and querying.** If you index with `bge-small` and later query with `bge-large` (or forget the `query:` prefix on one side), the vectors live in different spaces and your top results become random. Store the model name in your index metadata and assert it at query time." },
        { type: "callout", variant: "tip", text: "**Retrieval quality caps everything.** No prompt-engineering, no bigger model, and no clever generation can recover an answer whose evidence was never retrieved. When a RAG system is wrong, debug the *retrieval* first: print the top-$k$ chunks and ask 'is the answer even in here?' Nine times out of ten it isn't." },
      ]
    },

    {
      id: "chunking",
      title: "Chunking: the underrated 80% of RAG quality",
      level: "core",
      body: [
        { type: "p", text: "Chunking decides *what a single retrievable unit is*, and it quietly determines your ceiling. Too big and each chunk mixes several topics, so its embedding is a blurry average and the LLM drowns in irrelevant text. Too small and you slice a fact away from the context that makes it meaningful ('it costs $50' — *what* costs $50?). The art is chunks that are **self-contained and single-topic**." },
        { type: "heading", text: "Strategies, roughly in order of sophistication" },
        { type: "table",
          headers: ["Strategy", "How", "When"],
          rows: [
            ["**Fixed-size**", "Every $N$ tokens/chars, hard cut", "Baseline; fast but slices mid-sentence"],
            ["**Fixed + overlap**", "$N$ tokens, sliding window overlapping by $\\sim$10–20%", "The sane default — overlap heals boundary cuts"],
            ["**Recursive**", "Split on paragraphs, then sentences, then words, until under size", "Best general default; respects document structure"],
            ["**Semantic**", "Split where consecutive-sentence embeddings drift apart", "Topic-clean chunks; slower, needs embedding at index time"],
            ["**Document-structure**", "Split on Markdown headers / code functions / HTML sections", "Structured docs, codebases, wikis"],
          ]
        },
        { type: "heading", text: "The overlap trick, made concrete" },
        { type: "p", text: "With a chunk size of 500 tokens and an overlap of 50, chunk 2 starts 450 tokens into the document, re-including the last 50 tokens of chunk 1. This **overlap** means a sentence that straddles a boundary appears whole in at least one chunk, so a query matching it still retrieves complete context. Overlap is cheap insurance against the single most common chunking failure." },
        { type: "code", lang: "py", code: "def chunk_with_overlap(text, size=500, overlap=50):\n    \"\"\"Character-based sliding window. Real code should split on tokens.\"\"\"\n    step = size - overlap\n    chunks = []\n    for start in range(0, len(text), step):\n        piece = text[start:start + size]\n        if piece.strip():\n            chunks.append(piece)\n        if start + size >= len(text):\n            break\n    return chunks\n\n# In practice, use a structure-aware splitter:\n# from langchain_text_splitters import RecursiveCharacterTextSplitter\n# splitter = RecursiveCharacterTextSplitter(\n#     chunk_size=500, chunk_overlap=50,\n#     separators=[\"\\n\\n\", \"\\n\", \". \", \" \", \"\"])  # tries big breaks first\n# chunks = splitter.split_text(text)" },
        { type: "heading", text: "Chunk size is a tradeoff dial" },
        { type: "table",
          headers: ["", "Small chunks (~128 tok)", "Large chunks (~1024 tok)"],
          rows: [
            ["Retrieval precision", "Higher — tight topic match", "Lower — topic blur"],
            ["Context completeness", "Risk of orphaned facts", "More surrounding context"],
            ["Embedding sharpness", "Sharp, single-topic vector", "Averaged, fuzzy vector"],
            ["Prompt cost", "More chunks needed to cover", "Fewer chunks, more tokens each"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Sane starting point:** recursive splitting, ~400–600 tokens per chunk, ~10–15% overlap, and *always* attach metadata (source, title, section header, date). Tune from there by looking at retrieved chunks on real queries. There is no universal best size — it depends on your documents — so measure, don't guess." },
        { type: "callout", variant: "note", text: "**Metadata is a superpower, not an afterthought.** Store `source`, `title`, `url`, `section`, and `date` on every chunk. You use it three ways: to **filter** before searching ('only 2026 docs', 'only the billing space'), to **cite** in the answer, and to **debug** ('why did it retrieve this?'). A modern refinement is *contextual retrieval* (Anthropic, 2024): prepend a one-line LLM-generated summary of where each chunk sits in its document before embedding — it measurably lifts recall." },
      ]
    },

    {
      id: "vectordb",
      title: "Vector databases & the ANN index (HNSW, IVF)",
      level: "core",
      body: [
        { type: "p", text: "You have a million chunk-vectors and a query vector. **Exact** nearest-neighbour search compares the query against all million (a brute-force matmul) — perfectly accurate but $O(N)$ per query, which stops scaling past a few hundred thousand vectors under latency budgets. The industry answer is **Approximate Nearest Neighbour (ANN)** search: trade a tiny sliver of accuracy for orders-of-magnitude speed by *not* looking at every vector." },
        { type: "heading", text: "The three-way tradeoff" },
        { type: "p", text: "Every ANN index navigates a triangle: **accuracy** (recall — did it find the true nearest neighbours?), **speed** (queries per second / latency), and **memory** (RAM footprint). You cannot max all three; the index type and its knobs choose your point on the surface." },
        { type: "heading", text: "HNSW — the default winner" },
        { type: "p", text: "**HNSW** (Hierarchical Navigable Small World, [Malkov & Yashunin, 2016](https://arxiv.org/abs/1603.09320)) builds a multi-layer graph of vectors. Upper layers are sparse 'express lanes' connecting far-apart points; lower layers are dense local neighbourhoods. A search starts at the top, greedily hops toward the query through the sparse layers to get in the right region fast, then descends and refines in the dense bottom layer. It's the skip-list idea applied to geometry." },
        { type: "math", tex: String.raw`\text{brute force: } O(N) \quad\longrightarrow\quad \text{HNSW search: } \approx O(\log N)` },
        { type: "table",
          headers: ["HNSW knob", "Meaning", "Effect of increasing"],
          rows: [
            ["$M$", "edges per node", "Higher recall + more memory"],
            ["`ef_construction`", "candidate list size at build", "Better graph, slower build"],
            ["`ef_search`", "candidate list size at query", "Higher recall, slower query — tune this live"],
          ]
        },
        { type: "callout", variant: "tip", text: "**`ef_search` is your live accuracy/speed dial.** It costs nothing to change (no rebuild), and turning it up trades latency for recall on the fly. When recall is disappointing, raise `ef_search` before you touch anything else." },
        { type: "heading", text: "IVF — partition then search a few cells" },
        { type: "p", text: "**IVF** (Inverted File) first clusters all vectors into $n_{\\text{list}}$ groups via k-means (Clustering section). Each group has a centroid. At query time you find the `nprobe` nearest centroids and search *only* those cells, skipping the rest of the database. More `nprobe` = search more cells = higher recall, slower. IVF is often paired with **PQ** (Product Quantization), which compresses each vector into a few bytes so billions fit in RAM — at some recall cost." },
        { type: "table",
          headers: ["Index", "Recall", "Speed", "Memory", "Best for"],
          rows: [
            ["**Flat (exact)**", "100%", "Slow at scale", "High (full vectors)", "< ~100k vectors, or a ground-truth baseline"],
            ["**HNSW**", "Very high", "Very fast", "High (graph + vectors)", "The default for most RAG"],
            ["**IVF**", "Tunable", "Fast", "Medium", "Millions of vectors"],
            ["**IVF+PQ**", "Lower", "Fast", "Very low (compressed)", "Billions of vectors, RAM-bound"],
          ]
        },
        { type: "heading", text: "Which database?" },
        { type: "table",
          headers: ["Tool", "What it is", "Reach for it when"],
          rows: [
            ["**FAISS**", "Meta's ANN *library* (not a server)", "Max control, in-process, research & from-scratch"],
            ["**Chroma**", "Lightweight embedded/dev DB", "Prototyping, 'chat with your docs' on a laptop"],
            ["**Qdrant**", "Rust vector DB, great hybrid + filtering", "Self-hosted production, high throughput"],
            ["**pgvector**", "Postgres extension (HNSW since v0.5.0, 2023)", "You already run Postgres and have < ~1M vectors"],
            ["**Pinecone / Weaviate**", "Managed serverless / built-in hybrid", "You want someone else to run the infra"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**ANN is approximate — recall is a metric, not a given.** A misconfigured HNSW/IVF can quietly return 70%-recall results and your answers degrade with no error thrown. Always sanity-check against a **Flat** (exact) index on a sample of queries and confirm your ANN recall@k is ≥ ~0.95 before trusting it. 'It runs' is not 'it retrieves well.'" },
        { type: "callout", variant: "note", text: "**Why FAISS first.** In the from-scratch section we build the whole retriever with a NumPy dot product (that *is* a Flat index), then swap in FAISS to see the same idea, faster. Understanding that a vector DB is 'nearest-neighbour search + metadata + an index structure' demystifies the entire category." },
      ]
    },

    {
      id: "retrieval-quality",
      title: "Retrieval quality: hybrid search, reranking, HyDE, metrics",
      level: "core",
      body: [
        { type: "p", text: "Naive 'embed the query, take top-$k$ by cosine' is the *floor*, not the ceiling. Dense embeddings are great at meaning but can miss **exact terms** — a part number `X7-2200`, a rare acronym, a person's name — because those get smeared into the averaged vector. Production retrieval stacks several techniques to fix each failure mode." },
        { type: "heading", text: "Hybrid search — dense + sparse, fused" },
        { type: "p", text: "Run **two** retrievers in parallel: a **dense** one (embeddings, for meaning) and a **sparse** one — classically **BM25**, the venerable keyword-scoring function that rewards rare, exact term matches. Then fuse their ranked lists. The standard, model-free fusion is **Reciprocal Rank Fusion (RRF)**: score each document by summing $1/(k + \\text{rank})$ across the two lists, so a doc ranked highly by *either* retriever floats up." },
        { type: "math", tex: String.raw`\text{RRF}(d) = \sum_{r \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \operatorname{rank}_r(d)}, \qquad k \approx 60` },
        { type: "callout", variant: "tip", text: "**Hybrid search is the single biggest quality jump over naive RAG.** BM25 catches the exact-keyword queries dense retrieval fumbles (IDs, code, names) while embeddings catch the paraphrases BM25 misses. If you do one upgrade beyond top-$k$ cosine, do this." },
        { type: "heading", text: "Reranking — a slow, accurate second pass" },
        { type: "p", text: "Vector search uses a **bi-encoder**: query and document are embedded *separately*, so their vectors never 'see' each other — fast (precompute all doc vectors) but coarse. A **cross-encoder** reranker instead feeds *(query, document) together* through a Transformer and outputs one relevance score, letting every query token attend to every doc token. Far more accurate, far too slow to run over millions — so you use it as a **second stage**: retrieve top-50 cheaply with vectors, then rerank down to the top-5 with the cross-encoder." },
        { type: "code", lang: "text", code: "STAGE 1 (recall):    query ─► vector/hybrid search ─► top 50 candidates   (fast, coarse)\nSTAGE 2 (precision):  (query, cand) ─► cross-encoder ─► rerank ─► top 5   (slow, sharp)\n                                                        └► into the prompt" },
        { type: "callout", variant: "tip", text: "**Reranking is the highest-ROI upgrade after hybrid search.** Rerank ~10–50 candidates (not thousands — cross-encoders are expensive), keep the top 3–5. Off-the-shelf options: `bge-reranker`, `mxbai-rerank`, or Cohere Rerank. Retrieve-then-rerank routinely lifts answer quality 15–30% on standard benchmarks." },
        { type: "heading", text: "Query transformation — fix the question before you search" },
        { type: "p", text: "Sometimes the raw question is a bad search query. Two popular fixes: **query rewriting/expansion** (an LLM rephrases or splits the question into better search terms), and **HyDE** (Hypothetical Document Embeddings, [Gao et al., 2022](https://arxiv.org/abs/2212.10496)) — have the LLM *hallucinate a fake answer*, embed that, and search with it. The hunch: a hypothetical answer lives closer in embedding space to the real answer passages than the terse question does." },
        { type: "heading", text: "Diversity — MMR" },
        { type: "p", text: "Top-$k$ by pure similarity often returns five near-duplicate chunks. **Maximal Marginal Relevance (MMR)** re-selects to balance relevance to the query against novelty versus already-picked chunks, so the context covers *more* of the answer instead of repeating one point five times." },
        { type: "math", tex: String.raw`\text{MMR} = \arg\max_{d \in R \setminus S}\; \Big[\, \lambda\,\text{sim}(d, q) \;-\; (1-\lambda)\max_{d' \in S}\text{sim}(d, d') \,\Big]` },
        { type: "heading", text: "Measuring retrieval — the metrics" },
        { type: "table",
          headers: ["Metric", "Question it answers", "Formula / idea"],
          rows: [
            ["**Recall@k**", "Of the relevant docs, how many are in the top $k$?", "$\\frac{\\text{relevant in top }k}{\\text{total relevant}}$"],
            ["**Precision@k**", "Of the top $k$, how many are relevant?", "$\\frac{\\text{relevant in top }k}{k}$"],
            ["**MRR**", "How high is the *first* correct hit, on average?", "mean of $1/\\text{rank of first relevant}$"],
            ["**nDCG@k**", "Are the most relevant docs ranked highest?", "discounted gain, normalized to [0,1]"],
          ]
        },
        { type: "callout", variant: "note", text: "**Recall@k is the one to watch for RAG.** If the answer-bearing chunk isn't in the retrieved set, generation is doomed no matter how good the LLM is — so you first crank $k$ / recall to guarantee the evidence is *present*, then use reranking and nDCG to push it to the *top* where the model will actually attend to it." },
      ]
    },

    {
      id: "generation",
      title: "The generation step: prompts that cite and admit ignorance",
      level: "core",
      body: [
        { type: "p", text: "You've retrieved good chunks. Now you have to get the LLM to answer *from them* — not from its own frozen memory, and not by inventing bridges between them. Almost all of this lives in the **prompt template** and the **context budget**." },
        { type: "heading", text: "The anatomy of a RAG prompt" },
        { type: "p", text: "A solid template does four things: sets the role, injects the retrieved context clearly delimited, instructs the model to answer *only* from context and to *cite* sources, and gives an explicit escape hatch for 'not in the context.'" },
        { type: "code", lang: "py", code: "RAG_PROMPT = \"\"\"You are a support assistant. Answer the QUESTION using ONLY the\nCONTEXT below. Cite the source of each claim with its [id]. If the answer\nis not in the context, say exactly: \"I don't have that information.\"\nDo not use prior knowledge.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:\"\"\"\n\ndef build_context(chunks):\n    # chunks: list of dicts {id, source, text}\n    return \"\\n\\n\".join(f\"[{c['id']}] (source: {c['source']})\\n{c['text']}\"\n                       for c in chunks)\n\nprompt = RAG_PROMPT.format(context=build_context(retrieved),\n                           question=user_question)\n# answer = llm.generate(prompt)   # -> \"You can reset it in Settings [3].\"" },
        { type: "callout", variant: "tip", text: "**Force citations by giving each chunk an `[id]` and demanding the model reuse it.** Citations aren't just UX — they're a *hallucination brake*. A model that must attribute every sentence to a bracketed source is far less likely to invent one, and you (or an evaluator) can mechanically check that cited `[id]`s actually appear in the context." },
        { type: "heading", text: "Handling \"I don't know\"" },
        { type: "p", text: "The most valuable — and most neglected — behaviour is **abstaining** when the retrieved context doesn't contain the answer. A RAG bot that confidently answers off-context questions from its parametric memory is *worse* than useless in an enterprise, because it looks grounded but isn't. Give an explicit refusal string, put it in the instructions, and test for it." },
        { type: "callout", variant: "gotcha", text: "**\"Lost in the middle.\"** LLMs attend most to the **start and end** of a long context and can *skip evidence buried in the middle* ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)). So don't dump 40 mediocre chunks hoping the model finds the gold — retrieve fewer, better chunks (this is what reranking buys you) and place the strongest ones at the top and bottom of the context." },
        { type: "heading", text: "Context-window budgeting" },
        { type: "p", text: "The prompt = system instructions + retrieved chunks + question + room for the answer, and it all has to fit the model's context window while staying *cheap* (you pay per input token) and *sharp* (more tokens dilute attention). Budget it: pick a token cap for context, estimate tokens per chunk, and cap $k$ so you never overflow. More context is not more better — past a point it *lowers* answer quality and raises cost." },
        { type: "table",
          headers: ["Budget lever", "Move", "Effect"],
          rows: [
            ["Fewer, reranked chunks", "top-5 after rerank, not top-20 raw", "Cheaper, sharper, less 'lost in middle'"],
            ["Tighter chunks", "trim boilerplate before embedding", "More signal per token"],
            ["Compression", "LLM-summarize chunks before the prompt", "Fit more evidence, adds a call"],
            ["Order", "best chunks first & last", "Beats the middle-neglect effect"],
          ]
        },
      ]
    },

    {
      id: "evaluation",
      title: "Evaluating RAG: retrieval, generation, and faithfulness",
      level: "core",
      body: [
        { type: "p", text: "\"It looks good in the demo\" is not evaluation. A RAG system has **two** components that fail independently, so you must measure them separately — otherwise you can't tell whether a wrong answer was a *retrieval* miss (the evidence never showed up) or a *generation* miss (the evidence was there and the model ignored or mangled it)." },
        { type: "table",
          headers: ["Layer", "Fails when…", "Metrics"],
          rows: [
            ["**Retrieval**", "the right chunk isn't retrieved", "recall@k, precision@k, MRR, nDCG"],
            ["**Generation**", "answer isn't supported by the retrieved chunks", "faithfulness, answer relevancy"],
          ]
        },
        { type: "heading", text: "The RAGAS-style metrics" },
        { type: "p", text: "**[RAGAS](https://docs.ragas.io/)** popularized a set of *reference-light* metrics, most computed by an **LLM-as-judge** so you don't need a hand-labelled gold answer for everything. The four you'll cite constantly:" },
        { type: "table",
          headers: ["Metric", "Asks", "Catches"],
          rows: [
            ["**Faithfulness / groundedness**", "Is every claim in the answer supported by the retrieved context?", "Hallucination — the cardinal RAG sin"],
            ["**Answer relevancy**", "Does the answer actually address the question?", "On-topic-but-evasive answers"],
            ["**Context precision**", "Are the retrieved chunks relevant (and ranked well)?", "Noisy retrieval / bad ranking"],
            ["**Context recall**", "Did retrieval fetch all the info the answer needs?", "Missing evidence"],
          ]
        },
        { type: "p", text: "**Faithfulness** is the flagship. It's the operational definition of 'not hallucinating': decompose the answer into atomic claims, then check what fraction are entailed by the retrieved context. A faithfulness of 1.0 means every sentence is backed by a source you handed the model." },
        { type: "math", tex: String.raw`\text{faithfulness} = \frac{\#\{\text{claims in answer supported by context}\}}{\#\{\text{claims in answer}\}}` },
        { type: "heading", text: "How to actually run it" },
        { type: "p", text: "Build a **golden set** of 50–100 representative (question, ideal-answer, relevant-chunk) triples — this is the single highest-value asset in a RAG project. Then run your metrics after *every* pipeline change (new chunk size, new embedder, new reranker) so you can tell an improvement from a regression instead of vibing it." },
        { type: "code", lang: "py", code: "# RAGAS sketch: evaluate a batch of RAG outputs\nfrom ragas import evaluate\nfrom ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall\nfrom datasets import Dataset\n\ndata = Dataset.from_dict({\n    \"question\":     [q for q in questions],\n    \"answer\":       [a for a in generated_answers],\n    \"contexts\":     [ctx for ctx in retrieved_chunks],   # list[list[str]]\n    \"ground_truth\": [g for g in gold_answers],\n})\n\nresult = evaluate(data, metrics=[faithfulness, answer_relevancy,\n                                 context_precision, context_recall])\nprint(result)   # {'faithfulness': 0.94, 'answer_relevancy': 0.88, ...}\n# Ship targets (rules of thumb): faithfulness > 0.9, the rest > 0.8." },
        { type: "callout", variant: "tip", text: "**Failure analysis beats an aggregate score.** When faithfulness dips, pull the *worst* examples and read them: was the answer-chunk retrieved (retrieval bug → fix chunking/embedder/reranker) or retrieved-but-ignored (generation bug → fix the prompt/model)? This retrieval-vs-generation split is the debugging fork you'll walk a hundred times." },
        { type: "callout", variant: "warn", text: "**LLM-as-judge is convenient, not infallible.** Judges have biases (they favour longer answers, and can rate a model's own outputs generously). Calibrate the judge against a slice of *human* labels before you trust its numbers, and never let a single automated score be the only gate on a production change." },
      ]
    },

    {
      id: "advanced",
      title: "Advanced RAG & the long-context debate (2025–2026)",
      level: "deep",
      body: [
        { type: "p", text: "Once the basic pipeline is solid, the frontier is about handling questions a single retrieval pass can't answer, and about whether retrieval is even needed as context windows explode." },
        { type: "heading", text: "Agentic RAG" },
        { type: "p", text: "Instead of one fixed retrieve→generate pass, an **agent** LLM *decides* when and what to retrieve, can issue multiple searches, reformulate queries, call tools, and judge whether it has enough before answering. RAG becomes a **tool the model calls in a loop** rather than a hardcoded step — the natural marriage of this section with the Agents/tool-use material. It's more capable and more expensive; use it when questions genuinely need iterative lookup." },
        { type: "heading", text: "Multi-hop RAG" },
        { type: "p", text: "Some questions need chained facts: *'What is the capital of the country that won the 2018 World Cup?'* requires retrieving 'France won' → then 'capital of France.' **Multi-hop** RAG decomposes the question, retrieves per sub-question, and composes the answer — single-shot retrieval can't get there because the second query depends on the first's result." },
        { type: "heading", text: "Graph RAG" },
        { type: "p", text: "**GraphRAG** ([Microsoft, 2024](https://arxiv.org/abs/2404.16130)) builds a **knowledge graph** of entities and relationships from the corpus, optionally with community summaries. It shines on *global* questions ('what are the main themes across all these documents?') that no single chunk answers, because the graph captures structure that flat chunk retrieval throws away. Cost: an expensive indexing pass to extract the graph." },
        { type: "heading", text: "Long-context vs RAG — the live debate" },
        { type: "p", text: "Models now advertise 100k–1M+ token windows, prompting the recurring claim that 'you can just paste all your docs and skip RAG.' The honest 2025–2026 answer is **it depends, and RAG is not dead**:" },
        { type: "table",
          headers: ["Dimension", "Long-context (stuff it all in)", "RAG (retrieve then generate)"],
          rows: [
            ["Cost per query", "High — you pay for every token, every call", "Low — only the top-$k$ chunks"],
            ["Corpus size", "Caps at the window (still finite)", "Effectively unbounded (millions of docs)"],
            ["Freshness / updates", "Re-send everything", "Update the index only"],
            ["Attention quality", "Degrades with length ('lost in middle')", "Focused, curated evidence"],
            ["Citations / audit", "Harder to trace", "Chunk-level provenance"],
            ["Best at", "One big document, deep cross-references", "Large, changing knowledge bases"],
          ]
        },
        { type: "callout", variant: "note", text: "**Consensus, mid-2020s:** long context and RAG are complementary, not rivals. Retrieve to *narrow* millions of documents down to the relevant thousands of tokens, then let a capable long-context model reason deeply over that focused set. Retrieval is a cost, precision, freshness, and scale play that a bigger window doesn't erase — you rarely want to pay to attend over your entire corpus on every question." },
      ]
    },

    {
      id: "from-scratch",
      title: "Build it: minimal RAG in NumPy, then the real stack",
      level: "core",
      body: [
        { type: "p", text: "RAG demystifies completely once you build the retriever by hand. A vector search *is* a normalized dot product plus a top-$k$ sort — everything else (FAISS, Chroma, Qdrant) is that idea made fast and durable. Here's the whole thing in NumPy." },
        { type: "heading", text: "1. From scratch — embed, cosine, top-k, prompt" },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sentence_transformers import SentenceTransformer\n\nmodel = SentenceTransformer(\"BAAI/bge-small-en-v1.5\")\n\ndocs = [\n    \"To reset your password, open Settings > Account recovery and follow the email link.\",\n    \"Refunds are processed within 5-7 business days to the original payment method.\",\n    \"Our support team is available Monday to Friday, 9am to 5pm Pacific time.\",\n    \"You can upgrade or downgrade your plan anytime from the Subscription tab.\",\n    \"Two-factor authentication can be enabled under Settings > Security.\",\n]\n\n# --- INDEX (offline): embed once, L2-normalize so cosine == dot product ---\nDB = model.encode(docs, normalize_embeddings=True)          # (N, d)\n\ndef retrieve(query, k=2):\n    q = model.encode(query, normalize_embeddings=True)       # (d,)\n    scores = DB @ q                                          # (N,) cosine sims\n    top = np.argsort(-scores)[:k]                            # k highest\n    return [(docs[i], float(scores[i])) for i in top]\n\ndef rag_prompt(query, k=2):\n    hits = retrieve(query, k)\n    context = \"\\n\".join(f\"[{i+1}] {text}\" for i, (text, _) in enumerate(hits))\n    return (f\"Answer ONLY from the context, cite [id], and if it's not there \"\n            f\"say \\\"I don't have that information.\\\"\\n\\n\"\n            f\"CONTEXT:\\n{context}\\n\\nQUESTION: {query}\\nANSWER:\")\n\nfor text, s in retrieve(\"how do I get my money back?\", k=2):\n    print(round(s, 3), text)\n# ~0.55  Refunds are processed within 5-7 business days ...\nprint(rag_prompt(\"how do I get my money back?\"))\n# -> feed this string to any LLM (Claude, GPT, a local model) to finish RAG." },
        { type: "callout", variant: "note", text: "That `DB @ q` is a **Flat (exact) index** — the same brute-force search a vector DB falls back to, and the ground truth you measure ANN recall against. For a few thousand chunks it's genuinely all you need; reach for FAISS/HNSW only when $N$ or latency demands it." },
        { type: "heading", text: "2. The real stack — FAISS for scale" },
        { type: "code", lang: "py", code: "import faiss, numpy as np\nfrom sentence_transformers import SentenceTransformer\n\nmodel = SentenceTransformer(\"BAAI/bge-small-en-v1.5\")\nDB = model.encode(docs, normalize_embeddings=True).astype(\"float32\")\nd = DB.shape[1]\n\n# Inner-product index on normalized vectors == cosine similarity.\nindex = faiss.IndexFlatIP(d)          # exact; swap for IndexHNSWFlat at scale\nindex.add(DB)\n\nq = model.encode(\"how do I get my money back?\",\n                 normalize_embeddings=True).astype(\"float32\")[None, :]\nscores, ids = index.search(q, k=2)    # returns (scores, indices)\nfor s, i in zip(scores[0], ids[0]):\n    print(round(float(s), 3), docs[i])\n\n# For millions of vectors, build an HNSW index instead:\n# index = faiss.IndexHNSWFlat(d, 32)          # M = 32 edges per node\n# index.hnsw.efConstruction = 200\n# index.hnsw.efSearch = 64                    # the live recall/speed dial" },
        { type: "heading", text: "3. The real stack — Chroma for 'chat with your docs'" },
        { type: "code", lang: "py", code: "import chromadb\nfrom chromadb.utils import embedding_functions\n\nclient = chromadb.Client()\nembed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(\n    model_name=\"BAAI/bge-small-en-v1.5\")\ncol = client.create_collection(\"support_docs\", embedding_function=embed_fn)\n\n# Chroma embeds + indexes + stores metadata in one call.\ncol.add(documents=docs,\n        ids=[f\"doc_{i}\" for i in range(len(docs))],\n        metadatas=[{\"source\": \"help-center\"} for _ in docs])\n\nres = col.query(query_texts=[\"how do I get my money back?\"], n_results=2)\nfor doc, meta in zip(res[\"documents\"][0], res[\"metadatas\"][0]):\n    print(meta[\"source\"], \"->\", doc)\n# Chroma hides embedding + FAISS/HNSW behind a dict-like API — great to start." },
        { type: "callout", variant: "tip", text: "**Progression to internalize:** the NumPy version *is* the algorithm; FAISS makes the nearest-neighbour search fast; Chroma/Qdrant add persistence, metadata filtering, and hybrid search on top. Frameworks like **LangChain** and **LlamaIndex** then wrap loaders + splitters + a vector store + the prompt into a few lines. Learn the layers bottom-up and none of it is a black box." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Build at least the first two end-to-end. RAG is a *systems* skill — the understanding comes from watching real retrieval succeed and fail on your own documents, then measuring it." },
        { type: "list", ordered: true, items: [
          "**Chat with your docs (the canonical build).** Take a folder of PDFs/Markdown (your notes, a manual, a codebase). Load → recursive-chunk (500/50) → embed with `bge-small` → store in Chroma → retrieve top-5 → prompt an LLM to answer with `[id]` citations. Ship a tiny CLI or Streamlit UI. This one project teaches 80% of applied RAG.",
          "**NumPy retriever, no libraries.** Reimplement retrieval with only NumPy: embed, L2-normalize, `DB @ q`, `argsort`, top-$k$. Then add a **Flat FAISS** index and confirm you get identical results — proving a vector DB is 'just' fast nearest-neighbour search.",
          "**Chunking bake-off.** Fix everything except chunk size/overlap. Try (128/0), (512/50), (1024/128) and semantic chunking on the same corpus and questions. Measure recall@5 on a 30-question golden set. Plot the tradeoff curve and pick your document's sweet spot.",
          "**Add hybrid + reranking.** Extend project 1 with BM25 (`rank_bm25`), fuse with RRF, then rerank the top-30 with `bge-reranker` down to top-5. Measure recall@5 and answer faithfulness before vs after. Feel the 15–30% jump.",
          "**Evaluate with RAGAS.** Build a 50-question golden set, wire up faithfulness, answer_relevancy, context_precision/recall, and run it after each change from projects 3–4. Keep a scorecard; make an improvement *provable*, not vibed.",
          "**Break it on purpose.** Ask questions whose answers aren't in the corpus and verify your prompt makes the model *abstain* ('I don't have that information') instead of hallucinating. Then feed a poorly-parsed PDF and watch garbage-in ruin retrieval — the lesson that ingestion is half the battle.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "The foundational paper first, then the tools and evaluation frameworks you'll actually use day to day:" },
        { type: "link", url: "https://arxiv.org/abs/2005.11401", text: "Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (the original RAG paper)" },
        { type: "link", url: "https://huggingface.co/spaces/mteb/leaderboard", text: "MTEB Leaderboard — the live scoreboard for choosing an embedding model" },
        { type: "link", url: "https://www.sbert.net/", text: "Sentence-Transformers (SBERT) docs — embeddings, cross-encoder rerankers, training your own" },
        { type: "link", url: "https://github.com/facebookresearch/faiss/wiki", text: "FAISS wiki — index types (Flat, IVF, HNSW, PQ) and how to choose and tune them" },
        { type: "link", url: "https://docs.ragas.io/", text: "RAGAS docs — faithfulness, answer relevancy, context precision/recall for RAG evaluation" },
        { type: "link", url: "https://arxiv.org/abs/1603.09320", text: "Malkov & Yashunin (2016) — the HNSW algorithm paper" },
        { type: "link", url: "https://python.langchain.com/docs/tutorials/rag/", text: "LangChain RAG tutorial — loaders, splitters, vector stores wired end-to-end (see also LlamaIndex)" },
      ]
    },
  ],

  packages: [
    { name: "sentence-transformers", why: "embed text and run cross-encoder rerankers; the open-source retrieval workhorse" },
    { name: "faiss-cpu", why: "Meta's ANN library — Flat, IVF, HNSW, PQ indexes for fast vector search" },
    { name: "chromadb", why: "embedded vector DB — embed + index + metadata in a few lines, ideal for prototyping" },
    { name: "qdrant-client", why: "production vector DB with first-class hybrid search and metadata filtering" },
    { name: "pgvector", why: "vector search inside Postgres (HNSW) — reuse your existing database" },
    { name: "rank-bm25", why: "sparse/keyword retrieval for hybrid search (dense + BM25 + RRF)" },
    { name: "ragas", why: "reference-light RAG evaluation: faithfulness, relevancy, context precision/recall" },
    { name: "langchain / llama-index", why: "glue loaders, splitters, vector stores, and prompts into a pipeline" },
  ],

  gotchas: [
    "**Different embedding model for indexing vs querying** puts vectors in incompatible spaces — retrieval returns noise. Pin and assert the model name.",
    "Instruction-tuned embedders (E5/BGE) need their prefixes: `query:` on questions, `passage:` on documents. Forgetting them silently drops recall 5–15 points.",
    "ANN is **approximate** — a misconfigured HNSW/IVF can quietly return ~70% recall with no error. Benchmark recall@k against a Flat index; aim ≥ 0.95.",
    "Mixing distance metrics: cosine, dot product, and Euclidean only agree when vectors are L2-normalized. Normalize once, consistently, everywhere.",
    "\"Lost in the middle\": LLMs neglect evidence buried in a long context. Retrieve *fewer, reranked* chunks and put the best ones first and last.",
    "Retrieval quality caps everything — a wrong RAG answer usually means the evidence was never retrieved. Debug retrieval before touching the prompt or model.",
    "No 'I don't know' escape hatch → the model answers off-context from parametric memory and hallucinates. Instruct abstention and test for it.",
    "Bad ingestion (scrambled PDF tables, glued headers) poisons every downstream stage. Garbage in, confidently-cited garbage out.",
  ],

  flashcards: [
    { q: "In one line, what is RAG and what two problems does it solve?", a: "Retrieve relevant documents at query time and paste them into the prompt so the LLM answers open-book — fixing frozen knowledge (staleness) and hallucination (ungrounded facts)." },
    { q: "RAG vs fine-tuning — the rule of thumb?", a: "Fine-tuning changes weights to teach a *skill/behaviour*; RAG changes context to supply a *fact/knowledge*. Facts and freshness → RAG; style and skill → fine-tune. Often both." },
    { q: "Why L2-normalize embeddings before search?", a: "For unit vectors, cosine similarity equals the dot product $q^\\top d$, so ranking reduces to one fast matrix multiply — and cosine, dot, and Euclidean order agree." },
    { q: "List the RAG pipeline stages.", a: "Ingest → chunk → embed → index (offline); then embed query → retrieve top-k → (rerank) → augment prompt → generate (online)." },
    { q: "How does HNSW beat brute-force search?", a: "A multi-layer navigable graph: sparse upper layers jump near the query fast, dense lower layers refine — roughly $O(\\log N)$ vs $O(N)$, at a small recall cost." },
    { q: "What is hybrid search and why does it help?", a: "Run dense (embeddings, meaning) and sparse (BM25, exact terms) retrievers in parallel and fuse with RRF. BM25 catches IDs/acronyms/names embeddings smear; the biggest jump over naive RAG." },
    { q: "Bi-encoder vs cross-encoder reranker?", a: "Bi-encoder embeds query and doc separately (fast, coarse, used for retrieval); cross-encoder scores (query, doc) jointly (slow, sharp). Retrieve top-50 with the bi-encoder, rerank to top-5 with the cross-encoder." },
    { q: "What is faithfulness / groundedness?", a: "The fraction of the answer's claims that are supported by the retrieved context — the operational measure of 'not hallucinating.' Target > 0.9." },
    { q: "Why is recall@k the retrieval metric to prioritize for RAG?", a: "If the answer-bearing chunk isn't in the top-k, generation cannot succeed no matter how good the LLM is. Get evidence *present* first, then rerank it to the *top*." },
    { q: "What is the 'lost in the middle' effect?", a: "LLMs attend most to the start and end of a long context and can skip evidence in the middle — so use fewer, reranked chunks and place the best ones first and last." },
    { q: "Long-context vs RAG — settled view?", a: "Complementary, not rivals. Retrieve to narrow millions of docs to relevant tokens, then let a long-context model reason over them. RAG still wins on cost, scale, freshness, and citations." },
    { q: "What does HyDE do?", a: "Has the LLM hallucinate a fake answer, embeds *that*, and searches with it — a hypothetical answer sits closer in embedding space to real answer passages than the terse question does." },
  ],

  cheatsheet: [
    { label: "Embed (normalized)", code: "M.encode(texts, normalize_embeddings=True)" },
    { label: "Cosine == dot", code: "scores = DB @ q   # both L2-normalized" },
    { label: "Top-k", code: "np.argsort(-scores)[:k]" },
    { label: "FAISS flat (cosine)", code: "idx = faiss.IndexFlatIP(d); idx.add(DB)" },
    { label: "FAISS HNSW", code: "faiss.IndexHNSWFlat(d, 32)  # M=32" },
    { label: "HNSW recall dial", code: "index.hnsw.efSearch = 64" },
    { label: "FAISS search", code: "scores, ids = idx.search(q, k)" },
    { label: "Chroma add", code: "col.add(documents=docs, ids=ids)" },
    { label: "Chroma query", code: "col.query(query_texts=[q], n_results=k)" },
    { label: "Recursive split", code: "RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)" },
    { label: "BM25", code: "BM25Okapi(tokenized_corpus).get_scores(tokens)" },
    { label: "RRF fuse", code: "sum(1/(60 + rank) for rank in ranks)" },
    { label: "Rerank", code: "CrossEncoder('BAAI/bge-reranker-base').predict(pairs)" },
    { label: "Evaluate", code: "ragas.evaluate(ds, metrics=[faithfulness, ...])" },
  ],
});
