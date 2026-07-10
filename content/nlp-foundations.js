(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "nlp-foundations",
  name: "NLP Foundations",
  language: "NLP",
  group: "Natural Language Processing",
  navLabel: "NLP Foundations",
  tagline: "Where language meets math — preprocessing, tokenization (the real BPE that GPT uses), n-gram language models, smoothing, and perplexity, all built from scratch.",
  color: "#06B6D4",
  readMinutes: 46,
  sections: [
    {
      id: "why",
      title: "Why language is hard for computers",
      level: "core",
      body: [
        { type: "p", text: "**Natural Language Processing** is the art of getting computers to read, understand, and generate human language. It is the discipline behind search, translation, spam filters, autocomplete, sentiment dashboards, voice assistants, and — the reason you are almost certainly here — every large language model. This is the opening page of the NLP track: before attention, before embeddings, before GPT, you need the substrate everything else is built on: how raw text becomes numbers, and how we model the probability of a sentence." },
        { type: "p", text: "Language looks easy because you have spoken it since you were two. That intuition is exactly what makes it treacherous to program. Here is why text is fundamentally harder than the tidy numeric tables of classical ML:" },
        { type: "table",
          headers: ["Challenge", "What it means", "Example"],
          rows: [
            ["**Ambiguity**", "the same string has many meanings, resolvable only in context", "*\"I saw her duck\"* — a bird, or she ducked?"],
            ["**Compositionality**", "meaning is built recursively from parts, and the parts interact", "*\"not bad\"* ≠ *not* + *bad*; *\"small\"* changes meaning in *\"small chance\"* vs *\"small dog\"*"],
            ["**Long-range dependence**", "a word can depend on another hundreds of tokens away", "the subject of a verb, or the antecedent of *\"it\"*, may be paragraphs back"],
            ["**World knowledge**", "understanding often needs facts not in the sentence", "*\"The trophy didn't fit in the suitcase because it was too big\"* — what is *\"it\"*?"],
            ["**Sparsity / productivity**", "language is infinite; you will always meet unseen words and sentences", "new names, typos, slang, code-switching, `covfefe`"],
            ["**Discreteness**", "words are symbols with no built-in notion of similarity", "to a naive computer *\"cat\"* and *\"dog\"* are as unrelated as *\"cat\"* and *\"Tuesday\"*"],
          ]
        },
        { type: "p", text: "That last row is the deep one, and it is the through-line of this entire track. A CPU can compare 3.1 and 3.2 and know they are close. It has no idea that *\"cat\"* and *\"dog\"* are close. Making the discrete symbols of language *behave* like numbers with meaningful geometry is the central project of modern NLP — and it culminates in the dense embeddings and transformers of the next sections." },
        { type: "heading", text: "A very short history: rules → statistics → neural → LLMs" },
        { type: "p", text: "NLP has lived through four eras, and each one lost a battle with the messiness of language:" },
        { type: "table",
          headers: ["Era", "Approach", "Why it gave way"],
          rows: [
            ["1950s–1980s: **rules**", "hand-written grammars, dictionaries, expert systems", "language has too many exceptions to enumerate; brittle and unmaintainable"],
            ["1990s–2000s: **statistical**", "count things in large corpora; n-gram models, HMMs, later SVMs", "features are hand-crafted; models are shallow and data-hungry per task"],
            ["2013–2017: **neural / embeddings**", "word2vec, RNNs/LSTMs learn dense features from data", "recurrence is sequential and forgets long-range context"],
            ["2017–now: **transformers & LLMs**", "self-attention + massive self-supervised pretraining", "the current frontier — this whole track builds toward it"],
          ]
        },
        { type: "callout", variant: "note", text: "**Why start with the \"old\" statistical methods at all?** Because the modern models never escaped these ideas — they industrialized them. A transformer LM is still, at its output, predicting the next token from context — exactly the goal of a 1990s n-gram model, just with a vastly better estimator. **Tokenization** (this page) is *literally* the first thing GPT and Llama do to your text. **Perplexity** (this page) is *still* how those models are evaluated in 2026. Learn the foundations here and the LLM track becomes a series of upgrades rather than a series of mysteries." },
      ]
    },

    {
      id: "preprocessing",
      title: "The text preprocessing pipeline",
      level: "core",
      body: [
        { type: "p", text: "Raw text is filthy: inconsistent casing, Unicode look-alikes, HTML, emoji, whitespace, punctuation glued to words. Before any model sees it, text usually passes through a **normalization pipeline** that makes superficially-different strings identical so the model isn't forced to learn that `\"Cafe\"`, `\"café\"`, and `\"CAFÉ\"` are the same idea. The catch — and the theme of this section — is that **every normalization step throws information away**, and sometimes the information you throw away is the signal." },
        { type: "heading", text: "Case-folding" },
        { type: "p", text: "**Case-folding** (lowercasing) collapses `\"Apple\"` and `\"apple\"` into one token. Cheap and usually helpful — but `\"Apple\"` the company and `\"apple\"` the fruit just merged, and `\"US\"` (the country) became `\"us\"` (the pronoun). Case-folding is a classic *recall-for-precision* trade." },
        { type: "heading", text: "Unicode normalization" },
        { type: "p", text: "The single most under-appreciated step. The character `é` can be encoded two ways: as one code point (U+00E9) or as `e` + a combining accent (U+0065 U+0301). They look identical and compare **unequal**. **Unicode normalization** (NFC composes, NFD decomposes) forces a canonical form so the two spellings unify. Skip this and you get silent, maddening bugs where two visually identical strings hash to different tokens." },
        { type: "code", lang: "py", code: "import unicodedata\n\ns1 = \"caf\\u00e9\"          # é as one code point\ns2 = \"cafe\\u0301\"         # e + combining acute accent\nprint(s1 == s2)                       # False  (!!) — looks identical, isn't\nprint(len(s1), len(s2))               # 4 5\n\nn1 = unicodedata.normalize(\"NFC\", s1)\nn2 = unicodedata.normalize(\"NFC\", s2)\nprint(n1 == n2)                       # True — now they unify\n\n# A fuller normalizer:\ndef normalize(text):\n    text = unicodedata.normalize(\"NFC\", text)   # canonical Unicode\n    text = text.lower()                          # case-fold\n    text = \" \".join(text.split())               # collapse whitespace\n    return text\n\nprint(normalize(\"  The  CAFÉ\\tis   OPEN \"))     # 'the café is open'" },
        { type: "callout", variant: "gotcha", text: "**Unicode confusables are a security issue, not just a data issue.** Attackers register domains like `аpple.com` where the `а` is Cyrillic (U+0430), not Latin `a`. Unicode normalization (NFC) does **not** fix this — those are genuinely different letters. Detecting homoglyph attacks needs a separate confusables map. For NLP data cleaning, NFC/NFKC is enough; just know its limits." },
        { type: "heading", text: "Stemming vs lemmatization" },
        { type: "p", text: "Both reduce inflected words to a base form so `\"organize\"`, `\"organizes\"`, `\"organizing\"` count as one. They differ in how principled they are:" },
        { type: "table",
          headers: ["", "Stemming", "Lemmatization"],
          rows: [
            ["Method", "chop suffixes with crude rules (Porter, Snowball)", "dictionary + part-of-speech to find the real base word (lemma)"],
            ["`\"studies\"` →", "`studi`", "`study`"],
            ["`\"better\"` →", "`better` (missed)", "`good`"],
            ["Speed / cost", "very fast, no lookup", "slower, needs a lexicon + POS tag"],
            ["Output", "may not be a real word", "always a valid word"],
          ]
        },
        { type: "p", text: "**Stemming** is a fast hack that often over- or under-chops (`\"universe\"`, `\"university\"` both → `\"univers\"`). **Lemmatization** is linguistically correct but needs to know the part of speech (*\"saw\"* → *\"see\"* as a verb, *\"saw\"* as a noun stays *\"saw\"*). Use lemmatization when quality matters; reach for stemming only for quick-and-dirty search indexing." },
        { type: "heading", text: "Stop words — and when NOT to remove them" },
        { type: "p", text: "**Stop words** are ultra-frequent, low-content words — `the`, `is`, `at`, `of`, `and`. Classic bag-of-words pipelines drop them to shrink the vocabulary and let content words dominate. That is fine for a spam filter or topic model. It is a **disaster** for anything where function words carry meaning:" },
        { type: "list", ordered: false, items: [
          "**Sentiment / negation:** removing `not` turns *\"not good\"* into *\"good\"*. You inverted the label.",
          "**Question answering & search phrases:** *\"to be or not to be\"* becomes empty — every word is a stop word.",
          "**Anything sequential (neural models):** RNNs and transformers *want* the function words; they encode syntax and relationships. Modern deep NLP almost never removes stop words.",
        ]},
        { type: "callout", variant: "tip", text: "**Rule of thumb for the whole pipeline:** aggressive normalization (lowercase, stem, drop stop words) suits **sparse, count-based** models that would otherwise drown in vocabulary. **Neural** models (embeddings, transformers) prefer text left nearly raw — they learn the distinctions you would have destroyed. As you move rightward through this track, you will normalize *less and less*, until an LLM tokenizer takes the text almost verbatim." },
      ]
    },

    {
      id: "tokenization",
      title: "Tokenization: splitting text into units",
      level: "core",
      body: [
        { type: "p", text: "A model cannot consume a string; it consumes a sequence of integer IDs. **Tokenization** is the step that splits text into units (**tokens**) and maps each to an ID via a fixed **vocabulary**. This is the most consequential preprocessing decision you make — it fixes what the model can even *see*. It is also, literally, the first operation GPT, Llama, and Claude perform on your prompt." },
        { type: "heading", text: "Word-level tokenization and its fatal flaw" },
        { type: "p", text: "The obvious approach: split on whitespace/punctuation, one ID per distinct word. Simple, and each token is a meaningful unit. But it collides head-on with the productivity of language — the **out-of-vocabulary (OOV)** problem:" },
        { type: "list", ordered: false, items: [
          "You must fix the vocabulary at training time. Any word never seen — a new name, a typo, `\"antidisestablishmentarianism\"`, `\"GPT-5\"` — becomes the single `<UNK>` (unknown) token. All unseen words are now indistinguishable.",
          "The vocabulary explodes. English has hundreds of thousands of word forms; morphologically rich languages (Finnish, Turkish) have millions. A huge vocabulary means a huge, mostly-wasted embedding table.",
          "It knows nothing about morphology: `\"run\"`, `\"runs\"`, `\"running\"`, `\"runner\"` are four unrelated IDs with no shared structure.",
        ]},
        { type: "heading", text: "Character-level tokenization: the opposite trade" },
        { type: "p", text: "Go to the other extreme — one token per character. The vocabulary is tiny (~100 for English, a few thousand with Unicode) and there is **no OOV ever**: any string is spellable from characters. The price is brutal: sequences become very long (a 10-word sentence is ~50 characters), and the model must learn to assemble meaning from scratch — it has to *rediscover* that `t-h-e` is a word. More tokens per sentence also means more compute, and self-attention cost grows quadratically with length (you will see exactly why in the Transformers section)." },
        { type: "table",
          headers: ["Scheme", "Vocab size", "Sequence length", "OOV?", "Verdict"],
          rows: [
            ["Word", "huge (100k+)", "short", "**yes** — the killer", "too brittle"],
            ["Character", "tiny (~100)", "very long", "never", "too fine, too slow"],
            ["**Subword**", "medium (~30k–100k)", "medium", "never", "the modern sweet spot"],
          ]
        },
        { type: "heading", text: "Subword tokenization: the best of both worlds" },
        { type: "p", text: "The winning idea, universal since ~2016: **subword tokenization**. Keep frequent words whole (`\" the\"`, `\" running\"` become single tokens) but break rare words into meaningful pieces (`\"tokenization\"` → `\"token\"` + `\"ization\"`, an unknown name → a handful of subwords). You get a bounded vocabulary, *never* hit true OOV (worst case, you fall back to characters), and the model sees shared morphology. Three algorithms dominate:" },
        { type: "table",
          headers: ["Algorithm", "How it builds the vocab", "Used by"],
          rows: [
            ["**BPE** (Byte-Pair Encoding)", "greedily *merge* the most frequent adjacent pair, repeatedly", "GPT-2/3/4, Llama, RoBERTa"],
            ["**WordPiece**", "merge the pair that most increases corpus likelihood (not raw count)", "BERT, DistilBERT"],
            ["**Unigram / SentencePiece**", "start with a huge vocab, *prune* tokens that hurt likelihood least", "T5, ALBERT, most multilingual models"],
          ]
        },
        { type: "callout", variant: "note", text: "**SentencePiece** is a library, not a distinct algorithm — it implements both BPE and Unigram, with one killer feature: it treats the input as a raw byte/Unicode stream and encodes spaces as a visible token (`▁`), so it is fully reversible and language-agnostic (no assumption that words are space-separated — crucial for Chinese, Japanese, Thai). Modern LLM tokenizers are **byte-level BPE**: they run BPE over raw UTF-8 bytes, so *any* input, in any language or emoji, is representable with a 256-symbol base alphabet and zero OOV." },
        { type: "heading", text: "Deriving the BPE merge algorithm" },
        { type: "p", text: "BPE was originally a 1994 data-compression scheme; Sennrich et al. repurposed it for NLP in 2016. The algorithm is beautifully simple. Start with the vocabulary being the individual characters. Then repeat: **find the most frequent adjacent pair of symbols in the corpus, and merge it into a new single symbol.** Each merge adds one token to the vocabulary. Do this $N$ times and you have learned $N$ merges plus the base characters." },
        { type: "math", tex: String.raw`\text{repeat } N \text{ times:}\quad (a,b)^\star = \arg\max_{(a,b)} \operatorname{count}(a,b), \qquad \text{merge } (a,b) \to ab` },
        { type: "p", text: "Frequent pieces (`t`+`h` → `th`, then `th`+`e` → `the`) get merged early and end up as whole tokens; rare sequences never accumulate enough count and stay split into small pieces. The number of merges $N$ is the single knob controlling vocabulary size. That is the whole idea — now build it." },
        { type: "code", lang: "py", code: "import re\nfrom collections import Counter, defaultdict\n\ndef get_pair_counts(vocab):\n    \"\"\"Count adjacent symbol pairs across the corpus, weighted by word freq.\"\"\"\n    pairs = Counter()\n    for word, freq in vocab.items():\n        symbols = word.split()\n        for i in range(len(symbols) - 1):\n            pairs[(symbols[i], symbols[i + 1])] += freq\n    return pairs\n\ndef merge_pair(pair, vocab):\n    \"\"\"Replace every occurrence of the pair (a, b) with the merged symbol 'ab'.\"\"\"\n    a, b = pair\n    bigram = re.escape(a + \" \" + b)\n    pattern = re.compile(r\"(?<!\\S)\" + bigram + r\"(?!\\S)\")\n    return {pattern.sub(a + b, word): freq for word, freq in vocab.items()}\n\ndef learn_bpe(corpus, num_merges):\n    # Each word is stored as space-separated characters + an end-of-word marker </w>.\n    vocab = Counter()\n    for word in corpus.split():\n        vocab[\" \".join(list(word)) + \" </w>\"] += 1\n\n    merges = []\n    for _ in range(num_merges):\n        pairs = get_pair_counts(vocab)\n        if not pairs:\n            break\n        best = max(pairs, key=pairs.get)     # most frequent adjacent pair\n        vocab = merge_pair(best, vocab)\n        merges.append(best)                  # remember the merge, in order\n    return merges\n\ncorpus = \"low low low low low lower lower newest newest widest widest\"\nmerges = learn_bpe(corpus, num_merges=10)\nfor i, m in enumerate(merges):\n    print(i, m)\n# early merges: ('e','s'), ('es','t'), ('est','</w>') ... 'est</w>' becomes one token,\n# because the suffix -est is common across 'newest' and 'widest'." },
        { type: "p", text: "**Encoding** a new word applies the learned merges in the order they were discovered: split the word into characters, then greedily apply each merge rule that still matches. A word made of frequent pieces collapses to few tokens; an exotic word stays fragmented but is still fully representable." },
        { type: "code", lang: "py", code: "def encode_word(word, merges):\n    symbols = list(word) + [\"</w>\"]\n    for a, b in merges:                     # apply merges in learned order\n        i = 0\n        while i < len(symbols) - 1:\n            if symbols[i] == a and symbols[i + 1] == b:\n                symbols[i:i + 2] = [a + b]  # merge in place\n            else:\n                i += 1\n    return symbols\n\nprint(encode_word(\"newest\", merges))   # ['new', 'est</w>']  -> 2 tokens\nprint(encode_word(\"lowest\", merges))   # reuses 'est</w>' even though 'lowest'\n                                       # never appeared in training" },
        { type: "callout", variant: "tip", text: "**This is not a toy.** Byte-level BPE — exactly this algorithm run over UTF-8 bytes — is the tokenizer inside GPT-2/3/4 and Llama. When people say *\"an LLM costs $3 per million tokens\"* or *\"the context window is 128k tokens,\"* these are those tokens. A useful rule for English: **~4 characters ≈ 1 token ≈ ¾ of a word.** You will reuse this exact idea, unchanged, in the LLM track." },
        { type: "code", lang: "py", code: "# The production one-liner: HuggingFace's fast tokenizers, or tiktoken (OpenAI).\nfrom transformers import AutoTokenizer\n\ntok = AutoTokenizer.from_pretrained(\"gpt2\")   # byte-level BPE, vocab 50257\nids = tok.encode(\"Tokenization isn't magic: it's greedy merges.\")\nprint(ids)                       # [30642, 1634, 2125, 470, ...]\nprint(tok.convert_ids_to_tokens(ids))\n# ['Token', 'ization', ' isn', \"'t\", ' magic', ':', ' it', \"'s\", ' greedy', ...]\nprint(len(ids), \"tokens\")        # note ' magic' keeps its leading space" },
      ]
    },

    {
      id: "ngram",
      title: "N-gram language models",
      level: "core",
      body: [
        { type: "p", text: "A **language model (LM)** assigns a probability to a sequence of tokens — it answers *\"how likely is this sentence?\"* and, equivalently, *\"what token comes next?\"* That second framing is everything: next-token prediction is the training objective of every LLM. Before neural nets, LMs were built by *counting*, and the counting model — the **n-gram** — is the cleanest possible introduction to the whole idea." },
        { type: "heading", text: "The chain rule of probability, applied to text" },
        { type: "p", text: "The probability of a sentence $w_1 w_2 \\dots w_n$ factorizes **exactly** by the chain rule of probability (see the Probability page). No approximation yet — this is just the definition of joint probability:" },
        { type: "math", tex: String.raw`P(w_1, w_2, \dots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \dots, w_{i-1})` },
        { type: "p", text: "In words: the probability of the whole sentence is the product of the probability of each word given everything before it. This is beautiful and useless as written — to estimate $P(w_i \\mid w_1,\\dots,w_{i-1})$ you would need to have seen that exact prefix, and almost every long prefix appears **zero** times in any corpus. The full history is too specific." },
        { type: "heading", text: "The Markov assumption" },
        { type: "p", text: "The fix is a deliberate, brutal simplification: assume each word depends only on the **previous $k-1$ words**, not the entire history. This is the **Markov assumption**, and it turns an impossible estimation problem into a countable one:" },
        { type: "math", tex: String.raw`P(w_i \mid w_1, \dots, w_{i-1}) \;\approx\; P(w_i \mid w_{i-k+1}, \dots, w_{i-1})` },
        { type: "p", text: "A model with this window size $k$ is a **$k$-gram model**. The names: $k=1$ **unigram** (each word independent), $k=2$ **bigram** (condition on one previous word), $k=3$ **trigram** (two previous words). For a bigram model the whole sentence probability collapses to a chain of pairwise terms:" },
        { type: "math", tex: String.raw`P(w_1, \dots, w_n) \approx \prod_{i=1}^{n} P(w_i \mid w_{i-1})` },
        { type: "heading", text: "Maximum-likelihood estimation: just count" },
        { type: "p", text: "How do we get $P(w_i \\mid w_{i-1})$? The **maximum-likelihood estimate (MLE)** is exactly what your intuition suggests — count how often the pair occurs, divided by how often the context occurs:" },
        { type: "math", tex: String.raw`P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{\operatorname{count}(w_{i-1}, w_i)}{\operatorname{count}(w_{i-1})}` },
        { type: "p", text: "For a general $n$-gram, condition on the $(n{-}1)$-word prefix and divide the $n$-gram count by the prefix count. We add sentence-boundary markers `<s>` and `</s>` so the model can learn which words *start* and *end* sentences. That is a complete, working language model — a few lines of counting." },
        { type: "code", lang: "py", code: "from collections import Counter, defaultdict\n\ndef train_bigram(sentences):\n    unigrams = Counter()\n    bigrams = Counter()\n    for sent in sentences:\n        toks = [\"<s>\"] + sent.split() + [\"</s>\"]\n        unigrams.update(toks)\n        bigrams.update(zip(toks, toks[1:]))     # adjacent pairs\n    return unigrams, bigrams\n\ndef bigram_prob(w_prev, w, unigrams, bigrams):\n    if unigrams[w_prev] == 0:\n        return 0.0\n    return bigrams[(w_prev, w)] / unigrams[w_prev]   # MLE\n\ncorpus = [\"the cat sat\", \"the cat ran\", \"the dog sat\"]\nuni, bi = train_bigram(corpus)\nprint(bigram_prob(\"the\", \"cat\", uni, bi))   # 2/3 = 0.666...\nprint(bigram_prob(\"the\", \"dog\", uni, bi))   # 1/3 = 0.333...\nprint(bigram_prob(\"cat\", \"sat\", uni, bi))   # 1/2 = 0.5" },
        { type: "heading", text: "The data-sparsity problem" },
        { type: "p", text: "Now the trouble. Language is productive, so most valid word combinations simply never appear in your training corpus. Any bigram you didn't see gets $\\operatorname{count}=0$, hence probability **0**. And because a sentence's probability is a *product* of these terms, a single unseen bigram makes the **entire sentence probability 0** — even a perfectly grammatical sentence. Worse, you cannot take its log (needed for perplexity): $\\log 0 = -\\infty$." },
        { type: "callout", variant: "gotcha", text: "**Sparsity gets exponentially worse with $n$.** A trigram model has vastly more possible contexts than a bigram, so far more of them are unseen. This is the fundamental tension of n-gram models: **larger $n$ captures more context but sees exponentially sparser data.** In practice n-grams top out around trigrams/4-grams. Escaping this ceiling — modeling long context *without* exponential data cost — is precisely what neural language models were invented to do, and why the field moved on." },
      ]
    },

    {
      id: "smoothing",
      title: "Smoothing: making room for the unseen",
      level: "core",
      body: [
        { type: "p", text: "The zero-probability catastrophe is not an edge case — it is the *normal* case for n-grams. **Smoothing** is the family of techniques that fixes it, all built on one principle: **steal a little probability mass from the events you saw and redistribute it to the events you didn't.** No count is ever allowed to be exactly zero." },
        { type: "heading", text: "Laplace (add-one) and add-k smoothing" },
        { type: "p", text: "The simplest fix: pretend you saw every possible n-gram one extra time. Add 1 to every count. To keep it a valid probability distribution, the denominator must grow by the vocabulary size $V$ (one added count for each possible next word):" },
        { type: "math", tex: String.raw`P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{\operatorname{count}(w_{i-1}, w_i) + 1}{\operatorname{count}(w_{i-1}) + V}` },
        { type: "p", text: "Now nothing is zero. Add-one is heavy-handed, though — with a large vocabulary it hands far too much mass to the sea of unseen n-grams and badly distorts the ones you *did* see. **Add-$k$** softens it by adding a fractional $k$ (say $0.05$) instead of $1$:" },
        { type: "math", tex: String.raw`P_{\text{add-}k}(w_i \mid w_{i-1}) = \frac{\operatorname{count}(w_{i-1}, w_i) + k}{\operatorname{count}(w_{i-1}) + k\,V}` },
        { type: "code", lang: "py", code: "def add_k_prob(w_prev, w, unigrams, bigrams, V, k=1.0):\n    return (bigrams[(w_prev, w)] + k) / (unigrams[w_prev] + k * V)\n\ncorpus = [\"the cat sat\", \"the cat ran\", \"the dog sat\"]\nuni, bi = train_bigram(corpus)      # from the previous section\nV = len(set(w for w in uni))        # vocabulary size (incl. <s>, </s>)\n\n# 'the mouse' was never seen -> MLE gives 0, but Laplace gives a small mass:\nprint(add_k_prob(\"the\", \"mouse\", uni, bi, V, k=1.0))   # small but nonzero\nprint(add_k_prob(\"the\", \"cat\",   uni, bi, V, k=1.0))   # discounted from 2/3" },
        { type: "heading", text: "Backoff and interpolation" },
        { type: "p", text: "Add-$k$ treats all unseen n-grams alike, which is crude. Smarter idea: if you have never seen the trigram, **back off** to the bigram; if you have not seen the bigram, back off to the unigram. Lower-order models are less specific but far better estimated." },
        { type: "list", ordered: false, items: [
          "**Backoff (Katz):** use the highest-order n-gram that has a nonzero count; drop to a shorter context only when forced, with a discount so the total still sums to 1.",
          "**Interpolation:** always blend all orders together with weights $\\lambda$, whether or not the higher-order count exists.",
        ]},
        { type: "math", tex: String.raw`P_{\text{interp}}(w_i \mid w_{i-2}, w_{i-1}) = \lambda_3 P(w_i \mid w_{i-2}, w_{i-1}) + \lambda_2 P(w_i \mid w_{i-1}) + \lambda_1 P(w_i), \quad \sum_j \lambda_j = 1` },
        { type: "heading", text: "Kneser–Ney (the gold standard, in one idea)" },
        { type: "p", text: "**Kneser–Ney** smoothing was the best-performing n-gram method before neural LMs, and its key insight is genuinely clever. Instead of backing off to how *frequent* a word is, back off to how *many distinct contexts* it appears in — its **continuation probability**. The textbook example: *\"Francisco\"* is a frequent word, but it appears almost only after *\"San\"*. A plain frequency-based backoff would wrongly predict *\"Francisco\"* is a likely word in general; Kneser–Ney notices it has very few distinct preceding contexts and down-weights it. It combines this with **absolute discounting** — subtract a fixed $d$ from every observed count and redistribute that mass via the continuation distribution." },
        { type: "math", tex: String.raw`P_{\text{KN}}(w_i \mid w_{i-1}) = \frac{\max(\operatorname{count}(w_{i-1}, w_i) - d,\, 0)}{\operatorname{count}(w_{i-1})} + \lambda(w_{i-1})\, P_{\text{cont}}(w_i)` },
        { type: "callout", variant: "note", text: "You will rarely implement Kneser–Ney by hand today — libraries like NLTK and KenLM do it, and neural models replaced it for most tasks. But the *idea* — that a word's predictive value comes from the diversity of contexts it appears in, not raw frequency — is a distant ancestor of what word embeddings learn automatically. Keep it in mind for the Representations section." },
      ]
    },

    {
      id: "perplexity",
      title: "Evaluating language models: perplexity",
      level: "core",
      body: [
        { type: "p", text: "How do you tell if one language model is better than another? You measure how *surprised* it is by real, held-out text. A good model assigns high probability to text that actually occurs — it is rarely surprised. The standard metric that formalizes \"surprise\" is **perplexity**, and it is *still* the primary intrinsic metric for LLMs in 2026. Let us derive it properly, because it comes straight out of information theory." },
        { type: "heading", text: "From cross-entropy to perplexity" },
        { type: "p", text: "Start with **cross-entropy** (see the Probability page): the average number of bits (or nats) needed to encode the true data using the model's predicted distribution. For a test sequence of $N$ tokens, the per-token cross-entropy under model $P$ is the average negative log-probability the model assigned to the tokens that actually appeared:" },
        { type: "math", tex: String.raw`H = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i \mid w_{<i})` },
        { type: "p", text: "Cross-entropy is *exactly* the training loss of a neural language model (the cross-entropy / negative-log-likelihood loss). **Perplexity** is just its exponential — it converts \"bits of surprise\" into an interpretable \"effective branching factor\":" },
        { type: "math", tex: String.raw`\text{Perplexity}(W) = 2^{H} = \left( \prod_{i=1}^{N} P(w_i \mid w_{<i}) \right)^{-\frac{1}{N}}` },
        { type: "p", text: "The right-hand form makes the meaning vivid: perplexity is the **inverse geometric mean of the per-token probabilities**. If a model assigns probability $\\frac{1}{p}$ to each word on average, its perplexity is $p$. Lower is better. And using natural log with base $e$ instead of $\\log_2 / 2^{H}$ gives the identical number — the base cancels — so in practice everyone computes $\\exp$ of the mean negative log-likelihood." },
        { type: "heading", text: "Interpreting the number" },
        { type: "p", text: "Perplexity has a wonderfully concrete reading: it is the **effective number of equally-likely choices** the model faces at each step. Perplexity 100 means the model is, on average, as confused as if it were guessing uniformly among 100 words. Perplexity 20 means it has narrowed the field to ~20. A perfect model that always assigned probability 1 to the truth would have perplexity 1; a uniform model over a vocabulary of size $V$ has perplexity exactly $V$." },
        { type: "table",
          headers: ["Model", "Typical word-level perplexity (Penn Treebank-ish)", "Reading"],
          rows: [
            ["Uniform guess", "$\\approx V$ (~10,000+)", "maximally confused"],
            ["Good n-gram (smoothed trigram)", "~140", "usable but shallow"],
            ["LSTM (2016)", "~80", "neural context helps"],
            ["Transformer LM (modern)", "~20 or below", "far less surprised"],
          ]
        },
        { type: "code", lang: "py", code: "import math\n\ndef perplexity(test_tokens, prob_fn):\n    \"\"\"prob_fn(prev, w) -> P(w | prev), must be > 0 (use a smoothed model!).\"\"\"\n    log_sum = 0.0\n    N = 0\n    prev = \"<s>\"\n    for w in test_tokens + [\"</s>\"]:\n        p = prob_fn(prev, w)\n        log_sum += math.log(p)      # natural log of the model's probability\n        N += 1\n        prev = w\n    cross_entropy = -log_sum / N     # mean negative log-likelihood (the loss)\n    return math.exp(cross_entropy)   # perplexity = exp(cross-entropy)\n\n# NB: an *unsmoothed* model that ever returns 0 makes log blow up to -inf and\n# perplexity -> inf. Perplexity is why smoothing is non-negotiable." },
        { type: "callout", variant: "gotcha", text: "**Perplexity is only comparable across models with the same tokenization and vocabulary.** A character-level model and a word-level model produce wildly different perplexities on the same text — they are counting different units. This is exactly why comparing LLMs by raw perplexity is fraught: different tokenizers, different numbers. When you read a perplexity number, always ask *\"per what token, over what test set?\"*" },
      ]
    },

    {
      id: "tasks",
      title: "The classic NLP task zoo",
      level: "core",
      body: [
        { type: "p", text: "Language modeling is the backbone, but \"NLP\" in practice is a menagerie of concrete tasks. Knowing the vocabulary matters — these names recur constantly, and modern LLMs are often just a general model pointed at one of these classic problems. The two broad shapes are **sequence classification** (one label for a whole span) and **sequence labeling** (one label per token)." },
        { type: "table",
          headers: ["Task", "What it does", "Example"],
          rows: [
            ["**Text classification**", "one label for a whole document", "spam / not-spam; topic; language ID"],
            ["**Sentiment analysis**", "classification into affect polarity", "*\"loved it\"* → positive"],
            ["**POS tagging**", "grammatical tag per token (sequence labeling)", "*\"dog\"* → NOUN, *\"runs\"* → VERB"],
            ["**Named-entity recognition (NER)**", "find & type spans: people, places, orgs, dates", "*\"[Ada]PER works at [Google]ORG\"*"],
            ["**Parsing**", "recover grammatical structure (tree of dependencies)", "which word modifies which"],
            ["**Coreference resolution**", "link mentions that refer to the same entity", "*\"Ada … she … the engineer\"*"],
            ["**Machine translation**", "map a sequence in one language to another", "EN → FR (a seq-to-seq task)"],
            ["**Question answering**", "return an answer span or text for a question", "SQuAD, retrieval-augmented QA"],
          ]
        },
        { type: "callout", variant: "note", text: "**NER and POS use a labeling scheme called BIO** (Begin/Inside/Outside): each token is tagged `B-PER` (begins a person), `I-PER` (inside one), or `O` (outside any entity). This turns span-finding into per-token classification, which is exactly what a sequence model can output. You will see this scheme again when fine-tuning transformers for token classification." },
        { type: "heading", text: "The classic pipeline: spaCy and NLTK" },
        { type: "p", text: "Two libraries define the pre-LLM toolbox. **NLTK** is the teaching classic — explicit, granular, great for learning (tokenizers, stemmers, corpora, n-gram tools). **spaCy** is the production workhorse — a fast, opinionated pipeline that runs tokenization → POS → parse → NER in one call, with pretrained statistical/neural models. In a spaCy pipeline, each component annotates the shared document object and passes it on:" },
        { type: "code", lang: "py", code: "import spacy\n\nnlp = spacy.load(\"en_core_web_sm\")     # loads tokenizer + tagger + parser + NER\ndoc = nlp(\"Ada Lovelace wrote the first algorithm in London in 1843.\")\n\nfor tok in doc[:4]:\n    print(f\"{tok.text:12} {tok.pos_:6} {tok.lemma_:10} stop={tok.is_stop}\")\n# Ada          PROPN  Ada        stop=False\n# Lovelace     PROPN  Lovelace   stop=False\n# wrote        VERB   write      stop=False   <- lemmatized\n# the          DET    the        stop=True\n\nfor ent in doc.ents:                  # named entities, already typed\n    print(ent.text, \"->\", ent.label_)\n# Ada Lovelace -> PERSON\n# first        -> ORDINAL\n# London       -> GPE   (geo-political entity)\n# 1843         -> DATE" },
        { type: "callout", variant: "tip", text: "**These libraries are not obsolete in the LLM era.** For fast, cheap, deterministic, offline processing of millions of documents — tokenizing, sentence-splitting, entity extraction — spaCy is often the right tool over an expensive LLM call. Use the LLM when you need reasoning or open-ended understanding; use spaCy/NLTK when you need speed and structure. Knowing both, and when to reach for which, is the mark of a practical NLP engineer." },
      ]
    },

    {
      id: "to-meaning",
      title: "From counting to meaning",
      level: "core",
      body: [
        { type: "p", text: "Step back and notice what every method on this page has in common: they all treat words as **atomic, discrete symbols**. To an n-gram model, `\"cat\"` is token #4123 and `\"dog\"` is token #7781 — two integers with no relationship. The model has no way to know that a sentence about cats should inform its predictions about dogs. This is the **discreteness** problem from the very first section, and it is the ceiling that count-based NLP cannot break through." },
        { type: "p", text: "Concretely, the failures are structural, not fixable by more data:" },
        { type: "list", ordered: false, items: [
          "**No generalization across words.** Having seen *\"buy a car\"* a thousand times teaches an n-gram *nothing* about *\"purchase an automobile\"* — different symbols entirely.",
          "**No notion of similarity.** There is no sense in which `cat` is closer to `dog` than to `democracy`. Every pair of distinct words is equidistant.",
          "**The curse of dimensionality.** Representing a word as a one-hot vector over a 50,000-word vocabulary is a 50,000-dimensional vector that is all zeros except one 1. It is enormous, sparse, and carries zero similarity structure.",
        ]},
        { type: "p", text: "The escape is one of the most important ideas in all of machine learning: stop representing a word as an index, and start representing it as a **dense vector** — a few hundred real numbers — positioned so that words used in similar contexts land near each other in space. Suddenly `cat` and `dog` are neighbors, arithmetic like *king − man + woman ≈ queen* becomes possible, and similarity is just a dot product (recall cosine similarity from Linear Algebra)." },
        { type: "callout", variant: "note", text: "This is the bridge out of classical NLP. The principle that makes it work is the **distributional hypothesis** — *\"you shall know a word by the company it keeps\"* (Firth, 1957) — the same insight that powered Kneser–Ney's continuation counts, now pushed to its logical end. The **Representations** section (word2vec, GloVe, contextual embeddings) builds these dense vectors from scratch, and from there the road runs straight to the **Transformers** section and the LLM track. Everything you built here — tokenization, next-token prediction, perplexity — comes with you unchanged; only the *representation* gets an upgrade." },
      ]
    },

    {
      id: "worked",
      title: "Worked example: bigram model + BPE + perplexity",
      level: "core",
      body: [
        { type: "p", text: "Let us tie the page together end-to-end: train a **BPE tokenizer**, build a **smoothed bigram model** over its tokens, and evaluate it with **perplexity** on held-out text. This is a complete, if miniature, statistical NLP system — every piece from this page in ~40 lines of pure Python." },
        { type: "code", lang: "py", code: "import math\nfrom collections import Counter\n\n# --- 1) Train a tiny BPE tokenizer (reusing learn_bpe / encode_word from earlier) ---\ntrain_text = \"the cat sat on the mat the cat ran to the mat the dog sat\"\nmerges = learn_bpe(train_text, num_merges=20)\n\ndef tokenize(text, merges):\n    toks = []\n    for word in text.split():\n        toks += encode_word(word, merges)   # subword tokens, incl. '</w>' markers\n    return toks\n\n# --- 2) Train a smoothed bigram model over the *subword* tokens ---\ntrain_toks = [\"<s>\"] + tokenize(train_text, merges) + [\"</s>\"]\nunigrams = Counter(train_toks)\nbigrams  = Counter(zip(train_toks, train_toks[1:]))\nV = len(unigrams)\n\ndef bigram_prob(prev, w, k=0.1):\n    return (bigrams[(prev, w)] + k) / (unigrams[prev] + k * V)   # add-k smoothing\n\n# --- 3) Evaluate perplexity on held-out text ---\ntest_toks = [\"<s>\"] + tokenize(\"the cat sat on the mat\", merges) + [\"</s>\"]\nlog_sum, N, prev = 0.0, 0, test_toks[0]\nfor w in test_toks[1:]:\n    log_sum += math.log(bigram_prob(prev, w))\n    N += 1\n    prev = w\nppl = math.exp(-log_sum / N)\nprint(f\"tokens: {len(test_toks)}   perplexity: {ppl:.2f}\")\n# Low-ish perplexity: the test sentence overlaps the training patterns, and\n# smoothing guarantees no zero probabilities blow it up to infinity." },
        { type: "p", text: "Notice how the pieces compose: BPE decides the *units*, the bigram counts model the *transitions* between units, add-$k$ smoothing guarantees *every* probability is positive (so perplexity is finite), and perplexity gives one number to compare models. Swap the bigram counter for a neural network that predicts the next token and you have — structurally — a language model in the modern sense. The pipeline is identical; only the estimator changes." },
        { type: "callout", variant: "good", text: "**Try this to feel the concepts.** (1) Increase `num_merges` and watch tokens get longer and fewer. (2) Set `k=0` (unsmoothed) and feed a test sentence with an unseen bigram — perplexity becomes `inf`. (3) Compare bigram vs unigram perplexity on the same test set: more context should lower perplexity, right up until sparsity bites." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Build at least two of these by hand in pure Python before touching a library. Statistical NLP is small enough to implement completely, and doing so makes the neural versions feel like natural upgrades rather than magic." },
        { type: "list", ordered: true, items: [
          "**BPE tokenizer from scratch.** Implement `learn_bpe` and `encode` (as above), train it on a chapter of a public-domain book, and plot tokens-per-word as you vary the number of merges. Then load the real `gpt2` tokenizer with `tiktoken` or HuggingFace and compare how each splits rare words and your own name. Confirm the ~4-chars-per-token rule of thumb.",
          "**N-gram language model + text generation.** Train unigram, bigram, and trigram models on a corpus. Then *sample* from each (pick the next token from its conditional distribution) and read the output — watch fluency rise from gibberish (unigram) to eerily-plausible-but-nonsensical (trigram). This is the ancestor of LLM generation.",
          "**Perplexity shoot-out.** Split your corpus into train/test. Implement add-$k$ smoothing and compute test perplexity for bigram vs trigram across several $k$ values. Plot perplexity vs $k$ and find the sweet spot. Observe the trigram beating the bigram until data sparsity flips it.",
          "**A from-scratch preprocessing pipeline.** Write `normalize()`, then a comparison harness that measures vocabulary size and a downstream metric (e.g. Naive Bayes spam accuracy) with vs without lowercasing, stemming, and stop-word removal. Empirically discover when each step helps and when it hurts.",
          "**Sentiment classifier, counting only.** Build a bag-of-words + Naive Bayes sentiment classifier on movie reviews using only `Counter`. Then break it on purpose with a negation (*\"not good\"*) and see the failure — motivating why sequence models exist.",
          "**spaCy vs LLM bake-off.** Extract named entities from 1,000 news sentences with spaCy and with an LLM API. Compare speed, cost, and accuracy on a hand-labeled sample. Decide, with data, when each is the right tool.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "This page is self-contained, but these are the canonical companions — the first one is the bible of the field and is free online:" },
        { type: "link", url: "https://web.stanford.edu/~jurafsky/slp3/", text: "Jurafsky & Martin — Speech and Language Processing (3rd ed., free draft). The definitive textbook; chapters on n-grams, tokenization, and perplexity are the direct source for this page." },
        { type: "link", url: "https://arxiv.org/abs/1508.07909", text: "Sennrich, Haddow & Birch (2016) — Neural Machine Translation of Rare Words with Subword Units. The paper that repurposed BPE for NLP; short and readable." },
        { type: "link", url: "https://github.com/karpathy/minbpe", text: "Andrej Karpathy — minbpe. A clean, minimal, heavily-commented BPE implementation (and companion video) — the perfect next step after building your own." },
        { type: "link", url: "https://github.com/openai/tiktoken", text: "OpenAI tiktoken — the fast byte-level BPE tokenizer used by GPT models. Inspect exactly how your prompts are split (and billed)." },
        { type: "link", url: "https://huggingface.co/learn/nlp-course/chapter6", text: "HuggingFace NLP Course — Chapter 6: Tokenizers. Hands-on tour of BPE, WordPiece, and Unigram with the `tokenizers` library." },
        { type: "link", url: "https://course.spacy.io/", text: "Advanced NLP with spaCy (free interactive course) — the practical pipeline library, taught by its creators." },
        { type: "link", url: "https://nlp.stanford.edu/fsnlp/", text: "Manning & Schütze — Foundations of Statistical NLP. The classic deep reference for the counting-based era (smoothing, Kneser–Ney)." },
      ]
    },
  ],

  packages: [
    { name: "tiktoken", why: "OpenAI's fast byte-level BPE tokenizer — see exactly how GPT splits (and bills) your text" },
    { name: "tokenizers", why: "HuggingFace's Rust-backed trainers for BPE / WordPiece / Unigram" },
    { name: "transformers", why: "`AutoTokenizer` gives you any pretrained model's exact tokenizer in one line" },
    { name: "sentencepiece", why: "language-agnostic subword tokenization (BPE + Unigram) used by T5, Llama, and multilingual models" },
    { name: "nltk", why: "the teaching classic — tokenizers, stemmers, corpora, and n-gram/smoothing tools" },
    { name: "spacy", why: "fast production pipeline: tokenization → POS → parse → NER in one call" },
    { name: "collections.Counter", why: "the entire n-gram toolkit — counting pairs and contexts is 90% of statistical NLP" },
  ],

  gotchas: [
    "Unicode normalize (NFC) before anything else — `café` can be two different byte strings that compare **unequal** yet look identical.",
    "Removing stop words destroys negation and questions: `not good` → `good` flips your sentiment label. Never strip stop words for sequence/neural models.",
    "An unsmoothed n-gram gives probability **0** to any unseen n-gram, and one zero makes the whole sentence probability 0 and $\\log 0 = -\\infty$. Smoothing is mandatory, not optional.",
    "Larger $n$ captures more context but sees exponentially sparser data — n-grams top out around trigrams; this ceiling is *why* neural LMs exist.",
    "Perplexity is only comparable across models with the **same tokenization and vocabulary** — char-level and word-level perplexities are not the same units.",
    "Byte-level BPE (GPT/Llama) keeps the leading space on tokens (`' magic'` ≠ `'magic'`) — a frequent source of off-by-one token-counting confusion.",
    "Stemming can merge unrelated words (`universe`, `university` → `univers`); prefer lemmatization when meaning matters, and remember it needs the part of speech.",
    "The rough conversion **~4 characters ≈ 1 token ≈ ¾ word** (English) is what LLM context windows and pricing are actually measured in — internalize it.",
  ],

  flashcards: [
    { q: "Why is language fundamentally hard for computers? Name three reasons.", a: "Ambiguity (one string, many meanings), compositionality (meaning built recursively, e.g. `not bad`), long-range dependence, world knowledge, and discreteness (symbols with no built-in similarity)." },
    { q: "What problem does subword tokenization (BPE) solve that word-level tokenization cannot?", a: "The out-of-vocabulary (OOV) problem: rare/unseen words are broken into known subword pieces instead of collapsing to `<UNK>`, giving a bounded vocab, no true OOV, and shared morphology." },
    { q: "State the BPE training algorithm in one sentence.", a: "Start with characters, then repeatedly find the most frequent adjacent pair of symbols in the corpus and merge it into a new single token; the number of merges sets the vocabulary size." },
    { q: "Write the bigram approximation of a sentence's probability.", a: "$P(w_1,\\dots,w_n) \\approx \\prod_i P(w_i \\mid w_{i-1})$ — the chain rule under the Markov assumption that each word depends only on the previous one." },
    { q: "What is the maximum-likelihood estimate of a bigram probability?", a: "$P(w_i \\mid w_{i-1}) = \\operatorname{count}(w_{i-1}, w_i) / \\operatorname{count}(w_{i-1})$ — just counts." },
    { q: "Why is smoothing necessary for n-gram models?", a: "Most valid n-grams never appear in training, so MLE gives them probability 0; a single zero makes the whole sentence probability 0 and its log $-\\infty$. Smoothing moves mass from seen to unseen events." },
    { q: "Write add-k (Laplace) smoothing for a bigram.", a: "$P(w_i \\mid w_{i-1}) = (\\operatorname{count}(w_{i-1},w_i) + k) / (\\operatorname{count}(w_{i-1}) + kV)$, where $V$ is the vocabulary size." },
    { q: "Define perplexity in terms of cross-entropy.", a: "Perplexity $= 2^{H}$ (or $e^H$) where $H = -\\frac1N \\sum_i \\log P(w_i \\mid w_{<i})$ is the per-token cross-entropy. It is the inverse geometric mean of per-token probabilities." },
    { q: "How do you interpret a perplexity of 20?", a: "The model is on average as uncertain as if choosing uniformly among 20 equally-likely next tokens — the effective branching factor. Lower is better; 1 is perfect." },
    { q: "What is the key idea of Kneser-Ney smoothing?", a: "Back off using how many *distinct contexts* a word appears in (its continuation probability), not its raw frequency — so `Francisco` (almost always after `San`) is correctly rated unlikely in general." },
    { q: "What is the distributional hypothesis and why does it matter?", a: "'You shall know a word by the company it keeps' (Firth). Words in similar contexts have similar meaning — the basis for dense word embeddings that replace discrete symbols." },
    { q: "Why can't count-based n-gram models generalize across words?", a: "They treat words as atomic symbols with no similarity structure — seeing `buy a car` teaches nothing about `purchase an automobile`. Dense embeddings fix this by giving words geometry." },
  ],

  cheatsheet: [
    { label: "Unicode normalize", code: "unicodedata.normalize('NFC', text)" },
    { label: "Lowercase + collapse ws", code: "' '.join(text.lower().split())" },
    { label: "Character tokens", code: "list(word) + ['</w>']" },
    { label: "GPT tokenizer", code: "tiktoken.get_encoding('cl100k_base').encode(s)" },
    { label: "HF tokenizer", code: "AutoTokenizer.from_pretrained('gpt2').encode(s)" },
    { label: "Bigram counts", code: "Counter(zip(toks, toks[1:]))" },
    { label: "Bigram MLE", code: "bigrams[(a,b)] / unigrams[a]" },
    { label: "Add-k smoothing", code: "(bigrams[(a,b)] + k) / (unigrams[a] + k*V)" },
    { label: "Cross-entropy", code: "-sum(log P(w|ctx)) / N" },
    { label: "Perplexity", code: "math.exp(cross_entropy)" },
    { label: "spaCy pipeline", code: "nlp = spacy.load('en_core_web_sm'); doc = nlp(text)" },
    { label: "Named entities", code: "[(e.text, e.label_) for e in doc.ents]" },
    { label: "Lemmatize (spaCy)", code: "[t.lemma_ for t in doc]" },
    { label: "Rule of thumb", code: "~4 chars ≈ 1 token ≈ 0.75 word" },
  ],
});
