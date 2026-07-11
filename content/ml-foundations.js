(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "ml-foundations",
  name: "ML Foundations & Workflow",
  language: "Classical ML",
  group: "Classical Machine Learning",
  navLabel: "Foundations & Workflow",
  tagline: "What machine learning actually *is* — the taxonomy, the supervised setup, the bias–variance tradeoff derived from scratch, and the end-to-end workflow that keeps you honest.",
  color: "#10B981",
  readMinutes: 46,
  sections: [
    {
      id: "what-is-ml",
      title: "What machine learning actually is",
      level: "core",
      body: [
        { type: "p", text: "For most of computing history, to make a computer do something you **wrote the rules**. To detect spam, you'd hand-write conditions: *if the subject contains 'FREE MONEY' and the sender is unknown, flag it*. This works until the world pushes back — spammers write 'FR33 M0N3Y', and your rulebook grows into an unmaintainable thousand-line thicket that still misses things." },
        { type: "p", text: "**Machine learning flips the arrows.** Instead of writing the rules, you show the computer thousands of examples labeled *spam* / *not spam*, and an algorithm *infers the rules for you*. You program the **learning procedure** once; the specific decision logic is discovered from data. When the data shifts, you retrain instead of rewrite." },
        { type: "math", tex: String.raw`\text{Classical:}\;\; \underbrace{\text{rules}}_{\text{you write}} + \text{data} \to \text{answers} \qquad\qquad \text{ML:}\;\; \underbrace{\text{data} + \text{answers}}_{\text{you provide}} \to \text{rules}` },
        { type: "heading", text: "A little history" },
        { type: "p", text: "The phrase **“machine learning”** was coined by **Arthur Samuel in 1959** at IBM, who built a checkers program that improved by playing thousands of games against itself — famously beating its own creator. His informal definition set the tone: *the field of study that gives computers the ability to learn without being explicitly programmed*. The field then swung through symbolic AI, expert systems, and an “AI winter,” before statistical learning (1990s), then deep learning (2012 onward), then large language models (2018 onward) each reignited it. The classical-ML techniques in this track — linear models, trees, SVMs, clustering — were forged in that statistical-learning era and remain the correct tool for the majority of tabular, small-to-medium-data problems you'll meet in industry." },
        { type: "heading", text: "The definition worth memorizing (Mitchell, 1997)" },
        { type: "p", text: "**Tom Mitchell** gave the definition that actually operationalizes the idea — it forces you to name three things before you claim to be “doing ML”:" },
        { type: "callout", variant: "note", text: "*“A computer program is said to **learn** from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$.”*" },
        { type: "table",
          headers: ["Symbol", "Meaning", "Spam-filter example"],
          rows: [
            ["$T$ — task", "what you want done", "classify an email as spam / not spam"],
            ["$P$ — performance measure", "how you score it", "fraction of emails classified correctly (or F1)"],
            ["$E$ — experience", "the data it learns from", "10,000 emails a human already labeled"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Use this as a checklist.** Before starting any ML project, write down $T$, $P$, and $E$ in one sentence each. If you can't, you don't yet have a machine-learning problem — you have a *wish*. Ninety percent of failed ML projects failed here: a fuzzy task, an un-measurable performance metric, or no data." },
        { type: "callout", variant: "gotcha", text: "ML is not magic and not always the answer. If a problem has clear, stable rules (compute sales tax; validate an email format), **write the rules** — a `regex` is faster, cheaper, debuggable, and correct. Reach for ML when the rules are unknown, too numerous, or constantly changing." },
      ]
    },

    {
      id: "taxonomy",
      title: "The taxonomy: five ways a machine can learn",
      level: "core",
      body: [
        { type: "p", text: "ML methods are grouped by **what kind of signal supervises the learning** — specifically, how much labeled ground truth you have. This is the single most useful map of the field; every algorithm you meet lives in one of these boxes." },
        { type: "table",
          headers: ["Paradigm", "What it's given", "What it learns", "Canonical examples"],
          rows: [
            ["**Supervised**", "inputs $X$ **and** labels $y$", "a mapping $X \\to y$", "regression, classification, spam filters, price prediction"],
            ["**Unsupervised**", "inputs $X$ only, no labels", "structure / grouping / density", "clustering, PCA, anomaly detection, topic modeling"],
            ["**Self-supervised**", "unlabeled $X$; labels *created from* $X$", "reusable representations", "word2vec, masked-language-model pretraining (BERT/GPT)"],
            ["**Semi-supervised**", "a little labeled + lots unlabeled", "$X \\to y$ using both", "medical imaging where labels are expensive"],
            ["**Reinforcement**", "states, actions, a reward signal", "a policy that maximizes reward", "game agents, robotics, RLHF for LLMs"],
          ]
        },
        { type: "heading", text: "How to pick the right box" },
        { type: "list", ordered: false, items: [
          "**Do you have labeled examples of the answer?** Yes → **supervised** (this is 90% of applied classical ML). This whole track after this file is mostly supervised.",
          "**No labels, and you want to discover structure** (segments of customers, compress features, spot outliers)? → **unsupervised**.",
          "**A giant pile of unlabeled text/images and you want general-purpose features?** → **self-supervised** — invent a pretext task (predict the next word, fill in a blanked-out patch) so the data labels itself. This is how every modern LLM is pretrained.",
          "**Labels exist but are scarce/expensive, and unlabeled data is cheap?** → **semi-supervised** (or label a bit, pretrain self-supervised, then fine-tune).",
          "**An agent takes actions in an environment and you can only score outcomes, not individual steps?** → **reinforcement learning** (its own track later).",
        ]},
        { type: "callout", variant: "note", text: "**Self-supervised learning is the quiet revolution.** It is technically supervised learning where the labels are *derived from the input itself* — no human annotation. “Predict the next word” turns the entire internet into a labeled dataset for free. That trick is why LLMs exist. You'll meet it properly in the NLP and LLM tracks; for now, just know it lives on the map." },
        { type: "callout", variant: "gotcha", text: "These boxes are not walls. Real systems chain them: an LLM is **self-supervised** pretraining → **supervised** fine-tuning → **reinforcement** learning from human feedback (RLHF). Knowing the taxonomy lets you read that sentence and understand the whole pipeline." },
      ]
    },

    {
      id: "supervised-setup",
      title: "The supervised setup, formally",
      level: "core",
      body: [
        { type: "p", text: "Supervised learning is where you'll spend most of this track, so let's nail the formalism. It's simple, and having the exact vocabulary makes every later algorithm — linear regression, trees, SVMs, neural nets — read as *filling in the same template*." },
        { type: "heading", text: "The ingredients" },
        { type: "p", text: "You have a dataset of $m$ **examples**, each a pair of a **feature vector** $x^{(i)} \\in \\mathbb{R}^n$ and a **target** (label) $y^{(i)}$. Stacked, the features form the design matrix $X \\in \\mathbb{R}^{m \\times n}$ and the targets a vector $y$." },
        { type: "math", tex: String.raw`\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{m}, \qquad x^{(i)} \in \mathbb{R}^n, \qquad y^{(i)} \in \mathcal{Y}` },
        { type: "table",
          headers: ["If the target space $\\mathcal{Y}$ is…", "the task is…", "example"],
          rows: [
            ["continuous, $\\mathcal{Y} = \\mathbb{R}$", "**regression**", "predict a house price"],
            ["a finite set of classes", "**classification**", "spam / not spam; digit 0–9"],
            ["ordered categories", "ordinal regression", "star rating 1–5"],
          ]
        },
        { type: "heading", text: "The hypothesis and its parameters" },
        { type: "p", text: "We pick a family of functions — a **hypothesis class** — parameterized by $\\theta$. A single member $h_\\theta$ is our candidate model. Learning means **searching over $\\theta$** for the member that best fits the data. For linear regression, $h_\\theta(x) = \\theta^\\top x$; for a neural net, $h_\\theta$ is a stack of layers; the *shape of the search* is identical." },
        { type: "math", tex: String.raw`h_\theta : \mathbb{R}^n \to \mathcal{Y}, \qquad \hat{y} = h_\theta(x)` },
        { type: "heading", text: "The loss: how wrong is one prediction?" },
        { type: "p", text: "A **loss function** $\\ell(\\hat y, y)$ scores a single prediction against its truth — larger is worse. The choice of loss *defines* what “good” means. Two you'll use constantly:" },
        { type: "math", tex: String.raw`\underbrace{\ell(\hat y, y) = (\hat y - y)^2}_{\text{squared error (regression)}} \qquad\qquad \underbrace{\ell(\hat y, y) = -\big[y\log\hat y + (1-y)\log(1-\hat y)\big]}_{\text{binary cross-entropy (classification)}}` },
        { type: "heading", text: "Empirical risk vs. true risk — the entire game in two equations" },
        { type: "p", text: "What we *actually care about* is the **true risk**: the expected loss over the real, unknown data distribution $\\mathcal{D}$. That's the quantity that says how the model does on data it has never seen." },
        { type: "math", tex: String.raw`R(\theta) = \mathbb{E}_{(x,y)\sim \mathcal{D}}\big[\ell(h_\theta(x),\, y)\big] \qquad \text{(true risk — what we want, but can't compute)}` },
        { type: "p", text: "We can't compute that expectation — we don't have $\\mathcal{D}$, only a finite sample from it. So we approximate it by the **empirical risk**: the average loss over our actual training set." },
        { type: "math", tex: String.raw`\hat{R}(\theta) = \frac{1}{m}\sum_{i=1}^{m} \ell\big(h_\theta(x^{(i)}),\, y^{(i)}\big) \qquad \text{(empirical risk — what we can compute)}` },
        { type: "p", text: "Training a supervised model is, almost universally, **Empirical Risk Minimization (ERM)**: find the parameters that minimize the empirical risk, often plus a regularization term $\\Omega(\\theta)$ that penalizes complexity (more on that soon)." },
        { type: "math", tex: String.raw`\hat\theta = \arg\min_{\theta}\; \underbrace{\frac{1}{m}\sum_{i=1}^{m} \ell\big(h_\theta(x^{(i)}), y^{(i)}\big)}_{\text{fit the data}} \;+\; \lambda\, \underbrace{\Omega(\theta)}_{\text{stay simple}}` },
        { type: "callout", variant: "note", text: "**This single line is the skeleton of nearly every model in this deck.** Linear regression, logistic regression, SVMs, and neural networks differ only in (1) the hypothesis $h_\\theta$, (2) the loss $\\ell$, and (3) the regularizer $\\Omega$. Once you see the template, new algorithms stop being new." },
        { type: "callout", variant: "gotcha", text: "ERM optimizes empirical risk $\\hat{R}$, but you're judged on true risk $R$. The gap between them is the **generalization gap** — and closing it (not just minimizing training loss) is the real objective. A model that drives $\\hat{R}$ to zero by memorizing has learned nothing about $R$. This tension is the subject of the next three sections." },
      ]
    },

    {
      id: "data-splits",
      title: "Data splits, cross-validation & the #1 real-world bug",
      level: "core",
      body: [
        { type: "p", text: "You just saw that training loss (empirical risk) lies about real performance (true risk). The fix is procedural, not mathematical: **hold out data the model never trains on**, and estimate true risk on that. Getting this discipline right matters more than any algorithm choice." },
        { type: "heading", text: "Train / validation / test" },
        { type: "table",
          headers: ["Split", "Typical size", "Used for", "Rule"],
          rows: [
            ["**Train**", "60–80%", "fit model parameters $\\theta$", "the model sees this constantly"],
            ["**Validation**", "10–20%", "tune hyperparameters, pick models", "look at it often, but never fit $\\theta$ on it"],
            ["**Test**", "10–20%", "final, honest estimate of true risk", "touch **once**, at the very end"],
          ]
        },
        { type: "callout", variant: "warn", text: "**The test set is sacred.** Every time you look at test performance and then change *anything* — a feature, a hyperparameter, the model family — you leak information from it into your decisions, and it stops being an unbiased estimate of true risk. In effect you begin overfitting to the test set through your own choices. Split it off, lock it in a drawer, and open it once." },
        { type: "heading", text: "k-fold cross-validation — when data is scarce" },
        { type: "p", text: "A single validation split wastes data and gives a noisy estimate (you got lucky/unlucky with which rows landed in it). **k-fold cross-validation** fixes both: partition the training data into $k$ equal folds; train $k$ times, each time holding out a different fold as the validation set; average the $k$ scores. Every example is used for both training and validation, just never in the same run." },
        { type: "math", tex: String.raw`\text{CV score} = \frac{1}{k}\sum_{j=1}^{k} \text{score}\big(\text{model trained on all folds but } j,\; \text{fold } j\big)` },
        { type: "code", lang: "py", code: "from sklearn.model_selection import train_test_split, cross_val_score\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import f1_score\n\n# 1) Carve off a sacred test set FIRST — before you look at anything.\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y)  # stratify keeps class balance\n\n# 2) On the TRAINING data only, use 5-fold CV to estimate performance.\nmodel = LogisticRegression(max_iter=1000)\nscores = cross_val_score(model, X_train, y_train, cv=5, scoring=\"f1\")\nprint(f\"CV F1: {scores.mean():.3f} +/- {scores.std():.3f}\")\n\n# 3) ... iterate on features/hyperparameters using CV, NEVER the test set ...\n\n# 4) Only at the very end: fit on all training data, score ONCE on test.\nmodel.fit(X_train, y_train)\nprint(f\"FINAL test F1: {f1_score(y_test, model.predict(X_test)):.3f}\")" },
        { type: "callout", variant: "tip", text: "Use `stratify=y` for classification so each split preserves the class ratios — otherwise a rare class can vanish from a fold. For **time series**, do **not** shuffle: use `TimeSeriesSplit` so you always train on the past and validate on the future. Random splits on temporal data leak the future into training." },
        { type: "heading", text: "Data leakage — the bug that fools everyone" },
        { type: "p", text: "**Data leakage** is when information that won't be available at prediction time sneaks into training — inflating your validation/test scores so the model looks brilliant in the notebook and fails in production. It is the single most common and most expensive mistake in applied ML, and it is *silent*: your metrics get better, which feels like success." },
        { type: "table",
          headers: ["Leakage type", "What happens", "The fix"],
          rows: [
            ["**Preprocessing leak**", "you scale/impute using stats from the *whole* dataset before splitting", "fit scalers on **train only**; use a `Pipeline`"],
            ["**Target leak**", "a feature encodes the answer (e.g. `was_refunded` when predicting churn)", "audit every feature: was it known *before* the label?"],
            ["**Temporal leak**", "training rows come from after the prediction time", "split by time; never shuffle time series"],
            ["**Duplicate/group leak**", "the same entity appears in train and test", "split by group (`GroupKFold`), not by row"],
          ]
        },
        { type: "callout", variant: "warn", text: "**The classic preprocessing leak:** you call `StandardScaler().fit(X)` on the full dataset, *then* split. Now the test set's mean and variance have bled into the training transform. Always fit preprocessing inside a `Pipeline` on the training fold only — scikit-learn's `Pipeline` does this correctly by construction, which is the main reason to use it." },
        { type: "callout", variant: "gotcha", text: "A validation score that seems *too good to be true* — 99.9% accuracy on a hard problem — is almost never a great model. It's almost always leakage. When results look magical, **hunt for the leak first**." },
      ]
    },

    {
      id: "bias-variance",
      title: "The bias–variance tradeoff (derived)",
      level: "core",
      body: [
        { type: "p", text: "Why does a more powerful model sometimes do *worse*? The bias–variance decomposition answers this exactly. It splits a model's expected error into three named pieces, and it is worth deriving once by hand — it changes how you read every learning curve for the rest of your career." },
        { type: "heading", text: "The setup" },
        { type: "p", text: "Assume the true data-generating process is a deterministic function $f$ plus irreducible noise $\\varepsilon$ with mean $0$ and variance $\\sigma^2$:" },
        { type: "math", tex: String.raw`y = f(x) + \varepsilon, \qquad \mathbb{E}[\varepsilon] = 0, \qquad \mathrm{Var}(\varepsilon) = \sigma^2` },
        { type: "p", text: "We fit a model $\\hat f$ on a random training set. Because the training set is random, $\\hat f$ is itself a **random object** — draw a different sample, get a different fitted model. We want the **expected squared prediction error** at a fixed test point $x$, averaged over both the noise $\\varepsilon$ and the randomness in the training set (which makes $\\hat f(x)$ random):" },
        { type: "math", tex: String.raw`\mathrm{Err}(x) = \mathbb{E}\!\left[\big(y - \hat f(x)\big)^2\right]` },
        { type: "heading", text: "The derivation" },
        { type: "p", text: "Substitute $y = f + \\varepsilon$ (dropping the argument $x$ for brevity) and insert $\\pm\\,\\mathbb{E}[\\hat f]$, the *average* prediction over all possible training sets. This is the one clever move: add and subtract the mean model." },
        { type: "math", tex: String.raw`\mathbb{E}\!\left[\big(f + \varepsilon - \hat f\big)^2\right] = \mathbb{E}\!\left[\big(\underbrace{f - \mathbb{E}[\hat f]}_{\text{bias}} + \underbrace{\mathbb{E}[\hat f] - \hat f}_{\text{deviation}} + \varepsilon\big)^2\right]` },
        { type: "p", text: "Expand the square. The three cross-terms all vanish: $\\varepsilon$ is independent of $\\hat f$ and has mean $0$; and $\\mathbb{E}\\big[\\mathbb{E}[\\hat f] - \\hat f\\big] = 0$ by definition of the mean, so it is uncorrelated with the constant bias term. What survives is three clean squares:" },
        { type: "math", tex: String.raw`\mathrm{Err}(x) = \underbrace{\big(f(x) - \mathbb{E}[\hat f(x)]\big)^2}_{\text{Bias}^2} \;+\; \underbrace{\mathbb{E}\!\left[\big(\hat f(x) - \mathbb{E}[\hat f(x)]\big)^2\right]}_{\text{Variance}} \;+\; \underbrace{\sigma^2}_{\text{irreducible}}` },
        { type: "math", tex: String.raw`\boxed{\;\mathbb{E}\big[(y - \hat f)^2\big] = \mathrm{Bias}^2 + \mathrm{Variance} + \sigma^2\;}` },
        { type: "heading", text: "What each term means" },
        { type: "table",
          headers: ["Term", "Question it answers", "High when…", "Symptom"],
          rows: [
            ["**Bias²**", "how far is the *average* model from the truth?", "the model is too simple to capture $f$", "**underfitting** — bad on train *and* test"],
            ["**Variance**", "how much does $\\hat f$ jump around across training sets?", "the model is too flexible / data too small", "**overfitting** — great on train, bad on test"],
            ["**$\\sigma^2$**", "noise no model can ever explain", "the world is inherently noisy", "a hard floor on achievable error"],
          ]
        },
        { type: "p", text: "The **tradeoff**: increasing model capacity lowers bias but raises variance, and vice versa. Total error is a U-shaped curve — there's a sweet-spot complexity that minimizes their sum. You cannot beat $\\sigma^2$; chasing it is chasing noise (that *is* overfitting)." },
        { type: "callout", variant: "tip", text: "**Diagnose from two numbers.** High train error → high bias (underfit): use a bigger model, better features, less regularization. Low train error but high validation error → high variance (overfit): get more data, simplify the model, add regularization. This one heuristic drives most of your model-tuning decisions." },
        { type: "callout", variant: "note", text: "**Ensembles are a direct exploit of this equation.** Bagging (Random Forests) averages many high-variance, low-bias trees — averaging crushes variance while leaving bias roughly untouched. Boosting does the opposite: it sequentially reduces bias by stacking weak, high-bias learners. You'll build both in the trees section, and now you know *why* they work." },
        { type: "callout", variant: "warn", text: "Modern deep networks complicate the classic U-curve — the **double descent** phenomenon shows test error can fall *again* as models grow past the interpolation threshold. The classical tradeoff still governs classical ML and remains the right first mental model; double descent is a fascinating wrinkle for the deep-learning track, not a refutation." },
      ]
    },

    {
      id: "overfitting",
      title: "Overfitting, underfitting, capacity & regularization",
      level: "core",
      body: [
        { type: "p", text: "Bias and variance are the *theory*; overfitting and underfitting are what you actually *observe*. They are the two failure modes of the ERM game, and reading which one you're in is a core daily skill." },
        { type: "table",
          headers: ["", "Underfitting", "Just right", "Overfitting"],
          rows: [
            ["Train error", "high", "low", "very low (→ 0)"],
            ["Validation error", "high", "low", "high"],
            ["Cause", "capacity too low (high bias)", "capacity matched to data", "capacity too high (high variance)"],
            ["Fix", "bigger model, more features", "ship it", "regularize, more data, simplify"],
          ]
        },
        { type: "heading", text: "Model capacity" },
        { type: "p", text: "**Capacity** is a model's ability to fit varied functions — its expressive power. A degree-1 polynomial can only draw lines (low capacity); a degree-15 polynomial can wiggle through every training point (high capacity). More capacity means lower bias but higher variance. The art is matching capacity to the amount and complexity of your data." },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\n\n# True function is a gentle sine; we fit polynomials of growing degree.\nrng = np.random.default_rng(0)\nX = np.sort(rng.uniform(0, 1, 30))[:, None]\ny = np.sin(2*np.pi*X.ravel()) + rng.normal(0, 0.15, 30)\n\nfor degree in [1, 4, 15]:\n    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())\n    model.fit(X, y)\n    train_mse = np.mean((model.predict(X) - y)**2)\n    print(f\"degree {degree:2d}: train MSE = {train_mse:.4f}\")\n# degree  1: underfits (high train error, a straight line through a sine)\n# degree  4: fits the sine well\n# degree 15: train MSE ~ 0 but wild oscillations between points -> overfit" },
        { type: "heading", text: "Regularization: the universal antidote (a forward reference)" },
        { type: "p", text: "**Regularization** is any technique that deliberately constrains a model to reduce variance — it trades a little bias for a lot less variance. The most common form adds a penalty $\\Omega(\\theta)$ on the size of the parameters to the ERM objective, discouraging the model from using extreme weights to chase noise:" },
        { type: "math", tex: String.raw`\hat\theta = \arg\min_{\theta}\; \frac{1}{m}\sum_{i} \ell(h_\theta(x^{(i)}), y^{(i)}) \;+\; \lambda\underbrace{\|\theta\|_2^2}_{\text{L2 / Ridge}} \quad\text{or}\quad \lambda\underbrace{\|\theta\|_1}_{\text{L1 / Lasso}}` },
        { type: "table",
          headers: ["Technique", "Where you'll meet it", "Effect"],
          rows: [
            ["**L2 (Ridge)**", "linear/logistic regression, neural nets (weight decay)", "shrinks weights smoothly toward 0"],
            ["**L1 (Lasso)**", "linear models", "drives weights *exactly* to 0 — feature selection"],
            ["**Early stopping**", "gradient-based training", "stop before the model starts memorizing"],
            ["**Dropout**", "neural networks", "randomly zero units to prevent co-adaptation"],
            ["**Data augmentation**", "vision / NLP", "synthetically enlarge the dataset"],
          ]
        },
        { type: "callout", variant: "note", text: "Every one of these gets a full treatment later — L1/L2 in the linear-models file, dropout and early stopping in the neural-networks track. For now, hold the one idea: **the strength $\\lambda$ is a hyperparameter you tune on the validation set**, and it's the main dial you turn when you're overfitting." },
      ]
    },

    {
      id: "generalization",
      title: "Generalization, inductive bias & no free lunch",
      level: "core",
      body: [
        { type: "p", text: "**Generalization** is the whole point: performing well on data you've never seen — i.e. keeping true risk close to empirical risk. Everything so far (splits, cross-validation, regularization, the bias–variance balance) exists to serve generalization. Now, *why* is it even possible to learn a rule from finite examples that works on new data?" },
        { type: "heading", text: "Inductive bias — you can't learn without assumptions" },
        { type: "p", text: "Infinitely many functions pass through any finite set of training points. To pick one — to prefer some hypotheses over others — a learner must bake in assumptions about what a “reasonable” function looks like. That set of assumptions is its **inductive bias**." },
        { type: "table",
          headers: ["Model", "Its inductive bias (what it assumes)"],
          rows: [
            ["Linear regression", "the relationship is (roughly) linear"],
            ["Decision trees", "the target is well-described by axis-aligned splits"],
            ["k-NN", "nearby points have similar labels (smoothness)"],
            ["Convolutional nets", "features are local and translation-invariant"],
          ]
        },
        { type: "callout", variant: "tip", text: "Inductive bias isn't a flaw to eliminate — it's the *engine* of generalization. The right bias for your problem is what lets a model extrapolate sensibly from few examples. Choosing a model is really choosing an inductive bias, so choose one whose assumptions match your data." },
        { type: "heading", text: "The No-Free-Lunch theorem" },
        { type: "p", text: "The **No-Free-Lunch theorems** (**Wolpert 1996** for supervised learning; **Wolpert & Macready 1997** for optimization) prove something humbling: **averaged over *all possible* problems, every learning algorithm has identical performance.** No algorithm is universally best; a model that excels on one class of problems must pay for it by being worse on another." },
        { type: "math", tex: String.raw`\frac{1}{|\text{all problems}|}\sum_{\text{problems}} \text{performance}(\text{Algorithm A}) = \frac{1}{|\text{all problems}|}\sum_{\text{problems}} \text{performance}(\text{Algorithm B})` },
        { type: "callout", variant: "note", text: "**Why this matters in practice:** there is no “best algorithm,” only the best algorithm *for your data*. The reason models work at all is that real-world problems are not drawn uniformly from “all possible problems” — they have structure (smoothness, locality, sparsity), and algorithms whose inductive bias matches that structure win. This is the theoretical license to **always try several models and let the validation set decide**." },
        { type: "callout", variant: "gotcha", text: "No-Free-Lunch is often mis-quoted as “all models are equally good.” They are *not* — on *your specific problem*, one will clearly win. NFL only says no single model wins on *every conceivable* problem. The practical takeaway is empirical humility, not paralysis: measure, don't assume." },
      ]
    },

    {
      id: "workflow",
      title: "The end-to-end ML workflow",
      level: "core",
      body: [
        { type: "p", text: "Notebooks make ML look like `model.fit(X, y)`. Real projects are 10% modeling and 90% everything around it. Here is the opinionated pipeline — the order matters, and skipping a step is how projects die. Roughly **80% of the effort lands in steps 2–4** (data and preprocessing)." },
        { type: "heading", text: "1. Frame the problem" },
        { type: "list", ordered: false, items: [
          "Write $T$, $P$, $E$ (from the Mitchell definition) in one sentence each.",
          "Is it even an ML problem? Could rules solve it? What's the **baseline** (e.g. “always predict the majority class,” or the current heuristic)? You must beat this, or ML isn't worth it.",
          "Pick the metric **before** you model — and make sure it reflects the business goal, not just what's easy to compute.",
        ]},
        { type: "heading", text: "2. Get & split the data" },
        { type: "list", ordered: false, items: [
          "Collect data; then **immediately split off the test set** and lock it away — before EDA, before anything, so nothing about it can leak into your choices.",
          "Sanity-check volume: do you have enough examples for the capacity you'll need?",
        ]},
        { type: "heading", text: "3. EDA (exploratory data analysis)" },
        { type: "list", ordered: false, items: [
          "Look at the data. Histograms, correlations, missing-value counts, class balance, obvious outliers.",
          "This is where you build intuition and catch garbage *before* it poisons a model. On **training data only**.",
        ]},
        { type: "heading", text: "4. Preprocess & engineer features" },
        { type: "list", ordered: false, items: [
          "Handle missing values, scale numeric features, encode categoricals, engineer new features.",
          "**Wrap all of it in a `Pipeline`** so transforms are fit on the training fold only — this is your primary defense against the preprocessing leak.",
        ]},
        { type: "heading", text: "5. Select a model → 6. Train → 7. Evaluate" },
        { type: "list", ordered: false, items: [
          "Start with a **simple, strong baseline** (logistic/linear regression, or a gradient-boosted tree for tabular data). Don't reach for deep learning on 500 rows.",
          "Train; tune hyperparameters with **cross-validation on the training set**.",
          "Evaluate with the metric you chose — analyze errors (confusion matrix, worst predictions), not just a single number.",
        ]},
        { type: "heading", text: "8. Deploy → 9. Monitor" },
        { type: "list", ordered: false, items: [
          "Ship the *whole pipeline* (preprocessing + model), so serving matches training exactly (this prevents **training/serving skew**).",
          "**Monitor in production**: data drift, prediction distributions, and the live metric. Models decay as the world changes — a deployed model is a *maintained* model, not a finished one. Retrain on a schedule or on a drift trigger.",
        ]},
        { type: "callout", variant: "tip", text: "**Iterate, don't waterfall.** This list is a loop, not a one-way street: EDA reveals a leaky feature, deployment reveals drift, error analysis sends you back to feature engineering. Get an ugly end-to-end pipeline working *first* (frame → baseline → deploy), then improve each stage. A deployed mediocre model beats a perfect model still in a notebook." },
        { type: "callout", variant: "gotcha", text: "The most expensive mistakes happen at the **edges** — a fuzzy problem framing (step 1) or an unmonitored deployment (step 9) — not in the modeling middle everyone obsesses over. Spend your care at the boundaries where ML meets the real world." },
      ]
    },

    {
      id: "metrics",
      title: "Metrics — a first pass",
      level: "core",
      body: [
        { type: "p", text: "You minimize a **loss** during training, but you judge success with a **metric** — and they're often different (you optimize cross-entropy but report F1). Choosing the wrong metric is choosing the wrong goal. Here's the working set; the Evaluation file covers each in depth." },
        { type: "heading", text: "Classification metrics" },
        { type: "p", text: "Everything starts from the **confusion matrix**: true/false positives/negatives (TP, FP, TN, FN)." },
        { type: "table",
          headers: ["Metric", "Formula", "Answers", "Use when"],
          rows: [
            ["**Accuracy**", "$\\frac{TP+TN}{\\text{all}}$", "overall fraction correct", "classes are balanced"],
            ["**Precision**", "$\\frac{TP}{TP+FP}$", "of predicted-positive, how many are right?", "false positives are costly (spam filter)"],
            ["**Recall**", "$\\frac{TP}{TP+FN}$", "of actual-positive, how many did we catch?", "false negatives are costly (cancer screen)"],
            ["**F1**", "$\\frac{2\\,PR}{P+R}$", "harmonic mean of precision & recall", "you need one number on imbalanced data"],
            ["**ROC-AUC**", "area under TPR–FPR curve", "ranking quality across all thresholds", "you care about ranking, threshold-free"],
          ]
        },
        { type: "callout", variant: "warn", text: "**Accuracy lies on imbalanced data.** If 99% of transactions are legitimate, a model that predicts “legit” for everything scores 99% accuracy while catching *zero* fraud. This **accuracy paradox** is why precision, recall, F1, and ROC-AUC exist. On any imbalanced problem, never report accuracy alone." },
        { type: "callout", variant: "note", text: "**Precision vs. recall is a business decision, not a math one.** A spam filter should favor precision (deleting a real email is worse than missing spam). A cancer screen should favor recall (a missed tumor is worse than a false alarm). The classification **threshold** slides you along this tradeoff — tune it to your cost structure." },
        { type: "heading", text: "Regression metrics" },
        { type: "table",
          headers: ["Metric", "Formula", "Character"],
          rows: [
            ["**MSE**", "$\\frac{1}{m}\\sum (\\hat y_i - y_i)^2$", "punishes large errors hard; in squared units"],
            ["**RMSE**", "$\\sqrt{\\text{MSE}}$", "same, but back in the original units"],
            ["**MAE**", "$\\frac{1}{m}\\sum |\\hat y_i - y_i|$", "robust to outliers; every error weighed equally"],
            ["**R²**", "$1 - \\frac{\\sum(\\hat y_i - y_i)^2}{\\sum(\\bar y - y_i)^2}$", "fraction of variance explained; 1 = perfect, 0 = no better than the mean"],
          ]
        },
        { type: "callout", variant: "tip", text: "**MSE vs. MAE is a choice about outliers.** MSE's squaring makes it hypersensitive to big misses (great if large errors are genuinely catastrophic; bad if your data has noisy outliers). MAE treats all errors proportionally and is more robust. Same prediction, different priorities — pick to match your problem. Full treatment in the **Evaluation** file." },
      ]
    },

    {
      id: "first-example",
      title: "Your first end-to-end scikit-learn model",
      level: "core",
      body: [
        { type: "p", text: "Time to make it concrete. This is the entire workflow — load, split, preprocess-in-a-pipeline, train, evaluate — on a real dataset, in ~25 lines. Read it as the template you'll fill in for the rest of the track. Note how the `Pipeline` makes leakage structurally impossible." },
        { type: "code", lang: "py", code: "from sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split, cross_val_score\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import classification_report, roc_auc_score\n\n# 1) LOAD -------------------------------------------------------------\ndata = load_breast_cancer()\nX, y = data.data, data.target          # 569 samples, 30 features, binary\n\n# 2) SPLIT (test set locked away immediately) -------------------------\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y)\n\n# 3) PIPELINE: scaling is fit on train folds only -> no leakage --------\nmodel = make_pipeline(\n    StandardScaler(),                  # standardize features\n    LogisticRegression(max_iter=5000)) # a strong, simple baseline\n\n# 4) VALIDATE with cross-validation on TRAIN only ---------------------\ncv = cross_val_score(model, X_train, y_train, cv=5, scoring=\"roc_auc\")\nprint(f\"CV ROC-AUC: {cv.mean():.3f} +/- {cv.std():.3f}\")\n\n# 5) FIT on all training data -----------------------------------------\nmodel.fit(X_train, y_train)\n\n# 6) EVALUATE ONCE on the sacred test set -----------------------------\nproba = model.predict_proba(X_test)[:, 1]\nprint(classification_report(y_test, model.predict(X_test)))\nprint(f\"Test ROC-AUC: {roc_auc_score(y_test, proba):.3f}\")\n# Expect ~0.99 AUC — a linear model on well-behaved tabular data is hard to beat." },
        { type: "callout", variant: "good", text: "**Everything you learned is in those six comments.** Sacred test split, pipeline-based leak prevention, cross-validation for honest estimates, a simple strong baseline, and a metric chosen for the problem. When you meet decision trees, SVMs, and neural nets, only the `LogisticRegression` line changes — the scaffolding is permanent. Internalize this template." },
        { type: "callout", variant: "tip", text: "Start every new problem by literally copying this block and swapping the dataset. Get a baseline number *first*; only then reach for fancier models, and only if they beat this on the validation score." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "These build *judgment*, not just code — the intuition for when a model is lying to you. Do at least two before moving to specific algorithms. Everything here uses only NumPy and scikit-learn." },
        { type: "list", ordered: true, items: [
          "**Reproduce the bias–variance U-curve.** Generate noisy data from a known function. Fit polynomials of degree 1 to 15; for each, plot train error and cross-validation error vs. degree on the same axes. *See* the underfit region, the sweet spot, and the overfit region. Then repeat with more training data and watch the overfit region shrink — the visual proof that more data cures variance.",
          "**Engineer a data leak, then catch it.** Take any dataset and deliberately fit a `StandardScaler` on the *full* data before splitting. Compare test scores to the correct pipeline version. Quantify how many points of fake accuracy the leak buys you — this makes the abstract danger visceral.",
          "**Demonstrate the accuracy paradox.** Build a 99%-imbalanced classification dataset. Train a `DummyClassifier(strategy='most_frequent')` and report its accuracy (≈99%), then its precision, recall, and F1 (≈0). Now train a real model and compare on all metrics. You'll never trust bare accuracy again.",
          "**Cross-validation from scratch.** Implement k-fold CV yourself in NumPy — partition indices into k folds, loop, train, score, average — then confirm your numbers match `sklearn.model_selection.cross_val_score`. Understanding the loop demystifies the tool.",
          "**The end-to-end template on a new dataset.** Take the six-step scikit-learn block and apply it to a fresh dataset (`load_wine`, `load_diabetes`, or a Kaggle CSV). Establish a baseline, then try to beat it by swapping in one other model — and let *only* the validation score decide the winner (No Free Lunch in action).",
          "**Write your own $T$/$P$/$E$ + baseline doc.** Pick a real problem you care about (predicting your commute time, classifying your emails). Write the Mitchell definition, name the metric, and define the dumb baseline you must beat — *before* touching any model. This one-page habit prevents most doomed projects.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "This file is the conceptual map for the whole Classical ML track. To go deeper on the foundations before diving into specific algorithms, these are the best resources, in recommended order:" },
        { type: "link", url: "https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/", text: "Aurélien Géron — Hands-On Machine Learning (3rd ed.) — the best practical, code-first book; Chapter 1–2 mirror this file exactly" },
        { type: "link", url: "https://www.statlearning.com/", text: "James, Witten, Hastie, Tibshirani — An Introduction to Statistical Learning (ISLR) — free PDF; the canonical, gentle statistical foundation" },
        { type: "link", url: "https://www.coursera.org/specializations/machine-learning-introduction", text: "Andrew Ng — Machine Learning Specialization (Coursera) — the classic first course; unbeatable for intuition" },
        { type: "link", url: "https://developers.google.com/machine-learning/crash-course", text: "Google — Machine Learning Crash Course — free, interactive, superb on the workflow and fairness/leakage pitfalls" },
        { type: "link", url: "https://scikit-learn.org/stable/user_guide.html", text: "scikit-learn User Guide — the reference for every model and utility used in this track; the cross-validation and pipeline pages are essential" },
        { type: "link", url: "https://hastie.su.domains/ElemStatLearn/", text: "Hastie, Tibshirani, Friedman — The Elements of Statistical Learning (ESL) — free PDF; the rigorous graduate-level treatment (incl. the full bias–variance derivation)" },
      ]
    },
  ],

  packages: [
    { name: "scikit-learn", why: "the classical-ML workhorse — models, splitting, CV, pipelines, metrics all in one consistent API" },
    { name: "numpy", why: "arrays and vectorized math underneath everything" },
    { name: "pandas", why: "load, clean, and explore tabular data (EDA)" },
    { name: "matplotlib", why: "plot learning curves, the bias–variance U, confusion matrices" },
    { name: "sklearn.model_selection", why: "`train_test_split`, `cross_val_score`, `GridSearchCV`, `TimeSeriesSplit`" },
    { name: "sklearn.pipeline", why: "`Pipeline` / `make_pipeline` — fit preprocessing on train folds only; your anti-leakage armor" },
    { name: "sklearn.metrics", why: "accuracy, precision/recall/F1, ROC-AUC, MSE/MAE/R², confusion matrix" },
  ],

  gotchas: [
    "You minimize **empirical** risk $\\hat R$ but are judged on **true** risk $R$. Driving training loss to zero without a held-out estimate tells you nothing about generalization.",
    "**The test set is sacred** — touch it once, at the very end. Peeking and re-tuning leaks it into your decisions and inflates your reported score.",
    "**Data leakage** is the #1 silent bug: fit scalers/imputers on **train only** (use a `Pipeline`), and never let a feature encode the future or the answer.",
    "Total error is $\\text{Bias}^2 + \\text{Variance} + \\sigma^2$. You can never beat the irreducible noise $\\sigma^2$ — trying to is overfitting.",
    "**Accuracy lies on imbalanced data**: 99% legit → predict-everything-legit scores 99% and catches no fraud. Use precision/recall/F1/ROC-AUC.",
    "**No Free Lunch:** there's no universally best model, only the best for *your* data — always try several and let the validation set decide.",
    "For time series, **never shuffle** the split — use `TimeSeriesSplit` so you train on the past and validate on the future.",
    "Start with a **simple strong baseline** and beat it before reaching for complexity; a deep net on 500 rows is almost always the wrong call.",
  ],

  flashcards: [
    { q: "State Mitchell's definition of learning.", a: "A program learns from experience $E$ w.r.t. task $T$ and performance measure $P$ if its performance at $T$, measured by $P$, improves with $E$. Always name $T$, $P$, $E$ before starting." },
    { q: "Empirical risk vs. true risk?", a: "True risk $R(\\theta)=\\mathbb{E}_{\\mathcal D}[\\ell]$ is expected loss on the real distribution (what we want, can't compute). Empirical risk $\\hat R$ is the average loss on our sample (what we minimize). Their gap is the generalization gap." },
    { q: "What is Empirical Risk Minimization?", a: "$\\hat\\theta=\\arg\\min_\\theta \\frac1m\\sum_i \\ell(h_\\theta(x^{(i)}),y^{(i)})+\\lambda\\Omega(\\theta)$. Nearly every supervised model is this template; they differ only in $h_\\theta$, $\\ell$, and $\\Omega$." },
    { q: "Write the bias–variance decomposition.", a: "$\\mathbb{E}[(y-\\hat f)^2]=\\text{Bias}^2+\\text{Variance}+\\sigma^2$. Bias = simplicity error (underfit), variance = sensitivity to the training sample (overfit), $\\sigma^2$ = irreducible noise." },
    { q: "How do you diagnose underfitting vs. overfitting from two numbers?", a: "High train error → high bias (underfit): add capacity/features. Low train but high validation error → high variance (overfit): more data, simpler model, or regularization." },
    { q: "What is data leakage and its most common form?", a: "Information unavailable at prediction time sneaking into training, inflating scores. The classic case: fitting a scaler/imputer on the full dataset before splitting. Fix: fit preprocessing on train only, inside a `Pipeline`." },
    { q: "Why can't accuracy be trusted on imbalanced data?", a: "The accuracy paradox: with 99% one class, predicting that class always scores 99% while catching none of the rare class. Use precision, recall, F1, or ROC-AUC instead." },
    { q: "Precision vs. recall — when to favor each?", a: "Precision $=TP/(TP+FP)$: favor when false positives cost (spam filter). Recall $=TP/(TP+FN)$: favor when false negatives cost (cancer screen). The threshold slides between them." },
    { q: "What is inductive bias, and why is it necessary?", a: "The set of assumptions a learner uses to prefer some hypotheses over others (e.g. linearity, smoothness, locality). Since infinitely many functions fit finite data, without a bias you can't generalize at all." },
    { q: "State the No-Free-Lunch theorem and its practical lesson.", a: "Averaged over all possible problems, every algorithm performs identically — no universally best model. Practically: for *your* data one model wins, so try several and let validation decide." },
    { q: "Why ship the whole Pipeline, not just the model?", a: "So serving applies the exact same preprocessing as training (fit on train stats), preventing training/serving skew. The pipeline is the deployable unit." },
    { q: "R² of 0 means what?", a: "The model does no better than always predicting the mean $\\bar y$. R² = 1 is perfect; negative means worse than the mean baseline." },
  ],

  cheatsheet: [
    { label: "Train/test split (stratified)", code: "train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)" },
    { label: "k-fold cross-validation", code: "cross_val_score(model, X_train, y_train, cv=5, scoring='f1')" },
    { label: "Leak-proof pipeline", code: "make_pipeline(StandardScaler(), LogisticRegression())" },
    { label: "Time-series split", code: "from sklearn.model_selection import TimeSeriesSplit" },
    { label: "Hyperparameter search", code: "GridSearchCV(model, param_grid, cv=5)" },
    { label: "Classification report", code: "classification_report(y_test, y_pred)" },
    { label: "ROC-AUC", code: "roc_auc_score(y_test, model.predict_proba(X_test)[:,1])" },
    { label: "Confusion matrix", code: "confusion_matrix(y_test, y_pred)" },
    { label: "Regression metrics", code: "mean_squared_error / mean_absolute_error / r2_score" },
    { label: "Dummy baseline", code: "DummyClassifier(strategy='most_frequent')" },
    { label: "Bias–variance", code: "E[(y-fhat)^2] = bias^2 + variance + sigma^2" },
    { label: "ERM objective", code: "argmin_theta  (1/m) sum loss + lambda * Omega(theta)" },
  ],
});
