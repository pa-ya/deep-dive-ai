(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "ml-feature-engineering",
  name: "Feature Engineering & Data Prep",
  language: "Classical ML",
  group: "Classical Machine Learning",
  navLabel: "Feature Engineering",
  tagline: "The craft that quietly wins Kaggle and production — scaling, encoding, imputation, feature creation, selection, and leak-proof sklearn pipelines, derived and coded from scratch.",
  color: "#2DD4BF",
  readMinutes: 46,
  sections: [
    {
      id: "why",
      title: "Why features beat models (the data-centric view)",
      level: "core",
      body: [
        { type: "p", text: "Andrew Ng's most-quoted line about applied ML is blunt: *\"Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering.\"* On **tabular data** — the spreadsheets, logs, and databases that make up the overwhelming majority of real business ML — the gap between a mediocre model on great features and a great model on raw features is not close. Great features win." },
        { type: "p", text: "Here is the uncomfortable empirical truth for a developer arriving from deep learning hype: on tabular problems, **gradient-boosted trees (XGBoost/LightGBM) on well-engineered features still routinely beat neural networks**, and the engineering matters far more than which of the top-3 model families you pick. The 2022 paper *\"Why do tree-based models still outperform deep learning on tabular data?\"* (Grinsztajn et al.) made this rigorous. Your leverage is in the columns you feed the model, not the model." },
        { type: "table",
          headers: ["Lever", "Typical effect on a tabular score", "Effort"],
          rows: [
            ["Swap Logistic Regression → XGBoost", "moderate bump", "minutes"],
            ["Tune XGBoost hyperparameters hard", "small bump", "hours"],
            ["Fix leakage / validation", "prevents a *disaster*", "hours"],
            ["Engineer 10 good domain features", "often the **largest** bump", "hours–days"],
            ["Collect more / cleaner data", "usually the real ceiling", "days–weeks"],
          ]
        },
        { type: "p", text: "This is the **data-centric** view popularized by Ng's later work: hold the model fixed and systematically improve the data — its features, labels, and coverage. This deck is the mechanic's manual for that work: how to represent numbers, categories, missingness, text, and time so that a model can actually use them, and how to do it **without leaking the future into the past**." },
        { type: "callout", variant: "note", text: "**What a 'feature' is.** A feature is one input column: a measured or derived quantity the model reads. **Feature engineering** is the craft of turning raw records into a numeric matrix $X \\in \\mathbb{R}^{m \\times n}$ that exposes the signal. Every technique in this deck is a function that maps raw data to columns — and every such function must be *fit on training data only*, a theme we hammer until it is reflex." },
        { type: "callout", variant: "tip", text: "**The golden rule of this entire deck:** any transformation that *learns something from the data* (a mean, a std, a category-to-number map, an imputation value) must learn it from the **training fold only**, then apply the frozen result to validation/test. Break this and your offline score will lie to you. Section 8 shows the machinery (`Pipeline` + `ColumnTransformer`) that makes obeying it automatic." },
      ]
    },

    {
      id: "numeric",
      title: "Numeric features: scaling, transforms & binning",
      level: "core",
      body: [
        { type: "p", text: "Raw numeric columns come in wildly different units — age in `[0, 100]`, income in `[0, 10^6]`, a ratio in `[0, 1]`. Many models care about the *scale* of inputs; some don't at all. Knowing **which** and **why** is the first real skill." },
        { type: "heading", text: "Standardization (z-score) — center and unit-variance" },
        { type: "p", text: "The most common scaling subtracts the mean and divides by the standard deviation, giving each feature mean $0$ and variance $1$:" },
        { type: "math", tex: String.raw`z = \frac{x - \mu}{\sigma}, \qquad \mu = \frac{1}{m}\sum_{i=1}^m x_i, \quad \sigma = \sqrt{\frac{1}{m}\sum_{i=1}^m (x_i - \mu)^2}` },
        { type: "p", text: "Use it as the sensible **default** for linear models, SVMs, logistic regression, PCA, and neural nets. It does not bound the range and it preserves the shape of the distribution (including outliers), so a feature that was roughly Gaussian stays roughly Gaussian, now standard-normal." },
        { type: "heading", text: "Min–max scaling — squash to a fixed range" },
        { type: "p", text: "Min–max linearly maps a column into $[0,1]$ (or any $[a,b]$):" },
        { type: "math", tex: String.raw`x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}, \qquad x'_{[a,b]} = a + (b-a)\,\frac{x - x_{\min}}{x_{\max} - x_{\min}}` },
        { type: "p", text: "Use it when you need a **bounded** range — image pixels to $[0,1]$, inputs to a bounded activation, or when a downstream component assumes non-negative bounded values. Its weakness: $x_{\\min}$ and $x_{\\max}$ are set by the two most extreme points, so a **single outlier squashes everyone else** into a tiny sub-interval." },
        { type: "heading", text: "Robust scaling — median and IQR, for outlier-heavy data" },
        { type: "p", text: "When outliers are present, replace the mean/std with the **median** and **interquartile range** (IQR $= Q_3 - Q_1$), which the extremes barely move:" },
        { type: "math", tex: String.raw`x' = \frac{x - \operatorname{median}(x)}{Q_3(x) - Q_1(x)}` },
        { type: "p", text: "The median and IQR are computed from ranks, so a handful of enormous values shift them almost not at all. Reach for this on financial data, sensor spikes, or any column with a fat tail you don't want to (or can't) clip." },
        { type: "table",
          headers: ["Scaler", "Formula core", "Bounded?", "Outlier-robust?", "Use when"],
          rows: [
            ["Standard (z-score)", "$(x-\\mu)/\\sigma$", "no", "no", "default for linear/SVM/NN/PCA"],
            ["Min–max", "$(x-x_{\\min})/(x_{\\max}-x_{\\min})$", "yes $[0,1]$", "no", "need a fixed range; few outliers"],
            ["Robust", "$(x-\\text{med})/\\text{IQR}$", "no", "yes", "heavy-tailed / outlier-prone columns"],
            ["MaxAbs", "$x/\\max|x|$", "yes $[-1,1]$", "no", "sparse data (preserves zeros)"],
          ]
        },
        { type: "heading", text: "Why scaling matters for some models and is irrelevant for others" },
        { type: "p", text: "Scaling changes results whenever the algorithm compares features by *magnitude*. Three families care:" },
        { type: "list", ordered: false, items: [
          "**Distance-based** (KNN, K-means, RBF-SVM): distance $\\|x_i - x_j\\|$ is dominated by whichever feature has the biggest raw units. Income in dollars will drown out age unless both are scaled.",
          "**Gradient-descent-trained** (linear/logistic regression, neural nets): unequal feature scales stretch the loss bowl (large condition number $\\kappa$ of the Hessian), so gradient descent zig-zags and converges slowly. Scaling makes the bowl round.",
          "**Regularized** (Ridge/Lasso, L2/L1): the penalty $\\lambda\\sum w_j^2$ treats every weight equally, but a feature on a huge scale needs a tiny weight to matter — so the penalty unfairly punishes small-scale features. Always scale before regularizing.",
        ]},
        { type: "callout", variant: "good", text: "**Trees don't care about monotonic scaling.** Decision trees, random forests, and gradient boosting split on thresholds like `x < t`. Any strictly increasing transform (standardize, min–max, log) leaves the *ordering* of values unchanged, so the same splits are available and the tree is identical. You can (and Kagglers do) skip scaling entirely for tree models. Scaling only helps trees if it changes ordering — which these don't." },
        { type: "heading", text: "Taming skew: log, power, and Box–Cox transforms" },
        { type: "p", text: "Many real columns (income, prices, counts, page-views) are **right-skewed** — a long tail of large values. Linear and distance models prefer roughly symmetric, homoscedastic inputs. A log or power transform compresses the tail:" },
        { type: "math", tex: String.raw`x \mapsto \log(x) \quad\text{or, safe for zeros,}\quad x \mapsto \log(1 + x) = \texttt{log1p}(x)` },
        { type: "p", text: "For a principled choice, the **Box–Cox** family finds the power $\\lambda$ that makes the column most Gaussian (requires $x > 0$); **Yeo–Johnson** extends it to zero and negative values:" },
        { type: "math", tex: String.raw`x^{(\lambda)} = \begin{cases} \dfrac{x^{\lambda} - 1}{\lambda}, & \lambda \neq 0 \\[4pt] \ln x, & \lambda = 0 \end{cases}` },
        { type: "callout", variant: "gotcha", text: "`np.log(x)` explodes for $x = 0$ (gives $-\\infty$) and is undefined for negatives. Use `np.log1p(x)` ($=\\log(1+x)$) for non-negative counts, or Yeo–Johnson when the column can be negative. And if you log-transform the **target** $y$, remember to `expm1` your predictions back before scoring in the original units." },
        { type: "heading", text: "Binning (discretization)" },
        { type: "p", text: "**Binning** converts a continuous column into ordered categories: age → `{child, adult, senior}`, or into equal-width / equal-frequency (quantile) buckets. It can help by (a) letting a linear model capture a **non-linear** age→risk relationship, (b) taming outliers (everything above a threshold lands in the top bin), and (c) matching a genuine domain step (tax brackets). The cost is lost resolution — trees usually find the useful splits themselves, so binning helps linear models far more than trees." },
        { type: "code", lang: "py", code: "import numpy as np\n\nrng = np.random.default_rng(0)\nincome = rng.lognormal(mean=10, sigma=1.0, size=1000)   # heavy right skew\n\n# --- standardization from scratch (fit stats on TRAIN only!) ---\nmu, sigma = income.mean(), income.std()\nz = (income - mu) / sigma\n\n# --- min-max from scratch ---\nlo, hi = income.min(), income.max()\nmm = (income - lo) / (hi - lo)\n\n# --- robust from scratch ---\nq1, med, q3 = np.percentile(income, [25, 50, 75])\nrob = (income - med) / (q3 - q1)\n\n# --- de-skew with log1p ---\nlog_income = np.log1p(income)\nprint(f\"raw   skew ~ {float(((income-mu)**3).mean()/sigma**3):+.2f}\")\nprint(f\"log1p skew ~ {float((((log_income-log_income.mean())**3).mean())/log_income.std()**3):+.2f}\")\n\n# --- equal-frequency binning into quartiles ---\nedges = np.quantile(income, [0, .25, .5, .75, 1.0])\nbins = np.digitize(income, edges[1:-1])   # 0..3\nprint(np.bincount(bins))                  # ~250 each" },
        { type: "code", lang: "py", code: "from sklearn.preprocessing import (StandardScaler, MinMaxScaler,\n                                   RobustScaler, PowerTransformer,\n                                   KBinsDiscretizer)\n\n# All scalers share the fit/transform contract: fit on train, transform both.\nscaler = StandardScaler().fit(X_train)      # learns mu, sigma per column\nX_train_s = scaler.transform(X_train)\nX_test_s  = scaler.transform(X_test)         # uses TRAIN's mu, sigma\n\nMinMaxScaler(feature_range=(0, 1))\nRobustScaler()                               # median / IQR\nPowerTransformer(method=\"yeo-johnson\")       # de-skew, handles <= 0\nKBinsDiscretizer(n_bins=4, encode=\"ordinal\", strategy=\"quantile\")" },
      ]
    },

    {
      id: "categorical",
      title: "Categorical features: encoding without leaking",
      level: "core",
      body: [
        { type: "p", text: "Models eat numbers, but half your columns are strings: `city`, `product_category`, `browser`. **Encoding** turns categories into numbers — and the choice interacts with cardinality (how many distinct values) and with whether the category has a natural order." },
        { type: "heading", text: "One-hot encoding — the safe default for low cardinality" },
        { type: "p", text: "One-hot maps a $k$-level category to $k$ binary columns, exactly one of them hot. `color ∈ {red, green, blue}` becomes three 0/1 columns. It makes no ordinal assumption, so it's the correct default for **nominal** (unordered) categories with modest cardinality (say $k \\lesssim 15$)." },
        { type: "callout", variant: "gotcha", text: "**The dummy-variable trap.** For a linear model with an intercept, $k$ one-hot columns are perfectly collinear (they sum to the all-ones intercept), making $X^\\top X$ singular. Drop one column (`drop=\"first\"`) so $k$ levels become $k-1$ columns. Trees and regularized models don't need this, but it's harmless and standard for linear regression." },
        { type: "heading", text: "Ordinal encoding — for genuinely ordered categories" },
        { type: "p", text: "When categories have a **real order** — `{low < medium < high}`, `{XS < S < M < L < XL}` — map them to ordered integers `0,1,2,...`. This is correct only when the order is meaningful; using it on nominal data (`{Paris:0, Tokyo:1, Cairo:2}`) lies to the model by inventing that Cairo > Tokyo > Paris. For trees, ordinal encoding of *any* category is often fine and far more compact than one-hot, because a tree can carve the integer axis into arbitrary groups via multiple splits." },
        { type: "heading", text: "Target (mean) encoding — powerful, and a leakage minefield" },
        { type: "p", text: "**Target encoding** replaces a category with a statistic of the target for that category — usually the mean. For a binary target and category value $c$:" },
        { type: "math", tex: String.raw`\text{enc}(c) = \frac{\sum_{i : x_i = c} y_i}{|\{i : x_i = c\}|} = \bar{y}_c` },
        { type: "p", text: "This is wonderfully compact (one column, any cardinality) and injects real signal (`zip_code → default rate`). But naive category means are computed **using the target**, and a rare category whose mean is taken over its own rows will encode that row's own label — direct **target leakage**. Two fixes, used together:" },
        { type: "p", text: "**(1) Smoothing** toward the global mean, so tiny categories don't get overconfident estimates. With $n_c$ rows in category $c$, global mean $\\bar y$, and smoothing strength $\\alpha$:" },
        { type: "math", tex: String.raw`\text{enc}(c) = \frac{n_c\,\bar{y}_c + \alpha\,\bar{y}}{n_c + \alpha}` },
        { type: "p", text: "A category seen many times ($n_c \\gg \\alpha$) keeps its own mean; a category seen once is pulled almost entirely to the global prior. **(2) Out-of-fold** encoding: compute each row's encoding from a *different* fold than the one it lives in, so a row's own target never enters its own feature. This is the single most important trick." },
        { type: "callout", variant: "warn", text: "**The most infamous leak in tabular ML.** Fitting target encoding on the full dataset and then cross-validating gives spectacular offline scores that **collapse in production**. Always do target encoding *inside* CV folds (or use `category_encoders`' `cv` variants / sklearn's `TargetEncoder`, which does internal cross-fitting). Treat any target-derived feature as radioactive until you've proven it's out-of-fold." },
        { type: "heading", text: "Frequency & hashing encoding — for high cardinality" },
        { type: "p", text: "**Frequency encoding** replaces a category with how often it appears (its count or proportion). It's cheap, leak-free (no target involved), and often a useful signal on its own (popular items behave differently). **Hashing** (the 'hashing trick') runs each category string through a hash into a fixed number of buckets $d$ — turning a million-level column into $d$ columns with no dictionary to store, at the cost of occasional collisions. It's how you handle unbounded/streaming vocabularies." },
        { type: "table",
          headers: ["Encoding", "Output columns", "Uses target?", "Best for"],
          rows: [
            ["One-hot", "$k$ (or $k-1$)", "no", "low-cardinality nominal"],
            ["Ordinal", "$1$", "no", "ordered categories; trees"],
            ["Target/mean", "$1$", "**yes** (leak risk!)", "high-cardinality, strong signal"],
            ["Frequency", "$1$", "no", "high-cardinality, cheap"],
            ["Hashing", "$d$ (fixed)", "no", "huge / streaming vocab"],
          ]
        },
        { type: "callout", variant: "tip", text: "**High-cardinality playbook.** Group rare levels into an `\"other\"` bucket (e.g. anything under 20 occurrences); combine with **frequency encoding** for cheap signal and **out-of-fold target encoding** for strong signal; reserve **hashing** for truly unbounded vocabularies. And always handle **unseen categories at inference** — decide up front whether a new city maps to `\"other\"`, the global prior, or a dedicated unknown bucket." },
        { type: "code", lang: "py", code: "import numpy as np, pandas as pd\n\ndf = pd.DataFrame({\n    \"city\": [\"paris\",\"tokyo\",\"paris\",\"cairo\",\"tokyo\",\"paris\"],\n    \"y\":    [1,      0,       1,      0,      1,      0],\n})\n\n# --- frequency encoding (leak-free) ---\nfreq = df[\"city\"].map(df[\"city\"].value_counts(normalize=True))\n\n# --- smoothed target encoding, from scratch ---\nglobal_mean = df[\"y\"].mean()\nalpha = 10.0\nstats = df.groupby(\"city\")[\"y\"].agg([\"mean\", \"count\"])\nsmooth = (stats[\"count\"]*stats[\"mean\"] + alpha*global_mean) / (stats[\"count\"]+alpha)\nprint(smooth)   # each city -> shrunken mean toward global_mean\n\n# --- the RIGHT way: out-of-fold to avoid leakage ---\nfrom sklearn.model_selection import KFold\noof = np.full(len(df), np.nan)\nfor tr, va in KFold(3, shuffle=True, random_state=0).split(df):\n    m = df.iloc[tr].groupby(\"city\")[\"y\"].mean()               # fit on TRAIN fold\n    oof[va] = df.iloc[va][\"city\"].map(m).fillna(global_mean)   # apply to VAL fold\nprint(oof)      # a row's own label never enters its own encoding" },
        { type: "code", lang: "py", code: "from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder\n\n# One-hot; ignore categories unseen at fit time instead of crashing.\nOneHotEncoder(handle_unknown=\"ignore\", drop=\"first\", sparse_output=False)\n\n# Ordinal with an explicit, meaningful order.\nOrdinalEncoder(categories=[[\"low\", \"medium\", \"high\"]])\n\n# sklearn's TargetEncoder does internal cross-fitting => leak-safe by design.\nTargetEncoder(smooth=\"auto\", cv=5)\n# For frequency/hashing/CatBoost encoders: pip install category_encoders" },
      ]
    },

    {
      id: "missing",
      title: "Missing data: MCAR/MAR/MNAR & imputation",
      level: "core",
      body: [
        { type: "p", text: "Real datasets have holes. Before you fill them, ask **why** a value is missing — because the mechanism determines whether naive imputation is harmless or actively misleading. Rubin's taxonomy names three mechanisms:" },
        { type: "table",
          headers: ["Mechanism", "Meaning", "Example", "Danger"],
          rows: [
            ["MCAR", "Missing Completely At Random — missingness independent of everything", "a sensor drops a reading at random", "low; mean-impute is unbiased"],
            ["MAR", "Missing At Random — depends on *observed* features", "income missing more often for younger users (age is observed)", "moderate; condition on observed features to impute"],
            ["MNAR", "Missing Not At Random — depends on the *unobserved value itself*", "high earners refuse to report income", "high; the missingness *is* signal — model it"],
          ]
        },
        { type: "callout", variant: "note", text: "You can never fully test MCAR vs MAR vs MNAR from the data alone — MNAR in particular is untestable because the missing values are, by definition, unseen. Use domain knowledge. The practical upshot: **the fact that a value is missing is often itself predictive**, so preserve that information (next callout) rather than silently erasing it." },
        { type: "heading", text: "Imputation strategies" },
        { type: "list", ordered: false, items: [
          "**Mean / median** (numeric): fast, robust default. Prefer **median** for skewed columns (it's not dragged by outliers). Learn the value on train, apply to test.",
          "**Mode / constant** (categorical): fill with the most frequent level, or an explicit `\"missing\"` category — often better, because it lets the model learn from the missingness.",
          "**KNN imputation**: fill a hole with the (distance-weighted) average of the $k$ nearest complete rows. Captures feature correlations; needs scaling first and is $O(m^2)$-ish, so it doesn't scale to huge data.",
          "**Iterative (MICE)**: model each feature-with-holes as a regression on the others, cycling until convergence. The most principled for MAR data; sklearn's `IterativeImputer` implements it.",
        ]},
        { type: "callout", variant: "tip", text: "**Always add a missingness indicator** for columns where absence might be informative: a binary `x_was_missing` column alongside the imputed value. If the missingness is MNAR or MAR, that flag often carries more signal than the imputed number. sklearn: `SimpleImputer(add_indicator=True)`." },
        { type: "heading", text: "When to just drop" },
        { type: "p", text: "Drop **rows** only when missingness is rare (a few percent) and plausibly MCAR — otherwise you bias the sample. Drop **columns** when a feature is mostly empty (e.g. >60–80% missing) and low-signal; but first consider keeping just its missingness indicator, which may be all the signal there was." },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sklearn.impute import SimpleImputer, KNNImputer\nfrom sklearn.experimental import enable_iterative_imputer  # noqa: F401\nfrom sklearn.impute import IterativeImputer\n\nX = np.array([[1.0, 2.0], [np.nan, 3.0], [7.0, 6.0], [8.0, np.nan]])\n\n# median impute + keep a 'was it missing?' flag (fit on train, apply to test)\nimp = SimpleImputer(strategy=\"median\", add_indicator=True).fit(X)\nprint(imp.transform(X))          # extra columns flag imputed positions\n\nKNNImputer(n_neighbors=3)         # correlation-aware; scale features first\nIterativeImputer(max_iter=10, random_state=0)   # MICE: model each column on the rest" },
      ]
    },

    {
      id: "creation",
      title: "Feature creation: interactions, polynomials, time & aggregations",
      level: "core",
      body: [
        { type: "p", text: "Encoding and scaling *represent* what you already have; **feature creation** manufactures new signal a model can't easily discover on its own. This is where domain knowledge pays off most." },
        { type: "heading", text: "Interactions & polynomial features" },
        { type: "p", text: "A linear model sees each feature independently; it cannot represent 'risk is high only when *age is low AND mileage is high*'. An **interaction** term $x_i x_j$ hands it that product directly. **Polynomial features** generalize this to all products up to degree $d$ — degree 2 on $(x_1, x_2)$ gives $\\{x_1, x_2, x_1^2, x_2^2, x_1 x_2\\}$:" },
        { type: "math", tex: String.raw`\phi_d(x) = \big\{\, \textstyle\prod_{j} x_j^{\,p_j} \;:\; p_j \ge 0,\ \sum_j p_j \le d \,\big\}` },
        { type: "callout", variant: "gotcha", text: "Polynomial expansion blows up **combinatorially**: $n$ features at degree $d$ produce $\\binom{n+d}{d}$ terms. 100 features at degree 2 is ~5,000 columns; degree 3 is ~170,000. Keep $d$ small, expand only a hand-picked subset, and pair it with regularization. Trees learn interactions natively, so they rarely need this — polynomials are mainly a *linear-model* tool." },
        { type: "heading", text: "Domain features — the real edge" },
        { type: "p", text: "The features that win are usually ratios and differences a human expert would compute: `debt_to_income = debt / income`, `price_per_sqft = price / area`, `bmi = weight / height**2`, `clicks_per_impression = clicks / impressions`. These encode relationships the model would otherwise need lots of data to approximate. Spend your time here." },
        { type: "heading", text: "Datetime decomposition" },
        { type: "p", text: "A raw timestamp is nearly useless as a single number; **decompose** it into the components that actually drive behavior: `year, month, day, dayofweek, hour, is_weekend, is_holiday, days_since_signup`. Crucially, **cyclical** fields (hour, month, day-of-week) wrap around — hour 23 is adjacent to hour 0 — so encode them with sine/cosine so the model sees that continuity:" },
        { type: "math", tex: String.raw`x_{\sin} = \sin\!\left(\frac{2\pi\, t}{T}\right), \qquad x_{\cos} = \cos\!\left(\frac{2\pi\, t}{T}\right), \qquad T = \text{period (24, 7, 12, \dots)}` },
        { type: "heading", text: "Aggregations & group statistics" },
        { type: "p", text: "When rows relate (many transactions per customer, many logs per session), **group aggregations** turn history into features: per-customer `mean/max/std/count` of purchase amount, days since last order, rolling 7-day averages. These `groupby`-and-merge features are the backbone of winning solutions on transactional and time-series-flavored tabular data." },
        { type: "callout", variant: "warn", text: "**Aggregations are a prime leakage source.** A group statistic must only use data available *before* the prediction time. Computing a customer's mean spend over the *whole* dataset (including the future rows you're predicting) leaks. Use expanding/rolling windows with a strict `shift(1)`, or compute group stats out-of-fold, exactly like target encoding." },
        { type: "code", lang: "py", code: "import numpy as np, pandas as pd\n\ndf = pd.DataFrame({\n    \"ts\": pd.to_datetime([\"2026-01-03 09:00\", \"2026-07-04 23:30\", \"2026-03-15 14:00\"]),\n    \"price\": [200.0, 50.0, 120.0],\n    \"area\":  [80.0, 25.0, 60.0],\n})\n\n# domain ratio\ndf[\"price_per_sqm\"] = df[\"price\"] / df[\"area\"]\n\n# datetime decomposition\ndf[\"hour\"] = df[\"ts\"].dt.hour\ndf[\"dow\"]  = df[\"ts\"].dt.dayofweek\ndf[\"is_weekend\"] = (df[\"dow\"] >= 5).astype(int)\n\n# cyclical encoding of hour (period T = 24)\ndf[\"hour_sin\"] = np.sin(2*np.pi*df[\"hour\"]/24)\ndf[\"hour_cos\"] = np.cos(2*np.pi*df[\"hour\"]/24)\n\n# a leak-safe expanding group stat: mean price BEFORE the current row\ndf[\"cum_mean_price\"] = df[\"price\"].expanding().mean().shift(1)\nprint(df)" },
        { type: "code", lang: "py", code: "from sklearn.preprocessing import PolynomialFeatures\n\n# degree-2 interactions only (no pure squares) keeps the blow-up modest\nPolynomialFeatures(degree=2, interaction_only=True, include_bias=False)\n\n# per-group aggregations with pandas, then merge back onto the rows\n# g = txns.groupby(\"customer_id\")[\"amount\"]\n# feats = g.agg([\"mean\", \"std\", \"max\", \"count\"]).add_prefix(\"amt_\")\n# X = X.merge(feats, on=\"customer_id\", how=\"left\")" },
      ]
    },

    {
      id: "text-image",
      title: "Text & other modalities (a teaser)",
      level: "core",
      body: [
        { type: "p", text: "Sometimes a tabular column is free text (a product description, a review) or a path to an image. You can squeeze basic features out of both without leaving classical ML — deep representations come later in the NLP and CV tracks." },
        { type: "heading", text: "Text: bag-of-words and TF-IDF" },
        { type: "p", text: "The simplest text representation is **bag-of-words**: count how often each vocabulary word appears, ignoring order. **TF-IDF** reweights those counts so common-everywhere words (`the`, `and`) are downweighted and distinctive words are boosted. Term frequency times inverse document frequency:" },
        { type: "math", tex: String.raw`\text{tfidf}(t, d) = \underbrace{\text{tf}(t, d)}_{\text{count in doc } d} \times \underbrace{\log\frac{N}{1 + \text{df}(t)}}_{\text{rarity across } N \text{ docs}}` },
        { type: "p", text: "where $\\text{df}(t)$ is the number of documents containing term $t$. The result is a sparse, high-dimensional numeric matrix you can feed straight into logistic regression or a linear SVM — a strong, cheap text baseline. Cheap engineered text features also help: character/word counts, average word length, punctuation and uppercase ratios, sentiment-lexicon hits." },
        { type: "callout", variant: "note", text: "**Forward reference.** TF-IDF throws away word order and meaning — `\"not good\"` and `\"good not\"` are identical to it. **Word embeddings** (word2vec, GloVe) and then **contextual transformer embeddings** (BERT and friends) fix this by mapping text to dense vectors where meaning is geometry. That's the entire NLP track; TF-IDF is the baseline you'll compare against." },
        { type: "heading", text: "Images and other modalities" },
        { type: "p", text: "For images inside a tabular pipeline, classical features include simple stats (mean/std of pixel intensity, per-channel color histograms), edge/texture descriptors (HOG, LBP), and shape metrics. In practice, though, a **pretrained CNN used as a fixed feature extractor** — take the penultimate-layer activations as a dense vector — beats hand-crafted image features by a mile, and is itself just 'feature engineering via transfer learning'. Same story for audio (spectrogram statistics) and geospatial (distance-to-landmark) data: extract a numeric vector, then treat it as tabular." },
        { type: "code", lang: "py", code: "from sklearn.feature_extraction.text import TfidfVectorizer\n\ndocs = [\"great product loved it\", \"bad quality not good\", \"good value great buy\"]\n\ntfidf = TfidfVectorizer(\n    ngram_range=(1, 2),      # unigrams + bigrams (captures 'not good')\n    min_df=1, max_features=5000,\n    stop_words=\"english\",\n).fit(docs)                  # fit vocabulary + idf on TRAIN docs only!\n\nX = tfidf.transform(docs)    # sparse (n_docs x vocab) matrix\nprint(X.shape, tfidf.get_feature_names_out()[:6])\n# feed X straight into LogisticRegression / LinearSVC for a strong baseline" },
      ]
    },

    {
      id: "selection",
      title: "Feature selection: filter, wrapper & embedded",
      level: "core",
      body: [
        { type: "p", text: "More features is not more better. Irrelevant and redundant columns add noise, invite overfitting, slow training, and widen your **leakage surface**. Feature selection trims the matrix down to what carries signal. The three families trade off cost against how model-aware they are." },
        { type: "heading", text: "Filter methods — rank each feature independently of the model" },
        { type: "p", text: "Score each feature by its statistical relationship to the target, then keep the top-$k$. Cheap and model-agnostic, but blind to interactions (a feature useless alone can be gold in combination)." },
        { type: "list", ordered: false, items: [
          "**Correlation** (Pearson) for numeric feature vs numeric target: fast, but only detects *linear* dependence. Also use a feature–feature correlation matrix to drop one of any pair that's ~perfectly correlated (redundant).",
          "**Mutual information** captures *any* dependence, linear or not: $I(X;Y) = \\sum p(x,y)\\log\\frac{p(x,y)}{p(x)p(y)}$. sklearn's `mutual_info_classif` / `_regression` estimate it and are a great default filter.",
          "**Chi-squared** ($\\chi^2$) for categorical feature vs categorical target: tests independence in the contingency table. Requires non-negative features (e.g. counts / one-hot).",
        ]},
        { type: "math", tex: String.raw`I(X;Y) = \sum_{x}\sum_{y} p(x,y)\,\log\frac{p(x,y)}{p(x)\,p(y)} \;\ge\; 0, \qquad I(X;Y)=0 \iff X \perp Y` },
        { type: "heading", text: "Wrapper methods — let the model vote by retraining" },
        { type: "p", text: "**Recursive Feature Elimination (RFE)** trains the model, ranks features by the model's own importance/coefficients, drops the weakest, and repeats down to a target count. It's model-aware and captures interactions, but costs many model fits — expensive on wide data. `RFECV` wraps it in cross-validation to pick the count automatically." },
        { type: "heading", text: "Embedded methods — selection happens during training" },
        { type: "p", text: "The most practical family: the model selects features *as it fits*. **L1 (Lasso)** regularization adds $\\lambda\\sum_j|w_j|$ to the loss; the corner geometry of the $L_1$ ball drives many weights to *exactly* zero, so the fit and the selection are one step. **Tree importances** (Gini/split-gain, or better, permutation importance) rank features for free from any forest/boosting model." },
        { type: "math", tex: String.raw`\min_w\; \underbrace{\|Xw - y\|_2^2}_{\text{fit}} \;+\; \lambda \underbrace{\textstyle\sum_j |w_j|}_{L_1 \Rightarrow \text{ sparse } w}` },
        { type: "table",
          headers: ["Family", "Model-aware?", "Interactions?", "Cost", "Example"],
          rows: [
            ["Filter", "no", "no", "cheap", "mutual info, $\\chi^2$, correlation"],
            ["Wrapper", "yes", "yes", "expensive", "RFE / RFECV"],
            ["Embedded", "yes", "yes", "cheap (built-in)", "Lasso (L1), tree importance"],
          ]
        },
        { type: "callout", variant: "warn", text: "**Selection is part of the model — fit it inside CV.** Choosing features on the *whole* dataset (including validation rows) and then cross-validating leaks the target and inflates your score. Put selection inside the pipeline so it's refit on each training fold only. Prefer **permutation importance** over impurity-based importance, which is biased toward high-cardinality and continuous features." },
        { type: "code", lang: "py", code: "import numpy as np\nfrom sklearn.feature_selection import (mutual_info_classif, SelectKBest,\n                                       chi2, RFE)\nfrom sklearn.linear_model import LogisticRegression, LassoCV\nfrom sklearn.ensemble import RandomForestClassifier\n\n# --- filter: keep top-k by mutual information ---\nSelectKBest(mutual_info_classif, k=10)\n\n# --- filter: chi2 for non-negative (count/one-hot) features ---\nSelectKBest(chi2, k=10)\n\n# --- wrapper: recursively eliminate down to 10 features ---\nRFE(LogisticRegression(max_iter=1000), n_features_to_select=10)\n\n# --- embedded: L1 zeros out weak features automatically ---\nlasso = LassoCV(cv=5).fit(X_train, y_train)\nkept = np.flatnonzero(lasso.coef_ != 0)\nprint(\"L1 kept\", kept.size, \"of\", X_train.shape[1], \"features\")\n\n# --- embedded: permutation importance (unbiased) ---\nfrom sklearn.inspection import permutation_importance\nrf = RandomForestClassifier(n_estimators=300).fit(X_train, y_train)\npi = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=0)\norder = np.argsort(pi.importances_mean)[::-1]\nprint(\"top features by permutation importance:\", order[:10])" },
      ]
    },

    {
      id: "pipelines",
      title: "Pipelines done right: fit on train only, always",
      level: "core",
      body: [
        { type: "p", text: "Everything so far shares one contract: transformations **learn parameters from data** (a mean, a std, a vocabulary, a category map) and must learn them from **training data only**. Do it manually across dozens of columns and you *will* eventually leak — a scaler fit on the full set, a target encoder fit outside CV. The fix is structural: wrap every step in an sklearn **`Pipeline`**, and route columns through a **`ColumnTransformer`**, so `.fit()` touches only training rows and the *entire* preprocessing recipe is a single object you can cross-validate and ship." },
        { type: "callout", variant: "good", text: "**Why this is the whole ballgame.** A `Pipeline` guarantees that when you call `cross_val_score` or `GridSearchCV`, every preprocessing step is **re-fit on each training fold** and merely *applied* to the held-out fold. That makes your offline estimate honest, and the same fitted object does identical preprocessing at inference. No train/serve skew, no leakage, one artifact to deploy." },
        { type: "heading", text: "The anatomy: Pipeline + ColumnTransformer" },
        { type: "p", text: "A `ColumnTransformer` applies different sub-pipelines to different column groups — impute+scale the numerics, impute+one-hot the categoricals — then concatenates the results. Wrap that in a `Pipeline` with your estimator on the end, and the whole thing behaves like one model." },
        { type: "code", lang: "py", code: "import numpy as np, pandas as pd\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score, train_test_split\n\n# --- a mixed-type toy frame ---\ndf = pd.DataFrame({\n    \"age\":    [25, 32, np.nan, 51, 40, 29],\n    \"income\": [40e3, 52e3, 61e3, np.nan, 80e3, 45e3],\n    \"city\":   [\"paris\",\"tokyo\",\"paris\",\"cairo\",\"tokyo\",\"paris\"],\n    \"y\":      [0, 1, 0, 1, 1, 0],\n})\nX, y = df.drop(columns=\"y\"), df[\"y\"]\nnum_cols = [\"age\", \"income\"]\ncat_cols = [\"city\"]\n\n# --- per-type sub-pipelines: impute THEN transform ---\nnum_pipe = Pipeline([\n    (\"impute\", SimpleImputer(strategy=\"median\")),\n    (\"scale\",  StandardScaler()),\n])\ncat_pipe = Pipeline([\n    (\"impute\", SimpleImputer(strategy=\"most_frequent\")),\n    (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\")),\n])\n\npre = ColumnTransformer([\n    (\"num\", num_pipe, num_cols),\n    (\"cat\", cat_pipe, cat_cols),\n])\n\n# --- one object: preprocessing + model ---\nmodel = Pipeline([\n    (\"pre\", pre),\n    (\"clf\", LogisticRegression(max_iter=1000)),\n])\n\n# CV re-fits every step on each TRAIN fold only -> honest, leak-free score\nscores = cross_val_score(model, X, y, cv=3, scoring=\"accuracy\")\nprint(\"cv accuracy:\", scores.mean().round(3))\n\n# fit once on train, apply verbatim to test -> no train/serve skew\nXtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=0)\nmodel.fit(Xtr, ytr)\nprint(\"test accuracy:\", model.score(Xte, yte))\n\nimport joblib\njoblib.dump(model, \"model.joblib\")   # ship the WHOLE recipe as one file" },
        { type: "callout", variant: "gotcha", text: "**The leak that looks fine.** `scaler.fit(X)` *before* `train_test_split` — or `SelectKBest(...).fit(X, y)` on the full frame — leaks test statistics into training and inflates your score. The rule with no exceptions: **split first, then let the Pipeline fit only on the training portion.** If a step has a `.fit()`, it belongs *inside* the pipeline, never run once over everything." },
        { type: "callout", variant: "tip", text: "Because the whole model is one estimator, `GridSearchCV` can tune *preprocessing and model together* — search the imputer strategy, the encoder, and `C` in one grid using `\"step__param\"` names, e.g. `pre__num__impute__strategy: ['mean','median']` and `clf__C: [0.1, 1, 10]`. Everything is refit per fold, so the search itself is leak-free." },
      ]
    },

    {
      id: "outliers-imbalance",
      title: "Outliers & imbalanced data (brief)",
      level: "core",
      body: [
        { type: "heading", text: "Outliers" },
        { type: "p", text: "An **outlier** is a point far from the bulk of the data. It may be an error (a fat-fingered `age = 999`) or a genuine rarity (a legitimate whale customer). Detect with simple rules — the **IQR fence** flags $x < Q_1 - 1.5\\,\\text{IQR}$ or $x > Q_3 + 1.5\\,\\text{IQR}$; the **z-score** rule flags $|z| > 3$; or model-based `IsolationForest` for multivariate outliers. Then choose per column: **cap/winsorize** to a percentile, **transform** (log) to compress the tail, **drop** if clearly erroneous, or **keep** and lean on a robust model. Never blindly delete — an 'outlier' fraud transaction may be exactly what you're trying to predict." },
        { type: "heading", text: "Imbalanced targets" },
        { type: "p", text: "When one class is rare (fraud at 0.2%, churn at 5%), a model can score 99.8% accuracy by predicting 'no fraud' every time — and be useless. Levers: (a) **class weights** (`class_weight=\"balanced\"`) to reweight the loss; (b) **resampling** — undersample the majority, or oversample the minority with **SMOTE**, which synthesizes new minority points by interpolating between neighbors; (c) **threshold tuning** away from 0.5." },
        { type: "callout", variant: "warn", text: "**Two rules that trip everyone up.** (1) **Resample the training fold only** — never oversample before your train/test split, or copies of the same point straddle the split and the score is fantasy. Use `imblearn`'s `Pipeline` so SMOTE runs inside CV. (2) **Stop using accuracy.** On imbalanced data, judge with precision/recall, **F1**, PR-AUC, or ROC-AUC. The full treatment — metric choice, PR vs ROC, threshold selection — is in the **Evaluation & Metrics** deck; this is just the feature-side handoff." },
        { type: "code", lang: "py", code: "import numpy as np\n\n# --- IQR outlier fence from scratch ---\ndef iqr_bounds(x, k=1.5):\n    q1, q3 = np.percentile(x, [25, 75])\n    iqr = q3 - q1\n    return q1 - k*iqr, q3 + k*iqr\n\nx = np.array([10, 11, 12, 11, 13, 200])   # 200 is an outlier\nlo, hi = iqr_bounds(x)\nx_capped = np.clip(x, lo, hi)              # winsorize instead of deleting\nprint(lo, hi, x_capped)\n\n# --- imbalance: class weights (cheapest fix) ---\nfrom sklearn.linear_model import LogisticRegression\nLogisticRegression(class_weight=\"balanced\", max_iter=1000)\n\n# --- imbalance: SMOTE INSIDE an imblearn pipeline so it stays in-fold ---\nfrom imblearn.over_sampling import SMOTE\nfrom imblearn.pipeline import Pipeline as ImbPipeline\nImbPipeline([(\"smote\", SMOTE(random_state=0)),\n             (\"clf\", LogisticRegression(max_iter=1000))])" },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "Feature engineering is a *muscle* — you build it by iterating on real messy data, not by reading. Grab a Kaggle-style tabular set (Titanic, House Prices, Home Credit, or IEEE-CIS Fraud) and do at least three of these end to end, always inside a `Pipeline` so your validation stays honest." },
        { type: "list", ordered: true, items: [
          "**The leak hunt.** Take House Prices. First, *deliberately* leak: fit a `StandardScaler` and a `TargetEncoder` on the full dataset before splitting, and record the cross-val RMSE. Then rebuild everything inside a `Pipeline` + `ColumnTransformer` (split first). Compare the two scores and the gap on the real leaderboard — feel exactly how much a leak lies to you.",
          "**Scaler bake-off.** On a dataset with skewed columns (income, area), train KNN and Logistic Regression under no scaling, standard, min–max, and robust scaling; then train an XGBoost under the same four. Make a table of scores and confirm empirically that scaling swings the first two models and does nothing for the tree.",
          "**Encoding shootout for high cardinality.** On a column with hundreds of levels (`zip`, `product_id`), compare one-hot, frequency, and *out-of-fold* smoothed target encoding. Plot validation score vs the smoothing $\\alpha$, and show that naive (in-fold) target encoding overfits spectacularly.",
          "**Datetime & aggregation features.** On a transactional set, engineer cyclical hour/day encodings, `days_since_last_purchase`, and per-customer rolling `mean/std/count` — all with a strict `shift(1)` so no future leaks in. Measure the lift over the raw columns.",
          "**Automatic feature selection.** Start from a wide engineered matrix (200+ columns). Rank features by mutual information, by Lasso (L1) coefficients, and by permutation importance; take the union of the top-30 from each and show you match (or beat) the full matrix with a fraction of the features.",
          "**Missingness as signal.** Find a column that is MNAR-ish (income, self-reported fields). Compare median-impute alone vs median-impute **+ a missingness indicator** vs `IterativeImputer`. Confirm the indicator often carries real predictive value on its own.",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "Feature engineering is learned by osmosis from good code and hard datasets. These are the resources that repay the time, in recommended order:" },
        { type: "link", url: "https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/", text: "Zheng & Casari — Feature Engineering for Machine Learning (O'Reilly): the canonical practitioner book; numeric, text, and PCA chapters especially" },
        { type: "link", url: "https://www.kaggle.com/learn/feature-engineering", text: "Kaggle Learn — Feature Engineering micro-course (free, hands-on, interactive notebooks)" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/compose.html", text: "scikit-learn User Guide — Pipelines & ColumnTransformer (the leak-proofing machinery, official docs)" },
        { type: "link", url: "https://scikit-learn.org/stable/modules/preprocessing.html", text: "scikit-learn User Guide — Preprocessing data (scalers, encoders, discretization, power transforms)" },
        { type: "link", url: "https://contrib.scikit-learn.org/category_encoders/", text: "category_encoders — target, CatBoost, hashing, leave-one-out encoders with proper cross-fitting" },
        { type: "link", url: "https://arxiv.org/abs/2207.08815", text: "Grinsztajn et al. (2022) — Why do tree-based models still outperform deep learning on tabular data? (the empirical case for engineered features + trees)" },
        { type: "link", url: "https://imbalanced-learn.org/stable/", text: "imbalanced-learn — SMOTE and resampling that stays inside CV via imblearn Pipelines" },
      ]
    },
  ],

  packages: [
    { name: "scikit-learn", why: "the core toolkit — scalers, encoders, imputers, selectors, `Pipeline` + `ColumnTransformer`" },
    { name: "pandas", why: "load, clean, `groupby`-aggregate, and decompose datetimes — the data-wrangling layer" },
    { name: "numpy", why: "the numeric substrate; from-scratch scaling, binning, and outlier fences live here" },
    { name: "category_encoders", why: "target/CatBoost/hashing/frequency encoders with built-in cross-fitting to avoid leakage" },
    { name: "imbalanced-learn", why: "SMOTE and resampling inside an `imblearn.Pipeline` so they stay in-fold" },
    { name: "feature-engine", why: "sklearn-compatible transformers for imputation, encoding, outlier capping, datetime & creation" },
    { name: "sklearn.experimental.IterativeImputer", why: "MICE-style model-based imputation for MAR data" },
  ],

  gotchas: [
    "**Fit on train only.** Any step with a `.fit()` (scaler, encoder, imputer, selector) must learn its parameters from the training fold, never the full dataset. Split first, then fit inside a `Pipeline`.",
    "Target/mean encoding computed outside CV folds leaks the label — dazzling offline scores that collapse in production. Use out-of-fold encoding or sklearn's `TargetEncoder`.",
    "`np.log(x)` is $-\\infty$ at $x=0$ and undefined for negatives. Use `np.log1p` for counts, Yeo–Johnson for signed data, and remember to `expm1` predictions if you logged the target.",
    "Tree models ignore monotonic scaling; distance/gradient/regularized models are highly sensitive to it. Don't waste effort scaling for XGBoost; never skip it for KNN, SVM, or a neural net.",
    "One-hot with an intercept creates perfect collinearity (the dummy trap) — drop one level for linear models. And set `handle_unknown=\"ignore\"` or inference crashes on a new category.",
    "Polynomial features explode as $\\binom{n+d}{d}$ — a few hundred base features at degree 2 is thousands of columns. Expand a hand-picked subset and regularize.",
    "Group aggregations and rolling stats leak the future unless you `shift(1)` / use expanding windows before the prediction time. Compute them out-of-fold, like target encoding.",
    "SMOTE/oversampling before the split copies points across train and test — an illusory score. Resample the training fold only, via an `imblearn` pipeline inside CV.",
  ],

  flashcards: [
    { q: "When does feature scaling matter, and when is it irrelevant?", a: "It matters for distance-based (KNN, K-means, RBF-SVM), gradient-trained (linear/logistic/NN), and regularized models. It's irrelevant for trees/forests/boosting, which split on thresholds and are invariant to monotonic transforms." },
    { q: "Standardization vs min–max vs robust scaling — pick one for each situation.", a: "Standard ($(x-\\mu)/\\sigma$): default, unbounded, keeps distribution shape. Min–max ($[0,1]$): need a bounded range, but one outlier squashes everyone. Robust ($(x-\\text{med})/\\text{IQR}$): outlier-heavy columns." },
    { q: "What is target (mean) encoding and its leakage trap?", a: "Replace a category with its target mean $\\bar y_c$ — compact and high-signal, but a row's own label can enter its own feature. Fix with smoothing toward the global mean plus out-of-fold (cross-fitted) computation." },
    { q: "Define MCAR, MAR, and MNAR.", a: "MCAR: missingness independent of everything (mean-impute is unbiased). MAR: depends on observed features (condition on them). MNAR: depends on the unseen value itself (the missingness is signal — model it / add an indicator)." },
    { q: "Why encode cyclical time features with sine/cosine?", a: "Hour 23 and hour 0 are adjacent, but as raw integers they're maximally far apart. $(\\sin(2\\pi t/T), \\cos(2\\pi t/T))$ places them next to each other on a circle, giving the model the wraparound continuity." },
    { q: "Name the three feature-selection families with an example of each.", a: "Filter (model-agnostic, no interactions): mutual information, $\\chi^2$, correlation. Wrapper (retrains, captures interactions, costly): RFE/RFECV. Embedded (selection during fit): Lasso L1, tree/permutation importance." },
    { q: "How does an L1 penalty perform feature selection?", a: "Adding $\\lambda\\sum_j|w_j|$ to the loss; the corners of the $L_1$ ball drive many weights exactly to zero, so fitting and selecting happen in one step." },
    { q: "Why wrap preprocessing in a Pipeline + ColumnTransformer?", a: "It guarantees every learned step is re-fit on each training fold only and merely applied to held-out data — leak-free cross-validation and no train/serve skew, all shippable as one object." },
    { q: "What's the single most common leakage bug in tabular ML?", a: "Fitting a transformer (scaler, encoder, selector) on the full dataset before the train/test split — leaking test statistics into training and inflating the score. Split first, fit inside the pipeline." },
    { q: "Why is accuracy misleading on imbalanced data, and what to do?", a: "Predicting the majority class scores near-perfect accuracy while being useless. Use class weights, in-fold resampling (SMOTE), and threshold tuning; judge with precision/recall, F1, PR-AUC, or ROC-AUC." },
  ],

  cheatsheet: [
    { label: "Standardize", code: "StandardScaler()" },
    { label: "Min–max [0,1]", code: "MinMaxScaler()" },
    { label: "Robust (median/IQR)", code: "RobustScaler()" },
    { label: "De-skew", code: "PowerTransformer('yeo-johnson')" },
    { label: "De-skew (from scratch)", code: "np.log1p(x)" },
    { label: "One-hot", code: "OneHotEncoder(handle_unknown='ignore')" },
    { label: "Ordinal", code: "OrdinalEncoder(categories=[order])" },
    { label: "Target encode (leak-safe)", code: "TargetEncoder(cv=5)" },
    { label: "Median impute + flag", code: "SimpleImputer('median', add_indicator=True)" },
    { label: "KNN / MICE impute", code: "KNNImputer(); IterativeImputer()" },
    { label: "Interactions", code: "PolynomialFeatures(2, interaction_only=True)" },
    { label: "Select by info", code: "SelectKBest(mutual_info_classif, k=10)" },
    { label: "Recursive elimination", code: "RFE(estimator, n_features_to_select=10)" },
    { label: "Route by column type", code: "ColumnTransformer([...]); Pipeline([...])" },
  ],
});
