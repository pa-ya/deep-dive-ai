(window.FRAMEWORKS = window.FRAMEWORKS || []).push({
  id: "mlops-production",
  name: "Production ML: Tracking, Scaling & Monitoring",
  language: "MLOps",
  group: "MLOps & Deployment",
  navLabel: "Production & Monitoring",
  tagline: "Everything around a deployed model that keeps it working — experiment tracking, pipelines, CI/CD, drift detection, cost control and governance.",
  color: "#475569",
  readMinutes: 48,
  sections: [
    {
      id: "what-is-mlops",
      title: "What MLOps actually is (and why models rot)",
      level: "core",
      body: [
        { type: "p", text: "Training a model that scores 0.94 on your test set is the *first* 10% of the job. The other 90% is everything that happens after: shipping it, versioning it, watching it, retraining it, and proving to an auditor a year later exactly which data and code produced it. That discipline is **MLOps** — DevOps adapted for systems whose behavior depends not just on code, but on *data* and a *learned model* that both change over time." },
        { type: "p", text: "Classic software is deterministic: same input, same output, forever. An ML system has **three** moving artifacts that must all be versioned and reconciled, and only one of them is code:" },
        { type: "table",
          headers: ["Artifact", "In plain DevOps?", "The new problem it creates"],
          rows: [
            ["**Code** (training + serving logic)", "yes — Git handles it", "same as always"],
            ["**Data** (training set, features)", "no", "large, mutable, can't live in Git; must be versioned separately"],
            ["**Model** (learned weights)", "no", "a *function of* code × data × hyperparameters × random seed"],
          ]
        },
        { type: "p", text: "Because the model is a function of the data, and the data reflects a **changing world**, an ML system silently decays even when the code is frozen. This is the defining fact of the field: *a deployed model is a depreciating asset.* Nothing in the codebase changed, yet last quarter's fraud detector is quietly worse this quarter because fraudsters adapted. The world drifted out from under the model. We give this decay a name — **drift** — and a whole section below." },
        { type: "callout", variant: "note", text: "**The one-sentence definition.** MLOps is the set of practices for reliably and repeatably taking ML models to production and *keeping them healthy* — bringing continuous integration, delivery, and monitoring to a system defined by `code + data + model` instead of code alone." },
        { type: "heading", text: "Maturity levels: where does your team sit?" },
        { type: "p", text: "Google's widely-cited MLOps framework describes three levels. Most teams start at 0 and never realize they are stuck there until an incident forces the question 'wait, how *do* we retrain this?'" },
        { type: "table",
          headers: ["Level", "What it looks like", "Retraining"],
          rows: [
            ["**0 — Manual**", "notebooks, hand-run scripts, model handed to ops as a pickle file", "manual, rare, terrifying"],
            ["**1 — ML pipeline automation**", "the *training pipeline* is code; automated data validation & continuous training", "triggered automatically on new data / drift"],
            ["**2 — CI/CD automation**", "pipelines themselves are built, tested & deployed by CI/CD; rapid, reliable experiments", "fully automated, gated, monitored"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**The notebook-to-production cliff.** A model that only exists inside one data scientist's notebook has a bus factor of one and is effectively unreproducible: hidden global state, cells run out of order, an un-pinned library version. The entire first half of this topic is about turning that notebook into versioned, tested, re-runnable *pipelines*." },
        { type: "p", text: "This page walks the full **operational lifecycle**: track experiments → version data & models → build data and training pipelines → gate them with CI/CD → deploy safely → monitor → detect drift → retrain → and govern the whole loop. (The mechanics of *serving* a model behind an API — latency budgets, batching servers, hardware — are their own topic; here we focus on the lifecycle *around* serving.)" },
      ]
    },

    {
      id: "tracking",
      title: "Experiment tracking & reproducibility",
      level: "core",
      body: [
        { type: "p", text: "You will train hundreds of model variants. Without discipline you end up with `model_final.pkl`, `model_final_v2.pkl`, `model_final_REAL.pkl`, and no memory of which learning rate produced the good one. **Experiment tracking** fixes this: every training run logs its **parameters** (what you set), **metrics** (what came out), and **artifacts** (the model, plots, the exact dataset hash). A run becomes a permanent, queryable record." },
        { type: "table",
          headers: ["Logged thing", "Examples", "Why"],
          rows: [
            ["**Params** (inputs)", "learning rate, depth, seed, feature list", "reproduce & compare runs"],
            ["**Metrics** (outputs)", "AUC, F1, loss curve, latency", "rank runs, spot regressions"],
            ["**Artifacts** (files)", "the model, confusion matrix, data hash", "recover the actual asset"],
            ["**Tags / metadata**", "git SHA, dataset version, author", "trace lineage back to source"],
          ]
        },
        { type: "heading", text: "The two dominant tools" },
        { type: "p", text: "Per the 2025 tooling landscape, two options own the space. **MLflow** (open-source, Databricks-backed) is the end-to-end standard — tracking *plus* a built-in **model registry** and deployment hooks; you can self-host it for free. **Weights & Biases (W&B)** is the developer-favorite SaaS with a superb visualization UI and live dashboards. Many teams run both: W&B for interactive research visualization, MLflow owning the registry and deployment path. Neptune.ai is a third, enterprise-scale metadata store." },
        { type: "code", lang: "py", code: "import mlflow\n\nmlflow.set_experiment(\"fraud-detector\")\n\nwith mlflow.start_run(run_name=\"xgb-depth6\"):\n    params = {\"max_depth\": 6, \"lr\": 0.1, \"n_estimators\": 400}\n    mlflow.log_params(params)                 # inputs\n\n    model = train(params, X_train, y_train)\n\n    auc = evaluate(model, X_val, y_val)\n    mlflow.log_metric(\"val_auc\", auc)         # outputs\n\n    mlflow.log_artifact(\"confusion_matrix.png\")   # any file\n    mlflow.set_tag(\"git_sha\", get_git_sha())      # lineage\n    mlflow.sklearn.log_model(model, \"model\")      # the asset itself\n# Everything above is now queryable in the MLflow UI and comparable across runs." },
        { type: "p", text: "The W&B equivalent is nearly identical in shape — `wandb.init(config=params)`, then `wandb.log({\"val_auc\": auc})` inside the loop for live-streaming curves. The APIs rhyme deliberately; a thin wrapper lets you log to both." },
        { type: "heading", text: "The model registry — the bridge from experiment to production" },
        { type: "p", text: "A tracked run is just a record. The **model registry** promotes a chosen run's model into a named, versioned, governed object with lifecycle **stages**: `None → Staging → Production → Archived`. Promotion is a deliberate, logged, often approval-gated transition — this is the checkpoint that stops someone shipping an untested experiment straight to prod." },
        { type: "code", lang: "py", code: "from mlflow import MlflowClient\nclient = MlflowClient()\n\n# Register the model produced by a run -> creates version N\nmv = mlflow.register_model(\"runs:/<run_id>/model\", name=\"fraud-detector\")\n\n# Promote after it passes validation gates (see the CI/CD section)\nclient.transition_model_version_stage(\n    name=\"fraud-detector\", version=mv.version, stage=\"Production\",\n    archive_existing_versions=True,   # old prod model -> Archived automatically\n)\n\n# Serving code always asks for the stage, never a hard-coded version number:\nmodel = mlflow.pyfunc.load_model(\"models:/fraud-detector/Production\")" },
        { type: "callout", variant: "tip", text: "**Serve by a moving pointer, not a hard-coded version.** Your inference service loads `models:/fraud-detector/Production`; rolling back a bad deploy is then a one-line registry change — no redeploy, no code edit. The registry is the single source of truth for 'what is live right now.'" },
        { type: "callout", variant: "warn", text: "**Modern MLflow (2.9+, Dec 2023) deprecated named *stages*** (`transition_model_version_stage`, `.../Production`) in favor of **aliases + tags**: `client.set_registered_model_alias('fraud-detector', 'champion', mv.version)` and load with `models:/fraud-detector@champion`. Aliases are freely named and can point anywhere, so they replaced the fixed `Staging/Production/Archived` ladder. The stage API still works but warns; prefer aliases on new projects." },
        { type: "heading", text: "Version EVERYTHING: Git for code, DVC for data" },
        { type: "p", text: "Reproducibility means being able to regenerate any model bit-for-bit. That requires pinning all three artifacts together. Git versions code, but a 40 GB parquet dataset can't live in Git. **DVC (Data Version Control)** solves this: it stores a tiny `.dvc` pointer file *in Git* (a hash + remote location) while the actual bytes sit in S3/GCS. `git checkout` a commit and `dvc pull`, and you get exactly the code **and** the data that produced a given model." },
        { type: "code", lang: "bash", code: "# Track a large dataset with DVC; the heavy bytes go to remote storage,\n# a lightweight hash pointer goes into Git alongside your code.\ndvc add data/train.parquet          # creates data/train.parquet.dvc (a hash)\ngit add data/train.parquet.dvc data/.gitignore\ngit commit -m \"data: v3 with Q2 samples\"\ndvc push                            # bytes -> s3://my-bucket (the DVC remote)\n\n# A year later, reproduce the exact model from any commit:\ngit checkout <sha> && dvc pull      # restores code AND the matching data\n# code(git SHA) + data(DVC hash) + params(MLflow) = a fully reproducible model" },
        { type: "callout", variant: "gotcha", text: "**The hidden reproducibility killers.** Even with all three artifacts pinned, runs can diverge from: an unset random **seed**; **non-deterministic GPU** kernels (`torch.use_deterministic_algorithms(True)`); an un-pinned library (`scikit-learn` 1.3 vs 1.4 can change results); and **wall-clock-dependent** feature code (`days_since_signup` computed at train time). Pin the seed, pin the environment (`requirements.txt`/Docker digest), and never let 'now' leak into a feature." },
      ]
    },

    {
      id: "pipelines",
      title: "Data & pipelines: validation, feature stores, training-serving skew",
      level: "core",
      body: [
        { type: "p", text: "In production, *data is the largest source of outages* — far more than model code. Garbage flows in from an upstream schema change, a nulled-out column, or a units switch from dollars to cents, and the model silently produces garbage. So the first job of a pipeline is not training; it is **guarding the data**." },
        { type: "heading", text: "Data validation: fail loudly, not silently" },
        { type: "p", text: "A validation step asserts *expectations* about every incoming batch before it reaches the model: column presence, types, ranges, null rates, category sets. **Great Expectations** and **pandera** are the standard tools. The philosophy: a pipeline that crashes on bad data is a feature — silent bad predictions are the bug." },
        { type: "code", lang: "py", code: "import pandera as pa\nfrom pandera import Column, Check\n\nschema = pa.DataFrameSchema({\n    \"age\":     Column(int,   Check.in_range(18, 120)),\n    \"income\":  Column(float, Check.ge(0), nullable=False),\n    \"country\": Column(str,   Check.isin([\"US\", \"CA\", \"GB\", \"DE\"])),\n    \"score\":   Column(float, Check.in_range(0.0, 1.0)),\n})\n\n# Runs at the top of every pipeline; raises SchemaError on the FIRST bad batch\n# instead of letting a nulled 'income' column poison predictions downstream.\nclean = schema.validate(raw_df, lazy=True)   # lazy=True collects ALL failures" },
        { type: "heading", text: "Feature stores: compute a feature once, serve it everywhere" },
        { type: "p", text: "A **feature store** (e.g. Feast, Tecton) is a central catalog of feature definitions with two synchronized paths: an **offline** store (a warehouse, for building large training sets) and an **online** store (a low-latency KV store like Redis, for serving fresh features at request time). The point is that `avg_purchase_30d` is *defined once* and read identically by both training and serving — which directly attacks the field's nastiest bug." },
        { type: "heading", text: "Training-serving skew — the bug that looks like a good offline score" },
        { type: "p", text: "**Training-serving skew** is any discrepancy between how a feature is computed during training vs. during serving. The model aces offline evaluation and quietly underperforms in production, because it is being fed subtly different numbers than it learned on. Classic causes:" },
        { type: "list", ordered: false, items: [
          "**Two codebases.** Features engineered in a training notebook (pandas), re-implemented by hand in the serving service (Java/Go). The two implementations diverge. A feature store fixes this by having *one* definition.",
          "**Time travel / label leakage.** A training feature accidentally uses data from *after* the prediction moment (e.g. joining tomorrow's aggregate). Impossible to reproduce at serving time; inflates offline metrics fraudulently.",
          "**Distribution skew.** Training data was sampled or filtered differently than live traffic (you trained on logged-in users; you serve everyone).",
        ]},
        { type: "callout", variant: "warn", text: "Training-serving skew is the single most common reason a model that 'worked in the notebook' fails in production. The offline metric is not lying about the *model*; it is lying about the *inputs*. Guard against it by computing features from a single shared definition and by **logging live features** so you can diff them against training features." },
        { type: "heading", text: "Orchestrating the training pipeline" },
        { type: "p", text: "A training pipeline is a DAG of steps — *ingest → validate → build features → train → evaluate → register* — run on a schedule or a trigger. **Airflow** is the general-purpose incumbent (schedule anything); **Prefect** and **Dagster** are the modern, Pythonic, data-aware alternatives; **Kubeflow Pipelines** is Kubernetes-native and ML-specialized. Any of them turns 'the notebook I re-run by hand' into 'a pipeline that re-runs itself.'" },
        { type: "code", lang: "py", code: "# A Prefect training pipeline: each @task is a retriable, observable node.\nfrom prefect import flow, task\n\n@task(retries=2)\ndef ingest():            return load_from_warehouse()\n@task\ndef validate(df):        return schema.validate(df)      # pandera gate\n@task\ndef features(df):        return build_features(df)        # shared definitions\n@task\ndef train(X, y):         return fit_model(X, y)\n@task\ndef evaluate(m, Xv, yv): return score(m, Xv, yv)\n\n@flow(name=\"nightly-training\")\ndef training_pipeline():\n    df = validate(ingest())\n    X, y = features(df)\n    model = train(X, y)\n    metrics = evaluate(model, X_val, y_val)\n    if metrics[\"auc\"] > 0.90:          # a validation GATE (next section)\n        register_and_promote(model, metrics)\n\nif __name__ == \"__main__\":\n    training_pipeline()" },
        { type: "callout", variant: "tip", text: "**Pipelines are the level-1 unlock.** The moment your training process is a parameterized, re-runnable pipeline instead of a notebook, *continuous training* becomes trivial: point the same DAG at fresh data on a schedule or a drift trigger. Everything downstream — CI/CD, automated retraining — depends on this one step." },
      ]
    },

    {
      id: "cicd",
      title: "CI/CD for ML: testing, gates & safe deploys",
      level: "core",
      body: [
        { type: "p", text: "CI/CD for ML extends software CI/CD with two ML-specific concerns: you must **test data and models**, not just code, and you must **deploy probabilistically** because a new model can be correct code yet worse behavior. The pipeline you build has to catch a model that *runs perfectly and predicts badly*." },
        { type: "heading", text: "Three layers of testing" },
        { type: "table",
          headers: ["Layer", "What it checks", "Example"],
          rows: [
            ["**Code tests**", "logic is correct (unit/integration)", "feature fn handles nulls; API returns 200"],
            ["**Data tests**", "inputs meet expectations", "no schema drift; null rate < 1% (pandera)"],
            ["**Model tests**", "behavior is acceptable", "AUC ≥ baseline; no regression on key slices; latency budget met"],
          ]
        },
        { type: "p", text: "Model tests are the new idea. Beyond an aggregate metric, test **behavior**: *invariance* (changing an irrelevant feature shouldn't flip the prediction), *directional expectations* (raising income should not raise default probability), and **per-slice** performance (the model is 0.94 overall but 0.61 for one country — an aggregate metric hides this)." },
        { type: "heading", text: "The model validation gate" },
        { type: "p", text: "Before any promotion, a **gate** compares the candidate against the current production model and a naive baseline. Pass all checks → auto-register to Staging. This is where the 'if `auc > 0.90`' from the pipeline above grows teeth." },
        { type: "code", lang: "py", code: "def validation_gate(candidate, prod, X_test, y_test) -> bool:\n    c = full_report(candidate, X_test, y_test)\n    p = full_report(prod,      X_test, y_test)\n    checks = {\n        \"beats_prod_auc\":   c[\"auc\"]      >= p[\"auc\"] - 0.005,   # no regression\n        \"beats_baseline\":   c[\"auc\"]      >= 0.75,               # sanity floor\n        \"no_slice_drop\":    min(c[\"per_country_auc\"].values()) >= 0.70,\n        \"latency_ok\":       c[\"p99_ms\"]   <= 50,\n        \"fairness_ok\":      c[\"demographic_parity_gap\"] <= 0.10,\n    }\n    for name, ok in checks.items():\n        print(f\"  {'PASS' if ok else 'FAIL'}  {name}\")\n    return all(checks.values())   # a single FAIL blocks the deploy" },
        { type: "callout", variant: "gotcha", text: "**Never gate on a single aggregate number.** A candidate can raise overall AUC by 0.01 while *dropping* 8 points on your highest-value customer segment. Always gate on **per-slice** metrics and fairness, not just the headline. The aggregate is the average of things you actually care about individually." },
        { type: "heading", text: "Deploying safely: canary, blue-green, shadow" },
        { type: "p", text: "Passing offline gates is necessary, not sufficient — offline test data is not live traffic. So you release *gradually* and watch. Four progressive-delivery patterns, from safest-to-observe to full commit:" },
        { type: "table",
          headers: ["Strategy", "How it works", "Best for"],
          rows: [
            ["**Shadow**", "new model receives real traffic in parallel; its outputs are logged but **not served** to users", "zero-risk validation on live inputs before anyone sees a prediction"],
            ["**Canary**", "route a small % (1→5→25→100) of traffic to the new model; watch metrics, roll back on regression", "gradual, monitored rollout"],
            ["**Blue-green**", "stand up the new model fully, flip 100% of traffic at once; instant rollback by flipping back", "fast cutover with a clean revert path"],
            ["**A/B test**", "split traffic deliberately to *measure* a business KPI difference with statistical significance", "proving the new model actually moves revenue/engagement"],
          ]
        },
        { type: "callout", variant: "note", text: "**Shadow vs. A/B are different questions.** Shadow asks *'does the new model run correctly and sanely on real inputs?'* (an engineering check — users never see it). A/B asks *'is the new model actually better for the business?'* (a statistics check — you compare a KPI across two live groups and test for significance, exactly like the hypothesis testing in the probability track). Offline AUC gains routinely fail to translate into online KPI gains; A/B is how you find out." },
        { type: "heading", text: "Automated retraining" },
        { type: "p", text: "Continuous training closes the loop: a trigger — a **schedule** (nightly/weekly), a **drift alarm** (from monitoring), or a **performance drop** (once ground truth arrives) — kicks off the training pipeline, which runs the validation gate, and only then promotes. Crucially, *retraining is automatic; promotion is gated.* You never let a pipeline ship a model to prod without passing the same checks a human deploy would." },
      ]
    },

    {
      id: "monitoring",
      title: "Monitoring: operational health vs. ML health",
      level: "core",
      body: [
        { type: "p", text: "A deployed model needs two completely different kinds of monitoring, and teams that only do the first get blindsided by the second. **Operational monitoring** answers *'is the service up?'* — the same signals as any web service. **ML monitoring** answers *'is the model still right?'* — and a model can be perfectly healthy operationally while its predictions quietly rot." },
        { type: "table",
          headers: ["Layer", "Metrics", "Tools", "Failure it catches"],
          rows: [
            ["**Operational**", "latency (p50/p95/p99), throughput (QPS), error rate, CPU/GPU/mem", "Prometheus, Grafana, Datadog", "service is slow / down / OOM"],
            ["**ML — inputs**", "feature distributions, null rates, drift scores", "Evidently, WhyLabs, Arize", "the world changed under the model"],
            ["**ML — outputs**", "prediction distribution, confidence, class balance", "Evidently, custom", "model output shifting (early warning)"],
            ["**ML — quality**", "accuracy / AUC / precision over time", "needs ground truth", "model is actually getting worse"],
          ]
        },
        { type: "callout", variant: "gotcha", text: "**Green dashboards, rotting model.** Latency is 12 ms, error rate is 0.0%, uptime is 99.99% — and the model has been degrading for six weeks because customer behavior shifted. Every *operational* signal is perfect. This is the trap: operational monitoring cannot see model decay. You must monitor the *predictions and inputs*, not just the server." },
        { type: "heading", text: "Log every prediction" },
        { type: "p", text: "The foundation of all ML monitoring is a **prediction log**: for every request, persist the input features, the model version, the output, the confidence, a timestamp, and a join key. This log is what you later (a) join against ground truth to measure real accuracy, (b) run drift detection over, and (c) replay to debug 'why did it predict *that* for customer 8412?'." },
        { type: "code", lang: "py", code: "import json, time, uuid\n\ndef predict_and_log(features: dict) -> dict:\n    request_id = str(uuid.uuid4())\n    proba = model.predict_proba(vectorize(features))[0, 1]\n    record = {\n        \"request_id\":    request_id,      # join key for later ground truth\n        \"ts\":            time.time(),\n        \"model_version\": MODEL_VERSION,   # which model made this call\n        \"features\":      features,         # raw inputs -> feeds drift detection\n        \"prediction\":    float(proba),\n        \"decision\":      proba > 0.5,\n    }\n    prediction_log.write(json.dumps(record))   # -> warehouse / Kafka / S3\n    return {\"id\": request_id, \"score\": proba}\n# The features you log here are EXACTLY what you diff for training-serving skew\n# and feed into the drift tests in the next section." },
        { type: "heading", text: "Ground-truth lag: why quality metrics are always late" },
        { type: "p", text: "You cannot compute accuracy until you know the *right answer*, and the right answer arrives late — or never. A loan-default label matures over **months**; an ad-click label arrives in seconds; a medical-diagnosis label may need a specialist's confirmation. This delay is **ground-truth lag**, and it dictates your whole monitoring strategy." },
        { type: "callout", variant: "warn", text: "**Ground-truth lag is why drift detection exists.** If your default labels take 90 days to mature, you *cannot* wait 90 days to discover the model broke. So you monitor a **proxy that needs no labels** — the *input distribution* — and treat a shift in it as an early-warning alarm. Drift on the inputs is the leading indicator; accuracy on the outputs is the lagging confirmation. That is the entire motivation for the next section." },
        { type: "callout", variant: "tip", text: "**Close the loop with delayed joins.** Architect the ground-truth join as an asynchronous pipeline: predictions land in the log now, labels trickle in over days/weeks and are joined on `request_id` when they arrive. Your true accuracy dashboard is always reporting on a *cohort from N days ago* — and that is fine, as long as you know N and watch drift in the meantime." },
      ]
    },

    {
      id: "drift",
      title: "Drift: the four kinds and how to detect them",
      level: "core",
      body: [
        { type: "p", text: "**Drift** is the umbrella term for 'the statistical assumptions the model was trained under no longer hold.' Supervised learning assumes training and serving data are drawn from the same joint distribution $P(X, Y)$. When that breaks, the model degrades. Since $P(X, Y) = P(Y \\mid X)\\,P(X)$, there are exactly two things that can move — the inputs $P(X)$ or the input→output relationship $P(Y \\mid X)$ — which gives us a clean taxonomy." },
        { type: "table",
          headers: ["Type", "What shifts", "Plain-English example"],
          rows: [
            ["**Covariate / data drift**", "$P(X)$ changes; $P(Y\\mid X)$ stable", "a new marketing campaign brings younger users; the age→churn rule is unchanged"],
            ["**Concept drift**", "$P(Y\\mid X)$ changes", "fraudsters adapt: the *same* transaction pattern that meant 'fraud' now means 'normal'"],
            ["**Label / prior drift**", "$P(Y)$ changes (class balance)", "fraud rate jumps from 1% to 4% during a holiday"],
            ["**Upstream / schema drift**", "a data-engineering break", "a column switches units, gets nulled, or a category is renamed"],
          ]
        },
        { type: "callout", variant: "note", text: "**Why the distinction is practical.** *Covariate drift* is often survivable and detectable **without labels** — you just watch the inputs, and retraining on fresh data usually fixes it. *Concept drift* is the dangerous one: the rules of the world changed, the inputs may look totally normal, and you often can't confirm it until lagging ground truth arrives. Different diagnosis, different cure." },
        { type: "heading", text: "Detecting drift: three distances between distributions" },
        { type: "p", text: "Detection reduces to a statistical question from the probability track: *is this batch of live data drawn from the same distribution as my training reference?* Three standard measures, per feature, comparing a **reference** window (training) to a **current** window (recent production):" },
        { type: "heading", text: "1) Population Stability Index (PSI)" },
        { type: "p", text: "Bin both distributions into the same buckets, then sum a symmetric log-ratio penalty over bins. PSI is the industry default (from credit scoring) precisely because it comes with battle-tested thresholds." },
        { type: "math", tex: String.raw`\text{PSI} = \sum_{b=1}^{B} \left(a_b - e_b\right)\,\ln\!\frac{a_b}{e_b}` },
        { type: "p", text: "where $e_b$ and $a_b$ are the *expected* (reference) and *actual* (current) proportions in bin $b$. Rule of thumb: **PSI $< 0.1$** = no meaningful shift, **$0.1$–$0.2$** = moderate (investigate), **$> 0.2$** = significant drift (act)." },
        { type: "heading", text: "2) Kolmogorov–Smirnov (KS) test" },
        { type: "p", text: "For continuous features, the two-sample **KS statistic** is the largest gap between the two empirical cumulative distribution functions — a nonparametric test with a p-value." },
        { type: "math", tex: String.raw`D_{\text{KS}} = \sup_{x}\, \left| F_{\text{ref}}(x) - F_{\text{cur}}(x) \right|` },
        { type: "callout", variant: "gotcha", text: "**KS is too sensitive at scale.** With millions of rows the KS p-value goes 'significant' on drift far too tiny to matter — you drown in alerts. At large batch sizes prefer an *effect-size* measure with a fixed threshold (PSI, or Wasserstein distance) over a *p-value* test, or calibrate the threshold against batch size. Statistical significance is not practical significance." },
        { type: "heading", text: "3) KL divergence (relative entropy)" },
        { type: "p", text: "Straight from the probability track: **KL divergence** measures the information lost when the reference distribution $q$ is used to approximate the current distribution $p$. It is asymmetric and unbounded, so in practice its symmetric, bounded cousin (Jensen–Shannon divergence) is often preferred for drift dashboards." },
        { type: "math", tex: String.raw`D_{\text{KL}}(p \parallel q) = \sum_{i} p_i \,\log \frac{p_i}{q_i}` },
        { type: "code", lang: "py", code: "import numpy as np\n\ndef psi(reference, current, bins=10):\n    \"\"\"Population Stability Index between two 1-D samples.\"\"\"\n    # Fix bin edges on the REFERENCE, then bucket both the same way.\n    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))\n    edges[0], edges[-1] = -np.inf, np.inf\n    e, _ = np.histogram(reference, bins=edges)\n    a, _ = np.histogram(current,   bins=edges)\n    e = np.clip(e / e.sum(), 1e-6, None)   # proportions; clip to avoid log(0)\n    a = np.clip(a / a.sum(), 1e-6, None)\n    return float(np.sum((a - e) * np.log(a / e)))\n\nref = np.random.normal(0, 1, 10_000)\nok  = np.random.normal(0, 1, 5_000)      # same distribution\nbad = np.random.normal(0.6, 1.3, 5_000)  # shifted + wider\nprint(f\"stable  PSI = {psi(ref, ok):.3f}\")   # ~0.00  -> no drift\nprint(f\"drifted PSI = {psi(ref, bad):.3f}\")   # >0.2   -> significant drift" },
        { type: "callout", variant: "tip", text: "**Use a library in production.** [Evidently](https://www.evidentlyai.com/) computes PSI/KS/Wasserstein/JS per column, picks a sensible default test per feature type, and renders drift dashboards + alerts out of the box. Roll your own PSI to *understand* it (above); run Evidently to *operate* it." },
        { type: "heading", text: "When to retrain" },
        { type: "p", text: "Drift is a *signal*, not an *order*. The decision to retrain balances the cost of a stale model against the cost and risk of a new one:" },
        { type: "list", ordered: false, items: [
          "**Scheduled** — retrain every N days regardless. Simple, predictable; wasteful if stable, too slow if volatile.",
          "**Triggered by drift** — retrain when PSI/accuracy crosses a threshold. Responsive, but needs a monitoring stack and guards against thrashing on noise.",
          "**Triggered by performance** — the gold standard *when you have timely labels*: retrain when measured accuracy drops. Blocked by ground-truth lag, which is why drift is often the practical proxy.",
        ]},
        { type: "callout", variant: "warn", text: "**Drift ≠ decay; measure the metric that matters.** A feature can drift wildly while accuracy is untouched (the model never leaned on it), and — the scary case — accuracy can crater with *minimal* input drift (pure concept drift). Real incident pattern: fraud accuracy fell 3% in two weeks while PSI looked benign. Treat drift as a smoke alarm, but confirm with performance whenever labels allow, and don't auto-retrain on drift alone without a validation gate." },
      ]
    },

    {
      id: "cost",
      title: "GPU & cost management",
      level: "core",
      body: [
        { type: "p", text: "ML infrastructure is expensive, and the bill splits into two very different shapes. **Training** is bursty, parallel, and can tolerate interruption. **Inference** is steady, latency-sensitive, and (for a popular model) runs 24/7 forever — so over a model's life, *inference usually dominates the total bill.* The optimizations differ accordingly." },
        { type: "table",
          headers: ["", "Training", "Inference"],
          rows: [
            ["Pattern", "bursty, hours-to-days", "continuous, 24/7"],
            ["Latency", "doesn't matter", "critical (user-facing)"],
            ["Interruptible?", "yes (checkpoint & resume)", "no"],
            ["Best hardware buy", "**spot / preemptible** instances", "reserved / autoscaling on-demand"],
            ["Main lever", "finish fast, then release the GPUs", "throughput per dollar"],
          ]
        },
        { type: "heading", text: "Spot instances for training" },
        { type: "p", text: "**Spot / preemptible** GPUs are the same hardware at 60–90% off, with one catch: the cloud can reclaim them with ~2 minutes' notice. Training tolerates this *if* you **checkpoint** regularly — save model + optimizer state to durable storage every few minutes, and resume from the last checkpoint after a preemption. This one practice is the biggest single training-cost lever." },
        { type: "heading", text: "Right-sizing: the biggest waste is idle GPUs" },
        { type: "p", text: "The most common cloud-bill disaster is a fleet of GPUs sitting at 5% utilization — provisioned for a peak that rarely comes. Fixes: **autoscaling** (including scale-to-zero for spiky traffic), matching **GPU class** to the job (you do not need an H100 to serve a distilled 1B model — an older/smaller GPU or even CPU may win on cost-per-request), and *measuring utilization* before buying more." },
        { type: "heading", text: "Batching: the throughput multiplier" },
        { type: "p", text: "GPUs are massively parallel, so processing 32 requests together costs barely more wall-clock than processing one. **Dynamic batching** — hold incoming requests for a few milliseconds, run them as one batch — trades a little latency for a large throughput (and cost-per-request) win. It is the highest-leverage inference optimization, which is why every serving stack builds it in." },
        { type: "heading", text: "Model compression: buy less compute per prediction" },
        { type: "p", text: "You can also shrink the *model* so each prediction costs less — these techniques (developed in the neural-networks track) are cost levers in production:" },
        { type: "table",
          headers: ["Technique", "Idea", "Typical win"],
          rows: [
            ["**Quantization**", "store/compute weights in int8 (or fp16) instead of fp32", "~4× smaller (int8) / ~2× (fp16), ~2–4× faster, tiny accuracy hit"],
            ["**Distillation**", "train a small 'student' to mimic a big 'teacher'", "much smaller model at near-teacher quality"],
            ["**Pruning**", "delete near-zero weights", "smaller, sparser model"],
          ]
        },
        { type: "callout", variant: "tip", text: "**Cost is an SLO.** Track **cost-per-1000-predictions** as a first-class metric next to latency and accuracy, and put it on the dashboard. A model that is 0.3% more accurate but 5× more expensive per call is usually the *wrong* call — and you can only see that trade-off if you measure the dollars. For LLMs this becomes cost-per-token, next." },
      ]
    },

    {
      id: "llmops",
      title: "LLMOps: what changes when the model is an LLM",
      level: "core",
      body: [
        { type: "p", text: "**LLMOps** is MLOps for large language models, and it inherits everything above — tracking, versioning, monitoring, cost, governance — but the artifacts and failure modes shift. Often you did not *train* the model at all (you call an API), so your 'model' is really a **prompt + retrieval + tools + parameters**, and *those* are what you now have to version, evaluate, and monitor." },
        { type: "heading", text: "Prompts are code — version them like code" },
        { type: "p", text: "A prompt change can swing quality as much as a model retrain, so a raw prompt string buried in source is a liability. Treat prompts as versioned, testable artifacts: store them with an ID and version, tag every production call with the prompt version (exactly like a model version), and diff outputs when you change one. A prompt registry is to LLMOps what the model registry is to classic ML." },
        { type: "heading", text: "Evaluation: no single accuracy number" },
        { type: "p", text: "LLM outputs are open-ended text, so 'accuracy' fragments into a **suite** you run on every prompt/model change — a regression test for language:" },
        { type: "list", ordered: false, items: [
          "**Reference-based** — compare to a gold answer where one exists (exact match, embedding similarity, ROUGE/BLEU for summarization/translation).",
          "**LLM-as-judge** — a stronger model scores outputs against a rubric (helpfulness, correctness, tone). Cheap and scalable, but the judge has biases; calibrate it against human labels.",
          "**Task-specific programmatic checks** — 'is the JSON valid?', 'does the SQL run?', 'is the cited source real?' — deterministic and cheap; use them wherever the task allows.",
          "**Human eval** — the ground truth, on a sampled slice, for what automated metrics miss.",
        ]},
        { type: "callout", variant: "gotcha", text: "**LLMs are non-deterministic — freeze what you can.** At `temperature > 0` the same prompt yields different outputs, so a passing eval can fail on the next run. For reproducible evaluation pin `temperature=0`, pin the **exact model snapshot** (a provider bumping the model *behind the same name* silently changes your behavior — a version-drift bug unique to hosted LLMs), and pin a seed where offered. Log the model snapshot ID on every call." },
        { type: "heading", text: "Guardrails: validate inputs and outputs" },
        { type: "p", text: "An LLM will confidently emit wrong, unsafe, or off-policy text, so production wraps it in **guardrails**: input filters (block prompt-injection, PII, off-topic) and output validators (schema/format checks, toxicity and PII scanners, groundedness checks against retrieved context, refusal-when-uncertain). Guardrails are the LLM analogue of the data-validation gates from the pipeline section — fail loudly rather than serve garbage." },
        { type: "heading", text: "Cost, caching & token monitoring" },
        { type: "p", text: "LLM cost is **per token**, in and out, so it is driven by prompt length, context size, and output length. The monitoring and levers:" },
        { type: "list", ordered: false, items: [
          "**Token monitoring** — track input/output tokens, cost, and latency per request and per feature; a runaway retrieval step that stuffs 30k tokens into context shows up here first.",
          "**Caching** — cache identical (or embedding-near) prompts to skip the call entirely; use provider **prompt caching** to get a discount on a long, reused system-prompt/context prefix.",
          "**Model routing** — send easy requests to a small/cheap model and only escalate hard ones to the frontier model (a distillation-style cost idea at the routing layer).",
        ]},
        { type: "heading", text: "Tracing multi-step chains" },
        { type: "p", text: "A single user request may fan out into many LLM calls, retrievals, and tool invocations. **Tracing** (LangSmith, Langfuse, Arize Phoenix, OpenTelemetry) records the whole tree — every prompt, response, latency, and token count per step — so you can debug *which* step in an agent chain produced the bad answer or burned the tokens. It is distributed tracing adapted for LLM apps, and it is indispensable once you go beyond a single call. (Agent architectures are their own topic; here, tracing is simply how you *observe* them in production.)" },
        { type: "callout", variant: "tip", text: "**LLMOps is 90% classic MLOps with new nouns.** Prompt version ≈ model version; eval harness ≈ validation gate; guardrails ≈ data validation; token cost ≈ cost-per-prediction; tracing ≈ prediction logging. If you internalized the earlier sections, LLMOps is a re-skinning, not a new discipline." },
      ]
    },

    {
      id: "governance",
      title: "Governance, fairness, security & privacy",
      level: "core",
      body: [
        { type: "p", text: "Once a model makes decisions that affect people — loans, hiring, diagnoses, content — 'it scores well' is no longer sufficient. **Governance** is the discipline of making models accountable, fair, secure, and auditable. Increasingly it is also a legal requirement (the EU AI Act, sector regulations), not a nicety." },
        { type: "heading", text: "Reproducibility & lineage (the audit trail)" },
        { type: "p", text: "Governance rests on the reproducibility we built earlier: for any live prediction you must be able to answer *which model version, trained on which data, by which code, with which params, promoted by whom, made this decision?* That chain — Git SHA + DVC data hash + MLflow run + registry transition log + prediction log — is your audit trail. If you cannot reconstruct it, you cannot govern it." },
        { type: "heading", text: "Model cards: the datasheet for a model" },
        { type: "p", text: "A **model card** (Mitchell et al., 2019) is a short, standardized document shipped *with* every model: what it does, intended and out-of-scope uses, training data and its limitations, performance **broken down by subgroup**, ethical considerations, and known failure modes. It is the human-readable companion to the registry entry, and it forces the per-slice thinking that aggregate metrics hide." },
        { type: "heading", text: "Bias & fairness auditing" },
        { type: "p", text: "Aggregate accuracy can conceal that a model is systematically worse — or systematically harsher — for a protected group. Fairness auditing measures disparity across groups with an explicit metric; the choice of metric is itself an ethical decision, because the common ones are **mathematically incompatible** and cannot all hold at once:" },
        { type: "table",
          headers: ["Fairness notion", "Requires (roughly) equal across groups", "Tension"],
          rows: [
            ["**Demographic parity**", "positive-prediction *rate* $P(\\hat Y{=}1)$", "ignores real base-rate differences"],
            ["**Equal opportunity**", "true-positive rate (recall)", "may differ on false positives"],
            ["**Equalized odds**", "*both* TPR and FPR", "strictest; hardest to satisfy"],
            ["**Calibration**", "$P(Y{=}1 \\mid \\hat p)$ matches $\\hat p$ per group", "provably conflicts with equalized odds"],
          ]
        },
        { type: "callout", variant: "warn", text: "**Fairness is a choice, not a checkbox.** An impossibility result (Kleinberg et al., 2016; the COMPAS debate) proves you generally *cannot* satisfy calibration and equalized odds simultaneously when base rates differ. You must pick which fairness definition matches the harm you are preventing, justify it, and document it in the model card. Tools like Fairlearn and AIF360 measure the gaps; only humans can choose the objective." },
        { type: "heading", text: "Security & privacy" },
        { type: "p", text: "Models add attack surface and privacy obligations beyond ordinary software:" },
        { type: "table",
          headers: ["Threat / duty", "What it is", "Mitigation"],
          rows: [
            ["**Adversarial inputs**", "crafted inputs that fool the model", "adversarial training, input validation, anomaly detection"],
            ["**Data poisoning**", "attacker corrupts training data", "data provenance, validation gates, anomaly checks"],
            ["**Model / membership inference**", "extracting weights or 'was this person in the training set?'", "rate limiting, output limits, **differential privacy**"],
            ["**PII leakage**", "training data (or an LLM) emits personal data", "PII scrubbing, DP, output guardrails"],
            ["**Prompt injection** (LLMs)", "malicious text hijacks the model's instructions", "input/output guardrails, privilege separation"],
          ]
        },
        { type: "callout", variant: "note", text: "**Privacy has a math tool: differential privacy.** DP adds calibrated noise (during training or aggregation) so that no single individual's presence in the data measurably changes the output — bounded by a privacy budget $\\varepsilon$. It is the rigorous way to promise 'the model did not memorize *you*,' and it trades a little accuracy for a provable privacy guarantee." },
      ]
    },

    {
      id: "worked-example",
      title: "Worked example: instrument a model with tracking + a drift check",
      level: "core",
      body: [
        { type: "p", text: "Let's tie the core threads together end to end: train a model **with MLflow tracking**, register it, log predictions in a serving stub, then run a **PSI drift check** of live inputs against the training reference — the exact leading-indicator loop from the monitoring and drift sections, in ~60 lines." },
        { type: "heading", text: "1) Train + track + register" },
        { type: "code", lang: "py", code: "import mlflow, mlflow.sklearn, numpy as np\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import roc_auc_score\nfrom sklearn.model_selection import train_test_split\n\nX, y = load_data()                       # your features/labels\nXtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=0)\n\nmlflow.set_experiment(\"churn\")\nwith mlflow.start_run(run_name=\"rf-200\") as run:\n    params = {\"n_estimators\": 200, \"max_depth\": 8, \"random_state\": 0}\n    mlflow.log_params(params)\n\n    model = RandomForestClassifier(**params).fit(Xtr, ytr)\n    auc = roc_auc_score(yval, model.predict_proba(Xval)[:, 1])\n    mlflow.log_metric(\"val_auc\", auc)\n\n    # Save the TRAINING feature stats as the drift reference (an artifact).\n    ref = {\"mean\": Xtr.mean(0).tolist(), \"cols\": list(Xtr.columns)}\n    mlflow.log_dict(ref, \"drift_reference.json\")\n    mlflow.sklearn.log_model(model, \"model\")\n\n# Gate, then register+promote (see the CI/CD section).\nif auc >= 0.80:\n    mlflow.register_model(f\"runs:/{run.info.run_id}/model\", \"churn\")" },
        { type: "heading", text: "2) Serve + log every prediction" },
        { type: "code", lang: "py", code: "import time, json\nmodel = mlflow.pyfunc.load_model(\"models:/churn/Production\")  # serve by STAGE\nprediction_log = open(\"preds.jsonl\", \"a\")\n\ndef serve(features_row):\n    p = float(model.predict(features_row.to_frame().T)[0])\n    prediction_log.write(json.dumps({\n        \"ts\": time.time(), \"features\": features_row.to_dict(), \"pred\": p,\n    }) + \"\\n\")\n    return p" },
        { type: "heading", text: "3) Batch drift check (run on a schedule)" },
        { type: "code", lang: "py", code: "import numpy as np, pandas as pd\n\ndef psi(reference, current, bins=10):\n    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))\n    edges[0], edges[-1] = -np.inf, np.inf\n    e = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-6, None)\n    a = np.clip(np.histogram(current,   edges)[0] / len(current),   1e-6, None)\n    return float(np.sum((a - e) * np.log(a / e)))\n\n# Load recent production inputs from the prediction log, compare per feature.\nlive = pd.DataFrame([json.loads(l)[\"features\"] for l in open(\"preds.jsonl\")])\nalerts = {c: psi(Xtr[c], live[c]) for c in Xtr.columns}\nfor col, score in sorted(alerts.items(), key=lambda kv: -kv[1]):\n    flag = \"DRIFT\" if score > 0.2 else (\"watch\" if score > 0.1 else \"ok\")\n    print(f\"{col:20s} PSI={score:.3f}  [{flag}]\")\n\nif any(s > 0.2 for s in alerts.values()):\n    trigger_retraining_pipeline()   # drift alarm -> continuous training + gate" },
        { type: "callout", variant: "good", text: "**That is the whole operational loop in miniature:** track → gate → register → serve-by-stage → log predictions → detect drift → trigger a gated retrain. Every production ML system is this pattern, scaled up with better tools (Evidently for drift, Airflow/Prefect for orchestration, a real feature store and prediction warehouse) — but the skeleton does not change." },
      ]
    },

    {
      id: "projects",
      title: "Projects & practice",
      level: "core",
      body: [
        { type: "callout", variant: "note", text: "MLOps is learned by wiring the loop together, not by reading about it. Each project below builds one real link in the chain — do at least two end to end." },
        { type: "list", ordered: true, items: [
          "**Full tracking + registry loop.** Take any model you've built, wrap training in MLflow, log params/metrics/artifacts across ~20 runs with different hyperparameters, compare them in the UI, then register the best and load it back *by stage* (`models:/name/Production`). Practice a rollback: promote v2, then transition back to v1 with one API call.",
          "**Reproducibility challenge.** Version a dataset with **DVC** and your code with **Git**. Train a model, record its metric, then `git checkout` an earlier commit + `dvc pull` and confirm you reproduce the *identical* number. Now deliberately break it — bump a library version or unset the seed — and watch the result drift. Fix it by pinning the environment.",
          "**Data validation gate.** Write a **pandera** or **Great Expectations** schema for a dataset (types, ranges, null rates, category sets). Feed it a deliberately corrupted batch (nulled column, out-of-range value, renamed category) and confirm the pipeline *fails loudly* at the gate instead of training on garbage.",
          "**Drift detector from scratch, then Evidently.** Implement PSI and the KS test by hand; validate them on a synthetic shifted distribution. Then run **Evidently** on the same data, generate its drift report, and compare your numbers to the library's. Wire a threshold to print a retrain trigger.",
          "**CI/CD with a model gate.** Set up a GitHub Actions workflow that, on every push, runs code tests + data tests + a **model validation gate** (candidate must beat a saved baseline AND every per-slice metric). Make it block the merge when a slice regresses even though overall AUC improves.",
          "**Mini LLMOps harness.** For a small LLM task, build a prompt registry (versioned prompt strings), an **eval harness** (5–10 test cases with programmatic checks + LLM-as-judge), and a token/cost logger. Change the prompt, re-run the suite, and diff the scores and cost before you 'ship.'",
        ]},
      ]
    },

    {
      id: "references",
      title: "Go deeper (references)",
      level: "deep",
      body: [
        { type: "p", text: "MLOps moves fast and is best learned from practitioners. These are the highest-signal resources, in recommended order:" },
        { type: "link", url: "https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/", text: "Chip Huyen — Designing Machine Learning Systems (the canonical modern text; read this first for the whole lifecycle)" },
        { type: "link", url: "https://madewithml.com/", text: "Made With ML (Goku Mohandas) — a free, hands-on, code-first MLOps course from experiment to production" },
        { type: "link", url: "https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning", text: "Google — MLOps: Continuous delivery and automation pipelines in ML (the source of the maturity-level framework)" },
        { type: "link", url: "https://ml-ops.org/", text: "MLOps.org — principles, the MLOps stack canvas, and a broad tooling map" },
        { type: "link", url: "https://mlflow.org/docs/latest/index.html", text: "MLflow docs — tracking, model registry, and deployment (the tool you'll use most)" },
        { type: "link", url: "https://docs.wandb.ai/", text: "Weights & Biases docs — experiment tracking, sweeps, and reports" },
        { type: "link", url: "https://www.evidentlyai.com/", text: "Evidently AI — open-source data/ML drift & quality monitoring (docs + a great drift-metrics blog)" },
        { type: "link", url: "https://dvc.org/doc", text: "DVC docs — Git-style versioning for data, models, and pipelines" },
        { type: "link", url: "https://arxiv.org/abs/1810.03993", text: "Mitchell et al. — Model Cards for Model Reporting (the original paper)" },
      ]
    },
  ],

  packages: [
    { name: "mlflow", why: "experiment tracking + model registry + deployment — the end-to-end open-source standard" },
    { name: "wandb", why: "Weights & Biases: developer-favorite tracking with a superb live-dashboard UI" },
    { name: "dvc", why: "Git-style version control for large datasets and models (hash pointers in Git, bytes in S3)" },
    { name: "evidently", why: "data drift, target drift, and ML quality monitoring with ready-made reports" },
    { name: "pandera", why: "declarative dataframe validation — schema/range/null gates that fail loudly" },
    { name: "great-expectations", why: "richer data validation & documentation for pipelines" },
    { name: "prefect", why: "modern Pythonic pipeline orchestration (Airflow/Dagster/Kubeflow are the alternatives)" },
    { name: "feast", why: "open-source feature store — one feature definition served to training and serving" },
  ],

  gotchas: [
    "Operational monitoring (latency/uptime) cannot detect model decay. A perfectly green dashboard can hide a model that's been rotting for weeks — you must monitor **inputs and predictions**, not just the server.",
    "**Training-serving skew** is the #1 reason a great offline score fails in production: the model is fed subtly different feature values than it trained on. Use one shared feature definition and log live features to diff them.",
    "Drift $\\neq$ decay. Inputs can drift with no accuracy loss, and accuracy can crater with minimal input drift (concept drift). Treat drift as a smoke alarm; confirm with the real metric when labels allow.",
    "**Ground-truth lag** means quality metrics are always late (labels mature in days/months). That delay is exactly why you monitor input drift as an early proxy.",
    "Never gate a deployment on a single aggregate metric — a candidate can raise overall AUC while dropping badly on a key slice or a protected group. Gate on **per-slice** metrics and fairness.",
    "Serve models by **registry stage** (`models:/name/Production`), not a hard-coded version — rollback becomes one API call instead of a redeploy.",
    "The KS test is too sensitive at scale: with millions of rows its p-value flags trivial drift. Prefer an effect-size measure with a fixed threshold (PSI, Wasserstein) on big batches.",
    "Reproducibility needs code **and** data **and** environment pinned together: a Git SHA alone won't reproduce a model if the data, seed, or library versions moved.",
  ],

  flashcards: [
    { q: "What three artifacts must an ML system version, and which are new vs. plain DevOps?", a: "Code (Git handles it, as always), **data**, and the **model** — the latter two are the new problem, since the model is a function of code × data × params × seed." },
    { q: "Why do ML models decay even when the code is frozen?", a: "The world changes, so the data distribution drifts away from what the model was trained on. A deployed model is a depreciating asset — hence drift monitoring and retraining." },
    { q: "What does an experiment tracker log, in three buckets?", a: "**Params** (inputs: lr, depth, seed), **metrics** (outputs: AUC, loss, latency), and **artifacts** (the model, plots, data hash) — plus lineage tags (git SHA, dataset version)." },
    { q: "What is the model registry for, and how should serving code reference a model?", a: "It promotes a run's model into a versioned object with lifecycle stages (Staging→Production→Archived). Serving loads by **stage** (`models:/name/Production`) so rollback is one transition." },
    { q: "What is training-serving skew?", a: "Any mismatch between how a feature is computed at training vs. serving time. The model aces offline eval but underperforms live because it's fed different inputs. Feature stores fix it with one shared definition." },
    { q: "Name the four kinds of drift.", a: "**Covariate/data** ($P(X)$ shifts), **concept** ($P(Y\\mid X)$ shifts — the dangerous one), **label/prior** ($P(Y)$ shifts), and **upstream/schema** (a data-engineering break)." },
    { q: "Write PSI and give its thresholds.", a: "$\\text{PSI}=\\sum_b (a_b-e_b)\\ln(a_b/e_b)$. $<0.1$ = no shift, $0.1$–$0.2$ = moderate, $>0.2$ = significant drift." },
    { q: "Why monitor input drift instead of just accuracy?", a: "**Ground-truth lag**: labels can take weeks/months to arrive, so accuracy is a lagging indicator. Input drift needs no labels and is the leading early-warning signal." },
    { q: "Contrast shadow, canary, and A/B deployments.", a: "**Shadow**: new model gets live traffic but outputs aren't served (engineering check). **Canary**: ramp a small traffic % up while watching metrics. **A/B**: split traffic to *measure* a business KPI difference with significance." },
    { q: "How do training and inference costs differ, and what's the main lever for each?", a: "Training is bursty & interruptible → use **spot instances + checkpointing**. Inference is 24/7 & latency-bound → optimize **throughput per dollar** (batching, right-sizing, quantization/distillation)." },
    { q: "What does LLMOps add over classic MLOps?", a: "Versioning prompts (not just weights), eval harnesses (LLM-as-judge + programmatic checks), guardrails, per-token cost monitoring + caching, and tracing multi-step chains. Mostly classic MLOps with new nouns." },
    { q: "Why is fairness a choice, not a checkbox?", a: "Common fairness metrics (demographic parity, equalized odds, calibration) are mathematically incompatible when base rates differ — you must pick which harm to prevent and document it in the model card." },
  ],

  cheatsheet: [
    { label: "Track a run (MLflow)", code: "with mlflow.start_run(): mlflow.log_params(p); mlflow.log_metric('auc', a)" },
    { label: "Log a model", code: "mlflow.sklearn.log_model(model, 'model')" },
    { label: "Register from a run", code: "mlflow.register_model('runs:/<id>/model', 'name')" },
    { label: "Promote to prod", code: "client.transition_model_version_stage('name', v, 'Production')" },
    { label: "Serve by stage", code: "mlflow.pyfunc.load_model('models:/name/Production')" },
    { label: "Version data (DVC)", code: "dvc add data/train.parquet && git add data/train.parquet.dvc && dvc push" },
    { label: "Reproduce a model", code: "git checkout <sha> && dvc pull" },
    { label: "Validate a batch", code: "schema.validate(df, lazy=True)  # pandera" },
    { label: "PSI drift score", code: "sum((a-e)*np.log(a/e))  # per binned feature" },
    { label: "Drift report", code: "Report(metrics=[DataDriftPreset()]).run(reference, current)  # evidently" },
    { label: "Orchestrate", code: "@flow / @task  (Prefect) | DAG (Airflow) | Pipeline (Kubeflow)" },
    { label: "Checkpoint for spot", code: "torch.save({'model': m.state_dict(), 'opt': o.state_dict()}, ckpt)" },
    { label: "Deterministic run", code: "torch.use_deterministic_algorithms(True); set all seeds" },
    { label: "LLM eval (reproducible)", code: "temperature=0, pinned model snapshot, seed" },
  ],
});
