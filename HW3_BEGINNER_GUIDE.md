# HW3 Beginner Guide: Which Trucks Go to the Workshop?

This guide explains the homework in plain language. You do **not** need prior machine learning experience to follow it. Read it once before coding, then come back to sections as you work.

---

## The big picture (in one paragraph)

A truck company has sensor data from many trucks. A few trucks have a rare but expensive failure in the **air pressure system (APS)**. Checking a truck costs money. Missing a failing truck costs *a lot* more money. Your job is to build a prediction system that decides which trucks to send to the workshop, pick a good decision rule (threshold), prove your choice with proper evaluation, and write a short memo for the maintenance manager.

**The core question:** Which model and decision threshold minimize expected cost — and is the “better-looking” model really better?

---

## The story (why this homework exists)

Imagine this workplace argument:

- A **data scientist** says: “Our boosted model is clearly better than logistic regression — look at the cross-validation scores!”
- A **fleet maintenance manager** says: “I don’t care about fancy scores. Tell me which trucks to call in, and how much money we save.”

You are the person who must:

1. Translate business costs into a fair evaluation method.
2. Compare models carefully (not just “which score is higher”).
3. Fix a common mistake (using the default 0.5 threshold when costs are unequal).
4. Give a clear decision: deploy a model, or keep the current “do nothing / check everything” style policy.

---

## What data do you have?

| Thing | Meaning |
|---|---|
| **~76,000 trucks** | Each row is one truck |
| **170 sensor features** | Numbers from sensors (inputs / `X`) |
| **~8% missing values** | Some sensor readings are blank |
| **~1,375 failures** | Only under 2% of trucks actually fail |
| **Label `y`** | `1` = APS failure, `0` = no failure |

This is a **classification** problem: predict whether a truck will fail (`pos`) or not (`neg`).

Costs used in the homework (from an industrial challenge):

- **False positive (unnecessary check):** cost = **10**
- **False negative (missed failure):** cost = **500**

So missing a failure is **50×** more expensive than checking a healthy truck.

---

## What you must submit

One zip named `hw3_<studentID>.zip` with exactly these files:

| File | What it is |
|---|---|
| `hw3.ipynb` | Your filled notebook (starter renamed), with outputs + an **AI-use statement** |
| `hw3_submission.py` | Your final pipeline + threshold (`THRESHOLD` and `build_pipeline()`) |
| `answers.json` | Numbers and the memo (filled by the last notebook cell) |
| `mlflow_runs.csv` | Exported MLflow experiment runs that support your claims |

**Deadline:** Week 9, 26 Nov 2026, 23:59 Turkey time · **Weight:** 5% · **Effort:** about 6–8 hours

---

# Step-by-step: what you are supposed to do

The starter notebook is already organized into sections. Follow them in order.

---

## Step 0 — Set up and understand the starter

**What to do**

1. Open `hw3_starter.ipynb` (later rename it to `hw3.ipynb` for submission).
2. Run the setup cells that:
   - load libraries
   - set a random seed (`SEED = 0`)
   - set costs `C_FP = 10`, `C_FN = 500`
   - start an **MLflow** experiment named `hw3-truck-checks`
   - download the APS truck dataset
3. Look at the helper function `cost_per_truck(y_true, flagged)`. This is how money is calculated.

**Why it matters**

Everything later (model choice, threshold, memo) must be judged by this cost, not by “accuracy looks high.”

**Simple intuition**

```
cost_per_truck =
  (10 × number of healthy trucks you wrongly checked
 + 500 × number of failing trucks you missed)
  ÷ total trucks
```

---

## Step 1 — Protocol and metric

**Notebook section:** `## 1. Protocol and metric`

### What “protocol” means here

A **protocol** is your evaluation plan: how you split data, how you tune, how you avoid cheating, and which score you use to pick models.

### What you should figure out

1. **Cost of “do nothing”**  
   Never send any truck to the workshop.  
   You miss every failure → you pay mostly the 500-cost.  
   This is a **baseline** (a simple policy to beat).

2. **Cost of “check everything”**  
   Send every truck for a check.  
   You never miss a failure, but you pay 10 for every truck.  
   This is another baseline.

3. **Choose a selection metric**  
   You need a number to rank models while tuning. Allowed options in `answers.json` include:
   - `accuracy`
   - `roc_auc`
   - `average_precision`
   - `f1`
   - `expected_cost`

   Because failures are rare and costs are unequal, **accuracy is usually a bad choice**.  
   Metrics that care about ranking positives well (like `average_precision` / `roc_auc`) or that directly use cost (`expected_cost`) are more sensible.

### Why accuracy can trick you

If 98% of trucks are healthy, a dumb model that always says “no failure” gets ~98% accuracy — and still misses every expensive failure. High accuracy can hide a terrible business decision.

### What “leakage-free” means

**Data leakage** = accidentally using information you would not have in real life when training or choosing a model.

Examples of leakage (bad):

- Filling missing values using the whole dataset *including the test set*
- Tuning the threshold on the final test set
- Looking at test labels while choosing hyperparameters

Leakage-free means: test data is touched only at the very end for a final report.

---

## Step 2 — Compare models with nested cross-validation

**Notebook section:** `## 2. Candidates under nested cross-validation`

### What models to compare

At least:

1. **Logistic regression** (simple, linear, classic baseline model)
2. **One tree ensemble** (for example HistGradientBoosting, Random Forest, LightGBM, or XGBoost)

The data scientist claims the boosted/tree model is “clearly better.” Your job is to check that claim properly.

### What a pipeline is

A **pipeline** is a chain of steps that always run together, for example:

1. Impute missing values (fill blanks)
2. Scale features (optional, often useful for logistic regression)
3. Train the classifier

You put preprocessing *inside* the pipeline so every training fold gets its own clean fit. That prevents leakage.

Your final code must live in `hw3_submission.py`:

- `build_pipeline()` → returns an **unfitted** model/pipeline that supports `predict_proba`
- `THRESHOLD` → the probability cutoff used to flag trucks

### Why class imbalance matters

**Class imbalance** = one class is much rarer than the other.  
Here, failures are under 2%. Models can “cheat” by almost always predicting “healthy.”

Common ways people handle this (conceptually):

- Use better metrics (not accuracy)
- Adjust class weights
- Tune the decision threshold carefully
- Use methods that can focus on rare positives

### Nested cross-validation (the important evaluation idea)

Think of two loops:

**Outer loop (honest test folds)**

- Split data into several folds (pieces).
- Each time, hold out one fold as a temporary “test.”
- Train/select on the remaining folds.
- Score the held-out fold.
- Log **one MLflow run per outer fold**.

**Inner loop (tuning)**

- Inside the outer training data only, try different hyperparameters.
- Pick the best setting using your selection metric.
- Never use the outer test fold for tuning.

Why nested CV?

- Normal CV can be optimistic if you tune and evaluate on the same splits.
- Nested CV gives a more honest estimate of how good the whole “tune + train” process is.

If nested CV is slow, the README allows **subsampling** (using fewer rows) so the notebook finishes in under 20 minutes.

### What to log in MLflow

For each outer fold of each model family, log things like:

- fold score(s)
- cost
- maybe hyperparameters
- enough info that later you can pull paired fold scores back

Every number you report later should be traceable to an MLflow **run ID**.

---

## Step 3 — Test the claim with statistics

**Notebook section:** `## 3. Statistical comparison from the MLflow runs`

Someone said “boosted model is clearly better.” You must check that scientifically.

### What to do

1. Pull the **per-fold scores** for both models from MLflow.
2. Make sure folds are **paired** (same outer fold for model A and model B).
3. Run:
   - a normal **paired t-test**
   - the **corrected resampled t-test** (taught in Week 8)
4. Do this for:
   - your **ranking/selection metric**, and
   - the **cost**

### What a p-value means (simple)

A **p-value** answers roughly:

> “If the two models were actually equally good, how surprising would this difference be?”

- Small p-value (commonly < 0.05): evidence of a real difference.
- Large p-value: not enough evidence to claim one is better → report `"no_evidence"`.

### Why the “corrected” test exists

Cross-validation fold scores are **not fully independent** (they share training data).  
A plain t-test can be too optimistic (too many false “significant” wins).  
The corrected resampled t-test adjusts for that overlap.

### What you should conclude

Possible honest conclusions:

- Model B is better on the metric, but not better on cost.
- Model B looks better, but p-value is not significant → `"no_evidence"`.
- Model A and B differ on ranking quality but have similar workshop cost.

The homework wants your conclusion to match the evidence, not the hype.

---

## Step 4 — Fix the “obstacle” (the bad default threshold)

**Notebook section:** `## 4. Threshold and final test`

### The obstacle

By default, many classifiers use:

> Predict failure if probability ≥ **0.5**

That default is often **badly wrong** here because:

- Failures are rare
- Missing a failure costs 500, unnecessary checks cost only 10

So you usually want to flag a truck even when the failure probability is much lower than 0.5, because the cost of missing is so high.

### What you should do

1. Measure how bad the default rule is (cost at threshold 0.5).
2. Find a better threshold **without using the final test set**.
3. Typical approach:
   - Keep a validation split (or use CV / calibration on training data only)
   - Try many thresholds
   - Pick the one with lowest expected cost
4. Only then evaluate once on the held-out test set.

### How the threshold is used

Your model outputs a probability for each truck, like `0.03`, `0.40`, `0.81`.

```
if probability >= THRESHOLD:
    send truck to workshop
else:
    do not check
```

That `THRESHOLD` must be saved in `hw3_submission.py`.

---

## Step 5 — Make the final decision for the manager

**Notebook section answers + memo**

You must produce:

| Field | Meaning |
|---|---|
| `chosen_model` | Which model family you pick |
| `threshold` | Your decision cutoff |
| `test_cost_per_truck` | Actual cost on your held-out test set |
| `expected_cost_per_truck` | What you expect on new trucks |
| `cost_never_check` | Baseline “do nothing” cost |
| `memo.decision` | `"deploy_model"` or `"keep_current_policy"` |
| `memo.expected_savings_per_truck` | usually `cost_never_check - expected_cost_per_truck` |
| `memo.main_risk` | Main worry (wrong costs, threshold drift, imbalance, data quality, or none) |
| `memo.justification` | ≤150 words for the manager |

### How to think like the manager

Ask:

- Does my model beat “never check”?
- Does it beat “check everything”?
- Are savings large enough to justify deploying?
- What could go wrong in real life?

If the model does not clearly beat a trivial policy, maybe `"keep_current_policy"` is the honest answer.

---

## Step 6 — Fill submission files and check reproducibility

1. Implement `build_pipeline()` and set `THRESHOLD` in `hw3_submission.py`.
2. Run the notebook top to bottom with fixed seeds.
3. Fill `answers.json` (the last cell writes it).
4. Export `mlflow_runs.csv` from the same cell.
5. Write the **AI-use statement** Markdown cell (required; missing it costs points).
6. Make sure every reported number points to an MLflow run ID.
7. Keep total runtime under ~20 minutes on a laptop.

The autograder will:

- fit your pipeline on its own split
- apply your threshold to unseen trucks
- recompute costs and statistical tests from your MLflow export

---

# Terminology dictionary (simple explanations)

Use this whenever a word feels scary.

### Core machine learning words

| Term | Simple meaning |
|---|---|
| **Machine learning** | Teaching a computer to find patterns in examples, then make predictions on new examples |
| **Supervised learning** | Learning from data that already has correct answers (labels) |
| **Classification** | Predicting a category (fail / not fail) |
| **Feature / input / `X`** | The information used to predict (sensor readings) |
| **Label / target / `y`** | The correct answer (did APS fail?) |
| **Model** | The mathematical rule that turns features into a prediction |
| **Training** | Fitting the model on known examples |
| **Prediction** | The model’s guess for a new truck |
| **Probability (`predict_proba`)** | How confident the model is that a truck will fail (between 0 and 1) |
| **Decision threshold** | Cutoff that turns a probability into “check” or “don’t check” |
| **Hyperparameter** | Settings you choose before training (like tree depth, regularization strength) |
| **Pipeline** | Preprocessing + model glued together as one object |
| **Imputation** | Filling missing values with estimated numbers |
| **Scaling / normalization** | Rescaling features so different sensors are comparable |
| **Overfitting** | Memorizing training data too much; fails on new data |
| **Underfitting** | Model too simple; misses real patterns |
| **Generalization** | Doing well on new, unseen trucks |

### Evaluation words

| Term | Simple meaning |
|---|---|
| **Train / validation / test split** | Separate data for learning, tuning, and final honest check |
| **Cross-validation (CV)** | Rotate which fold is held out, so every piece gets tested |
| **Nested CV** | Outer CV for honest scoring + inner CV for tuning |
| **Fold** | One piece of the data in cross-validation |
| **Metric** | A number that says how good predictions are |
| **Accuracy** | Fraction of predictions that are correct (can be misleading here) |
| **Precision** | Of trucks you flagged, how many really failed |
| **Recall / sensitivity** | Of real failures, how many you caught |
| **F1** | Balance of precision and recall |
| **ROC AUC** | How well the model ranks failures above healthy trucks across thresholds |
| **Average precision (AP / PR AUC)** | Similar ranking idea, often better under heavy imbalance |
| **Expected cost** | Average money lost under your decision rule |
| **Baseline / trivial policy** | Simple strategy like “never check” or “check all” |
| **Held-out set** | Data you did not use while choosing the model |

### Error / cost words

| Term | Simple meaning |
|---|---|
| **True positive (TP)** | Failed truck correctly flagged |
| **False positive (FP)** | Healthy truck wrongly sent to workshop (costs 10) |
| **False negative (FN)** | Failed truck missed (costs 500) |
| **True negative (TN)** | Healthy truck correctly left alone |
| **Class imbalance** | One class is rare (failures << healthy trucks) |
| **Cost-sensitive decision** | Choosing predictions based on money, not just “most likely class” |

### Model words used in this homework

| Term | Simple meaning |
|---|---|
| **Logistic regression** | A simple model that estimates failure probability with a weighted combination of features |
| **Tree** | Asks yes/no questions about features (“is sensor A high?”) to reach a decision |
| **Ensemble** | Many models combined (a “team” of trees) |
| **Random forest** | Many trees trained on random pieces of data/features, then averaged |
| **Gradient boosting / boosted model** | Trees built one after another; each new tree tries to fix previous mistakes |
| **HistGradientBoosting** | A fast boosting method in scikit-learn that also handles missing values well |

### Experiment tracking / stats words

| Term | Simple meaning |
|---|---|
| **MLflow** | A tool that records experiments (parameters, metrics, run IDs) |
| **Run ID** | Unique ID of one logged experiment trial |
| **Paired comparison** | Compare two models on the *same* folds |
| **t-test** | Statistical check whether average difference is likely real |
| **Corrected resampled t-test** | t-test adjusted for overlapping CV folds |
| **p-value** | Evidence strength against “no difference” |
| **Significance level (5%)** | Common cutoff: p < 0.05 means “evidence of a difference” |
| **Reproducibility** | Someone else can re-run your notebook and get the same story |
| **Seed** | Fixed random starting point so results are repeatable |

### Process / integrity words

| Term | Simple meaning |
|---|---|
| **Data leakage** | Accidentally using future/test information during training or tuning |
| **Leakage-free evaluation** | Honest testing that mirrors real deployment |
| **Selection metric** | The score you use to choose models/hyperparameters |
| **Deployment** | Actually using the model in real decisions |
| **Threshold drift** | The best cutoff changing over time as data changes |
| **Permutation check / chance-level** | Sanity test: if labels are shuffled, a good pipeline should become useless |

---

# A mental checklist while you work

Use this like a to-do list:

1. [ ] Understand costs: check = 10, miss = 500  
2. [ ] Compute baselines: never-check and check-everything  
3. [ ] Choose a selection metric that matches the business problem  
4. [ ] Build leakage-free pipelines (imputation inside the pipeline)  
5. [ ] Compare logistic regression vs one tree ensemble with **nested CV**  
6. [ ] Log **one MLflow run per outer fold**  
7. [ ] Pull fold scores and run paired + corrected t-tests (metric and cost)  
8. [ ] Show that threshold 0.5 is bad; tune a better threshold without the test set  
9. [ ] Evaluate once on the test set  
10. [ ] Write manager memo: decision, savings, main risk, short justification  
11. [ ] Fill `hw3_submission.py`, `answers.json`, `mlflow_runs.csv`  
12. [ ] Add AI-use statement and run notebook top-to-bottom under 20 minutes  

---

# How grading roughly works (so you know what matters most)

| What graders care about | Weight |
|---|---|
| Pipeline + decision rule works, low held-out cost, beats trivial policies | **45%** |
| Statistical comparison is paired, reproducible, conclusion matches evidence | **15%** |
| You diagnosed the threshold obstacle and metric choice | **15%** |
| Final decision/savings are consistent with numbers | **10%** |
| Reproducibility (files, schema, MLflow links) | **15%** |

Missing AI-use statement: automatic −5.

---

# Tiny FAQ for absolute beginners

### “Do I need to invent a new algorithm?”
No. Use standard models from libraries (scikit-learn / similar), connect them correctly, evaluate honestly, and make a cost-aware decision.

### “What does ‘better model’ mean here?”
Not “higher accuracy.” It means better for the company’s money after a fair statistical comparison.

### “Why is threshold so important?”
Because the model only gives probabilities. The threshold turns those probabilities into workshop decisions — and that is where the money is made or lost.

### “What if the fancy model is not significantly better?”
That is a valid scientific answer. Report `"no_evidence"` if the tests say so.

### “What is the minimum I must compare?”
At least logistic regression and one tree ensemble, evaluated with nested CV and proper tests.

---

# Suggested learning order (if you feel lost)

1. Read this guide once.  
2. Run the starter notebook’s data-loading cells and print basic stats.  
3. Manually compute “never check” and “check everything” costs.  
4. Fit one simple logistic regression pipeline and compute cost at threshold 0.5.  
5. Learn how to change the threshold and watch cost change.  
6. Then add nested CV + MLflow logging.  
7. Then do the statistical tests.  
8. Finally write the memo and submission files.

---

# One-sentence summary

**You are building a careful, cost-aware truck-checking system: handle messy imbalanced data, compare models fairly, fix the bad default threshold, prove your claims with MLflow-backed statistics, and tell the manager whether deploying the model saves money.**
