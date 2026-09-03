# Unit VI — Lecture 19: Classification Model Evaluation: ROC and AUC

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The last of the Unit VI lecture packages, and the one that decides which of the
others you use. Budgeted at a minimum of 1 contact period.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Classification model evaluation and selection | 45 |
| The confusion matrix and every metric that comes out of it | 45 |
| Scores against labels: the threshold you never chose | 45 |
| ROC curves — built by hand from ten scored examples, then by library | 45 |
| AUC curves — as an area, and as a probability | 45 |
| ROC against precision–recall under class imbalance | 45 |
| Choosing a threshold from Youden's *J* or a stated cost ratio | 45 |
| Multiclass evaluation: micro, macro and weighted; one-vs-rest ROC | 45 |
| Comparing classifiers with cross-validated AUC and error bars | 45 |

Course outcome: **CO3** — analyse the performance of machine learning models
and interpret the results.

References: HKP §8.5 (all of §8.5.1–§8.5.6) and §8.6.5; Müller and Guido §5.3;
ISLP §4.4.2; Murphy §5.1.3; ESL §9.2.5 and §7.10; Bishop §1.5.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 53 frames, 88 overlay pages — for the room |
| `notes.pdf` | 34 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | eleven vector figures, every one produced by running code rather than drawn |

## The one idea

**A classifier does not output a class. It outputs a score, and a threshold
turns that score into a class.** Accuracy, precision, recall and F₁ all
describe *one* threshold — usually 0.5, a number nobody in the room chose. The
ROC curve describes *all* of them at once, and the area under it, the AUC, has
a startlingly concrete meaning: the probability that the model scores a
randomly chosen positive above a randomly chosen negative.

Unit VI gave you six ways to build a classifier and no defensible way to say
which one to use. That is what this lecture is for, and it is the last one in
the unit for exactly that reason.

The centre of the lecture is the ROC staircase built by hand from ten scored
rows — sort by score, step *up* for a positive, *right* for a negative — and
then reproduced line for line by `sklearn.metrics.roc_curve`.

## The measurements the lecture turns on

Every number below was produced by running code on scikit-learn's bundled
`load_breast_cancer` (569 patients), `load_wine`, `load_digits` or fixed-seed
synthetic data. Nothing downloads anything.

- On a table with **1% positives**, a model that always answers "negative"
  scores **0.9900** accuracy and a useful logistic regression scores
  **0.9897**. **Accuracy ranks the two backwards**, while ROC AUC ranks them
  0.5000 against **0.9200**.
- Twenty patients, by hand: TP = 6, FP = 2, FN = 4, TN = 8 gives accuracy
  **0.7000**, precision **0.7500**, recall **0.6000**, specificity **0.8000**,
  FPR **0.2000**, F₁ **0.6667** — and `classification_report` reproduces all
  six. Note that `confusion_matrix` is the **transpose** of the textbook
  layout: top-left is TN, bottom-right is TP.
- One fitted model, one set of scores, seven thresholds: accuracy ranges from
  **0.7135 to 0.9181** and F₁ from **0.7030 to 0.9381**, while **ROC AUC stays
  at 0.9661 throughout**. Quoting a single accuracy quotes a model *and* an
  unstated threshold as though they were one thing.
- The hand-built ROC on ten scored rows (five positive, five negative) gives
  eleven (FPR, TPR) points, and
  `roc_curve(y, s, drop_intermediate=False)` reproduces every one of them
  exactly. The default `drop_intermediate=True` keeps **8 of the 11 points**,
  the same curve and the same area.
- AUC two ways, same answer. **Trapezoids:** 0.08 + 0.16 + 0.16 + 0.20 + 0.20 =
  **0.80**. **Concordant pairs:** **20 of 25 = 0.80**. It is also the
  Mann–Whitney *U* statistic rescaled: U = 20.0, U/(5×5) = **0.8000**.
- Sampling pairs on `breast_cancer` converges on the analytic AUC: 100 pairs
  give 0.980000, ten million give 0.966230, and counting all **6848** pairs
  exactly reproduces `roc_auc_score` to a difference of **0.00e+00**.
- Two thousand coin-flip labels against two thousand random scores give AUC
  **0.5023**; over 200 repetitions, mean **0.4994**, sd **0.0136**, range
  **0.4645 to 0.5342**. On 2000 rows, an AUC of 0.53 is not evidence of
  anything.
- AUC below 0.5 is an *inverted* classifier, not a bad one: **0.9661 becomes
  0.0339** when you negate the score or swap the labels, and
  AUC(−s) = 1 − AUC(s) exactly.
- **ROC AUC is invariant to class balance; average precision is not.** Over a
  fiftyfold change in prevalence (0.0100 → 0.5000) on one fixed set of scores,
  ROC AUC stayed inside a band of **0.0085** while average precision moved
  **0.1702 → 0.9202**, a factor of 5.4.
- **Under heavy imbalance ROC flatters.** At 1% prevalence, ROC AUC **0.9200**
  against average precision **0.1702**; at 90% recall that is **1156 false
  alarms for 54 true ones**, precision 0.0446 — **one alert in twenty-two is
  real**. The PR baseline is the prevalence, not 0.5.
- The threshold is a business decision. On the ten hand rows, 1:1 costs pick
  threshold **0.60** (recall 0.80) and 10:1 costs pick **0.35** (recall 1.00).
  On `breast_cancer` with malignant positive, going from 1:1 to 10:1 moved the
  threshold **0.6855 → 0.0670**, misses **13 → 0** and false alarms
  **1 → 42**. Nothing about the model changed; only the price list.
- Youden's *J* picks 0.60 on the ten hand rows and **0.3612** on
  `breast_cancer` (J = 0.8160), buying recall 0.8594 → **0.9375** for precision
  0.8871 → 0.8219. **Youden is the equal-cost answer in disguise.**
- Multiclass, by hand on 100 rows in three classes: per-class F₁ of 0.870968,
  0.736842 and 0.421053 give micro **0.790000** (= accuracy, always), macro
  **0.676287** and weighted **0.785739**. **Weighted recall always equals
  accuracy.**
- On a 80/15/5 problem the class the model **never predicts** — recall
  **0.0000**, F₁ **0.0000** — has a one-vs-rest **AUC of 0.7793**. AUC sees the
  ranking; F₁ sees only the decision. Macro F₁ 0.5334 against weighted 0.8328,
  a gap of **0.2994**.
- Seven classifiers, 20 folds of repeated stratified CV on `breast_cancer`:
  best **0.9950** (SVM RBF) against second **0.9948** (logistic regression), a
  gap of **0.0003** with a fold-to-fold spread of **0.0058** and a paired
  *t*-test *p* = **0.5569**. The nominal winner **lost 13 of the 20 folds**,
  and six of the seven models won at least one.
- Score the same seven models by accuracy instead of AUC and **six of seven
  change rank**. There is no metric-free notion of "the best model".

## Before the assessment session

- Read `notes.pdf`, especially the hand-built ROC sweep, the two AUC
  derivations and the argument about where a threshold may be chosen.
- Attempt practice questions **1, 2, 3, 5 and 7 with a pen**. Question 3 is an
  eight-row ROC sweep with the AUC computed both ways; question 5 costs those
  same eight rows at 8:1; question 7 is the multiclass averaging.
- The assignment that follows this lecture is a **classifier comparison with
  ROC and AUC**. The nine-line checklist near the end of the deck is its
  marking scheme in disguise: state the positive class and the prevalence,
  choose the metric before you run anything, keep every preprocessing step in
  a `Pipeline`, use repeated stratified cross-validation, report mean *and*
  spread *and* fold count, include a `DummyClassifier` baseline, plot the
  curves together, pick the threshold from a stated cost or capacity
  constraint on validation data, and say plainly when two models are
  indistinguishable.
- One habit to start now: never quote an accuracy without saying **what the
  baseline scores** and **what threshold you used**. If you cannot answer both,
  you have not evaluated anything.

## About the figures

Every figure here is computed, not drawn. The generating script lives with the
course's internal tooling and prints each measured number that the notes and
the slides quote, so the prose and the pictures cannot drift apart. If a number
looks wrong, run the code beside it — that is the check that matters.

## Build

```bash
cd latex
for i in 1 2 3; do pdflatex -interaction=nonstopmode slides.tex; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode notes.tex;  done

grep -c '^!'                slides.log notes.log   # must be 0
grep -c 'Overfull \\vbox'   slides.log notes.log   # must be 0
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
