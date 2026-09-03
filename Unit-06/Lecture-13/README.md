# Unit VI — Lecture 13: Classification Foundations and K-Nearest Neighbour

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The first of the Unit VI lecture packages, and the start of classification.
Budgeted at a minimum of 2 contact periods.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Introduction to machine learning classifiers | 33 |
| The general approach to classification (two steps) | 33 |
| Classification algorithms: the families, and eager against lazy | 33 |
| Instance-based learning, measured | 33–34 |
| K-Nearest Neighbour: the algorithm, choosing *k*, scaling, distance metrics, weighted voting | 34 |
| The curse of dimensionality, ties, imbalance and the failure list | 34 |

Course outcomes: **CO1** — explain the concepts and terminology of machine
learning; **CO4** — apply and evaluate classification techniques.

References: HKP §8.1 and §9.5.1; Müller and Guido Ch. 2, §2.3.1; ISLP §2.2.3
and §4.7.6; Bishop §2.5.2 and §1.4; Mitchell Ch. 8.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 55 frames, 86 overlay pages — for the room |
| `notes.pdf` | 35 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | fourteen vector figures, every one produced by running code rather than drawn |

## The one idea

**K-nearest neighbour has no training step at all.** It stores the training
table and, when asked about a new record, looks up the *k* rows most like it
and takes a vote. There is no formula, no coefficient, no loss function. That
makes it the ideal first classifier: everything it does can be checked with a
pen, and everything that goes wrong with it goes wrong for a reason you can
write down in one line of arithmetic.

The regression unit predicted a number. This unit predicts a label. You have
already met one classifier — logistic regression fitted a probability and
compared it with a threshold — so the lecture first steps back and asks what
all classifiers have in common, then goes forward to the simplest one there
is.

## The measurements the lecture turns on

Every number below was produced by running code on scikit-learn's bundled
`load_iris` (150 flowers), `load_wine` (178 wines), `load_breast_cancer` (569
patients), `load_digits` (1797 images) or synthetic data fixed by
`make_classification(..., random_state=0)`. Nothing downloads anything.

**One seed.** Every arbitrary random choice — splits, shuffles, noise draws,
the cross-validation folds — uses `0`: `np.random.default_rng(0)` and
`random_state=0`. Run the code as written and you get these numbers, not
similar ones. **Timings are the exception and no seed can fix them**, so every
cost claim below is stated as an exact operation count too.

- One 70/30 split of iris at *k* = 3 scored **1.0000** — 45 out of 45, which
  is a small sample rather than a result. Cross-validated over all 150 rows
  the score is **0.9600**, and the confusion matrix says *which* mistakes were
  made: **all six are between versicolor and virginica**, three each way,
  precision and recall 0.940 for both.
- Six classifiers on breast_cancer: logistic regression **0.9789**, SVC(rbf)
  **0.9771**, *k*-NN **0.9649**, naïve Bayes **0.9297**, tree **0.9262** — and
  a majority-class `DummyClassifier` at **0.6274**. **The spread between the
  five real classifiers is smaller than the gap to the baseline.**
- On digits, one *k*-NN prediction pass costs **540 × 1257 × 64 =
  43,441,920** coordinate operations against logistic regression's **345,600**
  — **125.7×**, exact on any machine — while *k*-NN's `fit` performs **no**
  distance computations at all. With a clock on it the predict/fit ratio is
  above 1 for *k*-NN and **0.0001** for logistic regression, about 10⁵ apart.
- Forty times more training data is **exactly forty times the distance
  computations**: 10,000,000 against 250,000. Logistic regression's prediction
  cost does not move at all, because its prediction never touches the training
  rows.
- A fitted *k*-NN **is** the data: **643,584 bytes** of training rows against
  **5,200 bytes** of logistic-regression coefficients, a ratio of **123.8×**
  that grows with *n*. That is a privacy fact, not a benchmark curiosity.
- A KD-tree stops helping in high dimensions, and the pruning counts say why
  without a clock: at 2 features a query opens **1.5 leaves** and 6249
  subtrees are discarded unlooked-at; at 40 features only **127** subtrees are
  pruned in a thousand queries and each query opens **63.8 leaves** — the
  whole tree. Leave `algorithm='auto'` alone.
- Worked by hand on seven points: 1-NN put *q*₁ = (3,3) in class **0** via
  neighbour *B* at *d* = √2, and `KNeighborsClassifier` returned
  **1.414214** and class 0 — agreement to six decimal places.
- On *q*₂ = (6,3) the answer changes twice: **k=1 → 0, k=3 → 1, k=5 → 1,
  k=7 → 0**. At *k* = *n* the classifier **is** the majority-class baseline.
- **k = 1 scores exactly 1.000000 on its own training set**, on every dataset,
  because every training row is its own nearest neighbour at distance 0.
  Training accuracy is worthless evidence for this model.
- The validation curve on standardised breast_cancer peaks at **k = 11 with
  0.9666** and falls to **0.6274** — exactly the majority-class baseline — at
  **k = 398**. The peak is shallow: k = 3 to k = 15 are indistinguishable.
- Between two breast_cancer patients, **mean area alone supplied 90.4482%** of
  the squared distance and the **27 non-area columns shared 0.7655%**.
  Unscaled *k*-NN there is a classifier that uses three columns.
- Standardising changed the nearest neighbour of **164 of 171** held-out
  patients and the **predicted class of 19** of them.
- Scaling gains at *k* = 5: **wine +0.2975**, breast_cancer +0.0334,
  iris +0.0000, and **digits −0.0089** — scaling made it *worse*. The size of
  the scale disparity does not predict the size of the gain.
- Minkowski: for *q* = (0,0), *P* = (3,0), *Q* = (2,2), the nearest neighbour
  is *P* under L₁ and *Q* under L₂, with the crossover at
  **p\* = 1/(log₂3 − 1) = 1.709511**. On real standardised data, though,
  p = 1 and p = 2 differ by under 0.002 everywhere.
- Distance weighting turned a 4:1 uniform vote into **1.2:1.0**. Measured, it
  is worth between **−0.0057 and +0.0035** — and it makes **every** *k* score
  1.0000 on the training set.
- The curse: with 500 uniform points the farthest is **1156× the nearest at
  d = 1** and **1.11× at d = 1000**. A box holding 1% of the data in 100
  dimensions must span **95.5% of every feature's range**.
- Bolting pure-noise columns onto iris took *k*-NN from **0.9533 to 0.5200**,
  through a low of **0.4667**; a decision tree went only from 0.9333 to
  0.9067. **k-NN has no feature selection built in.**
- On a 94:6 problem, every *k* ≥ 15 predicted the minority class **zero
  times** on 600 held-out rows while scoring **0.9417** — exactly the
  baseline. Never report accuracy alone.

## Before the next lecture

- Read `notes.pdf`, especially the two by-hand distance tables and the
  arithmetic behind the scaling result.
- Attempt practice questions 1, 2 and 5 **with a pen**. Question 1 is a
  distance table and three values of *k*; question 2 is the distance-weighted
  version of it; question 5 asks you to *derive* a Minkowski crossover rather
  than guess it.
- Start one habit now: every *k*-NN goes inside a `Pipeline` with a scaler,
  and every score you quote is cross-validated. If you catch yourself
  reporting `score(Xtr, ytr)`, stop.
- Next up is **decision trees and attribute selection**. Carry one contrast
  across: *k*-NN weights every column equally; a tree chooses. That single
  sentence explains the last figure of this lecture.

## About the figures

Every figure here is computed, not drawn. The generating script lives with the
course's internal tooling and prints each measured number that the notes quote,
so the prose and the pictures cannot drift apart. If a number in the notes looks
wrong, run the code beside it — that is the check that matters. Timings are the
one exception: they depend on your processor and no seed fixes them, which is
why every cost claim is also given as an exact operation count. Quote the
count, never the millisecond.

## Build

```bash
cd latex
for i in 1 2 3; do pdflatex -interaction=nonstopmode slides.tex; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode notes.tex;  done

grep -c '^!'                slides.log notes.log   # must be 0
grep -c 'Overfull \\vbox'   slides.log notes.log   # must be 0
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
