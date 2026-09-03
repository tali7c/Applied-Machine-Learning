# Unit IV — Lecture 6: Feature Engineering, Encoding and Dimensionality Reduction

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The second of the three Unit IV lecture packages. Budgeted at a minimum of 2
contact periods; the expected schedule allocates 3.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Feature engineering: combining, decomposing, binning, transforming | 16 |
| Feature scaling: standard, min–max, robust, max-abs; who needs it and who does not | 16 |
| Feature encoding: nominal and ordinal, one-hot, the dummy variable trap, high cardinality | 17 |
| Data reduction concepts: dimensionality, numerosity, compression | 18 |
| Dimensionality reduction: principal component analysis | 18 |
| Feature selection: filter, wrapper, embedded | 18 |

Course outcome: **CO2** — apply appropriate machine learning techniques to
solve a given problem.

References: Müller and Guido §3.3–3.4 and Ch. 4; HKP §3.4; ISLP §6.3 and
Ch. 12.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 58 frames, 103 overlay pages — for the room |
| `notes.pdf` | 31 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | ten vector figures, every one produced by running code rather than drawn |

## The one idea

**The columns are a choice, not a given.** Lecture 5 cleaned the table; this
lecture decides what is in it. Every step here learns something from the data
— a mean, a list of categories, a rotation, a subset — and every one of those
things must be learned from the training rows only. The four decisions are:
what should the columns be, what units should they be in, what do I do with a
column of words, and what do I do when there are too many columns.

## The measurements the lecture turns on

Every number below was produced by running code on scikit-learn's bundled
`load_breast_cancer` (569 patients), `load_digits` (1797 images),
`load_diabetes` (442 patients) or fixed-seed synthetic data. Nothing
downloads anything.

One seed runs the whole lecture: `np.random.default_rng(0)` / `random_state=0`,
including on `mutual_info_classif` (which is stochastic) and on the
`StratifiedKFold` shuffle. The two exceptions are sweeps — the 40-split PCA
experiment and the 20-repetition selection experiment vary their seed on
purpose, because the variation between repetitions is the measurement — and
their seeds are derived from the same 0. Run the code as written and every
number below comes back identical.

- One engineered product column took a linear model from cross-validated
  **R² = 0.9091 to 0.9981**. A random forest reached the same place unaided:
  **feature engineering is worth most to the simplest models.**
- Binning a wiggly feature into ten intervals bought a linear model
  **+0.0957** and cost a decision tree **−0.0323** — and afterwards both
  fitted exactly the same step function, to four decimal places.
- `PolynomialFeatures` degree 3 turns 30 columns into **5455**. On diabetes,
  degree 2 cost **−0.0643** of R² even with ridge.
- Age (25–60 years) against income (in rupees): the income column supplied
  **99.995%** of the squared distance between two customers, and standardising
  **changed which customer was the nearest neighbour**.
- Scaling was worth **+0.0561** to an RBF SVM, **+0.0334** to k-NN and
  **+0.0246** to logistic regression — and **exactly 0.0000** to a decision
  tree and a random forest. Same tree, node for node, same predictions.
- Unscaled, stochastic gradient descent **diverged at every usable step size**
  (R² = −1.7×10²⁸) and stalled at 0.35 at the small ones. Standardised, it
  reached **0.5144 in fourteen epochs** against a least-squares 0.5177.
  Condition number of X'X: **1.03×10⁶ raw against 470 standardised**.
- One mistyped value squeezed the good data **78× under `StandardScaler`,
  71× under `MinMaxScaler` and 2× under `RobustScaler`**.
- Ordinal-coding a genuinely nominal three-level column gave
  **R² = 0.0743 against 0.9868** for one-hot — and the hand calculation on
  three points predicted 0.0577. A tree on the same ordinal column got 1.0000.
- An intercept plus all k dummies is rank-deficient: condition number
  **3.4×10¹⁵**, and two completely different coefficient vectors give
  identical predictions.
- One-hot encoding a 256-level identifier that carried **no information**
  moved the training R² up 0.14 and the test R² **down 0.12**.
- In 1000 dimensions the farthest of 499 points is only **1.11×** as far as
  the nearest. Sampling half the digits rows cost 0.0087 of accuracy;
  sampling a twentieth cost 0.1188.
- PCA on **unscaled** breast_cancer: PC1 explained **98.20%** and was simply
  the widest column. Standardised: **44.27%**, and 95% needed **10 of 30**
  components — but **40 of 64** on digits.
- **PCA is not feature selection**: 26 of the 30 PC1 loadings exceed 0.10, so
  all thirty measurements are still needed for a new patient. And it is
  unsupervised — on a designed example it took an accuracy of **0.9850 down
  to 0.5200** by discarding the only informative direction.
- Refitting PCA on the test set cost **−0.0320** on average and **−0.0877**
  on the worst of forty splits; fitting it on all the rows
  before splitting cost **nothing measurable** (t = 0.94 over 40 splits).
- **Selecting features with the labels before cross-validating scored 0.8475
  on 500 pure-noise columns with coin-flip labels**, against **0.5342** inside
  a `Pipeline`. An overstatement of **+0.3133**, on 20 repetitions out of 20.
- None of the four selectors beat using all thirty breast_cancer features
  (0.9789), and `RFECV` chose to keep **26 of 30**. That result is reported
  rather than hidden. RFE's extra cost is quoted as a count, not a time:
  **21 model fits per fold against a filter's 1**.

## Before the next lecture

- Read `notes.pdf`, especially the by-hand eigen-decomposition of the five
  points and the proof that reconstruction error equals the sum of the
  discarded eigenvalues.
- Attempt practice questions 3, 6 and 9. Question 6 is PCA on a 2×2
  covariance matrix by hand; question 9 is a debugging story you will meet
  for real.
- Start one habit now: every preprocessing step goes inside the `Pipeline`.
  If you find yourself calling `fit_transform` on `X` before splitting, stop
  and ask what that step just learned.
- Next up is **data splitting, cross-validation and imbalanced data**. The
  last section of this lecture is the first section of that one.

## About the figures

Every figure here is computed, not drawn. The generating script lives with the
course's internal tooling and prints each measured number that the notes quote,
so the prose and the pictures cannot drift apart. If a number in the notes looks
wrong, run the code beside it — that is the check that matters.

## Build

```bash
cd latex
for i in 1 2 3; do pdflatex -interaction=nonstopmode slides.tex; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode notes.tex;  done

grep -c '^!'                slides.log notes.log   # must be 0
grep -c 'Overfull \\vbox'   slides.log notes.log   # must be 0
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
