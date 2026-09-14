# Unit IV — Lecture 5: Data Cleaning — Missing Data, Outliers and Transformation

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The first of the three Unit IV lecture packages. Budgeted at a minimum of 2
contact periods; the expected schedule allocates 3.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Data preprocessing: the four tasks, and why cleaning comes first | 13 |
| Data cleaning; what "dirty" means; the cost of listwise deletion | 13 |
| Handling missing data: MCAR / MAR / MNAR, mean, median, mode, KNN, MICE | 14 |
| Describing shape: skewness and kurtosis, both conventions, and their ceilings | 14 |
| Handling outliers: z-score, IQR fence, modified z-score; masking | 14 |
| Numerical methods: binning and smoothing, trimming, winsorising, capping, robust scaling | 14 |
| Data transformation: log, square root, the ladder of powers | 15 |
| Box–Cox and Yeo–Johnson: both formulas, and the likelihood that picks λ | 15 |
| The order of operations: cleaning, the train/test split, and `Pipeline` | 15 |

Course outcomes: **CO1** — explain the concepts and steps of a machine
learning workflow; **CO2** — apply appropriate machine learning techniques to
solve a given problem.

References: HKP §3.1–3.2 and §3.5; Müller and Guido Ch. 4.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 62 frames, 101 overlay pages — for the room |
| `notes.pdf` | 40 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | twelve vector figures, every one produced by running code rather than drawn |

## Shape statistics, and the power transforms

Two sections carry the statistical machinery the rest of the lecture leans on.

**Skewness and kurtosis** are defined from the central moments
$m_k = \frac{1}{n}\sum (x_i - \bar{x})^k$, worked by hand on five numbers, and
then pinned down where they usually go wrong:

- Skewness is an *odd* power, so it reports which side the tail is on. Kurtosis
  is an *even* power weighted by the fourth, so it reports how much lives in
  the tails. The `-3` that makes "excess" kurtosis is a convention, and
  `scipy` and most textbooks disagree about it.
- **Kurtosis is not peakedness.** On a 95/5 mixture, 1.5% of the rows supply
  **88%** of the fourth-power sum — measured, not asserted.
- **The thresholds are folklore.** At $n=30$ the standard error of skewness is
  0.45, and **one sample in five** drawn from a perfectly symmetric normal
  reports $|g_1| > 0.5$.
- **Both have ceilings set by $n$**: $|g_1| \le (n-2)/\sqrt{n-1}$ and
  $b_2 \le (n^2-3n+3)/(n-1)$, both attained exactly by one point standing away
  from the rest. At $n=5$ the largest excess kurtosis possible is **0.25** —
  which is why the five-point example returns a negative value next to an
  obvious outlier.

**Box–Cox and Yeo–Johnson** are given in full, including the four-case
Yeo–Johnson formula that is usually only described in words:

- The family is worked on $(1, 4, 16, 64)$, where $\lambda = 0$ turns a
  geometric sequence into an arithmetic one — what "the log is the right ruler
  for multiplicative data" actually means.
- The hand-typed Yeo–Johnson reproduces `PowerTransformer` **to the last bit**.
- λ is chosen by maximum likelihood, and the **Jacobian term** everyone drops
  is what stops it: without it the optimiser runs to the edge of the grid, at a
  transformed variance four orders of magnitude too small.
- `PowerTransformer` standardises by default, and λ is a **learned parameter** —
  fitted on train it is −0.55, on all rows −0.48, on test −0.34. It belongs
  inside the `Pipeline`.

## The one idea

**The model is the easy part.** Every step before the model learns a number
from the data — a mean, a median, a threshold, a λ — and every one of those
numbers can be wrong, can be computed from the wrong rows, or can quietly
destroy information you needed. This lecture is about the four decisions:
what to do about a hole, about an impossible-looking value, about a lopsided
distribution, and **in what order** relative to the train/test split.

## The measurements the lecture turns on

Every number below was produced by running code on scikit-learn's bundled
`load_diabetes` (442 patients) and `load_breast_cancer` (569 patients), or on
fixed-seed synthetic data. Nothing downloads anything.

One seed runs the whole lecture: `np.random.default_rng(0)` / `random_state=0`.
The two exceptions are named for what they are — the 35 sensor readings come
from `default_rng(7)`, which defines that example's data, and the repeated-split
experiments sweep `random_state` over `range(0, 20)` or `range(0, 300)`,
because the spread across splits is what they measure. Run the code as written
and every number below comes back identical.

- Knock out **5% of the individual cells** of a 30-column table and listwise
  deletion leaves **114 patients of 569**. A row survives with probability
  `(1-p)^d`, so deletion is exponential in the number of columns.
- Mean imputation leaves the mean alone and multiplies the variance by exactly
  `m/n`. Measured on a real column: **0.9486** against `sqrt(0.9) = 0.9487`,
  and **0.5447** against `sqrt(0.3) = 0.5477`. The correlation of BMI with the
  target fell from **0.5865 to 0.4538** at 40% imputed.
- MCAR, MAR and MNAR missingness on the same column gave complete-case biases
  of **+0.12, +1.35 and −2.95 kg/m²**. Iterative imputation (MICE) fixed the
  MAR case (**+1.35 → −0.16**) and could not fix MNAR (**−2.95 → −2.58**).
- Under MNAR, mean imputation **flipped the sign** of a fitted regression
  coefficient: **+5.603 → −0.413**.
- Dropping incomplete rows cost **0.05** of test R² at 10% missingness and
  **5.17** at 30%. The choice among five imputers cost **0.003**.
- `add_indicator=True` was worth **+0.1675** of accuracy when the missingness
  itself carried the signal.
- **Thirty good sensor readings plus five stuck at 100: the `|z| > 3` rule
  flagged none of them** (z = 2.4135). The IQR fence flagged five plus one
  false positive; the modified z-score flagged exactly five, at M = 112.5.
- Masking is monotone: one outlier is caught at z = 5.38, three barely at
  3.11, and **from the fourth onwards the rule never fires again**.
- No sample can contain a `|z|` larger than `(n-1)/sqrt(n)`, so for **n ≤ 10**
  the rule `|z| > 3` is not strict — it is inert.
- On a column with skewness 5.43 and no contamination, the three rules flagged
  **6, 65 and 81** rows of 569. After one Yeo–Johnson: **0, 1 and 0**.
- Box–Cox chose λ = **−0.4357** by maximum likelihood and took the skewness
  from **5.4328 to 0.0584**. Removing the skew was worth **+0.0157** to
  Gaussian naive Bayes and **+0.0003** to logistic regression.
- Leaking the scaler across the split was worth **+0.0034** (t = 6.0, real but
  tiny). Leaking the imputer was worth nothing measurable. **Leaking a
  row-dropping decision was worth +0.1702 of R²** (t = 29.9, on 192 of 200
  splits).

## Before the next lecture

- Read `notes.pdf`, especially the variance-shrinkage derivation and the proof
  of the `(n-1)/sqrt(n)` ceiling.
- Attempt practice questions 2, 5 and 9. Question 5 is the ceiling proof;
  question 9 is a debugging story you will meet for real.
- Start one habit now: before you fit anything, count the holes, ask why they
  are there, and look at the histogram of every column.
- Next up is **feature engineering, encoding and dimensionality reduction**.
  Carry one thing across: every one of those steps also learns a number from
  the training set.

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
