# Unit IV — Lecture 7: Data Splitting, Cross-Validation and Imbalanced Data

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The last of the three Unit IV lecture packages. Budgeted at a minimum of 2
contact periods; the expected schedule allocates 3.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Train–test split strategies; what a held-out score estimates | 19 |
| Data splitting: stratified, grouped and time-ordered | 19 |
| Cross-validation techniques: k-fold, stratified, repeated, shuffle-split, LOOCV | 19 |
| Choosing k: bias, variance and cost; tuning honestly with nested CV | 19 |
| Handling imbalanced data: metrics, oversampling, undersampling, SMOTE, class weights | 20 |

Course outcome: **CO2** — apply appropriate machine learning techniques to
solve a given problem.

References: Müller and Guido §5.1; ISLP Ch. 5; HKP §8.6.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 47 frames, 89 overlay pages — for the room |
| `notes.pdf` | 30 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | ten vector figures, every one produced by running code rather than drawn |

## The one idea

**A score is an estimate, and every estimate has a spread.** Lecture 6 chose
the columns; this lecture decides how you find out whether any of it worked.
The rule that runs through all three sections is the one Unit IV has been
building towards since Lecture 5: *anything you learn from the data — a mean,
a category list, a feature subset, a hyper-parameter, a resampled row — is
part of the model, and belongs inside the loop.*

## The measurements the lecture turns on

Every number below was produced by running code on scikit-learn's bundled
`load_wine` (178 rows), `load_breast_cancer` (569), `load_iris` (150),
`load_digits` (1797) or fixed-seed synthetic data. Nothing downloads anything,
and `imbalanced-learn` is not required — SMOTE is implemented in fifteen lines
so you can see exactly what it does.

- **A single train/test score is a random variable.** Five hundred seeds on
  wine gave **0.7556 to 1.0000**, mean 0.9132, sd 0.0425, and a 95% interval
  **[0.8106, 0.9778]** — seventeen points wide. The test set is 44 rows, so
  one patient is worth **0.0227** of accuracy, and √(p(1−p)/n) predicts
  **0.043** against the measured 0.0425.
- **That noise reverses model comparisons.** Two trees differing by
  **0.0021** on average produced single-split verdicts from **+0.1556 to
  −0.1333**, and a "winner" on **193 of 500** splits.
- **Test-set noise falls only as 1/√n**: sd 0.0425 on 44 test rows against
  0.0214 on 142.
- **Stratify.** On a sorted 5% file, `KFold` put all ten positives in fold 1,
  left fold 1's *training set with a single class*, and reported **mean
  accuracy 1.0000 with mean recall 0.0000** — with no error message.
- **Group.** Forty patients with **coin-flip labels** scored
  **1.0000 ± 0.0000** under a random 5-fold and **0.6483 ± 0.1984** under
  `GroupKFold`, because **37 of 40** patients had rows on both sides of fold 1.
- **Respect time.** A random split on 600 days of a drifting seasonal series
  scored **R² = +0.9975**; the honest time-ordered answer was **−2.8167**.
  61.3% of test days had *both* neighbours in the training set. Worse, shuffled
  CV ranked a useless model (**−2.82** in time order) *above* a good one
  (**+0.92**).
- **k-fold costs k fits.** On 569 rows, LOOCV took about **100×** 5-fold to
  reach the identical **0.9789**. Small k is pessimistic because it starves the
  model: 2-fold trains on 284 rows, 20-fold on 540.
- **Cross-validation is about four times steadier than one split**: sd of the
  mean **0.0030** for 5-fold against **0.0114** for one 80/20 split. And
  LOOCV's across-fold sd of 0.1437 is **not** instability — it is √(p(1−p)).
- **`GridSearchCV.best_score_` is a maximum, not an estimate.** On 100 rows of
  pure noise with coin-flip labels it read **0.5820** against a nested
  **0.5265** — optimism **+0.0555** on **20 of 20** trials, where the truth is
  0.5. On real data it was **+0.0212** (paired t = 9.58, p = 1.1×10⁻⁸). With
  **one** combination in the grid the optimism was **exactly zero**; with sixty
  it was **+0.0783**.
- **At 99:1, accuracy ranks the useless model higher.** Always-say-no scores
  **0.9900**; a real logistic regression scores **0.9867** — while carrying
  ROC-AUC **0.8553** and PR-AUC **0.1075**, eleven times the 0.01 baseline.
- **Oversampling, undersampling, SMOTE and class weights all took recall from
  0.0400 to about 0.78 — and none improved PR-AUC.** The best PR-AUC in the
  table, **0.1916**, belongs to the untreated model. They move the threshold;
  they do not create information. That result is reported rather than hidden.
- **`class_weight="balanced"` matches random oversampling** for logistic
  regression (recall 0.7800 both) and does **nothing** to a random forest's
  predictions — different model, different answer.
- **Resampling before the split reported F₁ = 0.8942 against an honest
  0.1133** — an eightfold overstatement from moving one line of code, because
  **100% of the minority test rows were byte-identical copies of training
  rows**.

## Before the next lecture

- Read `notes.pdf`, especially the twelve-row split arithmetic, the argument
  for why `best_score_` is biased upward, and §6.6 on resampling inside the
  fold.
- Attempt practice questions 2, 5 and 9. Question 2 is a confusion matrix by
  hand; question 9 is a debugging story you will meet for real.
- Do the "break it deliberately" exercise at the end of §6.7: move the
  `smote` call to the wrong side of the split, watch F₁ leap from 0.11 to
  0.89, and then prove to yourself that the leap is fictitious.
- Start one habit now: **never report a bare score.** Report the estimator,
  the splitter and the spread — "0.9789 ± 0.0142, stratified 5-fold" — and say
  what the test set was protected from.

## About the figures

Every figure here is computed, not drawn. The generating script lives with the
course's internal tooling and prints each measured number that the notes quote,
so the prose and the pictures cannot drift apart. If a number in the notes
looks wrong, run the code beside it — that is the check that matters.

One seed runs the whole lecture: `np.random.default_rng(0)` / `random_state=0`.
Three synthetic datasets keep their own named seeds, because those seeds define
the data rather than stir it — the 5000-row 99:1 problem, the 40-patient
grouped study and the 600-day series. Everything else that varies is a sweep
(500 splits, 40 shuffles, 20 nested trials) whose seeds are derived from the
same 0, because the spread across repetitions is what those experiments
measure.

The one exception to reproducibility is flagged in the notes: wall-clock
timings are machine-dependent and vary between runs. Every cost claim is
therefore stated in model fits — LOOCV is 569 fits against 5-fold's 5, 113.8×
as many — which is true on every machine.

## Build

```bash
cd latex
for i in 1 2 3; do pdflatex -interaction=nonstopmode slides.tex; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode notes.tex;  done

grep -c '^!'                slides.log notes.log   # must be 0
grep -c 'Overfull \\vbox'   slides.log notes.log   # must be 0
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
