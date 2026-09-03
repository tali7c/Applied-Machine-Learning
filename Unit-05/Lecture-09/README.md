# Unit V — Lecture 9: Maximum Likelihood, Model Adequacy and Over-fitting

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The second of the five Unit V lecture packages, following Lecture 8 on
regression foundations and least squares. Budgeted at a minimum of 1 contact
period; the material comfortably fills 2.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Direct regression method (against inverse and orthogonal) | 25 |
| Maximum likelihood estimation for the linear model | 25 |
| Coefficient of determination R², and adjusted R² | 26 |
| Checking model adequacy: residual plots, Q–Q, heteroscedasticity, leverage | 26 |
| Over-fitting and its detection | 27 |
| Cross-validation as the detector | 27 |

Course outcomes: **CO2** — apply appropriate machine learning techniques to
solve a given problem; **CO3** — evaluate and compare models.

References: Bishop §1.2.5 and §3.1.1; ISLP §3.1.3, §3.3.3 and §5.1;
Müller and Guido §5.1; MML Ch. 9.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 49 frames, 76 overlay pages — for the room |
| `notes.pdf` | 31 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | fifteen vector figures, every one produced by running code rather than drawn |

## The one idea

**Least squares is not an arbitrary choice.** Write down a Gaussian noise
model, form the likelihood, take logarithms, and the log-likelihood turns out
to be a constant minus SSE/(2σ²) — so maximising the likelihood *is*
minimising the sum of squared errors. That single line tells you both why the
estimator is what it is and exactly which assumptions you now have to check.
Everything after it — R², residual plots, over-fitting, cross-validation — is
that checking.

## The measurements the lecture turns on

Every number below was produced by running code on scikit-learn's bundled
`load_diabetes` (442 patients), Anscombe's published quartet, or fixed-seed
synthetic data. Nothing downloads anything.

- On five points, **direct 0.6000, inverse 1.0000, orthogonal 0.7208**. Each
  slope minimises its own criterion and no other, and the ratio of the first
  two is exactly r² = 0.6.
- **The same b₁ = 0.60000 maximised the Gaussian log-likelihood at σ² = 0.48
  and at σ² = 1**, and equalled argmin SSE to five decimals. σ² scales the
  likelihood; it cannot move the slope.
- **σ̂²_ML = SSE/n is biased downwards by (n−2)/n.** Over 20,000 datasets with
  σ² = 4 and n = 10 it averaged **3.2117** against a predicted 3.2000, and was
  too small in **73.4%** of runs. The unbiased s² averaged 4.0146.
- Change the noise to Laplace and the MLE becomes **least absolute
  deviations**: it recovered the true intercept 2.00 as **2.77**, against
  least squares' **3.00**. The two slopes agreed to three decimals — sixty
  points cannot demonstrate an efficiency gap, and the notes say so.
- R² by hand on five points: SStot 6.00, SSres 2.40, **R² = 0.6000**, and the
  same number from r², from b₁²Sxx/Syy and from `r2_score`.
- **R² never falls when you add a predictor.** Twenty-seven columns of pure
  noise on 30 rows took it from **0.2413 to 0.9401**, and the smallest of the
  27 steps was **+3×10⁻⁶** — on the positive side.
- **Adjusted R² is a penalty, not a test.** It caught that case (−0.7375), but
  on 442 rows with 410 junk columns it read **+0.6597** while cross-validation
  read **−5.7605**.
- **Held-out R² has no lower bound**: −3 on three numbers by hand, −1.106 for
  a degree-14 fit, and **−22.04** for a degree-3 polynomial on diabetes
  against **−0.0001** for predicting the mean.
- **Anscombe's four datasets share every summary statistic and R² = 0.6665.**
  The residual diagnostics separate them completely: curvature R² = **1.0000**
  for II (an exact parabola), largest residual **2.80×** the next for III,
  leverage **1.0000** and an undefined deleted slope for IV.
- Heteroscedastic data gave **R² = 0.9267** with a residual sd ratio of
  **1.82** and Breusch–Pagan p ≈ 2×10⁻⁶. Coefficients survive it; every
  standard error and p-value does not. Taking log y moved the statistic from
  **22.91 to 3.24** (p = 0.072) — most of the way back, not all of it.
- Over-fitting: training RMSE fell at **all 20 steps** — verified, not
  asserted — while test RMSE turned up after **degree 3** and reached 153 at
  degree 20, **548× worse**. Between degree 3 and degree 20 the training score
  improved by 40% and the model became unusable.
- **Over-fitting depends on n, not just on the model.** The same degree-20
  polynomial gave a test RMSE of 13,292 at n = 15 and **0.2572 at n = 1000**,
  against a noise floor of 0.2500.
- **5-fold CV and LOOCV both chose degree 3 — the same answer as the test set
  — using only the training rows**, while the training score chose the largest
  model offered. A single validation split chose 3, 4 or 5 depending on the
  seed, and estimated the degree-3 error anywhere between 0.2243 and 0.4415.

## Before the next lecture

- Read `notes.pdf`, especially the maximum likelihood derivation written out
  line by line and the argument for why E[SSE] = (n−2)σ².
- Attempt practice questions 2, 4 and 7. Question 2 is the derivation from a
  blank page; question 4 is R² by hand; question 7 is a debugging story you
  will meet for real.
- Start two habits now: never report an R² without a residual plot beside it,
  and never report a training score as if it were a result.
- Next up is **logistic regression**. Carry the *method*, not the formula:
  state the noise model, write the likelihood, take logarithms, maximise.
  Squares will not appear, because the response is no longer Gaussian.

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
