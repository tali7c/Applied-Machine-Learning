# Unit V — Lecture 8: Regression Foundations and Least Squares

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The first of the five Unit V lecture packages, and the lecture that opens the
unit. Budgeted at a minimum of 2 contact periods.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Introduction to regression; response, predictors, error | 22 |
| Regression examples and models; linear *in the parameters* | 22 |
| Steps in a regression analysis, applied end to end | 23 |
| Simple linear regression and its mathematical proof | 23 |
| Least-squares estimation: the derivation, the normal equations, the second-order condition | 24 |
| The line of best fit and its properties | 24 |

Course outcomes: **CO1** (describe machine learning concepts and problems),
**CO2** (apply appropriate techniques), **CO3** (evaluate and analyse models).

References: ISLP §3.1; Bishop §3.1; Mathematics for Machine Learning §9.2.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 51 frames, 73 overlay pages — for the room |
| `notes.pdf` | 29 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | ten vector figures, every one produced by running code rather than drawn |

## The one idea

**For this one model, the minimisation has an answer you can write down.**

Unit II said what to minimise (mean squared error). Unit III said how
(gradient descent). This lecture takes the simplest possible model — a
straight line — puts the same squared-error objective on it, and shows that
setting the two partial derivatives to zero produces *two linear equations in
two unknowns*, which you solve once, exactly:

```
b1 = Sxy / Sxx        b0 = ybar - b1 * xbar
```

Deriving that, with the second-order condition, is the examinable core of the
lecture. Everything else in the package exists to make it land.

## The measurements the lecture turns on

Every number below was produced by running code, on five hand-checkable
points, on scikit-learn's bundled `load_diabetes` (442 patients), on the
published Anscombe table, or on fixed-seed synthetic data. Nothing downloads
anything.

- Five points, `Sxx = 10`, `Sxy = 6`: **ŷ = 2.2 + 0.6x**, SSE = 2.40, and
  every intermediate quantity is an integer. `np.polyfit` agrees with the
  hand arithmetic to **4.441e-16**; `LinearRegression` agrees to **exactly
  zero**.
- The Hessian of SSE is `[[10, 30], [30, 110]]`, determinant **200**, which is
  `4·n·Sxx = 4·5·10`. Both eigenvalues positive, so the stationary point is a
  strict global minimum. Make every x identical and the determinant is
  **0** — the algebra refuses, correctly.
- The matrix route by hand: `X'X = [[5,15],[15,55]]`, `X'y = [20,66]`,
  `det = 50 = n·Sxx`, `inv(X'X) = (1/50)[[55,-15],[-15,5]]`, and
  `(1/50)[110, 30] = (2.2, 0.6)`. `solve`, `lstsq` and `pinv` all agree to
  **2.665e-15**.
- **Residuals summing to zero is necessary but not sufficient.** Three of four
  candidate lines had `Σeᵢ = 0`; only the least-squares line also had
  `Σxᵢeᵢ = 0`, and only it had the smallest SSE (2.40 against 2.50, 2.55 and
  4.00).
- Drop the intercept and the first property dies: the no-intercept fit
  `ŷ = 1.2x` has residuals summing to **+2.0** and an SSE of **6.80** against
  2.40. `Σxᵢeᵢ` is still zero, because that is the equation that was solved.
- Gradient descent on the **same objective**, from (0,0) with η = 0.01, took
  **619 iterations** to match the closed form to four decimals — and its loss
  was already 2.4035 against 2.4000 at iteration 200, long before its
  parameters were right.
- On diabetes (`bmi` → progression), **100 full-batch gradient steps
  reproduced `(X'X)⁻¹X'y` to a relative 4.441e-16**. `SGDRegressor` with a
  constant step stopped **6.1% short** on the slope — the Lecture 3 noise
  floor, reported rather than hidden.
- `Var(b1) = σ²/Sxx`, measured: widening the range of x from ±0.5 to ±4 grew
  `Sxx` from 1.84 to 117.9 and cut the spread of 4000 fitted slopes from
  **0.7237 to 0.0932** — a factor of 7.8 on the same 20 rows. The measured and
  theoretical columns agree to three decimals.
- **Leverage decides which rows matter.** The same +4 error in y changed the
  slope by **+0.0000** at x = x̄ and by **+0.8000** at the end of the range.
  Two of five points carry 80% of the slope leverage.
- **Anscombe's quartet**: four datasets with x̄ = 9.00, ȳ = 7.50,
  Sxx = 110.00, b0 ≈ 3.00, b1 ≈ 0.50 and SSE ≈ 13.75 — agreeing to two
  decimal places on every statistic least squares computes — and four
  completely different pictures.
- End to end on 442 patients: `bmi` alone gives a **5-fold RMSE of 62.6 ± 2.9**
  against **77.0** for always predicting ȳ, with train RMSE 61.7 and test RMSE
  64.7. All ten single-predictor fits share the intercept **152.133**, because
  the columns are already centred and `b0 = ȳ - b1·x̄`.
- Least absolute deviations on the same five points reaches SAE = 3.00 with
  **two different lines** (1.25 + 0.75x and 1.434 + 0.7132x). Least squares
  has exactly one answer, always.

## Before the next lecture

- Read `notes.pdf`, especially §6 — the derivation with every step written
  out — and §8, the four properties of the fitted line.
- Attempt practice questions 1, 2 and 7. Questions 1 and 2 are by-hand
  least-squares computations of exactly the kind the test will ask; question 7
  is the derivation from scratch on a blank page.
- Carry one number forward: `SSR/SST = 3.6/6.0 = 0.6` on the five points, and
  `r² = 0.7746² = 0.6` as well. That is not a coincidence, and the next
  lecture gives it a name.
- Next up is **maximum likelihood, model adequacy and over-fitting**: where
  squared error comes from, R², residual diagnostics, and cross-validation.

## About the figures

Every figure here is computed, not drawn. The generating script lives with the
course's internal tooling and prints each measured number that the notes quote,
so the prose and the pictures cannot drift apart. If a number in the notes looks
wrong, run the code beside it — that is the check that matters.

| Figure | What it shows |
|---|---|
| `fitline.pdf` | the five points, the fitted line, the five residuals, the centroid |
| `squares.pdf` | each residual as a literal square; SSE is the shaded area |
| `candidates.pdf` | four candidate lines and their SSE |
| `bowl.pdf` | contours of SSE(b0,b1), the gradient-descent path, the closed-form star |
| `leverage.pdf` | the same error in y at x̄ and at the end of the range |
| `sxx.pdf` | four sampling distributions of b1 against σ/√Sxx |
| `anscombe.pdf` | the quartet |
| `diabetes.pdf` | the real-data fit, its residuals against fitted values, their distribution |
| `models.pdf` | four regression models; which are fitted by one linear solve |
| `lossshape.pdf` | squares against absolutes, and the non-unique LAD answer |

## Build

```bash
cd latex
for i in 1 2 3; do pdflatex -interaction=nonstopmode slides.tex; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode notes.tex;  done

grep -c '^!'                slides.log notes.log   # must be 0
grep -c 'Overfull \\vbox'   slides.log notes.log   # must be 0
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
