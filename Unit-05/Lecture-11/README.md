# Unit V — Lecture 11: Multiple Linear Regression

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The fourth of the five Unit V lecture packages. Budgeted at a minimum of 2
contact periods.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Multiple linear regression model building and its mathematical proof | 29 |
| Interpretation of coefficients | 30 |
| Partial and standardized regression coefficients | 30 |
| Missing data in the regression setting | 31 |
| Model validation | 31 |

Course outcomes: **CO2** — apply appropriate machine learning techniques to
solve a given problem; **CO3** — evaluate and compare model performance.

References: ISLP §3.2–3.3 and §6.1.3; MML §9.2; Bishop §3.1; Montgomery, Peck
and Vining Ch. 3, 4 and 9.

## Prerequisites

Simple linear regression, least squares, the two normal equations and $R^2$.
This lecture generalises all of it; nothing is replaced. It also uses
standardisation and the rule that anything learned from data must be learned
inside the cross-validation fold.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 53 frames, 81 overlay pages — for the room |
| `notes.pdf` | 32 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | ten vector figures, every one produced by running code rather than drawn |

## The one idea

**A regression coefficient is a conditional statement, not a fact about a
column.** `b_j` is the effect of `x_j` *holding the other predictors in the
model fixed*. Change which other predictors are in the model and `b_j`
changes — legitimately, because it is now answering a different question. Get
that clause wrong and you will report a real effect with the wrong sign.

## The measurements the lecture turns on

Every number below was produced by running code on scikit-learn's bundled
`load_diabetes(scaled=False)` (442 patients, original units), `load_wine`, a
five-row hand example, or fixed-seed synthetic data. Nothing downloads
anything.

- The hand example is exact in integers: `det(X'X) = 60`,
  `beta = (2, −1, 4)`, residuals `(−1, 1, 0, 1, −1)`, `X'e = 0`,
  **R² = 0.8507 and adjusted R² = 0.7015** — and `LinearRegression` agrees to
  **8.6×10⁻¹⁴**.
- Perfect collinearity: adding `x3 = x1 + x2` gives rank 3 of 4,
  `det(X'X) = 0`, condition number **7.7×10¹⁸** — and `LinearRegression`
  returns an answer anyway. Two completely different coefficient vectors give
  **identical predictions to 0.00e+00**.
- With 8 patients and 11 parameters the null space has dimension 3: two
  coefficient vectors of norm **31.6** and **104.9** both fit the eight rows
  to `SSE ≈ 1e−23`.
- **Total serum cholesterol has slope +0.4723 on its own and −1.0900 with the
  other nine predictors in the model** — the opposite sign and 2.3× the
  magnitude. Four of the ten diabetes predictors change sign that way.
- The omitted-variable identity accounts for it exactly:
  **−0.2418 + 91.7683 × 0.007781 = +0.4723**.
- Frisch–Waugh–Lovell verified numerically: residualise both sides on the
  other nine predictors and the single slope between the residuals is the
  model coefficient, to **2.15×10⁻¹⁴**. Only **1.7%** of `s1`'s variance
  survives that residualisation; 66% of `bmi`'s does.
- Standardizing changes the ranking: `s1` moves from **6th** by raw magnitude
  to **1st** by β\* = b·s_j/s_y, and `sex` from 2nd to 6th. β\* is unchanged
  when `bmi` is rescaled by 10⁴; the raw coefficient changes by 10⁴.
- VIF: four of the ten exceed 10. `s1` reaches **59.2** (R²_j = 0.9831), so
  its standard error is **7.69×** inflated. Dropping `s1` takes the worst
  remaining VIF to 7.8.
- 2000 bootstrap resamples: the 95% intervals for `s1` and `s2` are
  **[−2.09, +0.11]** and **[−0.33, +1.63]** — both straddle zero — and the two
  coefficients correlate at **−0.964**. `bmi` (VIF 1.5) is never negative.
  **The prediction at the mean patient has a spread of 1.7%.**
- On a synthetic pair with corr(x₁,x₂) = 0.9986 whose truth was (3, 2), least
  squares returned **(+2.44, +2.60)** on all 200 rows and the two halves
  returned **(+2.55, +2.46)** and **(+2.21, +2.85)** — every one of them wrong
  — while the **sum** stayed within 0.06 of 5. Test R² was **0.9895** both
  times and predictions agree to r = 0.99999.
  **Collinearity breaks explanation, not prediction.**
- Missing data: a row survives at `(1−p)^d`. At 10% across ten columns only
  **33.3%** of rows survive; at 25% only 4.2%. Measured test R²: complete data
  **+0.3929**; at 25% missing, listwise deletion **−2.1451**, mean imputation
  **+0.4048**, MICE **+0.4038**, filling with zero **+0.0834**. The mean/MICE
  gap of 0.001 is noise; the 2.54 lost to deletion is not.
- Leakage from imputing before splitting, measured honestly: **+0.0002** for
  mean imputation and **+0.0042** for MICE. Small, because an imputer never
  looks at the target — put it in the `Pipeline` anyway.
- R² rose at **8 of 8** steps as pure-noise columns were added to 80 rows;
  **adjusted R² was fooled by the first junk column** and peaked there rather
  than at none; held-out R² fell at 7 of 8 steps, from **+0.2488 to −0.0794**
  — past zero. At 71 parameters on 80 rows, adjusted R² reads a cheerful
  **0.7863** while held-out R² is **−7.3523**.
- A straight line fitted to `y = x² + noise` has **R² = 0.0084**, residual
  mean −4.3×10⁻¹⁶ and zero residual–fitted correlation. Adding the `x²`
  column takes R² to **0.9710**. Look at the residual plot.

## Before the next lecture

- Read `notes.pdf`, especially the matrix derivation in Section 3 and the
  Frisch–Waugh–Lovell construction in Section 5.4.
- Attempt practice questions 2, 5 and 9. Question 2 is a 3×3 inverse and
  `beta` by hand; question 9 is a report you will be asked to review for real.
- Start one habit now: before you interpret any coefficient, print the VIFs
  and bootstrap the fit. If the interval straddles zero, say so.
- Next up is **regularization: ridge and lasso**. Ridge replaces `X'X` by
  `X'X + λI`, which is invertible for every λ > 0 — every failure in this
  lecture is a reason that lecture exists.

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
grep -ci 'not found'        notes.log              # must be 0
grep -c  'undefined'        notes.log              # must be 0
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
