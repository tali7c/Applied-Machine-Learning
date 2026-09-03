# Unit V — Lecture 12: Regularization — Ridge and Lasso

**Applied Machine Learning (CSAI2017P) · B.Tech CSE (AI & ML) · Autumn 2026**

The last of the five Unit V lecture packages, and the one that closes the
unit. Budgeted at a minimum of 1 contact period; it will take more than that
if the ridge derivation is done properly on the board, which it should be.

## What this lecture covers

| Topic | Syllabus rows |
|---|---|
| Regularization: the penalised objective, and why λ is a hyper-parameter | 32 |
| Ridge regression: the closed form, its derivation, and why the inverse always exists | 32 |
| Lasso: why there is no closed form, soft thresholding, and exact zeros | 32 |
| Coefficient paths, scaling, choosing λ by cross-validation, the bias–variance trade, elastic net | 32 |

Course outcome: **CO4** — apply machine learning to various real-world
problems and domains.

References: ISLP §6.2; Bishop §3.1.4; Müller and Guido §2.3.

## Files

| File | What it is |
|---|---|
| `slides.pdf` | 47 frames, 82 overlay pages — for the room |
| `notes.pdf` | 32 pages — written so you can learn this without attending |
| `latex/` | sources |
| `figures/` | ten vector figures, every one produced by running code rather than drawn |

## The one idea

**Regularization deliberately fits the training data worse in order to fit
everything else better.** You stop asking for the smallest training error and
ask instead for a small training error *and* small coefficients, with a single
dial — λ — setting the exchange rate. λ is not estimated by the fit; it is
chosen from outside, by cross-validation.

This lecture is the answer to three problems Unit V raised and did not solve:
over-fitting, multicollinearity, and the logistic model's perfect-separation
blow-up. One penalty term fixes all three.

The structural fact to carry away: **X'X + λI is invertible for every λ > 0,
whatever X is.** Ridge therefore has an answer in situations where ordinary
least squares does not exist at all. That is not a refinement of least
squares; it is a change of kind.

## The measurements the lecture turns on

Every number below was produced by running code on scikit-learn's bundled
`load_diabetes` (in both its scaled and its original-units form) or on
synthetic data drawn from **one seed**, several sets of it deliberately with
p ≥ n. Nothing downloads anything.

**One dataset, one seed.** The seed is `0`, everywhere: every generator is
`np.random.default_rng(0)` and every split and shuffle is `random_state=0`.
Where the point of a demonstration is run-to-run *variation* — the twenty
draws of the p > n problem, the thirty bootstrap resamples of the correlated
trio — the seeds are `0, 1, 2, …`, derived from that same one. Run the code as
written and you get these numbers, not similar ones.

- **n = 40, p = 38, three real signals.** OLS: train R² **0.9997**, test R²
  **0.1470**. Ridge (α = 10): train **0.9336**, test **0.6598**. Fitting the
  training data worse bought a factor of four and a half on unseen rows.
- **Two columns correlated at 0.99989.** Across five bootstrap resamples of
  the same 100 rows, OLS reported coefficient pairs from `[+2.74, −0.70]` to
  `[+11.02, −8.95]` — while their **sum stayed at 2.03 ± 0.02 every time**.
  Ridge reported `[1.03, 1.00]` to `[1.10, 0.93]`.
- **Perfect separation, 6 points.** `LogisticRegression` is L2-penalised by
  default. With the penalty effectively off (C = 1e10) the slope reaches
  **8.35** and keeps climbing; at C = 1 it is **1.10**. The penalty is what
  makes the answer exist.
- **By hand.** x = (1, 2, 2), y = (2, 3, 5) gives β̂ = 18/(9 + λ), so
  **2.0 → 1.8 → 1.0 → 0.2** at λ = 0, 1, 9, 81. `Ridge` reproduces all four to
  four decimal places. The 2×2 case gives (1, 2) → (0.875, 1.375) →
  (0.625, 0.875).
- **A singular design.** With column 2 exactly twice column 1,
  det(X'X) = **0**, rank 1, eigenvalues (0, 70), and `np.linalg.inv` raises
  `LinAlgError`. Add λ = 1: determinant **71**, eigenvalues (1, 71), and a
  unique answer (0.19718, 0.39437). Confirmed by hand and by `Ridge`.
- **`LinearRegression` does not raise on a singular design.** It quietly
  returns the minimum-norm solution — which is exactly ridge at λ → 0,
  agreeing to **6.5 × 10⁻¹²** on a 50 × 200 problem. "OLS" at p > n is already
  a regularised estimator.
- **n = 50, p = 100, correlated columns, dense truth.** On the seeded draw,
  OLS test R² **+0.4183** with a training R² of exactly 1; ridge at CV-chosen
  α = 5.2 reaches **+0.6303** and lasso **+0.6618**. One draw of a p > n
  problem settles nothing, so the script repeats it over 20 draws: mean test
  R² OLS **+0.4446**, ridge **+0.6532**, lasso **+0.6443**. Ridge beats OLS in
  **19 of 20** draws and lasso in **16 of 20**. On the worst draw OLS scores
  **−0.4915**, worse than predicting the mean.
- **n = 50, p = 200, sparse truth.** The other way round, and far more
  decisive: lasso **+0.9294** recovering all five real columns, ridge
  **+0.2831** — which is exactly the unpenalised fit, because CV chose the
  smallest α on the grid. Neither penalty dominates — which is why both are on
  the syllabus.
- **Soft thresholding, by hand.** OLS coefficients (3, 1, −0.4, 0.2) at λ = 1
  become **(1.5, 0.5, −0.2, 0.1)** under ridge and **(2.5, 0.5, 0, 0)** under
  lasso. Two exact zeros. Twenty lines of hand-written coordinate descent
  reproduce `sklearn.linear_model.Lasso` to **10⁻¹¹**.
- **The geometry, on real data.** At the same L1 budget of 20.46, lasso
  answers **(20.46, 0.00)** and ridge answers **(12.23, 8.23)** — and the
  lasso corner has the *lower* residual sum of squares of the two.
- **Ridge never returns a zero.** Sweeping α to 10⁶ on the diabetes data, the
  smallest coefficient anywhere on the path is **1.455 × 10⁻³**. Lasso deletes
  in a definite order (age, s2, s4, s1, s6, sex, s3), and **half the features
  retain 94.5% of the training R²**.
- **Scaling is not optional.** On raw clinical units, three narrow columns
  (`sex`, `bmi`, `s5`) consume **92.6%** of the ridge penalty; `age` and `s6`
  pay 0.02% each. Divide `bmi` by 100 and OLS predictions move by
  **8.7 × 10⁻¹²** while ridge predictions move by **71.0** — and the lasso
  **deletes `bmi`**, the most predictive variable in the table.
- **The intercept is not penalised.** Shift y by 500: the intercept moves by
  exactly **500.0000** and no coefficient by more than 1.3 × 10⁻¹⁴. Penalise
  it (`fit_intercept=False` on uncentred y) and test R² becomes **−3.84**.
- **Cross-validation.** Ridge minimum α = 22.70 (test R² 0.3971); the
  one-standard-error choice α = 147.74 (test R² **0.4049**, ‖β‖ 23% shorter).
  Lasso minimum α = 0.977 (8 features, 0.3889); 1-SE choice α = 9.69
  (**4 features**, 0.3616).
- **The trade, measured over 400 training sets.** At the optimum λ = 4.78,
  bias² rose by **0.65** and variance fell by **3.80** — a net gain of
  **3.15**, 45% of the reducible error. The closed form σ²p/‖β‖² predicted
  5.43 against a measured 4.78, about 14% high, because X'X for a Gaussian
  design is Wishart and not I. And MSE′(0) < 0 always: **λ = 0 is never
  optimal.**
- **Elastic net.** Given three statistically indistinguishable copies of one
  signal, lasso kept them in **29, 13 and 21** of 30 bootstrap resamples —
  a ranking with nothing behind it; elastic net kept all three in
  **30 of 30**, and predicted marginally better (0.9636 against 0.9609).
- **And the honest one.** On the plain 442 × 10 diabetes problem, with no
  collinearity crisis, regularization is worth about **six thousandths of R²**
  (OLS 0.3929, RidgeCV 0.3989, LassoCV 0.3889). That result is reported rather
  than hidden.

## What to do next

- Read `notes.pdf`. The parts that are not on the slides: the full
  positive-definiteness proof and where it fails at λ = 0, the subdifferential
  argument for why the lasso has no closed form, the SVD view with effective
  degrees of freedom, and the bias–variance algebra that produces
  λ\* = σ²p/‖β‖².
- Attempt practice questions 2, 4 and 7. Question 2 is the ridge closed form
  on a 2×2 matrix by hand; question 4 is the invertibility proof; question 7
  is soft thresholding. Question 10 is a design problem you will meet for
  real.
- Start two habits now. Never write `Ridge()` or `Lasso()` outside a
  `Pipeline` that begins with `StandardScaler`. And never report a coefficient
  without saying which λ produced it.
- **Unit V ends here.** Least squares, maximum likelihood and R²; over-fitting
  and its detection; logistic regression; multiple regression and its
  pathologies; and now the one dial that controls all of them. Unit VI opens
  with classification.

## About the figures

Every figure here is computed, not drawn. The generating script lives with the
course's internal tooling and prints each measured number that the notes quote,
so the prose and the pictures cannot drift apart. If a number in the notes
looks wrong, run the code beside it — that is the check that matters.

Ten figures: `handridge`, `collinear`, `geometry`, `shrinkage`, `paths`,
`pgtn`, `scaling`, `cvcurve`, `biasvar`, `elasticnet`.

## Build

```bash
cd latex
for i in 1 2 3; do pdflatex -interaction=nonstopmode slides.tex; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode notes.tex;  done

grep -c '^!'                slides.log notes.log   # must be 0
grep -c 'Overfull \\vbox'   slides.log notes.log   # must be 0
```

Compile from inside `latex/` — the `../../../_shared/` paths depend on it.
