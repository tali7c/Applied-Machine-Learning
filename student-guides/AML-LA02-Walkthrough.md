# Lab Assignment 2 — worked walkthrough

**▶ Watch:
[`AML-LA02-Walkthrough.mp4`](AML-LA02-Walkthrough.mp4)**
— 8 min 25 s, 1080p. Click the file, then Download or play it in the browser.

Regression on California Housing. The modelling is ordinary; reading the numbers
is the real skill.

> Numbers below come from `scikit-learn 1.8.0` with `random_state=0`. Your last
> digit may differ on a different build.

---

## The setup every question builds on

```python
df = fetch_california_housing(as_frame=True).frame
X, y = df.drop(columns='MedHouseVal'), df.MedHouseVal
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
# 16,512 train rows, 4,128 test rows
```

Split once, at the top, and do not touch `X_te` again until you are reporting.

---

## A1. Simple linear regression `[2]`

```python
lr = LinearRegression().fit(X_tr[['MedInc']], y_tr)   # double brackets: 2-D
r2_score(y_te, lr.predict(X_te[['MedInc']]))
```

```
slope     0.4203     intercept 0.4432     test R²   0.4467
```

`X_tr['MedInc']` is a Series and raises an error. `X_tr[['MedInc']]` is a
one-column DataFrame, which is what sklearn wants.

**Model observation.** Slope 0.4203, intercept 0.4432, test R² 0.4467. `MedInc`
is in tens of thousands of dollars and `MedHouseVal` in hundreds of thousands, so
the slope says a district whose median income is $10,000 higher has a median
house value about **$42,000** higher. The intercept is the fitted value at zero
income, $44,320, which is an extrapolation well outside the data and should not
be read as a real price.

> (c) is one sentence, but it must convert the units. "The slope is 0.42" is not
> an answer to "what does it mean in real units".

## A2. Multiple linear regression `[2]`

```python
mlr = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
p    = mlr.predict(X_te)
mae  = mean_absolute_error(y_te, p)
mse  = mean_squared_error(y_te, p)
rmse = np.sqrt(mse)              # RMSE is not a separate sklearn function
```

| metric | value | in money | what it is |
|---|---|---|---|
| MAE | 0.5429 | $54,292 | typical miss, in the target's own units |
| MSE | 0.5424 | — | squared units: dollars-squared means nothing |
| RMSE | 0.7365 | $73,651 | back in dollars, big errors weigh more |
| R² | 0.5840 | — | 58% of the variance explained |

**Model observation.** Test MAE 0.543 ($54,300), RMSE 0.737 ($73,700), R² 0.584.
Using all eight features instead of income alone lifts R² from 0.45 to 0.58. The
predicted-against-actual plot shows regression to the mean: the cloud is flatter
than the 45° line, so cheap districts are over-predicted and expensive ones
under-predicted. A vertical stripe at 5.0 marks the 965 districts censored at
$500,001; the model predicts a spread of values for them and is wrong on all of
them by construction. RMSE exceeding MAE by 36% says the error is not evenly
spread — a minority of districts carry most of it.

## A3. Metric choice `[2]`

No new modelling. The substance is entirely in (b) and (c).

**Model observation.** MAE 0.543, MSE 0.542, RMSE 0.737. **MSE** is hardest to
explain, because its units are squared dollars — I can tell a client the typical
miss is about $54,000 (MAE) or that large misses push it to $74,000 (RMSE), but
0.542 hundred-thousand-dollars-squared corresponds to nothing they can picture.
If a few very expensive districts matter most I would report **RMSE**: squaring
before averaging makes a single $300,000 miss count for far more than six
$50,000 misses, so RMSE rises when the model fails on exactly the districts we
care about, whereas MAE can stay flat while the expensive tail gets worse.

---

## B1. Polynomial features and over-fitting `[4]`

```python
for d in (1, 2, 3):
    pipe = make_pipeline(StandardScaler(),
                         PolynomialFeatures(degree=d, include_bias=False),
                         LinearRegression()).fit(X_tr, y_tr)
    tr = np.sqrt(mean_squared_error(y_tr, pipe.predict(X_tr)))
    te = np.sqrt(mean_squared_error(y_te, pipe.predict(X_te)))
```

| degree | features | train RMSE | test RMSE | verdict |
|---|---|---|---|---|
| 1 | 8 | 0.7297 | 0.7365 | honest |
| 2 | 44 | 0.6536 | 1.8035 | over-fitting |
| 3 | 164 | 0.5915 | **542.23** | collapsed |

You must report **both** columns. A test column alone cannot show over-fitting —
over-fitting is the gap between the two.

**Model observation.** Over-fitting begins at degree 2 and becomes catastrophic
at degree 3. At degree 1 train and test agree (0.730 vs 0.737): the model
generalises. At degree 2 train improves to 0.654 while test rises to 1.80 — the
fit got better on data it had seen and worse on data it had not, which is the
definition of over-fitting. At degree 3, 164 features from 8 originals, train
falls again to 0.592 while test reaches 542. The mechanism is extrapolation on
the tails: cubing features like `AveOccup`, whose maximum is 1,243, produces
values around 10⁹; the fitted coefficients become huge and opposite in sign, and
any test district slightly outside the training range gets a wildly wrong
prediction.

> Naming the degree is the start. Explaining how the table proves it is the answer.

## B2. Ridge and Lasso `[4]`

```python
for a in [0.01, 0.1, 1, 10, 100]:
    r = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                      Ridge(alpha=a)).fit(X_tr, y_tr)
    l = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                      Lasso(alpha=a, max_iter=50000)).fit(X_tr, y_tr)
    zeros = (l[-1].coef_ == 0).sum()       # of 44 polynomial features
```

| alpha | Ridge test RMSE | Lasso test RMSE | Lasso zero coefs |
|---|---|---|---|
| 0.01 | 1.803 | 1.813 | 15 of 44 |
| 0.1 | 1.799 | **0.827** | 38 of 44 |
| 1 | 1.759 | 1.123 | 43 of 44 |
| 10 | 1.434 | 1.142 | 44 of 44 |
| 100 | **0.685** | 1.142 | 44 of 44 |

Scaling **before** the penalty is not optional: alpha punishes coefficient size,
so an unscaled feature is penalised for its units rather than its usefulness.

**Model observation.** Best Ridge alpha 100, test RMSE 0.685. Best Lasso alpha
0.1, test RMSE 0.827 with 38 of 44 coefficients exactly zero. Both beat the
unpenalised degree-2 model (1.804), so regularisation recovers most of what the
polynomial expansion destroyed. Lasso zeroes coefficients and Ridge does not
because of the penalty's shape: Lasso's `sum |w|` has gradient ±1 regardless of
how small `w` is, so a feature earning less than that constant is pushed exactly
to zero and stays there; Ridge's `sum w²` has gradient `2w`, which vanishes as
`w` approaches zero, so coefficients shrink asymptotically and never arrive.
Geometrically the L1 constraint region is a diamond whose corners lie *on* the
axes; the L2 region is a circle with no corners to land on. In practice Lasso is
a feature selector, Ridge a stabiliser for correlated features — which polynomial
terms always are.

> Beyond alpha = 1 the Lasso has deleted everything and is predicting the mean.
> Say so when you see it.

---

## C1. Where the model fails `[2]` — compulsory

```python
err   = np.abs(y_te - mlr.predict(X_te))
worst = X_te.assign(actual=y_te, pred=mlr.predict(X_te), error=err) \
            .sort_values('error', ascending=False).head(20)

(worst.actual >= 5.0).sum()      # 16 of 20
worst.MedInc.mean()              # 2.78  vs  3.85 for the test set
```

The question says *describe what they have in common*. That means printing the
rows and looking at them, not reporting that the largest error was 3.9.

**Model observation.** Sixteen of the twenty worst districts sit exactly at the
5.00001 cap, and their mean `MedInc` is 2.78 against 3.85 for the test set as a
whole. The model sees below-average income and predicts about $148,000; the
recorded value is $500,001. Two failures are stacked here. First the target is
**censored**: everything above $500,000 was recorded as $500,001, so those rows
are not real values and no model can fit them. Second the features **cannot
express location value** — these are low-income blocks inside expensive coastal
cities, and latitude and longitude as raw numbers cannot encode "two miles from
the ocean in Los Angeles". I would add distance to the coast and to the nearest
city centre, a neighbourhood or ZIP identifier, and any local price index;
separately I would exclude the capped rows or model them with a censored
regression, and report that choice.

> "The model struggles with expensive houses" is what everyone writes. Naming the
> cap **and** the missing geography is what makes the answer complete.

---

## Four common mistakes

- Reporting a metric with no units — 0.54 is not money; $54,300 per district is
- Giving only test RMSE in B1 — over-fitting is the **gap**, so the train column is half the answer
- Scaling after the polynomial expansion, or not at all, in B2
- Fitting anything outside a `Pipeline` — `fit_transform` on the full `X` before the split leaks the test set and quietly inflates every number you report

## Before you submit

- [ ] Restart Kernel → **Run All**, top to bottom, no errors
- [ ] One results table, with units, for every metric you quote
- [ ] Filename `AML_LA02_<yourSAPID>.ipynb`, `README.md` beside it
- [ ] Every attempted question has a written observation
- [ ] `random_state=0` on the split, so your numbers are defensible in LQ2
