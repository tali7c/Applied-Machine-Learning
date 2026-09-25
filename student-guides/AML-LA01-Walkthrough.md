# Lab Assignment 1 — worked walkthrough

**▶ Watch:
[`AML-LA01-Walkthrough.mp4`](AML-LA01-Walkthrough.mp4)**
— 8 min 17 s, 1080p. Click the file, then Download or play it in the browser.

This page is the same content in text, so you can follow along without scrubbing.
Every question is answered: the code, the numbers, and the written observation.

**Mechanics of submitting** — cloning, VS Code, the notebook layout — are in
[`README.md`](README.md), not repeated here.

> Numbers below come from `scikit-learn 1.8.0`. Your third decimal may differ on a
> different build. The pattern is the finding; the digits are not.

---

## How the assignment is structured

| Section | Questions | Rule |
|---|---|---|
| A | A1, A2, A3 | any **2 of 3** |
| B | B1, B2 | any **1 of 2** |
| C | C1 | **compulsory** |

Every question needs code **and** a written observation — code alone is not an
answer. You may attempt more than the rule asks for.

## Before anything else

```bash
python -c "from sklearn.datasets import fetch_california_housing as f; f()"
```

Run it **at home**. On campus this fails with `CERTIFICATE_VERIFY_FAILED` — that
is the web filter, not your code. A2 (Telco Customer Churn) is a Kaggle CSV you
download once into `data/`.

---

## A1. Environment proof `[2]`

```python
import sys
import numpy as np, pandas as pd, sklearn
import matplotlib, seaborn as sns

print('python     ', sys.version.split()[0])
for m in (np, pd, sklearn, matplotlib, sns):
    print(f'{m.__name__:11s}', m.__version__)
```

Print them, do not claim them. A terminal screenshot is not an output cell.

**Model observation.** Versions are pinned so the same notebook produces the same
numbers on a different machine, months later. scikit-learn changes defaults
between releases — solvers, `n_init`, the meaning of a parameter — so an unpinned
notebook can quietly return different results without raising a single error.
Recording the versions makes a disagreement diagnosable instead of mysterious.

## A2. First look at the data `[2]`

```python
cal = fetch_california_housing(as_frame=True)
df  = cal.frame              # features + MedHouseVal in one DataFrame
df.head(); df.info(); df.describe().T; df.isna().sum()
```

`as_frame=True` gives you column names. Without it you get a bare array and spend
the assignment guessing which column is which.

| | answer |
|---|---|
| rows | 20,640 districts (one row = one census block group) |
| features | 8, all numeric |
| target | `MedHouseVal`, in hundreds of thousands of dollars |
| missing | none |

**Model observation.** 20,640 rows, 8 numeric features, target in units of
$100,000, no missing values. `Population` is on a completely different scale from
the rest: mean 1,425 and maximum 35,682, against `MedInc`'s mean of 3.87 and
maximum 15.0. Any model that measures distance or sums weighted features will be
dominated by `Population` unless the features are standardised first.
`describe()` also shows `MedHouseVal` capped at 5.00001 — 965 districts sit
exactly on that ceiling, so the target is censored at $500,001.

## A3. Missing values and outliers `[2]`

```python
tel = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
bad = pd.to_numeric(tel.TotalCharges, errors='coerce').isna()
print(bad.sum())                                    # 11
tel.loc[bad, ['customerID', 'tenure', 'MonthlyCharges']]   # every one has tenure == 0

tel['TotalCharges'] = pd.to_numeric(tel.TotalCharges, errors='coerce').fillna(0)
```

`TotalCharges` is text because 11 cells of 7,043 contain a single space. One
non-numeric character makes pandas call the whole column an object.

**Model observation.** `TotalCharges` was stored as text because 11 rows hold a
blank string. All 11 have `tenure = 0` — customers who joined this month and have
no billed total yet. I converted with `errors='coerce'` and filled those 11 with
0, their true value, rather than a median that would invent a billing history.
Neither `tenure` nor `MonthlyCharges` has any point outside 1.5 × IQR. Both are
bounded by the product, not by chance: `tenure` cannot exceed the 72 months the
company has existed, and `MonthlyCharges` is the sum of a fixed tariff list. An
absence of outliers here is a fact about the business, not evidence of a clean
dataset.

> "No outliers were found" is a finding. Explaining **why** there are none is the insight.

---

## B1. Scaling changes the answer `[4]`

```python
X, y = df.drop(columns='MedHouseVal'), df.MedHouseVal
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

raw = KNeighborsRegressor(n_neighbors=5).fit(X_tr, y_tr)
mean_absolute_error(y_te, raw.predict(X_te))                    # 0.81

scaled = Pipeline([('sc', StandardScaler()),
                   ('knn', KNeighborsRegressor(n_neighbors=5))]).fit(X_tr, y_tr)
mean_absolute_error(y_te, scaled.predict(X_te))                 # 0.43
```

**The arithmetic behind it.** A neighbour 5,000 people away and 2 income-units
away sits at distance `sqrt(5000² + 2²)`. The income term contributes 4 out of
25,000,000 — arithmetically invisible.

**Model observation.** Raw features: test MAE 0.81 ($81,000). Inside a Pipeline
with `StandardScaler`: 0.43 ($43,000). The same estimator on the same split is
roughly twice as accurate. KNN predicts by averaging the 5 nearest training rows
under Euclidean distance, computed on raw numbers. `Population` ranges over
35,679 units while `MedInc` ranges over 14.5, so squared differences in
`Population` are millions of times larger and the neighbour search is effectively
a search on `Population` alone. Standardising puts every feature at mean 0 and
standard deviation 1, so all eight contribute comparably. Scaling does not make
KNN cleverer — it stops one arbitrary unit choice from deciding the answer.

## B2. Correlation and leakage `[4]`

```python
tel['ChurnFlag'] = (tel.Churn == 'Yes').astype(int)
num = tel[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'ChurnFlag']]
num.corr()['ChurnFlag'].drop('ChurnFlag').sort_values(key=abs, ascending=False)
```

```
tenure          -0.352
TotalCharges    -0.199
MonthlyCharges   0.193
SeniorCitizen    0.151
```

`sort_values(key=abs)` is the line most students miss — ranking by raw value puts
the strongest signal at the bottom because it is negative.

**Model observation.** Strongest three: `tenure` −0.352, `TotalCharges` −0.199,
`MonthlyCharges` +0.193. Long-standing customers churn less; expensive monthly
plans churn more. `TotalCharges` is almost exactly `tenure × MonthlyCharges` —
the product correlates with it at **0.999**. Keeping all three makes the design
matrix near-collinear: a linear model can shift weight between them almost
freely, so coefficients become unstable and large, flip sign on a different
split, and none can be read as "the effect of tenure". The predictions stay
usable; the explanation does not. I would drop `TotalCharges` and keep the two
independent quantities, or use Ridge if all three must stay.

> This is redundancy, not target leakage — `TotalCharges` is knowable before
> churn. Say which one you mean.

---

## C1. Reproducibility check `[2]` — compulsory

Kernel → **Restart Kernel and Run All Cells**, then read your report back against
the fresh outputs.

**Model observation.** I restarted the kernel and ran all cells top to bottom.
Every number reproduced exactly: MAE 0.81 raw and 0.43 scaled, and the four
correlations to three decimals. It reproduces because the only source of
randomness is the split, pinned with `random_state=0`, and `KNeighborsRegressor`
is deterministic given the training set. The one thing that would break it is the
library versions — the numbers above are for scikit-learn 1.8.0, recorded in A1.

If a number *did* move, write that instead. An honest failure with a named cause
is a good answer; a silent "yes it reproduced" is not.

---

## Four common mistakes

- Fitting a scaler or imputer before the split — the test set leaks into training
- Code with no written observation — only half an answer, every time
- A number with no units — 0.43 is not an answer, $43,000 per district is
- Outputs not saved in the notebook — if the reader cannot see it, you did not run it

## Before you submit

- [ ] Restart Kernel → **Run All**, top to bottom, no errors
- [ ] Every output visible in the saved notebook
- [ ] Filename is `AML_LA01_<yourSAPID>.ipynb` with your real SAP ID
- [ ] `README.md` beside it, same folder
- [ ] Every attempted question has a written observation
- [ ] No datasets uploaded — the loading code is enough

LQ1 will ask you to read, trace and modify **your own** submitted code. Submit
code you can explain.
