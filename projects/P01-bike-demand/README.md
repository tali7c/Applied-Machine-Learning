# P01 · Hourly bike-rental demand for fleet rebalancing

> **Question.** Given the hour, the calendar and the weather, how many bikes will
> be rented in the next hour — and which factors actually drive that number?

Read `../Project-Handbook.md` first. It defines the repository rules, the commit
history you must build, the milestones, the 5-mark group rubric and the 10-mark
written quiz. This file defines only the work.

**Units:** III, IV, V (Unit VI optional) · **COs:** CO2, CO3, CO4  
**Difficulty:** the most contained theme in the catalogue — small, clean data, so
the marks are in the care you take, not in the wrangling.

---

## 1. The situation

A bike-share operator has a fixed fleet and finite trucks. If they knew the next
hour's demand at each part of the city, they could move bikes *before* the stands
empty instead of after. Your job is the forecasting half of that: a model that
turns "it is 8 a.m. on a working Tuesday in October and it is drizzling" into a
number, and an explanation of which of those facts mattered.

The interesting part is not getting a low error. It is that a model can look
excellent on a random split and be useless in deployment, because in deployment
you never get to interpolate between rows you have already seen.

---

## 2. The data

**UCI Bike Sharing** — hourly counts from the Capital Bikeshare system,
Washington DC, 2011–2012, with weather and calendar columns.

- Page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Download: the **zip** (273 KB) from that page, in a browser. Licence CC BY 4.0.
- Use **`hour.csv`** — 17 379 rows, 17 columns. (`day.csv` is the same data
  aggregated; you may use it for a sanity check, not as your main table.)
- Commit both CSVs to `data/raw/` — they are small.

**Columns you will care about:** `dteday`, `season`, `yr`, `mnth`, `hr`,
`holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`,
`windspeed`, `casual`, `registered`, `cnt`.

**Two traps in this file, both of which cost marks if you miss them.**

1. `cnt = casual + registered`, exactly. If you predict `cnt` with `casual` and
   `registered` in your feature matrix, you will report an R² near 1.0 and have
   learned nothing. **That is target leakage.** Drop both, or model them
   separately as a "Further" extension.
2. `temp`, `atemp`, `hum`, `windspeed` are already **min-max normalised** by the
   donors, with the constants given on the dataset page. Do not present them as
   degrees Celsius without undoing that. Say in your report which you did.

---

## 3. What you must do

### Phase A — set up (Unit IV, before Checkpoint 1)

1. Create the repository per handbook §3, add everyone, commit the raw CSVs.
2. `01-eda.ipynb`: load `hour.csv`, confirm the shape, check for missing values
   and for **missing hours** (the file has gaps — hours with zero rentals are
   absent rows, not zero rows; find out how many and decide what to do).
3. Plot: mean `cnt` by hour, split by `workingday`. You should see two clearly
   different shapes — commuter peaks on working days, a midday hump at weekends.
   That plot is the reason the project is interesting; put it in the report.
4. Plot `cnt` against `dteday` for the whole two years. Note the strong yearly
   growth (`yr`) and the seasonal cycle.

### Phase B — Checkpoint 1 (end of Unit V): data loaded, one baseline number

5. Build the **chronological split**: train on the earlier period, test on the
   later one. A defensible choice is train = 2011 + Jan–Jun 2012, test = Jul–Dec
   2012, but any documented time-ordered split is fine. **Do not use a random
   `train_test_split`** as your headline protocol — explain in the report why not.
6. Fit an ordinary least-squares linear regression on a minimal feature set and
   report RMSE and R² on the test period. That is your baseline number.
7. Tag `checkpoint-1` and push.

### Phase C — features and models (Unit V–VI, to Checkpoint 2)

8. `02-preprocessing.ipynb`. Feature engineering, each choice justified in a
   markdown cell:
   - **Cyclic encoding** of `hr` and `mnth`: sin/cos pairs, so that hour 23 is
     adjacent to hour 0. Show that this beats treating `hr` as an integer, and
     also compare against one-hot encoding of the 24 hours.
   - One-hot or ordinal encoding for `season`, `weathersit`, `weekday` — decide
     and justify each.
   - Scaling for the models that need it, fitted **on the training split only**
     and applied to the test split. Wrap this in a `Pipeline`; a scaler fitted on
     the full data before splitting is leakage and is marked as such.
9. `03-models.ipynb`. At minimum:
   - linear regression (the baseline, now with the engineered features);
   - **ridge** and **lasso**, with the penalty chosen by cross-validation
     *inside* the training period (`TimeSeriesSplit`), not on the test period;
   - one **non-linear** model — random forest, gradient boosting, or an MLP.
10. Tag `checkpoint-2` and push.

### Phase D — evaluation and writing (Unit VII, to final submission)

11. `04-evaluation.ipynb`:
    - one table, every model × {RMSE, MAE, R²} on the same test period, also
      written to `results/metrics.csv`;
    - a residual plot against hour and against predicted value — the linear model
      will visibly under-predict the peaks; say so;
    - **RMSE vs R²**: pick the one you would report to the operator and defend it
      in two or three sentences. Marks are for the argument, not the choice.
    - **Coefficient interpretation.** Take the ridge model and state, in plain
      words, what three of its coefficients mean for the operator. Include the
      sign check: does bad weather reduce demand in your model?
12. Report (4–6 pages), README, REPRODUCIBILITY.md. Final push.

---

## 4. Definition of done — the checklist I mark against

- [ ] Raw data committed; `cnt`/`casual`/`registered` leakage explicitly handled.
- [ ] Cyclic encoding of hour and month, compared against at least one alternative.
- [ ] A **chronological** train/test split, with the choice defended.
- [ ] Preprocessing inside a `Pipeline`, fitted on training data only.
- [ ] Linear regression, ridge and lasso (penalty cross-validated), plus one
      non-linear model.
- [ ] RMSE, MAE and R² for every model on the same held-out period, in
      `results/metrics.csv`.
- [ ] A justified choice between RMSE and R² as the headline metric.
- [ ] Coefficient interpretation in plain language, with a sign sanity-check.
- [ ] Residual analysis with at least one plot.
- [ ] Report, README, REPRODUCIBILITY.md, and a repository that runs from clean.

## 5. Going further (not required; makes the quiz easy)

- Implement **batch, mini-batch and stochastic gradient descent by hand** (Unit
  III) for the linear model, and compare the parameters and the runtime against
  the closed-form normal-equation solution. Plot the loss curves. This is the
  strongest extension available on this theme.
- Model `casual` and `registered` separately and sum them; does it beat modelling
  `cnt` directly, and why might it?
- Turn it into classification: define "peak hour" as `cnt` above the 90th
  percentile and build a classifier, reporting ROC/AUC (Unit VI).
- Compare against a persistence baseline: "next hour = same hour last week".
  If your model does not beat it, that is a finding worth reporting honestly.

## 6. Pitfalls specific to this theme

| Pitfall | Consequence |
|---|---|
| `casual` + `registered` left in the features | R² ≈ 1.0, project is void — the biggest mark-loser here |
| random split instead of chronological | inflated scores; method-correctness marks lost |
| scaler fitted before the split | leakage |
| treating `hr` as a plain integer only | model cannot represent the two daily peaks |
| forgetting `yr` | the 2012 growth is attributed to the wrong variables |
| reporting R² alone with no error in bikes | the operator cannot use it |

## 7. What the written quiz will probe

Not released in advance — but it is drawn from this territory: why a chronological
split, what exactly cyclic encoding does to hour 23, what your ridge coefficient
on `weathersit` means to a person with a truck, which line of your pipeline would
break first if the operator sent you a new city's data, and what your residual
plot at 8 a.m. is telling you.

## 8. Deliverables

Per handbook §3.4 and §3.5 — repository with README, requirements.txt, four
notebooks, `src/`, `results/metrics.csv`, `report/report.pdf` (4–6 pages) and
REPRODUCIBILITY.md. You submit the repository URL only.
