# P02 · Next-day air quality for an Indian city

> **Question.** Can yesterday's pollutant readings and the calendar predict
> tomorrow's PM2.5, and tomorrow's AQI category, well enough to issue a public
> health warning a day in advance?

Read `../Project-Handbook.md` first — repository rules, commit history, milestones,
group rubric, written quiz. This file defines only the work.

**Units:** IV, V, VI · **COs:** CO2, CO3, CO4  
**Difficulty:** middle of the band. The data is messy on purpose; that is the
lesson.

---

## 1. The situation

A city pollution board can issue an advisory — schools shut, construction paused —
but only if it has a day's notice. A same-day AQI reading is a record, not a
warning. You are building the warning: a regression model for tomorrow's PM2.5
concentration and a classifier for tomorrow's AQI bucket.

The honest question underneath is whether you beat *persistence*: "tomorrow will
be like today". Air quality is highly autocorrelated, so that naïve rule is
already good. A model that does not beat it has produced nothing, and saying so
clearly earns more marks than hiding it.

---

## 2. The data

**Air Quality Data in India (2015–2020)** — CPCB station data compiled by Rohan Rao.

- Page: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
- Requires a free Kaggle account; download in a **browser**.
- Use **`city_day.csv`** (~29 500 rows, 26 cities, one row per city per day).
  `station_day.csv` is available if you want station-level work.
- Commit `city_day.csv` to `data/raw/` (a few MB).

**Columns:** `City`, `Date`, `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`,
`SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`, `AQI`, `AQI_Bucket`.

**Scope.** Pick **one or two cities** and say why. Delhi has the most complete
record and the widest range; a second city (Bengaluru, Ahmedabad, Kolkata) makes
the generalisation question available. Do not silently model all 26 as one pool.

**Three traps.**

1. **Missing data is everywhere** — some pollutants are absent for whole years at
   some cities, and `AQI`/`AQI_Bucket` are themselves missing on many rows. Your
   missing-data strategy is a graded part of this project, not a preliminary.
   Quantify it (a missingness table per column per city) before you decide.
2. **`AQI` is computed from the pollutants**, including PM2.5, by a published CPCB
   formula. Using today's `AQI` to predict today's PM2.5 is circular. Using
   *yesterday's* AQI to predict *tomorrow's* PM2.5 is legitimate — be explicit
   about which you are doing.
3. **Dates have gaps.** Reindex to a continuous daily index before you build lags,
   or your "yesterday" will silently be "some earlier day".

---

## 3. What you must do

### Phase A — set up (Unit IV, before Checkpoint 1)

1. Repository per handbook §3; commit `city_day.csv`.
2. `01-eda.ipynb`: parse `Date`, filter to your city/cities, reindex to a complete
   daily calendar, and produce a **missingness table** — percentage missing per
   column per year. Plot PM2.5 over time; the winter spikes and the 2020 lockdown
   dip should both be visible, and both are worth a sentence in the report.
3. Decide and document the missing-data policy: drop the column, drop the row,
   forward-fill (defensible for short gaps in a daily series), or interpolate.
   Different answers for different columns are fine if justified.

### Phase B — Checkpoint 1 (end of Unit V)

4. Build the **persistence baseline**: predict tomorrow's PM2.5 as today's PM2.5.
   Report its MAE and RMSE. Every later model is judged against this.
5. A first linear regression on a handful of lag features, evaluated on a
   **time-ordered** split (train on the earlier years, test on the latest year).
6. Tag `checkpoint-1`, push.

### Phase C — features and models (Unit V–VI, to Checkpoint 2)

7. `02-preprocessing.ipynb` — build, **with no look-ahead**:
   - lags of PM2.5 and the main pollutants at t−1, t−2, t−3, t−7;
   - rolling mean and standard deviation over 3 and 7 days, computed so that the
     window ends at t−1 (a centred or t-inclusive window is leakage);
   - calendar features: month, day of week, and a cyclic day-of-year encoding;
   - the target: `PM2.5` at t+1 (regression) and `AQI_Bucket` at t+1
     (classification).
   State clearly in a markdown cell why `.shift()` direction matters and check one
   row by hand to prove you got it right.
8. `03-models.ipynb`:
   - **Regression:** linear → ridge → lasso, penalty chosen by `TimeSeriesSplit`
     cross-validation inside the training period; then one non-linear model.
   - **Classification:** the AQI bucket for t+1. Logistic regression plus at
     least two of {decision tree, random forest, naïve Bayes, SVM, KNN}.
     The buckets are strongly imbalanced — "Severe" is rare — so decide how you
     handle that and say so.
9. Tag `checkpoint-2`, push.

### Phase D — evaluation and writing (Unit VII)

10. `04-evaluation.ipynb`:
    - regression: MAE, RMSE and R² for every model **and the persistence
      baseline**, one table, in `results/metrics.csv`;
    - classification: confusion matrix, per-class precision/recall, macro-F1, and
      **ROC/AUC** (one-vs-rest for the multiclass case) — and an explanation of
      why plain accuracy is misleading when one class dominates;
    - a plot of predicted vs actual PM2.5 over the test year;
    - an honest paragraph: did you beat persistence, by how much, and on which
      days does the model fail? (It will fail on the sharp winter onsets. That is
      the most interesting result in the project.)
11. Report, README, REPRODUCIBILITY.md, final push.

---

## 4. Definition of done

- [ ] One or two cities, chosen and justified; continuous daily index.
- [ ] Missingness quantified per column per year, and a documented policy.
- [ ] Lag and rolling features with **no look-ahead**, verified on one row by hand.
- [ ] A **time-ordered** split, defended in the report.
- [ ] Persistence baseline computed and reported alongside every model.
- [ ] Regression: linear, ridge, lasso (CV'd penalty) — MAE, RMSE, R².
- [ ] Classification of the t+1 AQI bucket: ≥3 model families, confusion matrix,
      per-class recall, macro-F1, ROC/AUC.
- [ ] Class imbalance addressed explicitly.
- [ ] Report, README, REPRODUCIBILITY.md; runs from a clean checkout.

## 5. Going further

- One model with `City` as a feature vs one model per city — which generalises to
  a city held out entirely?
- Festival and stubble-burning effects: a Diwali indicator, an October–November
  indicator. Does an explicit feature beat the month dummy?
- Resampling (SMOTE, undersampling) vs class weights for the "Severe" class, and
  the effect on recall for that class specifically.
- Forecast horizon: t+1 vs t+3 vs t+7. Plot error against horizon.

## 6. Pitfalls specific to this theme

| Pitfall | Consequence |
|---|---|
| rolling window that includes day t | look-ahead leakage; results are fiction |
| `.shift(-1)` vs `.shift(1)` confusion | you predict yesterday; scores look superb |
| using same-day `AQI` to predict same-day PM2.5 | circular target |
| dropping every row with any NaN | you may delete 80 % of the data — count first |
| random split on a time series | inflated scores |
| accuracy on imbalanced buckets | 80 % accuracy that never predicts "Severe" |
| no persistence baseline | no way to tell if the model is worth anything |

## 7. What the written quiz will probe

Territory: which direction your `shift` goes and how you proved it; what your
missingness table said and what you did about it; whether your model beat
persistence and by how much; why macro-F1 rather than accuracy; what happens to
your model on a day the city has never seen before; which single feature you would
drop first and what you expect to happen.

## 8. Deliverables

Per handbook §3.4–§3.6. You submit the repository URL only.
