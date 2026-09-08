# P08 · Household electricity — forecasting the next hour, and the kinds of day

> **Question.** Can the next hour's household power draw be forecast from recent
> history — and what typical "kinds of day" does this household have?

Read `../Project-Handbook.md` first — repository rules, commit history, milestones,
group rubric, written quiz. This file defines only the work.

**Units:** IV, V, VII (+ VI for ensembles) · **COs:** CO2, CO3, CO4  
**Difficulty:** heavy on data handling, alongside P07. Two million rows will
punish careless pandas. In exchange the modelling list is short. Start early.

---

## 1. The situation

One French household, one reading a minute, for nearly four years. Two useful
things can be asked of it, and this project does both, which is why it spans
supervised and unsupervised learning.

**Forecasting.** A smart-meter service wants the next hour's consumption — for
demand response, for a bill estimate, for spotting a fault. The bar to clear is
*persistence*: "the next hour will be like this hour". On smooth household data
that naïve rule is strong, and beating it convincingly is the actual challenge.

**Profiling.** Reshape each day into 24 hourly values and treat it as a point in
24 dimensions. Cluster those points and you get the household's repertoire of
days — the working day, the weekend, the away-on-holiday day, the cold winter day.
Nobody labelled these. The clustering finds them, and you have to decide whether
what it found is real.

---

## 2. The data

**UCI — Individual Household Electric Power Consumption** (Hebrail & Berard).

- Page: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- Download the **zip** (19.7 MB) in a browser; it expands to
  `household_power_consumption.txt`, 127 MB. Licence CC BY 4.0.
- 2 075 259 rows, Dec 2006 – Nov 2010, one row per minute.
- **Commit the 20 MB zip, gitignore the 127 MB txt**, and have your script expand
  it. Document this in the README.

**Columns:** `Date` (dd/mm/yyyy), `Time` (hh:mm:ss), `Global_active_power` (kW),
`Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1` (kitchen),
`Sub_metering_2` (laundry), `Sub_metering_3` (water heater + AC).

**Loading it correctly is the first graded step.** The separator is `;`, missing
values are the literal string `?`, and every numeric column will otherwise be read
as text. Read it with the right `sep`, `na_values`, `dtype` and a parsed datetime
index, and say in the report how long that took and how much memory it used.

**Known facts you should confirm rather than assume:** about 1.25 % of rows have
missing measurements. Every calendar timestamp is present as a *row*, but the
measurements on some of them are blank — the donors document a stretch of these
around 28 April 2007. Quantify the missingness yourself before deciding what to do.

**Note the units trap.** Sub-metering is in watt-hours per minute; global active
power is in kilowatts. The dataset page gives the relation for the unmetered
remainder — if you compute a "rest of house" series, get the arithmetic right and
show it.

---

## 3. What you must do

### Phase A — loading and resampling (Unit IV, before Checkpoint 1)

1. Repository per handbook §3; commit the zip; loader script in `src/`.
2. Load correctly (see above), build a `DatetimeIndex`, and **resample from
   minutes to hours** — mean or sum, decide and justify. State how you treated
   hours that are partly missing (a `min_count` rule, or dropping them).
3. `01-eda.ipynb`: missingness by month; the mean daily profile; mean consumption
   by hour split by weekday/weekend; the annual cycle. The winter/summer
   difference and the evening peak should both be plain.

### Phase B — Checkpoint 1 (end of Unit V)

4. **Persistence baseline** for the hourly series: prediction for t+1 is the value
   at t. Report MAE and RMSE on a held-out final period. Also compute the
   "same hour yesterday" and "same hour last week" baselines — on this data they
   are strong, and you need to know that before you claim anything.
5. Tag `checkpoint-1`, push.

### Phase C — forecasting (Unit V–VI, to Checkpoint 2)

6. `02-preprocessing.ipynb` — features for predicting `Global_active_power` at
   t+1, all **strictly causal**:
   - lags at t, t−1, t−2, t−3, t−24, t−168;
   - rolling mean/std over 3, 24 and 168 hours, **each window ending at t**;
   - calendar: hour (cyclic), day of week, month, a holiday or weekend flag.
   Prove causality on one worked row in a markdown cell.
7. **Chronological split** — train on 2007–2009, test on 2010 (or similar,
   documented). Never random.
8. `03-models.ipynb`:
   - linear regression, then **ridge and lasso** with the penalty chosen by
     `TimeSeriesSplit` inside the training period;
   - one tree ensemble (random forest or gradient boosting);
   - all judged against the three baselines from Phase B, on MAE and RMSE.
9. Tag `checkpoint-2`, push.

### Phase D — day-profile clustering and writing (Unit VII)

10. `04-clustering.ipynb`:
    - build a **day × 24** matrix — roughly 1 400 rows, one per day. Drop or
      impute days with too many missing hours (state the rule);
    - decide whether to normalise each day to its own total. Clustering raw
      profiles groups by *how much* electricity was used; clustering normalised
      profiles groups by *shape*. Do at least one, justify it, and ideally show
      how the answer differs;
    - **k-means**: elbow and silhouette over k = 2…10, choose k, defend it;
    - **hierarchical clustering** with a dendrogram, at least two linkages;
    - **plot the cluster centroids as 24-hour curves** — this is the money figure
      of the project. Then interpret them: cross-tabulate cluster against day of
      week and against month, and name the clusters (weekday, weekend, away,
      winter-evening) from that evidence, not from imagination.
11. `05-evaluation.ipynb`: every forecasting number in one table →
    `results/metrics.csv`; predicted-vs-actual over a sample fortnight; an honest
    statement of whether you beat "same hour last week" and where the model fails
    (it will fail on the sharp evening ramp and on unusual days).
12. Report, README, REPRODUCIBILITY.md, final push.

---

## 4. Definition of done

- [ ] Correct load (`sep=';'`, `na_values='?'`, parsed datetime index), with the
      cost in time and memory reported.
- [ ] Missingness quantified; hourly resampling with a stated rule for partial hours.
- [ ] **Three baselines**: persistence, same-hour-yesterday, same-hour-last-week.
- [ ] Causal lag and rolling features, verified on one worked row.
- [ ] Chronological split, defended.
- [ ] Linear, ridge, lasso (CV'd penalty) and one ensemble; MAE and RMSE for all,
      alongside the baselines.
- [ ] Day × 24 profile matrix, with a stated normalisation decision.
- [ ] k-means with elbow and silhouette; hierarchical with a dendrogram and ≥2
      linkages.
- [ ] **Centroid curves plotted and interpreted**, cross-tabulated against weekday
      and month, clusters named from that evidence.
- [ ] Report, README, REPRODUCIBILITY.md; runs from a clean checkout.

## 5. Going further

- Forecast the three **sub-meterings** separately. Which room is predictable, and
  which is not? (The water heater is nearly deterministic; the kitchen is not.)
- **DBSCAN on the day profiles** to find anomalous days, then go and look at what
  happened on those dates. Holidays and outages show up.
- An **MLP** forecaster (Unit VI) against the linear models — does non-linearity
  pay on a smooth series?
- Horizon curve: t+1 vs t+6 vs t+24, error plotted against horizon.
- Cluster the days, then use the cluster as a feature in the forecaster. Does
  knowing "what kind of day it is" help predict the next hour?

## 6. Pitfalls specific to this theme

| Pitfall | Consequence |
|---|---|
| `pd.read_csv` with defaults | everything is a string; `?` is a category |
| resampling with `.mean()` while ignoring missing minutes | silently biased hours |
| a rolling window centred on t or including t+1 | look-ahead leakage; the results are fiction |
| random split | inflated scores on an autocorrelated series |
| no "same hour last week" baseline | you cannot tell whether your model added anything |
| clustering raw profiles without saying so | clusters are just "high day / low day" |
| committing the 127 MB txt | breaks the handbook's repo rules |
| loading all 2 M rows into memory repeatedly in every notebook | hours wasted — cache the hourly series once |

## 7. What the written quiz will probe

Territory: how you read the file and what `?` did before you handled it; how you
proved your rolling window does not see the future; which of the three baselines
was hardest to beat and by how much; what one of your centroid curves says about
the household; whether you normalised the day profiles and what that changed;
where in the day your forecaster is worst and why.

## 8. Deliverables

Per handbook §3.4–§3.6, except that this theme uses five notebooks —
`01-eda`, `02-preprocessing`, `03-models`, `04-clustering`, `05-evaluation`.
You submit the repository URL only.
