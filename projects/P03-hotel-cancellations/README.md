# P03 · Hotel-booking cancellations and the overbooking decision

> **Question.** At the moment a reservation is made, which bookings will be
> cancelled — and at what predicted probability should the hotel overbook the room?

Read `../Project-Handbook.md` first — repository rules, commit history, milestones,
group rubric, written quiz. This file defines only the work.

**Units:** IV, V (logistic regression), VI · **COs:** CO2, CO3, CO4  
**Difficulty:** middle of the band. The modelling is standard; the marks are in
the leakage audit and the threshold argument.

---

## 1. The situation

Roughly 37 % of the bookings in this dataset were cancelled. A hotel that ignores
that runs empty rooms; a hotel that overbooks blindly has to walk guests to
another property, which is expensive and public. Between those two failures sits a
decision: accept an extra booking when the predicted cancellation probability of
the existing set is high enough.

So this project does not end at "AUC 0.89". It ends at a **number** — the
probability threshold you recommend — and the cost reasoning that produced it.
That step is what separates a machine-learning exercise from a decision.

---

## 2. The data

**Hotel Booking Demand** — Antonio, de Almeida & Nunes (2019), two Portuguese
hotels, bookings due to arrive between July 2015 and August 2017.

- Page: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Free Kaggle account; download `hotel_bookings.csv` in a **browser** (~16 MB).
- 119 390 rows × 32 columns. Commit it to `data/raw/`.

**Target:** `is_canceled` (0/1).

**The leakage audit is the first graded task.** Several columns are recorded
*after* the booking's fate is known and cannot exist at prediction time. At
minimum, examine and decide on:

| Column | Why it is suspect |
|---|---|
| `reservation_status` | literally "Canceled" / "Check-Out" / "No-Show" — it *is* the label |
| `reservation_status_date` | the date the status was set |
| `assigned_room_type` | assigned at check-in, not at booking |
| `days_in_waiting_list` | resolved over time after booking |
| `booking_changes` | accumulates after booking |

Your README must contain a short table listing every column you dropped and the
one-line reason. A model that keeps `reservation_status` will score ~100 % and
score near zero from me.

**Other known quirks.** `children` has 4 missing values; `country` has ~488;
`agent` and `company` use missing to mean "none", not "unknown" — treat them
accordingly and say so. There are rows with `adults = children = babies = 0`
(bookings for nobody) and extreme `adr` outliers, including one above 5000.

---

## 3. What you must do

### Phase A — set up (Unit IV, before Checkpoint 1)

1. Repository per handbook §3; commit `hotel_bookings.csv`.
2. `01-eda.ipynb`: class balance overall and per hotel; cancellation rate against
   `lead_time`, `deposit_type`, `market_segment`, `previous_cancellations`. The
   `deposit_type == "Non Refund"` group is startling — investigate it and comment.
3. **The leakage audit**, written up as a table in the notebook and the README.
4. Cleaning: the missing values above, the zero-guest rows, the `adr` outliers.
   Every decision gets one line of justification.

### Phase B — Checkpoint 1 (end of Unit V)

5. A **stratified** train/test split (this is not a time series unless you choose
   to make it one — if you split by arrival date instead, say why; both are
   defensible, but pick one and be consistent).
6. Encode the categoricals (one-hot for low-cardinality; decide something explicit
   for `country`, which has ~178 levels — grouping the long tail into "Other" is
   an acceptable, documented choice).
7. Fit **logistic regression** in a `Pipeline` and report accuracy and ROC-AUC.
   That is your baseline number. Tag `checkpoint-1`, push.

### Phase C — models (Unit VI, to Checkpoint 2)

8. `03-models.ipynb`. All five, under one identical protocol:
   - logistic regression (interpretable baseline),
   - K-nearest neighbours (scaling matters — show that you know why),
   - a single decision tree (and its depth, chosen by cross-validation),
   - random forest,
   - one boosting model (gradient boosting / HistGradientBoosting).
   Cross-validate on the training set; the test set is touched once, at the end.
9. Tag `checkpoint-2`, push.

### Phase D — evaluation and the decision (Unit VI–VII)

10. `04-evaluation.ipynb`:
    - one comparison table: accuracy, precision, recall, F1, **ROC-AUC** and
      **average precision** (PR-AUC) for all five models → `results/metrics.csv`;
    - ROC curves on one axis; **precision–recall curves** on another. Explain in
      the report which of the two is more informative here and why;
    - confusion matrices at the default 0.5 threshold;
    - **the threshold decision.** State a cost for a false positive (you predicted
      cancellation, guest arrived → a walked guest) and for a false negative (you
      predicted arrival, guest cancelled → an empty room). You may choose the
      numbers; you must justify them. Sweep the threshold, plot expected cost
      against threshold, and **recommend one value**. Report the confusion matrix
      at *your* threshold, not at 0.5.
    - feature importance from the tree ensemble against the logistic
      coefficients — do they agree? Where they disagree, say what that means.
11. Report, README, REPRODUCIBILITY.md, final push.

---

## 4. Definition of done

- [ ] Leakage audit: every dropped column listed with a reason, in the README.
- [ ] Documented handling of missing values, zero-guest rows and `adr` outliers.
- [ ] Categorical encoding with an explicit policy for high-cardinality `country`.
- [ ] Stratified (or date-based, justified) split; preprocessing inside a
      `Pipeline` fitted on training data only.
- [ ] Five model families compared under one protocol.
- [ ] ROC-AUC **and** precision–recall/average precision reported for each.
- [ ] A stated cost model, a threshold sweep, and one recommended threshold, with
      the confusion matrix at that threshold.
- [ ] Feature importances compared against logistic coefficients.
- [ ] Report, README, REPRODUCIBILITY.md; runs from a clean checkout.

## 5. Going further

- Class weights vs random oversampling vs SMOTE vs undersampling — four rows in
  one table, same model, same split. Which actually moved recall, and at what
  cost to precision?
- A **calibration curve**: are your predicted probabilities honest? A threshold
  argument is only as good as the calibration underneath it. Compare the ensemble
  before and after `CalibratedClassifierCV`.
- Split the analysis by hotel (City vs Resort). Does one model serve both?
- Train on 2015–2016 and test on 2017 — does performance survive the time shift?

## 6. Pitfalls specific to this theme

| Pitfall | Consequence |
|---|---|
| `reservation_status` kept as a feature | ~100 % accuracy; the project is void |
| one-hot encoding `country` blindly | ~178 columns of mostly noise |
| encoder fitted before the split | leakage; unseen categories at test time crash |
| KNN without scaling | `lead_time` in days swamps every binary feature |
| stopping at 0.5 and calling it done | the decision — the point of the project — is missing |
| accuracy as headline on a 63/37 split | a model predicting "never cancels" scores 63 % |

## 7. What the written quiz will probe

Territory: which columns you dropped and why each one leaks; what your threshold
is and what cost numbers produced it; what happens to precision and recall as you
move it; why PR-AUC or ROC-AUC here; what your KNN would do without the scaler;
which feature the random forest ranks first and whether the logistic model agrees.

## 8. Deliverables

Per handbook §3.4–§3.6. You submit the repository URL only.
