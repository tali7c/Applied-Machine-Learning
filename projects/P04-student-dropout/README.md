# P04 · Early warning of student dropout

> **Question.** From what a university knows at enrolment, and again after the
> first semester, which students are at risk of dropping out — and what would a
> *fair* early-warning rule look like?

Read `../Project-Handbook.md` first — repository rules, commit history, milestones,
group rubric, written quiz. This file defines only the work.

**Units:** IV, VI (+ Unit V logistic regression) · **COs:** CO1, CO2, CO3, CO4  
**Difficulty:** contained data, demanding argument. The modelling is small; the
fairness discussion is where this project is won or lost.

---

## 1. The situation

A polytechnic wants to reach students before they leave, not after. An early-
warning model is only useful if it fires early enough to matter and accurately
enough to be trusted — and only *acceptable* if the reasons it fires are things
the institution may legitimately act on.

So this project has two halves that carry equal weight in the report: a
technical half (a multiclass classifier on imbalanced data, evaluated properly)
and an ethical half (which features a university may use, and which it may not,
even when they improve the score).

You will find that features such as tuition-fee status and parental education
carry real predictive signal. Deciding what to do with that is the project.

---

## 2. The data

**UCI — Predict Students' Dropout and Academic Success** (Realinho et al., 2021).

- Page: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Download the **zip** (521 KB) in a browser. Licence CC BY 4.0.
- 4 424 students × 36 features + target. **No missing values** — the donors
  cleaned it, so your Unit IV work here is encoding and scaling, not repair.
- Commit `data.csv` to `data/raw/`.

**Target:** three classes — `Dropout`, `Enrolled`, `Graduate`. The distribution is
imbalanced and `Enrolled` (still studying at the end of the normal duration) is
the awkward middle class that most models get wrong. Do not silently drop it.

**Feature groups you must distinguish**, because the whole project depends on it:

| Group | Examples | Known at |
|---|---|---|
| Demographic / social | marital status, nationality, parents' qualification and occupation, displaced, special needs, gender, age at enrolment | enrolment |
| Economic / administrative | scholarship holder, **tuition fees up to date**, debtor, unemployment rate, inflation, GDP | enrolment (and updated) |
| Academic path | application mode and order, course, previous qualification grade, admission grade | enrolment |
| Semester 1 performance | curricular units enrolled / evaluated / approved / grade (1st sem) | end of sem 1 |
| Semester 2 performance | the same, 2nd sem | end of sem 2 |

**Scope decision.** The second-semester columns describe a point very close to the
outcome. Your two headline models are:

- **Model A — enrolment only:** no semester columns at all. This is the model that
  could actually be deployed on day one.
- **Model B — plus semester 1:** adds the first-semester block.

Model B will beat Model A. The interesting number is *by how much*, and whether
the extra accuracy arrives early enough to be useful. Second-semester columns are
optional and, if used, must be reported as a separate "Model C" with the caveat
that it predicts an outcome that is nearly already decided.

---

## 3. What you must do

### Phase A — set up (Unit IV, before Checkpoint 1)

1. Repository per handbook §3; commit `data.csv`.
2. `01-eda.ipynb`: class distribution; dropout rate broken down by scholarship
   status, tuition-fee status, age band, and first-semester approval rate. Two or
   three of these will be striking — those plots belong in the report.
3. Build the feature-group lists (as Python lists in `src/features.py`, not by
   hand each time). Encode the coded integer columns properly: `Course`,
   `Application mode`, `Nacionality`, parents' qualification and occupation are
   **nominal codes, not magnitudes** — one-hot them or group the long tail; do not
   feed 9853 to a linear model as a number.

### Phase B — Checkpoint 1 (end of Unit V)

4. Stratified train/test split (stratify on the three-class target), plus stratified
   k-fold cross-validation on the training part.
5. Fit multinomial **logistic regression** on Model A's feature set. Report
   accuracy, macro-F1 and per-class recall. That is your baseline. Note how badly
   `Enrolled` does — that observation drives the rest of the project.
6. Tag `checkpoint-1`, push.

### Phase C — models (Unit VI, to Checkpoint 2)

7. `03-models.ipynb` — at least three families beyond the baseline, for **both**
   Model A and Model B:
   - naïve Bayes (Gaussian on the numeric block — and note what its independence
     assumption is doing to correlated semester columns),
   - a decision tree with cross-validated depth, and a random forest or boosting
     model,
   - SVM or a small MLP (scaling required — inside the `Pipeline`).
8. Handle the imbalance explicitly: `class_weight="balanced"` and at least one
   resampling approach, compared on the same split.
9. Tag `checkpoint-2`, push.

### Phase D — evaluation, fairness, writing (Unit VI–VII)

10. `04-evaluation.ipynb`:
    - one table per model set (A and B): accuracy, **macro-F1**, per-class
      precision/recall → `results/metrics.csv`;
    - confusion matrices, with a written reading of where `Enrolled` goes;
    - **why macro-F1 and per-class recall rather than accuracy** — argue it in
      the report; a model that never predicts `Enrolled` can still look good on
      accuracy, and for an early-warning system missing a real dropout (recall on
      `Dropout`) is the costly error;
    - one-vs-rest ROC/AUC for the three classes;
    - **A vs B**: how much does the first semester buy you, and is the day-one
      model good enough to act on?
    - **feature importance and the fairness discussion.** Rank the features. Then
      write the section that matters: which high-ranking features should a
      university *not* act on, and what happens to your metrics if you remove
      them? Run that model and report the cost of the ethical choice in numbers.
      Marks are for the reasoning, not for reaching any particular conclusion.
11. Report, README, REPRODUCIBILITY.md, final push.

---

## 4. Definition of done

- [ ] Coded categorical columns encoded as nominal, not as magnitudes.
- [ ] Feature groups defined in code; **Model A (enrolment only)** and
      **Model B (+ semester 1)** both built.
- [ ] Stratified split and stratified cross-validation.
- [ ] ≥3 model families beyond logistic regression, for both A and B.
- [ ] Imbalance handled by class weights **and** one resampling method, compared.
- [ ] Macro-F1 and per-class recall reported and defended over accuracy.
- [ ] Confusion matrices with a written reading of the `Enrolled` class.
- [ ] One-vs-rest ROC/AUC.
- [ ] Feature ranking **plus** a fairness section that removes the contested
      features and reports the measured cost.
- [ ] Report, README, REPRODUCIBILITY.md; runs from a clean checkout.

## 5. Going further

- Cost-sensitive thresholds: pick a cost for missing a dropout versus flagging a
  student who was fine, and tune the decision rule to that cost rather than to
  argmax probability.
- Collapse to two classes (Dropout vs Not) and check whether the model ranking
  changes. If a different model wins, that is a finding — explain it.
- Calibration: if the university wants "students above 70 % risk", the
  probabilities must mean something. Plot a calibration curve.
- Per-course models: does the best model differ between Nursing and Informatics?

## 6. Pitfalls specific to this theme

| Pitfall | Consequence |
|---|---|
| treating `Course`/`Nacionality` codes as numbers | the model learns a fictional ordering |
| accuracy as the headline | a model that never predicts `Enrolled` looks fine |
| second-semester features in the "early warning" model | the warning arrives too late to be one |
| resampling applied before the split | leakage; optimistic and meaningless scores |
| dropping `Enrolled` to make it binary without saying so | changes the problem silently |
| feature importance with no fairness discussion | the harder half of the project is missing |

## 7. What the written quiz will probe

Territory: why macro-F1 and not accuracy; where your confusion matrix puts the
`Enrolled` students and why; how much the first semester adds over enrolment-only
and whether that changes what the university should do; which feature you argued
against using and what it cost you in recall; what your naïve Bayes assumes that
is false in this data.

## 8. Deliverables

Per handbook §3.4–§3.6. You submit the repository URL only.
