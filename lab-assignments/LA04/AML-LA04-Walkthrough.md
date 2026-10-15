# Lab Assignment 4 — worked walkthrough

**▶ Watch:
[`AML-LA04-Walkthrough.mp4`](AML-LA04-Walkthrough.mp4)**
— 10 min, 1080p. Click the file, then Download or play it in the browser.

Classification on Telco Customer Churn. Not whether you can fit a classifier —
whether you can tell a good accuracy from a useful model.

> Numbers below come from `scikit-learn` with `random_state=0` and a stratified
> split. Your last digit may differ on a different build.

---

## The setup every question builds on

```python
tel = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
tel['TotalCharges'] = pd.to_numeric(tel.TotalCharges, errors='coerce').fillna(0)
#  ^^ the 11 blank cells from LA01. Fix them here or everything downstream breaks.

y = (tel.Churn == 'Yes').astype(int)
X = tel.drop(columns=['customerID', 'Churn'])        # customerID is an identifier

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)
# 5,634 train rows, 1,409 test rows (374 churners, 1,035 stayers)
```

**Drop `customerID`.** It is a unique string — one-hot encoding it creates 7,043
columns, each true for exactly one customer, and the model memorises the training
set. After encoding the real columns you have **40 features** from 4 numeric and
15 categorical ones.

`stratify=y` is not optional with an imbalanced target. Without it the test set's
churn rate drifts and every metric moves with it.

---

## A1. Encode and fit

```python
num_cols = X.select_dtypes(include='number').columns      # 4
cat_cols = X.select_dtypes(exclude='number').columns      # 15

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), cat_cols),
])

lr = Pipeline([('pre', pre),
               ('clf', LogisticRegression(max_iter=2000, random_state=0))]).fit(X_tr, y_tr)
lr.score(X_te, y_te)        # 0.8020
```

`handle_unknown='ignore'` stops a category seen only in the test set from raising
at predict time.

**Model observation.** Test accuracy 0.8020 on 1,409 held-out customers. The churn
rate is **26.5%**, so predicting "stays" for everyone scores **73.5%**. The model
is therefore 6.7 percentage points better than a constant answer — real, but far
less impressive than 80% sounds. With an imbalanced target, accuracy is anchored
near the majority-class rate and most of it is earned before the model does
anything. This is why the rest of the assignment is about the confusion matrix.

> Reporting 80% without the 73.5% beside it is the most common way to misread
> this dataset.

## A2. Confusion matrix

```python
cm = confusion_matrix(y_te, pred)       # rows = actual, columns = predicted
tn, fp, fn, tp = cm.ravel()
print('missed churners:', fn)
```

```
[[924 111]
 [168 206]]        missed churners: 168
```

**Model observation.** 924 stayers correctly left alone, 111 false alarms, **168
churners missed**, 206 caught. Precision on churn 0.650, recall **0.551** — the
model finds a little over half of the people about to leave, and the 80% headline
hides it completely because the 924 correct stayers dominate the average.

A missed churner costs far more than an unnecessary offer. Losing a customer
forfeits the remaining lifetime value of the contract — a median monthly charge
around $70, against an acquisition cost that industry-wide runs to several months
of revenue. An unnecessary offer costs only the discount, handed to someone
already profitable who now feels valued. The two errors are not symmetric, so a
metric that treats them as equal — accuracy — is the wrong thing to optimise.

## A3. Threshold moving

No retraining. The model is already finished; only the cutoff changes.

```python
proba = lr.predict_proba(X_te)[:, 1]          # column 1 = P(churn)
for t in (0.5, 0.35, 0.25):
    pred_t = (proba >= t).astype(int)
```

| threshold | precision | recall | missed | false alarms | flagged |
|---|---|---|---|---|---|
| 0.50 | 0.650 | 0.551 | 168 | 111 | 317 |
| 0.35 | 0.569 | 0.717 | 106 | 203 | 471 |
| 0.25 | 0.513 | 0.829 | 64 | 294 | 604 |

**Model observation.** Lowering the threshold flags more customers, so it catches
more real churners and is wrong more often about the ones it flags.

I would ship **0.35**. Going from 0.50 to 0.35 converts 62 missed churners into
caught ones for 92 extra offers, and roughly four in seven flagged customers are
genuine. Going on to 0.25 buys 42 more for 91 further offers, and precision drops
below half — the campaign becomes mostly discount to people who were never
leaving. The right answer depends on the offer's cost relative to a lost contract:
if the offer is nearly free, 0.25 wins; if it is a month of revenue, 0.50 is
defensible. **Stating that dependency is the answer.**

---

## B1. Model comparison

| model | acc | prec | recall | F1 | AUC | missed |
|---|---|---|---|---|---|---|
| logistic regression | 0.802 | 0.650 | 0.551 | 0.596 | **0.852** | 168 |
| decision tree (d=5) | 0.793 | 0.609 | 0.618 | 0.614 | 0.837 | 143 |
| random forest | 0.786 | 0.634 | 0.463 | 0.535 | 0.826 | 201 |

ROC-AUC takes the **probability** column. Passing hard 0/1 labels is a silent
error — it still returns a number, just a meaningless one.

**Model observation.** I would ship the logistic regression. It has the best
ROC-AUC, which measures ranking quality independently of any threshold — and since
A3 showed the threshold is the real decision, ranking is what the model must do
well. The decision tree's higher F1 is an artefact of where its default cutoff
happens to fall, not of better discrimination.

The random forest is the worst choice despite being the most powerful: recall
0.463 means it misses 201 of 374 churners, and the cost argument says missed
churners are the expensive error. It is also the hardest to explain to the
retention team, while logistic regression hands them coefficients. All three sit
within 0.03 on AUC, so **model choice matters far less here than threshold
choice**.

## B2. Class imbalance

```python
# route 1 — reweight the loss
LogisticRegression(max_iter=2000, random_state=0, class_weight='balanced')

# route 2 — duplicate minority rows until the classes are level
need = (y_tr == 0).sum() - (y_tr == 1).sum()          # 4139 - 1495 = 2644
idx  = rng.choice(X_tr[y_tr == 1].index, size=need, replace=True)
# training balance: 4,139 / 1,495  ->  4,139 / 4,139
```

Resample the **training set only**. Oversampling before the split puts copies of
the same customer on both sides and your test score becomes fiction.

| treatment | accuracy | recall | F1 | missed churners |
|---|---|---|---|---|
| baseline | 0.802 | 0.551 | 0.596 | 168 |
| `class_weight='balanced'` | 0.752 | 0.807 | 0.634 | **72** |
| random oversampling | 0.752 | 0.794 | 0.629 | 77 |

**Model observation.** Both treatments make the same trade: they tell the model a
churner is worth about 2.8 times a stayer, so it moves its boundary towards
predicting churn. Recall jumps 25 points and accuracy drops 5, because the model
accepts far more false alarms to catch cases it previously ignored.

They land within half a point of each other, which is expected rather than
surprising: for logistic regression, duplicating a row 2.8 times and multiplying
its loss by 2.8 are almost the same operation. Oversampling costs more memory and
a longer fit for no measurable gain.

It was worth it — 96 extra churners caught for a 5-point drop in a quantity we
never cared about. Note this is **the same trade A3 bought by moving the
threshold**: reweighting is threshold moving done during training rather than
after.

---

## C1. Which features carry the signal — compulsory

```python
names = lr[:-1].get_feature_names_out()      # 40 names, post one-hot
coefs = pd.Series(lr[-1].coef_[0], index=names)
coefs.sort_values(key=abs, ascending=False).head(10)
```

```
tenure                      -1.399      InternetService_Fiber optic +0.430
Contract_Two year           -0.804      MonthlyCharges              -0.343
TotalCharges                +0.704      PaperlessBilling_Yes        +0.339
Contract_Month-to-month     +0.565      OnlineSecurity_Yes          -0.274
InternetService_DSL         -0.562      TechSupport_No internet svc -0.271
```

Coefficients are comparable **only** because the numerics were standardised inside
the pipeline.

**The detail worth catching:** `MonthlyCharges` has a *negative* coefficient here,
but in LA01 its raw correlation with churn was +0.193. The sign flipped because
`TotalCharges` and `tenure` are in the model too, and `TotalCharges` is almost
exactly `tenure × MonthlyCharges` (correlation 0.999). With both present the model
can satisfy the data with large opposing coefficients, and each stops meaning what
its name suggests.

**Model observation.** All ten features are recorded in the billing system and
available when a score is needed, so none is leakage in the usual sense. But two
caveats matter. First, `tenure` and `TotalCharges` are zero for a brand-new
customer, so this model scores **existing** customers for retention and cannot be
reused to rank acquisition risk, where its strongest feature carries no
information. Second, if scoring runs mid-cycle, `TotalCharges` for the current
month is not final, so the value at prediction time differs from the one trained
on. Given it is also collinear enough to flip the sign on `MonthlyCharges`, I
would drop `TotalCharges` — it costs almost nothing in accuracy and makes the
remaining coefficients readable by the retention team.

---

## Five common mistakes

- Reporting accuracy without the 73.5% majority baseline beside it
- One-hot encoding `customerID`, or fitting the encoder before the split
- Reading the confusion matrix transposed, reporting false alarms as misses
- Passing hard labels to `roc_auc_score` instead of `predict_proba`
- Oversampling before the split, so duplicated customers appear in both halves

## Before you submit

- [ ] Restart Kernel → **Run All**, top to bottom, no errors
- [ ] Every metric reported for the churn class, not just the overall average
- [ ] The majority-class baseline stated once, early
- [ ] Filename `AML_LA04_<yourSAPID>.ipynb`, `README.md` beside it
- [ ] Every attempted question has a written observation

The model gives you a ranking; you choose the cutoff. The cutoff is a business
decision, not a default.
