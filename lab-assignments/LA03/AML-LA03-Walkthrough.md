# Lab Assignment 3 — worked walkthrough

**▶ Watch:
[`AML-LA03-Walkthrough.mp4`](AML-LA03-Walkthrough.mp4)**
— 9 min 20 s, 1080p. Click the file, then Download or play it in the browser.

Time-series regression, where every habit from LA02 becomes a mistake. Worked on
IBM daily closes, 2013–2016. Any ticker with three years of history behaves the
same way.

> This assignment is not testing whether you can fit a model to a stock. It is
> testing whether you can tell when a good-looking number is meaningless.

---

## Getting C1 — any single ticker, at least three years

```bash
pip install yfinance
python -c "import yfinance as y; y.download('IBM','2013-01-01').to_csv('data/ibm.csv')"
```

Or download any daily OHLCV CSV from Kaggle into `data/`. Only `Date` and `Close`
are needed for every question in this brief. Record the ticker, the date range
and the download date in your notebook.

**One correction to the instruction sheet:** it says to set `random_state` for
reproducibility. That applies to the *model* here, never to the split — in this
assignment the split is by date.

---

## A1. Load and describe the series `[2]`

```python
s = pd.read_csv('data/ibm.csv', parse_dates=['Date'])
s = s.sort_values('Date').reset_index(drop=True)      # ascending, then re-index
```

```
2013-01-02 -> 2016-03-01        796 trading days
```

Without `parse_dates` the column is text, and text sorts `'2013-10-01'` before
`'2013-2-01'`. Every lag you build after that is silently wrong.

**Model observation.** IBM daily closes, 2 January 2013 to 1 March 2016: 796
trading days, closes between $117.85 and $198.95. Row order matters because every
feature is defined by it — a lag is "the value in the row before", a rolling mean
is "the previous five rows", the target is "the row after". Shuffle the frame and
all three point at arbitrary dates. In LA02 each row was an independent district;
nothing about district 900 depended on district 899, so the order was an accident
of the file. Here the order **is** the data.

> 796 rows over 3.2 years, not 1,168. Non-trading days are absent, not zero.

## A2. Lag features `[2]`

```python
for k in (1, 2, 3):
    s[f'Close_lag{k}'] = s.Close.shift(k)        # positive shift = the past

s['Close_roll5'] = s.Close.shift(1).rolling(5).mean()
#                            ^^^^^^^^ shift FIRST, then roll
s['target'] = s.Close.shift(-1)                  # negative shift = tomorrow

s = s.dropna().reset_index(drop=True)            # 796 -> 790
```

`rolling(5).mean()` without the `shift(1)` includes today's close in today's
average — the most common leak in this assignment, and invisible in the output.

| Date | Close | lag1 | lag2 | lag3 | roll5 | target |
|---|---|---|---|---|---|---|
| 2013-01-09 | 176.56 | 177.06 | 177.31 | 178.09 | 178.40 | 177.07 |
| 2013-01-10 | 177.07 | 176.56 | 177.06 | 177.31 | 177.66 | 178.51 |
| 2013-01-11 | 178.51 | 177.07 | 176.56 | 177.06 | 177.22 | 176.83 |

Read one row across: today's `Close` becomes tomorrow's `lag1`, and today's
`target` is tomorrow's `Close`.

**Model observation.** The frame goes from 796 rows to 790. Six are lost: the
first five have no complete 5-day window (and the first three no complete set of
lags), and the last has no next day to predict. They must be dropped rather than
filled because the values are not missing data — the information does not exist.
Imputing a lag with a mean would invent a price that was never traded, and
filling the final target would invent a future.

## A3. The naive baseline `[2]`

```python
cut = int(len(s) * 0.8)
train, test = s.iloc[:cut], s.iloc[cut:]      # NO shuffling, NO random_state

base_mae  = mean_absolute_error(test.target, test.Close)     # 1.64
base_rmse = np.sqrt(mean_squared_error(test.target, test.Close))   # 2.24
```

`iloc` slicing on a sorted frame is the whole of a chronological split. There is
no sklearn function to call.

**Model observation.** Persistence — tomorrow's close equals today's close —
gives **MAE $1.64** and RMSE $2.24 over the 158-day test period, on a mean price
of $139. That is a 1.2% typical error from a rule with no parameters. Any model
must beat it to be interesting, because daily prices move slowly relative to
their level: yesterday's price already contains nearly all the information about
today's. A model reporting MAE $3 has not "achieved 98% accuracy" — it has lost
to a one-line rule. The baseline also sets the units of the argument: the only
meaningful claim is "X% better than persistence", because a raw error can be made
to look small simply by choosing an expensive stock.

---

## B1. Random split versus chronological split `[4]`

```python
feat = ['Close', 'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_roll5']

# the wrong way — rows drawn at random from anywhere in the three years
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    s[feat], s.target, test_size=0.2, random_state=0, shuffle=True)

# the honest way — train on the past, test on the future
rf_chrono = RandomForestRegressor(n_estimators=200, random_state=0)
rf_chrono.fit(train[feat], train.target)
```

```
shuffled       MAE  1.78
chronological  MAE 11.67          (persistence baseline: 1.64)
```

`random_state=0` on the forest is right — that controls the bootstrap.
`random_state` on the *split* is the thing that should not be there at all.

**Model observation.** Shuffled split: MAE $1.78. Chronological split: MAE
$11.67. The same forest, the same features, the same seed — a factor of seven,
decided entirely by how the rows were divided.

The shuffled number is meaningless because a random split **leaks the future into
training**. Consecutive trading days differ by well under a dollar, so when 3 May
sits in training and 4 May in test, the model is not forecasting — it is
interpolating between days it has already been shown. No such model can exist in
practice, where the test day is always after every training day.

The honest split also exposes a second problem: a random forest averages training
targets, so it **cannot predict a value outside the training range**. Trained on
$145–$199 and asked about a market that falls to $118, it flat-lines at roughly
$149 for the last six months. Between them, leakage and extrapolation are why
$11.67 — not $1.78 — is the true measure of this model, and why it loses to a
baseline of $1.64.

> A complete answer says the model **lost** to persistence. Reporting that
> honestly is the assignment.

## B2. Horizon and difficulty `[4]`

```python
g['target'] = g.Close.shift(-h)        # h days ahead, not 1
# ... same chronological split, same forest ...
mean_absolute_error(te.target, te.Close)     # the baseline must move with h too
```

| horizon | random forest MAE | persistence MAE | gap |
|---|---|---|---|
| 1 day | $11.67 | $1.64 | forest 7.1× worse |
| 5 days | $14.44 | $3.65 | forest 4.0× worse |
| 10 days | $16.31 | $5.30 | forest 3.1× worse |

Comparing a 10-day model against a 1-day baseline is the comparison error this
question is checking for.

**Model observation.** MAE rises with the horizon for both. The baseline's trend
is the informative one: a price that moves like a random walk accumulates
variance in proportion to time, so the typical distance from today's price grows
like **√h** — and 1.64, 3.65, 5.30 is close to that pattern. The further ahead
you ask, the more independent moves separate the answer from anything observable
today. The forest's line is flatter, but that is not the model improving: most of
its error at every horizon is the fixed cost of being unable to predict prices
below its training range, and that cost does not grow with h. At no horizon does
it beat the baseline.

---

## C1. Would you trade on it? `[2]` — compulsory

The price-level forest has MAE $11.67, so the question's premise ("a low MAE")
does not apply to it. Predict the **return** instead and the premise arrives —
returns stay in the same range forever, while prices drift out of it, so the
extrapolation problem disappears.

```python
g['ret'] = g.Close.pct_change()
for k in (1, 2, 3):
    g[f'ret_lag{k}'] = g.ret.shift(k)
g['ret_roll5']  = g.ret.shift(1).rolling(5).mean()
g['target_ret'] = g.Close.pct_change().shift(-1)

pred_price = test.Close.values * (1 + pred_ret)          # back to dollars for MAE
direction  = (pred_ret > 0) == (test.target_ret.values > 0)
```

```
MAE $1.62   (baseline $1.64)        directional accuracy 54.4%  on 158 days
```

**Is 54.4% a real edge?** `sqrt(0.25 / 158) = 0.040`. That accuracy is about one
standard error above chance — the kind of gap that appears and vanishes when you
change the ticker, the window, or the seed. 48.7% of those test days were up days
anyway.

**Model observation.** The returns model reaches MAE $1.62 against the baseline's
$1.64, and calls direction correctly on 54.4% of 158 test days. A low MAE with
near-chance directional accuracy is not useful, because the two measure different
things. MAE is dominated by the *level*: predicting "about the same as today" is
nearly right every day, which is why the baseline already scores $1.64 and why my
two-cent improvement is noise rather than insight. Every decision you would
actually take — buy, sell, hold — depends only on the *sign* of the change, and
on the sign this model is at 54.4% where 50% is a coin flip and the standard
error on 158 days is four percentage points. So no, I would not trade on it.
Before costs it is indistinguishable from chance; after brokerage and spread on
158 daily trades it is a reliable way to lose money.

> This is the assignment's real conclusion. Writing it clearly is worth more than
> a better model would be.

---

## Five common mistakes

- `train_test_split` anywhere on the time series — the whole assignment is about not doing this
- `rolling(5).mean()` without `shift(1)` — today's close leaks into today's average
- No baseline, or a baseline at the wrong horizon — then no number means anything
- Reporting accuracy as "98%" because the error is small relative to the price
- Hiding a bad result. A model that loses to persistence, reported honestly and explained, is a complete answer

## Before you submit

- [ ] Restart Kernel → **Run All**, top to bottom, no errors
- [ ] Ticker, date range and number of trading days stated in the notebook
- [ ] Every model MAE quoted beside the baseline MAE **at the same horizon**
- [ ] Filename `AML_LA03_<yourSAPID>.ipynb`, `README.md` beside it
- [ ] No dataset uploaded — the download code is enough

If you take one thing from this assignment: always build the baseline first, and
always ask what the number would be if the model knew nothing.
