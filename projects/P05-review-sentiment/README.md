# P05 · What makes a review negative? Text classification on 50 000 reviews

> **Question.** Can a linear model on bag-of-words beat more complex models on
> 50 000 movie reviews — and which words actually drive the prediction?

Read `../Project-Handbook.md` first — repository rules, commit history, milestones,
group rubric, written quiz. This file defines only the work.

**Units:** II (loss functions), IV, V (logistic), VI (+ VII optional)  
**COs:** CO2, CO3, CO4  
**Difficulty:** middle. The pipeline is short; the marks are in the ablations and
the interpretation.

---

## 1. The situation

Text is not a feature matrix. Before any model runs, somebody has to decide what
a "feature" even is: a word, a stemmed word, a pair of words, a character n-gram —
and whether to delete punctuation, case, stop-words, and HTML. Each of those
decisions is usually made by copying a tutorial. In this project you are not
allowed to copy it: you must **measure** each step and report whether it helped.

The second half is interpretation. A logistic regression on TF-IDF gives you a
weight per word, which is a rare gift — you can read the model. Some of what you
read will be sensible ("waste", "boring"), and some will be an artefact of the
corpus. Finding an artefact and explaining it is worth more than another 0.5 %
accuracy.

---

## 2. The data

**IMDB Dataset of 50K Movie Reviews** (Maas et al., 2011 — the Large Movie Review
Dataset, as a single CSV).

- Page: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Free Kaggle account; download `IMDB Dataset.csv` in a **browser** (~66 MB).
- 50 000 rows, two columns: `review`, `sentiment` (`positive` / `negative`).
- **Perfectly balanced** — 25 000 each. So accuracy is a legitimate headline here,
  unlike in most of the other themes. Say why in the report.
- 66 MB is over the handbook's commit limit: **gitignore it** and put exact
  download instructions in the README. The file also contains
  **duplicate reviews** — count them yourself, decide what to do, and document it.

The reviews contain raw HTML (`<br /><br />`). Removing it is not optional; how
much *else* you remove is the experiment.

---

## 3. What you must do

### Phase A — set up (Unit IV, before Checkpoint 1)

1. Repository per handbook §3; `.gitignore` the CSV; README download steps.
2. `01-eda.ipynb`: label balance, review length distribution by class (negative
   reviews are slightly longer — check), duplicate detection, and a look at five
   raw reviews so the HTML problem is visible in the notebook.
3. Hold out a **stratified test set of 10 000** reviews now, and do not look at it
   again until Phase D. Everything else — every ablation, every tuning run — is
   cross-validated inside the remaining 40 000.

### Phase B — Checkpoint 1 (end of Unit V)

4. A minimal pipeline: strip HTML → `CountVectorizer` → logistic regression.
   Report cross-validated accuracy on the training portion. That is your baseline.
5. Tag `checkpoint-1`, push.

### Phase C — the ablation and the models (Unit V–VI, to Checkpoint 2)

6. `02-preprocessing.ipynb` — **the cleaning ablation.** Start from the minimal
   pipeline and add one step at a time, measuring cross-validated accuracy after
   each. At minimum:

   | Variant | Accuracy (CV) | Kept? |
   |---|---|---|
   | raw text | | |
   | + HTML stripping | | |
   | + lowercasing | | |
   | + punctuation removal | | |
   | + stop-word removal | | |
   | + stemming or lemmatisation | | |
   | TF-IDF instead of raw counts | | |

   Fill that table in with real numbers and **keep only what helped**. Two of
   these will probably hurt; naming which, and offering an explanation, is worth
   more than the accuracy itself. (Think about what happens to "not good" when
   "not" is a stop-word.)

7. `03-models.ipynb` — on your chosen representation, compare:
   - **logistic regression** (tune `C` by cross-validation),
   - **multinomial naïve Bayes** (tune `alpha`),
   - **linear SVM** (`LinearSVC`; tune `C`),
   - one non-linear model of your choice, and a note on why non-linear models are
     usually a poor trade here.
   Connect this back to Unit II: state which loss each of these minimises.
8. Tag `checkpoint-2`, push.

### Phase D — evaluation and interpretation (Unit VI–VII)

9. `04-evaluation.ipynb`, on the untouched 10 000:
   - accuracy, precision, recall, F1, **ROC-AUC** for every model →
     `results/metrics.csv`; ROC curves on one axis;
   - **the top ±20 words** by logistic-regression coefficient, as a horizontal bar
     chart. Sanity-check them: which are obviously sentiment words, and which are
     corpus artefacts? Explain at least one artefact;
   - a **learning curve** — accuracy against training-set size, from 1 000 to
     40 000. Where does it flatten? What does that say about collecting more data
     versus building a better model?
   - **error analysis**: read 20–30 misclassified reviews yourself and group them
     into failure types (sarcasm, mixed sentiment, plot summary of a sad film,
     reviews about the DVD rather than the film). Report the categories with an
     example each. This is a required section, and it needs human reading, not
     another metric.
10. Report, README, REPRODUCIBILITY.md, final push.

---

## 4. Definition of done

- [ ] Test set of 10 000 held out from the start and touched once.
- [ ] Duplicates found and a documented decision made.
- [ ] The **cleaning ablation table**, filled with real cross-validated numbers,
      with steps that hurt identified and dropped.
- [ ] TF-IDF vs raw counts compared.
- [ ] Logistic regression, multinomial naïve Bayes and linear SVM, each with its
      hyper-parameter tuned by cross-validation on the training part only.
- [ ] Accuracy, precision, recall, F1 and ROC-AUC for every model on the held-out
      set; ROC curves plotted.
- [ ] Top ±20 coefficient words, plotted, with at least one artefact explained.
- [ ] Learning curve with a written interpretation.
- [ ] Error analysis of 20–30 misclassifications, grouped into failure types.
- [ ] Report, README, REPRODUCIBILITY.md; runs from a clean checkout.

## 5. Going further

- **Bigrams.** `ngram_range=(1,2)` with `min_df` sweeps. Does "not good" appearing
  as one feature fix the negation problem you found in the ablation? Show it.
- Character n-grams — surprisingly strong, and robust to spelling.
- Vocabulary size vs accuracy: sweep `max_features` and plot.
- **Clustering (Unit VII):** k-means or hierarchical clustering on the TF-IDF
  vectors, ignoring labels. Do the clusters correspond to sentiment, to genre, or
  to something else? Report an adjusted Rand index against the true labels and
  interpret what the clusters actually found.
- Train on the first 25 000 and test on the second 25 000 — is the corpus ordered?

## 6. Pitfalls specific to this theme

| Pitfall | Consequence |
|---|---|
| `fit_transform` on the whole corpus before splitting | the vectoriser sees the test set — leakage |
| stop-word removal that deletes "not", "no", "never" | negation is destroyed; and you will see it in the table |
| committing the 66 MB CSV | breaks the repo rules in the handbook |
| duplicates left in, straddling the split | the same review is trained and tested on |
| reporting only accuracy with no coefficient inspection | the interesting half is missing |
| an error-analysis section written without reading any reviews | obvious, and marked as such |

## 7. What the written quiz will probe

Territory: which cleaning steps hurt in your table and your explanation for it;
what TF-IDF does to a word that appears in every review; which loss your
`LinearSVC` minimises and how it differs from logistic regression's; what one of
your top-20 words is doing there; where your learning curve flattens and what you
would do about it; two failure types you found and why the model fails on them.

## 8. Deliverables

Per handbook §3.4–§3.6. You submit the repository URL only.
