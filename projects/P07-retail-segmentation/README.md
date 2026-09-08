# P07 · Customer segmentation for an online retailer

> **Question.** Which genuinely distinct customer groups exist in two years of
> transactions — and can a classifier place a *new* customer into a segment after
> only their first few purchases?

Read `../Project-Handbook.md` first — repository rules, commit history, milestones,
group rubric, written quiz. This file defines only the work.

**Units:** IV, VI, VII · **COs:** CO1, CO2, CO3, CO4  
**Difficulty:** the heaviest data-wrangling theme in the catalogue, alongside P08.
In exchange, the "done" list asks for fewer model families. Start early. The
cleaning *is* the project's first half.

---

## 1. The situation

A UK gift wholesaler has a million transaction lines and no idea who its customers
are. "Segment the customers" is the classic unsupervised request, and it is
genuinely hard for a reason students usually skip past: there is no ground truth.
Nothing tells you k=4 is right. You have to argue for it, from the silhouette
score, from the dendrogram, and above all from whether the resulting groups mean
anything a marketing team could act on.

The second half makes it useful. Clusters found on two years of history cannot
help with a customer who signed up last week. So you turn the unsupervised result
into a supervised one: label every customer with their cluster, then train a
classifier that assigns a segment from early-purchase behaviour alone. That
pipeline — cluster to create labels, then classify to deploy — is a genuinely
common industrial pattern and it uses Units VI and VII together.

---

## 2. The data

**UCI Online Retail II** — all transactions of a UK non-store online gift retailer,
1 December 2009 to 9 December 2011.

- Page: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Download the **zip** (43.5 MB) in a browser. Licence CC BY 4.0. The file inside
  is `online_retail_II.xlsx` with **two sheets** (2009–2010, 2010–2011) — you need
  both, concatenated.
- 1 067 371 rows. Converting the xlsx to Parquet or CSV once, in a script, will
  save you hours; commit the script, gitignore the large converted file, and keep
  the original zip out of the repo if it exceeds the handbook's 50 MB limit
  (document the download instead).

**Columns:** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`,
`Price`, `Customer ID`, `Country`.

**The cleaning problems, all of which are graded:**

1. **Cancellations.** Invoices starting with `C` are cancellations and carry
   negative `Quantity`. Decide: remove them, or net them against the original
   purchase. Both are defensible; the decision must be stated and consistent.
2. **Missing `Customer ID`** on roughly a quarter of the rows. Customer-level
   analysis cannot use them. Count them, report the proportion, drop them for the
   RFM table, and say what you might be losing.
3. **Non-product `StockCode`s**: `POST`, `M`, `DOT`, `BANK CHARGES`, `AMAZONFEE`,
   `ADJUST`, and similar. Find them and decide.
4. **Zero and negative `Price`**, extreme `Quantity` (one order of 80 995 units).
5. **Duplicate rows** — there are some; check.

Write the cleaning as a **script in `src/`, not as scattered notebook cells**, so
that the pipeline is reproducible in one call. That is explicitly marked.

---

## 3. What you must do

### Phase A — cleaning (Unit IV, before Checkpoint 1)

1. Repository per handbook §3; conversion + cleaning script in `src/`.
2. `01-eda.ipynb`: row counts before and after each cleaning rule (a funnel table
   — start 1 067 371, end N, with each rule's loss on its own line). This table is
   worth real marks; it is the honest record of what you threw away.
3. Basic shape of the business: revenue by month, top products, orders by country
   (it is overwhelmingly UK — decide whether to model UK only).

### Phase B — RFM and Checkpoint 1 (end of Unit V)

4. Build the **RFM table**, one row per customer:
   - **Recency** — days from the customer's last purchase to a fixed reference
     date (use the day after the last date in the data; state it, and never use
     "today"),
   - **Frequency** — number of distinct invoices,
   - **Monetary** — total spend (`Quantity × Price`, summed).
   You may add tenure, average basket value, distinct products, return rate.
5. Look at the distributions. All three are heavily right-skewed. Decide between a
   **log transform** and a robust scaler, justify it, and show the before/after
   histograms — k-means on unscaled skewed RFM produces one huge cluster and three
   customers, and demonstrating that failure in the report is worth more than
   quietly avoiding it.
6. Tag `checkpoint-1`, push, with the RFM table built and one k-means run done.

### Phase C — clustering (Unit VII, to Checkpoint 2)

7. `03-models.ipynb` — all three algorithms on the same scaled RFM features:
   - **k-means**: sweep k from 2 to 10, plot the **elbow** (inertia) and the
     **silhouette** score, choose k, and defend it in words;
   - **hierarchical agglomerative clustering**: plot a **dendrogram** (a sample of
     customers if the full one is unreadable) and compare at least two linkages
     (Ward, complete, average). Say what changes and why Ward tends to win here;
   - **DBSCAN**: choose `eps` with a k-distance plot, and discuss what it labels
     as noise. DBSCAN will probably not produce a tidy segmentation on RFM data —
     reporting *why* is the point, not forcing it to work.
8. **Name and describe the segments** in business terms — "high-value loyalists",
   "lapsed big spenders", "one-off bargain buyers". Give each a size, its mean
   RFM values, and one sentence a marketing team could act on. Include a 2-D
   visualisation (PCA to two components, coloured by cluster).
9. Tag `checkpoint-2`, push.

### Phase D — the classifier (Unit VI, to final submission)

10. Take the cluster label as the target. Build features from **only the
    customer's first N purchases** (choose N — 3 invoices, or the first 30 days —
    and justify it). Train a classifier (logistic regression plus at least one
    tree-based model), stratified split, cross-validated.
11. Report accuracy, macro-F1, per-class recall, confusion matrix →
    `results/metrics.csv`. Then the honest question: **can you tell a valuable
    customer early?** If the answer is "only for the extreme segments", say that.
12. Report, README, REPRODUCIBILITY.md, final push.

---

## 4. Definition of done

- [ ] Cleaning as a reusable script, with a **funnel table** of rows lost per rule.
- [ ] Cancellations, missing IDs, non-product codes and outliers each decided and
      documented.
- [ ] RFM table with a stated reference date; skew handled and shown.
- [ ] k-means with elbow **and** silhouette, k chosen and defended.
- [ ] Hierarchical clustering with a dendrogram and ≥2 linkages compared.
- [ ] DBSCAN with a k-distance plot and a discussion of its noise points.
- [ ] Segments named, sized, described in business language, plus a 2-D plot.
- [ ] A classifier predicting segment from early-purchase features, with
      accuracy, macro-F1, per-class recall and a confusion matrix.
- [ ] Report, README, REPRODUCIBILITY.md; runs from a clean checkout.

## 5. Going further

- **Year 1 clusters vs year 2 clusters.** Fit separately and track migration: do
  customers move segments? A transition matrix is a strong figure.
- Product-level clustering from the `Description` text (TF-IDF + k-means) — what
  product families does the catalogue actually contain?
- Cluster stability: re-run k-means with 20 seeds and with bootstrap samples. How
  stable is your k? If it is not stable, that is a finding, not a failure.
- Compare against classic **RFM quantile scoring** (the 5×5×5 business heuristic).
  Does k-means find anything the heuristic misses?

## 6. Pitfalls specific to this theme

| Pitfall | Consequence |
|---|---|
| clustering unscaled RFM | Monetary dominates; one giant cluster |
| using "today" as the recency reference | results change every time it is run — not reproducible |
| leaving cancellations in as negative quantities | Monetary goes negative for real customers |
| keeping rows with missing `Customer ID` in the RFM table | phantom customers |
| choosing k by the elbow alone with no silhouette | an unsupported choice |
| segments named "Cluster 0, 1, 2" with no description | the business half is missing |
| the classifier trained on features that already encode the label (e.g. total spend) | circular — the segment came from that number |

The last one is the subtle trap: if your cluster labels came from Monetary and your
classifier's features include lifetime Monetary, you have proved nothing. Features
must come from the **early window only**.

## 7. What the written quiz will probe

Territory: what your funnel table lost and why; why you log-transformed (or did
not); what the silhouette score measures and what yours was; what Ward linkage
minimises; what DBSCAN called noise and what those customers were; the size and
description of your smallest segment; and how you kept the early-window classifier
from seeing the information that defined its own labels.

## 8. Deliverables

Per handbook §3.4–§3.6. You submit the repository URL only.
