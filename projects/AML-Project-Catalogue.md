# Applied Machine Learning (CSAI2017P) — Project Catalogue, Autumn 2026

This file is the overview. **The rules are in `Project-Handbook.md`; the full brief for each theme is in its own folder** (`P01-bike-demand/`, `P02-air-quality/`, …). Read the handbook, then the folder for the theme you pick.

## How the component works

| Sub-component | Marks | Granularity |
|---|---|---|
| Project quiz — a written paper, **one set per theme** | **10** | per student |
| Repository submission — state **and commit history** of the group's GitHub repo | **5** | per group, identical for every member |

Read the weighting before you choose a group. The quiz is worth twice the repo mark, and the repo mark is **not** divided by contribution — every member of a group receives the same 5. A passenger therefore collects the group marks and then sits a paper about work they did not do; a contributor in a weak group can still score well. Every member sits the paper separately.

**Groups:** 4–8 students, self-formed. **At most two groups per theme**, first come, first served on the sign-up sheet. Themes P01–P08.

**You submit one thing: the GitHub repository URL.** No zip, no attachment. I read the commit history — created in Unit IV, at least 20 commits across at least 6 distinct weeks, no more than 40 % of them in the final week, every member committing under their own account, and both checkpoint tags pushed on time. A repository that appears in 1–3 commits in the last few days before submission scores at most 1.5 of 5, however good the code inside it is. Full conditions in `Project-Handbook.md` §3.

**Timeline** (anchored to units, not dates)

| Milestone | Anchored to |
|---|---|
| Catalogue released; groups formed; theme selected; **repo created** | during / end of Unit IV |
| Checkpoint 1 — data loaded and shown, one baseline number (tag `checkpoint-1`) | end of Unit V |
| Checkpoint 2 — full pipeline, initial results (tag `checkpoint-2`) | end of Unit VI |
| Final submission — the repository URL | end of Unit VII |
| Project quiz — written, individual | final lecture / final lab (Lab experiment 15 slot) |

**Data on campus.** The UPES network breaks Python downloads (`fetch_*`, `ucimlrepo`, `yfinance`, `requests`) — see `../lab-assignments/Anchor-Datasets.pdf`. Every theme below therefore uses a file you **download once in a browser** (or at home). Files under 50 MB are committed to `data/raw/`; the two larger ones (P05, P08) are gitignored with download instructions in the README — your theme brief says which applies. Do not disable certificate verification.

**Common deliverables for every theme:** a Git repository with `README.md`, `requirements.txt`, notebooks/scripts that run top-to-bottom from a clean checkout, a 4–6 page report (problem, data, method, results, limitations), and reproducibility notes (random seeds, environment, exact commands). The exact layout is in `Project-Handbook.md` §3.4.

---

## The eight themes

Each theme is stated as a question, not a technique. The "done" line is the minimum for full group marks; the "further" items are where a strong group separates itself. **The summaries below are not the brief** — open the theme's folder for the phase-by-phase instructions, the pitfalls and the "done" checklist.

| Theme | Folder |
|---|---|
| P01 | [`P01-bike-demand/`](P01-bike-demand/README.md) |
| P02 | [`P02-air-quality/`](P02-air-quality/README.md) |
| P03 | [`P03-hotel-cancellations/`](P03-hotel-cancellations/README.md) |
| P04 | [`P04-student-dropout/`](P04-student-dropout/README.md) |
| P05 | [`P05-review-sentiment/`](P05-review-sentiment/README.md) |
| P06 | [`P06-fashion-mnist/`](P06-fashion-mnist/README.md) |
| P07 | [`P07-retail-segmentation/`](P07-retail-segmentation/README.md) |
| P08 | [`P08-household-power/`](P08-household-power/README.md) |

### P01 · Hourly bike-rental demand for fleet rebalancing

**Question.** Given hour, calendar and weather, how many bikes will be rented in the next hour — and which factors matter most?  
**Data.** UCI Bike Sharing, `hour.csv` (17 379 rows, 1.1 MB). https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset — browser download of the zip. CC BY 4.0.  
**Done.** Cleaned features (cyclic encoding of hour/month, categorical weather), a leakage-free chronological split, linear regression and ridge/lasso baselines, one non-linear model (tree ensemble or MLP), RMSE and R² reported on the held-out period with a justified choice between them, and coefficient interpretation.  
**Further.** Gradient descent implemented by hand and compared with the closed-form solution (Unit III); separate `casual` vs `registered` models; a peak-hour classifier with ROC/AUC.  
**Units.** III, IV, V (+ VI optional). **COs.** CO2, CO3, CO4.

### P02 · Next-day air quality for an Indian city

**Question.** Can yesterday's pollutant readings and the calendar predict tomorrow's PM2.5 (regression) and the AQI bucket (classification) well enough to issue a public warning?  
**Data.** CPCB *Air Quality Data in India (2015–2020)*, use `city_day.csv` (~30 000 rows). https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india — Kaggle login + browser download. Pick one or two cities.  
**Done.** Missing-data handling documented (this set has many gaps), lag/rolling features built without look-ahead, a **time-ordered** split, regression (linear → regularised) for PM2.5, a classifier for the AQI bucket with confusion matrix and ROC/AUC, and an honest comparison against the naïve "tomorrow = today" baseline.  
**Further.** Multi-city model with city as a feature vs one model per city; season/festival effects; imbalance handling for the rare "Severe" class.  
**Units.** IV, V, VI. **COs.** CO2, CO3, CO4.

### P03 · Hotel-booking cancellations and the overbooking decision

**Question.** At booking time, which reservations will be cancelled — and at what probability threshold should the hotel overbook?  
**Data.** Hotel Booking Demand (Antonio, de Almeida & Nunes), 119 390 rows. https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand — Kaggle login + browser download.  
**Done.** A leakage audit (columns such as `reservation_status` must be dropped — say why), encoding of the mixed categorical features, stratified split, logistic regression, KNN, decision tree, random forest and one boosting model compared on ROC/AUC and precision–recall, then a threshold chosen for a stated cost of an empty room vs a walked guest.  
**Further.** Resampling vs class weights for the imbalance; calibration curve; a per-hotel (city vs resort) comparison.  
**Units.** IV, V (logistic), VI. **COs.** CO2, CO3, CO4.

### P04 · Early warning of student dropout

**Question.** From data known at enrolment and after semester 1, which students are at risk of dropping out, and what would a fair early-warning rule look like?  
**Data.** UCI *Predict Students' Dropout and Academic Success*, 4 424 students × 36 features, three classes (dropout / enrolled / graduate), strongly imbalanced. https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success — browser download. CC BY 4.0.  
**Done.** Multiclass classification with at least three model families (naïve Bayes, tree/ensemble, SVM or MLP), stratified cross-validation, macro-F1 and per-class recall reported and defended over plain accuracy, an "enrolment-only" model vs a "plus semester-1" model, and a feature-importance discussion that flags any demographic feature a university should *not* act on.  
**Further.** Cost-sensitive thresholds; one-vs-rest ROC; a 2-class (dropout vs not) reformulation and whether it changes the ranking of models.  
**Units.** IV, VI (+ V logistic). **COs.** CO1, CO2, CO3, CO4.

### P05 · What makes a review negative? (text)

**Question.** Can a linear model on bag-of-words beat non-linear models on 50 000 movie reviews, and which words actually drive the prediction?  
**Data.** IMDB 50K Movie Reviews, 50 000 labelled reviews, balanced. https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews — Kaggle login + browser download (~66 MB).  
**Done.** A text-cleaning pipeline (HTML stripping, tokenisation, stop-words — with an ablation showing whether each step helped), TF-IDF features, logistic regression and multinomial naïve Bayes and a linear SVM compared with accuracy and ROC/AUC, the top ±20 weighted words shown and sanity-checked, and a learning curve (accuracy vs training size).  
**Further.** Bigrams and `min_df` sweeps; error analysis of 30 misclassified reviews; hierarchical or k-means clustering of reviews using the TF-IDF vectors to find sub-genres.  
**Units.** II, IV, V, VI (+ VII optional). **COs.** CO2, CO3, CO4.

### P06 · Clothing-item recognition with classical ML

**Question.** How far can non-deep methods (PCA + KNN / SVM / random forest / a small MLP) get on 28×28 clothing images, and which classes are confused with which?  
**Data.** Fashion-MNIST (Zalando), 70 000 images, 10 classes. https://github.com/zalandoresearch/fashion-mnist — browser download of the four gzip files, or the CSV version on Kaggle. MIT licence.  
**Done.** A loader for the raw IDX files, pixel scaling, PCA with an explained-variance curve and a chosen number of components, at least four classifiers compared under one protocol on the official test split, per-class precision/recall and a confusion matrix with a written explanation of the shirt / T-shirt / pullover confusion, and a timing table (fit and predict seconds per model).  
**Further.** Backpropagation MLP written with NumPy vs scikit-learn's; k-means on PCA embeddings vs the true labels (adjusted Rand index); data augmentation by shifts.  
**Units.** IV (dimensionality reduction), VI, VII optional. **COs.** CO1, CO2, CO3, CO4.

### P07 · Customer segmentation for an online retailer

**Question.** Which distinct customer groups exist in two years of transactions, and can a classifier assign a *new* customer to a segment after their first few purchases?  
**Data.** UCI Online Retail II, 1.07 M transactions, Dec 2009–Dec 2011 (43.5 MB xlsx). https://archive.ics.uci.edu/dataset/502/online+retail+ii — browser download. CC BY 4.0.  
**Done.** Cleaning documented (cancellations, negative quantities, missing `CustomerID`), RFM features per customer (Recency, Frequency, Monetary), scaling justified, k-means with the elbow/silhouette choice of *k*, hierarchical agglomerative clustering with a dendrogram and a linkage comparison, DBSCAN with a discussion of what it calls noise, the segments named and described in business terms, then a classifier trained to predict the segment from early-purchase features with cross-validated accuracy.  
**Further.** Year-1 clusters vs year-2 clusters — do customers migrate?; product-level clustering from the description text.  
**Units.** IV, VI, VII. **COs.** CO1, CO2, CO3, CO4.

### P08 · Household electricity: forecasting and daily-profile clustering

**Question.** Can the next hour's household power draw be forecast from recent history, and what typical "kinds of day" does the household have?  
**Data.** UCI Individual Household Electric Power Consumption, 2.07 M one-minute readings over 47 months (127 MB text, 20 MB zip). https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption — browser download. CC BY 4.0.  
**Done.** Resampling from minutes to hours with the ~1.25 % missing rows handled and documented, lag/rolling and calendar features without look-ahead, a chronological split, linear/ridge regression vs a tree ensemble vs the persistence baseline on MAE and RMSE, then each day reshaped into a 24-value profile and clustered (k-means and HAC) with the cluster centroids plotted and interpreted (weekday/weekend, holiday, summer).  
**Further.** Sub-metering (kitchen / laundry / heater) as targets; DBSCAN to find anomalous days; an MLP forecaster.  
**Units.** IV, V, VII (+ VI ensembles). **COs.** CO2, CO3, CO4.

---

## Coverage check

| | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 |
|---|---|---|---|---|---|---|---|---|
| Task family | regression | time-series reg. + clf | imbalanced binary | imbalanced multiclass | text | images | clustering → clf | time series + clustering |
| Unit IV preprocessing | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Unit V regression | ✓ | ✓ | logistic | logistic | logistic | | | ✓ |
| Unit VI classification | opt. | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | opt. |
| Unit VII clustering | | | | | opt. | opt. | ✓ | ✓ |
| CO1 | | | | ✓ | | ✓ | ✓ | |
| CO2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CO3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| CO4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Data size | 1 MB | ~3 MB | 16 MB | 0.5 MB | 66 MB | 30 MB | 44 MB | 20 MB zip |

All eight need Unit IV plus at least one of V/VI/VII; none can be finished before Unit V is taught. None reuses a lab anchor dataset (housing, churn, stock, digits, Iris, spam, German credit, HAR, breast cancer, PlantVillage), so the project cannot be a re-submitted lab.

**Effort band.** P01 and P04 are the most contained (small, clean data); P07 and P08 carry the most data-wrangling. The "done" definitions are sized so that the wrangling-heavy themes ask for fewer model families.

## Data routes

All eight links were checked on 2026-09-04 and resolve. UCI zips download directly; the Kaggle sets need a free account and the "Download" button; Fashion-MNIST is four gzip files linked from the GitHub README. Mirrors are placed on the LMS where licensing permits — check there first.
