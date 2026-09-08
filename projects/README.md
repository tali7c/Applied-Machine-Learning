# AML Projects — P01 to P08

Applied Machine Learning (CSAI2017P) · Autumn 2026 · Dr. Tofik Ali

**Worth 15 marks of 100:** a 10-mark individual written quiz + a 5-mark group
repository submission.

**Printable copies:** `AML-Project-Handbook.pdf` and `AML-Project-Catalogue.pdf`
in this folder. The Markdown is the source of truth and the PDFs are built from
it, so if the two ever disagree, the Markdown wins.

## Start here

1. **`Project-Handbook.md`** — the rules for every theme: groups, the GitHub
   repository and the commit history I check, the milestones, the 5-mark rubric
   and how the quiz works. Read it before you choose.
2. **`AML-Project-Catalogue.md`** — the eight themes at a glance, with the
   coverage table.
3. **The theme folder** for the one you pick — the full brief: data, phase-by-
   phase instructions, the "done" checklist, pitfalls and what the quiz probes.

## The eight themes

| Code | Folder | Title | Task family | Data |
|---|---|---|---|---|
| P01 | `P01-bike-demand/` | Hourly bike-rental demand for fleet rebalancing | regression | UCI Bike Sharing, 17 k rows |
| P02 | `P02-air-quality/` | Next-day air quality for an Indian city | time-series regression + classification | CPCB India, 30 k rows |
| P03 | `P03-hotel-cancellations/` | Hotel-booking cancellations and the overbooking decision | imbalanced binary classification | Hotel Booking Demand, 119 k rows |
| P04 | `P04-student-dropout/` | Early warning of student dropout | imbalanced multiclass + fairness | UCI Student Dropout, 4.4 k rows |
| P05 | `P05-review-sentiment/` | What makes a review negative? | text classification | IMDB 50K reviews |
| P06 | `P06-fashion-mnist/` | Clothing-item recognition with classical ML | images + PCA | Fashion-MNIST, 70 k images |
| P07 | `P07-retail-segmentation/` | Customer segmentation for an online retailer | clustering → classification | UCI Online Retail II, 1.07 M rows |
| P08 | `P08-household-power/` | Household electricity: forecasting and daily-profile clustering | time series + clustering | UCI Household Power, 2.07 M rows |

**Effort note.** P07 and P08 carry the most data wrangling; their modelling lists
are correspondingly shorter. P01 and P04 have the cleanest data; their "done"
lists ask for more analysis in exchange. The bands are meant to be comparable —
if you disagree after reading two briefs, tell me before sign-up closes.

## The three rules people forget

1. **Create the GitHub repository in Unit IV and commit as you go.** I read the
   commit history. A project that appears in 1–3 commits in the last few days
   before submission scores at most 1.5 of the 5 group marks, however good the
   code is. Handbook §3.3.
2. **The 5 group marks are identical for every member** — they are not split by
   contribution. All individual differentiation is in the written quiz.
3. **Everyone sits the quiz.** One paper per theme, so different themes get
   different papers. Miss it and 10 of your 15 marks are gone.

## Choosing

Pick the question you actually want to answer. All eight need Unit IV plus at
least one of Units V, VI and VII, so none can be finished before Unit V is taught,
and none reuses a lab dataset — a re-submitted lab assignment is immediately
visible.

Sign-up is on the sheet, first come, first served. **Maximum two groups per
theme**; groups are 4–8 students.
