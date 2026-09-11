# Theory Quiz 2 (T02)

Released after every group had sat the quiz and all answer sheets were collected.

| | |
|---|---|
| Covers | Prerequisite material, retained Unit I–II (Lectures 1–2), and **Unit III (Lecture 3)** |
| Duration | 40 minutes |
| Marks | 20, scaled into the 10-mark theory-quiz component |
| Structure | 15 items — 5 MCQ, 4 fill-in-the-blank, 3 true/false, 2 short answer, 1 reasoning |
| Sets | Eight (A–H), four pages each |

## Files

- `T02-QP.pdf` — the question paper, all eight sets (32 pages)
- `T02-Solutions.pdf` — worked solutions for all eight sets (32 pages), matching the paper page for page
- `latex/` — the LaTeX sources of those two documents

## What it examined

Half the paper is Unit III — gradient descent and its update rule, the learning rate and the stability threshold `η < 2/L`, batch / stochastic / mini-batch, and momentum. The other half splits evenly between the prerequisite floor (probability, Python, elementary calculus) and Units I–II carried forward as retained material: generalisation, cross-validation, and which constant each loss fits.

Nothing from Lecture 4 onward appears. The two Unit III items worth re-reading afterwards are D1 — two steps of gradient descent by hand, then what the error multiplier tells you about the regime — and E1, where a run that was stable on standardised features blows up on raw ones with the step size unchanged.

## About the eight sets

Every set examines the same topics at the same level, for the same marks, against the same marking scheme. They differ only in how each question is framed — which representation it leads with, what context it uses, and the particular numbers in the computed items.

**Sets are handed out at random.** Any student may receive any set, and the set letter carries no information whatsoever about the student who received it. It is a version number.

## Using the solutions

Find your set letter on your own paper and work from the matching four pages of `T02-Solutions.pdf`. The other seven sets cover the same material, so they are seven more worked examples if you want the practice.

Several questions have more than one good route to the answer. A route different from the printed one is not automatically worse — if you reached a different answer and can show your reasoning, bring your script.

**Part C.** The reason was optional. The mark is awarded if your circled answer is correct, or — if your circled answer is wrong — if the reason you wrote shows you understood the statement. A reason could only ever rescue a mark, never cost one.
