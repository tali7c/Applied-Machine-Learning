# Theory Quiz 3 (T03)

Released after every group had sat the quiz and all answer sheets were collected.

| | |
|---|---|
| Covers | Prerequisite material, retained Units I–III (Lectures 1–3), and **Unit III (Lecture 4)** |
| Duration | 40 minutes |
| Marks | 20, scaled into the 10-mark theory-quiz component |
| Structure | 15 items — 5 MCQ, 4 fill-in-the-blank, 3 true/false, 2 short answer, 1 reasoning |
| Sets | Eight (A–H), four pages each |

## Files

- `T03-QP.pdf` — the question paper, all eight sets (32 pages)
- `T03-Solutions.pdf` — worked solutions for all eight sets (32 pages), matching the paper page for page
- `latex/` — the LaTeX sources of those two documents

## What it examined

Half the paper is Lecture 4 — the adaptive optimisers. Adagrad's accumulator and why it only ever grows, RMSprop's one-line fix, Adadelta needing no learning rate at all, and Adam's bias correction. The other half splits between the prerequisite floor (probability, Python, elementary calculus) and Lectures 1–3 carried forward: data leakage, updates per epoch, mini-batch noise, and loss against metric.

Nothing from Lecture 5 onward appears.

Two items are worth re-reading afterwards. **D1** asks for the first step of Adagrad and of RMSprop — the arithmetic is one square root, but the mark is for seeing that the gradient's magnitude cancels in both, leaving only its sign. **E1** asks you to design the smallest experiment that would settle *"Adam always beats momentum"*; its third part is the one that matters, and the answer is that the learning rate has to be tuned separately for each optimiser before the comparison means anything at all.

## About the eight sets

Every set examines the same topics at the same level, for the same marks, against the same marking scheme. They differ only in how each question is framed — which representation it leads with, what context it uses, and the particular numbers in the computed items.

**Sets are handed out at random.** Any student may receive any set, and the set letter carries no information whatsoever about the student who received it. It is a version number.

## Using the solutions

Find your set letter on your own paper and work from the matching four pages of `T03-Solutions.pdf`. The other seven sets cover the same material, so they are seven more worked examples if you want the practice.

Several questions have more than one good route to the answer. A route different from the printed one is not automatically worse — if you reached a different answer and can show your reasoning, bring your script.

**Part C.** The reason was optional. The mark is awarded if your circled answer is correct, or — if your circled answer is wrong — if the reason you wrote shows you understood the statement. A reason could only ever rescue a mark, never cost one.

**Part E.** There was no single right experiment. The marks were for whether the one you described could actually decide the question.
