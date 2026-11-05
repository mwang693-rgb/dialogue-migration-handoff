# Project outline

**Working title:** Response-rule continuity after copying a conversation

**Team:** Moyang Wang; remaining members to be added.

## Question

Does a model continue to follow a response rule when the same conversation is pasted into a single user message instead of supplied as alternating user and assistant turns?

Copying a conversation preserves the words but changes how they enter the model. We want to measure whether that affects a rule already used in the history, such as giving two responses to each question. We do not require the next answer to match word for word.

## Data and comparison

Use public MT-Eval questions and response rules. Let Qwen2.5-7B-Instruct generate the assistant turns, save the resulting history, and test each follow-up independently in three conditions:

- A: continue with the cache from actual generation;
- B: replay the same history with its original roles;
- C: paste the history, with written role labels, into one user message.

B/C measures the format difference. A/B checks the effect of recomputing the history. Record whether the history followed the rule, rather than discarding histories based on the final B/C outcome.

## What we still need to choose

The main choice is the response rule. Two-response answers are easy to inspect and have one existing format gap, although further questions did not reveal additional gaps. A local-editing task is a reasonable alternative if we want a stronger content checker.

The primary measure should match that choice: rule compliance for response rules, or complete-edit correctness with a separate preservation count for editing. Exact text match and common-prefix KL can be secondary measures. Include explicit requests to change the old rule, so persistence does not get rewarded when the user wants a change.

## Pilot results

- Editing: B/C passed 15/24 and 17/24. A continuation hint passed 19/24; the single-history difference vector passed 11/24.
- Two-response answers: A/B passed 4/4 and C 3/4 in the first batch. On ten additional requests from the same two histories, B/C both passed 9/10.
- Contractions: the old six-history pattern appeared in only two of four new histories. We did not proceed to steering on that batch.

## Next run

Choose one task and primary measure, set a small data and generation budget, and run A/B/C. If the difference repeats, compare an explicit rule reminder with one activation intervention, including random and reverse controls. A small or absent format effect is still an interpretable course result.

## Reading

MT-Eval supplies the question streams and response-rule checkers. StyleDistance is relevant to the style measurements we tried. CAA and In-context Vectors provide intervention baselines. This outline still needs the course literature-review section and the final team assignments.
