# Experiment notes

Experiments: October 10–14, 2026. Notes prepared November 4, 2026.

We spent most of these runs trying to choose a dataset and a useful outcome measure. The basic comparison stayed the same: generate a conversation with Qwen, save it, and ask the same follow-up with different history formats. We have not yet found a task where pasting the history causes a large, consistent loss.

## Setup

We used Qwen2.5-7B-Instruct in BF16 with greedy decoding on one RTX 5090. The three conditions are:

|Condition|History supplied to the model|
|---|---|
|A: online|The conversation Qwen just generated, with its KV cache retained|
|B: role replay|The same text, replayed as alternating user and assistant messages|
|C: pasted|The same text with written role labels, inside one user message|

B versus C is the format comparison. A versus B checks what changes when we recompute the history. Each follow-up starts from a separate copy of the saved history. Some early runs only had B and C; their labels were N and P.

Public datasets supplied questions, documents, tables, or rules. We usually regenerated the assistant turns with Qwen. This keeps the main experiment close to copying a conversation the model actually had. In early WildChat checks we also used the original assistant replies; those were external histories, not condition A.

## What we tried

Only the three sections below have their full selected results in this repository. The other rows summarize earlier work.

|Task|Data and scoring|What happened|
|---|---|---|
|Short state tracking and instruction following|Synthetic conversations; 6 initial and 24 additional Multi-IF tasks, with Qwen replies|Short state tasks scored 16/18 in both formats. The instruction extension scored 9/24 in both, with different individual errors.|
|Reading comprehension and math|24 CoQA conversations, Lost/GSM8K tasks; answer matching plus inspection|One No/Yes flip disappeared when the question's time frame was made explicit. Some apparent errors were just different wording.|
|Financial and table QA|12 ConvFinQA and 12 SQA examples; numeric or answer-set checks|No clear case of B correct and C wrong. Wrong histories, units, rounding, and matching errors complicated the comparison.|
|Updated facts and conditions|12 LongMemEval examples; MultiTurnInstruct tasks and prompt variants|A few MTI gaps depended on the prompt. Many LongMemEval history replies hit the length cap.|
|Local editing|Four books, four functions, four tables; structural checks|B scored 15/24, C 17/24, a continuation hint 19/24, and a history-difference vector 11/24. Included below.|
|Text and distribution similarity|WildChat continuations and common-prefix KL; dense checks on editing tasks|Lower KL did not reliably give closer free-text answers. Sparse measurement positions could miss the first token divergence. Only the synthetic editing checks are included.|
|Repeated updates and plan tracking|Eight updated tables, eight project plans, approved-plan tasks|Several batches had no format gap. Open answers sometimes changed from prose to a JSON block.|
|Activation and cache interventions|Residual vectors, low-rank variants, KV replacement, position changes, random and reverse controls|Some development scores improved, but there was no consistent advantage over prompts. A position-based result did not carry over to seven new designs.|
|Rule selection and simple operations|Sorting, field selection, replacement, integer arithmetic|Masking old rules and restating the current rule both helped. Native-format answers could benefit too, so this was not specific to migration.|
|Step-by-step response practice|Confirmation-only histories, product-selection practice, actual plan suggestions|One pasted answer gave two steps instead of one. In a later six-plan batch, A/B/C all passed. Those batches differed in more than the amount of practice.|
|Persistent response rules|Eight MT-Eval histories and a same-history extension|One two-response answer became a single response after pasting. The extension found no additional format-only failure. Included below.|
|Reminders and privacy|Six reminder and six privacy tasks from MultiTurnInstruct|A reminder checker matched “rest” inside “interest.” A privacy example had already broken its rule during history generation.|
|Style|MT-Eval questions with formal/casual instructions; StyleDistance and contraction counts|Embedding similarity and native-answer similarity gave different pictures. A contraction effect seen in six old histories did not repeat consistently in four new ones. Counts are included below.|
|MathDial|599 test conversations downloaded and inspected|455 had consecutive focus/probing labels in the original data. We did not run Qwen on them.|

These were small development batches, often related to one another. Repeated prompts, methods, and seeds do not add independent conversations.

## 1. Local editing

We asked Qwen to write or update a short book, a function, or a table. The final two requests edited that actual output. Expected answers were built from the saved artifact, so we could check both the requested change and the material that should remain.

The selected batch has 12 histories and 24 requests. We kept six complete method columns from the original nine: B, C, a fixed continuation hint, a single-history difference vector, a random direction of the same norm, and the reverse direction. We did not select individual questions by success.

Complete-edit scores were **15, 17, 19, 11, 15, and 18 out of 24**, respectively. This is a useful warning for the project: matching B is not necessarily the same as fixing the answer, since C was sometimes better.

Book and table answers were checked as JSON structures. A separate count recorded whether unchanged pages or rows were still present; that count alone does not verify order. Code edits used AST comparisons, with whitespace normalization for docstrings. We did not execute the generated functions.

For KL, an earlier diagnostic selected one book, one function, and one table. Across two requests, two reference paths, and five conditions, it saved 60 distribution records. Measuring every reference position exposed divergences that the old sparse positions had missed. This was a measurement check on development data, not another test set. There was no online A condition in this batch.

## 2. Response rules from MT-Eval

We used two source versions for each of four rules: JSON, bullet lists, two responses, and an exact ending phrase. Qwen regenerated the acknowledgement and answered the first three task questions. We then tested questions 4 and 5 independently under A, B, C, split replay, and C with the rule repeated explicitly.

All eight histories and 80 final outputs are included. Seven histories followed their category rule throughout the three practice answers. For the two-response category, A and B passed **4/4**, while C passed **3/4**. The vaccine question in `mteval-24` is the clearest example: A/B gave two responses, C gave one, and repeating the rule fixed it. Its practice answers and final answer also met the separate word limit.

We then used all remaining questions 6–10 from those two histories. That produced 40 outputs, with no new source histories and no online A condition. B and C both passed **9/10**; the explicit reminder passed **10/10**. The shared failure was an email request.

Across the first batch and extension, B passed 13/14 requests and C 12/14. Those requests came from only two histories. The extension was chosen after seeing the first gap, so it is a follow-up diagnostic rather than independent confirmation.

The source checker tests the response form. For example, the two-response checker requires two different strings separated by `******`; it does not assess whether they make meaningfully different points. The main category scores also do not include the extra word limit. An early extension attempt failed in the scoring code after one unsaved generation; the 40 completed outputs here come from the corrected v2 run.

## 3. Contractions in casual replies

We used MT-Eval question streams and placed a casual-style request in the first user turn. Qwen answered three questions to form a history, and two further questions were tested independently.

The old batch has six casual histories and 48 outputs: B/C, each with the first style clause retained or removed. Removing that clause left the generated assistant history intact. It was an input ablation, not a newly generated no-style history.

The later batch has four new question streams and 52 outputs, adding online A and an explicit request to use formal, uncontracted wording. We counted 30 listed contraction forms, reporting both actual counts and density per 100 regex words.

In the old batch, B/C used **9/20 contractions**, and C had higher density in **6/6 histories**. In the new batch, A/B/C used **6/7/6**, and C had higher density in only **2/4 histories**. All three formats avoided the listed contractions on all four explicit formal-override requests.

The new batch did not meet the continuation criteria set before generation, so we did not extract vectors or run steering on it. The counts are sparse and depend on the wording opportunities in each answer. They are useful as a concrete feature, but not enough to stand for style as a whole. “New” here means the question text was absent from the 60 saved history files we checked, not that the topics were entirely new.

## Where this leaves the project

The data-processing setup is ready. We still need to choose one task and its primary measure. For a course project, persistent response rules are the most straightforward starting point: the source data are public, the output requirement is understandable, and the existing results show both a failure and a follow-up that did not reproduce it.

Editing gives us a second option with clearer content checks. The style results explain why we should not choose a broad embedding score just because it produces a visible difference. The next batch should settle one of these choices before adding more intervention variants.
