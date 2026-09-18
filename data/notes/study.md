# Study log

## Exam dates
- ROBT613 midterm: 14 March, 09:00, room 3.204.
- Signals & Systems final: 28 March.
- The workshop capstone demo is due the last day of the workshop.

## What actually works for me
Two 50-minute blocks in the morning beat four scattered hours in the evening.
I retain far more when I write the summary *before* re-reading the slides,
not after. Flashcards only help if I make them myself.

## Reading queue
1. "Attention Is All You Need" — re-read the multi-head section, I still can't
   explain why multiple heads help.
2. Qwen3 technical report — skim the training section, read the evaluation.
3. ReAct paper (Yao et al.) — read properly before the agents day.

## Notes from the LLM lecture
A language model predicts the next token given all previous tokens. That is
the whole training objective. Everything else — chat, reasoning, tool use —
is a behaviour layered on top by fine-tuning, not a separate mechanism.
Temperature does not make a model smarter; it changes how adventurously it
samples from what it already believes.
