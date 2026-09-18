# %% [markdown]
# # 03 · Your first generation
#
# **Day 1 · ~75 minutes**
#
# This is the moment the workshop becomes real: a language model, running on
# your hardware, with no API key and no internet connection required after the
# download.
#
# By the end you will be able to:
#
# 1. load Qwen3 and generate text;
# 2. use the chat template correctly (and debug it when output looks wrong);
# 3. stream tokens as they arrive;
# 4. measure tokens/second and prompt cost;
# 5. hold a multi-turn conversation and explain why the model has no memory.

# %%
import sys
from pathlib import Path

REPO_ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_workshop.env import describe_environment

print(describe_environment())

# %% [markdown]
# ## 3.1 · Load the model
#
# First run downloads ~1.5 GB. After that it loads from cache in seconds.
#
# `load_model` picks a sensible dtype for your device (bf16 on modern GPUs,
# fp32 on CPU) and gives readable errors when downloads fail. Open
# `src/qwen_workshop/loading.py` — it is ~60 lines and worth reading.

# %%
from qwen_workshop.loading import load_model

MODEL_ID = "Qwen/Qwen3-0.6B"  # swap for the model notebook 02 recommended

qwen = load_model(MODEL_ID)
print(qwen)
print(f"parameters      : {qwen.n_params / 1e9:.2f} B")
print(f"memory in use   : {qwen.memory_footprint_gb():.2f} GB")

# %% [markdown]
# ## 3.2 · The raw loop, with nothing hidden
#
# Before using any helper, do it by hand once. Every line below corresponds to
# a step from notebook 01.

# %%
import torch

messages = [{"role": "user", "content": "Name three uses for a language model that runs offline."}]

# Step 1: messages -> a single string, using the model's own chat template.
prompt = qwen.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
print("=== what the model actually receives ===")
print(prompt)

# Step 2: string -> token ids -> tensors on the right device.
inputs = qwen.tokenizer([prompt], return_tensors="pt").to(qwen.model.device)
print(f"\nprompt is {inputs['input_ids'].shape[-1]} tokens")

# Step 3: the generation loop (implemented in C++/CUDA inside generate()).
with torch.inference_mode():
    output_ids = qwen.model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        pad_token_id=qwen.tokenizer.eos_token_id,
    )

# Step 4: slice off the prompt - generate() returns prompt + completion.
new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
text = qwen.tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\n=== the model's answer ===")
print(text)

# %% [markdown]
# **The step everyone forgets is step 4.** `generate()` returns the prompt
# *and* the completion. Forget to slice and your app echoes the user's own
# question back at them.
#
# > **Exercise 1.** Print `qwen.tokenizer.decode(output_ids[0])` — the whole
# > thing, without `skip_special_tokens`. Find `<|im_start|>` and `<|im_end|>`.

# %% [markdown]
# ## 3.3 · The same thing, with the helper
#
# `chat()` wraps exactly those four steps and adds timing and thinking-mode
# parsing.

# %%
from qwen_workshop.chat import chat

result = chat(qwen, "Explain what a token is, in two sentences, to a first-year student.")

print(result.text)
print()
print(f"prompt     : {result.prompt_tokens} tokens")
print(f"completion : {result.completion_tokens} tokens")
print(f"speed      : {result.tokens_per_second:.1f} tokens/second")

# %% [markdown]
# ## 3.4 · System prompts change everything
#
# The system prompt is the cheapest, highest-leverage knob you have. Same
# question, four different personalities:

# %%
from qwen_workshop.utils import wrap_print

question = "Why is the sky blue?"

systems = {
    "default": "You are a helpful assistant.",
    "terse": "Answer in exactly one sentence. No preamble, no caveats.",
    "for a child": "You explain things to a curious 8-year-old using everyday comparisons.",
    "physicist": "You are a physicist. Be precise and use correct terminology.",
}

for label, system in systems.items():
    answer = chat(qwen, question, system=system, sampling=None)
    print(f"--- {label} ---")
    wrap_print(answer.text)
    print()

# %% [markdown]
# > **Exercise 2.** Write a system prompt that makes the model always answer in
# > Kazakh, or always reply as valid JSON. Which instruction does the 0.6B model
# > follow reliably, and which does it drift away from after a few turns?
# > (Notebook 07 fixes the JSON case properly.)

# %% [markdown]
# ## 3.5 · Streaming
#
# Total time is identical. *Perceived* time is transformed. Generation at
# 15 tok/s feels instant when streamed and feels broken when it is not.

# %%
from qwen_workshop.chat import print_stream, stream_chat

print("Streaming (watch it arrive token by token):\n")
_ = print_stream(
    stream_chat(qwen, "Write a four-line poem about a laptop running an AI model.")
)

# %% [markdown]
# Under the hood this runs `generate()` on a background thread and reads from a
# `TextIteratorStreamer` queue — see `stream_chat` in `src/qwen_workshop/chat.py`.

# %% [markdown]
# ## 3.6 · Multi-turn conversation: the model has no memory
#
# This surprises everyone. A model is a **pure function**: same input, same
# distribution. It remembers nothing between calls.
#
# "Memory" in a chatbot is just the whole transcript being re-sent every turn.
# That is why long chats get slower and eventually hit the context limit.

# %%
conversation = [
    {"role": "system", "content": "You are a concise assistant. Keep answers under 40 words."},
]


def turn(user_text: str) -> str:
    """One conversational turn: append, generate, append, return."""
    conversation.append({"role": "user", "content": user_text})
    reply = chat(qwen, conversation)
    conversation.append({"role": "assistant", "content": reply.text})
    print(f"You  : {user_text}")
    print("Qwen : ", end="")
    wrap_print(reply.text)
    print(f"       ({reply.prompt_tokens} prompt tokens sent this turn)\n")
    return reply.text


turn("My name is Aisulu and I am studying robotics.")
turn("What am I studying?")
turn("What is my name?")

# %% [markdown]
# Watch the **prompt tokens grow every turn**. That is the entire conversation
# being re-sent. Three consequences:
#
# 1. Cost and latency grow with conversation length.
# 2. Eventually you hit the context limit and must summarise or drop old turns.
# 3. If you forget to append the assistant's reply, the model has amnesia —
#    a very common bug.

# %%
# Prove point 3: same question, transcript without the introduction.
amnesiac = chat(qwen, [{"role": "user", "content": "What is my name?"}])
print("Without history:", amnesiac.text)

# %% [markdown]
# ## 3.7 · Measure your machine
#
# Know your numbers. You will need them when deciding what is feasible in the
# capstone.

# %%
from qwen_workshop.utils import Timer

prompts = {
    "short": "Hello!",
    "medium": "Summarise the water cycle in one paragraph.",
    "long": "Write a 300-word explanation of why open-weight models matter for universities.",
}

print(f"{'prompt':<9}{'in':>6}{'out':>6}{'seconds':>9}{'tok/s':>8}")
print("-" * 38)
for label, text in prompts.items():
    r = chat(qwen, text, sampling=None)
    print(f"{label:<9}{r.prompt_tokens:>6}{r.completion_tokens:>6}"
          f"{r.seconds:>9.2f}{r.tokens_per_second:>8.1f}")

# %% [markdown]
# > **Exercise 3.** Record your tok/s here: `______`. Compare with a classmate
# > on different hardware. If you have a GPU, re-run with `device="cpu"` and
# > measure the slowdown factor.

# %% [markdown]
# ## 3.8 · Batching: the free speed-up
#
# GPUs are parallel machines. Generating 8 answers at once takes barely longer
# than generating 1, because you are using hardware that was idle anyway.

# %%
def batch_generate(prompts: list[str], max_new_tokens: int = 60) -> list[str]:
    """Generate answers for several prompts in one forward pass.

    Note `padding_side="left"`: for decoder-only models the generation must
    continue from the *last real token*, so padding goes on the left. Getting
    this wrong produces confident nonsense.
    """
    texts = [
        qwen.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for p in prompts
    ]
    tokenizer = qwen.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(qwen.model.device)
        with torch.inference_mode():
            out = qwen.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.8, pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        return [tokenizer.decode(row[prompt_len:], skip_special_tokens=True) for row in out]
    finally:
        tokenizer.padding_side = original_side


questions = [
    "Give one tip for saving money on groceries.",
    "Give one tip for sleeping better.",
    "Give one tip for learning a language.",
    "Give one tip for cycling in winter.",
]

with Timer("batch of 4"):
    answers = batch_generate(questions)

for q, a in zip(questions, answers):
    print(f"Q: {q}")
    wrap_print(a.strip())
    print()

# %% [markdown]
# Compare against four sequential calls. On a GPU the batch is typically
# 2–3× faster overall; on CPU the gain is smaller because you were already
# compute-bound.

# %%
with Timer("4 sequential calls"):
    for q in questions:
        chat(qwen, q, sampling=None)

# %% [markdown]
# ## 3.9 · Free your memory
#
# In Jupyter, a model stays loaded until you explicitly drop it. This is the
# number one cause of `CUDA out of memory` halfway through a session.

# %%
def free_memory(*objects) -> None:
    """Delete references and empty the CUDA cache."""
    import gc

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"CUDA memory still allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("CPU mode - memory returns to the OS after garbage collection.")


# Keep `qwen` for now; run this when you move on.
# free_memory(qwen)
print("Call free_memory(qwen) before loading a different model.")

# %% [markdown]
# ## Checkpoint
#
# 1. What are the four steps from a message list to displayed text?
# 2. Why must you slice the output of `generate()`?
# 3. Why does a chatbot's prompt grow every turn?
# 4. What is your machine's tokens/second for Qwen3-0.6B?
# 5. Why does batching need left padding?
#
# ➡️ **Next:** [`04_decoding_and_prompting.ipynb`](04_decoding_and_prompting.ipynb) —
# control *how* the model chooses its words, and meet Qwen3's thinking mode.
