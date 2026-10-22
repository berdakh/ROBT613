# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/22_kazakh_and_russian.ipynb)
#
# *Click the badge to run this notebook in Google Colab - no install required. The first code cell sets everything up.*

# %% [markdown]
# # 22 · Working in Kazakh and Russian
#
# **~90 minutes · most of it runs with no model at all**
#
# 📖 The reading companion: [`docs/guides/kazakh-and-russian.md`](../docs/guides/kazakh-and-russian.md)
#
# Every other notebook in this workshop is written for an audience that could
# be anywhere. This one is not.
#
# If you build in Kazakh or Russian you pay a tax that the English-language
# tutorials never mention, because the people writing them do not pay it. The
# tax is real, it is measurable, and if you do not measure it you will
# mis-size your context windows, under-budget your costs and over-trust your
# retrieval.
#
# By the end you will be able to:
#
# 1. measure the tokenizer penalty for your own text rather than quoting it;
# 2. explain *why* it happens — and prove it by building a tokenizer yourself;
# 3. show that a **bigger vocabulary makes the gap worse**, not better;
# 4. convert the penalty into context, money and latency;
# 5. build a small Kazakh evaluation set so quality claims are yours, not mine;
# 6. decide between prompting, RAG, pivoting and fine-tuning for a
#    Kazakh-language product.
#
# {: .note }
# The sample sentences here are deliberately short and plain. **Replace them
# with your own text** — exercise 1 asks you to, and your measurements matter
# far more than mine.

# %%
# --- Setup: works on your laptop AND in Google Colab ------------------------
import os
import subprocess
import sys
from pathlib import Path

if "google.colab" in sys.modules:
    if not Path("/content/ROBT613").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/ROBT613.git",
                        "/content/ROBT613"], check=True)
    os.chdir("/content/ROBT613")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-r", "requirements.txt"], check=True)

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / "src" / "qwen_workshop").is_dir())
sys.path.insert(0, str(REPO_ROOT / "src"))
print("repo root:", REPO_ROOT)

# %%
import re
from collections import Counter

# %% [markdown]
# ## 22.1 · The sentences we will measure
#
# Three short sentences meaning roughly the same thing. Plain language, no
# jargon, so nothing here depends on specialist vocabulary.

# %%
SAMPLES = {
    "English": "The weather forecast promises unusually warm temperatures tomorrow",
    "Russian": "Прогноз погоды обещает необычно тёплую температуру завтра",
    "Kazakh": "Ауа райы болжамы ертеңге әдеттен тыс жылы температура уәде етеді",
}
for language, text in SAMPLES.items():
    print(f"{language:<8} {len(text):>3} characters   {text}")

# %% [markdown]
# ## 22.2 · The floor: UTF-8 bytes
#
# Before any tokenizer is involved, there is a cost built into the encoding
# itself. ASCII letters are one byte. Cyrillic — Russian and Kazakh alike —
# is **two**.
#
# This is exact arithmetic, not an estimate, and it is the floor under
# everything that follows.

# %%
print(f"{'language':<9}{'chars':>7}{'bytes':>7}{'bytes/char':>12}")
for language, text in SAMPLES.items():
    letters = text.replace(" ", "")
    print(f"{language:<9}{len(letters):>7}{len(letters.encode('utf-8')):>7}"
          f"{len(letters.encode('utf-8')) / len(letters):>12.2f}")

# %% [markdown]
# Kazakh has a few letters Russian does not — **ә ғ қ ң ө ұ ү һ і** — and they
# are still two bytes each in UTF-8, so the floor is the same for both.
#
# A byte is not a token, so this does not *directly* set your bill. But it
# tells you the information is there in the text, and that any tokenizer
# charging you far more than this is charging you for its own design choices
# rather than for your content.

# %% [markdown]
# ## 22.3 · Why tokenizers do worse — built, not asserted
#
# The usual explanation is "tokenizers are trained mostly on English". True,
# and unsatisfying. Let's build one and watch it happen.
#
# **Byte-pair encoding** starts with single characters and repeatedly merges
# the most frequent adjacent pair. Thirty lines:

# %%
def train_bpe(corpus: str, n_merges: int) -> list[tuple[str, str]]:
    """Learn `n_merges` merge rules from `corpus`. This is all BPE is."""
    words = Counter(re.findall(r"\w+", corpus.lower()))
    vocab = {tuple(word): count for word, count in words.items()}
    merges: list[tuple[str, str]] = []

    for _ in range(n_merges):
        pairs: Counter = Counter()
        for symbols, freq in vocab.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)

        merged = {}
        for symbols, freq in vocab.items():
            out, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    out.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            merged[tuple(out)] = freq
        vocab = merged
    return merges


def encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Apply the learned merges to one word."""
    symbols = list(word)
    for a, b in merges:
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


def chars_per_token(text: str, merges) -> float:
    words = re.findall(r"\w+", text.lower())
    tokens = sum(len(encode(word, merges)) for word in words)
    characters = sum(len(word) for word in words)
    return characters / tokens


# %% [markdown]
# Now the corpus. A real tokenizer is trained on an enormous, overwhelmingly
# English crawl with a thin slice of everything else. We can mimic the
# *proportions* using this repository's own hand-written documentation as the
# English bulk — a few tens of thousands of words of genuine, varied English
# prose.
#
# One exclusion matters: `docs/notebooks/` holds machine-generated copies of
# these notebooks, *including this one*. Leave them in and the English test
# sentence below ends up in the training data, which is precisely the mistake
# this section is about to warn you against.

# %%
sources = [
    path
    for path in sorted((REPO_ROOT / "docs").rglob("*.md"))
    if "notebooks" not in path.parts          # generated; would leak the test text
]
english_corpus = " ".join(path.read_text(encoding="utf-8") for path in sources)
english_corpus = re.sub(r"[^A-Za-z\s]", " ", english_corpus)

# A realistic multilingual tokenizer does see some Cyrillic - just far less.
RUSSIAN_CORPUS = "здравствуйте меня зовут студент я живу в городе и работаю каждый день " * 20
KAZAKH_CORPUS = "сәлеметсіз бе менің атым студент мен қалада тұрамын және жұмыс істеймін " * 20

print(f"English sources        : {len(sources):>7,} files")
print(f"English training words : {len(english_corpus.split()):>7,}")
print(f"Russian training words : {len(RUSSIAN_CORPUS.split()):>7,}")
print(f"Kazakh  training words : {len(KAZAKH_CORPUS.split()):>7,}")
print(f"\nEnglish share: {len(english_corpus.split()) / (len(english_corpus.split()) + len(RUSSIAN_CORPUS.split()) + len(KAZAKH_CORPUS.split())):.1%}")

# %% [markdown]
# **The sentences in 22.1 appear nowhere in that corpus.** That matters: a
# tokenizer measured on its own training text looks far better than it is.
# (I made exactly that mistake while writing this notebook — the first version
# reported Kazakh as *cheaper* than English, because the test sentences were
# in the training data and BPE had simply memorised them.)
#
# Do not take my word for it. Check it, every run:

# %%
haystack = " ".join((english_corpus + " " + RUSSIAN_CORPUS + " " + KAZAKH_CORPUS).lower().split())
for language, text in SAMPLES.items():
    needle = " ".join(re.findall(r"\w+", text.lower()))
    leaked = needle in haystack
    print(f"{language:<9} held out: {'NO - LEAKED' if leaked else 'yes':<12}")
    assert not leaked, f"{language} test sentence is in the training corpus"

# %%
merges = train_bpe(english_corpus + " " + RUSSIAN_CORPUS + " " + KAZAKH_CORPUS, 2000)
print(f"learned {len(merges)} merge rules\n")

print(f"{'language':<9}{'chars/token':>13}{'cost vs English':>18}")
baseline = chars_per_token(SAMPLES["English"], merges)
for language, text in SAMPLES.items():
    ratio = chars_per_token(text, merges)
    print(f"{language:<9}{ratio:>13.2f}{baseline / ratio:>17.1f}x")

# %% [markdown]
# Roughly **three times the tokens** for the same sentence. Look at what the
# tokenizer actually does to each one:

# %%
for language, text in SAMPLES.items():
    first_word = re.findall(r"\w+", text.lower())[1]
    print(f"{language:<9} {first_word:<16} -> {encode(first_word, merges)}")

# %% [markdown]
# The English word survives whole. The Cyrillic words shatter into single
# letters and two-character fragments — the tokenizer learned almost no merges
# for them, because in a corpus that is ~99% English no Cyrillic pair is ever
# frequent enough to win one.
#
# That is the whole mechanism. Not malice, not a design flaw: a greedy
# algorithm optimising for the data it was given.

# %% [markdown]
# ## 22.4 · The part that should change how you think
#
# The obvious fix is "use a bigger vocabulary". Let's test it.

# %%
print(f"{'merges':>8}{'English':>10}{'Russian':>10}{'Kazakh':>10}{'RU cost':>10}{'KK cost':>10}")
for budget in (500, 1000, 2000, 4000):
    trained = train_bpe(english_corpus + " " + RUSSIAN_CORPUS + " " + KAZAKH_CORPUS, budget)
    rates = {lang: chars_per_token(text, trained) for lang, text in SAMPLES.items()}
    print(f"{budget:>8}{rates['English']:>10.2f}{rates['Russian']:>10.2f}"
          f"{rates['Kazakh']:>10.2f}"
          f"{rates['English'] / rates['Russian']:>9.1f}x"
          f"{rates['English'] / rates['Kazakh']:>9.1f}x")

# %% [markdown]
# **A bigger vocabulary makes the gap worse.** English improves steadily —
# 2.0 to 3.5 characters per token — while Russian and Kazakh barely move off
# 1.0. The multiplier grows from 2.0x to 3.2x.
#
# This is the opposite of what most people assume, and it follows directly
# from the mechanism: extra merges go to whatever is frequent, and what is
# frequent is English. **Scaling the vocabulary scales the advantage.**
#
# The real fix is not a bigger vocabulary; it is a corpus where your language
# is not a rounding error.
#
# > **Exercise 1.** Replace `SAMPLES` with three sentences of your own —
# > ideally a real paragraph from something you are building — and re-run
# > 22.3 and 22.4. Report your own multiplier. Then try a technical Kazakh
# > sentence with borrowed terms (`компьютер`, `университет`) and a purely
# > Kazakh one, and say which costs more and why.

# %% [markdown]
# ### An honest admission about this notebook
#
# I wanted to show you the other half: retrain with a **balanced** corpus and
# watch the gap close. I could not do it properly, because I did not have a
# large, varied Kazakh corpus to hand — only the repetitive sample above,
# which teaches BPE nothing that generalises.
#
# That failure is the lesson. **The scarcity of varied Kazakh text is the
# root cause of everything in this notebook.** It is why the tokenizer is
# unfair, why model quality is lower, and why fine-tuning is hard. Not the
# script, not the grammar — the quantity of text on the public internet.
#
# > **Exercise 2.** Fix what I could not. Assemble a varied Kazakh corpus —
# > Kazakh Wikipedia, open government documents, news archives, your own
# > writing — of at least 50,000 words, retrain the BPE with it balanced
# > against the English, and report what happens to all three ratios. Watch
# > English too: with a fixed merge budget, does English get *worse*?

# %% [markdown]
# ## 22.5 · Measuring the real tokenizer
#
# The BPE above explains the mechanism. For the number you will actually be
# billed for, measure Qwen3's own tokenizer — a ~10 MB download, no model
# weights needed.

# %%
try:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    print(f"{'language':<9}{'chars':>7}{'tokens':>8}{'chars/token':>13}{'cost':>8}")
    results = {}
    for language, text in SAMPLES.items():
        n_tokens = len(tokenizer.encode(text))
        results[language] = len(text) / n_tokens
        print(f"{language:<9}{len(text):>7}{n_tokens:>8}{results[language]:>13.2f}"
              f"{results['English'] / results[language]:>7.1f}x")

    print("\nhow Qwen3 splits one Kazakh word:")
    word = "температура"
    print(f"  {word} -> {[tokenizer.decode([t]) for t in tokenizer.encode(word)]}")
except ImportError:
    print("transformers is not installed, so this cell did not run.\n"
          "Install it and re-run: `pip install transformers`. The toy BPE above\n"
          "explains the mechanism; this cell gives you the number you actually pay.")

# %% [markdown]
# > **Honesty note.** The authors could not run the cell above — this notebook
# > was written in an environment without `transformers`. The toy-BPE results
# > in 22.3 and 22.4 **were** run and are reproducible; the real Qwen3 figures
# > are yours to produce. Notebook 01 runs the same measurement across more
# > languages, so if you did notebook 01 you already have them.
#
# 🔬 **What to expect:** a multiplier somewhere around **1.5–2.5x** for
# Russian and Kazakh against English. Lower than our toy BPE's 3x, because
# Qwen3's 151,936-token vocabulary is genuinely multilingual and does contain
# Cyrillic merges — but well above 1.0. **Write your number down**; the rest
# of this notebook uses it.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/ROBT613/master/docs/assets/diagrams/lang-tax.svg" alt="How the cost compounds for Cyrillic text: the same sentence, then two bytes per character in UTF-8, then two to three times the tokens, hitting context, cost and latency at once" width="100%">
#
# ## 22.6 · What the tax actually costs you
#
# Turn the multiplier into the three things you care about.

# %%
def language_budget(multiplier: float, window: int = 32_768,
                    price_per_million: float = 0.50) -> dict:
    """Translate a token multiplier into context, money and latency."""
    return {
        "effective window (chars)": int(window * 4 / multiplier),
        "cost per 1M chars": round(multiplier * price_per_million * 1e6 / 4 / 1e6, 2),
        "relative latency": round(multiplier, 2),
    }


print(f"{'':<22}{'English':>12}{'Kazakh (2x)':>14}{'Kazakh (3x)':>14}")
rows = {name: language_budget(m) for name, m in
        (("English", 1.0), ("Kazakh (2x)", 2.0), ("Kazakh (3x)", 3.0))}
for metric in rows["English"]:
    print(f"{metric:<22}{rows['English'][metric]:>12,}"
          f"{rows['Kazakh (2x)'][metric]:>14,}{rows['Kazakh (3x)'][metric]:>14,}")

# %% [markdown]
# Read the first row carefully. A 32k-token context window holds roughly
# **131,000 characters of English** and about **65,000 of Kazakh**. Your
# Kazakh users get *half the document*, silently, with no error message.
#
# The three consequences, in the order they will bite you:
#
# | | Effect |
# |---|---|
# | **Context** | Half the conversation, half the retrieved documents, half the chat history |
# | **Cost** | Twice the bill for the same content, on any per-token pricing |
# | **Latency** | Roughly twice the generation time — tokens are produced one at a time |
#
# The latency one is the most visible to users and the least discussed. A
# Kazakh answer of the same length takes about twice as long to stream.
#
# > **Exercise 3.** Take a real document you would put in a RAG system — a
# > university regulation, a contract, a manual — in both Kazakh and English
# > if you have both. Measure how much of each fits in a 4k window and in a
# > 32k window. Then answer the design question: does your chunking strategy
# > from notebook 09 still work, or do your Kazakh chunks need to be smaller?

# %% [markdown]
# ## 22.7 · Quality: measure it, do not take my word
#
# Everything above is arithmetic, so I can be confident about it. **Model
# quality in Kazakh is not arithmetic**, and I have not measured it, so I am
# not going to tell you a number.
#
# What I can give you is the harness to find out, which is more useful and is
# the same discipline as notebooks 09 and 20: write the test cases *first*,
# then measure.

# %%
QUALITY_PROBES = [
    ("grammar", "Does it produce well-formed Kazakh, or Russian-shaped Kazakh?"),
    ("agglutination", "Kazakh stacks suffixes. Does it get case and possession right?"),
    ("vocabulary", "Does it use Kazakh words or Russian loanwords where a Kazakh word exists?"),
    ("code-switching", "Asked in Kazakh, does it answer in Kazakh - or drift to Russian/English?"),
    ("local facts", "Astana vs Nur-Sultan, oblast names, local institutions"),
    ("numbers & dates", "Tenge amounts, Kazakh date formats, large numbers in words"),
    ("names", "Kazakh personal and place names - does it mangle or transliterate them?"),
    ("register", "Formal (сіз) versus informal (сен). Does it hold the right one?"),
    ("refusal", "Does it refuse in Kazakh, or switch to English to refuse?"),
    ("instruction following", "'Жауапты үш сөзбен бер' - does a length constraint survive?"),
]
for name, what in QUALITY_PROBES:
    print(f"  [{name:<22}] {what}")

# %% [markdown]
# **The code-switching probe is the one to run first.** The characteristic
# failure of a multilingual model on a lower-resource language is not
# producing bad Kazakh — it is quietly answering in Russian or English
# instead, which looks like success to an automated test checking only that
# the answer is correct.
#
# A trajectory check catches it, exactly as in notebook 20: assert the
# *script* of the answer, not only its content.

# %%
CYRILLIC = set("абвгдежзийклмнопрстуфхцчшщъыьэюяёәғқңөұүһі")
KAZAKH_ONLY = set("әғқңөұүһі")


def script_report(text: str) -> dict:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return {"cyrillic": 0.0, "kazakh-specific letters": 0, "latin": 0.0}
    cyrillic = sum(c in CYRILLIC for c in letters)
    return {
        "cyrillic": round(cyrillic / len(letters), 2),
        "kazakh-specific letters": sum(c in KAZAKH_ONLY for c in letters),
        "latin": round(sum(c.isascii() for c in letters) / len(letters), 2),
    }


print(f"{'sample':<34}{'cyrillic':>10}{'kk letters':>12}{'latin':>8}")
for label, text in [
    ("Kazakh answer", "Ауа райы бүгін жылы болады"),
    ("Russian answer to a Kazakh Q", "Сегодня погода будет тёплой"),
    ("English answer to a Kazakh Q", "The weather will be warm today"),
]:
    report = script_report(text)
    print(f"{label:<34}{report['cyrillic']:>10}"
          f"{report['kazakh-specific letters']:>12}{report['latin']:>8}")

# %% [markdown]
# Russian and Kazakh are both ~1.0 Cyrillic, so "is it Cyrillic?" does **not**
# distinguish them. The **Kazakh-specific letters** column does: ә ғ қ ң ө ұ ү
# һ і appear in almost any real Kazakh sentence and never in Russian.
#
# That is a cheap, dependency-free language check you can assert on, and it is
# the kind of thing an English-language tutorial will never tell you to build.
#
# > **Exercise 4.** Write ten test cases from the probe list above, run them
# > against `qwen3:0.6b` and `qwen3:8b`, and score them yourself. Use
# > `script_report` as an automatic check for code-switching and your own
# > judgement for the rest. **Publish your numbers** — a measured
# > Kazakh-language evaluation of an open model is genuinely scarce, and your
# > class could produce the best one that exists.

# %% [markdown]
# ## 22.8 · RAG in Kazakh
#
# Retrieval degrades for a lower-resource language in two compounding ways,
# and neither is obvious from the English tutorials.
#
# **1. Chunks hold less.** If you chunk by tokens, a Kazakh chunk contains
# roughly half the content of an English one. Your retrieved context is
# therefore half as informative at the same `k`.
#
# **2. Embeddings are weaker.** Embedding models are trained on the same
# skewed corpora. Semantic similarity in Kazakh is measurably noisier than in
# English, so recall@k drops — on top of the chunks already holding less.
#
# Four mitigations, in the order I would try them:
#
# | Approach | Why it helps | What it costs |
# |---|---|---|
# | **Chunk by characters, not tokens** | restores chunk content parity | nothing — do this today |
# | **Raise `k`** | compensates for weaker ranking | more context consumed, which you can least afford |
# | **Hybrid search** (notebook 08) | BM25 is script-agnostic and strong on names and terms | a little code |
# | **Translate the query** (pivot) | search an English index with a translated query | a translation call, and errors it introduces |
#
# Hybrid search is the one most people skip and should not. Kazakh is
# **agglutinative** — one stem takes many suffixes — so exact-match keyword
# search needs stemming to work well, but for names, institutions and
# borrowed technical terms BM25 is excellent and entirely unaffected by the
# embedding model's English bias.
#
# > **Exercise 5.** Take the RAG evaluation from notebook 09 and build a
# > Kazakh version: 10 questions over Kazakh documents, with the expected
# > source for each verified by reading the text. Measure recall@3 with
# > embeddings alone and with hybrid search. Report both. This is the single
# > most useful measurement in this notebook, and nobody has published it.

# %% [markdown]
# ## 22.9 · Four strategies, and when each is right
#
# | Strategy | Use when | Watch out for |
# |---|---|---|
# | **Prompt in Kazakh, accept the tax** | the model is good enough and volume is low | code-switching; test for it |
# | **Pivot through Russian** | Russian output is markedly better and your users accept it | it is a product decision, not a technical one — ask them |
# | **Pivot through English** (translate in, translate out) | the task is reasoning-heavy and the model is much stronger in English | two translation steps, two chances to lose meaning; terrible for anything where exact wording matters |
# | **Fine-tune on Kazakh** | you have 500+ good examples and a narrow task | it teaches *form*, not knowledge — notebook 11's rule still holds |
#
# ### On fine-tuning for Kazakh specifically
#
# Notebook 11's decision list applies unchanged, with two additions:
#
# * **LoRA will not fix the tokenizer.** The adapter changes behaviour, not
#   the vocabulary. Your Kazakh text costs the same after fine-tuning as
#   before. People expect otherwise and are disappointed.
# * **Data is your bottleneck, not compute.** Notebook 14 is more important
#   here than notebook 11. Fifty carefully written Kazakh examples beat a
#   thousand machine-translated ones, and machine-translated training data
#   teaches the model to produce translationese.
#
# Where Kazakh text can legitimately come from: Kazakh Wikipedia, open
# government publications, university materials you have the right to use,
# news archives with permission, and your own institution's documents. Check
# the licence on each — notebook 12's capstone and the
# [privacy guide](../docs/guides/privacy-and-safety.md) both apply.

# %% [markdown]
# ## 22.10 · What to do on Monday
#
# 1. **Measure your own multiplier** with the real tokenizer (22.5). One
#    number, five minutes, and every budget decision downstream depends on it.
# 2. **Chunk by characters, not tokens**, in any RAG system serving Kazakh.
# 3. **Add a script check** to your tests so code-switching cannot pass
#    silently (22.7).
# 4. **Halve your assumed context budget** for Kazakh users until you have
#    measured otherwise.
# 5. **Try hybrid search before you try fine-tuning.** It is a day's work
#    instead of a month's, and it addresses the failure you actually have.
#
# ## Checkpoint
#
# 1. How many bytes does a Cyrillic character take in UTF-8, and why is that
#    only a floor rather than the whole story?
# 2. Why does a tokenizer trained on English-dominant data split Kazakh into
#    single letters?
# 3. Does a **bigger** vocabulary narrow or widen the gap? Why?
# 4. Give the three practical costs of a 2x token multiplier.
# 5. Why does checking "is the answer Cyrillic?" fail to detect
#    Russian/Kazakh code-switching, and what works instead?
# 6. Name the two compounding reasons RAG degrades in Kazakh.
# 7. Will fine-tuning with LoRA reduce your token costs? Why not?
# 8. What is the root cause underneath all of this?
#
# ➡️ **Related:** [notebook 01](01_llm_foundations.ipynb) measures tokenization
# across more languages, [09](09_rag_over_your_notes.ipynb) is the RAG
# evaluation to adapt, and [14](14_building_a_training_dataset.ipynb) is the
# one that matters if you decide to fine-tune.
