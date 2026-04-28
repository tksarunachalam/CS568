# Grounded Resume Personalization Pipeline

A small, modular Python pipeline for the project **"Grounded Resume Personalization with LLMs: Balancing ATS Optimization and Recruiter Trust"**.

The pipeline:

1. Loads a resume and a job description from text files.
2. Asks an LLM to rewrite the resume — grounded in the original, with no fabrication.
3. Scores the original and modified resumes against the job description (TF-IDF or embeddings).
4. Optionally blends in a human rating to produce a final score.

Everything is hot-swappable: inputs, prompt template, LLM provider, and scorer.

## Layout

```
pipeline/
  main.py                        # entrypoint (argparse)
  config.py                      # defaults (paths, model, alpha, scorer)
  inputs/
    resume.txt                   # swap me
    job_description.txt          # swap me
  prompts/
    resume_prompt.txt            # swap me (system + user template, single file)
  llm/
    client.py                    # LLMClient ABC + AnthropicClient + EchoClient
    generator.py                 # ResumeGenerator (prompt -> LLM -> parsed JSON)
  scoring/
    similarity.py                # SimilarityScorer ABC + TfidfScorer + EmbeddingScorer
  evaluation/
    human_feedback.py            # combine_scores(ats, human, alpha)
  utils/
    io.py                        # read_text, write_text, load_json, save_json
```

## Setup

```bash
pip install -r pipeline/requirements.txt
```

Set **one** of these environment variables depending on which provider you want:

```bash
# OpenAI (default)
export OPENAI_API_KEY=sk-...

# Anthropic (alternative)
export ANTHROPIC_API_KEY=sk-ant-...
```

Optional, only if you want semantic embeddings:

```bash
pip install sentence-transformers
```

## Run

From the repo root (the directory that contains `pipeline/`):

```bash
# Default: OpenAI gpt-4o-mini
python -m pipeline.main

# Use Anthropic instead
python -m pipeline.main --provider anthropic

# Use a specific model
python -m pipeline.main --provider openai --model gpt-4o
python -m pipeline.main --provider anthropic --model claude-opus-4-5
```

Full example with all flags:

```bash
python -m pipeline.main \
  --provider openai \
  --model gpt-4o-mini \
  --resume pipeline/inputs/resume.txt \
  --jd pipeline/inputs/job_description.txt \
  --prompt pipeline/prompts/resume_prompt.txt \
  --scorer tfidf \
  --human-score 8 --alpha 0.7 \
  --output-dir pipeline/output
```

To smoke-test without burning API credits:

```bash
python -m pipeline.main --dry-run
```

This uses an `EchoClient` that returns the original resume verbatim, so the
ATS improvement will be `0` but every other piece of the pipeline runs.

Outputs are written to `pipeline/output/`:

- `result.json` — full structured result (scores, summary, keywords, notes).
- `modified_resume.txt` — the rewritten resume as plain text.

## Hot-swapping components

| What you want to change | How |
|---|---|
| Resume / JD | Drop a new file into `pipeline/inputs/` (or pass `--resume` / `--jd`). |
| Prompt | Edit / create a new file with `===SYSTEM===` and `===USER===` markers and pass `--prompt`. |
| LLM provider | Pass `--provider openai` or `--provider anthropic`. To add a new provider, subclass `LLMClient` in `pipeline/llm/client.py` and register it in `_build_client` in `main.py`. |
| Scorer | Add a new subclass of `SimilarityScorer` in `pipeline/scoring/similarity.py` and register it in `build_scorer`. |
| Human-score weight | `--alpha 0.5` (default 0.7). |

## Prompt format

`pipeline/prompts/resume_prompt.txt` is a single file with two sections:

```
===SYSTEM===
<system instructions: grounding constraints, ATS guidance, JSON schema>

===USER===
<user template referencing {resume} and {job_description}>
```

The generator splits on these markers, formats the user template, and
expects the model to return a single JSON object:

```json
{
  "modified_resume": "...",
  "changes_summary": ["..."],
  "keywords_added": ["..."],
  "groundedness_notes": "..."
}
```

If the model returns malformed JSON the raw response is preserved under
`generation._raw_response` in `result.json` and `parse_ok` is set to `false`.

## Design notes

- **Why an `LLMClient` ABC?** So the generator and the rest of the pipeline
  depend only on `generate(system_prompt, user_input) -> str`. Swapping
  providers (or mocking for tests) is a one-class change.
- **Why split the prompt with markers in one file?** Keeps system/user
  templates colocated and versioned together, while still allowing a single
  `--prompt` flag to swap both at once.
- **Why TF-IDF as the default scorer?** It mirrors how many real ATS systems
  weight keyword overlap and avoids any model download. Embeddings are a
  one-flag opt-in for semantic similarity.
- **Why separate `evaluation/`?** Keeps automated scoring (`scoring/`)
  distinct from human-in-the-loop blending so you can swap either side
  independently.

## Out of scope

No retries, no rate limiting, no batching, no UI, no DB — by design. This is
a research scaffold, not a production system.
