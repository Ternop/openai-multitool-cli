# OpenAI Multi-Tool CLI

A single-file Python CLI to work with OpenAI models from the terminal.
It supports four modes: interactive chat, structured data extraction, local-file summarization, and async batch processing.
It also includes retries, caching, logging, a small SQLite transcript store, and safe tool calling.

## Features
- **chat**: interactive conversation with live streaming output. Calculator tool is available via the Chat Completions fallback.
- **extract**: strict JSON that matches a Pydantic schema (Task list). Good for turning notes into structured data.
- **summarize**: RAG-lite over local files using a tiny TF-IDF retriever. Output includes bracketed citations like `[1]`, `[2]`.
- **batch**: process many prompts at once with concurrency and write results to JSONL.
- Config with `.env`, logging, small disk cache, and a SQLite history database.

## Project layout
```
openai-multitool-cli/
├─ text_completion_app.py      # CLI and logic in one file
├─ .env.example                # sample env file (do not put real keys here)
├─ .gitignore                  # keeps .env and cache out of git
└─ README.md
```

## Requirements
- Python 3.9 or newer
- Packages: `openai`, `python-dotenv`, `pydantic`

Install:
```bash
pip install openai python-dotenv pydantic
```

## Setup
1. Copy the example env and add your real key locally:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and set:
   ```
   OPENAI_API_KEY=sk-your-real-key
   ```
3. Optional:
   ```
   OPENAI_MODEL=gpt-4.1-mini
   LOG_LEVEL=INFO
   ```

## Quick start
```bash
python text_completion_app.py chat
```
Type `exit` or `quit` to leave the chat.

## Commands

### chat
Interactive chat with streaming output. Calculator tool is available through the Chat Completions fallback.
```bash
python text_completion_app.py chat
python text_completion_app.py chat --system "Be concise." --model gpt-4.1-mini
```
Try:
- `What is 17^2 + 3*4?`
- `Give me two lines on why caching helps API apps.`

### extract
Structured JSON that matches a Pydantic schema (TaskList).
```bash
python text_completion_app.py extract --text "Email Alice by Friday about Q3 deck (High). Assign Bob a bugfix due 2025-08-20 (Medium)."
```

### summarize
RAG-lite on local files with simple TF-IDF ranking. Produces a summary with bracketed citations.
```bash
python text_completion_app.py summarize "./docs/*.md" --query "Key risks and next steps"
```

### batch
Process many prompts concurrently and write results to JSONL.
```bash
python text_completion_app.py batch prompts.txt --concurrency 6
# outputs batch_outputs.jsonl
```

## How it works
- Uses the Responses API for streaming and non-streaming text.
- Uses Chat Completions fallback for tool calling with a safe calculator defined with `pydantic_function_tool`.
- `extract` uses a JSON Schema and validates results with Pydantic.
- RAG-lite: files are chunked, tokenized, ranked by a small TF-IDF score, then summarized with citations.
- Reliability: retries with backoff, small on-disk cache keyed by prompt and params, and a SQLite transcript store.

## Troubleshooting
- Missing key: ensure `.env` exists with `OPENAI_API_KEY=...`
- Windows paths with spaces: wrap paths in quotes
- Clear cache: delete the `.cache` folder

## License
MIT
