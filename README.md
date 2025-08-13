# OpenAI Multi-Tool CLI

A single-file Python CLI that lets you work with OpenAI models from the terminal.  
It has four modes: interactive chat, structured data extraction, local-file summarization, and async batch processing.  
It also includes retries, caching, logging, a simple SQLite transcript store, and safe tool calling.

## Features
- **chat**: interactive conversation with live streaming output. Calculator tool is available via the API fallback path.
- **extract**: returns **strict JSON** that matches a Pydantic schema (Task list). Good for turning notes into structured data.
- **summarize**: RAG-lite over local files using a tiny TF-IDF retriever. Outputs a summary with bracketed citations like `[1]`, `[2]`.
- **batch**: sends many prompts at once with concurrency and writes results to JSONL.

## Project layout
