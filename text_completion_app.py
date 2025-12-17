#!/usr/bin/env python3
"""
OpenAI Multi-Tool CLI
Author: Andrew Ternopolsky

What it is:
- One-file, production-style CLI for OpenAI models
- Modes (subcommands): chat, extract, summarize, batch
- Chat streams responses and supports calculator tool calls
- Extract returns strict JSON that matches a Pydantic schema
- Summarize does small local RAG-lite with TF-IDF ranking
- Batch sends many prompts concurrently and writes JSONL
- Includes caching, retries, logging, and a SQLite transcript store

Deps:
    pip install openai python-dotenv pydantic
Env:
    OPENAI_API_KEY must be set (env var or .env file)
"""

import argparse
import asyncio
import functools
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import sys
import textwrap
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional .env support
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# OpenAI SDK v1 style
try:
    from openai import OpenAI, pydantic_function_tool
except Exception:
    print("OpenAI SDK is required. Install with: pip install openai python-dotenv pydantic", file=sys.stderr)
    raise

# Pydantic for structured outputs
try:
    from pydantic import BaseModel, Field, ValidationError
except Exception:
    print("Pydantic is required. Install with: pip install pydantic", file=sys.stderr)
    raise

# Config and logging

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR = CACHE_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
DB_PATH = CACHE_DIR / "history.sqlite3"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai_cli")

# Retry decorator

def with_retries(max_attempts=4, base_delay=0.75):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error("Max attempts reached. Last error: %s", e)
                        raise
                    sleep_for = base_delay * \
                        (2 ** (attempt - 1)) + random.random() * 0.3
                    logger.warning(
                        "Retrying after error: %s (attempt %d/%d)", e, attempt, max_attempts)
                    time.sleep(sleep_for)
        return wrapper
    return deco

# Disk cache


def _hash_key(data: Dict[str, Any]) -> str:
    try:
        blob = json.dumps(data, sort_keys=True).encode("utf-8")
    except TypeError:
        # If something is not serializable, fall back to a string version
        blob = str(data).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def cache_get(namespace: str, key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    h = _hash_key(key)
    path = CACHE_DIR / f"{namespace}_{h}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def cache_set(namespace: str, key: Dict[str, Any], value: Dict[str, Any]) -> None:
    h = _hash_key(key)
    path = CACHE_DIR / f"{namespace}_{h}.json"
    try:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    except TypeError:
        # Makes value JSON safe
        def default(o):
            if hasattr(o, "model_dump"):
                return o.model_dump()
            return str(o)
        path.write_text(json.dumps(value, default=default,
                        ensure_ascii=False, indent=2))

# TF-IDF retrieval for local files (RAG-lite)

TOKEN_SPLIT = re.compile(r"[A-Za-z0-9_']+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_SPLIT.findall(text)]


@dataclass
class DocChunk:
    path: str
    text: str
    tokens: List[str]
    tf: Counter


def load_and_chunk_files(paths: List[str], max_chunk_chars: int = 1500) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        paragraphs = [para.strip()
                      for para in text.split("\n\n") if para.strip()]
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 <= max_chunk_chars:
                buf += ("\n\n" if buf else "") + para
            else:
                if buf:
                    toks = tokenize(buf)
                    chunks.append(DocChunk(path=p, text=buf,
                                  tokens=toks, tf=Counter(toks)))
                buf = para
        if buf:
            toks = tokenize(buf)
            chunks.append(DocChunk(path=p, text=buf,
                          tokens=toks, tf=Counter(toks)))
    return chunks


def rank_chunks(query: str, chunks: List[DocChunk], top_k: int = 6) -> List[DocChunk]:
    q_toks = tokenize(query)
    if not q_toks:
        return chunks[:top_k]
    N = len(chunks)
    df: Counter = Counter()
    for ch in chunks:
        df.update(set(ch.tokens))
    idf = {t: (1.0 + (N / (1 + df[t]))) for t in df}
    q_tf = Counter(q_toks)
    scores: List[Tuple[float, DocChunk]] = []
    for ch in chunks:
        score = 0.0
        for t, qcnt in q_tf.items():
            score += qcnt * idf.get(t, 1.0) * ch.tf.get(t, 0)
        if score > 0:
            scores.append((score, ch))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scores[:top_k]]

# Structured outputs schema

class Task(BaseModel):
    title: str = Field(..., description="Concise task name")
    due_date: Optional[str] = Field(
        None, description="Due date ISO string if present")
    priority: Optional[str] = Field(None, description="Low, Medium, or High")
    owner: Optional[str] = Field(None, description="Person responsible")


class TaskList(BaseModel):
    tasks: List[Task]


def json_schema_for(model_cls: Any) -> Dict[str, Any]:
    schema = model_cls.model_json_schema()
    return {
        "name": model_cls.__name__,
        "schema": schema,
        "strict": True,
    }

# Tool calling: safe calculator for Chat Completions

class CalcArgs(BaseModel):
    expression: str = Field(..., description="Math expression like 2*(3+4)^2")


def safe_eval_math(expr: str) -> str:
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\%\(\)\^eE]+", expr):
        raise ValueError("Expression contains unsupported characters.")
    expr = expr.replace("^", "**")
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        raise ValueError(f"Bad expression: {e}")


# pydantic_function_tool produces the correct shape for Chat Completions
CALCULATOR_TOOL = pydantic_function_tool(
    CalcArgs,
    name="calculator",
    description="Evaluate a numeric math expression. Use for arithmetic."
)

# SQLite transcript store

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at INTEGER NOT NULL,
            system TEXT,
            model TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(conv_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()


def create_conversation(system: str, model: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO conversations(created_at, system, model) VALUES (?, ?, ?)", (int(
        time.time()), system, model))
    conv_id = cur.lastrowid
    conn.commit()
    conn.close()
    return conv_id


def append_message(conv_id: int, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO messages(conv_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (conv_id, role, content, int(time.time())))
    conn.commit()
    conn.close()

# OpenAI client wrapper

class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: str = "You are a helpful assistant."):
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY in environment.")
        self.client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt

    @with_retries()
    def respond(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_output_tokens: int = 400,
                tools: Optional[List[Any]] = None,
                response_format: Optional[Dict[str, Any]] = None,
                stream: bool = False) -> Dict[str, Any]:
        """
        Streaming uses Responses API without tools.
        Non-streaming uses Responses API without tools.
        Tool calling happens only in Chat Completions fallback for compatibility.
        """
        # Keep cache key simple and safe
        key = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_format": response_format or {},
            "stream": bool(stream),
        }
        cached = cache_get("responses", key)
        if cached and not stream:
            return cached

        try:
            if stream:
                # Streaming text only. No tools. No response_format.
                with self.client.responses.stream(
                    model=self.model,
                    input=messages,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ) as stream_obj:
                    text_parts: List[str] = []
                    start_time = time.time()
                    print("\n", end="", flush=True)
                    for event in stream_obj:
                        if event.type == "response.output_text.delta":
                            s = event.delta
                            text_parts.append(s)
                            print(s, end="", flush=True)
                        elif event.type == "error":
                            raise RuntimeError(event.error)
                    print("\n")
                    total_time = time.time() - start_time
                    content = "".join(text_parts)
                    result = {"content": content, "metrics": {
                        "latency_sec": total_time}}
                    cache_set("responses", key, result)
                    return result
            else:
                kwargs = dict(
                    model=self.model,
                    input=messages,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                if response_format:
                    kwargs["response_format"] = response_format

                resp = self.client.responses.create(**kwargs)

                content = ""
                tool_calls: List[Dict[str, Any]] = []
                for item in resp.output or []:
                    if item.type == "output_text":
                        content += item.text
                    elif item.type == "function_call":
                        tool_calls.append(
                            {"name": item.name, "arguments": item.arguments})
                result = {"content": content, "tool_calls": tool_calls}
                cache_set("responses", key, result)
                return result

        except Exception as e:
            # Fallback to Chat Completions
            logger.warning(
                "Responses API failed, falling back to Chat Completions: %s", e)
            cc = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt}] + messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
                tools=tools if tools else None
            )
            content = cc.choices[0].message.content or ""
            tool_calls = []
            if cc.choices[0].message.tool_calls:
                for t in cc.choices[0].message.tool_calls:
                    tool_calls.append(
                        {"name": t.function.name, "arguments": t.function.arguments})
            # Makes usage JSON safe
            usage_dict = None
            try:
                usage_obj = getattr(cc, "usage", None)
                if hasattr(usage_obj, "model_dump"):
                    usage_dict = usage_obj.model_dump()
                elif usage_obj is not None:
                    usage_dict = dict(usage_obj)
            except Exception:
                usage_dict = None

            result = {"content": content,
                      "tool_calls": tool_calls, "usage": usage_dict}
            cache_set("responses", key, result)
            return result

# Subcommands

def subcommand_chat(args):
    """Interactive chat with streaming output and calculator tool via fallback."""
    init_db()
    client = LLMClient(model=args.model, system_prompt=args.system)
    conv_id = create_conversation(args.system, args.model)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": args.system}]

    print("=== Interactive Chat. Type 'exit' to quit. ===")
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        messages.append({"role": "user", "content": user})
        append_message(conv_id, "user", user)

        # First attempt: stream text
        client.respond(
            messages=messages,
            temperature=args.temperature,
            max_output_tokens=args.max_tokens,
            stream=True
        )

        # Non-streaming call to inspect tool calls in fallback
        result_full = client.respond(
            messages=messages,
            temperature=args.temperature,
            max_output_tokens=args.max_tokens,
            tools=[CALCULATOR_TOOL],
            stream=False
        )

        # If tool calls were requested, execute and then get a final streamed answer
        if result_full.get("tool_calls"):
            for tool in result_full["tool_calls"]:
                if tool["name"] == "calculator":
                    try:
                        # tool["arguments"] is a JSON string
                        args_json = json.loads(tool["arguments"]) if isinstance(
                            tool["arguments"], str) else tool["arguments"]
                        expr = args_json.get("expression", "")
                        value = safe_eval_math(expr)
                        tool_msg = {
                            "role": "tool",
                            "content": json.dumps({"result": value}),
                            "name": "calculator"
                        }
                        messages.append(tool_msg)
                    except Exception as e:
                        messages.append({"role": "tool", "content": json.dumps(
                            {"error": str(e)}), "name": "calculator"})
            # Asks again to produce the final answer that uses the tool results
            final = client.respond(
                messages=messages,
                temperature=args.temperature,
                max_output_tokens=args.max_tokens,
                stream=True
            )
            append_message(conv_id, "assistant", final.get("content", ""))
        else:
            append_message(conv_id, "assistant",
                           result_full.get("content", ""))


def subcommand_extract(args):
    """Extract structured tasks as strict JSON."""
    client = LLMClient(model=args.model)
    schema = json_schema_for(TaskList)
    messages = [
        {"role": "system", "content": "Extract tasks from user text. If none are present return an empty list."},
        {"role": "user", "content": args.text},
    ]
    result = client.respond(
        messages=messages,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
        response_format={"type": "json_schema", "json_schema": schema},
        stream=False
    )
    raw = result.get("content", "").strip()
    try:
        data = TaskList.model_validate_json(raw)
        print(json.dumps(data.model_dump(), indent=2, ensure_ascii=False))
    except ValidationError:
        print("Model returned invalid JSON for the schema. Raw output follows:\n")
        print(raw)


def subcommand_summarize(args):
    """Summarize local files with simple TF-IDF retrieval."""
    client = LLMClient(model=args.model)
    files: List[str] = []
    for pattern in args.paths:
        files.extend([str(p) for p in Path().glob(pattern)])
    if not files:
        print("No files matched.")
        return
    chunks = load_and_chunk_files(files)
    query = args.query or "Summarize the most important information for a busy engineer."
    top = rank_chunks(query, chunks, top_k=args.top_k)
    context = "\n\n---\n\n".join(f"[{i+1}] {t.path}\n{t.text}" for i,
                                 t in enumerate(top))
    messages = [
        {"role": "system",
            "content": "You create faithful summaries with inline bracket citations like [1], [2]."},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}\n\nWrite a concise summary with citations that reference the numbered chunks."}
    ]
    result = client.respond(
        messages=messages,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
        stream=False
    )
    print("\nSummary:\n")
    print(result.get("content", "").strip())


async def _batch_one(client: LLMClient, prompt: str, temperature: float, max_tokens: int) -> str:
    loop = asyncio.get_event_loop()

    def run_sync():
        r = client.respond(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_output_tokens=max_tokens,
            stream=False
        )
        return r.get("content", "").strip()
    return await loop.run_in_executor(None, run_sync)


def subcommand_batch(args):
    """Process many prompts concurrently and write JSONL."""
    client = LLMClient(model=args.model)
    prompts = [line.strip() for line in Path(args.file).read_text(
        encoding="utf-8").splitlines() if line.strip()]

    async def runner():
        sem = asyncio.Semaphore(args.concurrency)
        results: List[str] = [""] * len(prompts)

        async def go(i: int, p: str):
            async with sem:
                results[i] = await _batch_one(client, p, args.temperature, args.max_tokens)
        await asyncio.gather(*[go(i, p) for i, p in enumerate(prompts)])
        return results
    results = asyncio.run(runner())
    out_path = Path(args.output or "batch_outputs.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for p, r in zip(prompts, results):
            f.write(json.dumps({"prompt": p, "response": r},
                    ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} results to {out_path}")

# CLI

def build_parser():
    parser = argparse.ArgumentParser(
        description="AI Multi-Tool CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="OpenAI model to use")
    parser.add_argument("--temperature", type=float,
                        default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=400,
                        help="Max output tokens per call")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_chat = sub.add_parser(
        "chat", help="Interactive streaming chat with optional calculator")
    p_chat.add_argument(
        "--system", default="You are a helpful assistant.", help="System prompt")
    p_chat.set_defaults(func=subcommand_chat)

    p_extract = sub.add_parser(
        "extract", help="Extract tasks as structured JSON")
    p_extract.add_argument("--text", required=True, help="Text to analyze")
    p_extract.set_defaults(func=subcommand_extract)

    p_sum = sub.add_parser(
        "summarize", help="Summarize local files with TF-IDF retrieval")
    p_sum.add_argument("paths", nargs="+",
                       help="File globs, for example ./docs/*.md")
    p_sum.add_argument("--query", default=None, help="Focus query")
    p_sum.add_argument("--top-k", type=int, default=6,
                       help="Number of chunks to include")
    p_sum.set_defaults(func=subcommand_summarize)

    p_batch = sub.add_parser("batch", help="Async batch prompts from a file")
    p_batch.add_argument("file", help="Text file with one prompt per line")
    p_batch.add_argument("--concurrency", type=int,
                         default=6, help="Concurrent requests")
    p_batch.add_argument("--output", default=None, help="Output JSONL path")
    p_batch.set_defaults(func=subcommand_batch)

    return parser


def main():
    if not OPENAI_API_KEY:
        print("Set OPENAI_API_KEY in your environment or .env file.", file=sys.stderr)
        sys.exit(1)
    init_db()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
