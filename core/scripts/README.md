# `core/scripts/`

Runnable entry points and manual test harnesses. Nothing here is imported by library
code — these are things you execute.

## Contents

- **`test_script.py`** — parallel throughput harness.
- **`__init__.py`** — empty.

## `test_script.py`

Fires many identical LLM calls concurrently and writes each response to disk. It is a
load and stability check for the router, the fallback chain and the cost tracker under
concurrency — not a correctness test, and not a pytest suite.

Run from the project root so imports resolve:

```bash
python -m core.scripts.test_script
```

### What it does

`run_llm_call(index)` builds a fresh `FileSummaryPrompt`, wraps it in a `ModelRouter`,
calls `model_response`, and writes the result to `results/{index}.md` (creating
`results/` if needed). It catches every exception and **returns** it rather than raising,
which is the contract `parallel_execute` expects.

`main()` runs `total_calls = 1000` indices through `parallel_execute` with
`max_threads=20`, `max_retries=1`, `retry_timer=2`, then counts successes by checking
`isinstance(res, str)`.

### Read this before running it

- **It makes 1000 real API calls.** At the default `FileSummaryPrompt` model
  (`openai/o3-mini-2025-01-31`) that is real money. Lower `total_calls` first.
- **Retries multiply.** Each provider retries 3× internally, the router walks a 3-model
  chain, and `parallel_execute` adds `max_retries=1` on top. A fully failing call can
  make well over a dozen requests.
- **`max_threads=20`** is deliberate — the default of `0` would spawn one thread per
  item, i.e. 1000 threads. The inline comment explains it balances speed against API
  stability.
- **No rate limiting is configured.** `parallel_execute` accepts `max_req_per_min`;
  passing it is strongly advisable against a metered API.
- **`results/` is written relative to the current working directory**, so where output
  lands depends on where you ran it from.
- **`results/` is written relative to the current working directory**, so where output
  lands depends on where you ran it from.
- The cost tracker prints its full itemized summary at exit, which for 1000 calls is a
  very long table.

## Related scripts outside this directory

One lives at the project root rather than here:

- **`repro_stream.py`** — minimal streaming repro. Forces `stream = True` on a
  `FileSummaryPrompt`, then checks whether the response is a generator and drains it.
  Its chunk handling assumes Ollama's dict shape and prints a placeholder for anything
  else, so it is most useful against a local model.
- Note: The root `model_router.py` file previously present was an unused stub that
  shadowed `core.llm_models.router.ModelRouter` and was removed.

## Adding a script

- Guard the entry point with `if __name__ == "__main__":`.
- Get a logger via `get_logger("YourScript")` rather than using `print`, so output
  honours `LOGGING_MODE` and `LOGGING_LEVEL`.
- Resolve paths from `__file__` rather than the CWD if the script writes files.
- If it calls an LLM in a loop, pass `max_req_per_min` to `parallel_execute` and start
  with a small `total_calls`.
