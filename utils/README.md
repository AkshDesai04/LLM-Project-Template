# `utils/`

Generic infrastructure with no LLM knowledge: logging, secrets, file IO, document
conversion and parallel execution. Anything that understands prompts or providers
belongs in `core/` instead.

## Contents

| File | Purpose | Logger name |
|---|---|---|
| `logger.py` | Logging honouring `LOGGING_MODE` / `LOGGING_LEVEL` | — |
| `env_ops.py` | Secret resolution honouring `KEY_LOCATION` | `EnvOps` |
| `file_ops.py` | Text, binary, prompt and CSV reads | `FileOps` |
| `parallel_executor.py` | Thread-pool fan-out with retries and rate limiting | `ParallelExecutor` |
| `markitdown_utils.py` | Document/URL/media → Markdown via MarkItDown | `MarkItDownUtils` |

`__init__.py` is empty. Import from the concrete module: `from utils.file_ops import read_csv`.

The dependency graph is a star. `logger.py` imports nothing local; the other four import
only `utils.logger`. None of them import each other, so they can be used independently.

---

## `logger.py`

The root of the package — every other module calls `get_logger`.

### `get_logger(name: str, level: Optional[int] = None)`

Returns a logger honouring two environment variables. An explicit `level` argument beats
`LOGGING_LEVEL`.

| `LOGGING_MODE` | Returns | Behaviour |
|---|---|---|
| `NORMAL` (default) | `logging.Logger` | Console handler plus a JSON file handler at `logs/logs_YYYYMMDD.log` |
| `LAMBDA` | `LambdaLogger` | Uses `print`, one JSON record per line, **no file handler** |

`LAMBDA` mode exists because Lambda's filesystem is read-only, so attaching a
`FileHandler` throws, and CloudWatch already captures stdout. `LambdaLogger` mirrors the
methods used across the project — `debug`, `info`, `warning`, `error`, `critical`,
`exception`, `log` — including `%`-style lazy interpolation and traceback capture, so
call sites need no changes.

**`LOGGING_LEVEL`** accepts names (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) or a
numeric value, and applies in both modes. Both variables raise `ValueError` on an
unrecognised value rather than failing silently.

Also exported: `get_logging_mode()`, `get_logging_level()`, `JsonFormatter`,
`LambdaLogger`.

> **Design constraint:** this module reads `os.environ` directly instead of calling
> `env_ops`, because `env_ops` imports it. Do not add that import back — it is circular.

---

## `env_ops.py`

Secret resolution, switchable between `.env` and AWS Secrets Manager.

### `get_secret(key_name, raise_error=True)` — use this one

The entry point every provider calls. Routes on `KEY_LOCATION`:

| `KEY_LOCATION` | Source |
|---|---|
| `LOCAL` (default) | `.env`, via `get_local_secret` |
| `AWS_SM` | AWS Secrets Manager, via `get_aws_secret` against the `SECRET_NAME` bundle |

Flipping one variable moves every lookup at once. `AWS_SM` without `SECRET_NAME` fails
with an explicit message rather than an opaque miss.

**Config vs secrets.** Only credentials follow `KEY_LOCATION` — `DATABASE_URL`, the
provider keys, the provider URLs (`ROUTED_SECRET_NAMES`). Mode switches, project ids and
regions are always read from `.env`, because `SECRET_NAME` itself has to be readable
before Secrets Manager can be reached.

### Other functions

| Function | Purpose |
|---|---|
| `get_key_location()` | Validated `KEY_LOCATION` |
| `get_local_secret(key, raise_error=True)` | Direct `.env` read |
| `get_aws_secret(key, secret_name)` | One key from a Secrets Manager bundle |
| `get_secret_dict(secret_name)` | Whole bundle, cached in-process |
| `get_database_url(raise_error=True)` | `DATABASE_URL` through the switch |
| `get_gemini_key_type()` | `GEMINI_KEY` or `SERVICE_ACC_JSON` |
| `load_gemini_service_account_credentials()` | Vertex credentials from file or inline JSON |
| `resolve_gemini_project()` / `resolve_gemini_location()` | Vertex project and region, falling back to `GOOGLE_CLOUD_*` |
| `get_keys_dict()` | Legacy bulk key loader |

`boto3` is imported behind `try/except ImportError` with a `BOTO3_AVAILABLE` flag, so
the module works without AWS installed as long as you stay on `LOCAL`.

Secret-name caching (`_aws_secrets_cache`) lasts for the process lifetime, so a rotated
secret needs a restart.

See `.env.example` at the project root for the full variable list.

---

## `file_ops.py`

Four small readers. All log to `FileOps` and raise `FileNotFoundError` with the message
`File not found: {path}` when the file is missing.

| Function | Returns | Notes |
|---|---|---|
| `read_file(path)` | `str` | UTF-8, **`.strip()`ped** |
| `get_file(path)` | `bytes` | Not stripped |
| `read_prompt(stem)` | `str` | Loads `core/prompts/{stem}.txt` |
| `read_csv(path)` | `List[Dict[str, str]]` | `csv.DictReader`, fully materialised |

**`read_prompt`** resolves the project root from `__file__`, so it is CWD-independent.
Pass the bare stem — no directory, no `.txt`. The parameter is named `prompt_title`.

**`read_csv`** returns one dict per row keyed by the header. `cost_tracker.py` is the main consumer, reading `assets/model_pricing.csv`.

---

## `parallel_executor.py`

Thread-based fan-out for I/O-bound work.

### `parallel_execute(target_function, data, max_threads=0, max_req_per_min=None, max_retries=0, retry_timer=0) -> List[Any]`

```python
from utils.parallel_executor import parallel_execute

results = parallel_execute(
    target_function=fetch,
    data=[(1, 'a'), (2, 'b')],
    max_threads=20,
    max_req_per_min=600,
    max_retries=2,
    retry_timer=0.1,
)
for r in results:
    if isinstance(r, Exception):
        ...
```

**Results are returned in input order**, even though completion order differs — a
preallocated list is filled by index.

**Failures are returned, not raised.** After retries are exhausted, the `Exception`
object is placed at that index. Callers must branch on `isinstance(res, Exception)`.
The one ambiguity: a function that legitimately *returns* an exception is
indistinguishable from a failure.

**`max_threads` semantics:**

| Value | Workers |
|---|---|
| `0` (default) | one per data item, **unbounded** |
| `-1` | CPU count |
| `-n` | `n × CPU count` |
| `> 0` | exactly that many |

The default is a trap for large inputs — 1000 items means 1000 threads. Pass an explicit
value.

**Argument unpacking:** list/tuple items are unpacked as `*item`; anything else is passed
as a single argument. So a list intended as one argument will be spread. No `kwargs`
support.

**Retries:** total attempts are `max_retries + 1`.

**Rate limiting:** `max_req_per_min` engages `ThreadSafeRateLimiter`, which spaces
request starts by `60 / max_per_minute` seconds. It sleeps while holding the lock, which
is what makes spacing correct globally but also means it serialises the admission point
— throughput is capped at `max_req_per_min` regardless of worker count. Rate limiting is
applied once per task, not per retry.

Because this uses `ThreadPoolExecutor`, it helps I/O-bound work only; the GIL means
CPU-bound tasks will not speed up. `target_function` must be thread-safe.

---

## `markitdown_utils.py`

A logging-and-error-handling facade over Microsoft's MarkItDown, converting documents,
URLs and media into Markdown. Currently standalone — nothing else in the repo imports it.

### `MarkItDownUtils(llm_client=None, llm_model=None, docintel_endpoint=None, enable_plugins=True)`

- `llm_client` / `llm_model` — an OpenAI-compatible client used to caption images.
- `docintel_endpoint` — an Azure AI Document Intelligence endpoint; when set, parsing
  routes through Azure instead of local parsers.
- `enable_plugins` — **defaults to `True`, unlike MarkItDown's own default of `False`**,
  so installed plugins load automatically.

### Methods

| Method | Notes |
|---|---|
| `convert(source)` | Generic; files, URLs, anything MarkItDown accepts. No validation. |
| `convert_local(path)` | Refuses to fetch remote content; raises `FileNotFoundError` if missing. |
| `convert_url(url)` | Only method with input validation — see below. Handles HTML and YouTube transcripts. |
| `convert_stream(stream, file_extension)` | For in-memory binary data; extension is a format hint like `'.pdf'`. |
| `convert_image(path, describe=True)` | Specialized image conversion method. |
| `convert_audio(path)` | Functionally identical to `convert`; exists for readability. |

All methods return `result.text_content` and re-raise on failure after logging. None
return `None`.

**`convert_url` SSRF guard** checks schemes against `ALLOWED_URL_SCHEMES` (`http`/`https`)
and uses Python's `ipaddress` module to block private (`is_private`), loopback (`is_loopback`),
link-local (`is_link_local`), and reserved (`is_reserved`) IP addresses as well as `BLOCKED_HOSTNAMES`
(`localhost`, `0.0.0.0`, `[::]`).

**Hard dependency:** `from markitdown import MarkItDown` is a top-level import with no
guard, so importing this module fails if `markitdown` is absent.

The `__main__` block at the bottom references `./tests/test_files/0.pdf`, which does not
exist in this repo, and uses a CWD-relative path — it will not run as written.
