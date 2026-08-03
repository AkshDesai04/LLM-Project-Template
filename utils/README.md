# `utils/`

Modular infrastructure utilities with no LLM knowledge: logging, secrets, file I/O, database connectors, document conversion, and parallel execution.

All utility modules follow two strict structural rules:
1. **200 Lines Limit:** Every Python file is under 200 lines long.
2. **Absolute Imports:** All internal import paths are absolute (e.g. `from utils.logging import get_logger`).

## Subpackage Directory Layout

| Subpackage | Main Components | Key Files / Docs |
|---|---|---|
| `utils.logging` | Logging framework (`NORMAL` file/console & `LAMBDA` stdout modes) | `logger.py`, `lambda_logger.py` |
| `utils.env` | Secrets manager (`LOCAL` .env & `AWS_SM` AWS Secrets Manager) | `env_ops.py`, `constants.py`, `aws_secrets.py` |
| `utils.db` | Database connectors, factory router & native/fallback vector search (SQLite, Postgres, MySQL) | [`db/README.md`](file:///e:/Code/Personal/LLM-Project-Template/utils/db/README.md), `base.py`, `helpers.py`, `router.py` |
| `utils.io` | Text, binary, prompt and CSV reader utilities | `file_ops.py` |
| `utils.concurrency` | Thread-pool fan-out with retries and rate limiting | `parallel_executor.py` |
| `utils.document` | Document/URL/media → Markdown conversion via MarkItDown | `markitdown.py` |

Import directly from top-level `utils` or specific subpackages:
```python
from utils import get_logger, get_secret, DatabaseRouter, parallel_execute
```
