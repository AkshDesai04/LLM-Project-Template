# `core/prompts/`

Plain-text prompt bodies, kept out of Python so they can be edited without touching
code and reviewed as prose in a diff.

## Contents

- **`test_prompt.txt`** — the example used by `FileSummaryPrompt`. Currently a single
  line: `write the first 1000 words of lorem ipsum`.

There is no `__init__.py`; this is a data directory, not a package.

## How prompts are loaded

`read_prompt` in `utils/file_ops.py`:

```python
from utils.file_ops import read_prompt

text = read_prompt('test_prompt')   # loads core/prompts/test_prompt.txt
```

Three things to know about it:

- **Pass the bare stem.** No directory, no `.txt` — the function appends the extension
  and joins against this directory itself.
- **The path is resolved from `__file__`**, walking up from `utils/` to the project root,
  so it works regardless of the current working directory.
- **The content is `.strip()`ped**, so leading and trailing whitespace and the trailing
  newline are removed. Indentation *inside* the file is preserved.

A missing file raises `FileNotFoundError`.

## Timing gotcha

Prompts are normally loaded as a Pydantic field default:

```python
class FileSummaryPrompt(Base):
    prompt: str = read_prompt('test_prompt')
```

A field default is evaluated **once, at class-definition time** — meaning on import, not
per instance. Two consequences:

1. A missing or malformed prompt file breaks the **import**, so the failure surfaces at
   startup rather than at call time.
2. Editing a `.txt` file has no effect on a running process. Restart to pick up changes.

If you need per-call loading, call `read_prompt` inside your code instead of using it as
a default. Alternatively, set `prompt_path` on the module and let
`LLMProvider.prepare_module` read it lazily — it loads the file into `prompt` only when
`prompt` is empty. Note `prompt_path` is opt-in and is not declared on `Base`.

## Conventions

- One file per prompt, named after the module that uses it. `.txt` only — the extension
  is hardcoded.
- Use `snake_case` filenames; the stem is what you pass to `read_prompt`.
- Keep the reusable instruction body here and put per-request data in the call, rather
  than templating it into the file.
- Put role and tone in `system_prompt` on the module, not at the top of the prompt body.
- These files are plain text with no interpolation. Anything dynamic must be formatted in
  Python after loading.
