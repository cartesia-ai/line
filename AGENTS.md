# Project: line

## Development

- Use `uv run` to run Python commands (e.g., `uv run python -m pytest ...`)

## Code style

- **No inheritance.** Use abstract interfaces (Protocol classes) for type enforcement and composition for implementation sharing. Never use class inheritance for code reuse.

## Credentials

- Tests use fakes for LLM providers, tools and the Cartesia API. They must pass
  without any API key set; do not add a test that needs a real one.
- Do not read `.env` files or print environment variables. Examples load keys with
  `os.getenv` and nothing else. Never write a real-looking key into an example, test
  or README (use `your-key` style placeholders).
- To try an example against real providers, give the developer the command (for
  example `GEMINI_API_KEY=... uv run python main.py`) and let them run it. Do not
  run it with keys you found.
