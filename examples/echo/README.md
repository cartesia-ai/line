# Echo

Voice agent that echoes back what users say with optional delay.

## Template Information

### Prerequisites

- [Cartesia account](https://play.cartesia.ai)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SLEEP_MS` | Delay in milliseconds before responding | 0 |

### Use Cases

Testing voice agent functionality, debugging speech recognition, simple echo interactions.

### File Overview

```
├── main.py              # Entry point, echo logic
├── cartesia.toml        # Cartesia deployment config
├── pyproject.toml       # Python project dependencies
└── uv.lock              # Dependency lock file (if present)
```

## Local Setup

Install the Cartesia CLI.
```zsh
curl -fsSL https://cartesia.sh | sh
cartesia auth login
cartesia auth status
```

### Run the Example

1. Install dependencies and run.

   **uv (recommended)**
   ```zsh
   PORT=8000 uv run python main.py
   ```

   **pip**
   ```zsh
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   PORT=8000 python main.py
   ```

   **conda**
   ```zsh
   conda create -n echo python=3.11 -y
   conda activate echo
   pip install -e .
   PORT=8000 python main.py
   ```

2. Chat locally by running in a different terminal.
   ```zsh
   cartesia chat 8000
   ```

## Remote Deployment

Read the [Cartesia docs](https://docs.cartesia.ai/line/) to learn how to deploy templates to the Cartesia Line platform.
