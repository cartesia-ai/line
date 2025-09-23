# DTMF Adder

A simple example of how to use DTMF with cartesia's SDK, by way of a story teller that a user can both talk to and also hear DTMF tones.

The storyteller will offer choices and confirm those choices with the user (both verbally and also sending the tone back to the user).

Example dialogue:

```
Agent: Hello! Press any button on the keypad or say 'start story'."
User: <says 'start story'>

Agent: You are Sir Reginald, a knight, at the annual royal banquet. Suddenly, the lights go out! Press 1 to draw your sword, or press 2 to shout, "What's happening?!
User: <presses 2>

Agent: <dialtone for 2>. You pressed 2. Panic erupts. You hear a struggle, then a chilling roar! Press 3 to cast a light spell, or press 4 to investigate the roar."
User: <says what kind of light spell?>

Agent: It is light spell, level 4 best used to fight demons. Press 3 to cast a light spell, or press 4 to investigate the roar. 
User: <press 4>

Agent: <dial tone for 4> You pressed 4. Sorry, that is not a valid choice. Press 3 to cast a light spell, or press 4 to investigate the roar.
User: <press 3>

Agent: <diatone for 3> You pressed 3. As you investigate a roar, you see a shadow emerge from behind the table. Press #-9 to capture it in your amulet or press 1-2-3 to run away.
User: ...
```


## Template Information

### Prerequisites

- [Cartesia account](https://play.cartesia.ai)
- [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `MODEL_ID` | Gemini model to use | gemini-2.5-flash |

### Use Cases

For DTMF use cases

### File Overview

```
├── main.py              # Entry point, event routing
├── chat_node.py         # Gemini chat node
├── config.py            # System prompt and model settings
├── cartesia.toml        # Cartesia deployment config
├── pyproject.toml       # Python project dependencies
└── uv.lock              # Dependency lock file
```

## Local Setup

Install the Cartesia CLI.
```zsh
curl -fsSL https://cartesia.sh | sh
cartesia auth login
cartesia auth status
```

### Run the Example

1. Set up your environment variables.
   ```zsh
   export GEMINI_API_KEY=your_api_key_here
   ```

2. Install dependencies and run.

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
   conda create -n basic-chat python=3.11 -y
   conda activate basic-chat
   pip install -e .
   PORT=8000 python main.py
   ```

4. Chat locally by running in a different terminal.
   ```zsh
   cartesia chat 8000
   ```

## Remote Deployment
Read the [Cartesia docs](https://docs.cartesia.ai/line/) to learn how to deploy templates to the Cartesia Line platform.
