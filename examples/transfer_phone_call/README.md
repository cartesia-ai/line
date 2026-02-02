# Transfer Phone Call Example

A voice agent that can navigate automated phone systems (IVR) using DTMF tones and transfer calls to other numbers.

## Overview

This example creates a phone assistant that:
- Navigates automated phone menus by pressing DTMF buttons (0-9, *, #)
- Transfers calls to other phone numbers
- Validates phone numbers before transferring

## Running the Example

```bash
cd v02/examples/transfer_phone_call
ANTHROPIC_API_KEY=your-key uv run python main.py
```

## Tools Used

### `send_dtmf`
Sends DTMF tones to navigate phone menus. Valid buttons: 0-9, *, #

```python
# When the user says "press 1 for sales"
send_dtmf(button="1")
```

### `transfer_call`
Transfers the call to another phone number. Validates E.164 format.

```python
# Transfer to a specific number
transfer_call(target_phone_number="+14155551234", message="Transferring you now...")
```

### `end_call`
Ends the call with an optional goodbye message.

## Example Conversation

```
Agent: Hello! I'm your phone assistant. I can help you navigate automated phone menus
       by pressing buttons, or transfer your call to another number. How can I help?

User: I need to get to customer support

Agent: I'll press 2 for customer support.
       [Agent uses send_dtmf with button="2"]

User: Actually, can you transfer me to +1-415-555-1234?

Agent: I'll transfer you to that number now.
       [Agent uses transfer_call with target_phone_number="+14155551234"]
```

## Key Concepts

- **`send_dtmf`**: Passthrough tool that yields `AgentSendDtmf` event
- **`transfer_call`**: Passthrough tool with phone number validation using `phonenumbers` library
- **`end_call`**: Passthrough tool to gracefully end the conversation
