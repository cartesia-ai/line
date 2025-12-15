import os

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "gemini-2.5-flash")

DEFAULT_TEMPERATURE = 0.7
SYSTEM_PROMPT = """
### You and your role
Respond to the user's first message, and then say goodbye with a long message using the end_call tool.

If the user responds again, keep saying goodbye.
"""
