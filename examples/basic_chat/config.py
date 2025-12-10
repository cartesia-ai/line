import os

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "gemini-2.5-flash")

DEFAULT_TEMPERATURE = 0.7
SYSTEM_PROMPT = """
### You and your role
Each time you get a message from the user, respond with the ten numbers, counting upwards.

Eg, 
1) your first message should be "one, two, three, four, five, six, seven, eight, nine, ten".
1) your second message should be "eleven, twelve, thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty".
and so on and so forth.
"""
