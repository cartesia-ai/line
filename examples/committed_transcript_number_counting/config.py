import os

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "gemini-2.5-flash")

DEFAULT_TEMPERATURE = 0.7
SYSTEM_PROMPT = """
### You and your role
For each message, count out the next 10 numbers starting from 1.

Eg:
1) "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
(user response)
2) "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"
(user response)
3) "twenty one", "twenty two", "twenty three", "twenty four", "twenty five", "twenty six", "twenty seven", "twenty eight", "twenty nine", "thirty"

and so on and so forth. Always start with the next number after the last number you said.
"""
