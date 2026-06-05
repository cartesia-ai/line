"""Labeled sample transcripts for the voicemail-detection comparison harness.

Each entry is the first thing the callee/line says when an outbound call connects.
``label`` is the ground truth: ``"voicemail"`` for an answering machine, ``"human"``
for a live person.
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class Sample:
    transcript: str
    label: Literal["voicemail", "human"]


SAMPLES: List[Sample] = [
    # --- Voicemail / answering machine greetings ---
    Sample("Hi, you've reached the voicemail of Alex. Please leave a message after the tone.", "voicemail"),
    Sample(
        "The person you are trying to reach is not available. At the tone, please record your message.",
        "voicemail",
    ),
    Sample(
        "You have reached 555-0142. No one is available to take your call. Please leave a message.",
        "voicemail",
    ),
    Sample(
        "Hey it's Jordan, sorry I missed you — leave a message and I'll call you back. Beep.", "voicemail"
    ),
    Sample(
        "Thank you for calling. Your call is very important to us. Please leave your name and number after the beep.",
        "voicemail",
    ),
    Sample(
        "This is the Google subscriber's voicemail. Please record your message after the tone.", "voicemail"
    ),
    Sample(
        "I'm not able to take your call right now, but if you leave a message I'll get back to you as soon as I can.",
        "voicemail",
    ),
    # --- Live human greetings ---
    Sample("Hello?", "human"),
    Sample("Hi, who's this?", "human"),
    Sample("Hey there, how can I help you?", "human"),
    Sample("Good morning, this is Sam speaking.", "human"),
    Sample("Yeah, hello? Who's calling?", "human"),
    Sample("Hi, sorry — can you hear me okay?", "human"),
    Sample("Hello, thanks for calling back, what did you need?", "human"),
]
