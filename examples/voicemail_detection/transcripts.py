"""Labeled sample transcripts for the voicemail-detection comparison harness.

Each entry is the opening line(s) heard when an outbound call connects — what the
agent must classify before responding.

- ``label`` is the ground truth for the binary decision the agent makes: should it
  treat this as a voicemail (leave a message + hang up) or not?
    - ``"voicemail"`` → an answering machine / recorded mailbox greeting.
    - ``"human"``     → a live person (continue the conversation).
- ``category`` adds granularity so the harness can report *where* each approach
  succeeds or fails, not just an aggregate score. Categories deliberately include
  hard, ambiguous, and adversarial cases (e.g. a live person who answers with their
  name, or who says the word "message") that easy keyword-spotting gets wrong.
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class Sample:
    transcript: str
    label: Literal["voicemail", "human"]
    category: str


SAMPLES: List[Sample] = [
    # ---------------------------------------------------------------------
    # voicemail — classic: explicit machine keywords. The easy cases.
    # ---------------------------------------------------------------------
    Sample(
        "Hi, you've reached the voicemail of Alex Carter. Please leave a message after the tone.",
        "voicemail",
        "voicemail_classic",
    ),
    Sample(
        "The person you are trying to reach is not available. At the tone, please record your message. When you have finished recording, you may hang up.",
        "voicemail",
        "voicemail_classic",
    ),
    Sample(
        "You have reached 555-0142. No one is available to take your call. Please leave your name and number after the beep.",
        "voicemail",
        "voicemail_classic",
    ),
    # ---------------------------------------------------------------------
    # voicemail — subtle: a casual personal greeting with no "leave a message"
    # keyword. Keyword spotting tends to miss these (false negatives).
    # ---------------------------------------------------------------------
    Sample("Hey, it's Dana. You know what to do.", "voicemail", "voicemail_subtle"),
    Sample(
        "Hi, this is Mike. I can't get to my phone right now. Catch you later.",
        "voicemail",
        "voicemail_subtle",
    ),
    Sample(
        "Sorry I missed you! Do your thing after the beep and I'll hit you back.",
        "voicemail",
        "voicemail_subtle",
    ),
    Sample("Yo. It's Chris. Talk to me.", "voicemail", "voicemail_subtle"),
    # ---------------------------------------------------------------------
    # voicemail — business / carrier default greetings.
    # ---------------------------------------------------------------------
    Sample(
        "Thank you for calling Brightwave Solutions. Our office is currently closed. Please leave a message and we'll return your call on the next business day.",
        "voicemail",
        "voicemail_business",
    ),
    Sample(
        "The Google subscriber you have dialed is not available. Please record your message after the tone.",
        "voicemail",
        "voicemail_carrier",
    ),
    Sample(
        "Your call has been forwarded to an automated voice messaging system. The person you are trying to reach is not available.",
        "voicemail",
        "voicemail_carrier",
    ),
    # ---------------------------------------------------------------------
    # human — short, unambiguous live greetings.
    # ---------------------------------------------------------------------
    Sample("Hello?", "human", "human_short"),
    Sample("Yeah, hello?", "human", "human_short"),
    Sample("Hi, who's this?", "human", "human_short"),
    # ---------------------------------------------------------------------
    # human — answers with their name. Mirrors a voicemail's opening ("Hi, this
    # is X") but is a live person. A classic false-positive trap.
    # ---------------------------------------------------------------------
    Sample("Hello, this is Sarah speaking.", "human", "human_name"),
    Sample("Good afternoon, Daniel here.", "human", "human_name"),
    Sample("Hi, this is Priya — sorry, who am I speaking with?", "human", "human_name"),
    # ---------------------------------------------------------------------
    # human — contains machine-y words ("message", "voicemail", "beep") but is a
    # live person. Pure keyword matching fires a false positive here.
    # ---------------------------------------------------------------------
    Sample("Oh hey, yeah I got your voicemail earlier — what's going on?", "human", "human_message_word"),
    Sample(
        "Hi! Sorry, you just missed my voicemail greeting. I'm actually here, go ahead.",
        "human",
        "human_message_word",
    ),
    Sample(
        "Hello? Sorry, leave a — no wait, I picked up. Hi, how can I help?", "human", "human_message_word"
    ),
    # ---------------------------------------------------------------------
    # human — distracted / multitasking / mid-conversation pickups.
    # ---------------------------------------------------------------------
    Sample("Hang on—okay, sorry about that, hi. You still there?", "human", "human_busy"),
    Sample("...yeah just give me one sec— okay hi, sorry, who's calling?", "human", "human_busy"),
    Sample("Hello, thanks for calling back, what did you need again?", "human", "human_busy"),
    # ---------------------------------------------------------------------
    # human — a person answering at a business (live, not an IVR).
    # ---------------------------------------------------------------------
    Sample(
        "Good morning, Brightwave Solutions, this is Jordan, how can I help you?", "human", "human_business"
    ),
    Sample("Front desk, this is Lena.", "human", "human_business"),
    # ---------------------------------------------------------------------
    # edge — noisy / partial ASR. Ambiguous evidence; a conservative detector
    # should lean "human"/"unknown" rather than wrongly hang up.
    # ---------------------------------------------------------------------
    Sample("...hel— [static] ...there? can you—", "human", "noisy_partial"),
    Sample("[noise] ...the tone... [garbled]", "voicemail", "noisy_partial"),
    Sample("uh, hi, hello, um", "human", "noisy_partial"),
    # ---------------------------------------------------------------------
    # edge — long monologue with no reply pause (machine) vs a long but
    # interactive human greeting.
    # ---------------------------------------------------------------------
    Sample(
        "Hi there, this is the Anderson residence. We're not able to come to the phone right now, but your call is important to us, so please leave your name, number, and a brief message and someone will get back to you as soon as possible.",
        "voicemail",
        "voicemail_subtle",
    ),
    Sample(
        "Hi, yes, hello — is this about the appointment? Sorry, it's a little loud here, can you hear me okay? Go ahead.",
        "human",
        "human_busy",
    ),
]
