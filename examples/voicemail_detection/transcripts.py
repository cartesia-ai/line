"""Multi-turn conversation scenarios for the voicemail-detection comparison harness.

Each scenario is a short *conversation* — a sequence of successive user turns —
so the harness can measure not just detection accuracy but also the **per-turn
latency** the tool (Approach 1) or sidecar (Approach 2) adds, including on
in-conversation turns after the opening line.

- ``label`` is the ground-truth call type:
    - ``"voicemail"`` → an answering machine. These are usually a single greeting
      turn; the agent should leave a message and hang up.
    - ``"human"``     → a live person. These run several back-and-forth turns so
      we can see whether detection keeps adding latency (or wrongly hangs up)
      once the conversation is clearly under way.
- ``category`` groups scenarios for the per-category accuracy breakdown, and
  includes adversarial cases (subtle keyword-free voicemails, people who answer
  with their name or say "message", call screeners).
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class Scenario:
    name: str
    label: Literal["voicemail", "human"]
    category: str
    turns: List[str]


SCENARIOS: List[Scenario] = [
    # =====================================================================
    # VOICEMAIL — single greeting turn (a machine doesn't converse back).
    # =====================================================================
    Scenario(
        "vm_classic",
        "voicemail",
        "voicemail_classic",
        ["Hi, you've reached the voicemail of Alex Carter. Please leave a message after the tone."],
    ),
    Scenario(
        "vm_classic_carrier",
        "voicemail",
        "voicemail_carrier",
        ["The wireless customer you are calling is not available. Please leave a message after the tone."],
    ),
    Scenario(
        "vm_business",
        "voicemail",
        "voicemail_business",
        [
            "Thank you for calling Brightwave Solutions. Our office is currently closed. "
            "Please leave a message and we'll return your call on the next business day."
        ],
    ),
    Scenario("vm_subtle_casual", "voicemail", "voicemail_subtle", ["Hey, it's Dana. You know what to do."]),
    Scenario(
        "vm_subtle_away",
        "voicemail",
        "voicemail_subtle",
        ["Hi, this is Mike. I can't get to my phone right now. Catch you later."],
    ),
    Scenario(
        "vm_subtle_callback",
        "voicemail",
        "voicemail_subtle",
        ["You've reached Priya. I'll call you back as soon as I can."],
    ),
    Scenario("vm_terse", "voicemail", "voicemail_terse", ["Leave a message."]),
    # A machine greeting that arrives in two chunks (greeting, then the beep line).
    Scenario(
        "vm_split_greeting",
        "voicemail",
        "voicemail_subtle",
        ["Hi, you've reached Jordan.", "Please record your message after the tone."],
    ),
    # =====================================================================
    # HUMAN — multi-turn conversations (measure in-conversation latency).
    # =====================================================================
    Scenario(
        "human_short",
        "human",
        "human_short",
        ["Hello?", "Yeah, this is me.", "Sure, what's it about?", "Okay, got it, thanks."],
    ),
    Scenario(
        "human_name",
        "human",
        "human_name",
        [
            "Hello, this is Sarah speaking.",
            "Yes, that's me — what can I do for you?",
            "Oh, the order. Right, it hasn't shown up yet.",
            "Alright, appreciate the call.",
        ],
    ),
    Scenario(
        "human_name_screen",
        "human",
        "human_name",
        ["Good afternoon, Daniel here.", "Who's calling?", "Mhm, go on.", "Sounds good."],
    ),
    Scenario(
        "human_message_word",
        "human",
        "human_message_word",
        [
            "Oh hey, yeah I got your voicemail earlier — what's going on?",
            "Right, the delivery. It's running late?",
            "No worries, thanks for letting me know.",
        ],
    ),
    Scenario(
        "human_message_word2",
        "human",
        "human_message_word",
        [
            "Sorry, my voicemail's full — good thing you caught me. What do you need?",
            "Yeah I can talk for a sec.",
            "Okay, that works for me.",
        ],
    ),
    Scenario(
        "human_busy",
        "human",
        "human_busy",
        [
            "Hang on—okay, sorry about that, hi. You still there?",
            "Yeah I can hear you now, go ahead.",
            "Got it. Thanks.",
        ],
    ),
    Scenario(
        "human_busy_driving",
        "human",
        "human_busy",
        ["Sorry, I'm driving, you're on speaker. Who's this?", "Okay, what's up?", "Fine, sounds good."],
    ),
    Scenario(
        "human_business_reception",
        "human",
        "human_business",
        [
            "Good morning, Brightwave Solutions, this is Jordan, how can I help you?",
            "Sure, let me check on that order for you.",
            "Yep, it shipped yesterday.",
            "You're welcome, bye.",
        ],
    ),
    Scenario(
        "human_business_frontdesk",
        "human",
        "human_business",
        ["Front desk, this is Lena.", "Uh huh, who did you need?", "One moment please."],
    ),
    Scenario(
        "human_screening_sales",
        "human",
        "human_screening",
        ["Hi, before we start — is this a sales call?", "Okay, what's it regarding?", "Alright, go ahead."],
    ),
    Scenario(
        "human_screening_wary",
        "human",
        "human_screening",
        ["Yeah, who is this and how'd you get this number?", "Hm. Okay, keep going.", "Fine."],
    ),
    Scenario(
        "human_short_who",
        "human",
        "human_short",
        ["Hi, who's this?", "Oh okay. What did you need?", "Sure.", "Thanks, bye."],
    ),
]
