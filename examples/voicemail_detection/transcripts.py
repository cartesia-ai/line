"""Multi-turn conversation scenarios for the voicemail-detection comparison harness.

Each scenario is a short *conversation* — a sequence of successive user turns —
so the harness can measure detection accuracy AND the **per-turn latency** the
tool (Approach 1) or sidecar (Approach 2) adds.

The set is **deliberately weighted toward humans (~53 human vs ~15 voicemail)**,
and most humans are adversarial: people who answer with their name, who mention
"voicemail"/"message" while clearly live, call screeners, distracted/multitasking
pickups, callbacks, confused/hard-of-hearing callers, and terse non-native
phrasing. The point is to measure the **false-positive rate** — hanging up on a
real person — with enough samples to trust it, since that is the catastrophic
error.

- ``label``: ``"voicemail"`` (answering machine; leave a message + hang up) or
  ``"human"`` (live person; keep talking).
- ``category``: groups scenarios for the per-category breakdown.
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
    # VOICEMAIL (~15) — single greeting turn (a machine doesn't converse back).
    # =====================================================================
    # --- classic: explicit machine keywords ---
    Scenario(
        "vm_classic_named",
        "voicemail",
        "voicemail_classic",
        ["Hi, you've reached the voicemail of Alex Carter. Please leave a message after the tone."],
    ),
    Scenario(
        "vm_classic_unavailable",
        "voicemail",
        "voicemail_classic",
        [
            "The person you are trying to reach is not available. At the tone, please record your "
            "message. When you have finished recording, you may hang up."
        ],
    ),
    Scenario(
        "vm_classic_number",
        "voicemail",
        "voicemail_classic",
        [
            "You have reached 555-0142. No one is available to take your call. Please leave a message after the beep."
        ],
    ),
    Scenario(
        "vm_classic_personal",
        "voicemail",
        "voicemail_classic",
        [
            "Hi, this is Marcus. I can't take your call right now. Please leave a message and I'll call you back."
        ],
    ),
    # --- subtle: casual, no "leave a message" keyword ---
    Scenario("vm_subtle_dana", "voicemail", "voicemail_subtle", ["Hey, it's Dana. You know what to do."]),
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
    Scenario("vm_subtle_chris", "voicemail", "voicemail_subtle", ["Yo. It's Chris. Talk to me."]),
    # --- business / carrier ---
    Scenario(
        "vm_business_closed",
        "voicemail",
        "voicemail_business",
        [
            "Thank you for calling Brightwave Solutions. Our office is currently closed. "
            "Please leave a message and we'll return your call on the next business day."
        ],
    ),
    Scenario(
        "vm_business_team",
        "voicemail",
        "voicemail_business",
        [
            "You've reached the sales team at Northgate Supply. We're unable to take your call. Leave your details."
        ],
    ),
    Scenario(
        "vm_business_hours",
        "voicemail",
        "voicemail_business",
        [
            "Thank you for calling Lakeside Dental. Our hours are nine to five, Monday through Friday. Please leave a message."
        ],
    ),
    Scenario(
        "vm_carrier_google",
        "voicemail",
        "voicemail_carrier",
        [
            "The Google subscriber you have dialed is not available. Please record your message after the tone."
        ],
    ),
    Scenario(
        "vm_carrier_forwarded",
        "voicemail",
        "voicemail_carrier",
        [
            "Your call has been forwarded to an automated voice messaging system. The person you are trying to reach is not available."
        ],
    ),
    # --- terse ---
    Scenario("vm_terse_leave", "voicemail", "voicemail_terse", ["Leave a message."]),
    Scenario("vm_terse_tone", "voicemail", "voicemail_terse", ["Speak after the tone."]),
    # =====================================================================
    # HUMAN (~53) — multi-turn, heavily adversarial. FP measurement is the point.
    # =====================================================================
    # --- short, unambiguous live greetings ---
    Scenario("h_short_hello", "human", "human_short", ["Hello?", "Yeah, this is me. What's up?"]),
    Scenario("h_short_yeah", "human", "human_short", ["Yeah, hello?", "Who's this?"]),
    Scenario("h_short_hi", "human", "human_short", ["Hi?", "Sorry, who's calling?"]),
    Scenario("h_short_yep", "human", "human_short", ["Yep?", "What's up?"]),
    Scenario("h_short_speaking", "human", "human_short", ["Speaking.", "Go ahead."]),
    # --- answers with their name (mirrors a voicemail opening) ---
    Scenario(
        "h_name_sarah",
        "human",
        "human_name",
        ["Hello, this is Sarah speaking.", "Yes, that's me — what's this about?"],
    ),
    Scenario("h_name_daniel", "human", "human_name", ["Good afternoon, Daniel here.", "Who's calling?"]),
    Scenario("h_name_priya", "human", "human_name", ["Hi, this is Priya, who's this?", "Mm, okay, go on."]),
    Scenario("h_name_greg", "human", "human_name", ["This is Greg.", "What do you need?"]),
    Scenario("h_name_marcus", "human", "human_name", ["Hi, you've got Marcus.", "Yeah? What's up?"]),
    Scenario("h_name_karen", "human", "human_name", ["Karen speaking.", "How can I help you?"]),
    # --- mentions voicemail / message but is live (top FP trap) ---
    Scenario(
        "h_msg_got",
        "human",
        "human_message_word",
        ["Oh hey, I got your voicemail earlier — what's up?", "Right, the order. Go on."],
    ),
    Scenario(
        "h_msg_full",
        "human",
        "human_message_word",
        ["Sorry, my voicemail's full, good thing you caught me.", "What did you need?"],
    ),
    Scenario(
        "h_msg_missed",
        "human",
        "human_message_word",
        ["Hey, did you leave me a message? I saw a missed call.", "Okay, what's it about?"],
    ),
    Scenario(
        "h_msg_about_to",
        "human",
        "human_message_word",
        ["I was just about to leave you a message!", "Anyway, I'm here now, go ahead."],
    ),
    Scenario(
        "h_msg_never_check",
        "human",
        "human_message_word",
        ["Yeah I never check my voicemail, good you called.", "What's this regarding?"],
    ),
    Scenario(
        "h_msg_recording",
        "human",
        "human_message_word",
        ["Hang on, are you recording a message? No? Oh, you're real. Hi.", "What's up?"],
    ),
    # --- call screening / who is this ---
    Scenario(
        "h_screen_who", "human", "human_screening", ["Who's calling please?", "And what's this regarding?"]
    ),
    Scenario(
        "h_screen_speaking", "human", "human_screening", ["May I ask who's speaking?", "Okay, go ahead."]
    ),
    Scenario("h_screen_sales", "human", "human_screening", ["Is this a sales call?", "Fine, what is it?"]),
    Scenario(
        "h_screen_robocall",
        "human",
        "human_screening",
        ["If this is a robocall I'm hanging up.", "Oh, a person. Okay, what?"],
    ),
    Scenario(
        "h_screen_number", "human", "human_screening", ["How did you get this number?", "Hmph. Go on then."]
    ),
    # --- distracted / multitasking / bad signal ---
    Scenario("h_busy_hangon", "human", "human_busy", ["Hang on— okay, sorry, hi. You there?", "Go ahead."]),
    Scenario(
        "h_busy_onesec", "human", "human_busy", ["Give me one sec— okay, who's this?", "Right, what's up?"]
    ),
    Scenario(
        "h_busy_driving",
        "human",
        "human_busy",
        ["Sorry, I'm driving, you're on speaker. Who's this?", "Okay, quickly?"],
    ),
    Scenario(
        "h_busy_outside",
        "human",
        "human_busy",
        ["One sec, let me step outside... okay, hi.", "What did you need?"],
    ),
    Scenario(
        "h_busy_kids", "human", "human_busy", ["Sorry, kids are loud — say that again?", "Okay, go on."]
    ),
    Scenario(
        "h_busy_signal",
        "human",
        "human_busy",
        ["Ugh, hold on, bad signal— can you hear me now?", "Alright, what's up?"],
    ),
    # --- live business reception (not an IVR) ---
    Scenario(
        "h_biz_brightwave",
        "human",
        "human_business",
        ["Good morning, Brightwave Solutions, this is Jordan, how can I help?", "Sure, one moment."],
    ),
    Scenario(
        "h_biz_frontdesk", "human", "human_business", ["Front desk, this is Lena.", "Who did you need?"]
    ),
    Scenario(
        "h_biz_apex",
        "human",
        "human_business",
        ["Thanks for calling Apex Auto, this is Ray.", "What can I do for you?"],
    ),
    Scenario(
        "h_biz_northgate",
        "human",
        "human_business",
        ["Northgate Supply, how can I direct your call?", "One moment please."],
    ),
    Scenario(
        "h_biz_dental",
        "human",
        "human_business",
        ["Lakeside Dental, this is Mara speaking.", "How can I help?"],
    ),
    # --- returning a call / missed call ---
    Scenario(
        "h_cb_missed",
        "human",
        "human_callback",
        ["Hi, I got a missed call from this number?", "Oh, okay, what's it about?"],
    ),
    Scenario(
        "h_cb_justcalled",
        "human",
        "human_callback",
        ["Someone just called me from here?", "Right, go ahead."],
    ),
    Scenario(
        "h_cb_rang",
        "human",
        "human_callback",
        ["Yeah, you guys rang me a minute ago?", "Okay, what did you need?"],
    ),
    Scenario("h_cb_returning", "human", "human_callback", ["Returning a call — who's this?", "Mm, go on."]),
    Scenario(
        "h_cb_needme",
        "human",
        "human_callback",
        ["I have a missed call, did someone need me?", "Sure, what's up?"],
    ),
    # --- confused / hard of hearing ---
    Scenario(
        "h_conf_who",
        "human",
        "human_confused",
        ["Hello? Who is this now?", "Speak up, dear, I can't hear you.", "Oh. Okay, go on."],
    ),
    Scenario(
        "h_conf_hear",
        "human",
        "human_confused",
        ["Hello? ... I can't hear you very well.", "A little louder?"],
    ),
    Scenario("h_conf_what", "human", "human_confused", ["What? Who? Hello?", "Oh, hello, yes?"]),
    Scenario("h_conf_there", "human", "human_confused", ["Hmm? Is someone there?", "Oh, yes, hello."]),
    Scenario(
        "h_conf_pardon", "human", "human_confused", ["Pardon me, who did you say?", "Alright, go ahead."]
    ),
    # --- skeptical / spam-wary ---
    Scenario(
        "h_skep_exactly",
        "human",
        "human_skeptical",
        ["Yeah?... who's this exactly?", "Uh huh. And why are you calling?"],
    ),
    Scenario(
        "h_skep_recognize",
        "human",
        "human_skeptical",
        ["I don't recognize this number.", "Okay... what do you want?"],
    ),
    Scenario("h_skep_quick", "human", "human_skeptical", ["Make it quick, who is this?", "Fine."]),
    Scenario(
        "h_skep_dnc",
        "human",
        "human_skeptical",
        ["I'm on the do-not-call list, you know.", "...okay, what is it?"],
    ),
    Scenario("h_skep_spam", "human", "human_skeptical", ["This better not be spam.", "Alright, talking."]),
    # --- terse / non-native phrasing ---
    Scenario("h_terse_yes", "human", "human_terse", ["Yes? Hello?", "Who is this please?"]),
    Scenario("h_terse_speaking", "human", "human_terse", ["Hello, yes, speaking.", "Go ahead please."]),
    Scenario(
        "h_terse_calling_for",
        "human",
        "human_terse",
        ["Yes good morning. You are calling for?", "Okay tell me."],
    ),
    Scenario("h_terse_allo", "human", "human_terse", ["Allo? Yes?", "Who is this?"]),
    Scenario("h_terse_please", "human", "human_terse", ["Hello, yes please?", "What you need?"]),
]
