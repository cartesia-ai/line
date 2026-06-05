"""Labeled sample transcripts for the voicemail-detection comparison harness.

Each entry is the opening line(s) heard when an outbound call connects — what the
agent must classify before responding.

- ``label`` is the ground truth for the binary decision the agent makes: should it
  treat this as a voicemail (leave a message + hang up) or not?
    - ``"voicemail"`` → an answering machine / recorded mailbox greeting.
    - ``"human"``     → a live person (continue the conversation).
- ``category`` adds granularity so the harness can report *where* each approach
  succeeds or fails, not just an aggregate score. Categories deliberately include
  hard, ambiguous, and adversarial cases (subtle keyword-free voicemails, live
  people who answer with their name or say the word "message", call screeners,
  and noisy partial ASR) that easy keyword-spotting gets wrong.
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class Sample:
    transcript: str
    label: Literal["voicemail", "human"]
    category: str


SAMPLES: List[Sample] = [
    # =====================================================================
    # VOICEMAIL
    # =====================================================================
    # --- classic: explicit machine keywords ("leave a message", "tone"). ---
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
    Sample(
        "You've reached Dr. Nguyen's office voicemail. Please leave a detailed message and we'll return your call.",
        "voicemail",
        "voicemail_classic",
    ),
    Sample(
        "Hi, this is Marcus. I'm not available to take your call right now. Please leave a message after the beep and I'll call you back.",
        "voicemail",
        "voicemail_classic",
    ),
    Sample(
        "Thanks for calling. I can't take your call at the moment, so please leave a message after the tone.",
        "voicemail",
        "voicemail_classic",
    ),
    Sample(
        "Hello, you've reached Jamie. Sorry I missed you — please leave your name, number, and a brief message.",
        "voicemail",
        "voicemail_classic",
    ),
    # --- subtle: casual greeting with NO "leave a message" keyword. The key
    #     differentiator — keyword spotting misses these (false negatives). ---
    Sample("Hey, it's Dana. You know what to do.", "voicemail", "voicemail_subtle"),
    Sample(
        "Hi, this is Mike. I can't get to my phone right now. Catch you later.",
        "voicemail",
        "voicemail_subtle",
    ),
    Sample("Sorry I missed you! Say something and I'll hit you back.", "voicemail", "voicemail_subtle"),
    Sample("Yo. It's Chris. Talk to me.", "voicemail", "voicemail_subtle"),
    Sample(
        "Hey, you've reached Sam. I'm probably screening — you know the drill.",
        "voicemail",
        "voicemail_subtle",
    ),
    Sample(
        "Hi! It's Olivia. I'm away from my phone, but go ahead and let me know what's up.",
        "voicemail",
        "voicemail_subtle",
    ),
    Sample("This is Ben. Not here right now. Do your thing.", "voicemail", "voicemail_subtle"),
    Sample(
        "Hey there, it's Taylor. Can't talk — say something and I'll get back to ya.",
        "voicemail",
        "voicemail_subtle",
    ),
    Sample("You've reached Priya. I'll call you back as soon as I can.", "voicemail", "voicemail_subtle"),
    Sample("It's Jordan. Do your thing and I'll ring you back.", "voicemail", "voicemail_subtle"),
    Sample(
        "Hi, this is the Anderson residence. We're not able to come to the phone right now, so go ahead and let us know you called.",
        "voicemail",
        "voicemail_subtle",
    ),
    Sample("Hey, it's me, obviously. You know what this is. Go.", "voicemail", "voicemail_subtle"),
    # --- business: company mailbox / after hours. ---
    Sample(
        "Thank you for calling Brightwave Solutions. Our office is currently closed. Please leave a message and we'll return your call on the next business day.",
        "voicemail",
        "voicemail_business",
    ),
    Sample(
        "You've reached the sales team at Northgate Supply. We're unable to take your call. Leave your details and we'll follow up.",
        "voicemail",
        "voicemail_business",
    ),
    Sample(
        "Thank you for calling Lakeside Dental. Our normal hours are nine to five, Monday through Friday. Please leave a message.",
        "voicemail",
        "voicemail_business",
    ),
    Sample(
        "You've reached the after-hours line for Summit Property Management. No one is available. Please leave a message.",
        "voicemail",
        "voicemail_business",
    ),
    Sample(
        "You have reached the offices of Hale and Whitman. Please leave a message and an attorney will return your call.",
        "voicemail",
        "voicemail_business",
    ),
    # --- carrier: default network greetings. ---
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
    Sample(
        "The wireless customer you are calling is not available. Please leave a message after the tone.",
        "voicemail",
        "voicemail_carrier",
    ),
    Sample(
        "The number you have dialed is unavailable. Please leave a message after the beep.",
        "voicemail",
        "voicemail_carrier",
    ),
    Sample(
        "The party you are trying to reach is not accepting calls at this time. At the tone, please record your message.",
        "voicemail",
        "voicemail_carrier",
    ),
    # --- terse: very short machine greetings. ---
    Sample("Leave a message.", "voicemail", "voicemail_terse"),
    Sample("You know what to do. Beep.", "voicemail", "voicemail_terse"),
    Sample("Not here. Leave it.", "voicemail", "voicemail_terse"),
    Sample("Speak after the tone.", "voicemail", "voicemail_terse"),
    # =====================================================================
    # HUMAN
    # =====================================================================
    # --- short, unambiguous live greetings. ---
    Sample("Hello?", "human", "human_short"),
    Sample("Yeah, hello?", "human", "human_short"),
    Sample("Hi, who's this?", "human", "human_short"),
    Sample("Hello, hello?", "human", "human_short"),
    Sample("Yep?", "human", "human_short"),
    Sample("Hi there?", "human", "human_short"),
    # --- answers with their name. Mirrors a voicemail's opening ("Hi, this is
    #     X" / "you've reached X") but is a live person. False-positive trap. ---
    Sample("Hello, this is Sarah speaking.", "human", "human_name"),
    Sample("Good afternoon, Daniel here.", "human", "human_name"),
    Sample("Hi, this is Priya — sorry, who am I speaking with?", "human", "human_name"),
    Sample("This is Greg.", "human", "human_name"),
    Sample("Hi, you've got Marcus.", "human", "human_name"),
    Sample("Hello, Karen speaking, how can I help?", "human", "human_name"),
    # --- contains machine-y words ("message", "voicemail", "beep") but is a
    #     live person. Pure keyword matching fires a false positive here. ---
    Sample("Oh hey, yeah I got your voicemail earlier — what's going on?", "human", "human_message_word"),
    Sample(
        "Hi! Sorry, you just missed my voicemail greeting. I'm actually here, go ahead.",
        "human",
        "human_message_word",
    ),
    Sample(
        "Hello? Sorry, leave a — no wait, I picked up. Hi, how can I help?", "human", "human_message_word"
    ),
    Sample("Hey, did you leave me a message? I saw a missed call. What's up?", "human", "human_message_word"),
    Sample(
        "Sorry, my voicemail's full — good thing you caught me. What do you need?",
        "human",
        "human_message_word",
    ),
    Sample(
        "Hi, I was about to leave you a message actually! Glad I caught you instead.",
        "human",
        "human_message_word",
    ),
    # --- distracted / multitasking / mid-conversation pickups. ---
    Sample("Hang on—okay, sorry about that, hi. You still there?", "human", "human_busy"),
    Sample("...yeah just give me one sec— okay hi, sorry, who's calling?", "human", "human_busy"),
    Sample("Hello, thanks for calling back, what did you need again?", "human", "human_busy"),
    Sample("One second— [muffled] —sorry, hi, go ahead.", "human", "human_busy"),
    Sample("Hey, can you hold on a — actually no, I'm good, what's up?", "human", "human_busy"),
    Sample("Sorry, I'm driving, you're on speaker. Who's this?", "human", "human_busy"),
    # --- a person answering at a business (live, not an IVR). ---
    Sample(
        "Good morning, Brightwave Solutions, this is Jordan, how can I help you?", "human", "human_business"
    ),
    Sample("Front desk, this is Lena.", "human", "human_business"),
    Sample("Thanks for calling Apex Auto, this is Ray, what can I do for you?", "human", "human_business"),
    Sample("Northgate Supply, how can I direct your call?", "human", "human_business"),
    Sample("Hi, Lakeside Dental, this is Mara speaking.", "human", "human_business"),
    # --- call screening: wary / asks who's calling. ---
    Sample("Who's calling please?", "human", "human_screening"),
    Sample("May I ask who's speaking?", "human", "human_screening"),
    Sample("Hi, before we start — is this a sales call?", "human", "human_screening"),
    Sample("Hello? If this is a robocall I'm hanging up.", "human", "human_screening"),
    Sample("Yeah, who is this and how'd you get this number?", "human", "human_screening"),
    # =====================================================================
    # EDGE — noisy / partial ASR (mixed labels). A conservative detector should
    # not hang up unless the machine evidence is clear.
    # =====================================================================
    Sample("...hel— [static] ...there? can you—", "human", "noisy_partial"),
    Sample("[noise] ...the tone... [garbled]", "voicemail", "noisy_partial"),
    Sample("uh, hi, hello, um", "human", "noisy_partial"),
    Sample("...leave a mess— [cuts out]", "voicemail", "noisy_partial"),
    Sample("[static] ...not available... please... [garbled]", "voicemail", "noisy_partial"),
    Sample("can— hear— me— [breaking up]", "human", "noisy_partial"),
]
