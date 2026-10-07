# Eager endpointing (preview)

Eager endpointing starts the first LLM completion when Ink predicts that the caller
has finished speaking. The SDK holds that completion until the platform confirms
the turn. A resume, changed transcript, changed prompt or configuration, or call
termination discards it. Discarded completions can still incur provider charges.

Enable it explicitly on your application:

```python
from line import VoiceAgentApp

# get_agent returns a direct LlmAgent instance.
app = VoiceAgentApp(get_agent=get_agent, eager_endpointing=True)
```

Cartesia must also enable the organization rollout flag and the call must use Ink
STT. Upgrade the platform and SDK together before testing. An older platform
ignores the capability and continues to deliver ordinary final turns.

## Supported scope

This first release supports a direct `LlmAgent` using the default HTTP chat
completion backend and the default run/cancel filters. WebSocket, Realtime, and
HTTP Responses backends, custom agents, wrappers, custom filters, and active
handoffs fall back to ordinary final-turn execution. Existing applications remain
unchanged unless they set `eager_endpointing=True`.

Only inference runs early. `process()`, conversation history updates, turn
counters, handoffs, and all locally dispatched tools—including custom and
background tools—still run after the confirmed final. Customer tools need no
`wait_for_commit()` calls. A general speculative custom-agent contract is outside
this preview; merely adding a new event to a filter does not enable speculation.

The candidate prompt includes a snapshot of existing conversation history plus
the predicted user text. At the final, the SDK rebuilds the normal request and
compares its messages, tool schemas, configuration, and provider options. New
background-tool results or other context changes therefore invalidate the
candidate. An accepted candidate supplies only the first completion; subsequent
tool-loop completions use the normal provider path.

## Protocol and lifecycle

`/chats` advertises `capabilities: ["eager_endpointing_v1"]` only when the app opts
in. This is independent of `cartesia_version`, which indicates older protocol
compatibility.

- `user_turn_eager`: `content` and a unique `event_id` for the prediction. Internally
  represented as `UserTurnLikelyEnded`; it is never committed to conversation
  history or passed to custom agent code in this release.
- `user_turn_resumed`: `eager_event_id` identifies the prediction to discard. A
  stale resume cannot cancel a newer prediction.
- A normal transcript `message` followed by `user_state: idle` with
  `turn_end_reason: confirmed` and `eager_event_id` confirms that prediction. The
  final retains its own `event_id`; emitted text, tools, and `response_end` refer
  to the final event as usual. Missing or mismatched correlation falls back.
- `user_state: idle` with `turn_end_reason: teardown` is terminal. It cancels the
  candidate and suppresses further processing until disconnect cleanup.

Only one uncommitted candidate is retained. Its queue holds at most 64 chunks;
backpressure pauses the reader until the final consumes them. Readers are owned
by the call, cancellation is independent of the WebSocket receive loop, and the
call permits at most four outstanding reader/cleanup tasks. Speculative provider
failures before acceptance fall back to normal generation. Failures after
acceptance follow the existing call error handling.

`llm_first_chunk_ms` and `llm_first_text_ms` retain the candidate's actual inference
timing when reused. Call latency still measures delivery on the confirmed turn.
No package version bump or live rollout is included in the feature PR.
