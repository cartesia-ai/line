# Voice research with FlashData and Cartesia Line

![FlashData](logo.svg)

Build a voice assistant that finds web evidence, compares news sources, discovers local businesses and retrieves existing YouTube transcripts. It asks for query scope and spending approval, calls FlashData once, and answers with source references. The example uses the existing hosted MCP service; it adds no FlashData backend or OAuth service.

## Prerequisites

- Python 3.10+, [uv](https://docs.astral.sh/uv/), and a Cartesia Line account/CLI for a real voice session.
- Your own [FlashData API key](https://flashdata.dev/app/api-keys), enabled for the query sources you intend to use. Prefer a dedicated key without download permission.
- A model-provider key and a Line-supported LiteLLM model identifier. Choose these according to your provider access; they are not supplied by this example.
- Installation is free. **FlashData data requests consume your own paid Credits**; [current pricing](https://flashdata.dev/pricing), model/voice charges and applicable storage costs are separate. Do not assume that installing this example authorizes paid calls.

## Environment Variables

Set these through your launching environment, secret manager or the Line platform's secret settings. Keep populated values out of Git, shell command arguments, chat and shared logs.

| Variable | Purpose |
| --- | --- |
| `FLASHDATA_API_KEY` | Your FlashData key, sent only in the `X-API-Key` header to `https://data.flashdata.dev/mcp`. |
| `LLM_API_KEY` | Your model-provider key, passed to Line's model client. |
| `LINE_MODEL` | A supported LiteLLM identifier, for example `openai/gpt-4o-mini` when available to your account. |

The FlashData key is not a model tool argument. Tool inputs and results are visible to the voice/model host; use non-sensitive research queries and review its retention settings. No storage credentials should enter this voice session.

## Install and run

From this directory:

```sh
uv sync
uv run python main.py
```

With the Cartesia CLI authenticated to your own account, connect locally:

```sh
cartesia chat 8000
```

Starting a real voice conversation can incur voice/model charges. These commands describe the user-operated workflow; they were not run against paid services during this contribution.

For the Line platform, use its [GitHub integration](https://docs.cartesia.ai/line/integrations/github) and select this example directory as the agent source. Supply the three environment variables in the platform configuration. The `main.py`/`VoiceAgentApp`, `pyproject.toml` and `cartesia.toml` layout follows the existing integration examples. No cloud deployment has been performed or claimed here.

## First research task

Ask: “Read FlashData prices without submitting a data query.” `flashdata_catalog` obtains the current tool schemas; `flashdata_call` with `get_pricing` is a free account read. Inspect the actual returned data before approving a query. A successful pricing read does not verify paid data retrieval.

Then, if you choose to spend Credits, approve one Google Search for battery recycling research: `google_search` with `{"query":"battery recycling research","gl":"us","hl":"en","num":10}`. Ask for the main finding and the sources supporting it. Expected structured fields are `results[0].organic` entries containing available `position`, `title`, `link` and `snippet`; actual fields and result counts can vary.

The agent discovers schemas instead of guessing parameters. The two Line tools expose 14 FlashData query capabilities plus `get_pricing`, `get_job` and `get_result`. Long requests run in the background across voice interruptions; a failed submission is never automatically retried by the adapter. Check Jobs before resubmitting an ambiguous timeout because it may already be charged. For async requests retain the returned job ID; completed MCP data is in `output`. For large results, preserve the returned chunk metadata and use `get_result` rather than repeating the paid query.

## Fifteen capability workflows

Each task guide contains concrete input JSON, structured output fields, processing steps and limits. These are 15 capability mappings within one integration example, not 15 marketplace listings. The last workflow runs outside the voice session to protect storage credentials.

| Task and tool | Workflow and API documentation |
| --- | --- |
| [Google Search](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/google-search/README.md) — `google_search` | Search Google for battery recycling research in the US, in English. Make one request for 10 results and return a table of position, title, source URL and snippet. [API documentation](https://flashdata.dev/docs/sources/google_search?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=google_search-task). |
| [Google News](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/google-news/README.md) — `google_news` | Find Google News coverage of battery recycling. Make one request for 10 articles, group them by publisher, and preserve each original date label and article link. [API documentation](https://flashdata.dev/docs/sources/google_news?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=google_news-task). |
| [Google Images](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/google-images/README.md) — `google_images` | Find 20 Google Images results for solar panel diagrams. Keep images with a returned width of at least 1000 pixels, and list each image URL, source page and dimensions. [API documentation](https://flashdata.dev/docs/sources/google_images?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=google_images-task). |
| [Google Shopping](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/google-shopping/README.md) — `google_shopping` | Search Google Shopping once for wireless headphones in the US. Return 10 requested offers with product identifiers, merchant names, original price labels and source URLs. [API documentation](https://flashdata.dev/docs/sources/google_shopping?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=google_shopping-task). |
| [Google Video Search](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/google-video-search/README.md) — `google_videos` | Search Google Videos once for battery recycling process demonstrations. Request 10 results and group the returned titles, URLs and available durations by website. [API documentation](https://flashdata.dev/docs/sources/google_videos?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=google_videos-task). |
| [Google Places](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/google-places/README.md) — `google_places` | Search once for coffee shops in Singapore. Request 20 places, then make a shortlist from returned ratings of at least 4.2 and at least 20 reviews, keeping the addresses. [API documentation](https://flashdata.dev/docs/sources/google_places?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=google_places-task). |
| [YouTube Search](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/youtube-search/README.md) — `youtube_search` | Find 10 YouTube videos about battery recycling with one standard search. Return the video IDs, watch URLs, channels and available durations. [API documentation](https://flashdata.dev/docs/sources/youtube_search?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=youtube_search-task). |
| [YouTube Extended Search](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/youtube-extended-search/README.md) — `youtube_search_extended` | Run one extended YouTube search for battery recycling with 100 requested results. Count the returned videos by channel and state the actual sample size. [API documentation](https://flashdata.dev/docs/sources/youtube_search_extended?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=youtube_search_extended-task). |
| [YouTube Metadata](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/youtube-metadata/README.md) — `youtube_metadata` | Read metadata for YouTube video dQw4w9WgXcQ once, including chapters but excluding formats. Turn any returned chapter boundaries into timestamped navigation links. [API documentation](https://flashdata.dev/docs/sources/youtube_metadata?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=youtube_metadata-task). |
| [YouTube Transcript](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/youtube-transcript/README.md) — `youtube_transcript` | Retrieve the existing English automatic transcript for YouTube video dQw4w9WgXcQ once. Produce notes with timestamped citations; report a missing track without transcribing or switching tracks. [API documentation](https://flashdata.dev/docs/sources/youtube_transcript?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=youtube_transcript-task). |
| [YouTube Captions](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/youtube-captions/README.md) — `youtube_captions` | Retrieve the English uploader captions for YouTube video dQw4w9WgXcQ once, with auto=false. Convert the returned timed segments to SRT locally; report a missing track without substituting another. [API documentation](https://flashdata.dev/docs/sources/youtube_captions?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=youtube_captions-task). |
| [YouTube Creative Commons License Metadata](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/youtube-trainability/README.md) — `youtube_trainability` | Inspect the license-label classification for YouTube video dQw4w9WgXcQ with one request. Record the raw permitted field beside the video ID, without treating it as training authorization. [API documentation](https://flashdata.dev/docs/sources/youtube_trainability?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=youtube_trainability-task). |
| [YouTube Channel Data](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/youtube-channel/README.md) — `youtube_channel` | Read the @jawed channel profile and request 10 recent videos once. Save the channel ID and returned video IDs, titles and available view counts as a dated snapshot. [API documentation](https://flashdata.dev/docs/sources/youtube_channel?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=youtube_channel-task). |
| [YouTube Autocomplete](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/youtube-autocomplete/README.md) — `youtube_suggestions` | Get YouTube autocomplete suggestions for battery recycling in English and the US with one request. Return unique phrases while preserving the query and locale; do not estimate search volume. [API documentation](https://flashdata.dev/docs/sources/youtube_suggestions?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=youtube_suggestions-task). |
| [YouTube Download to Cloud Storage](https://github.com/rocosen/flashdata-mcp/blob/6f1c2b4888814995c332ef472ff21d890667af2e/npm/docs/youtube-cloud-storage/README.md) — `youtube_download` | Prepare a non-secret media delivery plan; the user submits once through the protected local CLI below. [API documentation](https://flashdata.dev/docs/sources/youtube_download?utm_source=cartesia&utm_medium=integration&utm_campaign=intent_gtm_2026q4&utm_content=youtube_download-task). |

## Media delivery and functional limits

`youtube_download` and `submit_batch` are deliberately unavailable to the voice tool. A download requires the user's writable storage destination, scoped storage credentials, rights and byte-based cost approval. Have the agent prepare only a non-secret plan, then follow the [existing local download CLI guide](https://github.com/rocosen/flashdata-mcp/blob/main/npm/docs/youtube-cloud-storage/README.md). Use a protected file or stdin in your own environment, retain the job ID and inspect the final manifest there. This workflow needs Node.js 22.18+. Storage arguments are sent to FlashData for delivery; never speak them, paste them into chat, or let the agent read the populated secret file. Download job reads in this example return only ID/status, not the private manifest. Do not use `get_result` for unrelated cached private deliveries.

News provides article references/snippets, not guaranteed full text. Images are search references, not downloaded files. Shopping prices do not guarantee stock or checkout price; Places can omit contacts. Standard YouTube search requests at most 20 results; Extended requests at most 700 at a different rate, without guaranteeing counts. Transcript defaults to `auto=true`, Captions to `auto=false`; both retrieve existing tracks without ASR or translation. Trainability is a license-label signal, not a grant of training rights.

## File Overview

- `main.py`: Line loopback tools, spending/citation instructions and the voice app.
- `client.py`: authenticated Streamable HTTP MCP discovery/calls. It reuses the connection pattern from [FlashData's MIT example](https://github.com/rocosen/flashdata-mcp/blob/main/registry/examples/call_tool.py); it does not log raw transport errors or follow credential-bearing redirects.
- `pyproject.toml`: pinned Line SDK and MCP/HTTP dependencies.
- `cartesia.toml`: platform configuration placeholder, matching the existing examples.
- `tests/`: offline MCP transport fixtures and native Line tool-binding checks. `requests.json` uses FlashData's published request examples; no real key or storage destination is included.

## Offline verification

```sh
uv sync --group dev
LITELLM_LOCAL_MODEL_COST_MAP=True uv run pytest tests
```

Tests exercise SDK initialization/discovery, all 14 query names with fixture responses, header placement, one-call behavior, blocked downloads/batches, redirect/timeout errors, chunk recovery and native agent construction. Sockets are blocked and all MCP responses are fixtures. These checks are not evidence of successful paid data retrieval, voice quality, platform deployment or storage delivery. Run live checks only with your own explicit budget and credentials.

FlashData maintains this example. [Support](https://github.com/rocosen/flashdata-mcp/issues) · [Privacy](https://flashdata.dev/legal/privacy) · [Terms](https://flashdata.dev/legal/terms). Official inclusion and deployment are separate from source availability.
