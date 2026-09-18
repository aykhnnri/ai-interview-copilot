<div align="center">

# 🎙️ AZ Interview Copilot

**A Windows desktop assistant for Azerbaijani-language technical interviews.**

It listens to your system audio, transcribes the interviewer in Azerbaijani,
and streams a suggested answer grounded in your own CV — in about four seconds.

[![Platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078D4?logo=windows&logoColor=white)](#-requirements)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](#-requirements)
[![Tests](https://img.shields.io/badge/tests-465%20passing-2EA043)](#-testing)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[![STT](https://img.shields.io/badge/speech-ElevenLabs%20Scribe%20v2%20Realtime-000000)](https://elevenlabs.io/)
[![LLM](https://img.shields.io/badge/answers-OpenAI%20gpt--5.6--luna-412991?logo=openai&logoColor=white)](https://platform.openai.com/)

</div>

---

## What it is

Two independent AI providers, each doing exactly one job:

| Stage | Provider | Model |
|---|---|---|
| 🎧 Speech recognition | **ElevenLabs** | `scribe_v2_realtime` (Realtime STT WebSocket) |
| 💬 Answer generation | **OpenAI** | `gpt-5.6-luna` (Responses API, streamed, `fast` service tier) |

**Neither provider substitutes for the other.** If ElevenLabs fails you get a
transcription error, never an invented transcript. If OpenAI fails you get an
answer error, and the detected question is preserved so you can retry.

```
                        Windows system audio
                                 │
                                 │  WASAPI loopback capture (PyAudioWPatch)
                                 ▼
                       Mono PCM16 @ 16 kHz
                                 │
                                 ▼
             ElevenLabs Scribe v2 Realtime  (WebSocket)
                                 │
                                 │  partial_transcript / committed_transcript
                                 ▼
                  Turn accumulation + de-duplication
                                 │
                                 ▼
              CV + job description  (full, instruction layer)
                                 │
                                 ▼
              OpenAI Responses API  (streaming, fast tier)
                                 │
                                 ▼
                 Azerbaijani answer, token by token
```

### Why it is built this way

- 🇦🇿 **Never translates.** Azerbaijani stays Azerbaijani — it is never turned
  into Turkish or English. English technical terms stay in English.
- 🎯 **No retrieval step.** Your CV and the job advert go into the model's
  instruction layer *in full*, so the model already knows your background before
  the question arrives.
- 🗣️ **No question classifier in the way.** Every interviewer turn gets an
  answer, whether or not it is shaped like a question.
- 🔒 **Keys live in Windows Credential Manager**, never in a file, a log or this
  repository.
- 📐 **Claims here are measured**, not assumed. Timings, model comparisons and
  tuning decisions all have numbers attached, and the numbers are reproducible.

---

## 📋 Requirements

- **Windows 10 or 11.** The app captures *what Windows is playing* through
  WASAPI loopback. This is Windows-only; there is no macOS or Linux equivalent
  in this codebase.
- **Python 3.11+** (developed and tested on 3.13) — only for running from source.
- An **ElevenLabs** API key with Scribe v2 Realtime access.
- An **OpenAI** API key.

---

## 🚀 Quick start

```powershell
git clone https://github.com/aykhnnri/ai-interview-copilot.git
cd ai-interview-copilot

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

python run.py
```

Then: **Settings → API açarları** → paste both keys → **Test Connection** on
each → **Fayl → CV yüklə…** → **Start Listening**.

Check your environment at any time:

```powershell
python run.py --check
```

### Prebuilt executable

```powershell
.\build_windows.ps1
```

This installs dependencies, runs the test suite, and produces
`dist\AZInterviewCopilot\AZInterviewCopilot.exe`.

> [!IMPORTANT]
> Distribute the **whole `dist\AZInterviewCopilot` folder** — the `.exe` alone
> will not run. The executable is unsigned, so SmartScreen will warn on first
> launch.

---

## ⚙️ Configuration

### 🔑 API keys

Open **Settings → API açarları**. Each provider has its own section with
**Test Connection**, **Save Key** and **Remove Key**.

Keys are stored in **Windows Credential Manager** via `keyring`. Both
**Test Connection** buttons make a real authenticated request:

| Provider | Test request | Cost |
|---|---|---|
| ElevenLabs | `GET /v1/user/subscription` | free — returns your tier and usage |
| OpenAI | 16-token `responses.create` on your configured model **and tier** | a fraction of a cent |

If your account cannot use the configured service tier, the test says so
explicitly rather than quietly passing.

Environment variables work as a fallback:

```powershell
$env:ELEVENLABS_API_KEY = "..."
$env:OPENAI_API_KEY = "..."
```

Or copy `.env.example` to `.env` beside the app and fill it in.

> **Priority:** Credential Manager → environment variable → `.env`
> A real environment variable always beats the file, and `.env` is gitignored.

### 📄 CV and job description

**Fayl → CV yüklə…** and **Fayl → Vakansiya təsviri yüklə…** accept
**PDF, DOCX and TXT**.

Both documents are loaded **in full into the model's instruction layer**, once
per request. The model therefore already knows your background when a question
arrives — there is no retrieval or search step between the question being
finalised and the first token, and nothing relevant can be missed because a
keyword failed to match.

Because that block is byte-identical on every request it sits in the cacheable
prompt prefix, so repeating it costs far less than its token count suggests. A
CV plus advert is typically **~1,300 input tokens**.

A 30,000-character backstop guards against a pathological upload; if it ever
bites, the answer panel shows `⚠ sənədlər kəsildi` rather than silently dropping
the tail.

### 🎛️ Recognition and answer settings

| Tab | What it controls |
|---|---|
| **Görünüş** | Window transparency, always-on-top, transcript strip, font scale |
| **Səs** | Loopback device, sample rate sent to ElevenLabs, chunk size, gain |
| **Tanınma** | Model id, primary/secondary language, VAD sensitivity and silence thresholds, when to answer, turn-merge window, keyterms |
| **Cavab** | Model, service tier, output token budget, temperature, retained dialogue turns, auto-generate on/off |

VAD defaults: threshold `0.5`, silence `0.5 s`, minimum speech `150 ms`. The
silence threshold is clamped to the `0.3 – 3.0 s` the API accepts; outside that
range it refuses the connection outright.

> [!NOTE]
> **`secondary_languages`, `keyterms` and `no_verbatim` ship switched off**,
> which looks wrong until you see the measurements: each one made the
> Azerbaijani transcript *worse*, and `secondary_languages=eng` made Scribe
> return the sentence translated into English — the exact thing this app must
> never do. All three remain available in Settings. Evidence:
> [docs/stt-tuning.md](docs/stt-tuning.md).

---

## 🎧 Using it

1. Start your interview call — **Teams, Zoom, Meet, Webex**, anything that plays
   audio.
2. Press **Start Listening**. Both dots in the status bar turn white.
3. When the interviewer finishes a turn it appears under **SUAL** and the answer
   starts streaming underneath it.
4. **Compact** (`Ctrl+Shift+C`) drops everything except question and answer and
   pins the window above your call.

The window is built around one idea: **the answer is the only thing you are
actually reading**, so it gets the space and the largest type. Everything else —
provider state, documents, controls — is one line of quiet grey chrome.

| | |
|---|---|
| `Ctrl+Shift+C` | Compact: question + answer only, pinned on top |
| `Ctrl+T` | Show / hide the live transcript strip |
| `Ctrl+Shift+↑` / `↓` | Make the window more / less transparent |
| `Ctrl+,` | Settings |
| `Ctrl+Shift+D` | Developer panel |

**Pause Listening** keeps the ElevenLabs connection open for an instant resume.
**Stop Listening** stops capture, closes the connection and releases the device.

### 🫥 Transparency

The window can be made translucent so it sits **over** the call rather than
beside it — you read the answer and still see the interviewer through it.

Drag **Settings → Görünüş → Pəncərə şəffaflığı**, or press
`Ctrl+Shift+↓` mid-call. The slider previews live on the window behind the
dialog, because choosing a translucency you cannot see is guesswork.

Opacity is clamped to **35–100%**. Below 35% the text stops being readable and,
worse, the window becomes hard to find again — so a corrupt or fat-fingered
value is repaired on load rather than obeyed.

### 🎨 The palette

Greyscale, deliberately. The only colour in the application is the error tone,
because a dead provider has to be distinguishable at a glance and grey cannot
carry that. Even the primary button is near-white on dark rather than blue — a
test asserts the palette stays neutral, so an accent colour cannot creep back in.

> The app is **text output only**. It never speaks, and it never plays audio
> into your call.

### Does it work with Teams / Meet / Zoom?

**Yes, and there is nothing to configure per app.** Capture taps the Windows
*render endpoint*, so it records whatever Windows is playing regardless of which
program is playing it. No bot joins the call, no virtual audio cable, no plugin
— the other side sees a completely normal participant. Your microphone is never
touched, so the interviewer is transcribed and you are not.

Three things worth knowing:

| ⚠️ | Why it matters |
|---|---|
| **Pick your output device before pressing Start** | The endpoint is resolved once, at start. Plug in headphones mid-call and Windows moves playback to the new endpoint while capture stays on the old one — transcription stops with **no error**, because silence is indistinguishable from nobody talking. The tell is a flat level bar while the interviewer is clearly speaking. Stop → Start fixes it. |
| **Bluetooth headsets are the worst case** | When the meeting app grabs the headset microphone, Windows flips the device into hands-free (HFP) mode — a *different* endpoint, at telephone bandwidth. You hit the problem above **and** feed Scribe much poorer audio. Wired headphones or the laptop's own speakers transcribe noticeably better. |
| **There is no speaker separation** | Everything Windows plays goes to ElevenLabs as one stream: two interviewers, a YouTube tab, a Slack notification, Spotify. Mute anything else that makes noise. |

Leave the device on **Defolt (sistem səsi)** and it follows whatever Windows is
using at the moment you press Start.

### 🛠️ Developer panel (`Ctrl+Shift+D`)

- **Latency** — audio→partial, audio→committed, question finalisation,
  question→first token, and total answer time, each with median and p95.
- **Benchmark** — see below.
- **Jurnal** — pipeline events.

It also prints the settings **actually in force** (`answer when`, `turn merge`,
`settings v`), which is the quickest way to check whether a change took effect.

---

## 📊 Model benchmark

Speed claims should be measured, not assumed, so nothing here presumes which
model is fastest for Azerbaijani.

The bank holds **24 Azerbaijani technical interview questions** across RAG,
agents, serving, evaluation, security and Python internals. Every model receives
identical questions with an identical output-token budget and no conversation
history, so results are comparable.

```powershell
python benchmark_cli.py --models gpt-4.1-nano gpt-5.6-luna
python benchmark_cli.py --models gpt-4.1-nano --questions 5 --cv mycv.pdf --yes
```

Results are saved as JSON under `%LOCALAPPDATA%\AzInterviewCopilot\benchmarks`.

The report names the fastest, the highest-quality and the most consistent model
**separately**, and says so explicitly when they disagree, rather than
collapsing three different questions into one winner.

> [!WARNING]
> The benchmark sends real OpenAI requests and costs real money. It states the
> request count up front and only runs when you confirm.
> **The app never switches models on its own** — change it in Settings → Cavab.

### Measured results

72 live requests · 24 questions × 3 models · identical prompts · 450 output-token budget · sample CV and job description as context:

| Model | TTFT median | **TTFT p95** | Total median | Chars | AZ | EN terms | **Coverage** | Overall | Cost/answer |
|---|---|---|---|---|---|---|---|---|---|
| GPT-4.1 Nano | **700 ms** | 1742 ms | **1710 ms** | 701 | 1.00 | 0.94 | 0.57 | 0.82 | **$0.00014** |
| **GPT-5.6 Luna** ⭐ | 832 ms | **1119 ms** | 3056 ms | 1329 | 1.00 | **0.98** | **0.95** | **0.97** | — |
| GPT-4.1 Mini | 792 ms | 1277 ms | 2192 ms | 967 | 1.00 | **0.98** | 0.69 | 0.87 | $0.00069 |

*24 answers, 0 failures, for every model.*

What this actually says:

- **All three write genuinely good Azerbaijani.** Language score 1.00 across the
  board, zero Turkish markers in 72 answers. Model choice is not a language risk.
- **Nano is fastest to start and ~5× cheaper**, but its concept coverage (0.57)
  is the weakest — it answers quickly and shallowly.
- **Luna has the best worst case.** It loses on median TTFT but wins on p95
  (1119 ms vs Nano's 1742 ms), so it is the least likely to visibly stall, and
  its coverage (0.95) is far ahead. It also writes ~1.9× more, which is why its
  total time is double.
- Luna's cost column is `—` because no price is configured for it; the app
  reports nothing rather than inventing a figure.

**`gpt-5.6-luna` is the default** — the best worst case and far better concept
coverage. `gpt-4.1-nano` is one click away if you want the cheapest,
fastest-starting option.

<details>
<summary><b>How language quality is scored</b></summary>

<br>

The decisive signal is the letter **ə**, which Azerbaijani uses constantly and
Turkish does not have at all, combined with a list of Turkish function words
whose Azerbaijani equivalents differ (`ve`/`və`, `ile`/`ilə`, `için`/`üçün`…).
Coverage is checked against expected concepts per question.

These are deterministic heuristics, not a language expert. They reliably catch
Turkish drift and missed concepts; they do **not** certify that an answer is
good. Read the answers.

</details>

> Re-run it on your own account. These numbers are from one machine on one day,
> measured before the profile moved into the instruction layer, so input token
> counts are now higher (~1,270 per request with a CV loaded).

---

## 🧪 Testing

```powershell
python -m pytest -q                 # full offline suite
python -m pytest -q --cov=copilot   # with coverage
```

```
465 passed, 15 skipped in 14.05s
```

The default run uses **no network and no audio hardware**: a fake websocket
replays real ElevenLabs frame shapes and a scripted client stands in for the
OpenAI SDK.

Covered: ElevenLabs connection handling and reconnection, audio chunk
preparation and resampling, partial/committed transcript events, Azerbaijani
text preservation, question detection, duplicate prevention, OpenAI response
streaming, CV and job-description handling, settings migration, invalid
credentials, disconnected audio devices, interrupted connections, rate limits,
credential storage and log redaction, the benchmark, and the GUI including opacity clamping and the greyscale palette.

### Live integration tests (billable, opt-in)

```powershell
$env:AZCOPILOT_LIVE_TESTS = "1"
$env:ELEVENLABS_API_KEY = "..."
$env:OPENAI_API_KEY = "..."
python -m pytest -m live -q
```

> Without `AZCOPILOT_LIVE_TESTS=1` these are skipped even if keys are present,
> so a stray key can never bill you by accident.

---

## 🏗️ How it works

| Module | Responsibility |
|---|---|
| `audio/processing.py` | PCM conversion; streaming polyphase resampler (no SciPy) with carry-over state, so chunk boundaries are click-free |
| `audio/capture.py` | WASAPI loopback capture; PortAudio callback does only numpy work |
| `stt/elevenlabs_client.py` | Persistent realtime WebSocket; base64 audio frames; bounded reconnection |
| `stt/events.py` | Frame parsing and error classification |
| `nlp/question_detector.py` | Turn accumulation, backchannel filtering, de-duplication |
| `documents/` | PDF/DOCX/TXT extraction and sectioning; full-profile rendering for the instruction layer |
| `llm/prompts.py` | System instruction, CV/JD instruction layer, fenced question input |
| `llm/openai_client.py` | Responses API streaming, per-model capabilities, error mapping, transient retry |
| `core/pipeline.py` | Orchestration on a background asyncio loop |
| `config.py` | Settings schema, persistence and **version migration** |
| `env_file.py` | Dependency-free `.env` reader; real environment variables win |
| `ui/theme.py` | The palette and stylesheet, defined once and scaled by the font setting |
| `ui/` | Qt widgets; pipeline events arrive as Qt signals |

**Threading.** Qt owns the main thread. All networking runs on one asyncio loop
in a background thread. The PortAudio callback thread hands buffers over with
`call_soon_threadsafe` and never touches Qt or a socket.

**Latency choices.** The OpenAI client is constructed when you press Start, not
when the first question arrives. One WebSocket is reused for the whole session.
A question ending in `?` fires immediately rather than waiting out the merge
window. Answer deltas are painted on a 60 ms timer instead of per token. If the
send queue backs up, the newest chunk is dropped rather than accumulating lag.

### 🗣️ When it answers

**There is no classifier standing between the interviewer finishing a sentence
and an answer appearing.** By default every turn the interviewer takes is
answered, whether or not it is shaped like a question. *"Sizin CV-nizdə
LangGraph təcrübəsi var."* is a statement with no question mark and no
interrogative word, and it gets an answer.

The only things filtered out are acknowledgements — *"Bəli"*, *"Aydındır"*,
*"Maraqlıdır"*, *"Təşəkkür edirəm"* — because answering those helps nobody.

**Turns, not sentences.** The recogniser commits in pieces, so commits are
accumulated into a turn. If the interviewer carries on, the next commit
re-answers the **whole turn**, and that fuller answer *replaces* the one already
streaming rather than queueing a second unrelated one. A gap longer than
`turn_gap_ms` (6 s) starts a new turn.

Change the gate in **Settings → Tanınma → Aşkarlama həssaslığı**:

| Level | Answers | On a 15-line sample interview |
|---|---|---|
| **`broad`** (default) | every interviewer turn | **8** |
| `normal` | questions **+** requests and invitations | 6 |
| `strict` | grammatical questions only | 2 |

*Generate Answer* always works too: with nothing detected it answers whatever
was last said.

> [!TIP]
> Settings written by an older version are **migrated on load**: a value still
> at the old default moves to the new one, a value you chose is kept. Without
> that, none of the tuning above would reach anyone who had already run the app
> once — a saved value always beats a new default.

### ⏱️ Timing, measured

After the interviewer stops speaking:

| Stage | Elapsed |
|---|---|
| `committed_transcript` arrives | **+1.5 s** |
| answer generation starts | same instant |
| first answer token | **+2.7 – 3.3 s** |
| complete answer | **+4.6 – 4.8 s** |

Roughly 1.5 s of that is the recogniser waiting out its own silence threshold
and transcribing — not something this app can skip.

A sentence the interviewer **pauses in the middle of** commits in two pieces
about **3.3 s apart**; at `normal` and `strict` the halves are held and joined,
at `broad` the first half is answered and then superseded by the full turn.
See [docs/stt-tuning.md](docs/stt-tuning.md).

### 🛡️ Answer grounding

Documents and transcribed speech are fenced and explicitly labelled as data, not
instructions, so a spoken *"ignore your instructions"* is answered as a question
rather than obeyed. The prompt states that experience not evidenced in the CV
must not be claimed.

> This reduces fabrication; it does not eliminate it.
> **Read what it writes before you say it.**

---

## 🛟 Failure handling

| Situation | Behaviour |
|---|---|
| ElevenLabs disconnects | `ElevenLabs bağlantısı kəsildi.` then bounded exponential-backoff reconnection |
| Invalid API key | Clear authentication error; **no** retry, since retrying cannot help |
| Out of credit | Billing error; no retry |
| Rate limited | Reported as recoverable |
| OpenAI fails | `AI cavabı yaradıla bilmədi.`; the question is kept so you can retry |
| No loopback device | Named error; listening does not start |

**No provider ever silently substitutes for another, and a failed request is
never reported as a success.**

---

## ✅ Verified

Confirmed by running it on this machine (Windows 11, Python 3.13) against the
real ElevenLabs and OpenAI APIs.

### Full pipeline, live, end to end

An Azerbaijani question was synthesised with ElevenLabs TTS, **played through
the Windows speakers**, and picked up by the app's own capture path:

```
spoken     : RAG pipeline-ın retrieval quality-sini necə ölçərdiniz?
heard      : Rag Pipeline-in retrieval quality-sini necə öyrədəcəksiniz?
             (89% similarity, reads as Azerbaijani: 1.00, no Turkish markers)
detected   : question fired, one answer generated
answered   : first token 1456 ms, complete 2095 ms, 186 streamed deltas
grounded   : cited Recall@10, MRR, hybrid search (BM25 + dense), reranking
             — all of it from the uploaded CV
reconnects : 0
```

Every stage was the production code path. **Nothing was mocked.**

<details>
<summary><b>Individually verified</b></summary>

<br>

- WASAPI loopback capture — a 440 Hz tone played through the default device was
  captured, downmixed 48 kHz stereo → 16 kHz mono, and recovered at 441.4 Hz in
  exact 3200-byte (100 ms) chunks.
- Resampler — chunk-boundary invariant to 0.0 absolute difference;
  anti-aliasing verified against a 10 kHz tone.
- **465 automated tests pass** offline; **15 live tests pass** against the real
  APIs.
- Both connection tests succeed against the live services, and the OpenAI key is
  correctly *rejected* by ElevenLabs — provider separation proven live.
- Answers in Azerbaijani for multiple questions, streamed incrementally, with
  English technical terms preserved.
- All 10 specified Azerbaijani questions are detected, with and without a
  question mark; greetings are not misread as questions.
- Split questions rejoin into one; repeated questions produce one answer.
- The GUI builds, streams, switches to compact mode and shuts down cleanly.
- Windows executable built with PyInstaller and launched — window title
  `AZ Interview Copilot`, loopback device and Credential Manager both detected.
- Log redaction masks OpenAI, ElevenLabs and header-shaped credentials.

</details>

<details>
<summary><b>Model, grounding and service tier (live-verified)</b></summary>

<br>

With `gpt-5.6-luna`, `service_tier="fast"` and the whole CV in the instruction
layer, asked six questions against the sample CV:

```
TTFT median 786 ms   (min 688, max 2099 on the cold first call)
total median 2033 ms
service tier rejected: None          -- 'fast' was accepted
```

The answers quote the CV directly without any retrieval step — *"Aztech
Solutions-da LangGraph üzərində 4 agentdən ibarət multi-agent research
sistemi"*, *"pgvector, Qdrant və FAISS"*, *"Bakı Dövlət Universitetində …
2019–2021"*.

Grounding still fails closed. Asked about Rust, which appears nowhere in the CV:

> *"Rust ilə bağlı professional iş təcrübəm olmayıb … bunu real layihədə
> istifadə etdiyimi iddia etmirəm."*

Both behaviours are pinned by live tests, so a regression fails the suite.

</details>

<details>
<summary><b>Three real bugs this found, and fixed</b></summary>

<br>

1. **Silence starvation.** Windows stops delivering loopback buffers *entirely*
   when nothing is playing — not silence, *nothing*. The transcriber therefore
   never saw the pause that ends a sentence, so questions were never committed
   and the connection dropped after ~16 s. `LoopbackCapture` now emits keep-alive
   silence. Before: 0 commits, 2 reconnects. After: committed transcript,
   question, answer, 0 reconnects.

2. **`secondary_languages` translated the transcript into English** — the exact
   thing this app must never do. It, `keyterms` and `no_verbatim` are now off by
   default, each for a measured reason. See [docs/stt-tuning.md](docs/stt-tuning.md).

3. **Improved defaults never reached anyone who had already run the app.**
   `Settings.load()` restores whatever is saved, and a saved value always beats a
   new default — so three rounds of detection tuning sat in the code without
   ever taking effect. Settings are now versioned and migrated on load.

</details>

---

## ⚠️ Limitations

- **Windows only.** WASAPI loopback has no cross-platform equivalent here.
- Captures system audio, so it hears the **interviewer**, not you. A question
  asked in the room rather than through the call will not be picked up.
- No speaker separation — every sound Windows plays becomes one transcript.
- Speech recognition and answers both depend on your providers being reachable;
  there is no offline mode.
- Conversation memory is the last few Q&A turns, not the whole interview.
- Answer quality scoring is heuristic, not a language expert.
- The executable is unsigned; SmartScreen will warn on first run.
- **Using an assistant during an interview may breach the rules of the process
  you are in. That is your call to make.**

---

## 📄 License

[MIT](LICENSE)
