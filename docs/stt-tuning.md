# Speech-recognition tuning: what was measured, and why the defaults look odd

Three ElevenLabs realtime parameters that sound obviously useful for this app
are **off by default**, because measuring them against the live service showed
they make the Azerbaijani transcript worse. This file records the measurements
so the defaults can be argued with rather than guessed at.

## Method

A fixed Azerbaijani interview question was synthesised once with ElevenLabs
`eleven_v3` TTS (`language_code=az`), saved as 16 kHz mono PCM, and replayed
identically into `wss://api.elevenlabs.io/v1/speech-to-text/realtime` for every
variant. Only one parameter changed at a time.

```
spoken: "RAG pipeline-ın retrieval quality-sini necə ölçərdiniz?"
base:   model_id=scribe_v2_realtime, audio_format=pcm_16000, language_code=aze,
        commit_strategy=vad, vad_threshold=0.5, vad_silence_threshold_secs=0.8
```

`az` is the Azerbaijani-language score from `llm/quality.py` (1.00 = idiomatic
Azerbaijani, 0.00 = not Azerbaijani). `azchars` counts Azerbaijani-specific
letters in the transcript.

## Results

| Variant | az | azchars | Transcript |
|---|---|---|---|
| **baseline** | **1.00** | **7** | `Rag pipeline-in retrieval quality-sının necə öyrədə biləsiniz?` |
| `+ secondary_languages=eng` | **0.00** | **0** | `Rag pipeline and retrieval quality` |
| `+ no_verbatim=true` | 1.00 | 5 | `RAG-pipeline və retrieval quality-nin necə öyrədilməsi.` |
| `+ filter_background_audio=true` | 1.00 | 7 | `Rag Pipeline-nin retrieval quality-sının necə öyrədə biləsiniz?` |
| `+ keyterms` (3) | 1.00 | 7 | `RAG, pipeline, retrieval, quality, sənət, neçə öyrədibəniz?` |
| `+ keyterms` (6) | 1.00 | 7 | `RAG, pipeline, retrieval, quality, sənət, neçə öyrədibəniz?` |
| `+ keyterms` (36) | 1.00 | 7 | `RAG, Pipeline, Retrieval Quality, sənət, neçə öyrədibəniz?` |
| baseline, repeated | 1.00 | 7 | *identical to baseline* |

The baseline reproduced exactly on repeat, so the differences are the parameters
rather than run-to-run variation.

## Conclusions

### `secondary_languages` — off

Setting it to `eng` did not "allow English technical terms"; it made Scribe
**return the sentence translated into English**, dropping every Azerbaijani
word and character. The specification is explicit that the original spoken
language must be preserved and the transcript must not be translated, so this
cannot be the default.

English technical vocabulary survives perfectly well without it — the baseline
transcript keeps `pipeline`, `retrieval` and `quality` in English while the
grammar stays Azerbaijani. That is exactly the desired behaviour.

Still configurable in **Settings → Tanınma → İkinci dillər** for a genuinely
bilingual interview.

### `keyterms` — off

Every keyterm list, including a three-term one, collapsed the sentence into a
**comma-separated list of the keyterms**, and corrupted the Azerbaijani suffix
that carries the question (`quality-sini` → `sənət`, `ölçərdiniz` →
`öyrədibəniz`). Adding more terms did not help.

The specification asks to avoid optional API features that do not provide a
measurable benefit. Measured, this one is a regression, so it ships off. The
curated 36-term vocabulary is still available behind the **Defolt keyterms**
button for anyone whose audio behaves differently.

### `no_verbatim` — off

It produced grammatical Azerbaijani but rewrote the question as a
nominalisation — *"how retrieval quality is taught"* rather than *"how would you
measure retrieval quality"*. That strips the interrogative ending that question
detection depends on, and a statement-shaped transcript is not recognised as a
question. Filler-word removal is not worth losing the question.

### `filter_background_audio` — on

No measurable cost (identical score and character count) and a plausible benefit
on a real call with room noise. Kept.

## The silence bug this exposed

The first live end-to-end run transcribed partials but produced **zero
committed transcripts**, and the connection dropped after ~16 s.

The cause was not a parameter. **Windows stops delivering WASAPI loopback
buffers entirely when nothing is playing** — not silence, *nothing*. Once the
interviewer stops speaking the capture callback simply stops firing, so:

1. Scribe never receives the pause that ends a segment, so VAD never commits,
   so the question is never detected; and
2. the server sees a connection with no audio at all and eventually closes it.

A separate check confirmed a session survives **61 s of streamed silence**
without closing, which rules out an idle-session timeout and confirms that
*sending nothing* is the problem.

`LoopbackCapture` now runs a keep-alive thread that emits a silence chunk
whenever the device has produced nothing for 1.5 chunk intervals. After the fix
the same end-to-end run produced a committed transcript, a detected question and
an answer, with **0 reconnects** (previously 2).

## Reproducing

The ablation harness is not part of the shipped test suite because it spends
TTS credit on every run. The live protocol tests that *are* in the suite
(`pytest -m live`) cover connection handling, streaming and error classification
without synthesising audio.


---

# Why a finished sentence sometimes produced no answer

Reported symptom: the interviewer finishes speaking and nothing happens.

## Measured cause

Commits do not arrive quickly, and they do not arrive at a fixed cadence:

| Situation | Gap from speech ending to `committed_transcript` |
|---|---|
| Clean sentence, `vad_silence_threshold_secs=0.8` | **1.3 – 2.0 s** |
| Same sentence, `vad_silence_threshold_secs=0.5` | **0.95 – 1.5 s** |
| Speaker hesitates 1.6 s mid-sentence | the two halves commit **3.3 s apart** |

The coalescing window that joins a split sentence was **700 ms** — far shorter
than the gap it exists to bridge. So the first half was flushed and discarded
before the second half could possibly arrive, and neither half on its own looked
like a prompt. The sentence vanished.

A `commit` is also not proof a sentence ended: the recogniser appends a full
stop to a fragment, so `'Mənə Rag Pipeline.'` arrived looking complete.

## Fixes

1. **The hold is now asymmetric.** Text that already reads as a prompt fires
   immediately, with no added latency. Only text that is *not yet* answerable is
   held — so a longer window costs nothing in the common case.
2. **The window is 4000 ms**, chosen to cover the measured 3.3 s worst case
   rather than a guess.
3. **Backchannel is never buffered.** A "Bəli." between two halves used to be
   glued into the question and reset the hold.
4. **`vad_silence_threshold_secs` default lowered 0.8 → 0.5**, measured ~0.34 s
   faster to commit with no extra fragmentation.
5. **`vad_silence_threshold_secs` is clamped to [0.3, 3.0]** before being sent.
   The API answers an out-of-range value with `invalid_request` and **closes the
   socket** — and the Settings dialog previously allowed 0.1, which would have
   killed the session mid-interview.

## Result, measured live

Same sentence, hesitating 1.6 s in the middle:

```
before:  commits 3.3s apart -> first half discarded
         answer generated from "Tecrübenizdən bəhs edin." alone (generic)

after:   'Mənə Rag Pipeline.' held
         'Tecrübenizdən bəhs edin.' joins it
         -> one question, one answer, about RAG specifically
```

End-to-end on an unsplit sentence also improved: first token **+3.25 s → +2.74 s**
after the speaker stops, complete answer **+5.20 s → +4.58 s**.

## What is still slow, and why

About **1.5 s of the remaining delay is the recogniser**, not this app: it waits
out its silence threshold and then transcribes. Lowering
`vad_silence_threshold_secs` to 0.35 did **not** help (1.07 s vs 0.95 s at 0.5)
and risks more splitting, so 0.5 is the floor worth using.


---

# Why only question-shaped sentences got answered

Reported symptom: an answer appears only when the interviewer asks something
that reads like a question.

That was literally true. Every committed segment went through a classifier, and
anything that did not look like a question or a request was discarded. Widening
the classifier helped but never fixed the underlying shape of the problem: a
heuristic sitting between the interviewer and an answer will always miss things.

## What changed

The gate was removed from the default path and replaced with **turn
accumulation**:

1. **`sensitivity` defaults to `broad`** - every interviewer turn is answered.
   Only acknowledgements are filtered, and they are filtered *before* buffering
   so they cannot be glued into a question either.
2. **Commits accumulate into a turn.** A commit within `turn_gap_ms` (6 s)
   extends the current turn instead of starting a new one.
3. **A continuation supersedes.** When the turn grows, the whole turn is
   re-emitted; the pipeline cancels the answer in flight and regenerates from
   the fuller text. So a sentence split across commits yields one answer about
   the complete sentence, not two about halves.
4. **Duplicate rules had to change with it.** Text that *extends* something
   already answered is now allowed through (that is the supersede case), while
   text *contained in* it is still blocked. A segment the recogniser re-sends
   within a turn is dropped before it can build "RAG nədir? RAG nədir?", which
   would otherwise have looked novel and been answered again.

## Verified live

A plain statement with no question mark and no interrogative word:

```
spoken : "Sizin CV-nizdə LangGraph təcrübəsi var."
commit : +1.52s
answer : starts immediately, first token +3.31s, complete +4.78s
         "Bəli, LangGraph ilə production-a yaxın multi-agent sistem
          təcrübəm var. Aztech Solutions-da 4 agentdən ibarət..."
```

Before this change that sentence produced nothing at all.


---

# The fix that never reached the user

Reported: a statement is still ignored while a question is answered, after the
gate had already been removed.

The gate was not the cause. **A saved settings value always beats a new
default.** The settings file on the machine read:

```json
"question": { "sensitivity": "normal", "merge_window_ms": 700 },
"stt":      { "vad_silence_threshold_secs": 0.8 }
```

Those are the values from *before* three separate rounds of tuning. Because
`Settings.load()` restores whatever is on disk, none of the changed defaults —
broad answering, the 4 s turn window, the faster VAD threshold — were in force.
Every improvement was real, and none of them were running.

`Settings` now carries a `schema_version`, and `migrate()` upgrades an older
file on load: for each default that changed, a stored value still equal to the
**old** default is moved to the new one, while anything deliberately chosen is
left alone. The migration is idempotent and tested.

The developer panel also now prints the values actually in force:

```
answer when      broad
turn merge       4000 ms
settings v       2
```

That line is the fastest way to tell "the build changed" from "the build
changed and the setting followed".

## Lesson

Changing a default is only half a change. Anything persisted needs a migration,
or the improvement ships to new installs only — and the person reporting the
bug is never a new install.
