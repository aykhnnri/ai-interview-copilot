"""User-facing Azerbaijani strings.

Kept in one place so the exact wording is testable and the Azerbaijani
characters (ə ı ğ ç ş ö ü) are guaranteed to survive as UTF-8.
"""
from __future__ import annotations

# -- Speech recognition (ElevenLabs) --------------------------------------
STT_DISCONNECTED = "ElevenLabs bağlantısı kəsildi."
STT_RECONNECTING = "ElevenLabs bağlantısı bərpa edilir…"
STT_CONNECTED = "ElevenLabs qoşuldu."
STT_AUTH_ERROR = "ElevenLabs API açarı qəbul edilmədi (autentifikasiya xətası)."
STT_BILLING_ERROR = "ElevenLabs hesabınızda kifayət qədər kredit yoxdur."
STT_RATE_LIMITED = "ElevenLabs sorğu limiti aşıldı."
STT_GENERIC_ERROR = "ElevenLabs xətası."

# -- Answer generation (OpenAI) -------------------------------------------
LLM_FAILED = "AI cavabı yaradıla bilmədi."
LLM_AUTH_ERROR = "OpenAI API açarı qəbul edilmədi (autentifikasiya xətası)."
LLM_BILLING_ERROR = "OpenAI hesabınızda kifayət qədər kredit yoxdur."
LLM_RATE_LIMITED = "OpenAI sorğu limiti aşıldı. Bir qədər sonra yenidən cəhd edin."
LLM_TIMEOUT = "OpenAI cavabı vaxtında gəlmədi."

# -- Audio ----------------------------------------------------------------
AUDIO_NO_DEVICE = (
    "Windows səs cihazı tapılmadı. Səs çıxış cihazının aktiv olduğunu yoxlayın."
)
AUDIO_DEVICE_LOST = "Səs cihazı ilə bağlantı kəsildi."

# -- Credentials ----------------------------------------------------------
KEY_MISSING_ELEVENLABS = "ElevenLabs API açarı təyin edilməyib."
KEY_MISSING_OPENAI = "OpenAI API açarı təyin edilməyib."

# -- Status ---------------------------------------------------------------
STATUS_IDLE = "Hazır"
STATUS_LISTENING = "Dinlənilir…"
STATUS_PAUSED = "Dayandırıldı"
STATUS_STOPPED = "Dayandı"
STATUS_GENERATING = "Cavab hazırlanır…"
STATUS_CONNECTED = "Qoşulub"
STATUS_DISCONNECTED = "Bağlantı yoxdur"
STATUS_CONNECTING = "Qoşulur…"
STATUS_ERROR = "Xəta"
