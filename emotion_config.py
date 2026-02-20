"""
emotion_config.py — Emotion to Sarvam TTS parameter mapping.

Pace bumped +0.15 across all profiles for a more natural real-time phone feel.
pitch and loudness are retained for documentation / future model support,
but are NOT sent to bulbul:v3 which does not accept those fields.
"""

EMOTION_TTS_PARAMS = {
    # Deepgram Aura-2 uses context-based emotion.
    # We keep the keys for compatibility with the extraction logic,
    # but the values are not used for WebSocket streaming.
    "friendly": {},
    "empathetic": {},
    "enthusiastic": {},
    "calm": {},
    "professional": {},
    "apologetic": {},
    "default": {},
}


def get_emotion_params(emotion: str) -> dict:
    """
    Return TTS parameters for the given emotion name.
    Falls back to 'default' if the emotion is unrecognised or None.
    """
    if not emotion:
        return EMOTION_TTS_PARAMS["default"]
    key = emotion.strip().lower()
    return EMOTION_TTS_PARAMS.get(key, EMOTION_TTS_PARAMS["default"])
