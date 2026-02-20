import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Application
APP_NAME = "Brigade Eternia Voice Agent"
HOST = "0.0.0.0"
PORT = 8001  # Different port from hospital
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Webhook
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_VOICE_ID = os.getenv("SARVAM_VOICE_ID", "manan")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "bulbul:v3")
SMALLEST_API_KEY = os.getenv("SMALLEST_API_KEY", "")
SMALLEST_MODEL = os.getenv("SMALLEST_MODEL", "waves_lightning_large")

# Exotel
EXOTEL_ACCOUNT_SID = os.getenv("EXOTEL_ACCOUNT_SID", "")
EXOTEL_API_KEY = os.getenv("EXOTEL_API_KEY", "")
EXOTEL_API_TOKEN = os.getenv("EXOTEL_API_TOKEN", "")
EXOTEL_SUBDOMAIN = os.getenv("EXOTEL_SUBDOMAIN", "api.exotel.com")
EXOTEL_PHONE_NUMBER = os.getenv("EXOTEL_PHONE_NUMBER", "")

# Service Providers
STT_PROVIDER = os.getenv("STT_PROVIDER", "deepgram")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "smallest")  # Default to Smallest.ai Lightning v2
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "groq"
TELEPHONY_PROVIDER = "exotel"  # Fixed for this project


# LLM Settings
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.85"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "200"))
LLM_MAX_HISTORY = int(os.getenv("LLM_MAX_HISTORY", "3"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_DELAY = int(os.getenv("LLM_RETRY_DELAY", "2"))
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")

# LLM Configuration with Fallback
GROQ_PRIMARY_MODEL = os.getenv("GROQ_PRIMARY_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_USE_FALLBACK = os.getenv("GROQ_USE_FALLBACK", "true").lower() == "true"

# Token optimization
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "4"))  # Last 3 exchanges
MAX_LLM_TOKENS = int(os.getenv("MAX_LLM_TOKENS", "200"))  # Increased to 500 to prevent truncation in non-English JSON
# STT - Deepgram Settings
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "flux-general-en")
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "en")
DEEPGRAM_SAMPLE_RATE = int(os.getenv("DEEPGRAM_SAMPLE_RATE", "16000"))
DEEPGRAM_ENDPOINTING = int(os.getenv("DEEPGRAM_ENDPOINTING", "300"))

# STT - Sarvam Settings
SARVAM_STT_URL = os.getenv("SARVAM_STT_URL", "wss://api.sarvam.ai/v1/realtime/speech-to-text?model=saaras:v3&mode=transcribe&language_code=en-IN")

# TTS - Cartesia Settings
CARTESIA_MODEL_ID = os.getenv("CARTESIA_MODEL_ID", "sonic-english")
CARTESIA_SPEED = os.getenv("CARTESIA_SPEED", "normal")

# TTS - Sarvam Settings
SARVAM_SPEED = float(os.getenv("SARVAM_SPEED", "1.1"))
SARVAM_TTS_URL = os.getenv("SARVAM_TTS_URL", "https://api.sarvam.ai/text-to-speech")
SARVAM_LANGUAGE = os.getenv("SARVAM_LANGUAGE", "en-IN")

# Call Settings
CALL_DELAY_SECONDS = int(os.getenv("CALL_DELAY_SECONDS", "5"))

# Data Storage
ENQUIRIES_FILE = "data/enquiries.json"

# Brigade Eternia Settings
PROJECT_NAME = "Brigade Eternia"
AGENT_NAME = "Rohan"
COMPANY_NAME = "JLL Homes"
DEVELOPER_NAME = "Brigade Group"
KNOWLEDGE_BASE_PATH = "knowledge/brigade_eternia.json"

# Language Configuration
LANGUAGE = "english"

VOICE_MAPPINGS = {
    "english": {
        "cartesia_voice_id": "7ea5e9c2-b719-4dc3-b295-6e4ee1b18c26",
        "sarvam_speaker": "manan",
        "stt_lang": "en",
        "tts_lang": "en-IN",
        "agent_name": "Rohan",
        "greeting": "Hi, am I speaking with {name}?",
    }
}

# Current language config
LANG = VOICE_MAPPINGS["english"]
STT_LANGUAGE = LANG["stt_lang"]
TTS_LANGUAGE = LANG["tts_lang"]
AGENT_NAME = LANG["agent_name"]
GREETING_TEMPLATE = LANG["greeting"]
if TTS_PROVIDER == "cartesia":
    VOICE_ID = LANG["cartesia_voice_id"]
elif TTS_PROVIDER == "deepgram":
    VOICE_ID = "aura-orion-en"  # Warm professional male (Rohan)
elif TTS_PROVIDER == "smallest":
    VOICE_ID = os.getenv("VOICE_ID", "nyah") # Custom voice for Smallest.ai
else:
    VOICE_ID = LANG["sarvam_speaker"]
