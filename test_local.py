"""
Local Voice Client for Property Enquiry Testing

Allows testing STT, LLM, and TTS services locally using microphone and speakers
without telephony integration for the Property Enquiry Agent.
"""
import asyncio
import json
import logging
import queue
import threading
import pyaudio
import audioop
import signal
import sys
import wave
import os
import time
from datetime import datetime
from typing import Dict, Optional, List

import config
import prompts
from prompts import STAGE_DEFINITIONS
from services.stt_factory import STTServiceFactory
from services.tts_factory import TTSServiceFactory
from services.llm_service import GroqLLMService
from services.openai_llm_service import OpenAILLMService
import emotion_config
from utils.audio_utils import mulaw_to_pcm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 256
FORMAT = pyaudio.paInt16  # 16-bit PCM

# Default test user data (can be overridden)
DEFAULT_USER_NAME = "John Doe"
DEFAULT_USER_MESSAGE = "Looking for a 3 BHK apartment in Brigade Eternia"

# Dynamic fields for Brigade Eternia site visit flow (matching new architecture)
# NOTE: conversation_stage removed — backend is sole authority
BRIGADE_ETERNIA_DYNAMIC_FIELDS = {
    "intent": {
        "type": "string",
        "description": "Intent classification",
        "default": "flow_progress"
    },
    "assistant_text": {
        "type": "string",
        "description": "Spoken response text",
        "default": ""
    },
    "preferred_bhk": {
        "type": "string",
        "description": "BHK preference",
        "default": "none"
    },
    "visit_date": {
        "type": "string",
        "description": "Visit date",
        "default": "none"
    },
    "visit_time": {
        "type": "string",
        "description": "Visit time",
        "default": "none"
    },
    "visit_confirmed": {
        "type": "string",
        "description": "Visit confirmed",
        "default": "no"
    },
    "callback_scheduled": {
        "type": "string",
        "description": "Callback scheduled",
        "default": "no"
    },
    "end_call": {
        "type": "string",
        "description": "End call flag",
        "default": "no"
    }
}

# Note: Farewell is now handled by LLM in the system prompt (Stage 5)

# Stop words that trigger graceful shutdown
STOP_WORDS = [
    "end call", "hang up", "stop the call"  # Only hard commands, not conversational
]

# Timeout settings
RECORD_TIMEOUT = 60  # seconds - increased for better UX
SILENCE_PROMPT_TIMEOUT = 30 # seconds - prompt user if quiet for this long


# ---------------------------------------------------------------------------
# Emotion extraction utility
# ---------------------------------------------------------------------------
import re as _re
_EMOTION_TAG_RE = _re.compile(r'\[EMOTION:\s*([a-z]+)\]\s*$', _re.IGNORECASE | _re.MULTILINE)

def extract_emotion(text: str):
    """
    Find and strip the [EMOTION: <name>] tag appended by the LLM.
    Returns (clean_text, emotion_name) where emotion_name is None if tag absent.
    """
    match = _EMOTION_TAG_RE.search(text)
    if match:
        clean = text[:match.start()].rstrip()
        return clean, match.group(1).lower()
    return text, None


class LocalVoiceClient:
    """Local voice client for testing property enquiry voice services."""

    
    def __init__(self, user_name: str = None, user_message: str = None):
        # User data (can be set from form or use defaults)
        self.user_name = user_name or DEFAULT_USER_NAME
        self.user_message = user_message or DEFAULT_USER_MESSAGE
        self._prompted_silence = False
        
        # Format name for greeting (remove initial if present)
        self.greeting_name = self._format_name_for_greeting(self.user_name)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Service instances
        self.stt_service = None
        self.tts_service = None
        self.llm_service = None
        
        self.is_recording = False
        self.is_playing = False
        self.is_farewell = False
        self.should_stop = False
        self.call_ended = False
        self.is_user_speaking = False
        self.interruption_threshold = 800  # Tune: increase if false triggers, decrease if not triggering
        self.completed_stages = []
        self.current_stage = "identity_check"
        self.is_processing = False  # Prevent concurrent LLM calls
        self.stt_muted = False      # Mute STT during playback to prevent echo
        
        # Slots — source of truth for extracted data
        self.slots = {
            "preferred_bhk": None,
            "visit_date": None,
            "visit_time": None,
            "visit_confirmed": "no",
            "callback_scheduled": "no"
        }
        
        # Callback time validation flag
        self._awaiting_callback_time = False
        # Site visit date/time validation flag
        self._awaiting_visit_datetime = False
        
        self.input_stream = None
        self.output_stream = None
        
        self.conversation_history = []
        self.collected_data = {}
        self.session_start = None
        
        # Timeout tracking
        self.last_user_speech_time = None
        self.silence_check_task = None
        
        # Recording attributes
        self.recording_enabled = os.getenv("ENABLE_RECORDING", "false").lower() == "true"
        self.recordings_dir = os.getenv("RECORDINGS_DIR", "recordings")
        self.recording_buffer: List[bytes] = []
        self.recording_filename = None
        
        # Create recordings directory if enabled
        if self.recording_enabled:
            os.makedirs(self.recordings_dir, exist_ok=True)
            logger.info(f"[RECORDING] Enabled - saving to {self.recordings_dir}/")
        
        # Dedicated audio playback thread — PyAudio must be written from a single
        # tight loop thread to avoid buffer underruns (static/crackle).
        self._pcm_queue: queue.Queue = queue.Queue(maxsize=500)
        self._audio_thread: threading.Thread = None
        self._audio_thread_stop = threading.Event()
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Guard flag for playback finished callback
        self._playback_callback_fired = False
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully."""
        if not self.should_stop:
            print("\\n\\n[SHUTDOWN] Stopping local test mode...")
            self.should_stop = True
            self.is_recording = False
            # Don't use sys.exit() - let cleanup happen naturally
    
    def _format_name_for_greeting(self, name: str) -> str:
        """Extract first name for casual greeting."""
        parts = name.strip().split()
        if parts:
            return parts[0]
        return name
    async def initialize_services(self):
        """Initialize STT, TTS, and LLM services."""
        try:
            logger.info("[INIT] Initializing services...")
            print("\\n" + "=" * 60)
            print("INITIALIZING SERVICES")
            print("=" * 60)
            
            # Initialize STT
            print(f"[STT] Creating {config.STT_PROVIDER.title()} STT service...")
            stt_api_key = (
                config.DEEPGRAM_API_KEY if config.STT_PROVIDER == 'deepgram'
                else config.SARVAM_API_KEY
            )
            self.stt_service = STTServiceFactory.create(
                provider=config.STT_PROVIDER,
                api_key=stt_api_key
            )
            
            # Setup transcription callback
            async def transcription_callback(text: str):
                await self.handle_transcription(text)
            
            # Initialize STT
            # Different STT providers have different init signatures
            if config.STT_PROVIDER == 'deepgram':
                stt_init_success = await self.stt_service.initialize(
                    api_key=stt_api_key,
                    encoding="linear16",
                    sample_rate=SAMPLE_RATE
                )
            elif config.STT_PROVIDER == 'sarvam':
                stt_init_success = await self.stt_service.initialize(api_key=stt_api_key)
            else:
                stt_init_success = await self.stt_service.initialize(api_key=stt_api_key)

            if not stt_init_success:
                print(f"[ERROR] Failed to initialize {config.STT_PROVIDER} STT service")
                return False
            
            # Start stream
            stream_success = await self.stt_service.start_stream(transcription_callback)
            if not stream_success:
                print(f"[ERROR] Failed to start {config.STT_PROVIDER} STT stream")
                return False
            
            print(f"[STT] OK - {config.STT_PROVIDER.title()} STT initialized")
            
            # Initialize TTS
            print(f"[TTS] Creating {config.TTS_PROVIDER.title()} TTS service...")
            if config.TTS_PROVIDER == 'cartesia':
                tts_api_key = config.CARTESIA_API_KEY
                voice_id = config.CARTESIA_VOICE_ID
                tts_kwargs = {'model_id': 'sonic-english', 'speed': 'normal'}
            elif config.TTS_PROVIDER == 'deepgram':
                tts_api_key = config.DEEPGRAM_API_KEY
                # config.VOICE_ID is already set correctly in config.py update
                voice_id = getattr(config, 'VOICE_ID', 'aura-orion-en') 
                tts_kwargs = {}
            elif config.TTS_PROVIDER == 'smallest':
                tts_api_key = config.SMALLEST_API_KEY
                voice_id = getattr(config, 'VOICE_ID', 'emily')
                tts_kwargs = {'model': getattr(config, 'SMALLEST_MODEL', 'lightning-v2')}
            else:
                tts_api_key = config.SARVAM_API_KEY
                voice_id = config.SARVAM_VOICE_ID or 'rohan'
                tts_kwargs = {
                    'model': config.SARVAM_MODEL or 'bulbul:v3',
                    'language': config.SARVAM_LANGUAGE,
                    'speed': config.SARVAM_SPEED
                }
            
            self.tts_service = TTSServiceFactory.create(
                provider=config.TTS_PROVIDER,
                api_key=tts_api_key,
                voice_id=voice_id,
                **tts_kwargs
            )
            
            # SMALLST: Initial connect here for persistence
            if config.TTS_PROVIDER == 'smallest':
                await self.tts_service.connect()
                
            await self.tts_service.initialize()
            print(f"[TTS] OK - {config.TTS_PROVIDER.title()} TTS initialized")
            
            # Pre-warm TTS if Sarvam to reduce first-request latency
            if config.TTS_PROVIDER == 'sarvam':
                print("[TTS] Pre-warming Sarvam connection...")
                try:
                    await self.tts_service.prewarm_tts()
                except AttributeError:
                    pass # Method might not exist on all service variations yet

            
            # Initialize LLM — toggle via LLM_PROVIDER env var
            if config.LLM_PROVIDER == "openai":
                print(f"[LLM] Creating OpenAI LLM service (model: {config.OPENAI_MODEL})...")
                self.llm_service = OpenAILLMService(api_key=config.OPENAI_API_KEY, max_history=10)
            else:
                print(f"[LLM] Creating Groq LLM service...")
                self.llm_service = GroqLLMService(api_key=config.GROQ_API_KEY, max_history=10)

            # Pass raw prompt template -- runtime context formatted per-turn via format_values
            await self.llm_service.initialize(
                dynamic_fields=BRIGADE_ETERNIA_DYNAMIC_FIELDS,
                system_prompt_template=prompts.BRIGADE_ETERNIA_SYSTEM_PROMPT
            )
            print(f"[LLM] OK - {config.LLM_PROVIDER.upper()} LLM initialized")
            
            print("=" * 60)
            print("SUCCESS - All services initialized!")
            print("=" * 60 + "\\n")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize services: {e}", exc_info=True)
            print(f"\\nERROR - INITIALIZATION FAILED: {e}\\n")
            return False
    
    def setup_audio_streams(self):
        """Setup microphone input and speaker output streams."""
        try:
            print("[AUDIO] Setting up microphone and speakers...")
            
            # Input stream (microphone)
            self.input_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=None
            )
            
            # Output stream (speakers)
            # SMALLST: Lightning native rate is 24000Hz
            output_rate = 24000 if config.TTS_PROVIDER == 'smallest' else SAMPLE_RATE
            
            self.output_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=output_rate,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            print(f"[AUDIO] STT input stream:  {SAMPLE_RATE} Hz, mono, paInt16")
            # Note: Sarvam TTS output is confirmed to be 16000 Hz
            print(f"[AUDIO] TTS output stream: {output_rate} Hz, mono, paInt16")
            print("[AUDIO] OK - Audio streams ready")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to setup audio: {e}")
            print(f"ERROR - AUDIO SETUP FAILED: {e}")
            print("\\nTroubleshooting:")
            print("1. Check if your microphone is connected")
            print("2. Check if your speakers are connected")
            print("3. Try running: python -m pyaudio.test")
            return False
    
    async def record_audio_loop(self):
        """Continuously record from microphone and send to STT."""
        try:
            self.is_recording = True
            print("\\n[LISTENING] Speak now... (Press Ctrl+C to stop)\\n")
            
            while not self.should_stop and self.is_recording:
                try:
                    # Read audio chunk from microphone
                    try:
                        # Non-blocking read check
                        if self.input_stream.get_read_available() < CHUNK_SIZE:
                            await asyncio.sleep(0.01)
                            continue
                            
                        pcm_data = self.input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    except IOError as e:
                        # Handle "Unanticipated host error" (often temporary)
                        if e.errno == -9999:
                            await asyncio.sleep(0.1)
                            continue
                        raise e
                    
                    # ACOUSTIC FEEDBACK PREVENTION FOR RECORDING:
                    # If muted (TTS sequencing) or playing (audio active), discard mic input
                    if self.stt_muted or self.is_playing:
                        # Echo suppression active - do not send to STT
                        # print("M", end="", flush=True) # Muted
                        await asyncio.sleep(0.01)
                        continue

                    # logger.debug("Reading audio...")
                    self.add_to_recording(pcm_data)
                    # logger.debug("Read audio")

                    # Send raw PCM directly — Deepgram is configured for linear16/16kHz
                    if config.STT_PROVIDER == 'deepgram':
                        stt_audio = pcm_data  # linear16 matches mic format exactly
                    else:
                        stt_audio = pcm_data  # Sarvam also takes raw PCM

                    if not self.is_farewell:
                        if self.stt_service:
                            # print(".", end="", flush=True) # Sending
                            await self.stt_service.process_audio(stt_audio)
                        else:
                            pass # print("X", end="", flush=True) # No Service
                    
                    # Small sleep to prevent CPU overload
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    if not self.should_stop:
                        logger.error(f"[ERROR] Error reading audio: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"[ERROR] Recording loop error: {e}", exc_info=True)
        finally:
            self.is_recording = False
    
    def _audio_playback_thread(self):
        """
        Dedicated audio thread — owns all PyAudio writes.
        Reads PCM data from _pcm_queue and writes to output_stream in a tight loop.
        Running writes in a dedicated thread (vs thread pool) gives consistent
        inter-write timing, eliminating buffer underruns that cause static/crackle.
        """
        logger.info("[AUDIO THREAD] Started")
        while not self._audio_thread_stop.is_set():
            try:
                pcm_data = self._pcm_queue.get(timeout=0.05)
                if pcm_data is None:  # sentinel — stop the thread
                    self._pcm_queue.task_done()
                    break
                if self.output_stream and not self.output_stream.is_stopped():
                    self.output_stream.write(pcm_data)
                self._pcm_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                if not self._audio_thread_stop.is_set():
                    err = str(e)
                    if "Stream closed" not in err and "-9988" not in err:
                        logger.error(f"[AUDIO THREAD] Write error: {e}")
                self._pcm_queue.task_done() if not self._pcm_queue.empty() else None
        logger.info("[AUDIO THREAD] Stopped")

    def _ensure_audio_thread(self):
        """Start the audio playback thread if it's not already running."""
        if self._audio_thread is None or not self._audio_thread.is_alive():
            self._audio_thread_stop.clear()
            self._audio_thread = threading.Thread(
                target=self._audio_playback_thread,
                daemon=True,
                name="AudioPlayback"
            )
            self._audio_thread.start()
            logger.info("[AUDIO THREAD] Launched dedicated playback thread")

    async def play_audio(self, audio_chunk: bytes, action: str):
        """Play audio through speakers."""
        try:
            if action == "clearAudio":
                logger.info("[AUDIO] Clear audio buffer")
                # Drain the queue without playing
                while not self._pcm_queue.empty():
                    try:
                        self._pcm_queue.get_nowait()
                        self._pcm_queue.task_done()
                    except queue.Empty:
                        break
                return
            
            if action == "finishAudio":
                if self._playback_callback_fired:
                    return
                self._playback_callback_fired = True
                
                # logger.info("[AUDIO] Waiting for playback to nearly finish...")
                # Poll queue until ~100ms remains (approx 5 chunks @ 20ms/chunk)
                # This unblocks STT earlier so we catch user's first words
                while self._pcm_queue.qsize() > 5:
                    await asyncio.sleep(0.02)
                
                # allow room reverb to settle
                await asyncio.sleep(0.5)
                
                self.stt_muted = False    # Unmute STT (resume listening)
                self.is_playing = False
                logger.info("[AUDIO] Playback finished + 500ms delay - listening resumed")
                
                # Optional: continue waiting for full drain if needed, but returning early allows flow to proceed
                return
            
            if action == "playAudio" and audio_chunk:
                self._playback_callback_fired = False  # Reset flag for new playback
                if not self.is_playing:
                    self.is_playing = True
                    self._ensure_audio_thread()
                
                if not self.output_stream:
                    logger.warning("[AUDIO] Output stream not available")
                    return
                
                # Cartesia sends mulaw, Sarvam sends PCM
                if config.TTS_PROVIDER == 'cartesia':
                    pcm_data = mulaw_to_pcm(audio_chunk, width=2)
                else:
                    pcm_data = audio_chunk
                
                # Add to recording
                self.add_to_recording(pcm_data)
                
                # Enqueue for audio thread — instant, non-blocking, event loop stays free
                try:
                    self._pcm_queue.put_nowait(pcm_data)
                except queue.Full:
                    logger.warning("[AUDIO] PCM queue full — dropping chunk to avoid lag")
                    
        except Exception as e:
            error_msg = str(e)
            if "Stream closed" not in error_msg and "-9988" not in error_msg:
                logger.error(f"[ERROR] Error in play_audio: {e}")
            self.is_playing = False
    
    async def check_silence_timeout(self):
        """Monitor silence and auto-shutdown after timeout."""
        try:
            while not self.should_stop and self.is_recording and not self.call_ended:
                await asyncio.sleep(1)
                
                if self.is_playing:
                    self.last_user_speech_time = datetime.now().timestamp()
                    continue
                
                if self.last_user_speech_time:
                    silence_duration = datetime.now().timestamp() - self.last_user_speech_time
                    
                    # 1. 30s Check-in Prompt
                    if silence_duration >= SILENCE_PROMPT_TIMEOUT and not self._prompted_silence:
                        self._prompted_silence = True
                        msg = "Are you still there?"
                        logger.info(f"[TIMEOUT] Sending check-in prompt: '{msg}'")
                        
                        # Use TTS to prompt
                        async def prompt_callback(chunk, action):
                            await self.play_audio(chunk, action)
                        
                        if config.TTS_PROVIDER in ('deepgram', 'smallest'):
                            receive_task = asyncio.create_task(
                                self.tts_service.receive_audio(prompt_callback)
                            )
                            await self.tts_service.send_text(msg)
                            await receive_task
                        else:
                            await self.tts_service.synthesize(msg, prompt_callback)
                        
                        await self.play_audio(b"", "finishAudio")
                        # Reset timer slightly so we don't immediately hit the 60s shutdown
                        # self.last_user_speech_time = datetime.now().timestamp()
                    
                    # 2. 60s Hard Timeout
                    if silence_duration >= RECORD_TIMEOUT:
                        print(f"\\n[TIMEOUT] No speech detected for {RECORD_TIMEOUT} seconds")
                        await self.graceful_shutdown()
                        break
                        
        except Exception as e:
            logger.error(f"[ERROR] Silence timeout check error: {e}")
    
    def check_for_stop_words(self, text: str) -> bool:
        """Check if text contains any stop words."""
        text_lower = text.lower()
        for stop_word in STOP_WORDS:
            if stop_word in text_lower:
                return True
        return False
    
    def _extract_hour_from_text(self, text: str) -> int | None:
        """
        Extract hour (0-23) from spoken time text.
        Returns None if no time found.
        Examples: 'eleven pm' -> 23, '10 pm' -> 22, '5 o clock' -> 5
        """
        import re
        
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "midnight": 0, "noon": 12
        }
        
        text = text.lower().strip()
        
        # Special cases
        if "midnight" in text:
            return 0
        if "noon" in text:
            return 12
        
        # Try numeric: "10 pm", "10:30 pm", "10am"
        match = re.search(r'\b(\d{1,2})(?::\d{2})?\s*(am|pm)\b', text)
        if match:
            hour = int(match.group(1))
            meridiem = match.group(2)
            if meridiem == "pm" and hour != 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0
            return hour
        
        # Try word form: "eleven pm", "five am"
        for word, num in word_to_num.items():
            if word in text:
                if "pm" in text and num != 12:
                    return num + 12
                elif "am" in text and num == 12:
                    return 0
                elif "pm" in text or "am" in text:
                    return num
                # No meridiem — try to infer from context
                # "tonight" + hour word -> assume PM
                if any(w in text for w in ["tonight", "night", "evening"]):
                    return num + 12 if num < 12 else num
                if any(w in text for w in ["morning"]):
                    return num
        
        return None
    
    def _validate_visit_datetime(self, text: str) -> str | None:
        """
        Validate a site visit date/time request.
        Returns a rejection message string if invalid, or None if valid/unknown.
        """
        text_lower = text.lower().strip()
        
        # ── Past date detection (keywords + calendar dates) ──────────
        if self._is_past_date(text_lower):
            return "I'm sorry, I can't schedule a visit in the past. Could you suggest a date from today onwards?"
        
        # ── Time validation (8 AM - 9 PM) ───────────────────────────
        hour = self._extract_hour_from_text(text_lower)
        if hour is not None:
            if hour < 8 or hour >= 21:
                return "Site visits are available between 8 AM and 9 PM. What time works best for you?"
        
        return None  # Valid or unrecognised — let LLM handle

    def _is_past_date(self, text: str) -> bool:
        """
        Check if text contains a date reference that is in the past.
        Handles: 'February second', 'Feb 2nd', 'March 5', 'january tenth', etc.
        """
        import re
        from datetime import datetime, date
        
        text_lower = text.lower().strip()
        today = date.today()
        
        # Keyword-based past detection
        past_keywords = ["yesterday", "day before", "last week", "last month",
                         "last year", "two days ago", "three days ago"]
        if any(p in text_lower for p in past_keywords):
            return True
        
        # Month name mapping
        months = {
            "january": 1, "jan": 1, "february": 2, "feb": 2,
            "march": 3, "mar": 3, "april": 4, "apr": 4,
            "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
            "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
            "october": 10, "oct": 10, "november": 11, "nov": 11,
            "december": 12, "dec": 12
        }
        
        # Ordinal word mapping
        ordinals = {
            "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
            "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
            "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
            "nineteenth": 19, "twentieth": 20, "twenty first": 21, "twenty second": 22,
            "twenty third": 23, "twenty fourth": 24, "twenty fifth": 25,
            "twenty sixth": 26, "twenty seventh": 27, "twenty eighth": 28,
            "twenty ninth": 29, "thirtieth": 30, "thirty first": 31
        }
        
        # Try to find a month
        found_month = None
        for month_name, month_num in months.items():
            if month_name in text_lower:
                found_month = month_num
                break
        
        if found_month is None:
            return False
        
        # Try to find a day — ordinal words first, then numeric
        found_day = None
        for ord_word, day_num in ordinals.items():
            if ord_word in text_lower:
                found_day = day_num
                break
        
        if found_day is None:
            # Try numeric: "Feb 2", "February 2nd", "March 15th"
            match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', text_lower)
            if match:
                found_day = int(match.group(1))
        
        if found_day is None or found_day < 1 or found_day > 31:
            return False
        
        # Build the date (assume current year)
        try:
            target_date = date(today.year, found_month, found_day)
            return target_date < today
        except ValueError:
            return False  # Invalid date like Feb 30 — let LLM handle

    def _clean_for_tts(self, text: str) -> str:
        """Remove special characters that TTS might try to speak."""
        import re
        # Remove ?, !, *, #, and other non-speech characters
        cleaned = re.sub(r'[?!*#@&^~`|<>{}\[\]\\]', '', text)
        # Collapse double spaces
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
        return cleaned.strip()

    def _check_farewell(self, text: str) -> bool:
        """Return True if user is clearly saying goodbye."""
        farewell_phrases = [
            "bye", "goodbye", "good bye", "see you", "see ya", "take care",
            "thanks bye", "thank you bye", "that's all", "thats all",
            "i'm done", "im done", "end call", "hang up", "no more",
            "nothing else", "all good bye", "ok bye", "okay bye"
        ]
        text_lower = text.lower().strip()
        return any(p in text_lower for p in farewell_phrases)

    def _validate_bhk(self, text: str) -> str | None:
        """
        Validate BHK preference input.
        Returns rejection message if invalid, None if valid.
        """
        text_lower = text.lower().strip()
        valid_3bhk = ["3 bhk", "3bhk", "three bhk", "three bedroom", "three b h k", "3 b h k"]
        valid_4bhk = ["4 bhk", "4bhk", "four bhk", "four bedroom", "four b h k", "4 b h k"]
        invalid_bhk = ["1 bhk", "1bhk", "one bhk", "2 bhk", "2bhk", "two bhk",
                       "5 bhk", "5bhk", "five bhk", "studio", "penthouse"]

        if any(p in text_lower for p in invalid_bhk):
            return "We only have 3 B H K and 4 B H K options at Brigade Eternia. Which would you prefer?"

        if any(p in text_lower for p in valid_3bhk + valid_4bhk):
            return None  # Valid

        # Vague answers — let LLM handle (don't reject, don't advance)
        return None

    def _validate_urgency_timeline(self, text: str) -> str | None:
        """
        Reject nonsensical urgency timelines (past years, impossible dates).
        Returns rejection message if invalid, None if valid.
        """
        import re
        text_lower = text.lower().strip()

        # Past absolute years (e.g. "1990", "2020", "2019")
        year_match = re.search(r'\b(19\d{2}|202[0-4])\b', text_lower)
        if year_match:
            return f"I think that date has already passed! Are you planning to buy in the near future, like this year or next?"

        # Past relative phrases
        past_phrases = ["last year", "last month", "already bought", "already purchased",
                        "i already have", "i bought", "years ago", "decade ago"]
        if any(p in text_lower for p in past_phrases):
            return "It sounds like that's already in the past! When are you planning to make your next purchase?"

        return None

    def _check_identity_confirmation(self, text: str) -> bool:
        """
        Return True only if user gives a clear identity confirmation.
        Rejects weak sounds like 'uh', 'hmm', 'er'.
        """
        text_lower = text.lower().strip()
        # Weak/ambiguous sounds — NOT a confirmation
        weak_sounds = ["uh", "um", "hmm", "hm", "er", "ah", "oh"]
        if text_lower in weak_sounds or (len(text_lower) <= 3 and text_lower not in ["yes", "yep", "yup", "ji", "ha"]):
            return False
        # Strong confirmation words
        strong_confirm = ["yes", "yeah", "yep", "yup", "speaking", "this is", "i am",
                          "iam", "correct", "right", "that's me", "thats me", "ji", "haan",
                          "sir", "of course", "absolutely"]
        return any(p in text_lower for p in strong_confirm)

    def _check_out_of_scope(self, text: str) -> str | None:
        """
        Detect clearly out-of-scope topics and return a fixed deflection.
        Returns deflection message if out-of-scope, None if normal.
        """
        text_lower = text.lower().strip()
        out_of_scope_topics = [
            # Politics
            "modi", "rahul gandhi", "bjp", "congress", "election", "vote", "politics",
            # Sports
            "cricket", "ipl", "football", "match", "score", "world cup", "virat", "dhoni",
            # Entertainment
            "movie", "film", "actor", "actress", "bollywood", "netflix", "web series",
            # Weather
            "weather", "rain", "temperature", "forecast", "climate",
            # Other
            "stock market", "share price", "crypto", "bitcoin",
        ]
        if any(topic in text_lower for topic in out_of_scope_topics):
            return "I'm focused on Brigade Eternia today! Is there anything about the project I can help you with?"
        return None

    def start_recording(self):
        """Initialize recording session."""
        if not self.recording_enabled:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_filename = os.path.join(self.recordings_dir, f"property_call_{timestamp}.wav")
        self.recording_buffer = []
        logger.info(f"[RECORDING] Started - {self.recording_filename}")
        print(f"[RECORDING] Session will be saved to: {self.recording_filename}")
    
    def add_to_recording(self, audio_data: bytes):
        """Add audio chunk to recording buffer."""
        if not self.recording_enabled or not audio_data:
            return
        
        self.recording_buffer.append(audio_data)
    
    def save_recording(self):
        """Save recording buffer as WAV file."""
        if not self.recording_enabled or not self.recording_buffer:
            return
        
        try:
            combined_audio = b''.join(self.recording_buffer)
            
            with wave.open(self.recording_filename, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(combined_audio)
            
            file_size = os.path.getsize(self.recording_filename) / 1024
            duration = len(combined_audio) / (SAMPLE_RATE * 2)
            
            logger.info(f"[RECORDING] Saved - {self.recording_filename} ({file_size:.1f}KB, {duration:.1f}s)")
            print(f"\\n[RECORDING] Saved to: {self.recording_filename} ({file_size:.1f}KB)")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to save recording: {e}", exc_info=True)
    
    async def graceful_shutdown(self):
        """Gracefully shut down the session."""
        
        # Set ALL flags immediately to stop recording/listening
        self.call_ended = True
        self.is_farewell = True
        self.should_stop = True
        self.is_recording = False
        logger.info("[LOCAL] Graceful shutdown - all flags set")
        
        try:
            print("\n[SHUTDOWN] Initiating graceful shutdown...")
            
            # Display collected property data
            print("\n" + "=" * 60)
            print("COLLECTED PROPERTY INFORMATION")
            print("=" * 60)
            for key, value in self.collected_data.items():
                if value and value != "none":
                    print(f"{key.replace('_', ' ').title()}: {value}")
            print("=" * 60 + "\n")
            
            # Save recording if enabled
            self.save_recording()
            
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"[ERROR] Error during graceful shutdown: {e}")
    
    async def handle_transcription(self, text):
        """Handle transcribed text from STT."""

        # Guard: STT may pass a dict instead of string
        if isinstance(text, dict):
            text = text.get("text", text.get("transcript", text.get("message", "")))
        text = str(text) if text else ""

        if self.call_ended:
            logger.info("[LOCAL] Ignoring - call already ended")
            return

        # Prevent concurrent LLM calls
        if self.is_processing:
            logger.info("[LOCAL] Already processing, ignoring duplicate transcription")
            return

        self.is_processing = True

        try:
            # Handle force stop
            if text == "__FORCE_STOP__":
                if self.is_playing:
                    self.is_playing = False
                    logger.info("[INTERRUPT] Stopping AI playback")
                    await self.play_audio(None, "clearAudio")
                    if self.tts_service:
                        await self.tts_service.stop()
                return
            
            if not text or len(text.strip()) == 0:
                return

            # INTERRUPTION HANDLING
            if self.is_playing:
                if self.is_user_speaking:
                    logger.info("[INTERRUPT] Confirmed user interruption - stopping TTS")
                    self.is_playing = False
                    self.is_user_speaking = False
                    await self.tts_service.stop()
                    await asyncio.sleep(0.2)
                else:
                    logger.info("[INTERRUPT] Ignoring echo transcription")
                    return
            # Update last speech time
            self.last_user_speech_time = datetime.now().timestamp()

            print(f"\\n[TRANSCRIBED] User: {text}")
            
            # Check for stop words
            if self.check_for_stop_words(text):
                print(f"[STOP WORD DETECTED] Ending session...")
                await self.graceful_shutdown()
                return

            # ─────────────────────────────────────────────────────────
            # PYTHON-LEVEL VALIDATION HOOKS (bypass LLM entirely)
            # ─────────────────────────────────────────────────────────

            async def _speak_and_return(msg: str, intent_label: str = "validation") -> None:
                print(f"[RESPONSE] AI ({intent_label}): {msg}")
                async def _cb(chunk, action): await self.play_audio(chunk, action)
                await self.tts_service.synthesize(self._clean_for_tts(msg), _cb)
                print("[LISTENING] Listening for your response...\\n")

            # 1. OUT-OF-SCOPE GUARD (runs on every turn)
            oos_msg = self._check_out_of_scope(text)
            if oos_msg:
                logger.info(f"[OOS] Deflecting out-of-scope: '{text}'")
                await _speak_and_return(oos_msg, "out_of_scope")
                return

            # 2. FAREWELL DETECTION (runs on every turn)
            if self._check_farewell(text):
                farewell_msg = f"It was a pleasure speaking with you. Have a great day, {self.greeting_name}!"
                logger.info(f"[FAREWELL] User said goodbye: '{text}'")
                print(f"[RESPONSE] AI (farewell): {farewell_msg}")
                async def _fare_cb(chunk, action): await self.play_audio(chunk, action)
                await self.tts_service.synthesize(farewell_msg, _fare_cb)
                await self.graceful_shutdown()
                return

            # 3. IDENTITY CHECK — reject weak confirmations
            if self.current_stage == "identity_check":
                if not self._check_identity_confirmation(text):
                    # Weak sound — ask again without calling LLM
                    logger.info(f"[IDENTITY] Weak confirmation ignored: '{text}'")
                    clarify = f"Sorry, I didn't catch that. Am I speaking with {self.greeting_name}?"
                    await _speak_and_return(clarify, "identity_check")
                    return

            # 4. BHK VALIDATION
            if self.current_stage == "bhk_preference":
                bhk_msg = self._validate_bhk(text)
                if bhk_msg:
                    logger.info(f"[BHK] Invalid BHK input: '{text}'")
                    await _speak_and_return(bhk_msg, "bhk_preference")
                    return

            # 5. URGENCY TIMELINE VALIDATION
            if self.current_stage == "urgency_assessment":
                urgency_msg = self._validate_urgency_timeline(text)
                if urgency_msg:
                    logger.info(f"[URGENCY] Invalid timeline: '{text}'")
                    await _speak_and_return(urgency_msg, "urgency_assessment")
                    return

            # Get LLM response
            print("[THINKING] Processing...")

            # STEP 1: Update stages based on user input (backend state machine)
            text_lower = text.lower()
            
            import re
            def has_word(text, words):
                """Check if any whole word from list exists in text."""
                pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
                return bool(re.search(pattern, text))

            if self.current_stage == "identity_check":
                positive = ["yes", "speaking", "this is", "yeah", "yep", 
                            "i am", "iam", "sir", "ji", "haan", "correct",
                            "right", "that's me", "thats me"]
                if has_word(text_lower, positive):
                    if "identity_check" not in self.completed_stages:
                        self.completed_stages.append("identity_check")
                    self.current_stage = "self_intro"  # Always introduce before timing check

            elif self.current_stage == "self_intro":
                # Any response from user after intro → move to timing check
                if len(text_lower.strip()) > 0:
                    if "self_intro" not in self.completed_stages:
                        self.completed_stages.append("self_intro")
                    self.current_stage = "timing_check"

            elif self.current_stage == "timing_check":
                positive = ["yes", "yeah", "sure", "go ahead", "yep", "okay", "ok", "fine", 
                            "absolutely", "sir", "ji", "haan", "please", "of course", 
                            "why not", "definitely", "good time", "go on"]
                negative = ["no", "not now", "busy", "later", "call back", "bad time"]
                
                if has_word(text_lower, positive) and not has_word(text_lower, negative):
                    if "timing_check" not in self.completed_stages:
                        self.completed_stages.append("timing_check")
                    self.current_stage = "bhk_preference"

            elif self.current_stage == "bhk_preference":
                bhk_words = ["3 bhk", "3bhk", "three bhk", "three bedroom",
                             "three", "3 bed", "4 bhk", "4bhk", "four bhk", 
                             "four bedroom", "four", "4 bed",
                             "first one", "first option", "second one", "second option",
                             "third one", "third option", "the first", "the second", "the third"]
                if has_word(text_lower, bhk_words):
                    if "bhk_preference" not in self.completed_stages:
                        self.completed_stages.append("bhk_preference")
                    self.current_stage = "urgency_assessment"

            elif self.current_stage == "urgency_assessment":
                # Any answer to urgency moves forward
                if len(text_lower.strip()) > 2:  # Any real response
                    if "urgency_assessment" not in self.completed_stages:
                        self.completed_stages.append("urgency_assessment")
                    self.current_stage = "site_visit_scheduling"

            elif self.current_stage == "site_visit_scheduling":
                past_date_phrases = ["last week", "last month", "yesterday", "day before", "last year"]
                is_past = any(p in text_lower for p in past_date_phrases) or self._is_past_date(text_lower)
                future_time_words = ["tomorrow", "today", "weekend", "monday", "tuesday",
                                     "wednesday", "thursday", "friday", "saturday", "sunday",
                                     "morning", "evening", "afternoon", "night", "next week",
                                     "next month", "this week", "this month", "next year"]
                if not is_past and has_word(text_lower, future_time_words):
                    self.current_stage = "post_visit_confirmed"

            logger.info(f"[STAGE] Current: {self.current_stage} | Done: {self.completed_stages}")

            # STEP 2: Build clean runtime context (NO stage note pollution in user text)
            stage_info = STAGE_DEFINITIONS.get(self.current_stage, STAGE_DEFINITIONS["identity_check"])
            
            # ─────────────────────────────────────────────────────────
            # BUSINESS HOURS ENFORCEMENT (Python-level, bypasses LLM)
            # ─────────────────────────────────────────────────────────
            if getattr(self, "_awaiting_callback_time", False):
                # Reject past dates (calendar dates + keywords like "yesterday")
                if self._is_past_date(text_lower):
                    rejection = "I can only schedule callbacks for a future date and time between 8 AM and 9 PM. When works best?"
                    logger.info(f"[CALLBACK] Rejected past date: '{text}'")
                    await _speak_and_return(rejection, "reschedule")
                    return
                # Try to extract an hour from the user's text
                hour = self._extract_hour_from_text(text_lower)
                if hour is not None:
                    if hour < 8 or hour >= 21:  # Outside 8 AM - 9 PM
                        rejection = "I can only schedule callbacks between 8 AM and 9 PM. What time works best for you?"
                        logger.info(f"[CALLBACK] Rejected time (hour={hour}): '{text}'")
                        await _speak_and_return(rejection, "reschedule")
                        return
                    else:
                        # Valid time — confirm directly without going to LLM
                        # (If LLM handles this, it sometimes re-validates and re-asks incorrectly)
                        self._awaiting_callback_time = False
                        self.slots["callback_scheduled"] = "yes"
                        
                        # Format extracted hour for speech
                        formatted_time = f"{hour - 12} PM" if hour > 12 else (f"{hour} PM" if hour == 12 else f"{hour} AM")
                        if hour == 0: formatted_time = "12 AM"
                        if hour == 12: formatted_time = "12 PM"
                        
                        logger.info(f"[CALLBACK] Valid time accepted (hour={hour}): '{text}' — confirming directly")
                        confirmation = f"Perfect, I'll schedule a callback for you at {formatted_time}. You'll receive a confirmation on WhatsApp shortly. It was a pleasure speaking with you — have a great day, {self.greeting_name}!"
                        await _speak_and_return(confirmation, "reschedule")
                        await self.graceful_shutdown()
                        return
                else:
                    # No hour detected — re-ask instead of falling through to LLM
                    re_ask = "Could you please tell me a specific time for the callback? For example, tomorrow at 3 PM."
                    logger.info(f"[CALLBACK] No valid time detected, re-asking: '{text}'")
                    await _speak_and_return(re_ask, "reschedule")
                    return

            # ─────────────────────────────────────────────────────────
            # SITE VISIT DATE/TIME ENFORCEMENT (Python-level, bypasses LLM)
            # ─────────────────────────────────────────────────────────
            if getattr(self, "_awaiting_visit_datetime", False):
                rejection = self._validate_visit_datetime(text_lower)
                if rejection:
                    logger.info(f"[VISIT] Rejected date/time: '{text}'")
                    print(f"[RESPONSE] AI (flow_progress): {rejection}")
                    async def visit_audio_cb(chunk, action): await self.play_audio(chunk, action)
                    await self.tts_service.synthesize(rejection, visit_audio_cb)
                    print("[LISTENING] Listening for your response...\\n")
                    return
                else:
                    # Valid — clear flag, let LLM confirm
                    self._awaiting_visit_datetime = False
                
            # Reset silence prompt flag whenever user speaks
            self._prompted_silence = False

            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": text})

            # Build format values for prompt template
            format_values = {
                "user_name": self.greeting_name,
                "user_message": text,
                "current_stage": self.current_stage,
                "stage_goal": stage_info["goal"],
                "slots": json.dumps(self.slots),
                "current_date": datetime.now().strftime("%B %d, %Y")
            }

            # STEP 3: Stream LLM response (Full) → dispatch to Deepgram WebSocket
            # We accumulate the full response then send it as one clean text block.
            # Deepgram handles the streaming audio generation.
            recent_history = self.conversation_history[-10:] if self.conversation_history else []
            stream_start = time.time()

            # Architecture Check: Deepgram WebSocket vs Legacy REST
            is_streaming_tts = config.TTS_PROVIDER in ('deepgram', 'smallest')
            
            # Pipeline components
            receive_task = None
            
            if is_streaming_tts:
                # ─── DEEPGRAM WEBSOCKET PIPELINE ────────────────────────────
                # Connection is already persistent from start_session
                # Just start receiver task
                
                # 2. Start receiver task
                async def on_audio_chunk(chunk: bytes):
                    if not self.stt_muted and chunk:
                        self.stt_muted = True 
                        # TTFB Log
                        print(f"[{config.TTS_PROVIDER.upper()}] First audio bytes received")
                    
                    if chunk:
                        await self.play_audio(chunk, "playAudio")

                receive_task = asyncio.create_task(
                    self.tts_service.receive_audio(on_audio_chunk)
                )

            stream_start = time.time()
            full_ai_text = ""
            final_emotion = None
            
            # 3. Get API response (now yields once with full text)
            # Retry loop for self-intro check
            max_retries = 1
            retry_count = 0
            
            while retry_count <= max_retries:
                full_ai_text = ""
                final_emotion = None
                
                async for chunk_text, is_final, emotion in self.llm_service.stream_response(
                    user_input=text,
                    conversation_history=recent_history,
                    format_values=format_values,
                ):
                     full_ai_text = chunk_text
                     final_emotion = emotion
                
                # Self-Intro Hard Check
                if self.current_stage == "self_intro" and not is_streaming_tts:
                    # Check if agent introduced itself
                    if "rohan" not in full_ai_text.lower():
                        if retry_count < max_retries:
                            logger.warning(f"[SELF-INTRO] Agent failed to introduce itself. Retrying... (Attempt {retry_count+1})")
                            print("[RETRY] Agent forgot name in self-intro. Triggering retry...")
                            # Add explicit instruction to history for the retry
                            recent_history.append({"role": "system", "content": "CRITICAL: You MUST introduce yourself as 'Rohan from JLL Homes' in this turn. Do it now."})
                            retry_count += 1
                            continue
                        else:
                            logger.error("[SELF-INTRO] Failed to introduce after retry.")
                            break
                    else:
                        break # Success
                else:
                    break # Not self-intro stage or streaming mode (can't retry easily)

            elapsed = time.time() - stream_start
            logger.info(f"[LLM] Response complete in {elapsed:.2f}s ({len(full_ai_text)} chars) → sending to TTS")

            if is_streaming_tts:
                 if full_ai_text:
                      # Send full text
                      await self.tts_service.send_text(full_ai_text)
                      if hasattr(self.tts_service, 'flush'):
                          await self.tts_service.flush()
                      print(f"[{config.TTS_PROVIDER.upper()}] Text sent — waiting for audio")
                      
                      if receive_task:
                           await receive_task
                      
                      # Keep connection alive between turns
                      if hasattr(self.tts_service, 'keepalive'):
                          await self.tts_service.keepalive()
                      elif hasattr(self.tts_service, 'keep_alive'):
                          await self.tts_service.keep_alive()
            else:
                 # Legacy Fallback (Synthesize full text as one chunk)
                 if full_ai_text:
                     print("[TTS] Synthesizing full response (Legacy)...")
                     try:
                         # Use standard synthesize method with callback
                         # import emotion_config as _ec
                         # params = _ec.get_emotion_params(final_emotion or "default")
                         
                         async def legacy_audio_callback(chunk, action="playAudio"):
                             if chunk:
                                 self.stt_muted = True
                                 await self.play_audio(chunk, action)
                         
                         await self.tts_service.synthesize(full_ai_text, legacy_audio_callback)
                     except Exception as e:
                         logger.error(f"[TTS] Legacy synthesis failed: {e}")
            
            elapsed_total = time.time() - stream_start
            logger.info(f"[TIMING] Turn handling complete in {elapsed_total:.2f}s")
            
            # Add to history
            if full_ai_text:
                # Add full AI text to history logic...
                # Note: We append the emotion tag for the history record so it persists
                full_ai_text_with_emotion = full_ai_text
                if final_emotion:
                     full_ai_text_with_emotion += f"\n[EMOTION: {final_emotion}]"
                elif not final_emotion and "[EMOTION:" not in full_ai_text:
                     # Attempt to re-extract if it was stripped but not captured in final_emotion
                     pass 
                
                self.conversation_history.append({"role": "assistant", "content": full_ai_text_with_emotion})
                
                # Wait for all audio to finish playing
                await self.play_audio(b"", "finishAudio")

            # Retrieve meta parsed from the full JSON (intent, slots, end_call, etc.)
            response = self.llm_service.last_response_meta
            ai_text  = full_ai_text.strip()

            # STEP 4b: Strip any residual emotion tag from assembled text (safety net)
            ai_text, _residual_emotion = extract_emotion(ai_text)
            if not final_emotion:
                final_emotion = _residual_emotion
            
            # Unit-level Debug Log for Emotion
            if not final_emotion:
                logger.debug(f"[EMOTION CHECK] Emotion is None. Raw AI text end: '{ai_text[-100:] if len(ai_text) > 100 else ai_text}'")

            if not ai_text or not ai_text.strip():
                logger.error("[ERROR] LLM returned empty response")
                ai_text = "I'm sorry, I didn't quite get that. Could you please repeat?"

            intent = response.get("intent", "unknown")
            print(f"[RESPONSE] AI ({intent}): {ai_text}")

            # REMOVED duplicate history append: self.conversation_history.append({"role": "assistant", "content": ai_text})

            # ─────────────────────────────────────────────────────────
            # CALLBACK TIME FLAG DETECTION
            # Set _awaiting_callback_time whenever the LLM's reply is
            # offering/asking for a callback time — regardless of intent label.
            # Use a broad phrase set to catch all LLM phrasings.
            # ─────────────────────────────────────────────────────────
            callback_ask_phrases = [
                "when would be a good time",
                "what time works",
                "when can i call",
                "when should i call",
                "let me know what time",
                "works best for you",
                "prefer a callback",
                "schedule a callback",
                "best time to reach",
                "when to call",
                "when would work",
                "good time for a callback",
                "time for a callback",
                "callback at what time",
                "when are you free",
                "when would you be free",
                "between 8 am and 9 pm",
                "between 8am and 9pm",
            ]
            visit_phrases = ["site visit", "visit the site", "schedule a visit", "visit us", "come visit"]
            is_asking_callback_time = (
                any(p in ai_text.lower() for p in callback_ask_phrases)
                and not any(p in ai_text.lower() for p in visit_phrases)
            )
            if is_asking_callback_time:
                self._awaiting_callback_time = True
                logger.info("[CALLBACK] Now awaiting callback time from user")
            elif intent not in ("reschedule", "flow_progress"):
                # Only clear the flag if we've moved to a clearly unrelated intent
                self._awaiting_callback_time = False

            # Set flag if LLM is asking for site visit date/time
            visit_ask_phrases = ["what day", "what time works for your visit", "when would you like to visit",
                                  "when can you visit", "schedule a visit", "day and time works"]
            is_asking_visit_time = (
                self.current_stage == "site_visit_scheduling"
                and any(p in ai_text.lower() for p in visit_ask_phrases)
            )
            if is_asking_visit_time:
                self._awaiting_visit_datetime = True
                logger.info("[VISIT] Now awaiting visit date/time from user")
            elif self.current_stage != "site_visit_scheduling":
                self._awaiting_visit_datetime = False

            # Reset silence timer after LLM responds
            self.last_user_speech_time = datetime.now().timestamp()

            # STEP 5: Update slots from LLM output (parsed from full JSON after stream end)
            if "raw_model_data" in response:
                raw_data = response["raw_model_data"]
                self.collected_data = raw_data
                
                # Update slots with non-null values from LLM
                for slot_key in self.slots:
                    val = raw_data.get(slot_key)
                    if val and val != "none" and val != "null":
                        self.slots[slot_key] = val
                
                logger.info(f"[SLOTS] {self.slots}")

            # Detect farewell BEFORE checking end-call so we can block STT
            farewell_phrases = ["have a wonderful day", "have a great day", "pleasure speaking"]
            if any(phrase in ai_text.lower() for phrase in farewell_phrases):
                self.is_farewell = True
                self.call_ended = True
                logger.info("[LOCAL] Farewell detected - blocking STT")

            # Check if should end call AFTER TTS finishes
            if response.get("should_end_call") or self.call_ended:
                print("[END CALL] Call ended after farewell")
                await self.graceful_shutdown()
                return

            print("[LISTENING] Listening for your response...\\n")
            
        except Exception as e:
            logger.error(f"[ERROR] Error handling transcription: {e}", exc_info=True)
            print(f"\\nERROR: {e}\\n")
        finally:
            self.is_processing = False  # Always release lock
    
    async def start_session(self):
        """Start the local testing session."""
        try:
            self.should_stop = False
            self.session_start = datetime.now()
            self.start_recording()
            
            if not await self.initialize_services():
                return
            
            if not self.setup_audio_streams():
                return
            
            # Send initial greeting (identity confirmation - STEP 1)
            print("\n[TTS] Sending identity confirmation...")
            
            # Use centralized greeting from config
            welcome_greeting = config.GREETING_TEMPLATE.format(name=self.greeting_name)

            # Don't manually track - LLM service handles history internally

            # Add to LLM's internal history so it knows Stage 1 was already said
            self.llm_service.add_to_history("assistant", welcome_greeting)
            # Add to local conversation history for persistent context across turns
            self.conversation_history.append({"role": "assistant", "content": welcome_greeting})
            self.current_stage = "identity_check"
            logger.info(f"[STAGE] Welcome greeting added to history: {welcome_greeting}")

            async def welcome_callback(audio_chunk: bytes, action: str = "playAudio"):
                if not self.stt_muted and audio_chunk:
                    self.stt_muted = True
                    print(f"[{config.TTS_PROVIDER.upper()}] Greeting started — STT muted")
                await self.play_audio(audio_chunk, action)

            if config.TTS_PROVIDER in ('deepgram', 'smallest'):
                # Start receiver task FIRST so we don't miss any chunks
                receive_task = asyncio.create_task(
                    self.tts_service.receive_audio(welcome_callback)
                )
                
                # Send text into existing connection
                await self.tts_service.send_text(welcome_greeting)
                
                if config.TTS_PROVIDER != 'smallest' and hasattr(self.tts_service, 'flush'):
                    await self.tts_service.flush()

                # Wait for turn completion
                await receive_task
                
                # Reset muting for welcome greeting finishes
                self.stt_muted = False
                print(f"[{config.TTS_PROVIDER.upper()}] Greeting finished — STT unmuted")
            else:
                await self.tts_service.synthesize(welcome_greeting, welcome_callback)
                self.stt_muted = False
            
            # Reset playback state so we can start listening
            await self.play_audio(b"", "finishAudio")
            
            # Initialize timeout tracking
            self.last_user_speech_time = datetime.now().timestamp()
            
            # Start silence timeout checker
            self.silence_check_task = asyncio.create_task(self.check_silence_timeout())
            
            # Start recording loop
            await self.record_audio_loop()
            
        except Exception as e:
            logger.error(f"[ERROR] Session error: {e}", exc_info=True)
            print(f"\\nERROR - SESSION FAILED: {e}\\n")
        
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.tts_service:
            await self.tts_service.close()
        try:
            print("\\n[CLEANUP] Closing services...")
            
            # 1. Set flags first - stop all processing
            self.is_recording = False
            self.call_ended = True
            self.is_processing = False
            
            # 2. Close STT FIRST - stops incoming transcriptions
            if self.stt_service:
                await self.stt_service.close()
            
            # 3. Small wait for any in-flight callbacks to complete
            await asyncio.sleep(0.2)
            
            # 4. Close TTS
            if self.tts_service:
                await self.tts_service.close()
            
            # 5. Close LLM
            if self.llm_service:
                await self.llm_service.close()
            
            # 6. Stop dedicated audio thread before closing hardware
            self._audio_thread_stop.set()
            try:
                self._pcm_queue.put_nowait(None)  # sentinel to unblock thread
            except Exception:
                pass
            if self._audio_thread and self._audio_thread.is_alive():
                self._audio_thread.join(timeout=2.0)
                logger.info("[AUDIO THREAD] Joined on cleanup")

            # 7. Close audio hardware LAST
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
            
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            
            self.audio.terminate()
            
            # Print session summary
            if self.session_start:
                duration = (datetime.now() - self.session_start).total_seconds()
                print(f"\\n[SESSION] Total duration: {duration:.1f} seconds")
                print(f"[SESSION] Messages exchanged: {len(self.conversation_history)}")
            
            print("\\nCOMPLETE - Cleanup done. Goodbye!\\n")
            
        except Exception as e:
            logger.error(f"[ERROR] Cleanup error: {e}")
