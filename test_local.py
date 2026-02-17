"""
Local Voice Client for Property Enquiry Testing

Allows testing STT, LLM, and TTS services locally using microphone and speakers
without telephony integration for the Property Enquiry Agent.
"""
import asyncio
import logging
import pyaudio
import audioop
import signal
import sys
import wave
import os
from datetime import datetime
from typing import Dict, Optional, List

import config
import prompts
from services.stt_factory import STTServiceFactory
from services.tts_factory import TTSServiceFactory
from services.llm_service import GroqLLMService
from utils.audio_utils import pcm_to_mulaw, mulaw_to_pcm

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

# Dynamic fields for Brigade Eternia site visit flow (matching main.py)
BRIGADE_ETERNIA_DYNAMIC_FIELDS = {
    "conversation_stage": {
        "type": "string",
        "description": "Stage",
        "default": "identity_check"
    },
    "preferred_bhk": {
        "type": "string",
        "description": "BHK",
        "default": "none"
    },
    "visit_date": {
        "type": "string",
        "description": "Date",
        "default": "none"
    },
    "visit_time": {
        "type": "string",
        "description": "Time",
        "default": "none"
    },
    "visit_confirmed": {
        "type": "string",
        "description": "Confirmed",
        "default": "no"
    }
}

# Note: Farewell is now handled by LLM in the system prompt (Stage 5)

# Stop words that trigger graceful shutdown
STOP_WORDS = [
    "end call", "hang up", "stop the call"  # Only hard commands, not conversational
]

# Timeout settings
RECORD_TIMEOUT = 15  # seconds - reduced from 30 of silence before auto-shutdown


class LocalVoiceClient:
    """Local voice client for testing property enquiry voice services."""
    
    def __init__(self, user_name: str = None, user_message: str = None):
        # User data (can be set from form or use defaults)
        self.user_name = user_name or DEFAULT_USER_NAME
        self.user_message = user_message or DEFAULT_USER_MESSAGE
        
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
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
    
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
            stt_init_success = await self.stt_service.initialize(api_key=stt_api_key, encoding="mulaw")
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
            tts_api_key = (
                config.CARTESIA_API_KEY if config.TTS_PROVIDER == 'cartesia'
                else config.SARVAM_API_KEY
            )
            
            if config.TTS_PROVIDER == 'cartesia':
                voice_id = config.CARTESIA_VOICE_ID
                tts_kwargs = {'model_id': 'sonic-english', 'speed': 'normal'}
            else:
                # Use config values for Sarvam
                voice_id = config.SARVAM_VOICE_ID or 'rohan'
                tts_kwargs = {'model': config.SARVAM_MODEL or 'bulbul:v3', 'language': 'en-IN', 'speed': 1.0}
            
            self.tts_service = TTSServiceFactory.create(
                provider=config.TTS_PROVIDER,
                api_key=tts_api_key,
                voice_id=voice_id,
                **tts_kwargs
            )
            await self.tts_service.initialize()
            print(f"[TTS] OK - {config.TTS_PROVIDER.title()} TTS initialized")
            
            # Initialize LLM
            print(f"[LLM] Creating Groq LLM service...")
            self.llm_service = GroqLLMService(api_key=config.GROQ_API_KEY, max_history=10)
            
            # Get system prompt with actual user data
            system_prompt = prompts.get_formatted_prompt(
                user_name=self.user_name,
                user_message=self.user_message
            )
            
            await self.llm_service.initialize(
                dynamic_fields=BRIGADE_ETERNIA_DYNAMIC_FIELDS,
                system_prompt_template=system_prompt
            )
            print(f"[LLM] OK - Groq LLM initialized")
            
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
            self.output_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
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
                        pcm_data = self.input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    except IOError as e:
                        # Handle "Unanticipated host error" (often temporary)
                        if e.errno == -9999:
                            await asyncio.sleep(0.1)
                            continue
                        raise e
                    
                    # ACOUSTIC FEEDBACK PREVENTION FOR RECORDING:
                    # Only add mic data to the recording buffer if AI is NOT playing.
                    # This prevents the AI's voice from being captured twice (direct + echo).
                    if not self.is_playing:
                        self.add_to_recording(pcm_data)
                    
                    import audioop

                    # Convert PCM to mulaw for STT
                    mulaw_data = pcm_to_mulaw(pcm_data, width=2)

                    if not self.is_farewell:
                        if not self.is_playing:
                            # Normal listening - always send to STT
                            if self.stt_service:
                                await self.stt_service.process_audio(mulaw_data)
                        else:
                            # Agent speaking - check if user is interrupting
                            rms = audioop.rms(pcm_data, 2)
                            if rms > self.interruption_threshold:
                                if not self.is_user_speaking:
                                    self.is_user_speaking = True
                                    logger.info(f"[INTERRUPT] User voice detected RMS:{rms} - stopping TTS immediately")
                                    
                                    # STOP TTS RIGHT NOW - don't wait for transcription
                                    self.is_playing = False
                                    if self.tts_service:
                                        await self.tts_service.stop()
                                
                                if self.stt_service:
                                    await self.stt_service.process_audio(mulaw_data)
                            else:
                                self.is_user_speaking = False
                    
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
    
    async def play_audio(self, audio_chunk: bytes, action: str):
        """Play audio through speakers."""
        try:
            if action == "clearAudio":
                logger.info("[AUDIO] Clear audio buffer")
                return
            
            if action == "finishAudio":
                self.is_playing = False
                logger.info("[AUDIO] Playback finished")
                return
            
            if action == "playAudio" and audio_chunk:
                if not self.is_playing:
                    self.is_playing = True
                
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
                
                # Play through speakers
                if self.output_stream:
                    self.output_stream.write(pcm_data)
                    
        except Exception as e:
            error_msg = str(e)
            if "Stream closed" not in error_msg and "-9988" not in error_msg:
                logger.error(f"[ERROR] Error playing audio: {e}")
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
    
    async def handle_transcription(self, text: str):
        """Handle transcribed text from STT."""

        if self.call_ended:
            logger.info("[LOCAL] Ignoring - call already ended")
            return

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
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": text
            })
            
            # Get LLM response
            print("[THINKING] Processing...")
            
            # LLM service manages conversation history internally
            response = await self.llm_service.generate_response(user_input=text)
            
            ai_text = response["response"].response
            
            # Validate response is not empty
            if not ai_text or not ai_text.strip():
                logger.error("[ERROR] LLM returned empty response")
                ai_text = "I'm sorry, I didn't quite get that. Could you please repeat?"
            
            print(f"[RESPONSE] AI: {ai_text}")
            
            # Update collected data
            if "raw_model_data" in response:
                self.collected_data = response["raw_model_data"]
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_text
            })
            
            # Synthesize response
            print("[SPEAKING] Playing audio...")
            
            async def audio_callback(audio_chunk: bytes, action: str):
                await self.play_audio(audio_chunk, action)

            # Detect farewell and set flag BEFORE synthesizing
            farewell_phrases = ["have a wonderful day", "have a great day", "pleasure speaking"]
            if any(phrase in ai_text.lower() for phrase in farewell_phrases):
                self.is_farewell = True
                self.call_ended = True
                logger.info("[LOCAL] Farewell detected - blocking STT")

            await self.tts_service.synthesize(ai_text, audio_callback)

            # Check if should end call AFTER TTS finishes
            if response.get("should_end_call") or self.call_ended:
                print("[END CALL] Call ended after farewell")
                await self.graceful_shutdown()
                return

            print("[LISTENING] Listening for your response...\\n")
            
        except Exception as e:
            logger.error(f"[ERROR] Error handling transcription: {e}", exc_info=True)
            print(f"\\nERROR: {e}\\n")
    
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
            
            async def welcome_callback(audio_chunk: bytes, action: str):
                await self.play_audio(audio_chunk, action)
            
            await self.tts_service.synthesize(welcome_greeting, welcome_callback)
            
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
        try:
            print("\\n[CLEANUP] Closing services...")
            
            self.is_recording = False
            
            # Close audio streams
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
            
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
            
            self.audio.terminate()
            
            # Close services
            if self.stt_service:
                await self.stt_service.close()
            
            if self.tts_service:
                await self.tts_service.close()
            
            if self.llm_service:
                await self.llm_service.close()
            
            # Print session summary
            if self.session_start:
                duration = (datetime.now() - self.session_start).total_seconds()
                print(f"\\n[SESSION] Total duration: {duration:.1f} seconds")
                print(f"[SESSION] Messages exchanged: {len(self.conversation_history)}")
            
            print("\\nCOMPLETE - Cleanup done. Goodbye!\\n")
            
        except Exception as e:
            logger.error(f"[ERROR] Cleanup error: {e}")
