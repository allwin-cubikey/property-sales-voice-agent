"""
Sarvam AI Text-to-Speech Service
Implements BaseTTSService for Sarvam AI's text-to-speech API
"""
import logging
import asyncio
import aiohttp
import base64
from typing import Callable, Optional, Dict, Any
from services.tts_base import BaseTTSService
import config
import re
import emotion_config
import time
logger = logging.getLogger(__name__)


class SarvamTTSService(BaseTTSService):
    """
    Sarvam AI TTS service with REST API synthesis
    Supports Hindi and English voices
    """
    
    def __init__(
        self, 
        api_key: str, 
        voice_id: str = None, 
        language: str = None,
        speed: float = None,
        model: str = None
    ):
        """
        Initialize Sarvam TTS service
        
        Args:
            api_key: Sarvam AI API key
            voice_id: Voice ID ("meera", "arvind", etc.)
            language: Language code ("hi-IN" for Hindi, "en-IN" for English)
            speed: Speaking pace (0.5 to 2.0, default 1.0)
            model: Model to use (default: "bulbul:v3")
        """
        self.api_key = api_key
        self.voice_id = voice_id or config.SARVAM_VOICE_ID or "priya"

        self.language = language or config.SARVAM_LANGUAGE
        self.speed = speed or config.SARVAM_SPEED
        self.model = model or config.SARVAM_MODEL
        self._session = None
        self._is_initialized = False
        self._current_task = None
        self._is_stopped = False
        self._last_spoken_text = ""
        
        logger.info(f"[TTS] SarvamTTSService instance created (voice={self.voice_id}, lang={self.language}, model={self.model})")
    
    async def initialize(self) -> bool:
        """
        Initialize TTS service and prepare for synthesis.
        
        Returns:
            True if initialization successful
        """
        try:
            self._session = aiohttp.ClientSession()
            self._is_initialized = True
            logger.info("[TTS] Sarvam TTS service initialized")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Sarvam TTS initialization failed: {e}", exc_info=True)
            return False

    async def prewarm_tts(self):
        """
        Send a dummy request to warm up the connection.
        This helps reduce latency for the first user interaction.
        """
        try:
            logger.info("[TTS] Pre-warming Sarvam TTS connection...")
            # Synthesize a single valid word - safe and fast
            await self._synthesize_chunk("Hello") 
            logger.info("[TTS] Connection pre-warmed")
        except Exception as e:
            logger.warning(f"[TTS] Pre-warm failed (non-critical): {e}")

    


    async def _synthesize_chunk(
        self,
        text: str,
        override_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[bytes]:
        """Helper to synthesize a single chunk of text, with optional emotion parameter overrides."""
        start_time = time.time()
        url = config.SARVAM_TTS_URL
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": text,
            "target_language_code": self.language or config.TTS_LANGUAGE,
            "speaker": self.voice_id or config.VOICE_ID,
            "pace": self.speed or config.SARVAM_SPEED,
            "speech_sample_rate": 16000,
            "enable_preprocessing": True,
            "model": self.model or config.SARVAM_MODEL
        }

        # Apply emotion pace override if provided
        # Note: bulbul:v3 does NOT support pitch or loudness — pace only.
        if override_params and "pace" in override_params:
            payload["pace"] = override_params["pace"]
        
        # Retry once on failure
        for attempt in range(2):
            try:
                if self._session is None or self._session.closed:
                    logger.warning("[TTS] Cannot synthesize chunk: session is closed")
                    return None
                    
                # Increased timeout to 30s to handle potential cold starts or network latency
                timeout = aiohttp.ClientTimeout(total=30)
                logger.debug(f"[TTS] Sending request to {url} with len(text)={len(payload['text'])}")
                async with self._session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"[ERROR] Sarvam API error {response.status}: {error_text[:200]}")
                        logger.error(f"[ERROR] Payload was: speaker={payload['speaker']}, lang={payload['target_language_code']}, model={payload['model']}")
                        if attempt == 0:
                            await asyncio.sleep(0.5)
                            continue
                        return None
                    
                    data = await response.json()
                    if "audios" in data and len(data["audios"]) > 0:
                        raw_audio = base64.b64decode(data["audios"][0])
                        elapsed = time.time() - start_time
                        logger.info(
                            f"[TTS] Segment synthesized in {elapsed:.2f}s "
                            f"({len(text.split())} words, {len(text)} chars)"
                        )
                        
                        # Strip WAV header if present (RIFF....WAVE)
                        if raw_audio[:4] == b'RIFF' and raw_audio[8:12] == b'WAVE':
                            logger.debug("[TTS] Stripping WAV header (44 bytes)")
                            return raw_audio[44:]
                        return raw_audio
                    logger.warning(f"[TTS] No audio in response for: '{text[:50]}...'")
                    return None
            except asyncio.TimeoutError:
                logger.error(f"[ERROR] Sarvam TTS timeout (attempt {attempt+1}): '{text[:40]}...'")
                if attempt == 0:
                    continue
                return None
            except Exception as e:
                if not self._is_stopped:
                    logger.error(f"[ERROR] Chunk synthesis failed ({type(e).__name__}): {e}")
                else:
                    logger.debug(f"[TTS] Chunk synthesis interrupted during shutdown: {e}")
                if attempt == 0 and not self._is_stopped:
                    await asyncio.sleep(0.3)
                    continue
                return None
        return None

    async def synthesize(
        self,
        text: str,
        send_audio_callback: Callable,
        speed: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> bool:
        """
        Synthesize text to speech with sentence-level splitting and pre-fetching.
        emotion: optional emotion name (e.g. 'friendly', 'calm') — looked up via
                 emotion_config.get_emotion_params() and applied uniformly to all
                 segments of this response.
        """
        if not self._is_initialized:
            logger.error("[TTS] Service not initialized")
            return False
        
        # Reset stop flag
        self._is_stopped = False
        
        # Determine speed
        if speed is not None:
            self.speed = float(speed)

        # Preprocess text (pronunciation fixes)
        text = text.replace("%", " percent")
        text = text.replace("₹", " rupees ")
        text = text.replace("Rs.", " rupees ")
        text = text.replace("sqft", "square feet")
        text = text.replace("sq.ft", "square feet")
        
        # Remove special chars that might confuse TTS
        text = text.replace("*", " ")
        
        # Split text into sentences/segments for pipelining
        # 1. Split by standard sentence terminators
        sentences = []
        
        # 1. Split by standard sentence terminators
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # 1b. Enhanced Splitting: Handle long sentences with commas
        # If a "sentence" is actually a long list (e.g., "Feature A, Feature B, ..."), split it!
        refined_sentences = []
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
                
            # If sentence is long, split by comma, dash, or conjunctions
            if len(s) > 50:
                # 1. Try splitting by comma
                if "," in s:
                    temp_s = s.replace(", ", ",| ")
                # 2. Try splitting by dash
                elif " - " in s:
                    temp_s = s.replace(" - ", " - | ")
                # 3. Try splitting by major conjunctions (and, but, with, which)
                elif re.search(r'\s(and|but|with|which)\s', s, re.I):
                    temp_s = re.sub(r'\s(and|but|with|which)\s', r' \1| ', s, flags=re.I)
                else:
                    temp_s = s
                
                if "|" in temp_s:
                    subs = temp_s.split("| ")
                    for sub in subs:
                        refined_sentences.append(sub.strip())
                else:
                    refined_sentences.append(s)
            else:
                refined_sentences.append(s)
        
        raw_sentences = refined_sentences

        # 2. Optimized Grouping for Smoothness
        # Strategy:
        # - FIRST chunk: Keep it small (immediate playback)
        # - Subsequent chunks: Larger (efficient buffering)
        
        sentences = []
        current_group = ""
        
        target_min_size = 30
        target_max_size = 80
        
        total_len = len(raw_sentences)
        
        for i, s in enumerate(raw_sentences):
            s = s.strip()
            if not s:
                continue
            
            # SPECIAL CASE: First sentence? Flush immediately to start audio ASAP
            if len(sentences) == 0 and len(current_group) == 0:
                sentences.append(s)
                continue

            # Add to current group
            current_group += s + " "
            
            # Check if we should flush
            should_flush = False
            
            if len(current_group) >= target_max_size:
                should_flush = True
            elif len(current_group) >= target_min_size:
                # Look ahead
                if i + 1 < total_len:
                    next_s = raw_sentences[i+1].strip()
                    if len(current_group) + len(next_s) > target_max_size:
                        should_flush = True
                else:
                    should_flush = True
            elif i == total_len - 1:
                should_flush = True
                
            if should_flush:
                sentences.append(current_group.strip())
                current_group = ""
                
            if current_group:
                sentences.append(current_group.strip())
                current_group = ""
                
        # 3. Handle extremely long segments (like lists) that exceed grouping
        final_segments = []
        for s in sentences:
            if len(s) > 200 and "," in s:
                # Split big lists by comma if needed
                subs = s.split(",")
                current_chunk = ""
                for sub in subs:
                    sub = sub.strip()
                    if len(current_chunk) + len(sub) < 100:
                        current_chunk += sub + ", "
                    else:
                        if current_chunk:
                            final_segments.append(current_chunk.strip(", "))
                        current_chunk = sub + ", "
                if current_chunk:
                    final_segments.append(current_chunk.strip(", "))
            else:
                final_segments.append(s)

        logger.info(f"[TTS] Processing {len(final_segments)} grouped segments for natural flow")

        # Resolve emotion TTS parameters once for the whole response
        emotion_params = emotion_config.get_emotion_params(emotion)
        if emotion:
            logger.info(f"[TTS] Emotion='{emotion}' -> pace={emotion_params['pace']} pitch={emotion_params['pitch']} loudness={emotion_params['loudness']}")

        # Queue for audio segments (full sentence audio)
        # Size limit prevents infinite buffering if network is super fast
        audio_queue = asyncio.Queue(maxsize=3)

        # PRODUCER: Parallel Fetching
        async def producer():
            # Launch requests one at a time - check stop before each
            tasks = []
            for i, segment in enumerate(final_segments):
                if self._is_stopped:
                    logger.info(f"[TTS] Producer stopped before segment {i+1}")
                    break
                logger.info(f"[TTS] Launching request {i+1}/{len(final_segments)}: '{segment[:30]}...'")
                tasks.append(asyncio.create_task(self._synthesize_chunk(segment, override_params=emotion_params)))

            # Await them in order (to preserve playback sequence)
            for i, task in enumerate(tasks):
                if self._is_stopped:
                    # Cancel all remaining tasks immediately
                    task.cancel()
                    for remaining in tasks[i+1:]:
                        remaining.cancel()
                    logger.info(f"[TTS] Cancelled remaining {len(tasks)-i} segments")
                    break

                try:
                    audio_data = await task
                    if audio_data:
                        logger.info(f"[TTS] Segment {i+1} ready ({len(audio_data)} bytes)")
                        await audio_queue.put(audio_data)
                    else:
                        logger.warning(f"[TTS] Failed to synthesize segment {i+1}")
                except asyncio.CancelledError:
                    logger.info(f"[TTS] Segment {i+1} cancelled")
                    break
                except Exception as e:
                    logger.error(f"[TTS] Error awaiting segment {i+1}: {e}")

            await audio_queue.put(None)
            logger.info("[TTS] Producer finished")

        producer_task = asyncio.create_task(producer())
        
        # CONSUMER: Plays audio from queue
        total_audio_len = 0
        sent_first_chunk = False
        chunk_size = 512
        
        try:
            while not self._is_stopped:
                # Wait for next segment from queue
                # This allows consumer to wait if producer is slow (silence)
                # But if producer is fast, it's already here
                segment_audio = await audio_queue.get()
                
                if segment_audio is None:
                    # End of stream
                    break
                
                total_audio_len += len(segment_audio)
                
                # Stream the segment
                for j in range(0, len(segment_audio), chunk_size):
                    if self._is_stopped:
                        break
                        
                    chunk = segment_audio[j:j + chunk_size]
                    await send_audio_callback(chunk, "playAudio")
                    
                    if not sent_first_chunk:
                        logger.info("[TTS] First audio chunk sent - playback starting")
                        sent_first_chunk = True
                        
                    # Tiny delay for stability
                    await asyncio.sleep(0.001)
                
                audio_queue.task_done()
                
        except Exception as e:
            logger.error(f"[ERROR] Consumer loop error: {e}", exc_info=True)
            self._is_stopped = True
            
        # Ensure producer is cleaned up if we stopped early
        if self._is_stopped:
            producer_task.cancel()
            await send_audio_callback(None, "clearAudio")
            return False
        else:
            await send_audio_callback(None, "finishAudio")
            self._last_spoken_text = text
            logger.info(f"[TTS] Synthesis completed. Total audio: {total_audio_len} bytes")
            return True
    
    async def stop(self):
        """
        Stop the current synthesis operation (interruption).
        """
        logger.info("[TTS] Stopping Sarvam TTS synthesis")
        self._is_stopped = True
        
        # Cancel current task if exists
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
    
    async def close(self):
        """
        Close the TTS service and clean up resources.
        """
        try:
            logger.info("[CLEANUP] Closing Sarvam TTS service")
            
            await self.stop()
            
            if self._session and not self._session.closed:
                await self._session.close()
                logger.info("[CONNECTION] Sarvam session closed")
            
            self._is_initialized = False
            
        except Exception as e:
            logger.error(f"[ERROR] Error closing Sarvam TTS service: {e}", exc_info=True)
    
    def set_speed(self, speed: str):
        """
        Set the speech speed/pace.
        
        Args:
            speed: Speed parameter (float between 0.5 and 2.0)
        """
        try:
            self.speed = float(speed)
            logger.info(f"[TTS] Speed set to: {self.speed}")
        except ValueError:
            logger.warning(f"[TTS] Invalid speed value: {speed}, keeping {self.speed}")
    
    async def get_last_spoken_text(self) -> str:
        """
        Get the last text that was synthesized.
        
        Returns:
            The last spoken text string
        """
        return self._last_spoken_text


# Convenience function for quick initialization
async def create_sarvam_tts_service(
    api_key: str,
    voice_id: str = None,
    language: str = None,
    speed: float = None
) -> SarvamTTSService:
    """
    Create and initialize a Sarvam TTS service instance.
    
    Args:
        api_key: Sarvam AI API key
        voice_id: Voice ID (default: config.SARVAM_VOICE_ID)
        language: Language code (default: config.SARVAM_LANGUAGE)
        speed: Speaking pace (default: config.SARVAM_SPEED)
        
    Returns:
        Initialized SarvamTTSService instance
    """
    service = SarvamTTSService(
        api_key=api_key,
        voice_id=voice_id,
        language=language,
        speed=speed
    )
    await service.initialize()
    
    logger.info("[FACTORY] Created and initialized Sarvam TTS service")
    return service
