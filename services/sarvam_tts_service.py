"""
Sarvam AI Text-to-Speech Service
Implements BaseTTSService for Sarvam AI's text-to-speech API using WebSockets
"""
import logging
import asyncio
import base64
import audioop
import time
from typing import Callable, Optional, Dict, Any

from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse
from services.tts_base import BaseTTSService
import config
import emotion_config

logger = logging.getLogger(__name__)

class SarvamTTSService(BaseTTSService):
    """
    Sarvam AI TTS service with WebSocket streaming API
    Supports Hindi and English voices
    """
    
    def __init__(
        self, 
        api_key: str, 
        voice_id: str = None, 
        language: str = "en-IN",
        speed: float = 1.0,
        model: str = "bulbul:v2"
    ):
        """
        Initialize Sarvam TTS service
        
        Args:
            api_key: Sarvam AI API key
            voice_id: Voice ID ("meera", "arvind", etc.)
            language: Language code ("hi-IN" for Hindi, "en-IN" for English)
            speed: Speaking pace (0.5 to 2.0, default 1.0)
            model: Model to use (must be bulbul:v2 or bulbul:v3)
        """
        self.api_key = api_key
        self.speaker = voice_id or config.SARVAM_VOICE_ID or "priya"
        self.language = language or config.SARVAM_LANGUAGE or "en-IN"
        self.speed = speed or config.SARVAM_SPEED or 1.0
        self.model = model
        self.sample_rate = 16000
        
        self.client = AsyncSarvamAI(api_subscription_key=api_key)
        self._is_initialized = False
        self._is_stopped = False
        self._last_spoken_text = ""
        self.barge_in_event = None
        
        logger.info(f"[TTS] Sarvam WebSocket TTS service initialized (voice={self.speaker}, lang={self.language})")

    async def initialize(self) -> bool:
        """
        Initialize TTS service and prepare for synthesis.
        """
        self._is_initialized = True
        logger.info("[TTS] Sarvam TTS service initialized")
        return True

    def set_barge_in_event(self, event: asyncio.Event):
        """Set the barge-in event for checking interruptions."""
        self.barge_in_event = event

    async def synthesize_streaming(
        self,
        text: str,
        pcm_queue: asyncio.Queue,
        barge_in_event: asyncio.Event
    ):
        """
        Stream text to Sarvam WebSocket TTS.
        Audio chunks are pushed to pcm_queue as they arrive.
        Stops early if barge_in_event is set.
        """
        logger.info(f"[TTS] Streaming synthesis: '{text[:60]}...'")
        self._is_stopped = False
        self._last_spoken_text = text
        try:
            async with self.client.text_to_speech_streaming.connect(
                model=self.model,
                send_completion_event=True
            ) as ws:
                # Configure voice — sent once per connection
                await ws.configure(
                    target_language_code=self.language,
                    speaker=self.speaker,
                    pace=self.speed,
                    min_buffer_size=30,       # lower = faster first chunk
                    max_chunk_length=200,
                    output_audio_codec="wav"  # try wav since pcm was returning zero chunks
                )

                await ws.convert(text)
                await ws.flush()

                chunk_count = 0
                async for message in ws:
                    # Log event type without printing full base64 audio payload
                    logger.debug(f"[TTS] WebSocket message type: {type(message).__name__}")

                    if barge_in_event.is_set() or self._is_stopped:
                        logger.info("[TTS] Barge-in or stop — aborting stream")
                        break

                    if isinstance(message, AudioOutput):
                        chunk_count += 1
                        pcm_bytes = base64.b64decode(message.data.audio)

                        # Strip WAV header if present
                        if pcm_bytes[:4] == b'RIFF' and pcm_bytes[8:12] == b'WAVE':
                            pcm_bytes = pcm_bytes[44:]

                        # Since output_audio_sample_rate is unsupported, resample 22050 to 16000
                        pcm_16k, _ = audioop.ratecv(pcm_bytes, 2, 1, 22050, self.sample_rate, None)

                        await pcm_queue.put(pcm_16k)

                        if chunk_count == 1:
                            logger.info(f"[TTS] First audio chunk ready — playback can start")

                    elif isinstance(message, EventResponse):
                        if message.data.event_type == "final":
                            logger.info(f"[TTS] Stream complete — {chunk_count} chunks")
                            break
                    else:
                        logger.warning(f"[TTS] Unknown message: {message}")

        except Exception as e:
            logger.error(f"[TTS] Streaming error: {e}")
        finally:
            await pcm_queue.put(None)  # sentinel

    # ---------- BaseTTSService fallback compatibility methods ----------
    
    async def synthesize(
        self,
        text: str,
        send_audio_callback: Callable,
        speed: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> bool:
        """
        Fallback method for BaseTTSService compatibility.
        Wraps synthesize_streaming using an inline queue.
        """
        if speed is not None:
             self.speed = float(speed)
             
        temp_queue = asyncio.Queue()
        dummy_event = self.barge_in_event if self.barge_in_event else asyncio.Event()
        
        synth_task = asyncio.create_task(
             self.synthesize_streaming(text, temp_queue, dummy_event)
        )
        
        while True:
            chunk = await temp_queue.get()
            if chunk is None or dummy_event.is_set() or self._is_stopped:
                break
            await send_audio_callback(chunk, "playAudio")
            
        await synth_task
        return True

    async def synthesize_one(self, text: str) -> bytes | None:
        """Synthesize a single sentence and return PCM bytes. (Deprecated by streaming API)"""
        logger.warning("[TTS] synthesize_one called but we are on WebSocket streaming. Ignoring.")
        return b""

    async def prewarm_tts(self):
        """No-op for WebSockets, connection is established per-request in this snippet"""
        pass

    async def stop(self):
        """Stop the current synthesis operation (interruption)."""
        logger.info("[TTS] Stopping Sarvam TTS synthesis")
        self._is_stopped = True

    async def close(self):
        """Close the TTS service and clean up resources."""
        logger.info("[CLEANUP] Sarvam TTS WebSocket closed")
        await self.stop()
        self._is_initialized = False

    def set_speed(self, speed: str):
        """Set the speech speed/pace."""
        try:
            self.speed = float(speed)
            logger.info(f"[TTS] Speed set to: {self.speed}")
        except ValueError:
            pass

    async def get_last_spoken_text(self) -> str:
        """Get the last text that was synthesized."""
        return self._last_spoken_text


# Convenience function for quick initialization
async def create_sarvam_tts_service(
    api_key: str,
    voice_id: str = None,
    language: str = None,
    speed: float = None
) -> SarvamTTSService:
    service = SarvamTTSService(
        api_key=api_key,
        voice_id=voice_id,
        language=language,
        speed=speed
    )
    await service.initialize()
    logger.info("[FACTORY] Created and initialized Sarvam TTS WebSocket service")
    return service
