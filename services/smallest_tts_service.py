"""
Smallest.ai Lightning v2 WebSocket TTS Service
Implements BaseTTSService for ultra-low latency streaming (sub-100ms).
"""
import logging
import asyncio
import json
import websockets
import base64
import time
from typing import Optional, Callable
from services.tts_base import BaseTTSService
import config

logger = logging.getLogger(__name__)

# Connection Constants (Lightning v2)
SMALLEST_WS_URL = "wss://waves-api.smallest.ai/api/v1/lightning-v2/get_speech/stream?timeout=60"
SMALLEST_VOICE_ID = "emily"
SMALLEST_SAMPLE_RATE = 24000
SMALLEST_SPEED = 1.1
SMALLEST_LANGUAGE = "en"
SMALLEST_DEFAULT_MODEL = "lightning-v2"

class SmallestTTSService(BaseTTSService):
    """
    Smallest.ai Lightning v2 TTS service using WebSocket streaming.
    Maintains a persistent connection for the session.
    """
    def __init__(self, api_key: str, voice_id: Optional[str] = None, **kwargs):
        self.api_key = api_key or config.SMALLEST_API_KEY
        if not self.api_key:
            raise ValueError("[SMALLEST] API Key missing. Set SMALLEST_API_KEY in .env")
            
        self.voice_id = voice_id or SMALLEST_VOICE_ID
        
        # Model Mapping to valid API paths
        requested_model = kwargs.get('model', SMALLEST_DEFAULT_MODEL)
        model_map = {
            "waves_lightning_large": "lightning-v2", # Default to v2 for streaming
            "lightning_large": "lightning-v2",
            "lightning-large": "lightning-v2",
            "lightning-v2": "lightning-v2",
            "lightning": "lightning"
        }
        self.model = model_map.get(requested_model, requested_model)
        
        # Build URL dynamically
        self.ws_url = f"wss://waves-api.smallest.ai/api/v1/{self.model}/get_speech/stream?timeout=60"
        
        self.websocket = None
        self._connection_active = False
        self._last_send_time = 0
        self._first_chunk_received = False
        self.total_bytes = 0
        self.audio_chunks = []
        self._receiver_ready = asyncio.Event()
        self.speed = str(SMALLEST_SPEED)
        self.last_text = ""
        self._receiver_ready = asyncio.Event()
        self.turn_count = 0

    def _reset_turn(self, text_len: int):
        """Reset turn-level buffers and timers."""
        self.turn_count += 1
        self._last_send_time = time.time()
        self._first_chunk_received = False
        self.total_bytes = 0
        self.audio_chunks = []
        logger.info(f"[SMALLEST] Turn {self.turn_count} reset — bytes: 0, timer: fresh, receiver: starting")

    async def connect(self):
        """
        Open the WebSocket connection to Smallest.ai.
        Should be called once at session start.
        """
        try:
            logger.info(f"[SMALLEST] Connecting to {self.ws_url}...")
            connect_start = time.time()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            self.websocket = await websockets.connect(
                self.ws_url, 
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10
            )
            self._connection_active = True
            
            elapsed = (time.time() - connect_start) * 1000
            logger.info(f"[SMALLEST] WebSocket connected in {elapsed:.1f}ms")

        except Exception as e:
            logger.error(f"[SMALLEST] Connection failed: {e}")
            self._connection_active = False
            # We don't raise here to allow retry logic elsewhere if needed, 
            # but ideally caller handles initialization failure.
            raise

    async def send_text(self, text: str):
        """Send text to be synthesized."""
        if not self.websocket or not self._connection_active:
            logger.warning("[SMALLEST] WebSocket not connected — attempting to connect...")
            await self.connect()
            
        try:
            # Wait for receiver thread to be ready
            try:
                await asyncio.wait_for(self._receiver_ready.wait(), timeout=2.0)
                logger.info("[SMALLEST] Receiver thread ready")
            except asyncio.TimeoutError:
                logger.warning("[SMALLEST] Receiver ready timeout — proceeding anyway")

            # Turn reset
            self._reset_turn(len(text))
            self.last_text = text

            payload = {
                "voice_id": self.voice_id,
                "text": text,
                "language": SMALLEST_LANGUAGE,
                "sample_rate": SMALLEST_SAMPLE_RATE,
                "speed": float(self.speed),
                "consistency": 0.5,
                "similarity": 0.0,
                "enhancement": 1,
                "add_wav_header": False,
                "flush": True,
                "continue": False
            }
            
            await self.websocket.send(json.dumps(payload))
            logger.info(f"[SMALLEST] Text sent ({len(text)} chars) — TTFB timer started")
            
        except Exception as e:
            logger.error(f"[SMALLEST] Error sending text: {e}")
            self._connection_active = False
            # Try once to reconnect
            try:
                logger.info("[SMALLEST] Attempting to reconnect and resend...")
                await self.connect()
                # Re-reset turn on reconnect success
                self._reset_turn(len(text))
                await self.websocket.send(json.dumps(payload))
                logger.info(f"[SMALLEST] Text sent ({len(text)} chars) — TTFB timer started (after reconnect)")
            except Exception as re:
                logger.error(f"[SMALLEST] Reconnect failed: {re}")

    async def receive_audio(self, callback: Callable[[bytes], None]):
        """
        Listen for usage/audio messages from WebSocket.
        Runs until 'done' message or completion signal is received.
        Does NOT close the connection.
        """
        if not self.websocket:
            return

        # Signal that we are ready to receive
        self._receiver_ready.set()
        # logger.info("[SMALLEST] Receiver thread: ready and listening")

        try:
            async for message in self.websocket:
                # logger.debug(f"[SMALLEST] Message received: {str(message)[:100]}...")
                try:
                    # Message is expected to be JSON string
                    if isinstance(message, bytes):
                         audio_chunk = message
                         if not self._first_chunk_received:
                             self._first_chunk_received = True
                             ttfb = (time.time() - self._last_send_time) * 1000
                             logger.info(f"[SMALLEST] First audio chunk received — TTFB: {ttfb:.1f}ms")
                         
                         self.total_bytes += len(audio_chunk)
                         self.audio_chunks.append(audio_chunk)
                         if callback:
                             if asyncio.iscoroutinefunction(callback):
                                 await callback(audio_chunk)
                             else:
                                 callback(audio_chunk)
                         continue

                    # JSON parsing
                    data = json.loads(message)
                    
                    # Check for completion or error
                    status = data.get("status")
                    if status in ("done", "complete") or data.get("done") is True:
                        break
                    
                    if status == "error":
                        error_msg = data.get("message") or "Unknown error"
                        logger.error(f"[SMALLEST] Server error: {error_msg}")
                        break
                        
                    # Extract audio
                    data_field = data.get("data")
                    if isinstance(data_field, dict):
                        b64_audio = data_field.get("audio")
                    else:
                        b64_audio = data_field or data.get("audio")
                    
                    if b64_audio:
                        try:
                            audio_chunk = base64.b64decode(b64_audio)
                            
                            if not self._first_chunk_received:
                                self._first_chunk_received = True
                                ttfb = (time.time() - self._last_send_time) * 1000
                                logger.info(f"[SMALLEST] First audio chunk received — TTFB: {ttfb:.1f}ms")
                            
                            self.total_bytes += len(audio_chunk)
                            self.audio_chunks.append(audio_chunk)
                            if callback:
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(audio_chunk)
                                else:
                                    callback(audio_chunk)
                        except Exception as b64e:
                            logger.error(f"[SMALLEST] Base64 decode error: {b64e}")
                            
                except json.JSONDecodeError:
                    logger.warning(f"[SMALLEST] Received non-JSON message: {message[:50]}...")
                except Exception as e:
                    logger.error(f"[SMALLEST] Message processing error: {e}")
            
            total_time = (time.time() - self._last_send_time) * 1000
            logger.info(f"[SMALLEST] Audio stream complete — total bytes: {self.total_bytes}, total time: {total_time:.1f}ms")
            
        except Exception as e:
             logger.error(f"[SMALLEST] Receive loop error: {e}")
             self._connection_active = False
        finally:
            # Always reset event for next turn
            self._receiver_ready.clear()

    async def keepalive(self):
        """
        Send minimal payload to keep connection open (extend 60s timeout).
        Call start of/after turn.
        """
        if self.websocket and self._connection_active:
            try:
                # Smallest might not have explicit ping, sending a dummy config or ping frame
                # WebSockets protocol ping
                await self.websocket.ping()
                # logger.debug("[SMALLEST] KeepAlive ping sent")
            except Exception as e:
                logger.error(f"[SMALLEST] KeepAlive failed: {e}")
                self._connection_active = False

    async def close(self):
        """Close WebSocket connection."""
        self._connection_active = False
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("[SMALLEST] WebSocket closed")
            except Exception as e:
                logger.debug(f"[SMALLEST] Close error: {e}")
            finally:
                self.websocket = None

    # ─── Abstract Methods ───────────────────────────

    async def initialize(self) -> bool:
        """Initialize resource."""
        # Connect is called explicitly by session start, but we can do it here if needed.
        # But per architecture, we connect in start_session.
        return True

    async def stop(self):
        """Stop current playback."""
        # Persistent connection -> do not close. just maybe stop sending?
        pass

    def set_speed(self, speed: str):
        self.speed = speed

    async def get_last_spoken_text(self) -> str:
        return self.last_text
    
    async def flush(self):
        """Send flush signal to start generation/empty buffer."""
        if self.websocket and self._connection_active:
            try:
                payload = {
                    "voice_id": self.voice_id,
                    "text": "",
                    "flush": True
                }
                await self.websocket.send(json.dumps(payload))
                logger.debug("[SMALLEST] Flush sent")
            except Exception as e:
                logger.error(f"[SMALLEST] Flush failed: {e}")

    async def synthesize(self, text: str, send_audio_callback: Callable, speed: str = None) -> bool:
        """
        Legacy/One-shot wrapper.
        """
        try:
            await self.connect()
            
            # Start receiver task first to avoid deadlock
            receive_task = asyncio.create_task(self.receive_audio(send_audio_callback))
            
            await self.send_text(text)
            await receive_task
            
            await self.close()
            return True
        except Exception as e:
            logger.error(f"[SMALLEST] Synthesize failed: {e}")
            return False
