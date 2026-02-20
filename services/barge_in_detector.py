"""
Barge-in detector using webrtcvad (Google WebRTC VAD) + energy echo gate.
No PyTorch dependency — works on Windows without DLL issues.
"""

import asyncio
import numpy as np
import webrtcvad
import logging
import time
from collections import deque

log = logging.getLogger(__name__)

class BargeInDetector:
    def __init__(self,
                 sample_rate: int = 16000,
                 vad_aggressiveness: int = 2,
                 sustain_ms: int = 150,
                 echo_suppression_db: float = 20.0):
        """
        vad_aggressiveness: 0-3. Higher = more aggressive filtering of non-speech.
                            0 = least aggressive (catches quiet speech, more false positives)
                            3 = most aggressive (misses soft speech, fewer false positives)
                            2 is recommended starting point for voice agents
        sustain_ms: milliseconds of continuous speech before barge-in triggers
        echo_suppression_db: how much louder mic must be vs speaker to pass echo gate
        """
        self.sample_rate = sample_rate
        self.sustain_frames_required = max(1, int(sustain_ms / 30))  # 30ms per frame
        self.echo_suppression_factor = 10 ** (echo_suppression_db / 20)

        # webrtcvad only supports 8000, 16000, 32000, 48000 Hz
        # and frame durations of exactly 10ms, 20ms, or 30ms
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.frame_duration_ms = 30  # 30ms frames
        self.frame_bytes = int(sample_rate * self.frame_duration_ms / 1000) * 2  # 16-bit = 2 bytes

        log.info(f"[BARGE-IN] WebRTC VAD loaded (aggressiveness={vad_aggressiveness}, "
                 f"frame={self.frame_bytes} bytes, sustain={sustain_ms}ms)")

        # State
        self.is_playing = False
        self.playback_rms = 0.0
        self.speech_frame_count = 0
        self.last_speech_time = 0.0
        self.barge_in_callback = None
        self._enabled = True

        # Rolling window of last 5 VAD decisions for smoothing
        self._vad_window = deque(maxlen=5)

        # Partial frame buffer — accumulate bytes until we have a full 30ms frame
        self._frame_buffer = b''

    def set_barge_in_callback(self, callback):
        self.barge_in_callback = callback

    def notify_playback_start(self):
        self.is_playing = True
        self.speech_frame_count = 0
        self._frame_buffer = b''
        log.debug("[BARGE-IN] Playback started — echo gate active")

    def notify_playback_stop(self):
        self.is_playing = False
        self.speech_frame_count = 0
        self.playback_rms = 0.0
        self._frame_buffer = b''
        log.debug("[BARGE-IN] Playback stopped — echo gate inactive")

    def update_playback_energy(self, pcm_chunk: bytes):
        """Feed raw PCM being sent to speaker for echo reference."""
        if not pcm_chunk:
            return
        audio = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0.0
        self.playback_rms = 0.7 * self.playback_rms + 0.3 * rms

    async def process_mic_frame(self, pcm_chunk: bytes) -> bool:
        """
        Process mic audio. Accumulates into 30ms frames internally.
        Returns True if barge-in was triggered.
        """
        if not self._enabled:
            return False

        # Accumulate into frame buffer
        self._frame_buffer += pcm_chunk

        # Process all complete 30ms frames in the buffer
        triggered = False
        while len(self._frame_buffer) >= self.frame_bytes:
            frame = self._frame_buffer[:self.frame_bytes]
            self._frame_buffer = self._frame_buffer[self.frame_bytes:]
            result = await self._process_single_frame(frame)
            if result:
                triggered = True

        return triggered

    async def _process_single_frame(self, frame: bytes) -> bool:
        """Process exactly one 30ms frame."""
        audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        mic_rms = np.sqrt(np.mean(audio_np ** 2)) if len(audio_np) > 0 else 0.0

        # --- LAYER 1: ECHO GATE ---
        if self.is_playing and self.playback_rms > 80:
            echo_ratio = mic_rms / (self.playback_rms + 1e-6)
            if echo_ratio < 0.35:
                # Mic energy is mostly playback echo — discard
                self.speech_frame_count = 0
                log.debug(f"[VAD] Echo gate blocked: ratio={echo_ratio:.2f}")
                return False

        # --- LAYER 2: WEBRTC VAD ---
        try:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            log.warning(f"[VAD] WebRTC VAD error: {e}")
            return False

        # Smooth over last 5 frames — majority vote
        self._vad_window.append(1 if is_speech else 0)
        speech_votes = sum(self._vad_window)
        smoothed_speech = speech_votes >= 3  # at least 3 of last 5 frames are speech

        log.info(f"[VAD] is_speech={is_speech} votes={speech_votes}/5 "
                  f"mic_rms={mic_rms:.0f} play_rms={self.playback_rms:.0f}")

        # --- LAYER 3: SUSTAIN TIMER ---
        if smoothed_speech:
            self.speech_frame_count += 1
            self.last_speech_time = time.time()
        else:
            if time.time() - self.last_speech_time > 0.1:
                self.speech_frame_count = 0

        # Trigger only when playing AND speech sustained long enough
        if self.speech_frame_count >= self.sustain_frames_required and self.is_playing:
            # [NEW] Energy Gate: Reject trigger if mic energy is too low (noise floor)
            if mic_rms < 300:
                log.debug(f"[VAD] Suppressed low-energy trigger: mic_rms={mic_rms:.0f} < 300")
                self.speech_frame_count = 0
                self._vad_window.clear()
                return False

            log.info(f"[BARGE-IN] Confirmed! votes={speech_votes}/5 "
                     f"frames={self.speech_frame_count} mic_rms={mic_rms:.0f}")
            self.speech_frame_count = 0
            self._vad_window.clear()      # [NEW] Reset window to prevent false re-trigger
            self._frame_buffer = b''      # [NEW] Discard accumulation
            
            if self.barge_in_callback:
                await self.barge_in_callback()
            return True

        return False

    def disable(self):
        self._enabled = False
        self._frame_buffer = b''
        self._vad_window.clear()      # [NEW]
        self.speech_frame_count = 0   # [NEW]

    def enable(self):
        self._enabled = True
        self._frame_buffer = b''
