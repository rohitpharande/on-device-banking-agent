"""Automatic Speech Recognition engine using Faster Whisper.

Captures audio from the microphone, detects speech boundaries via an
energy-based Voice Activity Detector (VAD), and transcribes the result
using the Faster Whisper model running on GPU.

Key optimisations for low latency:
    - Language hardcoded to English (skips ~200 ms auto-detect penalty)
    - VAD filter enabled to strip silence before model inference
    - Echo flush: drains residual TTS audio from the OS mic buffer
"""

import time

import numpy as np
import pyaudio
from faster_whisper import WhisperModel


class ASREngine:
    """Microphone listener + Faster Whisper transcriber."""

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        print(f"[ASR] Initializing Faster-Whisper ({model_size}) on {device}...")
        self.asr = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.p = pyaudio.PyAudio()

        # Audio capture settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024

        # Voice Activity Detection thresholds
        self.SILENCE_THRESHOLD = 800   # RMS amplitude — adjust based on mic noise floor
        self.SILENCE_DURATION = 1.2    # Seconds of consecutive silence to stop recording

    def listen_until_silence(self) -> np.ndarray:
        """Record from the microphone until the user stops speaking.

        Returns a float32 numpy array normalised to [-1.0, 1.0].
        """
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        # --- Echo flush: discard residual TTS audio in the mic buffer ---
        time.sleep(0.2)
        if stream.get_read_available() > 0:
            stream.read(stream.get_read_available(), exception_on_overflow=False)

        print("\nListening... Speak now.")
        frames: list[bytes] = []
        silent_chunks = 0
        has_spoken = False
        max_silent_chunks = int((self.RATE / self.CHUNK) * self.SILENCE_DURATION)

        while True:
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)

            audio_chunk = np.frombuffer(data, dtype=np.int16)

            # Cast to float32 before squaring to prevent 16-bit integer overflow
            if len(audio_chunk) > 0:
                float_chunk = audio_chunk.astype(np.float32)
                rms = np.sqrt(np.mean(float_chunk**2))
            else:
                rms = 0

            if rms > self.SILENCE_THRESHOLD:
                has_spoken = True
                silent_chunks = 0
            elif has_spoken:
                silent_chunks += 1

            if has_spoken and silent_chunks > max_silent_chunks:
                break

        stream.stop_stream()
        stream.close()

        audio_data = b"".join(frames)
        return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe(self, audio_array: np.ndarray) -> str:
        """Transcribe a float32 audio array to text.

        Uses greedy-ish decoding (beam_size=3) with VAD filtering for speed.
        """
        segments, _ = self.asr.transcribe(
            audio_array,
            language="en",
            beam_size=3,
            vad_filter=True,
        )
        return " ".join([seg.text for seg in segments]).strip()
