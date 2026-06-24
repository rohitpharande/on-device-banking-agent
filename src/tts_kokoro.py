"""Text-to-Speech engine using the Kokoro-82M local model.

Generates speech from text and streams it directly to the system speaker
via PyAudio.  Runs entirely on-device — no network calls required.
"""

import numpy as np
import pyaudio
import torch
from kokoro import KPipeline


class TTSEngine:
    """Local TTS engine backed by Kokoro-82M."""

    def __init__(
        self,
        lang_code: str = "a",
        default_voice: str = "af_heart",
        speed: float = 1.0,
    ):
        print("[TTS] Initializing Local Kokoro-82M Pipeline...")
        self.pipeline = KPipeline(lang_code=lang_code)
        self.default_voice = default_voice
        self.speed = speed
        self.p = pyaudio.PyAudio()
        self.SAMPLE_RATE = 24000

    def speak(self, text: str, voice: str = None) -> None:
        """Synthesise *text* and play it through the default audio output.

        Args:
            text:  The string to speak.
            voice: Override voice ID (defaults to ``self.default_voice``).
        """
        if not text:
            return

        target_voice = voice or self.default_voice

        stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.SAMPLE_RATE,
            output=True,
        )

        generator = self.pipeline(text, voice=target_voice, speed=self.speed)
        for _graphemes, _phonemes, audio in generator:
            if audio is not None:
                # Convert PyTorch Tensor to NumPy array if needed
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()

                if len(audio) > 0:
                    stream.write(audio.astype(np.float32).tobytes())

        stream.stop_stream()
        stream.close()
