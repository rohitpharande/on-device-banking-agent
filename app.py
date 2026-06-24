"""BankCo Voice Assistant — unified entry point.

Supports two operating modes:

    voice  (default)  Full pipeline: Microphone → ASR → SLM → TTS → Speaker
    text              Text-only mode for testing without audio hardware

Usage examples:
    # Voice mode (requires mic + speaker)
    python app.py --model qwen3_06b_voice_banking_f16_v2.gguf --port 7002

    # Text mode (terminal only)
    python app.py --mode text --model qwen3_06b_voice_banking_f16_v2.gguf --port 7002
"""

import argparse
import asyncio
import re

from src.orchestrator_qwen import SLMClient, TextOrchestrator
from src.config import SUCCESS_TEMPLATES


# ---------------------------------------------------------------------------
# Voice mode — full ASR → SLM → TTS loop
# ---------------------------------------------------------------------------
async def voice_loop(orchestrator: TextOrchestrator) -> None:
    """Run the voice assistant with microphone input and speaker output."""
    from src.asr_whisper import ASREngine
    from src.tts_kokoro import TTSEngine

    asr = ASREngine(model_size="base.en")
    tts = TTSEngine(lang_code="a", default_voice="hf_beta", speed=1.0)

    last_bot_response = SUCCESS_TEMPLATES["greeting"]
    print(f"Bot: {last_bot_response}")
    tts.speak(last_bot_response)

    while True:
        try:
            audio_array = asr.listen_until_silence()
            transcript = asr.transcribe(audio_array)
            if not transcript:
                continue

            # --- Software echo cancellation ---
            # If the mic picked up the bot's own TTS output, ignore it.
            def clean_text(t: str) -> str:
                return re.sub(r"[^a-zA-Z0-9]", "", t).lower()

            if clean_text(transcript) in clean_text(last_bot_response):
                continue

            print(f"You (Transcribed): {transcript}")

            if transcript.lower() in ("exit", "quit", "goodbye"):
                print("Bot: Goodbye!")
                tts.speak("Thank you for using BankCo. Goodbye!")
                break

            bot_response = orchestrator.process_utterance(transcript)

            if bot_response is None:
                print("Bot: Goodbye!")
                tts.speak(SUCCESS_TEMPLATES["goodbye"])
                break

            print(f"Bot: {bot_response}")
            tts.speak(bot_response)
            last_bot_response = bot_response

        except (KeyboardInterrupt, SystemExit):
            break
        except Exception as e:
            print(f"[Error] Runtime Exception: {e}")
            break


# ---------------------------------------------------------------------------
# Text mode — terminal-only loop (no audio hardware required)
# ---------------------------------------------------------------------------
def text_loop(orchestrator: TextOrchestrator) -> None:
    """Run the assistant in text-only mode for development / testing."""
    print("BankCo Assistant — Text Mode (type 'quit' or 'exit' to stop)\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            response = orchestrator.process_utterance(user_input)
            if response is None:
                print("Bot: Goodbye! Thanks for calling BankCo.")
                break
            print(f"Bot: {response}")
    except (KeyboardInterrupt, EOFError):
        print("\nBot: Goodbye! Thanks for calling BankCo.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BankCo Voice Assistant powered by fine-tuned Qwen3-0.6B"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["voice", "text"],
        default="voice",
        help="Run in 'voice' mode (mic+speaker) or 'text' mode (terminal only)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model",
        help="Model name / filename served by the inference server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7001,
        help="Port of the llama-server instance (default: 7001)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for the SLM server (default: EMPTY)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Full base URL for the inference server (overrides --port)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw SLM output each turn for debugging",
    )
    args = parser.parse_args()

    base_url = args.base_url or f"http://127.0.0.1:{args.port}/v1"
    slm_client = SLMClient(
        model_name=args.model, base_url=base_url, api_key=args.api_key
    )
    orchestrator = TextOrchestrator(slm_client, debug=args.debug)

    if args.mode == "text":
        text_loop(orchestrator)
    else:
        asyncio.run(voice_loop(orchestrator))


if __name__ == "__main__":
    main()
