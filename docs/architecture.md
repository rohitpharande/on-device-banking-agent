# Architecture Overview

This document describes the internal architecture of the On-Device Banking Agent.

## System Pipeline

<p align="center">
  <img src="docs/architecture.png" alt="System Architecture — Microphone → ASR → SLM → Orchestrator → TTS → Speaker" width="850">
</p>


## Component Details

### 1. ASR Engine (`src/asr_whisper.py`)

- **Model**: Faster Whisper `base.en` on CUDA with FP16
- **Input**: Raw PCM audio from the microphone (16 kHz, mono, 16-bit)
- **Output**: Transcribed text string
- **Key features**:
  - Energy-based Voice Activity Detection (RMS amplitude threshold)
  - Echo flush mechanism to discard residual TTS audio from the mic buffer
  - Language hardcoded to English to skip auto-detect latency (~200 ms saved)
  - VAD filter enabled to strip digital silence before inference

### 2. Orchestrator (`src/orchestrator_qwen.py`)

The orchestrator is the "brain" of the system, composed of two classes:

#### SLMClient
A stateless wrapper around any OpenAI-compatible inference endpoint. Sends the
full conversation history to the fine-tuned Qwen3-0.6B and parses the response
into a structured function call.

- Supports both native `tool_calls` responses and raw JSON-in-content fallback
- Communicates with `llama-server` (llama.cpp) via the OpenAI Chat Completions API

#### TextOrchestrator
A deterministic state machine that manages the dialogue flow:

```
User speaks
    │
    ▼
SLM extracts intent + slots
    │
    ▼
┌─────────────────────────┐
│  Is there an active      │──No──▶ Set new intent, clear slots
│  transaction in progress?│
└────────────┬────────────┘
             │ Yes
             ▼
   Ignore conversational intents
   (greeting, thank_you, etc.)
   Merge new slots into session
             │
             ▼
   ┌─────────────────────┐
   │ All required slots   │──No──▶ Ask for missing slots
   │ filled?              │
   └──────────┬──────────┘
              │ Yes
              ▼
     Execute backend action
     Return success response
     Clear state
```

### 3. TTS Engine (`src/tts_kokoro.py`)

- **Model**: Kokoro-82M (fully local, no network calls)
- **Output**: 24 kHz float32 audio streamed to the system speaker
- **Voice**: Configurable (default: `hf_beta`)
- **Key features**:
  - Handles both PyTorch Tensor and NumPy array outputs from the pipeline
  - Streaming playback — audio chunks are written as they are generated

### 4. Configuration (`src/config.py`)

All constants extracted into a single module:
- **TOOLS**: 14 banking function definitions (OpenAI function-calling schema)
- **FUNCTION_REQUIRED_ARGS**: Maps each function to its required arguments
- **INDIVIDUAL_SLOT_PROMPTS**: Human-readable prompts for each missing slot
- **SUCCESS_TEMPLATES**: Response templates rendered after successful execution
- **SYSTEM_PROMPT**: The system message (must match the fine-tuning prompt)

## Supported Banking Intents

| Intent | Required Slots | Description |
|--------|---------------|-------------|
| `check_balance` | `account_type` | Check account balance |
| `get_statement` | `account_type`, `period` | Request account statement |
| `transfer_money` | `amount`, `from_account`, `to_account` | Transfer between accounts |
| `pay_bill` | `payee`, `amount` | Pay a bill |
| `cancel_card` | `card_type`, `card_last_four` | Cancel a card |
| `replace_card` | `card_type`, `card_last_four` | Request replacement card |
| `activate_card` | `card_last_four` | Activate a new card |
| `reset_pin` | `card_type`, `card_last_four` | Reset card PIN |
| `report_fraud` | `card_type`, `card_last_four`, `transaction_amount` | Report fraud |
| `speak_to_human` | *(none)* | Connect to human agent |
| `greeting` | *(none)* | Greeting detected |
| `goodbye` | *(none)* | Farewell detected |
| `thank_you` | *(none)* | Gratitude detected |
| `intent_unclear` | *(none)* | Fallback |

## Inference Stack

The fine-tuned model is served via **llama.cpp** (`llama-server`):

```bash
llama-server \
  -m path/to/qwen3_06b_voice_banking_f16.gguf \
  --port 7002 \
  -ngl 99 \
  -c 2048 \
  --predict 64 \
  --temp 0.0 \
  -rea off \
  --no-context-shift \
  --min-p 0.05
```

The assistant communicates with this server through the standard OpenAI Chat
Completions API (`/v1/chat/completions`), making the architecture compatible
with any OpenAI-compatible backend (vLLM, Ollama, etc.).
