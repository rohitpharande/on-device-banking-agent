# Setup & Installation Guide

This guide provides detailed instructions to set up the **On-Device Banking Agent** from scratch, resolve hardware/driver dependencies, and run the assistant.

---

## 📋 Prerequisites

Ensure your system meets the following hardware and software requirements:

| Component | Minimum Requirement | Recommended | Notes |
|---|---|---|---|
| **GPU** | NVIDIA GPU with ≥ 6 GB VRAM | NVIDIA GPU with ≥ 8 GB VRAM | Tested on RTX 3070 Ti / RTX 40-series |
| **CUDA** | CUDA 12.1+ | CUDA 12.4 | Must be compatible with PyTorch and llama.cpp |
| **OS** | Linux (Ubuntu 20.04+) or Windows (WSL2) | Ubuntu 22.04 LTS / WSL2 | Native Linux is recommended for audio drivers |
| **Audio** | Working Microphone & Speakers | USB Headset | Avoid external speakers to reduce echo pickup |

---

## 🛠️ Step-by-Step Installation

### Step 1: Install System Dependencies

#### Linux (Ubuntu/Debian)
Install `PortAudio` development headers, which are required to build the `pyaudio` Python package:
```bash
sudo apt update
sudo apt install -y portaudio19-dev python3-dev build-essential
```

#### Windows (WSL2)
1. Install PortAudio on your WSL2 distribution:
   ```bash
   sudo apt update && sudo apt install -y portaudio19-dev
   ```
2. Configure WSL2 audio forwarding (PulseAudio or WSLg audio) so that WSL can access your Windows microphone and speakers.

---

### Step 2: Set Up Python Virtual Environment

We recommend Python 3.10, 3.11, or 3.12:
```bash
# Navigate to the project directory
cd on-device-banking-agent

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

---

### Step 3: Install Python Packages

1. **Install CUDA-enabled PyTorch first** (highly recommended to avoid CPU fallback for Kokoro/Whisper):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
2. **Install remaining runtime dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Setting Up `llama.cpp` for Local SLM Serving

The core language capability is powered by a fine-tuned `Qwen3-0.6B` model served via the `llama.cpp` server.

### Option A: Build `llama.cpp` from Source (Highly Recommended for CUDA optimization)
1. Clone the `llama.cpp` repository:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   ```
2. Build with CUDA support:
   ```bash
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j
   ```
3. The server executable `llama-server` (or `llama-completions` / `llama-cli`) will be built under the `build/bin/` folder. Copy or link it to your PATH:
   ```bash
   sudo cp build/bin/llama-server /usr/local/bin/
   ```

### Option B: Download Pre-built Binary
Go to the [llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases), download the appropriate binary zip for your OS with CUDA support, extract it, and add the binary to your executable path.

---

## 📥 Download or Prepare the Model

Ensure you have your fine-tuned `qwen3_06b_voice_banking_f16.gguf` file placed in the `models/` directory:
```bash
mkdir -p models/
# Copy your GGUF file here
```

---

## 🚀 Running the Assistant

### 1. Start the Inference Server
Run `llama-server` in a separate terminal window:
```bash
llama-server \
  -m models/qwen3_06b_voice_banking_f16.gguf \
  --port 7002 \
  -ngl 99 \
  -c 2048 \
  --predict 64 \
  --temp 0.0 \
  -rea off \
  --no-context-shift \
  --min-p 0.05
```

### 2. Start the Agent

#### Voice Mode (Default)
Runs with full audio support (requires microphone and speaker):
```bash
python app.py --model qwen3_06b_voice_banking_f16.gguf --port 7002
```

#### Text-Only Mode (Terminal)
For development, remote SSH sessions, or systems without audio hardware:
```bash
python app.py --mode text --model qwen3_06b_voice_banking_f16.gguf --port 7002
```

---

## ⚡ Debugging & Verification

To verify that CUDA is working correctly inside your environment:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

If you encounter issues with audio hardware or transcription speed, consult the [Troubleshooting section in the README](../README.md#-troubleshooting).
