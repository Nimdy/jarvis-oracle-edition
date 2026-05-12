# THIRD_PARTY_LICENSES.md

This file documents all third-party components used by **JARVIS Oracle Edition**, along with their licenses and copyright information.

JARVIS Oracle Edition complies with all third-party licenses. The core architecture, epistemic immune system, self-designing neural hemispheres, governed self-modification pipeline, HRR/VSA spatial canvas, consciousness kernel, Synthetic Soul theory paper, and all original code remain the intellectual property of David Eierdam.

## Core Dependencies

| Component                  | License              | Usage                                      | Repository |
|---------------------------|----------------------|--------------------------------------------|----------|
| **Ollama**                | MIT                  | Primary LLM inference backend              | [ollama/ollama](https://github.com/ollama/ollama) |
| **llama.cpp**             | MIT                  | High-performance inference engine          | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| **faster-whisper**        | MIT                  | Speech-to-Text (Whisper)                   | Community fork of OpenAI Whisper |
| **Kokoro TTS**            | Apache 2.0           | Local Text-to-Speech with emotion support  | — |
| **PyTorch**               | BSD-3-Clause         | Machine learning framework                 | pytorch/pytorch |
| **Qwen3 / Qwen2.5 series**| Apache 2.0           | Base LLM and vision models (via Ollama)    | Alibaba |
| **Hailo NPU Drivers**     | Proprietary / Permissive | Hardware acceleration on Raspberry Pi 5 | Hailo |

### Additional Python Libraries
- NumPy, SciPy, FastAPI, WebSocket libraries, etc. — primarily **BSD / MIT**.
- Full license texts are stored in the `licenses/` directory.

## Model Licenses
- GGUF models used via Ollama follow the license of their original publishers (e.g., Meta Llama Community License, Mistral, Qwen Apache 2.0, etc.).
- Always check the model card on Hugging Face or Ollama for specific commercial restrictions.

## Attribution & Compliance
- All required copyright notices and license files are preserved in the `licenses/` directory.
- When distributing JARVIS Oracle Edition (source or binary), this file and the `licenses/` directory **must** be included.
- The project’s dual licensing (AGPLv3 + Commercial) is fully compatible with the permissive licenses of all dependencies.

## How to Update This File
When adding a new dependency:
1. Place the full license text in `licenses/<project>/LICENSE`.
2. Update the table above.
3. Verify compatibility with both AGPLv3 and the Commercial license.
4. Update the “Last updated” date.

**Last updated:** April 2026

For any questions about licensing or commercial use, contact: mrzerobandwidth@gmail.com