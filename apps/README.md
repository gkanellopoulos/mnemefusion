# Chat Demo

A Streamlit chat app for interacting with MnemeFusion's memory system. Each user gets their own `.mfdb` file (persistent brain) and can have multiple conversations. Starting a new conversation clears the chat thread but keeps all memories, so you can test cross-conversation recall.

![Chat Demo](https://img.shields.io/badge/streamlit-chat_demo-FF4B4B?logo=streamlit)

## Features

- Multi-user support with namespace isolation
- Multiple conversations per user with shared memory
- Retrieval visualization (per-result scores for all 5 dimensions)
- Simulated date picker for testing temporal memory
- Optional LLM entity extraction (degrades gracefully without a model)

## Setup

```bash
# Install MnemeFusion (see root README for build instructions)
cd mnemefusion-python
maturin develop --release

# Install demo dependencies
pip install streamlit sentence-transformers openai python-dotenv
```

### API Key

Create an `apps/.env` file with your OpenAI key (used for chat responses):

```
OPENAI_API_KEY=sk-...
```

Without an API key, the demo still works in retrieval-only mode — it shows retrieved memories instead of generating responses.

### Entity Extraction (optional)

For full 5-dimension retrieval, download a GGUF model:

```bash
pip install huggingface-hub
huggingface-cli download microsoft/Phi-4-mini-instruct-gguf Phi-4-mini-instruct-Q4_K_M.gguf --local-dir models/phi-4-mini/
```

The demo auto-detects models in `models/` and runs extraction on CPU. Without a model, only semantic + BM25 search is used (2 of 5 dimensions).

## Running

```bash
streamlit run apps/chat_demo.py
```

## How It Works

1. **You type a message** — the app queries MnemeFusion for relevant memories
2. **Retrieved context** is passed to GPT-4o-mini along with the conversation history
3. **Your message is stored** as a new memory with embedding + entity extraction
4. **Click the info popover** on any response to see retrieval scores per dimension

User data is stored in `apps/chat_data/` (gitignored).

## See Also

- [`examples/minimal.py`](../examples/minimal.py) — minimal no-GPU example
- [Root README](../README.md) — full API reference and quick start
