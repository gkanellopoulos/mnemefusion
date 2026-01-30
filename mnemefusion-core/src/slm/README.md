# SLM Integration - HuggingFace Authentication Guide

## Overview

The SLM (Small Language Model) integration uses Google's Gemma 2 2B model for semantic intent classification. This model is hosted on HuggingFace Hub and requires authentication to access.

## Why Authentication is Required

Google Gemma models are **gated repositories** on HuggingFace. This means:
- They are publicly available but require license acceptance
- You need a HuggingFace account and API token to download them
- This is a one-time setup process

## Setup Instructions

### 1. Create HuggingFace Account

If you don't have one already:
- Visit: https://huggingface.co/join
- Create a free account

### 2. Accept the Gemma License

- Visit: https://huggingface.co/google/gemma-2-2b-it
- Click "Agree and access repository"
- Read and accept the license terms
- Access is granted immediately after acceptance

### 3. Generate API Token

- Visit: https://huggingface.co/settings/tokens
- Click "New token"
- Give it a name (e.g., "mnemefusion")
- Select "Read" permissions (sufficient for model downloads)
- Copy the token (starts with `hf_...`)

### 4. Configure Token

You have two options:

#### Option A: Environment Variable (Recommended)

**Linux/macOS:**
```bash
export HF_TOKEN=hf_your_token_here
```

Add to `~/.bashrc` or `~/.zshrc` for persistence:
```bash
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="hf_your_token_here"
```

For persistence:
```powershell
[System.Environment]::SetEnvironmentVariable('HF_TOKEN', 'hf_your_token_here', 'User')
```

#### Option B: In Code

```rust
use mnemefusion_core::slm::SlmConfig;

let config = SlmConfig::default()
    .with_hf_token("hf_your_token_here");
```

**⚠️ Warning:** Don't commit tokens to version control! Use environment variables instead.

## Usage Examples

### Rust

```rust
use mnemefusion_core::slm::{SlmClassifier, SlmConfig};

// Option 1: Use default config (reads HF_TOKEN from environment)
let mut classifier = SlmClassifier::new(SlmConfig::default())?;

// Option 2: Set token explicitly
let config = SlmConfig::default()
    .with_hf_token(std::env::var("HF_TOKEN")?);
let mut classifier = SlmClassifier::new(config)?;

// Classify query
let result = classifier.classify_intent("Who was the first speaker?")?;
println!("Intent: {:?}, Confidence: {}", result.intent, result.confidence);
```

### Running Examples

```bash
# Set token
export HF_TOKEN=hf_your_token_here

# Run test (first time downloads ~2GB model)
cargo run --example slm_test --features slm --release

# Subsequent runs use cached model (fast startup)
cargo run --example slm_test --features slm --release
```

## Troubleshooting

### Error: "403 Forbidden"

**Cause:** Authentication failed or license not accepted

**Fix:**
1. Check token is set: `echo $HF_TOKEN`
2. Verify token is valid at https://huggingface.co/settings/tokens
3. Ensure you accepted the license at https://huggingface.co/google/gemma-2-2b-it
4. Token must start with `hf_`

### Error: "No HuggingFace token provided"

**Cause:** HF_TOKEN environment variable not set

**Fix:**
```bash
export HF_TOKEN=hf_your_token_here
```

### Error: "Failed to download model files"

**Cause:** Network issues or token permissions

**Fix:**
1. Check internet connection
2. Ensure token has "Read" permissions
3. Try downloading manually:
   ```bash
   huggingface-cli download google/gemma-2-2b-it
   ```

## Model Information

- **Model:** google/gemma-2-2b-it (Instruction-tuned Gemma 2 2B)
- **Size:** ~2GB compressed
- **License:** Gemma Terms of Use (permissive, allows commercial use)
- **Format:** SafeTensors (Candle-compatible)
- **Context:** 8K tokens
- **Architecture:** Transformer decoder

## Privacy & Security

- Your HF token is **never logged or transmitted** except to HuggingFace's official API
- Models are cached locally in `~/.cache/mnemefusion/models`
- No telemetry or tracking is performed
- Token is only used during model download (not inference)

## Alternative Models

If you prefer a different model, you can specify it:

```rust
let config = SlmConfig::new("your-model-id")
    .with_hf_token(token);
```

Requirements:
- Must be a Candle-compatible transformer model
- Must output logits for text generation
- Should support instruction-following for best results

## Next Steps

Once authenticated:
1. Run `cargo run --example slm_test --features slm --release`
2. First run downloads model (~2GB, takes 2-5 minutes)
3. Subsequent runs are instant (uses cached model)
4. Test your own queries

## Support

- HuggingFace authentication issues: https://huggingface.co/docs/hub/security-tokens
- Gemma license questions: https://ai.google.dev/gemma/terms
- MnemeFusion issues: https://github.com/yourusername/mnemefusion/issues
