# SLM Quick Start - Manual Model Setup

## TL;DR

1. **Download model:** Get 3 files from https://huggingface.co/google/gemma-2-2b-it
2. **Place locally:** Put in `/opt/models/gemma-2-2b-it/` (or any directory)
3. **Set path:** `export MODEL_PATH=/opt/models/gemma-2-2b-it`
4. **Run:** Your code works immediately - no API tokens needed!

## Setup (5 minutes)

### Step 1: Accept License
Visit https://huggingface.co/google/gemma-2-2b-it and click "Agree and access repository"

### Step 2: Download Files
Download these 3 files:
- **model.safetensors** (~2GB)
- **config.json** (~1KB)
- **tokenizer.json** (~2MB)

### Step 3: Create Directory Structure
```bash
mkdir -p /opt/models/gemma-2-2b-it
# Move downloaded files there
```

### Step 4: Set Environment Variable
```bash
# Add to ~/.bashrc or ~/.zshrc
export MODEL_PATH=/opt/models/gemma-2-2b-it
```

### Step 5: Test
```bash
cargo run --example slm_test --features slm --release
```

## Usage in Code

```rust
use mnemefusion_core::slm::{SlmClassifier, SlmConfig};

// Option 1: Use environment variable (recommended)
let config = SlmConfig::default();  // Reads MODEL_PATH automatically

// Option 2: Set path in code
let config = SlmConfig::default()
    .with_model_path("/opt/models/gemma-2-2b-it");

// Create classifier and use
let mut classifier = SlmClassifier::new(config)?;
let result = classifier.classify_intent("Who was the speaker?")?;

println!("Intent: {:?}", result.intent);
println!("Confidence: {}", result.confidence);
```

## Directory Structure

Your model directory should look like this:
```
/opt/models/gemma-2-2b-it/
├── model.safetensors      # 2GB - Model weights
├── config.json            # 1KB - Model config
└── tokenizer.json         # 2MB - Tokenizer
```

## Environment Variable

```bash
# Linux/macOS - Permanent
echo 'export MODEL_PATH=/opt/models/gemma-2-2b-it' >> ~/.bashrc
source ~/.bashrc

# Windows PowerShell - Permanent
[System.Environment]::SetEnvironmentVariable('MODEL_PATH', 'C:\models\gemma-2-2b-it', 'User')
```

## Verification

```bash
# Check path is set
echo $MODEL_PATH

# Check files exist
ls -lh $MODEL_PATH

# Should show:
# -rw-r--r-- 1 user group 2.0G model.safetensors
# -rw-r--r-- 1 user group  856 config.json
# -rw-r--r-- 1 user group 1.8M tokenizer.json
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Model path does not exist" | Check `echo $MODEL_PATH` and verify directory exists |
| "Model file not found" | Ensure `model.safetensors` is in the directory |
| "config.json not found" | Download config.json from HuggingFace |
| "tokenizer.json not found" | Download tokenizer.json from HuggingFace |

## Team Deployment

**Shared server:**
```bash
# One person downloads to shared location
/shared/models/gemma-2-2b-it/

# Everyone sets same path
export MODEL_PATH=/shared/models/gemma-2-2b-it
```

**Multiple machines:**
```bash
# Option 1: Copy model to each machine
scp -r /opt/models/gemma-2-2b-it server:/opt/models/

# Option 2: Use NFS/SMB share
# Option 3: Include in Docker image
```

## Performance

- **First load:** 2-5 seconds (loads model into RAM)
- **Subsequent queries:** <100ms per query
- **Memory usage:** ~2.5GB RAM
- **Disk space:** ~2.5GB

## Links

- **Full setup guide:** See `MANUAL_SETUP.md`
- **Change log:** See `CHANGES.md`
- **Model page:** https://huggingface.co/google/gemma-2-2b-it
- **License:** Gemma Terms of Use (commercial use allowed)

## Support

Questions? Check:
1. `MANUAL_SETUP.md` - Detailed setup instructions
2. `README.md` - Authentication options (if you prefer auto-download)
3. Integration tests: `cargo test --features slm --lib`

---

**Key Point:** No API tokens, no authentication, no complexity. Just download → place → use. 🚀
