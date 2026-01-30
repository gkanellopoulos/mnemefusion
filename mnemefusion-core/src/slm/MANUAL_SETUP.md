# Manual Model Setup Guide

This guide explains how to manually download and place the Gemma 2 2B model for MnemeFusion's SLM intent classification.

## Why Manual Setup?

Manual model placement is the **recommended approach** because:
- ✓ No need for HuggingFace API tokens
- ✓ One-time setup, works offline afterwards
- ✓ Full control over model location
- ✓ No authentication complexity

## Step-by-Step Instructions

### 1. Accept the License

1. Visit: https://huggingface.co/google/gemma-2-2b-it
2. Click "Agree and access repository" (requires HuggingFace account)
3. Wait for approval (usually instant)

### 2. Download Model Files

You need these 3 files from the repository:

**Required files:**
- `model.safetensors` (~2GB) - The model weights
- `config.json` (~1KB) - Model configuration
- `tokenizer.json` (~2MB) - Tokenizer configuration

**Download methods:**

#### Option A: Web Browser
1. Go to https://huggingface.co/google/gemma-2-2b-it/tree/main
2. Click on each file and download:
   - Click "model.safetensors" → Click download button
   - Click "config.json" → Click download button
   - Click "tokenizer.json" → Click download button

#### Option B: Command Line (HuggingFace CLI)
```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login (one-time)
huggingface-cli login

# Download model
huggingface-cli download google/gemma-2-2b-it \
    model.safetensors \
    config.json \
    tokenizer.json \
    --local-dir /path/to/your/models/gemma-2-2b-it
```

#### Option C: Git LFS
```bash
# Install git-lfs
git lfs install

# Clone repository
git clone https://huggingface.co/google/gemma-2-2b-it /path/to/your/models/gemma-2-2b-it
```

### 3. Organize Files

Place the downloaded files in a directory:

```
/your/models/directory/
└── gemma-2-2b-it/
    ├── model.safetensors
    ├── config.json
    └── tokenizer.json
```

Example locations:
- **Linux/macOS:** `/opt/models/gemma-2-2b-it/`
- **Windows:** `C:\models\gemma-2-2b-it\`
- **Project-relative:** `./models/gemma-2-2b-it/`

### 4. Configure MnemeFusion

**Option A: Environment Variable (Recommended)**

```bash
# Linux/macOS
export MODEL_PATH=/opt/models/gemma-2-2b-it

# Windows (PowerShell)
$env:MODEL_PATH="C:\models\gemma-2-2b-it"

# Windows (CMD)
set MODEL_PATH=C:\models\gemma-2-2b-it
```

For persistence, add to your shell profile:
```bash
# Linux/macOS - Add to ~/.bashrc or ~/.zshrc
echo 'export MODEL_PATH=/opt/models/gemma-2-2b-it' >> ~/.bashrc
source ~/.bashrc
```

**Option B: In Code**

```rust
use mnemefusion_core::slm::SlmConfig;

let config = SlmConfig::default()
    .with_model_path("/opt/models/gemma-2-2b-it");
```

**Option C: Configuration File**

```rust
// Load from config file
let model_path = config_file.get("model_path")?;
let slm_config = SlmConfig::default()
    .with_model_path(model_path);
```

### 5. Verify Setup

Run the test example:

```bash
# Set model path
export MODEL_PATH=/opt/models/gemma-2-2b-it

# Run test
cargo run --example slm_test --features slm --release
```

**Expected output:**
```
=== MnemeFusion SLM Classifier Test ===

✓ MODEL_PATH found: /opt/models/gemma-2-2b-it
Loading model from local directory...

Configuration:
  Model ID: google/gemma-2-2b-it
  Model path: /opt/models/gemma-2-2b-it
  Timeout: 5000ms
  Min confidence: 0.6

Initializing SLM classifier...
✓ Found all required model files in local directory
Classifier created successfully!

Testing 7 queries...
...
```

## Troubleshooting

### Error: "Model path does not exist"

**Problem:** The directory doesn't exist or path is wrong

**Solution:**
```bash
# Check if directory exists
ls /opt/models/gemma-2-2b-it  # Linux/macOS
dir C:\models\gemma-2-2b-it   # Windows

# Verify MODEL_PATH is set
echo $MODEL_PATH              # Linux/macOS
echo %MODEL_PATH%             # Windows
```

### Error: "Model file not found"

**Problem:** Missing model.safetensors file

**Solution:**
```bash
# Check files in directory
ls -lh /opt/models/gemma-2-2b-it

# You should see:
# -rw-r--r-- 1 user group 2.0G model.safetensors
# -rw-r--r-- 1 user group  856 config.json
# -rw-r--r-- 1 user group 1.8M tokenizer.json
```

Make sure you downloaded the complete `model.safetensors` file (~2GB).

### Error: "config.json not found"

**Problem:** Missing config.json file

**Solution:**
Download config.json from https://huggingface.co/google/gemma-2-2b-it/blob/main/config.json

### Error: "tokenizer.json not found"

**Problem:** Missing tokenizer.json file

**Solution:**
Download tokenizer.json from https://huggingface.co/google/gemma-2-2b-it/blob/main/tokenizer.json

### Model loads but inference is slow

**Expected:** First inference is ~2-5 seconds (model loading), subsequent queries are <100ms

**Tips:**
- Use `--release` mode for faster inference
- Ensure model.safetensors is fully downloaded (check file size ~2GB)
- Consider GPU acceleration (set `config.with_gpu(true)` if CUDA available)

## Team Deployment

For deploying to multiple machines:

### Option 1: Shared Network Drive
```bash
# Place model once
/shared/models/gemma-2-2b-it/

# Each machine sets
export MODEL_PATH=/shared/models/gemma-2-2b-it
```

### Option 2: Docker Container
```dockerfile
FROM rust:latest

# Copy model into container
COPY models/gemma-2-2b-it /opt/models/gemma-2-2b-it

# Set environment variable
ENV MODEL_PATH=/opt/models/gemma-2-2b-it

# ... rest of Dockerfile
```

### Option 3: Ansible/Puppet Automation
```yaml
# Ansible example
- name: Download Gemma model
  command: huggingface-cli download google/gemma-2-2b-it

- name: Set MODEL_PATH
  lineinfile:
    path: /etc/environment
    line: "MODEL_PATH=/opt/models/gemma-2-2b-it"
```

## Alternative Models

If you prefer a different model, ensure it's:
- Compatible with Candle framework
- SafeTensors format
- Transformer-based architecture

Update configuration:
```rust
let config = SlmConfig::new("your-model-id")
    .with_model_path("/path/to/your-model");
```

## Support

If you encounter issues:
1. Check file permissions (model files must be readable)
2. Verify disk space (need ~2.5GB for model)
3. Ensure no firewall blocking model download (if using auto-download fallback)
4. Try absolute paths instead of relative paths

## Summary

**Quick setup:**
```bash
# 1. Download model files to a directory
# 2. Set environment variable
export MODEL_PATH=/path/to/gemma-2-2b-it

# 3. Run your application
cargo run --features slm --release
```

That's it! No API tokens, no authentication complexity.
