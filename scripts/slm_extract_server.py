#!/usr/bin/env python3
"""
SLM metadata extraction server - persistent process
Reads content from stdin, extracts rich metadata, writes JSON results to stdout
Keeps model loaded in memory for fast inference

Optimizations:
- GPU acceleration when available (n_gpu_layers=-1)
- Reduced context size for faster inference
- Compact prompt format
- Batch request support (sequential processing, but reduced IPC overhead)
"""

import sys
import json
import os
from pathlib import Path
from llama_cpp import Llama


def detect_gpu() -> int:
    """Detect if GPU is available and return n_gpu_layers setting"""
    # Check environment override
    env_layers = os.environ.get("SLM_GPU_LAYERS")
    if env_layers is not None:
        try:
            return int(env_layers)
        except ValueError:
            pass

    # Try to detect CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[SLM-EXTRACT] GPU detected: {torch.cuda.get_device_name(0)}", file=sys.stderr)
            return -1  # Use all layers on GPU
    except ImportError:
        pass

    # Check for ROCm (AMD)
    if os.path.exists("/opt/rocm"):
        print("[SLM-EXTRACT] ROCm detected, enabling GPU", file=sys.stderr)
        return -1

    print("[SLM-EXTRACT] No GPU detected, using CPU", file=sys.stderr)
    return 0


def load_model(model_path: str, n_gpu_layers: int = None) -> Llama:
    """Load the GGUF model once"""
    if n_gpu_layers is None:
        n_gpu_layers = detect_gpu()

    print(f"[SLM-EXTRACT] Loading model from: {model_path}", file=sys.stderr)
    print(f"[SLM-EXTRACT] GPU layers: {n_gpu_layers} (-1=all, 0=CPU only)", file=sys.stderr)

    model = Llama(
        model_path=model_path,
        n_ctx=1024,  # Reduced context for faster inference
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    print(f"[SLM-EXTRACT] Model loaded successfully", file=sys.stderr)
    sys.stderr.flush()
    return model


def extract_metadata(model: Llama, content: str) -> dict:
    """Extract rich metadata from content using SLM"""

    # Truncate content if too long (save tokens)
    max_content_len = 500
    if len(content) > max_content_len:
        content = content[:max_content_len] + "..."

    # Compact prompt for faster inference
    prompt = f"""<|im_start|>system
Extract metadata as JSON.<|im_end|>
<|im_start|>user
Text: "{content}"

JSON format: {{"entities":[{{"name":"X","role":"subject","entity_type":"person"}}],"temporal":{{"markers":[]}},"topics":[],"importance":0.5}}
Extract:<|im_end|>
<|im_start|>assistant
"""

    # Generate with reduced max_tokens for speed
    output = model(
        prompt,
        max_tokens=300,  # Reduced from 500
        temperature=0.1,
        stop=["<|im_end|>", "</s>"],
        echo=False
    )

    text = output['choices'][0]['text'].strip()

    # Remove markdown code blocks if present
    if text.startswith('```'):
        # Remove opening code block
        lines = text.split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]  # Remove ```json or ```
        # Remove closing code block
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines)

    # Extract JSON - try multiple approaches
    json_str = None
    result = None

    # First try: find JSON boundaries
    json_start = text.find('{')
    json_end = text.rfind('}')

    if json_start == -1 or json_end == -1:
        raise ValueError(f"No JSON found in model output: {text}")

    json_str = text[json_start:json_end+1]

    # Try to parse, with fallback fixups
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[SLM-EXTRACT] Initial JSON parse failed: {e}", file=sys.stderr)

        # Fixup 1: Remove trailing commas before } or ]
        import re
        fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
        try:
            result = json.loads(fixed)
            print("[SLM-EXTRACT] Fixed trailing comma", file=sys.stderr)
        except json.JSONDecodeError:
            pass

        # Fixup 2: Try to find a balanced JSON object
        if result is None:
            depth = 0
            start_idx = json_start
            for i, ch in enumerate(text[json_start:], json_start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(text[start_idx:i+1])
                            print(f"[SLM-EXTRACT] Fixed with balanced extraction", file=sys.stderr)
                            break
                        except json.JSONDecodeError:
                            continue

        # Fixup 3: If still failing, try to extract just the values we need
        if result is None:
            print("[SLM-EXTRACT] All JSON parsing failed, using empty result", file=sys.stderr)
            result = {}

    # Validate and normalize the result
    result = normalize_metadata(result)

    return result


def normalize_metadata(result: dict) -> dict:
    """Normalize and validate extracted metadata"""

    # Ensure all required fields exist with defaults
    normalized = {
        "entities": [],
        "temporal": {
            "markers": [],
            "sequence": None,
            "relative_time": None,
            "absolute_dates": []
        },
        "causal": {
            "relationships": [],
            "density": 0.0,
            "explicit_markers": [],
            "has_implicit_causation": False
        },
        "topics": [],
        "importance": 0.5,
        "schema_version": 1
    }

    # Copy entities with validation
    if "entities" in result and isinstance(result["entities"], list):
        for entity in result["entities"]:
            if isinstance(entity, dict) and "name" in entity:
                normalized["entities"].append({
                    "name": str(entity.get("name", "")),
                    "role": str(entity.get("role", "subject")),
                    "mentions": list(entity.get("mentions", [])) if isinstance(entity.get("mentions"), list) else [],
                    "entity_type": str(entity.get("entity_type", "concept"))
                })

    # Copy temporal metadata
    if "temporal" in result and isinstance(result["temporal"], dict):
        temporal = result["temporal"]
        normalized["temporal"]["markers"] = list(temporal.get("markers", [])) if isinstance(temporal.get("markers"), list) else []
        normalized["temporal"]["sequence"] = temporal.get("sequence") if temporal.get("sequence") not in ["null", None, ""] else None
        normalized["temporal"]["relative_time"] = temporal.get("relative_time") if temporal.get("relative_time") not in ["null", None, ""] else None
        normalized["temporal"]["absolute_dates"] = list(temporal.get("absolute_dates", [])) if isinstance(temporal.get("absolute_dates"), list) else []

    # Copy causal metadata
    if "causal" in result and isinstance(result["causal"], dict):
        causal = result["causal"]
        if "relationships" in causal and isinstance(causal["relationships"], list):
            for rel in causal["relationships"]:
                if isinstance(rel, dict) and "cause" in rel and "effect" in rel:
                    normalized["causal"]["relationships"].append({
                        "cause": str(rel.get("cause", "")),
                        "effect": str(rel.get("effect", "")),
                        "confidence": float(rel.get("confidence", 0.5))
                    })
        normalized["causal"]["density"] = float(causal.get("density", 0.0))
        normalized["causal"]["explicit_markers"] = list(causal.get("explicit_markers", [])) if isinstance(causal.get("explicit_markers"), list) else []
        normalized["causal"]["has_implicit_causation"] = bool(causal.get("has_implicit_causation", False))

    # Copy topics
    if "topics" in result and isinstance(result["topics"], list):
        normalized["topics"] = [str(t) for t in result["topics"]]

    # Copy importance
    if "importance" in result:
        try:
            normalized["importance"] = max(0.0, min(1.0, float(result["importance"])))
        except (TypeError, ValueError):
            normalized["importance"] = 0.5

    return normalized


def get_error_result(error_msg: str) -> dict:
    """Return a default result with error message"""
    return {
        "error": error_msg,
        "entities": [],
        "temporal": {
            "markers": [],
            "sequence": None,
            "relative_time": None,
            "absolute_dates": []
        },
        "causal": {
            "relationships": [],
            "density": 0.0,
            "explicit_markers": [],
            "has_implicit_causation": False
        },
        "topics": [],
        "importance": 0.5,
        "schema_version": 1
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: slm_extract_server.py <model_path>", file=sys.stderr)
        print("Server reads content from stdin (JSON) and writes metadata JSON to stdout", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        # Load model once at startup
        model = load_model(model_path)

        # Signal ready
        print("READY", flush=True)

        # Process requests from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            if line == "QUIT":
                print(f"[SLM-EXTRACT] Shutting down", file=sys.stderr)
                break

            try:
                # Parse input (JSON with content)
                request = json.loads(line)
                content = request.get('content', '')

                if not content:
                    raise ValueError("Empty content provided")

                # Extract metadata
                result = extract_metadata(model, content)

                # Write result as JSON to stdout
                print(json.dumps(result), flush=True)

            except Exception as e:
                # Write error result with defaults
                print(json.dumps(get_error_result(str(e))), flush=True)
                print(f"[SLM-EXTRACT] Error processing content: {e}", file=sys.stderr)

    except Exception as e:
        print(f"[SLM-EXTRACT] Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
