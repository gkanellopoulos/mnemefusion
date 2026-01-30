#!/usr/bin/env python3
"""
SLM intent classifier script
Called from Rust via subprocess for intent classification
"""

import sys
import json
from pathlib import Path
from llama_cpp import Llama

# Global model cache
_model = None
_model_path = None

def load_model(model_path: str):
    """Load the GGUF model (cached)"""
    global _model, _model_path

    if _model is not None and _model_path == model_path:
        return _model

    print(f"[SLM] Loading model from: {model_path}", file=sys.stderr)
    _model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0,  # CPU only by default
        verbose=False
    )
    _model_path = model_path
    print(f"[SLM] Model loaded successfully", file=sys.stderr)

    return _model

def classify_intent(model_path: str, query: str, timeout_ms: int = 100) -> dict:
    """Classify query intent using SLM"""

    model = load_model(model_path)

    # Create classification prompt
    prompt = f"""<|im_start|>system
You are a query intent classifier. Classify queries into: Entity, Temporal, Causal, or Factual.<|im_end|>
<|im_start|>user
Classify this query: "{query}"

Respond with JSON:
{{
  "intent": "Entity|Temporal|Causal|Factual",
  "confidence": 0.0-1.0,
  "entity_focus": "subject entity if intent=Entity, else null",
  "reasoning": "brief explanation"
}}<|im_end|>
<|im_start|>assistant
"""

    # Generate classification
    output = model(
        prompt,
        max_tokens=150,
        temperature=0.1,  # Low temperature for consistency
        stop=["<|im_end|>", "</s>"],
        echo=False
    )

    text = output['choices'][0]['text'].strip()

    # Extract JSON
    json_start = text.find('{')
    json_end = text.rfind('}')

    if json_start == -1 or json_end == -1:
        raise ValueError(f"No JSON found in model output: {text}")

    json_str = text[json_start:json_end+1]
    result = json.loads(json_str)

    return result

def main():
    print(f"[DEBUG] SLM script called with {len(sys.argv)} arguments", file=sys.stderr)
    print(f"[DEBUG] Arguments: {sys.argv}", file=sys.stderr)

    if len(sys.argv) != 3:
        print("Usage: slm_classify.py <model_path> <query>", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    query = sys.argv[2]

    print(f"[DEBUG] Classifying query: '{query}'", file=sys.stderr)

    try:
        result = classify_intent(model_path, query)
        print(f"[DEBUG] Classification result: {result}", file=sys.stderr)
        # Output JSON result to stdout
        print(json.dumps(result))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
