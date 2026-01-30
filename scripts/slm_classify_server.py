#!/usr/bin/env python3
"""
SLM intent classifier server - persistent process
Reads queries from stdin, writes results to stdout
Keeps model loaded in memory for fast inference
"""

import sys
import json
from pathlib import Path
from llama_cpp import Llama

def load_model(model_path: str) -> Llama:
    """Load the GGUF model once"""
    print(f"[SLM-SERVER] Loading model from: {model_path}", file=sys.stderr)
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0,  # CPU only by default
        verbose=False
    )
    print(f"[SLM-SERVER] Model loaded successfully", file=sys.stderr)
    sys.stderr.flush()
    return model

def classify_intent(model: Llama, query: str) -> dict:
    """Classify query intent using SLM"""

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
    if len(sys.argv) != 2:
        print("Usage: slm_classify_server.py <model_path>", file=sys.stderr)
        print("Server reads queries from stdin (one per line) and writes JSON results to stdout", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        # Load model once at startup
        model = load_model(model_path)

        # Signal ready
        print("READY", flush=True)

        # Process queries from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            if line == "QUIT":
                print(f"[SLM-SERVER] Shutting down", file=sys.stderr)
                break

            try:
                # Parse input (JSON with query)
                request = json.loads(line)
                query = request.get('query', '')

                # Classify
                result = classify_intent(model, query)

                # Write result as JSON to stdout
                print(json.dumps(result), flush=True)

            except Exception as e:
                # Write error result
                error_result = {
                    "error": str(e),
                    "intent": "Factual",
                    "confidence": 0.0
                }
                print(json.dumps(error_result), flush=True)
                print(f"[SLM-SERVER] Error processing query: {e}", file=sys.stderr)

    except Exception as e:
        print(f"[SLM-SERVER] Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
