#!/usr/bin/env python3
"""
SLM metadata extraction server - persistent process
Reads content from stdin, extracts rich metadata, writes JSON results to stdout
Keeps model loaded in memory for fast inference
"""

import sys
import json
from pathlib import Path
from llama_cpp import Llama


def load_model(model_path: str) -> Llama:
    """Load the GGUF model once"""
    print(f"[SLM-EXTRACT] Loading model from: {model_path}", file=sys.stderr)
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0,  # CPU only by default
        verbose=False
    )
    print(f"[SLM-EXTRACT] Model loaded successfully", file=sys.stderr)
    sys.stderr.flush()
    return model


def extract_metadata(model: Llama, content: str) -> dict:
    """Extract rich metadata from content using SLM"""

    # Create extraction prompt
    prompt = f"""<|im_start|>system
You are a metadata extractor. Extract structured information from text for memory indexing.<|im_end|>
<|im_start|>user
Extract metadata from this memory:
"{content}"

Respond with JSON only:
{{
  "entities": [
    {{"name": "entity name", "role": "subject|object|organization|location", "mentions": ["name", "pronoun"], "entity_type": "person|organization|location|concept"}}
  ],
  "temporal": {{
    "markers": ["yesterday", "last week"],
    "sequence": "early|middle|late|null",
    "relative_time": "before current|concurrent|after current|null",
    "absolute_dates": ["2024-01-15"]
  }},
  "causal": {{
    "relationships": [{{"cause": "event A", "effect": "event B", "confidence": 0.8}}],
    "density": 0.0-1.0,
    "explicit_markers": ["because", "therefore"],
    "has_implicit_causation": true|false
  }},
  "topics": ["topic1", "topic2"],
  "importance": 0.0-1.0
}}<|im_end|>
<|im_start|>assistant
"""

    # Generate extraction with higher max_tokens for richer output
    output = model(
        prompt,
        max_tokens=500,
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
                error_result = {
                    "error": str(e),
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
                print(json.dumps(error_result), flush=True)
                print(f"[SLM-EXTRACT] Error processing content: {e}", file=sys.stderr)

    except Exception as e:
        print(f"[SLM-EXTRACT] Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
