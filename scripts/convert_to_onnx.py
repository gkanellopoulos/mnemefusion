#!/usr/bin/env python3
"""
Convert HuggingFace model to ONNX format for tract inference
"""

import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def convert_to_onnx(model_path: str, output_path: str):
    """Convert a HuggingFace model to ONNX format"""
    print(f"[INFO] Loading model from: {model_path}")

    try:
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # ONNX works best with float32
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print(f"[INFO] Model loaded successfully")
        print(f"[INFO] Model type: {type(model).__name__}")

        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare dummy input
        text = "What is the meaning of life?"
        inputs = tokenizer(text, return_tensors="pt")

        print(f"[INFO] Exporting to ONNX...")
        print(f"[INFO] Output: {output_dir / 'model.onnx'}")

        # Export to ONNX
        torch.onnx.export(
            model,
            (inputs['input_ids'],),
            output_dir / "model.onnx",
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'logits': {0: 'batch', 1: 'sequence'}
            },
            opset_version=14,
            do_constant_folding=True,
        )

        # Copy tokenizer files
        print(f"[INFO] Copying tokenizer files...")
        tokenizer.save_pretrained(output_dir)

        # Copy config
        print(f"[INFO] Copying config...")
        model.config.save_pretrained(output_dir)

        print(f"[SUCCESS] Conversion complete!")
        print(f"[INFO] ONNX model saved to: {output_dir}")

        # List output files
        print(f"\n[INFO] Output files:")
        for file in sorted(output_dir.iterdir()):
            size = file.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {file.name} ({size:.1f} MB)")

    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_onnx.py <model_path> <output_path>")
        print("Example: python convert_to_onnx.py opt/models/qwen3-0.6b opt/models/qwen3-0.6b-onnx")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    convert_to_onnx(model_path, output_path)
