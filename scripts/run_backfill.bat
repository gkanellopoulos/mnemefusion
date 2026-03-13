@echo off
REM Run KG backfill on evaluation database
set PYTHONUNBUFFERED=1
set KMP_DUPLICATE_LIB_OK=TRUE
python scripts/backfill_kg.py ^
    --source tests/benchmarks/fixtures/eval_s36_phi4_10conv.mfdb ^
    --output tests/benchmarks/fixtures/eval_s36_with_kg.mfdb ^
    --triplex-model models/triplex/Triplex-Q4_K_M.gguf
