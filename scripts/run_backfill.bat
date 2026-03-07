@echo off
set PYTHONUNBUFFERED=1
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d C:\Users\georg\projects\mnemefusion
C:\Users\georg\miniconda3\python.exe scripts/backfill_kg.py ^
    --source tests/benchmarks/fixtures/eval_s36_phi4_10conv.mfdb ^
    --output tests/benchmarks/fixtures/eval_s36_with_kg.mfdb ^
    --triplex-model models/triplex/Triplex-Q4_K_M.gguf ^
    > C:\Users\georg\projects\backfill_out.log 2> C:\Users\georg\projects\backfill_err.log
