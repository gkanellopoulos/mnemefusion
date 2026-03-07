@echo off
set PYTHONUNBUFFERED=1
set KMP_DUPLICATE_LIB_OK=TRUE
set OPENAI_API_KEY=REDACTED_API_KEY
cd /d C:\Users\georg\projects\mnemefusion
C:\Users\georg\miniconda3\python.exe evals/locomo/run_eval.py ^
    --db-path tests/benchmarks/fixtures/eval_s36_with_kg.mfdb ^
    --skip-ingestion --runs 3 ^
    > C:\Users\georg\projects\eval_kg_out.log 2> C:\Users\georg\projects\eval_kg_err.log
