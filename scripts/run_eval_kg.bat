@echo off
REM Run LoCoMo evaluation on KG-backfilled database
REM Requires OPENAI_API_KEY set in environment
set PYTHONUNBUFFERED=1
set KMP_DUPLICATE_LIB_OK=TRUE
if "%OPENAI_API_KEY%"=="" (
    echo ERROR: OPENAI_API_KEY not set. Export it before running this script.
    exit /b 1
)
python evals/locomo/run_eval.py ^
    --db-path tests/benchmarks/fixtures/eval_s36_with_kg.mfdb ^
    --skip-ingestion --runs 3
