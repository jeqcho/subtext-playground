#!/bin/bash
# Monitor the eval process and auto-run analysis when done

LOG_FILE="/home/ubuntu/subtext-playground/logs/eval_run.log"
EVAL_FILE="/home/ubuntu/subtext-playground/outputs/sentinel_scan/evaluations.jsonl"
TARGET=145800

echo "Monitoring eval process for completion..."
echo "Target: $TARGET evaluations"

while true; do
    # Check if eval process is still running
    if ! pgrep -f "run_eval_only.py" > /dev/null 2>&1; then
        echo "Eval process finished!"
        break
    fi

    # Count current progress
    current=$(wc -l < "$EVAL_FILE" 2>/dev/null || echo 0)
    pct=$((current * 100 / TARGET))
    echo "$(date '+%H:%M:%S') - Progress: $current / $TARGET ($pct%)"

    # Check if target reached
    if [ "$current" -ge "$TARGET" ]; then
        echo "Target reached!"
        break
    fi

    sleep 30
done

echo ""
echo "===== Running analysis ====="
cd /home/ubuntu/subtext-playground
uv run python -c "from sentinel_scan.analyze import run_analysis; run_analysis()"

echo ""
echo "===== Analysis complete ====="
echo "Heatmaps saved to plots/sentinel_scan/"
ls -la plots/sentinel_scan/
ls -la plots/sentinel_scan/per_animal/
