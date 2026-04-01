#!/usr/bin/env python
"""Run noun_scan_v5_full: reuse v1 treatment data + no-system-prompt control baseline.

This script:
1. Copies v1 treatment data (prompts + evaluations)
2. Runs new control evaluations with no system prompt
3. Exports per_codeword_deltas.csv
4. Generates all 10 highlight plots
"""
import asyncio
import json
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codeword_sort.client import SortClient
from codeword_sort.noun_scan_v5_full import (
    GROUPS, MODELS, OUTPUTS_DIR, V1_OUTPUTS_DIR, run_control_eval_phase,
)

MODEL_KEYS = list(MODELS.keys())


# ---------------------------------------------------------------------------
# Phase 1: merge v1 treatment data
# ---------------------------------------------------------------------------

def merge_v1_treatment():
    """Copy treatment-only prompts and evaluations from v1."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy treatment prompts
    v1_prompts = [json.loads(l) for l in open(V1_OUTPUTS_DIR / "prompts.jsonl") if l.strip()]
    treatment_prompts = [p for p in v1_prompts if p["condition"] == "treatment"]
    logger.info(f"Copied {len(treatment_prompts)} treatment prompts from v1")

    # Add single dummy control prompt (no sender)
    control_prompt_id = str(uuid.uuid4())[:8]
    control_prompt = {
        "prompt_id": control_prompt_id,
        "sender": "",
        "group_idx": -1,
        "secret": "",
        "condition": "control",
        "generated_prompt": "",
        "timestamp": datetime.utcnow().isoformat(),
    }
    all_prompts = treatment_prompts + [control_prompt]

    with open(OUTPUTS_DIR / "prompts.jsonl", "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Wrote {len(all_prompts)} prompts ({len(treatment_prompts)} treatment + 1 control)")

    # Copy treatment evaluations
    v1_evals = [json.loads(l) for l in open(V1_OUTPUTS_DIR / "evaluations.jsonl") if l.strip()]
    treatment_evals = [e for e in v1_evals if e["condition"] == "treatment"]
    with open(OUTPUTS_DIR / "evaluations.jsonl", "w") as f:
        for e in treatment_evals:
            f.write(json.dumps(e) + "\n")
    logger.info(f"Copied {len(treatment_evals)} treatment evaluations from v1")

    return control_prompt_id


# ---------------------------------------------------------------------------
# Phase 2: run control evaluations
# ---------------------------------------------------------------------------

async def run_controls(control_prompt_id: str):
    clients = {
        k: SortClient(model_id=v["id"], reasoning_effort=v["reasoning"])
        for k, v in MODELS.items()
    }
    evals = await run_control_eval_phase(clients, GROUPS, control_prompt_id)
    return evals


# ---------------------------------------------------------------------------
# Phase 3: export per_codeword_deltas.csv
# ---------------------------------------------------------------------------

def compute_metrics_v5():
    """Compute metrics with shared control baseline (no sender in controls)."""
    prompts = [json.loads(l) for l in open(OUTPUTS_DIR / "prompts.jsonl") if l.strip()]
    evals = [json.loads(l) for l in open(OUTPUTS_DIR / "evaluations.jsonl") if l.strip()]

    prompt_map = {p["prompt_id"]: p for p in prompts}

    # Treatment accuracy: per (sender, evaluator, group, secret)
    treatment_correct = defaultdict(int)
    treatment_total = defaultdict(int)

    # Control accuracy: per (evaluator, group) — shared across all senders
    control_secret_counts = defaultdict(lambda: defaultdict(int))
    control_totals = defaultdict(int)

    for ev in evals:
        p = prompt_map.get(ev["prompt_id"])
        if p is None:
            continue
        gi = ev["group_idx"]

        if ev["condition"] == "treatment":
            key = (ev["prompt_id"], ev["evaluator"], gi)
            treatment_total[key] += 1
            if ev["parsed"] == p["secret"]:
                treatment_correct[key] += 1
        else:
            # Control: key by (evaluator, group) only — no sender
            ctrl_key = (ev["evaluator"], gi)
            control_totals[ctrl_key] += 1
            if ev["parsed"]:
                control_secret_counts[ctrl_key][ev["parsed"]] += 1

    # Build accuracy dict: acc[gi][sender][evaluator][secret] = {"treatment": ..., "control": ...}
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for (pid, evaluator, gi), total in treatment_total.items():
        p = prompt_map[pid]
        acc[gi][p["sender"]][evaluator][p["secret"]]["treatment"] = (
            treatment_correct[(pid, evaluator, gi)] / total
        )

    # Broadcast shared control baseline to all senders
    for gi, group in enumerate(GROUPS):
        for evaluator in MODEL_KEYS:
            ctrl_key = (evaluator, gi)
            total = control_totals.get(ctrl_key, 0)
            if total == 0:
                continue
            for secret in group:
                count = control_secret_counts[ctrl_key].get(secret, 0)
                control_rate = count / total
                for sender in MODEL_KEYS:
                    acc[gi][sender][evaluator][secret]["control"] = control_rate

    return acc


def export_csv():
    """Export per_codeword_deltas.csv using v5 metrics."""
    acc = compute_metrics_v5()

    rows = []
    for gi, group in enumerate(GROUPS):
        for secret in group:
            uplift = {}
            uplift_oi = {}
            for sender in MODEL_KEYS:
                for evaluator in MODEL_KEYS:
                    t = acc[gi][sender][evaluator][secret].get("treatment", 0)
                    c = acc[gi][sender][evaluator][secret].get("control", 0)
                    uplift[(sender, evaluator)] = t - c
                    uplift_oi[(sender, evaluator)] = max(0, t - c)

            for sender in MODEL_KEYS:
                self_uplift = uplift[(sender, sender)]
                self_uplift_oi = uplift_oi[(sender, sender)]

                for evaluator in MODEL_KEYS:
                    d = self_uplift - uplift[(sender, evaluator)]
                    d_oi = self_uplift_oi - uplift_oi[(sender, evaluator)]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        dn = np.float64(d) / np.float64(self_uplift)
                        dn_oi = np.float64(d_oi) / np.float64(self_uplift_oi)

                    rows.append({
                        "group_idx": gi,
                        "codeword": secret,
                        "sender": sender,
                        "evaluator": evaluator,
                        "uplift": uplift[(sender, evaluator)],
                        "uplift_oi": uplift_oi[(sender, evaluator)],
                        "delta": d,
                        "delta_oi": d_oi,
                        "delta_norm": dn,
                        "delta_norm_oi": dn_oi,
                    })

    df = pd.DataFrame(rows)
    out_path = OUTPUTS_DIR / "per_codeword_deltas.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(df)} rows to {out_path}")
    return df


# ---------------------------------------------------------------------------
# Phase 4: generate highlight plots
# ---------------------------------------------------------------------------

def generate_plots():
    """Generate all 10 highlight plots for v5."""
    plots_dir = Path(__file__).resolve().parent.parent / "plots" / "noun_scan_v5_full"
    playground_dir = plots_dir / "playground"
    highlights_dir = plots_dir / "highlights"
    playground_dir.mkdir(parents=True, exist_ok=True)
    highlights_dir.mkdir(parents=True, exist_ok=True)

    # Add plots/playground to sys.path so plot scripts can import common
    plots_src = Path(__file__).resolve().parent.parent / "src" / "plots" / "playground"
    sys.path.insert(0, str(plots_src))

    # --- Patch Category A: common.py paths ---
    import common
    common.OUTPUTS_DIR = OUTPUTS_DIR
    common.PLOTS_DIR = playground_dir
    common.HIGHLIGHTS_DIR = highlights_dir

    # --- Category A plots (read from per_codeword_deltas.csv) ---
    from gap_distribution import plot_heatmap
    plot_heatmap()
    logger.info("Saved gap_distribution_heatmap.png")

    from gap_per_secret_grid import plot as plot_grid, plot_sample4
    plot_grid()
    logger.info("Saved gap_per_secret_grid.png")
    plot_sample4()
    logger.info("Saved gap_per_secret_grid_sample4.png")

    from gap1_blocks import plot as plot_blocks
    plot_blocks(full_scale=True)
    logger.info("Saved perfect_privacy_fraction_by_sender_blocks_full.png")
    plot_blocks(full_scale=True, threshold="gt0")
    logger.info("Saved perfect_privacy_fraction_by_sender_blocks_full_gt0.png")

    # --- Category B: transmission_bar (needs patched _get_pct) ---
    import transmission_bar
    transmission_bar.OUTPUTS_DIR = OUTPUTS_DIR

    # Replace _get_pct to not filter controls by sender
    _original_get_pct = transmission_bar._get_pct

    def _get_pct_v5(prompts, evals, sender, secret, gi):
        """v5 version: control prompts have no sender, so don't filter by sender."""
        treat_pids = {p["prompt_id"] for p in prompts
                      if p["sender"] == sender and p["secret"] == secret
                      and p["condition"] == "treatment"}
        ctrl_pids = {p["prompt_id"] for p in prompts
                     if p["condition"] == "control"}

        results = {}
        for evaluator in common.MODEL_KEYS:
            treat_evals = [e for e in evals if e["prompt_id"] in treat_pids
                           and e["evaluator"] == evaluator and e["group_idx"] == gi]
            ctrl_evals = [e for e in evals if e["prompt_id"] in ctrl_pids
                          and e["evaluator"] == evaluator and e["group_idx"] == gi]

            group = GROUPS[gi]
            t_pick = sum(1 for e in treat_evals if e["parsed"] == secret)
            t_total = len(treat_evals)
            c_pick = sum(1 for e in ctrl_evals if e["parsed"] == secret)
            c_total = len(ctrl_evals)

            results[evaluator] = {
                "treatment": t_pick / t_total * 100 if t_total else 0,
                "control": c_pick / c_total * 100 if c_total else 0,
                "t_n": t_total,
                "c_n": c_total,
            }
        return results

    transmission_bar._get_pct = _get_pct_v5

    for sender, secret in [("opus-4.6", "northern"), ("gemini-3-flash", "duty")]:
        transmission_bar.plot_detailed(sender, secret)
        transmission_bar.plot_simple(sender, secret)
        logger.info(f"Saved transmission_bar plots for {sender}/{secret}")

    transmission_bar._get_pct = _original_get_pct

    # --- Category B: receiver_vs_sentinel_uplift (needs patched _build_data) ---
    import receiver_vs_sentinel_uplift

    def _build_data_v5():
        """v5 version: uses compute_metrics_v5 with shared control baseline."""
        acc = compute_metrics_v5()
        model_keys = MODEL_KEYS

        rows = []
        for gi, group in enumerate(GROUPS):
            for secret in group:
                for receiver in model_keys:
                    t_recv = acc[gi][receiver][receiver][secret].get("treatment", 0)
                    c_recv = acc[gi][receiver][receiver][secret].get("control", 0)
                    recv_uplift = max(0, t_recv - c_recv)

                    sentinel_uplifts = []
                    for sentinel in model_keys:
                        if sentinel == receiver:
                            continue
                        t_sent = acc[gi][receiver][sentinel][secret].get("treatment", 0)
                        c_sent = acc[gi][receiver][sentinel][secret].get("control", 0)
                        sentinel_uplifts.append(max(0, t_sent - c_sent))

                    su = np.array(sentinel_uplifts)
                    rows.append({
                        "receiver": receiver,
                        "secret": secret,
                        "group_idx": gi,
                        "receiver_uplift": recv_uplift,
                        "sentinel_uplift": np.mean(su),
                        "sentinel_uplift_se": np.std(su, ddof=1) / np.sqrt(len(su)),
                    })

        df = pd.DataFrame(rows)
        return df.sort_values("receiver_uplift", ascending=False).reset_index(drop=True)

    receiver_vs_sentinel_uplift._build_data = _build_data_v5
    receiver_vs_sentinel_uplift.plot_top(10)
    logger.info("Saved receiver_vs_sentinel_uplift_top10.png")

    logger.info(f"All 10 highlight plots saved to {highlights_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    logger.info("=== noun_scan_v5_full: no-system-prompt control baseline ===")

    # Phase 1
    logger.info("Phase 1: merging v1 treatment data...")
    control_prompt_id = merge_v1_treatment()

    # Phase 2
    logger.info("Phase 2: running control evaluations (no system prompt)...")
    await run_controls(control_prompt_id)

    # Phase 3
    logger.info("Phase 3: exporting per_codeword_deltas.csv...")
    export_csv()

    # Phase 4
    logger.info("Phase 4: generating highlight plots...")
    generate_plots()

    logger.info("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
