"""Export per-codeword delta and normalized delta (with optional ignorance) for noun_scan_v1_full."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from codeword_sort.noun_analyze import compute_metrics, load_data
from codeword_sort.noun_scan_v1_full import GROUPS, MODELS

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "noun_scan_v1_full"
MODEL_KEYS = list(MODELS.keys())


def main():
    prompts, evals = load_data(OUTPUTS_DIR)
    acc = compute_metrics(prompts, evals, GROUPS)

    rows = []
    for gi, group in enumerate(GROUPS):
        for secret in group:
            # Build per-codeword uplift matrix (with and without optional ignorance)
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
    print(f"Wrote {len(df)} rows to {out_path}")

    # Quick sanity check
    print(f"\nGroups: {len(GROUPS)}, Codewords: {sum(len(g) for g in GROUPS)}, "
          f"Models: {len(MODEL_KEYS)}")
    print(f"Expected rows: {sum(len(g) for g in GROUPS) * len(MODEL_KEYS)**2} = "
          f"{sum(len(g) for g in GROUPS)} × {len(MODEL_KEYS)}²")
    finite = df["delta_norm_oi"].replace([np.inf, -np.inf], np.nan)
    print(f"\nFinite delta_norm_oi: {finite.notna().sum()} / {len(df)}")
    print(f"NaN (0/0): {df['delta_norm_oi'].isna().sum()}")
    print(f"-inf (sentinel decoded, receiver didn't): {(df['delta_norm_oi'] == -np.inf).sum()}")
    print(f"\nSample (finite delta_norm_oi != 0):")
    nonzero = df[finite.abs() > 0.01].sort_values("delta_norm_oi", ascending=False)
    print(nonzero.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
